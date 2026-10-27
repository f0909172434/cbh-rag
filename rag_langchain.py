"""
CBH RAG（LangChain + OpenRouter）：混合檢索 → 強制引用的結構化回答 → 程式驗證引用

流程：
  1. 查詢改寫：中文（或任何語言）問題 → 英文檢索句 + 關鍵字（文件是英文，BM25 需要英文）
  2. 混合檢索：FAISS 向量（原問題 + 英文改寫，cosine）＋ BM25（英文關鍵字），用 RRF 融合排名
     （直接用 faiss / rank_bm25，不依賴已停止維護的 langchain-community）
  3. 回答：LLM 以 Pydantic schema 輸出 {answer, claims[{statement, chunk_id, quote}]}
  4. 驗證：每個 quote 必須逐字出現在該 chunk 原文；沒通過的 claim 會要求模型重寫一次，仍失敗就剔除並標示

用法：
  python rag_langchain.py build                 # 建索引（約 840 chunks，embedding 費用 < US$0.01）
  python rag_langchain.py ask "BSC 的 Class II Type A2 和 B2 差在哪？"
"""
import json
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import ContentFilterFinishReasonError
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from trace_tools import norm

HERE = Path(__file__).parent
OUT = HERE / "rag_output"
INDEX_DIR = OUT / "faiss_index"
BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = os.environ.get("CBH_EMBED_MODEL", "qwen/qwen3-embedding-8b")
ANSWER_MODEL = os.environ.get("CBH_ANSWER_MODEL", "anthropic/claude-sonnet-5.5")
REWRITE_MODEL = os.environ.get("CBH_REWRITE_MODEL", "anthropic/claude-haiku-5.5")
TOP_K = 6
CANDIDATES = 10
MAX_CONTEXT_BLOCKS = 12   # 命中 chunk + Parent–Child 展開的相鄰 chunk 上限


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    env = HERE / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("找不到 OPENROUTER_API_KEY（環境變數或 .env）")
    return key


def embeddings():
    # check_embedding_ctx_length=False：直接送文字，不先用 tiktoken 轉 token id（非 OpenAI 模型需要）
    return OpenAIEmbeddings(model=EMBED_MODEL, base_url=BASE_URL, api_key=api_key(),
                            check_embedding_ctx_length=False, chunk_size=64)


def llm(model, **kw):
    return ChatOpenAI(model=model, base_url=BASE_URL, api_key=api_key(), temperature=0, **kw)


def load_chunks():
    src = OUT / "chunks_enriched.jsonl"
    if not src.exists():
        src = OUT / "chunks.jsonl"
    with open(src, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


def embed(texts):
    vecs = np.asarray(embeddings().embed_documents(texts), dtype="float32")
    faiss.normalize_L2(vecs)  # 正規化後內積 = cosine
    return vecs


# ---------------------------------------------------------------- 索引
def build():
    chunks = load_chunks()
    vecs = embed([c["embed_text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for old in ("index.pkl",):  # 舊版 langchain-community 格式
        (INDEX_DIR / old).unlink(missing_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    # 向量第 i 列 ↔ chunk_ids[i]；記下 hash，chunks 變動後可偵測索引過期
    (INDEX_DIR / "ids.json").write_text(json.dumps({
        "embed_model": EMBED_MODEL, "dim": int(vecs.shape[1]),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "chunk_hashes": [c["chunk_hash"] for c in chunks]}, ensure_ascii=False), encoding="utf-8")
    print(f"FAISS 索引完成：{len(chunks)} chunks, dim={vecs.shape[1]} → {INDEX_DIR}")


def load_index(chunks_by_id):
    meta = json.loads((INDEX_DIR / "ids.json").read_text(encoding="utf-8"))
    if meta["embed_model"] != EMBED_MODEL:
        raise SystemExit(f"索引是用 {meta['embed_model']} 建的，目前設定 {EMBED_MODEL}；請重跑 build")
    stale = [cid for cid, h in zip(meta["chunk_ids"], meta["chunk_hashes"])
             if chunks_by_id.get(cid, {}).get("chunk_hash") != h]
    if stale or len(meta["chunk_ids"]) != len(chunks_by_id):
        raise SystemExit(f"索引與 chunks 不一致（{len(stale)} 個 chunk 變動），請重跑 build")
    return faiss.read_index(str(INDEX_DIR / "index.faiss")), meta["chunk_ids"]


# ---------------------------------------------------------------- 檢索
class SearchQuery(BaseModel):
    english_query: str = Field(description="The question rewritten as one English search sentence")
    keywords: list[str] = Field(description="3-8 English keywords/acronyms likely to appear in a biosafety handbook")


def retrieve(question, chunks_by_id):
    try:
        rq = llm(REWRITE_MODEL).with_structured_output(SearchQuery, method="json_schema", strict=True).invoke(
            "Rewrite this question for searching the English-language Canadian Biosafety Handbook.\n"
            f"Question: {question}")
    except ContentFilterFinishReasonError:
        rq = SearchQuery(english_query=question, keywords=[])  # 改寫被拒時直接用原問題檢索
    index, ids = load_index(chunks_by_id)
    _, hits = index.search(embed([question, rq.english_query]), CANDIDATES)
    ranked_lists = [[ids[i] for i in row if i >= 0] for row in hits]
    bm25 = BM25Okapi([tokenize(chunks_by_id[cid]["embed_text"]) for cid in ids])
    bm25_scores = bm25.get_scores(tokenize(rq.english_query + " " + " ".join(rq.keywords)))
    ranked_lists.append([ids[i] for i in np.argsort(-bm25_scores)[:CANDIDATES]])
    # Reciprocal Rank Fusion：score = Σ 1 / (60 + rank)
    scores = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
    top = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
    return rq, [chunks_by_id[c] for c in top]


# ---------------------------------------------------------------- 回答 + 驗證
class Claim(BaseModel):
    statement: str = Field(description="One factual statement used in the answer, in the user's language")
    chunk_id: str = Field(description="The chunk_id of the context block that supports it")
    quote: str = Field(description="A short VERBATIM excerpt (copied exactly, in English) from that chunk proving the statement")


class Answer(BaseModel):
    answer: str = Field(description="The answer in the same language as the question; cite like [1], [2] matching claims order")
    claims: list[Claim]
    insufficient_context: bool = Field(description="True if the context does not contain the answer")


SYSTEM = """You answer questions using ONLY the Canadian Biosafety Handbook excerpts in <context>.
Rules:
- Every factual statement must be backed by a claim with the exact chunk_id and a verbatim quote copied from that block.
- Quotes must be copied character-for-character from the block text (English). Do not translate or paraphrase quotes.
- If the context does not answer the question, set insufficient_context=true and say so; do not use outside knowledge.
- Write the answer in the same language AND script as the question (Traditional Chinese questions get Traditional Chinese answers, never Simplified).
- Distinguish "must" (CBS requirement) from "should" (recommendation) when the text does.
- Blocks marked VLM-GENERATED are machine descriptions of tables/figures; prefer original text when both exist."""


def context_block(c):
    tag = "" if c.get("content_origin", "pdf_text") == "pdf_text" else " VLM-GENERATED"
    body = c.get("vlm_text") or c["text"]
    if c.get("figure_description_vlm"):
        body = c["text"] + "\n" + c["figure_description_vlm"]
    return f'<block chunk_id="{c["chunk_id"]}" source="{c["citation_label"]}"{tag}>\n{body}\n</block>'


def parent_of(section_id):
    """4.3.8.4 → 4.3.8；4.3 → 4.3（第 2 層以上就以自己為 parent）；ch24 等虛擬章不展開。"""
    parts = section_id.split(".")
    return ".".join(parts[:-1]) if len(parts) > 2 else section_id


def expand_parents(ctx, chunks_by_id, window=2):
    """Parent–Child：每個命中 chunk 補上原文順序前後 window 個、且屬於同一個 parent 章節的 chunk。
    小節很短時（例如 6.2.1.1–6.2.1.5 各只有一段）也能把兄弟小節一起帶進來。
    命中的排前面，展開的接在後面；總數上限 MAX_CONTEXT_BLOCKS。"""
    seq = list(chunks_by_id)
    pos = {cid: i for i, cid in enumerate(seq)}
    out = [c["chunk_id"] for c in ctx]
    for d in range(1, window + 1):
        for c in ctx:
            if not re.match(r"^\d", c["section_id"]):
                continue
            par = parent_of(c["section_id"])
            for j in (pos[c["chunk_id"]] - d, pos[c["chunk_id"]] + d):
                if 0 <= j < len(seq) and len(out) < MAX_CONTEXT_BLOCKS:
                    n = chunks_by_id[seq[j]]
                    sid = n["section_id"]
                    if (sid == par or sid.startswith(par + ".")) and n["chunk_id"] not in out:
                        out.append(n["chunk_id"])
    return [chunks_by_id[cid] for cid in out]


def check(claims, chunks_by_id):
    good, bad = [], []
    for cl in claims:
        c = chunks_by_id.get(cl.chunk_id)
        hay = "" if not c else norm(c["text"] + " " + c.get("vlm_text", "") + " " + c.get("figure_description_vlm", ""))
        (good if c and norm(cl.quote) in hay else bad).append(cl)
    return good, bad


def ask(question):
    chunks_by_id = {c["chunk_id"]: c for c in load_chunks()}
    rq, ctx = retrieve(question, chunks_by_id)
    ctx = expand_parents(ctx, chunks_by_id)
    model = llm(ANSWER_MODEL).with_structured_output(Answer, method="json_schema", strict=True)
    messages = [("system", SYSTEM),
                ("human", "<context>\n" + "\n\n".join(context_block(c) for c in ctx) +
                 f"\n</context>\n\nQuestion: {question}")]
    base = {"question": question, "search_query": rq.model_dump(), "retrieved": [c["chunk_id"] for c in ctx],
            "chunks": chunks_by_id}
    try:
        ans = model.invoke(messages)
    except ContentFilterFinishReasonError:
        # 模型的安全過濾拒答（例如毒素數量）：不硬繞，改成只回傳檢索到的原文出處讓人自己讀
        return {**base, "refused": True}
    good, bad = check(ans.claims, chunks_by_id)
    retried = False
    if bad:  # 給模型一次修正機會
        retried = True
        fb = "\n".join(f"- chunk_id={b.chunk_id} quote={b.quote!r}" for b in bad)
        messages += [("ai", ans.model_dump_json()),
                     ("human", "These quotes were NOT found verbatim in the cited blocks. Rewrite the whole answer; "
                               "copy quotes exactly, or drop statements you cannot support:\n" + fb)]
        ans = model.invoke(messages)
        good, bad = check(ans.claims, chunks_by_id)
    return {**base, "answer": ans, "verified": good, "rejected": bad, "retried": retried, "refused": False}


def show(res):
    print(f"Q: {res['question']}")
    print(f"檢索改寫: {res['search_query']['english_query']} | {res['search_query']['keywords']}")
    print(f"檢索到: {', '.join(res['retrieved'])}\n")
    if res["refused"]:
        print("模型的內容安全過濾拒絕產生回答。以下是檢索到的原文出處，請直接查閱：")
        for cid in res["retrieved"]:
            print("  -", res["chunks"][cid]["citation_label"])
        return
    a = res["answer"]
    print(a.answer, "\n")
    if a.insufficient_context:
        print("（模型判定：手冊內容不足以回答）")
    for i, cl in enumerate(a.claims, 1):
        c = res["chunks"].get(cl.chunk_id, {})
        ok = cl in res["verified"]
        mark = "✓" if ok else "✗ 未通過驗證"
        if ok and c.get("content_origin", "pdf_text") != "pdf_text" and norm(cl.quote) not in norm(c["text"]):
            mark = "⚠ 來自 VLM 產生內容"
        print(f"[{i}] {mark} {c.get('citation_label', cl.chunk_id)}\n    \"{cl.quote}\"")
    if res["rejected"]:
        print(f"\n⚠ {len(res['rejected'])} 個引用未通過驗證，對應敘述不可採信")
    if res["retried"]:
        print("（第一次回答有引用不符，已要求模型重寫一次）")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "ask":
        show(ask(" ".join(sys.argv[2:])))
