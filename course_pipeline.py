"""
對齊課程 M1 / M2 Notebook 的交付項目（在既有解析成果上補齊，不重做）：

M1
  - pages.jsonl：逐頁原文 + PDF 頁碼 + 印刷頁碼 + SHA-256
  - chunks_structure.jsonl：課程資料契約欄位（document_id, pdf_sha256, source_pages, source_boxes, text_sha256 …）
  - chunking_comparison.csv：fixed / recursive（LangChain splitter）/ structure 三種切法比較
  - audit：文字雜湊、頁碼範圍、bbox 是否在頁面內
M2
  - BM25、TF-IDF（char 3–5 gram）、Dense、三種 Hybrid（RRF）
  - Parent–Child 上下文還原（同節前後相鄰 chunk）
  - evidence_set.json（candidate_evidence_only）＋ 原文框選截圖
  - Golden QA：evaluation.csv（Hit@5 / MRR@10，各檢索方法比較）
  - ranking_comparison.csv、run_summary.json

用法：python course_pipeline.py /path/to/cbh-gcb-eng.pdf
"""
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import rag_langchain as R
from parse_cbh import page_lines

OUT = R.OUT
COURSE = OUT / "course"
MAX_WORDS, OVERLAP = 180, 25          # 與課程相同的比較基準（英文空白分詞，非 token）
BODY_PAGES = range(12, 359)           # PDF 頁 12–358：前言到附錄（排除封面、目錄、索引）

# 人工標記的評測題：目標章節依「章節標題」決定，不是用檢索結果反推（避免循環驗證）
# 命中規則：chunk 的 section_id 等於或位於目標章節之下（例如 11.3 也接受 11.3.x）
GOLD = [
    ("How to use the Canadian Biosafety Handbook", ["1.4"]),
    ("Risk management in biosafety", ["4.4"]),
    ("Biosecurity plans", ["6.2"]),
    ("生物安全手冊的指引是否具有強制法律效力？", ["1.3", "1.4"]),
    ("BSC 要多久認證一次？", ["11.3"]),
    ("What are the differences between the types of Class II biological safety cabinets?", ["11.1.2"]),
    ("HEPA 過濾器的過濾效率是多少？", ["10.2"]),
    ("使用離心機時要注意哪些安全事項？", ["12.1"]),
    ("高壓滅菌鍋要如何驗證是否有效？", ["15.4", "15.2"]),
    ("What should a biological spill response procedure include?", ["17.3"]),
    ("實驗室發生暴露事件時，什麼情況要通報 PHAC？", ["18.1"]),
    ("Pre-placement medical evaluation for laboratory personnel", ["7.2"]),
    ("緊急醫療聯絡卡上要寫哪些資訊？", ["7.7"]),
    ("訓練紀錄需要保存哪些內容？", ["8.5"]),
    ("How should infectious material be packaged and transported?", ["20.2"]),
    ("病原體庫存清單要記錄哪些資訊？", ["19.2"]),
    ("What is inward directional airflow and how is it verified?", ["10.1"]),
    ("朊病毒（prion）汙染要怎麼除汙？", ["15.12", "12.13", "13.8"]),
    ("哪一種手套材質適合搭配漂白水等化學消毒劑？", ["9.1"]),
    ("Working safely with non-human primates", ["13.9"]),
    ("生物醫療廢棄物要如何貯存與處置？", ["16.2", "16.1"]),
    ("不同化學消毒劑對各類微生物的效果比較", ["15.3"]),
]
DEMO_QUESTIONS = ["How to use the Canadian Biosafety Handbook", "BSC 要多久認證一次？"]


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ================================================================ M1
def m1_pages(doc, pdf_sha):
    pages = []
    doc_id = R.load_chunks()[0]["doc_id"]
    for i in range(len(doc)):
        text = doc[i].get_text()
        _, printed = page_lines(doc[i])
        pages.append({"document_id": doc_id, "pdf_sha256": pdf_sha, "pdf_page": i + 1, "printed_page": printed,
                      "text": text, "source_sha256": sha256(text)})
    save_jsonl(pages, COURSE / "pages.jsonl")
    assert [p["pdf_page"] for p in pages] == list(range(1, len(pages) + 1))
    assert all(sha256(p["text"]) == p["source_sha256"] for p in pages)
    return pages


def m1_contract(chunks, pdf_sha):
    """把既有 chunk 轉成課程 M1→M2 的資料契約欄位（保留原有全部 metadata）。"""
    rows = []
    for c in chunks:
        rows.append({
            **c,
            "method": "structure",
            "document_id": c["doc_id"],
            "pdf_sha256": pdf_sha,
            "pdf_page": c["pdf_page_start"],
            "source_pages": sorted({l["pdf_page"] for l in c["locations"]}),
            "source_boxes": [{"pdf_page": l["pdf_page"], "bbox": l["bbox"], "block_id": f"{c['chunk_id']}#{i}"}
                             for i, l in enumerate(c["locations"])],
            "text_sha256": sha256(c["text"]),
        })
    save_jsonl(rows, COURSE / "chunks_structure.jsonl")
    return rows


def audit(chunks, doc):
    issues = {}
    for c in chunks:
        p = []
        if sha256(c["text"]) != c["text_sha256"]:
            p.append("文字雜湊不符")
        for src in c["source_boxes"]:
            n = src["pdf_page"]
            if not 1 <= n <= len(doc):
                p.append(f"頁碼超出範圍：{n}")
                continue
            r = pymupdf.Rect(src["bbox"])
            if r.is_empty or not doc[n - 1].rect.contains(r):
                p.append(f"第 {n} 頁座標異常")
        if p:
            issues[c["chunk_id"]] = p
    return issues


def m1_chunking_comparison(doc, structure, sections):
    """固定長度 / 遞迴（LangChain）/ 章節感知 三種切法用同一份內文比較。"""
    parts, starts = [], []
    pos = 0
    for i in BODY_PAGES:
        t = doc[i - 1].get_text()
        # 去掉頁首與頁碼行，讓三種方法吃到一樣乾淨的內文
        t = "\n".join(l for l in t.splitlines()
                      if l.strip() and "Canadian Biosafety Handbook (CBH)" not in l
                      and not re.fullmatch(r"\s*[0-9ivxl]+\s*", l) and not l.startswith("CHAPTER "))
        starts.append((pos, i))
        parts.append(t)
        pos += len(t) + 2
    full = "\n\n".join(parts)

    def page_of(off):
        page = starts[0][1]
        for s, p in starts:
            if s > off:
                break
            page = p
        return page

    wl = lambda s: len(s.split())
    splitters = {
        "fixed": CharacterTextSplitter(separator=" ", chunk_size=MAX_WORDS, chunk_overlap=OVERLAP,
                                       length_function=wl, add_start_index=True),
        "recursive": RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " "], chunk_size=MAX_WORDS,
                                                    chunk_overlap=OVERLAP, length_function=wl, add_start_index=True),
    }
    norm_ws = lambda s: re.sub(r"\s+", " ", s)
    headings = [norm_ws(f"{sid} {v['title']}") for sid, v in sections.items() if re.match(r"^\d", sid)]

    def crosses(text):
        t = norm_ws(text)
        return any(t.find(h) > 0 for h in headings)

    stats = []
    for name, sp in splitters.items():
        docs = sp.create_documents([full])
        pages = [(page_of(d.metadata["start_index"]), page_of(d.metadata["start_index"] + len(d.page_content)))
                 for d in docs]
        texts = [d.page_content for d in docs]
        stats.append(summarize(name, texts, [a != b for a, b in pages], [crosses(t) for t in texts]))
    body = [c for c in structure if c["chunk_type"] in ("body", "resource_list")]
    stats.append(summarize("structure (本專案)", [c["text"] for c in structure],
                           [c["pdf_page_start"] != c["pdf_page_end"] for c in structure],
                           [False] * len(structure), extra={"body_chunks": len(body)}))
    df = pd.DataFrame(stats)
    df.to_csv(COURSE / "chunking_comparison.csv", index=False, encoding="utf-8-sig")
    return df


def summarize(name, texts, multi_page, cross, extra=None):
    words = [len(t.split()) for t in texts]
    mid = [not re.search(r"[.:;!?)\]”\"]\s*$", t.strip()) for t in texts]
    return {"method": name, "chunks": len(texts), "avg_words": round(float(np.mean(words)), 1),
            "max_words": int(max(words)), "cross_section_%": round(100 * float(np.mean(cross)), 1),
            "multi_page_%": round(100 * float(np.mean(multi_page)), 1),
            "ends_mid_sentence_%": round(100 * float(np.mean(mid)), 1), **(extra or {})}


# ================================================================ M2 檢索
class Retrievers:
    def __init__(self, chunks):
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.index, self.ids = R.load_index(self.by_id)
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        texts = [self.by_id[cid]["embed_text"] for cid in self.ids]
        self.bm25 = BM25Okapi([R.tokenize(t) for t in texts])
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=100000)
        self.tfidf_m = self.tfidf.fit_transform(texts)
        # Parent = 同一 section_id 的 chunk，依原文順序
        self.parents = defaultdict(list)
        for c in chunks:
            self.parents[(c["doc_id"], c["section_id"])].append(c["chunk_id"])

    def bm25_rank(self, q, k=30):
        s = self.bm25.get_scores(R.tokenize(q))
        return [self.ids[i] for i in np.argsort(-s)[:k]]

    def tfidf_rank(self, q, k=30):
        s = cosine_similarity(self.tfidf.transform([q]), self.tfidf_m).ravel()
        return [self.ids[i] for i in np.argsort(-s)[:k]]

    def dense_rank(self, q, k=30):
        _, hits = self.index.search(R.embed([q]), k)
        return [self.ids[i] for i in hits[0] if i >= 0]

    @staticmethod
    def rrf(lists, k0=60, limit=10):
        sc = defaultdict(float)
        for lst in lists:
            for r, cid in enumerate(lst):
                sc[cid] += 1 / (k0 + r + 1)
        return sorted(sc, key=sc.get, reverse=True)[:limit]

    def system_rank(self, q):
        rq, ctx = R.retrieve(q, self.by_id)
        return [c["chunk_id"] for c in ctx], rq

    def all_methods(self, q):
        bm, tf, de = self.bm25_rank(q), self.tfidf_rank(q), self.dense_rank(q)
        sysr, rq = self.system_rank(q)
        return {
            "BM25": bm[:10], "TF-IDF": tf[:10], "Dense": de[:10],
            "Hybrid BM25+TF-IDF": self.rrf([bm, tf]),
            "Hybrid Dense+BM25": self.rrf([de, bm]),
            "本系統（改寫+Dense×2+BM25）": sysr,
        }, rq

    def expand_parent(self, cid, neighbors=1, max_words=900):
        c = self.by_id[cid]
        sib = self.parents[(c["doc_id"], c["section_id"])]
        i = sib.index(cid)
        sel = sib[max(0, i - neighbors): i + neighbors + 1]
        text, used = [], []
        for s in sel:
            t = self.by_id[s]["text"]
            if sum(len(x.split()) for x in text) + len(t.split()) > max_words and s != cid:
                continue
            text.append(t)
            used.append(s)
        pages = sorted({l["pdf_page"] for s in used for l in self.by_id[s]["locations"]})
        return {"parent_key": f"{c['doc_id']}::{c['section_id']}", "child_ids": used, "focus_chunk_id": cid,
                "pages": pages, "context": "\n\n".join(text)}


def hit_and_rr(ranked, expected, k_hit=5):
    ok = lambda cid: any(cid.split(":")[1] == e or cid.split(":")[1].startswith(e + ".") for e in expected)
    hit = int(any(ok(c) for c in ranked[:k_hit]))
    rr = next((1 / (i + 1) for i, c in enumerate(ranked[:10]) if ok(c)), 0.0)
    return hit, rr


def evaluate(ret):
    rows = []
    for q, exp in GOLD:
        methods, rq = ret.all_methods(q)
        for m, ranked in methods.items():
            h, rr = hit_and_rr(ranked, exp)
            rows.append({"question": q, "target_sections": "|".join(exp), "method": m, "Hit@5": h,
                         "RR@10": round(rr, 3), "top1": ranked[0] if ranked else ""})
    df = pd.DataFrame(rows)
    df.to_csv(COURSE / "evaluation.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("method", sort=False).agg(**{"Hit@5": ("Hit@5", "mean"), "MRR@10": ("RR@10", "mean")}).round(3)
    zh = df[df["question"].str.contains(r"[一-鿿]")]
    summary["Hit@5 (中文題)"] = zh.groupby("method", sort=False)["Hit@5"].mean().round(3)
    summary.to_csv(COURSE / "evaluation_summary.csv", encoding="utf-8-sig")
    return df, summary


def ranking_comparison(ret):
    rows = []
    for q in DEMO_QUESTIONS:
        methods, _ = ret.all_methods(q)
        for m, ranked in methods.items():
            for r, cid in enumerate(ranked[:5], 1):
                c = ret.by_id[cid]
                rows.append({"question": q, "method": m, "rank": r, "chunk_id": cid, "section": c["section_id"],
                             "pdf_pages": f"{c['pdf_page_start']}-{c['pdf_page_end']}",
                             "excerpt": c["text"][:100].replace("\n", " ")})
    df = pd.DataFrame(rows)
    df.to_csv(COURSE / "ranking_comparison.csv", index=False, encoding="utf-8-sig")
    return df


def evidence_set(ret, contract_by_id, q, max_evidence=4):
    ranked, _ = ret.system_rank(q)
    ev = []
    for cid in ranked[:max_evidence]:
        c = contract_by_id[cid]
        valid_hash = sha256(c["text"]) == c["text_sha256"]
        valid_pages = all(isinstance(p, int) and p >= 1 for p in c["source_pages"])
        valid_boxes = all(len(b["bbox"]) == 4 for b in c["source_boxes"])
        parent = ret.expand_parent(cid)
        ev.append({"evidence_id": f"E{len(ev) + 1:03d}", "chunk_id": cid, "document_id": c["document_id"],
                   "section_id": c["section_id"], "section_title": c["section_title"],
                   "citation_label": c["citation_label"], "source_pages": c["source_pages"],
                   "printed_pages": [c["printed_page_start"], c["printed_page_end"]],
                   "source_boxes": c["source_boxes"], "text": c["text"], "text_sha256": c["text_sha256"],
                   "content_origin": c.get("content_origin", "pdf_text"),
                   "parent_context": {k: parent[k] for k in ("parent_key", "child_ids", "pages")},
                   "technical_check": valid_hash and valid_pages and valid_boxes})
    return {"query": q, "status": "candidate_evidence_only", "evidence": ev}


def preview(doc, ev, path):
    box = ev["source_boxes"][0]
    tmp = pymupdf.open()
    tmp.insert_pdf(doc, from_page=box["pdf_page"] - 1, to_page=box["pdf_page"] - 1)
    for b in ev["source_boxes"]:
        if b["pdf_page"] == box["pdf_page"]:
            tmp[0].draw_rect(pymupdf.Rect(b["bbox"]), color=(1, 0.1, 0.1), width=1.3)
    tmp[0].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(str(path))
    tmp.close()


def main(pdf_path):
    t0 = time.time()
    COURSE.mkdir(parents=True, exist_ok=True)
    pdf_sha = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    doc = pymupdf.open(pdf_path)
    chunks = R.load_chunks()
    sections = json.loads((OUT / "sections.json").read_text(encoding="utf-8"))

    print("== M1")
    m1_pages(doc, pdf_sha)
    contract = m1_contract(chunks, pdf_sha)
    issues = audit(contract, doc)
    print(f"技術稽核：{len(contract) - len(issues)}/{len(contract)} 無異常", list(issues.items())[:3])
    cmp_df = m1_chunking_comparison(doc, chunks, sections)
    print(cmp_df.to_string(index=False))

    print("\n== M2")
    ret = Retrievers(chunks)
    ranking_comparison(ret)
    ex = ret.expand_parent(ret.system_rank(DEMO_QUESTIONS[0])[0][0])
    print("Parent 展開示範：", ex["parent_key"], ex["child_ids"], "pages", ex["pages"])
    contract_by_id = {c["chunk_id"]: c for c in contract}
    evs = [evidence_set(ret, contract_by_id, q) for q in DEMO_QUESTIONS]
    (COURSE / "evidence_set.json").write_text(json.dumps(evs, ensure_ascii=False, indent=2), encoding="utf-8")
    for i, e in enumerate(evs, 1):
        preview(doc, e["evidence"][0], COURSE / f"evidence_preview_q{i}.png")
    df, summary = evaluate(ret)
    print(summary.to_string())

    run = {"pdf_sha256": pdf_sha, "chunks": len(contract), "audit_issues": len(issues),
           "gold_questions": len(GOLD), "evaluation": summary.reset_index().to_dict(orient="records"),
           "evidence_questions": DEMO_QUESTIONS, "seconds": round(time.time() - t0, 1),
           "models": {"embed": R.EMBED_MODEL, "rewrite": R.REWRITE_MODEL, "answer": R.ANSWER_MODEL}}
    (COURSE / "run_summary.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n輸出：{COURSE}")


if __name__ == "__main__":
    main(sys.argv[1])
