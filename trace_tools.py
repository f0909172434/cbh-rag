"""
來源追溯工具：
  1) verify  —— 檢查 LLM 回答裡的每個引用：chunk_id 是否存在、quote 是否真的出現在該 chunk 原文
  2) show    —— 把 chunk 在 PDF 原頁上的位置畫框輸出成 PNG（給人眼核對）

用法：
  python trace_tools.py verify answer.json
  python trace_tools.py show CBH-2E-2016:4.3.1:001 /path/to/cbh-gcb-eng.pdf
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).parent / "rag_output"


def load_chunks():
    src = OUT / "chunks_enriched.jsonl"
    if not src.exists():
        src = OUT / "chunks.jsonl"
    with open(src, encoding="utf-8") as f:
        return {c["chunk_id"]: c for c in map(json.loads, f)}


def norm(s):
    # 忽略連字號／破折號差異（PDF 行尾斷字、en dash 與 hyphen 混用），避免誤判成幻覺
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[-–—\u00ad]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def verify(answer_path):
    """answer.json 格式（要求 LLM 用此 JSON schema 輸出）：
    {"answer": "...", "claims": [{"statement": "...", "chunk_id": "...", "quote": "原文逐字摘錄"}]}
    """
    chunks = load_chunks()
    ans = json.loads(Path(answer_path).read_text(encoding="utf-8"))
    ok = True
    for i, cl in enumerate(ans.get("claims", []), 1):
        c = chunks.get(cl.get("chunk_id"))
        if not c:
            print(f"[{i}] ✗ chunk_id 不存在：{cl.get('chunk_id')}")
            ok = False
            continue
        q = norm(cl.get("quote", ""))
        if q in norm(c["text"]):
            print(f"[{i}] ✓ {c['citation_label']}")
        elif q in norm(c.get("vlm_text", "") + " " + c.get("figure_description_vlm", "")):
            # 只出現在 VLM 產生的內容裡：可用，但要標示不是 PDF 原文
            print(f"[{i}] ⚠ 引用的是 VLM 產生的內容（{c['generated_by']['model']}），非 PDF 原文：{c['citation_label']}")
        else:
            print(f"[{i}] ✗ quote 不在原文中（可能是幻覺或改寫）：{cl.get('quote')[:80]}")
            ok = False
    print("全部通過" if ok else "有引用未通過驗證，該回答不應直接採用")
    return ok


def show(chunk_id, pdf_path):
    import pymupdf
    c = load_chunks()[chunk_id]
    doc = pymupdf.open(pdf_path)
    outs = []
    for page_no in sorted({l["pdf_page"] for l in c["locations"]}):
        page = doc[page_no - 1]
        for loc in c["locations"]:
            if loc["pdf_page"] == page_no:
                page.draw_rect(pymupdf.Rect(loc["bbox"]), color=(1, 0, 0), width=1)
        fn = OUT / f"trace_{chunk_id.replace(':', '_')}_p{page_no}.png"
        page.get_pixmap(dpi=110).save(fn)
        outs.append(str(fn))
    print(c["citation_label"])
    print("\n".join(outs))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "verify":
        sys.exit(0 if verify(sys.argv[2]) else 1)
    elif cmd == "show":
        show(sys.argv[2], sys.argv[3])
