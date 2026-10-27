"""
用 OpenRouter 的 vision 模型補強 parse_cbh.py 抽不好的部分：
  1) table_quality == needs_review 的表格 → 看截圖重建 header/rows
  2) figure_caption → 看整頁圖描述圖的內容

原則（防幻覺）：
  - 原始 PDF 文字（text）一律保留，VLM 產生的內容放在獨立欄位，並標 content_origin / generated_by
  - 表格的每個 cell 會回頭比對 PDF 文字層，對不到的 cell 列在 vlm_unverified_cells
  - 結果寫到 rag_output/chunks_enriched.jsonl，不覆蓋原檔

用法：
  export OPENROUTER_API_KEY=...        # 或在本資料夾建立 .env，內容 OPENROUTER_API_KEY=...
  python openrouter_enrich.py /path/to/cbh-gcb-eng.pdf [--model anthropic/claude-sonnet-5.5]
"""
import argparse
import base64
import datetime
import json
import os
import re
import urllib.request
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
OUT = HERE / "rag_output"
API = "https://openrouter.ai/api/v1/chat/completions"

TABLE_PROMPT = """You are transcribing a table from the Canadian Biosafety Handbook.
The image(s) show {table_id}: "{caption}" (multiple images = the table continues across pages).
Transcribe it EXACTLY, verbatim, do not paraphrase or infer values that are not visible.
Flatten multi-level headers into one header row (join levels with " / ").
For merged/row-spanning cells, repeat the value in every row it spans.
Return only JSON: {{"header": [...], "rows": [[...], ...], "notes": "footnotes/legend text verbatim, or empty"}}"""

FIGURE_PROMPT = """This is page {page} of the Canadian Biosafety Handbook. It contains {figure_id}: "{caption}".
Describe ONLY what is visible in that figure, for use in a search index: its components, labels (quote them verbatim),
arrows/flow and what they connect. For a decision chart, list each decision node and its branches.
Do not add biosafety knowledge that is not shown in the figure. Plain text, max 250 words."""


def load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    env = HERE / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise SystemExit("找不到 OPENROUTER_API_KEY（環境變數或 .env）")
    return key


def call(key, model, prompt, images_png):
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
        for b in images_png]
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "X-Title": "CBH RAG enrichment"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_json(s):
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    return json.loads(s)


def norm(s):
    return re.sub(r"[\s\-–—]+", " ", s).strip().lower()


def to_md(header, rows):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    args = ap.parse_args()
    key = load_key()
    doc = pymupdf.open(args.pdf)
    chunks = [json.loads(l) for l in open(OUT / "chunks.jsonl", encoding="utf-8")]
    now = datetime.date.today().isoformat()
    log = []

    for c in chunks:
        if c["chunk_type"] == "table" and c.get("table_quality") == "needs_review":
            pages = sorted({l["pdf_page"] for l in c["locations"]})
            imgs = []
            for loc in c["locations"]:
                pg = doc[loc["pdf_page"] - 1]
                x0, y0, x1, y1 = loc["bbox"]
                clip = pymupdf.Rect(x0 - 8, y0 - 8, x1 + 8, y1 + 8) & pg.rect
                imgs.append(pg.get_pixmap(clip=clip, dpi=200).tobytes("png"))
            res = parse_json(call(key, args.model, TABLE_PROMPT.format(
                table_id=c["table_id"], caption=c["table_caption"]), imgs))
            # 每個 cell 回頭比對 PDF 文字層：VLM 只能「排版」，不能「生出」原文沒有的字
            page_text = norm(" ".join(doc[p - 1].get_text() for p in pages))
            unverified = [cell for r in res["rows"] for cell in r
                          if str(cell).strip() and norm(str(cell)) not in page_text]
            c["table_header_vlm"], c["table_rows_vlm"] = res["header"], res["rows"]
            c["vlm_text"] = f"[{c['table_id']}] {c['table_caption']}\n" + to_md(res["header"], res["rows"]) + \
                (f"\nNote: {res['notes']}" if res.get("notes") else "")
            c["vlm_unverified_cells"] = unverified
            c["embed_text"] = " > ".join(c["section_path"]) + "\n\n" + c["vlm_text"]
            c["content_origin"] = "pdf_text+vlm_table_layout"
            c["generated_by"] = {"model": args.model, "date": now, "task": "table_transcription"}
            log.append((c["table_id"], f"{len(res['rows'])} rows, unverified cells: {len(unverified)}"))

        elif c["chunk_type"] == "figure_caption":
            p = c["pdf_page_start"]
            img = doc[p - 1].get_pixmap(dpi=150).tobytes("png")
            caption = c["text"].split("] ", 1)[-1]
            desc = call(key, args.model, FIGURE_PROMPT.format(
                page=c["printed_page_start"], figure_id=c["figure_id"], caption=caption), [img]).strip()
            c["figure_description_vlm"] = desc
            c["embed_text"] = " > ".join(c["section_path"]) + "\n\n" + c["text"] + \
                "\n[Figure description, VLM-generated]\n" + desc
            c["content_origin"] = "pdf_caption+vlm_description"
            c["generated_by"] = {"model": args.model, "date": now, "task": "figure_description"}
            log.append((c["figure_id"], f"{len(desc.split())} words"))
        else:
            c.setdefault("content_origin", "pdf_text")

    with open(OUT / "chunks_enriched.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for k, v in log:
        print(k, "→", v)
    print(f"寫入 {OUT / 'chunks_enriched.jsonl'}")


if __name__ == "__main__":
    main()
