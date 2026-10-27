"""
Canadian Biosafety Handbook (CBH) 2nd Ed. → 結構化 RAG chunks（含來源追溯 metadata）

用法：
    python parse_cbh.py /path/to/cbh-gcb-eng.pdf out_dir/

依版面字型判斷結構（此 PDF 無書籤 outline，但排版字型非常一致）：
    22pt Futura-Book "Chapter"      → 章節分隔頁
    16pt Futura-Bold "Chapter N –"  → 章首標題
    13.5 / 12 / 10pt Futura-Book + 編號 → 第 2 / 3 / 4 層節標題
    9pt Futura-Light                → 內文
    ≤5.5pt 數字                     → 上標引用編號（從內文移除，記入 citation_refs）
    8pt Futura-Bold @ y≈600         → 印刷頁碼（直接讀，不用推算偏移）
    y < 45                          → 頁首（丟棄）
"""
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

DOC_ID = "CBH-2E-2016"
DOC_META = {
    "doc_id": DOC_ID,
    "doc_title": "Canadian Biosafety Handbook (CBH), 2nd Edition",
    "publisher": "Public Health Agency of Canada / Canadian Food Inspection Agency",
    "pub_date": "2016-03",
    "language": "en",
    "isbn_pdf": "978-1-100-25774-7",
}
SHORT_CITE = "CBH 2nd Ed."

HEADING_SIZES = {13.5: 2, 12.0: 3, 10.0: 4}
SEC_NUM_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.*)$")
CAPTION_RE = re.compile(r"^(Table|Figure)\s+(\d+\s*[–-]\s*\d+[a-z]?)\s*:?\s*(.*)$")
REG_REF_RE = re.compile(
    r"\b(CBS Matrix \d+(?:\.\d+)*|(?:HPTA|HPTR|HAA|HAR)\s+\d+(?:\(\d+\))*(?:\([a-z]\))?)"
)
XREF_RE = re.compile(r"\b(Chapter \d+|Section \d+(?:\.\d+)+|(?:Table|Figure) \d+[–-]\d+[a-z]?)")
MODAL_RE = re.compile(r"\b(must|should|may|is required|are required)\b", re.I)

SKIP_PDF_PAGES = set(range(1, 11))  # 封面、版權、目錄、圖表目錄
INDEX_START_PDF = 359               # 索引頁（不適合當知識來源）
TARGET_WORDS = 280                  # 約 380 tokens
OVERLAP_WORDS = 40


# ---------------------------------------------------------------- 1. 讀取版面
def is_super(span):
    return span["size"] <= 5.5


def page_lines(page):
    """把 span 合併成「行」，上標數字拆出成 citation refs，頁首頁尾分離。"""
    printed = None
    raw = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip() or s["text"] == " "]
            if not spans:
                continue
            y0 = l["bbox"][1]
            if y0 >= 590:
                if spans[0]["font"] == "Futura-Bold":
                    printed = "".join(s["text"] for s in spans).strip()
                continue
            if y0 < 45 and spans[0]["size"] == 11.0:
                continue  # 頁首 running header
            fonts = {s["font"] for s in spans}
            if all(f.startswith(("Arial", "Calibri", "FuturaBT")) for f in fonts):
                continue  # 圖片內文字
            raw.append({"bbox": list(l["bbox"]), "spans": spans})

    # 同一 y 的行合併（bullet「•」和文字是分開的 line）
    # 先依 y 分群（|dy|<3 視為同一列），列內依 x 排序；水平間距 <40pt 才合併（表格欄位不合併）
    raw.sort(key=lambda r: r["bbox"][1])
    rows = []
    for r in raw:
        if rows and abs(rows[-1][0]["bbox"][1] - r["bbox"][1]) < 3 and not is_super(r["spans"][0]):
            rows[-1].append(r)
        elif rows and is_super(r["spans"][0]) and r["bbox"][1] - rows[-1][0]["bbox"][1] < 4:
            rows[-1].append(r)  # 上標的 y 稍高
        else:
            rows.append([r])
    merged = []
    for row in rows:
        row.sort(key=lambda r: r["bbox"][0])
        cur = None
        for r in row:
            gap = r["bbox"][0] - cur["bbox"][2] if cur else None
            marker = cur and re.fullmatch(r"[•–-]|\(?[a-z0-9]{1,3}[.)]|\d+(?:\.\d+)+", "".join(s["text"] for s in cur["spans"]).strip())
            if cur and (gap < 6 or (marker and gap < 40) or is_super(r["spans"][0])):
                cur["spans"] += r["spans"]
                cur["bbox"] = [min(cur["bbox"][0], r["bbox"][0]), min(cur["bbox"][1], r["bbox"][1]),
                               max(cur["bbox"][2], r["bbox"][2]), max(cur["bbox"][3], r["bbox"][3])]
            else:
                cur = {"bbox": list(r["bbox"]), "spans": list(r["spans"])}
                merged.append(cur)

    lines = []
    for m in merged:
        text, cites, bold = "", [], []
        main = next((s for s in m["spans"] if not is_super(s)), m["spans"][0])
        for s in m["spans"]:
            t = s["text"]
            if is_super(s) and s is not main:
                if re.fullmatch(r"[\d,\s–-]+", t.strip()):
                    cites.append(t.strip())
                    continue
                text += t  # 序數 nd/st/th
                continue
            if s["font"] == "Futura-Book" and s["size"] == 9.0 and t.strip():
                bold.append(t.strip())
            text += t
        text = re.sub(r"[\t\xa0 ]+", " ", text).strip()
        if not text:
            continue
        lines.append({
            "text": text, "bbox": [round(v, 1) for v in m["bbox"]],
            "size": round(main["size"], 1), "font": main["font"],
            "cites": cites, "bold": bold,
        })
    lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return lines, printed


def expand_cites(cites):
    out = []
    for c in cites:
        for part in c.split(","):
            part = part.strip()
            if re.fullmatch(r"\d+[–-]\d+", part):
                a, b = map(int, re.split(r"[–-]", part))
                out += list(range(a, b + 1))
            elif part.isdigit():
                out.append(int(part))
    return out


# ---------------------------------------------------------------- 2. 結構解析
class Parser:
    def __init__(self, pdf_path):
        self.doc = pymupdf.open(pdf_path)
        self.blocks = []       # 段落/表格/詞彙等「原子單位」，之後再切 chunk
        self.sections = {}     # section_id -> info
        self.references = defaultdict(dict)  # chapter -> {n: text}
        self.chapter = None    # {"no", "title"}
        self.stack = []        # [(level, id, title)]
        self.mode = "body"     # body | references | table | glossary | abbrev
        self.para = None
        self.table = None
        self.pending_heading = None
        self.ref_cur = None

    # ---- 共同工具
    def section_ctx(self):
        if not self.chapter:
            return {"chapter_no": None, "chapter_title": None, "section_id": "front",
                    "section_title": "Front matter", "section_path": ["Front matter"]}
        path = [f"Chapter {self.chapter['no']} – {self.chapter['title']}"] + \
               [f"{sid} {t}" for _, sid, t in self.stack]
        sid, stitle = (self.stack[-1][1], self.stack[-1][2]) if self.stack else \
            (f"ch{self.chapter['no']}", self.chapter["title"])
        return {"chapter_no": self.chapter["no"], "chapter_title": self.chapter["title"],
                "section_id": sid, "section_title": stitle, "section_path": path}

    def loc(self, pno, printed, bbox):
        return {"pdf_page": pno, "printed_page": printed, "bbox": bbox}

    def flush_para(self):
        if self.para and self.para["text"].strip():
            self.blocks.append(self.para)
        self.para = None

    def flush_table(self):
        if self.table:
            self.blocks.append(build_table(self.table))
        self.table = None

    def new_block(self, btype, pno, printed, line, **extra):
        b = {"type": btype, **self.section_ctx(), "text": line["text"],
             "locs": [self.loc(pno, printed, line["bbox"])],
             "cites": list(line["cites"]), "bold": list(line["bold"]),
             "_last_y": line["bbox"][3], "_pno": pno}
        b.update(extra)
        return b

    def add_to(self, b, pno, printed, line, sep=" "):
        if b["text"].endswith("-") and not b["text"].endswith(" -"):
            b["text"] += line["text"]  # 行尾連字號：保留連字號直接接（non-enveloped），避免把複合詞黏成一個字
        else:
            b["text"] += sep + line["text"]
        last = b["locs"][-1]
        if last["pdf_page"] == pno:
            bb = last["bbox"]
            last["bbox"] = [min(bb[0], line["bbox"][0]), min(bb[1], line["bbox"][1]),
                            max(bb[2], line["bbox"][2]), max(bb[3], line["bbox"][3])]
        else:
            b["locs"].append(self.loc(pno, printed, line["bbox"]))
        b["cites"] += line["cites"]
        b["bold"] += line["bold"]
        b["_last_y"], b["_pno"] = line["bbox"][3], pno

    # ---- 主流程
    def run(self):
        for i, page in enumerate(self.doc):
            pno = i + 1
            if pno in SKIP_PDF_PAGES or pno >= INDEX_START_PDF:
                continue
            lines, printed = page_lines(page)
            if not lines:
                continue
            margin = 35 if printed is None or not printed.isdigit() or int(printed) % 2 == 0 else 53
            if pno in (11, 13):  # Preface / Abbreviations 分隔頁
                continue
            if any(l["size"] == 22.0 and l["font"] == "Futura-Book" for l in lines):
                continue  # 章節分隔頁（大字章名），章名從下一頁的章首標題取
            self.handle_page(pno, printed, lines, margin)
        self.flush_para()
        self.flush_table()
        return self

    def start_chapter(self, title_line):
        self.flush_para()
        self.flush_table()
        m = re.match(r"^(chapter|appendix)\s+(\w+)\s*–?\s*(.*)$", title_line, re.I)
        if m:
            no, title = m.group(2), m.group(3).strip()
            if m.group(1).lower() == "appendix":
                no = f"App-{no}"
        else:  # PREFACE / ABBREVIATIONS AND ACRONYMS
            title = title_line.strip()
            no = title.split()[0].capitalize()
        self.chapter = {"no": no, "title": title}
        self.stack = []
        self.mode = {"24": "glossary", "Abbreviations": "abbrev"}.get(no, "body")
        self.sections[f"ch{self.chapter['no']}"] = {"level": 1, "title": title}

    def handle_page(self, pno, printed, lines, margin):
        # 章首標題（16pt，可能跨兩行）
        head = [l for l in lines if l["size"] == 16.0]
        if head:
            self.start_chapter(" ".join(l["text"] for l in head))
            lines = [l for l in lines if l["size"] != 16.0]
            self.sections[f"ch{self.chapter['no']}"]["printed_page"] = printed
            self.sections[f"ch{self.chapter['no']}"]["pdf_page"] = pno

        if self.mode in ("glossary", "abbrev"):
            self.handle_two_column(pno, printed, lines, margin)
            return

        prev = None
        for ln in lines:
            t = ln["text"]
            # --- 節標題
            lvl = HEADING_SIZES.get(ln["size"]) if ln["font"] == "Futura-Book" else None
            m = SEC_NUM_RE.match(t) if lvl else None
            if m:
                self.flush_para()
                self.flush_table()
                self.mode = "body"
                sid, title = m.group(1), m.group(2)
                self.stack = [s for s in self.stack if s[0] < lvl] + [(lvl, sid, title)]
                self.sections[sid] = {"level": lvl, "title": title, "pdf_page": pno,
                                      "printed_page": printed}
                self.pending_heading = (ln["bbox"][3], ln["size"])
                prev = ln
                continue
            if lvl and self.pending_heading and ln["size"] == self.pending_heading[1] and \
                    ln["bbox"][1] - self.pending_heading[0] < 6 and t != "REFERENCES":
                # 標題換行
                l2, sid, title = self.stack[-1]
                self.stack[-1] = (l2, sid, title + " " + t)
                self.sections[sid]["title"] = self.stack[-1][2]
                self.pending_heading = (ln["bbox"][3], ln["size"])
                continue
            self.pending_heading = None
            if t == "REFERENCES" and ln["font"] == "Futura-Book":
                self.flush_para()
                self.flush_table()
                self.mode = "references"
                self.ref_cur = None
                continue

            # --- 參考文獻
            if self.mode == "references":
                # 文獻編號是 4.7pt，被 page_lines 當成上標拆進 cites；行首的才是編號
                if ln["cites"] and ln["cites"][0].isdigit() and ln["bbox"][0] <= margin + 2:
                    self.ref_cur = int(ln["cites"][0])
                    self.references[self.chapter["no"]][self.ref_cur] = t
                    continue
                if ln["size"] <= 5.5 or re.match(r"^\d+\s", t):
                    mm = re.match(r"^(\d+)\s*(.*)$", t)
                    if mm:
                        self.ref_cur = int(mm.group(1))
                        self.references[self.chapter["no"]][self.ref_cur] = mm.group(2)
                        continue
                if self.ref_cur is not None:
                    r = self.references[self.chapter["no"]]
                    r[self.ref_cur] = (r[self.ref_cur] + " " + t).strip()
                continue

            # --- 表格 / 圖說
            cm = CAPTION_RE.match(t)
            if cm and ln["font"] == "Futura-Book":
                self.flush_para()
                self.flush_table()
                kind, num, cap = cm.group(1), cm.group(2).replace(" ", "").replace("-", "–"), cm.group(3)
                if kind == "Table":
                    self.table = {**self.section_ctx(), "table_id": f"Table {num}",
                                  "caption": cap, "lines": [], "locs": [],
                                  "cites": list(ln["cites"]), "margin": margin,
                                  "_cap_y": ln["bbox"][3], "_pno": pno, "_printed": printed}
                    self.mode = "table"
                    self.table["locs"].append(self.loc(pno, printed, ln["bbox"]))
                else:
                    self.para = self.new_block("figure", pno, printed, ln,
                                               figure_id=f"Figure {num}")
                    self.para["text"] = f"[Figure {num}] {cap}".strip()
                    self.para["bold"] = []  # 圖說本身是粗體字型，不是詞彙表術語
                continue

            if self.mode == "table":
                tb = self.table
                if not tb["lines"] and tb["_pno"] == pno and ln["bbox"][1] - tb["_cap_y"] < 6:
                    if ln["font"] == "Futura-Book":  # 表名換行
                        tb["caption"] = (tb["caption"] + " " + t).strip()
                    else:                            # 表名下方的出處註記，例如 "Reproduced from the HPTR 10(2)."
                        tb["source_note"] = (tb.get("source_note", "") + " " + t).strip()
                    tb["cites"] += ln["cites"]
                    tb["_cap_y"] = ln["bbox"][3]
                    continue
                is_body = ln["size"] == 9.0 and abs(ln["bbox"][0] - margin) < 2 and \
                    (ln["bbox"][2] - ln["bbox"][0]) > 250 and ln["font"].startswith("Futura-Light") \
                    and (tb["_pno"] != pno or ln["bbox"][1] - tb["_cap_y"] > 30)
                if not is_body:
                    if self.table["_pno"] != pno:
                        self.table["locs"].append(self.loc(pno, printed, ln["bbox"]))
                        self.table["_pno"] = pno
                    else:
                        lb = self.table["locs"][-1]["bbox"]
                        self.table["locs"][-1]["bbox"] = [min(lb[0], ln["bbox"][0]), min(lb[1], ln["bbox"][1]),
                                                          max(lb[2], ln["bbox"][2]), max(lb[3], ln["bbox"][3])]
                    self.table["lines"].append({**ln, "pdf_page": pno})
                    self.table["cites"] += ln["cites"]
                    continue
                self.flush_table()
                self.mode = "body"

            # --- 圖說換行
            if self.para and self.para["type"] == "figure":
                if self.para["_pno"] == pno and ln["bbox"][1] - self.para["_last_y"] < 6:
                    self.add_to(self.para, pno, printed, ln)
                    continue
                self.flush_para()

            # --- 內文段落
            is_bullet = t.startswith("•") or re.match(r"^[a-z0-9]{1,2}\.\s|^\d+\.\s", t)
            if is_bullet and t.startswith("•"):
                ln = {**ln, "text": "- " + t.lstrip("• ").strip()}
            new_para = (self.para is None or is_bullet or self.para["_pno"] != pno and False or
                        (self.para["_pno"] == pno and ln["bbox"][1] - self.para["_last_y"] > 6))
            if self.para is not None and self.para["_pno"] != pno:
                # 跨頁：上一頁最後一段若沒有句號結尾，視為同一段延續
                new_para = is_bullet or self.para["text"].rstrip().endswith((".", ":", ";"))
            if new_para:
                self.flush_para()
                self.para = self.new_block("list_item" if is_bullet else "paragraph",
                                           pno, printed, ln)
            else:
                self.add_to(self.para, pno, printed, ln)
            prev = ln

    def handle_two_column(self, pno, printed, lines, margin):
        """Glossary / Abbreviations：左欄詞條、右欄定義；詞條垂直置中於定義段落。"""
        if self.mode == "abbrev":
            split = margin + 45
        else:
            split = margin + 70
        # 章首導言（滿版文字）在 glossary 第一頁
        left = [l for l in lines if l["bbox"][0] < split and l["bbox"][2] < split + 50]
        right = [l for l in lines if l["bbox"][0] >= split]
        intro = [l for l in lines if l not in left and l not in right]
        for l in intro:
            if self.para is None:
                self.para = self.new_block("paragraph", pno, printed, l)
            else:
                self.add_to(self.para, pno, printed, l)
        self.flush_para()
        # 定義段落
        defs = []
        for l in right:
            if defs and l["bbox"][1] - defs[-1]["y1"] < 6:
                defs[-1]["lines"].append(l)
                defs[-1]["y1"] = l["bbox"][3]
            else:
                defs.append({"y0": l["bbox"][1], "y1": l["bbox"][3], "lines": [l]})
        # 詞條（多行合併）
        terms = []
        for l in left:
            if terms and l["bbox"][1] - terms[-1]["y1"] < 4:
                terms[-1]["text"] += " " + l["text"]
                terms[-1]["y1"] = l["bbox"][3]
                terms[-1]["bbox"][2] = max(terms[-1]["bbox"][2], l["bbox"][2])
            else:
                terms.append({"text": l["text"], "y0": l["bbox"][1], "y1": l["bbox"][3],
                              "bbox": list(l["bbox"])})
        btype = "abbreviation" if self.mode == "abbrev" else "glossary_term"
        for d in defs:
            cy = (d["y0"] + d["y1"]) / 2
            cand = [t for t in terms if t["y0"] - 4 <= cy <= t["y1"] + 4 or d["y0"] - 2 <= t["y0"] <= d["y1"]]
            dtext = " ".join(x["text"] for x in d["lines"])
            bbox = [min(x["bbox"][0] for x in d["lines"]), d["y0"],
                    max(x["bbox"][2] for x in d["lines"]), d["y1"]]
            if cand:
                t = min(cand, key=lambda t: abs((t["y0"] + t["y1"]) / 2 - cy))
                terms.remove(t)
                bbox = [min(bbox[0], t["bbox"][0]), min(bbox[1], t["y0"]), bbox[2], max(bbox[3], t["y1"])]
                b = {"type": btype, **self.section_ctx(), "term": t["text"],
                     "text": f"{t['text']}: {dtext}", "locs": [self.loc(pno, printed, [round(v, 1) for v in bbox])],
                     "cites": [c for x in d["lines"] for c in x["cites"]], "bold": [],
                     "_last_y": d["y1"], "_pno": pno}
                self.blocks.append(b)
            elif self.blocks and self.blocks[-1]["type"] == btype:
                prev = self.blocks[-1]  # 跨頁的定義延續
                prev["text"] += " " + dtext
                prev["locs"].append(self.loc(pno, printed, [round(v, 1) for v in bbox]))
        for t in terms:  # 剩下沒配對到的（例如縮寫 HPTA Security Clearance 多行）
            if self.blocks and self.blocks[-1]["type"] == btype:
                self.blocks[-1]["term"] += " " + t["text"]
                self.blocks[-1]["text"] = self.blocks[-1]["term"] + ": " + \
                    self.blocks[-1]["text"].split(": ", 1)[1]


# ---------------------------------------------------------------- 3. 表格重建
def build_table(tb):
    """無框線表格重建：
    - 表頭 = 表格第一頁、第一個內文 cell 之上的 8pt Futura-Medium 行（可多層、含直排字）
    - 欄位 = 表頭「葉節點」（其 x 範圍內沒有更下層表頭）的中心點；cell 指派到最近的欄
    - 列 = 內文行依垂直空白切帶（band）；8pt Medium 的列標籤（可能跨列置中）指派到最近的帶
    複雜表格（直排表頭、跨列合併格）會標記 needs_review，並另存表格截圖供人工/VLM 核對。
    """
    lines = tb["lines"]
    first_pg = lines[0]["pdf_page"] if lines else None
    light_first = [l for l in lines if l["pdf_page"] == first_pg and l["font"] != "Futura-Medium"]
    y_body0 = min((l["bbox"][1] for l in light_first), default=1e9)
    is_med = lambda l: l["font"] == "Futura-Medium" and l["size"] == 8.0
    header = [l for l in lines if is_med(l) and l["pdf_page"] == first_pg and l["bbox"][3] <= y_body0 + 1]
    # 續頁重複的表頭：與首頁表頭文字相同的 Medium 行
    htexts = {l["text"] for l in header}
    rest = [l for l in lines if l not in header and not (is_med(l) and l["pdf_page"] != first_pg
                                                         and l["text"] in htexts)]
    notes = [l for l in rest if l["size"] < 8.5 and not is_med(l)]
    rest = [l for l in rest if l not in notes]
    labels = [l for l in rest if is_med(l)]
    cells = [l for l in rest if not is_med(l)]

    cx = lambda l: (l["bbox"][0] + l["bbox"][2]) / 2
    leaves = [h for h in header if not any(o is not h and o["bbox"][1] > h["bbox"][1] + 2 and
                                           h["bbox"][0] - 2 <= cx(o) <= h["bbox"][2] + 2 for o in header)]
    cols = []
    for h in sorted(leaves, key=cx):
        if cols and cx(h) - cols[-1]["x"] < 12:
            cols[-1]["members"].append(h)
        else:
            cols.append({"x": cx(h), "members": [h]})
    # 左側無表頭的列標籤欄（例如 Table 11–1 左上角空白格）
    left_cells = [l for l in cells if leaves and l["bbox"][2] < min(h["bbox"][0] for h in leaves) - 1]
    if left_cells and not labels:
        cols.insert(0, {"x": sorted(cx(l) for l in left_cells)[len(left_cells) // 2], "members": []})
    elif labels and (not cols or min(cx(l) for l in labels) < cols[0]["x"] - 20):
        cols.insert(0, {"x": min(cx(l) for l in labels), "members": []})
    if not cols:
        cols = [{"x": cx(l), "members": []} for l in sorted(cells, key=cx)[:1]]
    hdr = []
    for c in cols:
        hs = [h for h in header if h["bbox"][0] - 2 <= c["x"] <= h["bbox"][2] + 2]
        hdr.append(" ".join(h["text"] for h in sorted(hs, key=lambda h: h["bbox"][1])))

    def col_of(l):
        return min(range(len(cols)), key=lambda i: abs(cols[i]["x"] - cx(l)))

    # 換行的列標籤先合併（例如 "Highly" / "resistant"）
    merged_labels = []
    for lb in sorted(labels, key=lambda l: (l["pdf_page"], l["bbox"][1])):
        m = merged_labels[-1] if merged_labels else None
        if m and m["pdf_page"] == lb["pdf_page"] and lb["bbox"][1] - m["bbox"][3] < 3 and \
                abs(cx(lb) - cx(m)) < 30:
            m["text"] += " " + lb["text"]
            m["bbox"] = [min(m["bbox"][0], lb["bbox"][0]), m["bbox"][1],
                         max(m["bbox"][2], lb["bbox"][2]), lb["bbox"][3]]
        else:
            merged_labels.append({**lb, "bbox": list(lb["bbox"])})
    labels = merged_labels

    bands = []
    for l in sorted(cells, key=lambda l: (l["pdf_page"], l["bbox"][1])):
        if bands and bands[-1]["page"] == l["pdf_page"] and l["bbox"][1] < bands[-1]["y1"] + 4:
            bands[-1]["lines"].append(l)
            bands[-1]["y1"] = max(bands[-1]["y1"], l["bbox"][3])
        else:
            bands.append({"page": l["pdf_page"], "y0": l["bbox"][1], "y1": l["bbox"][3], "lines": [l]})
    for lb in labels:
        cand = [b for b in bands if b["page"] == lb["pdf_page"]]
        if cand:
            ly = (lb["bbox"][1] + lb["bbox"][3]) / 2
            min(cand, key=lambda b: abs((b["y0"] + b["y1"]) / 2 - ly))["lines"].append(lb)
    rows = []
    for b in bands:
        r = [""] * len(cols)
        for l in sorted(b["lines"], key=lambda l: l["bbox"][1]):
            c = 0 if is_med(l) else col_of(l)
            r[c] = (r[c] + " " + l["text"]).strip()
        rows.append(r)
    rotated = any((h["bbox"][3] - h["bbox"][1]) > 20 for h in header)
    suspicious = rotated or len(header) > len(cols) * 2 or \
        any(not c for r in rows for c in r) or \
        any("•" in r[0] for r in rows if len(r) > 1) or \
        any(r[0] and any(lb["text"] in c for c in r[1:]) for r in rows for lb in labels)
    quality = "needs_review" if suspicious else "ok"
    md = []
    if any(hdr):
        md.append("| " + " | ".join(hdr) + " |")
        md.append("|" + "---|" * len(hdr))
    for r in rows:
        md.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
    note_text = " ".join(n["text"] for n in notes)
    src = tb.get("source_note", "")
    text = f"[{tb['table_id']}] {tb['caption']}" + (f" ({src})" if src else "") + "\n" + \
        "\n".join(md) + (f"\nNote: {note_text}" if note_text else "")
    return {"type": "table", **{k: tb[k] for k in ("chapter_no", "chapter_title", "section_id",
                                                   "section_title", "section_path")},
            "table_id": tb["table_id"], "caption": tb["caption"], "source_note": src,
            "header": hdr, "rows": rows, "quality": quality,
            "text": text, "locs": tb["locs"], "cites": tb["cites"], "bold": []}


# ---------------------------------------------------------------- 4. 切 chunk + metadata
def page_range(locs):
    pdf = [l["pdf_page"] for l in locs]
    pr = [l["printed_page"] for l in locs if l["printed_page"]]
    return min(pdf), max(pdf), (pr[0] if pr else None), (pr[-1] if pr else None)


def citation_label(c):
    sec = c["section_id"]
    sec_txt = f"§{sec}" if re.match(r"^\d", sec) else (f"Ch.{c['chapter_no']}" if c["chapter_no"] else "Front matter")
    p = c["printed_page_start"]
    p2 = c["printed_page_end"]
    pages = f"p.{p}" if p == p2 else f"pp.{p}–{p2}"
    pdf = f"PDF p.{c['pdf_page_start']}" if c["pdf_page_start"] == c["pdf_page_end"] else \
        f"PDF pp.{c['pdf_page_start']}–{c['pdf_page_end']}"
    extra = f", {c['table_id']}" if c.get("table_id") else (f", term “{c['term']}”" if c.get("term") else "")
    return f"{SHORT_CITE} {sec_txt}{extra}, {pages} ({pdf})"


def make_chunks(blocks, references):
    chunks = []
    ordinal = defaultdict(int)

    def emit(group, ctype, extra=None):
        b0 = group[0]
        text = "\n".join(b["text"] for b in group).strip()
        locs = [l for b in group for l in b["locs"]]
        p0, p1, pp0, pp1 = page_range(locs)
        cites = sorted(set(expand_cites([c for b in group for c in b["cites"]])))
        sid = b0["section_id"]
        ordinal[sid] += 1
        n = ordinal[sid]
        chap_refs = references.get(b0["chapter_no"] or "", {})
        c = {
            "chunk_id": f"{DOC_ID}:{sid}:{n:03d}",
            "chunk_hash": hashlib.sha1(text.encode()).hexdigest()[:12],
            **DOC_META,
            "chunk_type": ctype,
            "chapter_no": b0["chapter_no"], "chapter_title": b0["chapter_title"],
            "section_id": sid, "section_title": b0["section_title"],
            "section_path": b0["section_path"],
            "pdf_page_start": p0, "pdf_page_end": p1,
            "printed_page_start": pp0, "printed_page_end": pp1,
            "locations": locs,
            "text": text,
            "embed_text": " > ".join(b0["section_path"]) + "\n\n" + text,
            "word_count": len(text.split()),
            "citation_refs": cites,
            "cited_sources": [{"n": i, "text": chap_refs.get(i)} for i in cites if chap_refs.get(i)],
            "regulatory_refs": sorted(set(REG_REF_RE.findall(text))),
            "cross_refs": sorted(set(x.replace("-", "–") for x in XREF_RE.findall(text))),
            "defined_terms_used": sorted(set(t.lower().strip(",.;:()") for b in group for t in b["bold"] if len(t) > 2)),
            "modal_verbs": sorted(set(m.lower() for m in MODAL_RE.findall(text))),
            **(extra or {}),
        }
        c["citation_label"] = citation_label(c)
        chunks.append(c)

    # 依 section 分組，不跨節
    groups, cur = [], []
    for b in blocks:
        if cur and (b["section_id"] != cur[-1]["section_id"] or b["type"] in
                    ("table", "glossary_term", "abbreviation", "figure") or
                    cur[-1]["type"] in ("table", "glossary_term", "abbreviation", "figure")):
            groups.append(cur)
            cur = []
        cur.append(b)
    if cur:
        groups.append(cur)

    for g in groups:
        t = g[0]["type"]
        if t == "table":
            emit(g, "table", {"table_id": g[0]["table_id"], "table_caption": g[0]["caption"],
                              "table_source_note": g[0]["source_note"], "table_quality": g[0]["quality"],
                              "table_header": g[0]["header"], "table_rows": g[0]["rows"]})
            continue
        if t in ("glossary_term", "abbreviation"):
            emit(g, t, {"term": g[0]["term"]})
            continue
        if t == "figure":
            emit(g, "figure_caption", {"figure_id": g[0]["figure_id"]})
            continue
        ctype = "resource_list" if g[0]["chapter_no"] == "25" else "body"
        # 段落累積到 TARGET_WORDS；超長段落以句子切開；重疊上一 chunk 尾端句子
        units = []
        for b in g:
            words = len(b["text"].split())
            if words <= TARGET_WORDS * 1.3:
                units.append(b)
                continue
            sents = re.split(r"(?<=[.;])\s+(?=[A-Z(])", b["text"])
            buf = ""
            for s in sents:
                if buf and len((buf + " " + s).split()) > TARGET_WORDS:
                    units.append({**b, "text": buf})
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                units.append({**b, "text": buf})
        acc, wc = [], 0
        for u in units:
            w = len(u["text"].split())
            if acc and wc + w > TARGET_WORDS:
                emit(acc, ctype)
                tail = " ".join(acc[-1]["text"].split()[-OVERLAP_WORDS:])
                ov = {**acc[-1], "text": "…" + tail, "cites": [], "bold": [],
                      "locs": acc[-1]["locs"][-1:]}
                acc, wc = [ov], OVERLAP_WORDS
            acc.append(u)
            wc += w
        if acc:
            emit(acc, ctype)
    return chunks


# ---------------------------------------------------------------- 5. 以目錄驗證
def parse_toc(doc):
    entries = []
    txt = "\n".join(doc[i].get_text() for i in range(2, 8))
    txt = re.sub(r"[\t\xa0]", " ", txt)
    for m in re.finditer(r"^\s*(\d+(?:\.\d+)+)\s*\n?\s*(.+?)(?:\s*\.\s*){2,}\s*(\d+)\s*$", txt, re.M | re.S):
        title = re.sub(r"\s+", " ", m.group(2)).strip(" .")
        if "\n" in m.group(0).strip().split("\n", 1)[-1] and len(title) > 120:
            continue
        entries.append({"section_id": m.group(1), "title": title, "printed_page": m.group(3)})
    return entries


def validate(doc, sections, chunks, refs):
    toc = parse_toc(doc)
    missing, page_mismatch = [], []
    for e in toc:
        s = sections.get(e["section_id"])
        if not s:
            missing.append(e)
        elif str(s.get("printed_page")) != e["printed_page"]:
            page_mismatch.append({**e, "found_printed_page": s.get("printed_page")})
    # 文字覆蓋率：輸出的字 vs 原始內文頁的字（粗估遺漏）
    import collections
    src = collections.Counter()
    for i in range(10, INDEX_START_PDF - 1):
        src.update(w.lower() for w in re.findall(r"[A-Za-z]{4,}", doc[i].get_text()))
    out = collections.Counter()
    for r in refs.values():
        for t in r.values():
            out.update(w.lower() for w in re.findall(r"[A-Za-z]{4,}", t))
    for c in chunks:
        if c["text"].startswith("…"):
            pass
        out.update(w.lower() for w in re.findall(r"[A-Za-z]{4,}", c["text"]))
    covered = sum(min(src[w], out[w]) for w in src) / max(1, sum(src.values()))
    types = collections.Counter(c["chunk_type"] for c in chunks)
    return {
        "toc_entries": len(toc), "sections_found": len([k for k in sections if not k.startswith("ch")]),
        "toc_missing": missing, "toc_page_mismatch": page_mismatch,
        "word_coverage_vs_source": round(covered, 4),
        "chunks_total": len(chunks), "chunks_by_type": dict(types),
        "chunks_without_printed_page": [c["chunk_id"] for c in chunks if not c["printed_page_start"]],
        "max_words": max(c["word_count"] for c in chunks),
    }


def main(pdf, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = Parser(pdf).run()
    refs = {ch: dict(sorted(r.items())) for ch, r in p.references.items()}
    chunks = make_chunks(p.blocks, refs)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "sections.json").write_text(json.dumps(p.sections, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    tables = [{k: c[k] for k in ("chunk_id", "table_id", "table_caption", "table_header", "table_rows", "table_quality",
                                 "section_id", "pdf_page_start", "pdf_page_end", "printed_page_start",
                                 "citation_label")} for c in chunks if c["chunk_type"] == "table"]
    crop_dir = out / "table_images"
    crop_dir.mkdir(exist_ok=True)
    for t in tables:
        c = next(c for c in chunks if c["chunk_id"] == t["chunk_id"])
        t["images"] = []
        for loc in c["locations"]:
            pg = p.doc[loc["pdf_page"] - 1]
            x0, y0, x1, y1 = loc["bbox"]
            clip = pymupdf.Rect(max(0, x0 - 8), max(0, y0 - 8), min(pg.rect.x1, x1 + 8), min(pg.rect.y1, y1 + 8))
            fn = f"{t['table_id'].replace(' ', '_').replace('–', '-')}_pdfp{loc['pdf_page']}.png"
            pg.get_pixmap(clip=clip, dpi=150).save(crop_dir / fn)
            t["images"].append(f"table_images/{fn}")
    (out / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = validate(p.doc, p.sections, chunks, refs)
    (out / "validation_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rag_output")
