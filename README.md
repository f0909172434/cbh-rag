# CBH 2nd Ed. → RAG 結構化資料（含來源追溯）

來源：*Canadian Biosafety Handbook (CBH), 2nd Edition, 2016*（PHAC/CFIA），PDF 共 366 頁。

把一份法規類 PDF 做成**可追溯來源、可驗證引用**的 RAG：依版面結構切塊、每個 chunk 帶章節／PDF 頁碼／印刷頁碼／bbox，混合檢索（多語系向量 + BM25），LLM 以結構化格式回答並附逐字引述，最後由程式驗證引述確實存在於原文。

## 授權與資料

- **程式碼**：MIT License（見 `LICENSE`）。
- **CBH PDF 與其衍生資料不在此 repo**：手冊載明僅可「個人或內部使用」重製。因此 chunks、逐頁文字、表格截圖、證據集、向量索引等含原文或頁面影像的檔案都沒有公開，請自行下載 PDF 後用腳本重新產生。repo 內只附不含原文的結果（`validation_report.json`、`rag_output/course/` 的評測與比較數據）。
- 手冊是 2016 年的指引，不等於現行法規要求。

## 快速開始

```bash
pip install -r requirements.txt
curl -LO https://www.canada.ca/content/dam/phac-aspc/migration/cbsg-nldcb/cbh-gcb/assets/pdf/cbh-gcb-eng.pdf
echo "OPENROUTER_API_KEY=你的key" > .env          # https://openrouter.ai/keys

python parse_cbh.py cbh-gcb-eng.pdf rag_output         # 1. PDF → chunks（免費、離線）
python openrouter_enrich.py cbh-gcb-eng.pdf            # 2. 選用：VLM 重建複雜表格、描述圖（約 US$0.5）
python rag_langchain.py build                          # 3. 建索引（< US$0.01）
python rag_langchain.py ask "BSC 要多久認證一次？"        # 4. 問答（每題約 US$0.01–0.03）
python course_pipeline.py cbh-gcb-eng.pdf              # 5. 選用：課程 M1/M2 交付物與評測
```

以下說明中提到的 `rag_output/` 檔案都是執行上述步驟後才會產生的。

## 檔案

| 檔案 | 內容 |
|---|---|
| `parse_cbh.py` | 解析器：PDF → 結構化 chunks（可重跑） |
| `trace_tools.py` | 來源追溯：`verify` 檢查 LLM 引用、`show` 在原 PDF 頁面框出位置 |
| `rag_output/chunks.jsonl` | **主要成品**：840 個 chunk，每行一個 JSON，可直接灌進向量庫 |
| `rag_output/sections.json` | 完整章節樹（331 節，含各節 pdf/印刷頁碼） |
| `rag_output/references.json` | 各章參考文獻（`{章: {編號: 書目}}`） |
| `rag_output/tables.json` | 6 張表格的結構化 header + rows，附截圖路徑 |
| `rag_output/table_images/` | 每張表格的原始 PDF 截圖（人工或 VLM 核對用） |
| `rag_output/validation_report.json` | 自動驗證結果 |
| `course_pipeline.py` / `rag_output/course/` | 對齊課程 M1/M2 的交付物（見下方對照表） |

## 方法選擇

| 方法 | 適合 | 這份 PDF 的情況 |
|---|---|---|
| **版面規則解析（本專案採用，PyMuPDF）** | 文字型 PDF、排版一致 | ✅ 沒有書籤 outline，但字型極一致：13.5/12/10pt Futura-Book = 第 2/3/4 層標題、4.7pt = 文獻編號、y≈600 = 印刷頁碼。完全可重現、可驗證 |
| Docling / Unstructured / Marker | 版面多樣、大量異質文件 | 可用，但通用模型抓不到這份文件的細節（上標引用編號、詞彙表兩欄配對、印刷頁碼） |
| Vision LLM（Claude / GPT-4o 看頁面圖） | 掃描檔、複雜表格、流程圖 | 只建議用在 4 張 `needs_review` 表格與 28 張圖 |
| 單純逐頁 / 固定長度切塊 | 快速原型 | ❌ 會切斷章節，來源無法精準指到哪一節 |

## chunk schema（追溯用欄位）

```jsonc
{
  "chunk_id": "CBH-2E-2016:4.3.8.4:001",   // 穩定 ID：文件:節:序號（LLM 引用就用這個）
  "chunk_hash": "f9bc3b1e8aa4",            // 內容 SHA1，可偵測原文是否變動
  "doc_id": "CBH-2E-2016", "doc_title": "...", "publisher": "...", "pub_date": "2016-03", "isbn_pdf": "...",
  "chunk_type": "body | table | glossary_term | abbreviation | figure_caption | resource_list",
  "chapter_no": "4", "chapter_title": "RISK FACTORS, ...",
  "section_id": "4.3.8.4", "section_title": "Synthetic Biological Devices and Systems",
  "section_path": ["Chapter 4 – ...", "4.3 Special Considerations", "4.3.8 Biotechnology", "4.3.8.4 ..."],
  "pdf_page_start": 65, "pdf_page_end": 65,          // PDF 實體頁（1-based）
  "printed_page_start": "47", "printed_page_end": "47", // 書上印的頁碼（從頁尾讀，不是用偏移推算）
  "locations": [{"pdf_page": 65, "printed_page": "47", "bbox": [x0,y0,x1,y1]}], // 可在原頁畫框的座標
  "text": "...",                // 原文（已移除上標引用數字）
  "embed_text": "Chapter 4 – ... > 4.3 ... > 4.3.8.4 ...\n\n...", // **embedding 用這個**（帶章節路徑）
  "citation_refs": [2, 3, 5],   // 內文上標引用編號
  "cited_sources": [{"n": 2, "text": "Madigan, M. T. ... (2010) ..."}], // 對應到的參考文獻（二階追溯）
  "regulatory_refs": ["CBS Matrix 4.10", "HPTA 33"], // 引用的法規/標準條文
  "cross_refs": ["Chapter 6", "Table 11–1"],          // 文件內交互參照
  "defined_terms_used": ["synthetic biology"],        // 內文粗體字 = 詞彙表（Ch.24）有定義的術語
  "modal_verbs": ["must", "should"],                  // 文件自述：must = CBS 強制要求，should = 建議
  "citation_label": "CBH 2nd Ed. §4.3.8.4, p.47 (PDF p.65)", // 直接給使用者看的引用字串
  // 表格另有：table_id, table_caption, table_source_note, table_header, table_rows, table_quality
  // 詞彙/縮寫另有：term
}
```

補充說明：
- **重疊（overlap）**：同一節內，第 2 個以後的 chunk 開頭會是 `…` + 前一個 chunk 最後 40 個字（`OVERLAP_WORDS`），所以相鄰 chunk 的 `text` 會有一小段重複，這是刻意的。
- **前置頁**：Preface 和 Abbreviations 被當成「虛擬章」，`chapter_no` 分別是 `Preface`、`Abbreviations`，所以引用標籤會出現 `Ch.Preface`，不是錯誤。附錄是 `App-A`、`App-B`。
- **斷字**：PDF 行尾有連字號的字會直接接回（`non-` + `enveloped` → `non-enveloped`）。`verify` 比對時會忽略連字號差異。

## 灌進 RAG 的建議

1. **Embedding**：用 `embed_text`，其餘欄位存成 metadata（Chroma / Qdrant / pgvector / Weaviate 都可）。
2. **Hybrid 檢索**：向量 + BM25。這份文件縮寫很多（BSC、HPTA、CL3），關鍵字比對表現比較好。
3. **Metadata 過濾**：例如「定義類問題」→ 優先找 `chunk_type in (glossary_term, abbreviation)`；「法規義務」→ `modal_verbs` 含 `must` 或 `regulatory_refs` 不為空。
4. **擴充上下文**：拿到某個 chunk 後，可以用 `section_id` 撈回同節的相鄰 chunk，用 `defined_terms_used` 帶入對應的詞彙定義。

## 強制引用的 Prompt（防幻覺）

```text
只能根據下方 <context> 回答。每個論點都必須附上 chunk_id 與該 chunk 原文的逐字引述（quote）。
如果 context 沒有答案，回答「此手冊未提及」，不要用外部知識補。
輸出 JSON：
{"answer": "...", "claims": [{"statement": "...", "chunk_id": "...", "quote": "原文逐字摘錄"}]}

<context>
[CBH-2E-2016:4.3.1:003] (CBH 2nd Ed. §4.3.1, Table 4–1, p.41)
...chunk text...
</context>
```

接著用程式驗證（這一步 LLM 無法造假）：

```bash
python trace_tools.py verify answer.json      # chunk_id 必須存在，quote 必須真的出現在原文
python trace_tools.py show CBH-2E-2016:4.3.8.4:001 /path/to/cbh-gcb-eng.pdf   # 在原頁框出來源位置
```

沒通過驗證的回答 → 不顯示，或標示「未驗證」並要求模型重新產生。

## 驗證結果（`validation_report.json`）

- 目錄（TOC）119 個節條目全部找到，印刷頁碼與目錄全部一致 ✅
- 共找到 331 個章節（含第 3、4 層）
- 輸出與原文的字詞覆蓋率 97.6%（缺的主要是標題與章節分隔頁，標題已存進 metadata）
- 840 個 chunk 全部有印刷頁碼；最長的 chunk 399 字（約 520 tokens）

## Chunking 策略：結構感知（structure-aware）+ 段落累積

1. **以章節為硬邊界**：chunk 不跨標題，每個 chunk 只屬於一個 `section_id`。
2. **段落 / 條列項目為最小單位**，跨頁且上一頁沒有句號結尾的段落會接回同一段。
3. **段落累積到約 280 字（約 380 tokens）就切**；單一段落超過約 360 字時，在句號、分號處切開。
4. **40 字重疊**（同一節內，前綴 `…`）。
5. **特殊內容一個一塊**：表格、詞彙表術語、縮寫、圖說。
6. **embedding 時前綴章節路徑**（`embed_text`）。

## OpenRouter VLM 補強（`openrouter_enrich.py` → `chunks_enriched.jsonl`）

- 模型：`anthropic/claude-sonnet-5.5`（可用 `--model` 換）；API key 放在 `.env`（已列入 `.gitignore`）
- 4 張 `needs_review` 表格 → `table_header_vlm` / `table_rows_vlm` / `vlm_text`，合併儲存格已向下填滿
  - **每一格都回頭比對 PDF 文字層**，對不到的列在 `vlm_unverified_cells`（本次只有 Table 15–2 有 4 格，3 格是換行造成的，1 格是符號 `+/L*`，請人工核對）
- 28 張圖 → `figure_description_vlm`（圖內標籤、流程分支，例如 Figure 18–2 決策圖）
- 每個 chunk 都有 `content_origin`（`pdf_text` / `pdf_text+vlm_table_layout` / `pdf_caption+vlm_description`）和 `generated_by`（模型、日期）
- `trace_tools.py verify` 會優先讀 `chunks_enriched.jsonl`；引述只出現在 VLM 內容時標 ⚠（可用，但不是 PDF 原文）
- **向量庫建議用 `chunks_enriched.jsonl`**（`embed_text` 已換成 VLM 重建的表格與圖描述）

## LangChain 問答管線（`rag_langchain.py`）

```bash
pip install -r requirements.txt
python rag_langchain.py build                                  # 建 FAISS 索引（qwen3-embedding-8b，< US$0.01）
python rag_langchain.py ask "發生實驗室事故後，怎麼判斷是否需要通報 PHAC？"
```

| 階段 | 做法 |
|---|---|
| 查詢改寫 | `claude-haiku-5.5` 把問題轉成英文檢索句與關鍵字（文件是英文，BM25 需要英文） |
| 混合檢索 | FAISS 向量（原問題 + 英文改寫，cosine）＋ BM25（英文關鍵字），RRF 融合取前 6。直接用 `faiss` / `rank_bm25`，不依賴已停止維護的 `langchain-community` |
| 索引一致性 | `faiss_index/ids.json` 記錄 embedding 模型與每個 chunk 的 hash；chunks 重新解析或換模型後，`ask` 會要求先重跑 `build`，避免引用到過期內容 |
| 回答 | `claude-sonnet-5.5`，Pydantic schema 強制輸出 `{answer, claims[{statement, chunk_id, quote}], insufficient_context}` |
| 驗證 | 每個 quote 必須逐字出現在該 chunk；不符就要求模型重寫一次，仍不符則標 ✗；只出現在 VLM 內容的標 ⚠ |
| 拒答處理 | 模型內容安全過濾拒答時，不嘗試繞過，改列出檢索到的原文出處讓使用者自行查閱 |

可用環境變數換模型：`CBH_ANSWER_MODEL`、`CBH_REWRITE_MODEL`、`CBH_EMBED_MODEL`。

注意：OpenRouter 上的這些 Claude 模型不支援強制 `tool_choice`，所以 structured output 用 `method="json_schema", strict=True`（`function_calling` 會回 400）。

測試結果（5 題）：表格查詢、比較題（引用 Table 11–1）、術語定義、流程判斷（Ch.18 通報規則）全部引用驗證通過；範圍外問題（COVID-19 疫苗劑數）正確回答「手冊未提及」。

## 對照課程 M1 / M2（`course_pipeline.py` → `rag_output/course/`）

```bash
python course_pipeline.py /path/to/cbh-gcb-eng.pdf
```

| 課程要求 | 本專案做法 | 狀態 |
|---|---|---|
| M1 PDF SHA-256 指紋 | `pdf_sha256` 寫入每個 chunk 與 `pages.jsonl` | ✅ |
| M1 逐頁解析 `pages.jsonl`（頁碼連續、雜湊可重算） | 另含印刷頁碼（從頁尾讀取，非固定偏移） | ✅ |
| M1 章節辨識（課程：regex 候選） | 依字型解析，目錄 119/119 節比對通過 | ✅ 更完整 |
| M1 三種 Chunking 比較 | `chunking_comparison.csv`（LangChain fixed / recursive vs structure） | ✅ |
| M1 資料契約 `chunks_structure.jsonl` | `document_id, pdf_sha256, section_id, pdf_page, source_pages, source_boxes, text, text_sha256, word_count` ＋ 原有全部 metadata | ✅ |
| M1 TF-IDF 基準（char 3–5 gram） | `course_pipeline.py` 的 `Retrievers.tfidf_rank` | ✅ |
| M1 技術稽核（雜湊、頁碼、bbox 在頁內） | 840/840 無異常 | ✅ |
| M1 來源框選視覺化 | `trace_tools.py show`、`evidence_preview_q*.png` | ✅ |
| M1 選修 Dense（課程：multilingual-e5） | `qwen3-embedding-8b`（多語系、4096 維，經 OpenRouter） | ✅ |
| M1 選修 Qdrant `:memory:` | 改用 FAISS 檔案索引（可持久化；Qdrant 記憶體模式重啟即消失） | ⚖️ 替代 |
| M2 BM25 + TF-IDF + RRF | ✅；另加 Dense+BM25、本系統（LLM 改寫 + Dense×2 + BM25） | ✅ |
| M2 Parent–Child 上下文還原 | 同一 parent 章節前後 2 個 chunk（含短小兄弟小節），已接入問答管線 | ✅ |
| M2 選修 Cross-Encoder Reranker | 未做（評測 Hit@5 已 100%，加 reranker 的邊際效益需更大的評測集才看得出） | ❌ 未做 |
| M2 Evidence Set（`candidate_evidence_only`） | `evidence_set.json`，含 technical_check、parent_context、content_origin | ✅ |
| M2 Golden QA（Hit@5） | 22 題（12 題中文）、6 種方法，`evaluation.csv` / `evaluation_summary.csv`，另算 MRR@10 | ✅ |
| M2 交付：兩題 BM25/TF-IDF/Hybrid 排序比較 | `ranking_comparison.csv` | ✅ |
| M2 `run_summary.json` | ✅ | ✅ |
| M3 方向：Citation Validator、拒答 | 已完成（`trace_tools.py verify`、`rag_langchain.py`） | ✅ 超前 |

### Chunking 比較結果（同一份內文、同樣 180 字上限）

| 方法 | chunks | 平均字數 | 跨章節 % | 跨頁 % | 句中截斷 % |
|---|---|---|---|---|---|
| fixed | 699 | 179.8 | **44.1** | 23.3 | **94.0** |
| recursive | 807 | 145.1 | 30.9 | 2.0 | 66.2 |
| structure（本專案） | 840 | 125.7 | **0.0** | 16.9 | **11.4** |

structure 的 11.4%「句中截斷」主要是詞彙表、縮寫、表格這類本來就沒有句號的 chunk；跨頁是段落本身跨頁，chunk 仍保留每頁的 bbox。

### 檢索評測（22 題，命中 = 前 5 名有目標章節或其子節）

| 方法 | Hit@5 | MRR@10 | Hit@5（中文題） |
|---|---|---|---|
| BM25 | 0.545 | 0.400 | 0.308 |
| TF-IDF | 0.500 | 0.394 | 0.308 |
| Dense | 0.955 | 0.886 | 1.000 |
| Hybrid BM25+TF-IDF（課程 M2 必修版） | 0.500 | 0.439 | 0.308 |
| Hybrid Dense+BM25 | 0.955 | 0.773 | 0.923 |
| **本系統（改寫 + Dense×2 + BM25）** | **1.000** | **0.920** | **1.000** |

評測題的目標章節是我依**章節標題**人工指定的（不是從檢索結果反推），但仍建議你檢查 `course_pipeline.py` 裡的 `GOLD` 是否同意這些標註。22 題偏少，數字只能當方向參考。

### 課程討論題

- **M1-5／M2-3 bbox 一致性 ≠ 法規正確性**：技術稽核只證明「這段文字真的在那一頁那個位置、沒被改過」；不證明這段文字回答了問題、條件完整、或在現行法規下仍有效。本專案把三層分開：`audit`（位置/雜湊）→ `verify`（回答的引述逐字存在）→ 內容是否支持結論仍需人工或評測集判斷。
- **M2-1 BM25+TF-IDF 為何不等於 Dense+Sparse Hybrid**：兩者都是詞彙比對，錯在同一類地方（同義詞、跨語言）。評測可見中文題兩者都只有 0.308，融合後沒有提升；加入 Dense 才補上語意與跨語言。
- **M2-2 section_id 誤判時 Parent Expansion 的風險**：會把不相關章節的文字當成上下文，模型可能據此作答。本專案章節由字型辨識並通過目錄 119/119 驗證，且展開只限同一 parent 章節、前後 2 個 chunk、總數上限 12。
- **M2-4 「現行有效法規」為何不能直接引 2016 手冊**：CBH 2nd Ed. 是 2016 年的指引，CBS 與 HPTA/HPTR 之後可能修訂；它本身也說明是「指引」而非要求（§1.4）。所有 chunk 都帶 `pub_date: 2016-03`，回答若涉及現行要求，應另外查核最新版 CBS／法規後才能下結論。

## 已知限制

- **表格**：`chunks.jsonl`（純 PDF 解析）中 9–1、11–1、15–1、15–2 為 `needs_review`；已在 `chunks_enriched.jsonl` 用 VLM 重建，只剩 Table 15–2 的 4 格要人工核對。
- **圖**：圖的內容描述來自 VLM（`chunks_enriched.jsonl`），並非原文；我抽查了 Figure 4–1，與原圖一致，其餘 27 張未逐一人工核對。
- 索引頁（PDF p.359 之後）和目錄頁刻意排除，不當知識來源。
- 重跑：`python parse_cbh.py cbh-gcb-eng.pdf rag_output`（需要 `pip install pymupdf`）
