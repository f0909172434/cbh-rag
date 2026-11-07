# Lab06：RAG + LINE Bot 教學示範

> **來源與授權**：本資料夾的程式與說明是課程授課教師提供的 Lab06 教材，**不在本 repo 的 MIT 授權範圍內**，著作權屬原作者。
> 本 repo 只做了以下調整：`.env.example` 改名（原檔名為 `.env copy.example`）、`config.yaml` 的 `pdf_path` 改成與根目錄 README 相同的 PDF 下載位置，並加上這段說明。
> FAISS 索引（`index/`）含手冊原文，依根目錄 `NOTICE` 不公開，請自行執行 `ingest.py` 重建。
>
> macOS / Linux 把下方指令中的 `.venv\Scripts\python.exe` 換成 `.venv/bin/python`，`copy` 換成 `cp`。

用 Canadian Biosafety Handbook（英文 PDF，366 頁）做知識庫，在 LINE 用繁中提問、繁中回答並附來源頁碼。
重點是「看得懂」：每個檔案都很短。

**以下所有指令都在 `labs\lab06_linebot` 目錄下執行**（PowerShell）。從專案根目錄進入：

```powershell
cd labs\lab06_linebot
```

## 架構圖

```
 [PDF] --ingest.py--> 切塊 --Voyage embedding--> [FAISS index/]
                                                      |
 LINE 使用者 -> LINE 平台 -> ngrok -> FastAPI /callback (app.py, 驗簽)
                                           |
                                      rag.py answer(question)
                                           |-- 1. retriever 取 top_k 個 chunk（<-- FAISS）
                                           |-- 2. chunk + 頁碼塞進 prompt -> Claude
                                           '-- 3. 回傳（繁中答案, 頁碼）
                                           |
              LINE 使用者 <- LINE 平台 <- reply_message（答案 + 空行 + 來源：p.X）
```

| 檔案 | 作用 |
|---|---|
| `config.yaml` | 模型、chunk、top_k、prompt 等設定（模型字串都在這） |
| `common.py` | 載入 `.env` 與 `config.yaml` |
| `ingest.py` | 建索引（只需跑一次） |
| `rag.py` | 檢索 + LLM（LCEL） |
| `ask.py` | 不用 LINE 的命令列測試 |
| `app.py` | LINE webhook |

## 1. 安裝（需 Python 3.11 與 uv）

沒有 uv 的話先裝：`winget install astral-sh.uv`（或 `pip install uv`），裝完重開終端機。

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

不用 `activate`（PowerShell 執行原則可能擋），後面所有指令都直接用 `.venv\Scripts\python.exe`。

## 2. 準備 PDF 與金鑰

- PDF：ingest 讀的是專案根目錄的 `cbh-gcb-eng.pdf`（`config.yaml` 的 `pdf_path`），沒有的話依根目錄 README 的 `curl` 指令下載。
- 金鑰：

```powershell
copy .env.example .env
```
編輯 `.env`，填入 `ANTHROPIC_API_KEY`、`VOYAGE_API_KEY`、`LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`。
（沒有 `.env` 時會退回專案根目錄的 `.env`。）

Voyage 免費帳號（未綁付款方式）會被限速（3 RPM / 10K TPM），ingest 會自動等待重試，實測約 35–40 分鐘。
綁付款方式後（https://dashboard.voyageai.com/，免費額度仍適用），可把 `config.yaml` 的 `ingest_batch_size` 調大，幾分鐘內完成。

## 3. 建索引

```powershell
.venv\Scripts\python.exe ingest.py
```
完成後會出現 `index\` 資料夾。執行時若看到 langchain-community 的 DeprecationWarning，可忽略。

## 4. 不用 LINE 先測試（建議教學先示範這步，再接 LINE）

```powershell
.venv\Scripts\python.exe ask.py "生物安全櫃多久要認證一次？"
```
輸出最後的「來源：p.X」是 PDF 檢視器的頁碼，不是手冊印刷頁碼（差約 18 頁）。

## 5. 啟動 webhook

```powershell
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```
瀏覽 http://localhost:8000/ 看到 `{"status":"ok"}` 就正常。
注意：第一則訊息才會載入索引與模型，會慢約 10 秒。

## 6. 用 ngrok 對外公開並設定 LINE

安裝 ngrok（https://ngrok.com 下載，或 `winget install ngrok.ngrok`），註冊後執行一次：
```powershell
ngrok config add-authtoken <你的token>
```
另開一個終端機：
```powershell
ngrok http 8000
```
複製 `https://xxx.ngrok-free.app` 網址，然後：

1. 到 LINE Developers 建立 Messaging API channel，取得 Channel secret 與 Channel access token（填回 `.env`，重啟 uvicorn）。
2. Messaging API 頁籤 -> Webhook URL 填 `https://xxx.ngrok-free.app/callback` -> Verify。
3. 開啟「Use webhook」。
4. 到 LINE Official Account Manager 關閉「自動回應訊息」。
5. 用手機掃 QR code 加好友，開始提問。

ngrok 免費網址每次重開都會變，要回 LINE 後台更新 Webhook URL 並按 Verify。

## 切換 LLM

改 `config.yaml` 的 `llm_model`（說明寫在檔內註解）。

## 常見錯誤

| 現象 | 原因與解法 |
|---|---|
| Webhook Verify 回 400 / 日誌 `Invalid signature` | `LINE_CHANNEL_SECRET` 填錯或沒重啟 uvicorn |
| 回覆「系統錯誤：找不到 index，請先執行 python ingest.py」 | 還沒建索引 |
| 回覆「系統錯誤」+ authentication / api key | `.env` 缺 `ANTHROPIC_API_KEY` 或 `VOYAGE_API_KEY` |
| 提問時回「系統錯誤」+ rate limit | 提問也要呼叫 Voyage，免費 3 RPM，連問 4 次就會遇到，等一分鐘再問 |
| Bot 沒反應 | ngrok 網址換了，要回 LINE 後台更新；或沒關自動回應 |
| ingest 一直印「限速，等 65 秒後重試」 | Voyage 未綁付款方式，免費帳號會被限速，ingest 會自動等待重試（約 35–40 分鐘），不是當掉 |
| `ngrok` 找不到指令 / 要求 authtoken | 先安裝 ngrok，再執行 `ngrok config add-authtoken <token>` |
