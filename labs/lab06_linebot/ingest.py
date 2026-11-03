"""建索引：PDF -> 切塊 -> Voyage embedding -> FAISS 存檔。執行：python ingest.py"""
import time
from voyageai.error import RateLimitError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from common import CFG, ROOT, INDEX_DIR


def embed_with_retry(fn, n):
    """遇到限速就等 65 秒重試（免費帳號 3 RPM / 10K TPM），同一批最多 10 次。"""
    for attempt in range(10):
        try:
            return fn()
        except RateLimitError:
            print(f"限速，等 65 秒後重試第 {n} 批（第 {attempt + 1} 次）", flush=True)
            time.sleep(65)
    raise RuntimeError(f"第 {n} 批重試 10 次仍被限速，放棄")


t0 = time.time()
pages = PyPDFLoader(str(ROOT / CFG["pdf_path"])).load()   # 每頁一個 Document，metadata 有 page（從 0 起算）
print(f"載入 {len(pages)} 頁")

splitter = RecursiveCharacterTextSplitter(chunk_size=CFG["chunk_size"], chunk_overlap=CFG["chunk_overlap"])
chunks = splitter.split_documents(pages)                  # 切塊後 metadata 會保留
print(f"切成 {len(chunks)} 個 chunk（範例：PDF 第 {chunks[0].metadata['page'] + 1} 頁）")

emb = VoyageAIEmbeddings(model=CFG["embedding_model"], batch_size=CFG["ingest_batch_size"])
size = CFG["ingest_batch_size"]
batches = [chunks[i:i + size] for i in range(0, len(chunks), size)]
db = None
for n, batch in enumerate(batches, 1):                    # 分批送，避免一次超過限速
    if db is None:
        db = embed_with_retry(lambda: FAISS.from_documents(batch, emb), n)   # 第一批：建立索引
    else:
        embed_with_retry(lambda: db.add_documents(batch), n)                 # 之後：加進索引
    print(f"第 {n}/{len(batches)} 批，累計 {db.index.ntotal} chunks", flush=True)
    if n % 10 == 0:
        db.save_local(str(INDEX_DIR))                     # checkpoint：中途中斷也不會全部重來（但不會自動續傳）

db.save_local(str(INDEX_DIR))
print(f"完成，已存到 {INDEX_DIR}，耗時 {time.time() - t0:.1f} 秒")
