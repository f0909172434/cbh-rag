"""RAG 核心：load_chain() 回傳 answer(question) -> (回答, 頁碼清單)。"""
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from common import CFG, INDEX_DIR


def load_retriever():
    if not INDEX_DIR.exists():
        raise FileNotFoundError("找不到 index，請先執行 python ingest.py")
    emb = VoyageAIEmbeddings(model=CFG["embedding_model"])
    db = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)  # 自己建的索引，可信
    return db.as_retriever(search_kwargs={"k": CFG["top_k"]})


def load_chain():
    retriever = load_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", CFG["system_prompt"] + "\n【檢索內容】\n{context}"),
        ("human", "{question}"),
    ])
    llm = init_chat_model(CFG["llm_model"])               # 例如 "anthropic:claude-haiku-5-5"
    chain = prompt | llm | StrOutputParser()               # LCEL：prompt -> LLM -> 純文字

    def answer(question):
        docs = retriever.invoke(question)                  # 1. 檢索
        context = "\n\n".join(f"[p.{d.metadata['page'] + 1}] {d.page_content}" for d in docs)
        text = chain.invoke({"context": context, "question": question})  # 2. 生成
        pages = sorted({d.metadata["page"] + 1 for d in docs})           # 3. 來源頁碼
        return text, pages

    return answer


NO_ANSWER_HINT = "找不到相關內容"   # 與 system_prompt 第 3 條一致


def format_reply(text, pages):
    """組出要回給使用者的字串；回答是「找不到」時不附頁碼，避免誤導。"""
    if NO_ANSWER_HINT in text or not pages:
        return text
    return f"{text}\n\n來源：" + ", ".join(f"p.{p}" for p in pages)
