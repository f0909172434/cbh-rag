"""LINE Bot webhook。啟動：uvicorn app:app --port 8000"""
import os
from fastapi import FastAPI, HTTPException, Request
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (ApiClient, Configuration, MessagingApi,
                                  ReplyMessageRequest, TextMessage as ReplyText)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import common  # noqa: F401  (載入 .env)
from rag import load_chain, format_reply

app = FastAPI()
parser = WebhookParser(os.getenv("LINE_CHANNEL_SECRET", ""))
line_conf = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""))
_answer = None  # 第一次收到訊息時才載入索引與模型


def get_answer():
    global _answer
    if _answer is None:
        _answer = load_chain()
    return _answer


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/callback")
async def callback(request: Request):
    body = (await request.body()).decode("utf-8")
    try:  # 驗簽：簽章不對就丟 InvalidSignatureError
        events = parser.parse(body, request.headers.get("X-Line-Signature", ""))
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
            try:
                text, pages = get_answer()(event.message.text)
                reply = format_reply(text, pages)
            except Exception as e:  # 教學用：錯誤直接回給使用者方便除錯
                reply = f"系統錯誤：{e}"
            with ApiClient(line_conf) as client:
                MessagingApi(client).reply_message(ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[ReplyText(text=reply[:5000])]))  # LINE 上限 5000 字
    return "OK"
