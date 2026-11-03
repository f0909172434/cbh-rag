"""命令列測試：python ask.py "生物安全櫃多久要認證一次？" """
import sys
from rag import load_chain, format_reply

question = " ".join(sys.argv[1:]) or "What is a biosafety cabinet?"
text, pages = load_chain()(question)
print(format_reply(text, pages))
