"""共用：讀 .env 與 config.yaml。"""
from pathlib import Path
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent          # labs/lab06_linebot
ROOT = HERE.parents[1]                # 專案根目錄

# 先找本資料夾的 .env，沒有就退回專案根目錄的 .env
load_dotenv(HERE / ".env") if (HERE / ".env").exists() else load_dotenv(ROOT / ".env")

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
INDEX_DIR = HERE / CFG["index_dir"]
