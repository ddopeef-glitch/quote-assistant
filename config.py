from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "询价助手 · 脱敏公开版"
APP_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8-sig").strip()
MASTER_DATA_FILE = ROOT / "主数据" / "标准产品价格表.xlsx"
OUTPUT_DIR = ROOT / "data" / "output"
