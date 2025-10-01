from __future__ import annotations
import os
from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    #db_path: str = os.environ.get("HRV_LAB_DB", "hrv_lab.sqlite3")
    enable_watcher: bool = os.environ.get("HRV_LAB_WATCHER", "0") == "1"
    inbox_dir: str = os.path.join("data", "inbox")
    archive_dir: str = os.path.join("data", "archive")
    db_path: str = os.environ.get(
        "HRV_LAB_DB",
        str(Path(__file__).resolve().parents[2] / "hrv_lab.sqlite3"),
    )
settings = Settings()

