from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import threading, time
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from gmail_fetch_elite_links import process_gmail as gmail_pull  
from ingest_hrv_inbound import run_once, cleanup_archive

# Config knobs you’ve been using
COMPUTE_ARGS = "--detrend linear --window hamming --interp cubic --fs 4 --nperseg 1024 --nfft 4096"

# Simple in-process lock 
_run_lock = threading.Lock()
_last_run_info = {"running": False, "started_at": None, "finished_at": None}

class RefreshResponse(BaseModel):
    running: bool
    started_at: Optional[float]
    finished_at: Optional[float]
    fetched_zips: int
    processed_files: int
    quarantined: int
    archived_deleted: int

app = FastAPI(title="HRV Data Refresh")

def _count_dir(path: Path) -> int:
    return sum(1 for p in path.glob("*") if p.is_file())

@app.post("/refresh", response_model=RefreshResponse)
def refresh_endpoint():
    # simple lock to avoid concurrent runs
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=423, detail="Refresh already running")

    _last_run_info.update({"running": True, "started_at": time.time(), "finished_at": None})
    fetched = processed = quarantined = cleaned = 0
    try:
        # 1) Pull new zips from Gmail into inbound/
        before = _count_dir(Path("data_pipeline/io/inbound"))
        gmail_pull()  # idempotent; labels messages after
        after = _count_dir(Path("data_pipeline/io/inbound"))
        fetched = max(0, after - before)

        # 2) Unzip + compute + DB + archive
        #    We pass compute args exactly as you’ve frozen them.
        run_once(dry_run=False, extra_compute_args=COMPUTE_ARGS.split())

        # 3) Optional: cleanup archives (30-day retention)
        cleanup_archive(retention_days=30)
        # (we won't count deletions precisely here; you can extend cleanup to return a count)

        # You can compute `processed` and `quarantined` by diffing archive/quarantine counts if you want:
        # processed = ...
        # quarantined = ...

        return RefreshResponse(
            running=False,
            started_at=_last_run_info["started_at"],
            finished_at=time.time(),
            fetched_zips=fetched,
            processed_files=processed,
            quarantined=quarantined,
            archived_deleted=cleaned,
        )
    finally:
        _last_run_info.update({"running": False, "finished_at": time.time()})
        _run_lock.release()

@app.get("/refresh/status", response_model=RefreshResponse)
def refresh_status():
    # a simple status endpoint you can poll to disable the button while running
    return RefreshResponse(
        running=_last_run_info["running"],
        started_at=_last_run_info["started_at"],
        finished_at=_last_run_info["finished_at"],
        fetched_zips=0, processed_files=0, quarantined=0, archived_deleted=0
    )
