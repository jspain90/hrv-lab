from __future__ import annotations
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.gmail_fetch import process_gmail as gmail_pull
from pipeline.ingest import run_once, cleanup_archive, INBOUND_DIR, ARCHIVE_DIR, QUARANTINE_DIR
from pipeline.weather_fetch import backfill_missing_weather
from pipeline.compute_residuals import compute_residuals_incremental
from app.core.settings import settings

COMPUTE_ARGS = "--detrend linear --window hamming --interp cubic --fs 4 --nperseg 1024 --nfft 4096"

def _repo_root() -> Path:
    """Get repository root directory."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "app").exists():
            return p
    return Path.cwd()

MODEL_PATH = _repo_root() / "models" / "hrv_baseline.pkl"

router = APIRouter(prefix="/refresh", tags=["refresh"])

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
    weather_dates_fetched: int
    new_predictions: int
    error_message: Optional[str] = None


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file())


def _count_processed_rows() -> Optional[int]:
    try:
        with sqlite3.connect(settings.db_path) as cx:
            cur = cx.execute("SELECT COUNT(*) FROM processed_heart_rate_readings")
            return int(cur.fetchone()[0])
    except sqlite3.Error:
        return None

@router.post("", response_model=RefreshResponse)
def trigger_refresh() -> RefreshResponse:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=423, detail="Refresh already running")

    _last_run_info.update({"running": True, "started_at": time.time(), "finished_at": None})

    fetched = processed = quarantined = archived_deleted = weather_fetched = new_predictions = 0
    errors: list[str] = []

    inbound_before = _count_files(INBOUND_DIR)
    archive_before = _count_files(ARCHIVE_DIR)
    quarantine_before = _count_files(QUARANTINE_DIR)
    processed_before = _count_processed_rows()

    try:
        # Step 1: Fetch Gmail attachments
        gmail_pull()
    except Exception as exc:  # pragma: no cover - loggable path
        errors.append(f"Gmail fetch failed: {exc}")
    else:
        inbound_after = _count_files(INBOUND_DIR)
        fetched = max(0, inbound_after - inbound_before)

    try:
        # Step 2: Process HRV files
        run_once(dry_run=False, extra_compute_args=COMPUTE_ARGS.split())
    except Exception as exc:  # pragma: no cover - loggable path
        errors.append(f"Pipeline ingest failed: {exc}")
    else:
        archive_after = _count_files(ARCHIVE_DIR)
        quarantine_after = _count_files(QUARANTINE_DIR)
        processed_after = _count_processed_rows()

        processed = max(0, (processed_after or 0) - (processed_before or 0)) if processed_after is not None and processed_before is not None else processed
        quarantined = max(0, quarantine_after - quarantine_before)

        # Step 3: Cleanup archives
        archive_count_pre_cleanup = archive_after
        cleanup_archive(retention_days=30)
        archive_after_cleanup = _count_files(ARCHIVE_DIR)
        archived_deleted = max(0, archive_count_pre_cleanup - archive_after_cleanup)

    # Step 4: Backfill weather data for missing dates
    try:
        weather_fetched = backfill_missing_weather(Path(settings.db_path))
    except Exception as exc:
        errors.append(f"Weather backfill failed: {exc}")

    # Step 5: Compute residuals for new HRV data (CRITICAL: Never retrain model!)
    try:
        new_predictions = compute_residuals_incremental(MODEL_PATH, Path(settings.db_path))
    except Exception as exc:
        errors.append(f"Residual computation failed: {exc}")

    _last_run_info.update({"running": False, "finished_at": time.time()})
    _run_lock.release()

    return RefreshResponse(
        running=False,
        started_at=_last_run_info["started_at"],
        finished_at=_last_run_info["finished_at"],
        fetched_zips=fetched,
        processed_files=processed,
        quarantined=quarantined,
        archived_deleted=archived_deleted,
        weather_dates_fetched=weather_fetched,
        new_predictions=new_predictions,
        error_message="; ".join(errors) if errors else None,
    )


@router.get("/status", response_model=RefreshResponse)
def refresh_status() -> RefreshResponse:
    return RefreshResponse(
        running=_last_run_info["running"],
        started_at=_last_run_info["started_at"],
        finished_at=_last_run_info["finished_at"],
        fetched_zips=0,
        processed_files=0,
        quarantined=0,
        archived_deleted=0,
        weather_dates_fetched=0,
        new_predictions=0,
        error_message=None,
    )
