# Create the orchestrator script that ingests inbound HRV files,
# computes metrics via an existing compute script that emits JSON,
# writes to SQLite, and archives/quarantines files.
#
# Assumptions:
# - You already created the following tables in hrv_lab.sqlite3:
#   processed_heart_rate_readings, hrv_results, standing_trials, exercise_sessions
# - Your compute script (compute_hrv_metrics.py) supports --json and prints a single JSON object.
# - Inbound files are TXT/CSV from Elite HRV exports.
#
# Usage examples:
#   python ingest_hrv_inbound.py --run-once
#   python ingest_hrv_inbound.py --dry-run --run-once
#
# You can also edit the CONFIG section at the top to match your paths.


import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import zipfile
import re

# =====================
# CONFIG
# =====================

PIPELINE_VERSION_FALLBACK = "hrv/1.0.0"

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "app").exists():   # detect repo root by presence of app/
            return p
    return Path.cwd()

DATA_ROOT = Path(os.environ.get("HRV_LAB_DATA_ROOT", str(_repo_root() / "data_pipeline"))).resolve()
DB_PATH = Path(
    os.environ.get(
        "HRV_LAB_DB",
        Path(__file__).resolve().parents[1] / "hrv_lab.sqlite3",
    )
)
INBOUND_DIR     = DATA_ROOT / "io" / "inbound"
ARCHIVE_DIR     = DATA_ROOT / "io" / "archive"
QUARANTINE_DIR  = DATA_ROOT / "io" / "quarantine"
ARCHIVE_ZIPS_DIR = ARCHIVE_DIR / "zips"
COMPUTE_SCRIPT  = Path("compute_hrv_metrics.py")  # your existing script

#TEmp debug
with sqlite3.connect(DB_PATH) as cx:
    print(cx.execute("PRAGMA database_list").fetchall())

# ensure directories exist
for d in (INBOUND_DIR, ARCHIVE_DIR, QUARANTINE_DIR, ARCHIVE_ZIPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Duration thresholds (seconds)
MIN_SEC   = 120   # discard below this
HRR_EXACT = 60   #HR recovery snapshot, will require additional work   
SHORT_MAX = 540   # 2–9 min -> hrv_results
STAND_MAX = 720   # 9–12 min -> standing_trials
# >= STAND_MAX -> exercise_sessions

# =====================
# Helpers
# =====================

FNAME_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})")

def ts_from_filename(name: str) -> str | None:
    m = FNAME_TS_RE.match(name)
    if not m:
        return None
    # build ISO: "YYYY-MM-DD HH-MM-SS" -> "YYYY-MM-DDTHH:MM:SS"
    raw = m.group(1)
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H-%M-%S")
        return dt.isoformat()  # naive local time; adjust to your taste
    except Exception:
        return None

def cleanup_archive(retention_days: int = 30) -> None:
    """Delete archived files older than retention_days from archive/ and archive/zips/."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for root in [ARCHIVE_DIR, ARCHIVE_ZIPS_DIR]:
        if not root.exists(): 
            continue
        for p in root.iterdir():
            if not p.is_file():
                continue
            try:
                # Use last modified; good enough for archive
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    p.unlink()
                    print(f"[CLEAN] deleted {p.name}")
            except Exception as e:
                print(f"[CLEAN] skipped {p.name}: {e}")

def is_safe_member(member: zipfile.ZipInfo) -> bool:
    # Reject directories, absolute paths, and traversal attempts
    name = member.filename
    if member.is_dir():
        return False
    if name.startswith("/") or name.startswith("\\"):
        return False
    norm = Path(name).as_posix()
    if ".." in norm.split("/"):
        return False
    return True

def extract_txts_from_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract only .txt files safely from zip into dest_dir, flattening subdirs."""
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for m in zf.infolist():
                if not is_safe_member(m):
                    continue
                if not m.filename.lower().endswith(".txt"):
                    continue
                if m.filename.startswith("!") or "!" in m.filename:
                    continue                
                # flatten: take only the filename part
                target_name = Path(m.filename).name
                # avoid collisions
                out_path = dest_dir / target_name
                if out_path.exists():
                    stem, suf = out_path.stem, out_path.suffix
                    i = 2
                    while True:
                        cand = dest_dir / f"{stem} ({i}){suf}"
                        if not cand.exists():
                            out_path = cand
                            break
                        i += 1
                with zf.open(m) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(out_path)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Bad zip: {e}")
    return extracted

def expand_any_zips():
    ensure_dirs()
    inbound = sorted([p for p in INBOUND_DIR.iterdir() if p.is_file()])
    for p in inbound:
        if p.suffix.lower() == ".zip":
            print(f"[UNZIP] {p.name}")
            try:
                files = extract_txts_from_zip(p, INBOUND_DIR)
                # archive the zip after successful extraction
                ARCHIVE_ZIPS_DIR.mkdir(parents=True, exist_ok=True)
                move_to(ARCHIVE_ZIPS_DIR, p)
                if files:
                    for f in files:
                        print(f"        -> {f.name}")
                else:
                    print("        (no .txt files found)")
            except Exception as e:
                print(f"[ERR  ] unzip {p.name}: {e}")
                qpath = move_to(QUARANTINE_DIR, p)
                (qpath.with_suffix(qpath.suffix + ".error.txt")).write_text(str(e), encoding="utf-8")


def normalize_win_path(p: str) -> str:
    """Convert Git Bash style /c/Users/... to C:/Users/... on Windows."""
    if os.name == "nt":
        m = re.match(r"^/([a-zA-Z])/(.*)", p)
        if m:
            return f"{m.group(1).upper()}:/{m.group(2)}"
    return p

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def ensure_dirs():
    for d in [INBOUND_DIR, ARCHIVE_DIR, QUARANTINE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def autodetect_sep(path: Path) -> Optional[str]:
    # Try pandas engine='python' sniff; fall back to candidates.
    try:
        pd.read_csv(path, engine="python", sep=None, nrows=5)
        return None  # autodetect works
    except Exception:
        for candidate in [",", "\t", ";", r"\s+"]:
            try:
                pd.read_csv(path, engine="python", sep=candidate, nrows=5)
                return candidate
            except Exception:
                continue
    return None

def peek_rr_duration_and_start(path: Path) -> Tuple[Optional[float], Optional[str]]:
    """
    Fast/robust duration probe for Elite HRV txt files.
    Supports headerless single-column files (just RR values per line),
    or basic delimited text. Returns (duration_sec, timestamp_iso_or_None).
    """
    try:
        # 1) Read as headerless, whitespace-separated to handle bare TXT
        df = pd.read_csv(path, engine="python", sep=r"\s+", header=None, nrows=1_000_000)

        # 2) Pick a numeric column (fallback: first column coerced)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            series = df[num_cols[0]]
        else:
            series = pd.to_numeric(df.iloc[:, 0], errors="coerce")

        rr = series.dropna().to_numpy(dtype=float)
        if rr.size == 0:
            return None, None

        # 3) Units heuristic (only for peek): if median < 5, assume seconds → convert to ms
        med = float(np.nanmedian(rr))
        rr_ms = rr * (1000.0 if med < 5.0 else 1.0)

        # 4) Basic sanity: drop non-positive intervals
        rr_ms = rr_ms[rr_ms > 0]
        if rr_ms.size == 0:
            return None, None

        duration_sec = float(np.sum(rr_ms) / 1000.0)

        # 5) Many Elite exports don't include a timestamp here; return None
        return duration_sec, None
    except Exception:
        return None, None

def choose_table(duration_sec: float) -> Optional[str]:
    if duration_sec == HRR_EXACT:
        return "heart_rate_recovery"
    if duration_sec < MIN_SEC:
        return None
    if duration_sec < SHORT_MAX:
        return "hrv_results"
    if duration_sec < STAND_MAX:
        return "standing_trials"
    return "exercise_sessions"

def insert_processed(cx: sqlite3.Connection, file_hash: str, file_name: str,
                     received_at: str, processed_at: str, duration_sec: float,
                     source_email_id: Optional[str], pipeline_version: str):
    cx.execute(
        """INSERT OR IGNORE INTO processed_heart_rate_readings
           (file_hash, file_name, received_at, processed_at, duration_sec, source_email_id, pipeline_version)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (file_hash, file_name, received_at, processed_at, duration_sec, source_email_id, pipeline_version)
    )

def already_processed(cx: sqlite3.Connection, file_hash: str) -> bool:
    cur = cx.execute("SELECT 1 FROM processed_heart_rate_readings WHERE file_hash = ? LIMIT 1", (file_hash,))
    return cur.fetchone() is not None

def run_compute(path: Path, extra_args: list[str]) -> dict:
    cmd = [sys.executable, str(COMPUTE_SCRIPT), "--input", str(path), "--json"]
    if extra_args:
        cmd.extend(extra_args)
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"compute script failed (rc={p.returncode}): {p.stderr.strip()}")
    try:
        return json.loads(p.stdout.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"compute script did not return JSON: {e}\nSTDOUT:\n{p.stdout[:1000]}")

def map_metrics_for_table(metrics: dict) -> dict:
    # Normalize key variants from different script versions
    out = {}
    def get(*names, default=None):
        for n in names:
            if n in metrics:
                return metrics[n]
        return default

    out["timestamp"] = get("timestamp")
    out["mean_rr_ms"] = get("MEAN_RR_MS", "MeanRR_MS")
    out["hr_bpm"] = get("HR_BPM", "HR")
    out["sdnn_ms"] = get("SDNN_MS")
    out["rmssd_ms"] = get("RMSSD_MS")
    out["pnn50"] = get("PNN50_PCT", "PNN50")
    out["total_power_ms2"] = get("TotalPower_MS2", "TP_MS2", default=None)
    out["vlf_ms2"] = get("VLF_MS2")
    out["lf_ms2"] = get("LF_MS2")
    out["hf_ms2"] = get("HF_MS2")
    out["lf_hf"] = get("LF_HF_Ratio", "LF_HF")
    out["vlf_peak_hz"] = get("VLF_PEAK_HZ")
    out["lf_peak_hz"] = get("LF_PEAK_HZ")
    out["hf_peak_hz"] = get("HF_PEAK_HZ")
    out["breath_rate_brpm"] = get("BREATH_RATE_BRPM")
    out["slow_fraction"] = get("SLOW_FRACTION_0_00_0_06_OVER_0_00_0_15")
    out["ese"] = get("ESE_TP_MINUS_2P5xHF")
    out["fs_interp_hz"] = get("FS_INTERP_HZ")
    out["detrend"] = get("DETREND")
    out["window"] = get("WINDOW")
    out["nperseg"] = get("NPERSEG")
    out["nfft"] = get("NFFT")
    out["interp"] = get("INTERP")
    out["pipeline_version"] = get("PIPELINE_VERSION", default=PIPELINE_VERSION_FALLBACK)
    return out

def insert_metrics(cx: sqlite3.Connection, table: str, file_hash: str, m: dict):
    cols = [
        "file_hash","timestamp","mean_rr_ms","hr_bpm","sdnn_ms","rmssd_ms","pnn50",
        "total_power_ms2","vlf_ms2","lf_ms2","hf_ms2","lf_hf",
        "vlf_peak_hz","lf_peak_hz","hf_peak_hz","breath_rate_brpm",
        "slow_fraction","ese","fs_interp_hz","detrend","window","nperseg","nfft","interp","pipeline_version"
    ]
    placeholders = ",".join(["?"]*len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    values = [
        file_hash,
        m.get("timestamp"),
        m.get("mean_rr_ms"), m.get("hr_bpm"), m.get("sdnn_ms"), m.get("rmssd_ms"), m.get("pnn50"),
        m.get("total_power_ms2"), m.get("vlf_ms2"), m.get("lf_ms2"), m.get("hf_ms2"), m.get("lf_hf"),
        m.get("vlf_peak_hz"), m.get("lf_peak_hz"), m.get("hf_peak_hz"), m.get("breath_rate_brpm"),
        m.get("slow_fraction"), m.get("ese"), m.get("fs_interp_hz"), m.get("detrend"), m.get("window"),
        m.get("nperseg"), m.get("nfft"), m.get("interp"), m.get("pipeline_version")
    ]
    cx.execute(sql, values)

def move_to(dst_dir: Path, src: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    # If name collision, add a suffix
    if target.exists():
        stem = target.stem
        suf = target.suffix
        i = 2
        while True:
            cand = target.with_name(f"{stem} ({i}){suf}")
            if not cand.exists():
                target = cand
                break
            i += 1
    shutil.move(str(src), str(target))
    return target

def process_file(cx: sqlite3.Connection, path: Path, *, dry_run: bool = False, extra_compute_args: list[str] = None):
    extra_compute_args = extra_compute_args or []
    raw_hash = file_sha256(path)

    if already_processed(cx, raw_hash):
        print(f"[SKIP] Already processed: {path.name}")
        move_to(ARCHIVE_DIR, path)
        return

    duration_sec, start_ts = peek_rr_duration_and_start(path)
    if duration_sec is None:
        # Can't parse → quarantine
        print(f"[QUAR] Cannot parse to get duration: {path.name}")
        qpath = move_to(QUARANTINE_DIR, path)
        (qpath.with_suffix(qpath.suffix + ".error.txt")).write_text("Failed to parse duration\n", encoding="utf-8")
        return

    table = choose_table(duration_sec)
    received_at = now_iso()

    if table is None:
        print(f"[DROP] Too short ({duration_sec:.1f}s): {path.name}")
        if not dry_run:
            with cx:
                insert_processed(cx, raw_hash, path.name, received_at, received_at, duration_sec, None, PIPELINE_VERSION_FALLBACK)
        move_to(ARCHIVE_DIR, path)
        return

    # Run compute
    print(f"[RUN ] {path.name}  duration={duration_sec:.1f}s  -> {table}")
    if dry_run:
        move_to(ARCHIVE_DIR, path)
        return

    try:
        metrics = run_compute(path, extra_compute_args)
        mapped = map_metrics_for_table(metrics)
        # Add timestamp to file if not present in returned metrics
        if not mapped.get("timestamp"):
            mapped["timestamp"] = ts_from_filename(path.name) or now_iso()
        # Use compute duration if available; else keep peek value
        duration_final = metrics.get("duration_sec", duration_sec)
        processed_at = now_iso()
        with cx:
            insert_metrics(cx, table, raw_hash, mapped)
            insert_processed(cx, raw_hash, path.name, received_at, processed_at, duration_final, None, mapped.get("pipeline_version", PIPELINE_VERSION_FALLBACK))
        move_to(ARCHIVE_DIR, path)
    except Exception as e:
        print(f"[ERR ] {path.name}: {e}")
        qpath = move_to(QUARANTINE_DIR, path)
        (qpath.with_suffix(qpath.suffix + ".error.txt")).write_text(str(e), encoding="utf-8")

def run_once(dry_run: bool = False, extra_compute_args: list[str] = None):
    ensure_dirs()
    # 1) expand any zips to .txt files first
    expand_any_zips()
    # 2) then process all .txt files
    inbound = sorted([p for p in INBOUND_DIR.iterdir()
                      if p.is_file() and p.suffix.lower() in (".txt", ".csv")])
    if not inbound:
        print("[INFO] No files in inbound.")
        return
    with sqlite3.connect(DB_PATH) as cx:
        for p in inbound:
            process_file(cx, p, dry_run=dry_run, extra_compute_args=extra_compute_args)

def main():
    ap = argparse.ArgumentParser(description="Ingest HRV inbound files, compute metrics, write to SQLite, archive files.")
    ap.add_argument("--run-once", action="store_true", help="Process current files and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to DB; still archive/quarantine files.")
    ap.add_argument("--compute-args", type=str, default="", help="Extra args to pass to compute script (e.g., '--detrend linear --window hamming').")
    ap.add_argument("--cleanup", action="store_true", help="Run archive cleanup after processing.")
    ap.add_argument("--retention-days", type=int, default=30, help="Archive retention window (days).")
    args = ap.parse_args()

    extra = []
    if args.compute_args.strip():
        # naive split
        extra = args.compute_args.split()

    if args.run_once:
        run_once(dry_run=args.dry_run, extra_compute_args=extra)
        if args.cleanup:
            cleanup_archive(args.retention_days)
    else:
        # Polling loop; replace with a proper watcher/cron later
        print("[INFO] Watching inbound directory. Press Ctrl+C to stop.")
        try:
            while True:
                run_once(dry_run=args.dry_run, extra_compute_args=extra)
                if args.cleanup:
                    cleanup_archive(args.retention_days)
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")

if __name__ == "__main__":
    main()
