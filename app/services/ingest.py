from __future__ import annotations
import io, os, glob, hashlib
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from sqlalchemy.orm import Session
from ..core.models import Reading, Metric
from ..core.settings import settings

COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "start time", "date time start", "date_time_start"],
    "posture": ["posture", "position"],
    "duration_s": ["duration", "duration_s", "duration (s)", "reading length"],
    "avg_hr": ["avg_hr", "average hr", "average_hr", "mean hr", "hr", "bpm"],
    "rmssd": ["rmssd", "rmssd (ms)"],
    "sdnn": ["sdnn", "sdnn (ms)"],
    "pnn50": ["pnn50", "%nn50", "pn50"],
    "lf": ["lf", "lf power", "low frequency power", "lf (ms^2)"],
    "hf": ["hf", "hf power", "high frequency power", "hf (ms^2)"],
    "total_power": ["total power", "total_power", "tp", "total power (ms^2)"],
    "lf_hf": ["lf/hf", "lf_hf", "lfhf", "lf/hf ratio"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            a_lower = a.lower()
            if a_lower in lower_cols:
                mapping[lower_cols[a_lower]] = canon
                break
    return df.rename(columns=mapping)

def parse_elite_csv(content: bytes) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(content))
    if df.empty:
        return []
    df = _normalize_columns(df)
    if "timestamp" not in df.columns:
        raise ValueError("CSV missing a timestamp-like column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])  # discard rows without valid time

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        row: Dict[str, Any] = {
            "started_at": pd.Timestamp(r["timestamp"]).to_pydatetime(),
            "posture": str(r.get("posture", "") or "") or None,
            "duration_s": int(r.get("duration_s", 0) or 0) if not pd.isna(r.get("duration_s", np.nan)) else None,
            "metrics": {}
        }
        for m in ["avg_hr", "rmssd", "sdnn", "pnn50", "lf", "hf", "total_power", "lf_hf"]:
            val = r.get(m, np.nan)
            if pd.isna(val):
                continue
            try:
                row["metrics"][m] = float(val)
            except Exception:
                pass
        rows.append(row)
    return rows

def ingest_content(content: bytes, filename: str, db: Session) -> int:
    # archive by content hash
    digest = hashlib.sha256(content).hexdigest()
    os.makedirs(settings.archive_dir, exist_ok=True)
    with open(os.path.join(settings.archive_dir, f"{digest}_{filename}"), "wb") as f:
        f.write(content)

    rows = parse_elite_csv(content)
    inserted = 0
    for r in rows:
        reading = Reading(
            started_at=r["started_at"],
            posture=(r.get("posture") or None),
            duration_s=r.get("duration_s"),
            source_file=filename,
            source_type="elite_csv",
        )
        for m, v in r["metrics"].items():
            unit = "ms^2" if m in {"lf", "hf", "total_power"} else ("ms" if m in {"sdnn", "rmssd"} else "")
            reading.metrics.append(Metric(metric=m, value=float(v), unit=unit))
        db.add(reading)
        inserted += 1
    db.commit()
    return inserted
