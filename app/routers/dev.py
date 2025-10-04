from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from ..core.db import get_db

router = APIRouter(prefix="/dev", tags=["dev"])

@router.get("/metric_counts")
def metric_counts(db: Session = Depends(get_db)):
    # Count records across all three HRV tables
    hrv_count = db.execute(text("SELECT COUNT(*) FROM hrv_results")).scalar()
    standing_count = db.execute(text("SELECT COUNT(*) FROM standing_trials")).scalar()
    exercise_count = db.execute(text("SELECT COUNT(*) FROM exercise_sessions")).scalar()

    return {
        "hrv_results": hrv_count,
        "standing_trials": standing_count,
        "exercise_sessions": exercise_count,
        "total": hrv_count + standing_count + exercise_count
    }

