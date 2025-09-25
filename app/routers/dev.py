from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..core.db import get_db

router = APIRouter(prefix="/dev", tags=["dev"])

@router.get("/metric_counts")
def metric_counts(db: Session = Depends(get_db)):
    rows = db.execute("SELECT metric, COUNT(*) as cnt FROM metrics GROUP BY metric ORDER BY cnt DESC").fetchall()
    return [{"metric": r[0], "count": r[1]} for r in rows]

