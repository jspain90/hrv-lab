from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import Reading, Metric
from ..core.schemas import SeriesOut
import numpy as np
import datetime as dt

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/series", response_model=SeriesOut)
def metric_series(
    metric: str = Query(..., description="metric name e.g. total_power, total_power_ln, rmssd"),
    posture: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    db: Session = Depends(get_db),
):
    base_metric = metric[:-3] if metric.endswith("_ln") else metric
    q = (
        db.query(Reading.started_at, Metric.value)
        .join(Metric, Metric.reading_id == Reading.id)
        .filter(Metric.metric == base_metric)
    )
    if posture:
        q = q.filter(Reading.posture == posture)
    if start:
        q = q.filter(Reading.started_at >= dt.datetime.fromisoformat(start))
    if end:
        q = q.filter(Reading.started_at <= dt.datetime.fromisoformat(end))
    rows = q.order_by(Reading.started_at.asc()).all()

    t: list[str] = []
    v: list[float] = []
    for ts, val in rows:
        t.append(ts.isoformat())
        v.append(float(np.log(val)) if metric.endswith("_ln") and val and val > 0 else float(val))
    return SeriesOut(t=t, v=v)
