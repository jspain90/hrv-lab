from __future__ import annotations
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.schemas import SeriesOut, ComplianceEventOut, DualSeriesOut
from ..core.models import ComplianceEvent, Intervention
import numpy as np
import datetime as dt

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/series", response_model=SeriesOut)
def metric_series(
    metric: str = Query(..., description="metric name e.g. total_power_ms2, total_power_ln, rmssd_ms, tp_residual, lfhf_residual, tp_residual_7d, lfhf_residual_7d"),
    posture: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    db: Session = Depends(get_db),
):
    # Check if requesting 7-day residuals
    if metric.endswith('_7d'):
        base_residual = metric[:-3]  # Remove '_7d' suffix
        # Calculate date 7 days ago
        seven_days_ago = (dt.datetime.now() - dt.timedelta(days=7)).strftime('%Y-%m-%d')

        query = """
            SELECT timestamp, {metric} as value
            FROM hrv_predictions
            WHERE timestamp >= '{start_date}'
        """.format(metric=base_residual, start_date=seven_days_ago)

        query += " ORDER BY timestamp ASC"

        result = db.execute(text(query))
        rows = result.fetchall()

        t: list[str] = []
        v: list[float] = []
        for row in rows:
            ts_str, val = row
            if ts_str and val is not None:
                t.append(ts_str)
                v.append(float(val))

        return SeriesOut(t=t, v=v)

    # Check if requesting residuals from predictions table
    if metric in ['tp_residual', 'lfhf_residual']:
        query = """
            SELECT timestamp, {metric} as value
            FROM hrv_predictions
            WHERE 1=1
        """.format(metric=metric)

        conditions = []
        if start:
            conditions.append(f"timestamp >= '{start}'")
        if end:
            conditions.append(f"timestamp <= '{end} 23:59:59'")

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += " ORDER BY timestamp ASC"

        result = db.execute(text(query))
        rows = result.fetchall()

        t: list[str] = []
        v: list[float] = []
        for row in rows:
            ts_str, val = row
            if ts_str and val is not None:
                t.append(ts_str)
                v.append(float(val))

        return SeriesOut(t=t, v=v)

    # Original logic for HRV metrics
    # Determine if we need to apply ln transform
    apply_ln = metric.endswith("_ln")
    base_metric = metric[:-3] if apply_ln else metric

    # Query hrv_results table
    query = """
        SELECT timestamp, {metric} as value FROM hrv_results
        WHERE 1=1
    """.format(metric=base_metric)

    conditions = []
    if start:
        conditions.append(f"timestamp >= '{start}'")
    if end:
        conditions.append(f"timestamp <= '{end} 23:59:59'")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += " ORDER BY timestamp ASC"

    result = db.execute(text(query))
    rows = result.fetchall()

    t: list[str] = []
    v: list[float] = []
    for row in rows:
        ts_str, val = row
        if ts_str and val is not None:
            t.append(ts_str)
            if apply_ln and val > 0:
                v.append(float(np.log(val)))
            else:
                v.append(float(val))

    return SeriesOut(t=t, v=v)

@router.get("/compliance-events", response_model=list[ComplianceEventOut])
def get_compliance_events(
    start: str | None = Query(None),
    end: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get all compliance events with intervention details, optionally filtered by date range."""
    query = db.query(
        ComplianceEvent.ts,
        ComplianceEvent.intervention_id,
        Intervention.name
    ).join(Intervention, ComplianceEvent.intervention_id == Intervention.id)

    if start:
        # Parse start date and convert to datetime
        start_dt = dt.datetime.strptime(start, '%Y-%m-%d')
        query = query.filter(ComplianceEvent.ts >= start_dt)

    if end:
        # Parse end date and add 23:59:59 to include the entire day
        end_dt = dt.datetime.strptime(end, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        query = query.filter(ComplianceEvent.ts <= end_dt)

    results = query.order_by(ComplianceEvent.ts.asc()).all()

    events = []
    for ts, intervention_id, intervention_name in results:
        events.append(ComplianceEventOut(
            timestamp=ts.strftime('%Y-%m-%d %H:%M:%S'),
            intervention_id=intervention_id,
            intervention_name=intervention_name
        ))

    return events

@router.get("/standing-trials", response_model=DualSeriesOut)
def standing_trials_series(
    start: str | None = Query(None),
    end: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get standing trial data with hr_bpm and lf_hf for dual-axis visualization."""
    query = """
        SELECT timestamp, hr_bpm, lf_hf
        FROM standing_trials
        WHERE 1=1
    """

    conditions = []
    if start:
        conditions.append(f"timestamp >= '{start}'")
    if end:
        conditions.append(f"timestamp <= '{end} 23:59:59'")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += " ORDER BY timestamp ASC"

    result = db.execute(text(query))
    rows = result.fetchall()

    t: list[str] = []
    v1: list[float] = []
    v2: list[float] = []
    for row in rows:
        ts_str, hr_bpm, lf_hf = row
        if ts_str and hr_bpm is not None and lf_hf is not None:
            t.append(ts_str)
            v1.append(float(hr_bpm))
            v2.append(float(lf_hf))

    return DualSeriesOut(t=t, v1=v1, v2=v2)

@router.get("/standing-trials-7d", response_model=DualSeriesOut)
def standing_trials_7d_series(
    db: Session = Depends(get_db),
):
    """Get standing trial data for last 7 days with hr_bpm and lf_hf."""
    seven_days_ago = (dt.datetime.now() - dt.timedelta(days=7)).strftime('%Y-%m-%d')

    query = """
        SELECT timestamp, hr_bpm, lf_hf
        FROM standing_trials
        WHERE timestamp >= '{start_date}'
        ORDER BY timestamp ASC
    """.format(start_date=seven_days_ago)

    result = db.execute(text(query))
    rows = result.fetchall()

    t: list[str] = []
    v1: list[float] = []
    v2: list[float] = []
    for row in rows:
        ts_str, hr_bpm, lf_hf = row
        if ts_str and hr_bpm is not None and lf_hf is not None:
            t.append(ts_str)
            v1.append(float(hr_bpm))
            v2.append(float(lf_hf))

    return DualSeriesOut(t=t, v1=v1, v2=v2)
