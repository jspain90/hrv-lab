from __future__ import annotations
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from ..core.db import get_db
from pydantic import BaseModel
import datetime as dt
import numpy as np

router = APIRouter(prefix="/analysis", tags=["analysis"])


class ResidualRecord(BaseModel):
    timestamp: str
    tp_actual: float | None
    tp_predicted: float | None
    tp_residual: float | None
    lfhf_actual: float | None
    lfhf_predicted: float | None
    lfhf_residual: float | None


class ResidualsSummary(BaseModel):
    count: int
    tp_mean_residual: float | None
    tp_std_residual: float | None
    lfhf_mean_residual: float | None
    lfhf_std_residual: float | None
    date_range: tuple[str, str] | None


@router.get("/residuals", response_model=list[ResidualRecord])
def get_residuals(
    start: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, le=5000, description="Max records to return"),
    db: Session = Depends(get_db)
):
    """
    Get HRV residuals (actual - predicted baseline).

    Residuals represent deviation from expected HRV controlling for:
    - Weather conditions
    - Time-of-day patterns
    - Temporal dependencies

    Positive residuals = better than expected HRV
    Negative residuals = worse than expected HRV
    """
    query = """
        SELECT
            timestamp,
            tp_actual,
            tp_predicted,
            tp_residual,
            lfhf_actual,
            lfhf_predicted,
            lfhf_residual
        FROM hrv_predictions
        WHERE 1=1
    """

    conditions = []
    if start:
        conditions.append(f"timestamp >= '{start}'")
    if end:
        conditions.append(f"timestamp <= '{end} 23:59:59'")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += f" ORDER BY timestamp ASC LIMIT {limit}"

    result = db.execute(text(query))
    rows = result.fetchall()

    if not rows:
        return []

    return [
        ResidualRecord(
            timestamp=row[0],
            tp_actual=row[1],
            tp_predicted=row[2],
            tp_residual=row[3],
            lfhf_actual=row[4],
            lfhf_predicted=row[5],
            lfhf_residual=row[6]
        )
        for row in rows
    ]


@router.get("/residuals/summary", response_model=ResidualsSummary)
def residuals_summary(
    start: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get summary statistics for residuals."""
    query = """
        SELECT
            COUNT(*) as count,
            AVG(tp_residual) as tp_mean,
            STDEV(tp_residual) as tp_std,
            AVG(lfhf_residual) as lfhf_mean,
            STDEV(lfhf_residual) as lfhf_std,
            MIN(timestamp) as min_date,
            MAX(timestamp) as max_date
        FROM hrv_predictions
        WHERE 1=1
    """

    conditions = []
    if start:
        conditions.append(f"timestamp >= '{start}'")
    if end:
        conditions.append(f"timestamp <= '{end} 23:59:59'")

    if conditions:
        query += " AND " + " AND ".join(conditions)

    result = db.execute(text(query))
    row = result.fetchone()

    # SQLite doesn't have STDEV, calculate manually
    if row and row[0] > 0:
        # Get values for std calculation
        std_query = """
            SELECT tp_residual, lfhf_residual
            FROM hrv_predictions
            WHERE 1=1
        """
        if conditions:
            std_query += " AND " + " AND ".join(conditions)

        residuals = db.execute(text(std_query)).fetchall()
        tp_residuals = [r[0] for r in residuals if r[0] is not None]
        lfhf_residuals = [r[1] for r in residuals if r[1] is not None]

        tp_std = float(np.std(tp_residuals)) if tp_residuals else None
        lfhf_std = float(np.std(lfhf_residuals)) if lfhf_residuals else None

        return ResidualsSummary(
            count=row[0],
            tp_mean_residual=round(row[1], 4) if row[1] else None,
            tp_std_residual=round(tp_std, 4) if tp_std else None,
            lfhf_mean_residual=round(row[3], 4) if row[3] else None,
            lfhf_std_residual=round(lfhf_std, 4) if lfhf_std else None,
            date_range=(row[5], row[6]) if row[5] and row[6] else None
        )

    return ResidualsSummary(
        count=0,
        tp_mean_residual=None,
        tp_std_residual=None,
        lfhf_mean_residual=None,
        lfhf_std_residual=None,
        date_range=None
    )


@router.get("/intervention/{intervention_id}/residuals")
def intervention_residuals(
    intervention_id: int,
    db: Session = Depends(get_db)
):
    """
    Get residuals during an intervention period vs baseline.

    This is the key analysis for determining intervention effectiveness.
    """
    # Get intervention details
    intervention_query = text("""
        SELECT start_date, duration_weeks
        FROM interventions
        WHERE id = :id
    """)
    intervention = db.execute(intervention_query, {"id": intervention_id}).fetchone()

    if not intervention:
        raise HTTPException(404, "Intervention not found")

    start_date = intervention[0]
    duration_weeks = intervention[1]

    # Calculate date range
    from datetime import datetime, timedelta
    start = datetime.fromisoformat(str(start_date))
    end = start + timedelta(weeks=duration_weeks)

    # Get residuals during intervention
    residuals_query = text("""
        SELECT
            AVG(tp_residual) as avg_tp_residual,
            AVG(lfhf_residual) as avg_lfhf_residual,
            COUNT(*) as n_readings
        FROM hrv_predictions
        WHERE timestamp >= :start AND timestamp <= :end
    """)

    result = db.execute(residuals_query, {
        "start": start.isoformat(),
        "end": end.isoformat()
    }).fetchone()

    return {
        "intervention_id": intervention_id,
        "start_date": str(start_date),
        "duration_weeks": duration_weeks,
        "n_readings": result[2] if result else 0,
        "avg_tp_residual": round(result[0], 4) if result and result[0] else None,
        "avg_lfhf_residual": round(result[1], 4) if result and result[1] else None,
        "interpretation": {
            "tp": "better than baseline" if result and result[0] and result[0] > 0 else "worse than baseline" if result and result[0] and result[0] < 0 else "no change",
            "lfhf": "better balance" if result and result[1] and abs(result[1]) < 0.1 else "dysregulated"
        }
    }
