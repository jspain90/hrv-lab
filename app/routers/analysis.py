from __future__ import annotations
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import Intervention, ComplianceEvent
from ..core.schemas import InterventionAnalysisOut
from ..analysis.tau_u import calculate_tau_u
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


@router.get("/intervention-effectiveness", response_model=list[InterventionAnalysisOut])
def get_intervention_effectiveness(db: Session = Depends(get_db)):
    """
    Analyze effectiveness of completed interventions using Tau-U.

    For each completed intervention (active=true, current_date > end_date):
    - Compare intervention period vs matched baseline period (same length)
    - Calculate Tau-U for TP Residual and Standing Trial HR
    - Include compliance percentage
    """
    today = dt.date.today()

    # Get all active interventions
    interventions = db.query(Intervention).filter_by(active=True).all()

    results = []
    for intervention in interventions:
        # Calculate dates
        intervention_start = intervention.start_date
        intervention_duration_days = intervention.duration_weeks * 7
        intervention_end = intervention_start + dt.timedelta(days=intervention_duration_days)

        # Only process completed interventions
        if today <= intervention_end:
            continue

        # Calculate matched baseline period (same length as intervention)
        baseline_start = intervention_start - dt.timedelta(days=intervention_duration_days)
        baseline_end = intervention_start - dt.timedelta(days=1)

        # === TP Residual Analysis ===
        # Baseline data
        tp_baseline_query = text("""
            SELECT tp_residual
            FROM hrv_predictions
            WHERE timestamp >= :baseline_start
            AND timestamp <= :baseline_end
            AND tp_residual IS NOT NULL
            ORDER BY timestamp ASC
        """)
        tp_baseline_result = db.execute(tp_baseline_query, {
            "baseline_start": baseline_start.isoformat(),
            "baseline_end": baseline_end.isoformat() + " 23:59:59"
        }).fetchall()
        tp_baseline = [row[0] for row in tp_baseline_result]

        # Intervention data
        tp_intervention_query = text("""
            SELECT tp_residual
            FROM hrv_predictions
            WHERE timestamp >= :intervention_start
            AND timestamp <= :intervention_end
            AND tp_residual IS NOT NULL
            ORDER BY timestamp ASC
        """)
        tp_intervention_result = db.execute(tp_intervention_query, {
            "intervention_start": intervention_start.isoformat(),
            "intervention_end": intervention_end.isoformat() + " 23:59:59"
        }).fetchall()
        tp_intervention = [row[0] for row in tp_intervention_result]

        # Calculate Tau-U for TP Residual
        tp_tau_u, tp_p_value, tp_effect = calculate_tau_u(tp_baseline, tp_intervention)

        # === Standing Trial HR Analysis ===
        # Baseline data
        hr_baseline_query = text("""
            SELECT hr_bpm
            FROM standing_trials
            WHERE timestamp >= :baseline_start
            AND timestamp <= :baseline_end
            AND hr_bpm IS NOT NULL
            ORDER BY timestamp ASC
        """)
        hr_baseline_result = db.execute(hr_baseline_query, {
            "baseline_start": baseline_start.isoformat(),
            "baseline_end": baseline_end.isoformat() + " 23:59:59"
        }).fetchall()
        hr_baseline = [row[0] for row in hr_baseline_result]

        # Intervention data
        hr_intervention_query = text("""
            SELECT hr_bpm
            FROM standing_trials
            WHERE timestamp >= :intervention_start
            AND timestamp <= :intervention_end
            AND hr_bpm IS NOT NULL
            ORDER BY timestamp ASC
        """)
        hr_intervention_result = db.execute(hr_intervention_query, {
            "intervention_start": intervention_start.isoformat(),
            "intervention_end": intervention_end.isoformat() + " 23:59:59"
        }).fetchall()
        hr_intervention = [row[0] for row in hr_intervention_result]

        # Calculate Tau-U for Standing HR
        hr_tau_u, hr_p_value, hr_effect = calculate_tau_u(hr_baseline, hr_intervention)

        # === Compliance Calculation ===
        total_expected_compliance = intervention.freq_per_week * intervention.duration_weeks
        completed_compliance = db.query(ComplianceEvent).filter(
            ComplianceEvent.intervention_id == intervention.id,
            ComplianceEvent.ts >= intervention_start,
            ComplianceEvent.ts <= intervention_end
        ).count()

        percent_compliance = (completed_compliance / total_expected_compliance * 100) if total_expected_compliance > 0 else 0.0

        # Store results
        results.append(InterventionAnalysisOut(
            intervention_id=intervention.id,
            intervention_name=intervention.name,
            start_date=intervention_start.isoformat(),
            end_date=intervention_end.isoformat(),
            baseline_start=baseline_start.isoformat(),
            baseline_end=baseline_end.isoformat(),
            tp_residual_tau_u=round(tp_tau_u, 3),
            tp_residual_p_value=round(tp_p_value, 4),
            tp_residual_effect_size=tp_effect,
            standing_hr_tau_u=round(hr_tau_u, 3),
            standing_hr_p_value=round(hr_p_value, 4),
            standing_hr_effect_size=hr_effect,
            percent_compliance=round(percent_compliance, 1),
            baseline_n=len(tp_baseline),  # Use TP baseline as representative sample size
            intervention_n=len(tp_intervention)
        ))

    return results
