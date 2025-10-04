from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import User, Intervention, QuickToken, ComplianceEvent
from ..core.schemas import InterventionIn, InterventionOut, QuickInterventionIn, ComplianceStatsOut
import uuid
import datetime as dt

router = APIRouter(prefix="/interventions", tags=["interventions"])

@router.post("", response_model=InterventionOut)
def create_intervention(payload: InterventionIn, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(name=payload.user_name).one_or_none()
    if not user:
        user = User(name=payload.user_name)
        db.add(user)
        db.flush()
    itv = Intervention(
        user_id=user.id,
        name=payload.name,
        start_date=payload.start_date,
        duration_weeks=payload.duration_weeks,
        freq_per_week=payload.freq_per_week,
        active=payload.active,
    )
    db.add(itv)
    db.commit()
    db.refresh(itv)

    token = QuickToken(intervention_id=itv.id, token=uuid.uuid4().hex, purpose="compliance_quick")
    db.add(token)
    db.commit()
    return itv

@router.get("", response_model=list[InterventionOut])
def list_interventions(db: Session = Depends(get_db)):
    return db.query(Intervention).order_by(Intervention.id.desc()).all()

@router.patch("/{intervention_id}/toggle-active")
def toggle_intervention_active(intervention_id: int, db: Session = Depends(get_db)):
    """Toggle the active status of an intervention."""
    intervention = db.query(Intervention).filter_by(id=intervention_id).first()
    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")

    intervention.active = not intervention.active
    db.commit()
    db.refresh(intervention)

    return {"success": True, "intervention_id": intervention.id, "active": intervention.active}

@router.post("/quick", status_code=201)
def create_quick_intervention(payload: QuickInterventionIn, db: Session = Depends(get_db)):
    # Get or create default user
    user = db.query(User).filter_by(name="owner").first()
    if not user:
        user = User(name="owner")
        db.add(user)
        db.flush()

    # Create intervention (active by default)
    itv = Intervention(
        user_id=user.id,
        name=payload.name,
        start_date=payload.start_date,
        duration_weeks=payload.duration_weeks,
        freq_per_week=payload.frequency_per_week,
        active=True,
    )
    db.add(itv)
    db.commit()
    db.refresh(itv)

    # Generate quick token for compliance
    token = QuickToken(
        intervention_id=itv.id,
        token=uuid.uuid4().hex,
        purpose="compliance_quick"
    )
    db.add(token)
    db.commit()

    return {"success": True, "intervention_id": itv.id, "token": token.token}

@router.put("/{intervention_id}", response_model=InterventionOut)
def update_intervention(
    intervention_id: int,
    payload: QuickInterventionIn,
    db: Session = Depends(get_db)
):
    """Update intervention details (name, start_date, duration_weeks, freq_per_week)."""
    intervention = db.query(Intervention).filter_by(id=intervention_id).first()
    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")

    # Update fields
    intervention.name = payload.name
    intervention.start_date = payload.start_date
    intervention.duration_weeks = payload.duration_weeks
    intervention.freq_per_week = payload.frequency_per_week

    db.commit()
    db.refresh(intervention)

    return intervention

@router.delete("/{intervention_id}")
def delete_intervention(intervention_id: int, db: Session = Depends(get_db)):
    """
    Delete an intervention and all associated data.

    Deletes in order:
    1. Compliance events linked to this intervention
    2. Quick tokens linked to this intervention
    3. The intervention itself
    """
    intervention = db.query(Intervention).filter_by(id=intervention_id).first()
    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")

    # Delete associated compliance events first
    compliance_count = db.query(ComplianceEvent).filter_by(intervention_id=intervention_id).count()
    db.query(ComplianceEvent).filter_by(intervention_id=intervention_id).delete()

    # Delete associated quick tokens
    token_count = db.query(QuickToken).filter_by(intervention_id=intervention_id).count()
    db.query(QuickToken).filter_by(intervention_id=intervention_id).delete()

    # Finally delete the intervention
    db.delete(intervention)
    db.commit()

    return {
        "success": True,
        "intervention_id": intervention_id,
        "deleted_compliance_events": compliance_count,
        "deleted_tokens": token_count
    }

@router.get("/compliance-stats", response_model=list[ComplianceStatsOut])
def get_compliance_stats(filter: str = "active", db: Session = Depends(get_db)):
    """
    Get compliance statistics for interventions.

    filter: 'active' for ongoing interventions, 'completed' for finished interventions
    """
    today = dt.date.today()

    # Get all active interventions
    interventions = db.query(Intervention).filter_by(active=True).all()

    results = []
    for intervention in interventions:
        # Calculate end_date
        end_date = intervention.start_date + dt.timedelta(days=intervention.duration_weeks * 7)

        # Apply filter
        if filter == "active":
            # Active: current_date > start_date AND current_date < end_date
            if not (today > intervention.start_date and today < end_date):
                continue
        elif filter == "completed":
            # Completed: current_date > end_date
            if not (today > end_date):
                continue
        else:
            continue

        # Count compliance events during the intervention period only
        completed_compliance = db.query(ComplianceEvent).filter(
            ComplianceEvent.intervention_id == intervention.id,
            ComplianceEvent.ts >= intervention.start_date,
            ComplianceEvent.ts <= end_date
        ).count()

        # Calculate metrics
        total_expected_compliance = intervention.freq_per_week * intervention.duration_weeks
        trial_length_days = (end_date - intervention.start_date).days
        trial_days_completed = (today - intervention.start_date).days

        # Ensure we don't exceed 100% for completed interventions
        if trial_days_completed > trial_length_days:
            trial_days_completed = trial_length_days

        percent_trial_completed = (trial_days_completed / trial_length_days * 100) if trial_length_days > 0 else 0
        percent_expected_compliance = (total_expected_compliance * percent_trial_completed / 100)

        # For active interventions: compare against expected at this point
        # For completed interventions: compare against total expected
        if filter == "active":
            percent_completed_compliance = (completed_compliance / percent_expected_compliance * 100) if percent_expected_compliance > 0 else 0
        else:
            percent_completed_compliance = (completed_compliance / total_expected_compliance * 100) if total_expected_compliance > 0 else 0

        results.append(ComplianceStatsOut(
            intervention_id=intervention.id,
            intervention_name=intervention.name,
            start_date=intervention.start_date.isoformat(),
            end_date=end_date.isoformat(),
            total_expected_compliance=total_expected_compliance,
            completed_compliance=completed_compliance,
            percent_trial_completed=round(percent_trial_completed, 1),
            percent_expected_compliance=round(percent_expected_compliance, 1),
            percent_completed_compliance=round(percent_completed_compliance, 1)
        ))

    return results
