from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import User, Intervention, QuickToken
from ..core.schemas import InterventionIn, InterventionOut, QuickInterventionIn
import uuid

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
