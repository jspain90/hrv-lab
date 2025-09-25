from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import User, Intervention, QuickToken
from ..core.schemas import InterventionIn, InterventionOut
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
        expected_metric=payload.expected_metric,
        expected_direction=payload.expected_direction,
        hypothesis_text=payload.hypothesis_text,
        posture_filter=payload.posture_filter,
        time_of_day_filter=payload.time_of_day_filter,
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
