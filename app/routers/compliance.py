from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import ComplianceEvent, Intervention, QuickToken
from ..core.schemas import ComplianceIn
import datetime as dt
import qrcode, io

router = APIRouter(prefix="", tags=["compliance"])

@router.post("/compliance")
def post_compliance(payload: ComplianceIn, db: Session = Depends(get_db)):
    itv = db.get(Intervention, payload.intervention_id)
    if not itv:
        raise HTTPException(404, "Intervention not found")
    ev = ComplianceEvent(
        intervention_id=itv.id,
        ts=payload.ts or dt.datetime.now(dt.timezone.utc),
        value_bool=payload.value_bool,
        notes=payload.notes,
        source="api",
    )
    db.add(ev)
    db.commit()
    return {"ok": True, "id": ev.id}

@router.get("/c/{token}")
def quick_compliance(token: str, db: Session = Depends(get_db)):
    t = db.query(QuickToken).filter_by(token=token, purpose="compliance_quick").one_or_none()
    if not t:
        raise HTTPException(404, "Invalid token")
    if t.expires_at and dt.datetime.now(dt.timezone.utc) > t.expires_at:
        raise HTTPException(410, "Token expired")
    if t.uses is not None and t.uses <= 0:
        raise HTTPException(410, "Token exhausted")
    ev = ComplianceEvent(intervention_id=t.intervention_id, source="quick", value_bool=True)
    db.add(ev)
    if t.uses is not None:
        t.uses -= 1
    db.commit()
    return PlainTextResponse("✓ Recorded. Have a great day.")

@router.get("/interventions/{iid}/qr")
def intervention_qr(iid: int, request: Request, db: Session = Depends(get_db)):
    itv = db.get(Intervention, iid)
    if not itv:
        raise HTTPException(404, "Intervention not found")
    token = (
        db.query(QuickToken)
        .filter_by(intervention_id=iid, purpose="compliance_quick")
        .order_by(QuickToken.id.asc())
        .first()
    )
    if not token:
        token = QuickToken(intervention_id=iid)
        db.add(token)
        db.commit()
    base = str(request.base_url).rstrip('/')
    quick_url = f"{base}/c/{token.token}"
    img = qrcode.make(quick_url)
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
