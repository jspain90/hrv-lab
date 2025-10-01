from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from ..core.db import get_db
from ..core.models import Intervention, QuickToken
from string import Template

router = APIRouter(tags=["dashboard"])

@router.get("/now", response_class=HTMLResponse)
def now_page(request: Request, db: Session = Depends(get_db)):
    base = str(request.base_url).rstrip('/')
    interventions = db.query(Intervention).order_by(Intervention.start_date.desc()).all()

    buttons = []
    for i in interventions:
        tok = db.query(QuickToken).filter_by(intervention_id=i.id, purpose="compliance_quick").first()
        href = f"/c/{tok.token}" if tok else f"/interventions/{i.id}/qr"
        label = f"✓ {i.name}" if tok else f"Make QR for {i.name}"
        buttons.append({"label": label, "href": href})

    return request.app.state.templates.TemplateResponse(
        "now.html",
        {"request": request, "buttons": buttons, "default_metric": "total_power_ln"},
    )

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "now.html",
        {"request": request, "buttons": items, "default_metric": "total_power_ln"},
    )
