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
        if tok:
            buttons.append(
                f"<a href='{base}/c/{tok.token}' "
                "style='display:inline-block;padding:12px 18px;margin:8px;font-size:16px;"
                "border:1px solid #ddd;border-radius:10px;text-decoration:none;'>"
                f"✓ {i.name}</a>"
            )
        else:
            buttons.append(f"<a href='{base}/interventions/{i.id}/qr'>Make QR for {i.name}</a>")
    html_buttons = "".join(buttons) or "<em>No interventions yet.</em>"

    tpl = Template(r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HRV-Lab — Now</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:24px}
    .wrap{max-width:960px;margin:auto}
    .card{padding:16px;border:1px solid #ddd;border-radius:12px;margin:12px 0}
    #chart{max-width:100%;border:1px solid #eee;border-radius:8px;margin-top:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>HRV-Lab — Now</h1>
    <div class="card"><h3>Quick compliance</h3><div>$buttons</div></div>
    <div class="card">
      <h3>Metric</h3>
      <select id="metric">
        <option value="total_power_ln" selected>Total Power (ln)</option>
        <option value="lf">LF</option>
        <option value="hf">HF</option>
        <option value="rmssd">RMSSD</option>
      </select>
      <button id="load">Load</button>
      <canvas id="chart" width="900" height="300"></canvas>
    </div>
  </div>
<script>
async function loadSeries(){
  const metric = document.getElementById('metric').value;
  const res = await fetch('$base/metrics/series?metric=' + encodeURIComponent(metric));
  const data = await res.json();
  draw(data.t, data.v, metric);
}
function draw(ts, vals, label){
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!ts.length){ ctx.fillText('No data for ' + label, 20, 20); return; }
  const W=c.width, H=c.height, pad=40;
  const nums = vals.filter(v=>Number.isFinite(v));
  const vmin = Math.min.apply(null, nums), vmax = Math.max.apply(null, nums);
  const xscale = (i)=> pad + (W-2*pad)*(i/((ts.length-1)||1));
  const yscale = (v)=> H-pad - (H-2*pad)*((v - vmin)/((vmax - vmin)||1));
  // axes
  ctx.strokeStyle='#999'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  // line
  ctx.strokeStyle='#000'; ctx.beginPath(); let started=false;
  for(let i=0;i<vals.length;i++){
    const v=vals[i]; if(!Number.isFinite(v)) continue;
    const x=xscale(i), y=yscale(v);
    if(!started){ ctx.moveTo(x,y); started=true; } else { ctx.lineTo(x,y); }
  }
  ctx.stroke();
}
document.getElementById('load').addEventListener('click', loadSeries);
loadSeries();
</script>
</body>
</html>""")

    return HTMLResponse(tpl.substitute(base=base, buttons=html_buttons))
