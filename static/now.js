async function loadSeries(){
  const metric = document.getElementById('metric').value;
  const res = await fetch('/metrics/series?metric=' + encodeURIComponent(metric));
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

//refresh button
const btn = document.getElementById('refreshBtn');
const out = document.getElementById('refreshOut');

async function refreshData() {
  if (!btn) return;
  btn.disabled = true;
  out.textContent = "Refreshing…";
  try {
    const res = await fetch('/refresh', { method: 'POST' });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || res.statusText);
    }
    const json = await res.json();
    out.textContent =
      `Done.\n` +
      `Fetched zips: ${json.fetched_zips}\n` +
      `Processed files: ${json.processed_files}\n` +
      `Quarantined: ${json.quarantined}\n` +
      `Archive deletions: ${json.archived_deleted}`;
  } catch (e) {
    out.textContent = "Error: " + (e.message || e);
  } finally {
    btn.disabled = false;
  }
}

if (btn) btn.addEventListener('click', refreshData);


  // axes
  ctx.strokeStyle='#999'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();

  // line
  ctx.beginPath();
  let started=false;
  for(let i=0;i<vals.length;i++){
    const v=vals[i]; if(!Number.isFinite(v)) continue;
    const x=xscale(i), y=yscale(v);
    if(!started){ ctx.moveTo(x,y); started=true; } else { ctx.lineTo(x,y); }
  }
  ctx.stroke();
}

document.getElementById('load').addEventListener('click', loadSeries);
loadSeries();
