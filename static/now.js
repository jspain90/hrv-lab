// Tab switching
document.querySelectorAll('.tab-button').forEach(button => {
  button.addEventListener('click', () => {
    const targetTab = button.dataset.tab;

    // Update active button
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');

    // Update active content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById('tab-' + targetTab).classList.add('active');
  });
});

async function loadSeries(chartNum){
  const metric = document.getElementById(`metric${chartNum}`).value;

  // Determine if it's a dual-series metric (standing trials)
  const isDualSeries = metric.startsWith('standing_');

  let seriesEndpoint;
  if (metric === 'standing_hr') {
    seriesEndpoint = '/metrics/standing-trials';
  } else if (metric === 'standing_hr_7d') {
    seriesEndpoint = '/metrics/standing-trials-7d';
  } else {
    seriesEndpoint = '/metrics/series?metric=' + encodeURIComponent(metric);
  }

  // Fetch both series data and compliance events in parallel
  const [seriesRes, eventsRes] = await Promise.all([
    fetch(seriesEndpoint),
    fetch('/metrics/compliance-events')
  ]);

  const seriesData = await seriesRes.json();
  const eventsData = await eventsRes.json();

  if (isDualSeries) {
    drawDualSeries(chartNum, seriesData.t, seriesData.v1, seriesData.v2, metric, eventsData);
  } else {
    draw(chartNum, seriesData.t, seriesData.v, metric, eventsData);
  }
  updateLegend(chartNum, eventsData);
}

// Color palette for interventions - deterministic based on intervention_id
const INTERVENTION_COLORS = [
  '#e74c3c', // red
  '#3498db', // blue
  '#2ecc71', // green
  '#9b59b6', // purple
  '#1abc9c', // teal
  '#e91e63', // pink
  '#f39c12', // orange
  '#34495e', // dark gray
  '#16a085', // dark teal
  '#8e44ad', // dark purple
  '#c0392b', // dark red
  '#27ae60', // dark green
];

function getInterventionColor(interventionId) {
  return INTERVENTION_COLORS[interventionId % INTERVENTION_COLORS.length];
}

function updateLegend(chartNum, complianceEvents) {
  const legendContainer = document.getElementById(`legend${chartNum}`);
  const legendItems = legendContainer.querySelector('.legend-items');

  if (!complianceEvents || complianceEvents.length === 0) {
    legendContainer.style.display = 'none';
    return;
  }

  // Extract unique interventions from compliance events
  const interventionMap = new Map();
  complianceEvents.forEach(event => {
    if (!interventionMap.has(event.intervention_id)) {
      interventionMap.set(event.intervention_id, {
        id: event.intervention_id,
        name: event.intervention_name,
        color: getInterventionColor(event.intervention_id)
      });
    }
  });

  // Build legend HTML
  let html = '';
  interventionMap.forEach(intervention => {
    html += `
      <div style="display: flex; align-items: center; gap: 6px;">
        <div style="width: 20px; height: 3px; background-color: ${intervention.color}; opacity: 0.3;"></div>
        <span style="font-size: 13px;">${intervention.name}</span>
      </div>
    `;
  });

  legendItems.innerHTML = html;
  legendContainer.style.display = 'block';
}

function drawDualSeries(chartNum, ts, primaryVals, secondaryVals, label, complianceEvents = []){
  const c = document.getElementById(`chart${chartNum}`);
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!ts.length){ ctx.fillText('No data for ' + label, 20, 20); return; }

  const W=c.width, H=c.height, pad=60;

  // Check if this is a 7-day view
  const is7Day = label.includes('_7d');

  // Convert timestamps to actual time values for proper spacing
  const timeValues = ts.map(t => new Date(t.replace(' ', 'T')).getTime());
  const minTime = Math.min(...timeValues);
  const maxTime = Math.max(...timeValues);
  const timeRange = maxTime - minTime;

  const xscale = (i) => {
    if (timeRange === 0) {
      // Single data point - center it
      return pad + (W-2*pad) / 2;
    }
    const time = timeValues[i];
    return pad + (W-2*pad)*((time - minTime)/timeRange);
  };

  // Scale for primary values (hr_bpm)
  const primary_nums = primaryVals.filter(v=>Number.isFinite(v));
  const primary_min = Math.min.apply(null, primary_nums);
  const primary_max = Math.max.apply(null, primary_nums);
  const yscale_primary = (v)=> H-pad - (H-2*pad)*((v - primary_min)/((primary_max - primary_min)||1));

  // Scale for secondary values (lf_hf) - uses same visual space
  const secondary_nums = secondaryVals.filter(v=>Number.isFinite(v));
  const secondary_min = Math.min.apply(null, secondary_nums);
  const secondary_max = Math.max.apply(null, secondary_nums);
  const yscale_secondary = (v)=> H-pad - (H-2*pad)*((v - secondary_min)/((secondary_max - secondary_min)||1));

  // Add time labels on x-axis (reuse logic from draw function)
  if (ts.length > 0) {
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    if (is7Day) {
      // For 7-day views: show day of week
      const dateGroups = new Map();
      ts.forEach((timestamp, idx) => {
        const date = new Date(timestamp.replace(' ', 'T'));
        const dateKey = date.toDateString();
        if (!dateGroups.has(dateKey)) {
          dateGroups.set(dateKey, { date: date, firstIdx: idx, lastIdx: idx });
        } else {
          const group = dateGroups.get(dateKey);
          group.lastIdx = idx;
        }
      });

      ctx.save();
      const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      dateGroups.forEach((group) => {
        const dayName = dayNames[group.date.getDay()];
        const midIdx = (group.firstIdx + group.lastIdx) / 2;
        const x = xscale(midIdx);
        const y = H - 10;
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(dayName, 0, 0);
        ctx.rotate(Math.PI / 4);
        ctx.translate(-x, -y);
      });
      ctx.restore();
    } else {
      // For all other views: show MM/YYYY
      const monthGroups = new Map();
      ts.forEach((timestamp, idx) => {
        const date = new Date(timestamp.replace(' ', 'T'));
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        if (!monthGroups.has(monthKey)) {
          monthGroups.set(monthKey, { date: date, firstIdx: idx, lastIdx: idx });
        } else {
          const group = monthGroups.get(monthKey);
          group.lastIdx = idx;
        }
      });

      ctx.save();
      monthGroups.forEach((group) => {
        const month = String(group.date.getMonth() + 1).padStart(2, '0');
        const year = group.date.getFullYear();
        const monthLabel = `${month}/${year}`;
        const midIdx = (group.firstIdx + group.lastIdx) / 2;
        const x = xscale(midIdx);
        const y = H - 25;
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(monthLabel, 0, 0);
        ctx.rotate(Math.PI / 4);
        ctx.translate(-x, -y);
      });
      ctx.restore();
    }
    ctx.textAlign = 'left';
  }

  // Draw lf_hf as area chart with 20% opacity (background layer)
  if (secondaryVals.length > 0) {
    ctx.fillStyle = 'rgba(100, 100, 100, 0.2)'; // Gray with 20% opacity
    ctx.beginPath();

    // Start at bottom-left of first point
    const firstX = xscale(0);
    ctx.moveTo(firstX, H - pad);

    // Draw line up to first data point
    if (Number.isFinite(secondaryVals[0])) {
      ctx.lineTo(firstX, yscale_secondary(secondaryVals[0]));
    }

    // Draw the area shape following the data
    for (let i = 1; i < secondaryVals.length; i++) {
      if (Number.isFinite(secondaryVals[i])) {
        const x = xscale(i);
        const y = yscale_secondary(secondaryVals[i]);
        ctx.lineTo(x, y);
      }
    }

    // Close the area by going back down to bottom at last point
    const lastX = xscale(secondaryVals.length - 1);
    ctx.lineTo(lastX, H - pad);

    ctx.closePath();
    ctx.fill();
  }

  // Draw compliance event vertical lines
  if (complianceEvents && complianceEvents.length > 0) {
    const timestampToIndex = (eventTs) => {
      for (let i = 0; i < ts.length; i++) {
        if (ts[i] >= eventTs) return i;
      }
      return ts.length - 1;
    };

    complianceEvents.forEach(event => {
      const idx = timestampToIndex(event.timestamp);
      const x = xscale(idx);
      const color = getInterventionColor(event.intervention_id);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, H - pad);
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    });
  }

  // Calculate linear regression on primary values (hr_bpm)
  function linearRegression(xArr, yArr) {
    const n = xArr.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
      sumX += xArr[i];
      sumY += yArr[i];
      sumXY += xArr[i] * yArr[i];
      sumX2 += xArr[i] * xArr[i];
      sumY2 += yArr[i] * yArr[i];
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const meanY = sumY / n;
    const ssTotal = sumY2 - n * meanY * meanY;
    const ssResidual = yArr.reduce((sum, y, i) => {
      const predicted = intercept + slope * xArr[i];
      return sum + Math.pow(y - predicted, 2);
    }, 0);
    const rSquared = 1 - (ssResidual / ssTotal);

    return { slope, intercept, rSquared };
  }

  // Draw orange regression line for hr_bpm
  const validIndices = [];
  const validPrimaryVals = [];
  for(let i=0; i<primaryVals.length; i++){
    if(Number.isFinite(primaryVals[i])) {
      validIndices.push(i);
      validPrimaryVals.push(primaryVals[i]);
    }
  }

  if (validIndices.length > 1) {
    const {slope, intercept, rSquared} = linearRegression(validIndices, validPrimaryVals);

    ctx.strokeStyle = '#ff8c00'; // Orange
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();

    const y1 = intercept + slope * validIndices[0];
    const y2 = intercept + slope * validIndices[validIndices.length - 1];

    ctx.moveTo(xscale(validIndices[0]), yscale_primary(y1));
    ctx.lineTo(xscale(validIndices[validIndices.length - 1]), yscale_primary(y2));
    ctx.stroke();

    // Display R-squared for 7-day views
    if (is7Day) {
      ctx.fillStyle = '#ff8c00';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`R² = ${rSquared.toFixed(3)}`, W - pad - 10, pad + 20);
      ctx.textAlign = 'left';
    }
  }

  // Draw hr_bpm data points (foreground layer)
  if (is7Day) {
    // 7-day view: connected line graph
    ctx.strokeStyle = '#d1d5db'; // Very light gray
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started=false;
    for(let i=0; i<primaryVals.length; i++){
      const v=primaryVals[i];
      if(!Number.isFinite(v)) continue;
      const x=xscale(i), y=yscale_primary(v);
      if(!started){ ctx.moveTo(x,y); started=true; } else { ctx.lineTo(x,y); }
    }
    ctx.stroke();
  } else {
    // All-time view: scatter plot (50% opacity)
    ctx.fillStyle = 'rgba(209, 213, 219, 0.5)'; // Very light gray with 50% opacity
    for(let i=0; i<primaryVals.length; i++){
      const v=primaryVals[i];
      if(!Number.isFinite(v)) continue;
      const x=xscale(i), y=yscale_primary(v);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function draw(chartNum, ts, vals, label, complianceEvents = []){
  const c = document.getElementById(`chart${chartNum}`);
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!ts.length){ ctx.fillText('No data for ' + label, 20, 20); return; }

  const W=c.width, H=c.height, pad=60; // Increased padding for rotated month labels
  const nums = vals.filter(v=>Number.isFinite(v));
  const vmin = Math.min.apply(null, nums), vmax = Math.max.apply(null, nums);
  const xscale = (i)=> pad + (W-2*pad)*(i/((ts.length-1)||1));
  const yscale = (v)=> H-pad - (H-2*pad)*((v - vmin)/((vmax - vmin)||1));

  // Check if this is a 7-day view (check for _7d suffix in metric name)
  const is7Day = label.includes('_7d');

  // Axes removed - cleaner look focusing on data and trends

  // Add time labels on x-axis
  if (ts.length > 0) {
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';

    if (is7Day) {
      // For 7-day views: show day of week (rotated like month labels)
      const dateGroups = new Map();
      ts.forEach((timestamp, idx) => {
        const date = new Date(timestamp.replace(' ', 'T'));
        const dateKey = date.toDateString();

        if (!dateGroups.has(dateKey)) {
          dateGroups.set(dateKey, {
            date: date,
            firstIdx: idx,
            lastIdx: idx
          });
        } else {
          const group = dateGroups.get(dateKey);
          group.lastIdx = idx;
        }
      });

      ctx.save(); // Save the current context state
      const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      dateGroups.forEach((group) => {
        const dayName = dayNames[group.date.getDay()];
        const midIdx = (group.firstIdx + group.lastIdx) / 2;
        const x = xscale(midIdx);
        const y = H - 10;

        // Rotate text 45 degrees
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4); // -45 degrees
        ctx.fillText(dayName, 0, 0);
        ctx.rotate(Math.PI / 4); // Rotate back
        ctx.translate(-x, -y);
      });
      ctx.restore(); // Restore the context state

    } else {
      // For all other views: show MM/YYYY for each month (rotated 45 degrees)
      const monthGroups = new Map();
      ts.forEach((timestamp, idx) => {
        const date = new Date(timestamp.replace(' ', 'T'));
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

        if (!monthGroups.has(monthKey)) {
          monthGroups.set(monthKey, {
            date: date,
            firstIdx: idx,
            lastIdx: idx
          });
        } else {
          const group = monthGroups.get(monthKey);
          group.lastIdx = idx;
        }
      });

      ctx.save(); // Save the current context state
      monthGroups.forEach((group) => {
        const month = String(group.date.getMonth() + 1).padStart(2, '0');
        const year = group.date.getFullYear();
        const monthLabel = `${month}/${year}`;
        const midIdx = (group.firstIdx + group.lastIdx) / 2;
        const x = xscale(midIdx);
        const y = H - 25;

        // Rotate text 45 degrees
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4); // -45 degrees
        ctx.fillText(monthLabel, 0, 0);
        ctx.rotate(Math.PI / 4); // Rotate back
        ctx.translate(-x, -y);
      });
      ctx.restore(); // Restore the context state
    }

    ctx.textAlign = 'left';
  }

  // For residuals, draw zero line in very light gray
  const isResidual = label.includes('residual');
  if (isResidual && vmin < 0 && vmax > 0) {
    const y0 = yscale(0);
    ctx.strokeStyle = '#d1d5db'; // Very light gray
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pad, y0);
    ctx.lineTo(W-pad, y0);
    ctx.stroke();

    // Label for zero line
    ctx.fillStyle = '#e4e6eb';
    ctx.font = '12px sans-serif';
    ctx.fillText('0 (baseline)', W-pad+5, y0+4);
  }

  // Calculate linear regression for trend line
  function linearRegression(xArr, yArr) {
    const n = xArr.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (let i = 0; i < n; i++) {
      sumX += xArr[i];
      sumY += yArr[i];
      sumXY += xArr[i] * yArr[i];
      sumX2 += xArr[i] * xArr[i];
      sumY2 += yArr[i] * yArr[i];
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Calculate R-squared
    const meanY = sumY / n;
    const ssTotal = sumY2 - n * meanY * meanY;
    const ssResidual = yArr.reduce((sum, y, i) => {
      const predicted = intercept + slope * xArr[i];
      return sum + Math.pow(y - predicted, 2);
    }, 0);
    const rSquared = 1 - (ssResidual / ssTotal);

    return { slope, intercept, rSquared };
  }

  // Draw orange regression line
  const validIndices = [];
  const validVals = [];
  for(let i=0; i<vals.length; i++){
    if(Number.isFinite(vals[i])) {
      validIndices.push(i);
      validVals.push(vals[i]);
    }
  }

  if (validIndices.length > 1) {
    const {slope, intercept, rSquared} = linearRegression(validIndices, validVals);

    ctx.strokeStyle = '#ff8c00'; // Orange
    ctx.lineWidth = 2;
    ctx.setLineDash([]); // Solid line instead of dashed
    ctx.beginPath();

    const y1 = intercept + slope * validIndices[0];
    const y2 = intercept + slope * validIndices[validIndices.length - 1];

    ctx.moveTo(xscale(validIndices[0]), yscale(y1));
    ctx.lineTo(xscale(validIndices[validIndices.length - 1]), yscale(y2));
    ctx.stroke();

    // Display R-squared for 7-day views in upper right corner
    if (is7Day) {
      ctx.fillStyle = '#ff8c00';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`R² = ${rSquared.toFixed(3)}`, W - pad - 10, pad + 20);
      ctx.textAlign = 'left';
    }
  }

  // Draw compliance event vertical lines
  if (complianceEvents && complianceEvents.length > 0) {
    // Helper function to parse timestamp and convert to index
    const timestampToIndex = (eventTs) => {
      // Find the closest timestamp in our data
      for (let i = 0; i < ts.length; i++) {
        if (ts[i] >= eventTs) {
          return i;
        }
      }
      return ts.length - 1; // If event is after all data, use last index
    };

    // Draw each compliance event as a thin vertical line
    complianceEvents.forEach(event => {
      const eventTimestamp = event.timestamp;
      const idx = timestampToIndex(eventTimestamp);
      const x = xscale(idx);

      // Get color for this intervention
      const color = getInterventionColor(event.intervention_id);

      // Draw vertical line with transparency
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, H - pad);
      ctx.stroke();
      ctx.globalAlpha = 1.0; // Reset alpha
    });
  }

  // Data visualization: scatter plot for long-term, line for 7-day
  if (is7Day) {
    // 7-day view: connected line graph (full opacity)
    ctx.strokeStyle = isResidual ? '#3498db' : '#d1d5db'; // Very light gray for non-residuals
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started=false;
    for(let i=0;i<vals.length;i++){
      const v=vals[i]; if(!Number.isFinite(v)) continue;
      const x=xscale(i), y=yscale(v);
      if(!started){ ctx.moveTo(x,y); started=true; } else { ctx.lineTo(x,y); }
    }
    ctx.stroke();
  } else {
    // Non-7-day view: scatter plot (50% opacity)
    const dotColor = isResidual ? 'rgba(52, 152, 219, 0.5)' : 'rgba(209, 213, 219, 0.5)'; // Very light gray with 50% opacity
    ctx.fillStyle = dotColor;

    for(let i=0;i<vals.length;i++){
      const v=vals[i]; if(!Number.isFinite(v)) continue;
      const x=xscale(i), y=yscale(v);

      // Draw circle at each data point
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI); // radius 3px
      ctx.fill();
    }
  }
}

// Refresh button
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
      `Archive deletions: ${json.archived_deleted}\n` +
      `Weather dates fetched: ${json.weather_dates_fetched}\n` +
      `New predictions: ${json.new_predictions}`;
  } catch (e) {
    out.textContent = "Error: " + (e.message || e);
  } finally {
    btn.disabled = false;
  }
}

if (btn) btn.addEventListener('click', refreshData);

// Metric charts - setup event listeners and load all three charts
[1, 2, 3].forEach(chartNum => {
  const select = document.getElementById(`metric${chartNum}`);
  if (select) {
    select.addEventListener('change', () => loadSeries(chartNum));
    loadSeries(chartNum);
  }
});

// Intervention form submission
const interventionForm = document.getElementById('interventionForm');
const interventionStatus = document.getElementById('interventionStatus');

if (interventionForm) {
  interventionForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = {
      name: document.getElementById('name').value,
      start_date: document.getElementById('start_date').value,
      duration_weeks: parseInt(document.getElementById('duration_weeks').value),
      frequency_per_week: parseInt(document.getElementById('frequency_per_week').value)
    };

    interventionStatus.textContent = 'Creating intervention...';
    interventionStatus.style.color = '#666';

    try {
      const res = await fetch('/interventions/quick', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!res.ok) {
        const error = await res.text();
        throw new Error(error || res.statusText);
      }

      const result = await res.json();
      interventionStatus.textContent = `Success! Intervention created.`;
      interventionStatus.style.color = '#4CAF50';
      interventionForm.reset();

      // Reload interventions list
      loadInterventionsList();

    } catch (error) {
      interventionStatus.textContent = `Error: ${error.message}`;
      interventionStatus.style.color = '#f44336';
    }
  });
}

// Load and display interventions list
async function loadInterventionsList() {
  const listContainer = document.getElementById('interventionsList');
  if (!listContainer) return;

  try {
    const res = await fetch('/interventions');
    if (!res.ok) throw new Error(res.statusText);

    const interventions = await res.json();

    if (interventions.length === 0) {
      listContainer.innerHTML = '<p style="color: #666;">No interventions yet.</p>';
      return;
    }

    // Create table
    let html = `
      <table style="width: 100%; border-collapse: collapse;">
        <thead>
          <tr style="border-bottom: 2px solid #ddd;">
            <th style="text-align: left; padding: 8px;">Name</th>
            <th style="text-align: left; padding: 8px;">Start Date</th>
            <th style="text-align: center; padding: 8px;">Duration (weeks)</th>
            <th style="text-align: center; padding: 8px;">Frequency/week</th>
            <th style="text-align: center; padding: 8px;">Active</th>
            <th style="text-align: center; padding: 8px;">Actions</th>
          </tr>
        </thead>
        <tbody>
    `;

    interventions.forEach(intervention => {
      const activeLabel = intervention.active ? 'Yes' : 'No';
      const activeColor = intervention.active ? '#10b981' : '#9ca3af';
      const toggleButtonText = intervention.active ? 'Deactivate' : 'Activate';
      const toggleButtonColor = intervention.active ? '#f59e0b' : '#10b981';

      // Escape intervention name for use in onclick attributes
      const escapedName = intervention.name.replace(/'/g, "\\'").replace(/"/g, '&quot;');

      html += `
        <tr style="border-bottom: 1px solid #eee;">
          <td style="padding: 8px;">${intervention.name}</td>
          <td style="padding: 8px;">${intervention.start_date}</td>
          <td style="text-align: center; padding: 8px;">${intervention.duration_weeks}</td>
          <td style="text-align: center; padding: 8px;">${intervention.freq_per_week}</td>
          <td style="text-align: center; padding: 8px; color: ${activeColor}; font-weight: bold;">${activeLabel}</td>
          <td style="text-align: center; padding: 8px;">
            <div style="display: flex; gap: 6px; justify-content: center; flex-wrap: wrap;">
              <button
                onclick="openEditModal(${intervention.id}, '${escapedName}', '${intervention.start_date}', ${intervention.duration_weeks}, ${intervention.freq_per_week})"
                style="padding: 6px 12px; font-size: 12px; background: #3b82f6; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Edit
              </button>
              <button
                onclick="toggleInterventionActive(${intervention.id})"
                style="padding: 6px 12px; font-size: 12px; background: ${toggleButtonColor}; color: white; border: none; border-radius: 4px; cursor: pointer;">
                ${toggleButtonText}
              </button>
              <button
                onclick="deleteIntervention(${intervention.id}, '${escapedName}')"
                style="padding: 6px 12px; font-size: 12px; background: #ef4444; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Delete
              </button>
            </div>
          </td>
        </tr>
      `;
    });

    html += '</tbody></table>';
    listContainer.innerHTML = html;

  } catch (error) {
    listContainer.innerHTML = `<p style="color: #f44336;">Error loading interventions: ${error.message}</p>`;
  }
}

// Toggle intervention active status
async function toggleInterventionActive(interventionId) {
  try {
    const res = await fetch(`/interventions/${interventionId}/toggle-active`, {
      method: 'PATCH'
    });

    if (!res.ok) throw new Error(res.statusText);

    await res.json();

    // Reload the interventions list to show updated status
    loadInterventionsList();

  } catch (error) {
    alert(`Error toggling intervention: ${error.message}`);
  }
}

// Helper function to escape HTML in intervention names
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Open edit modal and populate with intervention data
function openEditModal(id, name, startDate, durationWeeks, frequencyPerWeek) {
  const modal = document.getElementById('editModal');
  document.getElementById('edit_id').value = id;
  document.getElementById('edit_name').value = name;
  document.getElementById('edit_start_date').value = startDate;
  document.getElementById('edit_duration_weeks').value = durationWeeks;
  document.getElementById('edit_frequency_per_week').value = frequencyPerWeek;
  modal.style.display = 'flex';
}

// Close edit modal
function closeEditModal() {
  const modal = document.getElementById('editModal');
  modal.style.display = 'none';
}

// Handle edit form submission
document.getElementById('editInterventionForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();

  const interventionId = document.getElementById('edit_id').value;
  const payload = {
    name: document.getElementById('edit_name').value,
    start_date: document.getElementById('edit_start_date').value,
    duration_weeks: parseInt(document.getElementById('edit_duration_weeks').value),
    frequency_per_week: parseInt(document.getElementById('edit_frequency_per_week').value)
  };

  try {
    const res = await fetch(`/interventions/${interventionId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(res.statusText);

    closeEditModal();
    loadInterventionsList();
  } catch (error) {
    alert(`Error updating intervention: ${error.message}`);
  }
});

// Delete intervention with confirmation
async function deleteIntervention(interventionId, interventionName) {
  const confirmed = confirm(
    `Are you sure you want to delete "${interventionName}"?\n\n` +
    `This will also delete all associated compliance events and tokens for this intervention.\n\n` +
    `This action cannot be undone.`
  );

  if (!confirmed) return;

  try {
    const res = await fetch(`/interventions/${interventionId}`, {
      method: 'DELETE'
    });

    if (!res.ok) throw new Error(res.statusText);

    const result = await res.json();

    alert(
      `Successfully deleted intervention "${interventionName}".\n\n` +
      `Deleted ${result.deleted_compliance_events} compliance event(s) and ${result.deleted_tokens} token(s).`
    );

    loadInterventionsList();
  } catch (error) {
    alert(`Error deleting intervention: ${error.message}`);
  }
}

// Close modal when clicking outside of it
document.getElementById('editModal')?.addEventListener('click', (e) => {
  if (e.target.id === 'editModal') {
    closeEditModal();
  }
});

// Current compliance filter
let currentComplianceFilter = 'active';

// Load compliance statistics table
async function loadComplianceStats(filter = 'active') {
  const container = document.getElementById('complianceStatsContainer');
  if (!container) return;

  try {
    const res = await fetch(`/interventions/compliance-stats?filter=${filter}`);
    if (!res.ok) throw new Error(res.statusText);

    const stats = await res.json();

    if (stats.length === 0) {
      container.innerHTML = `<p style="color: #9ca3af;">No ${filter} interventions found.</p>`;
      return;
    }

    // Build table
    let html = `
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <thead>
          <tr style="border-bottom: 2px solid #3a3d44; background: #2a2d34;">
            <th style="padding: 12px 8px; text-align: left;">Intervention Name</th>
            <th style="padding: 12px 8px; text-align: center;">Start Date</th>
            <th style="padding: 12px 8px; text-align: center;">End Date</th>
            <th style="padding: 12px 8px; text-align: center;">Total Expected</th>
            <th style="padding: 12px 8px; text-align: center;">Completed</th>
            <th style="padding: 12px 8px; text-align: center;">% Trial Done</th>
            <th style="padding: 12px 8px; text-align: center;">Expected at %</th>
            <th style="padding: 12px 8px; text-align: center;">% Compliance</th>
          </tr>
        </thead>
        <tbody>
    `;

    stats.forEach(stat => {
      // Color code % compliance
      let complianceColor = '#9ca3af'; // gray
      if (stat.percent_completed_compliance >= 80) {
        complianceColor = '#10b981'; // green
      } else if (stat.percent_completed_compliance >= 60) {
        complianceColor = '#f59e0b'; // orange
      } else {
        complianceColor = '#ef4444'; // red
      }

      html += `
        <tr style="border-bottom: 1px solid #3a3d44;">
          <td style="padding: 8px;">${stat.intervention_name}</td>
          <td style="padding: 8px; text-align: center;">${stat.start_date}</td>
          <td style="padding: 8px; text-align: center;">${stat.end_date}</td>
          <td style="padding: 8px; text-align: center;">${stat.total_expected_compliance}</td>
          <td style="padding: 8px; text-align: center;">${stat.completed_compliance}</td>
          <td style="padding: 8px; text-align: center;">${stat.percent_trial_completed}%</td>
          <td style="padding: 8px; text-align: center;">${stat.percent_expected_compliance.toFixed(1)}</td>
          <td style="padding: 8px; text-align: center; color: ${complianceColor}; font-weight: bold;">
            ${stat.percent_completed_compliance}%
          </td>
        </tr>
      `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;

  } catch (error) {
    container.innerHTML = `<p style="color: #f44336;">Error loading compliance stats: ${error.message}</p>`;
  }
}

// Switch between active and completed interventions
function switchComplianceFilter(filter) {
  currentComplianceFilter = filter;

  // Update button styles
  const activeBtn = document.getElementById('activeFilterBtn');
  const completedBtn = document.getElementById('completedFilterBtn');

  if (filter === 'active') {
    activeBtn.style.background = '#3b82f6';
    activeBtn.style.fontWeight = 'bold';
    completedBtn.style.background = '#6b7280';
    completedBtn.style.fontWeight = 'normal';
  } else {
    activeBtn.style.background = '#6b7280';
    activeBtn.style.fontWeight = 'normal';
    completedBtn.style.background = '#3b82f6';
    completedBtn.style.fontWeight = 'bold';
  }

  // Reload data
  loadComplianceStats(filter);
}

// Load intervention analysis (Tau-U)
async function loadInterventionAnalysis() {
  const container = document.getElementById('interventionAnalysisContainer');
  if (!container) return;

  try {
    const res = await fetch('/analysis/intervention-effectiveness');
    if (!res.ok) throw new Error(res.statusText);

    const analyses = await res.json();

    if (analyses.length === 0) {
      container.innerHTML = `<p style="color: #9ca3af;">No completed interventions found for analysis.</p>`;
      return;
    }

    // Build table
    let html = `
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <thead>
          <tr style="border-bottom: 2px solid #3a3d44; background: #2a2d34;">
            <th style="padding: 12px 8px; text-align: left;">Intervention</th>
            <th style="padding: 12px 8px; text-align: center;">Intervention Period</th>
            <th style="padding: 12px 8px; text-align: center;">Baseline Period</th>
            <th style="padding: 12px 8px; text-align: center;">TP Residual Tau-U</th>
            <th style="padding: 12px 8px; text-align: center;">TP Effect</th>
            <th style="padding: 12px 8px; text-align: center;">Standing HR Tau-U</th>
            <th style="padding: 12px 8px; text-align: center;">HR Effect</th>
            <th style="padding: 12px 8px; text-align: center;">% Compliance</th>
            <th style="padding: 12px 8px; text-align: center;">Samples (B/I)</th>
          </tr>
        </thead>
        <tbody>
    `;

    analyses.forEach(analysis => {
      // Helper function to get color based on significance and direction
      function getEffectColor(tauU, pValue) {
        if (pValue >= 0.05) return '#9ca3af'; // Gray for non-significant
        return tauU > 0 ? '#10b981' : '#ef4444'; // Green for positive, red for negative
      }

      // Helper function to format effect size label
      function formatEffect(effectLabel) {
        const labels = {
          'negligible': 'Negligible',
          'small_to_medium': 'Small-Med',
          'medium_to_large': 'Med-Large',
          'large': 'Large',
          'insufficient_data': 'Insufficient'
        };
        return labels[effectLabel] || effectLabel;
      }

      const tpColor = getEffectColor(analysis.tp_residual_tau_u, analysis.tp_residual_p_value);
      const hrColor = getEffectColor(analysis.standing_hr_tau_u, analysis.standing_hr_p_value);

      // Compliance color
      let complianceColor = '#9ca3af';
      if (analysis.percent_compliance >= 80) {
        complianceColor = '#10b981';
      } else if (analysis.percent_compliance >= 60) {
        complianceColor = '#f59e0b';
      } else {
        complianceColor = '#ef4444';
      }

      html += `
        <tr style="border-bottom: 1px solid #3a3d44;">
          <td style="padding: 8px;">${analysis.intervention_name}</td>
          <td style="padding: 8px; text-align: center; font-size: 12px;">
            ${analysis.start_date}<br>to ${analysis.end_date}
          </td>
          <td style="padding: 8px; text-align: center; font-size: 12px;">
            ${analysis.baseline_start}<br>to ${analysis.baseline_end}
          </td>
          <td style="padding: 8px; text-align: center; color: ${tpColor}; font-weight: bold;">
            ${analysis.tp_residual_tau_u.toFixed(2)}<br>
            <span style="font-size: 11px; font-weight: normal; opacity: 0.8;">p=${analysis.tp_residual_p_value.toFixed(3)}</span>
          </td>
          <td style="padding: 8px; text-align: center; color: ${tpColor};">
            ${formatEffect(analysis.tp_residual_effect_size)}
          </td>
          <td style="padding: 8px; text-align: center; color: ${hrColor}; font-weight: bold;">
            ${analysis.standing_hr_tau_u.toFixed(2)}<br>
            <span style="font-size: 11px; font-weight: normal; opacity: 0.8;">p=${analysis.standing_hr_p_value.toFixed(3)}</span>
          </td>
          <td style="padding: 8px; text-align: center; color: ${hrColor};">
            ${formatEffect(analysis.standing_hr_effect_size)}
          </td>
          <td style="padding: 8px; text-align: center; color: ${complianceColor}; font-weight: bold;">
            ${analysis.percent_compliance.toFixed(1)}%
          </td>
          <td style="padding: 8px; text-align: center; font-size: 12px;">
            ${analysis.baseline_n} / ${analysis.intervention_n}
          </td>
        </tr>
      `;
    });

    html += '</tbody></table>';

    // Add legend
    html += `
      <div style="margin-top: 16px; padding: 12px; background: #2a2d34; border-radius: 8px; font-size: 13px;">
        <strong>Legend:</strong>
        <div style="display: flex; gap: 20px; margin-top: 8px; flex-wrap: wrap;">
          <div><span style="color: #10b981;">●</span> Significant positive effect (p < 0.05)</div>
          <div><span style="color: #ef4444;">●</span> Significant negative effect (p < 0.05)</div>
          <div><span style="color: #9ca3af;">●</span> Non-significant effect</div>
        </div>
        <div style="margin-top: 8px; opacity: 0.8;">
          <strong>Tau-U Interpretation:</strong> Measures non-overlap between baseline and intervention phases, controlling for baseline trend.
          Values range from -1 to +1, with larger absolute values indicating stronger effects.
        </div>
      </div>
    `;

    container.innerHTML = html;

  } catch (error) {
    container.innerHTML = `<p style="color: #f44336;">Error loading intervention analysis: ${error.message}</p>`;
  }
}

// Load interventions list when page loads
if (document.getElementById('interventionsList')) {
  loadInterventionsList();
}

// Load compliance stats when page loads
if (document.getElementById('complianceStatsContainer')) {
  loadComplianceStats('active');
}

// Load intervention analysis when page loads
if (document.getElementById('interventionAnalysisContainer')) {
  loadInterventionAnalysis();
}
