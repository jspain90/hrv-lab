# HRV Baseline Prediction & Residuals Pipeline

## Overview

This ML pipeline predicts baseline HRV metrics using weather, time-of-day, and temporal patterns. **Residuals** (actual - predicted) isolate intervention effects by controlling for confounding variables.

## What Was Built

### 1. Training Pipeline (`pipeline/train_baseline_model.py`)

**Features Created:**
- **HRV transforms**: log(RMSSD), log(SDNN), sqrt(pNN50), LF peak, HF peak
- **Time encoding**: Cyclic sin/cos for hour-of-day and day-of-week
- **Weather**: Temperature max, apparent temperature, surface pressure
- **Lagged features**: Previous day's RMSSD, SDNN, Total Power, temperature, pressure

**Targets:**
- `y_tp`: log(Total Power) = log(LF + HF) [excludes VLF intentionally]
- `y_lfhf`: log(LF/HF ratio)

**Models Compared:**
1. Linear Regression (baseline)
2. Ridge with Polynomial Features (degree=2)
3. XGBoost
4. ElasticNetCV with broad regularization (`l1_ratio`: 0.1-0.9)

**Results (April-July 2025):**
```
Total Power (log):
- WINNER: ElasticNet (MAE: 0.273, R²: 0.588)
- XGBoost close second (MAE: 0.274, R²: 0.616)

LF/HF Ratio (log):
- WINNER: XGBoost (MAE: 0.394, R²: 0.164)
- (Harder to predict - autonomic balance is volatile)
```

**Usage:**
```bash
# Train on April-July, split at June 16
python pipeline/train_baseline_model.py --start 2025-04-01 --end 2025-07-31 --split-date 2025-06-16

# Save best models
python pipeline/train_baseline_model.py --save-model models/hrv_baseline.pkl
```

### 2. Residuals Pipeline (`pipeline/compute_residuals.py`)

Generates predictions and residuals for all HRV data. Stores in `hrv_predictions` table.

**Database Schema:**
```sql
CREATE TABLE hrv_predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT UNIQUE,
    tp_actual REAL,
    tp_predicted REAL,
    tp_residual REAL,       -- This is what you care about!
    lfhf_actual REAL,
    lfhf_predicted REAL,
    lfhf_residual REAL,     -- And this!
    model_version TEXT,
    computed_at TEXT
);
```

**Interpretation:**
- **Positive residual** = Better than expected HRV (good!)
- **Negative residual** = Worse than expected HRV (concerning)
- **Near-zero residual** = Performing as expected given conditions

**Usage:**
```bash
# Compute residuals for all April-July data
python pipeline/compute_residuals.py --model models/hrv_baseline.pkl --start 2025-04-01 --end 2025-07-31

# Dry run (don't write to DB)
python pipeline/compute_residuals.py --model models/hrv_baseline.pkl --dry-run
```

### 3. Analysis API (`app/routers/analysis.py`)

**Endpoints:**

#### Get Residuals
```bash
GET /analysis/residuals?start=2025-04-01&end=2025-07-31&limit=1000
```
Returns residuals for date range.

#### Residuals Summary
```bash
GET /analysis/residuals/summary?start=2025-04-01&end=2025-07-31
```
Returns statistics: mean, std, count, date range.

#### Intervention Analysis
```bash
GET /analysis/intervention/{intervention_id}/residuals
```
**Key endpoint for n=1 analysis!** Returns:
- Average residuals during intervention period
- Interpretation (better/worse than baseline)
- Number of readings

Example response:
```json
{
  "intervention_id": 1,
  "start_date": "2025-04-15",
  "duration_weeks": 4,
  "n_readings": 28,
  "avg_tp_residual": 0.15,
  "avg_lfhf_residual": -0.02,
  "interpretation": {
    "tp": "better than baseline",
    "lfhf": "better balance"
  }
}
```

## How Residuals Work

### The Problem
Raw HRV changes could be due to:
- Your intervention ✓ (what you want to measure)
- Weather change ✗ (confounding variable)
- Time of day ✗ (confounding variable)
- Natural HRV variation ✗ (noise)

### The Solution
1. **Train model** to predict HRV from weather + time + past HRV
2. **Residual = Actual - Predicted**
3. Residual represents **what the model can't explain** = intervention effect

### Example
- Predicted TP (log): 8.5 (based on weather, time, etc.)
- Actual TP (log): 8.8
- **Residual: +0.3** → You're doing better than expected!

If barometric pressure drops (bad for POTS) but your residual stays positive, the intervention is working despite adverse conditions.

## Workflow for Intervention Analysis

### 1. Train Baseline Model (One-time or periodic)
```bash
python pipeline/train_baseline_model.py \
  --start 2025-04-01 \
  --end 2025-09-30 \
  --save-model models/hrv_baseline.pkl
```

### 2. Compute Residuals for All Data
```bash
python pipeline/compute_residuals.py \
  --model models/hrv_baseline.pkl
```

### 3. Analyze Intervention
```bash
# Via API
curl http://127.0.0.1:8000/analysis/intervention/1/residuals

# Or query database directly
sqlite3 hrv_lab.sqlite3 "
SELECT DATE(timestamp), AVG(tp_residual)
FROM hrv_predictions
WHERE DATE(timestamp) BETWEEN '2025-04-15' AND '2025-05-15'
GROUP BY DATE(timestamp)
ORDER BY DATE(timestamp)
"
```

### 4. Statistical Testing (Next Step)
- Compare intervention period residuals vs. baseline
- Use paired t-test, Wilcoxon signed-rank, or effect size (Cohen's d)
- Determine if change is statistically significant

## Files Created

| File | Purpose |
|------|---------|
| `pipeline/train_baseline_model.py` | Train ML models, compare performance |
| `pipeline/compute_residuals.py` | Generate predictions & residuals |
| `app/routers/analysis.py` | API endpoints for residuals |
| `models/hrv_baseline.pkl` | Trained models (saved) |
| `ML_PIPELINE_README.md` | This documentation |

## Next Steps

1. **Retrain periodically** as you collect more data (monthly?)
2. **Build statistical testing module** to compare intervention vs. baseline residuals
3. **Create visualization** showing predicted vs. actual over time
4. **Add alerts** for concerning patterns (persistent negative residuals)
5. **Expand features** if needed (sleep data, stress markers, diet logs)

## Key Insight

**You're not measuring raw HRV anymore. You're measuring HRV controlling for everything else.**

This is the difference between:
- "My HRV went up 10%" ← Could be weather
- "My residuals improved by 0.3 during intervention" ← Isolated effect

The residuals are your ground truth for whether interventions work.

## Dependencies

All packages already in `requirements.txt`:
- pandas, numpy, scikit-learn (ML)
- xgboost (gradient boosting)
- SQLAlchemy, FastAPI (API/DB)

## Questions?

- **Why exclude VLF?** Total Power = LF + HF is the standard autonomic balance metric. VLF has unclear physiological meaning.
- **Why log transform?** HRV metrics are highly skewed; log makes them more normally distributed.
- **Why ElasticNet?** Handles multicollinearity well and provides feature selection via L1 regularization.
- **Why TimeSeriesSplit?** Prevents data leakage in cross-validation for time series data.
