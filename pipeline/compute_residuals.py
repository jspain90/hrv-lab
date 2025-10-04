"""
Compute HRV Residuals (Actual - Predicted)

Uses trained baseline models to generate predictions and residuals for all HRV data.
Residuals represent deviation from expected baseline, controlling for:
- Weather conditions
- Time-of-day patterns
- Temporal dependencies (lagged HRV)

Residuals are stored in database for intervention analysis.

Usage:
    python compute_residuals.py --model models/hrv_baseline.pkl
    python compute_residuals.py --model models/hrv_baseline.pkl --start 2025-04-01 --end 2025-09-30
    python compute_residuals.py --model models/hrv_baseline.pkl --dry-run
"""

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Suppress xgboost warning from training script import
warnings.filterwarnings('ignore', message='XGBoost not installed')

# Import feature creation from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_baseline_model import load_hrv_weather_data, create_features, create_targets, _repo_root

DB_PATH = _repo_root() / "hrv_lab.sqlite3"


def ensure_predictions_table(cx: sqlite3.Connection):
    """Create hrv_predictions table if it doesn't exist."""
    cx.execute("""
        CREATE TABLE IF NOT EXISTS hrv_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL UNIQUE,
            tp_actual REAL,
            tp_predicted REAL,
            tp_residual REAL,
            lfhf_actual REAL,
            lfhf_predicted REAL,
            lfhf_residual REAL,
            model_version TEXT,
            computed_at TEXT NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON hrv_predictions(timestamp)")
    cx.commit()


def compute_and_store_residuals(
    model_path: Path,
    db_path: Path = DB_PATH,
    start_date: str = None,
    end_date: str = None,
    dry_run: bool = False
):
    """
    Load trained models and compute residuals for all HRV data.

    Args:
        model_path: Path to pickled model file
        db_path: Path to SQLite database
        start_date: Optional start date filter
        end_date: Optional end date filter
        dry_run: If True, don't write to database
    """
    # Load trained models
    print(f"Loading models from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    tp_model = model_data['models']['tp']
    lfhf_model = model_data['models']['lfhf']
    feature_columns = model_data['feature_columns']

    print(f"Model trained on: {model_data['data_range']}")
    print(f"Features: {len(feature_columns)} columns")

    # Load HRV and weather data
    print(f"\nLoading HRV and weather data...")
    df = load_hrv_weather_data(db_path, start_date, end_date)

    if len(df) == 0:
        print("ERROR: No data to process")
        return 1

    # Create features
    print("Creating features...")
    X = create_features(df)
    y_tp_actual, y_lfhf_actual = create_targets(df.loc[X.index])

    # Ensure feature columns match training
    if list(X.columns) != feature_columns:
        print("WARNING: Feature columns don't match training. Reordering...")
        X = X[feature_columns]

    print(f"Computing predictions for {len(X)} samples...")

    # Generate predictions
    tp_pred = tp_model.predict(X)
    lfhf_pred = lfhf_model.predict(X)

    # Compute residuals
    tp_residuals = y_tp_actual.values - tp_pred
    lfhf_residuals = y_lfhf_actual.values - lfhf_pred

    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': df.loc[X.index, 'timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'tp_actual': y_tp_actual.values,
        'tp_predicted': tp_pred,
        'tp_residual': tp_residuals,
        'lfhf_actual': y_lfhf_actual.values,
        'lfhf_predicted': lfhf_pred,
        'lfhf_residual': lfhf_residuals,
        'model_version': model_data.get('trained_date', 'unknown'),
        'computed_at': datetime.utcnow().isoformat()
    })

    # Print summary statistics
    print(f"\n{'='*60}")
    print("RESIDUALS SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal Power (log):")
    print(f"  Mean residual:   {tp_residuals.mean():.4f}")
    print(f"  Std residual:    {tp_residuals.std():.4f}")
    print(f"  MAE:             {np.abs(tp_residuals).mean():.4f}")
    print(f"\nLF/HF Ratio (log):")
    print(f"  Mean residual:   {lfhf_residuals.mean():.4f}")
    print(f"  Std residual:    {lfhf_residuals.std():.4f}")
    print(f"  MAE:             {np.abs(lfhf_residuals).mean():.4f}")

    # Show sample residuals
    print(f"\nSample residuals (first 5):")
    print(results[['timestamp', 'tp_residual', 'lfhf_residual']].head().to_string(index=False))

    if dry_run:
        print(f"\nDRY RUN: Would insert {len(results)} records into hrv_predictions")
        return 0

    # Store in database
    print(f"\nStoring {len(results)} predictions in database...")
    with sqlite3.connect(db_path) as cx:
        ensure_predictions_table(cx)

        # Insert or replace records
        for _, row in results.iterrows():
            cx.execute("""
                INSERT OR REPLACE INTO hrv_predictions
                (timestamp, tp_actual, tp_predicted, tp_residual,
                 lfhf_actual, lfhf_predicted, lfhf_residual,
                 model_version, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['timestamp'], row['tp_actual'], row['tp_predicted'], row['tp_residual'],
                row['lfhf_actual'], row['lfhf_predicted'], row['lfhf_residual'],
                row['model_version'], row['computed_at']
            ))

        cx.commit()

    print(f"OK Successfully stored predictions and residuals")

    # Verification
    with sqlite3.connect(db_path) as cx:
        count = cx.execute("SELECT COUNT(*) FROM hrv_predictions").fetchone()[0]
        print(f"OK Total records in hrv_predictions table: {count}")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")

    return 0


def compute_residuals_incremental(model_path: Path, db_path: Path = DB_PATH) -> int:
    """
    Compute residuals for NEW HRV data not yet in predictions table.

    Designed for automatic refresh workflow - finds HRV readings without predictions
    and computes residuals using the fixed baseline model.

    Args:
        model_path: Path to pickled baseline model (NEVER retrain this!)
        db_path: Path to SQLite database

    Returns:
        Number of new predictions computed (0 if error or no new data)
    """
    try:
        # Verify model exists
        if not model_path.exists():
            print(f"WARNING: Model file not found: {model_path}")
            return 0

        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        tp_model = model_data['models']['tp']
        lfhf_model = model_data['models']['lfhf']
        feature_columns = model_data['feature_columns']

        # Find HRV readings without predictions
        with sqlite3.connect(db_path) as cx:
            ensure_predictions_table(cx)

            # Get timestamps of HRV data not yet in predictions
            # Note: hrv_results uses ISO format (T separator), predictions use space separator
            # We need to normalize for comparison
            query = """
                SELECT h.timestamp
                FROM hrv_results h
                LEFT JOIN hrv_predictions p ON REPLACE(h.timestamp, 'T', ' ') = p.timestamp
                WHERE p.id IS NULL
                ORDER BY h.timestamp ASC
            """
            missing = pd.read_sql_query(query, cx)

        if len(missing) == 0:
            print("No new HRV data to process")
            return 0

        # Get date range for missing data
        start_date = pd.to_datetime(missing['timestamp'].min()).date().isoformat()
        end_date = pd.to_datetime(missing['timestamp'].max()).date().isoformat()

        print(f"Computing residuals for {len(missing)} new HRV readings ({start_date} to {end_date})")

        # Load HRV and weather data for date range
        df = load_hrv_weather_data(db_path, start_date, end_date)

        if len(df) == 0:
            print("ERROR: Could not load HRV/weather data")
            return 0

        # Create features
        X = create_features(df)
        y_tp_actual, y_lfhf_actual = create_targets(df.loc[X.index])

        # Ensure feature columns match training
        if list(X.columns) != feature_columns:
            X = X[feature_columns]

        # Generate predictions
        tp_pred = tp_model.predict(X)
        lfhf_pred = lfhf_model.predict(X)

        # Compute residuals
        tp_residuals = y_tp_actual.values - tp_pred
        lfhf_residuals = y_lfhf_actual.values - lfhf_pred

        # Create results dataframe
        results = pd.DataFrame({
            'timestamp': df.loc[X.index, 'timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            'tp_actual': y_tp_actual.values,
            'tp_predicted': tp_pred,
            'tp_residual': tp_residuals,
            'lfhf_actual': y_lfhf_actual.values,
            'lfhf_predicted': lfhf_pred,
            'lfhf_residual': lfhf_residuals,
            'model_version': model_data.get('trained_date', 'unknown'),
            'computed_at': datetime.utcnow().isoformat()
        })

        # Store in database
        with sqlite3.connect(db_path) as cx:
            for _, row in results.iterrows():
                cx.execute("""
                    INSERT OR REPLACE INTO hrv_predictions
                    (timestamp, tp_actual, tp_predicted, tp_residual,
                     lfhf_actual, lfhf_predicted, lfhf_residual,
                     model_version, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['timestamp'], row['tp_actual'], row['tp_predicted'], row['tp_residual'],
                    row['lfhf_actual'], row['lfhf_predicted'], row['lfhf_residual'],
                    row['model_version'], row['computed_at']
                ))
            cx.commit()

        print(f"OK Computed {len(results)} new predictions")
        return len(results)

    except Exception as e:
        print(f"ERROR computing residuals: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Compute HRV residuals from trained models")
    parser.add_argument("--model", type=str, required=True, help="Path to pickled model file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return 1

    return compute_and_store_residuals(
        model_path=model_path,
        db_path=DB_PATH,
        start_date=args.start,
        end_date=args.end,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
