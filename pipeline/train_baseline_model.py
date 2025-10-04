"""
HRV Baseline Prediction Model Training

Trains models to predict Total Power (LF+HF) and LF/HF ratio using:
- HRV metrics (RMSSD, SDNN, pNN50, peak frequencies)
- Weather data (temperature, pressure)
- Time-of-day features
- Lagged features (temporal dependencies)

Compares Linear Regression, Ridge with Polynomial Features, and XGBoost.
Saves best model for generating residuals (actual - predicted) for intervention analysis.

Usage:
    python train_baseline_model.py
    python train_baseline_model.py --start 2025-04-01 --end 2025-07-31
    python train_baseline_model.py --save-model models/hrv_baseline.pkl
"""

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "app").exists():
            return p
    return Path.cwd()

DB_PATH = _repo_root() / "hrv_lab.sqlite3"
MODELS_DIR = _repo_root() / "models"


def load_hrv_weather_data(db_path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load HRV data from hrv_results and merge with weather data.

    Returns DataFrame with all features for model training.
    """
    with sqlite3.connect(db_path) as cx:
        # Load HRV data
        query = """
            SELECT
                timestamp,
                rmssd_ms,
                sdnn_ms,
                pnn50,
                lf_ms2,
                hf_ms2,
                lf_hf,
                lf_peak_hz,
                hf_peak_hz,
                hr_bpm,
                total_power_ms2
            FROM hrv_results
            WHERE timestamp IS NOT NULL
        """

        if start_date:
            query += f" AND DATE(timestamp) >= '{start_date}'"
        if end_date:
            query += f" AND DATE(timestamp) <= '{end_date}'"

        query += " ORDER BY timestamp ASC"

        hrv_df = pd.read_sql_query(query, cx, parse_dates=['timestamp'])

        # Load weather data
        weather_query = """
            SELECT
                date,
                temperature_max_c,
                apparent_temperature_max_c,
                surface_pressure_hpa
            FROM weather_daily
            ORDER BY date ASC
        """
        weather_df = pd.read_sql_query(weather_query, cx)
        weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Merge on date
    hrv_df['date'] = hrv_df['timestamp'].dt.date
    hrv_df['date'] = pd.to_datetime(hrv_df['date'])

    merged = pd.merge(hrv_df, weather_df, on='date', how='left')

    print(f"Loaded {len(merged)} HRV readings from {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print(f"Weather data coverage: {merged['temperature_max_c'].notna().sum()}/{len(merged)} records")

    return merged


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for model training.

    Includes:
    - Log/sqrt transforms of HRV metrics
    - Cyclic time encoding
    - Weather features
    - Lagged features
    """
    X = pd.DataFrame(index=df.index)

    # HRV features (transformed)
    X['log_rmssd'] = np.log(df['rmssd_ms'] + 1e-6)
    X['log_sdnn'] = np.log(df['sdnn_ms'] + 1e-6)
    X['sqrt_pnn50'] = np.sqrt(df['pnn50'] + 1.0)
    X['lf_peak'] = df['lf_peak_hz']
    X['hf_peak'] = df['hf_peak_hz']
    X['hr_bpm'] = df['hr_bpm']

    # Time features (cyclic encoding)
    X['hour'] = df['timestamp'].dt.hour
    X['sin_hr'] = np.sin(2 * np.pi * X['hour'] / 24)
    X['cos_hr'] = np.cos(2 * np.pi * X['hour'] / 24)

    # Day of week (cyclical)
    X['day_of_week'] = df['timestamp'].dt.dayofweek
    X['sin_dow'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
    X['cos_dow'] = np.cos(2 * np.pi * X['day_of_week'] / 7)

    # Weather features
    X['temp_max'] = df['temperature_max_c'].fillna(df['temperature_max_c'].mean())
    X['pressure'] = df['surface_pressure_hpa'].fillna(df['surface_pressure_hpa'].mean())
    X['apparent_temp'] = df['apparent_temperature_max_c'].fillna(df['apparent_temperature_max_c'].mean())

    # NEW: Weather interactions and deltas
    X['temp_pressure_interaction'] = X['temp_max'] * X['pressure'] / 1000  # Scale down
    X['temp_delta'] = X['temp_max'].diff()  # Change from yesterday
    X['pressure_delta'] = X['pressure'].diff()

    # Lagged features (temporal dependencies)
    X['rmssd_lag1'] = X['log_rmssd'].shift(1)
    X['sdnn_lag1'] = X['log_sdnn'].shift(1)

    # Total power lag
    tp_series = np.log(df['lf_ms2'] + df['hf_ms2'] + 1e-6)
    X['tp_lag1'] = tp_series.shift(1)

    # NEW: More lag windows
    X['rmssd_lag2'] = X['log_rmssd'].shift(2)
    X['tp_lag2'] = tp_series.shift(2)
    X['tp_rolling_7d'] = tp_series.rolling(window=7, min_periods=1).mean()

    X['temp_lag1'] = X['temp_max'].shift(1)
    X['pressure_lag1'] = X['pressure'].shift(1)

    # NEW: Week of study (captures long-term trends)
    X['week_of_study'] = (df['timestamp'] - df['timestamp'].min()).dt.days // 7

    # Drop rows with NaN from lagging (first row)
    X = X.dropna()

    return X


def create_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Create target variables.

    Returns:
        y_tp: log(Total Power) = log(LF + HF)
        y_lfhf: log(LF/HF ratio)
    """
    # Total Power (LF + HF, excludes VLF intentionally)
    y_tp = np.log(df['lf_ms2'] + df['hf_ms2'] + 1e-6)

    # LF/HF Ratio
    y_lfhf = np.log(df['lf_hf'] + 1e-6)

    return y_tp, y_lfhf


def stratified_split(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.Series, pd.Series]:
    """
    Create stratified train/test split by sampling random weeks from each month.

    For each month:
    - If month has >= 21 days: sample 21 random days for training
    - If month has < 21 days: use 70% for training

    Returns:
        train_mask: Boolean series for training samples
        test_mask: Boolean series for test samples
    """
    np.random.seed(random_state)

    df = df.copy()
    df['month'] = df['timestamp'].dt.to_period('M')
    df['date_str'] = df['timestamp'].dt.date.astype(str)

    # Get unique dates per month
    dates_by_month = df.groupby('month')['date_str'].unique()

    train_dates = []
    test_dates = []

    for month, dates in dates_by_month.items():
        n_days = len(dates)

        if n_days < 21:
            # Use 70% for training
            n_train = max(1, int(n_days * 0.7))
            train_idx = np.random.choice(n_days, size=n_train, replace=False)
        else:
            # Sample 21 random days (3 weeks)
            train_idx = np.random.choice(n_days, size=21, replace=False)

        test_idx = np.setdiff1d(np.arange(n_days), train_idx)

        train_dates.extend(dates[train_idx])
        test_dates.extend(dates[test_idx])

    # Create masks
    train_mask = df['date_str'].isin(train_dates)
    test_mask = df['date_str'].isin(test_dates)

    return train_mask, test_mask


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}


def train_models(X_train, y_train, X_test, y_test, target_name="TP"):
    """
    Train and compare multiple models.

    Returns best model and results dictionary.
    """
    results = {}

    print(f"\n{'='*60}")
    print(f"Training Models for {target_name}")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 1. Linear Regression (baseline)
    print("\n1. Linear Regression")
    lr_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    lr_model.fit(X_train, y_train)
    results['linear'] = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    results['linear']['model'] = lr_model

    # 2. Ridge with Polynomial Features
    print("\n2. Ridge with Polynomial Features")
    ridge_poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TimeSeriesSplit(5)))
    ])
    ridge_poly_model.fit(X_train, y_train)
    results['ridge_poly'] = evaluate_model(ridge_poly_model, X_test, y_test, "Ridge + Polynomial")
    results['ridge_poly']['model'] = ridge_poly_model

    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        print("\n3. XGBoost")
        xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        results['xgboost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results['xgboost']['model'] = xgb_model

    # 4. ElasticNet (primary model with broader regularization)
    print("\n4. ElasticNetCV (Primary)")
    elasticnet_model = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNetCV(
            l1_ratio=np.linspace(0.1, 0.9, 5),
            alphas=np.logspace(-3, 1, 30),
            cv=TimeSeriesSplit(5),
            max_iter=10000,  # Fix convergence warnings
            random_state=42
        ))
    ])
    elasticnet_model.fit(X_train, y_train)
    results['elasticnet'] = evaluate_model(elasticnet_model, X_test, y_test, "ElasticNetCV")
    results['elasticnet']['model'] = elasticnet_model

    # Print selected hyperparameters for ElasticNet
    elastic_params = elasticnet_model.named_steps['elasticnet']
    print(f"  Selected l1_ratio: {elastic_params.l1_ratio_:.3f}")
    print(f"  Selected alpha: {elastic_params.alpha_:.6f}")

    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_model_name]['model']

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name.upper()} (MAE: {results[best_model_name]['mae']:.4f})")
    print(f"{'='*60}")

    return best_model, results, best_model_name


def main():
    parser = argparse.ArgumentParser(description="Train HRV baseline prediction models")
    parser.add_argument("--start", type=str, default="2025-04-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-09-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--split-method", type=str, choices=['stratified', 'date'], default="stratified",
                        help="Split method: stratified (random weeks per month) or date (temporal split)")
    parser.add_argument("--split-date", type=str, default="2025-07-01", help="Train/test split date (only for split-method=date)")
    parser.add_argument("--save-model", type=str, help="Path to save best model (pickle)")
    parser.add_argument("--target", choices=['tp', 'lfhf', 'both'], default='both', help="Which target to train")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for stratified split")

    args = parser.parse_args()

    # Load data
    print("Loading HRV and weather data...")
    df = load_hrv_weather_data(DB_PATH, args.start, args.end)

    if len(df) == 0:
        print("ERROR: No data loaded. Check date range and database.")
        return 1

    # Create features and targets
    print("\nCreating features...")
    X = create_features(df)
    y_tp, y_lfhf = create_targets(df.loc[X.index])

    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")

    # Train/test split
    if args.split_method == 'stratified':
        print(f"\nUsing stratified split (3 random weeks per month, seed={args.random_state})...")
        train_mask_raw, test_mask_raw = stratified_split(df, random_state=args.random_state)

        # Apply to feature index
        train_mask = train_mask_raw.loc[X.index]
        test_mask = test_mask_raw.loc[X.index]
    else:
        print(f"\nUsing temporal split at {args.split_date}...")
        split_date = pd.to_datetime(args.split_date)
        train_mask = df.loc[X.index, 'timestamp'] < split_date
        test_mask = df.loc[X.index, 'timestamp'] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_tp_train, y_tp_test = y_tp[train_mask], y_tp[test_mask]
    y_lfhf_train, y_lfhf_test = y_lfhf[train_mask], y_lfhf[test_mask]

    print(f"\nSplit Summary:")
    print(f"  Train: {train_mask.sum()} samples ({df.loc[X.index][train_mask]['timestamp'].min()} to {df.loc[X.index][train_mask]['timestamp'].max()})")
    print(f"  Test:  {test_mask.sum()} samples ({df.loc[X.index][test_mask]['timestamp'].min()} to {df.loc[X.index][test_mask]['timestamp'].max()})")

    # Train models
    models_to_save = {}

    if args.target in ['tp', 'both']:
        tp_model, tp_results, tp_best = train_models(
            X_train, y_tp_train, X_test, y_tp_test, target_name="Total Power (log)"
        )
        models_to_save['tp'] = tp_model

    if args.target in ['lfhf', 'both']:
        lfhf_model, lfhf_results, lfhf_best = train_models(
            X_train, y_lfhf_train, X_test, y_lfhf_test, target_name="LF/HF Ratio (log)"
        )
        models_to_save['lfhf'] = lfhf_model

    # Save models if requested
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': models_to_save,
                'feature_columns': list(X.columns),
                'trained_date': datetime.now().isoformat(),
                'data_range': (args.start, args.end),
                'split_method': args.split_method,
                'split_date': args.split_date if args.split_method == 'date' else None,
                'random_state': args.random_state if args.split_method == 'stratified' else None,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }, f)

        print(f"\nModels saved to: {save_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
