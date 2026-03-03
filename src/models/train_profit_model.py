"""
Train a LightGBM regressor to predict expected profit for a driver-route pair.

Input:  data/ml/training_dataset_v2.parquet  (from build_enhanced_dataset.py)
Output: models/profit_model_v2.pkl           (joblib-serialised LightGBM booster)
        models/feature_importance_v2.csv

Uses GroupShuffleSplit by driver_id so no driver appears in both train and val
sets (prevents data leakage from multiple routes per driver).

Usage:
    python src/models/train_profit_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATASET_PATH = ROOT / "data" / "ml" / "training_dataset_v2.parquet"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "profit_model_v2.pkl"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance_v2.csv"

FEATURE_COLS = [
    "route_distance_m",
    "route_duration_s",
    "corridor_cell_count",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "corridor_rider_count",
    "corridor_demand_density",
    "mean_rider_fare",
    "corridor_fare_density",
    "day_of_month",
    "time_bin_15min",
    "hour_sin",
    "hour_cos",
    "route_sinuosity",
    "route_avg_speed_ms",
    "bearing_sin",
    "bearing_cos",
    "straight_line_dist_m",
    "origin_landmark_dist_km",
    "dest_landmark_dist_km",
    "origin_jfk_km",
    "origin_lga_km",
    "origin_penn_km",
    "origin_times_sq_km",
    "dest_jfk_km",
    "dest_lga_km",
    "dest_penn_km",
    "dest_times_sq_km",
    "corridor_hist_pickups",
    "corridor_hist_dropoffs",
    "corridor_hist_pickup_density",
    "corridor_hist_dropoff_density",
    "corridor_hist_mean_fare",
    "corridor_hist_fare_density",
    "origin_cell_pickups",
    "origin_cell_mean_fare",
    "dest_cell_dropoffs",
]
TARGET_COL = "expected_profit"

VALIDATION_FRAC = 0.20
SPLIT_SEED = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "min_split_gain": 0.01,
    "verbose": -1,
}
NUM_ROUNDS = 2_000
EARLY_STOPPING = 80


def main() -> None:
    print("=== Train Profit Prediction Model ===")

    df = pd.read_parquet(DATASET_PATH)
    print(f"  Dataset loaded: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  Target stats: mean={df[TARGET_COL].mean():.2f}, "
          f"median={df[TARGET_COL].median():.2f}, "
          f"std={df[TARGET_COL].std():.2f}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    groups = df["driver_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRAC, random_state=SPLIT_SEED)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    n_train_drivers = len(set(groups[train_idx]))
    n_val_drivers = len(set(groups[val_idx]))
    print(f"  Train: {len(X_train):,} rows ({n_train_drivers:,} drivers)   "
          f"Val: {len(X_val):,} rows ({n_val_drivers:,} drivers)")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain, free_raw_data=False)

    print("\n  Training LightGBM regressor...")
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=True),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    print(f"\n  Validation metrics:")
    print(f"    RMSE:  ${rmse:.2f}")
    print(f"    MAE:   ${mae:.2f}")
    print(f"    R²:    {r2:.4f}")
    print(f"    Corr:  {np.corrcoef(y_val, y_pred_val)[0,1]:.4f}")

    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\n  Feature importance (gain):")
    for _, row in imp.iterrows():
        bar = "#" * int(row["importance"] / imp["importance"].max() * 30)
        print(f"    {row['feature']:30s}  {row['importance']:>10.0f}  {bar}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    imp.to_csv(IMPORTANCE_PATH, index=False)

    print(f"\n  Model saved: {MODEL_PATH}")
    print(f"  Feature importance saved: {IMPORTANCE_PATH}")
    print(f"  Best iteration: {model.best_iteration}")
    print("=== Training Complete ===")


if __name__ == "__main__":
    main()
