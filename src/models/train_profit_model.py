"""
Train the publication-facing LightGBM route-profit model.

Workflow:
  1. Evaluate tuned LightGBM under a temporal holdout when possible
     (Jan-Feb 2015 train, Mar 2015 validation).
  2. Write temporal_generalization.csv for the paper artifact.
  3. Retrain the final model on all Jan-Mar rows using the validated
     number of boosting rounds, then save it for April policy evaluation.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.domain_config import get_domain_config
from models.evaluation_split import build_eval_split

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


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the publication-facing LightGBM route-profit model")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--dataset", type=str, default="", help="Optional explicit dataset path")
    args = parser.parse_args()

    domain_config = get_domain_config(args.domain)
    dataset_path = Path(args.dataset) if args.dataset else domain_config.training_dataset_path()
    model_dir = ROOT / "models" if args.domain == "yellow" else domain_config.models_dir
    model_path = domain_config.model_path()
    importance_path = model_dir / "feature_importance_v2.csv"
    results_dir = ROOT / "results" if args.domain == "yellow" else domain_config.results_dir
    temporal_path = results_dir / "temporal_generalization.csv"

    print(f"=== Train Profit Prediction Model [{domain_config.display_name}] ===")

    df = pd.read_parquet(dataset_path)
    print(f"  Dataset loaded: {len(df):,} rows, {len(df.columns)} cols")
    if df.empty:
        raise ValueError(f"Training dataset is empty: {dataset_path}")
    missing_cols = [col for col in [*FEATURE_COLS, TARGET_COL, "driver_id"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Training dataset is missing required columns: {missing_cols}")
    print(
        f"  Target stats: mean={df[TARGET_COL].mean():.2f}, "
        f"median={df[TARGET_COL].median():.2f}, "
        f"std={df[TARGET_COL].std():.2f}"
    )

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    groups = df["driver_id"].values

    split = build_eval_split(df)
    train_idx, val_idx = split.train_idx, split.val_idx
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    n_train_drivers = len(set(groups[train_idx]))
    n_val_drivers = len(set(groups[val_idx]))
    print(f"  Split: {split.split_name} ({split.train_label} -> {split.val_label})")
    print(
        f"  Train: {len(X_train):,} rows ({n_train_drivers:,} drivers)   "
        f"Val: {len(X_val):,} rows ({n_val_drivers:,} drivers)"
    )

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain, free_raw_data=False)

    print("\n  Training LightGBM regressor on train split...")
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
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    mae = float(mean_absolute_error(y_val, y_pred_val))
    r2 = float(r2_score(y_val, y_pred_val))
    corr = _corr(y_val, y_pred_val)
    best_iteration = int(model.best_iteration or NUM_ROUNDS)

    print("\n  Validation metrics:")
    print(f"    RMSE:  ${rmse:.2f}")
    print(f"    MAE:   ${mae:.2f}")
    print(f"    R^2:   {r2:.4f}")
    print(f"    Corr:  {corr:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with temporal_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "split_name",
                "train_label",
                "val_label",
                "n_train_rows",
                "n_val_rows",
                "n_train_drivers",
                "n_val_drivers",
                "r2",
                "rmse",
                "mae",
                "corr",
                "best_iteration",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "LightGBM (tuned)",
                "split_name": split.split_name,
                "train_label": split.train_label,
                "val_label": split.val_label,
                "n_train_rows": len(X_train),
                "n_val_rows": len(X_val),
                "n_train_drivers": n_train_drivers,
                "n_val_drivers": n_val_drivers,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "best_iteration": best_iteration,
            }
        )

    print(f"\n  Retraining final model on all rows for {best_iteration} boosting rounds...")
    dall = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS, free_raw_data=False)
    final_model = lgb.train(
        LGB_PARAMS,
        dall,
        num_boost_round=best_iteration,
        valid_sets=[dall],
        valid_names=["train_all"],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    imp = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": final_model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    print("\n  Feature importance (gain):")
    for _, row in imp.iterrows():
        bar = "#" * int(row["importance"] / imp["importance"].max() * 30)
        print(f"    {row['feature']:30s}  {row['importance']:>10.0f}  {bar}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    imp.to_csv(importance_path, index=False)

    print(f"\n  Model saved: {model_path}")
    print(f"  Feature importance saved: {importance_path}")
    print(f"  Temporal validation saved: {temporal_path}")
    print(f"  Best iteration: {best_iteration}")
    print("=== Training Complete ===")


if __name__ == "__main__":
    main()
