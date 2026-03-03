"""
Feature ablation study: train LightGBM on feature subsets to quantify
each feature group's contribution to profit prediction and route ranking.

Feature groups:
  - Geometric:      route_distance_m, route_duration_s, corridor_cell_count,
                    route_sinuosity, route_avg_speed_ms, bearing_sin/cos,
                    straight_line_dist_m
  - Temporal:       hour_of_day, day_of_week, is_weekend, day_of_month,
                    time_bin_15min, hour_sin, hour_cos
  - Spatial Demand: corridor_rider_count, corridor_demand_density,
                    mean_rider_fare, corridor_fare_density,
                    corridor_hist_* (6), origin_cell_*, dest_cell_dropoffs
  - Landmark:       origin/dest distances to JFK, LGA, Penn, Times Sq,
                    origin/dest_landmark_dist_km

Experiments:
  1. All features
  2-5. Each group alone
  6-9. All minus each group
"""

import sys
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

DATASET_PATH = ROOT / "data" / "ml" / "training_dataset_v2.parquet"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 42
VAL_FRAC = 0.20
TARGET = "expected_profit"
EXCLUDE_COLS = {"driver_id", "route_idx", "expected_revenue", "driver_cost", "expected_profit"}

GEOMETRIC = [
    "route_distance_m", "route_duration_s", "corridor_cell_count",
    "route_sinuosity", "route_avg_speed_ms", "bearing_sin", "bearing_cos",
    "straight_line_dist_m",
]

TEMPORAL = [
    "hour_of_day", "day_of_week", "is_weekend", "day_of_month",
    "time_bin_15min", "hour_sin", "hour_cos",
]

SPATIAL_DEMAND = [
    "corridor_rider_count", "corridor_demand_density",
    "mean_rider_fare", "corridor_fare_density",
    "corridor_hist_pickups", "corridor_hist_dropoffs",
    "corridor_hist_pickup_density", "corridor_hist_dropoff_density",
    "corridor_hist_mean_fare", "corridor_hist_fare_density",
    "origin_cell_pickups", "origin_cell_mean_fare", "dest_cell_dropoffs",
]

LANDMARK = [
    "origin_landmark_dist_km", "dest_landmark_dist_km",
    "origin_jfk_km", "origin_lga_km", "origin_penn_km", "origin_times_sq_km",
    "dest_jfk_km", "dest_lga_km", "dest_penn_km", "dest_times_sq_km",
]

ALL_GROUPS = {
    "Geometric": GEOMETRIC,
    "Temporal": TEMPORAL,
    "Spatial Demand": SPATIAL_DEMAND,
    "Landmark": LANDMARK,
}


def rank_accuracy(df, val_idx, y_pred):
    val_df = df.iloc[val_idx][["driver_id", "expected_profit"]].copy()
    val_df["pred"] = y_pred
    correct, total = 0, 0
    for _, grp in val_df.groupby("driver_id"):
        if len(grp) < 2:
            continue
        total += 1
        if grp["expected_profit"].idxmax() == grp["pred"].idxmax():
            correct += 1
    return correct / total if total > 0 else 0


def train_and_evaluate(X_tr, y_tr, X_va, y_va, feat_names, df, val_idx):
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10,
        "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.7,
        "reg_alpha": 0.05, "reg_lambda": 0.5,
        "min_split_gain": 0.01, "verbose": -1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, feature_name=feat_names, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    y_pred = model.predict(X_va)
    r2 = r2_score(y_va, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
    mae = float(mean_absolute_error(y_va, y_pred))
    rank_acc = rank_accuracy(df, val_idx, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae, "rank_acc": rank_acc,
            "n_features": len(feat_names), "best_iter": model.best_iteration}


def main():
    print("=== Feature Ablation Study ===\n")

    print("  Loading dataset...", end=" ")
    df = pd.read_parquet(DATASET_PATH)
    all_feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"{len(df):,} rows, {len(all_feat_cols)} features")

    X_all = df[all_feat_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    groups = df["driver_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SPLIT_SEED)
    train_idx, val_idx = next(gss.split(X_all, y, groups=groups))
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}\n")

    experiments = [("All features", all_feat_cols)]

    for gname, gcols in ALL_GROUPS.items():
        present = [c for c in gcols if c in all_feat_cols]
        if present:
            experiments.append((f"Only {gname}", present))

    for gname, gcols in ALL_GROUPS.items():
        present = [c for c in gcols if c in all_feat_cols]
        remaining = [c for c in all_feat_cols if c not in present]
        if remaining and present:
            experiments.append((f"All minus {gname}", remaining))

    results = []
    for exp_name, feat_cols in experiments:
        print(f"  [{len(results)+1}/{len(experiments)}] {exp_name} "
              f"({len(feat_cols)} features)...")
        sys.stdout.flush()

        col_indices = [all_feat_cols.index(c) for c in feat_cols]
        X_sub = X_all[:, col_indices]

        result = train_and_evaluate(
            X_sub[train_idx], y[train_idx],
            X_sub[val_idx], y[val_idx],
            feat_cols, df, val_idx,
        )
        result["experiment"] = exp_name
        result["features"] = ", ".join(feat_cols[:5]) + ("..." if len(feat_cols) > 5 else "")
        results.append(result)

        print(f"    R2={result['r2']:.4f}  RMSE=${result['rmse']:.2f}  "
              f"Rank-1={result['rank_acc']:.1%}  "
              f"(iter={result['best_iter']})")
        sys.stdout.flush()

    # Summary table
    print(f"\n{'='*85}")
    print("ABLATION SUMMARY")
    print(f"{'='*85}")
    print(f"{'Experiment':30s} {'N_feat':>6s} {'R2':>8s} {'RMSE':>8s} {'Rank-1':>8s}")
    print("-" * 85)
    for r in results:
        print(f"{r['experiment']:30s} {r['n_features']:6d} "
              f"{r['r2']:8.4f} ${r['rmse']:7.2f} {r['rank_acc']:7.1%}")

    out_path = RESULTS_DIR / "ablation_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
