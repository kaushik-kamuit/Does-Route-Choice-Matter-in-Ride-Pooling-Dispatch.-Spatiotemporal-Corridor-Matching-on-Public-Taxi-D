"""Deep analysis of current model and dataset for improvement planning."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

ds = pd.read_parquet(ROOT / "data/ml/training_dataset.parquet")
print("=== DATASET SHAPE ===")
print(f"Rows: {len(ds):,}  Cols: {len(ds.columns)}")
print(f"Columns: {list(ds.columns)}")

print("\n=== TARGET STATS ===")
t = ds["expected_profit"]
print(f"mean={t.mean():.2f} median={t.median():.2f} std={t.std():.2f} min={t.min():.2f} max={t.max():.2f}")
print(f"zeros: {(t == 0).sum()}  negatives: {(t < 0).sum()}")

print("\n=== FEATURE CORRELATIONS WITH TARGET ===")
feats = ["route_distance_m", "route_duration_s", "corridor_cell_count", "hour_of_day",
         "day_of_week", "is_weekend", "corridor_rider_count", "corridor_demand_density",
         "mean_rider_fare", "corridor_fare_density", "feasible_rider_count", "matched_rider_count"]
for f in feats:
    corr = ds[f].corr(ds["expected_profit"])
    print(f"  {f:30s} r={corr:+.4f}")

print("\n=== MATCHED_RIDER_COUNT DISTRIBUTION ===")
m = ds["matched_rider_count"]
for v in range(4):
    print(f"  {v} riders: {(m == v).sum():,} ({(m == v).mean() * 100:.1f}%)")

print("\n=== ROUTES PER DRIVER ===")
rpd = ds.groupby("driver_id")["route_idx"].count()
for v in [1, 2, 3]:
    print(f"  {v} routes: {(rpd == v).sum():,}")
print(f"  Total drivers: {ds['driver_id'].nunique():,}")

print("\n=== PROFIT VARIANCE WITHIN DRIVERS ===")
profit_std = ds.groupby("driver_id")["expected_profit"].std().dropna()
print(f"  Mean within-driver profit std: ${profit_std.mean():.2f}")
print(f"  Median within-driver profit std: ${profit_std.median():.2f}")
diff = ds.groupby("driver_id")["expected_profit"].agg(["max", "min"])
diff["range"] = diff["max"] - diff["min"]
print(f"  Mean within-driver profit range: ${diff['range'].mean():.2f}")
print(f"  Median within-driver profit range: ${diff['range'].median():.2f}")
print(f"  Drivers with zero range: {(diff['range'] == 0).sum():,}")

print("\n=== MODEL ABLATION: REMOVE matched_rider_count + feasible_rider_count ===")
SPLIT_SEED = 42
VAL_FRAC = 0.20

X_all = ds[feats].values
y = ds["expected_profit"].values
groups = ds["driver_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SPLIT_SEED)
train_idx, val_idx = next(gss.split(X_all, y, groups=groups))

model_full = joblib.load(ROOT / "models/profit_model.pkl")
y_pred_full = model_full.predict(X_all[val_idx])
r2_full = r2_score(y[val_idx], y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y[val_idx], y_pred_full))

feats_no_match = [f for f in feats if f not in ("matched_rider_count", "feasible_rider_count")]
print(f"  Features (no match): {feats_no_match}")

import lightgbm as lgb
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
}

X_nm = ds[feats_no_match].values
dtrain_nm = lgb.Dataset(X_nm[train_idx], label=y[train_idx], feature_name=feats_no_match, free_raw_data=False)
dval_nm = lgb.Dataset(X_nm[val_idx], label=y[val_idx], feature_name=feats_no_match, reference=dtrain_nm, free_raw_data=False)
model_nm = lgb.train(LGB_PARAMS, dtrain_nm, num_boost_round=1000,
                     valid_sets=[dval_nm], valid_names=["val"],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
y_pred_nm = model_nm.predict(X_nm[val_idx])
r2_nm = r2_score(y[val_idx], y_pred_nm)
rmse_nm = np.sqrt(mean_squared_error(y[val_idx], y_pred_nm))

print(f"\n  Full model  (12 features): R²={r2_full:.4f}  RMSE=${rmse_full:.2f}")
print(f"  No-match model (10 feats): R²={r2_nm:.4f}  RMSE=${rmse_nm:.2f}")
print(f"  Delta R²: {r2_full - r2_nm:.4f}")
print(f"  Delta RMSE: ${rmse_nm - rmse_full:.2f}")

print("\n=== RANK ACCURACY COMPARISON ===")
for label, model_obj, X_use, feat_names in [
    ("Full (12 feats)", model_full, X_all, feats),
    ("No-match (10 feats)", model_nm, X_nm, feats_no_match),
]:
    val_df = ds.iloc[val_idx].copy()
    val_df["pred"] = model_obj.predict(X_use[val_idx])
    correct, total = 0, 0
    for _, grp in val_df.groupby("driver_id"):
        if len(grp) < 2:
            continue
        total += 1
        actual_best = grp["expected_profit"].idxmax()
        pred_best = grp["pred"].idxmax()
        if actual_best == pred_best:
            correct += 1
    acc = correct / total if total > 0 else 0
    print(f"  {label:25s}: rank-1 accuracy = {acc:.1%} ({correct:,}/{total:,})")

print("\n=== FEATURE IMPORTANCE: NO-MATCH MODEL ===")
imp = pd.DataFrame({
    "feature": feats_no_match,
    "importance": model_nm.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
for _, row in imp.iterrows():
    bar = "#" * int(row["importance"] / imp["importance"].max() * 40)
    print(f"  {row['feature']:30s} {row['importance']:>10.0f}  {bar}")
