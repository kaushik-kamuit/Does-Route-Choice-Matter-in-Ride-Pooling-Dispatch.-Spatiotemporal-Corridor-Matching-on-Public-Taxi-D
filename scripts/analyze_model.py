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

ds = pd.read_parquet(ROOT / "data/ml/training_dataset_v2.parquet")
print("=== DATASET SHAPE (v2) ===")
print(f"Rows: {len(ds):,}  Cols: {len(ds.columns)}")
print(f"Columns: {list(ds.columns)}")

EXCLUDE_COLS = {"driver_id", "route_idx", "expected_revenue", "driver_cost", "expected_profit"}
feats = [c for c in ds.columns if c not in EXCLUDE_COLS]

print("\n=== TARGET STATS ===")
t = ds["expected_profit"]
print(f"mean={t.mean():.2f} median={t.median():.2f} std={t.std():.2f} min={t.min():.2f} max={t.max():.2f}")
print(f"zeros: {(t == 0).sum()}  negatives: {(t < 0).sum()}")

print("\n=== FEATURE CORRELATIONS WITH TARGET ===")
for f in feats:
    corr = ds[f].corr(ds["expected_profit"])
    print(f"  {f:35s} r={corr:+.4f}")

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

SPLIT_SEED = 42
VAL_FRAC = 0.20

X_all = ds[feats].values
y = ds["expected_profit"].values
groups = ds["driver_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SPLIT_SEED)
train_idx, val_idx = next(gss.split(X_all, y, groups=groups))

model_v2 = joblib.load(ROOT / "models/profit_model_v2.pkl")
y_pred_v2 = model_v2.predict(X_all[val_idx])
r2_v2 = r2_score(y[val_idx], y_pred_v2)
rmse_v2 = np.sqrt(mean_squared_error(y[val_idx], y_pred_v2))

print(f"\n=== V2 MODEL METRICS ===")
print(f"  V2 model ({len(feats)} features): R²={r2_v2:.4f}  RMSE=${rmse_v2:.2f}")

print("\n=== RANK ACCURACY ===")
val_df = ds.iloc[val_idx].copy()
val_df["pred"] = model_v2.predict(X_all[val_idx])
correct, total = 0, 0
for _, grp in val_df.groupby("driver_id"):
    if len(grp) < 2:
        continue
    total += 1
    if grp["expected_profit"].idxmax() == grp["pred"].idxmax():
        correct += 1
acc = correct / total if total > 0 else 0
print(f"  Rank-1 accuracy = {acc:.1%} ({correct:,}/{total:,})")

print("\n=== FEATURE IMPORTANCE: V2 MODEL ===")
import lightgbm as lgb
imp = pd.DataFrame({
    "feature": feats,
    "importance": model_v2.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
for _, row in imp.iterrows():
    bar = "#" * int(row["importance"] / imp["importance"].max() * 40)
    print(f"  {row['feature']:35s} {row['importance']:>10.0f}  {bar}")
