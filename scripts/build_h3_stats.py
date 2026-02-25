"""
Pre-compute H3 cell-level demand statistics from the training riders.

Output: data/ml/h3_cell_stats.parquet
  - h3_cell: H3 cell ID
  - pickup_count: total pickups in this cell (train period)
  - dropoff_count: total dropoffs in this cell
  - mean_fare: average rider fare for pickups in this cell
  - median_fare: median rider fare
  - mean_distance: average trip distance for riders starting here
  - fare_std: fare standard deviation
  - per-hour demand (h0..h23): pickup count in each hour

These stats can be looked up for each corridor cell to create
aggregate spatial features without running match_riders.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import time

OUT_PATH = ROOT / "data" / "ml" / "h3_cell_stats.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=== Building H3 Cell Statistics ===")
t0 = time.time()

riders = pd.read_parquet(ROOT / "data/processed/riders.parquet")
train = riders[riders["split"] == "train"].reset_index(drop=True)
del riders
print(f"  Train riders: {len(train):,}")

print("  Computing pickup stats...")
pickup_stats = train.groupby("pickup_h3").agg(
    pickup_count=("fare_amount", "count"),
    mean_fare=("fare_amount", "mean"),
    median_fare=("fare_amount", "median"),
    fare_std=("fare_amount", "std"),
    mean_distance=("trip_distance_miles", "mean"),
    mean_pax=("passenger_count", "mean"),
).reset_index().rename(columns={"pickup_h3": "h3_cell"})

print("  Computing dropoff stats...")
dropoff_counts = train.groupby("dropoff_h3").size().reset_index()
dropoff_counts.columns = ["h3_cell", "dropoff_count"]

stats = pickup_stats.merge(dropoff_counts, on="h3_cell", how="outer")
stats["pickup_count"] = stats["pickup_count"].fillna(0).astype(int)
stats["dropoff_count"] = stats["dropoff_count"].fillna(0).astype(int)
stats["mean_fare"] = stats["mean_fare"].fillna(0)
stats["median_fare"] = stats["median_fare"].fillna(0)
stats["fare_std"] = stats["fare_std"].fillna(0)
stats["mean_distance"] = stats["mean_distance"].fillna(0)
stats["mean_pax"] = stats["mean_pax"].fillna(0)

print("  Computing hourly pickup counts per cell...")
hourly = train.groupby(["pickup_h3", "hour_of_day"]).size().unstack(fill_value=0)
hourly.columns = [f"h{h}" for h in hourly.columns]
hourly = hourly.reset_index().rename(columns={"pickup_h3": "h3_cell"})
stats = stats.merge(hourly, on="h3_cell", how="left")
for h in range(24):
    col = f"h{h}"
    if col not in stats.columns:
        stats[col] = 0

print("  Computing 15-min bin stats...")
dt = train["pickup_datetime"]
train["qh"] = (dt.dt.hour * 4 + dt.dt.minute // 15).astype(np.int8)
qh_stats = train.groupby(["pickup_h3", "qh"]).agg(
    qh_count=("fare_amount", "count"),
    qh_mean_fare=("fare_amount", "mean"),
).reset_index()
qh_stats.rename(columns={"pickup_h3": "h3_cell"}, inplace=True)
qh_path = ROOT / "data" / "ml" / "h3_qh_stats.parquet"
qh_stats.to_parquet(qh_path, compression="snappy", index=False)
print(f"  Saved QH stats: {len(qh_stats):,} rows -> {qh_path.name}")

stats.to_parquet(OUT_PATH, compression="snappy", index=False)
elapsed = time.time() - t0
print(f"\n  H3 stats: {len(stats):,} cells")
print(f"  Saved to: {OUT_PATH}")
print(f"  Time: {elapsed:.1f}s")
print(f"  Top pickup cells:")
for _, r in stats.nlargest(5, "pickup_count").iterrows():
    print(f"    {r['h3_cell']}  pickups={int(r['pickup_count']):,}  "
          f"mean_fare=${r['mean_fare']:.2f}")
