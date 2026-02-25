"""Explore rider/driver data for feature engineering opportunities."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

print("=== DRIVERS DATA ===")
drivers = pd.read_parquet(ROOT / "data/processed/drivers.parquet")
print(f"Rows: {len(drivers):,}")
print(f"Columns: {list(drivers.columns)}")
print(f"Train: {(drivers['split']=='train').sum():,}  Test: {(drivers['split']=='test').sum():,}")
print(f"\nSample dtypes:")
for c in drivers.columns:
    print(f"  {c:25s} {str(drivers[c].dtype):15s} nulls={drivers[c].isna().sum()}")

print(f"\npickup_datetime range: {drivers['pickup_datetime'].min()} to {drivers['pickup_datetime'].max()}")
print(f"trip_distance_miles: mean={drivers['trip_distance_miles'].mean():.1f} med={drivers['trip_distance_miles'].median():.1f}")
print(f"fare_amount: mean={drivers['fare_amount'].mean():.2f} med={drivers['fare_amount'].median():.2f}")

print("\n\n=== RIDERS DATA ===")
riders = pd.read_parquet(ROOT / "data/processed/riders.parquet")
print(f"Rows: {len(riders):,}")
print(f"Columns: {list(riders.columns)}")
print(f"Train: {(riders['split']=='train').sum():,}  Test: {(riders['split']=='test').sum():,}")
print(f"\nSample dtypes:")
for c in riders.columns:
    print(f"  {c:25s} {str(riders[c].dtype):15s} nulls={riders[c].isna().sum()}")

print(f"\npickup_datetime range: {riders['pickup_datetime'].min()} to {riders['pickup_datetime'].max()}")
print(f"fare_amount: mean={riders['fare_amount'].mean():.2f} med={riders['fare_amount'].median():.2f}")

print("\n=== H3 CELL STATISTICS (TRAIN RIDERS) ===")
train_riders = riders[riders["split"] == "train"]
pickup_counts = train_riders.groupby("pickup_h3").size()
print(f"Unique pickup H3 cells: {len(pickup_counts):,}")
print(f"Riders per cell: mean={pickup_counts.mean():.1f} med={pickup_counts.median():.0f} max={pickup_counts.max():,}")

print("\n=== HOURLY DEMAND PATTERNS ===")
hourly = train_riders.groupby("hour_of_day").size()
print("  Hour  Count")
for h in range(24):
    if h in hourly.index:
        bar = "#" * int(hourly[h] / hourly.max() * 40)
        print(f"  {h:2d}    {hourly[h]:>8,}  {bar}")

print("\n=== TOP 20 PICKUP H3 CELLS ===")
top_cells = pickup_counts.nlargest(20)
for cell, cnt in top_cells.items():
    print(f"  {cell}  {cnt:>6,} riders")

print("\n=== FARE DISTRIBUTION BY HOUR ===")
fare_by_hour = train_riders.groupby("hour_of_day")["fare_amount"].agg(["mean", "std", "count"])
for h in range(24):
    if h in fare_by_hour.index:
        r = fare_by_hour.loc[h]
        print(f"  {h:2d}h  mean=${r['mean']:.2f}  std=${r['std']:.2f}  n={int(r['count']):,}")

del drivers, riders, train_riders
