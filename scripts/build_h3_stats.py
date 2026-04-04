"""
Pre-compute H3 cell-level demand statistics from the training riders.

Supports domain-aware outputs so Yellow and Green artifacts stay separate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.domain_config import get_domain_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build H3 cell statistics from train riders")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    args = parser.parse_args()

    config = get_domain_config(args.domain)
    out_path = config.h3_stats_path()
    qh_path = config.h3_qh_stats_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qh_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Building H3 Cell Statistics ({config.display_name}) ===")
    t0 = time.time()

    riders = pd.read_parquet(config.riders_path())
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
    for hour in range(24):
        col = f"h{hour}"
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
    qh_stats.to_parquet(qh_path, compression="snappy", index=False)
    print(f"  Saved QH stats: {len(qh_stats):,} rows -> {qh_path}")

    stats.to_parquet(out_path, compression="snappy", index=False)
    elapsed = time.time() - t0
    print(f"\n  H3 stats: {len(stats):,} cells")
    print(f"  Saved to: {out_path}")
    print(f"  Time: {elapsed:.1f}s")
    print("  Top pickup cells:")
    for _, row in stats.nlargest(5, "pickup_count").iterrows():
        print(
            f"    {row['h3_cell']}  pickups={int(row['pickup_count']):,}  "
            f"mean_fare=${row['mean_fare']:.2f}"
        )


if __name__ == "__main__":
    main()
