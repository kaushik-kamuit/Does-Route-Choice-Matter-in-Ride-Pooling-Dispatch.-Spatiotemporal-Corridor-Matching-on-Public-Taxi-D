"""
Preprocess raw 2015 NYC TLC data into clean driver and rider datasets.

Steps:
  1. Rename columns to project conventions
  2. Clean outliers (extreme distance, fare, duration)
  3. Add derived temporal features
  4. Add H3 cells (resolution 9) for pickup and dropoff
  5. Mark temporal split (Jan-Mar = train, Apr = test)
  6. Split into drivers (>10 mi) and riders (0.5-10 mi)
  7. Save to data/processed/

Output:
  data/processed/<domain>/drivers.parquet  -- all driver trips
    Columns use origin/dest naming: origin_lat, origin_lng, dest_lat, dest_lng, origin_h3, dest_h3
  data/processed/<domain>/riders.parquet   -- sampled rider trips
    Columns use pickup/dropoff naming: pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, pickup_h3, dropoff_h3
"""

import gc
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.domain_config import DEFAULT_MONTH_WINDOW, YEAR, get_domain_config

H3_RESOLUTION = 9
RIDER_SAMPLE_FRAC = 0.25
RIDER_SAMPLE_SEED = 42

DRIVER_MIN_MILES = 10.0
RIDER_MIN_MILES = 0.5
RIDER_MAX_MILES = 10.0

DRIVER_COLUMN_RENAMES = {
    "pickup_lat": "origin_lat",
    "pickup_lng": "origin_lng",
    "dropoff_lat": "dest_lat",
    "dropoff_lng": "dest_lng",
    "pickup_h3": "origin_h3",
    "dropoff_h3": "dest_h3",
}

def load_raw(config, month: int) -> pd.DataFrame:
    path = config.raw_month_path(month)
    df = pd.read_parquet(path)
    return df.rename(columns=config.column_renames)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to remove invalid/outlier rows."""
    n_before = len(df)

    df["duration_min"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60.0

    mask = (
        df["trip_distance_miles"].between(0.3, 100)
        & df["fare_amount"].between(5.0, 200.0)
        & df["tip_amount"].between(0, 200)
        & (df["total_amount"] > 0)
        & df["duration_min"].between(1, 180)
        & df["passenger_count"].between(1, 6)
    )
    df = df.loc[mask].copy()
    print(f"    Cleaning: {n_before:,} -> {len(df):,} "
          f"(removed {n_before - len(df):,}, {(n_before - len(df)) / n_before * 100:.1f}%)")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["pickup_datetime"]
    df["date"] = dt.dt.date
    df["hour_of_day"] = dt.dt.hour.astype(np.int8)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    return df


def add_h3_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Compute H3 index for pickup and dropoff locations."""
    n = len(df)
    print(f"    Computing H3 cells for {n:,} rows (res={H3_RESOLUTION})...")
    t0 = time.time()

    pickup_lats = df["pickup_lat"].values
    pickup_lngs = df["pickup_lng"].values
    dropoff_lats = df["dropoff_lat"].values
    dropoff_lngs = df["dropoff_lng"].values

    pickup_h3 = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(pickup_lats, pickup_lngs)
    ]
    elapsed = time.time() - t0
    print(f"      Pickup H3 done in {elapsed:.1f}s "
          f"({n / elapsed:,.0f} rows/s)")

    t1 = time.time()
    dropoff_h3 = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(dropoff_lats, dropoff_lngs)
    ]
    elapsed2 = time.time() - t1
    print(f"      Dropoff H3 done in {elapsed2:.1f}s")

    df["pickup_h3"] = pickup_h3
    df["dropoff_h3"] = dropoff_h3
    print(f"      Unique pickup cells: {df['pickup_h3'].nunique():,}  "
          f"Unique dropoff cells: {df['dropoff_h3'].nunique():,}")
    return df


def add_split_label(df: pd.DataFrame, *, train_months: set[int], test_months: set[int], month_order: dict[int, int]) -> pd.DataFrame:
    df["split"] = np.where(df["month"].isin(train_months), "train", "test")
    ordered = df["month"].astype("Int64").map(month_order)
    if ordered.isna().any():
        missing = sorted(set(df.loc[ordered.isna(), "month"].dropna().astype(int).unique().tolist()))
        raise ValueError(f"Encountered months outside the configured window mapping: {missing}")
    df["service_window_pos"] = ordered.astype(np.int8)
    return df


def process_month(config, month: int, *, train_months: set[int], test_months: set[int], month_order: dict[int, int]) -> pd.DataFrame:
    print(f"\n  Loading {YEAR}-{month:02d}...")
    df = load_raw(config, month)
    print(f"    Raw rows: {len(df):,}")

    df = clean(df)
    df = add_temporal_features(df)
    df = add_h3_cells(df)
    df = add_split_label(df, train_months=train_months, test_months=test_months, month_order=month_order)
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw NYC TLC data into domain-specific driver/rider parquet files")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(DEFAULT_MONTH_WINDOW),
        help="Months to preprocess (default: Jan-Apr)",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit train months; default is all but the last processed month",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit test months; default is the last processed month",
    )
    args = parser.parse_args()

    config = get_domain_config(args.domain)
    out_dir = config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_months = sorted(dict.fromkeys(args.months))
    if len(ordered_months) < 2:
        raise ValueError("Expected at least two months so the last month can be held out for test.")
    train_months = set(args.train_months or ordered_months[:-1])
    test_months = set(args.test_months or ordered_months[-1:])
    month_order = {month: idx + 1 for idx, month in enumerate(ordered_months)}

    driver_chunks = []
    rider_chunks = []

    t_start = time.time()

    print(f"=== Preprocess {config.display_name} ({config.name}) ===")

    for month in ordered_months:
        print(f"\n{'='*60}")
        print(f"MONTH {month}")
        print(f"{'='*60}")

        df = process_month(
            config,
            month,
            train_months=train_months,
            test_months=test_months,
            month_order=month_order,
        )

        drivers = df.loc[df["trip_distance_miles"] > DRIVER_MIN_MILES].copy()
        drivers.rename(columns=DRIVER_COLUMN_RENAMES, inplace=True)
        riders_all = df.loc[
            df["trip_distance_miles"].between(RIDER_MIN_MILES, RIDER_MAX_MILES)
        ]
        riders = riders_all.sample(
            frac=RIDER_SAMPLE_FRAC, random_state=RIDER_SAMPLE_SEED
        ).copy()

        print(f"\n    Drivers (>{DRIVER_MIN_MILES} mi): {len(drivers):,}")
        print(f"    Riders  ({RIDER_MIN_MILES}-{RIDER_MAX_MILES} mi, "
              f"{RIDER_SAMPLE_FRAC:.0%} sample): {len(riders):,} "
              f"(of {len(riders_all):,} eligible)")

        driver_chunks.append(drivers)
        rider_chunks.append(riders)

        del df, drivers, riders_all, riders
        gc.collect()

    print(f"\n{'='*60}")
    print("SAVING")
    print(f"{'='*60}")

    all_drivers = pd.concat(driver_chunks, ignore_index=True)
    all_riders = pd.concat(rider_chunks, ignore_index=True)

    drv_path = out_dir / "drivers.parquet"
    rdr_path = out_dir / "riders.parquet"

    all_drivers.to_parquet(drv_path, compression="snappy", index=False)
    all_riders.to_parquet(rdr_path, compression="snappy", index=False)

    drv_mb = drv_path.stat().st_size / (1024 ** 2)
    rdr_mb = rdr_path.stat().st_size / (1024 ** 2)

    print(f"\n  drivers.parquet: {len(all_drivers):,} rows, {drv_mb:.1f} MB")
    print(f"  riders.parquet:  {len(all_riders):,} rows, {rdr_mb:.1f} MB")

    print(f"\n  --- Driver summary ---")
    print(f"  Train: {(all_drivers['split'] == 'train').sum():,}  "
          f"Test: {(all_drivers['split'] == 'test').sum():,}")
    print(f"  Distance: mean={all_drivers['trip_distance_miles'].mean():.1f} mi, "
          f"median={all_drivers['trip_distance_miles'].median():.1f} mi")
    print(f"  Fare: mean=${all_drivers['fare_amount'].mean():.2f}, "
          f"median=${all_drivers['fare_amount'].median():.2f}")

    print(f"\n  --- Rider summary ---")
    print(f"  Train: {(all_riders['split'] == 'train').sum():,}  "
          f"Test: {(all_riders['split'] == 'test').sum():,}")
    print(f"  Distance: mean={all_riders['trip_distance_miles'].mean():.1f} mi, "
          f"median={all_riders['trip_distance_miles'].median():.1f} mi")
    print(f"  Fare: mean=${all_riders['fare_amount'].mean():.2f}, "
          f"median=${all_riders['fare_amount'].median():.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
