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
  data/processed/drivers.parquet  -- all driver trips (~2-3M rows)
    Columns use origin/dest naming: origin_lat, origin_lng, dest_lat, dest_lng, origin_h3, dest_h3
  data/processed/riders.parquet   -- sampled rider trips (~10M rows)
    Columns use pickup/dropoff naming: pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, pickup_h3, dropoff_h3
"""

import gc
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

YEAR = 2015
MONTHS = [1, 2, 3, 4]
TRAIN_MONTHS = {1, 2, 3}
TEST_MONTHS = {4}

H3_RESOLUTION = 9
RIDER_SAMPLE_FRAC = 0.25
RIDER_SAMPLE_SEED = 42

DRIVER_MIN_MILES = 10.0
RIDER_MIN_MILES = 0.5
RIDER_MAX_MILES = 10.0

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

DRIVER_COLUMN_RENAMES = {
    "pickup_lat": "origin_lat",
    "pickup_lng": "origin_lng",
    "dropoff_lat": "dest_lat",
    "dropoff_lng": "dest_lng",
    "pickup_h3": "origin_h3",
    "dropoff_h3": "dest_h3",
}

COLUMN_RENAMES = {
    "tpepPickupDateTime": "pickup_datetime",
    "tpepDropoffDateTime": "dropoff_datetime",
    "startLat": "pickup_lat",
    "startLon": "pickup_lng",
    "endLat": "dropoff_lat",
    "endLon": "dropoff_lng",
    "tripDistance": "trip_distance_miles",
    "fareAmount": "fare_amount",
    "tipAmount": "tip_amount",
    "totalAmount": "total_amount",
    "passengerCount": "passenger_count",
}


def load_raw(month: int) -> pd.DataFrame:
    path = RAW_DIR / f"yellow_tripdata_{YEAR}-{month:02d}.parquet"
    df = pd.read_parquet(path)
    return df.rename(columns=COLUMN_RENAMES)


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


def add_split_label(df: pd.DataFrame) -> pd.DataFrame:
    df["split"] = np.where(df["month"].isin(TRAIN_MONTHS), "train", "test")
    return df


def process_month(month: int) -> pd.DataFrame:
    print(f"\n  Loading {YEAR}-{month:02d}...")
    df = load_raw(month)
    print(f"    Raw rows: {len(df):,}")

    df = clean(df)
    df = add_temporal_features(df)
    df = add_h3_cells(df)
    df = add_split_label(df)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    driver_chunks = []
    rider_chunks = []

    t_start = time.time()

    for month in MONTHS:
        print(f"\n{'='*60}")
        print(f"MONTH {month}")
        print(f"{'='*60}")

        df = process_month(month)

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

    drv_path = OUT_DIR / "drivers.parquet"
    rdr_path = OUT_DIR / "riders.parquet"

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
    print(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
