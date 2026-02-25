"""
Download NYC TLC Yellow Taxi 2015 data from Azure Open Datasets.

Uses the public Azure blob storage (no auth required).
Downloads 4 months: Jan-Mar (training), Apr (testing).
Applies quality filters and column pruning to manage size.
"""

import sys
import time
from pathlib import Path

import pandas as pd

YEAR = 2015
MONTHS = [1, 2, 3, 4]

AZURE_STORAGE_OPTIONS = {
    "account_name": "azureopendatastorage",
    "anon": True,
}
BASE_PATH = "az://nyctlc/yellow"

COLUMNS = [
    "tpepPickupDateTime",
    "tpepDropoffDateTime",
    "startLat",
    "startLon",
    "endLat",
    "endLon",
    "tripDistance",
    "fareAmount",
    "tipAmount",
    "totalAmount",
    "passengerCount",
]

NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.26, -73.68

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid/unusable rows. Keep all trip lengths for flexible driver/rider split."""
    mask = (
        df["startLat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["startLon"].between(NYC_LON_MIN, NYC_LON_MAX)
        & df["endLat"].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df["endLon"].between(NYC_LON_MIN, NYC_LON_MAX)
        & (df["tripDistance"] > 0.3)
        & df["fareAmount"].between(2.50, 300)
        & (df["passengerCount"] > 0)
    )
    return df.loc[mask].copy()


def download_month(year: int, month: int) -> pd.DataFrame:
    """Download one month of data from Azure, selecting only needed columns."""
    path = f"{BASE_PATH}/puYear={year}/puMonth={month}/"
    print(f"  Reading from Azure: {path}")
    df = pd.read_parquet(
        path,
        columns=COLUMNS,
        storage_options=AZURE_STORAGE_OPTIONS,
    )
    return df


def print_stats(df: pd.DataFrame, label: str) -> None:
    print(f"  [{label}] rows={len(df):,}  "
          f"tripDist: median={df['tripDistance'].median():.1f} mi, "
          f"mean={df['tripDistance'].mean():.1f} mi  "
          f"fare: median=${df['fareAmount'].median():.2f}, "
          f"mean=${df['fareAmount'].mean():.2f}")
    print(f"  [{label}] trips >5mi: {(df['tripDistance'] > 5).sum():,}  "
          f"trips >10mi: {(df['tripDistance'] > 10).sum():,}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_bytes = 0

    for month in MONTHS:
        print(f"\n{'='*60}")
        print(f"Processing {YEAR}-{month:02d}")
        print(f"{'='*60}")

        t0 = time.time()
        df = download_month(YEAR, month)
        dl_time = time.time() - t0
        print(f"  Downloaded {len(df):,} rows in {dl_time:.1f}s")
        print_stats(df, "raw")

        df = quality_filter(df)
        print_stats(df, "filtered")

        out_path = OUTPUT_DIR / f"yellow_tripdata_{YEAR}-{month:02d}.parquet"
        df.to_parquet(out_path, compression="snappy", index=False)

        file_mb = out_path.stat().st_size / (1024 * 1024)
        total_rows += len(df)
        total_bytes += out_path.stat().st_size

        print(f"  Saved: {out_path.name} ({file_mb:.1f} MB, {len(df):,} rows)")
        del df

    print(f"\n{'='*60}")
    print(f"DONE: {total_rows:,} total rows, {total_bytes / (1024**2):.1f} MB on disk")
    print(f"Files in: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
