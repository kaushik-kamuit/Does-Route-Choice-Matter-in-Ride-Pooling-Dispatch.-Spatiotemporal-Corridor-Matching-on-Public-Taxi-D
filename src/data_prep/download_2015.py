"""
Download NYC TLC Yellow or Green Taxi 2015 data from Azure Open Datasets.

Uses the public Azure blob storage (no auth required).
Supports domain-aware output paths so Yellow and Green artifacts stay isolated.
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.domain_config import DEFAULT_MONTH_WINDOW, YEAR, get_domain_config


AZURE_STORAGE_OPTIONS = {
    "account_name": "azureopendatastorage",
    "anon": True,
}

NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.26, -73.68


def _source_column(config, canonical_name: str) -> str:
    for raw_name, renamed in config.column_renames.items():
        if renamed == canonical_name:
            return raw_name
    raise KeyError(f"Domain {config.name!r} does not define a source column for {canonical_name!r}")


def quality_filter(df: pd.DataFrame, config) -> pd.DataFrame:
    """Remove invalid/unusable rows. Keep all trip lengths for flexible driver/rider split."""
    pickup_lat_col = _source_column(config, "pickup_lat")
    pickup_lng_col = _source_column(config, "pickup_lng")
    dropoff_lat_col = _source_column(config, "dropoff_lat")
    dropoff_lng_col = _source_column(config, "dropoff_lng")
    passenger_count_col = _source_column(config, "passenger_count")
    mask = (
        df[pickup_lat_col].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df[pickup_lng_col].between(NYC_LON_MIN, NYC_LON_MAX)
        & df[dropoff_lat_col].between(NYC_LAT_MIN, NYC_LAT_MAX)
        & df[dropoff_lng_col].between(NYC_LON_MIN, NYC_LON_MAX)
        & (df["tripDistance"] > 0.3)
        & df["fareAmount"].between(2.50, 300)
        & (df[passenger_count_col] > 0)
    )
    return df.loc[mask].copy()


def download_month(base_path: str, columns: tuple[str, ...], year: int, month: int) -> pd.DataFrame:
    """Download one month of data from Azure, selecting only needed columns."""
    path = f"{base_path}/puYear={year}/puMonth={month}/"
    print(f"  Reading from Azure: {path}")
    df = pd.read_parquet(
        path,
        columns=list(columns),
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
    import argparse

    parser = argparse.ArgumentParser(description="Download NYC TLC 2015 taxi data from Azure Open Datasets")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(DEFAULT_MONTH_WINDOW),
        help="Months to download (default: Jan-Apr)",
    )
    args = parser.parse_args()

    config = get_domain_config(args.domain)
    output_dir = config.raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_bytes = 0

    print(f"Domain: {config.display_name} ({config.name})")

    for month in args.months:
        print(f"\n{'='*60}")
        print(f"Processing {config.name} {YEAR}-{month:02d}")
        print(f"{'='*60}")

        t0 = time.time()
        df = download_month(config.azure_base_path, config.raw_columns, YEAR, month)
        dl_time = time.time() - t0
        print(f"  Downloaded {len(df):,} rows in {dl_time:.1f}s")
        print_stats(df, "raw")

        df = quality_filter(df, config)
        print_stats(df, "filtered")

        out_path = output_dir / f"{config.raw_filename_prefix}_{YEAR}-{month:02d}.parquet"
        df.to_parquet(out_path, compression="snappy", index=False)

        file_mb = out_path.stat().st_size / (1024 * 1024)
        total_rows += len(df)
        total_bytes += out_path.stat().st_size

        print(f"  Saved: {out_path.name} ({file_mb:.1f} MB, {len(df):,} rows)")
        del df

    print(f"\n{'='*60}")
    print(f"DONE: {total_rows:,} total rows, {total_bytes / (1024**2):.1f} MB on disk")
    print(f"Files in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
