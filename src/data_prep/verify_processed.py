"""Verify processed driver and rider datasets."""
from pathlib import Path
import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

for name in ["drivers", "riders"]:
    df = pd.read_parquet(PROC_DIR / f"{name}.parquet")
    pickup_h3_col = "pickup_h3" if "pickup_h3" in df.columns else "origin_h3"
    dropoff_h3_col = "dropoff_h3" if "dropoff_h3" in df.columns else "dest_h3"
    print(f"{'='*60}")
    print(f"  {name.upper()}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"{'='*60}")
    print(f"  Columns: {list(df.columns)}\n")
    print("  Dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:25s} {dtype}")

    print(f"\n  Split: train={df[df['split']=='train'].shape[0]:,}  "
          f"test={df[df['split']=='test'].shape[0]:,}")
    print(f"  H3 pickup cells:  {df[pickup_h3_col].nunique():,}")
    print(f"  H3 dropoff cells: {df[dropoff_h3_col].nunique():,}")

    print(f"\n  Key stats:")
    for col in ["trip_distance_miles", "fare_amount", "duration_min"]:
        s = df[col]
        print(f"    {col:25s}  min={s.min():.1f}  median={s.median():.1f}  "
              f"mean={s.mean():.1f}  max={s.max():.1f}")

    print(f"\n  First 2 rows:")
    print(df.head(2).to_string(index=False))
    print()

total_mb = sum(f.stat().st_size for f in PROC_DIR.glob("*.parquet")) / (1024**2)
print(f"Total disk: {total_mb:.1f} MB")
