"""Quick verification of downloaded 2015 data."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

df = pd.read_parquet(RAW_DIR / "yellow_tripdata_2015-01.parquet")

print("=== COLUMNS & DTYPES ===")
print(df.dtypes)
print(f"\nShape: {df.shape}")

print("\n=== FIRST 3 ROWS ===")
print(df.head(3).to_string())

print("\n=== COORDINATE CARDINALITY (proves real GPS, not zone centroids) ===")
for col in ["startLat", "startLon", "endLat", "endLon"]:
    print(f"  {col}: {df[col].nunique():>10,} unique values")

print("\n=== KEY COLUMN STATS ===")
cols = ["startLat", "startLon", "endLat", "endLon",
        "tripDistance", "fareAmount", "tipAmount", "totalAmount"]
print(df[cols].describe().round(3).to_string())

print("\n=== TRIP DISTANCE DISTRIBUTION ===")
bins = [0, 1, 2, 5, 10, 20, 50, 1000]
labels = ["0-1mi", "1-2mi", "2-5mi", "5-10mi", "10-20mi", "20-50mi", "50+mi"]
dist = pd.cut(df["tripDistance"], bins=bins, labels=labels).value_counts().sort_index()
for label, count in dist.items():
    pct = count / len(df) * 100
    print(f"  {label:>8s}: {count:>10,} ({pct:5.1f}%)")

print("\n=== FARE DISTRIBUTION ===")
bins_f = [0, 5, 10, 20, 50, 100, 300]
labels_f = ["$0-5", "$5-10", "$10-20", "$20-50", "$50-100", "$100-300"]
fdist = pd.cut(df["fareAmount"], bins=bins_f, labels=labels_f).value_counts().sort_index()
for label, count in fdist.items():
    pct = count / len(df) * 100
    print(f"  {label:>10s}: {count:>10,} ({pct:5.1f}%)")

print("\n=== DISK USAGE ===")
for f in sorted(RAW_DIR.glob("*.parquet")):
    mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {mb:.1f} MB")
total = sum(f.stat().st_size for f in RAW_DIR.glob("*.parquet"))
print(f"  TOTAL: {total / (1024**2):.1f} MB")
