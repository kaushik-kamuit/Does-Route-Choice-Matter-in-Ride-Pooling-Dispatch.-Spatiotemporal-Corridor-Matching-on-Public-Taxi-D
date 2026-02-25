"""
Augment v1 training dataset with v2 features WITHOUT re-running match_riders.

Strategy:
  - Keep v1's expected_profit/revenue/cost labels (already computed via match_riders)
  - Drop leaky features: matched_rider_count, feasible_rider_count
  - Add new features from driver metadata + H3 stats + route geometry
  - Produces training_dataset_v2.parquet in ~15 minutes instead of ~3 hours
"""
import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from spatial.router import OSRMRouter
from spatial.corridor import build_corridor
from spatial.h3_utils import haversine_m
import h3

V1_PATH = ROOT / "data/ml/training_dataset.parquet"
H3_STATS_PATH = ROOT / "data/ml/h3_cell_stats.parquet"
DRIVERS_PATH = ROOT / "data/processed/drivers.parquet"
OUTPUT_PATH = ROOT / "data/ml/training_dataset_v2.parquet"
CACHE_PATH = ROOT / "data/route_cache.db"

SAMPLE_SEED = 42
SAMPLE_SIZE = 100_000

LANDMARKS = {
    "jfk":          (40.6413, -73.7781),
    "lga":          (40.7769, -73.8740),
    "penn":         (40.7505, -73.9935),
    "times_sq":     (40.7580, -73.9855),
    "grand_cntrl":  (40.7527, -73.9772),
    "world_trade":  (40.7127, -74.0134),
}


def bearing_deg(lat1, lng1, lat2, lng2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    x = math.sin(dlng) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlng)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compute_landmark_dists(lat, lng):
    dists = {}
    for name, (llat, llng) in LANDMARKS.items():
        dists[name] = haversine_m((lat, lng), (llat, llng)) / 1000.0
    nearest_dist = min(dists.values())
    return nearest_dist, dists


def corridor_hist_stats(corridor_cells, h3_stats_dict):
    total_pu = 0
    total_do = 0
    fare_sum = 0.0
    fare_vals = []

    for cell in corridor_cells:
        row = h3_stats_dict.get(cell)
        if row is None:
            continue
        total_pu += row["pickup_count"]
        total_do += row["dropoff_count"]
        fare_sum += row["mean_fare"] * row["pickup_count"]
        if row["pickup_count"] > 0:
            fare_vals.append(row["mean_fare"])

    n = max(len(corridor_cells), 1)
    return {
        "corridor_hist_pickups": total_pu,
        "corridor_hist_dropoffs": total_do,
        "corridor_hist_pickup_density": total_pu / n,
        "corridor_hist_dropoff_density": total_do / n,
        "corridor_hist_mean_fare": float(np.mean(fare_vals)) if fare_vals else 0.0,
        "corridor_hist_fare_density": fare_sum / n,
    }


def main():
    print("=== Augment v1 -> v2 (fast, no match_riders) ===")
    t0 = time.time()

    v1 = pd.read_parquet(V1_PATH)
    print(f"  V1 dataset: {len(v1):,} rows, {v1.driver_id.nunique():,} drivers")

    v1.drop(columns=["matched_rider_count", "feasible_rider_count"], inplace=True, errors="ignore")
    print("  Dropped leaky features: matched_rider_count, feasible_rider_count")

    drivers_all = pd.read_parquet(DRIVERS_PATH)
    train = drivers_all[drivers_all["split"] == "train"].reset_index(drop=True)
    del drivers_all
    train = train.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"  Sampled drivers: {len(train):,}")

    h3_stats = pd.read_parquet(H3_STATS_PATH)
    h3_dict = {r["h3_cell"]: r.to_dict() for _, r in h3_stats.iterrows()}
    print(f"  H3 stats: {len(h3_dict):,} cells")

    router = OSRMRouter(cache_path=CACHE_PATH, cache_only=True)
    print(f"  Route cache: {router.cache_size:,} entries")

    origin_lats = train["origin_lat"].values
    origin_lngs = train["origin_lng"].values
    dest_lats = train["dest_lat"].values
    dest_lngs = train["dest_lng"].values
    pickup_dt = train["pickup_datetime"]
    hours = train["hour_of_day"].values
    day_of_months = pickup_dt.dt.day.values
    minutes_of_day = (pickup_dt.dt.hour * 60 + pickup_dt.dt.minute).values

    per_driver_data = {}
    per_route_corridors = {}

    print("\n  Phase 1: Computing per-driver features + corridor stats...")
    pbar = tqdm(range(len(train)), desc="  Drivers", unit="d", ncols=100)
    skipped_drivers = 0

    for i in pbar:
        olat, olng = float(origin_lats[i]), float(origin_lngs[i])
        dlat, dlng = float(dest_lats[i]), float(dest_lngs[i])
        hour = int(hours[i])
        dom = int(day_of_months[i])
        mod = int(minutes_of_day[i])

        straight_dist = haversine_m((olat, olng), (dlat, dlng))
        brg = bearing_deg(olat, olng, dlat, dlng)
        qh = hour * 4 + (mod % 60) // 15

        o_near_dist, o_dists = compute_landmark_dists(olat, olng)
        d_near_dist, d_dists = compute_landmark_dists(dlat, dlng)

        o_cell = h3.latlng_to_cell(olat, olng, 9)
        d_cell = h3.latlng_to_cell(dlat, dlng, 9)
        o_stats = h3_dict.get(o_cell, {})
        d_stats = h3_dict.get(d_cell, {})

        per_driver_data[i] = {
            "day_of_month": dom,
            "time_bin_15min": qh,
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "bearing_sin": math.sin(math.radians(brg)),
            "bearing_cos": math.cos(math.radians(brg)),
            "straight_line_dist_m": straight_dist,
            "origin_landmark_dist_km": o_near_dist,
            "dest_landmark_dist_km": d_near_dist,
            "origin_jfk_km": o_dists["jfk"],
            "origin_lga_km": o_dists["lga"],
            "origin_penn_km": o_dists["penn"],
            "origin_times_sq_km": o_dists["times_sq"],
            "dest_jfk_km": d_dists["jfk"],
            "dest_lga_km": d_dists["lga"],
            "dest_penn_km": d_dists["penn"],
            "dest_times_sq_km": d_dists["times_sq"],
            "origin_cell_pickups": o_stats.get("pickup_count", 0),
            "origin_cell_mean_fare": o_stats.get("mean_fare", 0.0),
            "dest_cell_dropoffs": d_stats.get("dropoff_count", 0),
        }

        try:
            routes = router.get_alternative_routes((olat, olng), (dlat, dlng), 3)
        except Exception:
            skipped_drivers += 1
            continue
        if not routes:
            skipped_drivers += 1
            continue

        for ri, route in enumerate(routes):
            corridor = build_corridor(route.polyline)
            sinuosity = route.distance_m / max(straight_dist, 1.0)
            avg_speed = route.distance_m / max(route.duration_s, 1.0)
            ch = corridor_hist_stats(corridor.corridor_cells, h3_dict)

            per_route_corridors[(i, ri)] = {
                "route_sinuosity": sinuosity,
                "route_avg_speed_ms": avg_speed,
                **ch,
            }

        if (i + 1) % 2000 == 0:
            pbar.set_postfix({"computed": len(per_route_corridors), "skip": skipped_drivers})

    pbar.close()
    print(f"  Per-driver data: {len(per_driver_data):,}")
    print(f"  Per-route data: {len(per_route_corridors):,}")
    print(f"  Skipped drivers: {skipped_drivers:,}")

    print("\n  Phase 2: Merging new features into v1...")

    new_cols = []
    for idx in tqdm(range(len(v1)), desc="  Merging", unit="row", ncols=100):
        did = int(v1.iloc[idx]["driver_id"])
        rid = int(v1.iloc[idx]["route_idx"])

        drv = per_driver_data.get(did, {})
        rte = per_route_corridors.get((did, rid), {})

        merged = {**drv, **rte}
        new_cols.append(merged)

    new_df = pd.DataFrame(new_cols)
    v2 = pd.concat([v1.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

    missing_mask = v2["route_sinuosity"].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing:,} rows missing corridor features (no cached route)")
        for col in new_df.columns:
            if col in v2.columns and v2[col].dtype in [np.float64, np.float32, np.int64]:
                v2[col] = v2[col].fillna(0)

    v2.to_parquet(OUTPUT_PATH, compression="snappy", index=False)
    elapsed = time.time() - t0

    print(f"\n=== v2 Dataset Complete ===")
    print(f"  Rows: {len(v2):,}")
    print(f"  Columns: {len(v2.columns)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size: {OUTPUT_PATH.stat().st_size / (1024**2):.1f} MB")
    print(f"\n  Feature columns:")
    skip = {"driver_id", "route_idx", "expected_revenue", "driver_cost", "expected_profit"}
    for c in sorted(v2.columns):
        if c not in skip:
            print(f"    {c}: {v2[c].dtype}  min={v2[c].min():.4f}  max={v2[c].max():.4f}")


if __name__ == "__main__":
    main()
