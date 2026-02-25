"""
Build the profit-labeled training dataset for the LightGBM route-profit predictor.

For each sampled train-split driver:
  1. Fetch up to 3 alternative routes from the SQLite route cache.
  2. Build an H3 corridor along each route.
  3. Run match_riders to get actual post-matching profit (not corridor-sum proxy).
  4. Compute features and the matched-profit label.

Output: data/ml/training_dataset.parquet  (one row per driver-route pair)

Usage:
    python src/models/build_dataset.py                  # default 100K sample, cache-only
    python src/models/build_dataset.py --sample 50000   # custom sample
    python src/models/build_dataset.py --fetch           # allow OSRM API for uncached routes
    python src/models/build_dataset.py --all             # every train driver
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from spatial.router import OSRMRouter
from spatial.corridor import build_corridor
from matching.rider_index import RiderIndex
from matching.matcher import match_riders, PLATFORM_SHARE, COST_PER_MILE, METERS_PER_MILE

DRIVERS_PATH = ROOT / "data" / "processed" / "drivers.parquet"
RIDERS_PATH = ROOT / "data" / "processed" / "riders.parquet"
CACHE_PATH = ROOT / "data" / "route_cache.db"
OUTPUT_PATH = ROOT / "data" / "ml" / "training_dataset.parquet"

DEFAULT_SAMPLE = 100_000
MAX_RIDERS = 1_000_000
SAMPLE_SEED = 42
MAX_ALTERNATIVES = 3
LOG_EVERY = 2_000
CHECKPOINT_EVERY = 5_000


def _extract_row(
    driver_id: int,
    route_idx: int,
    route,
    corridor,
    candidates: pd.DataFrame,
    rider_index: RiderIndex,
    minute_of_day: int,
    hour: int,
    day_of_week: int,
    is_weekend: int,
) -> dict:
    n_riders = len(candidates)
    corridor_cells = corridor.n_corridor_cells
    fares = candidates["fare_amount"].values if n_riders > 0 else np.array([])
    fare_sum = float(fares.sum()) if n_riders > 0 else 0.0

    matched, feasible = match_riders(
        corridor, route.polyline, rider_index, minute_of_day,
        seats=3, max_detour_min=4.0, seed=SAMPLE_SEED,
        candidates=candidates,
    )

    total_revenue = sum(m["fare_share"] for m in matched)
    driver_cost = route.distance_m / METERS_PER_MILE * COST_PER_MILE
    expected_profit = total_revenue - driver_cost

    return {
        "driver_id": driver_id,
        "route_idx": route_idx,
        "route_distance_m": route.distance_m,
        "route_duration_s": route.duration_s,
        "corridor_cell_count": corridor_cells,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "corridor_rider_count": n_riders,
        "corridor_demand_density": n_riders / max(corridor_cells, 1),
        "mean_rider_fare": float(fares.mean()) if n_riders > 0 else 0.0,
        "corridor_fare_density": fare_sum / max(corridor_cells, 1),
        "feasible_rider_count": len(feasible),
        "matched_rider_count": len(matched),
        "expected_revenue": total_revenue,
        "driver_cost": driver_cost,
        "expected_profit": expected_profit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Number of train drivers to sample (default {DEFAULT_SAMPLE:,})")
    parser.add_argument("--all", action="store_true",
                        help="Use ALL train drivers (overrides --sample)")
    parser.add_argument("--cache-only", action="store_true", default=True,
                        help="Only use cached routes, skip OSRM API calls (default)")
    parser.add_argument("--fetch", action="store_true",
                        help="Allow OSRM API calls for uncached routes")
    args = parser.parse_args()

    use_cache_only = args.cache_only and not args.fetch

    print("=== Build Training Dataset ===")

    drivers = pd.read_parquet(DRIVERS_PATH)
    train = drivers.loc[drivers["split"] == "train"].reset_index(drop=True)
    del drivers
    print(f"  Train drivers loaded: {len(train):,}")

    if not args.all and args.sample < len(train):
        train = train.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Sampled to {len(train):,} (seed={SAMPLE_SEED})")

    riders = pd.read_parquet(RIDERS_PATH)
    train_riders = riders.loc[riders["split"] == "train"].reset_index(drop=True)
    del riders
    if len(train_riders) > MAX_RIDERS:
        train_riders = train_riders.sample(n=MAX_RIDERS, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Train riders subsampled: {len(train_riders):,} / 8M (for build speed)")
    else:
        print(f"  Train riders loaded: {len(train_riders):,}")

    rider_index = RiderIndex(train_riders)

    router = OSRMRouter(cache_path=CACHE_PATH, cache_only=use_cache_only)
    print(f"  Route cache size: {router.cache_size:,}")
    if use_cache_only:
        print(f"  Mode: cache-only (no OSRM API calls, use --fetch to enable)")
    print()

    total = len(train)
    rows: list[dict] = []
    skipped = 0
    t_start = time.time()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_PATH.with_suffix(".partial.parquet")

    origin_lats = train["origin_lat"].values
    origin_lngs = train["origin_lng"].values
    dest_lats = train["dest_lat"].values
    dest_lngs = train["dest_lng"].values
    hours = train["hour_of_day"].values
    dows = train["day_of_week"].values
    weekends = train["is_weekend"].values
    pickup_dt = train["pickup_datetime"]
    minutes_of_day = (pickup_dt.dt.hour * 60 + pickup_dt.dt.minute).values

    pbar = tqdm(range(total), desc="  Building", unit="driver", ncols=100)
    for i in pbar:
        origin = (float(origin_lats[i]), float(origin_lngs[i]))
        dest = (float(dest_lats[i]), float(dest_lngs[i]))
        hour = int(hours[i])
        mod = int(minutes_of_day[i])
        dow = int(dows[i])
        wkend = int(weekends[i])

        try:
            routes = router.get_alternative_routes(origin, dest, MAX_ALTERNATIVES)
        except Exception:
            skipped += 1
            continue

        if not routes:
            skipped += 1
            continue

        for ri, route in enumerate(routes):
            corridor = build_corridor(route.polyline)
            candidates = rider_index.find_in_corridor(corridor.corridor_cells, mod)
            row = _extract_row(i, ri, route, corridor, candidates, rider_index,
                               mod, hour, dow, wkend)
            rows.append(row)

        if (i + 1) % 500 == 0:
            pbar.set_postfix({
                'rows': f'{len(rows):,}',
                'skip': skipped,
            })

        if (i + 1) % CHECKPOINT_EVERY == 0 and rows:
            pd.DataFrame(rows).to_parquet(
                checkpoint_path, compression="snappy", index=False
            )
            tqdm.write(f"  ** checkpoint saved: {len(rows):,} rows -> {checkpoint_path.name}")
    pbar.close()

    elapsed = time.time() - t_start

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, compression="snappy", index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print()
    print("=== Dataset Build Complete ===")
    print(f"  Drivers processed: {total - skipped:,} / {total:,}")
    print(f"  Skipped (no routes): {skipped:,}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / (1024**2):.1f} MB")
    print()
    print("  Column stats:")
    for col in ["expected_profit", "matched_rider_count", "feasible_rider_count",
                 "corridor_rider_count", "mean_rider_fare",
                 "route_distance_m", "corridor_demand_density"]:
        if col in df.columns:
            s = df[col]
            print(f"    {col:30s}  min={s.min():.2f}  median={s.median():.2f}  "
                  f"mean={s.mean():.2f}  max={s.max():.2f}")


if __name__ == "__main__":
    main()
