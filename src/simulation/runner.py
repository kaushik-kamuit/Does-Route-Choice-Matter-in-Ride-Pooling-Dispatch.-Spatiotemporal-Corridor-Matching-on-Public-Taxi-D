"""
Experiment runner: paired cold-start vs warm-up simulation.

For each test driver, runs BOTH strategies on the same rider pool and
records the outcomes. Repeats across multiple seeds (shuffled rider
tie-breaking order) for statistical robustness.

Usage:
    python src/simulation/runner.py                    # default 10K drivers, 5 seeds
    python src/simulation/runner.py --sample 5000      # custom sample
    python src/simulation/runner.py --seeds 3           # fewer seeds for speed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from matching.rider_index import RiderIndex
from models.predict import ProfitPredictor
from spatial.corridor import build_corridor
from spatial.router import OSRMRouter
from simulation.coldstart import run_coldstart
from simulation.warmup import run_warmup, _route_features
from simulation.data_types import DriverTrip

DRIVERS_PATH = ROOT / "data" / "processed" / "drivers.parquet"
RIDERS_PATH = ROOT / "data" / "processed" / "riders.parquet"
CACHE_PATH = ROOT / "data" / "route_cache.db"
H3_STATS_PATH = ROOT / "data" / "ml" / "h3_cell_stats.parquet"
RESULTS_DIR = ROOT / "results"

DEFAULT_SAMPLE = 10_000
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SEED = 42


def _build_driver_trips(df: pd.DataFrame) -> list[DriverTrip]:
    trips: list[DriverTrip] = []
    for i in range(len(df)):
        row = df.iloc[i]
        dep = row["pickup_datetime"] if "pickup_datetime" in df.columns else datetime(2015, 4, 1)
        trips.append(DriverTrip(
            driver_id=i,
            origin=(float(row["origin_lat"]), float(row["origin_lng"])),
            destination=(float(row["dest_lat"]), float(row["dest_lng"])),
            departure_time=dep,
            hour=int(row["hour_of_day"]),
            minute_of_day=dep.hour * 60 + dep.minute,
            trip_distance_miles=float(row["trip_distance_miles"]),
        ))
    return trips


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cold-start vs warm-up experiment")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Test drivers to sample (default {DEFAULT_SAMPLE:,})")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                        help=f"Number of seeds (default {len(DEFAULT_SEEDS)})")
    parser.add_argument("--all", action="store_true",
                        help="Use ALL test drivers (overrides --sample)")
    parser.add_argument("--cache-only", action="store_true", default=True,
                        help="Skip drivers without cached routes (default)")
    parser.add_argument("--fetch", action="store_true",
                        help="Allow OSRM API calls for uncached routes")
    args = parser.parse_args()

    use_cache_only = args.cache_only and not args.fetch
    seeds = DEFAULT_SEEDS[:args.seeds]

    print("=== Experiment Runner: Cold-Start vs Warm-Up ===")

    drivers_df = pd.read_parquet(DRIVERS_PATH)
    test_df = drivers_df.loc[drivers_df["split"] == "test"].reset_index(drop=True)
    del drivers_df
    print(f"  Test drivers available: {len(test_df):,}")

    if not args.all and args.sample < len(test_df):
        test_df = test_df.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Sampled to: {len(test_df):,} (seed={SAMPLE_SEED})")

    driver_trips = _build_driver_trips(test_df)

    RIDER_COLS = ["split", "pickup_h3", "dropoff_h3", "pickup_datetime",
                   "pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng",
                   "passenger_count", "fare_amount"]
    riders_df = pd.read_parquet(RIDERS_PATH, columns=RIDER_COLS)
    test_riders = riders_df.loc[riders_df["split"] == "test"].reset_index(drop=True)
    del riders_df
    MAX_TEST_RIDERS = 1_000_000
    if len(test_riders) > MAX_TEST_RIDERS:
        test_riders = test_riders.sample(n=MAX_TEST_RIDERS, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Test riders subsampled: {len(test_riders):,} (memory limit)")
    else:
        print(f"  Test riders loaded: {len(test_riders):,}")

    rider_index = RiderIndex(test_riders)

    router = OSRMRouter(cache_path=CACHE_PATH, cache_only=use_cache_only)
    print(f"  Route cache size: {router.cache_size:,}")
    if use_cache_only:
        print(f"  Mode: cache-only (use --fetch to enable OSRM API calls)")

    predictor = ProfitPredictor()
    print(f"  Profit model loaded")

    print(f"  Loading H3 cell stats...", end=" ")
    h3_stats = pd.read_parquet(H3_STATS_PATH)
    h3_stats_dict = {r["h3_cell"]: r.to_dict() for _, r in h3_stats.iterrows()}
    print(f"{len(h3_stats_dict):,} cells")

    print(f"  Seeds: {seeds}")

    dows = test_df["day_of_week"].values
    weekends = test_df["is_weekend"].values
    day_of_months = test_df["pickup_datetime"].dt.day.values

    total_drivers = len(driver_trips)
    total_iterations = total_drivers * len(seeds)
    print(f"  Total iterations: {total_iterations:,} "
          f"({total_drivers:,} drivers x {len(seeds)} seeds)")

    cat_counts = {}
    for d in driver_trips:
        cat = d.route_length_category
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"  Route categories: {cat_counts}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cs_path = RESULTS_DIR / "coldstart_outcomes.csv"
    wu_path = RESULTS_DIR / "warmup_outcomes.csv"
    CHECKPOINT_EVERY = 500

    coldstart_rows: list[dict] = []
    warmup_rows: list[dict] = []

    t_start = time.time()
    iteration = 0
    errors = 0
    skipped = 0
    n_seeds = len(seeds)
    processed_drivers = 0

    pbar = tqdm(driver_trips, desc="  Drivers", unit="driver", ncols=100)
    for i, driver in enumerate(pbar):
        try:
            routes = router.get_alternative_routes(
                driver.origin, driver.destination, max_alternatives=3
            )
            if not routes:
                skipped += n_seeds
                continue

            corridors = [build_corridor(r.polyline) for r in routes]

            dow_i = int(dows[i])
            wkend_i = int(weekends[i])
            dom_i = int(day_of_months[i])

            feature_list = [
                _route_features(
                    r, c, rider_index, r.polyline, driver.minute_of_day,
                    driver.hour, dow_i, wkend_i, dom_i,
                    driver.origin, driver.destination, h3_stats_dict,
                    seats=driver.seats, max_detour_min=driver.max_detour_minutes,
                )
                for r, c in zip(routes, corridors)
            ]
            ranking = predictor.rank_routes(feature_list)

            for seed in seeds:
                cs_outcome = run_coldstart(
                    driver, router, rider_index, seed=seed,
                    route=routes[0], corridor=corridors[0],
                )
                wu_outcome = run_warmup(
                    driver, router, rider_index, predictor,
                    day_of_week=dow_i,
                    is_weekend=wkend_i,
                    day_of_month=dom_i,
                    h3_stats_dict=h3_stats_dict,
                    seed=seed,
                    routes=routes, corridors=corridors, ranking=ranking,
                )

                if cs_outcome is None or wu_outcome is None:
                    skipped += 1
                    continue

                coldstart_rows.append(cs_outcome.to_dict())
                warmup_rows.append(wu_outcome.to_dict())
                iteration += 1

            processed_drivers += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                tqdm.write(f"    ERROR driver {i}: {e}")
            pbar.set_postfix({'err': errors})
            continue

        if processed_drivers % CHECKPOINT_EVERY == 0 and coldstart_rows:
            pd.DataFrame(coldstart_rows).to_csv(cs_path, index=False)
            pd.DataFrame(warmup_rows).to_csv(wu_path, index=False)
            tqdm.write(f"    checkpoint: {iteration:,} iterations saved")

        pbar.set_postfix({
            'done': processed_drivers,
            'CS': f'${cs_outcome.profit:.0f}',
            'WU': f'${wu_outcome.profit:.0f}',
            'err': errors
        })
    pbar.close()

    router.flush_cache()
    elapsed = time.time() - t_start

    cs_df = pd.DataFrame(coldstart_rows)
    wu_df = pd.DataFrame(warmup_rows)

    cs_df.to_csv(cs_path, index=False)
    wu_df.to_csv(wu_path, index=False)

    config = {
        "sample_size": len(driver_trips),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "errors": errors,
        "skipped_no_route": skipped,
        "elapsed_s": elapsed,
        "platform_share": 0.50,
        "cost_per_mile": 0.67,
        "max_detour_min": 4.0,
        "seats": 3,
        "route_categories": cat_counts,
    }
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("=== Experiment Complete ===")
    print(f"  Iterations:  {iteration:,} / {total_iterations:,}")
    print(f"  Skipped:     {skipped:,} (no cached routes)")
    print(f"  Errors:      {errors}")
    print(f"  Time:        {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()

    for label, df in [("Cold-Start", cs_df), ("Warm-Up", wu_df)]:
        if df.empty:
            continue
        print(f"  {label}:")
        print(f"    Mean profit:   ${df['profit'].mean():.2f}")
        print(f"    Median profit: ${df['profit'].median():.2f}")
        print(f"    Match rate:    {(df['matched_riders'] > 0).mean():.1%}")
        print(f"    Mean matches:  {df['matched_riders'].mean():.2f}")
        print()

    if not cs_df.empty and not wu_df.empty:
        delta = wu_df["profit"].mean() - cs_df["profit"].mean()
        print(f"  Warm-up advantage: ${delta:.2f} mean profit per driver")

    print(f"\n  Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
