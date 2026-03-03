"""
Experiment runner: paired multi-strategy simulation.

For each test driver, runs ALL strategies on the same rider pool and records
the outcomes.  Repeats across multiple seeds for statistical robustness.

Strategies:
  - cold-start:  default route (routes[0]), no ML
  - random:      uniform random among 3 OSRM alternatives
  - heuristic:   highest corridor_rider_count (no ML)
  - warmup:      ML-ranked best route (LightGBM)
  - oracle:      best actual profit (hindsight, upper bound)

Usage:
    python src/simulation/runner.py                          # 10K drivers, 5 seeds
    python src/simulation/runner.py --sample 5000            # custom sample
    python src/simulation/runner.py --seeds 3                # fewer seeds
    python src/simulation/runner.py --density 0.25           # 25% rider density
    python src/simulation/runner.py --density 0.10 --tag d10 # custom output tag
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
from simulation.baselines import run_oracle, run_random, run_heuristic
from simulation.data_types import DriverTrip

DRIVERS_PATH = ROOT / "data" / "processed" / "drivers.parquet"
RIDERS_PATH = ROOT / "data" / "processed" / "riders.parquet"
CACHE_PATH = ROOT / "data" / "route_cache.db"
H3_STATS_PATH = ROOT / "data" / "ml" / "h3_cell_stats.parquet"
RESULTS_DIR = ROOT / "results"

DEFAULT_SAMPLE = 10_000
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SEED = 42

STRATEGIES = ["coldstart", "random", "heuristic", "warmup", "oracle"]


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
    parser = argparse.ArgumentParser(description="Run multi-strategy experiment")
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
    parser.add_argument("--density", type=float, default=1.0,
                        help="Rider density fraction 0.0-1.0 (default 1.0)")
    parser.add_argument("--tag", type=str, default="",
                        help="Output file tag (e.g. 'd25' -> coldstart_d25.csv)")
    args = parser.parse_args()

    use_cache_only = args.cache_only and not args.fetch
    seeds = DEFAULT_SEEDS[:args.seeds]
    density = max(0.01, min(1.0, args.density))
    tag = args.tag or (f"d{int(density * 100)}" if density < 1.0 else "")
    suffix = f"_{tag}" if tag else ""

    print("=== Experiment Runner: Multi-Strategy Comparison ===")
    print(f"  Strategies: {', '.join(STRATEGIES)}")

    # --- Load drivers ---
    drivers_df = pd.read_parquet(DRIVERS_PATH)
    test_df = drivers_df.loc[drivers_df["split"] == "test"].reset_index(drop=True)
    del drivers_df
    print(f"  Test drivers available: {len(test_df):,}")

    if not args.all and args.sample < len(test_df):
        test_df = test_df.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Sampled to: {len(test_df):,} (seed={SAMPLE_SEED})")

    driver_trips = _build_driver_trips(test_df)

    # --- Load riders with optional density subsampling ---
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

    if density < 1.0:
        n_before = len(test_riders)
        test_riders = test_riders.sample(frac=density, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Density subsampling: {n_before:,} -> {len(test_riders):,} riders ({density:.0%})")

    rider_index = RiderIndex(test_riders)

    # --- Router + model ---
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
    print(f"  Density: {density:.0%}")

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

    # --- Output paths ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {s: RESULTS_DIR / f"{s}_outcomes{suffix}.csv" for s in STRATEGIES}
    CHECKPOINT_EVERY = 500

    rows: dict[str, list[dict]] = {s: [] for s in STRATEGIES}

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
                cs = run_coldstart(
                    driver, router, rider_index, seed=seed,
                    route=routes[0], corridor=corridors[0],
                )
                wu = run_warmup(
                    driver, router, rider_index, predictor,
                    day_of_week=dow_i, is_weekend=wkend_i,
                    day_of_month=dom_i, h3_stats_dict=h3_stats_dict,
                    seed=seed, routes=routes, corridors=corridors,
                    ranking=ranking,
                )
                ora = run_oracle(
                    driver, rider_index, seed=seed,
                    routes=routes, corridors=corridors,
                )
                rnd = run_random(
                    driver, rider_index, seed=seed,
                    routes=routes, corridors=corridors,
                )
                heu = run_heuristic(
                    driver, rider_index, seed=seed,
                    routes=routes, corridors=corridors,
                )

                outcomes = {"coldstart": cs, "warmup": wu, "oracle": ora,
                            "random": rnd, "heuristic": heu}

                any_none = any(v is None for v in outcomes.values())
                if any_none:
                    skipped += 1
                    continue

                for strat, outcome in outcomes.items():
                    rows[strat].append(outcome.to_dict())
                iteration += 1

            processed_drivers += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                tqdm.write(f"    ERROR driver {i}: {e}")
            continue

        if processed_drivers % CHECKPOINT_EVERY == 0 and rows["coldstart"]:
            for strat in STRATEGIES:
                pd.DataFrame(rows[strat]).to_csv(out_paths[strat], index=False)
            tqdm.write(f"    checkpoint: {iteration:,} iterations saved")

        pbar.set_postfix({
            'done': processed_drivers,
            'err': errors,
        })
    pbar.close()

    router.flush_cache()
    elapsed = time.time() - t_start

    # --- Save final results ---
    dfs: dict[str, pd.DataFrame] = {}
    for strat in STRATEGIES:
        df = pd.DataFrame(rows[strat])
        df.to_csv(out_paths[strat], index=False)
        dfs[strat] = df

    config = {
        "sample_size": len(driver_trips),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "density": density,
        "tag": tag,
        "errors": errors,
        "skipped_no_route": skipped,
        "elapsed_s": elapsed,
        "platform_share": 0.50,
        "cost_per_mile": 0.67,
        "max_detour_min": 4.0,
        "seats": 3,
        "route_categories": cat_counts,
        "strategies": STRATEGIES,
    }
    config_name = f"experiment_config{suffix}.json"
    with open(RESULTS_DIR / config_name, "w") as f:
        json.dump(config, f, indent=2)

    # --- Print summary ---
    print()
    print("=== Experiment Complete ===")
    print(f"  Iterations:  {iteration:,} / {total_iterations:,}")
    print(f"  Skipped:     {skipped:,} (no cached routes)")
    print(f"  Errors:      {errors}")
    print(f"  Time:        {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()

    for strat in STRATEGIES:
        df = dfs[strat]
        if df.empty:
            continue
        print(f"  {strat:12s}:  mean=${df['profit'].mean():.2f}  "
              f"med=${df['profit'].median():.2f}  "
              f"match_rate={( df['matched_riders'] > 0).mean():.1%}  "
              f"matches={df['matched_riders'].mean():.2f}")

    print()
    cs_mean = dfs["coldstart"]["profit"].mean() if not dfs["coldstart"].empty else 0
    for strat in STRATEGIES:
        if strat == "coldstart" or dfs[strat].empty:
            continue
        delta = dfs[strat]["profit"].mean() - cs_mean
        print(f"  {strat:12s} vs cold-start: {'+' if delta >= 0 else ''}{delta:.2f}")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    for strat in STRATEGIES:
        print(f"    {out_paths[strat].name}")


if __name__ == "__main__":
    main()
