"""
Experiment runner: paired multi-strategy simulation.

For each test driver, runs ALL strategies on the same rider pool and records
the outcomes.  Repeats across multiple seeds for statistical robustness.

Strategies:
  - cold-start:                default route (routes[0]), no ML
  - random:                    uniform random among 3 OSRM alternatives
  - heuristic_count:           highest corridor candidate count
  - heuristic_fare_density:    highest corridor fare density
  - heuristic_feasible_count:  highest feasible-rider count after exact filtering
  - heuristic_profit_proxy:    highest non-ML profit proxy using demand and route cost
  - warmup:                    ML-ranked best route (LightGBM)
  - oracle:                    best actual profit (hindsight, upper bound)

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

from data_prep.domain_config import get_domain_config
from matching.rider_index import RiderIndex
from models.predict import ProfitPredictor
from spatial.corridor import build_corridor
from spatial.router import OSRMRouter
from simulation.coldstart import run_coldstart
from simulation.domain_io import build_driver_trips
from simulation.experiment_config import ExperimentConfig
from simulation.warmup import run_warmup, _route_features
from simulation.baselines import (
    HEURISTIC_STRATEGIES,
    run_heuristic_count,
    run_heuristic_fare_density,
    run_heuristic_feasible_count,
    run_heuristic_profit_proxy,
    run_oracle,
    run_random,
)
DEFAULT_SAMPLE = 10_000
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SEED = 42

HEURISTIC_ALIAS = "heuristic"
STRATEGIES = ["coldstart", "random", *HEURISTIC_STRATEGIES, "warmup", "oracle"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-strategy experiment")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
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
    parser.add_argument("--index-bin-minutes", type=int, default=15,
                        help="Temporal bin size used for RiderIndex lookup")
    parser.add_argument("--candidate-window-bins", type=int, default=1,
                        help="Adjacent RiderIndex bins to search on each side")
    parser.add_argument("--max-request-offset-min", type=int, default=None,
                        help="Exact request-time offset in minutes after retrieval")
    parser.add_argument("--max-detour-min", type=float, default=4.0,
                        help="Maximum pickup/drop-off detour in minutes")
    parser.add_argument("--platform-share", type=float, default=0.50,
                        help="Platform fare share used for scenario profit")
    parser.add_argument("--cost-per-mile", type=float, default=0.67,
                        help="Driving cost per mile used for scenario profit")
    parser.add_argument("--urban-speed-kmh", type=float, default=40.0,
                        help="Urban speed proxy used to convert detour distance to minutes")
    parser.add_argument("--h3-resolution", type=int, default=9,
                        help="H3 resolution used for route-cell encoding")
    parser.add_argument("--corridor-k-ring", type=int, default=1,
                        help="k-ring width used to expand route cells into a corridor")
    parser.add_argument("--corridor-densify-step-m", type=float, default=80.0,
                        help="Polyline densification step in meters before H3 encoding")
    parser.add_argument("--scenario-name", type=str, default="primary",
                        help="Scenario label written to config metadata")
    parser.add_argument("--model-path", type=str, default="",
                        help="Optional path to a trained profit model")
    parser.add_argument("--max-riders", type=int, default=None,
                        help="Optional cap on loaded test riders for memory-constrained runs")
    args = parser.parse_args()

    use_cache_only = args.cache_only and not args.fetch
    seeds = DEFAULT_SEEDS[:args.seeds]
    density = max(0.01, min(1.0, args.density))
    tag = args.tag or (f"d{int(density * 100)}" if density < 1.0 else "")
    suffix = f"_{tag}" if tag else ""
    exp_config = ExperimentConfig(
        scenario_name=args.scenario_name,
        index_bin_minutes=args.index_bin_minutes,
        candidate_window_bins=args.candidate_window_bins,
        max_request_offset_min=args.max_request_offset_min,
        max_detour_min=args.max_detour_min,
        h3_resolution=args.h3_resolution,
        corridor_k_ring=args.corridor_k_ring,
        corridor_densify_step_m=args.corridor_densify_step_m,
        platform_share=args.platform_share,
        cost_per_mile=args.cost_per_mile,
        urban_speed_kmh=args.urban_speed_kmh,
    )
    domain_config = get_domain_config(args.domain)
    drivers_path = domain_config.drivers_path()
    riders_path = domain_config.riders_path()
    cache_path = domain_config.route_cache_path
    h3_stats_path = domain_config.h3_stats_path()
    results_dir = ROOT / "results" if args.domain == "yellow" else domain_config.results_dir

    print("=== Experiment Runner: Multi-Strategy Comparison ===")
    print(f"  Domain: {domain_config.display_name}")
    print(f"  Strategies: {', '.join(STRATEGIES)}")

    # --- Load drivers ---
    drivers_df = pd.read_parquet(drivers_path)
    test_df = drivers_df.loc[drivers_df["split"] == "test"].reset_index(drop=True)
    del drivers_df
    print(f"  Test drivers available: {len(test_df):,}")

    if not args.all and args.sample < len(test_df):
        test_df = test_df.sample(n=args.sample, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Sampled to: {len(test_df):,} (seed={SAMPLE_SEED})")

    driver_trips = build_driver_trips(
        test_df,
        seats=exp_config.seats,
        max_detour_min=exp_config.max_detour_min,
        platform_share=exp_config.platform_share,
        cost_per_mile=exp_config.cost_per_mile,
        urban_speed_kmh=exp_config.urban_speed_kmh,
    )

    # --- Load riders with optional density subsampling ---
    RIDER_COLS = ["split", "pickup_h3", "dropoff_h3", "pickup_datetime",
                   "pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng",
                   "passenger_count", "fare_amount"]
    riders_df = pd.read_parquet(riders_path, columns=RIDER_COLS)
    test_riders = riders_df.loc[riders_df["split"] == "test"].reset_index(drop=True)
    del riders_df

    actual_test_riders = len(test_riders)
    if args.max_riders is not None and len(test_riders) > args.max_riders:
        test_riders = test_riders.sample(n=args.max_riders, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Test riders subsampled: {len(test_riders):,} (user cap)")
    else:
        print(f"  Test riders loaded: {len(test_riders):,}")

    if density < 1.0:
        n_before = len(test_riders)
        test_riders = test_riders.sample(frac=density, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"  Density subsampling: {n_before:,} -> {len(test_riders):,} riders ({density:.0%})")

    rider_index = RiderIndex(
        test_riders,
        index_bin_minutes=exp_config.index_bin_minutes,
    )

    # --- Router + model ---
    router = OSRMRouter(cache_path=cache_path, cache_only=use_cache_only)
    print(f"  Route cache size: {router.cache_size:,}")
    if use_cache_only:
        print(f"  Mode: cache-only (use --fetch to enable OSRM API calls)")

    predictor = ProfitPredictor(args.model_path) if args.model_path else ProfitPredictor()
    print(f"  Profit model loaded")

    print(f"  Loading H3 cell stats...", end=" ")
    h3_stats = pd.read_parquet(h3_stats_path)
    h3_stats_dict = {r["h3_cell"]: r.to_dict() for _, r in h3_stats.iterrows()}
    print(f"{len(h3_stats_dict):,} cells")

    print(f"  Seeds: {seeds}")
    print(f"  Density: {density:.0%}")
    print(f"  Scenario: {exp_config.scenario_name}")
    print(f"  Index bin minutes: {exp_config.index_bin_minutes}")
    print(f"  Candidate window bins: {exp_config.candidate_window_bins}")
    print(f"  Exact request offset (min): {exp_config.max_request_offset_min}")
    print(f"  Max detour minutes: {exp_config.max_detour_min}")
    print(f"  H3 resolution: {exp_config.h3_resolution}")
    print(f"  Corridor k-ring: {exp_config.corridor_k_ring}")
    print(f"  Corridor densify step (m): {exp_config.corridor_densify_step_m}")

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
    results_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {s: results_dir / f"{s}_outcomes{suffix}.csv" for s in STRATEGIES}
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

            corridors = [
                build_corridor(
                    r.polyline,
                    resolution=exp_config.h3_resolution,
                    buffer_rings=exp_config.corridor_k_ring,
                    densify_step_m=exp_config.corridor_densify_step_m,
                )
                for r in routes
            ]

            dow_i = int(dows[i])
            wkend_i = int(weekends[i])
            dom_i = int(day_of_months[i])

            feature_list = [
                _route_features(
                    r, c, rider_index, r.polyline, driver.minute_of_day,
                    driver.hour, dow_i, wkend_i, dom_i,
                    driver.origin, driver.destination, h3_stats_dict,
                    seats=driver.seats, max_detour_min=driver.max_detour_minutes,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    query_datetime=driver.departure_time,
                )
                for r, c in zip(routes, corridors)
            ]
            ranking = predictor.rank_routes(feature_list)

            for seed in seeds:
                cs = run_coldstart(
                    driver, router, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    route=routes[0], corridor=corridors[0],
                )
                wu = run_warmup(
                    driver, router, rider_index, predictor,
                    day_of_week=dow_i, is_weekend=wkend_i,
                    day_of_month=dom_i, h3_stats_dict=h3_stats_dict,
                    seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                    ranking=ranking,
                )
                ora = run_oracle(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                )
                rnd = run_random(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                )
                heu_count = run_heuristic_count(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                )
                heu_fare = run_heuristic_fare_density(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                )
                heu_feasible = run_heuristic_feasible_count(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                )
                heu_proxy = run_heuristic_profit_proxy(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=exp_config.candidate_window_bins,
                    max_request_offset_min=exp_config.max_request_offset_min,
                    routes=routes, corridors=corridors,
                )

                outcomes = {
                    "coldstart": cs,
                    "random": rnd,
                    "heuristic_count": heu_count,
                    "heuristic_fare_density": heu_fare,
                    "heuristic_feasible_count": heu_feasible,
                    "heuristic_profit_proxy": heu_proxy,
                    "warmup": wu,
                    "oracle": ora,
                }

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

    heuristic_means = {
        strategy: dfs[strategy]["profit"].mean()
        for strategy in HEURISTIC_STRATEGIES
        if strategy in dfs and not dfs[strategy].empty
    }
    best_heuristic = max(heuristic_means, key=heuristic_means.get) if heuristic_means else None
    heuristic_alias_path = results_dir / f"{HEURISTIC_ALIAS}_outcomes{suffix}.csv"
    if best_heuristic is not None:
        dfs[best_heuristic].to_csv(heuristic_alias_path, index=False)
        dfs[HEURISTIC_ALIAS] = dfs[best_heuristic]

    config = {
        "scenario_name": exp_config.scenario_name,
        "sample_size": len(driver_trips),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "density": density,
        "tag": tag,
        "errors": errors,
        "skipped_no_route": skipped,
        "elapsed_s": elapsed,
        "index_bin_minutes": exp_config.index_bin_minutes,
        "candidate_window_bins": exp_config.candidate_window_bins,
        "max_request_offset_min": exp_config.max_request_offset_min,
        "max_detour_min": exp_config.max_detour_min,
        "h3_resolution": exp_config.h3_resolution,
        "corridor_k_ring": exp_config.corridor_k_ring,
        "corridor_densify_step_m": exp_config.corridor_densify_step_m,
        "platform_share": exp_config.platform_share,
        "cost_per_mile": exp_config.cost_per_mile,
        "urban_speed_kmh": exp_config.urban_speed_kmh,
        "rider_presample_frac": exp_config.rider_presample_frac,
        "seats": exp_config.seats,
        "actual_test_riders_before_density": actual_test_riders,
        "test_rider_cap": args.max_riders,
        "route_categories": cat_counts,
        "strategies": STRATEGIES,
        "heuristic_variants": HEURISTIC_STRATEGIES,
        "heuristic_selected_strategy": best_heuristic,
    }
    config_name = f"experiment_config{suffix}.json"
    with open(results_dir / config_name, "w") as f:
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

    if best_heuristic is not None:
        print(f"\n  Strongest heuristic alias: {best_heuristic} -> {heuristic_alias_path.name}")

    print(f"\n  Results saved to: {results_dir}")
    for strat in STRATEGIES:
        print(f"    {out_paths[strat].name}")
    if best_heuristic is not None:
        print(f"    {heuristic_alias_path.name}")


if __name__ == "__main__":
    main()
