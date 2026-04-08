from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.domain_config import get_domain_config
from dispatch import DispatchConfig, RollingDispatcher
from matching.rider_index import RiderIndex
from simulation.baselines import (
    run_heuristic_path_buffer,
)
from simulation.coldstart import run_coldstart
from simulation.domain_io import build_driver_trips, load_h3_stats_dict
from spatial.corridor import build_corridor
from spatial.router import COORD_PRECISION, OSRMRouter

RESULTS_DIR = ROOT / "results"
PYTHON = sys.executable
SAMPLE_SEED = 42
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DRIVER_COLS = [
    "split",
    "pickup_datetime",
    "origin_lat",
    "origin_lng",
    "dest_lat",
    "dest_lng",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "trip_distance_miles",
]
RIDER_COLS = [
    "split",
    "pickup_datetime",
    "pickup_h3",
    "dropoff_h3",
    "pickup_lat",
    "pickup_lng",
    "dropoff_lat",
    "dropoff_lng",
    "passenger_count",
    "fare_amount",
]
SINGLE_DRIVER_DENSITIES = [25, 10]
DISPATCH_DENSITIES = [10]
ALT_COUNT = 3


class DummyPredictor:
    """Minimal predictor interface for heuristic-only dispatch reruns."""

    def rank_routes(self, feature_rows: list[dict[str, float]]) -> list[tuple[int, float]]:
        return [(idx, 0.0) for idx in range(len(feature_rows))]


@dataclass(frozen=True)
class ScenarioSummary:
    mode: str
    density_pct: int
    policy: str
    profit_per_driver: float
    matched_riders_per_driver: float
    match_rate: float
    driver_sample_size: int
    n_seeds: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "density_pct": self.density_pct,
            "policy": self.policy,
            "profit_per_driver": self.profit_per_driver,
            "matched_riders_per_driver": self.matched_riders_per_driver,
            "match_rate": self.match_rate,
            "driver_sample_size": self.driver_sample_size,
            "n_seeds": self.n_seeds,
        }


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 72}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_yellow_ready() -> Path:
    config = get_domain_config("yellow")
    raw_months = [1, 2, 3, 4]
    missing = [month for month in raw_months if not config.raw_month_path(month).exists()]
    if missing:
        _run(
            [PYTHON, "src/data_prep/download_2015.py", "--domain", "yellow", "--months", *[str(month) for month in missing]],
            f"Download Yellow raw months {missing}",
        )
    if not config.drivers_path().exists() or not config.riders_path().exists():
        _run([PYTHON, "src/data_prep/preprocess.py", "--domain", "yellow", "--months", "1", "2", "3", "4"], "Preprocess Yellow Jan-Apr")
    if not config.h3_stats_path().exists():
        _run([PYTHON, "scripts/build_h3_stats.py", "--domain", "yellow"], "Build Yellow H3 stats")
    return config.route_cache_path


def _load_cache_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT cache_key FROM routes").fetchall()
    finally:
        con.close()
    return {row[0] for row in rows}


def _format_coord_series(series: pd.Series) -> pd.Series:
    fmt = f"{{:.{COORD_PRECISION}f}}".format
    return series.round(COORD_PRECISION).map(fmt)


def _filter_cached_test_drivers(drivers: pd.DataFrame, cache_keys: set[str]) -> pd.DataFrame:
    if not cache_keys:
        return drivers.iloc[0:0].copy()
    cache_key_series = (
        _format_coord_series(drivers["origin_lat"]) + "," +
        _format_coord_series(drivers["origin_lng"]) + "->" +
        _format_coord_series(drivers["dest_lat"]) + "," +
        _format_coord_series(drivers["dest_lng"]) +
        f"|alt={ALT_COUNT}"
    )
    return drivers.loc[cache_key_series.isin(cache_keys)].reset_index(drop=True)


def _sample_test_frames(
    sample: int,
    *,
    cache_only: bool,
    cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config = get_domain_config("yellow")
    drivers = pd.read_parquet(config.drivers_path(), columns=DRIVER_COLS)
    riders = pd.read_parquet(config.riders_path(), columns=RIDER_COLS)
    test_drivers = drivers.loc[drivers["split"] == "test"].reset_index(drop=True)
    test_riders = riders.loc[riders["split"] == "test"].reset_index(drop=True)
    coverage: dict[str, object] = {
        "cache_only": cache_only,
        "requested_driver_sample": sample,
        "available_test_drivers": int(len(test_drivers)),
        "available_test_riders": int(len(test_riders)),
    }
    if cache_only:
        cache_keys = _load_cache_keys(cache_path)
        cached_test_drivers = _filter_cached_test_drivers(test_drivers, cache_keys)
        coverage["cache_entries"] = int(len(cache_keys))
        coverage["cached_test_drivers"] = int(len(cached_test_drivers))
        if len(cached_test_drivers) < sample:
            raise ValueError(
                f"Requested {sample} cached drivers but only {len(cached_test_drivers)} "
                f"test drivers have cached routes in {cache_path.name}. "
                "Either lower --sample or rerun with --prefetch-missing."
            )
        test_drivers = cached_test_drivers
    if sample < len(test_drivers):
        test_drivers = test_drivers.sample(n=sample, random_state=SAMPLE_SEED).reset_index(drop=True)
    coverage["selected_driver_sample"] = int(len(test_drivers))
    return test_drivers, test_riders, coverage


def _routes_for_driver(router: OSRMRouter, driver) -> tuple[list, list] | None:
    try:
        routes = router.get_alternative_routes(driver.origin, driver.destination, max_alternatives=3)
    except Exception as exc:
        print(f"    Route fetch failed for driver {driver.driver_id}: {exc}")
        return None
    if not routes:
        return None
    corridors = [
        build_corridor(route.polyline, resolution=9, buffer_rings=1, densify_step_m=80.0)
        for route in routes
    ]
    return routes, corridors


def _aggregate_outcomes(df: pd.DataFrame) -> tuple[float, float, float]:
    by_driver = df.groupby("driver_id", as_index=False).agg(
        profit=("profit", "mean"),
        matched_riders=("matched_riders", "mean"),
    )
    match_rate = float((df["matched_riders"] > 0).mean() * 100.0)
    return (
        float(by_driver["profit"].mean()),
        float(by_driver["matched_riders"].mean()),
        match_rate,
    )


def run_single_driver_baseline(
    sample: int,
    seeds: list[int],
    *,
    cache_only: bool,
) -> tuple[list[ScenarioSummary], dict[str, object]]:
    config = get_domain_config("yellow")
    drivers_df, riders_df, coverage = _sample_test_frames(
        sample,
        cache_only=cache_only,
        cache_path=config.route_cache_path,
    )
    driver_trips = build_driver_trips(
        drivers_df,
        seats=3,
        max_detour_min=4.0,
        platform_share=0.50,
        cost_per_mile=0.67,
        urban_speed_kmh=40.0,
    )
    router = OSRMRouter(cache_path=config.route_cache_path, cache_only=cache_only, rate_limit=not cache_only)
    route_cache: dict[int, tuple[list, list]] = {}
    strategy_rows: dict[tuple[int, str], list[dict[str, object]]] = {}

    print("\n=== Single-driver path-buffer baseline rerun ===")
    print(f"  Drivers sampled: {len(driver_trips):,}")
    for idx, driver in enumerate(driver_trips, start=1):
        cached = _routes_for_driver(router, driver)
        if cached is None:
            continue
        route_cache[driver.driver_id] = cached
        if idx % 100 == 0:
            print(f"  Prefetched routes for {idx:,}/{len(driver_trips):,} drivers")
    router.flush_cache()

    routed_driver_count = len(route_cache)
    for density_pct in SINGLE_DRIVER_DENSITIES:
        print(f"\n  Density {density_pct}%")
        density_riders = riders_df.sample(frac=density_pct / 100.0, random_state=SAMPLE_SEED).reset_index(drop=True)
        rider_index = RiderIndex(density_riders, index_bin_minutes=15)
        for driver in driver_trips:
            cached = route_cache.get(driver.driver_id)
            if cached is None:
                continue
            routes, corridors = cached
            for seed in seeds:
                cold = run_coldstart(
                    driver,
                    router,
                    rider_index,
                    seed=seed,
                    candidate_window_bins=1,
                    max_request_offset_min=5,
                    route=routes[0],
                    corridor=corridors[0],
                )
                heu_path = run_heuristic_path_buffer(
                    driver, rider_index, seed=seed,
                    candidate_window_bins=1, max_request_offset_min=5,
                    routes=routes, corridors=corridors,
                )
                outcomes = {
                    "coldstart": cold,
                    "heuristic_path_buffer": heu_path,
                }
                for policy, outcome in outcomes.items():
                    if outcome is not None:
                        strategy_rows.setdefault((density_pct, policy), []).append(outcome.to_dict())

    router.flush_cache()
    summaries: list[ScenarioSummary] = []
    detailed_rows: list[dict[str, object]] = []
    for (density_pct, policy), rows in strategy_rows.items():
        df = pd.DataFrame(rows)
        profit_per_driver, matched_per_driver, match_rate = _aggregate_outcomes(df)
        summaries.append(
            ScenarioSummary(
                mode="single_driver",
                density_pct=density_pct,
                policy=policy,
                profit_per_driver=profit_per_driver,
                matched_riders_per_driver=matched_per_driver,
                match_rate=match_rate,
                driver_sample_size=routed_driver_count,
                n_seeds=len(seeds),
            )
        )
        detailed_rows.append(
            {
                "mode": "single_driver",
                "density_pct": density_pct,
                "policy": policy,
                "profit_per_driver": profit_per_driver,
                "matched_riders_per_driver": matched_per_driver,
                "match_rate": match_rate,
                "driver_sample_size": routed_driver_count,
                "n_seeds": len(seeds),
            }
        )
    pd.DataFrame(detailed_rows).to_csv(RESULTS_DIR / "path_buffer_single_driver_summary.csv", index=False)
    coverage["single_driver_routed_drivers"] = routed_driver_count
    return summaries, coverage


def run_dispatch_baseline(
    sample: int,
    seeds: list[int],
    *,
    cache_only: bool,
) -> tuple[list[ScenarioSummary], dict[str, object]]:
    config = get_domain_config("yellow")
    drivers_df, riders_df, coverage = _sample_test_frames(
        sample,
        cache_only=cache_only,
        cache_path=config.route_cache_path,
    )
    drivers_df = drivers_df.reset_index(drop=True)
    riders_df = riders_df.reset_index(drop=True)
    router = OSRMRouter(cache_path=config.route_cache_path, cache_only=cache_only, rate_limit=not cache_only)
    dispatcher = RollingDispatcher(
        DispatchConfig(
            domain="yellow",
            scenario_name="path_buffer_baseline",
            batch_seconds=60,
            density_pct=10,
            index_bin_minutes=15,
            candidate_window_bins=1,
            max_request_offset_min=5,
            max_detour_min=4.0,
            platform_share=0.50,
            cost_per_mile=0.67,
            urban_speed_kmh=40.0,
            h3_resolution=9,
            corridor_k_ring=1,
            corridor_densify_step_m=80.0,
            dispatch_policy="heuristic_path_buffer",
        ),
        domain_config=config,
        router=router,
        predictor=DummyPredictor(),
        h3_stats_dict=load_h3_stats_dict(config),
    )

    summaries: list[ScenarioSummary] = []
    rows: list[dict[str, object]] = []
    print("\n=== Dispatch path-buffer baseline rerun ===")
    for density_pct in DISPATCH_DENSITIES:
        dispatch_config = DispatchConfig(
            domain="yellow",
            scenario_name=f"path_buffer_dispatch_d{density_pct}",
            batch_seconds=60,
            density_pct=density_pct,
            index_bin_minutes=15,
            candidate_window_bins=1,
            max_request_offset_min=5,
            max_detour_min=4.0,
            platform_share=0.50,
            cost_per_mile=0.67,
            urban_speed_kmh=40.0,
            h3_resolution=9,
            corridor_k_ring=1,
            corridor_densify_step_m=80.0,
            dispatch_policy="heuristic_path_buffer",
        )
        dispatcher = RollingDispatcher(
            dispatch_config,
            domain_config=config,
            router=router,
            predictor=DummyPredictor(),
            h3_stats_dict=load_h3_stats_dict(config),
        )
        sampled_riders_df, rider_index, request_states, request_batches = dispatcher.prepare_rider_pool(riders_df)
        for policy in ["coldstart", "heuristic_path_buffer"]:
            seed_summaries = []
            for seed in seeds:
                _outcomes, _batches, summary = dispatcher.run_policy(
                    policy,
                    drivers_df,
                    riders_df,
                    seed=seed,
                    sampled_riders_df=sampled_riders_df,
                    rider_index=rider_index,
                    request_states=request_states,
                    request_batches=request_batches,
                )
                seed_summaries.append(summary)
            df = pd.DataFrame([summary.to_dict() for summary in seed_summaries])
            profit_per_driver = float(df["profit_per_launched_driver"].mean())
            matched_per_driver = float(df["mean_matched_riders_per_driver"].mean())
            match_rate = float(df["rider_service_rate"].mean() * 100.0)
            launched_mean = int(round(float(df["launched_drivers"].mean()))) if not df.empty else 0
            summaries.append(
                ScenarioSummary(
                    mode="dispatch",
                    density_pct=density_pct,
                    policy=policy,
                    profit_per_driver=profit_per_driver,
                    matched_riders_per_driver=matched_per_driver,
                    match_rate=match_rate,
                    driver_sample_size=launched_mean,
                    n_seeds=len(seeds),
                )
            )
            rows.append(
                {
                    "mode": "dispatch",
                    "density_pct": density_pct,
                    "policy": policy,
                    "profit_per_driver": profit_per_driver,
                    "matched_riders_per_driver": matched_per_driver,
                    "match_rate": match_rate,
                    "driver_sample_size": launched_mean,
                    "n_seeds": len(seeds),
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "path_buffer_dispatch_summary.csv", index=False)
    router.flush_cache()
    coverage["dispatch_driver_sample"] = int(len(drivers_df))
    return summaries, coverage


def write_comparison(
    single_driver_rows: list[ScenarioSummary],
    dispatch_rows: list[ScenarioSummary],
    *,
    sample_scope: str,
) -> None:
    rows = [summary.to_dict() for summary in [*single_driver_rows, *dispatch_rows]]
    df = pd.DataFrame(rows)
    single_driver_reference = pd.read_csv(RESULTS_DIR / "strong_baseline_comparison.csv")
    dispatch_reference = pd.read_csv(RESULTS_DIR / "dispatch_yellow_primary.csv")
    comparison_rows: list[dict[str, object]] = []
    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode]
        for density_pct in sorted(mode_df["density_pct"].unique(), reverse=True):
            subset = mode_df[mode_df["density_pct"] == density_pct].copy()
            if subset.empty:
                continue
            cold = subset.loc[subset["policy"] == "coldstart"]
            if cold.empty:
                continue
            cold_profit = float(cold.iloc[0]["profit_per_driver"])
            path_buffer = subset.loc[subset["policy"] == "heuristic_path_buffer"]
            if mode == "single_driver":
                ref = single_driver_reference.loc[
                    (single_driver_reference["density_pct"] == density_pct)
                    & (single_driver_reference["selected_for_paper"].astype(str) == "True")
                ]
                best_policy = ref.iloc[0]["heuristic_strategy"] if not ref.empty else None
                best_profit = float(ref.iloc[0]["mean_profit"]) if not ref.empty else None
            else:
                ref = dispatch_reference.loc[
                    (dispatch_reference["density_pct"] == density_pct)
                    & (dispatch_reference["selected_for_paper"].astype(str) == "True")
                ]
                best_policy = ref.iloc[0]["policy"] if not ref.empty else None
                best_profit = float(ref.iloc[0]["profit_per_launched_driver_mean"]) if not ref.empty else None
            comparison_rows.append(
                {
                    "mode": mode,
                    "density_pct": int(density_pct),
                    "sample_scope": sample_scope,
                    "coldstart_profit": cold_profit,
                    "path_buffer_profit": float(path_buffer.iloc[0]["profit_per_driver"]) if not path_buffer.empty else None,
                    "path_buffer_delta_vs_coldstart": float(path_buffer.iloc[0]["profit_per_driver"] - cold_profit) if not path_buffer.empty else None,
                    "paper_best_heuristic_policy": best_policy,
                    "paper_best_heuristic_profit": best_profit,
                    "path_buffer_vs_best_heuristic": (
                        float(path_buffer.iloc[0]["profit_per_driver"] - best_profit)
                        if not path_buffer.empty and best_profit is not None
                        else None
                    ),
                    "driver_sample_size": int(path_buffer.iloc[0]["driver_sample_size"]) if not path_buffer.empty else int(cold.iloc[0]["driver_sample_size"]),
                    "n_seeds": int(path_buffer.iloc[0]["n_seeds"]) if not path_buffer.empty else int(cold.iloc[0]["n_seeds"]),
                    "comparison_note": (
                        "Path-buffer rerun uses the same public Yellow assets but a cache-constrained driver subset; "
                        "paper-best heuristic values come from the validated full paper summaries."
                    ),
                }
            )
    out_path = RESULTS_DIR / "path_buffer_baseline_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a simple path-buffer baseline on Yellow public data")
    parser.add_argument("--sample", type=int, default=250, help="Driver sample for the baseline rerun (default 250)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to use (default 5)")
    parser.add_argument(
        "--prefetch-missing",
        action="store_true",
        help="Allow live OSRM fetches for uncached routes. Default is cache-only for reproducible reruns.",
    )
    args = parser.parse_args()

    ensure_yellow_ready()
    seeds = DEFAULT_SEEDS[: args.seeds]
    start = time.time()
    cache_only = not args.prefetch_missing

    single_driver_rows, single_driver_coverage = run_single_driver_baseline(
        args.sample,
        seeds,
        cache_only=cache_only,
    )
    dispatch_rows, dispatch_coverage = run_dispatch_baseline(
        args.sample,
        seeds,
        cache_only=cache_only,
    )
    sample_scope = "cached_route_subset" if cache_only else "live_or_cached_routes"
    write_comparison(single_driver_rows, dispatch_rows, sample_scope=sample_scope)

    metadata = {
        "domain": "yellow",
        "driver_sample_size": args.sample,
        "seeds": seeds,
        "single_driver_densities": SINGLE_DRIVER_DENSITIES,
        "dispatch_densities": DISPATCH_DENSITIES,
        "request_window_min": 5,
        "max_detour_min": 4.0,
        "batch_seconds": 60,
        "sample_scope": sample_scope,
        "single_driver_coverage": single_driver_coverage,
        "dispatch_coverage": dispatch_coverage,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "elapsed_s": time.time() - start,
    }
    (RESULTS_DIR / "path_buffer_baseline_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nPath-buffer baseline rerun complete.")
    print(f"  Single-driver summary: {RESULTS_DIR / 'path_buffer_single_driver_summary.csv'}")
    print(f"  Dispatch summary:      {RESULTS_DIR / 'path_buffer_dispatch_summary.csv'}")
    print(f"  Comparison summary:    {RESULTS_DIR / 'path_buffer_baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
