from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from matching.rider_index import RiderIndex
from rendezvous import MLMeetingPointSelector, RendezvousConfig, evaluate_driver_policies
from rendezvous.domain_io import build_driver_trips, load_domain_assets
from rendezvous.reporting import summarize_driver_outcomes, write_result_views
from spatial.router import OSRMRouter

DRIVER_COLUMNS = [
    "split",
    "pickup_datetime",
    "origin_lat",
    "origin_lng",
    "dest_lat",
    "dest_lng",
    "hour_of_day",
    "trip_distance_miles",
]

RIDER_COLUMNS = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled rendezvous route-choice study")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--density", type=int, default=100)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--scenario-name", type=str, default="primary")
    parser.add_argument("--meeting-k-ring", type=int, default=1)
    parser.add_argument("--max-walk-min", type=float, default=6.0)
    parser.add_argument("--occlusion-lambda", type=float, default=0.25)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--max-riders", type=int, default=None)
    args = parser.parse_args()

    config = RendezvousConfig(
        scenario_name=args.scenario_name,
        domain=args.domain,
        rider_density_pct=args.density,
        meeting_k_ring=args.meeting_k_ring,
        max_walk_min=args.max_walk_min,
        occlusion_lambda=args.occlusion_lambda,
    )
    domain_config, drivers_df, riders_df = load_domain_assets(
        args.domain,
        split="test",
        driver_columns=DRIVER_COLUMNS,
        rider_columns=RIDER_COLUMNS,
    )
    if args.sample < len(drivers_df):
        drivers_df = drivers_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    if args.max_riders is not None and args.max_riders < len(riders_df):
        riders_df = riders_df.sample(n=args.max_riders, random_state=42).reset_index(drop=True)
    if config.rider_density_pct < 100:
        riders_df = riders_df.sample(frac=config.rider_density_pct / 100.0, random_state=42).reset_index(drop=True)

    rider_index = RiderIndex(riders_df.reset_index(drop=True), index_bin_minutes=config.index_bin_minutes)
    router = OSRMRouter(cache_path=domain_config.route_cache_path, cache_only=not args.fetch)
    driver_trips = build_driver_trips(drivers_df, config)
    ml_selector = None
    if args.model_path:
        ml_selector = MLMeetingPointSelector.load(Path(args.model_path))

    outcome_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    seeds = list(range(42, 42 + args.seeds))

    for trip in driver_trips:
        routes = router.get_alternative_routes(trip.origin, trip.destination, max_alternatives=config.route_alternatives)
        if not routes:
            continue
        for seed in seeds:
            evaluation = evaluate_driver_policies(
                trip,
                rider_index,
                config,
                routes=routes,
                ml_selector=ml_selector,
                seed=seed,
            )
            for policy, plan in evaluation.plans.items():
                outcome_rows.append(
                    {
                        "driver_id": trip.driver_id,
                        "seed": seed,
                        "domain": args.domain,
                        "scenario_name": config.scenario_name,
                        "rider_density_pct": config.rider_density_pct,
                        "occlusion_lambda": config.occlusion_lambda,
                        "meeting_k_ring": config.meeting_k_ring,
                        **plan.to_dict(),
                    }
                )
            for route_eval in evaluation.route_evaluations:
                route_rows.append(
                    {
                        "driver_id": trip.driver_id,
                        "seed": seed,
                        "route_idx": route_eval.route_idx,
                        "candidate_count": route_eval.candidate_count,
                        "exact_time_candidate_count": route_eval.exact_time_candidate_count,
                        "feasible_opportunity_count": route_eval.feasible_opportunity_count,
                        "observable_opportunity_count": route_eval.observable_opportunity_count,
                        "nominal_route_value": route_eval.nominal_route_value,
                        "observable_route_value": route_eval.observable_route_value,
                        "route_distance_miles": route_eval.route.distance_m / 1609.34,
                    }
                )

    suffix = f"_{args.tag}" if args.tag else ""
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outcomes_df = pd.DataFrame(outcome_rows)
    routes_df = pd.DataFrame(route_rows)
    outcomes_df.to_csv(results_dir / f"rendezvous_driver_outcomes{suffix}.csv", index=False)
    routes_df.to_csv(results_dir / f"rendezvous_route_evaluations{suffix}.csv", index=False)
    (results_dir / f"rendezvous_config{suffix}.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    summary = summarize_driver_outcomes(outcomes_df)
    write_result_views(results_dir, summary)
    print(f"Wrote {len(outcomes_df):,} policy rows to {results_dir}")


if __name__ == "__main__":
    main()
