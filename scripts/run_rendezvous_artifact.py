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
from rendezvous.domain_io import apply_area_slice, build_driver_trips, load_domain_assets, load_urban_context_index
from rendezvous.reporting import summarize_driver_outcomes
from rendezvous.run_registry import create_run_artifact_dir, write_run_manifest
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
    "origin_h3",
    "dest_h3",
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
    parser.add_argument("--time-slice", type=str, default="all_day")
    parser.add_argument("--area-slice", type=str, default="all", choices=["all", "dense_core", "open_grid"])
    parser.add_argument("--hour-start", type=int, default=None)
    parser.add_argument("--hour-end", type=int, default=None)
    parser.add_argument("--density", type=int, default=100)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--scenario-name", type=str, default="primary")
    parser.add_argument("--meeting-k-ring", type=int, default=1)
    parser.add_argument("--max-walk-min", type=float, default=6.0)
    parser.add_argument("--occlusion-lambda", type=float, default=0.25)
    parser.add_argument(
        "--observability-ablation",
        type=str,
        default="full",
        choices=["full", "no_straightness", "no_turn", "no_ambiguity", "no_clutter"],
    )
    parser.add_argument(
        "--observability-profile",
        type=str,
        default="equal",
        choices=["equal", "calibrated"],
    )
    parser.add_argument("--walk-penalty-per-min", type=float, default=0.5)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--max-riders", type=int, default=None)
    parser.add_argument("--disable-urban-context", action="store_true")
    args = parser.parse_args()

    config = RendezvousConfig(
        scenario_name=args.scenario_name,
        domain=args.domain,
        time_slice=args.time_slice,
        area_slice=args.area_slice,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        rider_density_pct=args.density,
        meeting_k_ring=args.meeting_k_ring,
        max_walk_min=args.max_walk_min,
        occlusion_lambda=args.occlusion_lambda,
        observability_profile=args.observability_profile,
        observability_ablation=args.observability_ablation,
        walk_penalty_per_min=args.walk_penalty_per_min,
        use_urban_context=not args.disable_urban_context,
    )
    domain_config, drivers_df, riders_df = load_domain_assets(
        args.domain,
        split="test",
        driver_columns=DRIVER_COLUMNS,
        rider_columns=RIDER_COLUMNS,
    )
    if args.hour_start is not None and args.hour_end is not None:
        drivers_df = _filter_by_hour_range(drivers_df, args.hour_start, args.hour_end)
        riders_df = _filter_by_hour_range(riders_df, args.hour_start, args.hour_end)
    if args.sample < len(drivers_df):
        drivers_df = drivers_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    if args.max_riders is not None and args.max_riders < len(riders_df):
        riders_df = riders_df.sample(n=args.max_riders, random_state=42).reset_index(drop=True)
    if config.rider_density_pct < 100:
        riders_df = riders_df.sample(frac=config.rider_density_pct / 100.0, random_state=42).reset_index(drop=True)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = create_run_artifact_dir(
        results_dir,
        run_kind="driver",
        domain=args.domain,
        scenario_name=config.scenario_name,
        tag=args.tag or config.time_slice,
    )

    urban_context = load_urban_context_index(domain_config, config)
    drivers_df, riders_df = apply_area_slice(drivers_df, riders_df, urban_context, area_slice=config.area_slice)
    rider_index = RiderIndex(riders_df.reset_index(drop=True), index_bin_minutes=config.index_bin_minutes)
    router = OSRMRouter(cache_path=domain_config.route_cache_path, cache_only=not args.fetch)
    driver_trips = build_driver_trips(drivers_df, config)
    ml_selector = None
    if args.model_path:
        ml_selector = MLMeetingPointSelector.load(Path(args.model_path))

    outcome_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    opportunity_rows: list[dict[str, object]] = []
    seeds = list(range(42, 42 + args.seeds))
    skipped_no_route = 0

    for trip in driver_trips:
        routes = router.get_alternative_routes(trip.origin, trip.destination, max_alternatives=config.route_alternatives)
        if not routes:
            skipped_no_route += 1
            continue
        for seed in seeds:
            evaluation = evaluate_driver_policies(
                trip,
                rider_index,
                config,
                routes=routes,
                ml_selector=ml_selector,
                urban_context=urban_context,
                seed=seed,
            )
            for policy, plan in evaluation.plans.items():
                outcome_rows.append(
                    {
                        "driver_id": trip.driver_id,
                        "seed": seed,
                        "domain": args.domain,
                        "scenario_name": config.scenario_name,
                        "time_slice": config.time_slice,
                        "area_slice": config.area_slice,
                        "hour_start": config.hour_start,
                        "hour_end": config.hour_end,
                        "rider_density_pct": config.rider_density_pct,
                        "occlusion_lambda": config.occlusion_lambda,
                        "meeting_k_ring": config.meeting_k_ring,
                        "observability_profile": config.observability_profile,
                        "observability_ablation": config.observability_ablation,
                        "use_urban_context": config.use_urban_context,
                        "walk_penalty_per_min": config.walk_penalty_per_min,
                        "run_id": run_id,
                        **plan.to_dict(),
                    }
                )
            if seed == seeds[0]:
                for route_eval in evaluation.route_evaluations:
                    mean_route_walk = (
                        float(pd.Series([item.walk_min for item in route_eval.opportunities], dtype=float).mean())
                        if route_eval.opportunities
                        else 0.0
                    )
                    mean_route_observability = (
                        float(pd.Series([item.observability_score for item in route_eval.opportunities], dtype=float).mean())
                        if route_eval.opportunities
                        else 0.0
                    )
                    route_rows.append(
                        {
                            "driver_id": trip.driver_id,
                            "domain": args.domain,
                            "scenario_name": config.scenario_name,
                            "time_slice": config.time_slice,
                            "area_slice": config.area_slice,
                            "hour_start": config.hour_start,
                            "hour_end": config.hour_end,
                            "rider_density_pct": config.rider_density_pct,
                            "occlusion_lambda": config.occlusion_lambda,
                            "meeting_k_ring": config.meeting_k_ring,
                            "observability_profile": config.observability_profile,
                            "observability_ablation": config.observability_ablation,
                            "use_urban_context": config.use_urban_context,
                            "walk_penalty_per_min": config.walk_penalty_per_min,
                            "run_id": run_id,
                            "route_idx": route_eval.route_idx,
                            "candidate_count": route_eval.candidate_count,
                            "time_eligible_candidate_count": route_eval.time_eligible_candidate_count,
                            "feasible_opportunity_count": route_eval.feasible_opportunity_count,
                            "observable_opportunity_count": route_eval.observable_opportunity_count,
                            "nominal_route_value": route_eval.nominal_route_value,
                            "observable_route_value": route_eval.observable_route_value,
                            "walk_route_value": route_eval.walk_route_value,
                            "route_cost": route_eval.route_cost,
                            "route_distance_miles": route_eval.route.distance_m / 1609.34,
                            "route_duration_min": route_eval.route.duration_s / 60.0,
                            "mean_route_walk_min": mean_route_walk,
                            "mean_route_observability": mean_route_observability,
                            "polyline_json": json.dumps(route_eval.route.to_dict()["polyline"]),
                            "route_cells": ";".join(route_eval.route_cells),
                            "corridor_cells": ";".join(sorted(route_eval.corridor.corridor_cells)),
                        }
                    )
                    for opportunity in route_eval.opportunities:
                        opportunity_rows.append(
                            {
                                "driver_id": trip.driver_id,
                                "domain": args.domain,
                                "scenario_name": config.scenario_name,
                                "time_slice": config.time_slice,
                                "area_slice": config.area_slice,
                                "rider_density_pct": config.rider_density_pct,
                                "occlusion_lambda": config.occlusion_lambda,
                                "meeting_k_ring": config.meeting_k_ring,
                                "observability_profile": config.observability_profile,
                                "observability_ablation": config.observability_ablation,
                                "use_urban_context": config.use_urban_context,
                                "walk_penalty_per_min": config.walk_penalty_per_min,
                                "run_id": run_id,
                                "route_idx": route_eval.route_idx,
                                "route_cost": route_eval.route_cost,
                                "route_distance_miles": route_eval.route.distance_m / 1609.34,
                                "route_duration_min": route_eval.route.duration_s / 60.0,
                                "rider_id": opportunity.rider_id,
                                "anchor_cell": opportunity.anchor_cell,
                                "anchor_idx": opportunity.anchor_idx,
                                "pickup_h3": opportunity.pickup_h3,
                                "dropoff_h3": opportunity.dropoff_h3,
                                "fare_share": opportunity.fare_share,
                                "passenger_count": opportunity.passenger_count,
                                "walk_m": opportunity.walk_m,
                                "walk_min": opportunity.walk_min,
                                "anchor_progress": opportunity.anchor_progress,
                                "travel_fraction": opportunity.travel_fraction,
                                "ambiguity_count": opportunity.ambiguity_count,
                                "local_straightness": opportunity.local_straightness,
                                "turn_severity": opportunity.turn_severity,
                                "anchor_clutter": opportunity.anchor_clutter,
                                "urban_clutter_index": opportunity.urban_clutter_index,
                                "sidewalk_access_score": opportunity.sidewalk_access_score,
                                "building_height_proxy": opportunity.building_height_proxy,
                                "context_is_imputed": opportunity.context_is_imputed,
                                "observability_score": opportunity.observability_score,
                                "success_probability": opportunity.success_probability,
                            }
                        )

    outcomes_df = pd.DataFrame(outcome_rows)
    routes_df = pd.DataFrame(route_rows)
    opportunity_df = pd.DataFrame(opportunity_rows)
    outcomes_path = run_dir / "rendezvous_driver_outcomes.csv"
    routes_path = run_dir / "rendezvous_route_evaluations.csv"
    opportunities_path = run_dir / "rendezvous_route_opportunities.csv"
    config_path = run_dir / "rendezvous_config.json"
    outcomes_df.to_csv(outcomes_path, index=False)
    routes_df.to_csv(routes_path, index=False)
    opportunity_df.to_csv(opportunities_path, index=False)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    router.flush_cache()

    summary = summarize_driver_outcomes(outcomes_df)
    summary_path = run_dir / "rendezvous_driver_summary.csv"
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    run_stats = {
        "domain": args.domain,
        "scenario_name": config.scenario_name,
        "requested_drivers": len(driver_trips),
        "evaluated_drivers": len(driver_trips) - skipped_no_route,
        "skipped_no_route": skipped_no_route,
        "route_coverage_rate": (len(driver_trips) - skipped_no_route) / max(len(driver_trips), 1),
        "area_slice": config.area_slice,
        "seeds": len(seeds),
        "cache_only": not args.fetch,
        "model_path": args.model_path,
        "run_id": run_id,
    }
    run_stats_path = run_dir / "rendezvous_driver_run_stats.json"
    run_stats_path.write_text(json.dumps(run_stats, indent=2), encoding="utf-8")
    write_run_manifest(
        results_dir=results_dir,
        run_dir=run_dir,
        run_id=run_id,
        run_kind="driver",
        domain=args.domain,
        scenario_name=config.scenario_name,
        tag=args.tag or "",
        config=config.to_dict(),
        cli_args=vars(args),
        raw_outputs={
            "driver_outcomes": outcomes_path,
            "route_evaluations": routes_path,
            "route_opportunities": opportunities_path,
        },
        derived_outputs={
            "config": config_path,
            "driver_run_stats": run_stats_path,
            **({"driver_summary": summary_path} if summary_path.exists() else {}),
        },
        metadata=run_stats,
    )
    print(f"Wrote {len(outcomes_df):,} policy rows to {run_dir} ({run_id})")


def _filter_by_hour_range(df: pd.DataFrame, hour_start: int, hour_end: int) -> pd.DataFrame:
    hours = pd.to_datetime(df["pickup_datetime"]).dt.hour
    if hour_start <= hour_end:
        mask = (hours >= hour_start) & (hours < hour_end)
    else:
        mask = (hours >= hour_start) | (hours < hour_end)
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
