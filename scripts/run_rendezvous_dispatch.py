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

from rendezvous import ALL_POLICIES, MLMeetingPointSelector, RendezvousConfig, RendezvousDispatcher
from rendezvous.domain_io import apply_area_slice, load_domain_assets, load_urban_context_index
from rendezvous.reporting import summarize_dispatch
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
    parser = argparse.ArgumentParser(description="Run the rendezvous dispatch validation")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--time-slice", type=str, default="all_day")
    parser.add_argument("--area-slice", type=str, default="all", choices=["all", "dense_core", "open_grid"])
    parser.add_argument("--hour-start", type=int, default=None)
    parser.add_argument("--hour-end", type=int, default=None)
    parser.add_argument("--density", type=int, default=10)
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
    ml_selector = None
    if args.model_path:
        ml_selector = MLMeetingPointSelector.load(Path(args.model_path))
    router = OSRMRouter(cache_path=domain_config.route_cache_path, cache_only=not args.fetch)
    urban_context = load_urban_context_index(domain_config, config)
    drivers_df, riders_df = apply_area_slice(drivers_df, riders_df, urban_context, area_slice=config.area_slice)
    if args.sample < len(drivers_df):
        drivers_df = drivers_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = create_run_artifact_dir(
        results_dir,
        run_kind="dispatch",
        domain=args.domain,
        scenario_name=config.scenario_name,
        tag=args.tag or config.time_slice,
    )
    dispatcher = RendezvousDispatcher(config, router=router, ml_selector=ml_selector, urban_context=urban_context)
    sampled_riders_df, rider_index, request_states, request_batches = dispatcher.prepare_rider_pool(riders_df)

    outcome_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    policies = [policy for policy in ALL_POLICIES if policy != "ml_meeting_point_comparator" or ml_selector is not None]
    for policy in policies:
        for seed in range(42, 42 + args.seeds):
            outcomes, summary = dispatcher.run_policy(
                policy,
                drivers_df,
                riders_df,
                seed=seed,
                sampled_riders_df=sampled_riders_df,
                rider_index=rider_index,
                request_states=request_states,
                request_batches=request_batches,
            )
            outcome_rows.extend(
                [
                    {
                        **row.to_dict(),
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
                    }
                    for row in outcomes
                ]
            )
            summary_rows.append(
                {
                    **summary.to_dict(),
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
                }
            )

    outcomes_df = pd.DataFrame(outcome_rows)
    summaries_df = pd.DataFrame(summary_rows)
    outcomes_path = run_dir / "rendezvous_dispatch_outcomes.csv"
    summary_path = run_dir / "rendezvous_dispatch_summary.csv"
    config_path = run_dir / "rendezvous_dispatch_config.json"
    outcomes_df.to_csv(outcomes_path, index=False)
    summaries_df.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    router.flush_cache()

    dispatch_summary = summarize_dispatch(summaries_df)
    dispatch_policy_summary_path = run_dir / "rendezvous_dispatch_policy_summary.csv"
    if not dispatch_summary.empty:
        dispatch_summary.to_csv(dispatch_policy_summary_path, index=False)
    run_stats = {
        "domain": args.domain,
        "scenario_name": config.scenario_name,
        "requested_drivers": int(summaries_df["requested_drivers"].max()) if not summaries_df.empty else 0,
        "mean_route_coverage_rate": float(summaries_df["route_coverage_rate"].mean()) if not summaries_df.empty else 0.0,
        "mean_drivers_skipped_no_route": float(summaries_df["drivers_skipped_no_route"].mean()) if not summaries_df.empty else 0.0,
        "area_slice": config.area_slice,
        "seeds": args.seeds,
        "cache_only": not args.fetch,
        "model_path": args.model_path,
        "run_id": run_id,
    }
    run_stats_path = run_dir / "rendezvous_dispatch_run_stats.json"
    run_stats_path.write_text(
        json.dumps(run_stats, indent=2),
        encoding="utf-8",
    )
    write_run_manifest(
        results_dir=results_dir,
        run_dir=run_dir,
        run_id=run_id,
        run_kind="dispatch",
        domain=args.domain,
        scenario_name=config.scenario_name,
        tag=args.tag or "",
        config=config.to_dict(),
        cli_args=vars(args),
        raw_outputs={
            "dispatch_outcomes": outcomes_path,
            "dispatch_summary": summary_path,
        },
        derived_outputs={
            "config": config_path,
            "dispatch_run_stats": run_stats_path,
            **({"dispatch_policy_summary": dispatch_policy_summary_path} if dispatch_policy_summary_path.exists() else {}),
        },
        metadata=run_stats,
    )
    print(f"Wrote {len(outcomes_df):,} dispatch rows to {run_dir} ({run_id})")


def _filter_by_hour_range(df: pd.DataFrame, hour_start: int, hour_end: int) -> pd.DataFrame:
    hours = pd.to_datetime(df["pickup_datetime"]).dt.hour
    if hour_start <= hour_end:
        mask = (hours >= hour_start) & (hours < hour_end)
    else:
        mask = (hours >= hour_start) | (hours < hour_end)
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
