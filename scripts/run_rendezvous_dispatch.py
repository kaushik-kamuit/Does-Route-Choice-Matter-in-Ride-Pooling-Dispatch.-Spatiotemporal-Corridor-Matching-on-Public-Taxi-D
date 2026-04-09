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
from rendezvous.domain_io import load_domain_assets, load_urban_context_index
from rendezvous.reporting import summarize_dispatch, write_result_views
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
    parser = argparse.ArgumentParser(description="Run the rendezvous dispatch validation")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--density", type=int, default=10)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--scenario-name", type=str, default="primary")
    parser.add_argument("--meeting-k-ring", type=int, default=1)
    parser.add_argument("--max-walk-min", type=float, default=6.0)
    parser.add_argument("--occlusion-lambda", type=float, default=0.25)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--disable-urban-context", action="store_true")
    args = parser.parse_args()

    config = RendezvousConfig(
        scenario_name=args.scenario_name,
        domain=args.domain,
        rider_density_pct=args.density,
        meeting_k_ring=args.meeting_k_ring,
        max_walk_min=args.max_walk_min,
        occlusion_lambda=args.occlusion_lambda,
        use_urban_context=not args.disable_urban_context,
    )
    domain_config, drivers_df, riders_df = load_domain_assets(
        args.domain,
        split="test",
        driver_columns=DRIVER_COLUMNS,
        rider_columns=RIDER_COLUMNS,
    )
    if args.sample < len(drivers_df):
        drivers_df = drivers_df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    ml_selector = None
    if args.model_path:
        ml_selector = MLMeetingPointSelector.load(Path(args.model_path))
    router = OSRMRouter(cache_path=domain_config.route_cache_path, cache_only=not args.fetch)
    urban_context = load_urban_context_index(domain_config, config)
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
                [{**row.to_dict(), "domain": args.domain, "scenario_name": config.scenario_name, "rider_density_pct": config.rider_density_pct, "occlusion_lambda": config.occlusion_lambda, "meeting_k_ring": config.meeting_k_ring} for row in outcomes]
            )
            summary_rows.append(
                {
                    **summary.to_dict(),
                    "domain": args.domain,
                    "scenario_name": config.scenario_name,
                    "rider_density_pct": config.rider_density_pct,
                    "occlusion_lambda": config.occlusion_lambda,
                    "meeting_k_ring": config.meeting_k_ring,
                }
            )

    results_dir = ROOT / "results"
    suffix = f"_{args.tag}" if args.tag else ""
    outcomes_df = pd.DataFrame(outcome_rows)
    summaries_df = pd.DataFrame(summary_rows)
    outcomes_df.to_csv(results_dir / f"rendezvous_dispatch_outcomes{suffix}.csv", index=False)
    summaries_df.to_csv(results_dir / f"rendezvous_dispatch_summary{suffix}.csv", index=False)
    (results_dir / f"rendezvous_dispatch_config{suffix}.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )

    dispatch_summary = summarize_dispatch(summaries_df)
    write_result_views(results_dir, pd.DataFrame(), dispatch_summary)
    print(f"Wrote {len(outcomes_df):,} dispatch rows to {results_dir}")


if __name__ == "__main__":
    main()
