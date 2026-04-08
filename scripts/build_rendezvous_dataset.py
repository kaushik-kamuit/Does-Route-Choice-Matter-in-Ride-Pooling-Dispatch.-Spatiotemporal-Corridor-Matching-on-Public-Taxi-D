from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from matching.rider_index import RiderIndex
from rendezvous import RendezvousConfig, evaluate_driver_policies
from rendezvous.domain_io import build_driver_trips, load_domain_assets, load_urban_context_index
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
    parser = argparse.ArgumentParser(description="Build a meeting-point opportunity dataset")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--max-riders", type=int, default=None)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--disable-urban-context", action="store_true")
    args = parser.parse_args()

    config = RendezvousConfig(domain=args.domain, use_urban_context=not args.disable_urban_context)
    domain_config, drivers_df, riders_df = load_domain_assets(
        args.domain,
        split="train",
        driver_columns=DRIVER_COLUMNS,
        rider_columns=RIDER_COLUMNS,
    )
    if args.sample < len(drivers_df):
        drivers_df = drivers_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    if args.max_riders is not None and args.max_riders < len(riders_df):
        riders_df = riders_df.sample(n=args.max_riders, random_state=42).reset_index(drop=True)

    rider_index = RiderIndex(riders_df.reset_index(drop=True), index_bin_minutes=config.index_bin_minutes)
    router = OSRMRouter(cache_path=domain_config.route_cache_path, cache_only=not args.fetch)
    driver_trips = build_driver_trips(drivers_df, config)
    urban_context = load_urban_context_index(domain_config, config)

    rows: list[dict[str, object]] = []
    for trip in driver_trips:
        routes = router.get_alternative_routes(trip.origin, trip.destination, max_alternatives=config.route_alternatives)
        if not routes:
            continue
        evaluation = evaluate_driver_policies(
            trip,
            rider_index,
            config,
            routes=routes,
            urban_context=urban_context,
            seed=42,
        )
        for route_eval in evaluation.route_evaluations:
            for opportunity in route_eval.opportunities:
                rows.append(
                    {
                        "rider_id": opportunity.rider_id,
                        "route_idx": route_eval.route_idx,
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
                        "observability_score": opportunity.observability_score,
                        "success_probability": opportunity.success_probability,
                    }
                )

    output_path = domain_config.ml_dir / "rendezvous_meeting_point_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "rider_id",
            "route_idx",
            "walk_min",
            "anchor_progress",
            "travel_fraction",
            "ambiguity_count",
            "local_straightness",
            "turn_severity",
            "anchor_clutter",
            "urban_clutter_index",
            "sidewalk_access_score",
            "building_height_proxy",
            "observability_score",
            "success_probability",
        ],
    ).to_parquet(output_path, index=False)
    print(f"Wrote {len(rows):,} opportunities to {output_path}")


if __name__ == "__main__":
    main()
