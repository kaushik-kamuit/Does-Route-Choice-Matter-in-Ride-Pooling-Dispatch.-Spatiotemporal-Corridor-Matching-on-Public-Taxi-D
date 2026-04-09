from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--time-slice", type=str, default="all_day")
    parser.add_argument("--hour-start", type=int, default=None)
    parser.add_argument("--hour-end", type=int, default=None)
    parser.add_argument("--density", type=int, default=100)
    parser.add_argument("--occlusion-lambda", type=float, default=0.25)
    parser.add_argument(
        "--observability-profile",
        type=str,
        default="equal",
        choices=["equal", "calibrated"],
    )
    parser.add_argument("--max-riders", type=int, default=None)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--disable-urban-context", action="store_true")
    args = parser.parse_args()

    config = RendezvousConfig(
        domain=args.domain,
        time_slice=args.time_slice,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
        rider_density_pct=args.density,
        occlusion_lambda=args.occlusion_lambda,
        observability_profile=args.observability_profile,
        use_urban_context=not args.disable_urban_context,
    )
    domain_config, drivers_df, riders_df = load_domain_assets(
        args.domain,
        split="train",
        driver_columns=DRIVER_COLUMNS,
        rider_columns=RIDER_COLUMNS,
    )
    if args.hour_start is not None and args.hour_end is not None:
        drivers_df = _filter_by_hour_range(drivers_df, args.hour_start, args.hour_end)
        riders_df = _filter_by_hour_range(riders_df, args.hour_start, args.hour_end)
    if config.rider_density_pct < 100:
        riders_df = riders_df.sample(frac=config.rider_density_pct / 100.0, random_state=42).reset_index(drop=True)
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
                        "driver_id": trip.driver_id,
                        "rider_id": opportunity.rider_id,
                        "route_idx": route_eval.route_idx,
                        "anchor_idx": opportunity.anchor_idx,
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
                        "context_is_imputed": float(opportunity.context_is_imputed),
                        "observability_score": opportunity.observability_score,
                        "success_probability": opportunity.success_probability,
                        "observed_success": float(
                            _stable_uniform(
                                trip.driver_id,
                                route_eval.route_idx,
                                opportunity.rider_id,
                                opportunity.anchor_idx,
                            )
                            <= opportunity.success_probability
                        ),
                        "dataset_split": _dataset_split(trip.driver_id),
                        "domain": args.domain,
                        "time_slice": config.time_slice,
                        "hour_start": config.hour_start,
                        "hour_end": config.hour_end,
                        "rider_density_pct": config.rider_density_pct,
                        "occlusion_lambda": config.occlusion_lambda,
                        "observability_profile": config.observability_profile,
                        "use_urban_context": config.use_urban_context,
                    }
                )

    output_path = domain_config.ml_dir / "rendezvous_meeting_point_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "driver_id",
            "rider_id",
            "route_idx",
            "anchor_idx",
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
            "context_is_imputed",
            "observability_score",
            "success_probability",
            "observed_success",
            "dataset_split",
            "domain",
            "time_slice",
            "hour_start",
            "hour_end",
            "rider_density_pct",
            "occlusion_lambda",
            "observability_profile",
            "use_urban_context",
        ],
    ).to_parquet(output_path, index=False)
    router.flush_cache()
    print(f"Wrote {len(rows):,} opportunities to {output_path}")


def _stable_uniform(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16 - 1)


def _dataset_split(driver_id: int) -> str:
    bucket = int(hashlib.sha256(str(driver_id).encode("utf-8")).hexdigest()[:4], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "valid"
    return "test"


def _filter_by_hour_range(df: pd.DataFrame, hour_start: int, hour_end: int) -> pd.DataFrame:
    hours = pd.to_datetime(df["pickup_datetime"]).dt.hour
    if hour_start <= hour_end:
        mask = (hours >= hour_start) & (hours < hour_end)
    else:
        mask = (hours >= hour_start) | (hours < hour_end)
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    main()
