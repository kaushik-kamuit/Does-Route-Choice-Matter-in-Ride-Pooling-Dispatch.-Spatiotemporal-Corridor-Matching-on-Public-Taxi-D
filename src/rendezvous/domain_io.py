from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_prep.domain_config import DomainConfig, get_domain_config

from .config import RendezvousConfig
from .data_types import DriverTrip
from .urban_context import UrbanContextIndex


def load_domain_assets(
    domain: str,
    *,
    split: str | None = None,
    driver_columns: list[str] | None = None,
    rider_columns: list[str] | None = None,
) -> tuple[DomainConfig, pd.DataFrame, pd.DataFrame]:
    config = get_domain_config(domain)
    drivers = pd.read_parquet(config.drivers_path(), columns=driver_columns)
    riders = pd.read_parquet(config.riders_path(), columns=rider_columns)
    if split is not None:
        drivers = drivers.loc[drivers["split"] == split].reset_index(drop=True)
        riders = riders.loc[riders["split"] == split].reset_index(drop=True)
    return config, drivers, riders


def build_driver_trips(df: pd.DataFrame, config: RendezvousConfig) -> list[DriverTrip]:
    trips: list[DriverTrip] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        departure = pd.Timestamp(row["pickup_datetime"]).to_pydatetime()
        trips.append(
            DriverTrip(
                driver_id=idx,
                origin=(float(row["origin_lat"]), float(row["origin_lng"])),
                destination=(float(row["dest_lat"]), float(row["dest_lng"])),
                departure_time=departure,
                hour=int(row["hour_of_day"]),
                minute_of_day=departure.hour * 60 + departure.minute,
                trip_distance_miles=float(row["trip_distance_miles"]),
                seats=config.seats,
                platform_share=config.platform_share,
                cost_per_mile=config.cost_per_mile,
                walk_speed_kmh=config.walk_speed_kmh,
            )
        )
    return trips


def apply_area_slice(
    drivers_df: pd.DataFrame,
    riders_df: pd.DataFrame,
    urban_context: UrbanContextIndex,
    *,
    area_slice: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = (area_slice or "all").strip().lower()
    if key == "all" or not urban_context:
        return drivers_df.reset_index(drop=True), riders_df.reset_index(drop=True)
    if key not in {"dense_core", "open_grid"}:
        raise ValueError(f"Unsupported area slice '{area_slice}'. Expected one of: all, dense_core, open_grid")

    if "origin_h3" not in drivers_df.columns or "pickup_h3" not in riders_df.columns:
        raise ValueError("Area slicing requires origin_h3 on drivers and pickup_h3 on riders.")

    def _score(cell: object) -> float:
        features = urban_context.lookup(str(cell))
        return float(features.urban_clutter_index + features.building_height_proxy)

    driver_scores = drivers_df["origin_h3"].map(_score)
    rider_scores = riders_df["pickup_h3"].map(_score)
    combined = pd.concat([driver_scores, rider_scores], ignore_index=True)
    if combined.empty:
        return drivers_df.reset_index(drop=True), riders_df.reset_index(drop=True)

    lower = float(combined.quantile(0.25))
    upper = float(combined.quantile(0.75))
    if key == "dense_core":
        driver_mask = driver_scores >= upper
        rider_mask = rider_scores >= upper
    else:
        driver_mask = driver_scores <= lower
        rider_mask = rider_scores <= lower
    return (
        drivers_df.loc[driver_mask].reset_index(drop=True),
        riders_df.loc[rider_mask].reset_index(drop=True),
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_urban_context_index(config: DomainConfig, rendezvous_config: RendezvousConfig) -> UrbanContextIndex:
    if not rendezvous_config.use_urban_context:
        return UrbanContextIndex()
    resolution = rendezvous_config.urban_context_resolution or rendezvous_config.h3_resolution
    return UrbanContextIndex.from_parquet(config.urban_context_stats_path(resolution))
