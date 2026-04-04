from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data_prep.domain_config import DomainConfig, get_domain_config

from .data_types import DriverTrip


def load_domain_assets(
    domain: str,
    *,
    driver_columns: list[str] | None = None,
    rider_columns: list[str] | None = None,
    split: str | None = None,
) -> tuple[DomainConfig, pd.DataFrame, pd.DataFrame]:
    config = get_domain_config(domain)
    drivers = pd.read_parquet(config.drivers_path(), columns=driver_columns)
    riders = pd.read_parquet(config.riders_path(), columns=rider_columns)
    if split is not None:
        drivers = drivers.loc[drivers["split"] == split].reset_index(drop=True)
        riders = riders.loc[riders["split"] == split].reset_index(drop=True)
    return config, drivers, riders


def load_h3_stats_dict(config: DomainConfig) -> dict[str, dict]:
    h3_stats = pd.read_parquet(config.h3_stats_path())
    return {row["h3_cell"]: row.to_dict() for _, row in h3_stats.iterrows()}


def build_driver_trips(
    df: pd.DataFrame,
    *,
    seats: int,
    max_detour_min: float,
    platform_share: float,
    cost_per_mile: float,
    urban_speed_kmh: float,
) -> list[DriverTrip]:
    trips: list[DriverTrip] = []
    for i in range(len(df)):
        row = df.iloc[i]
        dep = row["pickup_datetime"] if "pickup_datetime" in df.columns else datetime(2015, 4, 1)
        trips.append(
            DriverTrip(
                driver_id=i,
                origin=(float(row["origin_lat"]), float(row["origin_lng"])),
                destination=(float(row["dest_lat"]), float(row["dest_lng"])),
                departure_time=dep,
                hour=int(row["hour_of_day"]),
                minute_of_day=dep.hour * 60 + dep.minute,
                trip_distance_miles=float(row["trip_distance_miles"]),
                seats=seats,
                max_detour_minutes=max_detour_min,
                platform_share=platform_share,
                cost_per_mile=cost_per_mile,
                urban_speed_kmh=urban_speed_kmh,
            )
        )
    return trips


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
