from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_prep.domain_config import DomainConfig, get_domain_config

from .config import RendezvousConfig
from .data_types import DriverTrip


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
