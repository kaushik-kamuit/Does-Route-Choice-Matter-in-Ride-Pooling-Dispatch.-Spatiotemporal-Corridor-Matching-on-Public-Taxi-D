from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

YEAR = 2015
DEFAULT_MONTH_WINDOW = (1, 2, 3, 4)


@dataclass(frozen=True)
class DomainConfig:
    name: str
    display_name: str
    azure_base_path: str
    raw_filename_prefix: str
    raw_columns: tuple[str, ...]
    column_renames: dict[str, str]
    legacy_raw_dir: Path | None = None
    legacy_processed_dir: Path | None = None
    legacy_ml_dir: Path | None = None
    legacy_route_cache_path: Path | None = None

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw" / self.name

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed" / self.name

    @property
    def ml_dir(self) -> Path:
        return ROOT / "data" / "ml" / self.name

    @property
    def results_dir(self) -> Path:
        return ROOT / "results" / self.name

    @property
    def models_dir(self) -> Path:
        return ROOT / "models" / self.name

    @property
    def urban_context_dir(self) -> Path:
        return ROOT / "data" / "urban_context"

    @property
    def route_cache_path(self) -> Path:
        if self.legacy_route_cache_path and self.legacy_route_cache_path.exists():
            return self.legacy_route_cache_path
        return ROOT / "data" / f"route_cache_{self.name}.db"

    def raw_month_path(self, month: int) -> Path:
        new_path = self.raw_dir / f"{self.raw_filename_prefix}_{YEAR}-{month:02d}.parquet"
        if new_path.exists():
            return new_path
        if self.legacy_raw_dir is not None:
            legacy_path = self.legacy_raw_dir / f"{self.raw_filename_prefix}_{YEAR}-{month:02d}.parquet"
            if legacy_path.exists():
                return legacy_path
        return new_path

    def drivers_path(self) -> Path:
        new_path = self.processed_dir / "drivers.parquet"
        if new_path.exists():
            return new_path
        if self.legacy_processed_dir is not None:
            legacy_path = self.legacy_processed_dir / "drivers.parquet"
            if legacy_path.exists():
                return legacy_path
        return new_path

    def riders_path(self) -> Path:
        new_path = self.processed_dir / "riders.parquet"
        if new_path.exists():
            return new_path
        if self.legacy_processed_dir is not None:
            legacy_path = self.legacy_processed_dir / "riders.parquet"
            if legacy_path.exists():
                return legacy_path
        return new_path

    def h3_stats_path(self) -> Path:
        new_path = self.ml_dir / "h3_cell_stats.parquet"
        if new_path.exists():
            return new_path
        if self.legacy_ml_dir is not None:
            legacy_path = self.legacy_ml_dir / "h3_cell_stats.parquet"
            if legacy_path.exists():
                return legacy_path
        return new_path

    def h3_qh_stats_path(self) -> Path:
        new_path = self.ml_dir / "h3_qh_stats.parquet"
        if new_path.exists():
            return new_path
        if self.legacy_ml_dir is not None:
            legacy_path = self.legacy_ml_dir / "h3_qh_stats.parquet"
            if legacy_path.exists():
                return legacy_path
        return new_path

    def training_dataset_path(self, tag: str = "") -> Path:
        stem = "training_dataset_v2"
        if tag:
            stem = f"{stem}_{tag}"
        new_path = self.ml_dir / f"{stem}.parquet"
        if new_path.exists():
            return new_path
        if self.legacy_ml_dir is not None:
            legacy_path = self.legacy_ml_dir / f"{stem}.parquet"
            if legacy_path.exists():
                return legacy_path
        return new_path

    def model_path(self, suffix: str = "") -> Path:
        filename = "profit_model_v2.pkl" if not suffix else f"profit_model_v2_{suffix}.pkl"
        legacy_path = ROOT / "models" / filename
        new_path = self.models_dir / filename
        if new_path.exists():
            return new_path
        if self.legacy_ml_dir is not None and legacy_path.exists():
            return legacy_path
        return new_path

    def urban_context_stats_path(self, resolution: int = 9) -> Path:
        return self.urban_context_dir / "processed" / f"urban_context_h3_res{resolution}.parquet"


YELLOW_CONFIG = DomainConfig(
    name="yellow",
    display_name="NYC Yellow Taxi",
    azure_base_path="az://nyctlc/yellow",
    raw_filename_prefix="yellow_tripdata",
    raw_columns=(
        "tpepPickupDateTime",
        "tpepDropoffDateTime",
        "startLat",
        "startLon",
        "endLat",
        "endLon",
        "tripDistance",
        "fareAmount",
        "tipAmount",
        "totalAmount",
        "passengerCount",
    ),
    column_renames={
        "tpepPickupDateTime": "pickup_datetime",
        "tpepDropoffDateTime": "dropoff_datetime",
        "startLat": "pickup_lat",
        "startLon": "pickup_lng",
        "endLat": "dropoff_lat",
        "endLon": "dropoff_lng",
        "tripDistance": "trip_distance_miles",
        "fareAmount": "fare_amount",
        "tipAmount": "tip_amount",
        "totalAmount": "total_amount",
        "passengerCount": "passenger_count",
    },
    legacy_raw_dir=ROOT / "data" / "raw",
    legacy_processed_dir=ROOT / "data" / "processed",
    legacy_ml_dir=ROOT / "data" / "ml",
    legacy_route_cache_path=ROOT / "data" / "route_cache.db",
)

GREEN_CONFIG = DomainConfig(
    name="green",
    display_name="NYC Green Taxi",
    azure_base_path="az://nyctlc/green",
    raw_filename_prefix="green_tripdata",
    raw_columns=(
        "lpepPickupDatetime",
        "lpepDropoffDatetime",
        "pickupLatitude",
        "pickupLongitude",
        "dropoffLatitude",
        "dropoffLongitude",
        "tripDistance",
        "fareAmount",
        "tipAmount",
        "totalAmount",
        "passengerCount",
    ),
    column_renames={
        "lpepPickupDatetime": "pickup_datetime",
        "lpepDropoffDatetime": "dropoff_datetime",
        "pickupLatitude": "pickup_lat",
        "pickupLongitude": "pickup_lng",
        "dropoffLatitude": "dropoff_lat",
        "dropoffLongitude": "dropoff_lng",
        "tripDistance": "trip_distance_miles",
        "fareAmount": "fare_amount",
        "tipAmount": "tip_amount",
        "totalAmount": "total_amount",
        "passengerCount": "passenger_count",
    },
)

DOMAIN_CONFIGS = {
    "yellow": YELLOW_CONFIG,
    "green": GREEN_CONFIG,
}


def get_domain_config(domain: str) -> DomainConfig:
    key = domain.lower().strip()
    if key not in DOMAIN_CONFIGS:
        raise ValueError(f"Unsupported domain '{domain}'. Expected one of: {', '.join(sorted(DOMAIN_CONFIGS))}")
    return DOMAIN_CONFIGS[key]
