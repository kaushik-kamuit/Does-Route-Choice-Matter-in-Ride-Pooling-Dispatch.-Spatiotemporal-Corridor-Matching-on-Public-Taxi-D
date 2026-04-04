from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    scenario_name: str = "primary"
    index_bin_minutes: int = 15
    candidate_window_bins: int = 1
    max_request_offset_min: int | None = None
    max_detour_min: float = 4.0
    rider_presample_frac: float = 0.25
    h3_resolution: int = 9
    corridor_k_ring: int = 1
    corridor_densify_step_m: float = 80.0
    platform_share: float = 0.50
    cost_per_mile: float = 0.67
    urban_speed_kmh: float = 40.0
    seats: int = 3

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


DEFAULT_CONFIG = ExperimentConfig()
