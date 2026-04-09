from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RendezvousConfig:
    scenario_name: str = "primary"
    domain: str = "yellow"
    time_slice: str = "all_day"
    area_slice: str = "all"
    hour_start: int | None = None
    hour_end: int | None = None
    route_alternatives: int = 3
    index_bin_minutes: int = 15
    candidate_window_bins: int = 1
    max_request_offset_min: int = 5
    h3_resolution: int = 9
    corridor_k_ring: int = 1
    corridor_densify_step_m: float = 80.0
    require_dropoff_in_corridor: bool = False
    meeting_k_ring: int = 1
    max_walk_min: float = 6.0
    walk_speed_kmh: float = 4.5
    walk_penalty_per_min: float = 0.5
    seats: int = 3
    platform_share: float = 0.50
    cost_per_mile: float = 0.67
    occlusion_lambda: float = 0.25
    observable_threshold: float = 0.80
    observability_profile: str = "equal"
    dispatch_batch_seconds: int = 60
    retire_failed_attempts: bool = True
    rider_density_pct: int = 100
    min_travel_fraction: float = 0.05
    use_urban_context: bool = True
    urban_context_resolution: int | None = None
    observability_ablation: str = "full"

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
