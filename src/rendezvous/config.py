from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RendezvousConfig:
    scenario_name: str = "primary"
    domain: str = "yellow"
    route_alternatives: int = 3
    index_bin_minutes: int = 15
    candidate_window_bins: int = 1
    max_request_offset_min: int = 5
    h3_resolution: int = 9
    corridor_k_ring: int = 1
    corridor_densify_step_m: float = 80.0
    meeting_k_ring: int = 1
    max_walk_min: float = 6.0
    walk_speed_kmh: float = 4.5
    seats: int = 3
    platform_share: float = 0.50
    cost_per_mile: float = 0.67
    occlusion_lambda: float = 0.25
    observable_threshold: float = 0.80
    dispatch_batch_seconds: int = 60
    rider_density_pct: int = 100
    min_travel_fraction: float = 0.05
    use_urban_context: bool = True
    urban_context_resolution: int | None = None

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
