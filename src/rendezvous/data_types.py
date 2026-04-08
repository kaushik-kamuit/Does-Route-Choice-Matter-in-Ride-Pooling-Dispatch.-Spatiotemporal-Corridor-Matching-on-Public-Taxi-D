from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from spatial.corridor import Corridor
from spatial.h3_utils import LatLng
from spatial.router import RouteInfo


METERS_PER_MILE = 1609.34


@dataclass(frozen=True)
class DriverTrip:
    driver_id: int
    origin: LatLng
    destination: LatLng
    departure_time: datetime
    hour: int
    minute_of_day: int
    trip_distance_miles: float
    seats: int = 3
    platform_share: float = 0.50
    cost_per_mile: float = 0.67
    walk_speed_kmh: float = 4.5


@dataclass(frozen=True)
class RendezvousOpportunity:
    rider_id: int
    anchor_cell: str
    anchor_idx: int
    pickup_h3: str
    dropoff_h3: str
    fare_share: float
    passenger_count: int
    walk_m: float
    walk_min: float
    anchor_progress: float
    travel_fraction: float
    ambiguity_count: int
    local_straightness: float
    turn_severity: float
    anchor_clutter: float
    observability_score: float
    success_probability: float

    @property
    def nominal_value(self) -> float:
        return self.fare_share

    @property
    def observable_value(self) -> float:
        return self.fare_share * self.success_probability


@dataclass(frozen=True)
class RouteOpportunityEvaluation:
    route_idx: int
    route: RouteInfo
    corridor: Corridor
    route_cells: tuple[str, ...]
    candidate_count: int
    exact_time_candidate_count: int
    feasible_opportunity_count: int
    observable_opportunity_count: int
    route_cost: float
    nominal_route_value: float
    observable_route_value: float
    opportunities: tuple[RendezvousOpportunity, ...]


@dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    route_idx: int
    score: float
    route_distance_miles: float
    route_cost: float
    expected_value: float
    nominal_revenue: float
    realized_revenue: float
    actual_profit: float
    successful_riders: int
    attempted_riders: int
    candidate_count: int
    exact_time_candidate_count: int
    feasible_opportunity_count: int
    observable_opportunity_count: int
    mean_observability: float
    mean_walk_min: float
    nominal_realized_gap: float
    selected_rider_ids: tuple[int, ...]
    successful_rider_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "route_idx": self.route_idx,
            "score": self.score,
            "route_distance_miles": self.route_distance_miles,
            "route_cost": self.route_cost,
            "expected_value": self.expected_value,
            "nominal_revenue": self.nominal_revenue,
            "realized_revenue": self.realized_revenue,
            "actual_profit": self.actual_profit,
            "successful_riders": self.successful_riders,
            "attempted_riders": self.attempted_riders,
            "candidate_count": self.candidate_count,
            "exact_time_candidate_count": self.exact_time_candidate_count,
            "feasible_opportunity_count": self.feasible_opportunity_count,
            "observable_opportunity_count": self.observable_opportunity_count,
            "mean_observability": self.mean_observability,
            "mean_walk_min": self.mean_walk_min,
            "nominal_realized_gap": self.nominal_realized_gap,
        }


@dataclass(frozen=True)
class DriverPolicyEvaluation:
    driver_id: int
    route_evaluations: tuple[RouteOpportunityEvaluation, ...]
    plans: dict[str, PolicyOutcome]


@dataclass(frozen=True)
class RequestState:
    rider_id: int
    pickup_datetime: pd.Timestamp
    expiration_time: pd.Timestamp


@dataclass(frozen=True)
class DispatchOutcome:
    policy: str
    seed: int
    batch_label: pd.Timestamp
    driver_id: int
    route_idx: int
    successful_riders: int
    attempted_riders: int
    realized_revenue: float
    actual_profit: float
    selected_score: float
    mean_wait_min: float
    mean_walk_min: float
    mean_observability: float
    open_requests_before: int

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "batch_label": self.batch_label.isoformat(),
            "driver_id": self.driver_id,
            "route_idx": self.route_idx,
            "successful_riders": self.successful_riders,
            "attempted_riders": self.attempted_riders,
            "realized_revenue": self.realized_revenue,
            "actual_profit": self.actual_profit,
            "selected_score": self.selected_score,
            "mean_wait_min": self.mean_wait_min,
            "mean_walk_min": self.mean_walk_min,
            "mean_observability": self.mean_observability,
            "open_requests_before": self.open_requests_before,
        }


@dataclass(frozen=True)
class DispatchSummary:
    policy: str
    seed: int
    launched_drivers: int
    served_riders: int
    total_profit: float
    profit_per_driver: float
    service_rate: float
    mean_wait_min: float
    mean_walk_min: float
    mean_observability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "launched_drivers": self.launched_drivers,
            "served_riders": self.served_riders,
            "total_profit": self.total_profit,
            "profit_per_driver": self.profit_per_driver,
            "service_rate": self.service_rate,
            "mean_wait_min": self.mean_wait_min,
            "mean_walk_min": self.mean_walk_min,
            "mean_observability": self.mean_observability,
        }
