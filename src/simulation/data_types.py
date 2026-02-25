"""
Shared data structures for the cold-start / warm-up simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from spatial.h3_utils import LatLng

METERS_PER_MILE = 1609.34


def categorize_route_length(distance_miles: float) -> str:
    if distance_miles < 13:
        return "short"
    if distance_miles < 18:
        return "medium"
    return "long"


@dataclass
class DriverTrip:
    driver_id: int
    origin: LatLng
    destination: LatLng
    departure_time: datetime
    hour: int
    minute_of_day: int = 0
    trip_distance_miles: float = 0.0
    seats: int = 3
    max_detour_minutes: float = 4.0

    @property
    def route_length_category(self) -> str:
        return categorize_route_length(self.trip_distance_miles)


@dataclass
class MatchResult:
    driver_id: int
    rider_id: int
    fare_share: float
    detour_minutes: float


@dataclass
class DriverOutcome:
    driver_id: int
    strategy: str
    route_distance_miles: float
    matched_riders: int
    total_revenue: float
    driving_cost: float
    profit: float
    route_rank_chosen: int
    predicted_profit: float
    compute_time_s: float
    route_length_category: str
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "driver_id": self.driver_id,
            "strategy": self.strategy,
            "route_distance_miles": self.route_distance_miles,
            "matched_riders": self.matched_riders,
            "total_revenue": self.total_revenue,
            "driving_cost": self.driving_cost,
            "profit": self.profit,
            "route_rank_chosen": self.route_rank_chosen,
            "predicted_profit": self.predicted_profit,
            "compute_time_s": self.compute_time_s,
            "route_length_category": self.route_length_category,
            "seed": self.seed,
        }
