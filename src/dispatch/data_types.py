from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from simulation.data_types import DriverTrip


@dataclass(frozen=True)
class DriverState:
    driver_trip: DriverTrip
    departure_batch: pd.Timestamp
    day_of_week: int
    is_weekend: int
    day_of_month: int


@dataclass(frozen=True)
class RequestState:
    rider_id: int
    pickup_datetime: pd.Timestamp
    expiration_time: pd.Timestamp


@dataclass(frozen=True)
class BatchMetrics:
    policy: str
    seed: int
    batch_label: pd.Timestamp
    launched_drivers: int
    open_requests_before: int
    open_requests_after: int
    served_riders: int
    mean_wait_min: float
    mean_eval_time_s: float
    runtime_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "batch_label": self.batch_label.isoformat(),
            "launched_drivers": self.launched_drivers,
            "open_requests_before": self.open_requests_before,
            "open_requests_after": self.open_requests_after,
            "served_riders": self.served_riders,
            "mean_wait_min": self.mean_wait_min,
            "mean_eval_time_s": self.mean_eval_time_s,
            "runtime_s": self.runtime_s,
        }


@dataclass(frozen=True)
class DispatchOutcome:
    policy: str
    seed: int
    batch_label: pd.Timestamp
    launched_at: pd.Timestamp
    driver_id: int
    route_rank_chosen: int
    route_distance_miles: float
    matched_riders: int
    matched_passengers: int
    total_revenue: float
    driving_cost: float
    profit: float
    predicted_profit: float
    selected_score: float
    retrieved_candidate_count: int
    exact_time_candidate_count: int
    available_candidate_count: int
    feasible_count: int
    mean_wait_min: float
    mean_detour_min: float
    compute_time_s: float
    route_length_category: str
    open_requests_before: int

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "batch_label": self.batch_label.isoformat(),
            "launched_at": self.launched_at.isoformat(),
            "driver_id": self.driver_id,
            "route_rank_chosen": self.route_rank_chosen,
            "route_distance_miles": self.route_distance_miles,
            "matched_riders": self.matched_riders,
            "matched_passengers": self.matched_passengers,
            "total_revenue": self.total_revenue,
            "driving_cost": self.driving_cost,
            "profit": self.profit,
            "predicted_profit": self.predicted_profit,
            "selected_score": self.selected_score,
            "retrieved_candidate_count": self.retrieved_candidate_count,
            "exact_time_candidate_count": self.exact_time_candidate_count,
            "available_candidate_count": self.available_candidate_count,
            "feasible_count": self.feasible_count,
            "mean_wait_min": self.mean_wait_min,
            "mean_detour_min": self.mean_detour_min,
            "compute_time_s": self.compute_time_s,
            "route_length_category": self.route_length_category,
            "open_requests_before": self.open_requests_before,
        }


@dataclass(frozen=True)
class DispatchSummary:
    policy: str
    seed: int
    launched_drivers: int
    total_profit: float
    profit_per_launched_driver: float
    served_riders: int
    rider_service_rate: float
    mean_wait_min: float
    mean_matched_riders_per_driver: float
    seat_occupancy: float
    mean_detour_min: float
    mean_eval_time_s: float
    mean_batch_runtime_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "launched_drivers": self.launched_drivers,
            "total_profit": self.total_profit,
            "profit_per_launched_driver": self.profit_per_launched_driver,
            "served_riders": self.served_riders,
            "rider_service_rate": self.rider_service_rate,
            "mean_wait_min": self.mean_wait_min,
            "mean_matched_riders_per_driver": self.mean_matched_riders_per_driver,
            "seat_occupancy": self.seat_occupancy,
            "mean_detour_min": self.mean_detour_min,
            "mean_eval_time_s": self.mean_eval_time_s,
            "mean_batch_runtime_s": self.mean_batch_runtime_s,
        }
