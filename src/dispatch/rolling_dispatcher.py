from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from data_prep.domain_config import DomainConfig
from matching.rider_index import RiderIndex
from models.predict import ProfitPredictor
from simulation.domain_io import build_driver_trips, load_h3_stats_dict
from simulation.route_evaluator import evaluate_driver_policies
from spatial.router import OSRMRouter

from .config import DispatchConfig
from .data_types import BatchMetrics, DispatchOutcome, DispatchSummary, DriverState, RequestState

ALL_POLICIES = [
    "coldstart",
    "random",
    "heuristic_count",
    "heuristic_fare_density",
    "heuristic_feasible_count",
    "heuristic_profit_proxy",
    "warmup",
    "oracle",
]


class RollingDispatcher:
    """Rolling-horizon dispatch-lite simulator built on the validated route evaluator."""

    def __init__(
        self,
        config: DispatchConfig,
        *,
        domain_config: DomainConfig,
        router: OSRMRouter,
        predictor: ProfitPredictor,
        h3_stats_dict: dict[str, dict] | None = None,
    ) -> None:
        self.config = config
        self.domain_config = domain_config
        self.router = router
        self.predictor = predictor
        self.h3_stats_dict = h3_stats_dict if h3_stats_dict is not None else load_h3_stats_dict(domain_config)
        self._batch_delta = pd.Timedelta(seconds=config.batch_seconds)

    def _batch_label(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(f"{self.config.batch_seconds}s")

    def _prepare_driver_states(self, drivers_df: pd.DataFrame) -> list[DriverState]:
        trips = build_driver_trips(
            drivers_df,
            seats=self.config.seats,
            max_detour_min=self.config.max_detour_min,
            platform_share=self.config.platform_share,
            cost_per_mile=self.config.cost_per_mile,
            urban_speed_kmh=self.config.urban_speed_kmh,
        )
        states: list[DriverState] = []
        dows = drivers_df["day_of_week"].to_numpy()
        weekends = drivers_df["is_weekend"].to_numpy()
        days = drivers_df["pickup_datetime"].dt.day.to_numpy()
        for idx, trip in enumerate(trips):
            states.append(
                DriverState(
                    driver_trip=trip,
                    departure_batch=self._batch_label(pd.Timestamp(trip.departure_time)),
                    day_of_week=int(dows[idx]),
                    is_weekend=int(weekends[idx]),
                    day_of_month=int(days[idx]),
                )
            )
        states.sort(key=lambda state: (state.departure_batch, state.driver_trip.departure_time, state.driver_trip.driver_id))
        return states

    def _prepare_request_states(self, riders_df: pd.DataFrame) -> dict[int, RequestState]:
        request_states: dict[int, RequestState] = {}
        for rider_id, row in riders_df.iterrows():
            pickup_ts = pd.Timestamp(row["pickup_datetime"])
            request_states[int(rider_id)] = RequestState(
                rider_id=int(rider_id),
                pickup_datetime=pickup_ts,
                expiration_time=pickup_ts + pd.Timedelta(minutes=self.config.max_request_offset_min),
            )
        return request_states

    def _sample_density(self, riders_df: pd.DataFrame) -> pd.DataFrame:
        density = max(1, min(100, self.config.density_pct))
        if density >= 100:
            return riders_df.reset_index(drop=True)
        sampled = riders_df.sample(frac=density / 100.0, random_state=42).reset_index(drop=True)
        return sampled

    def _group_states_by_batch(self, driver_states: list[DriverState]) -> dict[pd.Timestamp, list[DriverState]]:
        grouped: dict[pd.Timestamp, list[DriverState]] = defaultdict(list)
        for state in driver_states:
            grouped[state.departure_batch].append(state)
        return grouped

    def _group_requests_by_batch(self, riders_df: pd.DataFrame) -> dict[pd.Timestamp, list[int]]:
        grouped: dict[pd.Timestamp, list[int]] = defaultdict(list)
        for rider_id, row in riders_df.iterrows():
            grouped[self._batch_label(pd.Timestamp(row["pickup_datetime"]))].append(int(rider_id))
        return grouped

    @staticmethod
    def _provisional_sort_key(row: tuple[float, int, DriverState, list, float]) -> tuple[float, pd.Timestamp, int]:
        score, _driver_id, state, _routes, _eval_time = row
        # Higher policy scores should launch first; ties fall back to earliest departure and lower driver id.
        return (-score, pd.Timestamp(state.driver_trip.departure_time), state.driver_trip.driver_id)

    def prepare_rider_pool(
        self,
        riders_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, RiderIndex, dict[int, RequestState], dict[pd.Timestamp, list[int]]]:
        sampled_riders = self._sample_density(riders_df)
        rider_index = RiderIndex(sampled_riders, index_bin_minutes=self.config.index_bin_minutes)
        request_states = self._prepare_request_states(sampled_riders)
        request_batches = self._group_requests_by_batch(sampled_riders)
        return sampled_riders, rider_index, request_states, request_batches

    def _currently_available_riders(
        self,
        open_riders: set[int],
        request_states: dict[int, RequestState],
        query_time: pd.Timestamp,
    ) -> set[int]:
        return {
            rider_id
            for rider_id in open_riders
            if request_states[rider_id].pickup_datetime <= query_time <= request_states[rider_id].expiration_time
        }

    def run_policy(
        self,
        policy: str,
        drivers_df: pd.DataFrame,
        riders_df: pd.DataFrame,
        *,
        seed: int,
        sampled_riders_df: pd.DataFrame | None = None,
        rider_index: RiderIndex | None = None,
        request_states: dict[int, RequestState] | None = None,
        request_batches: dict[pd.Timestamp, list[int]] | None = None,
    ) -> tuple[list[DispatchOutcome], list[BatchMetrics], DispatchSummary]:
        if policy not in ALL_POLICIES:
            raise ValueError(f"Unsupported dispatch policy '{policy}'")

        full_rider_index = sampled_riders_df if sampled_riders_df is not None else self._sample_density(riders_df)
        rider_index = rider_index if rider_index is not None else RiderIndex(full_rider_index, index_bin_minutes=self.config.index_bin_minutes)
        request_states = request_states if request_states is not None else self._prepare_request_states(full_rider_index)
        driver_states = self._prepare_driver_states(drivers_df)
        driver_batches = self._group_states_by_batch(driver_states)
        request_batches = request_batches if request_batches is not None else self._group_requests_by_batch(full_rider_index)
        all_batches = sorted(set(driver_batches) | set(request_batches))

        open_riders: set[int] = set()
        outcomes: list[DispatchOutcome] = []
        batch_metrics: list[BatchMetrics] = []
        served_riders_total = 0
        total_wait_minutes = 0.0
        total_detour_minutes = 0.0

        for batch_label in all_batches:
            batch_start = batch_label
            process_time = batch_label + self._batch_delta
            for rider_id in request_batches.get(batch_label, []):
                open_riders.add(rider_id)

            expired = {
                rider_id
                for rider_id in open_riders
                if request_states[rider_id].expiration_time < batch_start
            }
            open_riders.difference_update(expired)

            active_drivers = driver_batches.get(batch_label, [])
            batch_open_before = len(open_riders)
            if not active_drivers:
                continue

            t_batch = time.perf_counter()
            provisional: list[tuple[float, int, DriverState, list, float]] = []
            eval_times: list[float] = []
            for state in active_drivers:
                departure_time = pd.Timestamp(state.driver_trip.departure_time)
                available_rider_ids = self._currently_available_riders(open_riders, request_states, departure_time)
                routes = self.router.get_alternative_routes(
                    state.driver_trip.origin,
                    state.driver_trip.destination,
                    max_alternatives=3,
                )
                if not routes:
                    continue
                t_eval = time.perf_counter()
                evaluation = evaluate_driver_policies(
                    state.driver_trip,
                    rider_index,
                    self.predictor,
                    self.h3_stats_dict,
                    day_of_week=state.day_of_week,
                    is_weekend=state.is_weekend,
                    day_of_month=state.day_of_month,
                    seed=seed,
                    candidate_window_bins=self.config.candidate_window_bins,
                    max_request_offset_min=self.config.max_request_offset_min,
                    h3_resolution=self.config.h3_resolution,
                    corridor_k_ring=self.config.corridor_k_ring,
                    corridor_densify_step_m=self.config.corridor_densify_step_m,
                    available_rider_ids=available_rider_ids,
                    routes=routes,
                )
                eval_time = time.perf_counter() - t_eval
                eval_times.append(eval_time)
                plan = evaluation.plans.get(policy)
                if plan is None:
                    continue
                provisional.append((plan.score, state.driver_trip.driver_id, state, routes, eval_time))

            provisional.sort(key=self._provisional_sort_key)
            batch_served = 0
            batch_waits: list[float] = []

            for _, _, state, routes, _initial_eval_time in provisional:
                if not routes:
                    continue
                open_before_driver = len(open_riders)
                departure_time = pd.Timestamp(state.driver_trip.departure_time)
                available_rider_ids = self._currently_available_riders(open_riders, request_states, departure_time)
                t_eval = time.perf_counter()
                evaluation = evaluate_driver_policies(
                    state.driver_trip,
                    rider_index,
                    self.predictor,
                    self.h3_stats_dict,
                    day_of_week=state.day_of_week,
                    is_weekend=state.is_weekend,
                    day_of_month=state.day_of_month,
                    seed=seed,
                    candidate_window_bins=self.config.candidate_window_bins,
                    max_request_offset_min=self.config.max_request_offset_min,
                    h3_resolution=self.config.h3_resolution,
                    corridor_k_ring=self.config.corridor_k_ring,
                    corridor_densify_step_m=self.config.corridor_densify_step_m,
                    available_rider_ids=available_rider_ids,
                    routes=routes,
                )
                eval_time = time.perf_counter() - t_eval
                eval_times.append(eval_time)
                plan = evaluation.plans.get(policy)
                if plan is None:
                    continue

                matched_rows = [match for match in plan.matched if match["rider_idx"] in available_rider_ids]
                matched_ids = [match["rider_idx"] for match in matched_rows]
                matched_passengers = sum(match["passenger_count"] for match in matched_rows)
                mean_wait_min = 0.0
                mean_detour_min = 0.0
                if matched_ids:
                    waits = [
                        max(0.0, (departure_time - request_states[rider_id].pickup_datetime).total_seconds() / 60.0)
                        for rider_id in matched_ids
                    ]
                    detours = [match["detour_minutes"] for match in matched_rows]
                    mean_wait_min = float(np.mean(waits)) if waits else 0.0
                    mean_detour_min = float(np.mean(detours)) if detours else 0.0
                    batch_waits.extend(waits)
                    served_riders_total += len(matched_ids)
                    total_wait_minutes += float(sum(waits))
                    total_detour_minutes += float(sum(detours))
                    batch_served += len(matched_ids)
                    open_riders.difference_update(matched_ids)

                outcome = DispatchOutcome(
                    policy=policy,
                    seed=seed,
                    batch_label=batch_label,
                    launched_at=departure_time,
                    driver_id=state.driver_trip.driver_id,
                    route_rank_chosen=plan.outcome.route_rank_chosen,
                    route_distance_miles=plan.outcome.route_distance_miles,
                    matched_riders=len(matched_ids),
                    matched_passengers=matched_passengers,
                    total_revenue=plan.outcome.total_revenue,
                    driving_cost=plan.outcome.driving_cost,
                    profit=plan.outcome.profit,
                    predicted_profit=plan.outcome.predicted_profit,
                    selected_score=plan.score,
                    retrieved_candidate_count=plan.retrieved_candidates_considered,
                    exact_time_candidate_count=plan.exact_time_candidates_considered,
                    available_candidate_count=plan.candidates_considered,
                    feasible_count=plan.feasible_count,
                    mean_wait_min=mean_wait_min,
                    mean_detour_min=mean_detour_min,
                    compute_time_s=eval_time,
                    route_length_category=plan.outcome.route_length_category,
                    open_requests_before=open_before_driver,
                )
                outcomes.append(outcome)

            batch_metrics.append(
                BatchMetrics(
                    policy=policy,
                    seed=seed,
                    batch_label=batch_label,
                    launched_drivers=len(active_drivers),
                    open_requests_before=batch_open_before,
                    open_requests_after=len(open_riders),
                    served_riders=batch_served,
                    mean_wait_min=float(np.mean(batch_waits)) if batch_waits else 0.0,
                    mean_eval_time_s=float(np.mean(eval_times)) if eval_times else 0.0,
                    runtime_s=time.perf_counter() - t_batch,
                )
            )

        launched_drivers = len(outcomes)
        total_profit = float(sum(outcome.profit for outcome in outcomes))
        mean_wait = total_wait_minutes / max(served_riders_total, 1)
        mean_detour = total_detour_minutes / max(served_riders_total, 1)
        mean_eval = float(np.mean([outcome.compute_time_s for outcome in outcomes])) if outcomes else 0.0
        mean_batch_runtime = float(np.mean([metric.runtime_s for metric in batch_metrics])) if batch_metrics else 0.0
        total_passengers = sum(outcome.matched_passengers for outcome in outcomes)
        seat_capacity = launched_drivers * self.config.seats

        summary = DispatchSummary(
            policy=policy,
            seed=seed,
            launched_drivers=launched_drivers,
            total_profit=total_profit,
            profit_per_launched_driver=total_profit / max(launched_drivers, 1),
            served_riders=served_riders_total,
            rider_service_rate=served_riders_total / max(len(full_rider_index), 1),
            mean_wait_min=mean_wait,
            mean_matched_riders_per_driver=float(np.mean([outcome.matched_riders for outcome in outcomes])) if outcomes else 0.0,
            seat_occupancy=total_passengers / max(seat_capacity, 1),
            mean_detour_min=mean_detour,
            mean_eval_time_s=mean_eval,
            mean_batch_runtime_s=mean_batch_runtime,
        )
        return outcomes, batch_metrics, summary
