from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from matching.rider_index import RiderIndex
from spatial.router import OSRMRouter

from .config import RendezvousConfig
from .data_types import DispatchOutcome, DispatchSummary, RequestState
from .domain_io import build_driver_trips
from .evaluator import ALL_POLICIES, evaluate_driver_policies
from .selectors import MLMeetingPointSelector
from .urban_context import UrbanContextIndex


class RendezvousDispatcher:
    def __init__(
        self,
        config: RendezvousConfig,
        *,
        router: OSRMRouter,
        ml_selector: MLMeetingPointSelector | None = None,
        urban_context: UrbanContextIndex | None = None,
    ) -> None:
        self.config = config
        self.router = router
        self.ml_selector = ml_selector
        self.urban_context = urban_context
        self._batch_delta = pd.Timedelta(seconds=config.dispatch_batch_seconds)

    def prepare_rider_pool(
        self,
        riders_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, RiderIndex, dict[int, RequestState], dict[pd.Timestamp, list[int]]]:
        if self.config.rider_density_pct >= 100:
            sampled = riders_df.reset_index(drop=True)
        else:
            sampled = riders_df.sample(frac=self.config.rider_density_pct / 100.0, random_state=42).reset_index(drop=True)

        rider_index = RiderIndex(sampled, index_bin_minutes=self.config.index_bin_minutes)
        request_states = {
            int(rider_id): RequestState(
                rider_id=int(rider_id),
                pickup_datetime=pd.Timestamp(row["pickup_datetime"]),
                expiration_time=pd.Timestamp(row["pickup_datetime"])
                + pd.Timedelta(minutes=self.config.max_request_offset_min),
            )
            for rider_id, row in sampled.iterrows()
        }
        request_batches: dict[pd.Timestamp, list[int]] = defaultdict(list)
        for rider_id, row in sampled.iterrows():
            request_batches[self._batch_label(pd.Timestamp(row["pickup_datetime"]))].append(int(rider_id))
        return sampled, rider_index, request_states, request_batches

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
    ) -> tuple[list[DispatchOutcome], DispatchSummary]:
        if policy not in ALL_POLICIES:
            raise ValueError(f"Unsupported policy '{policy}'")

        sampled_riders_df = sampled_riders_df if sampled_riders_df is not None else riders_df.reset_index(drop=True)
        rider_index = rider_index if rider_index is not None else RiderIndex(sampled_riders_df, index_bin_minutes=self.config.index_bin_minutes)
        request_states = request_states if request_states is not None else {
            int(rider_id): RequestState(
                rider_id=int(rider_id),
                pickup_datetime=pd.Timestamp(row["pickup_datetime"]),
                expiration_time=pd.Timestamp(row["pickup_datetime"])
                + pd.Timedelta(minutes=self.config.max_request_offset_min),
            )
            for rider_id, row in sampled_riders_df.iterrows()
        }
        request_batches = request_batches if request_batches is not None else self.prepare_rider_pool(sampled_riders_df)[3]

        driver_trips = build_driver_trips(drivers_df, self.config)
        driver_batches: dict[pd.Timestamp, list] = defaultdict(list)
        for trip in driver_trips:
            driver_batches[self._batch_label(pd.Timestamp(trip.departure_time))].append(trip)
        last_driver_batch = max(driver_batches) if driver_batches else None

        open_riders: set[int] = set()
        eligible_riders: set[int] = set()
        outcomes: list[DispatchOutcome] = []
        total_profit = 0.0
        total_wait_min = 0.0
        total_walk_min = 0.0
        total_observability = 0.0
        served_riders = 0
        drivers_skipped_no_route = 0

        for batch_label in sorted(set(driver_batches) | set(request_batches)):
            for rider_id in request_batches.get(batch_label, []):
                open_riders.add(rider_id)
                if last_driver_batch is not None and batch_label <= last_driver_batch:
                    eligible_riders.add(rider_id)

            expired = {
                rider_id
                for rider_id in open_riders
                if request_states[rider_id].expiration_time < batch_label
            }
            open_riders.difference_update(expired)

            for trip in driver_batches.get(batch_label, []):
                available = {
                    rider_id
                    for rider_id in open_riders
                    if request_states[rider_id].pickup_datetime
                    <= pd.Timestamp(trip.departure_time)
                    <= request_states[rider_id].expiration_time
                }
                routes = self.router.get_alternative_routes(
                    trip.origin,
                    trip.destination,
                    max_alternatives=self.config.route_alternatives,
                )
                if not routes:
                    drivers_skipped_no_route += 1
                    continue
                evaluation = evaluate_driver_policies(
                    trip,
                    rider_index,
                    self.config,
                    routes=routes,
                    available_rider_ids=available,
                    ml_selector=self.ml_selector,
                    urban_context=self.urban_context,
                    seed=seed,
                )
                plan = evaluation.plans.get(policy)
                if plan is None:
                    continue

                waits = [
                    max(
                        0.0,
                        (pd.Timestamp(trip.departure_time) - request_states[rider_id].pickup_datetime).total_seconds() / 60.0,
                    )
                    for rider_id in plan.successful_rider_ids
                ]
                retired_rider_ids = plan.selected_rider_ids if self.config.retire_failed_attempts else plan.successful_rider_ids
                open_riders.difference_update(retired_rider_ids)
                total_profit += plan.actual_profit
                total_wait_min += float(sum(waits))
                total_walk_min += plan.mean_walk_min * max(plan.attempted_riders, 1)
                total_observability += plan.mean_observability * max(plan.attempted_riders, 1)
                served_riders += plan.successful_riders
                outcomes.append(
                    DispatchOutcome(
                        policy=policy,
                        seed=seed,
                        batch_label=batch_label,
                        driver_id=trip.driver_id,
                        route_idx=plan.route_idx,
                        successful_riders=plan.successful_riders,
                        attempted_riders=plan.attempted_riders,
                        realized_revenue=plan.realized_revenue,
                        actual_profit=plan.actual_profit,
                        selected_score=plan.score,
                        mean_wait_min=float(np.mean(waits)) if waits else 0.0,
                        mean_walk_min=plan.mean_walk_min,
                        mean_observability=plan.mean_observability,
                        open_requests_before=len(available),
                        attempted_rider_ids=plan.selected_rider_ids,
                        successful_rider_ids=plan.successful_rider_ids,
                    )
                )

        launched = len(outcomes)
        attempted_total = sum(outcome.attempted_riders for outcome in outcomes)
        requested_drivers = len(driver_trips)
        route_ready_drivers = requested_drivers - drivers_skipped_no_route
        summary = DispatchSummary(
            policy=policy,
            seed=seed,
            requested_drivers=requested_drivers,
            launched_drivers=launched,
            drivers_skipped_no_route=drivers_skipped_no_route,
            route_coverage_rate=route_ready_drivers / max(requested_drivers, 1),
            eligible_riders=len(eligible_riders),
            served_riders=served_riders,
            total_profit=total_profit,
            profit_per_driver=total_profit / max(launched, 1),
            service_rate=served_riders / max(len(eligible_riders), 1),
            mean_wait_min=total_wait_min / max(served_riders, 1),
            mean_walk_min=total_walk_min / max(attempted_total, 1),
            mean_observability=total_observability / max(attempted_total, 1),
        )
        return outcomes, summary

    def _batch_label(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(f"{self.config.dispatch_batch_seconds}s")
