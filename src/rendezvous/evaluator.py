from __future__ import annotations

from math import cos, radians

import numpy as np
import pandas as pd
from shapely import LineString, Point

from matching.rider_index import RiderIndex
from spatial.corridor import build_corridor
from spatial.router import RouteInfo

from .config import RendezvousConfig
from .data_types import (
    CorridorCandidate,
    DriverPolicyEvaluation,
    DriverTrip,
    METERS_PER_MILE,
    PolicyOutcome,
    RendezvousOpportunity,
    RouteOpportunityEvaluation,
)
from .meeting_points import (
    anchor_clutter,
    build_route_anchor_cells,
    candidate_anchor_indices,
    cell_center,
    local_straightness,
    rider_walk_m,
    route_progress,
    turn_severity,
)
from .observability import compute_observability_score, pickup_success_probability, weights_for_mode, weights_for_profile
from .selectors import (
    DeterministicMeetingPointSelector,
    MLMeetingPointSelector,
    MeetingPointSelector,
    WalkAwareMeetingPointSelector,
)
from .urban_context import UrbanContextFeatures, UrbanContextIndex

NYC_LAT = 40.7
COS_LAT = cos(radians(NYC_LAT))

ALL_POLICIES = [
    "corridor_only",
    "time_only_baseline",
    "feasible_count_baseline",
    "walk_aware_rendezvous",
    "rendezvous_only",
    "rendezvous_observable",
    "ml_meeting_point_comparator",
]


def evaluate_driver_policies(
    driver: DriverTrip,
    rider_index: RiderIndex,
    config: RendezvousConfig,
    *,
    routes: list[RouteInfo],
    available_rider_ids: set[int] | None = None,
    ml_selector: MLMeetingPointSelector | None = None,
    urban_context: UrbanContextIndex | None = None,
    seed: int = 0,
) -> DriverPolicyEvaluation:
    route_evaluations: list[RouteOpportunityEvaluation] = []
    nominal_selector = DeterministicMeetingPointSelector(use_observability=False)
    observable_selector = DeterministicMeetingPointSelector(use_observability=True)
    walk_selector = WalkAwareMeetingPointSelector(walk_penalty_per_min=config.walk_penalty_per_min)
    for route_idx, route in enumerate(routes):
        corridor = build_corridor(
            route.polyline,
            resolution=config.h3_resolution,
            buffer_rings=config.corridor_k_ring,
            densify_step_m=config.corridor_densify_step_m,
        )
        candidates, stats = rider_index.find_in_corridor_with_stats(
            corridor.corridor_cells,
            driver.minute_of_day,
            window_bins=config.candidate_window_bins,
            max_request_offset_min=config.max_request_offset_min,
            query_datetime=driver.departure_time,
            require_dropoff_in_corridor=config.require_dropoff_in_corridor,
        )
        time_eligible_candidates = candidates
        if available_rider_ids is not None and not candidates.empty:
            candidates = candidates[candidates.index.isin(available_rider_ids)]

        route_cells = build_route_anchor_cells(
            route,
            resolution=config.h3_resolution,
            densify_step_m=config.corridor_densify_step_m,
        )
        opportunities = _build_opportunities(
            driver,
            route,
            route_cells,
            candidates,
            config,
            urban_context=urban_context,
        )
        route_cost = (route.distance_m / METERS_PER_MILE) * driver.cost_per_mile
        candidate_riders = tuple(
            CorridorCandidate(
                rider_id=int(rider_id),
                fare_share=float(row["fare_amount"]) * driver.platform_share,
                passenger_count=int(row["passenger_count"]),
            )
            for rider_id, row in candidates.iterrows()
        )
        nominal_selected = nominal_selector.select(opportunities, seats=driver.seats)
        observable_selected = observable_selector.select(opportunities, seats=driver.seats)
        walk_selected = walk_selector.select(opportunities, seats=driver.seats)
        route_evaluations.append(
            RouteOpportunityEvaluation(
                route_idx=route_idx,
                route=route,
                corridor=corridor,
                route_cells=route_cells,
                candidate_count=len(candidate_riders),
                time_eligible_candidate_count=len(time_eligible_candidates),
                feasible_opportunity_count=len(opportunities),
                observable_opportunity_count=sum(
                    1 for opportunity in opportunities if opportunity.success_probability >= config.observable_threshold
                ),
                route_cost=route_cost,
                nominal_route_value=sum(item.nominal_value for item in nominal_selected) - route_cost,
                observable_route_value=sum(item.observable_value for item in observable_selected) - route_cost,
                walk_route_value=sum(walk_selector.opportunity_value(item) for item in walk_selected) - route_cost,
                candidate_riders=candidate_riders,
                opportunities=tuple(opportunities),
            )
        )

    if not route_evaluations:
        return DriverPolicyEvaluation(driver_id=driver.driver_id, route_evaluations=tuple(), plans={})

    corridor_idx = _choose_route(route_evaluations, lambda row: float(row.candidate_count))
    time_only_idx = _choose_route(route_evaluations, lambda row: -row.route_cost)
    feasible_count_idx = _choose_route(route_evaluations, lambda row: float(row.feasible_opportunity_count))
    walk_idx = _choose_route(route_evaluations, lambda row: row.walk_route_value)
    nominal_idx = _choose_route(route_evaluations, lambda row: row.nominal_route_value)
    observable_idx = _choose_route(route_evaluations, lambda row: row.observable_route_value)
    ml_idx = None
    if ml_selector is not None:
        ml_idx = _choose_route(
            route_evaluations,
            lambda row: _selector_route_value(ml_selector, row.opportunities, seats=driver.seats) - row.route_cost,
        )

    plans = {
        "corridor_only": _build_corridor_policy_outcome(
            driver,
            route_evaluations[corridor_idx],
            config=config,
            score=float(route_evaluations[corridor_idx].candidate_count),
            seed=seed,
        ),
        "time_only_baseline": _build_policy_outcome(
            "time_only_baseline",
            driver,
            route_evaluations[time_only_idx],
            score=-route_evaluations[time_only_idx].route_cost,
            selector=nominal_selector,
            seed=seed + 7,
        ),
        "feasible_count_baseline": _build_policy_outcome(
            "feasible_count_baseline",
            driver,
            route_evaluations[feasible_count_idx],
            score=float(route_evaluations[feasible_count_idx].feasible_opportunity_count),
            selector=nominal_selector,
            seed=seed + 11,
        ),
        "walk_aware_rendezvous": _build_policy_outcome(
            "walk_aware_rendezvous",
            driver,
            route_evaluations[walk_idx],
            score=route_evaluations[walk_idx].walk_route_value,
            selector=walk_selector,
            seed=seed + 13,
        ),
        "rendezvous_only": _build_policy_outcome(
            "rendezvous_only",
            driver,
            route_evaluations[nominal_idx],
            score=route_evaluations[nominal_idx].nominal_route_value,
            selector=nominal_selector,
            seed=seed + 17,
        ),
        "rendezvous_observable": _build_policy_outcome(
            "rendezvous_observable",
            driver,
            route_evaluations[observable_idx],
            score=route_evaluations[observable_idx].observable_route_value,
            selector=observable_selector,
            seed=seed + 31,
        ),
    }
    if ml_selector is not None:
        assert ml_idx is not None
        ml_score = _selector_route_value(ml_selector, route_evaluations[ml_idx].opportunities, seats=driver.seats)
        plans["ml_meeting_point_comparator"] = _build_policy_outcome(
            "ml_meeting_point_comparator",
            driver,
            route_evaluations[ml_idx],
            score=ml_score - route_evaluations[ml_idx].route_cost,
            selector=ml_selector,
            seed=seed + 53,
        )

    return DriverPolicyEvaluation(
        driver_id=driver.driver_id,
        route_evaluations=tuple(route_evaluations),
        plans=plans,
    )


def _build_opportunities(
    driver: DriverTrip,
    route: RouteInfo,
    route_cells: tuple[str, ...],
    candidates: pd.DataFrame,
    config: RendezvousConfig,
    *,
    urban_context: UrbanContextIndex | None = None,
) -> list[RendezvousOpportunity]:
    if candidates.empty or not route_cells:
        return []

    route_line = _make_route_line(route.polyline)
    opportunities: list[RendezvousOpportunity] = []
    walk_speed_mps = max(driver.walk_speed_kmh, 1e-6) * 1000.0 / 3600.0
    base_weights = weights_for_profile(config.observability_profile, domain=config.domain)
    observability_weights = weights_for_mode(config.observability_ablation, base_weights=base_weights)

    for rider_id, row in candidates.iterrows():
        anchor_indices = candidate_anchor_indices(
            route_cells,
            str(row["pickup_h3"]),
            meeting_k_ring=config.meeting_k_ring,
        )
        if not anchor_indices:
            continue

        dropoff_point = Point(float(row["dropoff_lng"]) * COS_LAT, float(row["dropoff_lat"]))
        dropoff_fraction = float(route_line.project(dropoff_point, normalized=True))
        ambiguity_count = len(anchor_indices)

        for anchor_idx in anchor_indices:
            anchor_cell = route_cells[anchor_idx]
            walk_m = rider_walk_m(float(row["pickup_lat"]), float(row["pickup_lng"]), anchor_cell)
            walk_min = walk_m / walk_speed_mps / 60.0
            if walk_min > config.max_walk_min:
                continue

            anchor_lat, anchor_lng = cell_center(anchor_cell)
            anchor_point = Point(anchor_lng * COS_LAT, anchor_lat)
            anchor_fraction = float(route_line.project(anchor_point, normalized=True))
            travel_fraction = dropoff_fraction - anchor_fraction
            if travel_fraction < config.min_travel_fraction:
                continue

            straightness = local_straightness(route_cells, anchor_idx)
            turn = turn_severity(route_cells, anchor_idx)
            route_clutter = anchor_clutter(route_cells, anchor_idx)
            context_features = urban_context.lookup(anchor_cell) if urban_context else UrbanContextFeatures()
            clutter = min(
                4.0,
                route_clutter
                + context_features.urban_clutter_index
                + max(0.0, 1.0 - context_features.sidewalk_access_score)
                + 0.5 * context_features.building_height_proxy
                + 0.25 * context_features.street_complexity,
            )
            observability_score = compute_observability_score(
                local_straightness=straightness,
                turn_severity=turn,
                ambiguity_count=ambiguity_count,
                anchor_clutter=clutter,
                weights=observability_weights,
            )
            if context_features.is_imputed:
                observability_score *= 0.9
            success_probability = pickup_success_probability(
                observability_score,
                occlusion_lambda=config.occlusion_lambda,
            )
            opportunities.append(
                RendezvousOpportunity(
                    rider_id=int(rider_id),
                    anchor_cell=anchor_cell,
                    anchor_idx=anchor_idx,
                    pickup_h3=str(row["pickup_h3"]),
                    dropoff_h3=str(row["dropoff_h3"]),
                    fare_share=float(row["fare_amount"]) * driver.platform_share,
                    passenger_count=int(row["passenger_count"]),
                    walk_m=walk_m,
                    walk_min=walk_min,
                    anchor_progress=route_progress(anchor_idx, route_cells),
                    travel_fraction=travel_fraction,
                    ambiguity_count=ambiguity_count,
                    local_straightness=straightness,
                    turn_severity=turn,
                    anchor_clutter=clutter,
                    urban_clutter_index=context_features.urban_clutter_index,
                    sidewalk_access_score=context_features.sidewalk_access_score,
                    building_height_proxy=context_features.building_height_proxy,
                    context_is_imputed=context_features.is_imputed,
                    observability_score=observability_score,
                    success_probability=success_probability,
                )
            )

    return opportunities


def _build_corridor_policy_outcome(
    driver: DriverTrip,
    route_eval: RouteOpportunityEvaluation,
    *,
    config: RendezvousConfig,
    score: float,
    seed: int,
) -> PolicyOutcome:
    selected_candidates = _select_corridor_candidates(route_eval.candidate_riders, seats=driver.seats)
    opportunities_by_rider = _opportunities_by_rider(route_eval.opportunities)
    rng = np.random.default_rng(seed + driver.driver_id + route_eval.route_idx)

    realized_revenue = 0.0
    expected_revenue = 0.0
    nominal_revenue = 0.0
    successful_rider_ids: list[int] = []
    attempted_rider_ids: list[int] = []
    observability_values: list[float] = []
    walk_values: list[float] = []

    for candidate in selected_candidates:
        attempted_rider_ids.append(candidate.rider_id)
        nominal_revenue += candidate.fare_share
        chosen_anchor = _corridor_anchor_choice(opportunities_by_rider.get(candidate.rider_id, ()))
        if chosen_anchor is None:
            observability_values.append(0.0)
            walk_values.append(config.max_walk_min)
            continue

        expected_revenue += chosen_anchor.observable_value
        observability_values.append(chosen_anchor.observability_score)
        walk_values.append(chosen_anchor.walk_min)
        if rng.random() <= chosen_anchor.success_probability:
            realized_revenue += chosen_anchor.fare_share
            successful_rider_ids.append(candidate.rider_id)

    mean_observability = float(np.mean(observability_values)) if observability_values else 0.0
    mean_walk_min = float(np.mean(walk_values)) if walk_values else 0.0
    return PolicyOutcome(
        policy="corridor_only",
        route_idx=route_eval.route_idx,
        score=float(score),
        route_distance_miles=route_eval.route.distance_m / METERS_PER_MILE,
        route_cost=route_eval.route_cost,
        expected_value=expected_revenue - route_eval.route_cost,
        nominal_revenue=nominal_revenue,
        realized_revenue=realized_revenue,
        actual_profit=realized_revenue - route_eval.route_cost,
        successful_riders=len(successful_rider_ids),
        attempted_riders=len(selected_candidates),
        candidate_count=route_eval.candidate_count,
        time_eligible_candidate_count=route_eval.time_eligible_candidate_count,
        feasible_opportunity_count=route_eval.feasible_opportunity_count,
        observable_opportunity_count=route_eval.observable_opportunity_count,
        mean_observability=mean_observability,
        mean_walk_min=mean_walk_min,
        nominal_realized_gap=nominal_revenue - realized_revenue,
        selected_rider_ids=tuple(attempted_rider_ids),
        successful_rider_ids=tuple(successful_rider_ids),
    )


def _build_policy_outcome(
    policy: str,
    driver: DriverTrip,
    route_eval: RouteOpportunityEvaluation,
    *,
    score: float,
    selector: MeetingPointSelector,
    seed: int,
) -> PolicyOutcome:
    selected = selector.select(route_eval.opportunities, seats=driver.seats)
    rng = np.random.default_rng(seed + driver.driver_id + route_eval.route_idx)
    realized_revenue = 0.0
    successful_rider_ids: list[int] = []
    for opportunity in selected:
        if rng.random() <= opportunity.success_probability:
            realized_revenue += opportunity.fare_share
            successful_rider_ids.append(opportunity.rider_id)

    nominal_revenue = float(sum(opportunity.nominal_value for opportunity in selected))
    expected_revenue = float(sum(selector.opportunity_value(opportunity) for opportunity in selected))
    mean_observability = (
        float(np.mean([opportunity.observability_score for opportunity in selected])) if selected else 0.0
    )
    mean_walk_min = float(np.mean([opportunity.walk_min for opportunity in selected])) if selected else 0.0
    return PolicyOutcome(
        policy=policy,
        route_idx=route_eval.route_idx,
        score=float(score),
        route_distance_miles=route_eval.route.distance_m / METERS_PER_MILE,
        route_cost=route_eval.route_cost,
        expected_value=expected_revenue - route_eval.route_cost,
        nominal_revenue=nominal_revenue,
        realized_revenue=realized_revenue,
        actual_profit=realized_revenue - route_eval.route_cost,
        successful_riders=len(successful_rider_ids),
        attempted_riders=len(selected),
        candidate_count=route_eval.candidate_count,
        time_eligible_candidate_count=route_eval.time_eligible_candidate_count,
        feasible_opportunity_count=route_eval.feasible_opportunity_count,
        observable_opportunity_count=route_eval.observable_opportunity_count,
        mean_observability=mean_observability,
        mean_walk_min=mean_walk_min,
        nominal_realized_gap=nominal_revenue - realized_revenue,
        selected_rider_ids=tuple(opportunity.rider_id for opportunity in selected),
        successful_rider_ids=tuple(successful_rider_ids),
    )


def _select_corridor_candidates(
    candidates: tuple[CorridorCandidate, ...],
    *,
    seats: int,
) -> list[CorridorCandidate]:
    ranked = sorted(
        candidates,
        key=lambda item: (item.fare_share, -item.passenger_count, -item.rider_id),
        reverse=True,
    )
    if seats <= 0 or not ranked:
        return []

    best_by_capacity: dict[int, tuple[float, tuple[CorridorCandidate, ...]]] = {0: (0.0, tuple())}
    for candidate in ranked:
        weight = max(int(candidate.passenger_count), 0)
        if weight <= 0 or weight > seats:
            continue
        for used_capacity, (used_value, used_selection) in sorted(best_by_capacity.items(), reverse=True):
            new_capacity = used_capacity + weight
            if new_capacity > seats:
                continue
            new_value = used_value + candidate.fare_share
            new_selection = used_selection + (candidate,)
            current = best_by_capacity.get(new_capacity)
            if current is None or _candidate_selection_key(new_value, new_selection) > _candidate_selection_key(*current):
                best_by_capacity[new_capacity] = (new_value, new_selection)

    return list(max(best_by_capacity.values(), key=lambda item: _candidate_selection_key(item[0], item[1]))[1])


def _candidate_selection_key(
    total_value: float,
    selection: tuple[CorridorCandidate, ...],
) -> tuple[float, int, tuple[int, ...]]:
    rider_signature = tuple(sorted((item.rider_id for item in selection), reverse=True))
    return (round(float(total_value), 12), len(selection), rider_signature)


def _opportunities_by_rider(
    opportunities: tuple[RendezvousOpportunity, ...],
) -> dict[int, tuple[RendezvousOpportunity, ...]]:
    grouped: dict[int, list[RendezvousOpportunity]] = {}
    for opportunity in opportunities:
        grouped.setdefault(opportunity.rider_id, []).append(opportunity)
    return {rider_id: tuple(items) for rider_id, items in grouped.items()}


def _corridor_anchor_choice(
    opportunities: tuple[RendezvousOpportunity, ...],
) -> RendezvousOpportunity | None:
    if not opportunities:
        return None
    return min(
        opportunities,
        key=lambda item: (item.walk_min, -item.travel_fraction, item.anchor_idx),
    )


def _choose_route(route_evaluations: list[RouteOpportunityEvaluation], metric) -> int:
    best = max(route_evaluations, key=lambda row: (metric(row), -row.route_idx))
    return best.route_idx


def _selector_route_value(
    selector: MeetingPointSelector,
    opportunities: tuple[RendezvousOpportunity, ...],
    *,
    seats: int,
) -> float:
    return float(sum(selector.opportunity_value(opportunity) for opportunity in selector.select(opportunities, seats=seats)))


def _make_route_line(polyline) -> LineString:
    return LineString([(lng * COS_LAT, lat) for lat, lng in polyline])
