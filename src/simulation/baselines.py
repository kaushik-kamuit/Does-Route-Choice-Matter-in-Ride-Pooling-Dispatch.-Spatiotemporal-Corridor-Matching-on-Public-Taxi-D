"""
Baseline simulation strategies for comparison against ML warm-up.

Strategies:
  - Oracle:                   Runs match_riders on all routes, picks the best actual outcome.
  - Random:                   Picks uniformly among the available routes.
  - Heuristic count:          Highest candidate count inside the corridor.
  - Heuristic fare density:   Highest corridor fare density among candidates.
  - Heuristic feasible count: Highest feasible-rider count after exact filtering and geometry.
  - Heuristic profit proxy:   Highest hand-crafted profit proxy using route cost and corridor demand.

The paper-facing "heuristic" baseline is chosen downstream as the strongest
non-ML heuristic in the primary scenario and copied to heuristic_outcomes*.csv.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from matching.matcher import METERS_PER_MILE, match_riders
from matching.rider_index import RiderIndex
from spatial.corridor import Corridor
from spatial.router import RouteInfo

from .data_types import DriverOutcome, DriverTrip

HEURISTIC_STRATEGIES = [
    "heuristic_count",
    "heuristic_fare_density",
    "heuristic_feasible_count",
    "heuristic_profit_proxy",
]


def _outcome_from_match(
    driver: DriverTrip,
    route: RouteInfo,
    matched: list[dict],
    strategy: str,
    route_rank: int,
    elapsed: float,
    seed: int,
) -> DriverOutcome:
    total_revenue = sum(m["fare_share"] for m in matched)
    distance_miles = route.distance_m / METERS_PER_MILE
    driving_cost = distance_miles * driver.cost_per_mile
    profit = total_revenue - driving_cost
    return DriverOutcome(
        driver_id=driver.driver_id,
        strategy=strategy,
        route_distance_miles=distance_miles,
        matched_riders=len(matched),
        total_revenue=total_revenue,
        driving_cost=driving_cost,
        profit=profit,
        route_rank_chosen=route_rank + 1,
        predicted_profit=0.0,
        compute_time_s=elapsed,
        route_length_category=driver.route_length_category,
        seed=seed,
        hour=driver.hour,
    )


def _load_candidates(
    driver: DriverTrip,
    rider_index: RiderIndex,
    corridor: Corridor,
    *,
    candidate_window_bins: int,
    max_request_offset_min: int | None,
) -> pd.DataFrame:
    return rider_index.find_in_corridor(
        corridor.corridor_cells,
        driver.minute_of_day,
        window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        query_datetime=driver.departure_time,
    )


def _proxy_profit_score(
    driver: DriverTrip,
    route: RouteInfo,
    candidates: pd.DataFrame,
    corridor: Corridor,
) -> float:
    if candidates.empty:
        route_cost = (route.distance_m / METERS_PER_MILE) * driver.cost_per_mile
        return -route_cost

    mean_fare = float(candidates["fare_amount"].mean())
    corridor_density = len(candidates) / max(corridor.n_corridor_cells, 1)
    revenue_proxy = driver.platform_share * mean_fare * min(corridor_density, driver.seats)
    route_cost = (route.distance_m / METERS_PER_MILE) * driver.cost_per_mile
    return revenue_proxy - route_cost


def _select_and_match(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    strategy: str,
    best_idx: int,
    *,
    candidate_window_bins: int,
    max_request_offset_min: int | None,
    routes: list[RouteInfo],
    corridors: list[Corridor],
    preloaded_candidates: pd.DataFrame | None = None,
) -> DriverOutcome | None:
    route = routes[best_idx]
    corridor = corridors[best_idx]
    matched, _ = match_riders(
        corridor,
        route.polyline,
        rider_index,
        minute_of_day=driver.minute_of_day,
        query_datetime=driver.departure_time,
        seats=driver.seats,
        max_detour_min=driver.max_detour_minutes,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        platform_share=driver.platform_share,
        urban_speed_kmh=driver.urban_speed_kmh,
        seed=seed,
        candidates=preloaded_candidates,
    )
    return _outcome_from_match(driver, route, matched, strategy, best_idx, 0.0, seed)


def _route_actual_outcome(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    route: RouteInfo,
    corridor: Corridor,
    *,
    candidate_window_bins: int,
    max_request_offset_min: int | None,
    candidates: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict]]:
    return match_riders(
        corridor,
        route.polyline,
        rider_index,
        minute_of_day=driver.minute_of_day,
        query_datetime=driver.departure_time,
        seats=driver.seats,
        max_detour_min=driver.max_detour_minutes,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        platform_share=driver.platform_share,
        urban_speed_kmh=driver.urban_speed_kmh,
        seed=seed,
        candidates=candidates,
    )


def run_oracle(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Run match_riders on all routes and return the highest actual profit."""
    t0 = time.perf_counter()
    best_idx = -1
    best_route = None
    best_matched: list[dict] = []
    best_profit = -float("inf")

    for idx, (route, corridor) in enumerate(zip(routes, corridors)):
        matched, _ = _route_actual_outcome(
            driver,
            rider_index,
            seed,
            route,
            corridor,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
        )
        revenue = sum(m["fare_share"] for m in matched)
        cost = (route.distance_m / METERS_PER_MILE) * driver.cost_per_mile
        profit = revenue - cost
        if profit > best_profit:
            best_idx = idx
            best_route = route
            best_matched = matched
            best_profit = profit

    elapsed = time.perf_counter() - t0
    if best_route is None:
        return None
    return _outcome_from_match(driver, best_route, best_matched, "oracle", best_idx, elapsed, seed)


def run_random(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick a random route and match riders on it."""
    if not routes:
        return None
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + driver.driver_id)
    idx = int(rng.integers(0, len(routes)))
    route = routes[idx]
    corridor = corridors[idx]
    matched, _ = _route_actual_outcome(
        driver,
        rider_index,
        seed,
        route,
        corridor,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
    )
    elapsed = time.perf_counter() - t0
    return _outcome_from_match(driver, route, matched, "random", idx, elapsed, seed)


def run_heuristic_count(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick the route with the largest exact-window candidate count."""
    if not routes:
        return None
    t0 = time.perf_counter()
    best_idx = 0
    best_count = -1
    best_candidates: pd.DataFrame | None = None

    for idx, corridor in enumerate(corridors):
        candidates = _load_candidates(
            driver,
            rider_index,
            corridor,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
        )
        count = len(candidates)
        if count > best_count:
            best_idx = idx
            best_count = count
            best_candidates = candidates

    route = routes[best_idx]
    outcome = _select_and_match(
        driver,
        rider_index,
        seed,
        "heuristic_count",
        best_idx,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        routes=routes,
        corridors=corridors,
        preloaded_candidates=best_candidates,
    )
    if outcome is None:
        return None
    outcome.compute_time_s = time.perf_counter() - t0
    return outcome


def run_heuristic_fare_density(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick the route with the highest corridor fare density."""
    if not routes:
        return None
    t0 = time.perf_counter()
    best_idx = 0
    best_score = -float("inf")
    best_candidates: pd.DataFrame | None = None

    for idx, corridor in enumerate(corridors):
        candidates = _load_candidates(
            driver,
            rider_index,
            corridor,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
        )
        fare_density = (
            float(candidates["fare_amount"].sum()) / max(corridor.n_corridor_cells, 1)
            if not candidates.empty
            else 0.0
        )
        if fare_density > best_score:
            best_idx = idx
            best_score = fare_density
            best_candidates = candidates

    route = routes[best_idx]
    outcome = _select_and_match(
        driver,
        rider_index,
        seed,
        "heuristic_fare_density",
        best_idx,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        routes=routes,
        corridors=corridors,
        preloaded_candidates=best_candidates,
    )
    if outcome is None:
        return None
    outcome.compute_time_s = time.perf_counter() - t0
    return outcome


def run_heuristic_feasible_count(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick the route with the largest feasible rider set before seat fill."""
    if not routes:
        return None
    t0 = time.perf_counter()
    best_idx = 0
    best_score = -1
    best_matched: list[dict] | None = None
    best_route: RouteInfo | None = None

    for idx, (route, corridor) in enumerate(zip(routes, corridors)):
        candidates = _load_candidates(
            driver,
            rider_index,
            corridor,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
        )
        matched, feasible = _route_actual_outcome(
            driver,
            rider_index,
            seed,
            route,
            corridor,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
            candidates=candidates,
        )
        score = len(feasible)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_matched = matched
            best_route = route

    elapsed = time.perf_counter() - t0
    if best_route is None or best_matched is None:
        return None
    return _outcome_from_match(
        driver,
        best_route,
        best_matched,
        "heuristic_feasible_count",
        best_idx,
        elapsed,
        seed,
    )


def run_heuristic_profit_proxy(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick the route with the highest hand-crafted proxy profit."""
    if not routes:
        return None
    t0 = time.perf_counter()
    best_idx = 0
    best_score = -float("inf")
    best_candidates: pd.DataFrame | None = None

    for idx, (route, corridor) in enumerate(zip(routes, corridors)):
        candidates = _load_candidates(
            driver,
            rider_index,
            corridor,
            candidate_window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
        )
        score = _proxy_profit_score(driver, route, candidates, corridor)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_candidates = candidates

    route = routes[best_idx]
    outcome = _select_and_match(
        driver,
        rider_index,
        seed,
        "heuristic_profit_proxy",
        best_idx,
        candidate_window_bins=candidate_window_bins,
        max_request_offset_min=max_request_offset_min,
        routes=routes,
        corridors=corridors,
        preloaded_candidates=best_candidates,
    )
    if outcome is None:
        return None
    outcome.compute_time_s = time.perf_counter() - t0
    return outcome
