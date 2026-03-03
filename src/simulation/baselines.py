"""
Baseline simulation strategies for comparison against ML warm-up.

Strategies:
  - Oracle:    Runs match_riders on ALL routes, picks the best actual outcome.
               Upper bound on what any model could achieve.
  - Random:    Picks uniformly at random among available routes.
  - Heuristic: Picks the route with the highest corridor_rider_count (no ML).
"""

from __future__ import annotations

import time

import numpy as np

from matching.matcher import match_riders, COST_PER_MILE, METERS_PER_MILE
from matching.rider_index import RiderIndex
from spatial.corridor import Corridor, build_corridor
from spatial.router import OSRMRouter, RouteInfo

from .data_types import DriverTrip, DriverOutcome


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
    driving_cost = distance_miles * COST_PER_MILE
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


def run_oracle(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Run match_riders on ALL routes, return the outcome with highest profit."""
    t0 = time.perf_counter()
    best_outcome = None
    best_profit = -float("inf")

    for idx, (route, corridor) in enumerate(zip(routes, corridors)):
        matched, _ = match_riders(
            corridor, route.polyline, rider_index,
            minute_of_day=driver.minute_of_day,
            seats=driver.seats, max_detour_min=driver.max_detour_minutes,
            seed=seed,
        )
        revenue = sum(m["fare_share"] for m in matched)
        cost = (route.distance_m / METERS_PER_MILE) * COST_PER_MILE
        profit = revenue - cost

        if profit > best_profit:
            best_profit = profit
            best_outcome = (idx, route, matched)

    elapsed = time.perf_counter() - t0
    if best_outcome is None:
        return None
    idx, route, matched = best_outcome
    return _outcome_from_match(driver, route, matched, "oracle", idx, elapsed, seed)


def run_random(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick a random route, match riders on it."""
    if not routes:
        return None
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + driver.driver_id)
    idx = int(rng.integers(0, len(routes)))
    route = routes[idx]
    corridor = corridors[idx]

    matched, _ = match_riders(
        corridor, route.polyline, rider_index,
        minute_of_day=driver.minute_of_day,
        seats=driver.seats, max_detour_min=driver.max_detour_minutes,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return _outcome_from_match(driver, route, matched, "random", idx, elapsed, seed)


def run_heuristic(
    driver: DriverTrip,
    rider_index: RiderIndex,
    seed: int,
    *,
    routes: list[RouteInfo],
    corridors: list[Corridor],
) -> DriverOutcome | None:
    """Pick the route whose corridor contains the most riders (no ML)."""
    if not routes:
        return None
    t0 = time.perf_counter()
    best_idx = 0
    best_count = -1

    for idx, corridor in enumerate(corridors):
        candidates = rider_index.find_in_corridor(
            corridor.corridor_cells, driver.minute_of_day
        )
        count = len(candidates)
        if count > best_count:
            best_count = count
            best_idx = idx

    route = routes[best_idx]
    corridor = corridors[best_idx]
    matched, _ = match_riders(
        corridor, route.polyline, rider_index,
        minute_of_day=driver.minute_of_day,
        seats=driver.seats, max_detour_min=driver.max_detour_minutes,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return _outcome_from_match(
        driver, route, matched, "heuristic", best_idx, elapsed, seed
    )
