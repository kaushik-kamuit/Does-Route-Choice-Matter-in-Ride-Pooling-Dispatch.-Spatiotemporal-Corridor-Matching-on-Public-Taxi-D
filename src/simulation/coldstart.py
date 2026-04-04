"""
Cold-start simulation pipeline.

The driver provides origin + destination. The system fetches the default
route (routes[0] from the shared alt=3 request), builds an H3 corridor,
and matches riders along it. No ML, no route selection.
"""

from __future__ import annotations

import time

from matching.matcher import match_riders, METERS_PER_MILE
from matching.rider_index import RiderIndex
from spatial.corridor import Corridor, build_corridor
from spatial.router import OSRMRouter, RouteInfo

from .data_types import DriverTrip, DriverOutcome


def run_coldstart(
    driver: DriverTrip,
    router: OSRMRouter,
    rider_index: RiderIndex,
    seed: int = 0,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    *,
    route: RouteInfo | None = None,
    corridor: Corridor | None = None,
) -> DriverOutcome | None:
    """Execute the cold-start pipeline for a single driver.

    Args:
        route:    Pre-computed default route (skips cache lookup if provided).
        corridor: Pre-computed corridor for the route.

    Returns None if no route is available (cache miss).
    """
    t0 = time.perf_counter()

    if route is None:
        route = router.get_default_route(driver.origin, driver.destination)
    if route is None:
        return None

    if corridor is None:
        corridor = build_corridor(route.polyline)

    matched, _feasible = match_riders(
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
    )

    total_revenue = sum(m["fare_share"] for m in matched)
    distance_miles = route.distance_m / METERS_PER_MILE
    driving_cost = distance_miles * driver.cost_per_mile
    profit = total_revenue - driving_cost

    elapsed = time.perf_counter() - t0

    return DriverOutcome(
        driver_id=driver.driver_id,
        strategy="coldstart",
        route_distance_miles=distance_miles,
        matched_riders=len(matched),
        total_revenue=total_revenue,
        driving_cost=driving_cost,
        profit=profit,
        route_rank_chosen=1,
        predicted_profit=0.0,
        compute_time_s=elapsed,
        route_length_category=driver.route_length_category,
        seed=seed,
        hour=driver.hour,
    )
