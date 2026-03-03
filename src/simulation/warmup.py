"""
Warm-up simulation pipeline.

The driver provides origin + destination. The system fetches up to 3
alternative routes (same alt=3 request as cold-start for fairness),
builds H3 corridors along each, uses the trained ML model to predict
expected profit per route, ranks them, and selects the top route.
Riders are then matched along the chosen corridor.
"""

from __future__ import annotations

import math
import time

import h3
import numpy as np

from matching.matcher import match_riders, COST_PER_MILE, METERS_PER_MILE
from matching.rider_index import RiderIndex
from models.predict import ProfitPredictor
from spatial.corridor import Corridor, build_corridor
from spatial.h3_utils import haversine_m
from spatial.router import OSRMRouter, RouteInfo

from .data_types import DriverTrip, DriverOutcome

LANDMARKS = {
    "jfk":         (40.6413, -73.7781),
    "lga":         (40.7769, -73.8740),
    "penn":        (40.7505, -73.9935),
    "times_sq":    (40.7580, -73.9855),
    "grand_cntrl": (40.7527, -73.9772),
    "world_trade": (40.7127, -74.0134),
}


def _landmark_dists(lat: float, lng: float) -> tuple[float, dict[str, float]]:
    dists = {}
    for name, (llat, llng) in LANDMARKS.items():
        dists[name] = haversine_m((lat, lng), (llat, llng)) / 1000.0
    return min(dists.values()), dists


def _bearing_deg(lat1, lng1, lat2, lng2) -> float:
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    x = math.sin(dlng) * math.cos(rlat2)
    y = (math.cos(rlat1) * math.sin(rlat2)
         - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlng))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _corridor_hist(cells, h3_stats_dict: dict) -> dict[str, float]:
    total_pu = total_do = 0
    fare_sum = 0.0
    fare_vals = []
    for cell in cells:
        row = h3_stats_dict.get(cell)
        if row is None:
            continue
        total_pu += row["pickup_count"]
        total_do += row["dropoff_count"]
        fare_sum += row["mean_fare"] * row["pickup_count"]
        if row["pickup_count"] > 0:
            fare_vals.append(row["mean_fare"])
    n = max(len(cells), 1)
    return {
        "corridor_hist_pickups": total_pu,
        "corridor_hist_dropoffs": total_do,
        "corridor_hist_pickup_density": total_pu / n,
        "corridor_hist_dropoff_density": total_do / n,
        "corridor_hist_mean_fare": float(np.mean(fare_vals)) if fare_vals else 0.0,
        "corridor_hist_fare_density": fare_sum / n,
    }


def _route_features(
    route: RouteInfo,
    corridor: Corridor,
    rider_index: RiderIndex,
    polyline,
    minute_of_day: int,
    hour: int,
    day_of_week: int,
    is_weekend: int,
    day_of_month: int,
    origin: tuple[float, float],
    destination: tuple[float, float],
    h3_stats_dict: dict,
    seats: int = 3,
    max_detour_min: float = 4.0,
) -> dict[str, float]:
    """Build the feature dict the v2 profit model expects."""
    candidates = rider_index.find_in_corridor(corridor.corridor_cells, minute_of_day)
    n_riders = len(candidates)
    n_cells = corridor.n_corridor_cells
    fares = candidates["fare_amount"].values if n_riders > 0 else np.array([])
    fare_sum = float(fares.sum()) if n_riders > 0 else 0.0

    straight_dist = haversine_m(origin, destination)
    sinuosity = route.distance_m / max(straight_dist, 1.0)
    avg_speed = route.distance_m / max(route.duration_s, 1.0)

    brg = _bearing_deg(origin[0], origin[1], destination[0], destination[1])
    qh = hour * 4 + (minute_of_day % 60) // 15

    o_near, o_dists = _landmark_dists(*origin)
    d_near, d_dists = _landmark_dists(*destination)

    ch = _corridor_hist(corridor.corridor_cells, h3_stats_dict)

    o_cell = h3.latlng_to_cell(origin[0], origin[1], 9)
    d_cell = h3.latlng_to_cell(destination[0], destination[1], 9)
    o_stats = h3_stats_dict.get(o_cell, {})
    d_stats = h3_stats_dict.get(d_cell, {})

    return {
        "route_distance_m": route.distance_m,
        "route_duration_s": route.duration_s,
        "corridor_cell_count": n_cells,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "corridor_rider_count": n_riders,
        "corridor_demand_density": n_riders / max(n_cells, 1),
        "mean_rider_fare": float(fares.mean()) if n_riders > 0 else 0.0,
        "corridor_fare_density": fare_sum / max(n_cells, 1),
        "day_of_month": day_of_month,
        "time_bin_15min": qh,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "route_sinuosity": sinuosity,
        "route_avg_speed_ms": avg_speed,
        "bearing_sin": math.sin(math.radians(brg)),
        "bearing_cos": math.cos(math.radians(brg)),
        "straight_line_dist_m": straight_dist,
        "origin_landmark_dist_km": o_near,
        "dest_landmark_dist_km": d_near,
        "origin_jfk_km": o_dists["jfk"],
        "origin_lga_km": o_dists["lga"],
        "origin_penn_km": o_dists["penn"],
        "origin_times_sq_km": o_dists["times_sq"],
        "dest_jfk_km": d_dists["jfk"],
        "dest_lga_km": d_dists["lga"],
        "dest_penn_km": d_dists["penn"],
        "dest_times_sq_km": d_dists["times_sq"],
        **ch,
        "origin_cell_pickups": o_stats.get("pickup_count", 0),
        "origin_cell_mean_fare": o_stats.get("mean_fare", 0.0),
        "dest_cell_dropoffs": d_stats.get("dropoff_count", 0),
    }


def run_warmup(
    driver: DriverTrip,
    router: OSRMRouter,
    rider_index: RiderIndex,
    predictor: ProfitPredictor,
    day_of_week: int = 0,
    is_weekend: int = 0,
    day_of_month: int = 1,
    h3_stats_dict: dict | None = None,
    seed: int = 0,
    *,
    routes: list[RouteInfo] | None = None,
    corridors: list[Corridor] | None = None,
    ranking: list[tuple[int, float]] | None = None,
) -> DriverOutcome | None:
    """Execute the warm-up pipeline for a single driver.

    Args:
        routes:    Pre-fetched alternative routes (skips cache lookup).
        corridors: Pre-built corridors for each route.
        ranking:   Pre-computed (route_idx, predicted_profit) ranking.

    Returns None if no routes are available (cache miss).
    """
    t0 = time.perf_counter()

    if routes is None:
        routes = router.get_alternative_routes(
            driver.origin, driver.destination, max_alternatives=3
        )

    if not routes:
        return None

    if corridors is None:
        corridors = [build_corridor(r.polyline) for r in routes]

    if ranking is None:
        if h3_stats_dict is None:
            h3_stats_dict = {}
        feature_list = [
            _route_features(
                r, c, rider_index, r.polyline, driver.minute_of_day,
                driver.hour, day_of_week, is_weekend, day_of_month,
                driver.origin, driver.destination, h3_stats_dict,
                seats=driver.seats, max_detour_min=driver.max_detour_minutes,
            )
            for r, c in zip(routes, corridors)
        ]
        ranking = predictor.rank_routes(feature_list)

    if not ranking:
        return None

    best_idx, predicted_profit = ranking[0]

    best_route = routes[best_idx]
    best_corridor = corridors[best_idx]

    matched, _feasible = match_riders(
        best_corridor,
        best_route.polyline,
        rider_index,
        minute_of_day=driver.minute_of_day,
        seats=driver.seats,
        max_detour_min=driver.max_detour_minutes,
        seed=seed,
    )

    total_revenue = sum(m["fare_share"] for m in matched)
    distance_miles = best_route.distance_m / METERS_PER_MILE
    driving_cost = distance_miles * COST_PER_MILE
    profit = total_revenue - driving_cost

    elapsed = time.perf_counter() - t0

    return DriverOutcome(
        driver_id=driver.driver_id,
        strategy="warmup",
        route_distance_miles=distance_miles,
        matched_riders=len(matched),
        total_revenue=total_revenue,
        driving_cost=driving_cost,
        profit=profit,
        route_rank_chosen=best_idx + 1,
        predicted_profit=predicted_profit,
        compute_time_s=elapsed,
        route_length_category=driver.route_length_category,
        seed=seed,
        hour=driver.hour,
    )
