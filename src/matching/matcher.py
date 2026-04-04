"""
Shared matching logic used by both cold-start and warm-up pipelines.

Two-stage filter:
  Stage 1 (spatial):  RiderIndex.find_in_corridor -- pickup AND dropoff in
                      corridor, hour within window.
  Stage 2 (feasibility): Shapely line projection for directionality and
                          detour budget, plus seat capacity.

Greedy matching: feasible riders sorted by fare descending (with seed-based
tie-breaking); seats filled until capacity.

Performance: Shapely's line_locate_point and distance are backed by GEOS (C
library) and exposed as numpy ufuncs, so the entire Stage 2 runs in
vectorized C with no Python-level loops over polyline points.
"""

from __future__ import annotations

from datetime import datetime
from math import cos, radians
from typing import Sequence

import numpy as np
import pandas as pd
import shapely
from shapely import LineString

from spatial.h3_utils import LatLng
from spatial.corridor import Corridor
from matching.rider_index import RiderIndex

PLATFORM_SHARE = 0.50
COST_PER_MILE = 0.67
METERS_PER_MILE = 1609.34
URBAN_SPEED_MPS = 40_000 / 3600  # 40 km/h in m/s
MANHATTAN_FACTOR = 1.3
MIN_TRAVEL_FRACTION = 0.05

NYC_LAT = 40.7
COS_LAT = cos(radians(NYC_LAT))
DEG_TO_M = 111_320.0


def _make_route_line(polyline: Sequence[LatLng]) -> LineString:
    """Convert a (lat, lng) polyline to a Shapely LineString in
    locally-equalized Cartesian space (lng scaled by cos(lat))."""
    return LineString([(lng * COS_LAT, lat) for lat, lng in polyline])


def match_riders(
    corridor: Corridor,
    polyline: Sequence[LatLng],
    rider_index: RiderIndex,
    minute_of_day: int,
    query_datetime: datetime | pd.Timestamp | None = None,
    seats: int = 3,
    max_detour_min: float = 4.0,
    candidate_window_bins: int = 1,
    max_request_offset_min: int | None = None,
    platform_share: float = PLATFORM_SHARE,
    urban_speed_kmh: float = 40.0,
    seed: int = 0,
    candidates: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Match riders to a driver's corridor.

    Args:
        minute_of_day: Driver's departure as minute of day (0-1439).
        candidates: Pre-fetched corridor riders. If None, queries rider_index.

    Returns (matched, feasible) where each is a list of dicts with keys:
        rider_idx, fare_share, detour_minutes, passenger_count
    matched: riders actually assigned seats (greedy, fare-descending)
    feasible: all riders passing filters (before seat cap)
    """
    if candidates is None:
        candidates = rider_index.find_in_corridor(
            corridor.corridor_cells,
            minute_of_day,
            window_bins=candidate_window_bins,
            max_request_offset_min=max_request_offset_min,
            query_datetime=query_datetime,
        )
    if candidates.empty:
        return [], []

    route_line = _make_route_line(polyline)
    if route_line.length < 1e-9:
        return [], []

    pu_pts = shapely.points(
        candidates["pickup_lng"].values * COS_LAT,
        candidates["pickup_lat"].values,
    )
    do_pts = shapely.points(
        candidates["dropoff_lng"].values * COS_LAT,
        candidates["dropoff_lat"].values,
    )

    pu_fracs = shapely.line_locate_point(route_line, pu_pts, normalized=True)
    do_fracs = shapely.line_locate_point(route_line, do_pts, normalized=True)
    pu_dists_m = shapely.distance(route_line, pu_pts) * DEG_TO_M
    do_dists_m = shapely.distance(route_line, do_pts) * DEG_TO_M

    travel_frac = do_fracs - pu_fracs
    detour_m = 2 * (pu_dists_m + do_dists_m) * MANHATTAN_FACTOR
    urban_speed_mps = urban_speed_kmh * 1000.0 / 3600.0
    detour_min = detour_m / urban_speed_mps / 60.0

    pax = candidates["passenger_count"].values.astype(int)
    fares = candidates["fare_amount"].values

    mask = (
        (travel_frac >= MIN_TRAVEL_FRACTION)
        & (detour_min <= max_detour_min)
        & (pax <= seats)
    )

    indices = candidates.index.values[mask]
    fares_pass = fares[mask]
    detour_pass = detour_min[mask]
    pax_pass = pax[mask]

    n_pass = len(indices)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.01, 0.01, size=n_pass) if n_pass > 0 else np.empty(0)
    fare_shares = fares_pass * platform_share

    feasible: list[dict] = []
    for j in range(n_pass):
        feasible.append({
            "rider_idx": int(indices[j]),
            "fare_share": float(fare_shares[j]),
            "detour_minutes": float(detour_pass[j]),
            "passenger_count": int(pax_pass[j]),
            "sort_key": float(fare_shares[j]) + noise[j],
        })

    feasible.sort(key=lambda r: r["sort_key"], reverse=True)

    matched: list[dict] = []
    remaining_seats = seats
    for rider in feasible:
        if rider["passenger_count"] > remaining_seats:
            continue
        remaining_seats -= rider["passenger_count"]
        matched.append(rider)
        if remaining_seats <= 0:
            break

    return matched, feasible
