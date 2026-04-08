from __future__ import annotations

import math
from functools import lru_cache

import h3

from spatial.h3_utils import LatLng, haversine_m, polyline_to_h3_cells
from spatial.router import RouteInfo


def build_route_anchor_cells(
    route: RouteInfo,
    *,
    resolution: int,
    densify_step_m: float,
) -> tuple[str, ...]:
    return tuple(
        polyline_to_h3_cells(
            route.polyline,
            resolution=resolution,
            step_m=densify_step_m,
        )
    )


def candidate_anchor_indices(
    route_cells: tuple[str, ...],
    pickup_h3: str,
    *,
    meeting_k_ring: int,
) -> list[int]:
    indices: list[int] = []
    for idx, cell in enumerate(route_cells):
        try:
            distance = h3.grid_distance(cell, pickup_h3)
        except Exception:
            continue
        if distance <= meeting_k_ring:
            indices.append(idx)
    return indices


@lru_cache(maxsize=20_000)
def cell_center(cell: str) -> LatLng:
    lat, lng = h3.cell_to_latlng(cell)
    return (float(lat), float(lng))


def route_progress(anchor_idx: int, route_cells: tuple[str, ...]) -> float:
    if len(route_cells) <= 1:
        return 0.0
    return anchor_idx / float(len(route_cells) - 1)


def local_straightness(route_cells: tuple[str, ...], anchor_idx: int, radius: int = 2) -> float:
    start = max(0, anchor_idx - radius)
    end = min(len(route_cells) - 1, anchor_idx + radius)
    segment = [cell_center(cell) for cell in route_cells[start : end + 1]]
    if len(segment) < 2:
        return 1.0
    path_length = 0.0
    for idx in range(1, len(segment)):
        path_length += haversine_m(segment[idx - 1], segment[idx])
    if path_length <= 1e-9:
        return 1.0
    direct = haversine_m(segment[0], segment[-1])
    return max(0.0, min(1.0, direct / path_length))


def turn_severity(route_cells: tuple[str, ...], anchor_idx: int) -> float:
    if anchor_idx <= 0 or anchor_idx >= len(route_cells) - 1:
        return 0.0
    prev_pt = cell_center(route_cells[anchor_idx - 1])
    cur_pt = cell_center(route_cells[anchor_idx])
    next_pt = cell_center(route_cells[anchor_idx + 1])
    incoming = _bearing(prev_pt, cur_pt)
    outgoing = _bearing(cur_pt, next_pt)
    diff = abs(outgoing - incoming) % 360.0
    diff = min(diff, 360.0 - diff)
    return max(0.0, min(1.0, diff / 180.0))


def anchor_clutter(route_cells: tuple[str, ...], anchor_idx: int, ring_size: int = 1) -> float:
    if not route_cells:
        return 0.0
    anchor = route_cells[anchor_idx]
    neighborhood = set(h3.grid_disk(anchor, ring_size))
    if not neighborhood:
        return 0.0
    overlap = sum(1 for cell in route_cells if cell in neighborhood)
    return overlap / float(len(neighborhood))


def rider_walk_m(pickup_lat: float, pickup_lng: float, anchor_cell: str) -> float:
    return haversine_m((pickup_lat, pickup_lng), cell_center(anchor_cell))


def _bearing(a: LatLng, b: LatLng) -> float:
    lat1, lng1 = math.radians(a[0]), math.radians(a[1])
    lat2, lng2 = math.radians(b[0]), math.radians(b[1])
    dlng = lng2 - lng1
    x = math.sin(dlng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
