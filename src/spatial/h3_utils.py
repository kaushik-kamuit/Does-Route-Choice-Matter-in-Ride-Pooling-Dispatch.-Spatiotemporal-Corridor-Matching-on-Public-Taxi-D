"""
H3 spatial primitives used throughout the project.

All corridor-building and spatial matching code imports from here.
No H3 logic should be duplicated elsewhere.
"""

from __future__ import annotations

import math
from typing import Sequence

import h3

LatLng = tuple[float, float]

EARTH_RADIUS_M = 6_371_000


def geo_to_h3(lat: float, lng: float, resolution: int = 9) -> str:
    """Convert a single (lat, lng) point to an H3 cell index."""
    return h3.latlng_to_cell(lat, lng, resolution)


def haversine_m(a: LatLng, b: LatLng) -> float:
    """Great-circle distance in meters between two (lat, lng) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _interpolate_point(a: LatLng, b: LatLng, frac: float) -> LatLng:
    """Linear interpolation between two points. Good enough for short segments."""
    return (
        a[0] + (b[0] - a[0]) * frac,
        a[1] + (b[1] - a[1]) * frac,
    )


def densify_polyline(points: Sequence[LatLng], step_m: float = 200) -> list[LatLng]:
    """
    Insert intermediate points along a polyline so that consecutive
    points are no more than step_m meters apart.
    """
    if len(points) < 2:
        return list(points)

    result = [points[0]]
    for i in range(1, len(points)):
        seg_dist = haversine_m(points[i - 1], points[i])
        if seg_dist <= step_m:
            result.append(points[i])
            continue

        n_segments = math.ceil(seg_dist / step_m)
        for j in range(1, n_segments + 1):
            frac = j / n_segments
            result.append(_interpolate_point(points[i - 1], points[i], frac))
    return result


def polyline_to_h3_cells(
    polyline: Sequence[LatLng],
    resolution: int = 9,
    step_m: float = 200,
) -> list[str]:
    """
    Convert a polyline to an ordered list of unique H3 cells.

    1. Densify the polyline (interpolate every step_m meters).
    2. Map each point to its H3 cell.
    3. Deduplicate while preserving order (route direction).
    """
    dense = densify_polyline(polyline, step_m=step_m)
    seen: set[str] = set()
    ordered: list[str] = []
    for lat, lng in dense:
        cell = h3.latlng_to_cell(lat, lng, resolution)
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    return ordered


def expand_corridor(cells: Sequence[str], k: int = 1) -> set[str]:
    """
    Expand a set of H3 cells by k rings.

    k=0 returns the original cells.
    k=1 adds all immediate neighbors (hex ring 1).
    k=2 adds neighbors of neighbors, etc.
    """
    expanded: set[str] = set()
    for cell in cells:
        expanded.update(h3.grid_disk(cell, k))
    return expanded
