"""
Corridor building on top of H3 primitives.

A Corridor represents the spatial footprint of a driver's route:
  - route_cells:    H3 cells that the polyline passes directly through.
  - corridor_cells: route_cells expanded by buffer_rings (the matchable zone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .h3_utils import (
    LatLng,
    densify_polyline,
    expand_corridor,
    haversine_m,
    polyline_to_h3_cells,
)


@dataclass(frozen=True, slots=True)
class Corridor:
    route_cells: tuple[str, ...]
    corridor_cells: frozenset[str]
    resolution: int
    buffer_rings: int
    route_length_m: float

    def contains_cell(self, cell: str) -> bool:
        return cell in self.corridor_cells

    def overlap(self, other_cells: set[str]) -> set[str]:
        """Return cells present in both this corridor and the given set."""
        return self.corridor_cells & other_cells

    @property
    def n_route_cells(self) -> int:
        return len(self.route_cells)

    @property
    def n_corridor_cells(self) -> int:
        return len(self.corridor_cells)


def build_corridor(
    polyline: Sequence[LatLng],
    resolution: int = 9,
    buffer_rings: int = 1,
    densify_step_m: float = 80,
) -> Corridor:
    """
    Build a Corridor from a polyline (list of lat/lng points).

    Args:
        polyline: Ordered (lat, lng) waypoints describing the route.
        resolution: H3 resolution (9 = ~174 m hex edge, ~520 m corridor width with k=1).
        buffer_rings: Number of H3 k-rings to expand around the route.
        densify_step_m: Max distance between interpolated points (meters).

    Returns:
        Corridor with route cells and expanded corridor cells.
    """
    route_cells = polyline_to_h3_cells(
        polyline, resolution=resolution, step_m=densify_step_m
    )
    corridor_cells = expand_corridor(route_cells, k=buffer_rings)

    route_length = _polyline_length_m(polyline)

    return Corridor(
        route_cells=tuple(route_cells),
        corridor_cells=frozenset(corridor_cells),
        resolution=resolution,
        buffer_rings=buffer_rings,
        route_length_m=route_length,
    )


def build_straight_line_corridor(
    origin: LatLng,
    destination: LatLng,
    resolution: int = 9,
    buffer_rings: int = 1,
    densify_step_m: float = 80,
) -> Corridor:
    """
    Build a corridor from a straight line between origin and destination.

    Used as the --no-api fallback when OSRM routes are unavailable.
    """
    polyline = [origin, destination]
    return build_corridor(
        polyline,
        resolution=resolution,
        buffer_rings=buffer_rings,
        densify_step_m=densify_step_m,
    )


def _polyline_length_m(points: Sequence[LatLng]) -> float:
    """Total length of a polyline in meters."""
    total = 0.0
    for i in range(1, len(points)):
        total += haversine_m(points[i - 1], points[i])
    return total
