from __future__ import annotations

import sys
import unittest
from pathlib import Path

import h3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.meeting_points import build_route_anchor_cells, candidate_anchor_indices
from spatial.router import RouteInfo


class MeetingPointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = RouteInfo(
            polyline=((40.7500, -73.9900), (40.7600, -73.9800), (40.7700, -73.9700)),
            distance_m=2500.0,
            duration_s=600.0,
        )

    def test_anchor_generation_is_deterministic(self) -> None:
        first = build_route_anchor_cells(self.route, resolution=9, densify_step_m=80.0)
        second = build_route_anchor_cells(self.route, resolution=9, densify_step_m=80.0)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 0)

    def test_no_common_anchor_returns_empty_indices(self) -> None:
        route_cells = build_route_anchor_cells(self.route, resolution=9, densify_step_m=80.0)
        far_pickup_h3 = h3.latlng_to_cell(40.7000, -74.0500, 9)
        self.assertEqual(candidate_anchor_indices(route_cells, far_pickup_h3, meeting_k_ring=0), [])

    def test_candidate_anchors_expand_monotonically(self) -> None:
        route_cells = build_route_anchor_cells(self.route, resolution=9, densify_step_m=80.0)
        pickup_h3 = h3.latlng_to_cell(40.7550, -73.9850, 9)
        tight = candidate_anchor_indices(route_cells, pickup_h3, meeting_k_ring=0)
        wide = candidate_anchor_indices(route_cells, pickup_h3, meeting_k_ring=2)
        self.assertLessEqual(len(tight), len(wide))


if __name__ == "__main__":
    unittest.main()
