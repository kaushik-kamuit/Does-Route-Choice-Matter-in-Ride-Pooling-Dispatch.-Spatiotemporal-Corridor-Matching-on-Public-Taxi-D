from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation.baselines import _proxy_profit_score
from simulation.data_types import DriverTrip
from spatial.corridor import Corridor
from spatial.router import RouteInfo


class BaselineProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = RouteInfo(
            polyline=((40.75, -73.99), (40.76, -73.98)),
            distance_m=1609.34,
            duration_s=420.0,
        )
        self.corridor = Corridor(
            route_cells=("a", "b"),
            corridor_cells=frozenset({"a", "b", "c"}),
            resolution=9,
            buffer_rings=1,
            route_length_m=1609.34,
        )
        self.candidates = pd.DataFrame({"fare_amount": [10.0, 20.0, 30.0]})

    def _driver(self, *, platform_share: float = 0.50, cost_per_mile: float = 0.67) -> DriverTrip:
        return DriverTrip(
            driver_id=1,
            origin=(40.75, -73.99),
            destination=(40.76, -73.98),
            departure_time=datetime(2015, 4, 1, 10, 0, 0),
            hour=10,
            minute_of_day=600,
            trip_distance_miles=1.0,
            platform_share=platform_share,
            cost_per_mile=cost_per_mile,
        )

    def test_profit_proxy_increases_with_platform_share(self) -> None:
        low_share = _proxy_profit_score(self._driver(platform_share=0.40), self.route, self.candidates, self.corridor)
        high_share = _proxy_profit_score(self._driver(platform_share=0.60), self.route, self.candidates, self.corridor)
        self.assertGreater(high_share, low_share)

    def test_profit_proxy_decreases_with_higher_cost_per_mile(self) -> None:
        low_cost = _proxy_profit_score(self._driver(cost_per_mile=0.50), self.route, self.candidates, self.corridor)
        high_cost = _proxy_profit_score(self._driver(cost_per_mile=0.85), self.route, self.candidates, self.corridor)
        self.assertLess(high_cost, low_cost)

    def test_empty_candidate_proxy_reduces_to_negative_route_cost(self) -> None:
        score = _proxy_profit_score(self._driver(cost_per_mile=0.67), self.route, self.candidates.iloc[0:0], self.corridor)
        self.assertAlmostEqual(score, -0.67, places=6)


if __name__ == "__main__":
    unittest.main()
