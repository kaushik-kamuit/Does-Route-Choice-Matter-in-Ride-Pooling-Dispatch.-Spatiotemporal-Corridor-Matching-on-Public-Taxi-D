from __future__ import annotations

import sys
import unittest
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matching.rider_index import RiderIndex
from simulation.path_buffer import path_buffer_candidate_count
from simulation.route_evaluator import evaluate_driver_policies
from simulation.domain_io import build_driver_trips
from spatial.router import RouteInfo


class FakePredictor:
    def rank_routes(self, feature_list):
        indexed = list(enumerate(feature_list))
        indexed.sort(key=lambda item: item[1]["corridor_rider_count"], reverse=True)
        return [(idx, float(row["corridor_rider_count"])) for idx, row in indexed]


def _route(polyline, distance_m=2500.0) -> RouteInfo:
    return RouteInfo(polyline=tuple(polyline), distance_m=distance_m, duration_s=600.0)


def _rider(ts: str, pickup, dropoff, fare=12.0) -> dict:
    return {
        "split": "test",
        "pickup_datetime": pd.Timestamp(ts),
        "pickup_h3": h3.latlng_to_cell(pickup[0], pickup[1], 9),
        "dropoff_h3": h3.latlng_to_cell(dropoff[0], dropoff[1], 9),
        "pickup_lat": pickup[0],
        "pickup_lng": pickup[1],
        "dropoff_lat": dropoff[0],
        "dropoff_lng": dropoff[1],
        "passenger_count": 1,
        "fare_amount": fare,
    }


def _driver(ts: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "test",
                "pickup_datetime": pd.Timestamp(ts),
                "origin_lat": 40.7500,
                "origin_lng": -73.9900,
                "dest_lat": 40.7700,
                "dest_lng": -73.9700,
                "hour_of_day": pd.Timestamp(ts).hour,
                "day_of_week": pd.Timestamp(ts).dayofweek,
                "is_weekend": int(pd.Timestamp(ts).dayofweek >= 5),
                "trip_distance_miles": 12.0,
            }
        ]
    )


class PathBufferBaselineTests(unittest.TestCase):
    def test_path_buffer_count_prefers_route_with_closer_riders(self) -> None:
        route_a = _route(
            [
                (40.7500, -73.9900),
                (40.7600, -73.9800),
                (40.7700, -73.9700),
            ]
        )
        route_b = _route(
            [
                (40.7500, -73.9400),
                (40.7600, -73.9300),
                (40.7700, -73.9200),
            ]
        )
        riders = pd.DataFrame(
            [
                _rider(
                    "2015-04-01T10:00:00",
                    pickup=(40.7540, -73.9860),
                    dropoff=(40.7660, -73.9740),
                ),
                _rider(
                    "2015-04-01T10:00:30",
                    pickup=(40.7560, -73.9840),
                    dropoff=(40.7680, -73.9720),
                ),
                _rider(
                    "2015-04-01T10:00:00",
                    pickup=(40.7540, -73.9360),
                    dropoff=(40.7660, -73.9240),
                ),
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        count_a = path_buffer_candidate_count(
            rider_index,
            route_a.polyline,
            10 * 60,
            max_request_offset_min=5,
            query_datetime=pd.Timestamp("2015-04-01T10:00:10"),
        )
        count_b = path_buffer_candidate_count(
            rider_index,
            route_b.polyline,
            10 * 60,
            max_request_offset_min=5,
            query_datetime=pd.Timestamp("2015-04-01T10:00:10"),
        )
        self.assertGreater(count_a, count_b)

    def test_route_evaluator_exposes_path_buffer_policy(self) -> None:
        routes = [
            _route(
                [
                    (40.7500, -73.9900),
                    (40.7600, -73.9800),
                    (40.7700, -73.9700),
                ]
            ),
            _route(
                [
                    (40.7500, -73.9400),
                    (40.7600, -73.9300),
                    (40.7700, -73.9200),
                ]
            ),
        ]
        riders = pd.DataFrame(
            [
                _rider(
                    "2015-04-01T10:00:00",
                    pickup=(40.7540, -73.9860),
                    dropoff=(40.7660, -73.9740),
                ),
                _rider(
                    "2015-04-01T10:00:30",
                    pickup=(40.7560, -73.9840),
                    dropoff=(40.7680, -73.9720),
                ),
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        drivers = _driver("2015-04-01T10:00:10")
        trip = build_driver_trips(
            drivers,
            seats=3,
            max_detour_min=4.0,
            platform_share=0.50,
            cost_per_mile=0.67,
            urban_speed_kmh=40.0,
        )[0]
        evaluation = evaluate_driver_policies(
            trip,
            rider_index,
            FakePredictor(),
            {},
            day_of_week=int(drivers.iloc[0]["day_of_week"]),
            is_weekend=int(drivers.iloc[0]["is_weekend"]),
            day_of_month=int(drivers.iloc[0]["pickup_datetime"].day),
            seed=42,
            candidate_window_bins=1,
            max_request_offset_min=5,
            h3_resolution=9,
            corridor_k_ring=1,
            corridor_densify_step_m=80.0,
            available_rider_ids=set(riders.index.tolist()),
            routes=routes,
        )
        self.assertIn("heuristic_path_buffer", evaluation.plans)
        self.assertEqual(evaluation.plans["heuristic_path_buffer"].route_idx, 0)


if __name__ == "__main__":
    unittest.main()
