from __future__ import annotations

import sys
import unittest
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.domain_config import get_domain_config
from dispatch import DispatchConfig, RollingDispatcher
from simulation.route_evaluator import evaluate_driver_policies
from simulation.domain_io import build_driver_trips
from spatial.router import RouteInfo


class FakePredictor:
    def rank_routes(self, feature_list):
        indexed = list(enumerate(feature_list))
        indexed.sort(key=lambda item: item[1]["corridor_rider_count"], reverse=True)
        return [(idx, float(row["corridor_rider_count"])) for idx, row in indexed]


class FakeRouter:
    def __init__(self, routes):
        self._routes = routes

    def get_alternative_routes(self, origin, destination, max_alternatives=3):
        return self._routes[:max_alternatives]

    def flush_cache(self):
        return None


def _make_route(offset_lng: float = 0.0) -> RouteInfo:
    polyline = (
        (40.7500, -73.9900 + offset_lng),
        (40.7600, -73.9800 + offset_lng),
        (40.7700, -73.9700 + offset_lng),
    )
    return RouteInfo(polyline=polyline, distance_m=2500.0, duration_s=600.0)


def _make_route_with_distance(distance_m: float, offset_lng: float = 0.0) -> RouteInfo:
    polyline = (
        (40.7500, -73.9900 + offset_lng),
        (40.7600, -73.9800 + offset_lng),
        (40.7700, -73.9700 + offset_lng),
    )
    return RouteInfo(polyline=polyline, distance_m=distance_m, duration_s=600.0)


def _rider_row(ts: str, pickup, dropoff, fare: float = 12.0) -> dict:
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


def _driver_row(ts: str) -> dict:
    return {
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


class DispatchSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.routes = [_make_route(0.0), _make_route(0.02), _make_route(-0.02)]
        self.router = FakeRouter(self.routes)
        self.predictor = FakePredictor()
        self.h3_stats_dict = {}
        self.domain_config = get_domain_config("yellow")

    def _make_dispatcher(self, **kwargs) -> RollingDispatcher:
        config = DispatchConfig(domain="yellow", density_pct=100, **kwargs)
        return RollingDispatcher(
            config,
            domain_config=self.domain_config,
            router=self.router,
            predictor=self.predictor,
            h3_stats_dict=self.h3_stats_dict,
        )

    def test_driver_activates_and_serves_request_in_same_batch(self) -> None:
        drivers = pd.DataFrame([_driver_row("2015-04-01T10:00:10")])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:05",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, batch_metrics, summary = dispatcher.run_policy("coldstart", drivers, riders, seed=42)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].matched_riders, 1)
        self.assertEqual(summary.served_riders, 1)
        self.assertEqual(batch_metrics[0].launched_drivers, 1)

    def test_request_expires_before_late_driver(self) -> None:
        drivers = pd.DataFrame([_driver_row("2015-04-01T10:10:00")])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:00",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, _, summary = dispatcher.run_policy("coldstart", drivers, riders, seed=42)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].matched_riders, 0)
        self.assertEqual(summary.served_riders, 0)

    def test_request_at_exact_window_threshold_is_still_available(self) -> None:
        drivers = pd.DataFrame([_driver_row("2015-04-01T10:05:00")])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:00",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, _, summary = dispatcher.run_policy("coldstart", drivers, riders, seed=42)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].matched_riders, 1)
        self.assertEqual(summary.served_riders, 1)

    def test_future_request_in_same_batch_is_not_visible_to_earlier_driver(self) -> None:
        drivers = pd.DataFrame([_driver_row("2015-04-01T10:00:05")])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:50",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, _, summary = dispatcher.run_policy("coldstart", drivers, riders, seed=42)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].matched_riders, 0)
        self.assertEqual(summary.served_riders, 0)

    def test_same_rider_is_not_assigned_to_two_drivers(self) -> None:
        drivers = pd.DataFrame([
            _driver_row("2015-04-01T10:00:10"),
            _driver_row("2015-04-01T10:00:20"),
        ])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:05",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, _, summary = dispatcher.run_policy("warmup", drivers, riders, seed=42)
        self.assertEqual(sum(outcome.matched_riders for outcome in outcomes), 1)
        self.assertEqual(summary.served_riders, 1)

    def test_single_driver_dispatch_matches_route_evaluator(self) -> None:
        drivers = pd.DataFrame([_driver_row("2015-04-01T10:00:10")])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:05",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, _, _ = dispatcher.run_policy("warmup", drivers, riders, seed=42)

        from matching.rider_index import RiderIndex

        rider_index = RiderIndex(riders.reset_index(drop=True), index_bin_minutes=15)
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
            self.predictor,
            self.h3_stats_dict,
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
            routes=self.routes,
        )
        warmup_plan = evaluation.plans["warmup"]
        self.assertAlmostEqual(outcomes[0].profit, warmup_plan.outcome.profit, places=6)
        self.assertEqual(outcomes[0].matched_riders, warmup_plan.outcome.matched_riders)

    def test_heuristic_tie_break_does_not_use_realized_profit(self) -> None:
        drivers = pd.DataFrame([_driver_row("2015-04-01T10:00:10")])
        riders = pd.DataFrame(columns=[
            "split", "pickup_datetime", "pickup_h3", "dropoff_h3",
            "pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng",
            "passenger_count", "fare_amount",
        ])
        riders["pickup_datetime"] = pd.to_datetime(riders["pickup_datetime"])
        from matching.rider_index import RiderIndex

        routes = [
            _make_route_with_distance(3200.0, 0.0),
            _make_route_with_distance(2200.0, 0.02),
            _make_route_with_distance(2600.0, -0.02),
        ]
        rider_index = RiderIndex(riders, index_bin_minutes=15)
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
            self.predictor,
            self.h3_stats_dict,
            day_of_week=int(drivers.iloc[0]["day_of_week"]),
            is_weekend=int(drivers.iloc[0]["is_weekend"]),
            day_of_month=int(drivers.iloc[0]["pickup_datetime"].day),
            seed=42,
            candidate_window_bins=1,
            max_request_offset_min=5,
            h3_resolution=9,
            corridor_k_ring=1,
            corridor_densify_step_m=80.0,
            available_rider_ids=set(),
            routes=routes,
        )
        self.assertEqual(evaluation.plans["heuristic_count"].route_idx, 0)
        self.assertEqual(evaluation.plans["heuristic_feasible_count"].route_idx, 0)

    def test_coldstart_dispatch_tie_uses_departure_order(self) -> None:
        drivers = pd.DataFrame([
            _driver_row("2015-04-01T10:00:05"),
            _driver_row("2015-04-01T10:00:50"),
        ])
        drivers.loc[1, "trip_distance_miles"] = 20.0
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:00",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
            )
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5)
        outcomes, _, summary = dispatcher.run_policy("coldstart", drivers, riders, seed=42)
        matched = [outcome for outcome in outcomes if outcome.matched_riders == 1]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].driver_id, 0)
        self.assertEqual(summary.served_riders, 1)

    def test_summary_wait_is_rider_weighted(self) -> None:
        drivers = pd.DataFrame([
            _driver_row("2015-04-01T10:00:10"),
            _driver_row("2015-04-01T10:05:00"),
        ])
        riders = pd.DataFrame([
            _rider_row(
                "2015-04-01T10:00:00",
                pickup=(40.7540, -73.9860),
                dropoff=(40.7660, -73.9740),
                fare=12.0,
            ),
            _rider_row(
                "2015-04-01T10:00:05",
                pickup=(40.7545, -73.9855),
                dropoff=(40.7665, -73.9735),
                fare=11.0,
            ),
            _rider_row(
                "2015-04-01T10:00:00",
                pickup=(40.7542, -73.9858),
                dropoff=(40.7662, -73.9738),
                fare=1.0,
            ),
        ])
        dispatcher = self._make_dispatcher(max_request_offset_min=5, seats=2)
        outcomes, _, summary = dispatcher.run_policy("coldstart", drivers, riders, seed=42)
        self.assertEqual(sum(outcome.matched_riders for outcome in outcomes), 3)
        self.assertAlmostEqual(summary.mean_wait_min, 1.75, places=6)


if __name__ == "__main__":
    unittest.main()
