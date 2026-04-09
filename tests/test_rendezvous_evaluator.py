from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matching.rider_index import RiderIndex
from rendezvous import RendezvousConfig, evaluate_driver_policies
from rendezvous.data_types import DriverTrip
from rendezvous.meeting_points import build_route_anchor_cells
from rendezvous.urban_context import UrbanContextIndex
from spatial.router import RouteInfo


class RendezvousEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = RouteInfo(
            polyline=((40.7500, -73.9900), (40.7600, -73.9800), (40.7700, -73.9700)),
            distance_m=2500.0,
            duration_s=600.0,
        )
        self.driver = DriverTrip(
            driver_id=1,
            origin=(40.7500, -73.9900),
            destination=(40.7700, -73.9700),
            departure_time=datetime(2015, 4, 1, 10, 0, 0),
            hour=10,
            minute_of_day=600,
            trip_distance_miles=2.0,
        )

    def test_zero_anchor_overlap_yields_zero_feasible_opportunities(self) -> None:
        route_cell = h3.latlng_to_cell(40.7500, -73.9900, 9)
        neighbor = next(cell for cell in h3.grid_disk(route_cell, 1) if cell != route_cell)
        pickup_lat, pickup_lng = h3.cell_to_latlng(neighbor)
        dropoff_lat, dropoff_lng = h3.cell_to_latlng(h3.latlng_to_cell(40.7700, -73.9700, 9))
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": neighbor,
                    "dropoff_h3": h3.latlng_to_cell(dropoff_lat, dropoff_lng, 9),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "dropoff_lat": dropoff_lat,
                    "dropoff_lng": dropoff_lng,
                    "passenger_count": 1,
                    "fare_amount": 12.0,
                }
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=0)

        evaluation = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route],
            seed=42,
        )

        self.assertEqual(evaluation.route_evaluations[0].feasible_opportunity_count, 0)
        self.assertEqual(evaluation.plans["corridor_only"].attempted_riders, 1)
        self.assertEqual(evaluation.plans["corridor_only"].successful_riders, 0)

    def test_observable_value_is_not_higher_than_nominal_value(self) -> None:
        pickup_lat, pickup_lng = 40.7550, -73.9850
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(pickup_lat, pickup_lng, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                }
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=1, occlusion_lambda=0.25)

        evaluation = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route],
            seed=42,
        )

        route_eval = evaluation.route_evaluations[0]
        self.assertGreaterEqual(route_eval.nominal_route_value, route_eval.observable_route_value)

    def test_policy_realization_is_seed_reproducible(self) -> None:
        pickup_lat, pickup_lng = 40.7550, -73.9850
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(pickup_lat, pickup_lng, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                }
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=1)

        first = evaluate_driver_policies(self.driver, rider_index, config, routes=[self.route], seed=42)
        second = evaluate_driver_policies(self.driver, rider_index, config, routes=[self.route], seed=42)

        self.assertEqual(
            first.plans["rendezvous_observable"].successful_rider_ids,
            second.plans["rendezvous_observable"].successful_rider_ids,
        )

    def test_urban_context_can_lower_observable_value(self) -> None:
        pickup_lat, pickup_lng = 40.7550, -73.9850
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(pickup_lat, pickup_lng, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                }
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=1, occlusion_lambda=0.25)
        route_cells = build_route_anchor_cells(self.route, resolution=9, densify_step_m=80.0)
        poor_context = UrbanContextIndex.from_frame(
            pd.DataFrame(
                [
                    {
                        "h3_cell": cell,
                        "urban_clutter_index": 0.95,
                        "sidewalk_access_score": 0.05,
                        "building_height_proxy": 0.85,
                    }
                    for cell in route_cells
                ]
            )
        )
        good_context = UrbanContextIndex.from_frame(
            pd.DataFrame(
                [
                    {
                        "h3_cell": cell,
                        "urban_clutter_index": 0.05,
                        "sidewalk_access_score": 0.95,
                        "building_height_proxy": 0.05,
                    }
                    for cell in route_cells
                ]
            )
        )

        baseline = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route],
            urban_context=good_context,
            seed=42,
        )
        contextual = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route],
            urban_context=poor_context,
            seed=42,
        )

        self.assertLessEqual(
            contextual.route_evaluations[0].observable_route_value,
            baseline.route_evaluations[0].observable_route_value,
        )

    def test_rendezvous_only_uses_nominal_selector(self) -> None:
        pickup_lat, pickup_lng = 40.7550, -73.9850
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(pickup_lat, pickup_lng, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                }
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=1, occlusion_lambda=0.8)

        evaluation = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route],
            seed=42,
        )

        self.assertGreaterEqual(
            evaluation.plans["rendezvous_only"].nominal_revenue,
            evaluation.plans["rendezvous_observable"].nominal_revenue,
        )

    def test_time_only_baseline_prefers_shorter_route(self) -> None:
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(40.7550, -73.9850, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": 40.7550,
                    "pickup_lng": -73.9850,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                }
            ]
        )
        shorter_route = RouteInfo(
            polyline=((40.7500, -73.9900), (40.7590, -73.9810), (40.7700, -73.9700)),
            distance_m=1800.0,
            duration_s=500.0,
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=1)

        evaluation = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route, shorter_route],
            seed=42,
        )

        self.assertEqual(evaluation.plans["time_only_baseline"].route_idx, 1)

    def test_feasible_count_baseline_prefers_route_with_more_feasible_opportunities(self) -> None:
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(40.7550, -73.9850, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": 40.7550,
                    "pickup_lng": -73.9850,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                },
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(40.7650, -73.9750, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7720, -73.9680, 9),
                    "pickup_lat": 40.7650,
                    "pickup_lng": -73.9750,
                    "dropoff_lat": 40.7720,
                    "dropoff_lng": -73.9680,
                    "passenger_count": 1,
                    "fare_amount": 17.0,
                },
            ]
        )
        detour_route = RouteInfo(
            polyline=((40.7500, -73.9900), (40.7520, -73.9880), (40.7540, -73.9860)),
            distance_m=1600.0,
            duration_s=420.0,
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        config = RendezvousConfig(meeting_k_ring=1)

        evaluation = evaluate_driver_policies(
            self.driver,
            rider_index,
            config,
            routes=[self.route, detour_route],
            seed=42,
        )

        self.assertEqual(evaluation.plans["feasible_count_baseline"].route_idx, 0)

    def test_observability_ablation_can_raise_observable_value_in_cluttered_context(self) -> None:
        pickup_lat, pickup_lng = 40.7550, -73.9850
        riders = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "pickup_h3": h3.latlng_to_cell(pickup_lat, pickup_lng, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7690, -73.9710, 9),
                    "pickup_lat": pickup_lat,
                    "pickup_lng": pickup_lng,
                    "dropoff_lat": 40.7690,
                    "dropoff_lng": -73.9710,
                    "passenger_count": 1,
                    "fare_amount": 18.0,
                }
            ]
        )
        rider_index = RiderIndex(riders, index_bin_minutes=15)
        route_cells = build_route_anchor_cells(self.route, resolution=9, densify_step_m=80.0)
        cluttered_context = UrbanContextIndex.from_frame(
            pd.DataFrame(
                [
                    {
                        "h3_cell": cell,
                        "urban_clutter_index": 0.95,
                        "sidewalk_access_score": 0.05,
                        "building_height_proxy": 0.85,
                    }
                    for cell in route_cells
                ]
            )
        )
        full_config = RendezvousConfig(meeting_k_ring=1, occlusion_lambda=0.4, observability_ablation="full")
        ablated_config = RendezvousConfig(
            meeting_k_ring=1,
            occlusion_lambda=0.4,
            observability_ablation="no_clutter",
        )

        full_eval = evaluate_driver_policies(
            self.driver,
            rider_index,
            full_config,
            routes=[self.route],
            urban_context=cluttered_context,
            seed=42,
        )
        ablated_eval = evaluate_driver_policies(
            self.driver,
            rider_index,
            ablated_config,
            routes=[self.route],
            urban_context=cluttered_context,
            seed=42,
        )

        self.assertGreaterEqual(
            ablated_eval.route_evaluations[0].observable_route_value,
            full_eval.route_evaluations[0].observable_route_value,
        )


if __name__ == "__main__":
    unittest.main()
