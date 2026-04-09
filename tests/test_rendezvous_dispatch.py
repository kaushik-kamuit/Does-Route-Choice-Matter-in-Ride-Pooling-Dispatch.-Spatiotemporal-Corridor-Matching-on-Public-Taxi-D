from __future__ import annotations

import sys
import unittest
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous import RendezvousConfig, RendezvousDispatcher
from rendezvous.urban_context import UrbanContextIndex
from spatial.router import RouteInfo


class _Router:
    def __init__(self, route: RouteInfo) -> None:
        self._route = route

    def get_alternative_routes(self, *_args, **_kwargs):
        return [self._route]


class DispatchBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = RouteInfo(
            polyline=((40.7500, -73.9900), (40.7600, -73.9800), (40.7700, -73.9700)),
            distance_m=2500.0,
            duration_s=600.0,
        )
        self.router = _Router(self.route)
        self.drivers_df = pd.DataFrame(
            [
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:00:00"),
                    "origin_lat": 40.7500,
                    "origin_lng": -73.9900,
                    "dest_lat": 40.7700,
                    "dest_lng": -73.9700,
                    "hour_of_day": 10,
                    "trip_distance_miles": 2.0,
                },
                {
                    "pickup_datetime": pd.Timestamp("2015-04-01 10:01:00"),
                    "origin_lat": 40.7500,
                    "origin_lng": -73.9900,
                    "dest_lat": 40.7700,
                    "dest_lng": -73.9700,
                    "hour_of_day": 10,
                    "trip_distance_miles": 2.0,
                },
            ]
        )

    def test_failed_attempts_are_retired(self) -> None:
        pickup_lat, pickup_lng = 40.7550, -73.9850
        riders_df = pd.DataFrame(
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
        config = RendezvousConfig(occlusion_lambda=0.95, retire_failed_attempts=True)
        dispatcher = RendezvousDispatcher(
            config,
            router=self.router,
            urban_context=UrbanContextIndex(),
        )
        sampled_riders_df, rider_index, request_states, request_batches = dispatcher.prepare_rider_pool(riders_df)

        outcomes, summary = dispatcher.run_policy(
            "rendezvous_observable",
            self.drivers_df,
            riders_df,
            seed=0,
            sampled_riders_df=sampled_riders_df,
            rider_index=rider_index,
            request_states=request_states,
            request_batches=request_batches,
        )

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[1].open_requests_before, 0)
        self.assertEqual(summary.served_riders, 0)

    def test_service_rate_uses_eligible_riders_before_last_driver_batch(self) -> None:
        riders_df = pd.DataFrame(
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
                    "pickup_datetime": pd.Timestamp("2015-04-01 11:00:00"),
                    "pickup_h3": h3.latlng_to_cell(40.7560, -73.9840, 9),
                    "dropoff_h3": h3.latlng_to_cell(40.7680, -73.9720, 9),
                    "pickup_lat": 40.7560,
                    "pickup_lng": -73.9840,
                    "dropoff_lat": 40.7680,
                    "dropoff_lng": -73.9720,
                    "passenger_count": 1,
                    "fare_amount": 17.0,
                },
            ]
        )
        config = RendezvousConfig()
        dispatcher = RendezvousDispatcher(config, router=self.router, urban_context=UrbanContextIndex())
        sampled_riders_df, rider_index, request_states, request_batches = dispatcher.prepare_rider_pool(riders_df)

        _outcomes, summary = dispatcher.run_policy(
            "corridor_only",
            self.drivers_df.iloc[:1].reset_index(drop=True),
            riders_df,
            seed=0,
            sampled_riders_df=sampled_riders_df,
            rider_index=rider_index,
            request_states=request_states,
            request_batches=request_batches,
        )

        self.assertEqual(summary.eligible_riders, 1)


if __name__ == "__main__":
    unittest.main()
