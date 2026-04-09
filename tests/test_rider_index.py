from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matching.rider_index import RiderIndex


def _rider_row(
    rider_id: int,
    pickup_ts: str,
    pickup_h3: str = "pu_cell",
    dropoff_h3: str = "do_cell",
) -> dict[str, object]:
    ts = pd.Timestamp(pickup_ts)
    return {
        "rider_id": rider_id,
        "pickup_datetime": ts,
        "pickup_h3": pickup_h3,
        "dropoff_h3": dropoff_h3,
        "pickup_lat": 40.75,
        "pickup_lng": -73.99,
        "dropoff_lat": 40.76,
        "dropoff_lng": -73.98,
        "passenger_count": 1,
        "fare_amount": 10.0,
    }


class RiderIndexTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corridor = {"pu_cell", "do_cell"}

    def test_exact_request_offset_keeps_close_rider(self) -> None:
        riders = pd.DataFrame(
            [
                _rider_row(1, "2015-04-01 10:02:00"),
                _rider_row(2, "2015-04-01 10:08:00"),
            ]
        )
        index = RiderIndex(riders, index_bin_minutes=15)

        result = index.find_in_corridor(
            self.corridor,
            minute_of_day=10 * 60,
            window_bins=1,
            max_request_offset_min=5,
            query_datetime=pd.Timestamp("2015-04-01 10:00:00"),
        )

        self.assertEqual(result["rider_id"].tolist(), [1])

    def test_midnight_wraparound_respects_exact_offset(self) -> None:
        riders = pd.DataFrame(
            [
                _rider_row(1, "2015-04-01 23:58:00"),
                _rider_row(2, "2015-04-01 23:51:00"),
            ]
        )
        index = RiderIndex(riders, index_bin_minutes=15)

        result = index.find_in_corridor(
            self.corridor,
            minute_of_day=2,
            window_bins=1,
            max_request_offset_min=5,
            query_datetime=pd.Timestamp("2015-04-02 00:02:00"),
        )

        self.assertEqual(result["rider_id"].tolist(), [1])

    def test_legacy_behavior_is_preserved_without_exact_filter(self) -> None:
        riders = pd.DataFrame(
            [
                _rider_row(1, "2015-04-01 10:20:00"),
            ]
        )
        index = RiderIndex(riders, index_bin_minutes=15)

        result = index.find_in_corridor(
            self.corridor,
            minute_of_day=10 * 60,
            window_bins=1,
            max_request_offset_min=None,
        )

        self.assertEqual(result["rider_id"].tolist(), [1])

    def test_exact_filter_does_not_match_same_clock_time_on_other_days(self) -> None:
        riders = pd.DataFrame(
            [
                _rider_row(1, "2015-04-01 10:02:00"),
                _rider_row(2, "2015-04-15 10:03:00"),
            ]
        )
        index = RiderIndex(riders, index_bin_minutes=15)

        result = index.find_in_corridor(
            self.corridor,
            minute_of_day=10 * 60,
            window_bins=1,
            max_request_offset_min=5,
            query_datetime=pd.Timestamp("2015-04-01 10:00:00"),
        )

        self.assertEqual(result["rider_id"].tolist(), [1])

    def test_pickup_only_mode_keeps_off_corridor_dropoff(self) -> None:
        riders = pd.DataFrame(
            [
                _rider_row(1, "2015-04-01 10:02:00", pickup_h3="pu_cell", dropoff_h3="outside"),
            ]
        )
        index = RiderIndex(riders, index_bin_minutes=15)

        result = index.find_in_corridor(
            {"pu_cell"},
            minute_of_day=10 * 60,
            window_bins=1,
            max_request_offset_min=5,
            query_datetime=pd.Timestamp("2015-04-01 10:00:00"),
            require_dropoff_in_corridor=False,
        )

        self.assertEqual(result["rider_id"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
