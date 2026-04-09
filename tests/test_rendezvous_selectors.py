from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.data_types import RendezvousOpportunity
from rendezvous.selectors import DeterministicMeetingPointSelector, WalkAwareMeetingPointSelector


def _opportunity(rider_id: int, fare_share: float, passenger_count: int, anchor_idx: int) -> RendezvousOpportunity:
    return RendezvousOpportunity(
        rider_id=rider_id,
        anchor_cell=f"cell-{rider_id}",
        anchor_idx=anchor_idx,
        pickup_h3=f"pickup-{rider_id}",
        dropoff_h3=f"dropoff-{rider_id}",
        fare_share=fare_share,
        passenger_count=passenger_count,
        walk_m=100.0,
        walk_min=1.0,
        anchor_progress=0.5,
        travel_fraction=0.5,
        ambiguity_count=1,
        local_straightness=1.0,
        turn_severity=0.0,
        anchor_clutter=0.1,
        urban_clutter_index=0.1,
        sidewalk_access_score=0.9,
        building_height_proxy=0.2,
        context_is_imputed=False,
        observability_score=0.9,
        success_probability=0.9,
    )


class SelectorPackingTests(unittest.TestCase):
    def test_selector_uses_optimal_seat_packing(self) -> None:
        selector = DeterministicMeetingPointSelector(use_observability=False)
        opportunities = [
            _opportunity(rider_id=1, fare_share=6.0, passenger_count=2, anchor_idx=0),
            _opportunity(rider_id=2, fare_share=4.1, passenger_count=1, anchor_idx=1),
            _opportunity(rider_id=3, fare_share=4.0, passenger_count=1, anchor_idx=2),
        ]

        selected = selector.select(opportunities, seats=2)

        self.assertEqual(sorted(opportunity.rider_id for opportunity in selected), [2, 3])

    def test_walk_aware_selector_prefers_shorter_walk_when_fares_match(self) -> None:
        selector = WalkAwareMeetingPointSelector(walk_penalty_per_min=1.0)
        near = _opportunity(rider_id=1, fare_share=6.0, passenger_count=1, anchor_idx=0)
        far = _opportunity(rider_id=1, fare_share=6.0, passenger_count=1, anchor_idx=1)
        object.__setattr__(near, "walk_min", 1.0)
        object.__setattr__(far, "walk_min", 4.0)

        selected = selector.select([near, far], seats=1)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].anchor_idx, 0)


if __name__ == "__main__":
    unittest.main()
