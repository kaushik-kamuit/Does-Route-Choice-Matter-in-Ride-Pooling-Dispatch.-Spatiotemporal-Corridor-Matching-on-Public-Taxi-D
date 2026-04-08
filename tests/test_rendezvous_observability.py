from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.observability import compute_observability_score, pickup_success_probability


class ObservabilityTests(unittest.TestCase):
    def test_score_stays_in_bounds(self) -> None:
        score = compute_observability_score(
            local_straightness=1.2,
            turn_severity=-0.1,
            ambiguity_count=3,
            anchor_clutter=2.5,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_worse_geometry_lowers_score(self) -> None:
        good = compute_observability_score(
            local_straightness=0.95,
            turn_severity=0.05,
            ambiguity_count=1,
            anchor_clutter=0.1,
        )
        bad = compute_observability_score(
            local_straightness=0.35,
            turn_severity=0.85,
            ambiguity_count=4,
            anchor_clutter=2.0,
        )
        self.assertGreater(good, bad)

    def test_success_probability_is_bounded(self) -> None:
        low = pickup_success_probability(0.1, occlusion_lambda=0.25)
        high = pickup_success_probability(0.9, occlusion_lambda=0.25)
        self.assertGreaterEqual(low, 0.35)
        self.assertLessEqual(high, 0.95)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
