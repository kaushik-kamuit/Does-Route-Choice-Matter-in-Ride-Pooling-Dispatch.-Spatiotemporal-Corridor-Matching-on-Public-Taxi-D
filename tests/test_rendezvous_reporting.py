from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.reporting import write_result_views


class RendezvousReportingTests(unittest.TestCase):
    def test_write_result_views_preserves_domain_in_gap_summary(self) -> None:
        driver_summary = pd.DataFrame(
            [
                {
                    "policy": "corridor_only",
                    "domain": "yellow",
                    "scenario_name": "primary",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "mean_nominal_realized_gap": 10.0,
                },
                {
                    "policy": "corridor_only",
                    "domain": "green",
                    "scenario_name": "primary",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "mean_nominal_realized_gap": 4.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_result_views(Path(tmpdir), driver_summary)
            gap = pd.read_csv(Path(tmpdir) / "rendezvous_nominal_realized_gap.csv")

        self.assertEqual(set(gap["domain"]), {"yellow", "green"})
        self.assertEqual(
            gap.sort_values("domain")["mean_nominal_realized_gap"].tolist(),
            [4.0, 10.0],
        )

    def test_write_result_views_prefers_area_all_slice(self) -> None:
        driver_summary = pd.DataFrame(
            [
                {
                    "policy": "rendezvous_only",
                    "domain": "yellow",
                    "scenario_name": "primary",
                    "time_slice": "all_day",
                    "area_slice": "dense_core",
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "mean_nominal_realized_gap": 2.0,
                },
                {
                    "policy": "rendezvous_only",
                    "domain": "yellow",
                    "scenario_name": "primary",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "mean_nominal_realized_gap": 3.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_result_views(Path(tmpdir), driver_summary)
            primary = pd.read_csv(Path(tmpdir) / "rendezvous_primary_summary.csv")

        self.assertEqual(primary["area_slice"].tolist(), ["all"])


if __name__ == "__main__":
    unittest.main()
