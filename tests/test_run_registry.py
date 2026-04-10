from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.run_registry import backfill_legacy_runs, has_registered_runs, registered_file_paths


class RunRegistryTests(unittest.TestCase):
    def test_backfill_copies_legacy_driver_outputs_into_run_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            pd.DataFrame([{"driver_id": 1, "policy": "corridor_only", "actual_profit": 1.0}]).to_csv(
                results_dir / "rendezvous_driver_outcomes_primary.csv",
                index=False,
            )
            pd.DataFrame([{"driver_id": 1, "route_idx": 0}]).to_csv(
                results_dir / "rendezvous_route_evaluations_primary.csv",
                index=False,
            )
            pd.DataFrame([{"driver_id": 1, "route_idx": 0, "rider_id": 7}]).to_csv(
                results_dir / "rendezvous_route_opportunities_primary.csv",
                index=False,
            )
            pd.DataFrame([{"policy": "corridor_only", "mean_actual_profit": 1.0}]).to_csv(
                results_dir / "rendezvous_driver_summary_primary.csv",
                index=False,
            )
            (results_dir / "rendezvous_config_primary.json").write_text(
                json.dumps({"domain": "yellow", "scenario_name": "primary"}),
                encoding="utf-8",
            )
            (results_dir / "rendezvous_driver_run_stats_primary.json").write_text(
                json.dumps({"requested_drivers": 1, "evaluated_drivers": 1}),
                encoding="utf-8",
            )

            created = backfill_legacy_runs(results_dir)

            self.assertEqual(len(created), 1)
            self.assertTrue(has_registered_runs(results_dir))
            registered = registered_file_paths(results_dir, role="driver_outcomes", run_kind="driver")
            self.assertEqual(len(registered), 1)
            self.assertIn(str(results_dir / "runs"), str(registered[0]))
            self.assertNotEqual(registered[0], results_dir / "rendezvous_driver_outcomes_primary.csv")
            self.assertTrue(registered[0].exists())

    def test_backfill_is_idempotent_for_same_legacy_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            pd.DataFrame([{"seed": 42, "policy": "corridor_only", "profit_per_driver": 1.0}]).to_csv(
                results_dir / "rendezvous_dispatch_summary_sparse.csv",
                index=False,
            )
            pd.DataFrame([{"policy": "corridor_only"}]).to_csv(
                results_dir / "rendezvous_dispatch_outcomes_sparse.csv",
                index=False,
            )
            pd.DataFrame([{"policy": "corridor_only", "mean_profit_per_driver": 1.0}]).to_csv(
                results_dir / "rendezvous_dispatch_policy_summary_sparse.csv",
                index=False,
            )
            (results_dir / "rendezvous_dispatch_config_sparse.json").write_text(
                json.dumps({"domain": "yellow", "scenario_name": "sparse_high_occlusion"}),
                encoding="utf-8",
            )
            (results_dir / "rendezvous_dispatch_run_stats_sparse.json").write_text(
                json.dumps({"requested_drivers": 10}),
                encoding="utf-8",
            )

            first = backfill_legacy_runs(results_dir)
            second = backfill_legacy_runs(results_dir)

            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
