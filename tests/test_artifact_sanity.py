from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ArtifactSanityTests(unittest.TestCase):
    def test_selected_heuristic_is_best_non_ml_per_density(self) -> None:
        df = pd.read_csv(ROOT / "results" / "strong_baseline_comparison.csv")
        for density, group in df.groupby("density_pct"):
            selected = group[group["selected_for_paper"]]
            self.assertEqual(len(selected), 1, f"density {density} should have one selected heuristic")
            best = group.sort_values("mean_profit", ascending=False).iloc[0]
            self.assertEqual(
                selected.iloc[0]["heuristic_strategy"],
                best["heuristic_strategy"],
                f"density {density} selected heuristic should match the highest-profit heuristic",
            )

    def test_looser_request_windows_improve_reported_profits(self) -> None:
        df = pd.read_csv(ROOT / "results" / "window_sensitivity.csv")
        for density in (100, 25, 10):
            sub = df[df["density_pct"] == density].sort_values("matching_window_min")
            self.assertEqual(sub["matching_window_min"].tolist(), [2, 5, 10])
            for column in ("coldstart_profit", "heuristic_profit", "warmup_profit", "oracle_profit"):
                values = sub[column].tolist()
                self.assertLessEqual(values[0], values[1], f"{column} should improve from 2 to 5 minutes")
                self.assertLessEqual(values[1], values[2], f"{column} should improve from 5 to 10 minutes")

    def test_economics_sensitivity_moves_in_expected_direction(self) -> None:
        df = pd.read_csv(ROOT / "results" / "economics_sensitivity.csv").set_index("tag")
        base = float(df.loc["d10", "warmup_profit"])
        self.assertLessEqual(float(df.loc["econ_ps40_d10", "warmup_profit"]), base)
        self.assertGreaterEqual(float(df.loc["econ_ps60_d10", "warmup_profit"]), base)
        self.assertGreaterEqual(float(df.loc["econ_c50_d10", "warmup_profit"]), base)
        self.assertLessEqual(float(df.loc["econ_c85_d10", "warmup_profit"]), base)


if __name__ == "__main__":
    unittest.main()
