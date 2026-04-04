from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.evaluation_split import build_eval_split


class EvaluationSplitTests(unittest.TestCase):
    def test_prefers_temporal_month_holdout_when_months_present(self) -> None:
        df = pd.DataFrame(
            {
                "driver_id": [1, 1, 2, 2, 3, 3],
                "service_month": [1, 1, 2, 2, 3, 3],
            }
        )

        split = build_eval_split(df)

        self.assertEqual(split.split_name, "temporal_jan_feb_to_mar")
        self.assertEqual(df.iloc[split.train_idx]["service_month"].tolist(), [1, 1, 2, 2])
        self.assertEqual(df.iloc[split.val_idx]["service_month"].tolist(), [3, 3])

    def test_falls_back_to_group_shuffle_without_month_metadata(self) -> None:
        df = pd.DataFrame(
            {
                "driver_id": [1, 1, 2, 2, 3, 3, 4, 4],
            }
        )

        split = build_eval_split(df, prefer_temporal=True, group_test_size=0.25, split_seed=7)

        self.assertEqual(split.split_name, "group_shuffle_by_driver")
        self.assertGreater(len(split.train_idx), 0)
        self.assertGreater(len(split.val_idx), 0)
        self.assertTrue(set(split.train_idx).isdisjoint(set(split.val_idx)))


if __name__ == "__main__":
    unittest.main()
