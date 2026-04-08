from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.urban_context import asset_catalog_rows
from rendezvous.urban_context import UrbanContextFeatures, UrbanContextIndex


class UrbanContextTests(unittest.TestCase):
    def test_catalog_contains_recommended_assets(self) -> None:
        keys = {row["key"] for row in asset_catalog_rows()}
        self.assertTrue({"street_centerline", "sidewalk_centerline", "building_footprints", "pluto"}.issubset(keys))

    def test_missing_cell_uses_safe_defaults(self) -> None:
        index = UrbanContextIndex()
        self.assertEqual(index.lookup("892a100d2d7ffff"), UrbanContextFeatures())

    def test_from_frame_round_trips_scores(self) -> None:
        index = UrbanContextIndex.from_frame(
            pd.DataFrame(
                [
                    {
                        "h3_cell": "892a100d2d7ffff",
                        "urban_clutter_index": 0.7,
                        "sidewalk_access_score": 0.4,
                        "building_height_proxy": 0.6,
                        "building_intensity": 0.5,
                        "street_complexity": 0.3,
                        "elevation_complexity": 0.2,
                    }
                ]
            )
        )
        features = index.lookup("892a100d2d7ffff")
        self.assertAlmostEqual(features.urban_clutter_index, 0.7)
        self.assertAlmostEqual(features.sidewalk_access_score, 0.4)


if __name__ == "__main__":
    unittest.main()
