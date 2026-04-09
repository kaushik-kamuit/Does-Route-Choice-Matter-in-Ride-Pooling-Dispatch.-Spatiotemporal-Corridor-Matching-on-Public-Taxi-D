from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep.urban_context import _download_csv_pages, _with_socrata_paging, asset_catalog_rows, get_asset
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
        self.assertFalse(features.is_imputed)

    def test_missing_cell_uses_frame_medians(self) -> None:
        frame = pd.DataFrame(
            [
                {"h3_cell": "a", "urban_clutter_index": 0.2, "sidewalk_access_score": 0.8, "building_height_proxy": 0.1},
                {"h3_cell": "b", "urban_clutter_index": 0.6, "sidewalk_access_score": 0.4, "building_height_proxy": 0.5},
            ]
        )
        index = UrbanContextIndex.from_frame(frame)
        features = index.lookup("missing")
        self.assertTrue(features.is_imputed)
        self.assertAlmostEqual(features.urban_clutter_index, 0.4)
        self.assertAlmostEqual(features.sidewalk_access_score, 0.6)
        self.assertAlmostEqual(features.building_height_proxy, 0.3)

    def test_paging_url_adds_limit_and_offset(self) -> None:
        url = _with_socrata_paging("https://data.cityofnewyork.us/resource/abc.csv?$select=x", limit=500, offset=1000)
        self.assertIn("%24limit=500", url)
        self.assertIn("%24offset=1000", url)

    def test_download_csv_pages_appends_multiple_pages(self) -> None:
        class _Response:
            def __init__(self, text: str) -> None:
                self.text = text

            def raise_for_status(self) -> None:
                return None

        class _Session:
            def __init__(self, responses: list[str]) -> None:
                self._responses = responses

            def get(self, *_args, **_kwargs):
                return _Response(self._responses.pop(0))

        asset = get_asset("pluto")
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "pluto.csv"
            session = _Session(["latitude,longitude\n1,2\n3,4\n", "latitude,longitude\n5,6\n"])
            with patch("data_prep.urban_context.SOCRATA_PAGE_SIZE", 2):
                rows, pages = _download_csv_pages(session, asset, target, timeout_s=1)
            self.assertEqual(rows, 3)
            self.assertEqual(pages, 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "latitude,longitude\n1,2\n3,4\n5,6\n")


if __name__ == "__main__":
    unittest.main()
