from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.analysis import build_matched_observability_pairs, select_case_studies
from rendezvous.domain_io import apply_area_slice
from rendezvous.observability import weights_for_profile
from rendezvous.urban_context import UrbanContextIndex


class RendezvousAnalysisTests(unittest.TestCase):
    def test_matched_pair_builder_is_deterministic(self) -> None:
        route_df = pd.DataFrame(
            [
                {
                    "domain": "yellow",
                    "scenario_name": "sparse_high_occlusion",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "rider_density_pct": 10,
                    "occlusion_lambda": 0.4,
                    "meeting_k_ring": 1,
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "walk_penalty_per_min": 0.5,
                    "driver_id": 7,
                    "route_idx": 0,
                    "candidate_count": 5,
                    "feasible_opportunity_count": 2,
                    "route_distance_miles": 2.0,
                    "mean_route_walk_min": 3.0,
                    "mean_route_observability": 0.82,
                    "route_cost": 1.2,
                },
                {
                    "domain": "yellow",
                    "scenario_name": "sparse_high_occlusion",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "rider_density_pct": 10,
                    "occlusion_lambda": 0.4,
                    "meeting_k_ring": 1,
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "walk_penalty_per_min": 0.5,
                    "driver_id": 7,
                    "route_idx": 1,
                    "candidate_count": 4,
                    "feasible_opportunity_count": 2,
                    "route_distance_miles": 2.1,
                    "mean_route_walk_min": 3.2,
                    "mean_route_observability": 0.62,
                    "route_cost": 1.2,
                },
            ]
        )
        opportunity_df = pd.DataFrame(
            [
                {
                    "domain": "yellow",
                    "scenario_name": "sparse_high_occlusion",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "rider_density_pct": 10,
                    "occlusion_lambda": 0.4,
                    "meeting_k_ring": 1,
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "walk_penalty_per_min": 0.5,
                    "driver_id": 7,
                    "route_idx": 0,
                    "route_cost": 1.2,
                    "rider_id": 10,
                    "anchor_cell": "892a100d2d7ffff",
                    "anchor_idx": 0,
                    "pickup_h3": "892a100d2d7ffff",
                    "dropoff_h3": "892a100d2d7ffff",
                    "fare_share": 8.0,
                    "passenger_count": 1,
                    "walk_m": 250.0,
                    "walk_min": 3.0,
                    "anchor_progress": 0.3,
                    "travel_fraction": 0.4,
                    "ambiguity_count": 1,
                    "local_straightness": 0.9,
                    "turn_severity": 0.2,
                    "anchor_clutter": 0.3,
                    "urban_clutter_index": 0.4,
                    "sidewalk_access_score": 0.8,
                    "building_height_proxy": 0.5,
                    "context_is_imputed": False,
                    "observability_score": 0.82,
                    "success_probability": 0.90,
                },
                {
                    "domain": "yellow",
                    "scenario_name": "sparse_high_occlusion",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "rider_density_pct": 10,
                    "occlusion_lambda": 0.4,
                    "meeting_k_ring": 1,
                    "observability_profile": "calibrated",
                    "observability_ablation": "full",
                    "use_urban_context": True,
                    "walk_penalty_per_min": 0.5,
                    "driver_id": 7,
                    "route_idx": 1,
                    "route_cost": 1.2,
                    "rider_id": 10,
                    "anchor_cell": "892a100d2d7ffff",
                    "anchor_idx": 0,
                    "pickup_h3": "892a100d2d7ffff",
                    "dropoff_h3": "892a100d2d7ffff",
                    "fare_share": 8.0,
                    "passenger_count": 1,
                    "walk_m": 270.0,
                    "walk_min": 3.2,
                    "anchor_progress": 0.3,
                    "travel_fraction": 0.4,
                    "ambiguity_count": 1,
                    "local_straightness": 0.7,
                    "turn_severity": 0.3,
                    "anchor_clutter": 0.6,
                    "urban_clutter_index": 0.5,
                    "sidewalk_access_score": 0.7,
                    "building_height_proxy": 0.6,
                    "context_is_imputed": False,
                    "observability_score": 0.62,
                    "success_probability": 0.72,
                },
            ]
        )
        first_pairs, first_summary = build_matched_observability_pairs(route_df, opportunity_df, seeds=[42, 43], iterations=100)
        second_pairs, second_summary = build_matched_observability_pairs(route_df, opportunity_df, seeds=[42, 43], iterations=100)
        self.assertFalse(first_pairs.empty)
        self.assertEqual(first_pairs.to_dict(orient="records"), second_pairs.to_dict(orient="records"))
        self.assertEqual(first_summary.to_dict(orient="records"), second_summary.to_dict(orient="records"))

    def test_case_selection_prefers_requested_mix(self) -> None:
        matched = pd.DataFrame(
            [
                {
                    "domain": "yellow",
                    "scenario_name": "sparse_high_occlusion",
                    "time_slice": "morning_peak",
                    "area_slice": "all",
                    "driver_id": idx,
                    "high_route_idx": 0,
                    "low_route_idx": 1,
                    "profit_delta": 1.0 + idx,
                    "observability_gap": 0.2,
                    "higher_observability_wins": 1,
                    "tie": 0,
                    "distance_pct_diff": 0.05,
                    "walk_min_diff": 0.2,
                }
                for idx in range(6)
            ]
            + [
                {
                    "domain": "green",
                    "scenario_name": "sparse_high_occlusion",
                    "time_slice": "all_day",
                    "area_slice": "all",
                    "driver_id": 100 + idx,
                    "high_route_idx": 0,
                    "low_route_idx": 1,
                    "profit_delta": 1.0,
                    "observability_gap": 0.18,
                    "higher_observability_wins": 1,
                    "tie": 0,
                    "distance_pct_diff": 0.04,
                    "walk_min_diff": 0.3,
                }
                for idx in range(2)
            ]
        )
        selected = select_case_studies(matched, total_cases=8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(int((selected["domain"] == "yellow").sum()), 6)
        self.assertEqual(int((selected["domain"] == "green").sum()), 2)

    def test_green_calibrated_profile_falls_back_to_yellow(self) -> None:
        weights = weights_for_profile("calibrated", domain="green")
        self.assertGreater(weights.straightness + weights.turn + weights.ambiguity + weights.clutter, 0.0)

    def test_area_slice_filters_by_context_score(self) -> None:
        drivers = pd.DataFrame({"origin_h3": ["a", "b"], "pickup_datetime": ["2015-04-01", "2015-04-01"]})
        riders = pd.DataFrame({"pickup_h3": ["a", "b"], "pickup_datetime": ["2015-04-01", "2015-04-01"]})
        index = UrbanContextIndex.from_frame(
            pd.DataFrame(
                [
                    {"h3_cell": "a", "urban_clutter_index": 0.1, "building_height_proxy": 0.1},
                    {"h3_cell": "b", "urban_clutter_index": 0.9, "building_height_proxy": 0.9},
                ]
            )
        )
        dense_drivers, dense_riders = apply_area_slice(drivers, riders, index, area_slice="dense_core")
        open_drivers, open_riders = apply_area_slice(drivers, riders, index, area_slice="open_grid")
        self.assertEqual(dense_drivers["origin_h3"].tolist(), ["b"])
        self.assertEqual(dense_riders["pickup_h3"].tolist(), ["b"])
        self.assertEqual(open_drivers["origin_h3"].tolist(), ["a"])
        self.assertEqual(open_riders["pickup_h3"].tolist(), ["a"])


if __name__ == "__main__":
    unittest.main()
