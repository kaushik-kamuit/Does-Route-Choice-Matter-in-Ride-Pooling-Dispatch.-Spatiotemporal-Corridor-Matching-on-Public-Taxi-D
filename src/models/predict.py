"""
Thin inference wrapper around the trained LightGBM profit model.

Used by the warm-up simulation pipeline to rank alternative routes
by predicted profit.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

FEATURE_COLS = [
    "route_distance_m",
    "route_duration_s",
    "corridor_cell_count",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "corridor_rider_count",
    "corridor_demand_density",
    "mean_rider_fare",
    "corridor_fare_density",
    "day_of_month",
    "time_bin_15min",
    "hour_sin",
    "hour_cos",
    "route_sinuosity",
    "route_avg_speed_ms",
    "bearing_sin",
    "bearing_cos",
    "straight_line_dist_m",
    "origin_landmark_dist_km",
    "dest_landmark_dist_km",
    "origin_jfk_km",
    "origin_lga_km",
    "origin_penn_km",
    "origin_times_sq_km",
    "dest_jfk_km",
    "dest_lga_km",
    "dest_penn_km",
    "dest_times_sq_km",
    "corridor_hist_pickups",
    "corridor_hist_dropoffs",
    "corridor_hist_pickup_density",
    "corridor_hist_dropoff_density",
    "corridor_hist_mean_fare",
    "corridor_hist_fare_density",
    "origin_cell_pickups",
    "origin_cell_mean_fare",
    "dest_cell_dropoffs",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "profit_model_v2.pkl"


class ProfitPredictor:
    """Load a trained model once, then call predict / rank_routes many times."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self._model = joblib.load(model_path)

    def predict(self, features: dict[str, float]) -> float:
        """Return the predicted expected profit for a single route."""
        x = np.array([[features[c] for c in FEATURE_COLS]])
        return float(self._model.predict(x)[0])

    def predict_batch(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        """Predict profit for multiple routes at once."""
        x = np.array([[row[c] for c in FEATURE_COLS] for row in feature_rows])
        return self._model.predict(x)

    def rank_routes(
        self, feature_list: list[dict[str, float]]
    ) -> list[tuple[int, float]]:
        """
        Return (route_index, predicted_profit) sorted by profit descending.

        The first element is the best route to recommend to the driver.
        """
        preds = self.predict_batch(feature_list)
        indexed = list(enumerate(preds))
        indexed.sort(key=lambda t: t[1], reverse=True)
        return [(idx, float(pred)) for idx, pred in indexed]
