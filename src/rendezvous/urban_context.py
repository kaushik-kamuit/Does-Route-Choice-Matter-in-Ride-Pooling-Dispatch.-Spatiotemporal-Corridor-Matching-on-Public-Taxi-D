from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_prep.urban_context import PROCESSED_DIR


@dataclass(frozen=True)
class UrbanContextFeatures:
    urban_clutter_index: float = 0.5
    sidewalk_access_score: float = 0.5
    building_height_proxy: float = 0.5
    building_intensity: float = 0.5
    street_complexity: float = 0.5
    elevation_complexity: float = 0.5
    is_imputed: bool = True


class UrbanContextIndex:
    def __init__(
        self,
        features_by_cell: dict[str, UrbanContextFeatures] | None = None,
        *,
        default_features: UrbanContextFeatures | None = None,
    ) -> None:
        self._features_by_cell = features_by_cell or {}
        self._default_features = default_features or UrbanContextFeatures()

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> 'UrbanContextIndex':
        features_by_cell: dict[str, UrbanContextFeatures] = {}
        if frame.empty:
            return cls(features_by_cell, default_features=UrbanContextFeatures())
        required = {'h3_cell'}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Urban context frame is missing required columns: {sorted(missing)}')
        default_features = _default_features_from_frame(frame)
        for row in frame.itertuples(index=False):
            features_by_cell[str(row.h3_cell)] = UrbanContextFeatures(
                urban_clutter_index=float(getattr(row, 'urban_clutter_index', default_features.urban_clutter_index) or default_features.urban_clutter_index),
                sidewalk_access_score=float(getattr(row, 'sidewalk_access_score', default_features.sidewalk_access_score) or default_features.sidewalk_access_score),
                building_height_proxy=float(getattr(row, 'building_height_proxy', default_features.building_height_proxy) or default_features.building_height_proxy),
                building_intensity=float(getattr(row, 'building_intensity', default_features.building_intensity) or default_features.building_intensity),
                street_complexity=float(getattr(row, 'street_complexity', default_features.street_complexity) or default_features.street_complexity),
                elevation_complexity=float(getattr(row, 'elevation_complexity', default_features.elevation_complexity) or default_features.elevation_complexity),
                is_imputed=False,
            )
        return cls(features_by_cell, default_features=default_features)

    @classmethod
    def from_parquet(cls, path: Path) -> 'UrbanContextIndex':
        if not path.exists():
            return cls()
        return cls.from_frame(pd.read_parquet(path))

    @classmethod
    def load_default(cls, *, resolution: int = 9) -> 'UrbanContextIndex':
        return cls.from_parquet(PROCESSED_DIR / f'urban_context_h3_res{resolution}.parquet')

    def lookup(self, cell: str) -> UrbanContextFeatures:
        return self._features_by_cell.get(str(cell), self._default_features)

    def __bool__(self) -> bool:
        return bool(self._features_by_cell)


def _default_features_from_frame(frame: pd.DataFrame) -> UrbanContextFeatures:
    def _median(column: str, fallback: float) -> float:
        if column not in frame.columns:
            return fallback
        series = pd.to_numeric(frame[column], errors='coerce').dropna()
        if series.empty:
            return fallback
        return float(series.median())

    return UrbanContextFeatures(
        urban_clutter_index=_median('urban_clutter_index', 0.5),
        sidewalk_access_score=_median('sidewalk_access_score', 0.5),
        building_height_proxy=_median('building_height_proxy', 0.5),
        building_intensity=_median('building_intensity', 0.5),
        street_complexity=_median('street_complexity', 0.5),
        elevation_complexity=_median('elevation_complexity', 0.5),
        is_imputed=True,
    )
