from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_prep.urban_context import PROCESSED_DIR


@dataclass(frozen=True)
class UrbanContextFeatures:
    urban_clutter_index: float = 0.0
    sidewalk_access_score: float = 1.0
    building_height_proxy: float = 0.0
    building_intensity: float = 0.0
    street_complexity: float = 0.0
    elevation_complexity: float = 0.0


class UrbanContextIndex:
    def __init__(self, features_by_cell: dict[str, UrbanContextFeatures] | None = None) -> None:
        self._features_by_cell = features_by_cell or {}

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> 'UrbanContextIndex':
        features_by_cell: dict[str, UrbanContextFeatures] = {}
        if frame.empty:
            return cls(features_by_cell)
        required = {'h3_cell'}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Urban context frame is missing required columns: {sorted(missing)}')
        for row in frame.itertuples(index=False):
            features_by_cell[str(row.h3_cell)] = UrbanContextFeatures(
                urban_clutter_index=float(getattr(row, 'urban_clutter_index', 0.0) or 0.0),
                sidewalk_access_score=float(getattr(row, 'sidewalk_access_score', 1.0) or 1.0),
                building_height_proxy=float(getattr(row, 'building_height_proxy', 0.0) or 0.0),
                building_intensity=float(getattr(row, 'building_intensity', 0.0) or 0.0),
                street_complexity=float(getattr(row, 'street_complexity', 0.0) or 0.0),
                elevation_complexity=float(getattr(row, 'elevation_complexity', 0.0) or 0.0),
            )
        return cls(features_by_cell)

    @classmethod
    def from_parquet(cls, path: Path) -> 'UrbanContextIndex':
        if not path.exists():
            return cls()
        return cls.from_frame(pd.read_parquet(path))

    @classmethod
    def load_default(cls, *, resolution: int = 9) -> 'UrbanContextIndex':
        return cls.from_parquet(PROCESSED_DIR / f'urban_context_h3_res{resolution}.parquet')

    def lookup(self, cell: str) -> UrbanContextFeatures:
        return self._features_by_cell.get(str(cell), UrbanContextFeatures())

    def __bool__(self) -> bool:
        return bool(self._features_by_cell)
