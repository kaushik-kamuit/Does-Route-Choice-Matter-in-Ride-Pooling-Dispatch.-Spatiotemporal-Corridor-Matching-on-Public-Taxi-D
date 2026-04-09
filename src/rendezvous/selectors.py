from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .data_types import RendezvousOpportunity

FEATURE_NAMES = [
    "walk_min",
    "anchor_progress",
    "travel_fraction",
    "ambiguity_count",
    "local_straightness",
    "turn_severity",
    "anchor_clutter",
    "urban_clutter_index",
    "sidewalk_access_score",
    "building_height_proxy",
    "context_is_imputed",
]


class MeetingPointSelector(ABC):
    @abstractmethod
    def opportunity_value(self, opportunity: RendezvousOpportunity) -> float:
        raise NotImplementedError

    def select(self, opportunities: Iterable[RendezvousOpportunity], *, seats: int) -> list[RendezvousOpportunity]:
        best_per_rider: dict[int, RendezvousOpportunity] = {}
        for opportunity in opportunities:
            current = best_per_rider.get(opportunity.rider_id)
            if current is None or self.opportunity_value(opportunity) > self.opportunity_value(current):
                best_per_rider[opportunity.rider_id] = opportunity

        ranked = sorted(
            best_per_rider.values(),
            key=lambda opportunity: (self.opportunity_value(opportunity), -opportunity.anchor_idx),
            reverse=True,
        )
        if seats <= 0 or not ranked:
            return []

        best_by_capacity: dict[int, tuple[float, tuple[RendezvousOpportunity, ...]]] = {0: (0.0, tuple())}
        for opportunity in ranked:
            weight = max(int(opportunity.passenger_count), 0)
            if weight <= 0 or weight > seats:
                continue
            for used_capacity, (used_value, used_selection) in sorted(best_by_capacity.items(), reverse=True):
                new_capacity = used_capacity + weight
                if new_capacity > seats:
                    continue
                new_value = used_value + self.opportunity_value(opportunity)
                new_selection = used_selection + (opportunity,)
                current = best_by_capacity.get(new_capacity)
                if current is None or _selection_key(new_value, new_selection) > _selection_key(*current):
                    best_by_capacity[new_capacity] = (new_value, new_selection)

        best_value, best_selection = max(
            best_by_capacity.values(),
            key=lambda item: _selection_key(item[0], item[1]),
        )
        _ = best_value
        return list(best_selection)


class DeterministicMeetingPointSelector(MeetingPointSelector):
    def __init__(self, *, use_observability: bool) -> None:
        self.use_observability = use_observability

    def opportunity_value(self, opportunity: RendezvousOpportunity) -> float:
        if self.use_observability:
            return opportunity.observable_value
        return opportunity.nominal_value


class WalkAwareMeetingPointSelector(MeetingPointSelector):
    def __init__(self, *, walk_penalty_per_min: float = 0.5) -> None:
        self.walk_penalty_per_min = max(0.0, float(walk_penalty_per_min))

    def opportunity_value(self, opportunity: RendezvousOpportunity) -> float:
        return max(0.0, opportunity.fare_share - self.walk_penalty_per_min * opportunity.walk_min)


class MLMeetingPointSelector(MeetingPointSelector):
    def __init__(self, model: GradientBoostingRegressor | None = None) -> None:
        self.model = model if model is not None else GradientBoostingRegressor(
            random_state=42,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
        )
        self._is_fit = model is not None

    def fit(self, opportunities: Iterable[RendezvousOpportunity]) -> "MLMeetingPointSelector":
        rows = list(opportunities)
        if not rows:
            return self
        x = pd.DataFrame([feature_vector(row) for row in rows], columns=FEATURE_NAMES)
        y = np.asarray([row.success_probability for row in rows], dtype=float)
        self.model.fit(x, y)
        self._is_fit = True
        return self

    def opportunity_value(self, opportunity: RendezvousOpportunity) -> float:
        if not self._is_fit:
            return opportunity.observable_value
        x = pd.DataFrame([feature_vector(opportunity)], columns=FEATURE_NAMES)
        prediction = float(self.model.predict(x)[0])
        probability = max(0.0, min(1.0, prediction))
        return opportunity.fare_share * probability

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "MLMeetingPointSelector":
        return cls(joblib.load(path))


def feature_vector(opportunity: RendezvousOpportunity) -> list[float]:
    return [
        opportunity.walk_min,
        opportunity.anchor_progress,
        opportunity.travel_fraction,
        float(opportunity.ambiguity_count),
        opportunity.local_straightness,
        opportunity.turn_severity,
        opportunity.anchor_clutter,
        opportunity.urban_clutter_index,
        opportunity.sidewalk_access_score,
        opportunity.building_height_proxy,
        float(opportunity.context_is_imputed),
    ]


def _selection_key(
    total_value: float,
    selection: tuple[RendezvousOpportunity, ...],
) -> tuple[float, int, int, tuple[int, ...]]:
    rider_signature = tuple(sorted((item.rider_id for item in selection), reverse=True))
    anchor_signature = sum(item.anchor_idx for item in selection)
    return (round(float(total_value), 12), len(selection), -anchor_signature, rider_signature)
