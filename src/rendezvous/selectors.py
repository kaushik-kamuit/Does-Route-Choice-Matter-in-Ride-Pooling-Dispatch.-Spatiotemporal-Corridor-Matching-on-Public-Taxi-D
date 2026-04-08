from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
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
    "observability_score",
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

        selected: list[RendezvousOpportunity] = []
        remaining = seats
        ranked = sorted(
            best_per_rider.values(),
            key=lambda opportunity: (self.opportunity_value(opportunity), -opportunity.anchor_idx),
            reverse=True,
        )
        for opportunity in ranked:
            if opportunity.passenger_count > remaining:
                continue
            remaining -= opportunity.passenger_count
            selected.append(opportunity)
            if remaining <= 0:
                break
        return selected


class DeterministicMeetingPointSelector(MeetingPointSelector):
    def __init__(self, *, use_observability: bool) -> None:
        self.use_observability = use_observability

    def opportunity_value(self, opportunity: RendezvousOpportunity) -> float:
        if self.use_observability:
            return opportunity.observable_value
        return opportunity.nominal_value


class MLMeetingPointSelector(MeetingPointSelector):
    def __init__(self, model: GradientBoostingRegressor | None = None) -> None:
        self.model = model if model is not None else GradientBoostingRegressor(random_state=42)
        self._is_fit = model is not None

    def fit(self, opportunities: Iterable[RendezvousOpportunity]) -> "MLMeetingPointSelector":
        rows = list(opportunities)
        if not rows:
            return self
        x = np.asarray([feature_vector(row) for row in rows], dtype=float)
        y = np.asarray([row.success_probability for row in rows], dtype=float)
        self.model.fit(x, y)
        self._is_fit = True
        return self

    def opportunity_value(self, opportunity: RendezvousOpportunity) -> float:
        if not self._is_fit:
            return opportunity.observable_value
        prediction = float(self.model.predict(np.asarray([feature_vector(opportunity)], dtype=float))[0])
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
        opportunity.observability_score,
    ]
