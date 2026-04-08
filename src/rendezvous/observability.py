from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservabilityWeights:
    straightness: float = 0.25
    turn: float = 0.25
    ambiguity: float = 0.25
    clutter: float = 0.25


DEFAULT_WEIGHTS = ObservabilityWeights()


def compute_observability_score(
    *,
    local_straightness: float,
    turn_severity: float,
    ambiguity_count: int,
    anchor_clutter: float,
    weights: ObservabilityWeights = DEFAULT_WEIGHTS,
) -> float:
    straightness_score = _clip01(local_straightness)
    turn_score = 1.0 - _clip01(turn_severity)
    ambiguity_score = 1.0 / max(int(ambiguity_count), 1)
    clutter_score = 1.0 / (1.0 + max(anchor_clutter, 0.0))

    weighted = (
        weights.straightness * straightness_score
        + weights.turn * turn_score
        + weights.ambiguity * ambiguity_score
        + weights.clutter * clutter_score
    )
    total_weight = (
        weights.straightness
        + weights.turn
        + weights.ambiguity
        + weights.clutter
    )
    if total_weight <= 1e-9:
        return 0.0
    return _clip01(weighted / total_weight)


def pickup_success_probability(
    observability_score: float,
    *,
    occlusion_lambda: float,
    base_success: float = 0.95,
    min_success: float = 0.35,
) -> float:
    probability = base_success - occlusion_lambda * (1.0 - _clip01(observability_score))
    return max(min_success, min(base_success, probability))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
