from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings


@dataclass(frozen=True)
class ObservabilityWeights:
    straightness: float = 0.25
    turn: float = 0.25
    ambiguity: float = 0.25
    clutter: float = 0.25


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ObservabilityWeights()
_WARNED_FALLBACK_DOMAINS: set[str] = set()


def weights_for_mode(mode: str, base_weights: ObservabilityWeights | None = None) -> ObservabilityWeights:
    key = (mode or "full").strip().lower()
    base = base_weights or DEFAULT_WEIGHTS
    if key == "full":
        return base
    if key == "no_straightness":
        return ObservabilityWeights(
            straightness=0.0,
            turn=base.turn,
            ambiguity=base.ambiguity,
            clutter=base.clutter,
        )
    if key == "no_turn":
        return ObservabilityWeights(
            straightness=base.straightness,
            turn=0.0,
            ambiguity=base.ambiguity,
            clutter=base.clutter,
        )
    if key == "no_ambiguity":
        return ObservabilityWeights(
            straightness=base.straightness,
            turn=base.turn,
            ambiguity=0.0,
            clutter=base.clutter,
        )
    if key == "no_clutter":
        return ObservabilityWeights(
            straightness=base.straightness,
            turn=base.turn,
            ambiguity=base.ambiguity,
            clutter=0.0,
        )
    else:
        raise ValueError(
            f"Unsupported observability ablation '{mode}'. Expected one of: full, no_ambiguity, no_clutter, no_straightness, no_turn"
        )


def weights_for_profile(profile: str, *, domain: str) -> ObservabilityWeights:
    key = (profile or "equal").strip().lower()
    if key == "equal":
        return DEFAULT_WEIGHTS
    if key == "calibrated":
        return _load_calibrated_weights(domain)
    raise ValueError(f"Unsupported observability profile '{profile}'. Expected one of: calibrated, equal")


def _load_calibrated_weights(domain: str) -> ObservabilityWeights:
    path = ROOT / "models" / f"observability_weights_{domain}.json"
    if not path.exists():
        fallback = ROOT / "models" / "observability_weights_yellow.json"
        if domain != "yellow" and fallback.exists():
            if domain not in _WARNED_FALLBACK_DOMAINS:
                warnings.warn(
                    f"Calibrated observability weights not found for domain '{domain}'. "
                    f"Falling back to Yellow-calibrated weights at {fallback}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _WARNED_FALLBACK_DOMAINS.add(domain)
            path = fallback
        else:
            raise FileNotFoundError(f"Calibrated observability weights not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ObservabilityWeights(
        straightness=float(payload["weights"]["straightness"]),
        turn=float(payload["weights"]["turn"]),
        ambiguity=float(payload["weights"]["ambiguity"]),
        clutter=float(payload["weights"]["clutter"]),
    )


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
