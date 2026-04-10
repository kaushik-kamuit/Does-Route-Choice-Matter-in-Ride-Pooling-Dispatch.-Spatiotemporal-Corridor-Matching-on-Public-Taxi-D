from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .data_types import RendezvousOpportunity
from .selectors import DeterministicMeetingPointSelector


MATCH_GROUP_KEYS = [
    "domain",
    "scenario_name",
    "time_slice",
    "area_slice",
    "rider_density_pct",
    "occlusion_lambda",
    "meeting_k_ring",
    "observability_profile",
    "observability_ablation",
    "use_urban_context",
    "walk_penalty_per_min",
]


@dataclass(frozen=True)
class MatchedPairRule:
    max_candidate_diff: int = 2
    max_feasible_diff: int = 1
    max_distance_pct_diff: float = 0.10
    max_walk_min_diff: float = 0.5


def build_matched_observability_pairs(
    route_df: pd.DataFrame,
    opportunity_df: pd.DataFrame,
    *,
    seeds: Iterable[int],
    iterations: int = 5000,
    seed: int = 42,
    rule: MatchedPairRule | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if route_df.empty or opportunity_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rule = rule or MatchedPairRule()
    route_df = route_df.copy()
    opportunity_df = opportunity_df.copy()
    route_df = route_df.dropna(
        subset=[
            "route_cost",
            "mean_route_observability",
            "mean_route_walk_min",
            "candidate_count",
            "feasible_opportunity_count",
            "route_distance_miles",
        ]
    ).reset_index(drop=True)
    opportunity_df = opportunity_df.dropna(
        subset=[
            "route_cost",
            "fare_share",
            "passenger_count",
            "walk_min",
            "observability_score",
            "success_probability",
        ]
    ).reset_index(drop=True)
    if route_df.empty or opportunity_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    route_df["area_slice"] = route_df.get("area_slice", "all").fillna("all")
    opportunity_df["area_slice"] = opportunity_df.get("area_slice", "all").fillna("all")

    route_key_cols = [column for column in MATCH_GROUP_KEYS if column in route_df.columns]
    route_group_cols = route_key_cols + ["driver_id", "route_idx"]
    if "route_cost" not in route_df.columns:
        raise ValueError("Route evaluations must include route_cost for matched-pair analysis.")

    opportunity_map = _group_opportunities(opportunity_df)
    selector = DeterministicMeetingPointSelector(use_observability=True)
    rows: list[dict[str, object]] = []

    for group_values, driver_df in route_df.groupby(route_key_cols + ["driver_id"], sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_map = dict(zip(route_key_cols + ["driver_id"], group_values))
        driver_routes = driver_df.sort_values("route_idx").reset_index(drop=True)
        if len(driver_routes) < 2:
            continue
        for left_idx, right_idx in combinations(driver_routes.index.tolist(), 2):
            left = driver_routes.iloc[left_idx]
            right = driver_routes.iloc[right_idx]
            if not _is_matched_pair(left, right, rule):
                continue
            if float(left["mean_route_observability"]) == float(right["mean_route_observability"]):
                continue
            high = left if float(left["mean_route_observability"]) > float(right["mean_route_observability"]) else right
            low = right if high is left else left
            high_key = _route_key(high, route_key_cols)
            low_key = _route_key(low, route_key_cols)
            high_opportunities = opportunity_map.get(high_key, ())
            low_opportunities = opportunity_map.get(low_key, ())
            if not high_opportunities or not low_opportunities:
                continue
            for current_seed in seeds:
                high_profit = _route_actual_profit(
                    int(high["driver_id"]),
                    int(high["route_idx"]),
                    float(high["route_cost"]),
                    high_opportunities,
                    selector=selector,
                    seed=int(current_seed),
                )
                low_profit = _route_actual_profit(
                    int(low["driver_id"]),
                    int(low["route_idx"]),
                    float(low["route_cost"]),
                    low_opportunities,
                    selector=selector,
                    seed=int(current_seed),
                )
                rows.append(
                    {
                        **{key: group_map[key] for key in route_key_cols},
                        "driver_id": int(group_map["driver_id"]),
                        "seed": int(current_seed),
                        "high_route_idx": int(high["route_idx"]),
                        "low_route_idx": int(low["route_idx"]),
                        "candidate_count_diff": abs(int(high["candidate_count"]) - int(low["candidate_count"])),
                        "feasible_count_diff": abs(
                            int(high["feasible_opportunity_count"]) - int(low["feasible_opportunity_count"])
                        ),
                        "distance_pct_diff": _distance_pct_diff(
                            float(high["route_distance_miles"]),
                            float(low["route_distance_miles"]),
                        ),
                        "walk_min_diff": abs(float(high["mean_route_walk_min"]) - float(low["mean_route_walk_min"])),
                        "observability_gap": float(high["mean_route_observability"]) - float(low["mean_route_observability"]),
                        "high_observability": float(high["mean_route_observability"]),
                        "low_observability": float(low["mean_route_observability"]),
                        "high_route_distance_miles": float(high["route_distance_miles"]),
                        "low_route_distance_miles": float(low["route_distance_miles"]),
                        "high_route_walk_min": float(high["mean_route_walk_min"]),
                        "low_route_walk_min": float(low["mean_route_walk_min"]),
                        "high_actual_profit": float(high_profit),
                        "low_actual_profit": float(low_profit),
                        "profit_delta": float(high_profit - low_profit),
                        "higher_observability_wins": int(high_profit > low_profit),
                        "tie": int(abs(high_profit - low_profit) <= 1e-9),
                    }
                )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df, pd.DataFrame()
    summary_df = _summarize_matched_pairs(pair_df, route_key_cols=route_key_cols, iterations=iterations, seed=seed)
    return pair_df, summary_df


def select_case_studies(
    matched_pairs: pd.DataFrame,
    *,
    preferred_yellow: int = 6,
    preferred_green: int = 2,
    total_cases: int = 8,
) -> pd.DataFrame:
    if matched_pairs.empty:
        return pd.DataFrame()
    pair_cols = [
        "domain",
        "scenario_name",
        "time_slice",
        "area_slice",
        "driver_id",
        "high_route_idx",
        "low_route_idx",
    ]
    ranked = (
        matched_pairs.groupby(pair_cols, as_index=False)
        .agg(
            mean_profit_delta=("profit_delta", "mean"),
            mean_observability_gap=("observability_gap", "mean"),
            win_rate=("higher_observability_wins", "mean"),
            mean_distance_pct_diff=("distance_pct_diff", "mean"),
            mean_walk_min_diff=("walk_min_diff", "mean"),
        )
        .sort_values(
            ["domain", "mean_observability_gap", "win_rate", "mean_profit_delta"],
            ascending=[True, False, False, False],
        )
    )
    chosen_rows: list[pd.DataFrame] = []
    chosen_keys: set[tuple[object, ...]] = set()

    def _append_focus(frame: pd.DataFrame, limit: int | None = None) -> None:
        nonlocal chosen_rows, chosen_keys
        if frame.empty:
            return
        frame = frame[
            ~frame.apply(lambda row: tuple(row[column] for column in pair_cols) in chosen_keys, axis=1)
        ]
        if frame.empty:
            return
        if limit is not None:
            frame = frame.head(limit)
        if frame.empty:
            return
        chosen_rows.append(frame)
        for _, row in frame.iterrows():
            chosen_keys.add(tuple(row[column] for column in pair_cols))

    yellow_focus = ranked[
        (ranked["domain"] == "yellow")
        & (ranked["scenario_name"] == "sparse_high_occlusion")
        & (ranked["time_slice"] == "morning_peak")
    ]
    _append_focus(yellow_focus, preferred_yellow)

    yellow_all_day = ranked[
        (ranked["domain"] == "yellow")
        & (ranked["scenario_name"] == "sparse_high_occlusion")
        & (ranked["time_slice"] == "all_day")
    ]
    _append_focus(yellow_all_day, max(preferred_yellow - sum(len(frame) for frame in chosen_rows if (frame["domain"] == "yellow").all()), 0))

    green_focus = ranked[
        (ranked["domain"] == "green")
        & (ranked["scenario_name"] == "sparse_high_occlusion")
        & (ranked["time_slice"] == "all_day")
    ]
    _append_focus(green_focus, preferred_green)

    combined = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else pd.DataFrame(columns=ranked.columns)
    if len(combined) < total_cases:
        remainder = ranked[
            ~ranked.apply(lambda row: tuple(row[column] for column in pair_cols) in chosen_keys, axis=1)
        ].head(total_cases - len(combined))
        if not remainder.empty:
            combined = pd.concat([combined, remainder], ignore_index=True)
    return combined.head(total_cases).reset_index(drop=True)


def _group_opportunities(df: pd.DataFrame) -> dict[tuple[object, ...], tuple[RendezvousOpportunity, ...]]:
    key_cols = [column for column in MATCH_GROUP_KEYS if column in df.columns] + ["driver_id", "route_idx"]
    grouped: dict[tuple[object, ...], tuple[RendezvousOpportunity, ...]] = {}
    for key_values, group_df in df.groupby(key_cols, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        opportunities = tuple(
            RendezvousOpportunity(
                rider_id=int(row.rider_id),
                anchor_cell=str(row.anchor_cell),
                anchor_idx=int(row.anchor_idx),
                pickup_h3=str(row.pickup_h3),
                dropoff_h3=str(row.dropoff_h3),
                fare_share=float(row.fare_share),
                passenger_count=int(row.passenger_count),
                walk_m=float(row.walk_m),
                walk_min=float(row.walk_min),
                anchor_progress=float(row.anchor_progress),
                travel_fraction=float(row.travel_fraction),
                ambiguity_count=int(row.ambiguity_count),
                local_straightness=float(row.local_straightness),
                turn_severity=float(row.turn_severity),
                anchor_clutter=float(row.anchor_clutter),
                urban_clutter_index=float(row.urban_clutter_index),
                sidewalk_access_score=float(row.sidewalk_access_score),
                building_height_proxy=float(row.building_height_proxy),
                context_is_imputed=_as_bool(row.context_is_imputed),
                observability_score=float(row.observability_score),
                success_probability=float(row.success_probability),
            )
            for row in group_df.itertuples(index=False)
        )
        grouped[tuple(key_values)] = opportunities
    return grouped


def _route_actual_profit(
    driver_id: int,
    route_idx: int,
    route_cost: float,
    opportunities: tuple[RendezvousOpportunity, ...],
    *,
    selector: DeterministicMeetingPointSelector,
    seed: int,
    seats: int = 3,
) -> float:
    selected = selector.select(opportunities, seats=seats)
    realized_revenue = 0.0
    for opportunity in selected:
        if _stable_uniform(driver_id, route_idx, opportunity.rider_id, opportunity.anchor_idx, seed) <= opportunity.success_probability:
            realized_revenue += opportunity.fare_share
    return float(realized_revenue - route_cost)


def _route_key(row: pd.Series, route_key_cols: list[str]) -> tuple[object, ...]:
    return tuple(row[column] for column in route_key_cols) + (int(row["driver_id"]), int(row["route_idx"]))


def _is_matched_pair(left: pd.Series, right: pd.Series, rule: MatchedPairRule) -> bool:
    if abs(int(left["candidate_count"]) - int(right["candidate_count"])) > rule.max_candidate_diff:
        return False
    if abs(int(left["feasible_opportunity_count"]) - int(right["feasible_opportunity_count"])) > rule.max_feasible_diff:
        return False
    if _distance_pct_diff(float(left["route_distance_miles"]), float(right["route_distance_miles"])) > rule.max_distance_pct_diff:
        return False
    if abs(float(left["mean_route_walk_min"]) - float(right["mean_route_walk_min"])) > rule.max_walk_min_diff:
        return False
    return True


def _distance_pct_diff(left: float, right: float) -> float:
    baseline = max(min(abs(left), abs(right)), 1e-9)
    return abs(left - right) / baseline


def _summarize_matched_pairs(
    df: pd.DataFrame,
    *,
    route_key_cols: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    group_cols = route_key_cols
    for group_values, group_df in df.groupby(group_cols, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        deltas = group_df["profit_delta"].to_numpy(dtype=float)
        if len(deltas) == 0:
            continue
        draws = rng.choice(deltas, size=(max(int(iterations), 1), len(deltas)), replace=True).mean(axis=1)
        effect_size = float(np.mean(deltas) / max(np.std(deltas, ddof=1), 1e-9)) if len(deltas) > 1 else np.nan
        summary_rows.append(
            {
                **{key: value for key, value in zip(group_cols, group_values)},
                "metric": "profit_delta",
                "n_seed_pairs": int(len(group_df)),
                "n_unique_pairs": int(
                    group_df[["driver_id", "high_route_idx", "low_route_idx"]].drop_duplicates().shape[0]
                ),
                "mean_profit_delta": float(np.mean(deltas)),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "higher_observability_win_rate": float(np.mean(group_df["higher_observability_wins"].to_numpy(dtype=float))),
                "tie_rate": float(np.mean(group_df["tie"].to_numpy(dtype=float))),
                "loss_rate": float(
                    1.0
                    - np.mean(group_df["higher_observability_wins"].to_numpy(dtype=float))
                    - np.mean(group_df["tie"].to_numpy(dtype=float))
                ),
                "mean_observability_gap": float(np.mean(group_df["observability_gap"].to_numpy(dtype=float))),
                "mean_distance_pct_diff": float(np.mean(group_df["distance_pct_diff"].to_numpy(dtype=float))),
                "mean_walk_min_diff": float(np.mean(group_df["walk_min_diff"].to_numpy(dtype=float))),
                "effect_size": effect_size,
            }
        )
    return pd.DataFrame(summary_rows)


def _stable_uniform(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
