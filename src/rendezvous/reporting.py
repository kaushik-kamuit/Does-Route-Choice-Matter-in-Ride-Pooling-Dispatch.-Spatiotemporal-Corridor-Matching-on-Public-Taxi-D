from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def summarize_driver_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = _normalize_grouping_columns(df)
    keys = _group_keys(df)
    return (
        df.groupby(keys, as_index=False)
        .agg(
            mean_actual_profit=("actual_profit", "mean"),
            mean_expected_value=("expected_value", "mean"),
            mean_successful_riders=("successful_riders", "mean"),
            mean_attempted_riders=("attempted_riders", "mean"),
            mean_nominal_realized_gap=("nominal_realized_gap", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_time_eligible_candidate_count=("time_eligible_candidate_count", "mean"),
            mean_feasible_opportunity_count=("feasible_opportunity_count", "mean"),
            mean_observable_opportunity_count=("observable_opportunity_count", "mean"),
            mean_walk_min=("mean_walk_min", "mean"),
            mean_observability=("mean_observability", "mean"),
            n_rows=("driver_id", "count"),
        )
        .sort_values(keys)
    )


def summarize_dispatch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = _normalize_grouping_columns(df)
    keys = _group_keys(df)
    return (
        df.groupby(keys, as_index=False)
        .agg(
            mean_profit_per_driver=("profit_per_driver", "mean"),
            mean_total_profit=("total_profit", "mean"),
            mean_service_rate=("service_rate", "mean"),
            mean_wait_min=("mean_wait_min", "mean"),
            mean_walk_min=("mean_walk_min", "mean"),
            mean_observability=("mean_observability", "mean"),
            mean_route_coverage_rate=("route_coverage_rate", "mean"),
            mean_drivers_skipped_no_route=("drivers_skipped_no_route", "mean"),
            mean_eligible_riders=("eligible_riders", "mean"),
            n_runs=("seed", "count"),
        )
        .sort_values(keys)
    )


def bootstrap_mean_intervals(
    df: pd.DataFrame,
    *,
    value_col: str,
    unit_cols: list[str],
    iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = _normalize_grouping_columns(df)
    keys = _group_keys(df)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group_values, group_df in df.groupby(keys, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        unit_df = (
            group_df.groupby(unit_cols, as_index=False)[value_col]
            .mean()
            .sort_values(unit_cols)
            .reset_index(drop=True)
        )
        values = unit_df[value_col].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        draw_count = max(int(iterations), 1)
        draws = rng.choice(values, size=(draw_count, len(values)), replace=True).mean(axis=1)
        row = {key: value for key, value in zip(keys, group_values)}
        row.update(
            {
                "metric": value_col,
                "unit_count": int(len(values)),
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_policy_deltas(
    df: pd.DataFrame,
    *,
    value_col: str,
    unit_cols: list[str],
    reference_policy: str,
    iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = _normalize_grouping_columns(df)
    keys = [key for key in _group_keys(df) if key != "policy"]
    group_cols = keys + unit_cols + ["policy"]
    reduced = (
        df[group_cols + [value_col]]
        .groupby(group_cols, as_index=False)[value_col]
        .mean()
    )
    pivot = reduced.pivot_table(index=keys + unit_cols, columns="policy", values=value_col)
    if reference_policy not in pivot.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index_values, row in pivot.iterrows():
        if not isinstance(index_values, tuple):
            index_values = (index_values,)
        index_map = dict(zip(keys + unit_cols, index_values))
        reference_value = row.get(reference_policy)
        if pd.isna(reference_value):
            continue
        for policy, policy_value in row.items():
            if policy == reference_policy or pd.isna(policy_value):
                continue
            rows.append(
                {
                    **{key: index_map[key] for key in keys},
                    **{key: index_map[key] for key in unit_cols},
                    "reference_policy": reference_policy,
                    "policy": policy,
                    "delta": float(policy_value - reference_value),
                }
            )

    if not rows:
        return pd.DataFrame()
    delta_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for group_values, group_df in delta_df.groupby(keys + ["reference_policy", "policy"], sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        deltas = group_df["delta"].to_numpy(dtype=float)
        draws = rng.choice(deltas, size=(max(int(iterations), 1), len(deltas)), replace=True).mean(axis=1)
        record = {key: value for key, value in zip(keys + ["reference_policy", "policy"], group_values)}
        record.update(
            {
                "metric": value_col,
                "unit_count": int(len(deltas)),
                "mean_delta": float(deltas.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "share_positive": float(np.mean(deltas > 0.0)),
                "share_zero": float(np.mean(np.isclose(deltas, 0.0))),
                "share_negative": float(np.mean(deltas < 0.0)),
                "share_nonnegative": float(np.mean(deltas >= 0.0)),
                "mean_abs_delta": float(np.mean(np.abs(deltas))),
                "effect_size": float(deltas.mean() / max(np.std(deltas, ddof=1), 1e-9)) if len(deltas) > 1 else np.nan,
            }
        )
        summary_rows.append(record)
    return pd.DataFrame(summary_rows)


def write_result_views(results_dir: Path, driver_summary: pd.DataFrame, dispatch_summary: pd.DataFrame | None = None) -> None:
    if not driver_summary.empty:
        primary = _default_slice(driver_summary)
        primary = primary[primary["scenario_name"] == "primary"].copy()
        if not primary.empty:
            primary.to_csv(results_dir / "rendezvous_primary_summary.csv", index=False)

            gap = (
                primary.groupby([column for column in ["domain", "policy"] if column in primary.columns], as_index=False)[
                    "mean_nominal_realized_gap"
                ]
                .mean()
                .sort_values(["domain", "mean_nominal_realized_gap"] if "domain" in primary.columns else ["mean_nominal_realized_gap"], ascending=[True, False] if "domain" in primary.columns else False)
            )
            gap.to_csv(results_dir / "rendezvous_nominal_realized_gap.csv", index=False)

            comparator = primary[primary["policy"].isin(["rendezvous_observable", "ml_meeting_point_comparator"])].copy()
            if not comparator.empty:
                comparator.to_csv(results_dir / "rendezvous_meeting_point_comparison.csv", index=False)

        sensitivity = _default_slice(driver_summary)
        sort_cols = [column for column in ["occlusion_lambda", "policy"] if column in sensitivity.columns]
        if sort_cols:
            sensitivity = sensitivity.sort_values(sort_cols)
        sensitivity.to_csv(results_dir / "rendezvous_occlusion_sensitivity.csv", index=False)

    if dispatch_summary is not None and not dispatch_summary.empty:
        _default_slice(dispatch_summary).to_csv(results_dir / "rendezvous_dispatch_policy_summary.csv", index=False)


def _default_slice(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.copy()
    if "time_slice" in subset.columns and "all_day" in set(subset["time_slice"].dropna().astype(str)):
        subset = subset[subset["time_slice"] == "all_day"]
    if "area_slice" in subset.columns and "all" in set(subset["area_slice"].dropna().astype(str)):
        subset = subset[subset["area_slice"] == "all"]
    if "observability_profile" in subset.columns and "calibrated" in set(subset["observability_profile"].dropna().astype(str)):
        subset = subset[subset["observability_profile"] == "calibrated"]
    if "observability_ablation" in subset.columns:
        subset = subset[subset["observability_ablation"] == "full"]
    if "use_urban_context" in subset.columns:
        subset = subset[subset["use_urban_context"] == True]  # noqa: E712
    return subset


def _group_keys(df: pd.DataFrame) -> list[str]:
    preferred = [
        "policy",
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
    return [column for column in preferred if column in df.columns]


def _normalize_grouping_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    defaults: dict[str, object] = {
        "time_slice": "all_day",
        "area_slice": "all",
        "observability_profile": "equal",
        "observability_ablation": "full",
        "use_urban_context": True,
        "walk_penalty_per_min": 0.5,
    }
    for column, default in defaults.items():
        if column in normalized.columns:
            normalized[column] = normalized[column].fillna(default)
    return normalized
