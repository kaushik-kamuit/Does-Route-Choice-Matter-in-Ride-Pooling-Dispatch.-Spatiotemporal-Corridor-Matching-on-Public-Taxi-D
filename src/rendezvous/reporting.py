from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_driver_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = ["policy", "domain", "scenario_name", "rider_density_pct", "occlusion_lambda", "meeting_k_ring"]
    return (
        df.groupby(keys, as_index=False)
        .agg(
            mean_actual_profit=("actual_profit", "mean"),
            mean_expected_value=("expected_value", "mean"),
            mean_successful_riders=("successful_riders", "mean"),
            mean_attempted_riders=("attempted_riders", "mean"),
            mean_nominal_realized_gap=("nominal_realized_gap", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
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
    keys = ["policy", "domain", "scenario_name", "rider_density_pct", "occlusion_lambda", "meeting_k_ring"]
    return (
        df.groupby(keys, as_index=False)
        .agg(
            mean_profit_per_driver=("profit_per_driver", "mean"),
            mean_total_profit=("total_profit", "mean"),
            mean_service_rate=("service_rate", "mean"),
            mean_wait_min=("mean_wait_min", "mean"),
            mean_walk_min=("mean_walk_min", "mean"),
            mean_observability=("mean_observability", "mean"),
            n_runs=("seed", "count"),
        )
        .sort_values(keys)
    )


def write_result_views(results_dir: Path, driver_summary: pd.DataFrame, dispatch_summary: pd.DataFrame | None = None) -> None:
    if not driver_summary.empty:
        primary = driver_summary[driver_summary["scenario_name"] == "primary"].copy()
        if not primary.empty:
            primary.to_csv(results_dir / "rendezvous_primary_summary.csv", index=False)

            gap = (
                primary.groupby("policy", as_index=False)["mean_nominal_realized_gap"]
                .mean()
                .sort_values("mean_nominal_realized_gap", ascending=False)
            )
            gap.to_csv(results_dir / "rendezvous_nominal_realized_gap.csv", index=False)

            comparator = primary[primary["policy"].isin(["rendezvous_observable", "ml_meeting_point_comparator"])].copy()
            if not comparator.empty:
                comparator.to_csv(results_dir / "rendezvous_meeting_point_comparison.csv", index=False)

        sensitivity = driver_summary.sort_values(["occlusion_lambda", "policy"])
        sensitivity.to_csv(results_dir / "rendezvous_occlusion_sensitivity.csv", index=False)

    if dispatch_summary is not None and not dispatch_summary.empty:
        dispatch_summary.to_csv(results_dir / "rendezvous_dispatch_policy_summary.csv", index=False)
