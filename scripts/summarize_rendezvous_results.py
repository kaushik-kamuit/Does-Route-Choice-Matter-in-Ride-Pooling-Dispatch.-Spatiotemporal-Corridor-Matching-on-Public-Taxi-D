from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.reporting import (
    bootstrap_mean_intervals,
    paired_policy_deltas,
    summarize_dispatch,
    summarize_driver_outcomes,
    write_result_views,
)


def _load_many(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((ROOT / "results").glob(pattern))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_json_rows(pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "results").glob(pattern)):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    results_dir = ROOT / "results"
    driver_outcomes = _load_many("rendezvous_driver_outcomes*.csv")
    dispatch_summary = _load_many("rendezvous_dispatch_summary*.csv")
    driver_run_stats = _load_json_rows("rendezvous_driver_run_stats*.json")
    dispatch_run_stats = _load_json_rows("rendezvous_dispatch_run_stats*.json")

    driver_summary = summarize_driver_outcomes(driver_outcomes)
    dispatch_policy_summary = summarize_dispatch(dispatch_summary)
    write_result_views(results_dir, driver_summary, dispatch_policy_summary)

    if not driver_summary.empty:
        driver_summary.to_csv(results_dir / "rendezvous_policy_summary.csv", index=False)
        driver_ci = bootstrap_mean_intervals(
            driver_outcomes,
            value_col="actual_profit",
            unit_cols=["driver_id", "seed"],
            iterations=1000,
        )
        if not driver_ci.empty:
            driver_ci.to_csv(results_dir / "rendezvous_policy_bootstrap_ci.csv", index=False)
        corridor_deltas = paired_policy_deltas(
            driver_outcomes,
            value_col="actual_profit",
            unit_cols=["driver_id", "seed"],
            reference_policy="corridor_only",
            iterations=1000,
        )
        if not corridor_deltas.empty:
            corridor_deltas.to_csv(results_dir / "rendezvous_pairwise_deltas_vs_corridor.csv", index=False)
        rendezvous_deltas = paired_policy_deltas(
            driver_outcomes,
            value_col="actual_profit",
            unit_cols=["driver_id", "seed"],
            reference_policy="rendezvous_only",
            iterations=1000,
        )
        if not rendezvous_deltas.empty:
            rendezvous_deltas.to_csv(results_dir / "rendezvous_pairwise_deltas_vs_rendezvous_only.csv", index=False)
        if "observability_ablation" in driver_summary.columns:
            ablation_summary = driver_summary[
                driver_summary["observability_ablation"] != "full"
            ].copy()
            if not ablation_summary.empty:
                ablation_summary.to_csv(results_dir / "rendezvous_observability_ablation_summary.csv", index=False)
    if not dispatch_policy_summary.empty:
        dispatch_policy_summary.to_csv(results_dir / "rendezvous_dispatch_policy_summary.csv", index=False)
        dispatch_ci = bootstrap_mean_intervals(
            dispatch_summary,
            value_col="profit_per_driver",
            unit_cols=["seed"],
            iterations=1000,
        )
        if not dispatch_ci.empty:
            dispatch_ci.to_csv(results_dir / "rendezvous_dispatch_bootstrap_ci.csv", index=False)
        dispatch_deltas = paired_policy_deltas(
            dispatch_summary,
            value_col="profit_per_driver",
            unit_cols=["seed"],
            reference_policy="corridor_only",
            iterations=1000,
        )
        if not dispatch_deltas.empty:
            dispatch_deltas.to_csv(results_dir / "rendezvous_dispatch_pairwise_deltas_vs_corridor.csv", index=False)
    if not driver_run_stats.empty:
        driver_run_stats.to_csv(results_dir / "rendezvous_driver_run_coverage.csv", index=False)
    if not dispatch_run_stats.empty:
        dispatch_run_stats.to_csv(results_dir / "rendezvous_dispatch_run_coverage.csv", index=False)


if __name__ == "__main__":
    main()
