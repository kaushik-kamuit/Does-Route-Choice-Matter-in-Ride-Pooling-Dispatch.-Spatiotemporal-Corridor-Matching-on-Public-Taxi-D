from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.reporting import summarize_dispatch, summarize_driver_outcomes, write_result_views


def _load_many(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((ROOT / "results").glob(pattern))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    results_dir = ROOT / "results"
    driver_outcomes = _load_many("rendezvous_driver_outcomes*.csv")
    dispatch_summary = _load_many("rendezvous_dispatch_summary*.csv")

    driver_summary = summarize_driver_outcomes(driver_outcomes)
    dispatch_policy_summary = summarize_dispatch(dispatch_summary)
    write_result_views(results_dir, driver_summary, dispatch_policy_summary)

    if not driver_summary.empty:
        driver_summary.to_csv(results_dir / "rendezvous_policy_summary.csv", index=False)
    if not dispatch_policy_summary.empty:
        dispatch_policy_summary.to_csv(results_dir / "rendezvous_dispatch_policy_summary.csv", index=False)


if __name__ == "__main__":
    main()
