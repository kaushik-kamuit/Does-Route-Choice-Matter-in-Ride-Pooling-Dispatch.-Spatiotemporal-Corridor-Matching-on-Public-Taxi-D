from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.selectors import FEATURE_NAMES, MLMeetingPointSelector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ML meeting-point comparator")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--dataset", type=str, default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else ROOT / "data" / "ml" / args.domain / "rendezvous_meeting_point_dataset.parquet"
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    if df.empty:
        raise SystemExit(f"Dataset is empty: {dataset_path}. Rebuild with a larger sample or use --fetch.")
    selector = MLMeetingPointSelector()
    selector.model.fit(df[FEATURE_NAMES], df["success_probability"])
    selector_path = ROOT / "models" / f"rendezvous_meeting_point_model_{args.domain}.joblib"
    selector.save(selector_path)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": selector.model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_path = ROOT / "models" / f"rendezvous_meeting_point_feature_importance_{args.domain}.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_path, index=False)
    print(f"Saved model to {selector_path}")
    print(f"Saved feature importance to {importance_path}")


if __name__ == "__main__":
    main()
