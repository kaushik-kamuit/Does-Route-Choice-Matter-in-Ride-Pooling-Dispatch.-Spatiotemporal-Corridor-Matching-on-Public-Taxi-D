from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

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
    target_col = "success_probability" if "success_probability" in df.columns else "observed_success"
    train_df, valid_df, test_df = _split_dataset(df)
    if train_df.empty:
        raise SystemExit(f"Training split is empty in {dataset_path}")

    selector = MLMeetingPointSelector()
    selector.model.fit(train_df[FEATURE_NAMES], train_df[target_col])
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
    metrics_rows = []
    metrics_json: dict[str, dict[str, float | int | None]] = {}
    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if split_df.empty:
            continue
        preds = np.clip(selector.model.predict(split_df[FEATURE_NAMES]), 0.0, 1.0)
        target = split_df[target_col].to_numpy(dtype=float)
        metrics = {
            "n_rows": int(len(split_df)),
            "target_mean": float(np.mean(target)),
            "mae": float(mean_absolute_error(target, preds)),
            "rmse": float(mean_squared_error(target, preds) ** 0.5),
        }
        if "success_probability" in split_df.columns and target_col != "success_probability":
            proxy = split_df["success_probability"].to_numpy(dtype=float)
            metrics["proxy_rmse"] = float(mean_squared_error(proxy, preds) ** 0.5)
        if "observed_success" in split_df.columns:
            observed = split_df["observed_success"].to_numpy(dtype=float)
            metrics["observed_rmse"] = float(mean_squared_error(observed, preds) ** 0.5)
            metrics["observed_mae"] = float(mean_absolute_error(observed, preds))
            if len(np.unique(observed)) > 1:
                metrics["observed_roc_auc"] = float(roc_auc_score(observed, preds))
            else:
                metrics["observed_roc_auc"] = None
        metrics_json[split_name] = metrics
        metrics_rows.append({"split": split_name, **metrics})

    metrics_path = ROOT / "models" / f"rendezvous_meeting_point_metrics_{args.domain}.json"
    metrics_csv_path = ROOT / "models" / f"rendezvous_meeting_point_metrics_{args.domain}.csv"
    metrics_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)
    print(f"Saved model to {selector_path}")
    print(f"Saved feature importance to {importance_path}")
    print(f"Saved evaluation metrics to {metrics_path}")


def _split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "dataset_split" in df.columns:
        return (
            df[df["dataset_split"] == "train"].reset_index(drop=True),
            df[df["dataset_split"] == "valid"].reset_index(drop=True),
            df[df["dataset_split"] == "test"].reset_index(drop=True),
        )

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    n_train = max(int(n * 0.7), 1)
    n_valid = max(int(n * 0.2), 1) if n >= 3 else 0
    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    valid_df = shuffled.iloc[n_train : n_train + n_valid].reset_index(drop=True)
    test_df = shuffled.iloc[n_train + n_valid :].reset_index(drop=True)
    return train_df, valid_df, test_df


if __name__ == "__main__":
    main()
