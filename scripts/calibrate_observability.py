from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.observability import pickup_success_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate observability weights on non-test opportunity data")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()

    dataset_path = (
        Path(args.dataset)
        if args.dataset
        else ROOT / "data" / "ml" / args.domain / "rendezvous_meeting_point_dataset.parquet"
    )
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    if df.empty:
        raise SystemExit(f"Dataset is empty: {dataset_path}")

    calibration_df = df[df["dataset_split"].isin(["train", "valid"])].reset_index(drop=True)
    if calibration_df.empty:
        raise SystemExit("Calibration split is empty; cannot calibrate observability weights.")

    candidate_weights = list(_grid_weights(step=args.grid_step))
    baseline_weights = {"straightness": 0.25, "turn": 0.25, "ambiguity": 0.25, "clutter": 0.25}
    best_weights = baseline_weights
    best_score = float("inf")
    for weights in candidate_weights:
        preds = _predicted_success(calibration_df, weights)
        score = brier_score_loss(calibration_df["observed_success"].to_numpy(dtype=float), preds)
        if score < best_score - 1e-12:
            best_score = score
            best_weights = weights

    metrics = {
        "equal": _evaluate_profile(df, baseline_weights),
        "calibrated": _evaluate_profile(df, best_weights),
    }
    output_json = {
        "domain": args.domain,
        "dataset_path": str(dataset_path),
        "grid_step": args.grid_step,
        "weights": best_weights,
        "metrics": metrics,
    }

    output_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"observability_weights_{args.domain}.json"
    csv_path = output_dir / f"observability_calibration_metrics_{args.domain}.csv"
    json_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")

    rows = []
    for profile_name, splits in metrics.items():
        for split_name, split_metrics in splits.items():
            rows.append({"profile": profile_name, "split": split_name, **split_metrics})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved calibrated weights to {json_path}")
    print(f"Saved calibration metrics to {csv_path}")


def _evaluate_profile(df: pd.DataFrame, weights: dict[str, float]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for split_name in ["train", "valid", "test"]:
        split_df = df[df["dataset_split"] == split_name].reset_index(drop=True)
        if split_df.empty:
            continue
        preds = _predicted_success(split_df, weights)
        observed = split_df["observed_success"].to_numpy(dtype=float)
        metrics: dict[str, float | int | None] = {
            "n_rows": int(len(split_df)),
            "mean_prediction": float(np.mean(preds)),
            "mean_observed": float(np.mean(observed)),
            "brier": float(brier_score_loss(observed, preds)),
        }
        if len(np.unique(observed)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(observed, preds))
        else:
            metrics["roc_auc"] = None
        result[split_name] = metrics
    return result


def _predicted_success(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    straightness = np.clip(df["local_straightness"].to_numpy(dtype=float), 0.0, 1.0)
    turn = 1.0 - np.clip(df["turn_severity"].to_numpy(dtype=float), 0.0, 1.0)
    ambiguity = 1.0 / np.maximum(df["ambiguity_count"].to_numpy(dtype=float), 1.0)
    clutter = 1.0 / (1.0 + np.maximum(df["anchor_clutter"].to_numpy(dtype=float), 0.0))
    weighted = (
        weights["straightness"] * straightness
        + weights["turn"] * turn
        + weights["ambiguity"] * ambiguity
        + weights["clutter"] * clutter
    )
    total = max(sum(weights.values()), 1e-9)
    observability = np.clip(weighted / total, 0.0, 1.0)
    lambdas = df["occlusion_lambda"].to_numpy(dtype=float) if "occlusion_lambda" in df.columns else np.full(len(df), 0.25)
    return np.asarray(
        [pickup_success_probability(score, occlusion_lambda=float(lambda_value)) for score, lambda_value in zip(observability, lambdas)],
        dtype=float,
    )


def _grid_weights(step: float) -> list[dict[str, float]]:
    ticks = max(int(round(1.0 / step)), 1)
    combinations = []
    for a, b, c in itertools.product(range(ticks + 1), repeat=3):
        d = ticks - a - b - c
        if d < 0:
            continue
        weights = {
            "straightness": a * step,
            "turn": b * step,
            "ambiguity": c * step,
            "clutter": d * step,
        }
        if abs(sum(weights.values()) - 1.0) <= 1e-9:
            combinations.append(weights)
    return combinations


if __name__ == "__main__":
    main()
