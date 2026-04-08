"""
ML model insight plots.

1. Feature importance (horizontal bar)
2. Predicted vs actual profit scatter (held-out validation split only)
3. Route rank accuracy analysis (held-out validation split only)

Uses GroupShuffleSplit by driver_id to match the training script's split,
ensuring evaluation is on truly unseen drivers.

Usage:
    python old-scripts/plot_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET_PATH = ROOT / "data" / "ml" / "training_dataset_v2.parquet"
MODEL_PATH = ROOT / "models" / "profit_model_v2.pkl"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance_v2.csv"
PLOTS_DIR = ROOT / "results" / "plots"

FEATURE_COLS = [
    "route_distance_m",
    "route_duration_s",
    "corridor_cell_count",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "corridor_rider_count",
    "corridor_demand_density",
    "mean_rider_fare",
    "corridor_fare_density",
    "day_of_month",
    "time_bin_15min",
    "hour_sin",
    "hour_cos",
    "route_sinuosity",
    "route_avg_speed_ms",
    "bearing_sin",
    "bearing_cos",
    "straight_line_dist_m",
    "origin_landmark_dist_km",
    "dest_landmark_dist_km",
    "origin_jfk_km",
    "origin_lga_km",
    "origin_penn_km",
    "origin_times_sq_km",
    "dest_jfk_km",
    "dest_lga_km",
    "dest_penn_km",
    "dest_times_sq_km",
    "corridor_hist_pickups",
    "corridor_hist_dropoffs",
    "corridor_hist_pickup_density",
    "corridor_hist_dropoff_density",
    "corridor_hist_mean_fare",
    "corridor_hist_fare_density",
    "origin_cell_pickups",
    "origin_cell_mean_fare",
    "dest_cell_dropoffs",
]

SPLIT_SEED = 42
VALIDATION_FRAC = 0.20

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def _load_val_split() -> pd.DataFrame:
    """Load dataset and return only the held-out validation partition."""
    df = pd.read_parquet(DATASET_PATH)
    X = df[FEATURE_COLS].values
    y = df["expected_profit"].values
    groups = df["driver_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRAC, random_state=SPLIT_SEED)
    _train_idx, val_idx = next(gss.split(X, y, groups=groups))
    return df.iloc[val_idx].reset_index(drop=True)


def plot_feature_importance() -> None:
    imp = pd.read_csv(IMPORTANCE_PATH).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("LightGBM Feature Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(val_df: pd.DataFrame) -> None:
    model = joblib.load(MODEL_PATH)

    X = val_df[FEATURE_COLS].values
    y_true = val_df["expected_profit"].values
    y_pred = model.predict(X)

    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 50_000, replace=False)
        y_true_s, y_pred_s = y_true[idx], y_pred[idx]
    else:
        y_true_s, y_pred_s = y_true, y_pred

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_s, y_pred_s, alpha=0.15, s=4, color="#4C72B0")

    lo = min(y_true_s.min(), y_pred_s.min())
    hi = max(y_true_s.max(), y_pred_s.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect")

    ax.set_xlabel("Actual Profit ($)")
    ax.set_ylabel("Predicted Profit ($)")
    ax.set_title(f"Predicted vs Actual Profit — Held-Out Val (R² = {r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "predicted_vs_actual.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Validation R²: {r2:.4f}")


def plot_rank_accuracy(val_df: pd.DataFrame) -> None:
    """For each driver in the validation set with multiple routes, check if
    the ML-predicted best route is actually the most profitable route."""
    model = joblib.load(MODEL_PATH)

    X = val_df[FEATURE_COLS].values
    val_df = val_df.copy()
    val_df["predicted_profit"] = model.predict(X)

    groups = val_df.groupby("driver_id")

    correct = 0
    total = 0
    rank_confusion = np.zeros((3, 3), dtype=int)

    for _, grp in groups:
        if len(grp) < 2:
            continue
        total += 1

        actual_rank = grp["expected_profit"].values.argsort()[::-1]
        pred_rank = grp["predicted_profit"].values.argsort()[::-1]

        if actual_rank[0] == pred_rank[0]:
            correct += 1

        n_routes = min(len(grp), 3)
        for r in range(n_routes):
            actual_pos = int(np.where(actual_rank == r)[0][0])
            pred_pos = int(np.where(pred_rank == r)[0][0])
            if actual_pos < 3 and pred_pos < 3:
                rank_confusion[pred_pos, actual_pos] += 1

    accuracy = correct / total if total > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(["Correct", "Incorrect"], [accuracy * 100, (1 - accuracy) * 100],
           color=["#55a868", "#c44e52"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Percentage")
    ax.set_title(f"Top-1 Route Selection Accuracy: {accuracy:.1%}")
    ax.set_ylim(0, 100)

    ax = axes[1]
    im = ax.imshow(rank_confusion, cmap="Blues")
    ax.set_xlabel("Actual Rank")
    ax.set_ylabel("Predicted Rank")
    ax.set_title("Rank Confusion Matrix (Held-Out Drivers)")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["1st", "2nd", "3rd"])
    ax.set_yticklabels(["1st", "2nd", "3rd"])

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(rank_confusion[i, j]),
                    ha="center", va="center", fontsize=12,
                    color="white" if rank_confusion[i, j] > rank_confusion.max() * 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("ML Route Ranking Quality (Validation Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rank_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  Rank-1 accuracy: {accuracy:.1%} ({correct:,}/{total:,})")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_feature_importance()
    print("  [1/3] Feature importance")

    val_df = _load_val_split()
    print(f"  Validation split: {len(val_df):,} rows")

    plot_predicted_vs_actual(val_df)
    print("  [2/3] Predicted vs actual (held-out)")

    plot_rank_accuracy(val_df)
    print("  [3/3] Rank accuracy (held-out)")

    print(f"\n  All ML plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
