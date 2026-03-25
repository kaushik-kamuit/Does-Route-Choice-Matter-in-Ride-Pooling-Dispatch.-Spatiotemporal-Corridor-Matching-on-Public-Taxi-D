"""
Publication-quality figures for the paper (TR-C / IEEE T-ITS style).

Generates:
  Figure 1: Density vs warm-up advantage (main result, 95% CI, annotations)
  Figure 2: Strategy comparison — mean profit with 95% CIs
  Figure 3: Per-driver profit difference distribution + by route length
  Figure 4: Model quality — feature importance (top 15) + predicted vs actual
  Figure 5: Mean profit by strategy across densities
  Figure 6: Interactive HTML map — 3 alternate routes and H3 corridors (long route)

Usage:
    python visualizations/plot_paper_figures.py
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_PATH = ROOT / "models" / "profit_model_v2.pkl"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance_v2.csv"
DATASET_PATH = ROOT / "data" / "ml" / "training_dataset_v2.parquet"

sys.path.insert(0, str(ROOT / "src"))

STRATEGIES = ["coldstart", "random", "heuristic", "warmup", "oracle"]
STRATEGY_LABELS = {
    "coldstart": "Cold-Start",
    "random": "Random",
    "heuristic": "Heuristic",
    "warmup": "ML Warm-Up",
    "oracle": "Oracle",
}
PALETTE = {
    "coldstart": "#4C72B0",
    "random": "#8DA0CB",
    "heuristic": "#66C2A5",
    "warmup": "#DD8452",
    "oracle": "#E78AC3",
}
CATEGORY_ORDER = ["short", "medium", "long"]

# Feature list for model figure (must match train_profit_model.py)
FEATURE_COLS = [
    "route_distance_m", "route_duration_s", "corridor_cell_count",
    "hour_of_day", "day_of_week", "is_weekend", "corridor_rider_count",
    "corridor_demand_density", "mean_rider_fare", "corridor_fare_density",
    "day_of_month", "time_bin_15min", "hour_sin", "hour_cos", "route_sinuosity",
    "route_avg_speed_ms", "bearing_sin", "bearing_cos", "straight_line_dist_m",
    "origin_landmark_dist_km", "dest_landmark_dist_km", "origin_jfk_km",
    "origin_lga_km", "origin_penn_km", "origin_times_sq_km", "dest_jfk_km",
    "dest_lga_km", "dest_penn_km", "dest_times_sq_km", "corridor_hist_pickups",
    "corridor_hist_dropoffs", "corridor_hist_pickup_density",
    "corridor_hist_dropoff_density", "corridor_hist_mean_fare",
    "corridor_hist_fare_density", "origin_cell_pickups", "origin_cell_mean_fare",
    "dest_cell_dropoffs",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def _load_strategy(name: str, suffix: str = "") -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{name}_outcomes{suffix}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_all(suffix: str = "") -> dict[str, pd.DataFrame]:
    return {s: _load_strategy(s, suffix) for s in STRATEGIES if _load_strategy(s, suffix) is not None}


# ---------------------------------------------------------------------------
# Figure 1: Density vs warm-up advantage (main result)
# ---------------------------------------------------------------------------
def fig1_density_advantage() -> None:
    path = RESULTS_DIR / "density_results.csv"
    if not path.exists():
        print("  [Fig 1] Skip: density_results.csv not found")
        return
    df = pd.read_csv(path).sort_values("density_pct", ascending=False)
    if len(df) < 2:
        print("  [Fig 1] Skip: need >= 2 density levels")
        return

    ci = df["delta_sem"].values * 1.96
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        df["density_pct"], df["delta_mean"], yerr=ci,
        fmt="o-", color="#DD8452", capsize=5, capthick=1.2, linewidth=2, markersize=9
    )
    # Annotate every density so the difference is clear at each level
    for i, row in df.iterrows():
        pct = row["delta_pct"]
        ci_val = ci[df.index.get_loc(i)]
        y_annot = row["delta_mean"] + ci_val + 0.12
        ax.annotate(
            f"+${row['delta_mean']:.2f} ({pct:.1f}%)",
            xy=(row["density_pct"], y_annot),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax.set_xlabel("Rider density (%)")
    ax.set_ylabel("Mean profit difference: Warm-Up − Cold-Start ($)")
    ax.set_title("Warm-up advantage over cold-start increases at lower rider density")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(df["density_pct"].max() + 5, df["density_pct"].min() - 5)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_fig1_density_advantage.png", bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 1] paper_fig1_density_advantage.png")


# ---------------------------------------------------------------------------
# Figure 2: Strategy comparison at each density (one panel per density)
# ---------------------------------------------------------------------------
def fig2_strategy_comparison() -> None:
    densities = [100, 75, 50, 25, 10]
    # Build (density, strategy -> mean, sem) for all available
    data = []
    for d in densities:
        suffix = f"_d{d}" if d < 100 else ""
        dfs = _load_all(suffix)
        if "coldstart" not in dfs or len(dfs) < 2:
            continue
        present = [s for s in STRATEGIES if s in dfs]
        for s in present:
            agg = dfs[s].groupby("driver_id")["profit"].mean()
            data.append({
                "density_pct": d,
                "strategy": s,
                "mean": agg.mean(),
                "sem": agg.sem(),
                "present_order": present,
            })
    if not data:
        print("  [Fig 2] Skip: need at least coldstart and one other strategy")
        return

    # One panel per density so difference is seen properly at each
    n_dens = len(densities)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat = axes.flat
    for idx, d in enumerate(densities):
        ax = axes_flat[idx]
        row = next((r for r in data if r["density_pct"] == d and r["strategy"] == "coldstart"), None)
        if row is None:
            ax.set_visible(False)
            continue
        present = row["present_order"]
        means = []
        sems = []
        for s in present:
            r = next((x for x in data if x["density_pct"] == d and x["strategy"] == s), None)
            if r:
                means.append(r["mean"])
                sems.append(r["sem"])
        if not means:
            ax.set_visible(False)
            continue
        ci = [x * 1.96 for x in sems]
        colors = [PALETTE[s] for s in present]
        x = np.arange(len(present))
        ax.bar(x, means, yerr=ci, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        cs_mean = means[present.index("coldstart")] if "coldstart" in present else 0
        for i, (xi, m) in enumerate(zip(x, means)):
            delta = m - cs_mean
            sign = "+" if delta >= 0 else ""
            ax.text(xi, m + ci[i] + 0.3, f"{sign}${delta:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in present], rotation=15, ha="right")
        ax.set_ylabel("Mean profit ($)")
        ax.set_title(f"{d}% rider density")
        ax.axhline(cs_mean, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_ylim(0, None)
    # Hide unused subplot
    if n_dens < len(axes_flat):
        axes_flat[n_dens].set_visible(False)
    fig.suptitle("Strategy comparison: mean profit with 95% CI at each rider density (5,000 drivers)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_fig2_strategy_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 2] paper_fig2_strategy_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: Per-driver profit difference — interpretable (log-scale hist + violin)
# ---------------------------------------------------------------------------
def fig3_profit_difference_distribution() -> None:
    cs = _load_strategy("coldstart")
    wu = _load_strategy("warmup")
    if cs is None or wu is None:
        print("  [Fig 3] Skip: coldstart or warmup outcomes missing")
        return

    cs_agg = cs.groupby("driver_id")["profit"].mean()
    wu_agg = wu.groupby("driver_id")["profit"].mean()
    merged = pd.DataFrame({"cs": cs_agg, "wu": wu_agg}).dropna()
    delta = merged["wu"] - merged["cs"]
    mean_diff = delta.mean()

    pct_win = (delta > 0).mean() * 100
    pct_lose = (delta < 0).mean() * 100
    pct_tie = (delta == 0).mean() * 100
    n_zero = (delta == 0).sum()

    # Merge route_length_category from coldstart
    cs_first = cs.groupby("driver_id").first().reset_index()[["driver_id", "route_length_category"]]
    merged = merged.reset_index().merge(cs_first, on="driver_id")
    merged["delta"] = merged["wu"] - merged["cs"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: histogram with log y-axis so the tails are visible (not dominated by zero bar)
    ax = axes[0]
    bins = np.linspace(delta.quantile(0.01), delta.quantile(0.99), 45)
    counts, bin_edges, _ = ax.hist(
        delta, bins=bins, color="#DD8452", edgecolor="black", linewidth=0.3, alpha=0.85
    )
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(mean_diff, color="red", linewidth=1.5, linestyle="--", label=f"Mean = +${mean_diff:.2f}")
    # Summary so the story is clear: Better / Worse / Tied with counts
    ax.text(
        0.97, 0.95,
        f"Better off: {pct_win:.1f}%\nWorse off: {pct_lose:.1f}%\nTied (no change): {pct_tie:.1f}%\n  ({int(n_zero):,} drivers at $0)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
    )
    ax.set_xlabel("Profit difference: ML Warm-Up − Cold-Start ($)")
    ax.set_ylabel("Number of drivers (log scale)")
    ax.set_title("Per-driver profit difference distribution")
    ax.legend(loc="upper right")

    # Right: violin plot by route length (shows density; interpretable vs box + many outliers)
    ax = axes[1]
    by_cat = [merged[merged["route_length_category"] == c]["delta"].values for c in CATEGORY_ORDER]
    vp = ax.violinplot(
        by_cat, positions=[0, 1, 2], showmeans=True, showmedians=True,
        widths=0.72,
    )
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor("#DD8452")
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([c.capitalize() for c in CATEGORY_ORDER])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Profit difference ($)")
    ax.set_xlabel("Route length category")
    ax.set_title("By route length (violin = density)")
    fig.suptitle("Distribution of per-driver profit difference (Warm-Up − Cold-Start)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_fig3_profit_difference.png", bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 3] paper_fig3_profit_difference.png")


# ---------------------------------------------------------------------------
# Figure 4: Model quality — feature importance (top 15) + predicted vs actual
# ---------------------------------------------------------------------------
def fig4_model_quality() -> None:
    if not IMPORTANCE_PATH.exists():
        print("  [Fig 4] Skip: feature_importance_v2.csv not found")
        return
    imp = pd.read_csv(IMPORTANCE_PATH).sort_values("importance", ascending=True).tail(15)
    # Human-readable labels
    imp = imp.copy()
    imp["label"] = imp["feature"].str.replace("_", " ").str.title()

    if not DATASET_PATH.exists() or not MODEL_PATH.exists():
        print("  [Fig 4] Skip: dataset or model not found; plotting feature importance only")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp["label"], imp["importance"], color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Importance (gain)")
        ax.set_title("Profit model: top 15 features (LightGBM)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "paper_fig4_model_quality.png", bbox_inches="tight")
        plt.close(fig)
        print("  [Fig 4] paper_fig4_model_quality.png (importance only)")
        return

    df = pd.read_parquet(DATASET_PATH)
    X = df[FEATURE_COLS].values
    y = df["expected_profit"].values
    groups = df["driver_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, val_idx = next(gss.split(X, y, groups=groups))
    X_val, y_true = X[val_idx], y[val_idx]
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.barh(imp["label"], imp["importance"], color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 15 features")

    ax = axes[1]
    n = len(y_true)
    if n > 30000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 30000, replace=False)
        y_true_s, y_pred_s = y_true[idx], y_pred[idx]
    else:
        y_true_s, y_pred_s = y_true, y_pred
    ax.scatter(y_true_s, y_pred_s, alpha=0.2, s=6, color="#4C72B0")
    lo = min(y_true_s.min(), y_pred_s.min())
    hi = max(y_true_s.max(), y_pred_s.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual profit ($)")
    ax.set_ylabel("Predicted profit ($)")
    ax.set_title(f"Predicted vs actual (validation)\nR² = {r2:.3f}, RMSE = ${rmse:.2f}")
    ax.legend(loc="upper left")

    fig.suptitle("Profit model: feature importance and predictive performance", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_fig4_model_quality.png", bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 4] paper_fig4_model_quality.png")


# ---------------------------------------------------------------------------
# Figure 5: Mean profit by strategy across densities
# ---------------------------------------------------------------------------
def fig5_profit_by_density() -> None:
    densities = [100, 75, 50, 25, 10]
    strategies_to_plot = ["coldstart", "warmup", "oracle"]
    data = []  # density_pct, strategy, mean, sem
    for d in densities:
        suffix = f"_d{d}" if d < 100 else ""
        for s in strategies_to_plot:
            df = _load_strategy(s, suffix)
            if df is None:
                continue
            agg = df.groupby("driver_id")["profit"].mean()
            data.append({
                "density_pct": d,
                "strategy": s,
                "mean": agg.mean(),
                "sem": agg.sem(),
            })
    if not data:
        print("  [Fig 5] Skip: no density outcome files")
        return
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in strategies_to_plot:
        sub = df[df["strategy"] == s].sort_values("density_pct", ascending=False)
        if len(sub) == 0:
            continue
        ci = sub["sem"].values * 1.96
        ax.errorbar(
            sub["density_pct"], sub["mean"], yerr=ci,
            fmt="o-", label=STRATEGY_LABELS[s], color=PALETTE[s],
            capsize=4, linewidth=1.5, markersize=7
        )

    # Annotate difference at each density so it's seen properly: Warm-Up − Cold-Start
    densities_sorted = sorted(df["density_pct"].unique(), reverse=True)
    for d in densities_sorted:
        r_cs = df[(df["density_pct"] == d) & (df["strategy"] == "coldstart")]
        r_wu = df[(df["density_pct"] == d) & (df["strategy"] == "warmup")]
        if r_cs.empty or r_wu.empty:
            continue
        cs_mean = r_cs["mean"].iloc[0]
        wu_mean = r_wu["mean"].iloc[0]
        diff = wu_mean - cs_mean
        mid_y = (cs_mean + wu_mean) / 2
        ax.annotate(
            f"+${diff:.2f}",
            xy=(d, mid_y),
            xytext=(d - 4, mid_y),
            fontsize=8,
            fontweight="bold",
            color="#DD8452",
            ha="right",
            va="center",
        )
    ax.set_xlabel("Rider density (%)")
    ax.set_ylabel("Mean profit per driver ($)")
    ax.set_title("Mean profit by strategy across rider density (annotations: Warm-Up − Cold-Start)")
    ax.legend(loc="lower right")
    ax.set_xlim(105, 5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_fig5_profit_by_density.png", bbox_inches="tight")
    plt.close(fig)
    print("  [Fig 5] paper_fig5_profit_by_density.png")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating paper figures...")
    fig1_density_advantage()
    fig2_strategy_comparison()
    fig3_profit_difference_distribution()
    fig4_model_quality()
    fig5_profit_by_density()
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "visualizations" / "plot_corridor_map.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0:
            print(r.stdout or "  [Fig 6] corridor_map.html")
        else:
            print(f"  [Fig 6] Skip corridor map: {r.stderr or r.stdout}")
    except Exception as e:
        print(f"  [Fig 6] Skip corridor map: {e}")
    print(f"\nPaper figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
