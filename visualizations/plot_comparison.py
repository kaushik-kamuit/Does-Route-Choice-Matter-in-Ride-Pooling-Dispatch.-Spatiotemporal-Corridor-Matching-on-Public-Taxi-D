"""
Publication-quality comparison plots: cold-start vs warm-up.

Reads results/{coldstart,warmup}_outcomes.csv and produces:
  1. Mean profit bar chart with 95% CI
  2. Profit distribution box plots (faceted by route length)
  3. Cumulative profit line chart
  4. Match rate bar chart
  5. Revenue vs cost stacked bar
  6. Compute time bar chart

Also writes results/summary.txt with statistical tests.

Usage:
    python visualizations/plot_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

CATEGORY_ORDER = ["short", "medium", "long"]
STRATEGY_ORDER = ["coldstart", "warmup"]
STRATEGY_LABELS = {
    "coldstart": "Cold-Start", "warmup": "Warm-Up",
    "random": "Random", "heuristic": "Heuristic", "oracle": "Oracle",
}
PALETTE = {
    "coldstart": "#4C72B0", "warmup": "#DD8452",
    "random": "#8DA0CB", "heuristic": "#66C2A5", "oracle": "#E78AC3",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.figsize": (8, 5),
})


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cs = pd.read_csv(RESULTS_DIR / "coldstart_outcomes.csv")
    wu = pd.read_csv(RESULTS_DIR / "warmup_outcomes.csv")
    return cs, wu


def _combined(cs: pd.DataFrame, wu: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([cs, wu], ignore_index=True)


# -----------------------------------------------------------------------
# Plot 1: Mean profit bar chart with 95% CI
# -----------------------------------------------------------------------
def plot_profit_bar(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    df = _combined(cs, wu)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})

    # Overall
    ax = axes[0]
    agg = df.groupby("strategy")["profit"].agg(["mean", "sem"]).reindex(STRATEGY_ORDER)
    ci95 = agg["sem"] * 1.96
    bars = ax.bar(
        [STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
        agg["mean"],
        yerr=ci95,
        color=[PALETTE[s] for s in STRATEGY_ORDER],
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Mean Profit ($)")
    ax.set_title("Overall")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Per category
    ax = axes[1]
    agg_cat = (
        df.groupby(["route_length_category", "strategy"])["profit"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    x_pos = np.arange(len(CATEGORY_ORDER))
    width = 0.35
    for i, strat in enumerate(STRATEGY_ORDER):
        sub = agg_cat[agg_cat["strategy"] == strat].set_index("route_length_category").reindex(CATEGORY_ORDER)
        ci = sub["sem"] * 1.96
        ax.bar(
            x_pos + i * width - width / 2,
            sub["mean"],
            width,
            yerr=ci,
            label=STRATEGY_LABELS[strat],
            color=PALETTE[strat],
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.capitalize() for c in CATEGORY_ORDER])
    ax.set_ylabel("Mean Profit ($)")
    ax.set_title("By Route Length")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Mean Profit: Cold-Start vs Warm-Up", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "profit_bar.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 2: Profit distribution box plots
# -----------------------------------------------------------------------
def plot_profit_box(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    df = _combined(cs, wu)
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, cat in zip(axes, CATEGORY_ORDER):
        sub = df[df["route_length_category"] == cat]
        sns.boxplot(
            data=sub,
            x="strategy_label",
            y="profit",
            hue="strategy_label",
            palette={STRATEGY_LABELS[s]: PALETTE[s] for s in STRATEGY_ORDER},
            ax=ax,
            showfliers=False,
            legend=False,
        )
        ax.set_title(f"{cat.capitalize()} Routes")
        ax.set_xlabel("")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Profit ($)")
    fig.suptitle("Profit Distribution by Route Length", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "profit_box.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 3: Cumulative profit line chart
# -----------------------------------------------------------------------
def plot_cumulative_profit(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Use first seed for a clean single-run view
    first_seed = cs["seed"].min()
    cs_s = cs[cs["seed"] == first_seed].reset_index(drop=True)
    wu_s = wu[wu["seed"] == first_seed].reset_index(drop=True)

    ax.plot(cs_s["profit"].cumsum().values, label="Cold-Start", color=PALETTE["coldstart"])
    ax.plot(wu_s["profit"].cumsum().values, label="Warm-Up", color=PALETTE["warmup"])
    ax.set_xlabel("Driver #")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title("Cumulative Profit Over Drivers")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cumulative_profit.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 4: Match rate bar chart
# -----------------------------------------------------------------------
def plot_match_rate(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    df = _combined(cs, wu)
    df["matched"] = (df["matched_riders"] > 0).astype(int)

    rates = (
        df.groupby(["route_length_category", "strategy"])["matched"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(CATEGORY_ORDER))
    width = 0.35
    for i, strat in enumerate(STRATEGY_ORDER):
        sub = rates[rates["strategy"] == strat].set_index("route_length_category").reindex(CATEGORY_ORDER)
        ax.bar(
            x_pos + i * width - width / 2,
            sub["matched"] * 100,
            width,
            label=STRATEGY_LABELS[strat],
            color=PALETTE[strat],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.capitalize() for c in CATEGORY_ORDER])
    ax.set_ylabel("Match Rate (%)")
    ax.set_title("Percentage of Drivers Matching >= 1 Rider")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "match_rate.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 5: Revenue vs cost stacked bar
# -----------------------------------------------------------------------
def plot_revenue_cost(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = {}
    for strat, sdf in [("coldstart", cs), ("warmup", wu)]:
        metrics[strat] = {
            "revenue": sdf["total_revenue"].mean(),
            "cost": sdf["driving_cost"].mean(),
        }

    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    revenues = [metrics[s]["revenue"] for s in STRATEGY_ORDER]
    costs = [metrics[s]["cost"] for s in STRATEGY_ORDER]

    ax.bar(labels, revenues, label="Revenue", color="#55a868", edgecolor="black", linewidth=0.5)
    ax.bar(labels, [-c for c in costs], label="Cost", color="#c44e52", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Dollars ($)")
    ax.set_title("Mean Revenue vs Driving Cost per Driver")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "revenue_cost.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 6: Compute time bar chart
# -----------------------------------------------------------------------
def plot_compute_time(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    means = {
        "coldstart": cs["compute_time_s"].mean() * 1000,
        "warmup": wu["compute_time_s"].mean() * 1000,
    }
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    values = [means[s] for s in STRATEGY_ORDER]

    ax.bar(labels, values, color=[PALETTE[s] for s in STRATEGY_ORDER],
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Time per Driver (ms)")
    ax.set_title("Compute Time Comparison")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "compute_time.png", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Statistical summary
# -----------------------------------------------------------------------
def write_summary(cs: pd.DataFrame, wu: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("STATISTICAL SUMMARY: Cold-Start vs Warm-Up")
    lines.append("=" * 70)

    for cat in ["all"] + CATEGORY_ORDER:
        if cat == "all":
            cs_sub, wu_sub = cs, wu
            label = "OVERALL"
        else:
            cs_sub = cs[cs["route_length_category"] == cat]
            wu_sub = wu[wu["route_length_category"] == cat]
            label = f"ROUTE LENGTH: {cat.upper()}"

        lines.append(f"\n--- {label} ---")
        lines.append(f"{'':20s} {'Cold-Start':>12s} {'Warm-Up':>12s}")
        lines.append("-" * 50)

        for metric in ["profit", "matched_riders", "total_revenue", "driving_cost"]:
            cs_m = cs_sub[metric].mean()
            wu_m = wu_sub[metric].mean()
            lines.append(f"  Mean {metric:20s}  {cs_m:>10.2f}   {wu_m:>10.2f}")

        cs_mr = (cs_sub["matched_riders"] > 0).mean() * 100
        wu_mr = (wu_sub["matched_riders"] > 0).mean() * 100
        lines.append(f"  Match rate (%)         {cs_mr:>10.1f}   {wu_mr:>10.1f}")

        for stat_name, fn in [("Median", "median"), ("Std", "std"), ("Min", "min"), ("Max", "max")]:
            cs_v = getattr(cs_sub["profit"], fn)()
            wu_v = getattr(wu_sub["profit"], fn)()
            lines.append(f"  {stat_name} profit           {cs_v:>10.2f}   {wu_v:>10.2f}")

        cs_agg = cs_sub.groupby("driver_id")["profit"].mean()
        wu_agg = wu_sub.groupby("driver_id")["profit"].mean()
        merged = pd.DataFrame({"cs": cs_agg, "wu": wu_agg}).dropna()
        if len(merged) > 1:
            diff = merged["wu"] - merged["cs"]
            t_stat, p_val = stats.ttest_rel(merged["wu"], merged["cs"])
            try:
                w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
                lines.append(f"  Wilcoxon test:   W={w_stat:.0f}, p={w_pval:.2e}")
            except ValueError:
                pass
            lines.append(f"\n  Paired t-test:   t={t_stat:.3f}, p={p_val:.2e}")
            lines.append(f"  Mean difference: ${diff.mean():.2f} (WU - CS)")
            lines.append(f"  N drivers:       {len(merged):,}")

    summary = "\n".join(lines)
    print(summary)
    with open(RESULTS_DIR / "summary.txt", "w") as f:
        f.write(summary)
    print(f"\n  Summary saved to: {RESULTS_DIR / 'summary.txt'}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cs, wu = _load()
    print(f"  Cold-start rows: {len(cs):,}   Warm-up rows: {len(wu):,}")

    plot_profit_bar(cs, wu)
    print("  [1/6] Profit bar chart")

    plot_profit_box(cs, wu)
    print("  [2/6] Profit box plots")

    plot_cumulative_profit(cs, wu)
    print("  [3/6] Cumulative profit")

    plot_match_rate(cs, wu)
    print("  [4/6] Match rate")

    plot_revenue_cost(cs, wu)
    print("  [5/6] Revenue vs cost")

    plot_compute_time(cs, wu)
    print("  [6/6] Compute time")

    write_summary(cs, wu)

    print(f"\n  All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
