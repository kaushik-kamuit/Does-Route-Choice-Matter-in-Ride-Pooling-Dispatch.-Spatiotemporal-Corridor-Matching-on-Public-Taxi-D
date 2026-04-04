"""
Extended analysis and publication-quality plots for the multi-strategy experiment.

Generates:
  1. Baseline comparison bar chart (all 5 strategies)
  2. Winner/loser histogram (per-driver profit difference distribution)
  3. Heterogeneity analysis (time-of-day, geography, route-choice)
  4. Enhanced statistical summary (Cohen's d, bootstrap CI, economic framing)
  5. Density vs advantage line chart (if density experiment results exist)
  6. Ablation heatmap (if ablation results exist)

Usage:
    python visualizations/plot_extended.py
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
REALISM_PRIMARY_PATH = RESULTS_DIR / "realism_primary_summary.csv"

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

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.figsize": (8, 5),
})


def _load_strategy(name: str, suffix: str = "") -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{name}_outcomes{suffix}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_all(suffix: str = "") -> dict[str, pd.DataFrame]:
    dfs = {}
    for s in STRATEGIES:
        df = _load_strategy(s, suffix)
        if df is not None:
            dfs[s] = df
    return dfs


# -----------------------------------------------------------------------
# 1. Baseline comparison bar chart
# -----------------------------------------------------------------------
def plot_baseline_comparison(dfs: dict[str, pd.DataFrame]) -> None:
    present = [s for s in STRATEGIES if s in dfs]
    if len(present) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    means, cis, colors, labels = [], [], [], []
    for s in present:
        df = dfs[s]
        agg = df.groupby("driver_id")["profit"].mean()
        means.append(agg.mean())
        cis.append(agg.sem() * 1.96)
        colors.append(PALETTE.get(s, "#999999"))
        labels.append(STRATEGY_LABELS.get(s, s))

    x = np.arange(len(present))
    bars = ax.bar(x, means, yerr=cis, color=colors, capsize=5,
                  edgecolor="black", linewidth=0.5)

    cs_mean = dfs["coldstart"].groupby("driver_id")["profit"].mean().mean() if "coldstart" in dfs else 0
    for xi, m in zip(x, means):
        delta = m - cs_mean
        sign = "+" if delta >= 0 else ""
        ax.text(xi, m + cis[x.tolist().index(xi)] + 0.3,
                f"{sign}${delta:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Profit per Driver ($)")
    ax.set_title("Strategy Comparison: Mean Profit with 95% CI")
    ax.axhline(cs_mean, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [1] Baseline comparison bar chart")


# -----------------------------------------------------------------------
# 2. Winner/loser histogram
# -----------------------------------------------------------------------
def plot_winner_loser(dfs: dict[str, pd.DataFrame]) -> None:
    if "coldstart" not in dfs or "warmup" not in dfs:
        return
    cs_agg = dfs["coldstart"].groupby("driver_id")["profit"].mean()
    wu_agg = dfs["warmup"].groupby("driver_id")["profit"].mean()
    merged = pd.DataFrame({"cs": cs_agg, "wu": wu_agg}).dropna()
    delta = merged["wu"] - merged["cs"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(delta.quantile(0.01), delta.quantile(0.99), 60)
    ax.hist(delta, bins=bins, color="#DD8452", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-")
    ax.axvline(delta.mean(), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean: ${delta.mean():.2f}")

    pct_win = (delta > 0).mean() * 100
    pct_lose = (delta < 0).mean() * 100
    pct_tie = (delta == 0).mean() * 100
    ax.text(0.97, 0.95,
            f"Better: {pct_win:.1f}%\nWorse: {pct_lose:.1f}%\nTied: {pct_tie:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel("Profit Difference: Warm-Up minus Cold-Start ($)")
    ax.set_ylabel("Number of Drivers")
    ax.set_title("Per-Driver Profit Difference Distribution")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "winner_loser.png", bbox_inches="tight")
    plt.close(fig)
    print("  [2] Winner/loser histogram")


# -----------------------------------------------------------------------
# 3. Heterogeneity: time-of-day
# -----------------------------------------------------------------------
def plot_heterogeneity_time(dfs: dict[str, pd.DataFrame]) -> None:
    if "coldstart" not in dfs or "warmup" not in dfs:
        return
    cs = dfs["coldstart"].copy()
    wu = dfs["warmup"].copy()

    driver_hours = cs.groupby("driver_id").first()
    if "hour" not in cs.columns:
        return

    cs_agg = cs.groupby("driver_id").agg({"profit": "mean", "seed": "first"})
    wu_agg = wu.groupby("driver_id")["profit"].mean()

    hour_info = cs.groupby("driver_id").first()
    if "hour" not in hour_info.columns:
        return

    combined = pd.DataFrame({
        "cs_profit": cs_agg["profit"],
        "wu_profit": wu_agg,
    }).dropna()

    hours = cs.drop_duplicates("driver_id").set_index("driver_id")
    if "hour" not in hours.columns:
        return

    combined["hour"] = hours.reindex(combined.index).get("hour", np.nan)
    combined = combined.dropna(subset=["hour"])
    combined["hour"] = combined["hour"].astype(int)
    combined["delta"] = combined["wu_profit"] - combined["cs_profit"]

    def _bucket(h):
        if 7 <= h < 10:
            return "Morning\n(7-10)"
        elif 10 <= h < 16:
            return "Midday\n(10-16)"
        elif 16 <= h < 20:
            return "Evening\n(16-20)"
        else:
            return "Night\n(20-7)"

    combined["period"] = combined["hour"].apply(_bucket)
    period_order = ["Morning\n(7-10)", "Midday\n(10-16)", "Evening\n(16-20)", "Night\n(20-7)"]
    combined["period"] = pd.Categorical(combined["period"], categories=period_order, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    agg = combined.groupby("period")["delta"].agg(["mean", "sem", "count"])
    ci95 = agg["sem"] * 1.96
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["mean"], yerr=ci95, color="#DD8452", capsize=5,
                  edgecolor="black", linewidth=0.5)
    for xi, row in zip(x, agg.itertuples()):
        ax.text(xi, row.mean + row.sem * 1.96 + 0.1,
                f"n={row.count}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, fontsize=10)
    ax.set_ylabel("Warm-Up Advantage ($)")
    ax.set_title("Warm-Up Advantage by Time of Day")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "heterogeneity_time.png", bbox_inches="tight")
    plt.close(fig)
    print("  [3] Heterogeneity: time-of-day")


# -----------------------------------------------------------------------
# 4. Heterogeneity: route choice analysis
# -----------------------------------------------------------------------
def plot_route_choice(dfs: dict[str, pd.DataFrame]) -> None:
    if "warmup" not in dfs or "coldstart" not in dfs:
        return
    wu = dfs["warmup"].copy()
    cs = dfs["coldstart"].copy()

    wu_agg = wu.groupby("driver_id").agg(
        wu_profit=("profit", "mean"),
        route_chosen=("route_rank_chosen", "first"),
    )
    cs_agg = cs.groupby("driver_id")["profit"].mean().rename("cs_profit")
    combined = wu_agg.join(cs_agg, how="inner")
    combined["delta"] = combined["wu_profit"] - combined["cs_profit"]
    combined["chose_alt"] = combined["route_chosen"] > 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Route choice distribution
    ax = axes[0]
    choice_counts = combined["route_chosen"].value_counts().sort_index()
    ax.bar(choice_counts.index, choice_counts.values,
           color=["#4C72B0", "#DD8452", "#55a868"][:len(choice_counts)],
           edgecolor="black", linewidth=0.5)
    for xi, v in zip(choice_counts.index, choice_counts.values):
        ax.text(xi, v + 10, f"{v/len(combined)*100:.1f}%", ha="center", fontsize=10)
    ax.set_xlabel("Route Chosen (1 = default)")
    ax.set_ylabel("Number of Drivers")
    ax.set_title("ML Route Selection Distribution")

    # Delta when choosing alternative
    ax = axes[1]
    for label, mask, color in [
        ("Same as default", ~combined["chose_alt"], "#4C72B0"),
        ("Alternative", combined["chose_alt"], "#DD8452"),
    ]:
        sub = combined[mask]
        ax.bar(label, sub["delta"].mean(),
               yerr=sub["delta"].sem() * 1.96,
               color=color, capsize=5, edgecolor="black", linewidth=0.5)
        ax.text(
            ["Same as default", "Alternative"].index(label),
            sub["delta"].mean() + sub["delta"].sem() * 1.96 + 0.1,
            f"n={len(sub)}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Mean Profit Difference vs Cold-Start ($)")
    ax.set_title("Advantage When ML Picks Alternative Route")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "route_choice.png", bbox_inches="tight")
    plt.close(fig)
    print("  [4] Route choice analysis")


# -----------------------------------------------------------------------
# 5. Density vs advantage (reads density experiment results)
# -----------------------------------------------------------------------
def plot_density_advantage() -> None:
    densities = [100, 75, 50, 25, 10]
    records = []
    for d in densities:
        suffix = f"_d{d}" if d < 100 else ""
        cs = _load_strategy("coldstart", suffix)
        wu = _load_strategy("warmup", suffix)
        heu = _load_strategy("heuristic", suffix)
        ora = _load_strategy("oracle", suffix)
        if cs is None or wu is None:
            continue

        cs_agg = cs.groupby("driver_id")["profit"].mean()
        wu_agg = wu.groupby("driver_id")["profit"].mean()
        merged = pd.DataFrame({"cs": cs_agg, "wu": wu_agg}).dropna()
        delta = merged["wu"] - merged["cs"]
        match_rate_cs = (cs["matched_riders"] > 0).mean() * 100
        match_rate_wu = (wu["matched_riders"] > 0).mean() * 100

        row = {
            "density_pct": d,
            "delta_mean": delta.mean(),
            "delta_sem": delta.sem(),
            "delta_pct": (
                delta.mean() / (abs(merged["cs"].mean()) if merged["cs"].mean() < 0 else merged["cs"].mean()) * 100
                if merged["cs"].mean() != 0 else 0
            ),
            "cs_mean": merged["cs"].mean(),
            "wu_mean": merged["wu"].mean(),
            "match_rate_cs": match_rate_cs,
            "match_rate_wu": match_rate_wu,
            "n_drivers": len(merged),
            "matching_window_min": 5,
            "index_bin_minutes": 15,
            "candidate_window_bins": 1,
            "max_detour_min": 4.0,
            "rider_pool_semantics": "retained_25pct_sample",
        }
        if heu is not None:
            row["heuristic_profit"] = heu.groupby("driver_id")["profit"].mean().mean()
            row["warmup_vs_heuristic"] = row["wu_mean"] - row["heuristic_profit"]
        if ora is not None:
            row["oracle_profit"] = ora.groupby("driver_id")["profit"].mean().mean()
        records.append(row)

    if len(records) < 2:
        print("  [5] Density plot: skipped (need >= 2 density levels)")
        return

    df = pd.DataFrame(records).sort_values("density_pct")
    if "delta_sem" not in df.columns:
        df["delta_sem"] = 0.0
    df.to_csv(RESULTS_DIR / "density_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(df["density_pct"], df["delta_mean"],
                yerr=df["delta_sem"] * 1.96,
                fmt="o-", color="#DD8452", capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Retained-Sample Density (%)")
    ax.set_ylabel("Warm-Up Advantage ($)")
    ax.set_title("Warm-Up Advantage vs Rider Density")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.invert_xaxis()
    ax.text(
        0.03, 0.96,
        "5-minute exact request window\n25% rider pre-sample retained before density subsampling",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax = axes[1]
    if "match_rate_cs" in df.columns and "match_rate_wu" in df.columns:
        ax.plot(df["density_pct"], df["match_rate_cs"], "o-", color="#4C72B0",
                label="Cold-Start", linewidth=2, markersize=8)
        ax.plot(df["density_pct"], df["match_rate_wu"], "s-", color="#DD8452",
                label="Warm-Up", linewidth=2, markersize=8)
        ax.set_xlabel("Retained-Sample Density (%)")
        ax.set_ylabel("Match Rate (%)")
        ax.set_title("Match Rate vs Rider Density")
        ax.legend()
        ax.invert_xaxis()
    else:
        ax.plot(df["density_pct"], df["cs_mean"], "o-", color="#4C72B0",
                label="Cold-Start", linewidth=2, markersize=8)
        ax.plot(df["density_pct"], df["wu_mean"], "s-", color="#DD8452",
                label="Warm-Up", linewidth=2, markersize=8)
        ax.set_xlabel("Retained-Sample Density (%)")
        ax.set_ylabel("Mean Profit ($)")
        ax.set_title("Absolute Profit vs Rider Density")
        ax.legend()
        ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "density_advantage.png", bbox_inches="tight")
    plt.close(fig)
    print("  [5] Density vs advantage line chart")


# -----------------------------------------------------------------------
# 6. Ablation heatmap
# -----------------------------------------------------------------------
def plot_ablation_heatmap() -> None:
    path = RESULTS_DIR / "ablation_results.csv"
    if not path.exists():
        print("  [6] Ablation heatmap: skipped (no results)")
        return

    df = pd.read_csv(path)
    metrics = df[["experiment", "r2", "rmse", "rank_acc"]].set_index("experiment")
    metrics.columns = ["R²", "RMSE ($)", "Rank-1 Accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, col in zip(axes, metrics.columns):
        vals = metrics[col].values.reshape(-1, 1)
        cmap = "YlOrRd_r" if "RMSE" in col else "YlOrRd"
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics.index, fontsize=9)
        ax.set_xticks([])
        ax.set_title(col, fontsize=12)
        for i, v in enumerate(vals.flatten()):
            fmt = f"{v:.4f}" if "R²" in col else (f"${v:.2f}" if "RMSE" in col else f"{v:.1%}")
            ax.text(0, i, fmt, ha="center", va="center", fontsize=10,
                    color="white" if v > vals.mean() and "RMSE" not in col else "black")
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Feature Ablation Study", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ablation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  [6] Ablation heatmap")


# -----------------------------------------------------------------------
# 7. Enhanced statistical summary
# -----------------------------------------------------------------------
def write_extended_summary(dfs: dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("EXTENDED STATISTICAL SUMMARY: Multi-Strategy Comparison")
    lines.append("=" * 78)

    # Strategy comparison table
    lines.append("\n--- STRATEGY COMPARISON ---")
    lines.append(f"{'Strategy':15s} {'Mean $':>10s} {'Median $':>10s} "
                 f"{'Std $':>10s} {'Match%':>8s} {'vs CS':>10s}")
    lines.append("-" * 78)
    cs_mean = 0
    for s in STRATEGIES:
        if s not in dfs:
            continue
        df = dfs[s]
        agg = df.groupby("driver_id")["profit"].mean()
        mr = (df["matched_riders"] > 0).mean() * 100
        if s == "coldstart":
            cs_mean = agg.mean()
        delta = agg.mean() - cs_mean
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {STRATEGY_LABELS.get(s, s):13s} {agg.mean():10.2f} {agg.median():10.2f} "
            f"{agg.std():10.2f} {mr:7.1f}% {sign}${abs(delta):.2f}"
        )

    # Effect size metrics (warm-up vs cold-start)
    if "coldstart" in dfs and "warmup" in dfs:
        cs_agg = dfs["coldstart"].groupby("driver_id")["profit"].mean()
        wu_agg = dfs["warmup"].groupby("driver_id")["profit"].mean()
        merged = pd.DataFrame({"cs": cs_agg, "wu": wu_agg}).dropna()
        diff = merged["wu"] - merged["cs"]
        n = len(merged)

        lines.append("\n--- EFFECT SIZE: Warm-Up vs Cold-Start ---")

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(merged["wu"], merged["cs"])
        lines.append(f"  Paired t-test:    t={t_stat:.3f}, p={t_pval:.2e}")

        # Wilcoxon
        try:
            w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
            lines.append(f"  Wilcoxon:         W={w_stat:.0f}, p={w_pval:.2e}")
        except ValueError:
            pass

        # Cohen's d
        pooled_std = np.sqrt((merged["cs"].var() + merged["wu"].var()) / 2)
        cohens_d = diff.mean() / pooled_std if pooled_std > 0 else 0
        lines.append(f"  Cohen's d:        {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            effect_label = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_label = "small"
        elif abs(cohens_d) < 0.8:
            effect_label = "medium"
        else:
            effect_label = "large"
        lines.append(f"  Effect category:  {effect_label}")

        # Bootstrap CI
        rng = np.random.default_rng(42)
        n_boot = 10_000
        boot_means = np.array([
            rng.choice(diff.values, size=n, replace=True).mean()
            for _ in range(n_boot)
        ])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        lines.append(f"  Mean difference:  ${diff.mean():.2f}")
        lines.append(f"  95% CI (t-based): [${diff.mean() - diff.sem()*1.96:.2f}, "
                     f"${diff.mean() + diff.sem()*1.96:.2f}]")
        lines.append(f"  95% CI (boot):    [${ci_lo:.2f}, ${ci_hi:.2f}]")
        lines.append(f"  N drivers:        {n:,}")

        # Winner/loser breakdown
        pct_win = (diff > 0).mean() * 100
        pct_lose = (diff < 0).mean() * 100
        pct_tie = (diff == 0).mean() * 100
        lines.append(f"\n  Drivers better off:  {pct_win:.1f}%")
        lines.append(f"  Drivers worse off:   {pct_lose:.1f}%")
        lines.append(f"  Drivers tied:        {pct_tie:.1f}%")
        lines.append(f"  Mean gain (winners): ${diff[diff > 0].mean():.2f}")
        lines.append(f"  Mean loss (losers):  ${diff[diff < 0].mean():.2f}")

        # Economic framing
        lines.append("\n--- ECONOMIC IMPACT ---")
        per_trip = diff.mean()
        daily_trips = 50_000
        lines.append(f"  Per trip improvement:      ${per_trip:.2f}")
        lines.append(f"  Per {daily_trips:,} trips/day:   ${per_trip * daily_trips:,.0f}")
        lines.append(f"  Annual (365 days):         ${per_trip * daily_trips * 365:,.0f}")

    # Oracle gap analysis
    if "coldstart" in dfs and "warmup" in dfs and "oracle" in dfs:
        cs_agg = dfs["coldstart"].groupby("driver_id")["profit"].mean()
        wu_agg = dfs["warmup"].groupby("driver_id")["profit"].mean()
        or_agg = dfs["oracle"].groupby("driver_id")["profit"].mean()
        merged = pd.DataFrame({"cs": cs_agg, "wu": wu_agg, "oracle": or_agg}).dropna()

        oracle_gap = merged["oracle"].mean() - merged["cs"].mean()
        ml_captured = merged["wu"].mean() - merged["cs"].mean()
        capture_pct = ml_captured / oracle_gap * 100 if oracle_gap > 0 else 0

        lines.append("\n--- ORACLE GAP ANALYSIS ---")
        lines.append(f"  Oracle advantage vs CS:  ${oracle_gap:.2f}")
        lines.append(f"  ML captured:             ${ml_captured:.2f}")
        lines.append(f"  ML capture rate:         {capture_pct:.1f}%")
        lines.append(f"  Remaining headroom:      ${oracle_gap - ml_captured:.2f}")

    summary = "\n".join(lines)
    print(summary)
    out_path = RESULTS_DIR / "extended_summary.txt"
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"\n  Extended summary saved to: {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dfs = _load_all()

    if not dfs:
        print("  No outcome files found. Run the simulation first.")
        return

    print(f"  Loaded strategies: {list(dfs.keys())}")
    for s, df in dfs.items():
        print(f"    {s}: {len(df):,} rows")
    print()

    plot_baseline_comparison(dfs)
    plot_winner_loser(dfs)
    plot_heterogeneity_time(dfs)
    plot_route_choice(dfs)
    plot_density_advantage()
    plot_ablation_heatmap()
    write_extended_summary(dfs)

    print(f"\n  All extended plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
