from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAPER_FIG_DIR = ROOT / "paper_rendezvous" / "figures"
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#d8d8d8",
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)
POLICY_COLORS = {
    "corridor_only": "#4d4d4d",
    "time_only_baseline": "#7f7f7f",
    "feasible_count_baseline": "#e69f00",
    "walk_aware_rendezvous": "#009e73",
    "rendezvous_only": "#0072b2",
    "rendezvous_observable": "#d55e00",
    "ml_meeting_point_comparator": "#cc79a7",
}
POLICY_HATCHES = {
    "corridor_only": "///",
    "time_only_baseline": "\\\\\\",
    "feasible_count_baseline": "xx",
    "walk_aware_rendezvous": "..",
    "rendezvous_only": "++",
    "rendezvous_observable": "oo",
    "ml_meeting_point_comparator": "**",
}
POLICY_MARKERS = {
    "corridor_only": "s",
    "time_only_baseline": "^",
    "feasible_count_baseline": "D",
    "walk_aware_rendezvous": "P",
    "rendezvous_only": "o",
    "rendezvous_observable": "X",
    "ml_meeting_point_comparator": "*",
}
POLICY_LABELS = {
    "corridor_only": "Corridor only",
    "time_only_baseline": "Shortest-route baseline",
    "feasible_count_baseline": "Feasible-count baseline",
    "walk_aware_rendezvous": "Walk-aware rendezvous",
    "rendezvous_only": "Rendezvous only",
    "rendezvous_observable": "Rendezvous + observability",
    "ml_meeting_point_comparator": "ML comparator",
}
POLICY_AXIS_LABELS = {
    "corridor_only": "Corridor\nonly",
    "time_only_baseline": "Shortest\nroute",
    "feasible_count_baseline": "Feasible\ncount",
    "walk_aware_rendezvous": "Walk-aware\nrendezvous",
    "rendezvous_only": "Rendezvous\nonly",
    "rendezvous_observable": "Rendezvous +\nobservability",
    "ml_meeting_point_comparator": "ML\ncomparator",
}
MAIN_POLICY_ORDER = [
    "corridor_only",
    "rendezvous_only",
    "rendezvous_observable",
    "ml_meeting_point_comparator",
]
STRONG_BASELINE_ORDER = [
    "corridor_only",
    "time_only_baseline",
    "feasible_count_baseline",
    "walk_aware_rendezvous",
    "rendezvous_only",
    "rendezvous_observable",
    "ml_meeting_point_comparator",
]
SCENARIO_LABELS = {
    "primary": "Primary",
    "sparse_high_occlusion": "Sparse\nhigh occlusion",
    "very_sparse_low_occlusion": "Very sparse\nlow occlusion",
    "very_sparse_extreme_occlusion": "Very sparse\nextreme occlusion",
}
TIME_SLICE_LABELS = {
    "all_day": "All day",
    "morning_peak": "Morning peak",
    "evening_peak": "Evening peak",
}


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _filter_default_slice(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if "time_slice" in filtered.columns:
        filtered = filtered[filtered["time_slice"] == "all_day"]
    if "observability_profile" in filtered.columns:
        filtered = filtered[filtered["observability_profile"] == "calibrated"]
    if "observability_ablation" in filtered.columns:
        filtered = filtered[filtered["observability_ablation"] == "full"]
    if "use_urban_context" in filtered.columns:
        filtered = filtered[filtered["use_urban_context"] == True]  # noqa: E712
    return filtered


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=250, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / name, dpi=250, bbox_inches="tight")
    pdf_name = Path(name).with_suffix(".pdf").name
    fig.savefig(PLOTS_DIR / pdf_name, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / pdf_name, bbox_inches="tight")
    plt.close(fig)


def _save_alias(source_name: str, alias_name: str) -> None:
    source_plot = PLOTS_DIR / source_name
    source_paper = PAPER_FIG_DIR / source_name
    if source_plot.exists():
        (PLOTS_DIR / alias_name).write_bytes(source_plot.read_bytes())
    if source_paper.exists():
        (PAPER_FIG_DIR / alias_name).write_bytes(source_paper.read_bytes())
    source_plot_pdf = PLOTS_DIR / Path(source_name).with_suffix(".pdf").name
    source_paper_pdf = PAPER_FIG_DIR / Path(source_name).with_suffix(".pdf").name
    if source_plot_pdf.exists():
        (PLOTS_DIR / Path(alias_name).with_suffix(".pdf").name).write_bytes(source_plot_pdf.read_bytes())
    if source_paper_pdf.exists():
        (PAPER_FIG_DIR / Path(alias_name).with_suffix(".pdf").name).write_bytes(source_paper_pdf.read_bytes())


def _style_axis(ax: plt.Axes, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", linestyle="-", alpha=0.65)
    ax.set_axisbelow(True)


def _bar(
    ax: plt.Axes,
    x,
    heights,
    *,
    policies,
    width=0.8,
    label: str | None = None,
):
    containers = []
    for xpos, height, policy in zip(x, heights, policies):
        containers.extend(
            ax.bar(
                xpos,
                height,
                width=width,
                color=POLICY_COLORS[policy],
                edgecolor="#222222",
                linewidth=0.7,
                hatch=POLICY_HATCHES[policy],
                label=label if label and len(containers) == 0 else None,
            )
        )
    return containers


def fig1_concept() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    corridor_x = np.linspace(0.8, 9.2, 240)
    center_y = 2.0 + 0.10 * np.sin(corridor_x * 0.9)
    upper_y = center_y + 0.42
    lower_y = center_y - 0.42
    ax.fill_between(corridor_x, lower_y, upper_y, color="#dceaf7", alpha=0.95, zorder=1)
    ax.plot(corridor_x, center_y, color="#1f4e79", linewidth=2.8, zorder=2)

    anchor_x = np.array([2.0, 4.7, 7.7])
    anchor_y = 2.0 + 0.10 * np.sin(anchor_x * 0.9)
    ax.scatter(anchor_x, anchor_y, s=95, color="#d55e00", edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(anchor_x + np.array([-0.35, 0.15, 0.55]), anchor_y + np.array([0.62, 0.68, 0.58]), s=58, color="#4d4d4d", zorder=5)
    ax.scatter([4.7], [anchor_y[1]], s=170, facecolor="none", edgecolor="#0072b2", linewidth=1.8, zorder=3)

    ax.annotate("Route corridor exposure", xy=(1.2, 2.55), xytext=(0.8, 3.35),
                arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=0.9), color="#1f4e79", fontsize=8.5)
    ax.annotate("Feasible rendezvous anchors", xy=(4.7, anchor_y[1]), xytext=(3.4, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="#d55e00", lw=0.9), color="#d55e00", fontsize=8.5)
    ax.annotate("Nearby riders", xy=(7.9, 2.7), xytext=(8.1, 3.35),
                arrowprops=dict(arrowstyle="-|>", color="#4d4d4d", lw=0.9), color="#4d4d4d", fontsize=8.5)
    ax.text(5.0, 0.24, "Routes differ not just in exposure, but in the feasible and observable meeting opportunities they create.",
            ha="center", va="bottom", fontsize=8.4, color="#444444")
    ax.axis("off")
    _save(fig, "rendezvous_fig1_concept.png")


def fig2_primary() -> None:
    df = _load(RESULTS_DIR / "rendezvous_primary_summary.csv")
    if df is None or df.empty:
        return
    df = _filter_default_slice(df)
    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    order = [policy for policy in MAIN_POLICY_ORDER if policy in set(df["policy"])]
    sub = df.set_index("policy").loc[order].reset_index()
    x = np.arange(len(sub))
    _bar(ax, x, sub["mean_actual_profit"], policies=sub["policy"].tolist(), width=0.72)
    _style_axis(ax, ylabel="Mean Actual Profit")
    ax.set_title("Primary Single-Driver Policy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in sub["policy"]], rotation=0)
    for idx, value in enumerate(sub["mean_actual_profit"]):
        ax.text(idx, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, "rendezvous_fig2_primary.png")


def fig2_matched_pairs() -> None:
    df = _load(RESULTS_DIR / "rendezvous_observability_matched_summary.csv")
    if df is None or df.empty:
        return
    focus = df[
        (df["scenario_name"] == "sparse_high_occlusion")
        & (df["time_slice"].isin(["all_day", "morning_peak"]))
        & (df["area_slice"] == "all")
    ].copy()
    if focus.empty:
        return
    focus = focus.sort_values("time_slice")
    labels = [TIME_SLICE_LABELS.get(item, item.replace("_", " ")) for item in focus["time_slice"]]
    x = np.arange(len(focus))
    fig, ax1 = plt.subplots(figsize=(6.9, 3.7))
    ax1.bar(x, focus["mean_profit_delta"], color="#d55e00", alpha=0.88, edgecolor="#222222", linewidth=0.7, hatch="oo")
    yerr = np.vstack(
        [
            focus["mean_profit_delta"].to_numpy(dtype=float) - focus["ci_low"].to_numpy(dtype=float),
            focus["ci_high"].to_numpy(dtype=float) - focus["mean_profit_delta"].to_numpy(dtype=float),
        ]
    )
    ax1.errorbar(x, focus["mean_profit_delta"], yerr=yerr, fmt="none", ecolor="#1d3557", capsize=4)
    ax1.axhline(0.0, color="#cccccc", linewidth=0.8)
    _style_axis(ax1, ylabel="Higher - Lower Observability\nProfit Delta")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Matched Route-Pair Observability Isolation")

    ax2 = ax1.twinx()
    ax2.plot(x, focus["higher_observability_win_rate"], marker="o", markersize=6, linewidth=2.2, color="#0072b2")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Higher-Observability Win Rate")
    _save(fig, "rendezvous_fig2_matched_pairs.png")


def fig3_gap() -> None:
    df = _load(RESULTS_DIR / "rendezvous_nominal_realized_gap.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    order = [policy for policy in MAIN_POLICY_ORDER if policy in set(df["policy"])]
    sub = df.set_index("policy").loc[order].reset_index()
    x = np.arange(len(sub))
    _bar(ax, x, sub["mean_nominal_realized_gap"], policies=sub["policy"].tolist(), width=0.72)
    _style_axis(ax, ylabel="Mean Nominal - Realized Gap")
    ax.set_title("Nominal vs. Realized Service Gap")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in sub["policy"]], rotation=0)
    for idx, value in enumerate(sub["mean_nominal_realized_gap"]):
        ax.text(idx, value + 0.2, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    _save(fig, "rendezvous_fig3_gap.png")


def fig4_dispatch() -> None:
    df = _load(RESULTS_DIR / "rendezvous_dispatch_policy_summary.csv")
    if df is None or df.empty:
        return
    df = _filter_default_slice(df)
    focus = df[df["scenario_name"].isin(["primary", "sparse_high_occlusion"])].copy()
    if focus.empty:
        focus = df.copy()
    policies = [policy for policy in MAIN_POLICY_ORDER if policy in set(focus["policy"])]
    scenarios = list(dict.fromkeys(focus["scenario_name"].tolist()))

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    x = np.arange(len(scenarios))
    width = 0.18 if len(policies) >= 4 else 0.22
    offsets = np.linspace(-width * (len(policies) - 1) / 2.0, width * (len(policies) - 1) / 2.0, len(policies))
    for offset, policy in zip(offsets, policies):
        values = []
        for scenario in scenarios:
            sub = focus[(focus["scenario_name"] == scenario) & (focus["policy"] == policy)]
            values.append(float(sub["mean_profit_per_driver"].iloc[0]) if not sub.empty else np.nan)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            edgecolor="#222222",
            linewidth=0.6,
            hatch=POLICY_HATCHES[policy],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n")) for scenario in scenarios])
    _style_axis(ax, ylabel="Mean Profit per Driver")
    ax.set_title("Dispatch Validation in Primary and Hard Regimes")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.25)
    _save(fig, "rendezvous_fig4_dispatch.png")


def fig5_ml_comparator() -> None:
    df = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    if df is None or df.empty:
        return
    df = _filter_default_slice(df)
    focus = df[
        df["scenario_name"].isin(["primary", "sparse_high_occlusion"])
        & df["policy"].isin(["rendezvous_observable", "ml_meeting_point_comparator"])
    ].copy()
    if focus.empty:
        return
    scenarios = list(dict.fromkeys(focus["scenario_name"].tolist()))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(scenarios))
    width = 0.28
    for offset, policy, color in [
        (-width / 2.0, "rendezvous_observable", "#0d6efd"),
        (width / 2.0, "ml_meeting_point_comparator", "#dc3545"),
    ]:
        values = []
        for scenario in scenarios:
            sub = focus[(focus["scenario_name"] == scenario) & (focus["policy"] == policy)]
            values.append(float(sub["mean_actual_profit"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=POLICY_LABELS[policy],
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
            hatch=POLICY_HATCHES[policy],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n")) for scenario in scenarios])
    _style_axis(ax, ylabel="Mean Actual Profit")
    ax.set_title("Deterministic vs ML Meeting-Point Ranking")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, "rendezvous_fig5_ml_comparator.png")
    _save_alias("rendezvous_fig5_ml_comparator.png", "rendezvous_appendix_ml_comparator.png")


def fig6_sensitivity() -> None:
    df = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    if df is None or df.empty:
        return
    df = _filter_default_slice(df)
    focus = df[df["rider_density_pct"] == 10].copy()
    if focus.empty:
        focus = df.copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for policy in [policy for policy in MAIN_POLICY_ORDER if policy in set(focus["policy"])]:
        sub = focus[focus["policy"] == policy].sort_values("occlusion_lambda")
        ax.plot(
            sub["occlusion_lambda"],
            sub["mean_actual_profit"],
            marker=POLICY_MARKERS[policy],
            markersize=5.5,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            linewidth=2.0,
        )
    _style_axis(ax, xlabel="Occlusion Lambda", ylabel="Mean Actual Profit")
    ax.set_title("Occlusion Sensitivity in Very Sparse Demand")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(bottom=0.27)
    _save(fig, "rendezvous_fig6_sensitivity.png")


def fig7_strong_baselines() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    ci = _load(RESULTS_DIR / "rendezvous_policy_bootstrap_ci.csv")
    if summary is None or summary.empty or ci is None or ci.empty:
        return

    focus = _filter_default_slice(summary)
    focus = focus[focus["scenario_name"].isin(["primary", "sparse_high_occlusion"])].copy()
    ci_focus = _filter_default_slice(ci)
    ci_focus = ci_focus[(ci_focus["scenario_name"].isin(["primary", "sparse_high_occlusion"])) & (ci_focus["metric"] == "actual_profit")].copy()
    if focus.empty or ci_focus.empty:
        return

    scenarios = ["primary", "sparse_high_occlusion"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        sub = focus[focus["scenario_name"] == scenario].copy()
        sub_ci = ci_focus[ci_focus["scenario_name"] == scenario].copy()
        order = [policy for policy in STRONG_BASELINE_ORDER if policy in set(sub["policy"])]
        sub = sub.set_index("policy").loc[order].reset_index()
        sub_ci = sub_ci.set_index("policy").loc[order].reset_index()
        x = np.arange(len(sub))
        y = sub["mean_actual_profit"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub_ci["ci_low"].to_numpy(dtype=float),
                sub_ci["ci_high"].to_numpy(dtype=float) - y,
            ]
        )
        for xpos, value, policy, low_err, high_err in zip(x, y, sub["policy"], yerr[0], yerr[1]):
            ax.bar(
                xpos,
                value,
                color=POLICY_COLORS[policy],
                edgecolor="#222222",
                linewidth=0.6,
                hatch=POLICY_HATCHES[policy],
                yerr=np.array([[low_err], [high_err]]),
                capsize=3,
            )
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario.replace("_", " ")))
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in sub["policy"]], rotation=0)
        ax.axhline(0.0, color="#cccccc", linewidth=0.8)
        _style_axis(ax)
        for idx, value in enumerate(y):
            ax.text(idx, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Mean Actual Profit")
    fig.suptitle("Stronger Baseline Comparison with Bootstrap 95% Intervals", y=1.02)
    _save(fig, "rendezvous_fig7_strong_baselines.png")


def fig8_context_ablation() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    if summary is None or summary.empty:
        return
    focus = summary[
        (summary["scenario_name"] == "sparse_high_occlusion")
        & (summary["time_slice"] == "all_day")
        & (summary["observability_profile"] == "calibrated")
        & (summary["observability_ablation"] == "full")
        & (summary["policy"].isin(["rendezvous_only", "rendezvous_observable", "ml_meeting_point_comparator"]))
    ].copy()
    if focus.empty:
        return

    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    condition_order = [(True, "With urban context"), (False, "No urban context")]
    policy_order = ["rendezvous_only", "rendezvous_observable", "ml_meeting_point_comparator"]
    x = np.arange(len(policy_order))
    width = 0.28
    for offset, (use_context, label) in zip([-width / 2.0, width / 2.0], condition_order):
        values = []
        for policy in policy_order:
            sub = focus[(focus["policy"] == policy) & (focus["use_urban_context"] == use_context)]
            values.append(float(sub["mean_actual_profit"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color="#0072b2" if use_context else "#b8b8b8",
            edgecolor="#222222",
            linewidth=0.6,
            hatch="///" if use_context else "\\\\\\",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in policy_order])
    _style_axis(ax, ylabel="Mean Actual Profit")
    ax.set_title("Urban-Context Ablation in Sparse High Occlusion")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.26)
    _save(fig, "rendezvous_fig8_context_ablation.png")


def fig8_green_robustness() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_green_policy_summary.csv")
    dispatch = _load(RESULTS_DIR / "rendezvous_green_dispatch_policy_summary.csv")
    if summary is None or summary.empty:
        return
    focus = summary[
        (summary["scenario_name"].isin(["primary", "sparse_high_occlusion"]))
        & (summary["time_slice"] == "all_day")
        & (summary["observability_profile"] == "calibrated")
        & (summary["observability_ablation"] == "full")
        & (summary["use_urban_context"] == True)  # noqa: E712
        & (summary["policy"].isin(["corridor_only", "rendezvous_only", "rendezvous_observable"]))
    ].copy()
    if focus.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    scenarios = ["primary", "sparse_high_occlusion"]
    x = np.arange(len(scenarios))
    width = 0.22
    policies = ["corridor_only", "rendezvous_only", "rendezvous_observable"]
    offsets = np.linspace(-width, width, len(policies))
    for offset, policy in zip(offsets, policies):
        values = []
        for scenario in scenarios:
            sub = focus[(focus["scenario_name"] == scenario) & (focus["policy"] == policy)]
            values.append(float(sub["mean_actual_profit"].iloc[0]) if not sub.empty else np.nan)
        axes[0].bar(
            x + offset,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            edgecolor="#222222",
            linewidth=0.6,
            hatch=POLICY_HATCHES[policy],
            label=POLICY_LABELS[policy],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n")) for scenario in scenarios])
    _style_axis(axes[0], ylabel="Mean Actual Profit")
    axes[0].set_title("Green Single-Driver Transfer")
    axes[0].axhline(0.0, color="#cccccc", linewidth=0.8)

    if dispatch is not None and not dispatch.empty:
        dispatch_focus = dispatch[
            (dispatch["scenario_name"] == "sparse_high_occlusion")
            & (dispatch["time_slice"] == "all_day")
            & (dispatch["observability_profile"] == "calibrated")
            & (dispatch["observability_ablation"] == "full")
            & (dispatch["use_urban_context"] == True)  # noqa: E712
            & (dispatch["policy"].isin(policies))
        ].copy()
        bars = []
        for policy in policies:
            sub = dispatch_focus[dispatch_focus["policy"] == policy]
            bars.append(float(sub["mean_profit_per_driver"].iloc[0]) if not sub.empty else np.nan)
        axes[1].bar(
            np.arange(len(policies)),
            bars,
            color=[POLICY_COLORS[policy] for policy in policies],
            edgecolor="#222222",
            linewidth=0.6,
            hatch=None,
        )
        axes[1].set_xticks(np.arange(len(policies)))
        axes[1].set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in policies])
        _style_axis(axes[1], ylabel="Mean Profit per Driver")
        axes[1].set_title("Green Dispatch Transfer")
        axes[1].axhline(0.0, color="#cccccc", linewidth=0.8)
    else:
        axes[1].axis("off")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.05, -0.10), ncol=1)
    fig.subplots_adjust(bottom=0.24, wspace=0.30)
    _save(fig, "rendezvous_fig8_green_robustness.png")


def fig9_time_slice_robustness() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    dispatch = _load(RESULTS_DIR / "rendezvous_dispatch_policy_summary.csv")
    if summary is None or summary.empty or dispatch is None or dispatch.empty:
        return

    policies = ["corridor_only", "rendezvous_only", "rendezvous_observable"]
    single = summary[
        (summary["domain"] == "yellow")
        &
        (summary["scenario_name"] == "sparse_high_occlusion")
        & (summary["observability_profile"] == "calibrated")
        & (summary["observability_ablation"] == "full")
        & (summary["use_urban_context"] == True)  # noqa: E712
        & (summary["policy"].isin(policies))
        & (summary["time_slice"].isin(["all_day", "morning_peak", "evening_peak"]))
    ].copy()
    dispatch_focus = dispatch[
        (dispatch["domain"] == "yellow")
        &
        (dispatch["scenario_name"] == "sparse_high_occlusion")
        & (dispatch["observability_profile"] == "calibrated")
        & (dispatch["observability_ablation"] == "full")
        & (dispatch["use_urban_context"] == True)  # noqa: E712
        & (dispatch["policy"].isin(policies))
        & (dispatch["time_slice"].isin(["all_day", "morning_peak"]))
    ].copy()
    if single.empty or dispatch_focus.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=False)
    slice_order = ["all_day", "morning_peak", "evening_peak"]
    x = np.arange(len(slice_order))
    for policy in policies:
        sub = single[single["policy"] == policy].set_index("time_slice")
        values = [float(sub.loc[label, "mean_actual_profit"]) if label in sub.index else np.nan for label in slice_order]
        axes[0].plot(
            x,
            values,
            marker=POLICY_MARKERS[policy],
            markersize=5.5,
            linewidth=2.0,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TIME_SLICE_LABELS[label] for label in slice_order])
    _style_axis(axes[0], ylabel="Mean Actual Profit")
    axes[0].set_title("Single-Driver Sparse High Occlusion")

    dispatch_slice_order = ["all_day", "morning_peak"]
    x2 = np.arange(len(dispatch_slice_order))
    width = 0.22
    offsets = np.linspace(-width, width, len(policies))
    for offset, policy in zip(offsets, policies):
        sub = dispatch_focus[dispatch_focus["policy"] == policy].set_index("time_slice")
        values = [float(sub.loc[label, "mean_profit_per_driver"]) if label in sub.index else np.nan for label in dispatch_slice_order]
        axes[1].bar(
            x2 + offset,
            values,
            width=width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            edgecolor="#222222",
            linewidth=0.6,
            hatch=POLICY_HATCHES[policy],
        )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([TIME_SLICE_LABELS[label] for label in dispatch_slice_order])
    _style_axis(axes[1], ylabel="Mean Profit per Driver")
    axes[1].set_title("Dispatch Sparse High Occlusion")
    axes[1].axhline(0.0, color="#cccccc", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    _save(fig, "rendezvous_fig9_time_slice_robustness.png")


def main() -> None:
    fig1_concept()
    fig2_primary()
    fig2_matched_pairs()
    fig3_gap()
    fig4_dispatch()
    fig5_ml_comparator()
    fig6_sensitivity()
    fig7_strong_baselines()
    fig8_context_ablation()
    fig8_green_robustness()
    fig9_time_slice_robustness()


if __name__ == "__main__":
    main()
