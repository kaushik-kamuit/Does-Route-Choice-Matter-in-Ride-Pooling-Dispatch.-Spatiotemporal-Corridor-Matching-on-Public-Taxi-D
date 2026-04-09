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
POLICY_COLORS = {
    "corridor_only": "#6c757d",
    "time_only_baseline": "#8d99ae",
    "feasible_count_baseline": "#ffb703",
    "walk_aware_rendezvous": "#2a9d8f",
    "rendezvous_only": "#198754",
    "rendezvous_observable": "#0d6efd",
    "ml_meeting_point_comparator": "#dc3545",
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
    plt.close(fig)


def fig1_concept() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.plot([0.5, 9.5], [2.0, 2.0], color="#264653", linewidth=4)
    ax.scatter([2.0, 5.0, 8.0], [2.0, 2.0, 2.0], s=120, color="#2a9d8f", zorder=3)
    ax.scatter([1.6, 4.7, 8.5], [2.6, 3.0, 2.9], s=80, color="#e76f51", zorder=3)
    ax.annotate("Route corridor", xy=(1.0, 1.55), color="#264653")
    ax.annotate("Feasible rendezvous", xy=(4.4, 1.25), color="#2a9d8f")
    ax.annotate("Rider requests", xy=(7.4, 3.15), color="#e76f51")
    ax.text(6.0, 0.5, "Observable anchors stay valuable under occlusion", fontsize=9, color="#444444")
    ax.axis("off")
    _save(fig, "rendezvous_fig1_concept.png")


def fig2_primary() -> None:
    df = _load(RESULTS_DIR / "rendezvous_primary_summary.csv")
    if df is None or df.empty:
        return
    df = _filter_default_slice(df)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    order = [policy for policy in MAIN_POLICY_ORDER if policy in set(df["policy"])]
    sub = df.set_index("policy").loc[order].reset_index()
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_actual_profit"], color=[POLICY_COLORS[policy] for policy in sub["policy"]])
    ax.set_ylabel("Mean actual profit")
    ax.set_title("Primary single-driver policy comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in sub["policy"]], rotation=0)
    for idx, value in enumerate(sub["mean_actual_profit"]):
        ax.text(idx, value + 0.25, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, "rendezvous_fig2_primary.png")


def fig3_gap() -> None:
    df = _load(RESULTS_DIR / "rendezvous_nominal_realized_gap.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = [policy for policy in MAIN_POLICY_ORDER if policy in set(df["policy"])]
    sub = df.set_index("policy").loc[order].reset_index()
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_nominal_realized_gap"], color=[POLICY_COLORS[policy] for policy in sub["policy"]])
    ax.set_ylabel("Mean nominal-realized gap")
    ax.set_title("Nominal vs realized service gap")
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

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(scenarios))
    width = 0.18 if len(policies) >= 4 else 0.22
    offsets = np.linspace(-width * (len(policies) - 1) / 2.0, width * (len(policies) - 1) / 2.0, len(policies))
    for offset, policy in zip(offsets, policies):
        values = []
        for scenario in scenarios:
            sub = focus[(focus["scenario_name"] == scenario) & (focus["policy"] == policy)]
            values.append(float(sub["mean_profit_per_driver"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=POLICY_LABELS[policy], color=POLICY_COLORS[policy])

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n")) for scenario in scenarios])
    ax.set_ylabel("Mean profit per driver")
    ax.set_title("Dispatch validation in primary and hard regimes")
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
        ax.bar(x + offset, values, width=width, label=POLICY_LABELS[policy], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n")) for scenario in scenarios])
    ax.set_ylabel("Mean actual profit")
    ax.set_title("Deterministic vs ML meeting-point ranking")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, "rendezvous_fig5_ml_comparator.png")


def fig6_sensitivity() -> None:
    df = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    if df is None or df.empty:
        return
    df = _filter_default_slice(df)
    focus = df[df["rider_density_pct"] == 10].copy()
    if focus.empty:
        focus = df.copy()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for policy in [policy for policy in MAIN_POLICY_ORDER if policy in set(focus["policy"])]:
        sub = focus[focus["policy"] == policy].sort_values("occlusion_lambda")
        ax.plot(
            sub["occlusion_lambda"],
            sub["mean_actual_profit"],
            marker="o",
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
    ax.set_xlabel("Occlusion lambda")
    ax.set_ylabel("Mean actual profit")
    ax.set_title("Occlusion sensitivity in very sparse demand")
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
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
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
        ax.bar(x, y, color=[POLICY_COLORS[policy] for policy in sub["policy"]], yerr=yerr, capsize=3)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario.replace("_", " ")))
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in sub["policy"]], rotation=0)
        ax.axhline(0.0, color="#cccccc", linewidth=0.8)
        for idx, value in enumerate(y):
            ax.text(idx, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Mean actual profit")
    fig.suptitle("Stronger baseline comparison with bootstrap 95% intervals", y=1.02)
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

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    condition_order = [(True, "With urban context"), (False, "No urban context")]
    policy_order = ["rendezvous_only", "rendezvous_observable", "ml_meeting_point_comparator"]
    x = np.arange(len(policy_order))
    width = 0.28
    for offset, (use_context, label) in zip([-width / 2.0, width / 2.0], condition_order):
        values = []
        for policy in policy_order:
            sub = focus[(focus["policy"] == policy) & (focus["use_urban_context"] == use_context)]
            values.append(float(sub["mean_actual_profit"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=label, color="#0d6efd" if use_context else "#adb5bd")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in policy_order])
    ax.set_ylabel("Mean actual profit")
    ax.set_title("Urban-context ablation in sparse high occlusion")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.26)
    _save(fig, "rendezvous_fig8_context_ablation.png")


def fig9_time_slice_robustness() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    dispatch = _load(RESULTS_DIR / "rendezvous_dispatch_policy_summary.csv")
    if summary is None or summary.empty or dispatch is None or dispatch.empty:
        return

    policies = ["corridor_only", "rendezvous_only", "rendezvous_observable"]
    single = summary[
        (summary["scenario_name"] == "sparse_high_occlusion")
        & (summary["observability_profile"] == "calibrated")
        & (summary["observability_ablation"] == "full")
        & (summary["use_urban_context"] == True)  # noqa: E712
        & (summary["policy"].isin(policies))
        & (summary["time_slice"].isin(["all_day", "morning_peak", "evening_peak"]))
    ].copy()
    dispatch_focus = dispatch[
        (dispatch["scenario_name"] == "sparse_high_occlusion")
        & (dispatch["observability_profile"] == "calibrated")
        & (dispatch["observability_ablation"] == "full")
        & (dispatch["use_urban_context"] == True)  # noqa: E712
        & (dispatch["policy"].isin(policies))
        & (dispatch["time_slice"].isin(["all_day", "morning_peak"]))
    ].copy()
    if single.empty or dispatch_focus.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    slice_order = ["all_day", "morning_peak", "evening_peak"]
    x = np.arange(len(slice_order))
    for policy in policies:
        sub = single[single["policy"] == policy].set_index("time_slice")
        values = [float(sub.loc[label, "mean_actual_profit"]) if label in sub.index else np.nan for label in slice_order]
        axes[0].plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TIME_SLICE_LABELS[label] for label in slice_order])
    axes[0].set_ylabel("Mean actual profit")
    axes[0].set_title("Single-driver sparse high occlusion")

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
        )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([TIME_SLICE_LABELS[label] for label in dispatch_slice_order])
    axes[1].set_ylabel("Mean profit per driver")
    axes[1].set_title("Dispatch sparse high occlusion")
    axes[1].axhline(0.0, color="#cccccc", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    _save(fig, "rendezvous_fig9_time_slice_robustness.png")


def main() -> None:
    fig1_concept()
    fig2_primary()
    fig3_gap()
    fig4_dispatch()
    fig5_ml_comparator()
    fig6_sensitivity()
    fig7_strong_baselines()
    fig8_context_ablation()
    fig9_time_slice_robustness()


if __name__ == "__main__":
    main()
