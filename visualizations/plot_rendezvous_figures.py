from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAPER_FIG_DIR = ROOT / "paper_rendezvous" / "figures"
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.titlesize": 10.4,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "figure.titlesize": 10.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#d0d6de",
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
    "time_only_baseline": "",
    "feasible_count_baseline": "",
    "walk_aware_rendezvous": "",
    "rendezvous_only": "",
    "rendezvous_observable": "",
    "ml_meeting_point_comparator": "",
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
    "very_sparse_low_occlusion": "Boundary\nlow occlusion",
    "very_sparse_extreme_occlusion": "Boundary\nextreme occlusion",
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


def _filter_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    if "domain" in df.columns:
        return df[df["domain"] == domain].copy()
    return df.copy()


def _canonical_slice(
    df: pd.DataFrame,
    *,
    domain: str | None = None,
    scenario_names: list[str] | None = None,
    time_slice: str | None = "all_day",
    area_slice: str | None = "all",
    rider_density_pct: int | None = None,
    observability_profile: str | None = "calibrated",
    observability_ablation: str | None = "full",
    use_urban_context: bool | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if domain is not None and "domain" in filtered.columns:
        filtered = filtered[filtered["domain"] == domain]
    if scenario_names is not None and "scenario_name" in filtered.columns:
        filtered = filtered[filtered["scenario_name"].isin(scenario_names)]
    if time_slice is not None and "time_slice" in filtered.columns:
        filtered = filtered[filtered["time_slice"] == time_slice]
    if area_slice is not None and "area_slice" in filtered.columns:
        filtered = filtered[filtered["area_slice"] == area_slice]
    if rider_density_pct is not None and "rider_density_pct" in filtered.columns:
        filtered = filtered[filtered["rider_density_pct"] == rider_density_pct]
    if observability_profile is not None and "observability_profile" in filtered.columns:
        filtered = filtered[filtered["observability_profile"] == observability_profile]
    if observability_ablation is not None and "observability_ablation" in filtered.columns:
        filtered = filtered[filtered["observability_ablation"] == observability_ablation]
    if use_urban_context is not None and "use_urban_context" in filtered.columns:
        filtered = filtered[filtered["use_urban_context"] == use_urban_context]
    return filtered.copy()


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PAPER_FIG_DIR / name, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    pdf_name = Path(name).with_suffix(".pdf").name
    fig.savefig(PLOTS_DIR / pdf_name, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PAPER_FIG_DIR / pdf_name, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _finalize_axes_text(fig: plt.Figure, size: float = 10.0) -> None:
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_fontsize() is not None:
            text.set_fontsize(size)


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
    ax.grid(axis="y", linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)


def _load_ci(
    *,
    file_name: str,
    metric: str,
    domain: str,
    scenario_names: list[str],
    area_slice: str = "all",
    rider_density_pct: int | None = None,
    time_slice: str | None = "all_day",
    use_urban_context: bool | None = True,
    observability_profile: str | None = "calibrated",
    observability_ablation: str | None = "full",
) -> pd.DataFrame:
    ci = _load(RESULTS_DIR / file_name)
    if ci is None or ci.empty:
        return pd.DataFrame()
    ci = _canonical_slice(
        ci,
        domain=domain,
        scenario_names=scenario_names,
        time_slice=time_slice,
        area_slice=area_slice,
        rider_density_pct=rider_density_pct,
        observability_profile=observability_profile,
        observability_ablation=observability_ablation,
        use_urban_context=use_urban_context,
    )
    if "metric" in ci.columns:
        ci = ci[ci["metric"] == metric].copy()
    return ci


def _box_anchor(patch: FancyBboxPatch, side: str) -> tuple[float, float]:
    x = patch.get_x()
    y = patch.get_y()
    w = patch.get_width()
    h = patch.get_height()
    if side == "left":
        return (x, y + h / 2.0)
    if side == "right":
        return (x + w, y + h / 2.0)
    if side == "top":
        return (x + w / 2.0, y + h)
    return (x + w / 2.0, y)


def _draw_round_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str = "#f7f9fc",
    edgecolor: str = "#7a8793",
    fontsize: float = 8.2,
    weight: str = "normal",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f2933",
        weight=weight,
        linespacing=1.15,
        zorder=2,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    lw: float = 1.0,
    color: str = "#4b5563",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": lw,
            "color": color,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=3,
    )


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


def fig0_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    _draw_round_box(ax, x=0.03, y=0.08, w=0.24, h=0.84, text="", facecolor="#edf3fb")
    _draw_round_box(ax, x=0.31, y=0.08, w=0.40, h=0.84, text="", facecolor="#f4f7fb")
    _draw_round_box(ax, x=0.75, y=0.08, w=0.22, h=0.84, text="", facecolor="#faf6ee")
    ax.text(0.15, 0.885, "1. Offline Data and Context", ha="center", va="center", fontsize=11.3, weight="bold", color="#1f2933")
    ax.text(0.51, 0.885, "2. Online Route Valuation", ha="center", va="center", fontsize=11.3, weight="bold", color="#1f2933")
    ax.text(0.86, 0.885, "3. Evaluation", ha="center", va="center", fontsize=11.3, weight="bold", color="#1f2933")

    left_y = [0.74, 0.59, 0.44, 0.29, 0.14]
    left_labels = [
        "Urban Context\nStreet, sidewalk, building,\nland-use layers",
        "Observability Calibration\nYellow-based geometry\nand context profile",
        "Mobility Demand\nNYC Yellow/Green trips",
        "Driver / Rider Split\nProxy roles and\ntrain/test partition",
        "Routes and H3 Assets\nOSRM alternatives,\ncorridors, anchors",
    ]
    left_boxes = [
        _draw_round_box(ax, x=0.045, y=y, w=0.205, h=0.105, text=label, facecolor="#ffffff", fontsize=8.0)
        for y, label in zip(left_y, left_labels)
    ]

    mid_boxes = {
        "active": _draw_round_box(ax, x=0.35, y=0.57, w=0.13, h=0.11, text="Active Query\nDriver trip, rider snapshot,\nroute set, seats", facecolor="#ffffff", fontsize=8.2),
        "candidate": _draw_round_box(ax, x=0.51, y=0.57, w=0.13, h=0.11, text="Candidate Retrieval\nCorridor-based spatial\nand temporal filtering", facecolor="#ffffff", fontsize=8.05),
        "opportunity": _draw_round_box(ax, x=0.35, y=0.35, w=0.13, h=0.11, text="Opportunity Generation\nRoute-induced rendezvous\nopportunities", facecolor="#ffffff", fontsize=8.05),
        "feasibility": _draw_round_box(ax, x=0.51, y=0.35, w=0.13, h=0.12, text="Feasibility + Observability\nWalk, timing, and\npickup-reliability screening", facecolor="#edf7f1", fontsize=7.9),
        "scoring": _draw_round_box(ax, x=0.43, y=0.16, w=0.18, h=0.13, text="Route Scoring\nExposure-only,\nrendezvous-aware, and\nobservability-aware policies", facecolor="#ffffff", fontsize=8.0),
    }

    right_y = [0.72, 0.56, 0.40, 0.24, 0.10]
    right_labels = [
        "Single-Driver Study\nIsolated route-choice analysis",
        "Dispatch Validation",
        "Matched Route Pairs\nExposure-controlled comparison",
        "Green Transfer\nCross-domain robustness",
        "Case Studies\nLocal-geometry validation",
    ]
    right_boxes = [
        _draw_round_box(ax, x=0.78, y=y, w=0.16, h=0.09, text=label, facecolor="#ffffff", fontsize=8.1)
        for y, label in zip(right_y, right_labels)
    ]

    _arrow(ax, _box_anchor(mid_boxes["active"], "right"), _box_anchor(mid_boxes["candidate"], "left"), color="#65758b")
    _arrow(ax, _box_anchor(mid_boxes["active"], "bottom"), _box_anchor(mid_boxes["opportunity"], "top"), color="#65758b")
    _arrow(ax, _box_anchor(mid_boxes["candidate"], "bottom"), _box_anchor(mid_boxes["feasibility"], "top"), color="#65758b")
    _arrow(ax, _box_anchor(mid_boxes["opportunity"], "right"), _box_anchor(mid_boxes["feasibility"], "left"), color="#65758b")
    _arrow(ax, _box_anchor(mid_boxes["feasibility"], "bottom"), _box_anchor(mid_boxes["scoring"], "top"), color="#65758b")

    _arrow(ax, _box_anchor(left_boxes[0], "right"), _box_anchor(mid_boxes["feasibility"], "left"), rad=-0.01, color="#65758b")
    _arrow(ax, _box_anchor(left_boxes[1], "right"), _box_anchor(mid_boxes["feasibility"], "left"), color="#65758b")
    _arrow(ax, _box_anchor(left_boxes[2], "right"), _box_anchor(mid_boxes["active"], "left"), color="#65758b")
    _arrow(ax, _box_anchor(left_boxes[3], "right"), _box_anchor(mid_boxes["active"], "left"), rad=0.06, color="#65758b")
    _arrow(ax, _box_anchor(left_boxes[4], "right"), _box_anchor(mid_boxes["candidate"], "left"), rad=0.02, color="#65758b")

    for box in right_boxes:
        _arrow(ax, _box_anchor(mid_boxes["scoring"], "right"), _box_anchor(box, "left"), color="#65758b")

    ax.text(
        0.5,
        0.03,
        "Route value should depend on feasible and observable rendezvous opportunities, not corridor proximity alone.",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#2f2f2f",
    )
    _save(fig, "rendezvous_fig0_pipeline_overview.png")


def fig1_concept() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)

    corridor_x = np.linspace(0.7, 9.3, 260)
    center_y = 2.1 + 0.14 * np.sin(corridor_x * 0.85)
    upper_y = center_y + 0.48
    lower_y = center_y - 0.48
    ax.fill_between(corridor_x, lower_y, upper_y, color="#dceaf7", alpha=0.98, zorder=1)
    ax.plot(corridor_x, center_y, color="#143d59", linewidth=2.9, zorder=3)

    anchor_x = np.array([2.0, 5.0, 7.8])
    anchor_y = 2.1 + 0.14 * np.sin(anchor_x * 0.85)
    rider_x = np.array([1.5, 4.4, 8.5])
    rider_y = np.array([3.25, 3.45, 3.05])
    selected_idx = 1

    ax.scatter(anchor_x, anchor_y, s=88, color="#2a9d8f", edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter(rider_x, rider_y, s=72, color="#6f1d9b", edgecolor="white", linewidth=0.8, zorder=6)
    for idx, (rx, ry, ax_x, ax_y) in enumerate(zip(rider_x, rider_y, anchor_x, anchor_y)):
        linestyle = "--" if idx != selected_idx else "-"
        alpha = 0.65 if idx != selected_idx else 0.95
        ax.plot([rx, ax_x], [ry, ax_y], color="#6c757d", linestyle=linestyle, linewidth=1.2, alpha=alpha, zorder=4)
    ax.scatter(
        [anchor_x[selected_idx]],
        [anchor_y[selected_idx]],
        s=190,
        marker="*",
        color="#d62828",
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    ax.scatter(
        [anchor_x[selected_idx]],
        [anchor_y[selected_idx]],
        s=280,
        facecolor="none",
        edgecolor="#0072b2",
        linewidth=1.8,
        zorder=4,
    )

    ax.annotate(
        "Expanded route corridor",
        xy=(1.15, 2.72),
        xytext=(0.85, 4.02),
        fontsize=12.4,
        fontweight="bold",
        color="#143d59",
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color="#143d59", lw=1.1, shrinkA=2, shrinkB=3),
    )
    ax.annotate(
        "Corridor-near riders",
        xy=(8.35, 3.10),
        xytext=(7.55, 4.08),
        fontsize=12.4,
        fontweight="bold",
        color="#6f1d9b",
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color="#6f1d9b", lw=1.1, shrinkA=2, shrinkB=3),
    )
    ax.annotate(
        "Route-anchor meeting\ncandidates",
        xy=(1.92, anchor_y[0]),
        xytext=(0.78, 0.70),
        fontsize=11.8,
        fontweight="bold",
        color="#2a9d8f",
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=1.1, shrinkA=2, shrinkB=3),
    )
    ax.annotate(
        "Best feasible and observable\ncommon meeting point",
        xy=(anchor_x[selected_idx], anchor_y[selected_idx]),
        xytext=(5.65, 0.60),
        fontsize=11.8,
        fontweight="bold",
        color="#d62828",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=1.1, shrinkA=2, shrinkB=3),
    )
    ax.text(
        5.0,
        0.16,
        "Routes should be valued by the common meeting opportunities they create, not by proximity alone.",
        ha="center",
        va="bottom",
        fontsize=12.2,
        fontweight="bold",
        color="#444444",
    )
    ax.axis("off")
    _finalize_axes_text(fig, size=10.6)
    _save(fig, "rendezvous_fig1_concept.png")


def fig2_primary() -> None:
    df = _load(RESULTS_DIR / "rendezvous_primary_summary.csv")
    if df is None or df.empty:
        return
    df = _canonical_slice(df, domain="yellow", scenario_names=["primary"], area_slice="all")
    order = [policy for policy in MAIN_POLICY_ORDER if policy in set(df["policy"])]
    if not order:
        return
    ci = _load_ci(
        file_name="rendezvous_policy_bootstrap_ci.csv",
        metric="actual_profit",
        domain="yellow",
        scenario_names=["primary"],
        rider_density_pct=100,
        time_slice="all_day",
        use_urban_context=True,
    )
    if ci.empty:
        return
    sub = df.set_index("policy").loc[order].reset_index()
    sub_ci = ci.set_index("policy").loc[order].reset_index()
    y = np.arange(len(order))
    mean_vals = sub["mean_actual_profit"].to_numpy(dtype=float)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.9),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 2.25], "wspace": 0.05},
    )
    for ax in [ax_left, ax_right]:
        ax.grid(axis="x", linestyle="-", alpha=0.45)
        ax.set_axisbelow(True)
    for yi, policy, mean in zip(y, order, mean_vals):
        ci_row = sub_ci[sub_ci["policy"] == policy].iloc[0]
        ci_low = float(ci_row["ci_low"])
        ci_high = float(ci_row["ci_high"])
        for ax in [ax_left, ax_right]:
            ax.barh(yi, mean, color=POLICY_COLORS[policy], edgecolor="#303030", height=0.58)
            ax.errorbar(mean, yi, xerr=[[mean - ci_low], [ci_high - mean]], fmt="none", ecolor="#1f2933", elinewidth=1.1, capsize=3)
        if policy == "corridor_only":
            ax_left.text(mean + 0.10, yi, f"{mean:.2f}", va="center", ha="left", fontsize=8.0)
        else:
            ax_right.text(mean + 0.14, yi, f"{mean:.2f}", va="center", ha="left", fontsize=8.0)
    ax_left.set_xlim(0.0, 8.4)
    ax_right.set_xlim(15.0, 18.8)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([POLICY_LABELS[p] for p in order])
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_left.invert_yaxis()
    ax_left.set_title("Primary Single-Driver Policy Comparison")
    ax_left.set_xlabel("Mean Actual Profit")
    ax_right.set_xlabel("Mean Actual Profit")
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    d = 0.008
    kwargs = dict(transform=ax_left.transAxes, color="#444444", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    _save(fig, "rendezvous_fig2_primary.png")


def fig2_matched_pairs() -> None:
    df = _load(RESULTS_DIR / "rendezvous_observability_matched_summary.csv")
    if df is None or df.empty:
        return
    focus = _canonical_slice(
        df,
        domain="yellow",
        scenario_names=["sparse_high_occlusion"],
        time_slice=None,
        area_slice="all",
        rider_density_pct=10,
        observability_profile="calibrated",
        observability_ablation="full",
        use_urban_context=True,
    )
    focus = focus[focus["time_slice"].isin(["all_day", "morning_peak"])].copy()
    focus = focus.drop_duplicates(subset=["time_slice"], keep="first")
    if focus.empty:
        return
    focus["time_sort"] = focus["time_slice"].map({"all_day": 0, "morning_peak": 1}).fillna(99)
    focus = focus.sort_values("time_sort").drop(columns=["time_sort"])
    labels = [TIME_SLICE_LABELS.get(item, item.replace("_", " ")) for item in focus["time_slice"]]
    y = np.arange(len(focus))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.45), gridspec_kw={"wspace": 0.26, "width_ratios": [1.12, 0.88]})
    ax_delta, ax_win = axes
    delta_vals = focus["mean_profit_delta"].to_numpy(dtype=float)
    ci_low = focus["ci_low"].to_numpy(dtype=float)
    ci_high = focus["ci_high"].to_numpy(dtype=float)
    n_pairs = focus["n_unique_pairs"].to_numpy(dtype=float)
    for yy, lo, hi in zip(y, ci_low, ci_high):
        ax_delta.hlines(yy, lo, hi, color="#1d3557", linewidth=2.0, zorder=2)
    ax_delta.scatter(delta_vals, y, s=82, color="#e6862a", edgecolor="#2c2c2c", linewidth=0.7, zorder=3)
    ax_delta.axvline(0.0, color="#c8cdd3", linewidth=0.9)
    _style_axis(ax_delta, xlabel="Profit delta")
    ax_delta.grid(axis="x", linestyle="-", alpha=0.6)
    ax_delta.grid(axis="y", visible=False)
    ax_delta.set_title("Higher minus lower observability", pad=8, fontweight="bold")
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(labels)
    ax_delta.set_ylim(-0.55, len(y) - 0.45)
    xmax = max(float(np.nanmax(ci_high)), float(np.nanmax(delta_vals))) + 0.75
    ax_delta.set_xlim(0.0, xmax)
    for yy, value, pairs in zip(y, delta_vals, n_pairs):
        ax_delta.text(value + 0.12, yy + 0.07, f"{value:.2f}", ha="left", va="bottom", fontsize=8.8, color="#2c2c2c")
        ax_delta.text(0.06, yy - 0.18, f"n={int(pairs)} matched pairs", ha="left", va="center", fontsize=8.8, color="#5b6570")

    win_vals = focus["higher_observability_win_rate"].to_numpy(dtype=float)
    ax_win.hlines(y, 0.5, win_vals, color="#8db7dd", linewidth=6.0, alpha=0.95, zorder=1)
    ax_win.scatter(win_vals, y, s=78, color="#4f8fc7", edgecolor="#2c2c2c", linewidth=0.7, zorder=3)
    ax_win.axvline(0.5, color="#aab2bd", linewidth=0.9, linestyle="--")
    _style_axis(ax_win, xlabel="Win rate")
    ax_win.grid(axis="x", linestyle="-", alpha=0.6)
    ax_win.grid(axis="y", visible=False)
    ax_win.set_title("Higher-observability win rate", pad=8, fontweight="bold")
    ax_win.set_yticks(y)
    ax_win.set_yticklabels(labels)
    ax_win.tick_params(axis="y", length=0)
    ax_win.set_xlim(0.0, 1.0)
    ax_win.set_ylim(-0.55, len(y) - 0.45)
    for yy, value in zip(y, win_vals):
        ax_win.text(
            min(value + 0.035, 0.94),
            yy + 0.07,
            f"{value:.2f}",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#2c2c2c",
        )
    fig.suptitle("Matched route-pair observability isolation", y=1.02, fontweight="bold")
    _finalize_axes_text(fig, size=8.8)
    _save(fig, "rendezvous_fig2_matched_pairs.png")


def fig3_gap() -> None:
    df = _load(RESULTS_DIR / "rendezvous_primary_summary.csv")
    if df is None or df.empty:
        return
    df = _canonical_slice(df, domain="yellow", scenario_names=["primary"], area_slice="all")
    if df.empty:
        return
    df = (
        df.groupby("policy", as_index=False)["mean_nominal_realized_gap"]
        .mean()
        .sort_values("mean_nominal_realized_gap", ascending=False)
    )
    order = [policy for policy in MAIN_POLICY_ORDER if policy in set(df["policy"])]
    sub = df.set_index("policy").loc[order].reset_index()
    y = np.arange(len(sub))
    values = sub["mean_nominal_realized_gap"].to_numpy(dtype=float)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(6.9, 3.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.05},
    )
    for ax in [ax_left, ax_right]:
        ax.grid(axis="x", linestyle="-", alpha=0.45)
        ax.set_axisbelow(True)
    for yi, value, policy in zip(y, values, sub["policy"]):
        for ax in [ax_left, ax_right]:
            ax.barh(yi, value, color=POLICY_COLORS[policy], edgecolor="#303030", height=0.58)
        if policy == "corridor_only":
            ax_right.text(value + 0.20, yi, f"{value:.1f}", va="center", ha="left", fontsize=8.8)
        else:
            ax_left.text(value + 0.12, yi, f"{value:.1f}", va="center", ha="left", fontsize=8.8)
    ax_left.set_xlim(0.0, 6.2)
    ax_right.set_xlim(22.0, 24.8)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels([POLICY_LABELS[policy] for policy in sub["policy"]])
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_left.invert_yaxis()
    ax_left.set_title("Nominal vs. Realized Service Gap", fontweight="bold")
    ax_left.set_xlabel("Mean Nominal - Realized Gap")
    ax_right.set_xlabel("Mean Nominal - Realized Gap")
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    d = 0.008
    kwargs = dict(transform=ax_left.transAxes, color="#444444", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    _finalize_axes_text(fig, size=8.8)
    _save(fig, "rendezvous_fig3_gap.png")


def fig4_dispatch() -> None:
    df = _load(RESULTS_DIR / "rendezvous_dispatch_policy_summary.csv")
    if df is None or df.empty:
        return
    primary = _canonical_slice(
        df,
        domain="yellow",
        scenario_names=["primary"],
        area_slice="all",
        rider_density_pct=100,
    )
    hard = _canonical_slice(
        df,
        domain="yellow",
        scenario_names=["sparse_high_occlusion"],
        area_slice="all",
        rider_density_pct=10,
    )
    focus = pd.concat([primary, hard], ignore_index=True)
    if focus.empty:
        focus = _canonical_slice(df, domain="yellow", area_slice="all")
    ci = _load_ci(
        file_name="rendezvous_dispatch_bootstrap_ci.csv",
        metric="profit_per_driver",
        domain="yellow",
        scenario_names=["primary", "sparse_high_occlusion"],
        rider_density_pct=None,
        time_slice="all_day",
        use_urban_context=True,
    )
    policies = [policy for policy in ["corridor_only", "rendezvous_only", "rendezvous_observable"] if policy in set(focus["policy"])]
    if not policies:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"wspace": 0.28})
    for ax, scenario, title in zip(axes, ["primary", "sparse_high_occlusion"], ["Primary dispatch", "Sparse high occlusion dispatch"]):
        sub = focus[focus["scenario_name"] == scenario].set_index("policy")
        sub_ci = ci[ci["scenario_name"] == scenario].set_index("policy") if not ci.empty else pd.DataFrame()
        shown_policies = [policy for policy in policies if policy in sub.index]
        vals = [float(sub.loc[policy, "mean_profit_per_driver"]) for policy in shown_policies]
        xpos = np.arange(len(shown_policies))
        ax.bar(xpos, vals, color=[POLICY_COLORS[p] for p in shown_policies], edgecolor="#2d2d2d", linewidth=0.7, width=0.62)
        for idx, policy in enumerate(shown_policies):
            label_y = vals[idx] + (0.16 if vals[idx] >= 0 else 0.14)
            if not sub_ci.empty and policy in sub_ci.index:
                mean = vals[idx]
                ci_row = sub_ci.loc[[policy]].iloc[0]
                low = mean - float(ci_row["ci_low"])
                high = float(ci_row["ci_high"]) - mean
                ax.errorbar(idx, mean, yerr=[[low], [high]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.1)
                label_y = mean + high + 0.10 if mean >= 0 else mean + 0.14
            ax.text(
                idx,
                label_y,
                f"{vals[idx]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )
        ax.axhline(0.0, color="#bdbdbd", linewidth=0.9)
        ax.set_xticks(xpos)
        ax.set_xticklabels([POLICY_AXIS_LABELS[p] for p in shown_policies])
        _style_axis(ax, ylabel="Mean Profit per Driver" if scenario == "primary" else None)
        ax.set_title(title, fontweight="bold")
    fig.suptitle("Dispatch Validation in Primary and Hard Regimes", y=1.02, fontweight="bold")
    _finalize_axes_text(fig, size=8.8)
    _save(fig, "rendezvous_fig4_dispatch.png")


def fig5_ml_comparator() -> None:
    df = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    if df is None or df.empty:
        return
    primary = _canonical_slice(
        df,
        domain="yellow",
        scenario_names=["primary"],
        area_slice="all",
        rider_density_pct=100,
        use_urban_context=True,
    )
    hard = _canonical_slice(
        df,
        domain="yellow",
        scenario_names=["sparse_high_occlusion"],
        area_slice="all",
        rider_density_pct=10,
        use_urban_context=True,
    )
    focus = pd.concat([primary, hard], ignore_index=True)
    focus = focus[focus["policy"].isin(["rendezvous_observable", "ml_meeting_point_comparator"])].copy()
    valid_scenarios = []
    for scenario in ["primary", "sparse_high_occlusion"]:
        scenario_df = focus[focus["scenario_name"] == scenario]
        policies = set(scenario_df["policy"])
        if {"rendezvous_observable", "ml_meeting_point_comparator"}.issubset(policies):
            valid_scenarios.append(scenario)
    focus = focus[focus["scenario_name"].isin(valid_scenarios)].copy()
    if focus.empty:
        return
    ci = _load_ci(
        file_name="rendezvous_policy_bootstrap_ci.csv",
        metric="actual_profit",
        domain="yellow",
        scenario_names=["primary", "sparse_high_occlusion"],
        rider_density_pct=None,
        time_slice="all_day",
        use_urban_context=True,
    )
    scenarios = [item for item in ["primary", "sparse_high_occlusion"] if item in set(focus["scenario_name"])]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.0, 3.4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        scenario_df = focus[focus["scenario_name"] == scenario].set_index("policy")
        shown = ["rendezvous_observable", "ml_meeting_point_comparator"]
        vals = [float(scenario_df.loc[p, "mean_actual_profit"]) for p in shown]
        xpos = np.arange(len(shown))
        ax.bar(
            xpos,
            vals,
            color=[POLICY_COLORS[p] for p in shown],
            edgecolor="#2d2d2d",
            linewidth=0.7,
            width=0.58,
        )
        for idx, policy in enumerate(shown):
            ci_sub = ci[(ci["scenario_name"] == scenario) & (ci["policy"] == policy)]
            if not ci_sub.empty:
                mean = vals[idx]
                low = mean - float(ci_sub["ci_low"].iloc[0])
                high = float(ci_sub["ci_high"].iloc[0]) - mean
                ax.errorbar(idx, mean, yerr=[[low], [high]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.1)
            ax.text(idx, vals[idx] + (0.18 if vals[idx] >= 0 else -0.18), f"{vals[idx]:.2f}", ha="center", va="bottom" if vals[idx] >= 0 else "top", fontsize=7.8)
        ax.text(0.5, 0.96, f"ML - deterministic = {vals[1] - vals[0]:+.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7.9, color="#4b5563")
        ax.axhline(0.0, color="#cccccc", linewidth=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["Deterministic\nobservability", "ML\ncomparator"])
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario.replace("_", " ")))
        _style_axis(ax, ylabel="Mean Actual Profit" if scenario == scenarios[0] else None)
    fig.suptitle("Deterministic vs ML Meeting-Point Ranking", y=1.02)
    _save(fig, "rendezvous_fig5_ml_comparator.png")
    _save_alias("rendezvous_fig5_ml_comparator.png", "rendezvous_appendix_ml_comparator.png")


def fig6_sensitivity() -> None:
    df = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    if df is None or df.empty:
        return
    focus = _canonical_slice(
        df,
        domain="yellow",
        scenario_names=["very_sparse_low_occlusion", "very_sparse_extreme_occlusion"],
        area_slice="all",
        rider_density_pct=10,
    )
    if focus.empty:
        return
    scenario_order = ["very_sparse_low_occlusion", "very_sparse_extreme_occlusion"]
    policies = [policy for policy in MAIN_POLICY_ORDER if policy in set(focus["policy"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, scenario in zip(axes, scenario_order):
        sub = focus[focus["scenario_name"] == scenario].set_index("policy")
        shown = [p for p in policies if p in sub.index]
        vals = [float(sub.loc[p, "mean_actual_profit"]) for p in shown]
        ypos = np.arange(len(shown))
        ax.barh(ypos, vals, color=[POLICY_COLORS[p] for p in shown], edgecolor="#303030", height=0.58)
        ax.axvline(0.0, color="#bdbdbd", linewidth=0.9)
        for yi, val in zip(ypos, vals):
            ax.text(val + (0.08 if val >= 0 else -0.08), yi, f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=7.8)
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario.replace("_", " ")))
        ax.set_yticks(ypos)
        ax.set_yticklabels([POLICY_LABELS[p] for p in shown])
        _style_axis(ax, xlabel="Mean Actual Profit")
        ax.invert_yaxis()
    fig.suptitle("Boundary-Case Sensitivity at Density 10", y=1.02)
    _save(fig, "rendezvous_fig6_sensitivity.png")


def fig7_strong_baselines() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    ci = _load(RESULTS_DIR / "rendezvous_policy_bootstrap_ci.csv")
    if summary is None or summary.empty or ci is None or ci.empty:
        return

    focus = _filter_default_slice(summary)
    focus = _canonical_slice(
        focus,
        domain="yellow",
        scenario_names=["primary", "sparse_high_occlusion"],
        area_slice="all",
    )
    ci_focus = _canonical_slice(
        _filter_default_slice(ci),
        domain="yellow",
        scenario_names=["primary", "sparse_high_occlusion"],
        area_slice="all",
    )
    ci_focus = ci_focus[ci_focus["metric"] == "actual_profit"].copy()
    if focus.empty or ci_focus.empty:
        return

    scenarios = ["primary", "sparse_high_occlusion"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, scenario in zip(axes, scenarios):
        sub = focus[focus["scenario_name"] == scenario].copy()
        sub_ci = ci_focus[ci_focus["scenario_name"] == scenario].copy()
        order = [
            policy
            for policy in STRONG_BASELINE_ORDER
            if policy in set(sub["policy"]) and policy in set(sub_ci["policy"])
        ]
        if not order:
            continue
        sub = sub.set_index("policy").loc[order].reset_index()
        sub_ci = sub_ci.set_index("policy").loc[order].reset_index()
        y = np.arange(len(sub))
        means = sub["mean_actual_profit"].to_numpy(dtype=float)
        ax.barh(y, means, color=[POLICY_COLORS[p] for p in sub["policy"]], edgecolor="#303030", height=0.56)
        yerr = np.vstack(
            [
                means - sub_ci["ci_low"].to_numpy(dtype=float),
                sub_ci["ci_high"].to_numpy(dtype=float) - means,
            ]
        )
        ax.errorbar(means, y, xerr=yerr, fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.0)
        for yi, mean in zip(y, means):
            ax.text(mean + 0.08, yi, f"{mean:.2f}", va="center", ha="left", fontsize=7.5, color="#1f2933")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario.replace("_", " ")))
        ax.axvline(0.0, color="#cccccc", linewidth=0.8)
        _style_axis(ax, xlabel="Mean Actual Profit")
        ax.set_yticks(y)
        ax.set_yticklabels([POLICY_LABELS[p] for p in sub["policy"]])
        ax.invert_yaxis()
    axes[0].set_ylabel("Policy")
    fig.suptitle("Stronger Baseline Comparison with Bootstrap 95% Intervals", y=1.02)
    _save(fig, "rendezvous_fig7_strong_baselines.png")


def fig8_context_ablation() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    ci = _load(RESULTS_DIR / "rendezvous_policy_bootstrap_ci.csv")
    if summary is None or summary.empty:
        return
    focus = _canonical_slice(
        summary,
        domain="yellow",
        scenario_names=["sparse_high_occlusion"],
        area_slice="all",
        rider_density_pct=10,
        use_urban_context=None,
    )
    focus = focus[
        focus["policy"].isin(["rendezvous_only", "rendezvous_observable", "ml_meeting_point_comparator"])
    ].copy()
    if focus.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    policy_order = ["rendezvous_only", "rendezvous_observable", "ml_meeting_point_comparator"]
    x = np.arange(len(policy_order))
    width = 0.32
    ci_focus = pd.DataFrame()
    if ci is not None and not ci.empty:
        ci_focus = _canonical_slice(
            ci,
            domain="yellow",
            scenario_names=["sparse_high_occlusion"],
            area_slice="all",
            rider_density_pct=10,
            use_urban_context=None,
        )
        if "metric" in ci_focus.columns:
            ci_focus = ci_focus[ci_focus["metric"] == "actual_profit"].copy()
    for idx, policy in enumerate(policy_order):
        no_ctx = focus[(focus["policy"] == policy) & (focus["use_urban_context"] == False)]  # noqa: E712
        yes_ctx = focus[(focus["policy"] == policy) & (focus["use_urban_context"] == True)]  # noqa: E712
        if no_ctx.empty or yes_ctx.empty:
            continue
        x0 = float(no_ctx["mean_actual_profit"].iloc[0])
        x1 = float(yes_ctx["mean_actual_profit"].iloc[0])
        ax.bar(idx - width / 2, x0, width=width, color="#b7bec8", edgecolor="#2d2d2d", linewidth=0.7, label="No urban context" if idx == 0 else None)
        ax.bar(idx + width / 2, x1, width=width, color=POLICY_COLORS[policy], edgecolor="#2d2d2d", linewidth=0.7, label="With urban context" if idx == 0 else None)
        ax.text(idx, max(x0, x1) + 0.07, f"\N{GREEK CAPITAL LETTER DELTA}={x1 - x0:+.2f}", fontsize=7.8, color="#4b5563", ha="center")
        if not ci_focus.empty:
            no_ci = ci_focus[(ci_focus["policy"] == policy) & (ci_focus["use_urban_context"] == False)]  # noqa: E712
            yes_ci = ci_focus[(ci_focus["policy"] == policy) & (ci_focus["use_urban_context"] == True)]  # noqa: E712
            if not no_ci.empty:
                ax.errorbar(idx - width / 2, x0, yerr=[[x0 - float(no_ci["ci_low"].iloc[0])], [float(no_ci["ci_high"].iloc[0]) - x0]], fmt="none", ecolor="#5d6773", capsize=3, linewidth=1.0)
            if not yes_ci.empty:
                ax.errorbar(idx + width / 2, x1, yerr=[[x1 - float(yes_ci["ci_low"].iloc[0])], [float(yes_ci["ci_high"].iloc[0]) - x1]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_AXIS_LABELS[p] for p in policy_order])
    _style_axis(ax, ylabel="Mean Actual Profit")
    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.set_title("Urban-Context Ablation in Sparse High Occlusion")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "rendezvous_fig8_context_ablation.png")


def fig8_green_robustness() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_green_policy_summary.csv")
    dispatch = _load(RESULTS_DIR / "rendezvous_green_dispatch_policy_summary.csv")
    if summary is None or summary.empty:
        return
    focus = _canonical_slice(
        summary,
        domain="green",
        scenario_names=["primary", "sparse_high_occlusion"],
        area_slice="all",
        rider_density_pct=None,
        use_urban_context=True,
    )
    focus = focus[focus["policy"].isin(["corridor_only", "rendezvous_only", "rendezvous_observable"])].copy()
    if focus.empty:
        return

    ci_single = _load_ci(
        file_name="rendezvous_policy_bootstrap_ci.csv",
        metric="actual_profit",
        domain="green",
        scenario_names=["primary", "sparse_high_occlusion"],
        rider_density_pct=None,
        time_slice="all_day",
        use_urban_context=True,
    )
    ci_dispatch = _load_ci(
        file_name="rendezvous_dispatch_bootstrap_ci.csv",
        metric="profit_per_driver",
        domain="green",
        scenario_names=["sparse_high_occlusion"],
        rider_density_pct=10,
        time_slice="all_day",
        use_urban_context=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"wspace": 0.28})
    scenarios = ["primary", "sparse_high_occlusion"]
    policies = ["corridor_only", "rendezvous_only", "rendezvous_observable"]
    x = np.arange(len(scenarios))
    width = 0.22
    for i, policy in enumerate(policies):
        values = []
        for scenario in scenarios:
            sub = focus[(focus["scenario_name"] == scenario) & (focus["policy"] == policy)]
            values.append(float(sub["mean_actual_profit"].iloc[0]) if not sub.empty else np.nan)
        xpos = x + (i - 1) * width
        axes[0].bar(xpos, values, width=width, color=POLICY_COLORS[policy], edgecolor="#303030", linewidth=0.7, label=POLICY_LABELS[policy])
        for xi, scenario, value in zip(xpos, scenarios, values):
            ci_sub = ci_single[(ci_single["policy"] == policy) & (ci_single["scenario_name"] == scenario)]
            if ci_sub.empty or np.isnan(value):
                continue
            axes[0].errorbar(xi, value, yerr=[[value - float(ci_sub["ci_low"].iloc[0])], [float(ci_sub["ci_high"].iloc[0]) - value]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SCENARIO_LABELS.get(scenario, scenario.replace("_", "\n")) for scenario in scenarios])
    _style_axis(axes[0], ylabel="Mean Actual Profit")
    axes[0].set_title("Green Single-Driver Transfer")
    axes[0].axhline(0.0, color="#cccccc", linewidth=0.8)

    if dispatch is not None and not dispatch.empty:
        dispatch_focus = _canonical_slice(
            dispatch,
            domain="green",
            scenario_names=["sparse_high_occlusion"],
            area_slice="all",
            use_urban_context=True,
        )
        dispatch_focus = dispatch_focus[dispatch_focus["policy"].isin(policies)].copy()
        bars = []
        for policy in policies:
            sub = dispatch_focus[dispatch_focus["policy"] == policy]
            bars.append(float(sub["mean_profit_per_driver"].iloc[0]) if not sub.empty else np.nan)
        xpos = np.arange(len(policies))
        axes[1].bar(xpos, bars, color=[POLICY_COLORS[p] for p in policies], edgecolor="#303030", linewidth=0.7, width=0.58)
        for idx, policy in enumerate(policies):
            val = bars[idx]
            if np.isnan(val):
                continue
            ci_sub = ci_dispatch[ci_dispatch["policy"] == policy]
            if not ci_sub.empty:
                axes[1].errorbar(idx, val, yerr=[[val - float(ci_sub["ci_low"].iloc[0])], [float(ci_sub["ci_high"].iloc[0]) - val]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.0)
        axes[1].set_xticks(xpos)
        axes[1].set_xticklabels([POLICY_AXIS_LABELS[policy] for policy in policies])
        _style_axis(axes[1], ylabel="Mean Profit per Driver")
        axes[1].set_title("Green Dispatch Transfer")
        axes[1].axhline(0.0, color="#cccccc", linewidth=0.8)
    else:
        axes[1].axis("off")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.07, -0.12), ncol=1)
    fig.subplots_adjust(bottom=0.24)
    _save(fig, "rendezvous_fig8_green_robustness.png")


def fig9_time_slice_robustness() -> None:
    summary = _load(RESULTS_DIR / "rendezvous_policy_summary.csv")
    dispatch = _load(RESULTS_DIR / "rendezvous_dispatch_policy_summary.csv")
    ci_single = _load(RESULTS_DIR / "rendezvous_policy_bootstrap_ci.csv")
    ci_dispatch = _load(RESULTS_DIR / "rendezvous_dispatch_bootstrap_ci.csv")
    if summary is None or summary.empty or dispatch is None or dispatch.empty:
        return

    policies = ["corridor_only", "rendezvous_only", "rendezvous_observable"]
    single = _canonical_slice(
        summary,
        domain="yellow",
        scenario_names=["sparse_high_occlusion"],
        time_slice=None,
        area_slice="all",
        rider_density_pct=10,
        use_urban_context=True,
    )
    single = single[
        single["policy"].isin(policies)
        & single["time_slice"].isin(["all_day", "morning_peak", "evening_peak"])
    ].copy()
    dispatch_focus = _canonical_slice(
        dispatch,
        domain="yellow",
        scenario_names=["sparse_high_occlusion"],
        time_slice=None,
        area_slice="all",
        rider_density_pct=10,
        use_urban_context=True,
    )
    dispatch_focus = dispatch_focus[
        dispatch_focus["policy"].isin(policies)
        & dispatch_focus["time_slice"].isin(["all_day", "morning_peak"])
    ].copy()
    if single.empty or dispatch_focus.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=False, gridspec_kw={"wspace": 0.24})
    slice_order = ["all_day", "morning_peak", "evening_peak"]
    x = np.arange(len(slice_order))
    width = 0.22
    ci_single_focus = pd.DataFrame()
    if ci_single is not None and not ci_single.empty:
        ci_single_focus = _canonical_slice(
            ci_single,
            domain="yellow",
            scenario_names=["sparse_high_occlusion"],
            time_slice=None,
            area_slice="all",
            rider_density_pct=10,
            use_urban_context=True,
        )
        if "metric" in ci_single_focus.columns:
            ci_single_focus = ci_single_focus[ci_single_focus["metric"] == "actual_profit"].copy()
    for i, policy in enumerate(policies):
        sub = single[single["policy"] == policy].set_index("time_slice")
        values = [float(sub.loc[label, "mean_actual_profit"]) if label in sub.index else np.nan for label in slice_order]
        xpos = x + (i - 1) * width
        axes[0].bar(xpos, values, width=width, color=POLICY_COLORS[policy], edgecolor="#303030", linewidth=0.7, label=POLICY_LABELS[policy])
        if not ci_single_focus.empty:
            ci_sub = ci_single_focus[ci_single_focus["policy"] == policy].set_index("time_slice")
            for xi, label, val in zip(xpos, slice_order, values):
                if label not in ci_sub.index or np.isnan(val):
                    continue
                axes[0].errorbar(xi, val, yerr=[[val - float(ci_sub.loc[label, "ci_low"])], [float(ci_sub.loc[label, "ci_high"]) - val]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TIME_SLICE_LABELS[label] for label in slice_order])
    _style_axis(axes[0], ylabel="Mean Actual Profit")
    axes[0].set_title("Single-Driver Sparse High Occlusion")
    axes[0].axhline(0.0, color="#cccccc", linewidth=0.8)

    dispatch_slice_order = ["all_day", "morning_peak"]
    x2 = np.arange(len(dispatch_slice_order))
    ci_dispatch_focus = pd.DataFrame()
    if ci_dispatch is not None and not ci_dispatch.empty:
        ci_dispatch_focus = _canonical_slice(
            ci_dispatch,
            domain="yellow",
            scenario_names=["sparse_high_occlusion"],
            time_slice=None,
            area_slice="all",
            rider_density_pct=10,
            use_urban_context=True,
        )
        if "metric" in ci_dispatch_focus.columns:
            ci_dispatch_focus = ci_dispatch_focus[ci_dispatch_focus["metric"] == "profit_per_driver"].copy()
    for i, policy in enumerate(policies):
        sub = dispatch_focus[dispatch_focus["policy"] == policy].set_index("time_slice")
        values = [float(sub.loc[label, "mean_profit_per_driver"]) if label in sub.index else np.nan for label in dispatch_slice_order]
        xpos = x2 + (i - 1) * width
        axes[1].bar(xpos, values, width=width, color=POLICY_COLORS[policy], edgecolor="#303030", linewidth=0.7, label=POLICY_LABELS[policy])
        if not ci_dispatch_focus.empty:
            ci_sub = ci_dispatch_focus[ci_dispatch_focus["policy"] == policy].set_index("time_slice")
            for xi, label, val in zip(xpos, dispatch_slice_order, values):
                if label not in ci_sub.index or np.isnan(val):
                    continue
                axes[1].errorbar(xi, val, yerr=[[val - float(ci_sub.loc[label, "ci_low"])], [float(ci_sub.loc[label, "ci_high"]) - val]], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1.0)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([TIME_SLICE_LABELS[label] for label in dispatch_slice_order])
    _style_axis(axes[1], ylabel="Mean Profit per Driver")
    axes[1].set_title("Dispatch Sparse High Occlusion")
    axes[1].axhline(0.0, color="#cccccc", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(bottom=0.28, wspace=0.30)
    _save(fig, "rendezvous_fig9_time_slice_robustness.png")


def main() -> None:
    fig0_pipeline_overview()
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
