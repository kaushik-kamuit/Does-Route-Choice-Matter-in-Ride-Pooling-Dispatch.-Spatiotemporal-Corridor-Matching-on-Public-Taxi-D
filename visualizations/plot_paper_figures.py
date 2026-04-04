"""
Publication figures for the dispatch-first paper package.

This module renders all paper figures from already-summarized CSV outputs.
The plotting style is rebuilt around scientific-figure guidance from:

- IEEE Author Center graphics guidance
- PLOS "Ten Simple Rules for Better Figures"
- ACS Energy Letters guidance on scientific figure clarity

The design goals are:
- small multiples instead of crowded omnibus plots
- direct labels over oversized legends
- restrained annotation
- strong alignment and whitespace
- clear separation between mechanism figures and result figures
- locked user-provided source assets for the architecture and corridor map
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAPER_FIG_DIR = ROOT / "paper" / "figures"

REALISM_PRIMARY_PATH = RESULTS_DIR / "realism_primary_summary.csv"
STRATEGY_GAP_PATH = RESULTS_DIR / "strategy_gap_results.csv"
DISPATCH_DENSITY_CI_PATH = RESULTS_DIR / "dispatch_density_ci_summary.csv"
DOMAIN_TRANSFER_CI_PATH = RESULTS_DIR / "domain_transfer_ci_summary.csv"
FUNNEL_PATH = RESULTS_DIR / "matching_ball_funnel_summary.csv"
SENSITIVITY_GRID_PATH = RESULTS_DIR / "sensitivity_grid_summary.csv"
MODEL_FAMILY_PATH = RESULTS_DIR / "model_feature_family_summary.csv"
MODEL_CALIBRATION_PATH = RESULTS_DIR / "model_calibration_summary.csv"

ARCHITECTURE_SOURCE = PAPER_FIG_DIR / "paper_fig1_dispatch_architecture_source.png"
CORRIDOR_MAP_SOURCE = PAPER_FIG_DIR / "paper_fig2a_corridor_map.jpg"

POLICY_ORDER = ["coldstart", "best_heuristic", "warmup", "oracle"]
POLICY_LABELS = {
    "coldstart": "Cold-start",
    "best_heuristic": "Best heuristic",
    "warmup": "ML warm-up",
    "oracle": "Oracle",
}
POLICY_COLORS = {
    "coldstart": "#4C78A8",
    "best_heuristic": "#59A14F",
    "warmup": "#F28E2B",
    "oracle": "#B07AA1",
}
POLICY_MARKERS = {
    "coldstart": "o",
    "best_heuristic": "s",
    "warmup": "D",
    "oracle": "^",
}

FAMILY_LABELS = {
    "Spatial demand": "Spatial demand",
    "Geometry": "Geometry",
    "Landmark": "Landmark",
    "Temporal": "Temporal",
}
FAMILY_COLORS = {
    "Spatial demand": "#4C78A8",
    "Geometry": "#72B7B2",
    "Landmark": "#B279A2",
    "Temporal": "#F28E2B",
}
FUNNEL_LABELS = {
    "retrieved_candidates": "Retrieved corridor candidates",
    "available_exact_time_candidates": "Dispatch-available after exact-time filter",
    "feasible_after_detour_seat": "Detour / seat feasible",
    "matched_riders": "Matched riders",
}
FUNNEL_COLORS = ["#4C78A8", "#7AA6D1", "#F1B555", "#F58518"]


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.3,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "axes.grid": False,
        "lines.linewidth": 1.3,
        "patch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, filename: str, aliases: list[str] | None = None) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = [filename] + (aliases or [])
    for name in names:
        fig.savefig(PLOTS_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(PAPER_FIG_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
        pdf_name = Path(name).with_suffix(".pdf").name
        fig.savefig(PLOTS_DIR / pdf_name, bbox_inches="tight", facecolor="white")
        fig.savefig(PAPER_FIG_DIR / pdf_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] {filename}")


def _style_axis(ax: plt.Axes, *, grid_axis: str | None = "x") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E3E8EF", linewidth=0.8)
    ax.spines["left"].set_color("#64748B")
    ax.spines["bottom"].set_color("#64748B")
    ax.tick_params(colors="#334155", length=3, pad=3)
    ax.set_axisbelow(True)


def _panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=7)


def _style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D7DEE8")
    frame.set_linewidth(0.7)


def _policy_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=POLICY_MARKERS[p],
            linestyle="none",
            markerfacecolor=POLICY_COLORS[p],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.0,
            label=POLICY_LABELS[p],
        )
        for p in POLICY_ORDER
    ]


def _selected_dispatch_density() -> pd.DataFrame | None:
    df = _load_csv(DISPATCH_DENSITY_CI_PATH)
    if df is None or df.empty:
        return None
    df = df[df["domain"] == "yellow"].copy()
    blocks: list[pd.DataFrame] = []
    for density in [100, 25, 10]:
        sub = df[df["density_pct"] == density].copy()
        if sub.empty:
            continue
        keep = sub[sub["policy"].isin(["coldstart", "warmup", "oracle"])].copy()
        heur = sub[sub["selected_for_paper"] == True].copy()
        if not heur.empty:
            heur = heur.head(1).copy()
            heur["policy"] = "best_heuristic"
            keep = pd.concat([keep, heur], ignore_index=True)
        blocks.append(keep)
    if not blocks:
        return None
    out = pd.concat(blocks, ignore_index=True)
    out["policy"] = pd.Categorical(out["policy"], POLICY_ORDER, ordered=True)
    return out.sort_values(["density_pct", "policy"])


def _selected_primary_by_domain() -> pd.DataFrame | None:
    df = _load_csv(DOMAIN_TRANSFER_CI_PATH)
    if df is None or df.empty:
        return None
    blocks: list[pd.DataFrame] = []
    for domain in ["yellow", "green"]:
        sub = df[df["domain"] == domain].copy()
        if sub.empty:
            continue
        keep = sub[sub["policy"].isin(["coldstart", "warmup", "oracle"])].copy()
        heur = sub[sub["selected_for_paper"] == True].copy()
        if not heur.empty:
            heur = heur.head(1).copy()
            heur["policy"] = "best_heuristic"
            keep = pd.concat([keep, heur], ignore_index=True)
        blocks.append(keep)
    if not blocks:
        return None
    out = pd.concat(blocks, ignore_index=True)
    out["policy"] = pd.Categorical(out["policy"], POLICY_ORDER, ordered=True)
    return out.sort_values(["policy", "domain"])


def fig1_architecture() -> None:
    if not ARCHITECTURE_SOURCE.exists():
        print("  [Fig 1] Skip: architecture source image missing")
        return

    image = plt.imread(ARCHITECTURE_SOURCE)
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.imshow(image)
    ax.axis("off")
    _save(fig, "paper_fig1_dispatch_architecture_v2.png")


def fig2_matching_ball_mechanism() -> None:
    funnel = _load_csv(FUNNEL_PATH)
    if funnel is None or funnel.empty or not CORRIDOR_MAP_SOURCE.exists():
        print("  [Fig 2] Skip: funnel summary or corridor map missing")
        return

    stage_order = [
        "retrieved_candidates",
        "available_exact_time_candidates",
        "feasible_after_detour_seat",
        "matched_riders",
    ]
    funnel = funnel.set_index("stage").reindex(stage_order).reset_index()
    values = funnel["mean_per_launched_driver"].to_numpy()
    maximum = float(values.max())

    fig = plt.figure(figsize=(7.1, 3.35))
    gs = GridSpec(
        1,
        2,
        width_ratios=[1.58, 1.0],
        left=0.055,
        right=0.985,
        top=0.92,
        bottom=0.14,
        wspace=0.20,
        figure=fig,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.imshow(plt.imread(CORRIDOR_MAP_SOURCE))
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    _panel_title(ax_map, "(a) Route corridors define candidate space")
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#D5DCE5")
        spine.set_linewidth(0.8)

    ax_flow = fig.add_subplot(gs[0, 1])
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 4.35)
    ax_flow.axis("off")
    _panel_title(ax_flow, "(b) Filtering pipeline")

    y_positions = [3.45, 2.45, 1.45, 0.45]
    widths = 0.28 + 0.60 * (values / maximum)
    x_center = 0.53
    box_h = 0.62

    for idx, (_, row) in enumerate(funnel.iterrows()):
        width = float(widths[idx])
        x0 = x_center - width / 2
        y0 = y_positions[idx] - box_h / 2
        patch = FancyBboxPatch(
            (x0, y0),
            width,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=FUNNEL_COLORS[idx],
            edgecolor="white",
            linewidth=0.8,
        )
        ax_flow.add_patch(patch)

        label = FUNNEL_LABELS[row["stage"]]
        value = float(row["mean_per_launched_driver"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])

        ax_flow.text(
            x_center,
            y_positions[idx] + 0.08,
            label,
            ha="center",
            va="center",
            fontsize=7.1,
            color="#1F2937",
            fontweight="semibold",
        )
        ax_flow.text(
            x_center,
            y_positions[idx] - 0.13,
            f"{value:.2f} per launched driver",
            ha="center",
            va="center",
            fontsize=6.7,
            color="#334155",
        )
        ax_flow.text(
            x_center,
            y_positions[idx] - 0.28,
            f"95% interval [{ci_low:.2f}, {ci_high:.2f}]",
            ha="center",
            va="center",
            fontsize=6.1,
            color="#475569",
        )

        if idx < len(funnel) - 1:
            next_value = float(funnel.iloc[idx + 1]["mean_per_launched_driver"])
            retention = 100.0 * next_value / value if value else 0.0
            arrow = FancyArrowPatch(
                (x_center, y_positions[idx] - box_h / 2 - 0.05),
                (x_center, y_positions[idx + 1] + box_h / 2 + 0.05),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="#94A3B8",
            )
            ax_flow.add_patch(arrow)
            ax_flow.text(
                x_center + 0.22,
                (y_positions[idx] + y_positions[idx + 1]) / 2,
                f"{retention:.0f}% retained",
                ha="left",
                va="center",
                fontsize=6.2,
                color="#64748B",
            )

    _save(fig, "paper_fig2_matching_ball_mechanism.png")


def fig3_dispatch_density() -> None:
    df = _selected_dispatch_density()
    if df is None or df.empty:
        print("  [Fig 3] Skip: dispatch density summary missing")
        return

    densities = [100, 25, 10]
    density_titles = {
        100: "100% benchmark",
        25: "25% practical sparse",
        10: "10% stress test",
    }
    profit_min = float(df["profit_per_launched_driver_ci_low"].min()) - 0.25
    profit_max = float(df["profit_per_launched_driver_ci_high"].max()) + 0.20
    matched_min = max(0.0, float(df["mean_matched_riders_per_driver_ci_low"].min()) - 0.03)
    matched_max = float(df["mean_matched_riders_per_driver_ci_high"].max()) + 0.06

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.1, 4.35),
        sharey="row",
    )

    y = np.arange(len(POLICY_ORDER))
    for col, density in enumerate(densities):
        sub = df[df["density_pct"] == density].set_index("policy").reindex(POLICY_ORDER).reset_index()
        band = "#F8FBFF" if density != 10 else "#FFF6EF"
        for row in range(2):
            axes[row, col].set_facecolor(band)

        profit_ax = axes[0, col]
        for idx, (_, row) in enumerate(sub.iterrows()):
            mean = float(row["profit_per_launched_driver_mean"])
            low = float(row["profit_per_launched_driver_ci_low"])
            high = float(row["profit_per_launched_driver_ci_high"])
            profit_ax.errorbar(
                mean,
                idx,
                xerr=np.array([[mean - low], [high - mean]]),
                fmt=POLICY_MARKERS[row["policy"]],
                markersize=5.4,
                linestyle="none",
                color=POLICY_COLORS[row["policy"]],
                ecolor=POLICY_COLORS[row["policy"]],
                elinewidth=1.1,
                capsize=2.3,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
        profit_ax.axvline(0, color="#A1A1AA", linestyle="--", linewidth=0.9, zorder=1)
        _style_axis(profit_ax, grid_axis="x")
        profit_ax.set_xlim(profit_min, profit_max)
        profit_ax.set_yticks(y)
        if col == 0:
            profit_ax.set_yticklabels([POLICY_LABELS[p] for p in POLICY_ORDER])
        else:
            profit_ax.set_yticklabels([])
        profit_ax.invert_yaxis()
        _panel_title(profit_ax, f"(a{col + 1}) {density_titles[density]}")
        if col == 0:
            profit_ax.set_xlabel("Loss per launched driver ($)")
        else:
            profit_ax.set_xlabel("Loss / driver ($)")

        warm = float(sub[sub["policy"] == "warmup"]["profit_per_launched_driver_mean"].iloc[0])
        cold = float(sub[sub["policy"] == "coldstart"]["profit_per_launched_driver_mean"].iloc[0])
        profit_ax.text(
            0.97,
            0.08,
            f"ML vs cold-start: {warm - cold:+.2f}",
            transform=profit_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            color="#7C4A14",
        )

        match_ax = axes[1, col]
        for idx, (_, row) in enumerate(sub.iterrows()):
            mean = float(row["mean_matched_riders_per_driver_mean"])
            low = float(row["mean_matched_riders_per_driver_ci_low"])
            high = float(row["mean_matched_riders_per_driver_ci_high"])
            match_ax.errorbar(
                mean,
                idx,
                xerr=np.array([[mean - low], [high - mean]]),
                fmt=POLICY_MARKERS[row["policy"]],
                markersize=5.4,
                linestyle="none",
                color=POLICY_COLORS[row["policy"]],
                ecolor=POLICY_COLORS[row["policy"]],
                elinewidth=1.1,
                capsize=2.3,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
        _style_axis(match_ax, grid_axis="x")
        match_ax.set_xlim(matched_min, matched_max)
        match_ax.set_yticks(y)
        if col == 0:
            match_ax.set_yticklabels([POLICY_LABELS[p] for p in POLICY_ORDER])
        else:
            match_ax.set_yticklabels([])
        match_ax.invert_yaxis()
        if col == 0:
            match_ax.set_xlabel("Matched riders per launched driver")
        else:
            match_ax.set_xlabel("Matched riders / driver")
        if col == 0:
            _panel_title(match_ax, "(b1) Dispatch throughput")
        else:
            _panel_title(match_ax, f"(b{col + 1}) Throughput")

    fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.11, wspace=0.18, hspace=0.36)
    _save(fig, "paper_fig3_dispatch_density.png")


def fig4_cross_domain() -> None:
    df = _selected_primary_by_domain()
    if df is None or df.empty:
        print("  [Fig 4] Skip: domain transfer summary missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95), sharey=True)
    metrics = [
        ("profit_per_launched_driver", "Loss per launched driver ($)", "(a) Dispatch loss"),
        ("mean_matched_riders_per_driver", "Matched riders per launched driver", "(b) Dispatch throughput"),
    ]
    y = np.arange(len(POLICY_ORDER))

    for ax, (metric, xlabel, title) in zip(axes, metrics):
        for idx, policy in enumerate(POLICY_ORDER):
            sub = df[df["policy"] == policy].set_index("domain")
            if "yellow" not in sub.index or "green" not in sub.index:
                continue
            y_val = float(sub.loc["yellow", f"{metric}_mean"])
            g_val = float(sub.loc["green", f"{metric}_mean"])
            ax.plot([y_val, g_val], [idx, idx], color=POLICY_COLORS[policy], linewidth=1.6, alpha=0.65, zorder=2)

            for domain, marker in [("yellow", "o"), ("green", "D")]:
                row = sub.loc[domain]
                mean = float(row[f"{metric}_mean"])
                low = float(row[f"{metric}_ci_low"])
                high = float(row[f"{metric}_ci_high"])
                ax.errorbar(
                    mean,
                    idx,
                    xerr=np.array([[mean - low], [high - mean]]),
                    fmt=marker,
                    markersize=5.3 if domain == "yellow" else 5.0,
                    linestyle="none",
                    color=POLICY_COLORS[policy],
                    ecolor=POLICY_COLORS[policy],
                    elinewidth=1.0,
                    capsize=2.3,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    zorder=3,
                )

        _style_axis(ax, grid_axis="x")
        ax.set_yticks(y)
        ax.set_yticklabels([POLICY_LABELS[p] for p in POLICY_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        _panel_title(ax, title)

    legend = axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#64748B", markeredgecolor="white", markeredgewidth=0.7, markersize=6, label="Yellow"),
            Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="#64748B", markeredgecolor="white", markeredgewidth=0.7, markersize=5.7, label="Green"),
        ],
        loc="lower right",
        frameon=True,
        borderpad=0.3,
        handletextpad=0.4,
    )
    _style_legend(legend)

    fig.subplots_adjust(left=0.19, right=0.99, top=0.88, bottom=0.18, wspace=0.30)
    _save(fig, "paper_fig4_cross_domain.png")


def fig5_single_driver_mechanism() -> None:
    df = _load_csv(REALISM_PRIMARY_PATH)
    gap_df = _load_csv(STRATEGY_GAP_PATH)
    if df is None or gap_df is None or df.empty or gap_df.empty:
        print("  [Fig 5] Skip: single-driver summaries missing")
        return

    df = df[df["density_pct"].isin([25, 10])].copy().sort_values("density_pct", ascending=False)
    gap_df = gap_df[gap_df["comparison"] == "warmup_vs_heuristic"].copy().sort_values("density_pct", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), gridspec_kw={"width_ratios": [1.02, 1.06]})

    ax = axes[0]
    x = np.arange(len(df))
    heur_gain = (df["heuristic_profit"] - df["coldstart_profit"]).to_numpy()
    ml_gain = (df["warmup_profit"] - df["heuristic_profit"]).to_numpy()
    oracle_headroom = (df["oracle_profit"] - df["warmup_profit"]).to_numpy()

    ax.bar(x, heur_gain, width=0.55, color=POLICY_COLORS["best_heuristic"], edgecolor="white", label="Recovered by best heuristic", zorder=3)
    ax.bar(x, ml_gain, bottom=heur_gain, width=0.55, color=POLICY_COLORS["warmup"], edgecolor="white", label="Additional ML lift", zorder=4)
    ax.scatter(x, heur_gain + ml_gain + oracle_headroom, marker="D", s=28, color=POLICY_COLORS["oracle"], edgecolor="white", linewidth=0.7, zorder=5, label="Remaining oracle headroom")

    for idx, total in enumerate(heur_gain + ml_gain):
        ax.text(x[idx], total + 0.05, f"+${total:.2f}", ha="center", va="bottom", fontsize=6.5, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}% density" for v in df["density_pct"]])
    ax.set_ylabel("Improvement over cold-start ($/driver)")
    _panel_title(ax, "(a) Gain decomposition in the key sparse regimes")
    _style_axis(ax, grid_axis="y")
    legend = ax.legend(loc="upper left", frameon=True, borderpad=0.3, handletextpad=0.4)
    _style_legend(legend)

    ax = axes[1]
    y = np.arange(len(gap_df))
    means = gap_df["mean_diff"].to_numpy()
    low = means - gap_df["boot_low"].to_numpy()
    high = gap_df["boot_high"].to_numpy() - means
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        markersize=5.0,
        linestyle="none",
        color=POLICY_COLORS["warmup"],
        ecolor="#6B7280",
        elinewidth=1.1,
        capsize=2.6,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    ax.axvline(0, color="#A1A1AA", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(v)}%" for v in gap_df["density_pct"]])
    ax.invert_yaxis()
    ax.set_xlabel("Warm-up vs strongest heuristic ($/driver)")
    ax.set_ylabel("Retained density")
    _panel_title(ax, "(b) Isolated ML lift across densities")
    _style_axis(ax, grid_axis="x")

    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.17, wspace=0.28)
    _save(fig, "paper_fig5_single_driver_mechanism.png")


def fig6_model_support() -> None:
    family = _load_csv(MODEL_FAMILY_PATH)
    calib = _load_csv(MODEL_CALIBRATION_PATH)
    if family is None or calib is None or family.empty or calib.empty:
        print("  [Fig 6] Skip: model support summaries missing")
        return

    family = family.sort_values("share_pct", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95), gridspec_kw={"width_ratios": [1.02, 1.18]})

    ax = axes[0]
    family = family.sort_values("share_pct", ascending=True).copy()
    ypos = np.arange(len(family))
    ax.barh(
        ypos,
        family["share_pct"],
        color=[FAMILY_COLORS[g] for g in family["group"]],
        edgecolor="white",
        height=0.58,
        zorder=3,
    )
    for yi, (_, row) in zip(ypos, family.iterrows()):
        ax.text(
            float(row["share_pct"]) + 1.0,
            yi,
            f"{float(row['share_pct']):.1f}%",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#334155",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([FAMILY_LABELS.get(g, g) for g in family["group"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total feature importance (%)")
    _panel_title(ax, "(a) Importance concentration by feature family")
    _style_axis(ax, grid_axis="x")

    ax = axes[1]
    actual = calib["actual_mean"].to_numpy()
    pred = calib["pred_mean"].to_numpy()
    q25 = calib["pred_q25"].to_numpy()
    q75 = calib["pred_q75"].to_numpy()
    lo = min(float(np.min(actual)), float(np.min(pred)))
    hi = max(float(np.max(actual)), float(np.max(pred)))
    ax.fill_between(actual, q25, q75, color="#F6C58D", alpha=0.35, zorder=1, label="Prediction IQR")
    ax.plot([lo, hi], [lo, hi], color="#7C7C7C", linestyle="--", linewidth=1.0, zorder=2, label="Ideal calibration")
    ax.plot(actual, pred, color=POLICY_COLORS["warmup"], linewidth=1.5, zorder=3)
    ax.scatter(actual, pred, s=24, color=POLICY_COLORS["warmup"], edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xlabel("Mean actual profit by decile ($)")
    ax.set_ylabel("Mean predicted profit ($)")
    _panel_title(ax, "(b) Temporal-holdout calibration")
    _style_axis(ax, grid_axis="both")
    legend = ax.legend(loc="upper left", frameon=True, borderpad=0.3, handletextpad=0.4)
    _style_legend(legend)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.17, wspace=0.26)
    _save(fig, "paper_fig6_model_support.png")


def fig7_sensitivity() -> None:
    df = _load_csv(SENSITIVITY_GRID_PATH)
    if df is None or df.empty:
        print("  [Fig 7] Skip: sensitivity grid summary missing")
        return
    df = df[(df["domain"] == "yellow") & (df["density_pct"] == 10)].copy()
    if df.empty:
        print("  [Fig 7] Skip: no Yellow 10% sensitivity rows")
        return

    heat = (
        df.pivot_table(
            index="max_detour_min",
            columns="matching_window_min",
            values="warmup_minus_coldstart_mean",
            aggfunc="mean",
        )
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )

    row_means = heat.mean(axis=1)
    col_means = heat.mean(axis=0)

    fig = plt.figure(figsize=(5.35, 3.55))
    gs = GridSpec(
        2,
        2,
        width_ratios=[1.0, 0.34],
        height_ratios=[0.35, 1.0],
        left=0.12,
        right=0.97,
        top=0.90,
        bottom=0.15,
        wspace=0.08,
        hspace=0.10,
        figure=fig,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heat)

    x = np.arange(len(col_means))
    ax_top.plot(x, col_means.to_numpy(), color=POLICY_COLORS["warmup"], marker="o", markersize=4.5, linewidth=1.4)
    ax_top.fill_between(x, 0, col_means.to_numpy(), color="#F9D4A8", alpha=0.25)
    ax_top.set_ylabel("Avg gain")
    ax_top.set_xticks([])
    _panel_title(ax_top, "Sensitivity of warm-up gain")
    _style_axis(ax_top, grid_axis="y")

    matrix = heat.to_numpy()
    im = ax_heat.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=float(matrix.min()), vmax=float(matrix.max()))
    for i, detour in enumerate(heat.index):
        for j, window in enumerate(heat.columns):
            val = float(heat.loc[detour, window])
            ax_heat.text(j, i, f"+{val:.2f}", ha="center", va="center", fontsize=6.6, color="#2B2118", fontweight="semibold")
    if 4 in heat.index and 5 in heat.columns:
        i = list(heat.index).index(4)
        j = list(heat.columns).index(5)
        ax_heat.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1F2937", linewidth=1.5))
    ax_heat.set_xticks(np.arange(len(heat.columns)))
    ax_heat.set_xticklabels([f"{int(v)} min" for v in heat.columns])
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels([f"{int(v)} min" for v in heat.index])
    ax_heat.set_xlabel("Exact request window")
    ax_heat.set_ylabel("Detour bound")

    y = np.arange(len(row_means))
    ax_right.barh(y, row_means.to_numpy(), color="#E7C47A", edgecolor="white", height=0.58)
    ax_right.set_xlabel("Avg")
    ax_right.tick_params(axis="y", labelleft=False)
    _style_axis(ax_right, grid_axis="x")

    cbar = fig.colorbar(im, ax=[ax_heat, ax_right], fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("Gain ($/launched driver)", rotation=90)

    _save(fig, "paper_fig7_sensitivity.png")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating paper figures from summary CSVs...")
    fig1_architecture()
    fig2_matching_ball_mechanism()
    fig3_dispatch_density()
    fig4_cross_domain()
    fig5_single_driver_mechanism()
    fig6_model_support()
    fig7_sensitivity()
    print(f"\nPaper figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
