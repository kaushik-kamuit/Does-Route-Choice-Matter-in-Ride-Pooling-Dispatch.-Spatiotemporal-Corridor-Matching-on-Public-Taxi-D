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


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
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
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for policy, color in {
        "corridor_only": "#6c757d",
        "rendezvous_only": "#198754",
        "rendezvous_observable": "#0d6efd",
        "ml_meeting_point_comparator": "#dc3545",
    }.items():
        sub = df[df["policy"] == policy]
        if sub.empty:
            continue
        ax.plot(sub["rider_density_pct"], sub["mean_actual_profit"], marker="o", label=policy, color=color)
    ax.set_xlabel("Retained-sample density (%)")
    ax.set_ylabel("Mean actual profit")
    ax.set_title("Primary policy comparison")
    ax.legend(frameon=False)
    _save(fig, "rendezvous_fig2_primary.png")


def fig3_gap() -> None:
    df = _load(RESULTS_DIR / "rendezvous_nominal_realized_gap.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(df["policy"], df["mean_nominal_realized_gap"], color=["#6c757d", "#198754", "#0d6efd", "#dc3545"][: len(df)])
    ax.set_ylabel("Mean nominal-realized gap")
    ax.set_title("Nominal vs realized service gap")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "rendezvous_fig3_gap.png")


def fig4_dispatch() -> None:
    df = _load(RESULTS_DIR / "rendezvous_dispatch_policy_summary.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(df["policy"], df["mean_profit_per_driver"], color="#0d6efd")
    ax.set_ylabel("Mean profit per driver")
    ax.set_title("Dispatch validation")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "rendezvous_fig4_dispatch.png")


def fig5_ml_comparator() -> None:
    df = _load(RESULTS_DIR / "rendezvous_meeting_point_comparison.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(df["policy"], df["mean_successful_riders"], color=["#198754", "#dc3545"][: len(df)])
    ax.set_ylabel("Mean successful riders")
    ax.set_title("Deterministic vs ML meeting-point selection")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "rendezvous_fig5_ml_comparator.png")


def fig6_sensitivity() -> None:
    df = _load(RESULTS_DIR / "rendezvous_occlusion_sensitivity.csv")
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for policy in sorted(df["policy"].unique()):
        sub = df[df["policy"] == policy]
        ax.plot(sub["occlusion_lambda"], sub["mean_actual_profit"], marker="o", label=policy)
    ax.set_xlabel("Occlusion lambda")
    ax.set_ylabel("Mean actual profit")
    ax.set_title("Occlusion sensitivity")
    ax.legend(frameon=False)
    _save(fig, "rendezvous_fig6_sensitivity.png")


def main() -> None:
    fig1_concept()
    fig2_primary()
    fig3_gap()
    fig4_dispatch()
    fig5_ml_comparator()
    fig6_sensitivity()


if __name__ == "__main__":
    main()
