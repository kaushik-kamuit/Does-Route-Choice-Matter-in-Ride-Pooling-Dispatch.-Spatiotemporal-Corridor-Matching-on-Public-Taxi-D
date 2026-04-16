from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PRIMARY_STRATEGIES = ["coldstart", "random", "heuristic", "warmup", "oracle"]
HEURISTIC_VARIANTS = [
    "heuristic_count",
    "heuristic_fare_density",
    "heuristic_feasible_count",
    "heuristic_profit_proxy",
]
PRIMARY_DENSITIES = [100, 75, 50, 25, 10]
WINDOW_SWEEP = {
    "2": {100: "w2", 25: "w2_d25", 10: "w2_d10"},
    "5": {100: "", 25: "d25", 10: "d10"},
    "10": {100: "w10", 25: "w10_d25", 10: "w10_d10"},
}
DETOUR_SWEEP = {"2": "w5_det2_d10", "4": "d10", "6": "w5_det6_d10"}
H3_SWEEP = [
    ("baseline_d10", "d10", 9, 1, 10),
    ("h3r8_k1_d10", "h3r8_k1_d10", 8, 1, 10),
    ("h3r9_k0_d10", "h3r9_k0_d10", 9, 0, 10),
    ("h3r9_k2_d10", "h3r9_k2_d10", 9, 2, 10),
]
ECON_SWEEP = [
    ("baseline", "d10", 0.50, 0.67, 10),
    ("econ_ps40_d10", "econ_ps40_d10", 0.40, 0.67, 10),
    ("econ_ps60_d10", "econ_ps60_d10", 0.60, 0.67, 10),
    ("econ_c50_d10", "econ_c50_d10", 0.50, 0.50, 10),
    ("econ_c85_d10", "econ_c85_d10", 0.50, 0.85, 10),
]


@dataclass
class ScenarioStats:
    tag: str
    density_pct: int
    strategy: str
    mean_profit: float
    mean_matched_riders: float
    mean_revenue: float
    mean_cost: float
    match_rate: float
    mean_compute_time_s: float
    median_compute_time_s: float
    p95_compute_time_s: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def load_config(tag: str) -> dict[str, object]:
    path = RESULTS / f"experiment_config{suffix(tag)}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_outcomes(tag: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for strategy in [*PRIMARY_STRATEGIES, *HEURISTIC_VARIANTS]:
        path = RESULTS / f"{strategy}_outcomes{suffix(tag)}.csv"
        if path.exists():
            data[strategy] = pd.read_csv(path)
    return data


def aggregate_strategy(df: pd.DataFrame) -> tuple[float, float, float, float, float, float, float, float]:
    by_driver = df.groupby("driver_id", as_index=False).agg(
        profit=("profit", "mean"),
        matched_riders=("matched_riders", "mean"),
        total_revenue=("total_revenue", "mean"),
        driving_cost=("driving_cost", "mean"),
        compute_time_s=("compute_time_s", "mean"),
    )
    match_rate = float((df["matched_riders"] > 0).mean() * 100.0)
    return (
        float(by_driver["profit"].mean()),
        float(by_driver["matched_riders"].mean()),
        float(by_driver["total_revenue"].mean()),
        float(by_driver["driving_cost"].mean()),
        match_rate,
        float(by_driver["compute_time_s"].mean()),
        float(by_driver["compute_time_s"].median()),
        float(by_driver["compute_time_s"].quantile(0.95)),
    )


def infer_density_pct(tag: str) -> int:
    if not tag:
        return 100
    for token in tag.split("_"):
        if token.startswith("d") and token[1:].isdigit():
            return int(token[1:])
    if tag in {"w2", "w10"}:
        return 100
    return 10


def collect(tag: str) -> tuple[dict[str, ScenarioStats], dict[str, object]]:
    density_pct = infer_density_pct(tag)
    rows: dict[str, ScenarioStats] = {}
    data = load_outcomes(tag)
    for strategy, df in data.items():
        (
            mean_profit,
            mean_match,
            mean_rev,
            mean_cost,
            match_rate,
            mean_compute,
            median_compute,
            p95_compute,
        ) = aggregate_strategy(df)
        rows[strategy] = ScenarioStats(
            tag=tag or "primary_full",
            density_pct=density_pct,
            strategy=strategy,
            mean_profit=mean_profit,
            mean_matched_riders=mean_match,
            mean_revenue=mean_rev,
            mean_cost=mean_cost,
            match_rate=match_rate,
            mean_compute_time_s=mean_compute,
            median_compute_time_s=median_compute,
            p95_compute_time_s=p95_compute,
        )
    return rows, load_config(tag)


def relative_improvement(delta: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    denom = abs(baseline) if baseline < 0 else baseline
    return (delta / denom) * 100.0


def main() -> None:
    primary_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    strong_baseline_rows: list[dict[str, object]] = []

    for density in PRIMARY_DENSITIES:
        tag = "" if density == 100 else f"d{density}"
        stats, config = collect(tag)
        cs = stats["coldstart"]
        wu = stats["warmup"]
        heu = stats["heuristic"]
        ora = stats["oracle"]

        primary_rows.append({
            "density_pct": density,
            "rider_pool_semantics": "retained_25pct_sample",
            "matching_window_min": int(config.get("max_request_offset_min", 5) or 5),
            "index_bin_minutes": int(config.get("index_bin_minutes", 15)),
            "candidate_window_bins": int(config.get("candidate_window_bins", 1)),
            "max_detour_min": float(config.get("max_detour_min", 4)),
            "heuristic_selected_strategy": config.get("heuristic_selected_strategy", "heuristic"),
            "coldstart_profit": cs.mean_profit,
            "warmup_profit": wu.mean_profit,
            "heuristic_profit": heu.mean_profit,
            "oracle_profit": ora.mean_profit,
            "warmup_delta": wu.mean_profit - cs.mean_profit,
            "warmup_gain_pct": relative_improvement(wu.mean_profit - cs.mean_profit, cs.mean_profit),
            "warmup_vs_heuristic": wu.mean_profit - heu.mean_profit,
        })

        for strategy in PRIMARY_STRATEGIES:
            entry = stats[strategy]
            runtime_rows.append({
                "scenario_family": "primary_density",
                "tag": tag or "primary_full",
                "density_pct": density,
                "strategy": strategy,
                "mean_compute_time_s": entry.mean_compute_time_s,
                "median_compute_time_s": entry.median_compute_time_s,
                "p95_compute_time_s": entry.p95_compute_time_s,
            })

        for heuristic_strategy in HEURISTIC_VARIANTS:
            if heuristic_strategy not in stats:
                continue
            entry = stats[heuristic_strategy]
            strong_baseline_rows.append({
                "density_pct": density,
                "heuristic_strategy": heuristic_strategy,
                "selected_for_paper": heuristic_strategy == config.get("heuristic_selected_strategy"),
                "mean_profit": entry.mean_profit,
                "vs_coldstart": entry.mean_profit - cs.mean_profit,
                "vs_warmup": entry.mean_profit - wu.mean_profit,
            })

    write_csv(RESULTS / "realism_primary_summary.csv", primary_rows)
    write_csv(RESULTS / "runtime_profile.csv", runtime_rows)
    write_csv(RESULTS / "strong_baseline_comparison.csv", strong_baseline_rows)

    window_rows: list[dict[str, object]] = []
    for window_min, mapping in WINDOW_SWEEP.items():
        for density_pct, tag in mapping.items():
            stats, config = collect(tag)
            cs = stats["coldstart"]
            wu = stats["warmup"]
            heu = stats["heuristic"]
            ora = stats["oracle"]
            window_rows.append({
                "matching_window_min": int(window_min),
                "density_pct": density_pct,
                "tag": tag or "primary_full",
                "heuristic_selected_strategy": config.get("heuristic_selected_strategy", "heuristic"),
                "coldstart_profit": cs.mean_profit,
                "warmup_profit": wu.mean_profit,
                "heuristic_profit": heu.mean_profit,
                "oracle_profit": ora.mean_profit,
                "warmup_delta": wu.mean_profit - cs.mean_profit,
                "warmup_vs_heuristic": wu.mean_profit - heu.mean_profit,
            })
    write_csv(RESULTS / "realism_window_sensitivity.csv", window_rows)
    write_csv(RESULTS / "window_sensitivity.csv", window_rows)

    detour_rows: list[dict[str, object]] = []
    for detour_min, tag in DETOUR_SWEEP.items():
        stats, config = collect(tag)
        cs = stats["coldstart"]
        wu = stats["warmup"]
        heu = stats["heuristic"]
        ora = stats["oracle"]
        detour_rows.append({
            "matching_window_min": int(config.get("max_request_offset_min", 5) or 5),
            "density_pct": 10,
            "max_detour_min": int(float(detour_min)),
            "tag": tag,
            "heuristic_selected_strategy": config.get("heuristic_selected_strategy", "heuristic"),
            "coldstart_profit": cs.mean_profit,
            "warmup_profit": wu.mean_profit,
            "heuristic_profit": heu.mean_profit,
            "oracle_profit": ora.mean_profit,
            "warmup_delta": wu.mean_profit - cs.mean_profit,
            "warmup_vs_heuristic": wu.mean_profit - heu.mean_profit,
        })
    write_csv(RESULTS / "realism_detour_sensitivity.csv", detour_rows)
    write_csv(RESULTS / "detour_sensitivity.csv", detour_rows)

    h3_rows: list[dict[str, object]] = []
    for _, tag, resolution, k_ring, density_pct in H3_SWEEP:
        stats, config = collect(tag)
        cs = stats["coldstart"]
        wu = stats["warmup"]
        heu = stats["heuristic"]
        ora = stats["oracle"]
        h3_rows.append({
            "tag": tag or "primary_full",
            "density_pct": density_pct,
            "h3_resolution": int(config.get("h3_resolution", resolution)),
            "corridor_k_ring": int(config.get("corridor_k_ring", k_ring)),
            "matching_window_min": int(config.get("max_request_offset_min", 5) or 5),
            "heuristic_selected_strategy": config.get("heuristic_selected_strategy", "heuristic"),
            "coldstart_profit": cs.mean_profit,
            "heuristic_profit": heu.mean_profit,
            "warmup_profit": wu.mean_profit,
            "oracle_profit": ora.mean_profit,
            "warmup_delta": wu.mean_profit - cs.mean_profit,
            "warmup_vs_heuristic": wu.mean_profit - heu.mean_profit,
        })
    write_csv(RESULTS / "h3_corridor_sensitivity.csv", h3_rows)

    econ_rows: list[dict[str, object]] = []
    for _, tag, platform_share, cost_per_mile, density_pct in ECON_SWEEP:
        stats, config = collect(tag)
        cs = stats["coldstart"]
        wu = stats["warmup"]
        heu = stats["heuristic"]
        ora = stats["oracle"]
        econ_rows.append({
            "tag": tag,
            "density_pct": density_pct,
            "platform_share": float(config.get("platform_share", platform_share)),
            "cost_per_mile": float(config.get("cost_per_mile", cost_per_mile)),
            "matching_window_min": int(config.get("max_request_offset_min", 5) or 5),
            "heuristic_selected_strategy": config.get("heuristic_selected_strategy", "heuristic"),
            "coldstart_profit": cs.mean_profit,
            "heuristic_profit": heu.mean_profit,
            "warmup_profit": wu.mean_profit,
            "oracle_profit": ora.mean_profit,
            "warmup_delta": wu.mean_profit - cs.mean_profit,
            "warmup_vs_heuristic": wu.mean_profit - heu.mean_profit,
        })
    write_csv(RESULTS / "economics_sensitivity.csv", econ_rows)

    headline = next(row for row in primary_rows if row["density_pct"] == 10)
    write_csv(
        RESULTS / "paper_primary_summary.csv",
        [
            {
                "headline_scenario": "5-minute exact window, 10% retained-sample density",
                "coldstart_profit": headline["coldstart_profit"],
                "heuristic_profit": headline["heuristic_profit"],
                "heuristic_selected_strategy": headline["heuristic_selected_strategy"],
                "warmup_profit": headline["warmup_profit"],
                "oracle_profit": headline["oracle_profit"],
                "warmup_delta": headline["warmup_delta"],
                "warmup_gain_pct": headline["warmup_gain_pct"],
                "warmup_vs_heuristic": headline["warmup_vs_heuristic"],
            }
        ],
    )

    scenario_rows = [
        {"parameter": "retained_rider_sample", "value": "25%", "note": "Applied in preprocess.py before density subsampling"},
        {"parameter": "headline_matching_window", "value": "5 min", "note": "Exact request offset after bin retrieval"},
        {"parameter": "index_bin_minutes", "value": "15", "note": "Used only for coarse RiderIndex lookup"},
        {"parameter": "candidate_window_bins", "value": "1", "note": "Adjacent bins searched to avoid boundary misses"},
        {"parameter": "headline_max_detour_min", "value": "4", "note": "Primary realism scenario"},
        {"parameter": "headline_h3_resolution", "value": "9", "note": "Primary route-cell resolution"},
        {"parameter": "headline_corridor_k_ring", "value": "1", "note": "Primary corridor width"},
        {"parameter": "platform_share", "value": "0.50", "note": "Scenario profit assumption"},
        {"parameter": "cost_per_mile", "value": "$0.67", "note": "Scenario profit assumption"},
        {"parameter": "urban_speed_proxy", "value": "40 km/h", "note": "Used for detour-minute conversion"},
        {"parameter": "seats", "value": "3", "note": "Driver seat capacity assumption"},
        {"parameter": "primary_heuristic", "value": headline["heuristic_selected_strategy"], "note": "Strongest non-ML heuristic in the primary scenario"},
    ]
    write_csv(RESULTS / "scenario_assumptions.csv", scenario_rows)

    temporal_path = RESULTS / "temporal_generalization.csv"
    temporal_line = "Temporal holdout: not generated"
    if temporal_path.exists():
        temporal_df = pd.read_csv(temporal_path)
        if not temporal_df.empty:
            row = temporal_df.iloc[0]
            temporal_line = (
                f"Temporal validation ({row['train_label']} -> {row['val_label']}): "
                f"R^2={float(row['r2']):.3f}, RMSE=${float(row['rmse']):.2f}"
            )

    summary = "\n".join([
        "=== Realism-First Q2 Upgrade Summary ===",
        "Headline scenario: 5-minute exact request window, 10% retained-sample density",
        f"Cold-start profit: ${headline['coldstart_profit']:.2f}",
        f"Best heuristic ({headline['heuristic_selected_strategy']}): ${headline['heuristic_profit']:.2f}",
        f"ML warm-up profit: ${headline['warmup_profit']:.2f}",
        f"Oracle profit:     ${headline['oracle_profit']:.2f}",
        f"Warm-up delta:     +${headline['warmup_delta']:.2f}",
        f"Warm-up gain:      {headline['warmup_gain_pct']:.1f}%",
        f"Warm-up vs heuristic: +${headline['warmup_vs_heuristic']:.2f}",
        temporal_line,
        "",
        "Benchmark semantics: 100% means the full retained 25% rider sample, not full city demand.",
    ])
    (RESULTS / "realism_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
