"""
Validate the publication-facing realism-first artifact package.

The checks are intentionally lightweight and dependency-free:
  - realism-first result tables exist
  - required paper figures exist
  - scenario metadata reflects the headline 5-minute setup
  - the manuscript discloses retained-sample semantics and scenario-profit assumptions
  - key manuscript numbers match the current CSV outputs
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PAPER = ROOT / "paper"


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.exists():
        fail(f"Missing required file: {path}")


def load_csv(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    fail(f"Could not find row where {key}={value}")
    raise AssertionError("unreachable")


def fmt_money(value: str | float) -> str:
    return f"{float(value):.2f}"


def fmt_pct(value: str | float, decimals: int = 1) -> str:
    return f"{float(value):.{decimals}f}"


def validate_realism_tables() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    primary_rows = load_csv(RESULTS / "realism_primary_summary.csv")
    window_rows = load_csv(RESULTS / "window_sensitivity.csv")
    detour_rows = load_csv(RESULTS / "detour_sensitivity.csv")
    strong_baseline_rows = load_csv(RESULTS / "strong_baseline_comparison.csv")
    load_csv(RESULTS / "h3_corridor_sensitivity.csv")
    economics_rows = load_csv(RESULTS / "economics_sensitivity.csv")
    load_csv(RESULTS / "runtime_profile.csv")
    load_csv(RESULTS / "paper_primary_summary.csv")
    scenario_rows = load_csv(RESULTS / "scenario_assumptions.csv")
    load_csv(RESULTS / "route_gain_decomposition_summary.csv")
    load_csv(RESULTS / "ml_gap_comparison_summary.csv")
    require_file(RESULTS / "realism_summary.txt")

    observed_primary = {int(row["density_pct"]) for row in primary_rows}
    expected_primary = {10, 25, 50, 75, 100}
    if observed_primary != expected_primary:
        fail(
            f"realism_primary_summary.csv has densities {sorted(observed_primary)}; "
            f"expected {sorted(expected_primary)}"
        )

    if any(row["matching_window_min"] != "5" for row in primary_rows):
        fail("Primary realism summary must use a 5-minute exact request window")
    if any(row["rider_pool_semantics"] != "retained_25pct_sample" for row in primary_rows):
        fail("Primary realism summary must disclose retained-sample rider semantics")

    full_row = find_row(primary_rows, "density_pct", "100")
    sparse_row = find_row(primary_rows, "density_pct", "10")
    if float(sparse_row["warmup_delta"]) <= 0:
        fail("Headline sparse scenario should have a positive warm-up delta")
    if float(sparse_row["coldstart_profit"]) >= float(full_row["coldstart_profit"]):
        fail("10% sparse scenario should be economically harsher than the full retained-sample benchmark")
    if float(sparse_row["warmup_profit"]) >= float(full_row["warmup_profit"]):
        fail("10% sparse scenario should have lower absolute profit than the full retained-sample benchmark")

    observed_window = {
        (int(row["matching_window_min"]), int(row["density_pct"]))
        for row in window_rows
    }
    expected_window = {
        (2, 100), (2, 25), (2, 10),
        (5, 100), (5, 25), (5, 10),
        (10, 100), (10, 25), (10, 10),
    }
    if observed_window != expected_window:
        fail(
            f"Window sensitivity grid mismatch: {sorted(observed_window)} vs expected {sorted(expected_window)}"
        )

    observed_detour = {int(float(row["max_detour_min"])) for row in detour_rows}
    if observed_detour != {2, 4, 6}:
        fail(f"Detour sensitivity should include 2, 4, 6 minute bounds; found {sorted(observed_detour)}")

    for density in expected_primary:
        density_rows = [row for row in strong_baseline_rows if int(row["density_pct"]) == density]
        if not density_rows:
            fail(f"Missing strong-baseline comparison rows for density {density}")
        selected = [row for row in density_rows if row["selected_for_paper"] == "True"]
        if len(selected) != 1:
            fail(f"Density {density} should have exactly one selected strongest heuristic")
        best = max(density_rows, key=lambda row: float(row["mean_profit"]))
        if selected[0]["heuristic_strategy"] != best["heuristic_strategy"]:
            fail(
                f"Selected heuristic at {density}% should be {best['heuristic_strategy']} "
                f"but found {selected[0]['heuristic_strategy']}"
            )

    for density in (100, 25, 10):
        density_rows = sorted(
            (row for row in window_rows if int(row["density_pct"]) == density),
            key=lambda row: int(row["matching_window_min"]),
        )
        if [int(row["matching_window_min"]) for row in density_rows] != [2, 5, 10]:
            fail(f"Window sensitivity rows for density {density} should be 2, 5, 10 minutes")
        for metric in ("coldstart_profit", "heuristic_profit", "warmup_profit", "oracle_profit"):
            values = [float(row[metric]) for row in density_rows]
            if not (values[0] <= values[1] <= values[2]):
                fail(
                    f"{metric} should improve monotonically as the exact request window loosens "
                    f"for density {density}; found {values}"
                )

    econ_by_tag = {row["tag"]: row for row in economics_rows}
    for tag in ("d10", "econ_ps40_d10", "econ_ps60_d10", "econ_c50_d10", "econ_c85_d10"):
        if tag not in econ_by_tag:
            fail(f"Missing economics sensitivity row: {tag}")
    if not (
        float(econ_by_tag["econ_ps40_d10"]["warmup_profit"])
        <= float(econ_by_tag["d10"]["warmup_profit"])
        <= float(econ_by_tag["econ_ps60_d10"]["warmup_profit"])
    ):
        fail("Warm-up profit should improve as platform share increases in economics sensitivity")
    if not (
        float(econ_by_tag["econ_c85_d10"]["warmup_profit"])
        <= float(econ_by_tag["d10"]["warmup_profit"])
        <= float(econ_by_tag["econ_c50_d10"]["warmup_profit"])
    ):
        fail("Warm-up profit should worsen as cost per mile increases in economics sensitivity")

    params = {row["parameter"]: row["value"] for row in scenario_rows}
    required_params = {
        "retained_rider_sample": "25%",
        "headline_matching_window": "5 min",
        "index_bin_minutes": "15",
        "candidate_window_bins": "1",
        "headline_max_detour_min": "4",
        "platform_share": "0.50",
        "cost_per_mile": "$0.67",
        "urban_speed_proxy": "40 km/h",
        "seats": "3",
    }
    for key, expected_value in required_params.items():
        if params.get(key) != expected_value:
            fail(f"Scenario assumption {key} expected {expected_value!r} but found {params.get(key)!r}")

    print("Realism-first summaries verified.")
    return full_row, sparse_row, params


def validate_model_outputs() -> tuple[dict[str, str], dict[str, str], dict[str, str] | None]:
    temporal_rows = load_csv(RESULTS / "temporal_generalization.csv")
    model_rows = load_csv(RESULTS / "model_comparison.csv")
    ablation_rows = load_csv(RESULTS / "ablation_results.csv")
    domain_temporal_rows = load_csv(RESULTS / "domain_temporal_generalization.csv") if (RESULTS / "domain_temporal_generalization.csv").exists() else []

    temporal = temporal_rows[0]
    if temporal["train_label"] != "Jan-Feb 2015" or temporal["val_label"] != "Mar 2015":
        fail("Temporal holdout should use Jan-Feb 2015 for training and Mar 2015 for validation")

    tuned = find_row(model_rows, "model", "LightGBM (tuned)")
    ridge = find_row(model_rows, "model", "Ridge (linear)")
    baseline = find_row(model_rows, "model", "LightGBM (baseline)")
    mlp = find_row(model_rows, "model", "MLP (Neural Net)")
    if float(tuned["r2"]) < float(baseline["r2"]):
        fail("Tuned LightGBM should not underperform the baseline LightGBM in R^2")
    if float(tuned["r2"]) < float(ridge["r2"]):
        fail("Tuned LightGBM should not underperform the Ridge baseline in R^2")
    if float(mlp["r2"]) <= float(ridge["r2"]):
        fail("MLP should provide a nonlinear fit stronger than Ridge on the temporal holdout")

    for experiment in ("All features", "All minus Temporal", "All minus Spatial Demand"):
        find_row(ablation_rows, "experiment", experiment)

    print(
        "Model outputs verified: "
        f"R2={float(tuned['r2']):.4f} RMSE=${float(tuned['rmse']):.2f} Rank-1={float(tuned['rank_acc'])*100:.2f}%"
    )
    green_temporal = next((row for row in domain_temporal_rows if row.get("domain") == "green"), None)
    return tuned, temporal, green_temporal


def validate_strategy_gaps() -> dict[str, str]:
    rows = load_csv(RESULTS / "strategy_gap_results.csv")
    route_rows = load_csv(RESULTS / "strategy_gap_route_breakdown.csv")
    require_file(RESULTS / "strategy_gap_summary.txt")

    required = {
        (density, comparison)
        for density in ("100", "75", "50", "25", "10")
        for comparison in ("warmup_vs_heuristic", "warmup_vs_coldstart", "oracle_vs_warmup")
    }
    observed = {(row["density_pct"], row["comparison"]) for row in rows}
    missing = required - observed
    if missing:
        fail(f"Missing paired-gap rows: {sorted(missing)}")

    ten_gap = next(
        row for row in rows
        if row["comparison"] == "warmup_vs_heuristic" and row["density_pct"] == "10"
    )
    if "long" not in {row["route_length_category"] for row in route_rows}:
        fail("Route breakdown must include the long-route category")

    print(
        "Strategy gaps verified: "
        f"10% warmup-vs-heuristic gap=${float(ten_gap['mean_diff']):.2f} "
        f"CI=[{float(ten_gap['boot_low']):.2f}, {float(ten_gap['boot_high']):.2f}]"
    )
    return ten_gap


def validate_paper_package(
    full_row: dict[str, str],
    sparse_row: dict[str, str],
    scenario_params: dict[str, str],
    tuned_row: dict[str, str],
    temporal_row: dict[str, str],
    green_temporal_row: dict[str, str] | None,
    heuristic_gap_row: dict[str, str],
    dispatch_rows: dict[str, dict[str, str]] | None,
) -> None:
    for name in (
        "paper_fig1_dispatch_architecture_v2.png",
        "paper_fig2_matching_ball_mechanism.png",
        "paper_fig3_dispatch_density.png",
        # paper_fig4_cross_domain.png removed — data covered by Table tab:domain_transfer
        "paper_fig5_single_driver_mechanism.png",
        "paper_fig6_model_support.png",
        "paper_fig7_sensitivity.png",
    ):
        require_file(PLOTS / name)
        require_file(PAPER / "figures" / name)

    manuscript = PAPER.joinpath("ieee_submission.tex").read_text(encoding="utf-8")
    references = PAPER.joinpath("references.bib").read_text(encoding="utf-8")

    required_snippets = [
        "rolling-horizon",
        "dispatch simulator",
        "retained-sample",
        "25\\%",
        "5-minute exact request window",
        "15-minute",
        "January--February 2015",
        "March 2015",
        "April 2015",
        "within the route set",
        "scenario profit",
        "not calibrated platform margins",
        "strongest non-ML heuristic",
        "feasible-rider count",
        "request window",
        "detour",
        "platform share",
        "cost per mile",
        "60-second",
        "NYC Green",
        "single-driver",
        "secondary evidence",
        "paper_fig1_dispatch_architecture_v2.png",
        "paper_fig2_matching_ball_mechanism.png",
        "paper_fig3_dispatch_density.png",
        # paper_fig4_cross_domain.png removed
        "paper_fig5_single_driver_mechanism.png",
        "paper_fig6_model_support.png",
        "paper_fig7_sensitivity.png",
        "fig:dispatch_architecture",
        "fig:matching_ball",
        "fig:dispatch_density",
        # fig:cross_domain removed with paper_fig4
        "fig:single_driver_density",
        "fig:model_support",
        "fig:sensitivity",
        "matching-ball",
        "Disk}_1(h)",
        "retrieved corridor candidates",
        "dispatch-available",
    ]
    for snippet in required_snippets:
        if snippet not in manuscript:
            fail(f"Manuscript is missing expected realism-first snippet: {snippet}")

    value_snippets = [
        f"\\${fmt_money(abs(float(sparse_row['coldstart_profit'])))}",
        f"\\${fmt_money(abs(float(sparse_row['warmup_profit'])))}",
        f"\\${fmt_money(abs(float(sparse_row['heuristic_profit'])))}",
        f"\\${fmt_money(abs(float(sparse_row['oracle_profit'])))}",
        fmt_money(heuristic_gap_row["mean_diff"]),
        fmt_money(heuristic_gap_row["boot_low"]),
        fmt_money(heuristic_gap_row["boot_high"]),
        f"{float(temporal_row['r2']):.3f}",
        fmt_money(temporal_row["rmse"]),
        f"{float(tuned_row['r2']):.3f}",
        fmt_money(tuned_row["rmse"]),
    ]
    if green_temporal_row is not None:
        value_snippets.extend([
            f"{float(green_temporal_row['r2']):.3f}",
            fmt_money(green_temporal_row["rmse"]),
        ])
    if dispatch_rows is not None:
        for key in ("yellow_coldstart", "yellow_heuristic", "yellow_warmup", "yellow_oracle", "green_coldstart", "green_heuristic", "green_warmup", "green_oracle"):
            row = dispatch_rows[key]
            value_snippets.append(fmt_money(abs(float(row["profit_per_launched_driver_mean"]))))
            value_snippets.append(fmt_money(row["mean_wait_min_mean"]))
        value_snippets.extend([
            f"{float(dispatch_rows['yellow_warmup']['mean_eval_time_s_mean']) * 1000:.1f}",
            f"{float(dispatch_rows['yellow_warmup']['mean_batch_runtime_s_mean']) * 1000:.1f}",
        ])
    for snippet in value_snippets:
        if snippet not in manuscript:
            fail(f"Manuscript is missing a current numeric anchor: {snippet}")

    for key in (
        "alonso2017",
        "agatz2012",
        "santi2014",
        "tachet2017",
        "chen2021",
        "lin2018fleet",
        "xu2018dispatch",
        "tang2019deepvalue",
        "li2019meanfield",
        "zhou2019ovdm",
        "jin2019coride",
        "xu2020reposition",
        "cheng2019queueing",
        "tang2021value",
        "suhr2019fair",
        "shi2021fair",
        "raman2021fair",
        "stumpe2024",
        "zhou2026",
        "ke2017",
        "azureyellow",
        "azuregreen",
        "osrmapi",
        "h3docs",
    ):
        if (
            f"@article{{{key}" not in references
            and f"@misc{{{key}" not in references
            and f"@inproceedings{{{key}" not in references
            and f"@incollection{{{key}" not in references
        ):
            fail(f"Bibliography key missing from references.bib: {key}")

    print(
        "Paper package verified: "
        f"dispatch headline is ${float(dispatch_rows['yellow_coldstart']['profit_per_launched_driver_mean']):.2f} -> "
        f"${float(dispatch_rows['yellow_warmup']['profit_per_launched_driver_mean']):.2f}"
    )


def validate_dispatch_outputs_if_present() -> dict[str, dict[str, str]] | None:
    primary_path = RESULTS / "dispatch_primary_ci_summary.csv"
    density_path = RESULTS / "dispatch_density_ci_summary.csv"
    domain_path = RESULTS / "domain_transfer_ci_summary.csv"
    funnel_path = RESULTS / "matching_ball_funnel_summary.csv"
    sensitivity_path = RESULTS / "sensitivity_grid_summary.csv"
    if not primary_path.exists():
        print("Dispatch outputs not present; skipping dispatch validation.")
        return None

    yellow_rows = [row for row in load_csv(primary_path) if row["domain"] == "yellow"]
    observed_policies = {row["policy"] for row in yellow_rows}
    required_policies = {"coldstart", "warmup", "oracle"}
    if not required_policies.issubset(observed_policies):
        fail(
            f"dispatch_primary_ci_summary.csv must contain at least {sorted(required_policies)} for Yellow; "
            f"found {sorted(observed_policies)}"
        )

    heuristic_rows = [
        row for row in yellow_rows
        if row["policy"].startswith("heuristic_")
    ]
    if not heuristic_rows:
        fail("dispatch_primary_ci_summary.csv must contain at least one Yellow heuristic policy row")
    selected = [row for row in yellow_rows if row.get("selected_for_paper") == "True"]
    if len(selected) != 1:
        fail("dispatch_primary_ci_summary.csv should mark exactly one selected Yellow dispatch heuristic")
    best_profit = max(float(row["profit_per_launched_driver_mean"]) for row in heuristic_rows)
    tied_best = {
        row["policy"]
        for row in heuristic_rows
        if abs(float(row["profit_per_launched_driver_mean"]) - best_profit) < 1e-9
    }
    if selected[0]["policy"] not in tied_best:
        fail(
            f"Selected dispatch heuristic should be one of {sorted(tied_best)} "
            f"but found {selected[0]['policy']}"
        )

    density_rows_all = load_csv(density_path)
    for domain in ("yellow", "green"):
        density_rows_domain = [row for row in density_rows_all if row["domain"] == domain]
        observed_density = {int(row["density_pct"]) for row in density_rows_domain}
        if observed_density != {100, 25, 10}:
            fail(f"{domain} dispatch density CI summary should contain densities 100, 25, 10; found {sorted(observed_density)}")
    for density in (100, 25, 10):
        density_rows = [row for row in density_rows_all if row["domain"] == "yellow" and int(row["density_pct"]) == density]
        density_selected = [row for row in density_rows if row.get("selected_for_paper") == "True"]
        if len(density_selected) != 1:
            fail(f"dispatch density {density}% should mark exactly one selected heuristic")

    for field in ("rider_service_rate_mean", "mean_wait_min_mean", "mean_batch_runtime_s_mean"):
        if field not in yellow_rows[0]:
            fail(f"dispatch_primary_ci_summary.csv is missing required dispatch metric column: {field}")

    green_rows = [row for row in load_csv(primary_path) if row["domain"] == "green"]
    green_selected = next((row for row in green_rows if row.get("selected_for_paper") == "True"), None)
    transfer_rows = load_csv(domain_path)
    funnel_rows = load_csv(funnel_path)
    sensitivity_rows = load_csv(sensitivity_path)

    print(
        "Dispatch outputs verified: "
        f"selected heuristic is {selected[0]['policy']} at "
        f"{float(selected[0]['profit_per_launched_driver_mean']):.2f} profit/driver"
    )
    rows = {
        "yellow_coldstart": find_row(yellow_rows, "policy", "coldstart"),
        "yellow_heuristic": selected[0],
        "yellow_warmup": find_row(yellow_rows, "policy", "warmup"),
        "yellow_oracle": find_row(yellow_rows, "policy", "oracle"),
    }
    if green_rows and green_selected is not None:
        rows.update({
            "green_coldstart": find_row(green_rows, "policy", "coldstart"),
            "green_heuristic": green_selected,
            "green_warmup": find_row(green_rows, "policy", "warmup"),
            "green_oracle": find_row(green_rows, "policy", "oracle"),
        })
        transfer_required = {"coldstart", green_selected["policy"], "warmup", "oracle"}
        observed_transfer = {row["policy"] for row in transfer_rows}
        if not transfer_required.issubset(observed_transfer):
            fail(f"domain_transfer_ci_summary.csv missing required policies: {sorted(transfer_required - observed_transfer)}")

    observed_stages = {row["stage"] for row in funnel_rows}
    required_stages = {"retrieved_candidates", "available_exact_time_candidates", "feasible_after_detour_seat", "matched_riders"}
    if observed_stages != required_stages:
        fail(f"matching_ball_funnel_summary.csv should contain {sorted(required_stages)}; found {sorted(observed_stages)}")

    observed_grid = {
        (int(row["matching_window_min"]), int(float(row["max_detour_min"])))
        for row in sensitivity_rows
        if row["domain"] == "yellow" and row["density_pct"] == "10"
    }
    expected_grid = {
        (2, 2), (5, 2), (10, 2),
        (2, 4), (5, 4), (10, 4),
        (2, 6), (5, 6), (10, 6),
    }
    if observed_grid != expected_grid:
        fail(f"sensitivity_grid_summary.csv should contain the full Yellow 10% 3x3 grid; found {sorted(observed_grid)}")
    return rows


def main() -> None:
    full_row, sparse_row, scenario_params = validate_realism_tables()
    tuned_row, temporal_row, green_temporal_row = validate_model_outputs()
    heuristic_gap_row = validate_strategy_gaps()
    dispatch_rows = validate_dispatch_outputs_if_present()
    validate_paper_package(
        full_row,
        sparse_row,
        scenario_params,
        tuned_row,
        temporal_row,
        green_temporal_row,
        heuristic_gap_row,
        dispatch_rows,
    )
    print("Artifact consistency PASSED.")


if __name__ == "__main__":
    try:
        main()
    except StopIteration:
        fail("Expected row not found while validating paper artifacts")
