from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DISPATCH_ROOT = RESULTS / "dispatch"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance_v2.csv"
MODEL_PATH = ROOT / "models" / "profit_model_v2.pkl"
DATASET_PATH = ROOT / "data" / "ml" / "training_dataset_v2.parquet"

sys.path.insert(0, str(ROOT / "src"))

from models.evaluation_split import build_eval_split

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

METRICS = [
    "total_profit",
    "profit_per_launched_driver",
    "launched_drivers",
    "served_riders",
    "rider_service_rate",
    "mean_wait_min",
    "mean_matched_riders_per_driver",
    "seat_occupancy",
    "mean_detour_min",
    "mean_eval_time_s",
    "mean_batch_runtime_s",
]

HEURISTIC_POLICIES = {
    "heuristic_count",
    "heuristic_fare_density",
    "heuristic_feasible_count",
    "heuristic_profit_proxy",
}


def _feature_group(name: str) -> str:
    if name.startswith("corridor_hist_") or name.startswith("corridor_") or "cell_" in name or name == "mean_rider_fare":
        return "Spatial demand"
    if "hour" in name or "day_" in name or "time_bin" in name or name == "is_weekend":
        return "Temporal"
    if "landmark" in name or "_jfk_" in name or "_lga_" in name or "_penn_" in name or "times_sq" in name:
        return "Landmark"
    return "Geometry"


def _metric_stats(series: pd.Series) -> tuple[float, float, float, int]:
    clean = series.dropna().astype(float)
    n = int(clean.shape[0])
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mean = float(clean.mean())
    if n == 1:
        return mean, mean, mean, 1
    sem = float(clean.std(ddof=1) / math.sqrt(n))
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci, n


def _aggregate_seed_summary(seed_df: pd.DataFrame, meta: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, group in seed_df.groupby("policy", sort=False):
        row: dict[str, object] = {
            "domain": meta["domain"],
            "scenario_name": meta["scenario_name"],
            "density_pct": meta["density_pct"],
            "matching_window_min": meta["max_request_offset_min"],
            "max_detour_min": meta["max_detour_min"],
            "batch_seconds": meta["batch_seconds"],
            "driver_sample_size": meta["driver_sample_size"],
            "policy": policy,
        }
        n_seeds = 0
        for metric in METRICS:
            mean, ci_low, ci_high, n = _metric_stats(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
            n_seeds = max(n_seeds, n)
        row["n_seeds"] = n_seeds
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["selected_for_paper"] = False
    heuristic_rows = df[df["policy"].isin(HEURISTIC_POLICIES)]
    if not heuristic_rows.empty:
        best_idx = heuristic_rows["profit_per_launched_driver_mean"].idxmax()
        df.loc[best_idx, "selected_for_paper"] = True
        selected_policy = df.loc[best_idx, "policy"]
    else:
        selected_policy = ""
    df["selected_heuristic_policy"] = selected_policy
    return df


def _load_dispatch_scenarios() -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    if not DISPATCH_ROOT.exists():
        return scenarios
    for domain_dir in DISPATCH_ROOT.iterdir():
        if not domain_dir.is_dir():
            continue
        for scenario_dir in domain_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            config_path = scenario_dir / "dispatch_config.json"
            seed_summary_path = scenario_dir / "dispatch_seed_summary.csv"
            outcomes_path = scenario_dir / "dispatch_outcomes.csv"
            if not config_path.exists() or not seed_summary_path.exists() or not outcomes_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            seed_df = pd.read_csv(seed_summary_path)
            outcomes_df = pd.read_csv(outcomes_path)
            driver_sample_size = int(config.get("driver_sample_size", 0)) or (
                int(seed_df["launched_drivers"].max()) if not seed_df.empty else 0
            )
            scenarios.append(
                {
                    "domain": str(config["domain"]),
                    "scenario_name": str(config["scenario_name"]),
                    "config": config,
                    "seed_df": seed_df,
                    "outcomes_df": outcomes_df,
                    "driver_sample_size": driver_sample_size,
                }
            )
    return scenarios


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [Saved] {path.name}")


def summarize_dispatch_outputs() -> None:
    scenarios = _load_dispatch_scenarios()
    if not scenarios:
        print("  [Dispatch summary] No dispatch scenarios found; skipping.")
        return

    aggregated: list[pd.DataFrame] = []
    scenario_map: dict[tuple[str, str], pd.DataFrame] = {}
    for entry in scenarios:
        meta = {
            "domain": entry["domain"],
            "scenario_name": entry["scenario_name"],
            "density_pct": entry["config"]["density_pct"],
            "max_request_offset_min": entry["config"]["max_request_offset_min"],
            "max_detour_min": entry["config"]["max_detour_min"],
            "batch_seconds": entry["config"]["batch_seconds"],
            "driver_sample_size": entry["driver_sample_size"],
        }
        agg = _aggregate_seed_summary(entry["seed_df"], meta)
        if not agg.empty:
            aggregated.append(agg)
            scenario_map[(entry["domain"], entry["scenario_name"])] = agg

    if not aggregated:
        print("  [Dispatch summary] Aggregation produced no rows; skipping.")
        return

    all_dispatch = pd.concat(aggregated, ignore_index=True)
    _write(RESULTS / "dispatch_primary_ci_summary.csv", all_dispatch[all_dispatch["scenario_name"] == "primary_dispatch_d10_w5_det4"])
    _write(
        RESULTS / "dispatch_density_ci_summary.csv",
        all_dispatch[all_dispatch["scenario_name"].str.startswith("dispatch_density_")].copy(),
    )

    primary = all_dispatch[all_dispatch["scenario_name"] == "primary_dispatch_d10_w5_det4"].copy()
    if not primary.empty:
        primary["is_primary_selected_policy"] = primary["policy"].eq(primary["selected_heuristic_policy"])
        _write(RESULTS / "domain_transfer_ci_summary.csv", primary)

    sensitivity = all_dispatch[all_dispatch["scenario_name"].str.startswith("dispatch_sensitivity_")].copy()
    sensitivity_rows: list[dict[str, object]] = []
    if not sensitivity.empty:
        for (domain, scenario_name), group in sensitivity.groupby(["domain", "scenario_name"], sort=False):
            warmup = group[group["policy"] == "warmup"]
            cold = group[group["policy"] == "coldstart"]
            if warmup.empty or cold.empty:
                continue
            delta_mean = float(warmup["profit_per_launched_driver_mean"].iloc[0] - cold["profit_per_launched_driver_mean"].iloc[0])
            delta_low = float(warmup["profit_per_launched_driver_ci_low"].iloc[0] - cold["profit_per_launched_driver_ci_high"].iloc[0])
            delta_high = float(warmup["profit_per_launched_driver_ci_high"].iloc[0] - cold["profit_per_launched_driver_ci_low"].iloc[0])
            sensitivity_rows.append(
                {
                    "domain": domain,
                    "scenario_name": scenario_name,
                    "matching_window_min": int(warmup["matching_window_min"].iloc[0]),
                    "max_detour_min": float(warmup["max_detour_min"].iloc[0]),
                    "density_pct": int(warmup["density_pct"].iloc[0]),
                    "driver_sample_size": int(warmup["driver_sample_size"].iloc[0]),
                    "n_seeds": int(warmup["n_seeds"].iloc[0]),
                    "warmup_profit_mean": float(warmup["profit_per_launched_driver_mean"].iloc[0]),
                    "coldstart_profit_mean": float(cold["profit_per_launched_driver_mean"].iloc[0]),
                    "warmup_minus_coldstart_mean": delta_mean,
                    "warmup_minus_coldstart_ci_low": delta_low,
                    "warmup_minus_coldstart_ci_high": delta_high,
                }
            )
    _write(RESULTS / "sensitivity_grid_summary.csv", pd.DataFrame(sensitivity_rows))

    funnel_rows: list[dict[str, object]] = []
    primary_yellow = next(
        (entry for entry in scenarios if entry["domain"] == "yellow" and entry["scenario_name"] == "primary_dispatch_d10_w5_det4"),
        None,
    )
    if primary_yellow is not None:
        outcomes = primary_yellow["outcomes_df"]
        warmup = outcomes[outcomes["policy"] == "warmup"].copy()
        required_cols = {
            "retrieved_candidate_count",
            "available_candidate_count",
            "feasible_count",
            "matched_riders",
        }
        if not warmup.empty and required_cols.issubset(warmup.columns):
            per_seed = (
                warmup.groupby("seed", as_index=False)
                .agg(
                    retrieved_candidates=("retrieved_candidate_count", "mean"),
                    available_exact_time_candidates=("available_candidate_count", "mean"),
                    feasible_after_detour_seat=("feasible_count", "mean"),
                    matched_riders=("matched_riders", "mean"),
                )
            )
            for stage in (
                "retrieved_candidates",
                "available_exact_time_candidates",
                "feasible_after_detour_seat",
                "matched_riders",
            ):
                mean, ci_low, ci_high, n = _metric_stats(per_seed[stage])
                funnel_rows.append(
                    {
                        "domain": "yellow",
                        "scenario_name": "primary_dispatch_d10_w5_det4",
                        "policy": "warmup",
                        "stage": stage,
                        "mean_per_launched_driver": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_seeds": n,
                        "driver_sample_size": int(primary_yellow["driver_sample_size"]),
                    }
                )
        else:
            print("  [Dispatch summary] Retrieval-stage funnel columns missing in current primary dispatch outputs; skipping funnel summary.")
    _write(RESULTS / "matching_ball_funnel_summary.csv", pd.DataFrame(funnel_rows))

    summarize_gain_decomposition_from_public_summaries()


def summarize_gain_decomposition_from_public_summaries() -> None:
    dispatch_density_path = RESULTS / "dispatch_density_ci_summary.csv"
    realism_path = RESULTS / "realism_primary_summary.csv"
    strategy_gap_path = RESULTS / "strategy_gap_results.csv"

    dispatch_df = pd.read_csv(dispatch_density_path) if dispatch_density_path.exists() else pd.DataFrame()
    realism_df = pd.read_csv(realism_path) if realism_path.exists() else pd.DataFrame()
    strategy_gap_df = pd.read_csv(strategy_gap_path) if strategy_gap_path.exists() else pd.DataFrame()

    gain_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []

    for density in (100, 25, 10):
        dispatch_sub = dispatch_df[
            (dispatch_df["domain"] == "yellow")
            & (dispatch_df["density_pct"] == density)
        ].copy()
        if not dispatch_sub.empty:
            cold = dispatch_sub[dispatch_sub["policy"] == "coldstart"]
            warm = dispatch_sub[dispatch_sub["policy"] == "warmup"]
            oracle = dispatch_sub[dispatch_sub["policy"] == "oracle"]
            heur = dispatch_sub[dispatch_sub["selected_for_paper"] == True]
            if not cold.empty and not warm.empty and not oracle.empty and not heur.empty:
                cold_row = cold.iloc[0]
                warm_row = warm.iloc[0]
                oracle_row = oracle.iloc[0]
                heur_row = heur.iloc[0]
                gain_rows.append(
                    {
                        "density_pct": density,
                        "layer": "dispatch",
                        "coldstart_profit": float(cold_row["profit_per_launched_driver_mean"]),
                        "heuristic_profit": float(heur_row["profit_per_launched_driver_mean"]),
                        "warmup_profit": float(warm_row["profit_per_launched_driver_mean"]),
                        "oracle_profit": float(oracle_row["profit_per_launched_driver_mean"]),
                        "route_aware_gain": float(warm_row["profit_per_launched_driver_mean"] - cold_row["profit_per_launched_driver_mean"]),
                        "heuristic_recovery": float(heur_row["profit_per_launched_driver_mean"] - cold_row["profit_per_launched_driver_mean"]),
                        "ml_residual": float(warm_row["profit_per_launched_driver_mean"] - heur_row["profit_per_launched_driver_mean"]),
                        "oracle_headroom": float(oracle_row["profit_per_launched_driver_mean"] - warm_row["profit_per_launched_driver_mean"]),
                        "selected_heuristic_policy": str(heur_row["policy"]),
                    }
                )
                gap_rows.append(
                    {
                        "density_pct": density,
                        "dispatch_gap_mean": float(warm_row["profit_per_launched_driver_mean"] - heur_row["profit_per_launched_driver_mean"]),
                        "dispatch_gap_low": float(warm_row["profit_per_launched_driver_ci_low"] - heur_row["profit_per_launched_driver_ci_high"]),
                        "dispatch_gap_high": float(warm_row["profit_per_launched_driver_ci_high"] - heur_row["profit_per_launched_driver_ci_low"]),
                        "selected_dispatch_heuristic_policy": str(heur_row["policy"]),
                    }
                )

        isolated = realism_df[realism_df["density_pct"] == density].copy()
        if not isolated.empty:
            row = isolated.iloc[0]
            gain_rows.append(
                {
                    "density_pct": density,
                    "layer": "single_driver",
                    "coldstart_profit": float(row["coldstart_profit"]),
                    "heuristic_profit": float(row["heuristic_profit"]),
                    "warmup_profit": float(row["warmup_profit"]),
                    "oracle_profit": float(row["oracle_profit"]),
                    "route_aware_gain": float(row["warmup_profit"] - row["coldstart_profit"]),
                    "heuristic_recovery": float(row["heuristic_profit"] - row["coldstart_profit"]),
                    "ml_residual": float(row["warmup_profit"] - row["heuristic_profit"]),
                    "oracle_headroom": float(row["oracle_profit"] - row["warmup_profit"]),
                    "selected_heuristic_policy": str(row["heuristic_selected_strategy"]),
                }
            )

        gap = strategy_gap_df[
            (strategy_gap_df["density_pct"] == density)
            & (strategy_gap_df["comparison"] == "warmup_vs_heuristic")
        ].copy()
        if not gap.empty:
            row = gap.iloc[0]
            matching_gap_row = next((item for item in gap_rows if item["density_pct"] == density), None)
            if matching_gap_row is None:
                matching_gap_row = {"density_pct": density}
                gap_rows.append(matching_gap_row)
            matching_gap_row.update(
                {
                    "single_driver_gap_mean": float(row["mean_diff"]),
                    "single_driver_gap_low": float(row["boot_low"]),
                    "single_driver_gap_high": float(row["boot_high"]),
                }
            )
            if "dispatch_gap_mean" in matching_gap_row and float(row["mean_diff"]) != 0.0:
                matching_gap_row["dispatch_to_single_driver_ratio"] = float(
                    matching_gap_row["dispatch_gap_mean"] / float(row["mean_diff"])
                )

    _write(RESULTS / "route_gain_decomposition_summary.csv", pd.DataFrame(gain_rows))
    _write(RESULTS / "ml_gap_comparison_summary.csv", pd.DataFrame(gap_rows).sort_values("density_pct"))


def summarize_model_support() -> None:
    if not IMPORTANCE_PATH.exists() or not MODEL_PATH.exists() or not DATASET_PATH.exists():
        print("  [Model summary] Missing model artifacts; skipping.")
        return

    imp = pd.read_csv(IMPORTANCE_PATH)
    family = (
        imp.assign(group=imp["feature"].map(_feature_group))
        .groupby("group", as_index=False)["importance"]
        .sum()
    )
    family["share_pct"] = family["importance"] / family["importance"].sum() * 100.0
    _write(RESULTS / "model_feature_family_summary.csv", family.sort_values("share_pct", ascending=False))

    df = pd.read_parquet(DATASET_PATH)
    X = df[FEATURE_COLS].values
    y = df["expected_profit"].values
    split = build_eval_split(df)
    val_idx = split.val_idx
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X[val_idx])
    cal = pd.DataFrame({"actual": y[val_idx], "pred": pred})
    cal["decile"] = pd.qcut(cal["actual"], 10, duplicates="drop")
    cal_summary = (
        cal.groupby("decile", observed=False)
        .agg(
            actual_mean=("actual", "mean"),
            pred_mean=("pred", "mean"),
            pred_q25=("pred", lambda s: s.quantile(0.25)),
            pred_q75=("pred", lambda s: s.quantile(0.75)),
            n=("pred", "size"),
        )
        .reset_index(drop=True)
    )
    _write(RESULTS / "model_calibration_summary.csv", cal_summary)


def main() -> None:
    print("Summarizing dispatch and model evidence...")
    summarize_dispatch_outputs()
    summarize_model_support()
    summarize_gain_decomposition_from_public_summaries()
    print("Evidence summary complete.")


if __name__ == "__main__":
    main()
