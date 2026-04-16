from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from data_prep.domain_config import DEFAULT_MONTH_WINDOW, YEAR, DomainConfig, get_domain_config
from dispatch import DispatchConfig, RollingDispatcher
from dispatch.rolling_dispatcher import ALL_POLICIES
from models.predict import ProfitPredictor
from simulation.domain_io import load_domain_assets, load_h3_stats_dict
from spatial.router import OSRMRouter

RESULTS_DIR = ROOT / "results"
DISPATCH_ROOT = RESULTS_DIR / "dispatch"
PYTHON = sys.executable

YELLOW_DENSITIES = [100, 25, 10]
YELLOW_WINDOWS = [2, 5, 10]
YELLOW_DETOURS = [2.0, 4.0, 6.0]

DRIVER_COLS = [
    "split",
    "pickup_datetime",
    "origin_lat",
    "origin_lng",
    "dest_lat",
    "dest_lng",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "trip_distance_miles",
]
RIDER_COLS = [
    "split",
    "pickup_datetime",
    "pickup_h3",
    "dropoff_h3",
    "pickup_lat",
    "pickup_lng",
    "dropoff_lat",
    "dropoff_lng",
    "passenger_count",
    "fare_amount",
]

HEURISTIC_POLICIES = [
    "heuristic_count",
    "heuristic_fare_density",
    "heuristic_feasible_count",
    "heuristic_profit_proxy",
]

GREEN_TRAINING_SAMPLE = 20_000


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 72}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _ensure_months_downloaded(domain: str, months: list[int]) -> None:
    config = get_domain_config(domain)
    missing = [month for month in months if not config.raw_month_path(month).exists()]
    if not missing:
        return
    _run(
        [PYTHON, "src/data_prep/download_2015.py", "--domain", domain, "--months", *[str(month) for month in missing]],
        f"Download raw {domain} months {missing}",
    )


def _dataset_is_usable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["expected_profit", "driver_id"])
    except Exception:
        return False
    return not df.empty


def _estimate_window_thresholds(domain: str, months: list[int]) -> tuple[int, int]:
    from data_prep.preprocess import clean, load_raw

    config = get_domain_config(domain)
    last_month = months[-1]
    df = load_raw(config, last_month)
    df = clean(df)
    distances = df["trip_distance_miles"]
    test_drivers = int((distances > 10.0).sum())
    eligible_test_riders = int(distances.between(0.5, 10.0).sum())
    retained_test_riders = int(eligible_test_riders * 0.25)
    return test_drivers, retained_test_riders


def _select_green_window() -> list[int] | None:
    for start_month in range(1, 10):
        months = list(range(start_month, start_month + 4))
        _ensure_months_downloaded("green", months)
        test_drivers, retained_riders = _estimate_window_thresholds("green", months)
        if test_drivers >= 2_000 and retained_riders >= 100_000:
            return months
    return None


def _ensure_domain_ready(domain: str, months: list[int]) -> None:
    config = get_domain_config(domain)
    _ensure_months_downloaded(domain, months)

    drivers_path = config.drivers_path()
    riders_path = config.riders_path()
    if not drivers_path.exists() or not riders_path.exists():
        _run(
            [
                PYTHON,
                "src/data_prep/preprocess.py",
                "--domain",
                domain,
                "--months",
                *[str(month) for month in months],
            ],
            f"Preprocess {domain} window {months}",
        )

    if not config.h3_stats_path().exists():
        _run([PYTHON, "scripts/build_h3_stats.py", "--domain", domain], f"Build H3 stats ({domain})")

    dataset_path = config.training_dataset_path()
    if not _dataset_is_usable(dataset_path):
        build_cmd = [
            PYTHON,
            "scripts/build_enhanced_dataset.py",
            "--domain",
            domain,
            "--max-request-offset-min",
            "5",
            "--max-detour-min",
            "4",
        ]
        if domain == "green":
            build_cmd.extend(["--sample", str(GREEN_TRAINING_SAMPLE), "--fetch-routes"])
        _run(build_cmd, f"Build training dataset ({domain})")

    if not config.model_path().exists():
        _run([PYTHON, "src/models/train_profit_model.py", "--domain", domain], f"Train model ({domain})")


def _load_test_frames(domain: str, *, sample: int | None) -> tuple[DomainConfig, pd.DataFrame, pd.DataFrame]:
    config, drivers_df, riders_df = load_domain_assets(
        domain,
        driver_columns=DRIVER_COLS,
        rider_columns=RIDER_COLS,
        split="test",
    )
    if sample is not None and sample < len(drivers_df):
        drivers_df = drivers_df.sample(n=sample, random_state=42).reset_index(drop=True)
    return config, drivers_df.reset_index(drop=True), riders_df.reset_index(drop=True)


def _scenario_dir(domain: str, tag: str) -> Path:
    path = DISPATCH_ROOT / domain / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _summarize_scenario(summary_rows: list[dict[str, object]], config: DispatchConfig, *, driver_sample_size: int) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    grouped = df.groupby("policy", as_index=False).agg(
        total_profit=("total_profit", "mean"),
        profit_per_launched_driver=("profit_per_launched_driver", "mean"),
        launched_drivers=("launched_drivers", "mean"),
        served_riders=("served_riders", "mean"),
        rider_service_rate=("rider_service_rate", "mean"),
        mean_wait_min=("mean_wait_min", "mean"),
        mean_matched_riders_per_driver=("mean_matched_riders_per_driver", "mean"),
        seat_occupancy=("seat_occupancy", "mean"),
        mean_detour_min=("mean_detour_min", "mean"),
        mean_eval_time_s=("mean_eval_time_s", "mean"),
        mean_batch_runtime_s=("mean_batch_runtime_s", "mean"),
    )
    grouped["domain"] = config.domain
    grouped["density_pct"] = config.density_pct
    grouped["matching_window_min"] = config.max_request_offset_min
    grouped["max_detour_min"] = config.max_detour_min
    grouped["batch_seconds"] = config.batch_seconds
    grouped["scenario_name"] = config.scenario_name
    grouped["driver_sample_size"] = driver_sample_size
    grouped["n_seeds"] = df["seed"].nunique()
    if not grouped.empty:
        heuristic_mask = grouped["policy"].isin(HEURISTIC_POLICIES)
        best_heuristic = grouped.loc[heuristic_mask].sort_values("profit_per_launched_driver", ascending=False).head(1)
        grouped["selected_for_paper"] = False
        if not best_heuristic.empty:
            grouped.loc[grouped["policy"] == best_heuristic.iloc[0]["policy"], "selected_for_paper"] = True
            grouped["selected_heuristic_policy"] = best_heuristic.iloc[0]["policy"]
        else:
            grouped["selected_heuristic_policy"] = ""
    return grouped


def _run_dispatch_scenario(
    domain: str,
    config: DispatchConfig,
    *,
    sample: int | None,
    seeds: list[int],
    fetch: bool,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    domain_config, drivers_df, riders_df = _load_test_frames(domain, sample=sample)
    router = OSRMRouter(
        cache_path=domain_config.route_cache_path,
        cache_only=not fetch and domain_config.route_cache_path.exists(),
    )
    predictor = ProfitPredictor(domain_config.model_path())
    h3_stats_dict = load_h3_stats_dict(domain_config)
    dispatcher = RollingDispatcher(
        config,
        domain_config=domain_config,
        router=router,
        predictor=predictor,
        h3_stats_dict=h3_stats_dict,
    )

    scenario_dir = _scenario_dir(domain, config.scenario_name)
    outcome_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    sampled_riders_df, rider_index, request_states, request_batches = dispatcher.prepare_rider_pool(riders_df)

    active_policies = policies or ALL_POLICIES
    for policy in active_policies:
        print(f"\nRunning dispatch scenario '{config.scenario_name}' [{domain}] policy={policy}")
        for seed in seeds:
            outcomes, batch_metrics, summary = dispatcher.run_policy(
                policy,
                drivers_df,
                riders_df,
                seed=seed,
                sampled_riders_df=sampled_riders_df,
                rider_index=rider_index,
                request_states=request_states,
                request_batches=request_batches,
            )
            outcome_rows.extend([{**row.to_dict(), "domain": domain, "scenario_name": config.scenario_name} for row in outcomes])
            batch_rows.extend([{**row.to_dict(), "domain": domain, "scenario_name": config.scenario_name} for row in batch_metrics])
            summary_rows.append({**summary.to_dict(), "domain": domain, "scenario_name": config.scenario_name})

    router.flush_cache()

    _save_rows(scenario_dir / "dispatch_outcomes.csv", outcome_rows)
    _save_rows(scenario_dir / "dispatch_batch_metrics.csv", batch_rows)
    _save_rows(scenario_dir / "dispatch_seed_summary.csv", summary_rows)
    config_payload = {**config.to_dict(), "driver_sample_size": len(drivers_df)}
    (scenario_dir / "dispatch_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    scenario_summary = _summarize_scenario(summary_rows, config, driver_sample_size=len(drivers_df))
    scenario_summary.to_csv(scenario_dir / "dispatch_summary.csv", index=False)
    return scenario_summary


def _write_dispatch_public_outputs(
    yellow_primary: pd.DataFrame,
    green_primary: pd.DataFrame | None,
    yellow_density: list[pd.DataFrame],
    yellow_sensitivity: list[pd.DataFrame],
) -> None:
    yellow_primary.to_csv(RESULTS_DIR / "dispatch_yellow_primary.csv", index=False)
    if green_primary is not None:
        green_primary.to_csv(RESULTS_DIR / "dispatch_green_primary.csv", index=False)

    density_df = pd.concat(yellow_density, ignore_index=True) if yellow_density else pd.DataFrame()
    if not density_df.empty:
        density_df.to_csv(RESULTS_DIR / "dispatch_density_summary.csv", index=False)
        density_df[
            density_df["policy"].isin(["coldstart", "warmup", "oracle"]) | density_df["selected_for_paper"]
        ].to_csv(RESULTS_DIR / "dispatch_service_wait_summary.csv", index=False)

    sensitivity_df = pd.concat(yellow_sensitivity, ignore_index=True) if yellow_sensitivity else pd.DataFrame()
    window_df = (
        sensitivity_df[sensitivity_df["max_detour_min"] == 4.0].copy()
        if not sensitivity_df.empty
        else pd.DataFrame()
    )
    if not window_df.empty:
        window_df.to_csv(RESULTS_DIR / "dispatch_window_sensitivity.csv", index=False)

    detour_df = (
        sensitivity_df[sensitivity_df["matching_window_min"] == 5].copy()
        if not sensitivity_df.empty
        else pd.DataFrame()
    )
    if not detour_df.empty:
        detour_df.to_csv(RESULTS_DIR / "dispatch_detour_sensitivity.csv", index=False)

    if green_primary is not None:
        transfer_rows = []
        for policy in ["coldstart", "warmup", "oracle", *HEURISTIC_POLICIES]:
            y_row = yellow_primary[yellow_primary["policy"] == policy]
            g_row = green_primary[green_primary["policy"] == policy]
            if y_row.empty or g_row.empty:
                continue
            transfer_rows.append(
                {
                    "policy": policy,
                    "yellow_profit_per_driver": float(y_row.iloc[0]["profit_per_launched_driver"]),
                    "green_profit_per_driver": float(g_row.iloc[0]["profit_per_launched_driver"]),
                    "yellow_service_rate": float(y_row.iloc[0]["rider_service_rate"]),
                    "green_service_rate": float(g_row.iloc[0]["rider_service_rate"]),
                    "yellow_mean_wait_min": float(y_row.iloc[0]["mean_wait_min"]),
                    "green_mean_wait_min": float(g_row.iloc[0]["mean_wait_min"]),
                }
            )
        pd.DataFrame(transfer_rows).to_csv(RESULTS_DIR / "domain_transfer_summary.csv", index=False)

        temporal_rows = []
        for domain in ("yellow", "green"):
            path = (ROOT / "results" if domain == "yellow" else get_domain_config(domain).results_dir) / "temporal_generalization.csv"
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    row["domain"] = domain
                    temporal_rows.append(row)
        pd.DataFrame(temporal_rows).to_csv(RESULTS_DIR / "domain_temporal_generalization.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rolling dispatch artifact and public-domain robustness study")
    parser.add_argument("--sample", type=int, default=1000, help="Test-driver sample for dispatch runs (paper headline default)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (default 5)")
    parser.add_argument("--skip-green", action="store_true", help="Skip the Green robustness run")
    parser.add_argument("--skip-yellow", action="store_true", help="Skip all Yellow-domain runs")
    parser.add_argument("--green-only", action="store_true", help="Run only the Green primary scenario")
    parser.add_argument("--fetch", action="store_true", help="Allow OSRM fetches when routes are missing from cache")
    parser.add_argument("--primary-only", action="store_true", help="Run only the Yellow primary scenario (plus optional Green primary)")
    parser.add_argument("--skip-primary", action="store_true", help="Skip primary dispatch scenarios and resume density/sensitivity work")
    parser.add_argument("--skip-density", action="store_true", help="Skip Yellow density sweeps")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip Yellow window/detour sensitivity sweeps")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46][: args.seeds]
    if args.green_only:
        args.skip_yellow = True

    yellow_primary = None
    yellow_density: list[pd.DataFrame] = []
    yellow_sensitivity_runs: list[pd.DataFrame] = []
    reduced_policies = ["coldstart", "heuristic_feasible_count", "warmup", "oracle"]
    if not args.skip_yellow:
        yellow_months = list(DEFAULT_MONTH_WINDOW)
        _ensure_domain_ready("yellow", yellow_months)

        if not args.skip_primary:
            yellow_primary = _run_dispatch_scenario(
                "yellow",
                DispatchConfig(
                    domain="yellow",
                    scenario_name="primary_dispatch_d10_w5_det4",
                    density_pct=10,
                    max_request_offset_min=5,
                    max_detour_min=4.0,
                ),
                sample=args.sample,
                seeds=seeds,
                fetch=args.fetch,
            )
            selected_heuristic = yellow_primary.loc[yellow_primary["selected_for_paper"], "policy"].iloc[0]
            reduced_policies = ["coldstart", selected_heuristic, "warmup", "oracle"]

        if not args.primary_only and not args.skip_density:
            yellow_density = [
                _run_dispatch_scenario(
                    "yellow",
                    DispatchConfig(
                        domain="yellow",
                        scenario_name=f"dispatch_density_d{density}",
                        density_pct=density,
                        max_request_offset_min=5,
                        max_detour_min=4.0,
                    ),
                    sample=args.sample,
                    seeds=seeds,
                    fetch=args.fetch,
                    policies=reduced_policies,
                )
                for density in YELLOW_DENSITIES
            ]

        if not args.primary_only and not args.skip_sensitivity:
            for window_min in YELLOW_WINDOWS:
                for detour_min in YELLOW_DETOURS:
                    yellow_sensitivity_runs.append(_run_dispatch_scenario(
                        "yellow",
                        DispatchConfig(
                            domain="yellow",
                            scenario_name=f"dispatch_sensitivity_w{window_min}_det{int(detour_min)}_d10",
                            density_pct=10,
                            max_request_offset_min=window_min,
                            max_detour_min=detour_min,
                        ),
                        sample=min(args.sample, 2000),
                        seeds=seeds,
                        fetch=args.fetch,
                        policies=reduced_policies,
                    ))

    green_primary = None
    green_density: list[pd.DataFrame] = []
    if not args.skip_green:
        green_window = _select_green_window()
        if green_window is not None:
            _ensure_domain_ready("green", green_window)
            if not args.skip_primary:
                green_primary = _run_dispatch_scenario(
                    "green",
                    DispatchConfig(
                        domain="green",
                        scenario_name="primary_dispatch_d10_w5_det4",
                        density_pct=10,
                        max_request_offset_min=5,
                        max_detour_min=4.0,
                    ),
                    sample=min(args.sample, 3000),
                    seeds=seeds,
                    fetch=True if not args.fetch else args.fetch,
                    policies=reduced_policies,
                )
            if not args.primary_only and not args.skip_density:
                green_density = [
                    _run_dispatch_scenario(
                        "green",
                        DispatchConfig(
                            domain="green",
                            scenario_name=f"dispatch_density_d{density}",
                            density_pct=density,
                            max_request_offset_min=5,
                            max_detour_min=4.0,
                        ),
                        sample=min(args.sample, 3000),
                        seeds=seeds,
                        fetch=True if not args.fetch else args.fetch,
                        policies=reduced_policies,
                    )
                    for density in YELLOW_DENSITIES
                ]
        else:
            print("Green 2015 did not satisfy the public robustness thresholds; skipping Green domain run.")

    if yellow_primary is not None:
        _write_dispatch_public_outputs(yellow_primary, green_primary, yellow_density, yellow_sensitivity_runs)
    elif green_primary is not None:
        green_primary.to_csv(RESULTS_DIR / "dispatch_green_primary.csv", index=False)
    _run([PYTHON, "scripts/summarize_dispatch_results.py"], "Summarize dispatch evidence")


if __name__ == "__main__":
    main()
