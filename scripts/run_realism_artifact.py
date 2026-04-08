"""
Run the realism-first artifact pipeline end to end.

Phases:
  0. Archive the legacy optimistic 45-minute-window artifacts
  1. Rebuild the training dataset for the headline 5-minute exact window
  2. Retrain the profit model and regenerate model-comparison outputs
  3. Run the headline density sweep at the 5-minute exact request window
  4. Run request-window and detour sensitivity scenarios
  5. Regenerate summaries, figures, and the paper-facing validator outputs

The headline scenario is:
  - retained rider pre-sample: 25% (applied in preprocessing)
  - exact request window: 5 minutes
  - retained-sample densities: 100%, 75%, 50%, 25%, 10%
  - max detour: 4 minutes
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "ml"
LEGACY_DIR = ROOT / "artifacts" / "legacy_optimistic_45min"

PRIMARY_DENSITIES = [
    (1.00, ""),
    (0.75, "d75"),
    (0.50, "d50"),
    (0.25, "d25"),
    (0.10, "d10"),
]

WINDOW_SENSITIVITY = [
    (2, 1.00, "w2"),
    (2, 0.25, "w2_d25"),
    (2, 0.10, "w2_d10"),
    (10, 1.00, "w10"),
    (10, 0.25, "w10_d25"),
    (10, 0.10, "w10_d10"),
]

DETOUR_SENSITIVITY = [
    (2.0, 0.10, "w5_det2_d10"),
    (6.0, 0.10, "w5_det6_d10"),
]

H3_SENSITIVITY = [
    (8, 1, 0.10, "h3r8_k1_d10"),
    (9, 0, 0.10, "h3r9_k0_d10"),
    (9, 2, 0.10, "h3r9_k2_d10"),
]

ECON_SENSITIVITY = [
    (0.40, 0.67, 0.10, "econ_ps40_d10"),
    (0.60, 0.67, 0.10, "econ_ps60_d10"),
    (0.50, 0.50, 0.10, "econ_c50_d10"),
    (0.50, 0.85, 0.10, "econ_c85_d10"),
]


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 72}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  FAILED ({label}) exit code {result.returncode}")
        raise SystemExit(result.returncode)
    print(f"\n  Completed {label} in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def archive_legacy_artifact() -> None:
    if LEGACY_DIR.exists():
        print(f"Legacy artifact already archived: {LEGACY_DIR}")
        return

    print(f"Archiving legacy optimistic artifact to: {LEGACY_DIR}")
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    for pattern in (
        "*.csv",
        "*.txt",
        "*.json",
    ):
        for path in RESULTS_DIR.glob(pattern):
            _copy_if_exists(path, LEGACY_DIR / "results" / path.name)

    _copy_if_exists(PLOTS_DIR, LEGACY_DIR / "results" / "plots")

    for pattern in ("profit_model_v2*.pkl", "feature_importance_v2*.csv"):
        for path in MODELS_DIR.glob(pattern):
            _copy_if_exists(path, LEGACY_DIR / "models" / path.name)

    _copy_if_exists(DATA_DIR / "training_dataset_v2.parquet", LEGACY_DIR / "data" / "training_dataset_v2.parquet")
    print("Legacy optimistic artifact archived.")


def build_headline_dataset(py: str, ds_sample: int | None) -> None:
    cmd = [
        py, "scripts/build_enhanced_dataset.py",
        "--index-bin-minutes", "15",
        "--candidate-window-bins", "1",
        "--max-request-offset-min", "5",
        "--max-detour-min", "4",
    ]
    if ds_sample is not None:
        cmd += ["--sample", str(ds_sample)]
    _run(cmd, "Phase 1A: Build headline 5-minute training dataset")


def train_models(py: str) -> None:
    _run([py, "src/models/train_profit_model.py"], "Phase 1B: Train LightGBM headline model")
    _run([py, "scripts/compare_models.py"], "Phase 1C: Compare model families")
    _run([py, "scripts/ablation_study.py"], "Phase 1D: Run feature ablation study")


def run_density_sweep(py: str, sample: int | None, seeds: int, model_path: str) -> None:
    base = [
        py, "src/simulation/runner.py",
        "--seeds", str(seeds),
        "--index-bin-minutes", "15",
        "--candidate-window-bins", "1",
        "--max-request-offset-min", "5",
        "--max-detour-min", "4",
        "--model-path", model_path,
    ]
    if sample is not None:
        base += ["--sample", str(sample)]

    for density, tag in PRIMARY_DENSITIES:
        cmd = base.copy()
        if density < 1.0:
            cmd += ["--density", str(density), "--tag", tag]
        _run(cmd, f"Phase 2: Headline density sweep ({int(density * 100)}%)")


def run_window_sensitivity(py: str, sample: int | None, seeds: int, model_path: str) -> None:
    base = [
        py, "src/simulation/runner.py",
        "--seeds", str(seeds),
        "--index-bin-minutes", "15",
        "--candidate-window-bins", "1",
        "--max-detour-min", "4",
        "--model-path", model_path,
    ]
    if sample is not None:
        base += ["--sample", str(sample)]

    for window_min, density, tag in WINDOW_SENSITIVITY:
        cmd = base.copy() + [
            "--max-request-offset-min", str(window_min),
            "--density", str(density),
            "--tag", tag,
        ]
        _run(
            cmd,
            f"Phase 3A: Request-window sensitivity ({window_min} min, {int(density * 100)}%)",
        )


def run_detour_sensitivity(py: str, sample: int | None, seeds: int, model_path: str) -> None:
    base = [
        py, "src/simulation/runner.py",
        "--seeds", str(seeds),
        "--index-bin-minutes", "15",
        "--candidate-window-bins", "1",
        "--max-request-offset-min", "5",
        "--model-path", model_path,
    ]
    if sample is not None:
        base += ["--sample", str(sample)]

    for detour_min, density, tag in DETOUR_SENSITIVITY:
        cmd = base.copy() + [
            "--max-detour-min", str(detour_min),
            "--density", str(density),
            "--tag", tag,
        ]
        _run(
            cmd,
            f"Phase 3B: Detour sensitivity ({detour_min:.0f} min, {int(density * 100)}%)",
        )


def regenerate_publication_outputs(py: str) -> None:
    _run([py, "old-scripts/plot_comparison.py"], "Phase 4A: Comparison plots")
    _run([py, "scripts/summarize_realism_results.py"], "Phase 4B: Realism scenario summaries")
    _run([py, "scripts/analyze_strategy_gaps.py"], "Phase 4C: Paired strategy gap analysis")
    _run([py, "old-scripts/plot_extended.py"], "Phase 4D: Extended analysis plots")
    _run([py, "visualizations/plot_paper_figures.py"], "Phase 4E: Paper figure generation")
    _run([py, "scripts/validate_paper_consistency.py"], "Phase 4F: Artifact consistency validation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the realism-first paper artifact pipeline")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Simulation test-driver sample size (default 5000)")
    parser.add_argument("--ds-sample", type=int, default=None,
                        help="Optional training driver sample size for dataset rebuild")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of simulation seeds (default 5)")
    parser.add_argument("--skip-archive", action="store_true",
                        help="Skip archiving the existing optimistic artifact")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip rebuilding the headline training dataset")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip retraining and model comparison")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip headline and sensitivity simulations")
    parser.add_argument("--skip-post", action="store_true",
                        help="Skip post-processing, figure generation, and validation")
    parser.add_argument("--max-riders", type=int, default=None,
                        help="Optional rider cap passed to dataset/simulation for memory-constrained runs")
    parser.add_argument("--sensitivity-sample", type=int, default=2000,
                        help="Driver sample used for secondary sensitivity sweeps (default 2000)")
    args = parser.parse_args()

    py = sys.executable
    model_path = str(MODELS_DIR / "profit_model_v2.pkl")
    total_start = time.time()

    if not args.skip_archive:
        archive_legacy_artifact()

    if not args.skip_dataset:
        build_cmd_sample = args.ds_sample
        cmd = [
            py, "scripts/build_enhanced_dataset.py",
            "--index-bin-minutes", "15",
            "--candidate-window-bins", "1",
            "--max-request-offset-min", "5",
            "--max-detour-min", "4",
        ]
        if build_cmd_sample is not None:
            cmd += ["--sample", str(build_cmd_sample)]
        if args.max_riders is not None:
            cmd += ["--max-riders", str(args.max_riders)]
        _run(cmd, "Phase 1A: Build headline 5-minute training dataset")

    if not args.skip_train:
        train_models(py)

    if not args.skip_sim:
        base_sim_args = []
        if args.max_riders is not None:
            base_sim_args = ["--max-riders", str(args.max_riders)]
        def _run_with_base(cmd: list[str], label: str) -> None:
            _run(cmd + base_sim_args, label)

        base = [
            py, "src/simulation/runner.py",
            "--seeds", str(args.seeds),
            "--scenario-name", "primary",
            "--index-bin-minutes", "15",
            "--candidate-window-bins", "1",
            "--max-request-offset-min", "5",
            "--max-detour-min", "4",
            "--model-path", model_path,
        ]
        if args.sample is not None:
            base += ["--sample", str(args.sample)]
        for density, tag in PRIMARY_DENSITIES:
            cmd = base.copy()
            if density < 1.0:
                cmd += ["--density", str(density), "--tag", tag]
            _run_with_base(cmd, f"Phase 2: Headline density sweep ({int(density * 100)}%)")

        base_window = [
            py, "src/simulation/runner.py",
            "--seeds", str(args.seeds),
            "--scenario-name", "window_sensitivity",
            "--index-bin-minutes", "15",
            "--candidate-window-bins", "1",
            "--max-detour-min", "4",
            "--model-path", model_path,
        ]
        if args.sensitivity_sample is not None:
            base_window += ["--sample", str(args.sensitivity_sample)]
        for window_min, density, tag in WINDOW_SENSITIVITY:
            cmd = base_window.copy() + [
                "--max-request-offset-min", str(window_min),
                "--density", str(density),
                "--tag", tag,
            ]
            _run_with_base(
                cmd,
                f"Phase 3A: Request-window sensitivity ({window_min} min, {int(density * 100)}%)",
            )

        base_detour = [
            py, "src/simulation/runner.py",
            "--seeds", str(args.seeds),
            "--scenario-name", "detour_sensitivity",
            "--index-bin-minutes", "15",
            "--candidate-window-bins", "1",
            "--max-request-offset-min", "5",
            "--model-path", model_path,
        ]
        if args.sensitivity_sample is not None:
            base_detour += ["--sample", str(args.sensitivity_sample)]
        for detour_min, density, tag in DETOUR_SENSITIVITY:
            cmd = base_detour.copy() + [
                "--max-detour-min", str(detour_min),
                "--density", str(density),
                "--tag", tag,
            ]
            _run_with_base(
                cmd,
                f"Phase 3B: Detour sensitivity ({detour_min:.0f} min, {int(density * 100)}%)",
            )

        base_h3 = [
            py, "src/simulation/runner.py",
            "--seeds", str(args.seeds),
            "--scenario-name", "h3_sensitivity",
            "--index-bin-minutes", "15",
            "--candidate-window-bins", "1",
            "--max-request-offset-min", "5",
            "--max-detour-min", "4",
            "--model-path", model_path,
        ]
        if args.sensitivity_sample is not None:
            base_h3 += ["--sample", str(args.sensitivity_sample)]
        for resolution, k_ring, density, tag in H3_SENSITIVITY:
            cmd = base_h3.copy() + [
                "--h3-resolution", str(resolution),
                "--corridor-k-ring", str(k_ring),
                "--density", str(density),
                "--tag", tag,
            ]
            _run_with_base(
                cmd,
                f"Phase 3C: H3/corridor sensitivity (res={resolution}, k={k_ring}, {int(density * 100)}%)",
            )

        base_econ = [
            py, "src/simulation/runner.py",
            "--seeds", str(args.seeds),
            "--scenario-name", "economics_sensitivity",
            "--index-bin-minutes", "15",
            "--candidate-window-bins", "1",
            "--max-request-offset-min", "5",
            "--max-detour-min", "4",
            "--model-path", model_path,
        ]
        if args.sensitivity_sample is not None:
            base_econ += ["--sample", str(args.sensitivity_sample)]
        for platform_share, cost_per_mile, density, tag in ECON_SENSITIVITY:
            cmd = base_econ.copy() + [
                "--platform-share", str(platform_share),
                "--cost-per-mile", str(cost_per_mile),
                "--density", str(density),
                "--tag", tag,
            ]
            _run_with_base(
                cmd,
                f"Phase 3D: Economics sensitivity (share={platform_share:.2f}, cost={cost_per_mile:.2f}, {int(density * 100)}%)",
            )

    if not args.skip_post:
        regenerate_publication_outputs(py)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 72}")
    print("  REALISM-FIRST ARTIFACT COMPLETE")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
