"""
Single entry point to run the full warm-up vs cold-start experiment pipeline.

Phases:
  2. Build ML training dataset + train profit predictor
  3. Run paired cold-start / warm-up simulation
  4. Generate comparison plots and statistical summary

Usage:
    python run_all.py                 # full pipeline
    python run_all.py --no-api        # skip OSRM, use straight-line corridors
    python run_all.py --sample 5000   # fewer test drivers for speed
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED ({label}) exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  Completed {label} in {elapsed:.0f}s ({elapsed/60:.1f} min)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--no-api", action="store_true",
                        help="Use straight-line corridors instead of OSRM routes")
    parser.add_argument("--sample", type=int, default=None,
                        help="Override sample size for simulation runner")
    parser.add_argument("--ds-sample", type=int, default=None,
                        help="Override sample size for build_dataset")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of experiment seeds (default 5)")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip dataset building (reuse existing)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (reuse existing)")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation (reuse existing results)")
    args = parser.parse_args()

    if args.no_api:
        os.environ["CARPOOL_NO_API"] = "1"

    t_total = time.time()

    # --- Phase 2A: Build dataset ---
    if not args.skip_dataset:
        ds_cmd = [sys.executable, "src/models/build_dataset.py"]
        if args.ds_sample:
            ds_cmd += ["--sample", str(args.ds_sample)]
        _run(ds_cmd, "Phase 2A: Build Training Dataset")

    # --- Phase 2B: Train model ---
    if not args.skip_train:
        _run([sys.executable, "src/models/train_profit_model.py"],
             "Phase 2B: Train Profit Model")

    # --- Phase 3: Simulation ---
    if not args.skip_sim:
        sim_cmd = [sys.executable, "src/simulation/runner.py"]
        if args.sample:
            sim_cmd += ["--sample", str(args.sample)]
        sim_cmd += ["--seeds", str(args.seeds)]
        _run(sim_cmd, "Phase 3: Run Simulation")

    # --- Phase 4: Visualization ---
    _run([sys.executable, "visualizations/plot_comparison.py"],
         "Phase 4A: Comparison Plots")

    _run([sys.executable, "visualizations/plot_model.py"],
         "Phase 4B: Model Insight Plots")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results: {ROOT / 'results'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
