"""
Run the simulation at multiple rider density levels and then generate
all extended plots and statistics.

Usage:
    python scripts/run_density_experiments.py                # all densities
    python scripts/run_density_experiments.py --sample 5000  # fewer drivers
    python scripts/run_density_experiments.py --skip-full    # skip 100% run if already done
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DENSITIES = [0.75, 0.50, 0.25, 0.10]


def run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  {label}: {status} ({elapsed:.0f}s)")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip the full-density (100%%) run")
    args = parser.parse_args()

    py = sys.executable
    runner = str(ROOT / "src" / "simulation" / "runner.py")

    if not args.skip_full:
        run([py, runner, "--sample", str(args.sample), "--seeds", str(args.seeds)],
            "Full density (100%)")

    for density in DENSITIES:
        tag = f"d{int(density * 100)}"
        run([py, runner,
             "--sample", str(args.sample),
             "--seeds", str(args.seeds),
             "--density", str(density),
             "--tag", tag],
            f"Density {density:.0%} (tag={tag})")

    print("\n" + "="*60)
    print("  Generating plots and statistics...")
    print("="*60)

    run([py, str(ROOT / "visualizations" / "plot_comparison.py")],
        "Basic comparison plots")
    run([py, str(ROOT / "visualizations" / "plot_extended.py")],
        "Extended analysis plots")

    print("\n  All experiments complete!")
    print(f"  Results in: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
