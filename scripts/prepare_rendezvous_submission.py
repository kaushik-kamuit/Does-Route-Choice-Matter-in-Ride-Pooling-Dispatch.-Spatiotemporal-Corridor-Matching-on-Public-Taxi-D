from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the paper-facing rendezvous outputs in the safe registry-backed order."
    )
    parser.add_argument("--skip-backfill", action="store_true", help="Skip legacy run registration.")
    parser.add_argument("--skip-summaries", action="store_true", help="Skip summary rebuild.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip publication figure rebuild.")
    parser.add_argument("--skip-case-studies", action="store_true", help="Skip case-study overlay rebuild.")
    parser.add_argument("--skip-overleaf", action="store_true", help="Skip slim Overleaf package sync.")
    args = parser.parse_args()

    commands: list[list[str]] = []
    if not args.skip_backfill:
        commands.append([sys.executable, str(ROOT / "scripts" / "backfill_rendezvous_run_registry.py")])
    if not args.skip_summaries:
        commands.append([sys.executable, str(ROOT / "scripts" / "summarize_rendezvous_results.py")])
    if not args.skip_figures:
        commands.append([sys.executable, str(ROOT / "visualizations" / "plot_rendezvous_figures.py")])
    if not args.skip_case_studies:
        commands.append([sys.executable, str(ROOT / "scripts" / "build_case_studies.py")])
    if not args.skip_overleaf:
        commands.append([sys.executable, str(ROOT / "scripts" / "sync_overleaf_package.py")])

    for command in commands:
        result = subprocess.run(command, cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
