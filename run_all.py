"""
Single entry point for the Q2-targeted artifact suite.

By default this runs:
  1. the realism-first single-driver artifact
  2. the rolling-horizon dispatch artifact
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-driver and dispatch artifact suites")
    parser.add_argument("--single-driver-only", action="store_true", help="Run only the realism-first single-driver artifact")
    parser.add_argument("--dispatch-only", action="store_true", help="Run only the rolling dispatch artifact")
    args, passthrough = parser.parse_known_args()
    dispatch_only_flags = {"--primary-only", "--skip-green", "--fetch"}
    realism_args: list[str] = []
    dispatch_args: list[str] = []
    i = 0
    while i < len(passthrough):
        token = passthrough[i]
        target = dispatch_args if token in dispatch_only_flags else realism_args
        target.append(token)
        if i + 1 < len(passthrough) and not passthrough[i + 1].startswith("--"):
            target.append(passthrough[i + 1])
            i += 1
        i += 1

    commands: list[list[str]] = []
    if not args.dispatch_only:
        commands.append([sys.executable, str(ROOT / "scripts" / "run_realism_artifact.py"), *realism_args])
    if not args.single_driver_only:
        commands.append([sys.executable, str(ROOT / "scripts" / "run_dispatch_artifact.py"), *dispatch_args, *realism_args])

    for cmd in commands:
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
