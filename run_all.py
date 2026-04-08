from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paper 2 rendezvous experiment suite")
    parser.add_argument("--single-driver-only", action="store_true", help="Run only the controlled route-choice study")
    parser.add_argument("--dispatch-only", action="store_true", help="Run only the dispatch validation study")
    args, passthrough = parser.parse_known_args()

    commands: list[list[str]] = []
    if not args.dispatch_only:
        commands.append([sys.executable, str(ROOT / "scripts" / "run_rendezvous_artifact.py"), *passthrough])
    if not args.single_driver_only:
        commands.append([sys.executable, str(ROOT / "scripts" / "run_rendezvous_dispatch.py"), *passthrough])

    for command in commands:
        result = subprocess.run(command, cwd=str(ROOT))
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
