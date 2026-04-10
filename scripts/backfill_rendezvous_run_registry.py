from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rendezvous.run_registry import backfill_legacy_runs


def main() -> None:
    results_dir = ROOT / "results"
    created = backfill_legacy_runs(results_dir)
    if created:
        print(f"Registered {len(created)} legacy runs in {results_dir / 'runs'}")
    else:
        print("No new legacy runs needed registration.")


if __name__ == "__main__":
    main()
