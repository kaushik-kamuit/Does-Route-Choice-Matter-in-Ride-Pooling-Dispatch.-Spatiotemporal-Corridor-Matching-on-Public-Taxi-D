"""
Deprecated compatibility wrapper for the public training-dataset entry point.

Historically the repository exposed ``python src/models/build_dataset.py`` with
older assumptions and a narrower feature set. The publication artifact is now
built through ``scripts/build_enhanced_dataset.py``. Keeping both full
implementations in the repo invites accidental divergence, so this module now
forwards to the enhanced builder while preserving the familiar CLI entry point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "build_enhanced_dataset.py"), *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    main()
