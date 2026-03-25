"""
Lightweight paper consistency checks.

Purpose:
  - Catch stale manuscript/README numbers after reruns.
  - Verify paper-facing counts line up with current result files.
  - Flag known reviewer-facing issues before submission.

Usage:
    python scripts/validate_paper_consistency.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_FILES = [
    ROOT / "README.md",
    ROOT / "paper" / "MANUSCRIPT.md",
    ROOT / "paper" / "main_full.tex",
    ROOT / "paper" / "README.md",
]

STALE_PATTERNS = {
    "10,000 test drivers": "Paper-facing documents should reflect the current 5,000-driver results.",
    "For each of the 10,000 test drivers": "README paired-design section is stale.",
    "R^2 = 0.764": "Old model figure metric still present.",
    "RMSE = \\$8.44": "Old model figure metric still present.",
    "(Optional.)": "Placeholder text should not remain in the paper package.",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_single_row_csv(path: Path, key_field: str, key_value: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row[key_field] == key_value:
            return row
    raise KeyError(f"{key_value!r} not found in {path}")


def main() -> int:
    problems: list[str] = []

    for path in TEXT_FILES:
        if not path.exists():
            problems.append(f"Missing expected file: {path}")
            continue
        text = _read_text(path)
        for pattern, message in STALE_PATTERNS.items():
            if pattern in text:
                problems.append(f"{path.name}: {message} Found pattern {pattern!r}.")

    density_path = ROOT / "results" / "density_results.csv"
    if not density_path.exists():
        problems.append("Missing results/density_results.csv")
    else:
        with density_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 5:
            problems.append(f"density_results.csv expected 5 density rows, found {len(rows)}.")
        for row in rows:
            if row.get("n_drivers") != "5000":
                problems.append(
                    f"density_results.csv density={row.get('density_pct')} has n_drivers={row.get('n_drivers')} instead of 5000."
                )

    model_path = ROOT / "results" / "model_comparison.csv"
    if model_path.exists():
        tuned = _load_single_row_csv(model_path, "model", "LightGBM (tuned)")
        print(
            "Model comparison anchor:",
            f"R2={float(tuned['r2']):.4f}",
            f"RMSE=${float(tuned['rmse']):.2f}",
            f"rank-1={float(tuned['rank_acc']):.1%}",
        )
    else:
        problems.append("Missing results/model_comparison.csv")

    ablation_path = ROOT / "results" / "ablation_results.csv"
    if ablation_path.exists():
        ablation = _load_single_row_csv(ablation_path, "experiment", "All features")
        print(
            "Ablation anchor:",
            f"R2={float(ablation['r2']):.4f}",
            f"RMSE=${float(ablation['rmse']):.2f}",
            f"rank-1={float(ablation['rank_acc']):.1%}",
        )
    else:
        problems.append("Missing results/ablation_results.csv")

    tex_path = ROOT / "paper" / "main_full.tex"
    if tex_path.exists():
        tex = _read_text(tex_path)
        cite_keys: set[str] = set()
        for match in re.findall(r"\\cite\{([^}]*)\}", tex):
            for key in match.split(","):
                key = key.strip()
                if key:
                    cite_keys.add(key)
        bib_keys = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
        missing_bib = sorted(cite_keys - bib_keys)
        if missing_bib:
            problems.append(f"Undefined bibliography keys in main_full.tex: {', '.join(missing_bib)}")
    else:
        problems.append("Missing paper/main_full.tex")

    if problems:
        print("Consistency check FAILED:\n")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Consistency check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
