from __future__ import annotations

import re
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "paper_rendezvous"
TARGET_DIR = ROOT / "paper_rendezvous_overleaf"
MAIN_TEX = SOURCE_DIR / "ieee_submission.tex"


def _parse_includegraphics(tex: str) -> list[Path]:
    matches = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    return [SOURCE_DIR / match for match in matches]


def _parse_bibliographies(tex: str) -> list[Path]:
    bibs: list[Path] = []
    for block in re.findall(r"bibliography\{([^}]+)\}", tex):
        for name in [item.strip() for item in block.split(",") if item.strip()]:
            bibs.append(SOURCE_DIR / f"{name}.bib")
    return bibs


def _parse_inputs(tex: str) -> list[Path]:
    inputs: list[Path] = []
    for command in ("input", "include"):
        for block in re.findall(rf"{command}\{{([^}}]+)\}}", tex):
            candidate = SOURCE_DIR / block
            if candidate.suffix:
                inputs.append(candidate)
            else:
                inputs.append(candidate.with_suffix(".tex"))
    return inputs


def _copy_file(source: Path) -> None:
    relative = source.relative_to(SOURCE_DIR)
    target = TARGET_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".png":
        with Image.open(source) as image:
            image.save(target, optimize=True)
    else:
        shutil.copy2(source, target)


def _write_manifest(files: list[Path]) -> None:
    lines = ["Overleaf package manifest", ""]
    for path in sorted(files):
        relative = path.relative_to(TARGET_DIR)
        lines.append(f"{relative.as_posix()} ({path.stat().st_size} bytes)")
    (TARGET_DIR / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not MAIN_TEX.exists():
        raise SystemExit(f"Missing main TeX file: {MAIN_TEX}")

    tex = MAIN_TEX.read_text(encoding="utf-8")
    dependencies = {MAIN_TEX}
    dependencies.update(_parse_includegraphics(tex))
    dependencies.update(_parse_bibliographies(tex))
    dependencies.update(_parse_inputs(tex))

    missing = [path for path in sorted(dependencies) if not path.exists()]
    if missing:
        message = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing dependency files:\n{message}")

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for dependency in sorted(dependencies):
        _copy_file(dependency)

    copied_files = [TARGET_DIR / path.relative_to(SOURCE_DIR) for path in dependencies]
    _write_manifest(copied_files)
    print(f"Created slim Overleaf package in {TARGET_DIR}")
    print(f"Copied {len(copied_files)} files.")


if __name__ == "__main__":
    main()
