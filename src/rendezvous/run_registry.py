from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

RUNS_DIRNAME = "runs"
MANIFEST_NAME = "manifest.json"

DRIVER_RAW_ROLES = {
    "driver_outcomes": "rendezvous_driver_outcomes{suffix}.csv",
    "route_evaluations": "rendezvous_route_evaluations{suffix}.csv",
    "route_opportunities": "rendezvous_route_opportunities{suffix}.csv",
}

DRIVER_DERIVED_ROLES = {
    "driver_summary": "rendezvous_driver_summary{suffix}.csv",
    "config": "rendezvous_config{suffix}.json",
    "driver_run_stats": "rendezvous_driver_run_stats{suffix}.json",
}

DISPATCH_RAW_ROLES = {
    "dispatch_outcomes": "rendezvous_dispatch_outcomes{suffix}.csv",
    "dispatch_summary": "rendezvous_dispatch_summary{suffix}.csv",
}

DISPATCH_DERIVED_ROLES = {
    "dispatch_policy_summary": "rendezvous_dispatch_policy_summary{suffix}.csv",
    "config": "rendezvous_dispatch_config{suffix}.json",
    "dispatch_run_stats": "rendezvous_dispatch_run_stats{suffix}.json",
}


def create_run_artifact_dir(
    results_dir: Path,
    *,
    run_kind: str,
    domain: str,
    scenario_name: str,
    tag: str = "",
) -> tuple[str, Path]:
    run_id = build_run_id(run_kind=run_kind, domain=domain, scenario_name=scenario_name, tag=tag)
    run_dir = results_dir / RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def build_run_id(*, run_kind: str, domain: str, scenario_name: str, tag: str = "") -> str:
    parts = [run_kind, domain, scenario_name]
    if tag:
        parts.append(tag)
    slug = "-".join(_slugify(part) for part in parts if part)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug}-{timestamp}-{uuid4().hex[:8]}"


def write_run_manifest(
    *,
    results_dir: Path,
    run_dir: Path,
    run_id: str,
    run_kind: str,
    domain: str,
    scenario_name: str,
    tag: str,
    config: dict[str, object],
    cli_args: dict[str, object],
    raw_outputs: dict[str, Path],
    derived_outputs: dict[str, Path],
    metadata: dict[str, object] | None = None,
) -> Path:
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": run_kind,
        "domain": domain,
        "scenario_name": scenario_name,
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "cli_args": cli_args,
        "raw_outputs": _relative_path_map(results_dir, raw_outputs),
        "derived_outputs": _relative_path_map(results_dir, derived_outputs),
        "metadata": metadata or {},
    }
    manifest_path = run_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def list_run_manifests(results_dir: Path) -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    runs_dir = results_dir / RUNS_DIRNAME
    if not runs_dir.exists():
        return manifests
    for manifest_path in sorted(runs_dir.glob(f"*/{MANIFEST_NAME}")):
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    return manifests


def has_registered_runs(results_dir: Path) -> bool:
    return any((results_dir / RUNS_DIRNAME).glob(f"*/{MANIFEST_NAME}"))


def registered_file_paths(results_dir: Path, *, role: str, run_kind: str | None = None) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for manifest in list_run_manifests(results_dir):
        manifest_kind = str(manifest.get("run_kind", ""))
        if run_kind is not None and manifest_kind != run_kind:
            continue
        for section in ("raw_outputs", "derived_outputs"):
            outputs = manifest.get(section, {})
            if not isinstance(outputs, dict):
                continue
            relative = outputs.get(role)
            if not relative:
                continue
            path = results_dir / str(relative)
            if path in seen or not path.exists():
                continue
            seen.add(path)
            paths.append(path)
    return paths


def backfill_legacy_runs(results_dir: Path) -> list[str]:
    created_run_ids: list[str] = []
    results_dir.mkdir(parents=True, exist_ok=True)
    created_run_ids.extend(
        _backfill_legacy_kind(
            results_dir,
            run_kind="driver",
            primary_pattern="rendezvous_driver_outcomes*.csv",
            raw_roles=DRIVER_RAW_ROLES,
            derived_roles=DRIVER_DERIVED_ROLES,
        )
    )
    created_run_ids.extend(
        _backfill_legacy_kind(
            results_dir,
            run_kind="dispatch",
            primary_pattern="rendezvous_dispatch_summary*.csv",
            raw_roles=DISPATCH_RAW_ROLES,
            derived_roles=DISPATCH_DERIVED_ROLES,
        )
    )
    return created_run_ids


def _backfill_legacy_kind(
    results_dir: Path,
    *,
    run_kind: str,
    primary_pattern: str,
    raw_roles: dict[str, str],
    derived_roles: dict[str, str],
) -> list[str]:
    created: list[str] = []
    existing_suffixes = {
        str(manifest.get("metadata", {}).get("legacy_suffix", ""))
        for manifest in list_run_manifests(results_dir)
        if str(manifest.get("run_kind", "")) == run_kind and manifest.get("metadata", {}).get("legacy_suffix") is not None
    }
    for primary_path in sorted(results_dir.glob(primary_pattern)):
        suffix = _extract_suffix(primary_path.stem, primary_pattern.replace("*.csv", ""))
        if suffix in existing_suffixes:
            continue
        config_name = derived_roles["config"].format(suffix=suffix)
        config_path = results_dir / config_name
        config = _read_json(config_path)
        if not config:
            continue
        domain = str(config.get("domain", "unknown"))
        scenario_name = str(config.get("scenario_name", "unspecified"))
        tag = suffix[1:] if suffix.startswith("_") else suffix
        run_id = f"legacy-{_slugify(run_kind)}-{_slugify(domain)}-{_slugify(scenario_name)}"
        if tag:
            run_id = f"{run_id}-{_slugify(tag)}"
        run_dir = results_dir / RUNS_DIRNAME / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        raw_outputs = _copy_role_files(results_dir, run_dir, raw_roles, suffix)
        derived_outputs = _copy_role_files(results_dir, run_dir, derived_roles, suffix)
        stats_role = "driver_run_stats" if run_kind == "driver" else "dispatch_run_stats"
        metadata = _read_json(results_dir / derived_roles[stats_role].format(suffix=suffix))
        metadata["legacy_suffix"] = suffix
        write_run_manifest(
            results_dir=results_dir,
            run_dir=run_dir,
            run_id=run_id,
            run_kind=run_kind,
            domain=domain,
            scenario_name=scenario_name,
            tag=tag,
            config=config,
            cli_args={},
            raw_outputs=raw_outputs,
            derived_outputs=derived_outputs,
            metadata=metadata,
        )
        created.append(run_id)
    return created


def _copy_role_files(
    results_dir: Path,
    run_dir: Path,
    roles: dict[str, str],
    suffix: str,
) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for role, template in roles.items():
        source = results_dir / template.format(suffix=suffix)
        if not source.exists():
            continue
        destination = run_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        copied[role] = destination
    return copied


def _relative_path_map(results_dir: Path, path_map: dict[str, Path]) -> dict[str, str]:
    return {
        key: value.relative_to(results_dir).as_posix()
        for key, value in path_map.items()
        if value.exists()
    }


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_suffix(stem: str, prefix: str) -> str:
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return ""


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or "run"
