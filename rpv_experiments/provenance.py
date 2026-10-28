from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import (
    code_fingerprint,
    config_sha256,
    current_git_commit,
    resolved_output_root,
    seeds_for_stage,
    units_for_stage,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_matrix_versions(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    stage: str = "formal",
    experiment_ids: Iterable[str] | None = None,
    reject_extra_runs: bool = True,
) -> dict[str, Any]:
    """Fail closed unless every selected run uses one semantic experiment version.

    Git commits are retained as audit metadata, but are deliberately excluded
    from scientific identity. Documentation-only commits may therefore coexist
    when code, config, data/control fingerprints, and run fingerprints agree.
    """
    root = resolved_output_root(cfg)
    expected_versions = {
        "code_fingerprint": code_fingerprint(),
        "config_sha256": config_sha256(config_path),
    }
    observed = {key: set() for key in expected_versions}
    observed_git_commits: set[str] = set()
    records: list[dict[str, Any]] = []
    incomplete: list[str] = []
    declared_run_dirs: set[Path] = set()
    selected_experiments = (
        {str(value) for value in experiment_ids}
        if experiment_ids is not None
        else None
    )
    selected_units = [
        unit
        for unit in units_for_stage(cfg, stage)
        if selected_experiments is None
        or str(unit["experiment_id"]) in selected_experiments
    ]
    for unit in selected_units:
        for seed in seeds_for_stage(cfg, stage):
            run_dir = (
                root
                / "runs"
                / str(unit["experiment_id"])
                / str(unit["dataset"])
                / f"seed_{int(seed)}"
            )
            declared_run_dirs.add(run_dir.resolve())
            paths = {
                "state": run_dir / "state.json",
                "metrics": run_dir / "metrics.json",
                "provenance": run_dir / "provenance.json",
            }
            if any(not path.exists() for path in paths.values()):
                incomplete.append(str(run_dir))
                continue
            payloads = {name: _read_json(path) for name, path in paths.items()}
            if payloads["state"].get("status") != "complete":
                incomplete.append(str(run_dir))
                continue
            fingerprints = {payload.get("fingerprint") for payload in payloads.values()}
            if len(fingerprints) != 1 or None in fingerprints:
                raise RuntimeError(f"Run artifact fingerprint mismatch: {run_dir}")
            for field in expected_versions:
                values = {payload.get(field) for payload in payloads.values()}
                if len(values) != 1 or None in values:
                    raise RuntimeError(f"Run artifact {field} mismatch: {run_dir}")
                observed[field].update(values)
            git_values = {payload.get("git_commit") for payload in payloads.values()}
            if len(git_values) != 1 or None in git_values:
                raise RuntimeError(f"Run artifact git_commit record mismatch: {run_dir}")
            observed_git_commits.add(str(next(iter(git_values))))
            records.append(
                {
                    "run_dir": str(run_dir),
                    "state_path": str(paths["state"]),
                    "metrics_path": str(paths["metrics"]),
                    "provenance_path": str(paths["provenance"]),
                    "fingerprint": next(iter(fingerprints)),
                }
            )
    # Do not silently fold stale or ad-hoc runs into a formal analysis.  The
    # declared matrix is the complete universe of admissible runs; an extra
    # metrics file usually means a previous protocol or a wrong config was
    # copied into the current output root.
    if reject_extra_runs:
        extra_run_dirs = []
        for run_dir_path in sorted((root / "runs").glob("E*/*/seed_*")):
            if not run_dir_path.is_dir():
                continue
            run_dir = run_dir_path.resolve()
            if run_dir not in declared_run_dirs:
                extra_run_dirs.append(str(run_dir))
        if extra_run_dirs:
            raise RuntimeError(
                "Formal matrix contains undeclared run artifacts; refusing to mix experiment versions: "
                + ", ".join(extra_run_dirs[:5])
            )
    if incomplete:
        raise RuntimeError(
            f"Formal matrix is incomplete; missing or unfinished {len(incomplete)} declared runs"
        )
    for field, expected in expected_versions.items():
        values = observed[field]
        if len(values) != 1 or next(iter(values)) != expected:
            raise RuntimeError(
                "Formal matrix mixes incompatible experiment versions: "
                f"{field}={sorted(str(value) for value in values)}, current={expected}"
            )
    return {
        "stage": stage,
        "run_count": len(records),
        "git_commits_recorded": sorted(observed_git_commits),
        "validation_git_commit": current_git_commit(),
        **expected_versions,
        "runs": records,
    }
