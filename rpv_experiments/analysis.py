from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpv_data.utils import atomic_json, sha256_file

from .config import resolved_output_root
from .provenance import validate_matrix_versions


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _scientific_family(experiment_id: str, comparator: str) -> str:
    if comparator == "random_control_gate":
        return "random_control_ablation"
    number = int(str(experiment_id).removeprefix("E"))
    if 1 <= number <= 3:
        return "synthetic_mechanism"
    if 4 <= number <= 9:
        return "real_primary"
    if number == 10:
        return "locked_confirmation"
    raise ValueError(f"Unknown experiment family: {experiment_id}")


def apply_holm_by_claim_family(rows: list[dict[str, Any]]) -> None:
    """Correct only within predeclared scientific claim and comparator families."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        family = _scientific_family(str(row["experiment_id"]), str(row["comparator"]))
        row["holm_family"] = family
        groups[(family, str(row["comparator"]))].append(index)
    for indices in groups.values():
        adjusted = holm_adjust(
            [float(rows[index]["p_one_sided_no_improvement"]) for index in indices]
        )
        for index, value in zip(indices, adjusted):
            rows[index]["holm_adjusted_p"] = float(value)
            rows[index]["significant_improvement"] = bool(
                float(rows[index]["ci95_low"]) > 0.0 and value < 0.05
            )


def _circular_block_mean(values: np.ndarray, block_length: int, rng: np.random.Generator) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Bootstrap values must be a nonempty vector")
    indices = []
    while len(indices) < len(array):
        start = int(rng.integers(0, len(array)))
        indices.extend((start + offset) % len(array) for offset in range(int(block_length)))
    sampled = array[np.asarray(indices[: len(array)], dtype=np.int64)]
    finite = sampled[np.isfinite(sampled)]
    return float(np.mean(finite)) if finite.size else float("nan")


def paired_seed_block_bootstrap(
    differences_by_seed: list[np.ndarray],
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if not differences_by_seed:
        raise ValueError("At least one paired seed is required")
    lengths = {len(np.asarray(values)) for values in differences_by_seed}
    if len(lengths) != 1:
        raise ValueError("Paired seed arrays must have equal origin counts")
    point = float(np.nanmean(np.concatenate([np.asarray(value, dtype=np.float64) for value in differences_by_seed])))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled_seed_indices = rng.integers(0, len(differences_by_seed), size=len(differences_by_seed))
        seed_means = [
            _circular_block_mean(differences_by_seed[int(index)], block_length, rng)
            for index in sampled_seed_indices
        ]
        draws[replicate] = np.nanmean(seed_means)
    finite = draws[np.isfinite(draws)]
    if finite.size != len(draws):
        raise RuntimeError("Non-finite bootstrap replicate encountered")
    return {
        "difference_comparator_minus_matched_gate": point,
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "p_one_sided_no_improvement": float((1 + np.sum(finite <= 0.0)) / (len(finite) + 1)),
        "replicates": int(replicates),
        "block_length": int(block_length),
    }


def paired_tail_noninferiority_bootstrap(
    matched_by_seed: list[np.ndarray],
    fixed_by_seed: list[np.ndarray],
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if not matched_by_seed or len(matched_by_seed) != len(fixed_by_seed):
        raise ValueError("Tail bootstrap requires paired matched-gate and fixed-branch seeds")
    lengths = {
        len(np.asarray(values))
        for pair in zip(matched_by_seed, fixed_by_seed)
        for values in pair
    }
    if len(lengths) != 1:
        raise ValueError("Tail arrays must have equal paired origin counts")

    def degradation(matched_values: np.ndarray, fixed_values: np.ndarray) -> float:
        valid = np.isfinite(matched_values) & np.isfinite(fixed_values)
        if not np.any(valid):
            return float("nan")
        matched_mean = float(np.mean(matched_values[valid]))
        fixed_mean = float(np.mean(fixed_values[valid]))
        return (matched_mean - fixed_mean) / max(fixed_mean, 1e-12)

    point = degradation(
        np.concatenate([np.asarray(values, dtype=np.float64) for values in matched_by_seed]),
        np.concatenate([np.asarray(values, dtype=np.float64) for values in fixed_by_seed]),
    )
    if not np.isfinite(point):
        raise RuntimeError("No q95 events were available for the tail endpoint")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    origin_count = next(iter(lengths))
    for replicate in range(int(replicates)):
        sampled_seed_indices = rng.integers(0, len(matched_by_seed), size=len(matched_by_seed))
        sampled_matched: list[np.ndarray] = []
        sampled_fixed: list[np.ndarray] = []
        for seed_index in sampled_seed_indices:
            indices: list[int] = []
            while len(indices) < origin_count:
                start = int(rng.integers(0, origin_count))
                indices.extend((start + offset) % origin_count for offset in range(int(block_length)))
            selected = np.asarray(indices[:origin_count], dtype=np.int64)
            sampled_matched.append(np.asarray(matched_by_seed[int(seed_index)], dtype=np.float64)[selected])
            sampled_fixed.append(np.asarray(fixed_by_seed[int(seed_index)], dtype=np.float64)[selected])
        draws[replicate] = degradation(np.concatenate(sampled_matched), np.concatenate(sampled_fixed))
    finite = draws[np.isfinite(draws)]
    if finite.size != len(draws):
        raise RuntimeError("A tail bootstrap replicate contained no q95 events")
    return {
        "relative_degradation_matched_gate_vs_validation_fixed": point,
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "replicates": int(replicates),
        "block_length": int(block_length),
    }


def _load(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _origin_mae(target: np.ndarray, prediction: np.ndarray, horizon: int) -> np.ndarray:
    index = int(horizon) - 1
    return np.mean(np.abs(np.asarray(prediction[:, index, :], dtype=np.float64) - np.asarray(target[:, index, :], dtype=np.float64)), axis=1)


def _origin_pooled_horizon_mae(
    target: np.ndarray,
    prediction: np.ndarray,
    horizons: list[int],
) -> np.ndarray:
    indices = np.asarray([int(horizon) - 1 for horizon in horizons], dtype=np.int64)
    y = np.asarray(target[:, indices, :], dtype=np.float64)
    pred = np.asarray(prediction[:, indices, :], dtype=np.float64)
    return np.mean(np.abs(pred - y), axis=(1, 2))


def _origin_tail_mae(
    target: np.ndarray,
    prediction: np.ndarray,
    horizon: int,
    q95_thresholds: np.ndarray,
) -> np.ndarray:
    index = int(horizon) - 1
    y = np.asarray(target[:, index, :], dtype=np.float64)
    pred = np.asarray(prediction[:, index, :], dtype=np.float64)
    event = y >= np.asarray(q95_thresholds, dtype=np.float64).reshape(1, -1)
    count = event.sum(axis=1)
    summed = np.where(event, np.abs(pred - y), 0.0).sum(axis=1)
    return np.divide(
        summed,
        count,
        out=np.full(len(y), np.nan, dtype=np.float64),
        where=count > 0,
    )


def analyze_runs(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    replicates: int | None = None,
    bootstrap_seed: int = 20260814,
) -> Path:
    root = resolved_output_root(cfg)
    if str(cfg.get("mode")) == "formal":
        if cfg.get("schema") == "selective_prior_injection_v2":
            from .decision import verify_mechanism_authorization

            verify_mechanism_authorization(cfg, config_path)
        validate_matrix_versions(cfg, config_path, stage="formal")
    replicate_count = int(
        replicates
        if replicates is not None
        else cfg.get("statistics", {}).get("paired_moving_block_bootstrap_replicates", 5000)
    )
    grouped: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for metrics_path in sorted((root / "runs").glob("E*/*/seed_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for horizon in metrics["horizons"]:
            grouped[(metrics["experiment_id"], metrics["dataset"], int(horizon))].append(metrics_path.parent)
    if not grouped:
        raise FileNotFoundError(f"No complete runs under {root / 'runs'}")
    statistics = cfg.get("statistics", {})
    comparisons = tuple(
        dict.fromkeys(
            [
                *statistics.get("primary_comparators", ["static_mix", "loss_gate"]),
                *statistics.get("ablation_comparators", ["random_control_gate"]),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for (experiment_id, dataset, horizon), run_dirs in sorted(grouped.items()):
        run_dirs = sorted(run_dirs, key=lambda path: int(path.name.removeprefix("seed_")))
        target_paths = [path / "predictions" / "test" / "final_target.npy" for path in run_dirs]
        matched_paths = [path / "predictions" / "test" / "final_matched_control_gate.npy" for path in run_dirs]
        for comparator in comparisons:
            comparator_paths = [path / "predictions" / "test" / f"final_{comparator}.npy" for path in run_dirs]
            required = target_paths + matched_paths + comparator_paths
            if any(not path.exists() for path in required):
                continue
            differences = [
                _origin_mae(_load(target_path), _load(comparator_path), horizon)
                - _origin_mae(_load(target_path), _load(matched_path), horizon)
                for target_path, matched_path, comparator_path in zip(target_paths, matched_paths, comparator_paths)
            ]
            matched_mae = float(
                np.mean([
                    np.mean(_origin_mae(_load(target_path), _load(matched_path), horizon))
                    for target_path, matched_path in zip(target_paths, matched_paths)
                ])
            )
            comparator_mae = float(
                np.mean([
                    np.mean(_origin_mae(_load(target_path), _load(comparator_path), horizon))
                    for target_path, comparator_path in zip(target_paths, comparator_paths)
                ])
            )
            block_length = max(24, int(horizon))
            result = paired_seed_block_bootstrap(
                differences,
                block_length=block_length,
                replicates=replicate_count,
                seed=int(bootstrap_seed + len(rows) * 104729),
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "comparator": comparator,
                    "seeds": len(run_dirs),
                    "matched_gate_mae": matched_mae,
                    "comparator_mae": comparator_mae,
                    "relative_improvement": (comparator_mae - matched_mae) / max(comparator_mae, 1e-12),
                    **result,
                }
            )
        q95_paths = [path / "q95_thresholds.npy" for path in run_dirs]
        if all(path.exists() for path in target_paths + matched_paths + q95_paths):
            selected_fixed = []
            for run_dir in run_dirs:
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                branch = str(
                    metrics["validation_selected_fixed_branch_by_horizon"].get(str(horizon))
                )
                if branch not in {"all_only", "prior_only"}:
                    raise ValueError(
                        f"Invalid validation-selected fixed branch at horizon {horizon}: {branch}"
                    )
                selected_fixed.append(run_dir / "predictions" / "test" / f"final_{branch}.npy")
            if all(path.exists() for path in selected_fixed):
                matched_tail = [
                    _origin_tail_mae(_load(target), _load(matched), horizon, _load(q95))
                    for target, matched, q95 in zip(target_paths, matched_paths, q95_paths)
                ]
                fixed_tail = [
                    _origin_tail_mae(_load(target), _load(fixed), horizon, _load(q95))
                    for target, fixed, q95 in zip(target_paths, selected_fixed, q95_paths)
                ]
                tail_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "seeds": len(run_dirs),
                        **paired_tail_noninferiority_bootstrap(
                            matched_tail,
                            fixed_tail,
                            block_length=max(24, int(horizon)),
                            replicates=replicate_count,
                            seed=int(bootstrap_seed + 1_000_003 + len(tail_rows) * 104729),
                        ),
                    }
                )
    if not rows:
        raise RuntimeError("No complete paired prediction artifacts were found")
    apply_holm_by_claim_family(rows)
    secondary_frame = pd.DataFrame(rows)
    unit_groups: dict[tuple[str, str], list[Path]] = {}
    for (experiment_id, dataset, _), run_dirs in grouped.items():
        key = (experiment_id, dataset)
        normalized = sorted(run_dirs, key=lambda path: int(path.name.removeprefix("seed_")))
        if key in unit_groups and unit_groups[key] != normalized:
            raise RuntimeError(f"Inconsistent seed runs across horizons for {experiment_id}/{dataset}")
        unit_groups[key] = normalized
    primary_rows: list[dict[str, Any]] = []
    for (experiment_id, dataset), run_dirs in sorted(unit_groups.items()):
        first_metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
        horizons = sorted(int(value) for value in first_metrics["horizons"])
        target_paths = [path / "predictions" / "test" / "final_target.npy" for path in run_dirs]
        matched_paths = [
            path / "predictions" / "test" / "final_matched_control_gate.npy"
            for path in run_dirs
        ]
        for comparator in comparisons:
            comparator_paths = [
                path / "predictions" / "test" / f"final_{comparator}.npy"
                for path in run_dirs
            ]
            if any(
                not path.exists()
                for path in [*target_paths, *matched_paths, *comparator_paths]
            ):
                continue
            matched_origin = [
                _origin_pooled_horizon_mae(_load(target), _load(matched), horizons)
                for target, matched in zip(target_paths, matched_paths)
            ]
            comparator_origin = [
                _origin_pooled_horizon_mae(_load(target), _load(compared), horizons)
                for target, compared in zip(target_paths, comparator_paths)
            ]
            differences = [
                compared - matched
                for compared, matched in zip(comparator_origin, matched_origin)
            ]
            matched_mae = float(np.mean([np.mean(values) for values in matched_origin]))
            comparator_mae = float(np.mean([np.mean(values) for values in comparator_origin]))
            result = paired_seed_block_bootstrap(
                differences,
                block_length=max(24, max(horizons)),
                replicates=replicate_count,
                seed=int(bootstrap_seed + 2_000_003 + len(primary_rows) * 104729),
            )
            primary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset": dataset,
                    "pooled_horizons": ",".join(str(value) for value in horizons),
                    "comparator": comparator,
                    "seeds": len(run_dirs),
                    "matched_gate_mae": matched_mae,
                    "comparator_mae": comparator_mae,
                    "relative_improvement": (
                        comparator_mae - matched_mae
                    )
                    / max(comparator_mae, 1e-12),
                    **result,
                }
            )
    if not primary_rows:
        raise RuntimeError("No complete unit-level primary comparisons were found")
    apply_holm_by_claim_family(primary_rows)
    primary_frame = pd.DataFrame(primary_rows)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "paired_comparisons.csv"
    primary_frame.to_csv(csv_path, index=False)
    secondary_path = output / "secondary_horizon_comparisons.csv"
    secondary_frame.to_csv(secondary_path, index=False)
    tail_path = output / "tail_noninferiority.csv"
    pd.DataFrame(tail_rows).to_csv(tail_path, index=False)
    atomic_json(
        output / "analysis_manifest.json",
        {
            "schema": (
                "selective_prior_injection_analysis_v2"
                if cfg.get("schema") == "selective_prior_injection_v2"
                else "selective_prior_injection_analysis_v1"
            ),
            "primary_method": "matched_control_gate",
            "comparators": list(comparisons),
            "bootstrap_replicates": replicate_count,
            "block_length_policy": "max(24, forecast_horizon) forecast origins",
            "familywise_correction": (
                "Holm separately within real-primary, synthetic-mechanism, "
                "locked-confirmation, and bounded random-ablation claim families, "
                "and separately for each comparator"
            ),
            "tail_test": "IUT: every real cell upper 95% CI must be below the frozen relative margin",
            "primary_endpoint": "dataset-unit MAE pooled equally across predeclared report horizons",
            "primary_rows": len(primary_frame),
            "paired_comparisons_sha256": sha256_file(csv_path),
            "secondary_horizon_rows": len(secondary_frame),
            "secondary_horizon_comparisons_sha256": sha256_file(secondary_path),
            "tail_rows": len(tail_rows),
            "tail_noninferiority_sha256": sha256_file(tail_path),
        },
    )
    return output
