"""Locked-test statistics for the frozen FINAL_V1 CREA experiment.

The module is deliberately post-hoc.  It reads completed prediction artifacts,
checks the causal aggregation contract, and produces dataset-equal summaries
and paired chronological block-bootstrap intervals.  It never selects a
checkpoint, eta, gate, dataset, or seed from test outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_DATASETS = {"pems03", "pems04", "pems07", "pems08", "metr_la"}
EXPECTED_SEEDS = (42, 3407, 101, 2026, 777)
METHODS = ("causal_hedge", "static_mix", "loss_gate", "all_only", "prior_only")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_array(root: Path, split: str, stem: str) -> np.ndarray:
    path = root / "predictions" / split / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _exact_raw_target(
    report_root: Path,
    run_root: Path,
    metrics: dict[str, Any],
    saved_target: np.ndarray,
) -> np.ndarray:
    """Reconstruct the exact raw-scale targets used before float32 inverse rounding.

    Exact zeros can become residual values of roughly 1e-5 after a float32
    standardize/inverse-transform round trip.  The traffic MAPE convention masks
    raw ground-truth zeros, so its mask must be formed from the raw series rather
    than from those rounded residuals.
    """
    series_path = report_root.parent.parent / "data" / "processed" / str(metrics["dataset"]) / "series.npz"
    if not series_path.exists():
        raise FileNotFoundError(series_path)
    with np.load(series_path, allow_pickle=False) as archive:
        raw = np.asarray(archive["raw_values"], dtype=np.float64)
        observed = np.asarray(archive["observed_mask"], dtype=bool)
    train_end = int(metrics["split_boundaries"]["forecaster_train"][1])
    frame = pd.DataFrame(np.where(observed, raw, np.nan)).ffill()
    medians = pd.DataFrame(np.where(observed[:train_end], raw[:train_end], np.nan)).median(axis=0)
    frame = frame.fillna(medians)
    imputed = frame.to_numpy(dtype=np.float64)
    origins = _load_array(run_root, "test", "origin_indices").astype(np.int64)
    horizon = int(saved_target.shape[1])
    exact = imputed[origins[:, None] + np.arange(horizon, dtype=np.int64)[None, :]]
    if exact.shape != saved_target.shape:
        raise ValueError(f"Raw and saved target shapes differ: {exact.shape} != {saved_target.shape}")
    if not np.allclose(exact, saved_target, rtol=0.0, atol=1e-3):
        difference = float(np.max(np.abs(exact - saved_target)))
        raise ValueError(f"Raw and saved targets are misaligned; max difference={difference}")
    return exact


def _block_indices(
    size: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample chronological origins with circular moving blocks."""
    if int(size) <= 0:
        raise ValueError("Cannot bootstrap an empty origin vector")
    if int(block_length) <= 0:
        raise ValueError("block_length must be positive")
    indices: list[int] = []
    while len(indices) < int(size):
        start = int(rng.integers(0, int(size)))
        indices.extend((start + offset) % int(size) for offset in range(int(block_length)))
    return np.asarray(indices[: int(size)], dtype=np.int64)


def _metric_rows(
    target: np.ndarray,
    mape_target: np.ndarray,
    predictions: dict[str, np.ndarray],
    horizons: list[int],
    *,
    dataset: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    if target.ndim != 3:
        raise ValueError(f"Expected target [origin,horizon,node], got {target.shape}")
    if mape_target.shape != target.shape:
        raise ValueError("MAPE target and saved target shapes do not match")
    if any(value.shape != target.shape for value in predictions.values()):
        raise ValueError("Prediction and target shapes do not match")
    rows: list[dict[str, Any]] = []
    comparison_errors: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        comparator: {} for comparator in ("over_static_mix", "over_loss_gate")
    }
    for horizon in horizons:
        index = int(horizon) - 1
        y = np.asarray(target[:, index, :], dtype=np.float64)
        y_mape = np.asarray(mape_target[:, index, :], dtype=np.float64)
        errors: dict[str, np.ndarray] = {}
        for name, prediction in predictions.items():
            errors[name] = np.asarray(prediction[:, index, :], dtype=np.float64) - y
        mae = {name: float(np.mean(np.abs(value))) for name, value in errors.items()}
        rmse = {name: float(np.sqrt(np.mean(value * value))) for name, value in errors.items()}
        valid = np.abs(y_mape) > 1e-5
        if not np.any(valid):
            raise ValueError(f"No nonzero MAPE targets for {dataset}, seed={seed}, h={horizon}")
        mape = {
            name: float(100.0 * np.mean(np.abs(predictions[name][:, index, :][valid] - y_mape[valid]) / np.abs(y_mape[valid])))
            for name in predictions
        }
        gains = {
            "over_static_mix": (mae["static_mix"] - mae["causal_hedge"])
            / max(mae["static_mix"], 1e-12),
            "over_loss_gate": (mae["loss_gate"] - mae["causal_hedge"])
            / max(mae["loss_gate"], 1e-12),
        }
        for comparator, baseline_name in (
            ("over_static_mix", "static_mix"),
            ("over_loss_gate", "loss_gate"),
        ):
            comparison_errors[comparator][str(horizon)] = (
                # Aggregate nodes within each chronological origin first.
                # Resampling these 1-D origin MAEs is algebraically identical
                # to resampling the full [origin,node] absolute-error matrix,
                # while avoiding huge bootstrap copies for PEMS07.
                np.mean(np.abs(errors[baseline_name]), axis=1),
                np.mean(np.abs(errors["causal_hedge"]), axis=1),
            )
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": int(seed),
            "horizon": int(horizon),
            "test_origins": int(target.shape[0]),
            "mae_causal_hedge": mae["causal_hedge"],
            "mae_static_mix": mae["static_mix"],
            "mae_loss_gate": mae["loss_gate"],
            "rmse_causal_hedge": rmse["causal_hedge"],
            "rmse_static_mix": rmse["static_mix"],
            "rmse_loss_gate": rmse["loss_gate"],
            "mape_percent_causal_hedge": mape["causal_hedge"],
            "mape_percent_static_mix": mape["static_mix"],
            "mape_percent_loss_gate": mape["loss_gate"],
            "mape_percent_all_only": mape["all_only"],
            "mape_percent_prior_only": mape["prior_only"],
            "relative_gain_over_static_mix": float(gains["over_static_mix"]),
            "relative_gain_over_loss_gate": float(gains["over_loss_gate"]),
        }
        rows.append(row)
    return rows, comparison_errors


def _discover_runs(report_root: Path, seeds: list[int]) -> list[Path]:
    roots: list[Path] = []
    for seed in seeds:
        roots.extend(sorted(report_root.glob(f"runs/E*/*/seed_{int(seed)}")))
    if len(roots) != len(seeds) * len(EXPECTED_DATASETS):
        raise RuntimeError(
            f"FINAL_V1 requires {len(seeds) * len(EXPECTED_DATASETS)} runs, found {len(roots)}"
        )
    datasets = {root.parent.name for root in roots}
    if datasets != EXPECTED_DATASETS:
        raise RuntimeError(f"Unexpected FINAL_V1 dataset set: {sorted(datasets)}")
    for root in roots:
        state = _read_json(root / "state.json")
        if state.get("status") != "complete":
            raise RuntimeError(f"Run is not complete: {root}")
        causal = _read_json(root / "causal_aggregation.json")
        if causal.get("test_selection_use") != "none":
            raise RuntimeError(f"Test was used for selection: {root}")
    return roots


def _bootstrap_macro(
    comparison_errors: dict[tuple[str, int, str, int], tuple[np.ndarray, np.ndarray]],
    datasets: list[str],
    seeds: list[int],
    horizons: list[int],
    *,
    comparator: str,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(int(seed))
    seed_values = np.asarray(seeds, dtype=np.int64)
    # Resample seed identities once per draw, as in the declared seed-level
    # bootstrap, then vectorize the expensive chronological-origin sampling.
    sampled_seed_positions = rng.integers(
        0, len(seed_values), size=(int(replicates), len(seed_values))
    )
    dataset_draws = np.empty((int(replicates), len(datasets)), dtype=np.float64)
    for dataset_index, dataset in enumerate(datasets):
        per_seed_horizon = np.empty(
            (int(replicates), len(seed_values), len(horizons)), dtype=np.float64
        )
        for seed_index, selected_seed in enumerate(seed_values):
            for horizon_index, horizon in enumerate(horizons):
                errors = comparison_errors[
                    (comparator, int(selected_seed), dataset, int(horizon))
                ]
                baseline_mae, causal_mae = _paired_block_means(
                    errors[0], errors[1], int(replicates), block_length, rng
                )
                per_seed_horizon[:, seed_index, horizon_index] = (
                    baseline_mae - causal_mae
                ) / np.maximum(baseline_mae, 1e-12)
        draw_rows = per_seed_horizon[
            np.arange(int(replicates))[:, None], sampled_seed_positions, :
        ]
        dataset_draws[:, dataset_index] = np.mean(draw_rows, axis=(1, 2))
    draws = np.mean(dataset_draws, axis=1)
    return {
        "estimate": float(np.mean(draws)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "p_one_sided_nonpositive": float((1 + np.sum(draws <= 0.0)) / (len(draws) + 1)),
        "replicates": int(replicates),
        "block_length": int(block_length),
    }


def _ratio_of_aggregate_mae(
    errors: tuple[np.ndarray, np.ndarray], indices: np.ndarray
) -> float:
    """Compute the locked endpoint after paired origin resampling.

    The baseline and CREA errors are resampled with the same chronological
    origin indices, then each aggregate MAE is formed, and only then is the
    relative gain calculated.  This is intentionally not the mean of
    origin-level ratios.
    """
    baseline, causal = errors
    baseline_mae = float(np.mean(np.asarray(baseline)[indices]))
    causal_mae = float(np.mean(np.asarray(causal)[indices]))
    return (baseline_mae - causal_mae) / max(baseline_mae, 1e-12)


def _block_index_matrix(
    size: int,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized circular moving-block indices for all bootstrap draws."""
    if int(size) <= 0:
        raise ValueError("Cannot bootstrap an empty origin vector")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    if int(block_length) <= 0:
        raise ValueError("block_length must be positive")
    blocks = (int(size) + int(block_length) - 1) // int(block_length)
    starts = rng.integers(0, int(size), size=(int(replicates), blocks))
    offsets = np.arange(int(block_length), dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % int(size)
    return indices.reshape(int(replicates), -1)[:, : int(size)]


def _circular_block_sum(values: np.ndarray, block_length: int) -> np.ndarray:
    """Sum every circular block of ``block_length`` in a 1-D vector."""
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if value.size == 0:
        raise ValueError("Cannot bootstrap an empty origin vector")
    extended = np.concatenate([value, value[: int(block_length) - 1]])
    prefix = np.concatenate([[0.0], np.cumsum(extended, dtype=np.float64)])
    return prefix[int(block_length) : int(block_length) + value.size] - prefix[: value.size]


def _paired_block_means(
    baseline: np.ndarray,
    causal: np.ndarray,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired moving-block means without materializing indices."""
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    ours = np.asarray(causal, dtype=np.float64).reshape(-1)
    if base.shape != ours.shape or base.size == 0:
        raise ValueError("Paired bootstrap vectors must be non-empty and aligned")
    full_blocks, remainder = divmod(int(base.size), int(block_length))
    blocks = full_blocks + int(remainder > 0)
    starts = rng.integers(0, base.size, size=(int(replicates), blocks))
    base_sums = np.zeros(int(replicates), dtype=np.float64)
    ours_sums = np.zeros(int(replicates), dtype=np.float64)
    if full_blocks:
        base_full = _circular_block_sum(base, int(block_length))
        ours_full = _circular_block_sum(ours, int(block_length))
        base_sums += np.sum(base_full[starts[:, :full_blocks]], axis=1)
        ours_sums += np.sum(ours_full[starts[:, :full_blocks]], axis=1)
    if remainder:
        base_tail = _circular_block_sum(base, int(remainder))
        ours_tail = _circular_block_sum(ours, int(remainder))
        base_sums += base_tail[starts[:, -1]]
        ours_sums += ours_tail[starts[:, -1]]
    return base_sums / base.size, ours_sums / ours.size


def write_final_v1_stats(
    report_root: Path,
    *,
    seeds: list[int] | None = None,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260815,
    block_length: int = 24,
) -> Path:
    root = report_root.resolve()
    selected_seeds = [int(value) for value in (seeds or EXPECTED_SEEDS)]
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seeds must be unique")
    run_roots = _discover_runs(root, selected_seeds)
    detail_rows: list[dict[str, Any]] = []
    comparison_errors: dict[
        tuple[str, int, str, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    provenance: list[dict[str, Any]] = []
    for run_root in run_roots:
        metrics = _read_json(run_root / "metrics.json")
        dataset = str(metrics["dataset"])
        seed = int(metrics["seed"])
        horizons = [int(value) for value in metrics["report_horizons"]]
        target = _load_array(run_root, "test", "targets")
        mape_target = _exact_raw_target(root, run_root, metrics, target)
        predictions = {
            name: _load_array(run_root, "test", f"final_{name}")
            for name in METHODS
        }
        rows, gains = _metric_rows(
            target,
            mape_target,
            predictions,
            horizons,
            dataset=dataset,
            seed=seed,
        )
        detail_rows.extend(rows)
        for comparator, by_horizon in gains.items():
            for horizon, values in by_horizon.items():
                comparison_errors[(comparator, seed, dataset, int(horizon))] = values
        provenance.append(
            {
                "dataset": dataset,
                "seed": seed,
                "config_sha256": metrics.get("config_sha256"),
                "code_fingerprint": metrics.get("code_fingerprint"),
                "git_commit": metrics.get("git_commit"),
                "metrics_sha256": _sha256(run_root / "metrics.json"),
                "causal_aggregation_sha256": _sha256(run_root / "causal_aggregation.json"),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values(["dataset", "seed", "horizon"])
    datasets = sorted({str(value) for value in detail["dataset"]})
    horizons = sorted({int(value) for value in detail["horizon"]})
    per_seed: list[dict[str, Any]] = []
    for seed in selected_seeds:
        subset = detail[detail["seed"] == int(seed)]
        per_dataset: dict[str, dict[str, float]] = {}
        for dataset in datasets:
            item = subset[subset["dataset"] == dataset]
            per_dataset[dataset] = {
                "over_static_mix": float(item["relative_gain_over_static_mix"].mean()),
                "over_loss_gate": float(item["relative_gain_over_loss_gate"].mean()),
            }
        per_seed.append(
            {
                "seed": int(seed),
                "per_dataset": per_dataset,
                "macro_over_static_mix": float(np.mean([v["over_static_mix"] for v in per_dataset.values()])),
                "macro_over_loss_gate": float(np.mean([v["over_loss_gate"] for v in per_dataset.values()])),
            }
        )
    bootstrap = {
        "over_static_mix": _bootstrap_macro(
            comparison_errors,
            datasets,
            selected_seeds,
            horizons,
            comparator="over_static_mix",
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "over_loss_gate": _bootstrap_macro(
            comparison_errors,
            datasets,
            selected_seeds,
            horizons,
            comparator="over_loss_gate",
            block_length=block_length,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        ),
    }
    summary = {
        "schema": "selective_prior_causal_hedge_final_v1_stats_v2",
        "report_root": str(root),
        "datasets": datasets,
        "seeds": selected_seeds,
        "horizons": horizons,
        "primary_endpoint": "dataset_equal_macro_mean_of_horizon_relative_MAE_gains",
        "secondary_metrics": ["MAE", "RMSE", "MAPE_percent"],
        "mape_definition": "100 * mean(abs(prediction-target)/abs(target)) over exact raw-scale targets with abs(target)>1e-5; secondary only",
        "test_selection_use": "none",
        "full_test_origins": True,
        "per_seed": per_seed,
        "overall_mean": {
            "over_static_mix": float(np.mean([row["macro_over_static_mix"] for row in per_seed])),
            "over_loss_gate": float(np.mean([row["macro_over_loss_gate"] for row in per_seed])),
        },
        "overall_std": {
            "over_static_mix": float(np.std([row["macro_over_static_mix"] for row in per_seed], ddof=1))
            if len(per_seed) > 1 else 0.0,
            "over_loss_gate": float(np.std([row["macro_over_loss_gate"] for row in per_seed], ddof=1))
            if len(per_seed) > 1 else 0.0,
        },
        "paired_chronological_block_bootstrap": bootstrap,
        "paper_evidence_gate": bool(
            all(row["macro_over_static_mix"] > 0.0 for row in per_seed)
            and all(row["macro_over_loss_gate"] > 0.0 for row in per_seed)
        ),
        "provenance": provenance,
    }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "final_v1_per_dataset_horizon_seed.csv", index=False)
    pd.DataFrame(per_seed).to_json(output / "final_v1_per_seed_summary.json", orient="records", indent=2)
    (output / "final_v1_stats.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output / "final_v1_stats.json"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--seeds", default="42,3407,101,2026,777")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--block-length", type=int, default=24)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    print(
        write_final_v1_stats(
            args.report_root,
            seeds=seeds,
            bootstrap_replicates=args.bootstrap,
            block_length=args.block_length,
        )
    )


if __name__ == "__main__":
    main()
