"""Evaluate validation-selected static mixtures of frozen CREA predictors.

Author: Jia Rong. Loads A/R checkpoints and writes per-node mixing, dense
global mixing, and oracle results to a separate analysis directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mae_at(target: np.ndarray, prediction: np.ndarray, horizon_index: int) -> float:
    return float(np.mean(np.abs(np.asarray(prediction)[:, horizon_index, :] - np.asarray(target)[:, horizon_index, :])))


def dense_alpha_losses(
    target: np.ndarray,
    pred_a: np.ndarray,
    pred_r: np.ndarray,
    grid: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [grid,node] and [grid] validation MAE without test access."""
    y = np.asarray(target, dtype=np.float32)
    a = np.asarray(pred_a, dtype=np.float32)
    r = np.asarray(pred_r, dtype=np.float32)
    if y.shape != a.shape or y.shape != r.shape or y.ndim != 2:
        raise ValueError("Expected aligned [origin,node] arrays")
    node_losses = np.empty((len(grid), y.shape[1]), dtype=np.float64)
    global_losses = np.empty(len(grid), dtype=np.float64)
    delta = r - a
    for start in range(0, len(grid), max(1, int(chunk_size))):
        stop = min(len(grid), start + max(1, int(chunk_size)))
        alpha = grid[start:stop].astype(np.float32)[:, None, None]
        mixed = a[None, :, :] + alpha * delta[None, :, :]
        errors = np.abs(mixed - y[None, :, :])
        node_losses[start:stop] = errors.mean(axis=1, dtype=np.float64)
        global_losses[start:stop] = errors.mean(axis=(1, 2), dtype=np.float64)
    return node_losses, global_losses


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> str:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint has no model state: {path}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return sha256_file(path)


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root))

    from rpv_experiments.config import load_config, units_for_stage, resolved_data_root, current_git_commit  # noqa: E402
    from rpv_experiments.data import fixed_masks, load_bundle  # noqa: E402
    from rpv_experiments.runner import _datasets_for_unit  # noqa: E402
    from rpv_experiments.training import build_forecaster, predict_raw  # noqa: E402

    config_path = source_root / args.config
    cfg, resolved_config_path = load_config(config_path)
    units = [unit for unit in units_for_stage(cfg, "formal") if str(unit["dataset"]) == args.dataset]
    if len(units) != 1:
        raise ValueError(f"Expected one formal unit for {args.dataset}, found {len(units)}")
    unit = units[0]
    seed = int(args.seed)
    output_root = Path(args.output_root).resolve()
    run_output = output_root / "runs" / str(unit["experiment_id"]) / args.dataset / f"seed_{seed}"
    complete_path = run_output / "complete.json"
    if complete_path.exists() and not args.force:
        return json.loads(complete_path.read_text(encoding="utf-8"))

    torch.set_num_threads(max(1, int(args.threads)))
    torch.set_num_interop_threads(max(1, min(int(args.threads), 4)))
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data_root = resolved_data_root(cfg)
    bundle = load_bundle(data_root, args.dataset, cfg["split"])
    datasets = _datasets_for_unit(cfg, unit, bundle)
    masks = fixed_masks(bundle, device=device)
    training = dict(cfg["training"])
    training["micro_batch_size"] = int(training.get("micro_batch_size", training.get("batch_size", 64)))
    training["num_workers"] = 0
    report_root = Path(args.original_report_root).resolve()
    checkpoint_dir = report_root / "runs" / str(unit["experiment_id"]) / args.dataset / f"seed_{seed}" / "checkpoints"
    checkpoint_paths = {
        "all": checkpoint_dir / "all_only.pt",
        "prior": checkpoint_dir / "prior_only.pt",
    }
    for path in checkpoint_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    models: dict[str, torch.nn.Module] = {}
    checkpoint_sha: dict[str, str] = {}
    for route, checkpoint_path in checkpoint_paths.items():
        model = build_forecaster(bundle, int(unit["history"]), int(unit["max_horizon"]), training)
        checkpoint_sha[route] = load_checkpoint(model, checkpoint_path, device)
        models[route] = model

    grid = np.linspace(0.0, 1.0, int(round(1.0 / float(args.alpha_step))) + 1, dtype=np.float64)
    split_results: dict[str, dict[str, Any]] = {}
    posthoc_rows: list[dict[str, Any]] = []

    for split in ("gate_validation", "test"):
        targets, pred_a = predict_raw(models["all"], datasets[split], masks["all"], training, device)
        prior_targets, pred_r = predict_raw(models["prior"], datasets[split], masks["prior"], training, device)
        if not np.array_equal(targets, prior_targets):
            raise RuntimeError(f"Target alignment failed for {args.dataset} seed {seed} {split}")
        split_dir = run_output / "predictions" / split
        atomic_save_array(split_dir / "targets.npy", targets.astype(np.float32))
        atomic_save_array(split_dir / "pred_all.npy", pred_a.astype(np.float32))
        atomic_save_array(split_dir / "pred_prior.npy", pred_r.astype(np.float32))
        atomic_save_array(split_dir / "origin_indices.npy", datasets[split].starts.astype(np.int64))
        split_results[split] = {
            "shape": list(targets.shape),
            "target_sha256": hashlib.sha256(np.ascontiguousarray(targets).tobytes()).hexdigest(),
            "origin_sha256": hashlib.sha256(np.ascontiguousarray(datasets[split].starts).tobytes()).hexdigest(),
        }

        if split == "gate_validation":
            alpha_global = np.empty(pred_a.shape[1], dtype=np.float64)
            alpha_node = np.empty((pred_a.shape[1], pred_a.shape[2]), dtype=np.float64)
            validation_losses_global: list[np.ndarray] = []
            validation_losses_node: list[np.ndarray] = []
            for h in range(pred_a.shape[1]):
                node_losses, global_losses = dense_alpha_losses(
                    targets[:, h, :], pred_a[:, h, :], pred_r[:, h, :], grid, int(args.grid_chunk)
                )
                validation_losses_node.append(node_losses)
                validation_losses_global.append(global_losses)
                alpha_global[h] = grid[int(np.argmin(global_losses))]
                alpha_node[h] = grid[np.argmin(node_losses, axis=0)]
            validation_losses_node_array = np.stack(validation_losses_node, axis=0)
            validation_losses_global_array = np.stack(validation_losses_global, axis=0)
            posthoc_dir = run_output / "posthoc"
            atomic_save_array(posthoc_dir / "alpha_grid.npy", grid)
            atomic_save_array(posthoc_dir / "alpha_global_by_horizon.npy", alpha_global)
            atomic_save_array(posthoc_dir / "alpha_node_by_horizon.npy", alpha_node)
            atomic_save_array(posthoc_dir / "validation_loss_global_by_horizon_grid.npy", validation_losses_global_array)
            atomic_save_array(posthoc_dir / "validation_loss_node_by_horizon_grid.npy", validation_losses_node_array)
            alpha_selection = {
                "grid_step": float(args.alpha_step),
                "grid_size": int(len(grid)),
                "selection_split": "gate_validation",
                "test_selection_used": False,
                "alpha_global_by_horizon": alpha_global.tolist(),
                "node_alpha_min": float(alpha_node.min()),
                "node_alpha_max": float(alpha_node.max()),
                "node_alpha_mean": float(alpha_node.mean()),
            }
            atomic_save_json(posthoc_dir / "alpha_selection.json", alpha_selection)

        else:
            posthoc_dir = run_output / "posthoc"
            alpha_global = np.load(posthoc_dir / "alpha_global_by_horizon.npy", allow_pickle=False)
            alpha_node = np.load(posthoc_dir / "alpha_node_by_horizon.npy", allow_pickle=False)
            global_prediction = pred_a + alpha_global[None, :, None].astype(np.float32) * (pred_r - pred_a)
            node_prediction = pred_a + alpha_node[None, :, :].astype(np.float32) * (pred_r - pred_a)
            oracle_prediction = np.where(
                np.abs(pred_r - targets) < np.abs(pred_a - targets), pred_r, pred_a
            ).astype(np.float32)
            atomic_save_array(posthoc_dir / "test_global_dense_static.npy", global_prediction)
            atomic_save_array(posthoc_dir / "test_per_node_static.npy", node_prediction)
            atomic_save_array(posthoc_dir / "test_pointwise_oracle.npy", oracle_prediction)
            metrics: dict[str, Any] = {
                "selection_split": "gate_validation",
                "test_selection_used": False,
                "horizons": {},
                "full_horizon_mae": {},
            }
            report_horizons = (3, 6, 12)
            for horizon in report_horizons:
                h = horizon - 1
                row = {
                    "horizon": horizon,
                    "all_only_mae": mae_at(targets, pred_a, h),
                    "prior_only_mae": mae_at(targets, pred_r, h),
                    "global_dense_static_mae": mae_at(targets, global_prediction, h),
                    "per_node_static_mae": mae_at(targets, node_prediction, h),
                    "pointwise_oracle_mae": mae_at(targets, oracle_prediction, h),
                }
                row["per_node_gain_over_global"] = float(
                    (row["global_dense_static_mae"] - row["per_node_static_mae"])
                    / max(row["global_dense_static_mae"], 1e-12)
                )
                row["global_gain_over_all"] = float(
                    (row["all_only_mae"] - row["global_dense_static_mae"])
                    / max(row["all_only_mae"], 1e-12)
                )
                metrics["horizons"][str(horizon)] = row
                posthoc_rows.append({"dataset": args.dataset, "seed": seed, **row})
            metrics["full_horizon_mae"] = {
                "all_only": float(np.mean(np.abs(pred_a - targets))),
                "prior_only": float(np.mean(np.abs(pred_r - targets))),
                "global_dense_static": float(np.mean(np.abs(global_prediction - targets))),
                "per_node_static": float(np.mean(np.abs(node_prediction - targets))),
                "pointwise_oracle": float(np.mean(np.abs(oracle_prediction - targets))),
            }
            atomic_save_json(posthoc_dir / "test_metrics.json", metrics)

    result = {
        "schema": "dfsa_replay_per_node_alpha_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": seed,
        "experiment_id": str(unit["experiment_id"]),
        "source_root": str(source_root),
        "source_commit": current_git_commit(),
        "config": str(resolved_config_path),
        "config_sha256": sha256_file(resolved_config_path),
        "original_report_root": str(report_root),
        "checkpoint_paths": {key: str(value) for key, value in checkpoint_paths.items()},
        "checkpoint_sha256": checkpoint_sha,
        "data_root": str(data_root),
        "selection_split": "gate_validation",
        "test_selection_used": False,
        "alpha_grid_step": float(args.alpha_step),
        "alpha_grid_size": int(len(grid)),
        "route_inference": "independent frozen all_only=A and prior_only=R checkpoints",
        "postprocessing": "global-per-horizon and node-per-horizon convex alpha selected on gate_validation",
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "host": platform.node(),
        "split_results": split_results,
        "status": "complete",
    }
    atomic_save_json(complete_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", default="configs/experiment_final_v1.yaml")
    parser.add_argument("--original-report-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--grid-chunk", type=int, default=8)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_one(args)
    print(json.dumps({"status": result.get("status"), "dataset": result.get("dataset"), "seed": result.get("seed")}, sort_keys=True))


if __name__ == "__main__":
    main()
