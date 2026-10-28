from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from rpv_data.utils import atomic_json, sha256_file

from .config import (
    code_fingerprint,
    config_sha256,
    current_git_commit,
    jsonable_config,
    lock_path,
    protocol_version,
    resolved_data_root,
    resolved_output_root,
    seeds_for_stage,
    units_for_stage,
    verify_external_lock,
)
from .data import (
    DatasetBundle,
    WindowDataset,
    base_graph_path,
    fixed_masks,
    load_bundle,
    load_controls,
    load_matched_controls,
    node_error_scale,
    window_starts,
)
from .gating import (
    apply_gate_shrinkage_by_horizon,
    choose_best_fixed_by_horizon,
    choose_gate_shrinkage_by_horizon,
    choose_static_alphas_by_horizon,
    gate_metrics,
    mix_with_horizon_alphas,
    prior_value_targets_from_files,
    regression_metrics_numpy,
)
from .provenance import validate_matrix_versions
from .training import (
    gate_probabilities,
    matched_gate_probabilities,
    predict_raw,
    resolve_device,
    train_fixed_forecaster,
    train_matched_prior_gate,
    train_prior_gate,
    train_shared_forecaster,
)

SPLITS = ("forecaster_train", "forecaster_validation", "gate_train", "gate_validation", "test")
GATE_SPLITS = ("gate_train", "gate_validation", "test")
BASE_HEAVY_COMPONENTS = ("all_only", "prior_only", "shared_matched")


def _artifact_schema(cfg: dict[str, Any], stem: str) -> str:
    """Version run artifacts with the active protocol without mutating v1 outputs."""
    return f"{stem}_{protocol_version(cfg)}"


def _uses_random_control_ablation(unit: dict[str, Any]) -> bool:
    return bool(unit.get("random_control_ablation", False))


def _uses_local_context_baseline(unit: dict[str, Any]) -> bool:
    return bool(unit.get("local_context_baseline", False))


def _requires_synthetic_e1_diagnostics(
    cfg: dict[str, Any], unit: dict[str, Any]
) -> bool:
    """Return whether E1 must carry the synthetic mechanism labels.

    FINAL_V1 deliberately uses real PEMS03 as E1.  It reuses the independent
    frozen-expert routing path from v4, but it is not a synthetic mechanism
    protocol and therefore has no ``ground_truth_prior_valid`` or regime
    labels.  Keeping this predicate explicit prevents the final traffic
    matrix from failing *after* all predictions have been generated merely
    because the v3/v4 diagnostic hook was applied to a real dataset.
    """
    return (
        str(cfg.get("schema")) in {
            "selective_prior_injection_v3",
            "selective_prior_injection_v4",
        }
        and str(unit.get("experiment_id")) == "E1"
    )


def _plans_all_only_training(unit: dict[str, Any]) -> bool:
    dataset = str(unit.get("dataset"))
    return not (
        str(unit.get("experiment_id")) == "E3"
        and not dataset.endswith("corruption_000")
    )


def _heavy_components(
    unit: dict[str, Any], *, independent_frozen_experts: bool = False
) -> tuple[str, ...]:
    if independent_frozen_experts:
        # V4 has no shared forecaster: A-only and R-only are the two frozen
        # experts, while every gate/control variant is post-training.
        components = ["all_only", "prior_only"]
        if _uses_local_context_baseline(unit):
            components.insert(0, "local_only")
        return tuple(components)
    components = list(BASE_HEAVY_COMPONENTS)
    if not _plans_all_only_training(unit):
        components.remove("all_only")
    if _uses_local_context_baseline(unit):
        components.insert(0, "local_only")
    if _uses_random_control_ablation(unit):
        components.append("shared_random")
    return tuple(components)


def _control_partitions(cfg: dict[str, Any], total: int) -> tuple[list[int], list[int]]:
    """Return exposure and held-out specificity-probe control indices."""
    method = cfg.get("method", {})
    exposure_count = int(method.get("control_exposure_count", max(1, int(total) - 4)))
    probe_indices = [
        int(index)
        for index in method.get("control_probe_indices", range(exposure_count, int(total)))
    ]
    allow_probe_only = cfg.get("schema") in {
        "selective_prior_injection_v4",
        "selective_prior_injection_v5",
        "selective_prior_injection_final_v1",
    }
    if exposure_count < (0 if allow_probe_only else 1) or exposure_count > int(total):
        lower = "zero" if allow_probe_only else "one"
        raise ValueError(f"control_exposure_count must be between {lower} and the control count")
    if not probe_indices or any(index < 0 or index >= int(total) for index in probe_indices):
        raise ValueError("control_probe_indices contain an unavailable control")
    if set(probe_indices) & set(range(exposure_count)):
        raise ValueError("Specificity probe controls must be held out from forecaster exposure")
    return list(range(exposure_count)), probe_indices


def _stable_int(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _atomic_array(path: Path, array: np.ndarray, dtype: np.dtype | str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=dtype), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_compressed_array(path: Path, array: np.ndarray, dtype: np.dtype | str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=np.asarray(array, dtype=dtype))
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _series_content_fingerprint(bundle: DatasetBundle) -> str:
    """Hash semantic series content rather than NPZ container bytes."""
    digest = hashlib.sha256()
    for name, value in (
        ("raw_values", bundle.raw_values),
        ("observed_mask", bundle.observed_mask),
        ("timestamps", bundle.timestamps),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_content_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _unit_output(cfg: dict[str, Any], unit: dict[str, Any], seed: int) -> Path:
    return (
        resolved_output_root(cfg)
        / "runs"
        / str(unit["experiment_id"])
        / str(unit["dataset"])
        / f"seed_{int(seed)}"
    )


def _selected_units(
    cfg: dict[str, Any],
    stage: str,
    experiment_ids: Iterable[str] | None,
    datasets: Iterable[str] | None,
) -> list[dict[str, Any]]:
    allowed_experiments = {str(value) for value in experiment_ids} if experiment_ids else None
    allowed_datasets = {str(value) for value in datasets} if datasets else None
    units = units_for_stage(cfg, stage)
    return [
        unit
        for unit in units
        if (allowed_experiments is None or unit["experiment_id"] in allowed_experiments)
        and (allowed_datasets is None or unit["dataset"] in allowed_datasets)
    ]


def matrix_plan(
    cfg: dict[str, Any],
    config_path: Path,
    stage: str,
    experiment_ids: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if stage == "external":
        verify_external_lock(cfg, config_path)
    plan: list[dict[str, Any]] = []
    for unit in _selected_units(cfg, stage, experiment_ids, datasets):
        for unit_seed in seeds_for_stage(cfg, stage):
            if seed is not None and int(seed) != int(unit_seed):
                continue
            plan.append(
                {
                    "stage": stage,
                    "experiment_id": unit["experiment_id"],
                    "title": unit["experiment_title"],
                    "dataset": unit["dataset"],
                    "seed": int(unit_seed),
                    "history": int(unit["history"]),
                    "max_horizon": int(unit["max_horizon"]),
                    "report_horizons": [int(value) for value in unit["report_horizons"]],
                    "locked": bool(unit["locked"]),
                    "heavy_forecasters": list(
                        _heavy_components(
                            unit,
                            independent_frozen_experts=(
                                cfg.get("schema")
                                in {
                                    "selective_prior_injection_v4",
                                    "selective_prior_injection_v5",
                                    "selective_prior_injection_final_v1",
                                }
                            ),
                        )
                    ),
                    "output": str(_unit_output(cfg, unit, unit_seed)),
                }
            )
    return plan


def _unit_fingerprint(
    cfg: dict[str, Any],
    config_path: Path,
    unit: dict[str, Any],
    seed: int,
    bundle: DatasetBundle,
    control_provenance: list[dict[str, Any]],
) -> str:
    payload = {
        "schema": _artifact_schema(cfg, "selective_prior_injection_run"),
        "config_sha256": config_sha256(config_path),
        "code_fingerprint": code_fingerprint(),
        "unit": jsonable_config(unit),
        "seed": int(seed),
        "series_content_sha256": _series_content_fingerprint(bundle),
        "prior_graph_sha256": sha256_file(base_graph_path(bundle)),
        "matched_controls": control_provenance,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fixed_component_fingerprint(
    cfg: dict[str, Any],
    unit: dict[str, Any],
    seed: int,
    bundle: DatasetBundle,
    component: str,
) -> str:
    """Fingerprint a fixed branch only from inputs that can affect that branch."""
    payload: dict[str, Any] = {
        "schema": _artifact_schema(cfg, "selective_prior_fixed_component"),
        "component": component,
        "code_fingerprint": code_fingerprint(),
        "series_content_sha256": _series_content_fingerprint(bundle),
        "split": jsonable_config(cfg["split"]),
        "history": int(unit["history"]),
        "max_horizon": int(unit["max_horizon"]),
        "training": jsonable_config(cfg["training"]),
        "max_windows": jsonable_config(cfg["max_windows"]),
        "window_sampling_namespace": int(cfg["randomness"]["window_sampling_namespace"]),
        "window_sampling_group": str(unit.get("window_sampling_group", bundle.name)),
        "window_origin_sha256s": _window_origin_hashes(cfg, unit, bundle),
        "seed": int(seed),
    }
    if component == "prior_only":
        payload["prior_graph_sha256"] = sha256_file(base_graph_path(bundle))
    elif component not in {"all_only", "local_only"}:
        raise ValueError(f"Unsupported fixed component fingerprint: {component}")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _shared_component_fingerprint(
    cfg: dict[str, Any],
    unit: dict[str, Any],
    seed: int,
    bundle: DatasetBundle,
    component: str,
    controls: list[dict[str, Any]],
) -> str:
    if component not in {"shared_matched", "shared_random"}:
        raise ValueError(f"Unsupported shared component fingerprint: {component}")
    payload = {
        "schema": _artifact_schema(cfg, "selective_prior_shared_component"),
        "component": component,
        "code_fingerprint": code_fingerprint(),
        "series_content_sha256": _series_content_fingerprint(bundle),
        "prior_graph_sha256": sha256_file(base_graph_path(bundle)),
        "controls": controls,
        "split": jsonable_config(cfg["split"]),
        "history": int(unit["history"]),
        "max_horizon": int(unit["max_horizon"]),
        "training": jsonable_config(cfg["training"]),
        "max_windows": jsonable_config(cfg["max_windows"]),
        "window_sampling_namespace": int(cfg["randomness"]["window_sampling_namespace"]),
        "window_sampling_group": str(unit.get("window_sampling_group", bundle.name)),
        "window_origin_sha256s": _window_origin_hashes(cfg, unit, bundle),
        "control_exposure_count": int(cfg["method"].get("control_exposure_count", len(controls))),
        "control_probe_indices": [int(value) for value in cfg["method"].get("control_probe_indices", [])],
        "shared_control_loss_weight": float(
            cfg["method"].get("shared_control_loss_weight", 0.20)
        ),
        "shared_validation_control_weight": float(
            cfg["method"].get("shared_validation_control_weight", 0.0)
        ),
        "shared_distillation_weight": float(
            cfg["method"].get("shared_distillation_weight", 0.0)
        ),
        "shared_checkpoint_selection": str(
            cfg["method"].get("shared_checkpoint_selection", "mean_route_mae")
        ),
        "seed": int(seed),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cached_all_only_forecaster(
    cfg: dict[str, Any],
    unit: dict[str, Any],
    seed: int,
    bundle: DatasetBundle,
    datasets: dict[str, WindowDataset],
    mask: torch.Tensor,
    training: dict[str, Any],
    device: torch.device,
    local_checkpoint: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train A-only once per component fingerprint and reuse it across E3 corruption units."""
    fingerprint = _fixed_component_fingerprint(cfg, unit, seed, bundle, "all_only")
    expected_origin_hashes = {
        split: _array_content_sha256(dataset.starts)
        for split, dataset in datasets.items()
    }
    cache_dir = resolved_output_root(cfg) / "component_cache" / "all_only"
    cache_checkpoint = cache_dir / f"{fingerprint}.pt"
    marker = cache_dir / f"{fingerprint}.complete.json"
    lock = cache_dir / f"{fingerprint}.lock"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = None
    deadline = time.time() + 12 * 60 * 60
    while not (cache_checkpoint.exists() and marker.exists()):
        try:
            lock.mkdir()
            owner = True
            atomic_json(
                lock / "owner.json",
                {"pid": os.getpid(), "created_unix": time.time(), "fingerprint": fingerprint},
            )
        except FileExistsError:
            owner = False
        if owner:
            try:
                model = train_fixed_forecaster(
                    bundle,
                    datasets["forecaster_train"],
                    datasets["forecaster_validation"],
                    mask,
                    training,
                    int(seed),
                    device,
                    cache_checkpoint,
                    fingerprint,
                    False,
                )
                atomic_json(
                    marker,
                    {
                        "schema": _artifact_schema(cfg, "selective_prior_component_cache"),
                        "fingerprint": fingerprint,
                        "checkpoint_sha256": sha256_file(cache_checkpoint),
                        "window_sampling_group": str(
                            unit.get("window_sampling_group", bundle.name)
                        ),
                        "window_origin_sha256s": expected_origin_hashes,
                    },
                )
            finally:
                if lock.exists():
                    shutil.rmtree(lock)
            break
        else:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for A-only component cache: {lock}")
            time.sleep(5.0)
    marker_payload = _read_json(marker)
    expected_marker_schema = _artifact_schema(cfg, "selective_prior_component_cache")
    if marker_payload.get("schema") != expected_marker_schema:
        raise RuntimeError(f"A-only cache marker schema mismatch: {marker}")
    if marker_payload.get("fingerprint") != fingerprint:
        raise RuntimeError(f"A-only cache marker fingerprint mismatch: {marker}")
    if marker_payload.get("window_sampling_group") != str(
        unit.get("window_sampling_group", bundle.name)
    ):
        raise RuntimeError(f"A-only cache sampling-group mismatch: {marker}")
    if marker_payload.get("window_origin_sha256s") != expected_origin_hashes:
        raise RuntimeError(
            "A-only cache origin hashes do not match the current window contract: "
            f"{marker}"
        )
    if marker_payload.get("checkpoint_sha256") != sha256_file(cache_checkpoint):
        raise RuntimeError(f"A-only cache checkpoint hash mismatch: {cache_checkpoint}")
    if model is None:
        model = train_fixed_forecaster(
            bundle,
            datasets["forecaster_train"],
            datasets["forecaster_validation"],
            mask,
            training,
            int(seed),
            device,
            cache_checkpoint,
            fingerprint,
            True,
        )
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_checkpoint, local_checkpoint)
    cache_history = cache_checkpoint.with_suffix(".history.json")
    if cache_history.exists():
        shutil.copy2(cache_history, local_checkpoint.with_suffix(".history.json"))
    provenance = {
        "schema": _artifact_schema(cfg, "selective_prior_component_reuse"),
        "component": "all_only",
        "fingerprint": fingerprint,
        "cache_checkpoint": str(cache_checkpoint),
        "cache_checkpoint_sha256": sha256_file(cache_checkpoint),
        "window_sampling_group": str(unit.get("window_sampling_group", bundle.name)),
        "window_origin_sha256s": expected_origin_hashes,
    }
    atomic_json(local_checkpoint.with_suffix(".reuse.json"), provenance)
    return model, provenance


def _window_seed(cfg: dict[str, Any], sampling_key: str, split: str) -> int:
    """Derive a window seed from the declared sampling group, not graph identity."""
    namespace = int(cfg.get("randomness", {}).get("window_sampling_namespace", 20260813))
    return _stable_int(namespace, sampling_key, split)


def _window_origin_hashes(
    cfg: dict[str, Any], unit: dict[str, Any], bundle: DatasetBundle
) -> dict[str, str]:
    """Hash the exact origin arrays used by every split.

    The E3 A-only cache deliberately ignores the prior graph, but it must not
    ignore the sampled training windows.  This helper is also used in the
    component fingerprints so a cache can only be reused when the complete
    window contract is identical.
    """
    history = int(unit["history"])
    horizon = int(unit["max_horizon"])
    sampling_key = str(unit.get("window_sampling_group", bundle.name))
    hashes: dict[str, str] = {}
    for split in SPLITS:
        maximum = unit.get("max_windows", {}).get(split, cfg["max_windows"].get(split))
        starts = window_starts(
            bundle,
            split,  # type: ignore[arg-type]
            history,
            horizon,
            None if maximum is None else int(maximum),
            _window_seed(cfg, sampling_key, split),
        )
        hashes[split] = _array_content_sha256(starts)
    return hashes


def _datasets_for_unit(cfg: dict[str, Any], unit: dict[str, Any], bundle: DatasetBundle) -> dict[str, WindowDataset]:
    history = int(unit["history"])
    horizon = int(unit["max_horizon"])
    sampling_key = str(unit.get("window_sampling_group", bundle.name))
    datasets: dict[str, WindowDataset] = {}
    for split in SPLITS:
        maximum = unit.get("max_windows", {}).get(split, cfg["max_windows"].get(split))
        starts = window_starts(
            bundle,
            split,  # type: ignore[arg-type]
            history,
            horizon,
            None if maximum is None else int(maximum),
            _window_seed(cfg, sampling_key, split),
        )
        datasets[split] = WindowDataset(bundle, starts, history, horizon)
    return datasets


def _save_prediction(path: Path, array: np.ndarray, cfg: dict[str, Any], *, matched: bool = False) -> Path:
    storage = cfg.get("prediction_storage", {})
    dtype = storage.get("matched_dtype", "float32") if matched else storage.get("prediction_dtype", "float32")
    if matched and bool(storage.get("matched_compressed", True)):
        actual = path.with_suffix(".npz")
        _atomic_compressed_array(actual, array, dtype=dtype)
    else:
        actual = path
        _atomic_array(actual, array, dtype=dtype)
    return actual


def _save_shared_predictions(
    cfg: dict[str, Any],
    directory: Path,
    split: str,
    model: torch.nn.Module,
    dataset: WindowDataset,
    masks: dict[str, torch.Tensor],
    matched_masks: list[torch.Tensor],
    training: dict[str, Any],
    device: torch.device,
    artifact_root: str = "predictions",
) -> dict[str, Any]:
    split_dir = directory / artifact_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    targets, pred_all = predict_raw(model, dataset, masks["all"], training, device)  # type: ignore[arg-type]
    prior_targets, pred_prior = predict_raw(model, dataset, masks["prior"], training, device)  # type: ignore[arg-type]
    if not np.allclose(targets, prior_targets, atol=0.0, rtol=0.0):
        raise RuntimeError("Prediction routes produced misaligned targets")
    _save_prediction(split_dir / "targets.npy", targets, cfg)
    _save_prediction(split_dir / "pred_all.npy", pred_all, cfg)
    _save_prediction(split_dir / "pred_prior.npy", pred_prior, cfg)
    matched_paths: list[Path] = []
    for index, mask in enumerate(matched_masks):
        control_targets, prediction = predict_raw(model, dataset, mask, training, device)  # type: ignore[arg-type]
        if not np.allclose(targets, control_targets, atol=0.0, rtol=0.0):
            raise RuntimeError(f"Matched control {index} produced misaligned targets")
        path = _save_prediction(
            split_dir / f"pred_matched_{index:03d}.npy", prediction, cfg, matched=True
        )
        matched_paths.append(path)
    _atomic_array(split_dir / "origin_indices.npy", dataset.starts, dtype=np.int64)
    return {
        "directory": split_dir,
        "targets": targets,
        "pred_all": pred_all,
        "pred_prior": pred_prior,
        "matched_paths": matched_paths,
    }


def _save_independent_expert_predictions(
    cfg: dict[str, Any],
    directory: Path,
    split: str,
    all_model: torch.nn.Module,
    prior_model: torch.nn.Module,
    dataset: WindowDataset,
    masks: dict[str, torch.Tensor],
    matched_masks: list[torch.Tensor],
    training: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Save predictions from independently trained A and R experts.

    Protocol v4 deliberately never trains a shared forecaster on matched
    controls.  The R expert is evaluated under held-out matched masks only as
    a counterfactual probe; its parameters are frozen from the true-prior
    route.  This preserves a meaningful intervention and removes the v2/v3
    control-exposure interference mechanism.
    """
    split_dir = directory / "predictions" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    targets, pred_all = predict_raw(all_model, dataset, masks["all"], training, device)
    prior_targets, pred_prior = predict_raw(prior_model, dataset, masks["prior"], training, device)
    if not np.allclose(targets, prior_targets, atol=0.0, rtol=0.0):
        raise RuntimeError("Independent expert routes produced misaligned targets")
    _save_prediction(split_dir / "targets.npy", targets, cfg)
    _save_prediction(split_dir / "pred_all.npy", pred_all, cfg)
    _save_prediction(split_dir / "pred_prior.npy", pred_prior, cfg)
    matched_paths: list[Path] = []
    for index, mask in enumerate(matched_masks):
        control_targets, prediction = predict_raw(prior_model, dataset, mask, training, device)
        if not np.allclose(targets, control_targets, atol=0.0, rtol=0.0):
            raise RuntimeError(f"Independent matched probe {index} produced misaligned targets")
        matched_paths.append(
            _save_prediction(
                split_dir / f"pred_matched_{index:03d}.npy", prediction, cfg, matched=True
            )
        )
    _atomic_array(split_dir / "origin_indices.npy", dataset.starts, dtype=np.int64)
    return {
        "directory": split_dir,
        "targets": targets,
        "pred_all": pred_all,
        "pred_prior": pred_prior,
        "matched_paths": matched_paths,
    }


def _node_ids(bundle: DatasetBundle) -> np.ndarray:
    path = bundle.root / "nodes.csv"
    if not path.exists():
        return np.asarray([str(index) for index in range(bundle.variables)])
    frame = pd.read_csv(path)
    column = "node_id" if "node_id" in frame.columns else frame.columns[-1]
    values = frame[column].astype(str).to_numpy()
    if len(values) != bundle.variables:
        raise ValueError(f"Node table length mismatch for {bundle.name}")
    return values


def _horizon_slice(array: np.ndarray, horizon: int) -> np.ndarray:
    if array.ndim != 3 or not 1 <= int(horizon) <= array.shape[1]:
        raise ValueError(f"Cannot take horizon {horizon} from shape {array.shape}")
    return array[:, int(horizon) - 1 : int(horizon), :]


def _synthetic_ground_truth(bundle: DatasetBundle, origins: np.ndarray, horizon: int) -> np.ndarray | None:
    path = bundle.root / "ground_truth_prior_valid.npy"
    if not path.exists():
        return None
    labels = np.load(path, allow_pickle=False).astype(np.uint8)
    target_indices = origins[:, None] + np.arange(1, horizon + 1, dtype=np.int64)[None, :] - 1
    if target_indices.max() >= len(labels):
        raise ValueError("Synthetic ground-truth labels do not cover requested targets")
    return np.repeat(labels[target_indices, None], bundle.variables, axis=2)


def _synthetic_evaluable_mask(
    bundle: DatasetBundle,
    origins: np.ndarray,
    horizon: int,
    history: int,
) -> np.ndarray | None:
    """Return a label-derived mask for stable-regime mechanism scoring."""
    regime_path = bundle.root / "ground_truth_regime.npy"
    if regime_path.exists():
        regimes = np.load(regime_path, allow_pickle=False).astype(np.int64)
        result = np.zeros((len(origins), int(horizon)), dtype=bool)
        for origin_index, origin in enumerate(np.asarray(origins, dtype=np.int64)):
            for forecast_index in range(int(horizon)):
                stop = int(origin) + forecast_index + 1
                start = int(origin) - int(history)
                if start < 0 or stop > len(regimes):
                    raise ValueError("Synthetic regime labels do not cover a window")
                window = regimes[start:stop]
                result[origin_index, forecast_index] = bool(
                    len(window) and np.all(window == window[-1])
                )
        return np.repeat(result[:, :, None], bundle.variables, axis=2)
    path = bundle.root / "ground_truth_evaluable.npy"
    if not path.exists():
        return None
    evaluable = np.load(path, allow_pickle=False).astype(bool)
    target_indices = (
        origins[:, None]
        + np.arange(1, horizon + 1, dtype=np.int64)[None, :]
        - 1
    )
    if target_indices.max() >= len(evaluable):
        raise ValueError("Synthetic evaluability mask does not cover requested targets")
    return np.repeat(evaluable[target_indices, None], bundle.variables, axis=2)


def _e1_grouped_prior_value_diagnostics(
    *,
    labels: np.ndarray,
    evaluable: np.ndarray,
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
    q_loss: np.ndarray,
    q_specificity: np.ndarray,
    matched_soft_win_rate: np.ndarray,
    benefit_probability: np.ndarray,
    specificity_probability: np.ndarray,
    combined_probability: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Separate E1 headroom, targets, and gate evidence by known regime."""
    arrays = {
        "labels": np.asarray(labels),
        "evaluable": np.asarray(evaluable),
        "target": np.asarray(target),
        "pred_all": np.asarray(pred_all),
        "pred_prior": np.asarray(pred_prior),
        "q_loss": np.asarray(q_loss),
        "q_specificity": np.asarray(q_specificity),
        "matched_soft_win_rate": np.asarray(matched_soft_win_rate),
        "benefit_probability": np.asarray(benefit_probability),
        "specificity_probability": np.asarray(specificity_probability),
        "combined_probability": np.asarray(combined_probability),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"E1 grouped diagnostic shape mismatch: {shapes}")
    selected = arrays["evaluable"].astype(bool)
    truth = arrays["labels"].astype(bool)
    groups = {"prior_valid": selected & truth, "prior_invalid": selected & ~truth}
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in groups.items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise RuntimeError(f"E1 has no evaluable {name} elements")
        all_mae = float(np.mean(np.abs(arrays["pred_all"][mask] - arrays["target"][mask])))
        prior_mae = float(np.mean(np.abs(arrays["pred_prior"][mask] - arrays["target"][mask])))
        result[name] = {
            "evaluable_elements": count,
            "shared_all_mae": all_mae,
            "shared_prior_mae": prior_mae,
            "shared_prior_relative_gain_over_all": (all_mae - prior_mae) / max(all_mae, 1e-12),
            "mean_q_loss": float(np.mean(arrays["q_loss"][mask])),
            "mean_q_specificity": float(np.mean(arrays["q_specificity"][mask])),
            "mean_matched_soft_win_rate": float(np.mean(arrays["matched_soft_win_rate"][mask])),
            "mean_benefit_head_probability": float(np.mean(arrays["benefit_probability"][mask])),
            "mean_specificity_head_probability": float(
                np.mean(arrays["specificity_probability"][mask])
            ),
            "mean_combined_probability": float(np.mean(arrays["combined_probability"][mask])),
            "activation_rate_greater_0_5": float(
                np.mean(arrays["combined_probability"][mask] > 0.5)
            ),
        }
    return result


def _evaluate(
    bundle: DatasetBundle,
    unit: dict[str, Any],
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    binary_prior_value: np.ndarray,
    static_alphas: dict[str, float],
    random_static_alphas: dict[str, float] | None,
    best_fixed_by_horizon: dict[str, str],
    origins: np.ndarray,
    validation_branch_predictions: dict[str, np.ndarray] | None = None,
    explicit_gate_validation_preservation: bool = False,
) -> dict[str, Any]:
    q95 = np.quantile(
        bundle.raw_values[bundle.boundaries["forecaster_train"][0] : bundle.boundaries["forecaster_train"][1]],
        0.95,
        axis=0,
    )
    rows: dict[str, Any] = {}
    for horizon in [int(value) for value in unit["report_horizons"]]:
        target = _horizon_slice(predictions["target"], horizon)
        methods = {
            name: _horizon_slice(value, horizon)
            for name, value in predictions.items()
            if name != "target"
        }
        method_metrics = {
            name: regression_metrics_numpy(target, prediction, q95_threshold=q95)
            for name, prediction in methods.items()
        }
        best_fixed_name = best_fixed_by_horizon.get(str(horizon))
        if best_fixed_name not in {"all_only", "prior_only"}:
            raise ValueError(
                f"Invalid validation-selected fixed branch at horizon {horizon}: {best_fixed_name}"
            )
        best_fixed_mae = method_metrics[best_fixed_name]["mae"]
        best_fixed_q95 = method_metrics[best_fixed_name]["q95_mae"]
        diagnostics: dict[str, float] = {
            "shared_all_test_relative_degradation": (
                method_metrics["shared_all"]["mae"] - method_metrics["all_only"]["mae"]
            )
            / max(method_metrics["all_only"]["mae"], 1e-12),
            "shared_prior_test_relative_degradation": (
                method_metrics["shared_prior"]["mae"] - method_metrics["prior_only"]["mae"]
            )
            / max(method_metrics["prior_only"]["mae"], 1e-12),
            "oracle_headroom_from_static": (
                method_metrics["static_mix"]["mae"] - method_metrics["oracle"]["mae"]
            )
            / max(method_metrics["static_mix"]["mae"], 1e-12),
            "matched_gate_relative_gain_over_static": (
                method_metrics["static_mix"]["mae"] - method_metrics["matched_control_gate"]["mae"]
            )
            / max(method_metrics["static_mix"]["mae"], 1e-12),
            "matched_gate_relative_gain_over_loss_gate": (
                method_metrics["loss_gate"]["mae"] - method_metrics["matched_control_gate"]["mae"]
            )
            / max(method_metrics["loss_gate"]["mae"], 1e-12),
            "matched_gate_relative_gain_over_best_fixed": (
                best_fixed_mae - method_metrics["matched_control_gate"]["mae"]
            )
            / max(best_fixed_mae, 1e-12),
            "matched_gate_q95_relative_degradation_from_best_fixed": (
                method_metrics["matched_control_gate"]["q95_mae"] - best_fixed_q95
            )
            / max(best_fixed_q95, 1e-12),
        }
        if validation_branch_predictions is not None:
            required_validation = {
                "target",
                "all_only",
                "prior_only",
                "shared_all",
                "shared_prior",
            }
            missing_validation = required_validation.difference(validation_branch_predictions)
            if missing_validation:
                raise ValueError(
                    "Validation branch diagnostics are missing arrays: "
                    + ", ".join(sorted(missing_validation))
                )
            validation_target = _horizon_slice(
                validation_branch_predictions["target"], horizon
            )
            validation_all = _horizon_slice(
                validation_branch_predictions["all_only"], horizon
            )
            validation_prior = _horizon_slice(
                validation_branch_predictions["prior_only"], horizon
            )
            validation_shared_all = _horizon_slice(
                validation_branch_predictions["shared_all"], horizon
            )
            validation_shared_prior = _horizon_slice(
                validation_branch_predictions["shared_prior"], horizon
            )
            validation_all_mae = float(np.mean(np.abs(validation_all - validation_target)))
            validation_prior_mae = float(np.mean(np.abs(validation_prior - validation_target)))
            shared_all_gate_validation_degradation = (
                float(np.mean(np.abs(validation_shared_all - validation_target)))
                - validation_all_mae
            ) / max(validation_all_mae, 1e-12)
            shared_prior_gate_validation_degradation = (
                float(np.mean(np.abs(validation_shared_prior - validation_target)))
                - validation_prior_mae
            ) / max(validation_prior_mae, 1e-12)
            diagnostics["shared_all_gate_validation_relative_degradation"] = shared_all_gate_validation_degradation
            diagnostics["shared_prior_gate_validation_relative_degradation"] = shared_prior_gate_validation_degradation
            if not explicit_gate_validation_preservation:
                diagnostics["shared_all_validation_relative_degradation"] = shared_all_gate_validation_degradation
                diagnostics["shared_prior_validation_relative_degradation"] = shared_prior_gate_validation_degradation
        if "random_control_gate" in method_metrics:
            diagnostics["matched_gate_relative_gain_over_random_control_gate"] = (
                method_metrics["random_control_gate"]["mae"]
                - method_metrics["matched_control_gate"]["mae"]
            ) / max(method_metrics["random_control_gate"]["mae"], 1e-12)
        gate_rows = {}
        for name, probability in probabilities.items():
            if name == "random_control_gate":
                all_branch = _horizon_slice(predictions["random_shared_all"], horizon)
                prior_branch = _horizon_slice(predictions["random_shared_prior"], horizon)
                static = _horizon_slice(predictions["random_static_mix"], horizon)
            else:
                all_branch = _horizon_slice(predictions["shared_all"], horizon)
                prior_branch = _horizon_slice(predictions["shared_prior"], horizon)
                static = _horizon_slice(predictions["static_mix"], horizon)
            gate_rows[name] = gate_metrics(
                _horizon_slice(probability, horizon),
                _horizon_slice(binary_prior_value, horizon),
                target,
                all_branch,
                prior_branch,
                static,
                methods[name],
            )
        ground_truth = _synthetic_ground_truth(bundle, origins, int(unit["max_horizon"]))
        if ground_truth is not None:
            labels = _horizon_slice(ground_truth, horizon).ravel()
            probability = _horizon_slice(
                probabilities["matched_control_gate"], horizon
            ).ravel()
            evaluable = _synthetic_evaluable_mask(
                bundle,
                origins,
                int(unit["max_horizon"]),
                int(unit["history"]),
            )
            if evaluable is not None:
                selected = _horizon_slice(evaluable, horizon).ravel().astype(bool)
                if not np.any(selected):
                    raise RuntimeError(
                        f"Synthetic mechanism horizon {horizon} has no evaluable targets"
                    )
                labels = labels[selected]
                probability = probability[selected]
                gate_rows["matched_control_gate"]["ground_truth_evaluable_fraction"] = (
                    float(np.mean(selected))
                )
            gate_rows["matched_control_gate"]["ground_truth_prevalence"] = float(labels.mean())
            gate_rows["matched_control_gate"]["ground_truth_auprc"] = (
                float(average_precision_score(labels, probability)) if np.unique(labels).size == 2 else float("nan")
            )
            activated = probability > 0.5
            valid = labels.astype(bool)
            invalid = ~valid
            gate_rows["matched_control_gate"]["ground_truth_valid_activation_rate"] = (
                float(np.mean(activated[valid])) if np.any(valid) else float("nan")
            )
            gate_rows["matched_control_gate"]["ground_truth_invalid_activation_rate"] = (
                float(np.mean(activated[invalid])) if np.any(invalid) else float("nan")
            )
        rows[str(horizon)] = {"methods": method_metrics, "gates": gate_rows, "diagnostics": diagnostics}
    return {
        "static_alpha_by_horizon": {str(key): float(value) for key, value in static_alphas.items()},
        "random_static_alpha_by_horizon": (
            {str(key): float(value) for key, value in random_static_alphas.items()}
            if random_static_alphas is not None
            else None
        ),
        "validation_selected_fixed_branch_by_horizon": dict(best_fixed_by_horizon),
        "horizons": rows,
    }


def _component_fingerprint(run_fingerprint: str, component: str) -> str:
    return hashlib.sha256(f"{run_fingerprint}:{component}".encode()).hexdigest()


def run_one(
    cfg: dict[str, Any],
    config_path: Path,
    unit: dict[str, Any],
    seed: int,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if unit.get("locked"):
        verify_external_lock(cfg, config_path)
    data_root = resolved_data_root(cfg)
    bundle = load_bundle(data_root, str(unit["dataset"]), cfg["split"])
    device = resolve_device(str(cfg["device"]))
    masks = fixed_masks(bundle, device=device)
    matched_masks, control_provenance = load_matched_controls(
        bundle,
        int(cfg["method"]["matched_controls"]),
        float(cfg["method"]["minimum_endpoint_change"]),
        device=device,
    )
    matched_exposure_indices, matched_probe_indices = _control_partitions(
        cfg, len(matched_masks)
    )
    matched_exposure_masks = [matched_masks[index] for index in matched_exposure_indices]
    use_random_control_ablation = _uses_random_control_ablation(unit)
    if use_random_control_ablation:
        random_masks, random_control_provenance = load_controls(
            bundle,
            int(cfg["method"]["matched_controls"]),
            float(cfg["method"]["minimum_endpoint_change"]),
            kind="random",
            device=device,
        )
    else:
        random_masks, random_control_provenance = [], []
    random_exposure_indices: list[int] = []
    random_probe_indices: list[int] = []
    random_exposure_masks: list[torch.Tensor] = []
    if random_masks:
        random_exposure_indices, random_probe_indices = _control_partitions(
            cfg, len(random_masks)
        )
        random_exposure_masks = [random_masks[index] for index in random_exposure_indices]
    all_control_provenance = control_provenance + random_control_provenance
    fingerprint = _unit_fingerprint(cfg, config_path, unit, seed, bundle, all_control_provenance)
    output = _unit_output(cfg, unit, seed)
    state_path = output / "state.json"
    if state_path.exists():
        state = _read_json(state_path)
        if state.get("status") == "complete" and state.get("fingerprint") == fingerprint and not overwrite:
            return _read_json(output / "metrics.json")
        if not resume and not overwrite:
            raise FileExistsError(f"Incomplete or stale run exists: {output}; use --resume or --overwrite")
    if output.exists() and overwrite:
        resolved_output = output.resolve()
        resolved_root = resolved_output_root(cfg).resolve()
        if resolved_root not in resolved_output.parents or "runs" not in resolved_output.parts:
            raise RuntimeError(f"Refusing to replace an output outside the declared run tree: {resolved_output}")
        shutil.rmtree(resolved_output)
    output.mkdir(parents=True, exist_ok=True)
    q95_thresholds = np.quantile(
        bundle.raw_values[
            bundle.boundaries["forecaster_train"][0] : bundle.boundaries["forecaster_train"][1]
        ],
        0.95,
        axis=0,
    )
    _atomic_array(output / "q95_thresholds.npy", q95_thresholds)
    atomic_json(
        state_path,
        {
            "schema": _artifact_schema(cfg, "selective_prior_injection_state"),
            "status": "running",
            "fingerprint": fingerprint,
            "experiment_id": unit["experiment_id"],
            "dataset": bundle.name,
            "seed": int(seed),
            "git_commit": current_git_commit(),
            "config_sha256": config_sha256(config_path),
            "code_fingerprint": code_fingerprint(),
        },
    )
    datasets = _datasets_for_unit(cfg, unit, bundle)
    window_origin_sha256s = {
        split: _array_content_sha256(dataset.starts)
        for split, dataset in datasets.items()
    }
    for split, dataset in datasets.items():
        _atomic_array(output / "splits" / f"{split}_origins.npy", dataset.starts, dtype=np.int64)
    _atomic_array(output / "node_ids.npy", _node_ids(bundle).astype("U"))
    training = cfg["training"]
    checkpoint_dir = output / "checkpoints"
    local = None
    if _uses_local_context_baseline(unit):
        local = train_fixed_forecaster(
            bundle,
            datasets["forecaster_train"],
            datasets["forecaster_validation"],
            masks["local"],
            training,
            int(seed),
            device,
            checkpoint_dir / "local_only.pt",
            _fixed_component_fingerprint(cfg, unit, seed, bundle, "local_only"),
            resume and not overwrite,
        )
    all_only, all_only_reuse = _cached_all_only_forecaster(
        cfg,
        unit,
        seed,
        bundle,
        datasets,
        masks["all"],
        training,
        device,
        checkpoint_dir / "all_only.pt",
    )
    prior_only = train_fixed_forecaster(
        bundle,
        datasets["forecaster_train"],
        datasets["forecaster_validation"],
        masks["prior"],
        training,
        int(seed),
        device,
        checkpoint_dir / "prior_only.pt",
        _fixed_component_fingerprint(cfg, unit, seed, bundle, "prior_only"),
        resume and not overwrite,
    )
    is_v3 = cfg.get("schema") == "selective_prior_injection_v3"
    # V5 keeps the independent frozen A/R experts and held-out controls from
    # v4.  Its new causal aggregation layer is applied after these artefacts
    # are written, so the expensive expert predictions remain identical.
    is_v4 = cfg.get("schema") in {
        "selective_prior_injection_v4",
        "selective_prior_injection_v5",
        "selective_prior_injection_final_v1",
    }
    if is_v4:
        # Independent frozen experts are the v4 scientific intervention.  The
        # variable named ``shared`` below is only a feature backbone for the
        # gate; no shared forecaster checkpoint is trained or used.
        shared = all_only
        exposures = []
    else:
        shared, exposures = train_shared_forecaster(
            bundle,
            datasets["forecaster_train"],
            datasets["forecaster_validation"],
            masks["all"],
            masks["prior"],
            matched_exposure_masks,
            [int(value) for value in cfg["method"]["shared_validation_controls"]],
            training,
            int(seed),
            device,
            checkpoint_dir / "shared_matched.pt",
            _shared_component_fingerprint(
                cfg,
                unit,
                seed,
                bundle,
                "shared_matched",
                [control_provenance[index] for index in matched_exposure_indices],
            ),
            resume and not overwrite,
            control_loss_weight=float(cfg["method"].get("shared_control_loss_weight", 0.20)),
            validation_control_weight=float(
                cfg["method"].get("shared_validation_control_weight", 0.0)
            ),
            teacher_all=all_only if is_v3 else None,
            teacher_prior=prior_only if is_v3 else None,
            distillation_weight=float(
                cfg["method"].get("shared_distillation_weight", 0.0)
            ),
            checkpoint_selection=str(
                cfg["method"].get("shared_checkpoint_selection", "mean_route_mae")
            ),
            checkpoint_schema=(
                "shared_mask_forecaster_v3" if is_v3 else "shared_mask_forecaster_v2"
            ),
        )
    random_shared = None
    random_exposures = None
    if use_random_control_ablation:
        random_shared, random_exposures = train_shared_forecaster(
            bundle,
            datasets["forecaster_train"],
            datasets["forecaster_validation"],
            masks["all"],
            masks["prior"],
            random_exposure_masks,
            [int(value) for value in cfg["method"]["shared_validation_controls"]],
            training,
            int(seed),
            device,
            checkpoint_dir / "shared_random.pt",
            _shared_component_fingerprint(
                cfg,
                unit,
                seed,
                bundle,
                "shared_random",
                [random_control_provenance[index] for index in random_exposure_indices],
            ),
            resume and not overwrite,
            control_loss_weight=float(cfg["method"].get("shared_control_loss_weight", 0.20)),
            validation_control_weight=float(
                cfg["method"].get("shared_validation_control_weight", 0.0)
            ),
            teacher_all=all_only if is_v3 else None,
            teacher_prior=prior_only if is_v3 else None,
            distillation_weight=float(
                cfg["method"].get("shared_distillation_weight", 0.0)
            ),
            checkpoint_selection=str(
                cfg["method"].get("shared_checkpoint_selection", "mean_route_mae")
            ),
            checkpoint_schema=(
                "shared_mask_forecaster_v3" if is_v3 else "shared_mask_forecaster_v2"
            ),
        )
    if is_v4:
        shared_predictions = {
            split: _save_independent_expert_predictions(
                cfg,
                output,
                split,
                all_only,
                prior_only,
                datasets[split],
                masks,
                matched_masks,
                training,
                device,
            )
            for split in GATE_SPLITS
        }
    else:
        shared_predictions = {
            split: _save_shared_predictions(
                cfg,
                output,
                split,
                shared,
                datasets[split],
                masks,
                matched_masks,
                training,
                device,
            )
            for split in GATE_SPLITS
        }
    random_shared_predictions = (
        {
            split: _save_shared_predictions(
                cfg,
                output,
                split,
                random_shared,
                datasets[split],
                masks,
                random_masks,
                training,
                device,
                artifact_root="predictions_random_controls",
            )
            for split in GATE_SPLITS
        }
        if random_shared is not None
        else None
    )
    epsilon = node_error_scale(bundle)
    targets_by_split = {
        split: prior_value_targets_from_files(
            shared_predictions[split]["targets"],
            shared_predictions[split]["pred_all"],
            shared_predictions[split]["pred_prior"],
            [shared_predictions[split]["matched_paths"][index] for index in matched_probe_indices],
            epsilon,
            cfg["method"],
        )
        for split in GATE_SPLITS
    }
    random_targets_by_split = (
        {
            split: prior_value_targets_from_files(
                random_shared_predictions[split]["targets"],
                random_shared_predictions[split]["pred_all"],
                random_shared_predictions[split]["pred_prior"],
                [
                    random_shared_predictions[split]["matched_paths"][index]
                    for index in random_probe_indices
                ],
                epsilon,
                cfg["method"],
            )
            for split in GATE_SPLITS
        }
        if random_shared_predictions is not None
        else None
    )
    for split, values in targets_by_split.items():
        for name, value in values.items():
            _atomic_array(output / "predictions" / split / f"target_{name}.npy", value)
    if random_targets_by_split is not None:
        for split, values in random_targets_by_split.items():
            for name, value in values.items():
                _atomic_array(output / "predictions_random_controls" / split / f"target_{name}.npy", value)
    loss_gate = train_prior_gate(
        shared,
        datasets["gate_train"],
        datasets["gate_validation"],
        shared_predictions["gate_train"]["pred_all"],
        shared_predictions["gate_train"]["pred_prior"],
        shared_predictions["gate_validation"]["pred_all"],
        shared_predictions["gate_validation"]["pred_prior"],
        targets_by_split["gate_train"]["q_loss"],
        targets_by_split["gate_validation"]["q_loss"],
        float(cfg["method"]["gate_lambda"]),
        training,
        _stable_int(seed, "loss_gate"),
        device,
        checkpoint_dir / "loss_gate.pt",
        _component_fingerprint(fingerprint, "loss_gate"),
        resume and not overwrite,
        checkpoint_schema="prior_gate_v4" if is_v4 else "prior_gate_v2",
    )
    if is_v3 or is_v4:
        matched_gate = train_matched_prior_gate(
            shared,
            datasets["gate_train"],
            datasets["gate_validation"],
            shared_predictions["gate_train"]["pred_all"],
            shared_predictions["gate_train"]["pred_prior"],
            shared_predictions["gate_validation"]["pred_all"],
            shared_predictions["gate_validation"]["pred_prior"],
            targets_by_split["gate_train"]["q_loss"],
            targets_by_split["gate_validation"]["q_loss"],
            targets_by_split["gate_train"]["q_specificity"],
            targets_by_split["gate_validation"]["q_specificity"],
            float(cfg["method"]["gate_lambda"]),
            float(cfg["method"].get("specificity_gate_lambda", 0.30)),
            training,
            _stable_int(seed, "matched_gate"),
            device,
            checkpoint_dir / "matched_control_gate.pt",
            _component_fingerprint(
                fingerprint,
                "matched_control_gate_v4" if is_v4 else "matched_control_gate_v3",
            ),
            resume and not overwrite,
            checkpoint_schema="matched_prior_gate_v4" if is_v4 else "matched_prior_gate_v3",
            combiner=str(cfg["method"].get("gate_combiner", "geometric_mean")),
        )
    else:
        matched_gate = train_prior_gate(
            shared,
            datasets["gate_train"],
            datasets["gate_validation"],
            shared_predictions["gate_train"]["pred_all"],
            shared_predictions["gate_train"]["pred_prior"],
            shared_predictions["gate_validation"]["pred_all"],
            shared_predictions["gate_validation"]["pred_prior"],
            targets_by_split["gate_train"]["q_loss"],
            targets_by_split["gate_validation"]["q_loss"],
            float(cfg["method"]["gate_lambda"]),
            training,
            _stable_int(seed, "matched_gate"),
            device,
            checkpoint_dir / "matched_control_gate.pt",
            _component_fingerprint(fingerprint, "matched_control_gate"),
            resume and not overwrite,
            train_q_specificity=targets_by_split["gate_train"]["q_specificity"],
            validation_q_specificity=targets_by_split["gate_validation"]["q_specificity"],
            specificity_lambda=float(
                cfg["method"].get("specificity_gate_lambda", 0.30)
            ),
        )
    random_gate = None
    if (
        random_shared is not None
        and random_shared_predictions is not None
        and random_targets_by_split is not None
    ):
        if is_v3:
            random_gate = train_matched_prior_gate(
                random_shared,
                datasets["gate_train"],
                datasets["gate_validation"],
                random_shared_predictions["gate_train"]["pred_all"],
                random_shared_predictions["gate_train"]["pred_prior"],
                random_shared_predictions["gate_validation"]["pred_all"],
                random_shared_predictions["gate_validation"]["pred_prior"],
                random_targets_by_split["gate_train"]["q_loss"],
                random_targets_by_split["gate_validation"]["q_loss"],
                random_targets_by_split["gate_train"]["q_specificity"],
                random_targets_by_split["gate_validation"]["q_specificity"],
                float(cfg["method"]["gate_lambda"]),
                float(cfg["method"].get("specificity_gate_lambda", 0.30)),
                training,
                _stable_int(seed, "random_control_gate"),
                device,
                checkpoint_dir / "random_control_gate.pt",
                _component_fingerprint(fingerprint, "random_control_gate_v3"),
                resume and not overwrite,
            )
        else:
            random_gate = train_prior_gate(
                random_shared,
                datasets["gate_train"],
                datasets["gate_validation"],
                random_shared_predictions["gate_train"]["pred_all"],
                random_shared_predictions["gate_train"]["pred_prior"],
                random_shared_predictions["gate_validation"]["pred_all"],
                random_shared_predictions["gate_validation"]["pred_prior"],
                random_targets_by_split["gate_train"]["q_loss"],
                random_targets_by_split["gate_validation"]["q_loss"],
                float(cfg["method"]["gate_lambda"]),
                training,
                _stable_int(seed, "random_control_gate"),
                device,
                checkpoint_dir / "random_control_gate.pt",
                _component_fingerprint(fingerprint, "random_control_gate"),
                resume and not overwrite,
                train_q_specificity=random_targets_by_split["gate_train"]["q_specificity"],
                validation_q_specificity=random_targets_by_split["gate_validation"]["q_specificity"],
                specificity_lambda=float(
                    cfg["method"].get("specificity_gate_lambda", 0.30)
                ),
            )
    validation = shared_predictions["gate_validation"]
    static_alphas = choose_static_alphas_by_horizon(
        validation["targets"], validation["pred_all"], validation["pred_prior"]
    )
    random_static_alphas = None
    if random_shared_predictions is not None:
        random_validation = random_shared_predictions["gate_validation"]
        random_static_alphas = choose_static_alphas_by_horizon(
            random_validation["targets"],
            random_validation["pred_all"],
            random_validation["pred_prior"],
        )
    validation_targets, validation_all_only = predict_raw(
        all_only, datasets["gate_validation"], masks["all"], training, device
    )
    validation_prior_targets, validation_prior_only = predict_raw(
        prior_only, datasets["gate_validation"], masks["prior"], training, device
    )
    if not np.allclose(validation_targets, validation_prior_targets, atol=0.0, rtol=0.0):
        raise RuntimeError("Fixed-branch validation predictions are misaligned")
    best_fixed_by_horizon = choose_best_fixed_by_horizon(
        validation_targets,
        validation_all_only,
        validation_prior_only,
    )
    test = shared_predictions["test"]
    validation_probability_loss = gate_probabilities(
        shared,
        loss_gate,
        datasets["gate_validation"],
        training,
        device,
        validation["pred_all"],
        validation["pred_prior"],
    )
    raw_probability_loss = gate_probabilities(
        shared,
        loss_gate,
        datasets["test"],
        training,
        device,
        test["pred_all"],
        test["pred_prior"],
    )
    matched_evidence = None
    validation_matched_evidence = None
    if is_v3 or is_v4:
        validation_matched_evidence = matched_gate_probabilities(
            shared,
            matched_gate,
            datasets["gate_validation"],
            training,
            device,
            validation["pred_all"],
            validation["pred_prior"],
        )
        matched_evidence = matched_gate_probabilities(
            shared,
            matched_gate,
            datasets["test"],
            training,
            device,
            test["pred_all"],
            test["pred_prior"],
        )
        raw_probability_matched = matched_evidence["combined"]
    else:
        validation_matched_raw = gate_probabilities(
            shared,
            matched_gate,
            datasets["gate_validation"],
            training,
            device,
            validation["pred_all"],
            validation["pred_prior"],
        )
        validation_matched_evidence = {"combined": validation_matched_raw}
        raw_probability_matched = gate_probabilities(
            shared,
            matched_gate,
            datasets["test"],
            training,
            device,
            test["pred_all"],
            test["pred_prior"],
        )
    calibration = None
    if is_v4:
        calibration = {
            "loss_gate": choose_gate_shrinkage_by_horizon(
                validation["targets"],
                validation["pred_all"],
                validation["pred_prior"],
                validation_probability_loss,
                static_alphas,
            ),
            "matched_control_gate": choose_gate_shrinkage_by_horizon(
                validation["targets"],
                validation["pred_all"],
                validation["pred_prior"],
                validation_matched_evidence["combined"],
                static_alphas,
            ),
        }
        probability_loss = apply_gate_shrinkage_by_horizon(
            raw_probability_loss, static_alphas, calibration["loss_gate"]
        )
        probability_matched = apply_gate_shrinkage_by_horizon(
            raw_probability_matched,
            static_alphas,
            calibration["matched_control_gate"],
        )
        validation_probability_loss_final = apply_gate_shrinkage_by_horizon(
            validation_probability_loss,
            static_alphas,
            calibration["loss_gate"],
        )
        validation_probability_matched_final = apply_gate_shrinkage_by_horizon(
            validation_matched_evidence["combined"],
            static_alphas,
            calibration["matched_control_gate"],
        )
    else:
        probability_loss = raw_probability_loss
        probability_matched = raw_probability_matched
        validation_probability_loss_final = validation_probability_loss
        validation_probability_matched_final = validation_matched_evidence["combined"]
    _, pred_all_only = predict_raw(all_only, datasets["test"], masks["all"], training, device)
    _, pred_prior_only = predict_raw(prior_only, datasets["test"], masks["prior"], training, device)
    static_prediction = mix_with_horizon_alphas(test["pred_all"], test["pred_prior"], static_alphas)
    loss_prediction = probability_loss * test["pred_prior"] + (1.0 - probability_loss) * test["pred_all"]
    matched_prediction = probability_matched * test["pred_prior"] + (1.0 - probability_matched) * test["pred_all"]
    all_error = np.abs(test["pred_all"] - test["targets"])
    prior_error = np.abs(test["pred_prior"] - test["targets"])
    oracle_prediction = np.where(prior_error < all_error, test["pred_prior"], test["pred_all"])
    final_predictions = {
        "target": test["targets"],
        "all_only": pred_all_only,
        "prior_only": pred_prior_only,
        "shared_all": test["pred_all"],
        "shared_prior": test["pred_prior"],
        "static_mix": static_prediction,
        "loss_gate": loss_prediction,
        "matched_control_gate": matched_prediction,
        "oracle": oracle_prediction,
    }
    if local is not None:
        _, pred_local = predict_raw(local, datasets["test"], masks["local"], training, device)
        final_predictions["local_only"] = pred_local
    probabilities = {
        "loss_gate": probability_loss,
        "matched_control_gate": probability_matched,
    }
    probability_random = None
    if (
        random_shared is not None
        and random_gate is not None
        and random_shared_predictions is not None
        and random_static_alphas is not None
    ):
        random_test = random_shared_predictions["test"]
        if is_v3:
            random_evidence = matched_gate_probabilities(
                random_shared,
                random_gate,
                datasets["test"],
                training,
                device,
                random_test["pred_all"],
                random_test["pred_prior"],
            )
            probability_random = random_evidence["combined"]
        else:
            random_evidence = None
            probability_random = gate_probabilities(
                random_shared,
                random_gate,
                datasets["test"],
                training,
                device,
                random_test["pred_all"],
                random_test["pred_prior"],
            )
        random_static_prediction = mix_with_horizon_alphas(
            random_test["pred_all"], random_test["pred_prior"], random_static_alphas
        )
        random_gate_prediction = (
            probability_random * random_test["pred_prior"]
            + (1.0 - probability_random) * random_test["pred_all"]
        )
        final_predictions.update(
            {
                "random_shared_all": random_test["pred_all"],
                "random_shared_prior": random_test["pred_prior"],
                "random_static_mix": random_static_prediction,
                "random_control_gate": random_gate_prediction,
            }
        )
        probabilities["random_control_gate"] = probability_random
    for name, value in final_predictions.items():
        _save_prediction(output / "predictions" / "test" / f"final_{name}.npy", value, cfg)
    _atomic_array(output / "predictions" / "test" / "gate_probability_loss.npy", probability_loss)
    _atomic_array(output / "predictions" / "test" / "gate_probability_matched.npy", probability_matched)
    # Persist validation-time baseline predictions/probabilities as well.  A
    # causal aggregator may use these only to choose its frozen update rate;
    # it must never infer a router hyperparameter from the locked test split.
    validation_static_prediction = mix_with_horizon_alphas(
        validation["pred_all"], validation["pred_prior"], static_alphas
    )
    validation_loss_prediction = (
        validation_probability_loss_final * validation["pred_prior"]
        + (1.0 - validation_probability_loss_final) * validation["pred_all"]
    )
    validation_matched_prediction = (
        validation_probability_matched_final * validation["pred_prior"]
        + (1.0 - validation_probability_matched_final) * validation["pred_all"]
    )
    _atomic_array(
        output / "predictions" / "gate_validation" / "final_static_mix.npy",
        validation_static_prediction,
    )
    _atomic_array(
        output / "predictions" / "gate_validation" / "final_loss_gate.npy",
        validation_loss_prediction,
    )
    _atomic_array(
        output / "predictions" / "gate_validation" / "final_matched_control_gate.npy",
        validation_matched_prediction,
    )
    _atomic_array(
        output / "predictions" / "gate_validation" / "gate_probability_loss.npy",
        validation_probability_loss_final,
    )
    if validation_matched_evidence is not None:
        _atomic_array(
            output / "predictions" / "gate_validation" / "gate_probability_matched.npy",
            validation_probability_matched_final,
        )
    if matched_evidence is not None:
        _atomic_array(
            output / "predictions" / "test" / "gate_probability_matched_benefit.npy",
            matched_evidence["benefit"],
        )
        _atomic_array(
            output / "predictions" / "test" / "gate_probability_matched_specificity.npy",
            matched_evidence["specificity"],
        )
    if probability_random is not None:
        _atomic_array(output / "predictions" / "test" / "gate_probability_random.npy", probability_random)
    metrics = _evaluate(
        bundle,
        unit,
        final_predictions,
        probabilities,
        targets_by_split["test"]["binary_prior_value"],
        static_alphas,
        random_static_alphas,
        best_fixed_by_horizon,
        datasets["test"].starts,
        validation_branch_predictions={
            "target": validation_targets,
            "all_only": validation_all_only,
            "prior_only": validation_prior_only,
            "shared_all": validation["pred_all"],
            "shared_prior": validation["pred_prior"],
        },
        explicit_gate_validation_preservation=is_v3,
    )
    e1_ground_truth = None
    e1_evaluable = None
    if _requires_synthetic_e1_diagnostics(cfg, unit):
        e1_ground_truth = _synthetic_ground_truth(
            bundle, datasets["test"].starts, int(unit["max_horizon"])
        )
        e1_evaluable = _synthetic_evaluable_mask(
            bundle,
            datasets["test"].starts,
            int(unit["max_horizon"]),
            int(unit["history"]),
        )
        if e1_ground_truth is None or e1_evaluable is None or matched_evidence is None:
            raise RuntimeError(
                "Protocol v3/v4 E1 requires ground truth, evaluability, and both gate heads"
            )
    for horizon_text, content in metrics["horizons"].items():
        horizon = int(horizon_text)
        index = horizon - 1
        diagnostics = content["diagnostics"]
        diagnostics.update(
            {
                "mean_target_q_loss": float(
                    np.mean(targets_by_split["test"]["q_loss"][:, index, :])
                ),
                "mean_target_q_specificity": float(
                    np.mean(
                        targets_by_split["test"]["q_specificity"][:, index, :]
                    )
                ),
                "mean_matched_soft_win_rate": float(
                    np.mean(
                        targets_by_split["test"]["matched_soft_win_rate"][
                            :, index, :
                        ]
                    )
                ),
                "binary_prior_value_prevalence": float(
                    np.mean(
                        targets_by_split["test"]["binary_prior_value"][
                            :, index, :
                        ]
                    )
                ),
            }
        )
        if matched_evidence is not None:
            diagnostics.update(
                {
                    "mean_gate_benefit_evidence": float(
                        np.mean(matched_evidence["benefit"][:, index, :])
                    ),
                    "mean_gate_specificity_evidence": float(
                        np.mean(matched_evidence["specificity"][:, index, :])
                    ),
                    "mean_gate_combined_probability": float(
                        np.mean(probability_matched[:, index, :])
                    ),
                }
            )
        if e1_ground_truth is not None and e1_evaluable is not None:
            diagnostics["e1_prior_value_groups"] = _e1_grouped_prior_value_diagnostics(
                labels=_horizon_slice(e1_ground_truth, horizon),
                evaluable=_horizon_slice(e1_evaluable, horizon),
                target=_horizon_slice(test["targets"], horizon),
                pred_all=_horizon_slice(test["pred_all"], horizon),
                pred_prior=_horizon_slice(test["pred_prior"], horizon),
                q_loss=_horizon_slice(targets_by_split["test"]["q_loss"], horizon),
                q_specificity=_horizon_slice(
                    targets_by_split["test"]["q_specificity"], horizon
                ),
                matched_soft_win_rate=_horizon_slice(
                    targets_by_split["test"]["matched_soft_win_rate"], horizon
                ),
                benefit_probability=_horizon_slice(matched_evidence["benefit"], horizon),
                specificity_probability=_horizon_slice(
                    matched_evidence["specificity"], horizon
                ),
                combined_probability=_horizon_slice(probability_matched, horizon),
            )
    metrics.update(
        {
            "schema": _artifact_schema(cfg, "selective_prior_injection_metrics"),
            "experiment_id": unit["experiment_id"],
            "experiment_title": unit["experiment_title"],
            "dataset": bundle.name,
            "seed": int(seed),
            "fingerprint": fingerprint,
            "git_commit": current_git_commit(),
            "config_sha256": config_sha256(config_path),
            "code_fingerprint": code_fingerprint(),
            "device": str(device),
            "history": int(unit["history"]),
            "max_horizon": int(unit["max_horizon"]),
            "report_horizons": [int(value) for value in unit["report_horizons"]],
            "control_exposures": exposures,
            "random_control_exposures": random_exposures,
            "matched_control_exposure_indices": matched_exposure_indices,
            "matched_control_probe_indices": matched_probe_indices,
            "random_control_exposure_indices": random_exposure_indices,
            "random_control_probe_indices": random_probe_indices,
            "all_only_component_reuse": all_only_reuse,
            "prior_graph_sha256": sha256_file(base_graph_path(bundle)),
            "matched_controls": control_provenance,
            "random_controls": random_control_provenance,
            "split_boundaries": bundle.boundaries,
            "window_counts": {name: len(dataset) for name, dataset in datasets.items()},
            "window_sampling_group": str(unit.get("window_sampling_group", bundle.name)),
            "window_origin_sha256s": window_origin_sha256s,
            "routing_contract": (
                "independent_frozen_experts"
                if is_v4
                else "shared_mask_forecaster_with_exposed_controls"
            ),
            "shared_forecaster_contract": {
                "routing_mode": str(
                    cfg["method"].get("routing_mode", "shared_mask_forecaster")
                ),
                "exposure_control_loss_weight": float(
                    cfg["method"].get("shared_control_loss_weight", 0.20)
                ),
                "validation_control_weight": float(
                    cfg["method"].get("shared_validation_control_weight", 0.0)
                ),
                "distillation_weight": float(
                    cfg["method"].get("shared_distillation_weight", 0.0)
                ),
                "checkpoint_selection": str(
                    cfg["method"].get(
                        "shared_checkpoint_selection", "mean_route_mae"
                    )
                ),
                "checkpoint_selection_split": str(
                    cfg["method"]["shared_checkpoint_selection_split"]
                ),
                "hard_preservation_gate_split": str(
                    cfg["method"]["shared_hard_preservation_gate_split"]
                ),
                "hard_preservation_metric_fields": (
                    []
                    if is_v4
                    else [
                        "shared_all_gate_validation_relative_degradation",
                        "shared_prior_gate_validation_relative_degradation",
                    ]
                ),
            },
            "gate_training_contract": {
                "targets": ["q_loss", "q_specificity"],
                "q_matched_use": "diagnostic_only",
                "specificity_lambda": float(
                    cfg["method"].get("specificity_gate_lambda", 0.30)
                ),
                "input_features": "local_global_horizon_plus_normalized_A_R_delta_and_magnitude",
                "head_contract": (
                    "separate_benefit_and_specificity_heads"
                    if is_v3 or is_v4
                    else "single_logit_two_losses"
                ),
                "combiner": str(
                    cfg["method"].get("gate_combiner", "legacy_single_logit")
                ),
                "probability_calibration": (
                    "validation_shrinkage_by_horizon" if is_v4 else "none"
                ),
            },
            **(
                {
                    "causal_aggregation_contract": {
                        "experts": list(
                            cfg["method"].get(
                                "causal_aggregation_experts",
                                ["all_only", "prior_only", "static_mix", "loss_gate"],
                            )
                        ),
                        "feedback": str(
                            cfg["method"].get(
                                "causal_feedback", "delayed_until_target_enters_history"
                            )
                        ),
                        "eta_grid": [
                            float(value)
                            for value in cfg["method"].get(
                                "causal_eta_grid", [0.01, 0.03, 0.1, 0.3, 1.0]
                            )
                        ],
                        "eta_selection_split": "gate_validation",
                        "test_selection_use": "none",
                        "validation_expert_predictions_are_calibrated": True,
                    }
                }
                if cfg.get("schema") in {"selective_prior_injection_v5", "selective_prior_injection_final_v1"}
                else {}
            ),
        }
    )
    if calibration is not None:
        metrics["gate_probability_calibration"] = calibration
    atomic_json(output / "metrics.json", metrics)
    atomic_json(
        output / "provenance.json",
        {
            "schema": _artifact_schema(cfg, "selective_prior_injection_provenance"),
            "fingerprint": fingerprint,
            "unit": jsonable_config(unit),
            "seed": int(seed),
            "series_sha256": sha256_file(bundle.root / "series.npz"),
            "prior_graph_sha256": sha256_file(base_graph_path(bundle)),
            "matched_controls": control_provenance,
            "random_controls": random_control_provenance,
            "matched_control_exposure_indices": matched_exposure_indices,
            "matched_control_probe_indices": matched_probe_indices,
            "random_control_exposure_indices": random_exposure_indices,
            "random_control_probe_indices": random_probe_indices,
            "all_only_component_reuse": all_only_reuse,
            "node_scale_source": "forecaster_train_0_to_60_percent_only",
            "test_selection_use": "none",
            "git_commit": current_git_commit(),
            "config_sha256": config_sha256(config_path),
            "code_fingerprint": code_fingerprint(),
            "window_sampling_group": str(unit.get("window_sampling_group", bundle.name)),
            "window_origin_sha256s": window_origin_sha256s,
            "routing_contract": (
                "independent_frozen_experts"
                if is_v4
                else "shared_mask_forecaster_with_exposed_controls"
            ),
            "shared_forecaster_contract": {
                "routing_mode": str(
                    cfg["method"].get("routing_mode", "shared_mask_forecaster")
                ),
                "exposure_control_loss_weight": float(
                    cfg["method"].get("shared_control_loss_weight", 0.20)
                ),
                "validation_control_weight": float(
                    cfg["method"].get("shared_validation_control_weight", 0.0)
                ),
                "distillation_weight": float(
                    cfg["method"].get("shared_distillation_weight", 0.0)
                ),
                "checkpoint_selection": str(
                    cfg["method"].get(
                        "shared_checkpoint_selection", "mean_route_mae"
                    )
                ),
                "checkpoint_selection_split": str(
                    cfg["method"]["shared_checkpoint_selection_split"]
                ),
                "hard_preservation_gate_split": str(
                    cfg["method"]["shared_hard_preservation_gate_split"]
                ),
                "hard_preservation_metric_fields": (
                    []
                    if is_v4
                    else [
                        "shared_all_gate_validation_relative_degradation",
                        "shared_prior_gate_validation_relative_degradation",
                    ]
                ),
            },
            "gate_training_contract": {
                "targets": ["q_loss", "q_specificity"],
                "q_matched_use": "diagnostic_only",
                "specificity_lambda": float(
                    cfg["method"].get("specificity_gate_lambda", 0.30)
                ),
                "input_features": "local_global_horizon_plus_normalized_A_R_delta_and_magnitude",
                "head_contract": (
                    "separate_benefit_and_specificity_heads"
                    if is_v3 or is_v4
                    else "single_logit_two_losses"
                ),
                "combiner": str(
                    cfg["method"].get("gate_combiner", "legacy_single_logit")
                ),
            },
        },
    )
    atomic_json(
        state_path,
        {
            "schema": _artifact_schema(cfg, "selective_prior_injection_state"),
            "status": "complete",
            "fingerprint": fingerprint,
            "git_commit": current_git_commit(),
            "config_sha256": config_sha256(config_path),
            "code_fingerprint": code_fingerprint(),
        },
    )
    return metrics


def run_matrix(
    cfg: dict[str, Any],
    config_path: Path,
    stage: str,
    *,
    experiment_ids: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if (
        stage == "formal"
        and cfg.get("schema") in {
            "selective_prior_injection_v3",
            "selective_prior_injection_v4",
        }
        and str(cfg.get("mode")) != "formal"
    ):
        raise RuntimeError(
            "The mechanism config cannot launch a formal matrix; first pass "
            "the Seed42 mechanism decision, then freeze a separate formal config"
        )
    if stage == "formal" and cfg.get("schema") == "selective_prior_injection_v2":
        # Keep the authorization inside the runner as well as the CLI preflight:
        # callers must not be able to bypass the mechanism gate programmatically.
        from .decision import verify_mechanism_authorization

        verify_mechanism_authorization(cfg, config_path)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    units_by_key = {
        (unit["experiment_id"], unit["dataset"]): unit
        for unit in _selected_units(cfg, stage, experiment_ids, datasets)
    }
    plan = matrix_plan(cfg, config_path, stage, experiment_ids, datasets, seed=seed)
    plan = [row for index, row in enumerate(plan) if index % num_shards == shard_index]
    if limit is not None:
        plan = plan[: int(limit)]
    results = []
    for row in plan:
        unit = units_by_key[(row["experiment_id"], row["dataset"])]
        results.append(run_one(cfg, config_path, unit, int(row["seed"]), resume=resume, overwrite=overwrite))
    return results


def summarize_runs(cfg: dict[str, Any]) -> Path:
    root = resolved_output_root(cfg)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("E*/*/seed_*/metrics.json")):
        metrics = _read_json(path)
        for horizon, content in metrics["horizons"].items():
            for method, values in content["methods"].items():
                rows.append(
                    {
                        "experiment_id": metrics["experiment_id"],
                        "dataset": metrics["dataset"],
                        "seed": metrics["seed"],
                        "horizon": int(horizon),
                        "method": method,
                        **values,
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No completed metrics under {root / 'runs'}")
    frame = pd.DataFrame(rows)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_dir / "per_seed_metrics.csv", index=False)
    group_columns = ["experiment_id", "dataset", "horizon", "method"]
    numeric = [column for column in ("mae", "rmse", "q95_mae", "q95_count") if column in frame.columns]
    aggregate = frame.groupby(group_columns, dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    aggregate.columns = ["_".join(str(value) for value in column if value) for column in aggregate.columns.to_flat_index()]
    aggregate.to_csv(summary_dir / "aggregate_metrics.csv", index=False)
    atomic_json(
        summary_dir / "summary_manifest.json",
        {
            "schema": _artifact_schema(cfg, "selective_prior_injection_summary"),
            "completed_seed_horizon_method_rows": len(frame),
            "per_seed_csv_sha256": sha256_file(summary_dir / "per_seed_metrics.csv"),
            "aggregate_csv_sha256": sha256_file(summary_dir / "aggregate_metrics.csv"),
        },
    )
    return summary_dir


def _required_development_runs(cfg: dict[str, Any]) -> list[Path]:
    paths = []
    for unit in units_for_stage(cfg, "formal"):
        for seed in seeds_for_stage(cfg, "formal"):
            paths.append(_unit_output(cfg, unit, seed) / "state.json")
    return paths


def create_study_lock(cfg: dict[str, Any], config_path: Path) -> Path:
    if cfg.get("schema") == "selective_prior_injection_v2":
        from .decision import verify_mechanism_authorization

        verify_mechanism_authorization(cfg, config_path)
    matrix_version = validate_matrix_versions(cfg, config_path, stage="formal")
    missing = []
    state_hashes = []
    for state_path in _required_development_runs(cfg):
        metrics_path = state_path.parent / "metrics.json"
        provenance_path = state_path.parent / "provenance.json"
        if (
            not state_path.exists()
            or _read_json(state_path).get("status") != "complete"
            or not metrics_path.exists()
            or not provenance_path.exists()
        ):
            missing.append(str(state_path))
        else:
            state = _read_json(state_path)
            metrics = _read_json(metrics_path)
            provenance = _read_json(provenance_path)
            fingerprints = {state.get("fingerprint"), metrics.get("fingerprint"), provenance.get("fingerprint")}
            if len(fingerprints) != 1:
                missing.append(f"fingerprint mismatch in {state_path.parent}")
                continue
            state_hashes.append(
                {
                    "path": str(state_path),
                    "state_sha256": sha256_file(state_path),
                    "metrics_sha256": sha256_file(metrics_path),
                    "provenance_sha256": sha256_file(provenance_path),
                    "fingerprint": state.get("fingerprint"),
                }
            )
    if missing:
        raise RuntimeError(f"Cannot lock before all E1-E9 formal runs complete; missing {len(missing)} states")
    decision_path = resolved_output_root(cfg) / "analysis" / "go_no_go.json"
    if not decision_path.exists():
        raise RuntimeError("Cannot lock before the predeclared go/no-go decision is written")
    decision = _read_json(decision_path)
    if decision.get("proceed_external") is not True:
        raise RuntimeError("The predeclared E1-E9 go/no-go gates do not authorize E10")
    repository = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise RuntimeError("Study lock requires a clean Git worktree")
    external_data = []
    for unit in units_for_stage(cfg, "external"):
        bundle = load_bundle(resolved_data_root(cfg), str(unit["dataset"]), cfg["split"])
        controls_manifest = bundle.root / "spi_controls" / "manifest.json"
        if not controls_manifest.exists():
            raise FileNotFoundError(f"Prepare E10 controls before locking: {controls_manifest}")
        external_data.append(
            {
                "dataset": bundle.name,
                "series_sha256": sha256_file(bundle.root / "series.npz"),
                "prior_graph_sha256": sha256_file(base_graph_path(bundle)),
                "controls_manifest_sha256": sha256_file(controls_manifest),
            }
        )
    version = protocol_version(cfg)
    lock = {
        "schema": f"selective_prior_injection_lock_{version}",
        "selection_locked": True,
        "git_commit": current_git_commit(),
        "config_sha256": config_sha256(config_path),
        "code_fingerprint": code_fingerprint(),
        "development_matrix_version": {
            key: matrix_version[key]
            for key in (
                "run_count",
                "config_sha256",
                "code_fingerprint",
                "git_commits_recorded",
                "validation_git_commit",
            )
        },
        "development_states": state_hashes,
        "decision_sha256": sha256_file(decision_path),
        "external_data": external_data,
    }
    path = lock_path(cfg)
    atomic_json(path, lock)
    return path
