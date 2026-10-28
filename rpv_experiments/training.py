from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rpv_data.utils import atomic_json

from .data import DatasetBundle, WindowDataset
from .models import (
    MaskedVariableForecaster,
    MatchedPriorGate,
    PriorGate,
    matched_prior_probability,
)


def seed_everything(seed: int) -> None:
    bounded = int(seed) % (2**32)
    random.seed(bounded)
    np.random.seed(bounded)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    name = str(requested).lower()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Protocol requests CUDA, but CUDA is unavailable; CPU fallback is forbidden")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be cuda or cpu")


def _loader(dataset: WindowDataset, training: dict[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(training["micro_batch_size"]),
        shuffle=bool(shuffle),
        num_workers=int(training.get("num_workers", 0)),
        drop_last=False,
        pin_memory=False,
        generator=generator,
    )


def build_forecaster(bundle: DatasetBundle, history: int, horizon: int, training: dict[str, Any]) -> MaskedVariableForecaster:
    return MaskedVariableForecaster(
        history=int(history),
        horizon=int(horizon),
        variables=bundle.variables,
        hidden=int(training["hidden"]),
        layers=int(training["layers"]),
        heads=int(training["heads"]),
        dropout=float(training["dropout"]),
    )


def _validation_mae(
    model: MaskedVariableForecaster,
    dataset: WindowDataset,
    mask: torch.Tensor,
    training: dict[str, Any],
    device: torch.device,
) -> float:
    loader = _loader(dataset, training, shuffle=False, seed=0)
    total = count = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x, mask)
            total += float(torch.abs(prediction - y).sum().item())
            count += int(y.numel())
    if count == 0:
        raise RuntimeError("Empty validation loader")
    return total / count


def _validation_raw_mae(
    model: MaskedVariableForecaster,
    dataset: WindowDataset,
    mask: torch.Tensor,
    training: dict[str, Any],
    device: torch.device,
) -> float:
    """Validation MAE in the reported physical scale.

    Per-node standardization makes a normalized aggregate MAE a different
    objective from the raw-scale preservation gate used by the experiment.
    Protocol v3 selects on ``forecaster_validation`` and reports the hard gate
    on ``gate_validation``. Both use this raw-scale definition, while the
    distinct splits keep checkpoint selection separate from verification.
    """
    loader = _loader(dataset, training, shuffle=False, seed=0)
    scale = torch.as_tensor(
        dataset.bundle.std,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)
    total = count = 0
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x, mask)
            total += float((torch.abs(prediction - y) * scale).sum().item())
            count += int(y.numel())
    if count == 0:
        raise RuntimeError("Empty validation loader")
    return total / count


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
    accumulated_micro_batches: int = 1,
) -> None:
    if accumulated_micro_batches < 1:
        raise ValueError("accumulated_micro_batches must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(accumulated_micro_batches))
    clip = float(training.get("grad_clip", 0.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _branch_features(
    pred_all_raw: np.ndarray,
    pred_prior_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Return normalized A/R disagreement features as ``[origin,node,2*h]``."""
    all_array = np.asarray(pred_all_raw, dtype=np.float32)
    prior_array = np.asarray(pred_prior_raw, dtype=np.float32)
    if all_array.ndim != 3 or all_array.shape != prior_array.shape:
        raise ValueError("Branch predictions must share [origin,horizon,node] shape")
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32).view(1, 1, -1)
    std_tensor = torch.as_tensor(std, dtype=torch.float32).view(1, 1, -1)
    all_normalized = (torch.as_tensor(all_array) - mean_tensor) / std_tensor
    prior_normalized = (torch.as_tensor(prior_array) - mean_tensor) / std_tensor
    delta = (prior_normalized - all_normalized).transpose(1, 2)
    magnitude = torch.abs(delta)
    return torch.cat([delta, magnitude], dim=-1).contiguous()


def train_fixed_forecaster(
    bundle: DatasetBundle,
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    mask: torch.Tensor,
    training: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    fingerprint: str,
    resume: bool,
) -> MaskedVariableForecaster:
    seed_everything(seed)
    model = build_forecaster(bundle, train_dataset.history, train_dataset.horizon, training).to(device)
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {checkpoint_path}")
        if checkpoint.get("schema") != "fixed_forecaster_v2":
            raise RuntimeError(
                "Fixed forecaster checkpoint does not implement the active checkpoint contract: "
                f"{checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model"])
        return model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.MSELoss()
    best = float("inf")
    best_epoch = -1
    patience = int(training["patience"])
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        loader = _loader(train_dataset, training, shuffle=True, seed=seed + epoch * 104729)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_count = 0
        epoch_loss = 0.0
        samples = 0
        for batch_index, (x, y, _) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x, mask), y)
            loss.backward()
            micro_count += 1
            epoch_loss += float(loss.item()) * int(x.shape[0])
            samples += int(x.shape[0])
            if micro_count == int(training["gradient_accumulation_steps"]) or batch_index == len(loader) - 1:
                _optimizer_step(model, optimizer, training, micro_count)
                micro_count = 0
        validation = _validation_mae(model, validation_dataset, mask, training, device)
        history_rows.append({"epoch": epoch, "train_mse": epoch_loss / max(samples, 1), "validation_mae": validation})
        if validation < best - 1e-10:
            best, best_epoch, patience = validation, epoch, int(training["patience"])
            _save_checkpoint(
                checkpoint_path,
                {
                    "schema": "fixed_forecaster_v2",
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_mae": best,
                    "fingerprint": fingerprint,
                },
            )
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_epoch < 0:
        raise RuntimeError("Fixed forecaster produced no checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    atomic_json(checkpoint_path.with_suffix(".history.json"), {"best_epoch": best_epoch, "epochs": history_rows})
    return model


def train_shared_forecaster(
    bundle: DatasetBundle,
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    mask_all: torch.Tensor,
    mask_prior: torch.Tensor,
    matched_masks: list[torch.Tensor],
    validation_control_indices: list[int],
    training: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    fingerprint: str,
    resume: bool,
    control_loss_weight: float = 0.20,
    validation_control_weight: float = 0.0,
    teacher_all: MaskedVariableForecaster | None = None,
    teacher_prior: MaskedVariableForecaster | None = None,
    distillation_weight: float = 0.0,
    checkpoint_selection: str = "mean_route_mae",
    checkpoint_schema: str = "shared_mask_forecaster_v2",
) -> tuple[MaskedVariableForecaster, list[int]]:
    seed_everything(seed)
    control_loss_weight = float(control_loss_weight)
    validation_control_weight = float(validation_control_weight)
    if not 0.0 <= control_loss_weight <= 1.0:
        raise ValueError("control_loss_weight must be in [0,1]")
    if not 0.0 <= validation_control_weight <= 1.0:
        raise ValueError("validation_control_weight must be in [0,1]")
    distillation_weight = float(distillation_weight)
    if not 0.0 <= distillation_weight <= 1.0:
        raise ValueError("distillation_weight must be in [0,1]")
    if checkpoint_selection not in {"mean_route_mae", "minimax_raw_relative_degradation"}:
        raise ValueError(f"Unknown shared checkpoint selection: {checkpoint_selection}")
    if (teacher_all is None) != (teacher_prior is None):
        raise ValueError("A-only and R-only teachers must be provided together")
    if distillation_weight > 0.0 and teacher_all is None:
        raise ValueError("Shared distillation requires fixed A-only and R-only teachers")
    if checkpoint_selection == "minimax_raw_relative_degradation" and teacher_all is None:
        raise ValueError("Minimax preservation selection requires fixed branch teachers")
    if not matched_masks:
        raise ValueError("At least one exposure control is required")
    model = build_forecaster(bundle, train_dataset.history, train_dataset.horizon, training).to(device)
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {checkpoint_path}")
        if checkpoint.get("schema") != checkpoint_schema:
            raise RuntimeError(
                "Shared forecaster checkpoint does not implement the active exposure/checkpoint contract: "
                f"{checkpoint_path}"
            )
        if abs(float(checkpoint.get("control_loss_weight", -1.0)) - control_loss_weight) > 1e-12:
            raise RuntimeError(f"Shared forecaster control-loss contract mismatch: {checkpoint_path}")
        if abs(float(checkpoint.get("validation_control_weight", -1.0)) - validation_control_weight) > 1e-12:
            raise RuntimeError(f"Shared forecaster validation-selection contract mismatch: {checkpoint_path}")
        if abs(float(checkpoint.get("distillation_weight", -1.0)) - distillation_weight) > 1e-12:
            raise RuntimeError(f"Shared forecaster distillation contract mismatch: {checkpoint_path}")
        if checkpoint.get("checkpoint_selection") != checkpoint_selection:
            raise RuntimeError(f"Shared forecaster checkpoint-selection mismatch: {checkpoint_path}")
        saved_exposures = [int(value) for value in checkpoint.get("control_exposures", [])]
        if len(saved_exposures) != len(matched_masks) or (
            saved_exposures and max(saved_exposures) - min(saved_exposures) > 1
        ):
            raise RuntimeError(f"Shared forecaster control exposure contract mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
        return model, saved_exposures
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.MSELoss()
    best_key = (float("inf"), float("inf"))
    best_epoch = -1
    patience = int(training["patience"])
    counter = 0
    exposures = [0 for _ in matched_masks]
    history_rows: list[dict[str, float | int]] = []
    teacher_all_raw_mae = teacher_prior_raw_mae = float("nan")
    if teacher_all is not None and teacher_prior is not None:
        for teacher in (teacher_all, teacher_prior):
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        teacher_all_raw_mae = _validation_raw_mae(
            teacher_all, validation_dataset, mask_all, training, device
        )
        teacher_prior_raw_mae = _validation_raw_mae(
            teacher_prior, validation_dataset, mask_prior, training, device
        )
    for epoch in range(1, int(training["epochs"]) + 1):
        loader = _loader(train_dataset, training, shuffle=True, seed=seed + epoch * 104729)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_count = 0
        epoch_loss = 0.0
        samples = 0
        for batch_index, (x, y, _) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            control_index = counter % len(matched_masks)
            counter += 1
            exposures[control_index] += 1
            prediction_all = model(x, mask_all)
            prediction_prior = model(x, mask_prior)
            route_loss = (
                criterion(prediction_all, y) + criterion(prediction_prior, y)
            ) / 2.0
            preservation_loss = torch.zeros((), device=device)
            if teacher_all is not None and teacher_prior is not None:
                with torch.no_grad():
                    target_all = teacher_all(x, mask_all)
                    target_prior = teacher_prior(x, mask_prior)
                preservation_loss = (
                    criterion(prediction_all, target_all)
                    + criterion(prediction_prior, target_prior)
                ) / 2.0
            route_objective = (
                (1.0 - distillation_weight) * route_loss
                + distillation_weight * preservation_loss
            )
            control_loss = criterion(model(x, matched_masks[control_index]), y)
            # Controls remain in the exposure distribution, but cannot
            # consume the majority of the shared model's optimization budget.
            loss = (
                (1.0 - control_loss_weight) * route_objective
                + control_loss_weight * control_loss
            )
            loss.backward()
            micro_count += 1
            epoch_loss += float(loss.item()) * int(x.shape[0])
            samples += int(x.shape[0])
            if micro_count == int(training["gradient_accumulation_steps"]) or batch_index == len(loader) - 1:
                _optimizer_step(model, optimizer, training, micro_count)
                micro_count = 0
        validation_all = _validation_mae(model, validation_dataset, mask_all, training, device)
        validation_prior = _validation_mae(model, validation_dataset, mask_prior, training, device)
        validation_route = (validation_all + validation_prior) / 2.0
        validation_all_raw = _validation_raw_mae(
            model, validation_dataset, mask_all, training, device
        )
        validation_prior_raw = _validation_raw_mae(
            model, validation_dataset, mask_prior, training, device
        )
        all_relative_degradation = (
            (validation_all_raw - teacher_all_raw_mae)
            / max(teacher_all_raw_mae, 1e-12)
            if np.isfinite(teacher_all_raw_mae)
            else float("nan")
        )
        prior_relative_degradation = (
            (validation_prior_raw - teacher_prior_raw_mae)
            / max(teacher_prior_raw_mae, 1e-12)
            if np.isfinite(teacher_prior_raw_mae)
            else float("nan")
        )
        validation_controls = float("nan")
        if validation_control_indices:
            validation_controls = float(
                np.mean([
                    _validation_mae(model, validation_dataset, matched_masks[index], training, device)
                    for index in validation_control_indices
                ])
            )
        validation = (1.0 - validation_control_weight) * validation_route
        if np.isfinite(validation_controls):
            validation += validation_control_weight * validation_controls
        if checkpoint_selection == "minimax_raw_relative_degradation":
            selection_key = (
                max(all_relative_degradation, prior_relative_degradation),
                (validation_all_raw + validation_prior_raw) / 2.0,
            )
        else:
            selection_key = (validation, validation_route)
        history_rows.append(
            {
                "epoch": epoch,
                "train_balanced_mse": epoch_loss / max(samples, 1),
                "validation_all_mae": validation_all,
                "validation_prior_mae": validation_prior,
                "validation_route_mae": validation_route,
                "validation_controls_mae": validation_controls,
                "validation_selection_loss": validation,
                "validation_all_raw_mae": validation_all_raw,
                "validation_prior_raw_mae": validation_prior_raw,
                "teacher_all_raw_mae": teacher_all_raw_mae,
                "teacher_prior_raw_mae": teacher_prior_raw_mae,
                "validation_all_relative_degradation": all_relative_degradation,
                "validation_prior_relative_degradation": prior_relative_degradation,
                "selection_primary": selection_key[0],
                "selection_secondary": selection_key[1],
            }
        )
        if selection_key < best_key:
            best_key, best_epoch, patience = selection_key, epoch, int(training["patience"])
            _save_checkpoint(
                checkpoint_path,
                {
                    "schema": checkpoint_schema,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_composite_mae": validation,
                    "validation_route_mae": validation_route,
                    "validation_controls_mae": validation_controls,
                    "validation_all_raw_mae": validation_all_raw,
                    "validation_prior_raw_mae": validation_prior_raw,
                    "teacher_all_raw_mae": teacher_all_raw_mae,
                    "teacher_prior_raw_mae": teacher_prior_raw_mae,
                    "validation_all_relative_degradation": all_relative_degradation,
                    "validation_prior_relative_degradation": prior_relative_degradation,
                    "control_loss_weight": control_loss_weight,
                    "validation_control_weight": validation_control_weight,
                    "distillation_weight": distillation_weight,
                    "checkpoint_selection": checkpoint_selection,
                    "control_exposures": exposures,
                    "fingerprint": fingerprint,
                },
            )
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_epoch < 0:
        raise RuntimeError("Shared forecaster produced no checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    saved_exposures = [int(value) for value in checkpoint["control_exposures"]]
    if max(saved_exposures) - min(saved_exposures) > 1:
        raise RuntimeError("Round-robin control exposure is imbalanced")
    atomic_json(checkpoint_path.with_suffix(".history.json"), {"best_epoch": best_epoch, "epochs": history_rows})
    return model, saved_exposures


def predict_raw(
    model: MaskedVariableForecaster,
    dataset: WindowDataset,
    mask: torch.Tensor,
    training: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = _loader(dataset, training, shuffle=False, seed=0)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            prediction = model(x.to(device), mask).cpu().numpy()
            targets.append(dataset.bundle.inverse(y.numpy()))
            predictions.append(dataset.bundle.inverse(prediction))
    return np.concatenate(targets).astype(np.float32), np.concatenate(predictions).astype(np.float32)


def train_prior_gate(
    shared: MaskedVariableForecaster,
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    train_pred_all_raw: np.ndarray,
    train_pred_prior_raw: np.ndarray,
    validation_pred_all_raw: np.ndarray,
    validation_pred_prior_raw: np.ndarray,
    train_q_loss: np.ndarray,
    validation_q_loss: np.ndarray,
    gate_lambda: float,
    training: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    fingerprint: str,
    resume: bool,
    train_q_specificity: np.ndarray | None = None,
    validation_q_specificity: np.ndarray | None = None,
    specificity_lambda: float = 0.0,
    checkpoint_schema: str = "prior_gate_v2",
) -> PriorGate:
    seed_everything(seed)
    gate = PriorGate(
        shared.hidden,
        shared.horizon,
        dropout=float(training["dropout"]),
        branch_feature_dim=2 * shared.horizon,
    ).to(device)
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Gate checkpoint fingerprint mismatch: {checkpoint_path}")
        if checkpoint.get("schema") != checkpoint_schema or int(
            checkpoint.get("branch_feature_dim", -1)
        ) != 2 * shared.horizon:
            raise RuntimeError(
                "Gate checkpoint does not implement the v2 branch-evidence input contract: "
                f"{checkpoint_path}"
            )
        gate.load_state_dict(checkpoint["model"])
        return gate
    for parameter in shared.parameters():
        parameter.requires_grad_(False)
    shared.eval()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    bce = nn.BCEWithLogitsLoss()
    mean = torch.as_tensor(train_dataset.bundle.mean, dtype=torch.float32).view(1, 1, -1)
    std = torch.as_tensor(train_dataset.bundle.std, dtype=torch.float32).view(1, 1, -1)

    def normalized(array: np.ndarray) -> torch.Tensor:
        return (torch.as_tensor(array, dtype=torch.float32) - mean) / std

    train_all = normalized(train_pred_all_raw)
    train_prior = normalized(train_pred_prior_raw)
    val_all = normalized(validation_pred_all_raw)
    val_prior = normalized(validation_pred_prior_raw)
    train_q_loss_tensor = torch.as_tensor(train_q_loss, dtype=torch.float32)
    val_q_loss_tensor = torch.as_tensor(validation_q_loss, dtype=torch.float32)
    train_q_specificity_tensor = (
        torch.as_tensor(train_q_specificity, dtype=torch.float32)
        if train_q_specificity is not None
        else None
    )
    val_q_specificity_tensor = (
        torch.as_tensor(validation_q_specificity, dtype=torch.float32)
        if validation_q_specificity is not None
        else None
    )
    if (train_q_specificity_tensor is None) != (val_q_specificity_tensor is None):
        raise ValueError("train and validation specificity targets must be provided together")
    specificity_lambda = float(specificity_lambda)
    if specificity_lambda < 0.0:
        raise ValueError("specificity_lambda must be non-negative")
    best = float("inf")
    best_epoch = -1
    patience = int(training["patience"])
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        loader = _loader(train_dataset, training, shuffle=True, seed=seed + epoch * 65537)
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        micro_count = 0
        epoch_loss = 0.0
        samples = 0
        for batch_index, (x, y, index) in enumerate(loader):
            branch_all = train_all[index].to(device)
            branch_prior = train_prior[index].to(device)
            target_q_loss = train_q_loss_tensor[index].to(device)
            target_q_specificity = (
                train_q_specificity_tensor[index].to(device)
                if train_q_specificity_tensor is not None
                else None
            )
            branch_delta = (branch_prior - branch_all).transpose(1, 2)
            branch_features = torch.cat([branch_delta, torch.abs(branch_delta)], dim=-1)
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = shared.local_features(x)
            logits = gate(features, branch_features)
            probability = torch.sigmoid(logits)
            mixed = probability * branch_prior + (1.0 - probability) * branch_all
            loss = torch.mean((mixed - y) ** 2) + float(gate_lambda) * bce(logits, target_q_loss)
            if target_q_specificity is not None and specificity_lambda > 0.0:
                loss = loss + specificity_lambda * bce(logits, target_q_specificity)
            loss.backward()
            micro_count += 1
            if micro_count == int(training["gradient_accumulation_steps"]) or batch_index == len(loader) - 1:
                _optimizer_step(gate, optimizer, training, micro_count)
                micro_count = 0
            epoch_loss += float(loss.item()) * int(x.shape[0])
            samples += int(x.shape[0])
        gate.eval()
        validation_total = validation_count = 0
        with torch.no_grad():
            for x, y, index in _loader(validation_dataset, training, shuffle=False, seed=0):
                branch_all = val_all[index].to(device)
                branch_prior = val_prior[index].to(device)
                target_q_loss = val_q_loss_tensor[index].to(device)
                target_q_specificity = (
                    val_q_specificity_tensor[index].to(device)
                    if val_q_specificity_tensor is not None
                    else None
                )
                branch_delta = (branch_prior - branch_all).transpose(1, 2)
                branch_features = torch.cat([branch_delta, torch.abs(branch_delta)], dim=-1)
                x, y = x.to(device), y.to(device)
                logits = gate(shared.local_features(x), branch_features)
                probability = torch.sigmoid(logits)
                mixed = probability * branch_prior + (1.0 - probability) * branch_all
                loss = torch.mean((mixed - y) ** 2) + float(gate_lambda) * bce(logits, target_q_loss)
                if target_q_specificity is not None and specificity_lambda > 0.0:
                    loss = loss + specificity_lambda * bce(logits, target_q_specificity)
                validation_total += float(loss.item()) * int(x.shape[0])
                validation_count += int(x.shape[0])
        validation = validation_total / max(validation_count, 1)
        history_rows.append({"epoch": epoch, "train_gate_loss": epoch_loss / max(samples, 1), "validation_gate_loss": validation})
        if validation < best - 1e-10:
            best, best_epoch, patience = validation, epoch, int(training["patience"])
            _save_checkpoint(
                checkpoint_path,
                {
                    "schema": checkpoint_schema,
                    "model": gate.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best,
                    "specificity_lambda": specificity_lambda,
                    "branch_feature_dim": 2 * shared.horizon,
                    "fingerprint": fingerprint,
                },
            )
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_epoch < 0:
        raise RuntimeError("Gate produced no checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gate.load_state_dict(checkpoint["model"])
    atomic_json(checkpoint_path.with_suffix(".history.json"), {"best_epoch": best_epoch, "epochs": history_rows})
    return gate


def train_matched_prior_gate(
    shared: MaskedVariableForecaster,
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    train_pred_all_raw: np.ndarray,
    train_pred_prior_raw: np.ndarray,
    validation_pred_all_raw: np.ndarray,
    validation_pred_prior_raw: np.ndarray,
    train_q_loss: np.ndarray,
    validation_q_loss: np.ndarray,
    train_q_specificity: np.ndarray,
    validation_q_specificity: np.ndarray,
    gate_lambda: float,
    specificity_lambda: float,
    training: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    fingerprint: str,
    resume: bool,
    checkpoint_schema: str = "matched_prior_gate_v3",
    combiner: str = "geometric_mean",
) -> MatchedPriorGate:
    """Train v3 benefit and specificity heads without a shared-logit compromise."""
    seed_everything(seed)
    gate = MatchedPriorGate(
        shared.hidden,
        shared.horizon,
        dropout=float(training["dropout"]),
        branch_feature_dim=2 * shared.horizon,
    ).to(device)
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Gate checkpoint fingerprint mismatch: {checkpoint_path}")
        if checkpoint.get("schema") != checkpoint_schema:
            raise RuntimeError(f"Matched gate checkpoint schema mismatch: {checkpoint_path}")
        if checkpoint.get("combiner") != combiner:
            raise RuntimeError(f"Matched gate combiner mismatch: {checkpoint_path}")
        gate.load_state_dict(checkpoint["model"])
        return gate
    for parameter in shared.parameters():
        parameter.requires_grad_(False)
    shared.eval()
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    bce = nn.BCEWithLogitsLoss()
    mean = torch.as_tensor(train_dataset.bundle.mean, dtype=torch.float32).view(1, 1, -1)
    std = torch.as_tensor(train_dataset.bundle.std, dtype=torch.float32).view(1, 1, -1)

    def normalized(array: np.ndarray) -> torch.Tensor:
        return (torch.as_tensor(array, dtype=torch.float32) - mean) / std

    train_all = normalized(train_pred_all_raw)
    train_prior = normalized(train_pred_prior_raw)
    validation_all = normalized(validation_pred_all_raw)
    validation_prior = normalized(validation_pred_prior_raw)
    train_loss_target = torch.as_tensor(train_q_loss, dtype=torch.float32)
    validation_loss_target = torch.as_tensor(validation_q_loss, dtype=torch.float32)
    train_specificity_target = torch.as_tensor(train_q_specificity, dtype=torch.float32)
    validation_specificity_target = torch.as_tensor(
        validation_q_specificity, dtype=torch.float32
    )
    gate_lambda = float(gate_lambda)
    specificity_lambda = float(specificity_lambda)
    if gate_lambda < 0.0 or specificity_lambda < 0.0:
        raise ValueError("Matched gate supervision weights must be non-negative")
    best = float("inf")
    best_epoch = -1
    patience = int(training["patience"])
    history_rows: list[dict[str, float | int]] = []

    def batch_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor,
        branch_all: torch.Tensor,
        branch_prior: torch.Tensor,
        benefit_target: torch.Tensor,
        specificity_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_all = branch_all[indices].to(device)
        selected_prior = branch_prior[indices].to(device)
        branch_delta = (selected_prior - selected_all).transpose(1, 2)
        branch_features = torch.cat([branch_delta, torch.abs(branch_delta)], dim=-1)
        with torch.no_grad():
            local = shared.local_features(x)
        benefit_logits, specificity_logits = gate(local, branch_features)
        probability = matched_prior_probability(benefit_logits, specificity_logits)
        mixed = probability * selected_prior + (1.0 - probability) * selected_all
        prediction_loss = torch.mean((mixed - y) ** 2)
        benefit_loss = bce(benefit_logits, benefit_target[indices].to(device))
        specificity_loss = bce(
            specificity_logits,
            specificity_target[indices].to(device),
        )
        total = (
            prediction_loss
            + gate_lambda * benefit_loss
            + specificity_lambda * specificity_loss
        )
        return total, benefit_loss, specificity_loss

    for epoch in range(1, int(training["epochs"]) + 1):
        loader = _loader(train_dataset, training, shuffle=True, seed=seed + epoch * 65537)
        gate.train()
        optimizer.zero_grad(set_to_none=True)
        micro_count = 0
        epoch_total = epoch_benefit = epoch_specificity = 0.0
        samples = 0
        for batch_index, (x, y, index) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            total, benefit_loss, specificity_loss = batch_loss(
                x,
                y,
                index,
                train_all,
                train_prior,
                train_loss_target,
                train_specificity_target,
            )
            total.backward()
            micro_count += 1
            if micro_count == int(training["gradient_accumulation_steps"]) or batch_index == len(loader) - 1:
                _optimizer_step(gate, optimizer, training, micro_count)
                micro_count = 0
            batch_size = int(x.shape[0])
            epoch_total += float(total.item()) * batch_size
            epoch_benefit += float(benefit_loss.item()) * batch_size
            epoch_specificity += float(specificity_loss.item()) * batch_size
            samples += batch_size
        gate.eval()
        validation_total = validation_benefit = validation_specificity = 0.0
        validation_count = 0
        with torch.no_grad():
            for x, y, index in _loader(validation_dataset, training, shuffle=False, seed=0):
                x, y = x.to(device), y.to(device)
                total, benefit_loss, specificity_loss = batch_loss(
                    x,
                    y,
                    index,
                    validation_all,
                    validation_prior,
                    validation_loss_target,
                    validation_specificity_target,
                )
                batch_size = int(x.shape[0])
                validation_total += float(total.item()) * batch_size
                validation_benefit += float(benefit_loss.item()) * batch_size
                validation_specificity += float(specificity_loss.item()) * batch_size
                validation_count += batch_size
        validation = validation_total / max(validation_count, 1)
        history_rows.append(
            {
                "epoch": epoch,
                "train_gate_loss": epoch_total / max(samples, 1),
                "train_benefit_bce": epoch_benefit / max(samples, 1),
                "train_specificity_bce": epoch_specificity / max(samples, 1),
                "validation_gate_loss": validation,
                "validation_benefit_bce": validation_benefit / max(validation_count, 1),
                "validation_specificity_bce": validation_specificity / max(validation_count, 1),
            }
        )
        if validation < best - 1e-10:
            best, best_epoch, patience = validation, epoch, int(training["patience"])
            _save_checkpoint(
                checkpoint_path,
                {
                    "schema": checkpoint_schema,
                    "model": gate.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best,
                    "gate_lambda": gate_lambda,
                    "specificity_lambda": specificity_lambda,
                    "branch_feature_dim": 2 * shared.horizon,
                    "combiner": combiner,
                    "fingerprint": fingerprint,
                },
            )
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_epoch < 0:
        raise RuntimeError("Matched prior gate produced no checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gate.load_state_dict(checkpoint["model"])
    atomic_json(
        checkpoint_path.with_suffix(".history.json"),
        {"best_epoch": best_epoch, "epochs": history_rows},
    )
    return gate


def matched_gate_probabilities(
    shared: MaskedVariableForecaster,
    gate: MatchedPriorGate,
    dataset: WindowDataset,
    training: dict[str, Any],
    device: torch.device,
    pred_all_raw: np.ndarray,
    pred_prior_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return both evidence heads and their v3 mixture coefficient."""
    branch_features = _branch_features(
        pred_all_raw,
        pred_prior_raw,
        dataset.bundle.mean,
        dataset.bundle.std,
    )
    if len(branch_features) != len(dataset):
        raise ValueError("Matched gate branch features are not aligned with the dataset")
    benefit_rows: list[np.ndarray] = []
    specificity_rows: list[np.ndarray] = []
    combined_rows: list[np.ndarray] = []
    shared.eval()
    gate.eval()
    with torch.no_grad():
        for x, _, index in _loader(dataset, training, shuffle=False, seed=0):
            local = shared.local_features(x.to(device))
            branch = branch_features[index].to(device)
            benefit_logits, specificity_logits = gate(local, branch)
            benefit = torch.sigmoid(benefit_logits)
            specificity = torch.sigmoid(specificity_logits)
            combined = matched_prior_probability(benefit_logits, specificity_logits)
            benefit_rows.append(benefit.cpu().numpy())
            specificity_rows.append(specificity.cpu().numpy())
            combined_rows.append(combined.cpu().numpy())
    return {
        "benefit": np.concatenate(benefit_rows).astype(np.float32),
        "specificity": np.concatenate(specificity_rows).astype(np.float32),
        "combined": np.concatenate(combined_rows).astype(np.float32),
    }


def gate_probabilities(
    shared: MaskedVariableForecaster,
    gate: PriorGate,
    dataset: WindowDataset,
    training: dict[str, Any],
    device: torch.device,
    pred_all_raw: np.ndarray,
    pred_prior_raw: np.ndarray,
) -> np.ndarray:
    branch_features = _branch_features(
        pred_all_raw,
        pred_prior_raw,
        dataset.bundle.mean,
        dataset.bundle.std,
    )
    if len(branch_features) != len(dataset):
        raise ValueError("Gate branch features are not aligned with the dataset windows")
    probabilities: list[np.ndarray] = []
    shared.eval()
    gate.eval()
    with torch.no_grad():
        for x, _, index in _loader(dataset, training, shuffle=False, seed=0):
            features = shared.local_features(x.to(device))
            branch = branch_features[index].to(device)
            probabilities.append(torch.sigmoid(gate(features, branch)).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)
