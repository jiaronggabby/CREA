"""Causal, delayed-feedback aggregation of frozen forecasting experts.

The aggregator is deliberately post-training.  It never sees the current
forecast target before emitting a prediction: for horizon ``h`` the loss from
an origin is applied only after its target timestamp enters the next origin's
history.  The update-rate grid is selected on ``gate_validation`` only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def robust_error_scale(target: np.ndarray) -> np.ndarray:
    """Return a train/validation-only scale for bounded expert losses.

    The scale is deliberately a robust statistic of the supplied split.  The
    caller must compute it on a pre-test split and reuse it for the locked
    test stream; deriving it from test targets would leak future distribution
    information into the online update rule.
    """

    value = np.asarray(target, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("target must be [origin,horizon,node]")
    return np.maximum(
        np.median(np.abs(value - np.median(value, axis=0)), axis=0) + 1e-3,
        1e-6,
    )


def _validate_inputs(
    target: np.ndarray,
    expert_predictions: Sequence[np.ndarray],
    origins: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    y = np.asarray(target, dtype=np.float64)
    predictions = [np.asarray(value, dtype=np.float64) for value in expert_predictions]
    if y.ndim != 3 or not predictions:
        raise ValueError("target must be [origin,horizon,node] and experts cannot be empty")
    if any(value.shape != y.shape for value in predictions):
        raise ValueError("Every expert prediction must match target shape")
    order = np.asarray(origins, dtype=np.int64).reshape(-1)
    if order.shape[0] != y.shape[0] or np.any(np.diff(order) < 0):
        raise ValueError("origins must be sorted and aligned with target rows")
    return y, predictions, order


def causal_hedge_mix(
    target: np.ndarray,
    expert_predictions: Sequence[np.ndarray],
    origins: np.ndarray,
    eta_by_horizon: Sequence[float],
    *,
    error_scale: np.ndarray | None = None,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal mixture predictions and expert weights.

    ``target`` is used only to update weights after feedback becomes
    available.  For a forecast made at origin ``s``, horizon ``h`` targets
    timestamp ``s+h`` (the first target has offset zero); that loss is applied
    only at a later origin ``s' > s+h``.  This handles overlapping windows and
    avoids future leakage for multi-step forecasting.
    """

    y, predictions, order = _validate_inputs(target, expert_predictions, origins)
    experts, horizon, nodes = len(predictions), y.shape[1], y.shape[2]
    eta = np.asarray(eta_by_horizon, dtype=np.float64).reshape(-1)
    if eta.shape != (horizon,) or np.any(eta < 0.0):
        raise ValueError("eta_by_horizon must have one non-negative value per horizon")
    if error_scale is None:
        scale = robust_error_scale(y)
    else:
        scale = np.asarray(error_scale, dtype=np.float64)
        if scale.shape != (horizon, nodes):
            raise ValueError("error_scale must have shape [horizon,node]")
        scale = np.maximum(scale, 1e-6)
    if initial_weights is None:
        weights = np.full((experts, horizon, nodes), 1.0 / experts, dtype=np.float64)
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.shape != (experts, horizon, nodes):
            raise ValueError("initial_weights must have shape [experts,horizon,node]")
        if np.any(weights < 0.0):
            raise ValueError("initial_weights must be non-negative")
        denominator = weights.sum(axis=0, keepdims=True)
        if np.any(denominator <= 0.0):
            raise ValueError("Every horizon/node initial weight vector must be non-zero")
        weights /= denominator

    normalized_losses = _normalized_losses(y, predictions, scale)
    return _causal_hedge_mix_with_losses(
        y,
        predictions,
        order,
        eta,
        normalized_losses,
        initial_weights=weights,
    )


def _normalized_losses(
    target: np.ndarray,
    expert_predictions: Sequence[np.ndarray],
    scale: np.ndarray,
) -> np.ndarray:
    """Return clipped per-row losses once for reuse across eta candidates."""

    return np.stack(
        [
            np.clip(
                np.abs(prediction - target) / scale[None, :, :],
                0.0,
                10.0,
            )
            for prediction in expert_predictions
        ],
        axis=0,
    )


def _causal_hedge_mix_with_losses(
    target: np.ndarray,
    expert_predictions: Sequence[np.ndarray],
    origins: np.ndarray,
    eta: np.ndarray,
    normalized_losses: np.ndarray,
    *,
    initial_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Core delayed-feedback loop using precomputed normalized losses.

    The feedback rule is equivalent to the row-wise deque implementation, but
    is evaluated horizon by horizon with cumulative losses.  For a forecast
    row at origin ``s`` and horizon ``h`` (zero based), a previous row ``j`` is
    observable exactly when ``origin[j] + h < s``.  ``searchsorted`` therefore
    identifies the prefix of losses that may update the weights before each
    prediction.  Keeping the same prefix rule while vectorising the expensive
    expert/node operations makes full-origin post-processing practical for the
    larger PEMS graphs.
    """

    y = np.asarray(target, dtype=np.float64)
    predictions = [np.asarray(value, dtype=np.float64) for value in expert_predictions]
    experts, horizon, nodes = len(predictions), y.shape[1], y.shape[2]
    prediction_stack = np.stack(predictions, axis=0)
    initial = np.asarray(initial_weights, dtype=np.float64).copy()
    denominator = initial.sum(axis=0, keepdims=True)
    initial /= np.maximum(denominator, 1e-12)
    output = np.empty_like(y)
    weight_rows = np.empty((y.shape[0], experts, horizon, nodes), dtype=np.float32)
    # The cumulative prefix is rebuilt one horizon at a time to avoid a second
    # full [expert, origin, horizon, node] allocation.  This is algebraically
    # identical to multiplying by every newly available loss in the deque
    # implementation because the per-expert/node normalisers cancel.
    for h in range(horizon):
        ready = _ready_loss_prefix(
            normalized_losses[:, :, h, :], origins, h
        )
        weights_h = _weights_from_ready_losses(initial[:, h, :], ready, eta[h])
        weight_rows[:, :, h, :] = np.moveaxis(weights_h, 0, 1).astype(np.float32)
        output[:, h, :] = np.sum(
            prediction_stack[:, :, h, :] * weights_h,
            axis=0,
        )
    return output.astype(np.float32), weight_rows


def _ready_loss_prefix(
    losses: np.ndarray, origins: np.ndarray, horizon_index: int
) -> np.ndarray:
    """Return cumulative losses observable before each origin for one horizon."""

    values = np.asarray(losses, dtype=np.float64)
    cumulative = np.cumsum(values, axis=1)
    prefix_count = np.searchsorted(
        origins, origins - int(horizon_index), side="left"
    )
    ready = np.zeros_like(values, dtype=np.float64)
    has_ready = prefix_count > 0
    if np.any(has_ready):
        ready[:, has_ready, :] = cumulative[:, prefix_count[has_ready] - 1, :]
    return ready


def _weights_from_ready_losses(
    initial_weights: np.ndarray, ready_losses: np.ndarray, eta: float
) -> np.ndarray:
    """Compute normalized expert weights from already accumulated losses."""

    with np.errstate(divide="ignore"):
        logits = (
            np.log(np.maximum(np.asarray(initial_weights, dtype=np.float64), 1e-300))[
                :, None, :
            ]
            - float(eta) * np.asarray(ready_losses, dtype=np.float64)
        )
    logits -= np.max(logits, axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-300)
    return weights


def choose_hedge_eta_by_horizon(
    target: np.ndarray,
    expert_predictions: Sequence[np.ndarray],
    origins: np.ndarray,
    *,
    error_scale: np.ndarray | None = None,
    grid: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
) -> dict[str, float]:
    """Select one causal Hedge learning rate per horizon on validation only."""

    y, predictions, order = _validate_inputs(target, expert_predictions, origins)
    horizon = y.shape[1]
    scale = (
        np.asarray(error_scale, dtype=np.float64)
        if error_scale is not None
        else robust_error_scale(y)
    )
    values = [float(value) for value in grid]
    if not values or any(value < 0.0 for value in values):
        raise ValueError("Hedge eta grid must contain non-negative values")
    normalized_losses = _normalized_losses(y, predictions, scale)
    prediction_stack = np.stack(predictions, axis=0)
    initial = np.full(
        (len(predictions), y.shape[2]),
        1.0 / len(predictions),
        dtype=np.float64,
    )
    selected: dict[str, float] = {}
    for index in range(horizon):
        ready = _ready_loss_prefix(
            normalized_losses[:, :, index, :], order, index
        )
        best_eta = values[0]
        best_loss = float("inf")
        for candidate in values:
            weights = _weights_from_ready_losses(initial, ready, candidate)
            mixed = np.sum(
                prediction_stack[:, :, index, :] * weights,
                axis=0,
            )
            loss = float(np.mean(np.abs(mixed - y[:, index])))
            if loss < best_loss - 1e-12 or (abs(loss - best_loss) <= 1e-12 and candidate < best_eta):
                best_loss, best_eta = loss, candidate
        selected[str(index + 1)] = float(best_eta)
    return selected
