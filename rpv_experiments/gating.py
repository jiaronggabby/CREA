from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def _specificity_target(
    soft_win_rate: np.ndarray,
    method: dict[str, Any],
    sigmoid,
) -> np.ndarray:
    mode = str(method.get("specificity_target", "thresholded_win_rate"))
    rho = float(method["rho"])
    if mode == "thresholded_win_rate":
        return sigmoid((soft_win_rate - rho) / float(method["tau_w"]))
    if mode == "positive_excess_win_rate":
        chance = float(method.get("specificity_chance", 0.5))
        if not 0.0 <= chance < rho <= 1.0:
            raise ValueError("specificity_chance and rho must satisfy 0 <= chance < rho <= 1")
        # Chance-level performance against matched controls carries no
        # structure-specific evidence.  Evidence then grows continuously to
        # one at the strict verification threshold rho.  Unlike the v2
        # sigmoid centred at 0.8, this retains gradients for moderate but
        # genuine specificity without weakening the final binary certificate.
        return np.clip((soft_win_rate - chance) / (rho - chance), 0.0, 1.0)
    raise ValueError(f"Unknown specificity_target mode: {mode}")


def prior_value_targets(
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
    pred_matched: np.ndarray,
    epsilon: np.ndarray,
    method: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build loss-only and matched-specific soft targets at origin/horizon/node level."""
    y = np.asarray(target, dtype=np.float64)
    all_prediction = np.asarray(pred_all, dtype=np.float64)
    prior_prediction = np.asarray(pred_prior, dtype=np.float64)
    matched = np.asarray(pred_matched, dtype=np.float64)
    if y.shape != all_prediction.shape or y.shape != prior_prediction.shape:
        raise ValueError("target, pred_all, and pred_prior must share [origin,horizon,node] shape")
    if matched.ndim != 4 or matched.shape[1:] != y.shape:
        raise ValueError("pred_matched must be [control,origin,horizon,node]")
    eps = np.asarray(epsilon, dtype=np.float64).reshape(1, 1, -1)
    error_all = np.abs(all_prediction - y)
    error_prior = np.abs(prior_prediction - y)
    error_matched = np.abs(matched - y[None, ...])
    advantage = np.log((error_all + eps) / (error_prior + eps))
    matched_advantage = np.log((error_matched + eps[None, ...]) / (error_prior[None, ...] + eps[None, ...]))
    delta_a = float(np.log1p(float(method["delta_a_relative"])))
    delta_s = float(method["delta_s"])
    tau_a = float(method["tau_a"])
    tau_s = float(method["tau_s"])
    rho = float(method["rho"])
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
    q_a = sigmoid((advantage - delta_a) / tau_a)
    soft_win_rate = sigmoid((matched_advantage - delta_s) / tau_s).mean(axis=0)
    q_s = _specificity_target(soft_win_rate, method, sigmoid)
    # The v2 trainer supervises the gate with q_loss and q_specificity as two
    # separate terms.  Keep q_matched only as a diagnostic soft-AND target;
    # using a product here would make a moderately useful prior look like a
    # hard zero whenever specificity is uncertain.
    matched_target = np.minimum(q_a, q_s).astype(np.float32)
    hard_win_rate = (error_prior[None, ...] < error_matched).mean(axis=0)
    binary = ((advantage > delta_a) & (hard_win_rate >= rho)).astype(np.uint8)
    return {
        "q_loss": q_a.astype(np.float32),
        "q_specificity": q_s.astype(np.float32),
        "q_matched": matched_target,
        "binary_prior_value": binary,
        "advantage_log_ratio": advantage.astype(np.float32),
        "matched_soft_win_rate": soft_win_rate.astype(np.float32),
        "matched_hard_win_rate": hard_win_rate.astype(np.float32),
    }


def prior_value_targets_from_files(
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
    matched_paths: list[Path],
    epsilon: np.ndarray,
    method: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Memory-bounded equivalent of :func:`prior_value_targets`."""
    if not matched_paths:
        raise ValueError("matched_paths cannot be empty")
    y = np.asarray(target, dtype=np.float64)
    all_prediction = np.asarray(pred_all, dtype=np.float64)
    prior_prediction = np.asarray(pred_prior, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64).reshape(1, 1, -1)
    error_all = np.abs(all_prediction - y)
    error_prior = np.abs(prior_prediction - y)
    advantage = np.log((error_all + eps) / (error_prior + eps))
    delta_a = float(np.log1p(float(method["delta_a_relative"])))
    tau_a = float(method["tau_a"])
    tau_s = float(method["tau_s"])
    delta_s = float(method["delta_s"])
    rho = float(method["rho"])
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
    soft_win_sum = np.zeros_like(error_prior, dtype=np.float64)
    hard_win_sum = np.zeros_like(error_prior, dtype=np.float64)
    for path in matched_paths:
        loaded = np.load(path, mmap_mode="r" if path.suffix == ".npy" else None, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                prediction = np.asarray(loaded["prediction"], dtype=np.float64)
            finally:
                loaded.close()
        else:
            prediction = np.asarray(loaded, dtype=np.float64)
        if prediction.shape != y.shape:
            raise ValueError(f"Matched prediction shape mismatch in {path}: {prediction.shape} != {y.shape}")
        error = np.abs(prediction - y)
        ratio = np.log((error + eps) / (error_prior + eps))
        soft_win_sum += sigmoid((ratio - delta_s) / tau_s)
        hard_win_sum += error_prior < error
    soft_win_rate = soft_win_sum / len(matched_paths)
    hard_win_rate = hard_win_sum / len(matched_paths)
    q_a = sigmoid((advantage - delta_a) / tau_a)
    q_s = _specificity_target(soft_win_rate, method, sigmoid)
    return {
        "q_loss": q_a.astype(np.float32),
        "q_specificity": q_s.astype(np.float32),
        # See :func:`prior_value_targets`: training consumes q_loss and
        # q_specificity separately; this field is retained for diagnostics.
        "q_matched": np.minimum(q_a, q_s).astype(np.float32),
        "binary_prior_value": ((advantage > delta_a) & (hard_win_rate >= rho)).astype(np.uint8),
        "advantage_log_ratio": advantage.astype(np.float32),
        "matched_soft_win_rate": soft_win_rate.astype(np.float32),
        "matched_hard_win_rate": hard_win_rate.astype(np.float32),
    }


def choose_static_alpha(target: np.ndarray, pred_all: np.ndarray, pred_prior: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    losses = [
        float(np.mean(np.abs((alpha * pred_prior + (1.0 - alpha) * pred_all) - target)))
        for alpha in grid
    ]
    return float(grid[int(np.argmin(losses))])


def choose_static_alphas_by_horizon(
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
) -> dict[str, float]:
    """Choose one validation-frozen Static Mix weight for each forecast horizon."""
    y = np.asarray(target)
    all_prediction = np.asarray(pred_all)
    prior_prediction = np.asarray(pred_prior)
    if y.ndim != 3 or y.shape != all_prediction.shape or y.shape != prior_prediction.shape:
        raise ValueError("Static Mix inputs must share [origin,horizon,node] shape")
    return {
        str(horizon): choose_static_alpha(
            y[:, horizon - 1 : horizon, :],
            all_prediction[:, horizon - 1 : horizon, :],
            prior_prediction[:, horizon - 1 : horizon, :],
        )
        for horizon in range(1, y.shape[1] + 1)
    }


def mix_with_horizon_alphas(
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
    alphas: dict[str, float],
) -> np.ndarray:
    """Apply validation-selected Static Mix weights without pooling horizons."""
    all_prediction = np.asarray(pred_all)
    prior_prediction = np.asarray(pred_prior)
    if all_prediction.ndim != 3 or all_prediction.shape != prior_prediction.shape:
        raise ValueError("Static Mix predictions must share [origin,horizon,node] shape")
    missing = [str(horizon) for horizon in range(1, all_prediction.shape[1] + 1) if str(horizon) not in alphas]
    if missing:
        raise ValueError(f"Missing Static Mix alpha for horizons {missing}")
    alpha = np.asarray(
        [float(alphas[str(horizon)]) for horizon in range(1, all_prediction.shape[1] + 1)],
        dtype=np.float64,
    ).reshape(1, -1, 1)
    return alpha * prior_prediction + (1.0 - alpha) * all_prediction


def choose_best_fixed_by_horizon(
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
) -> dict[str, str]:
    """Freeze the lower-validation-MAE fixed branch separately at each horizon."""
    y = np.asarray(target)
    all_prediction = np.asarray(pred_all)
    prior_prediction = np.asarray(pred_prior)
    if y.ndim != 3 or y.shape != all_prediction.shape or y.shape != prior_prediction.shape:
        raise ValueError("Best-fixed inputs must share [origin,horizon,node] shape")
    selected: dict[str, str] = {}
    for horizon in range(1, y.shape[1] + 1):
        index = horizon - 1
        all_mae = float(np.mean(np.abs(all_prediction[:, index, :] - y[:, index, :])))
        prior_mae = float(np.mean(np.abs(prior_prediction[:, index, :] - y[:, index, :])))
        selected[str(horizon)] = "all_only" if all_mae <= prior_mae else "prior_only"
    return selected


def regression_metrics_numpy(
    target: np.ndarray,
    prediction: np.ndarray,
    q95_threshold: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    error = pred - y
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }
    if q95_threshold is not None:
        threshold = np.asarray(q95_threshold, dtype=np.float64).reshape(1, 1, -1)
        event = y >= threshold
        result["q95_mae"] = float(np.mean(np.abs(error)[event])) if np.any(event) else float("nan")
        result["q95_count"] = int(event.sum())
    return result


def gate_metrics(
    probability: np.ndarray,
    binary_target: np.ndarray,
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
    static_prediction: np.ndarray,
    mixed_prediction: np.ndarray,
) -> dict[str, Any]:
    probability_flat = np.asarray(probability, dtype=np.float64).ravel()
    binary_flat = np.asarray(binary_target, dtype=np.uint8).ravel()
    prevalence = float(binary_flat.mean())
    if np.unique(binary_flat).size < 2:
        auprc = float("nan")
    else:
        auprc = float(average_precision_score(binary_flat, probability_flat))
    brier = float(brier_score_loss(binary_flat, np.clip(probability_flat, 0.0, 1.0)))
    y = np.asarray(target, dtype=np.float64)
    all_error = np.abs(np.asarray(pred_all) - y)
    prior_error = np.abs(np.asarray(pred_prior) - y)
    oracle_prediction = np.where(prior_error < all_error, pred_prior, pred_all)
    static_mae = float(np.mean(np.abs(np.asarray(static_prediction) - y)))
    mixed_mae = float(np.mean(np.abs(np.asarray(mixed_prediction) - y)))
    oracle_mae = float(np.mean(np.abs(oracle_prediction - y)))
    denominator = static_mae - oracle_mae
    recovery = (static_mae - mixed_mae) / denominator if denominator > 1e-12 else float("nan")
    oracle_regret = mixed_mae - oracle_mae
    calibration_bins = []
    expected_calibration_error = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        if index == 9:
            selected = (probability_flat >= edges[index]) & (probability_flat <= edges[index + 1])
        else:
            selected = (probability_flat >= edges[index]) & (probability_flat < edges[index + 1])
        count = int(selected.sum())
        if count:
            mean_probability = float(probability_flat[selected].mean())
            empirical_rate = float(binary_flat[selected].mean())
            expected_calibration_error += (count / len(probability_flat)) * abs(mean_probability - empirical_rate)
        else:
            mean_probability = None
            empirical_rate = None
        calibration_bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "empirical_rate": empirical_rate,
            }
        )
    return {
        "prevalence": prevalence,
        "auprc": auprc,
        "brier": brier,
        # The continuous coefficient is the v3 E3 corruption-response
        # statistic. Keep hard activation as a separate diagnostic.
        "mean_probability": float(np.mean(probability_flat)),
        "activation_rate_greater_0_5": float(np.mean(probability_flat > 0.5)),
        "oracle_mae": oracle_mae,
        "oracle_regret": float(oracle_regret),
        "headroom_recovery": float(recovery),
        "expected_calibration_error": float(expected_calibration_error),
        "calibration_bins": calibration_bins,
    }


def choose_gate_shrinkage_by_horizon(
    target: np.ndarray,
    pred_all: np.ndarray,
    pred_prior: np.ndarray,
    raw_probability: np.ndarray,
    static_alphas: dict[str, float],
    *,
    grid_size: int = 21,
) -> dict[str, float]:
    """Calibrate a learned gate toward a validation-frozen Static Mix.

    The gate is a forecast combiner, so a validation-only convex shrinkage is a
    safer objective than asking the gate to extrapolate its raw probabilities.
    ``beta_h=0`` is exactly the Static Mix at horizon ``h``; therefore the
    calibration never selects a more aggressive gate on validation when it does
    not improve that horizon.  Test targets are never read here.
    """
    y = np.asarray(target, dtype=np.float64)
    all_prediction = np.asarray(pred_all, dtype=np.float64)
    prior_prediction = np.asarray(pred_prior, dtype=np.float64)
    probability = np.asarray(raw_probability, dtype=np.float64)
    if (
        y.ndim != 3
        or y.shape != all_prediction.shape
        or y.shape != prior_prediction.shape
        or y.shape != probability.shape
    ):
        raise ValueError("Gate shrinkage inputs must share [origin,horizon,node] shape")
    if int(grid_size) < 2:
        raise ValueError("grid_size must be at least two")
    grid = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    selected: dict[str, float] = {}
    for horizon in range(1, y.shape[1] + 1):
        key = str(horizon)
        if key not in static_alphas:
            raise ValueError(f"Missing Static Mix alpha for horizon {horizon}")
        alpha = float(static_alphas[key])
        index = horizon - 1
        best_beta = 0.0
        best_loss = float("inf")
        for beta in grid:
            weight = beta * probability[:, index, :] + (1.0 - beta) * alpha
            mixed = weight * prior_prediction[:, index, :] + (1.0 - weight) * all_prediction[:, index, :]
            loss = float(np.mean(np.abs(mixed - y[:, index, :])))
            if loss < best_loss - 1e-12:
                best_loss = loss
                best_beta = float(beta)
        selected[key] = best_beta
    return selected


def apply_gate_shrinkage_by_horizon(
    raw_probability: np.ndarray,
    static_alphas: dict[str, float],
    shrinkage: dict[str, float],
) -> np.ndarray:
    """Apply validation-frozen horizon-wise gate shrinkage to new origins."""
    probability = np.asarray(raw_probability, dtype=np.float64)
    if probability.ndim != 3:
        raise ValueError("raw_probability must have shape [origin,horizon,node]")
    result = np.empty_like(probability, dtype=np.float64)
    for horizon in range(1, probability.shape[1] + 1):
        key = str(horizon)
        if key not in static_alphas or key not in shrinkage:
            raise ValueError(f"Missing calibration for horizon {horizon}")
        alpha = float(static_alphas[key])
        beta = float(shrinkage[key])
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Gate shrinkage values must be in [0,1]")
        result[:, horizon - 1, :] = beta * probability[:, horizon - 1, :] + (1.0 - beta) * alpha
    return result.astype(np.float32)
