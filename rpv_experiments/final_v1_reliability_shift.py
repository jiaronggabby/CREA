"""Frozen-expert prior-reliability shift experiment for FINAL_V1.

The prior-only checkpoint is frozen.  Gate-validation is always evaluated with
the clean supplied graph and is used once to select eta and the initial test
weights.  On the locked test stream the clean prior is used before a fixed
midpoint, then a corrupted prior is switched on; eta is never re-selected
after that switch.  This is a mechanism analysis, not a primary endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rpv_data.graphs import (
    GraphArrays,
    endpoint_change_fraction,
    load_graph,
    rewire_directed,
)

from .causal_aggregation import (
    _normalized_losses,
    _ready_loss_prefix,
    _weights_from_ready_losses,
    robust_error_scale,
)
from .config import (
    load_config,
    resolved_data_root,
    resolved_output_root,
    units_for_stage,
)
from .data import (
    WindowDataset,
    base_graph_path,
    graph_to_mask,
    load_bundle,
    window_starts,
)
from .runner import _window_seed
from .training import build_forecaster, predict_raw, resolve_device

EXPECTED_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(root: Path, split: str, name: str) -> np.ndarray:
    path = root / "predictions" / split / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False).astype(np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_sha256(graph: GraphArrays) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(graph.edge_index, dtype=np.int64).tobytes())
    digest.update(np.asarray(graph.edge_weight, dtype=np.float32).tobytes())
    digest.update(str(int(graph.num_nodes)).encode("ascii"))
    return digest.hexdigest()


def _shift_graph(base: GraphArrays, requested: float, seed: int) -> tuple[GraphArrays, float]:
    if requested <= 0.0:
        return base, 0.0
    # For a nominal 100% corruption request, exact endpoint replacement can be
    # infeasible under the simultaneous degree and weight-bin constraints (the
    # feasible state space may retain a small unavoidable overlap).  Search a
    # fixed, deterministic set of unconstrained matched rewires and retain the
    # most changed graph.  The realized fraction is recorded in every row; it
    # is never relabelled as 1.0 and the result is rejected if the corruption is
    # not genuinely high.  This keeps the 100% row interpretable as a
    # predeclared maximum-feasible corruption rather than silently weakening
    # the null constraint.
    if requested >= 0.999:
        best_graph: GraphArrays | None = None
        best_fraction = -1.0
        for attempt in range(12):
            try:
                graph = rewire_directed(
                    base,
                    seed=int(seed) + attempt * 104729,
                    matched_weight_bins=True,
                    min_endpoint_change=0.0,
                )
            except (RuntimeError, ValueError):
                continue
            actual = float(endpoint_change_fraction(base, graph))
            if actual > best_fraction:
                best_graph, best_fraction = graph, actual
        if best_graph is None or best_fraction < 0.90:
            raise RuntimeError(
                "Could not create a genuinely high maximum-feasible prior "
                f"reliability shift: best endpoint change={best_fraction:.3f}."
            )
        return best_graph, best_fraction
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            graph = rewire_directed(
                base,
                seed=int(seed) + attempt * 104729,
                matched_weight_bins=True,
                min_endpoint_change=float(requested),
            )
            return graph, float(endpoint_change_fraction(base, graph))
        except (RuntimeError, ValueError) as error:
            last_error = error
    raise RuntimeError(
        f"Could not create frozen prior reliability shift {requested:.2f}: {last_error}"
    )


def _load_prior_model(
    checkpoint: Path,
    bundle: Any,
    unit: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "fixed_forecaster_v2":
        raise RuntimeError(f"Unexpected prior checkpoint schema: {checkpoint}")
    model = build_forecaster(bundle, int(unit["history"]), int(unit["max_horizon"]), training).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _causal_shift_summary(
    target: np.ndarray,
    all_prediction: np.ndarray,
    prior_prediction: np.ndarray,
    origins: np.ndarray,
    eta_vector: list[float],
    error_scale: np.ndarray,
    initial_weights: np.ndarray,
    transition: int,
) -> dict[str, Any]:
    """Compute shift metrics horizon-by-horizon without materializing weights.

    This is algebraically identical to ``causal_hedge_mix`` for the two
    frozen experts.  The shift audit only needs row-wise errors and prior
    weights, so keeping one horizon in memory avoids a multi-gigabyte
    ``[origin, expert, horizon, node]`` allocation on PEMS07.
    """

    y = np.asarray(target, dtype=np.float64)
    all_pred = np.asarray(all_prediction, dtype=np.float64)
    prior_pred = np.asarray(prior_prediction, dtype=np.float64)
    if y.shape != all_pred.shape or y.shape != prior_pred.shape or y.ndim != 3:
        raise ValueError("shift predictions and target must share [origin,horizon,node] shape")
    n_origins, horizon, nodes = y.shape
    if len(eta_vector) != horizon:
        raise ValueError("eta_vector must have one value per horizon")
    scale = np.maximum(np.asarray(error_scale, dtype=np.float64), 1e-6)
    if scale.shape != (horizon, nodes):
        raise ValueError("error_scale must have shape [horizon,node]")
    initial = np.asarray(initial_weights, dtype=np.float64)
    if initial.shape != (2, horizon, nodes):
        raise ValueError("initial_weights must have shape [2,horizon,node]")
    prior_error = np.mean(np.abs(prior_pred - y), axis=(1, 2))
    all_error = np.mean(np.abs(all_pred - y), axis=(1, 2))
    causal_error_sum = np.zeros(n_origins, dtype=np.float64)
    prior_weight_sum = np.zeros(n_origins, dtype=np.float64)
    for horizon_index in range(horizon):
        target_h = y[:, horizon_index, :]
        all_h = all_pred[:, horizon_index, :]
        prior_h = prior_pred[:, horizon_index, :]
        losses = np.stack(
            [
                np.clip(np.abs(all_h - target_h) / scale[horizon_index][None, :], 0.0, 10.0),
                np.clip(np.abs(prior_h - target_h) / scale[horizon_index][None, :], 0.0, 10.0),
            ],
            axis=0,
        )
        ready = _ready_loss_prefix(losses, origins, horizon_index)
        weights_h = _weights_from_ready_losses(
            initial[:, horizon_index, :], ready, float(eta_vector[horizon_index])
        )
        causal_error_sum += np.sum(
            np.abs(all_h * weights_h[0] + prior_h * weights_h[1] - target_h), axis=1
        )
        prior_weight_sum += np.sum(weights_h[1], axis=1)
    causal_error = causal_error_sum / float(horizon * nodes)
    prior_weight = prior_weight_sum / float(horizon * nodes)
    pre_weights = prior_weight[:transition]
    post_weights = prior_weight[transition:]
    pre_mean = float(np.mean(pre_weights)) if len(pre_weights) else float("nan")
    post_mean = float(np.mean(post_weights)) if len(post_weights) else float("nan")
    adaptation_delay = None
    if len(post_weights) and pre_mean > 0.0:
        threshold = 0.5 * pre_mean
        below = np.flatnonzero(post_weights <= threshold)
        if len(below):
            adaptation_delay = int(below[0])
    return {
        "prior_error": prior_error,
        "all_error": all_error,
        "causal_error": causal_error,
        "prior_weight": prior_weight,
        "prior_weight_first_half": pre_mean,
        "prior_weight_second_half": post_mean,
        "prior_weight_drop": float(pre_mean - post_mean),
        "adaptation_delay": adaptation_delay,
    }


def _unit_shift(
    cfg: dict[str, Any],
    unit: dict[str, Any],
    seed: int,
    report_root: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    run_root = report_root / "runs" / str(unit["experiment_id"]) / str(unit["dataset"]) / f"seed_{int(seed)}"
    state = _json(run_root / "state.json")
    if state.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {run_root}")
    causal = _json(run_root / "causal_aggregation.json")
    if causal.get("test_selection_use") != "none":
        raise RuntimeError(f"Test was used for selection: {run_root}")
    bundle = load_bundle(resolved_data_root(cfg), str(unit["dataset"]), cfg["split"])
    training = cfg["training"]
    sampling_group = str(unit.get("window_sampling_group", bundle.name))
    datasets = {}
    for split in ("gate_validation", "test"):
        maximum = unit.get("max_windows", {}).get(split, cfg["max_windows"].get(split))
        starts = window_starts(
            bundle,
            split,
            int(unit["history"]),
            int(unit["max_horizon"]),
            None if maximum is None else int(maximum),
            _window_seed(cfg, sampling_group, split),
        )
        datasets[split] = WindowDataset(bundle, starts, int(unit["history"]), int(unit["max_horizon"]))
    validation = {
        "target": _array(run_root, "gate_validation", "targets"),
        "all": _array(run_root, "gate_validation", "pred_all"),
        "prior": _array(run_root, "gate_validation", "pred_prior"),
    }
    test = {
        "target": _array(run_root, "test", "targets"),
        "all": _array(run_root, "test", "pred_all"),
        "prior": _array(run_root, "test", "pred_prior"),
    }
    checkpoint = run_root / "checkpoints" / "prior_only.pt"
    model = _load_prior_model(checkpoint, bundle, unit, training, device)
    base = load_graph(base_graph_path(bundle))
    scale = robust_error_scale(validation["target"])
    # The clean graph is the only graph allowed to determine eta and the
    # starting weights.  This is deliberately outside the corruption loop.
    # Reuse the eta selected by the already completed clean formal
    # aggregation.  Re-searching the same gate-validation grid for every
    # corruption level is redundant and dominated runtime; this remains a
    # train/validation-only choice and never reads test outcomes.
    eta = {
        str(key): float(value) for key, value in causal["eta_by_horizon"].items()
    }
    eta_vector = [
        float(eta[str(index + 1)]) for index in range(validation["target"].shape[1])
    ]
    # Only the final validation weights are needed.  Compute them directly
    # from the same delayed cumulative losses instead of materializing the
    # complete validation mixture and weight trajectory.
    normalized_losses = _normalized_losses(
        validation["target"], [validation["all"], validation["prior"]], scale
    )
    initial_weights = np.empty(
        (2, validation["target"].shape[1], validation["target"].shape[2]),
        dtype=np.float64,
    )
    uniform = np.full((2, validation["target"].shape[2]), 0.5, dtype=np.float64)
    for horizon_index, eta_value in enumerate(eta_vector):
        ready = _ready_loss_prefix(
            normalized_losses[:, :, horizon_index, :],
            datasets["gate_validation"].starts,
            horizon_index,
        )
        weights_h = _weights_from_ready_losses(uniform, ready, eta_value)
        initial_weights[:, horizon_index, :] = weights_h[:, -1, :]
    del normalized_losses
    transition = len(test["target"]) // 2
    post_change_dataset = WindowDataset(
        bundle,
        datasets["test"].starts[transition:],
        int(unit["history"]),
        int(unit["max_horizon"]),
    )
    # Inference-only batching does not alter the trained checkpoint or any
    # selection rule.  A larger fixed batch keeps the post-hoc mechanism audit
    # tractable on the full test stream, especially for PEMS07.
    posthoc_training = dict(training)
    posthoc_training["micro_batch_size"] = max(128, int(training["micro_batch_size"]))
    rows: list[dict[str, Any]] = []
    for fraction_index, requested in enumerate(EXPECTED_FRACTIONS):
        graph, actual = _shift_graph(
            base,
            float(requested),
            seed=int(cfg["randomness"]["matched_control_namespace"]) + int(seed) + fraction_index * 1009,
        )
        mask = graph_to_mask(graph, device=device)
        # Only the post-change half is evaluated with the corrupted relation.
        # The pre-change half reuses the clean, frozen prior predictions from
        # the formal run; no target-dependent choice is made here.  Predicting
        # only the post-change windows also avoids doing unused inference for
        # the clean half and for the zero-corruption row.
        corrupted_test_prior = None
        if float(actual) > 0.0:
            _, corrupted_test_prior = predict_raw(
                model, post_change_dataset, mask, posthoc_training, device
            )
        shifted_test_prior = np.asarray(test["prior"], dtype=np.float64).copy()
        if corrupted_test_prior is not None:
            shifted_test_prior[transition:] = corrupted_test_prior
        shift_summary = _causal_shift_summary(
            test["target"],
            test["all"],
            shifted_test_prior,
            datasets["test"].starts,
            eta_vector,
            scale,
            initial_weights,
            transition,
        )
        prior_error = shift_summary["prior_error"]
        all_error = shift_summary["all_error"]
        hedge_error = shift_summary["causal_error"]
        pre_mean = float(shift_summary["prior_weight_first_half"])
        post_mean = float(shift_summary["prior_weight_second_half"])
        adaptation_delay = shift_summary["adaptation_delay"]
        rows.append(
            {
                "dataset": str(unit["dataset"]),
                "experiment_id": str(unit["experiment_id"]),
                "seed": int(seed),
                "requested_endpoint_change_fraction": float(requested),
                "actual_endpoint_change_fraction": float(actual),
                "test_origins": int(test["target"].shape[0]),
                "transition_origin_index": int(transition),
                "prior_expert_mean_weight": float(np.mean(shift_summary["prior_weight"])),
                "prior_expert_weight_first_half": pre_mean,
                "prior_expert_weight_second_half": post_mean,
                "prior_expert_weight_drop": float(pre_mean - post_mean),
                "adaptation_delay_origins_to_half_pre_weight": adaptation_delay,
                "prior_only_mae": float(np.mean(prior_error)),
                "all_only_mae": float(np.mean(all_error)),
                "causal_two_expert_mae": float(np.mean(hedge_error)),
                "prior_harm_rate_first_half": float(np.mean(prior_error[:transition] > all_error[:transition])),
                "prior_harm_rate_second_half": float(np.mean(prior_error[transition:] > all_error[transition:])),
                "prior_harm_rate": float(np.mean(prior_error > all_error)),
                "causal_gain_over_all": float((np.mean(all_error) - np.mean(hedge_error)) / max(np.mean(all_error), 1e-12)),
                "eta_by_horizon": eta,
                "graph_sha256": _graph_sha256(graph),
                "prior_checkpoint_sha256": _sha256(checkpoint),
            }
        )
        del corrupted_test_prior, shifted_test_prior, shift_summary
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_reliability_shift(
    config_path: Path,
    *,
    datasets: list[str],
    seeds: list[int],
) -> Path:
    cfg, _ = load_config(config_path)
    report_root = resolved_output_root(cfg)
    device = resolve_device(str(cfg["device"]))
    units = [unit for unit in units_for_stage(cfg, "formal") if str(unit["dataset"]) in set(datasets)]
    if {str(unit["dataset"]) for unit in units} != set(datasets):
        raise RuntimeError("Requested reliability-shift datasets are not in FINAL_V1 formal units")
    rows = [
        row
        for seed in seeds
        for unit in units
        for row in _unit_shift(cfg, unit, int(seed), report_root, device)
    ]
    output = report_root / "analysis" / "prior_reliability_shift"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "final_v1_prior_reliability_shift.json"
    path.write_text(
        json.dumps(
            {
                "schema": "selective_prior_causal_hedge_final_v1_prior_reliability_shift_v2",
                "test_selection_use": "none",
                "eta_selection_split": "gate_validation_clean_formal_eta_reused",
                "test_change_point": "fixed_midpoint",
                "test_prior_schedule": "clean_first_half_then_corrupted_second_half",
                "corrupted_prediction_scope": "post_change_test_half_only",
                "posthoc_inference_batch_size": max(128, int(cfg["training"]["micro_batch_size"])),
                "datasets": datasets,
                "seeds": [int(value) for value in seeds],
                "fractions": list(EXPECTED_FRACTIONS),
                "nominal_100_percent_policy": "maximum_feasible_matched_rewire_under_degree_and_weight_constraints; realized_fraction_is_reported",
                "rows": rows,
                "mechanism_note": "The eta selected by the completed clean gate-validation formal aggregation is reused unchanged; clean validation recomputes two-expert initial weights once, and the locked test stream switches from clean to corrupted prior at a fixed midpoint without retuning.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--datasets", default="pems04,metr_la")
    parser.add_argument("--seeds", default="42,3407,101,2026,777")
    args = parser.parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    print(write_reliability_shift(args.config, datasets=datasets, seeds=seeds))


if __name__ == "__main__":
    main()
