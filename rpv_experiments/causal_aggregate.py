"""Run the locked v5 causal expert aggregation on a completed unit.

This command is intentionally post-training.  It reads the already-frozen
expert and baseline predictions, chooses Hedge rates on gate-validation, then
evaluates the test split once with delayed feedback.  No test statistic is
used to select an eta or an initial weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_aggregation import (
    causal_hedge_mix,
    choose_hedge_eta_by_horizon,
    robust_error_scale,
)


def _load(root: Path, split: str, name: str) -> np.ndarray:
    path = root / "predictions" / split / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _relative_gain(baseline: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> float:
    base = float(np.mean(np.abs(baseline - target)))
    value = float(np.mean(np.abs(candidate - target)))
    return (base - value) / max(base, 1e-12)


def aggregate_unit(run_root: Path) -> dict[str, object]:
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("routing_contract") != "independent_frozen_experts":
        raise RuntimeError("Causal aggregation requires independent frozen A/R experts")
    contract = metrics.get("causal_aggregation_contract", {})
    if contract.get("feedback") != "delayed_until_target_enters_history":
        raise RuntimeError("Missing locked delayed-feedback causal aggregation contract")
    if contract.get("eta_selection_split") != "gate_validation":
        raise RuntimeError("Causal aggregation eta must be selected on gate_validation")
    validation = {
        name: _load(run_root, "gate_validation", name)
        for name in ("targets", "pred_all", "pred_prior", "final_static_mix", "final_loss_gate")
    }
    test = {
        name: _load(run_root, "test", name)
        for name in ("targets", "pred_all", "pred_prior", "final_static_mix", "final_loss_gate")
    }
    validation_origins = _load(run_root, "gate_validation", "origin_indices")
    test_origins = _load(run_root, "test", "origin_indices")
    validation_experts = [
        validation["pred_all"],
        validation["pred_prior"],
        validation["final_static_mix"],
        validation["final_loss_gate"],
    ]
    test_experts = [
        test["pred_all"],
        test["pred_prior"],
        test["final_static_mix"],
        test["final_loss_gate"],
    ]
    eta_grid = contract.get(
        "eta_grid", metrics.get("causal_eta_grid", [0.01, 0.03, 0.1, 0.3, 1.0])
    )
    error_scale = robust_error_scale(validation["targets"])
    eta = choose_hedge_eta_by_horizon(
        validation["targets"],
        validation_experts,
        validation_origins,
        error_scale=error_scale,
        grid=[float(value) for value in eta_grid],
    )
    eta_vector = np.asarray(
        [float(eta[str(index + 1)]) for index in range(validation["targets"].shape[1])],
        dtype=np.float64,
    )
    validation_prediction, validation_weights = causal_hedge_mix(
        validation["targets"],
        validation_experts,
        validation_origins,
        eta_vector,
        error_scale=error_scale,
    )
    test_prediction, test_weights = causal_hedge_mix(
        test["targets"],
        test_experts,
        test_origins,
        eta_vector,
        error_scale=error_scale,
        # Warm-start from the normalized weights used at the last validation
        # forecast origin, not from a state flushed after all validation targets
        # arrive. Pending validation losses are not carried across the split;
        # only test-origin losses update this state during testing.
        initial_weights=validation_weights[-1],
    )
    np.save(run_root / "predictions" / "gate_validation" / "final_causal_hedge.npy", validation_prediction)
    np.save(run_root / "predictions" / "test" / "final_causal_hedge.npy", test_prediction)
    np.save(run_root / "predictions" / "test" / "causal_hedge_weights.npy", test_weights)
    per_horizon: dict[str, object] = {}
    for index in range(test["targets"].shape[1]):
        h = str(index + 1)
        target_test = test["targets"][:, index]
        target_validation = validation["targets"][:, index]
        per_horizon[h] = {
            "eta": float(eta[h]),
            "validation_mae": {
                "causal_hedge": float(np.mean(np.abs(validation_prediction[:, index] - target_validation))),
                "loss_gate": float(np.mean(np.abs(validation["final_loss_gate"][:, index] - target_validation))),
            },
            "test_mae": {
                "causal_hedge": float(np.mean(np.abs(test_prediction[:, index] - target_test))),
                "static_mix": float(np.mean(np.abs(test["final_static_mix"][:, index] - target_test))),
                "loss_gate": float(np.mean(np.abs(test["final_loss_gate"][:, index] - target_test))),
            },
            "test_relative_gain_over_static": _relative_gain(
                test["final_static_mix"][:, index], test_prediction[:, index], target_test
            ),
            "test_relative_gain_over_loss_gate": _relative_gain(
                test["final_loss_gate"][:, index], test_prediction[:, index], target_test
            ),
        }
    summary = {
        "schema": "selective_prior_causal_hedge_aggregation_v1",
        "run_root": str(run_root),
        "dataset": metrics.get("dataset"),
        "experiment_id": metrics.get("experiment_id"),
        "seed": metrics.get("seed"),
        "experts": ["all_only", "prior_only", "static_mix", "loss_gate"],
        "feedback_contract": "delayed_until_target_enters_history",
        "eta_selection_split": "gate_validation",
        "error_scale_source": "gate_validation_only",
        "test_initial_weights_source": "gate_validation_last_forecast_origin_weights",
        "pending_validation_feedback": "discarded_at_test_boundary",
        "test_selection_use": "none",
        "eta_by_horizon": eta,
        "per_horizon": per_horizon,
        "mean_test_relative_gain_over_static": _relative_gain(
            test["final_static_mix"], test_prediction, test["targets"]
        ),
        "mean_test_relative_gain_over_loss_gate": _relative_gain(
            test["final_loss_gate"], test_prediction, test["targets"]
        ),
    }
    (run_root / "causal_aggregation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    summary = aggregate_unit(args.run_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
