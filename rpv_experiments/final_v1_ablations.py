"""Post-hoc ablations for the frozen FINAL_V1 CREA predictions.

These ablations do not retrain or retune on test.  They reuse the frozen
expert predictions and select any alternative eta only on gate-validation.
The frozen-validation-weights variant is the deployable no-update ablation;
the no-delay variant is explicitly a non-causal diagnostic upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .causal_aggregation import (
    causal_hedge_mix,
    choose_hedge_eta_by_horizon,
    robust_error_scale,
)
from .final_v1_stats import _exact_raw_target

EXPECTED_DATASETS = {"pems03", "pems04", "pems07", "pems08", "metr_la"}
EXPECTED_SEEDS = (42, 3407, 101, 2026, 777)
ETA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(root: Path, split: str, name: str) -> np.ndarray:
    path = root / "predictions" / split / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False).astype(np.float64)


def _mix_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mape_target: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    raw = np.asarray(mape_target, dtype=np.float64)
    valid = np.abs(raw) > 1e-5
    if not np.any(valid):
        raise ValueError("No nonzero raw-scale targets remain for MAPE")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mape_percent": float(100.0 * np.mean(np.abs(np.asarray(prediction)[valid] - raw[valid]) / np.abs(raw[valid]))),
    }


def _global_eta(
    target: np.ndarray,
    experts: list[np.ndarray],
    origins: np.ndarray,
    error_scale: np.ndarray,
    grid: tuple[float, ...] = ETA_GRID,
) -> tuple[float, np.ndarray]:
    best_eta: float | None = None
    best_loss = float("inf")
    best_weights: np.ndarray | None = None
    for candidate in grid:
        prediction, weights = causal_hedge_mix(
            target,
            experts,
            origins,
            [candidate] * target.shape[1],
            error_scale=error_scale,
        )
        loss = float(np.mean(np.abs(prediction - target)))
        if loss < best_loss - 1e-12 or (
            abs(loss - best_loss) <= 1e-12 and (best_eta is None or candidate < best_eta)
        ):
            best_loss = loss
            best_eta = float(candidate)
            best_weights = weights[-1]
    if best_eta is None or best_weights is None:
        raise RuntimeError("Global eta selection produced no candidate")
    return best_eta, best_weights


def _non_delayed_mix(
    target: np.ndarray,
    experts: list[np.ndarray],
    eta_by_horizon: list[float],
    error_scale: np.ndarray,
    initial_weights: np.ndarray,
) -> np.ndarray:
    """Oracle same-origin feedback diagnostic; never a deployable method."""
    y = np.asarray(target, dtype=np.float64)
    predictions = [np.asarray(value, dtype=np.float64) for value in experts]
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    output = np.empty_like(y)
    for row in range(y.shape[0]):
        stack = np.stack([value[row] for value in predictions], axis=0)
        output[row] = np.sum(stack * weights, axis=0)
        losses = np.stack(
            [
                np.clip(
                    np.abs(value[row] - y[row]) / np.maximum(error_scale, 1e-6),
                    0.0,
                    10.0,
                )
                for value in predictions
            ],
            axis=0,
        )
        weights *= np.exp(
            -np.asarray(eta_by_horizon, dtype=np.float64)[None, :, None] * losses
        )
        weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    return output.astype(np.float32)


def _run_variants(report_root: Path, root: Path) -> list[dict[str, Any]]:
    metrics = _json(root / "metrics.json")
    causal = _json(root / "causal_aggregation.json")
    if causal.get("test_selection_use") != "none":
        raise RuntimeError(f"Test was used for selection: {root}")
    validation = {
        name: _array(root, "gate_validation", name)
        for name in ("targets", "pred_all", "pred_prior", "final_static_mix", "final_loss_gate")
    }
    test = {
        name: _array(root, "test", name)
        for name in (
            "targets",
            "pred_all",
            "pred_prior",
            "final_static_mix",
            "final_loss_gate",
            "final_causal_hedge",
        )
    }
    exact_test_target = _exact_raw_target(report_root, root, metrics, test["targets"])
    validation_origins = _array(root, "gate_validation", "origin_indices").reshape(-1).astype(np.int64)
    test_origins = _array(root, "test", "origin_indices").reshape(-1).astype(np.int64)
    scale = robust_error_scale(validation["targets"])
    eta_by_horizon = [
        float(causal["eta_by_horizon"][str(index + 1)])
        for index in range(validation["targets"].shape[1])
    ]
    experts = [
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
    _, validation_weights = causal_hedge_mix(
        validation["targets"], experts, validation_origins, eta_by_horizon, error_scale=scale
    )
    two_eta = choose_hedge_eta_by_horizon(
        validation["targets"], experts[:2], validation_origins, error_scale=scale, grid=ETA_GRID
    )
    two_eta_vector = [float(two_eta[str(index + 1)]) for index in range(validation["targets"].shape[1])]
    _, two_validation_weights = causal_hedge_mix(
        validation["targets"],
        experts[:2],
        validation_origins,
        two_eta_vector,
        error_scale=scale,
    )
    global_eta, global_initial = _global_eta(
        validation["targets"], experts, validation_origins, scale
    )
    variants: dict[str, np.ndarray] = {
        "causal_hedge": test["final_causal_hedge"],
        "w_o_prior_expert": test["pred_all"],
        "two_experts_all_prior": causal_hedge_mix(
            test["targets"],
            test_experts[:2],
            test_origins,
            two_eta_vector,
            error_scale=scale,
            initial_weights=two_validation_weights[-1],
        )[0],
        "global_eta": causal_hedge_mix(
            test["targets"],
            test_experts,
            test_origins,
            [global_eta] * test["targets"].shape[1],
            error_scale=scale,
            initial_weights=global_initial,
        )[0],
        "frozen_validation_weights": causal_hedge_mix(
            test["targets"],
            test_experts,
            test_origins,
            [0.0] * test["targets"].shape[1],
            error_scale=scale,
            initial_weights=validation_weights[-1],
        )[0],
        "no_delayed_feedback_diagnostic": _non_delayed_mix(
            test["targets"],
            test_experts,
            eta_by_horizon,
            scale,
            validation_weights[-1],
        ),
    }
    rows: list[dict[str, Any]] = []
    horizons = [int(value) for value in metrics["report_horizons"]]
    for variant, prediction in variants.items():
        for horizon in horizons:
            index = horizon - 1
            metric = _mix_metrics(
                test["targets"][:, index],
                prediction[:, index],
                exact_test_target[:, index],
            )
            rows.append(
                {
                    "dataset": str(metrics["dataset"]),
                    "seed": int(metrics["seed"]),
                    "horizon": horizon,
                    "variant": variant,
                    "eta_global": global_eta if variant == "global_eta" else None,
                    **metric,
                }
            )
    return rows


def write_final_v1_ablations(
    report_root: Path,
    *,
    seeds: list[int] | None = None,
) -> Path:
    root = report_root.resolve()
    selected = [int(value) for value in (seeds or EXPECTED_SEEDS)]
    roots = [
        path
        for seed in selected
        for path in sorted(root.glob(f"runs/E*/*/seed_{int(seed)}"))
    ]
    if len(roots) != len(selected) * len(EXPECTED_DATASETS):
        raise RuntimeError("FINAL_V1 ablations require every selected dataset/seed run")
    rows = [row for run_root in roots for row in _run_variants(root, run_root)]
    frame = pd.DataFrame(rows).sort_values(["variant", "dataset", "seed", "horizon"])
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "final_v1_ablation_per_dataset_horizon_seed.csv", index=False)
    formal_variants = [
        "w_o_prior_expert",
        "two_experts_all_prior",
        "frozen_validation_weights",
        "global_eta",
        "no_delayed_feedback_diagnostic",
    ]
    summary: list[dict[str, Any]] = []
    for (variant, seed), subset in frame[
        frame["variant"].isin(formal_variants)
    ].groupby(["variant", "seed"]):
        baseline = frame[
            (frame["variant"] == "causal_hedge") & (frame["seed"] == seed)
        ]
        per_dataset: dict[str, dict[str, float]] = {}
        for dataset in sorted(frame["dataset"].unique()):
            variant_dataset = subset[subset["dataset"] == dataset]
            baseline_dataset = baseline[baseline["dataset"] == dataset]
            variant_mae = float(variant_dataset["mae"].mean())
            baseline_mae = float(baseline_dataset["mae"].mean())
            variant_rmse = float(variant_dataset["rmse"].mean())
            baseline_rmse = float(baseline_dataset["rmse"].mean())
            per_dataset[dataset] = {
                "mae": variant_mae,
                "causal_hedge_mae": baseline_mae,
                "relative_degradation_vs_causal_hedge": (
                    variant_mae - baseline_mae
                )
                / max(baseline_mae, 1e-12),
                "rmse": variant_rmse,
                "causal_hedge_rmse": baseline_rmse,
                "relative_rmse_degradation_vs_causal_hedge": (
                    variant_rmse - baseline_rmse
                )
                / max(baseline_rmse, 1e-12),
            }
        summary.append(
            {
                "variant": str(variant),
                "seed": int(seed),
                "per_dataset": per_dataset,
                "dataset_equal_macro_relative_degradation_vs_causal_hedge": float(
                    np.mean(
                        [
                            value["relative_degradation_vs_causal_hedge"]
                            for value in per_dataset.values()
                        ]
                    )
                ),
                "dataset_equal_macro_relative_rmse_degradation_vs_causal_hedge": float(
                    np.mean(
                        [
                            value["relative_rmse_degradation_vs_causal_hedge"]
                            for value in per_dataset.values()
                        ]
                    )
                ),
            }
        )
    report = {
        "schema": "selective_prior_causal_hedge_final_v1_ablations_v2",
        "test_selection_use": "none",
        "variants": [
            "w_o_prior_expert",
            "two_experts_all_prior",
            "frozen_validation_weights",
            "global_eta",
            "no_delayed_feedback_diagnostic",
        ],
        "primary_ablation_endpoint": "dataset_equal_macro_relative_degradation_vs_causal_hedge",
        "noncausal_variant_note": (
            "no_delayed_feedback_diagnostic uses same-origin target feedback and is an oracle-style "
            "diagnostic, never a deployable competitor or primary claim."
        ),
        "summary": summary,
    }
    (output / "final_v1_ablations.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output / "final_v1_ablations.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--seeds", default="42,3407,101,2026,777")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    print(write_final_v1_ablations(args.report_root, seeds=seeds))


if __name__ == "__main__":
    main()
