"""Post-hoc efficiency audit for the frozen FINAL_V1 system.

This measures storage, checkpoint parameter counts, and the additional causal
aggregation wall time on representative completed runs.  It does not alter
predictions or select any method setting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .causal_aggregation import causal_hedge_mix, robust_error_scale


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(root: Path, split: str, name: str) -> np.ndarray:
    return np.load(root / "predictions" / split / f"{name}.npy", allow_pickle=False)


def _parameter_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    return int(sum(int(value.numel()) for value in state.values()))


def _run(root: Path) -> dict[str, Any]:
    causal = _json(root / "causal_aggregation.json")
    validation_target = _array(root, "gate_validation", "targets").astype(np.float64)
    test_target = _array(root, "test", "targets").astype(np.float64)
    validation_experts = [
        _array(root, "gate_validation", name).astype(np.float64)
        for name in ("pred_all", "pred_prior", "final_static_mix", "final_loss_gate")
    ]
    test_experts = [
        _array(root, "test", name).astype(np.float64)
        for name in ("pred_all", "pred_prior", "final_static_mix", "final_loss_gate")
    ]
    validation_origins = _array(root, "gate_validation", "origin_indices").reshape(-1)
    test_origins = _array(root, "test", "origin_indices").reshape(-1)
    eta = [
        float(causal["eta_by_horizon"][str(index + 1)])
        for index in range(test_target.shape[1])
    ]
    scale = robust_error_scale(validation_target)
    _, validation_weights = causal_hedge_mix(
        validation_target,
        validation_experts,
        validation_origins,
        eta,
        error_scale=scale,
    )
    # Time one complete frozen test aggregation call.  A warm-up would double
    # the expensive full-origin pass without changing the reported overhead.
    start = time.perf_counter()
    causal_hedge_mix(
        test_target,
        test_experts,
        test_origins,
        eta,
        error_scale=scale,
        initial_weights=validation_weights[-1],
    )
    elapsed = time.perf_counter() - start
    prediction_bytes = sum(
        int((root / "predictions" / "test" / f"{name}.npy").stat().st_size)
        for name in (
            "pred_all",
            "pred_prior",
            "final_static_mix",
            "final_loss_gate",
            "final_causal_hedge",
        )
    )
    return {
        "dataset": root.parent.name,
        "seed": int(_json(root / "metrics.json")["seed"]),
        "test_origins": int(test_target.shape[0]),
        "nodes": int(test_target.shape[2]),
        "horizons": int(test_target.shape[1]),
        "causal_aggregation_seconds": float(elapsed),
        "causal_aggregation_ms_per_origin": float(1000.0 * elapsed / test_target.shape[0]),
        "test_prediction_storage_mb": float(prediction_bytes / (1024.0 * 1024.0)),
        "all_only_parameter_count": _parameter_count(root / "checkpoints" / "all_only.pt"),
        "prior_only_parameter_count": _parameter_count(root / "checkpoints" / "prior_only.pt"),
        "additional_expert_count": 2,
        "aggregation_is_posttraining": True,
    }


def write_efficiency(report_root: Path, *, datasets: list[str], seed: int) -> Path:
    root = report_root.resolve()
    rows = []
    for dataset in datasets:
        matches = sorted(root.glob(f"runs/E*/{dataset}/seed_{int(seed)}"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one completed run for {dataset}, found {matches}")
        rows.append(_run(matches[0]))
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "final_v1_efficiency.json"
    path.write_text(
        json.dumps(
            {
                "schema": "selective_prior_causal_hedge_final_v1_efficiency_v1",
                "test_selection_use": "none",
                "measurement_note": "Post-hoc CPU timing of the frozen causal aggregation over saved predictions; no training or test-based selection.",
                "datasets": datasets,
                "seed": int(seed),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_root", type=Path)
    parser.add_argument("--datasets", default="pems07,metr_la")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    print(write_efficiency(args.report_root, datasets=datasets, seed=args.seed))


if __name__ == "__main__":
    main()
