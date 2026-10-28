"""Read-only standard traffic-benchmark reporting for FINAL_V1.

The locked primary endpoint remains horizons 3/6/12.  This module produces a
secondary table that is easier to compare with traffic-forecasting papers:
PEMS03/04/07/08 use the mean over all twelve forecast steps, while METR-LA
reports 3/6/12.  It never selects a method or hyperparameter from test data.
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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(root: Path, name: str) -> np.ndarray:
    return np.load(root / "predictions" / "test" / f"{name}.npy", allow_pickle=False)


def _metric(root: Path, horizons: list[int]) -> dict[str, Any]:
    metrics = _json(root / "metrics.json")
    target = np.asarray(_array(root, "targets"), dtype=np.float64)
    predictions = {
        name: np.asarray(_array(root, name), dtype=np.float64)
        for name in ("final_causal_hedge", "final_static_mix", "final_loss_gate")
    }
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        y = target[:, int(horizon) - 1]
        errors = {
            name: prediction[:, int(horizon) - 1] - y
            for name, prediction in predictions.items()
        }
        mae = {name: float(np.mean(np.abs(error))) for name, error in errors.items()}
        rows.append(
            {
                "dataset": str(metrics["dataset"]),
                "seed": int(metrics["seed"]),
                "horizon": int(horizon),
                "mae_causal_hedge": mae["final_causal_hedge"],
                "mae_static_mix": mae["final_static_mix"],
                "mae_loss_gate": mae["final_loss_gate"],
                "relative_gain_over_static_mix": (
                    mae["final_static_mix"] - mae["final_causal_hedge"]
                )
                / max(mae["final_static_mix"], 1e-12),
                "relative_gain_over_loss_gate": (
                    mae["final_loss_gate"] - mae["final_causal_hedge"]
                )
                / max(mae["final_loss_gate"], 1e-12),
            }
        )
    return {"rows": rows, "test_origins": int(target.shape[0])}


def write_standard_reporting(
    report_root: Path, *, seeds: list[int] | None = None
) -> Path:
    root = report_root.resolve()
    selected = [int(value) for value in (seeds or EXPECTED_SEEDS)]
    rows: list[dict[str, Any]] = []
    for seed in selected:
        for run_root in sorted(root.glob(f"runs/E*/*/seed_{seed}")):
            dataset = run_root.parent.name
            if dataset not in EXPECTED_DATASETS:
                continue
            horizons = list(range(1, 13)) if dataset != "metr_la" else [3, 6, 12]
            rows.extend(_metric(run_root, horizons)["rows"])
    frame = pd.DataFrame(rows).sort_values(["dataset", "seed", "horizon"])
    if len(frame) != len(selected) * (4 * 12 + 3):
        raise RuntimeError("Standard reporting requires all five datasets and selected seeds")
    summary: list[dict[str, Any]] = []
    for (dataset, seed), subset in frame.groupby(["dataset", "seed"]):
        summary.append(
            {
                "dataset": str(dataset),
                "seed": int(seed),
                "horizons": sorted(int(value) for value in subset["horizon"]),
                "mean_mae_causal_hedge": float(subset["mae_causal_hedge"].mean()),
                "mean_mae_static_mix": float(subset["mae_static_mix"].mean()),
                "mean_mae_loss_gate": float(subset["mae_loss_gate"].mean()),
                "mean_relative_gain_over_static_mix": float(
                    subset["relative_gain_over_static_mix"].mean()
                ),
                "mean_relative_gain_over_loss_gate": float(
                    subset["relative_gain_over_loss_gate"].mean()
                ),
            }
        )
    macro: dict[str, float] = {}
    for comparator in ("static_mix", "loss_gate"):
        per_seed = []
        for seed in selected:
            dataset_values = [
                next(
                    row[f"mean_relative_gain_over_{comparator}" ]
                    for row in summary
                    if row["dataset"] == dataset and row["seed"] == seed
                )
                for dataset in sorted(EXPECTED_DATASETS)
            ]
            per_seed.append(float(np.mean(dataset_values)))
        macro[f"over_{comparator}"] = {
            "per_seed": per_seed,
            "mean": float(np.mean(per_seed)),
            "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        }
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "final_v1_standard_traffic_per_horizon.csv", index=False)
    path = output / "final_v1_standard_traffic_reporting.json"
    path.write_text(
        json.dumps(
            {
                "schema": "selective_prior_causal_hedge_final_v1_standard_traffic_reporting_v1",
                "test_selection_use": "none",
                "primary_endpoint_unchanged": "horizons_3_6_12_dataset_equal_macro_ratio_of_aggregate_mae",
                "horizon_policy": {
                    "pems03_pems04_pems07_pems08": "1_to_12_mean",
                    "metr_la": "3_6_12",
                },
                "seeds": selected,
                "per_dataset_seed": summary,
                "dataset_equal_macro": macro,
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
    parser.add_argument("--seeds", default="42,3407,101,2026,777")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    print(write_standard_reporting(args.report_root, seeds=seeds))


if __name__ == "__main__":
    main()
