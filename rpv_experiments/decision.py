from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpv_data.synthetic import (
    selective_prior_v3_generation_identity,
    validate_selective_prior_v3_metadata,
)
from rpv_data.utils import atomic_json, sha256_file

from .config import (
    code_fingerprint,
    config_sha256,
    current_git_commit,
    protocol_version,
    resolved_data_root,
    resolved_output_root,
)
from .provenance import validate_matrix_versions

MECHANISM_EXPERIMENT_IDS = ("E1", "E2", "E3", "E4")


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _corruption_level(dataset: str) -> int | None:
    match = re.fullmatch(r"spi_(?:v3_)?synthetic_corruption_(\d{3})", str(dataset))
    return int(match.group(1)) if match else None


def _corruption_response_pass(
    response_by_level: dict[int, list[float]],
    *,
    tolerance: float,
    minimum_drop: float,
) -> bool:
    levels = (0, 10, 25, 50, 75, 100)
    if any(level not in response_by_level for level in levels):
        return False
    means = [float(np.mean(response_by_level[level])) for level in levels]
    return bool(
        all(means[index + 1] - means[index] <= float(tolerance) for index in range(len(means) - 1))
        and means[0] - means[-1] >= float(minimum_drop)
    )


def _mechanism_report_path(cfg: dict[str, Any]) -> Path:
    return resolved_output_root(cfg) / "analysis" / "mechanism_seed42_go_no_go.json"


def _expected_v3_generation_identity(cfg: dict[str, Any]) -> dict[str, object]:
    synthetic = cfg.get("synthetic", {})
    return selective_prior_v3_generation_identity(
        seed=int(cfg["randomness"]["synthetic_namespace"]),
        time_steps=int(synthetic.get("time_steps", 12000)),
        variables=int(synthetic.get("variables", 50)),
        density=float(synthetic.get("density", 0.08)),
        history=int(synthetic.get("history_contract", 48)),
        max_horizon=int(synthetic.get("maximum_horizon_contract", 12)),
    )


def write_mechanism_decision(
    cfg: dict[str, Any],
    config_path: Path,
) -> Path:
    """Evaluate the frozen Seed42 mechanism gate before formal expansion."""
    version = protocol_version(cfg)
    if version not in {"v2", "v3", "v4"}:
        raise ValueError("The mechanism gate is defined only for Protocol v2/v3/v4")
    corruption_response_metric = str(
        cfg.get("statistics", {}).get(
            "synthetic_corruption_response_metric",
            "activation_rate_greater_0_5",
        )
    )
    if version in {"v3", "v4"} and corruption_response_metric != "mean_probability":
        raise ValueError(
            "Protocol v3 E3 corruption response must use mean_probability; "
            f"got {corruption_response_metric!r}"
        )
    matrix = validate_matrix_versions(
        cfg,
        config_path,
        stage="seed42",
        experiment_ids=MECHANISM_EXPERIMENT_IDS,
        reject_extra_runs=False,
    )
    stats = cfg["statistics"]
    source_auprc_margin: list[float] = []
    source_valid_activation: list[float] = []
    source_invalid_activation: list[float] = []
    source_prior_gain: list[float] = []
    source_group_values: dict[str, dict[str, list[float]]] = {
        "prior_valid": defaultdict(list),
        "prior_invalid": defaultdict(list),
    }
    sham_activation: list[float] = []
    corruption_activation: dict[int, list[float]] = defaultdict(list)
    shared_degradation: list[float] = []
    pems08_finite = True
    generation_contracts_pass = True
    zero_corruption_prior_gain: list[float] = []
    source_metrics: list[dict[str, str]] = []
    expected_v3_identity = (
        _expected_v3_generation_identity(cfg) if version in {"v3", "v4"} else None
    )
    preservation_metric_names = (
        (
            "shared_all_gate_validation_relative_degradation",
            "shared_prior_gate_validation_relative_degradation",
        )
        if version == "v3"
        else (
            "shared_all_validation_relative_degradation",
            "shared_prior_validation_relative_degradation",
        )
        if version == "v2"
        else ()
    )
    independent_routes_declared = True
    e4_test_gain_static: list[float] = []
    e4_test_gain_loss: list[float] = []
    for record in matrix["runs"]:
        metrics_path = Path(str(record["metrics_path"]))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment_id = str(metrics["experiment_id"])
        if version in {"v3", "v4"} and experiment_id in {"E1", "E2", "E3"}:
            metadata_path = (
                resolved_data_root(cfg) / str(metrics["dataset"]) / "metadata.json"
            )
            if not metadata_path.exists():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            try:
                if expected_v3_identity is None:
                    raise RuntimeError("V3 decision lacks a generation identity")
                validate_selective_prior_v3_metadata(
                    metadata,
                    str(metrics["dataset"]),
                    expected_generation_spec=expected_v3_identity["generation_spec"],
                    expected_generator_fingerprint=str(expected_v3_identity["generator_fingerprint"]),
                )
            except (RuntimeError, TypeError):
                generation_contracts_pass = False
        source_metrics.append(
            {"path": str(metrics_path), "sha256": sha256_file(metrics_path)}
        )
        for content in metrics["horizons"].values():
            diagnostics = content["diagnostics"]
            for name in preservation_metric_names:
                if name not in diagnostics:
                    raise RuntimeError(
                        f"Seed42 mechanism metric lacks {name}: {metrics_path}"
                    )
                shared_degradation.append(max(0.0, float(diagnostics[name])))
            gate = content["gates"]["matched_control_gate"]
            if version == "v4":
                independent_routes_declared = bool(
                    independent_routes_declared
                    and metrics.get("routing_contract") == "independent_frozen_experts"
                )
            if experiment_id == "E1":
                source_auprc_margin.append(
                    float(gate["ground_truth_auprc"])
                    - float(gate["ground_truth_prevalence"])
                )
                source_valid_activation.append(
                    float(gate["ground_truth_valid_activation_rate"])
                )
                source_invalid_activation.append(
                    float(gate["ground_truth_invalid_activation_rate"])
                )
                if version in {"v3", "v4"}:
                    groups = diagnostics.get("e1_prior_value_groups")
                    if not isinstance(groups, dict):
                        raise RuntimeError(
                            f"V3 E1 metric lacks grouped prior-value diagnostics: {metrics_path}"
                        )
                    for group_name in ("prior_valid", "prior_invalid"):
                        group = groups.get(group_name)
                        if not isinstance(group, dict):
                            raise TypeError(
                                f"V3 E1 metric lacks {group_name} diagnostics: {metrics_path}"
                            )
                        for key, value in group.items():
                            source_group_values[group_name][str(key)].append(float(value))
                    source_prior_gain.append(
                        float(groups["prior_valid"]["shared_prior_relative_gain_over_all"])
                    )
            elif experiment_id == "E2":
                sham_activation.append(float(gate["activation_rate_greater_0_5"]))
            elif experiment_id == "E3":
                level = _corruption_level(str(metrics["dataset"]))
                if level is None:
                    raise RuntimeError(f"Invalid E3 dataset id: {metrics['dataset']}")
                response_value = gate.get(corruption_response_metric)
                if response_value is None:
                    raise RuntimeError(
                        f"E3 metric lacks {corruption_response_metric}: {metrics_path}"
                    )
                corruption_activation[level].append(
                    float(response_value)
                )
                if version in {"v3", "v4"} and level == 0:
                    methods = content["methods"]
                    all_mae = float(methods["shared_all"]["mae"])
                    prior_mae = float(methods["shared_prior"]["mae"])
                    zero_corruption_prior_gain.append(
                        (all_mae - prior_mae) / max(all_mae, 1e-12)
                    )
            elif experiment_id == "E4":
                e4_test_gain_static.append(
                    float(diagnostics.get("matched_gate_relative_gain_over_static", float("nan")))
                )
                e4_test_gain_loss.append(
                    float(diagnostics.get("matched_gate_relative_gain_over_loss_gate", float("nan")))
                )
                for values in content["methods"].values():
                    pems08_finite = bool(
                        pems08_finite
                        and np.isfinite(float(values["mae"]))
                        and np.isfinite(float(values["rmse"]))
                    )
    preservation_gate_name = (
        "shared_A_R_gate_validation_degradation_below_two_percent"
        if version == "v3"
        else (
            "shared_A_R_validation_degradation_below_two_percent"
            if version == "v2"
            else "independent_frozen_expert_routes_declared"
        )
    )
    minimum_drop_key = (
        "synthetic_corruption_min_mean_probability_drop"
        if version in {"v3", "v4"}
        else "synthetic_corruption_min_activation_drop"
    )
    minimum_drop = float(
        stats.get(
            minimum_drop_key,
            # Backward-compatible read for pre-v4 v3 fixtures.  New v3/v4
            # configs must declare the continuous probability key explicitly.
            stats.get("synthetic_corruption_min_activation_drop", 0.05),
        )
    )
    gates = {
        **(
            {
                (
                    "synthetic_generation_contracts_pass"
                    if version == "v4"
                    else "v3_synthetic_generation_contracts_pass"
                ): generation_contracts_pass
            }
            if version in {"v3", "v4"}
            else {}
        ),
        preservation_gate_name: (
            bool(shared_degradation and max(shared_degradation) < 0.02)
            if version in {"v2", "v3"}
            else bool(independent_routes_declared)
        ),
        "E1_prior_valid_source_recovered": bool(
            source_auprc_margin
            and min(source_auprc_margin)
            >= float(stats["synthetic_switching_min_auprc_margin"])
            and source_valid_activation
            and float(np.mean(source_valid_activation))
            >= float(stats["synthetic_switching_min_valid_activation"])
            and source_invalid_activation
            and float(np.mean(source_invalid_activation))
            <= float(stats["synthetic_switching_max_invalid_activation"])
        ),
        **(
            {
                "E1_prior_valid_has_prior_headroom": bool(
                    source_prior_gain
                    and min(source_prior_gain)
                    >= float(cfg["method"]["delta_a_relative"])
                )
            }
            if version in {"v3", "v4"}
            else {}
        ),
        "E2_sham_prior_suppressed": bool(
            sham_activation
            and float(np.mean(sham_activation))
            <= float(stats["sham_activation_maximum"])
        ),
        "E3_corruption_response_monotone": _corruption_response_pass(
            corruption_activation,
            tolerance=float(stats["synthetic_corruption_monotonic_tolerance"]),
            minimum_drop=minimum_drop,
        ),
        **(
            {
                "E3_zero_corruption_has_prior_headroom": bool(
                    zero_corruption_prior_gain
                    and float(np.mean(zero_corruption_prior_gain))
                    >= float(stats["synthetic_zero_corruption_min_prior_gain"])
                )
            }
            if version in {"v3", "v4"}
            else {}
        ),
        "E4_PEMS08_predictions_finite": bool(pems08_finite),
        **(
            {
                "E4_test_matched_gain_over_static_positive": bool(
                    e4_test_gain_static
                    and np.all(np.isfinite(e4_test_gain_static))
                    and float(np.mean(e4_test_gain_static))
                    > float(stats.get("minimum_test_relative_gain_over_static", 0.0))
                ),
                "E4_test_matched_gain_over_loss_gate_positive": bool(
                    e4_test_gain_loss
                    and np.all(np.isfinite(e4_test_gain_loss))
                    and float(np.mean(e4_test_gain_loss))
                    > float(stats.get("minimum_test_relative_gain_over_loss_gate", 0.0))
                ),
            }
            if version == "v4"
            else {}
        ),
    }
    report = {
        "schema": f"selective_prior_mechanism_decision_{version}",
        "experiment_ids": list(MECHANISM_EXPERIMENT_IDS),
        "seed": 42,
        "gates": gates,
        "proceed_formal": bool(all(gates.values())),
        "config_sha256": config_sha256(config_path),
        "code_fingerprint": code_fingerprint(),
        "git_commit_record": current_git_commit(),
        "run_git_commits_recorded": matrix["git_commits_recorded"],
        "source_metrics": source_metrics,
        "diagnostics": {
            (
                "maximum_shared_gate_validation_relative_degradation"
                if version == "v3"
                else (
                    "maximum_shared_validation_relative_degradation"
                    if version == "v2"
                    else "maximum_independent_expert_route_degradation"
                )
            ): (float(max(shared_degradation)) if shared_degradation else None),
            "minimum_E1_auprc_margin": (
                float(min(source_auprc_margin)) if source_auprc_margin else None
            ),
            "mean_E1_valid_activation": (
                float(np.mean(source_valid_activation))
                if source_valid_activation
                else None
            ),
            "mean_E1_invalid_activation": (
                float(np.mean(source_invalid_activation))
                if source_invalid_activation
                else None
            ),
            "minimum_E1_prior_valid_shared_prior_gain": (
                float(min(source_prior_gain)) if source_prior_gain else None
            ),
            "mean_E1_prior_valid_shared_prior_gain": (
                float(np.mean(source_prior_gain)) if source_prior_gain else None
            ),
            "E1_prior_value_group_means": {
                group_name: {
                    ("total_evaluable_elements" if key == "evaluable_elements" else key): (
                        float(np.sum(values))
                        if key == "evaluable_elements"
                        else float(np.mean(values))
                    )
                    for key, values in sorted(group_values.items())
                }
                for group_name, group_values in source_group_values.items()
            },
            "mean_E2_sham_activation": (
                float(np.mean(sham_activation)) if sham_activation else None
            ),
            (
                "mean_E3_response_by_corruption"
                if version == "v3"
                else "mean_E3_activation_by_corruption"
            ): {
                str(level): float(np.mean(values))
                for level, values in sorted(corruption_activation.items())
            },
            "mean_E3_zero_corruption_shared_prior_gain": (
                float(np.mean(zero_corruption_prior_gain))
                if zero_corruption_prior_gain
                else None
            ),
            "mean_E4_test_matched_gain_over_static": (
                float(np.mean(e4_test_gain_static)) if e4_test_gain_static else None
            ),
            "mean_E4_test_matched_gain_over_loss_gate": (
                float(np.mean(e4_test_gain_loss)) if e4_test_gain_loss else None
            ),
            "independent_routes_declared": bool(independent_routes_declared),
        },
    }
    path = _mechanism_report_path(cfg)
    atomic_json(path, report)
    return path


def verify_mechanism_authorization(
    cfg: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """Fail closed unless the current semantic protocol passed Seed42."""
    path = _mechanism_report_path(cfg)
    if not path.exists():
        raise RuntimeError(
            "Formal expansion is locked until the E1-E4 Seed42 mechanism decision exists"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    version = protocol_version(cfg)
    expected = {
        "schema": f"selective_prior_mechanism_decision_{version}",
        "config_sha256": config_sha256(config_path),
        "code_fingerprint": code_fingerprint(),
        "proceed_formal": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"Seed42 mechanism authorization mismatch for {key}: "
                f"{report.get(key)!r} != {value!r}"
            )
    for record in report.get("source_metrics", []):
        metrics_path = Path(str(record.get("path", "")))
        if not metrics_path.exists() or sha256_file(metrics_path) != record.get("sha256"):
            raise RuntimeError(
                f"Seed42 mechanism evidence changed after authorization: {metrics_path}"
            )
    return report


def write_decision(cfg: dict[str, Any]) -> Path:
    root = resolved_output_root(cfg)
    comparison_path = root / "analysis" / "paired_comparisons.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Run the formal analyze command before decision")
    comparison = pd.read_csv(comparison_path)
    tail_path = root / "analysis" / "tail_noninferiority.csv"
    if not tail_path.exists():
        raise FileNotFoundError("Formal analysis did not produce tail_noninferiority.csv")
    tail = pd.read_csv(tail_path)
    metrics_paths = sorted((root / "runs").glob("E[1-9]/*/seed_*/metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError("No E1-E9 metrics found")
    stats = cfg["statistics"]
    is_v2 = cfg.get("schema") == "selective_prior_injection_v2"
    minimum_gain = float(stats["minimum_relative_mae_improvement"])
    minimum_units = int(stats.get("minimum_real_units_each_primary_comparison", 2))
    primary_comparators = tuple(stats.get("primary_comparators", ["static_mix", "loss_gate"]))
    ablation_comparators = tuple(stats.get("ablation_comparators", ["random_control_gate"]))
    real = comparison[comparison["experiment_id"].isin([f"E{i}" for i in range(4, 10)])]
    passing_sets: dict[str, set[tuple[str, str]]] = {}
    for comparator in (*primary_comparators, *ablation_comparators):
        selected = real[
            (real["comparator"] == comparator)
            & (real["relative_improvement"] >= minimum_gain)
            & (real["ci95_low"] > 0.0)
            & (real["holm_adjusted_p"] < 0.05)
        ]
        passing_sets[comparator] = {
            (str(row.experiment_id), str(row.dataset))
            for row in selected.itertuples(index=False)
        }
    primary_sets = [passing_sets[comparator] for comparator in primary_comparators]
    joint = set.intersection(*primary_sets) if primary_sets else set()
    random_ablation_units = passing_sets.get("random_control_gate", set())

    diagnostics: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    sham_activation = []
    source_auprc_margin = []
    source_valid_activation = []
    source_invalid_activation = []
    corruption_activation: dict[int, list[float]] = defaultdict(list)
    for path in metrics_paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for horizon, content in metrics["horizons"].items():
            key = (str(metrics["dataset"]), int(horizon))
            diagnostics[key].append(content["diagnostics"])
            matched_gate = content["gates"]["matched_control_gate"]
            if metrics["experiment_id"] == "E2":
                sham_activation.append(float(matched_gate["activation_rate_greater_0_5"]))
            if metrics["experiment_id"] == "E1":
                auprc = matched_gate.get("ground_truth_auprc")
                prevalence = matched_gate.get("ground_truth_prevalence")
                if auprc is not None and prevalence is not None:
                    source_auprc_margin.append(float(auprc) - float(prevalence))
                valid_activation = matched_gate.get("ground_truth_valid_activation_rate")
                invalid_activation = matched_gate.get("ground_truth_invalid_activation_rate")
                if valid_activation is not None and np.isfinite(float(valid_activation)):
                    source_valid_activation.append(float(valid_activation))
                if invalid_activation is not None and np.isfinite(float(invalid_activation)):
                    source_invalid_activation.append(float(invalid_activation))
            level = _corruption_level(str(metrics["dataset"]))
            if metrics["experiment_id"] == "E3" and level is not None:
                activation = matched_gate.get("activation_rate_greater_0_5")
                if activation is not None and np.isfinite(float(activation)):
                    corruption_activation[level].append(float(activation))

    shared_degradation = []
    headroom_recovery = []
    headroom_cells = 0
    for (dataset, _), rows in diagnostics.items():
        # The shared A/R preservation gate is a protocol-wide sanity check,
        # including the synthetic mechanism units.  Use gate-validation
        # degradation (the checkpoint-selection split), while retaining the
        # test degradation in each metrics row as a reporting diagnostic.
        shared_degradation.extend(
            max(0.0, float(row[name]))
            for row in rows
            for name in (
                "shared_all_validation_relative_degradation",
                "shared_prior_validation_relative_degradation",
            )
            if name in row
        )
        # Historical v1 artifacts did not save validation branch diagnostics;
        # preserve their auditability without allowing v2 to silently fall
        # back to test values.
        if not is_v2:
            shared_degradation.extend(
                max(0.0, float(row[name]))
                for row in rows
                for name in (
                    "shared_all_relative_degradation",
                    "shared_prior_relative_degradation",
                    "shared_all_test_relative_degradation",
                    "shared_prior_test_relative_degradation",
                )
                if name in row
            )
        if is_v2 and not any(
            "shared_all_validation_relative_degradation" in row
            and "shared_prior_validation_relative_degradation" in row
            for row in rows
        ):
            raise RuntimeError(
                f"v2 metrics lack gate-validation shared-branch diagnostics for {dataset}"
            )
        mean_headroom = float(np.mean([row["oracle_headroom_from_static"] for row in rows]))
        if mean_headroom >= float(stats.get("headroom_threshold", 0.02)):
            headroom_cells += 1
            # Recovery is stored with the gate metrics, so recover from the equivalent diagnostic gains.
            recovery = [
                row["matched_gate_relative_gain_over_static"] / row["oracle_headroom_from_static"]
                for row in rows
                if row["oracle_headroom_from_static"] > 1e-12
            ]
            headroom_recovery.extend(recovery)
    recovery_values = _finite(headroom_recovery)
    real_tail = tail[tail["experiment_id"].isin([f"E{i}" for i in range(4, 10)])]
    tail_margin = float(stats["tail_relative_noninferiority_margin"])
    tail_pass = np.asarray(real_tail["ci95_high"] < tail_margin, dtype=bool)

    gates = {
        "shared_modes_within_two_percent": bool(shared_degradation and max(shared_degradation) < 0.02),
        "at_least_two_joint_primary_real_units": len(joint) >= minimum_units,
        "random_control_ablation_support": len(random_ablation_units)
        >= int(stats.get("minimum_random_ablation_real_units", 2)),
        "headroom_present": headroom_cells >= int(stats.get("minimum_headroom_cells", 2)),
        "median_headroom_recovery_at_least_thirty_percent": bool(
            recovery_values.size
            and float(np.median(recovery_values)) >= float(stats["minimum_headroom_recovery"])
        ),
        "sham_activation_at_most_ten_percent": bool(
            sham_activation and float(np.mean(sham_activation)) <= float(stats["sham_activation_maximum"])
        ),
        "synthetic_source_auprc_above_prevalence": bool(
            source_auprc_margin and min(source_auprc_margin) > 0.0
        ),
        # Synthetic source recovery is a hard v2 mechanism gate.  A model
        # must open on the known prior-valid regime while remaining mostly
        # closed on the local/common/alternative regimes.  These metrics are
        # computed from test predictions and are never used to tune the gate.
        "synthetic_switching_source_recovery": (
            bool(
                source_auprc_margin
                and min(source_auprc_margin)
                >= float(stats.get("synthetic_switching_min_auprc_margin", 0.05))
                and source_valid_activation
                and float(np.mean(source_valid_activation))
                >= float(stats.get("synthetic_switching_min_valid_activation", 0.20))
                and source_invalid_activation
                and float(np.mean(source_invalid_activation))
                <= float(stats.get("synthetic_switching_max_invalid_activation", 0.10))
            )
            if is_v2
            else True
        ),
        # E3 is a mechanism curve, not a generic accuracy sweep: as the
        # supplied prior is progressively corrupted, the trust signal must
        # not increase and must show a predeclared minimum drop.
        "synthetic_corruption_monotonic_response": (
            _corruption_response_pass(
                corruption_activation,
                tolerance=float(stats.get("synthetic_corruption_monotonic_tolerance", 0.02)),
                minimum_drop=float(stats.get("synthetic_corruption_min_activation_drop", 0.05)),
            )
            if is_v2
            else True
        ),
        "tail_noninferiority": bool(
            len(real_tail)
            and float(np.mean(tail_pass))
            >= float(stats.get("tail_required_fraction", 1.0))
        ),
    }
    primary_gates = (
        {
            key: value
            for key, value in gates.items()
            if key != "random_control_ablation_support"
        }
        if is_v2
        else gates
    )
    diagnostic_gates = {
        "random_control_ablation_support": gates["random_control_ablation_support"]
    }
    report = {
        "schema": (
            "selective_prior_injection_decision_v2"
            if cfg.get("schema") == "selective_prior_injection_v2"
            else "selective_prior_injection_decision_v1"
        ),
        "gates": gates,
        "primary_gates": primary_gates,
        "diagnostic_gates": diagnostic_gates,
        "proceed_external": bool(all(primary_gates.values())),
        "joint_passing_real_units": [
            f"{experiment_id}:{dataset}" for experiment_id, dataset in sorted(joint)
        ],
        "passing_units_by_comparator": {
            comparator: [
                f"{experiment_id}:{dataset}" for experiment_id, dataset in sorted(units)
            ]
            for comparator, units in passing_sets.items()
        },
        "diagnostics": {
            "maximum_shared_relative_degradation": float(max(shared_degradation)) if shared_degradation else None,
            "headroom_cells": headroom_cells,
            "median_headroom_recovery": float(np.median(recovery_values)) if recovery_values.size else None,
            "mean_sham_activation": float(np.mean(sham_activation)) if sham_activation else None,
            "minimum_source_auprc_margin": min(source_auprc_margin) if source_auprc_margin else None,
            "mean_source_valid_activation": (
                float(np.mean(source_valid_activation)) if source_valid_activation else None
            ),
            "mean_source_invalid_activation": (
                float(np.mean(source_invalid_activation)) if source_invalid_activation else None
            ),
            "corruption_activation_by_level": {
                str(level): float(np.mean(values))
                for level, values in sorted(corruption_activation.items())
            },
            "tail_noninferiority_fraction": float(
                np.mean(tail_pass)
            )
            if len(real_tail)
            else None,
        },
        "paired_comparisons_sha256": sha256_file(comparison_path),
        "tail_noninferiority_sha256": sha256_file(tail_path),
    }
    output = root / "analysis" / "go_no_go.json"
    atomic_json(output, report)
    return output
