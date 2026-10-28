from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FORMAL_EXPERIMENT_IDS = tuple(f"E{i}" for i in range(1, 11))
FORMAL_OPTIMIZATION_SEEDS = (42, 1384143919, 337881674, 822358677, 324271686)
FINAL_V1_OPTIMIZATION_SEEDS = (42, 3407, 101, 2026, 777)
SPLIT_KEYS = (
    "forecaster_train",
    "forecaster_validation",
    "gate_train",
    "gate_validation",
    "test",
)


def protocol_version(cfg: dict[str, Any]) -> str:
    schema = str(cfg.get("schema", ""))
    versions = {
        "selective_prior_injection_v1": "v1",
        "selective_prior_injection_v2": "v2",
        "selective_prior_injection_v3": "v3",
        "selective_prior_injection_v4": "v4",
        "selective_prior_injection_v5": "v5",
        "selective_prior_injection_final_v1": "final_v1",
    }
    if schema not in versions:
        raise ValueError(f"Unsupported selective-prior schema: {schema}")
    return versions[schema]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _under_external_root(value: str | Path, environment: str, conventional_prefix: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    external = os.environ.get(environment)
    if not external:
        return _resolve(path)
    parts = list(path.parts)
    if parts and parts[0].lower() == conventional_prefix.lower():
        parts = parts[1:]
    return Path(external).expanduser().resolve().joinpath(*parts)


def resolved_data_root(cfg: dict[str, Any]) -> Path:
    return _under_external_root(str(cfg["data_root"]), "RPV_DATA_ROOT", "data")


def resolved_output_root(cfg: dict[str, Any]) -> Path:
    return _under_external_root(str(cfg["output_root"]), "RPV_REPORT_ROOT", "reports")


def config_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable_config(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _validate_split(split: dict[str, Any]) -> None:
    missing = [key for key in SPLIT_KEYS if key not in split]
    if missing:
        raise ValueError(f"split is missing intervals: {missing}")
    values = [float(split[key]) for key in SPLIT_KEYS]
    if any(value <= 0 for value in values) or abs(sum(values) - 1.0) > 1e-12:
        raise ValueError("The five chronological split fractions must be positive and sum to one")
    if values != [0.60, 0.10, 0.10, 0.10, 0.10]:
        raise ValueError("The active protocol freezes the chronological split at 60/10/10/10/10")


def _validate_seeds(cfg: dict[str, Any], formal: bool) -> None:
    seeds = cfg.get("seeds", {})
    pilot = tuple(int(value) for value in seeds.get("pilot", []))
    full = tuple(int(value) for value in seeds.get("formal", []))
    if pilot != (42,):
        raise ValueError("The pilot seed is frozen to [42]")
    if formal:
        expected = FINAL_V1_OPTIMIZATION_SEEDS if cfg.get("schema") == "selective_prior_injection_final_v1" else FORMAL_OPTIMIZATION_SEEDS
        if full != expected:
            raise ValueError(f"Formal optimization seeds must be {list(expected)}")
    if len(set(full)) != len(full) or any(b - a == 1 for a in full for b in full if b > a):
        raise ValueError("Formal seeds must be unique and non-consecutive")


def _validate_method(cfg: dict[str, Any], formal: bool) -> None:
    method = cfg.setdefault("method", {})
    controls = int(method.get("matched_controls", 20))
    if formal and controls != 20:
        raise ValueError("Formal runs require exactly 20 matched structural controls")
    if controls < 2:
        raise ValueError("matched_controls must be at least two")
    expected = {
        "rho": 0.8,
        "delta_s": 0.0,
        "delta_a_relative": 0.005,
        "tau_a": 0.05,
        "tau_s": 0.05,
        "tau_w": 0.05,
        "gate_lambda": 0.3,
    }
    if formal:
        for key, frozen in expected.items():
            actual = float(method.get(key, frozen))
            if abs(actual - frozen) > 1e-12:
                raise ValueError(f"Formal method parameter {key} is frozen to {frozen}, got {actual}")
    method.setdefault("matched_controls", controls)
    for key, value in expected.items():
        method.setdefault(key, value)
    method.setdefault("minimum_endpoint_change", 0.80)
    if formal and abs(float(method["minimum_endpoint_change"]) - 0.80) > 1e-12:
        raise ValueError("Formal minimum_endpoint_change is frozen to 0.80")
    method.setdefault("shared_validation_controls", [0, 5, 10, 15])
    if any(int(index) < 0 or int(index) >= controls for index in method["shared_validation_controls"]):
        raise ValueError("shared_validation_controls contains an unavailable control index")
    exposure_count = int(method.get("control_exposure_count", max(1, controls - 4)))
    probe_indices = [int(index) for index in method.get("control_probe_indices", range(exposure_count, controls))]
    allow_probe_only = cfg.get("schema") in {
        "selective_prior_injection_v4",
        "selective_prior_injection_v5",
        "selective_prior_injection_final_v1",
    }
    if exposure_count < (0 if allow_probe_only else 1) or exposure_count > controls:
        lower = "zero" if allow_probe_only else "one"
        raise ValueError(f"control_exposure_count must be between {lower} and matched_controls")
    if not probe_indices or any(index < 0 or index >= controls for index in probe_indices):
        raise ValueError("control_probe_indices must contain valid matched-control indices")
    if set(probe_indices) & set(range(exposure_count)):
        raise ValueError("control probe controls must be held out from exposure controls")
    if (
        any(int(index) >= exposure_count for index in method["shared_validation_controls"])
        and not (allow_probe_only and not method["shared_validation_controls"])
    ):
        raise ValueError("shared_validation_controls must refer to exposure controls")
    method.setdefault("control_exposure_count", exposure_count)
    method.setdefault("control_probe_indices", probe_indices)
    method.setdefault("shared_control_loss_weight", 0.20)
    method.setdefault("shared_validation_control_weight", 0.0)
    method.setdefault("specificity_gate_lambda", 0.30)
    method.setdefault("specificity_target", "thresholded_win_rate")
    method.setdefault("specificity_chance", 0.50)
    method.setdefault("shared_distillation_weight", 0.0)
    method.setdefault("shared_checkpoint_selection", "mean_route_mae")
    method.setdefault("shared_checkpoint_selection_split", "forecaster_validation")
    method.setdefault("shared_hard_preservation_gate_split", "gate_validation")
    method.setdefault("gate_combiner", "legacy_single_logit")
    if not 0.0 <= float(method["shared_control_loss_weight"]) <= 1.0:
        raise ValueError("shared_control_loss_weight must be in [0,1]")
    if not 0.0 <= float(method["shared_validation_control_weight"]) <= 1.0:
        raise ValueError("shared_validation_control_weight must be in [0,1]")
    if float(method["specificity_gate_lambda"]) < 0.0:
        raise ValueError("specificity_gate_lambda must be non-negative")
    if not 0.0 <= float(method["shared_distillation_weight"]) <= 1.0:
        raise ValueError("shared_distillation_weight must be in [0,1]")
    if formal and cfg.get("schema") == "selective_prior_injection_v2":
        if exposure_count != 16 or probe_indices != [16, 17, 18, 19]:
            raise ValueError("Protocol v2 freezes 16 exposure and four held-out probe controls")
        if abs(float(method["shared_control_loss_weight"]) - 0.20) > 1e-12:
            raise ValueError("Protocol v2 shared_control_loss_weight is frozen to 0.20")
        if abs(float(method["shared_validation_control_weight"])) > 1e-12:
            raise ValueError("Protocol v2 selects checkpoints on A/R validation only")
        if abs(float(method["specificity_gate_lambda"]) - 0.30) > 1e-12:
            raise ValueError("Protocol v2 specificity_gate_lambda is frozen to 0.30")
    if cfg.get("schema") == "selective_prior_injection_v3":
        expected_v3 = {
            "matched_controls": 20,
            "control_exposure_count": 10,
            "control_probe_indices": list(range(10, 20)),
            "shared_validation_controls": [0, 3, 6, 9],
            "specificity_target": "positive_excess_win_rate",
            "specificity_chance": 0.50,
            "shared_control_loss_weight": 0.10,
            "shared_validation_control_weight": 0.0,
            "shared_distillation_weight": 0.25,
            "shared_checkpoint_selection": "minimax_raw_relative_degradation",
            "shared_checkpoint_selection_split": "forecaster_validation",
            "shared_hard_preservation_gate_split": "gate_validation",
            "gate_combiner": "geometric_mean",
        }
        for key, expected_value in expected_v3.items():
            if method.get(key) != expected_value:
                raise ValueError(
                    f"Protocol v3 method.{key} is frozen to {expected_value!r}, "
                    f"got {method.get(key)!r}"
                )
        response_metric = cfg.setdefault("statistics", {}).get(
            "synthetic_corruption_response_metric"
        )
        if response_metric != "mean_probability":
            raise ValueError(
                "Protocol v3 statistics.synthetic_corruption_response_metric "
                "is frozen to 'mean_probability'"
            )
    if cfg.get("schema") == "selective_prior_injection_v4":
        expected_v4 = {
            "matched_controls": 20,
            "control_exposure_count": 0,
            "control_probe_indices": list(range(20)),
            "shared_validation_controls": [],
            "specificity_target": "positive_excess_win_rate",
            "specificity_chance": 0.50,
            "shared_control_loss_weight": 0.0,
            "shared_validation_control_weight": 0.0,
            "shared_distillation_weight": 0.0,
            "shared_checkpoint_selection": "none",
            "shared_checkpoint_selection_split": "none",
            "shared_hard_preservation_gate_split": "none",
            "gate_combiner": "geometric_mean",
            "routing_mode": "independent_frozen_experts",
            "gate_probability_calibration": "validation_shrinkage_by_horizon",
        }
        for key, expected_value in expected_v4.items():
            if method.get(key) != expected_value:
                raise ValueError(
                    f"Protocol v4 method.{key} is frozen to {expected_value!r}, "
                    f"got {method.get(key)!r}"
                )
        response_metric = cfg.setdefault("statistics", {}).get(
            "synthetic_corruption_response_metric"
        )
        if response_metric != "mean_probability":
            raise ValueError(
                "Protocol v4 statistics.synthetic_corruption_response_metric "
                "is frozen to 'mean_probability'"
            )
    if cfg.get("schema") in {"selective_prior_injection_v5", "selective_prior_injection_final_v1"}:
        expected_v5 = {
            "matched_controls": 20,
            "control_exposure_count": 0,
            "control_probe_indices": list(range(20)),
            "shared_validation_controls": [],
            "specificity_target": "positive_excess_win_rate",
            "specificity_chance": 0.50,
            "shared_control_loss_weight": 0.0,
            "shared_validation_control_weight": 0.0,
            "shared_distillation_weight": 0.0,
            "shared_checkpoint_selection": "none",
            "shared_checkpoint_selection_split": "none",
            "shared_hard_preservation_gate_split": "none",
            "gate_combiner": "geometric_mean",
            "routing_mode": "causal_hedge_aggregation",
            "gate_probability_calibration": "validation_shrinkage_by_horizon",
            "causal_feedback": "delayed_until_target_enters_history",
            "causal_aggregation_experts": ["all_only", "prior_only", "static_mix", "loss_gate"],
            "causal_eta_grid": [0.01, 0.03, 0.1, 0.3, 1.0],
        }
        for key, expected_value in expected_v5.items():
            if method.get(key) != expected_value:
                raise ValueError(
                    f"Frozen causal-Hedge method.{key} is frozen to {expected_value!r}, "
                    f"got {method.get(key)!r}"
                )
        eta_grid = [float(value) for value in method.get("causal_eta_grid", [])]
        if any(value < 0.0 for value in eta_grid) or eta_grid != [0.01, 0.03, 0.1, 0.3, 1.0]:
            raise ValueError("Causal-Hedge eta grid is frozen to [0.01, 0.03, 0.1, 0.3, 1.0]")


def _validate_experiments(cfg: dict[str, Any], formal: bool) -> None:
    experiments = cfg.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    ids = tuple(str(experiment.get("id")) for experiment in experiments)
    if len(set(ids)) != len(ids):
        raise ValueError("Experiment ids must be unique")
    if formal:
        expected_ids = tuple(f"E{i}" for i in range(1, 6)) if cfg.get("schema") == "selective_prior_injection_final_v1" else FORMAL_EXPERIMENT_IDS
        if ids != expected_ids:
            raise ValueError(f"Formal protocol must declare {expected_ids} in order, got {ids}")
    random_control_units: set[tuple[str, str]] = set()
    local_context_units: set[tuple[str, str]] = set()
    e3_sampling_groups: set[str] = set()
    for experiment in experiments:
        if not experiment.get("title") or not experiment.get("units"):
            raise ValueError(f"Experiment {experiment.get('id')} needs a title and at least one unit")
        if str(experiment.get("id")) == "E10" and experiment.get("locked") is not True:
            raise ValueError("E10 must be declared locked")
        for unit in experiment["units"]:
            for key in ("dataset", "history", "max_horizon", "report_horizons"):
                if key not in unit:
                    raise ValueError(f"{experiment['id']} unit lacks {key}")
            history = int(unit["history"])
            max_horizon = int(unit["max_horizon"])
            horizons = [int(value) for value in unit["report_horizons"]]
            if history < 2 or max_horizon < 1 or not horizons:
                raise ValueError(f"Invalid history/horizon declaration for {unit['dataset']}")
            if any(value < 1 or value > max_horizon for value in horizons):
                raise ValueError(f"Reported horizon exceeds max_horizon for {unit['dataset']}")
            if "random_control_ablation" in unit and not isinstance(
                unit["random_control_ablation"], bool
            ):
                raise ValueError("random_control_ablation must be true or false")
            if "local_context_baseline" in unit and not isinstance(
                unit["local_context_baseline"], bool
            ):
                raise ValueError("local_context_baseline must be true or false")
            if bool(unit.get("random_control_ablation", False)):
                random_control_units.add((str(experiment["id"]), str(unit["dataset"])))
            if bool(unit.get("local_context_baseline", False)):
                local_context_units.add((str(experiment["id"]), str(unit["dataset"])))
            if str(experiment.get("id")) == "E3" and cfg.get("schema") != "selective_prior_injection_final_v1":
                group = str(unit.get("window_sampling_group", ""))
                if not group:
                    raise ValueError("Every E3 unit must declare window_sampling_group")
                e3_sampling_groups.add(group)
    if formal and cfg.get("schema") != "selective_prior_injection_final_v1":
        if e3_sampling_groups != {"spi_synthetic_corruption"}:
            raise ValueError(
                "Formal E3 units must share window_sampling_group=spi_synthetic_corruption"
            )
        expected_random_control_units = {
            ("E1", "spi_synthetic_switching"),
            ("E5", "pems04"),
            ("E8", "causalrivers_bavaria15m_s50_00"),
            ("E9", "coast_dev_los_angeles_bight"),
        }
        if random_control_units != expected_random_control_units:
            raise ValueError(
                "Formal random-control ablation units are frozen to one synthetic, traffic, river, and coast representative"
            )
        if local_context_units != expected_random_control_units:
            raise ValueError(
                "Formal Local-only context units are frozen to the same synthetic, traffic, river, and coast representatives"
            )


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = _resolve(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if cfg.get("schema") not in {
        "selective_prior_injection_v1",
        "selective_prior_injection_v2",
        "selective_prior_injection_v3",
        "selective_prior_injection_v4",
        "selective_prior_injection_v5",
        "selective_prior_injection_final_v1",
    }:
        raise ValueError(
            "Config must declare schema: selective_prior_injection_v1, "
            "selective_prior_injection_v2, selective_prior_injection_v3, "
            "or selective_prior_injection_v4, selective_prior_injection_v5, selective_prior_injection_final_v1"
        )
    formal = str(cfg.get("mode", "formal")) == "formal"
    _validate_split(cfg.setdefault("split", {}))
    _validate_seeds(cfg, formal=formal)
    _validate_method(cfg, formal=formal)
    _validate_experiments(cfg, formal=formal)
    training = cfg.setdefault("training", {})
    defaults = {
        "batch_size": 64,
        "micro_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "epochs": 300,
        "patience": 30,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "hidden": 64,
        "layers": 2,
        "heads": 4,
        "dropout": 0.1,
        "grad_clip": 1.0,
        "num_workers": 0,
    }
    for key, value in defaults.items():
        training.setdefault(key, value)
    if int(training["batch_size"]) != int(training["micro_batch_size"]) * int(training["gradient_accumulation_steps"]):
        raise ValueError("batch_size must equal micro_batch_size * gradient_accumulation_steps")
    cfg.setdefault("device", "cuda")
    cfg.setdefault("data_root", "data/processed")
    version = protocol_version(cfg)
    cfg.setdefault("output_root", f"reports/selective_prior_injection_{version}")
    cfg.setdefault("lock_file", f"locks/selective_prior_injection_{version}.json")
    if formal:
        if str(cfg["device"]).lower() != "cuda":
            raise ValueError(
                f"Formal {cfg.get('schema')} requires CUDA and forbids silent CPU fallback"
            )
        expected_output_root = f"reports/selective_prior_injection_{version}"
        if str(cfg["output_root"]).replace("\\", "/") != expected_output_root:
            raise ValueError(f"Formal output_root is frozen to {expected_output_root}")
        randomness = cfg.get("randomness", {})
        expected_namespaces = {
            "synthetic_namespace": 20260810,
            "matched_control_namespace": 20260811,
            "random_control_namespace": 20260812,
            "window_sampling_namespace": 20260813,
        }
        for key, value in expected_namespaces.items():
            if int(randomness.get(key, -1)) != value:
                raise ValueError(f"Formal randomness.{key} is frozen to {value}")
        storage = cfg.get("prediction_storage", {})
        if storage.get("prediction_dtype") != "float32" or storage.get("matched_dtype") != "float32":
            raise ValueError("Formal predictions and structural-control predictions must be stored as float32")
        if storage.get("matched_compressed") is not True:
            raise ValueError("Formal structural-control predictions use lossless compressed NPZ storage")
        statistics = cfg.get("statistics", {})
        if statistics.get("primary_comparators") != ["static_mix", "loss_gate"]:
            raise ValueError("Formal primary comparators are frozen to Static Mix and Loss-only Gate")
        if statistics.get("ablation_comparators") != ["random_control_gate"]:
            raise ValueError("Formal ablation comparator is frozen to Random-Control Gate")
        frozen_statistics = {
            "paired_moving_block_bootstrap_replicates": 5000,
            "minimum_relative_mae_improvement": 0.005,
            "minimum_real_units_each_primary_comparison": 2,
            "minimum_random_ablation_real_units": 2,
            "minimum_headroom_recovery": 0.30,
            "sham_activation_maximum": 0.10,
            "tail_relative_noninferiority_margin": 0.01,
            "synthetic_switching_min_auprc_margin": 0.05,
            "synthetic_switching_min_valid_activation": 0.20,
            "synthetic_switching_max_invalid_activation": 0.10,
            "synthetic_corruption_min_activation_drop": 0.05,
            "synthetic_corruption_monotonic_tolerance": 0.02,
        }
        for key, value in frozen_statistics.items():
            if key not in statistics or abs(float(statistics[key]) - value) > 1e-12:
                raise ValueError(f"Formal statistics.{key} is frozen to {value}")
    cfg.setdefault("max_windows", {})
    for split_name, default in {
        "forecaster_train": 4096,
        "forecaster_validation": 1024,
        "gate_train": 2048,
        "gate_validation": 1024,
        "test": 2048,
    }.items():
        cfg["max_windows"].setdefault(split_name, default)
    return cfg, config_path


def iter_units(cfg: dict[str, Any], experiment_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = {str(value) for value in experiment_ids} if experiment_ids is not None else None
    units: list[dict[str, Any]] = []
    for experiment in cfg["experiments"]:
        experiment_id = str(experiment["id"])
        if selected is not None and experiment_id not in selected:
            continue
        for index, raw in enumerate(experiment["units"]):
            unit = dict(raw)
            unit["experiment_id"] = experiment_id
            unit["experiment_title"] = str(experiment["title"])
            unit["locked"] = bool(experiment.get("locked", False))
            unit["unit_index"] = int(index)
            units.append(unit)
    return units


def seeds_for_stage(cfg: dict[str, Any], stage: str) -> list[int]:
    if stage == "seed42":
        return [int(value) for value in cfg["seeds"]["pilot"]]
    if stage in {"formal", "external"}:
        return [int(value) for value in cfg["seeds"]["formal"]]
    raise ValueError("stage must be seed42, formal, or external")


def units_for_stage(cfg: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    units = iter_units(cfg)
    if stage in {"seed42", "formal"}:
        return [unit for unit in units if not unit["locked"]]
    if stage == "external":
        return [unit for unit in units if unit["locked"]]
    raise ValueError("stage must be seed42, formal, or external")


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "rpv_experiments").glob("*.py"))
    paths += sorted((ROOT / "rpv_data").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def lock_path(cfg: dict[str, Any]) -> Path:
    return _resolve(str(cfg["lock_file"]))


def verify_external_lock(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    path = lock_path(cfg)
    if not path.exists():
        raise RuntimeError(f"E10 is locked; create the study lock first: {path}")
    with path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    version = protocol_version(cfg)
    expected = {
        "schema": f"selective_prior_injection_lock_{version}",
        "config_sha256": config_sha256(config_path),
        "code_fingerprint": code_fingerprint(),
        "selection_locked": True,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"Study lock mismatch for {key}: {lock.get(key)!r} != {value!r}")
    decision_path = resolved_output_root(cfg) / "analysis" / "go_no_go.json"
    if not decision_path.exists() or config_sha256(decision_path) != lock.get("decision_sha256"):
        raise RuntimeError("E10 study lock no longer matches the frozen go/no-go decision")
    for record in lock.get("development_states", []):
        state_path = Path(str(record["path"]))
        artifacts = {
            "state_sha256": state_path,
            "metrics_sha256": state_path.parent / "metrics.json",
            "provenance_sha256": state_path.parent / "provenance.json",
        }
        for field, artifact in artifacts.items():
            if not artifact.exists() or config_sha256(artifact) != record.get(field):
                raise RuntimeError(f"E10 development artifact mismatch: {artifact}")
    locked_data = {str(row["dataset"]): row for row in lock.get("external_data", [])}
    expected_datasets = {str(unit["dataset"]) for unit in units_for_stage(cfg, "external")}
    if set(locked_data) != expected_datasets:
        raise RuntimeError("Study lock external dataset list does not match E10")
    for dataset, record in locked_data.items():
        root = resolved_data_root(cfg) / dataset
        graph = root / "graph_physical.npz"
        if not graph.exists():
            graph = root / "graph_candidate.npz"
        paths = {
            "series_sha256": root / "series.npz",
            "prior_graph_sha256": graph,
            "controls_manifest_sha256": root / "spi_controls" / "manifest.json",
        }
        for field, current_path in paths.items():
            if not current_path.exists() or config_sha256(current_path) != record.get(field):
                raise RuntimeError(f"E10 locked input mismatch for {dataset}/{field}")
    return lock
