from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from rpv_data.downloads import download_dataset
from rpv_data.preprocessors import (
    preprocess_causalrivers,
    preprocess_coast,
    preprocess_metr_la,
    preprocess_pems04,
    preprocess_pems08,
    preprocess_pems_standard,
)
from rpv_data.synthetic import (
    generate_selective_prior_synthetic,
    generate_selective_prior_synthetic_v3,
    selective_prior_v3_generation_identity,
    validate_selective_prior_v3_metadata,
)
from rpv_data.utils import PROCESSED_ROOT, RAW_ROOT, atomic_json

from .analysis import analyze_runs
from .config import (
    load_config,
    protocol_version,
    resolved_data_root,
    resolved_output_root,
    units_for_stage,
    verify_external_lock,
)
from .controls import prepare_controls
from .data import fixed_masks, load_bundle, load_controls, load_matched_controls
from .decision import (
    verify_mechanism_authorization,
    write_decision,
    write_mechanism_decision,
)
from .final_v1_ablations import write_final_v1_ablations
from .final_v1_efficiency import write_efficiency
from .final_v1_reliability_shift import write_reliability_shift
from .final_v1_standard_reporting import write_standard_reporting
from .final_v1_stats import write_final_v1_stats
from .runner import create_study_lock, matrix_plan, run_matrix, summarize_runs
from .training import resolve_device


def _csv(value: str) -> list[str] | None:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    return values or None


def _unique_datasets(units: Iterable[dict]) -> list[str]:
    return list(dict.fromkeys(str(unit["dataset"]) for unit in units))


def _expected_v3_generation_identity(cfg: dict) -> dict[str, object]:
    synthetic = cfg.get("synthetic", {})
    return selective_prior_v3_generation_identity(
        seed=int(cfg["randomness"]["synthetic_namespace"]),
        time_steps=int(synthetic.get("time_steps", 12000)),
        variables=int(synthetic.get("variables", 50)),
        density=float(synthetic.get("density", 0.08)),
        history=int(synthetic.get("history_contract", 48)),
        max_horizon=int(synthetic.get("maximum_horizon_contract", 12)),
    )


def _selected_units(cfg: dict, stage: str, experiments: list[str] | None, datasets: list[str] | None) -> list[dict]:
    experiment_set = set(experiments or [])
    dataset_set = set(datasets or [])
    return [
        unit
        for unit in units_for_stage(cfg, stage)
        if (not experiment_set or unit["experiment_id"] in experiment_set)
        and (not dataset_set or unit["dataset"] in dataset_set)
    ]


def _prepare_data(cfg: dict, stage: str, experiments: list[str] | None, datasets: list[str] | None, download: bool) -> list[str]:
    selected = _selected_units(cfg, stage, experiments, datasets)
    names = _unique_datasets(selected)
    unit_by_name = {str(unit["dataset"]): unit for unit in selected}
    done: list[str] = []
    reusable: set[str] = set()
    required_files = {
        "series.npz",
        "nodes.csv",
        "metadata.json",
        "graph_physical.npz",
        "graph_identity.npz",
        "graph_all.npz",
    }
    # Protocol v4 deliberately reuses the v3 synthetic DGP, but changes the
    # forecaster/gating intervention.  Its data identity must therefore be
    # checked with exactly the same fail-closed generator contract.
    synthetic_identity_protocol = cfg.get("schema") in {
        "selective_prior_injection_v3",
        "selective_prior_injection_v4",
    }
    expected_v3_identity = (
        _expected_v3_generation_identity(cfg) if synthetic_identity_protocol else None
    )
    for name in names:
        root = resolved_data_root(cfg) / name
        present = {path.name for path in root.iterdir()} if root.exists() else set()
        if present and not required_files <= present:
            missing = sorted(required_files - present)
            raise RuntimeError(
                f"Partial processed dataset exists for {name}; preserve it and resolve missing files explicitly: {missing}"
            )
        if not required_files <= present:
            continue
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if ("synthetic_corruption_" in name) and (
            metadata.get("corruption_curve")
            != "nested_endpoint_replacement_without_original_endpoint_reentry"
            or "actual_endpoint_change_fraction" not in metadata
        ):
            continue
        if name.startswith("spi_v3_synthetic_"):
            if expected_v3_identity is None:
                raise RuntimeError(
                    f"Synthetic dataset {name} cannot be reused by a protocol without a generator identity"
                )
            validate_selective_prior_v3_metadata(
                metadata,
                name,
                expected_generation_spec=expected_v3_identity["generation_spec"],
                expected_generator_fingerprint=str(expected_v3_identity["generator_fingerprint"]),
            )
        if name == "metr_la" and (
            metadata.get("node_alignment", {}).get("policy")
            != "series columns reordered to official adjacency sensor order"
            or metadata.get("node_alignment", {}).get("sensor_id_to_ind_verified") is not True
        ):
            raise RuntimeError(
                "Existing METR-LA processed data predates mandatory graph/data node alignment; "
                f"archive it and rerun prepare-data explicitly: {root}"
            )
        bundle = load_bundle(resolved_data_root(cfg), name, cfg["split"])
        expected_resolution = unit_by_name[name].get("expected_resolution")
        if expected_resolution and str(metadata.get("resolution")) != str(expected_resolution):
            raise ValueError(
                f"Reusable {name} resolution {metadata.get('resolution')!r} != {expected_resolution!r}"
            )
        if bundle.name != name:
            raise ValueError(f"Reusable dataset identity mismatch: {bundle.name} != {name}")
        reusable.add(name)
        done.append(name)
    names = [name for name in names if name not in reusable]
    synthetic_v3 = [name for name in names if name.startswith("spi_v3_synthetic_")]
    if synthetic_v3:
        generate_selective_prior_synthetic_v3(
            seed=int(cfg["randomness"]["synthetic_namespace"]),
            T=int(cfg.get("synthetic", {}).get("time_steps", 12000)),
            n=int(cfg.get("synthetic", {}).get("variables", 50)),
            density=float(cfg.get("synthetic", {}).get("density", 0.08)),
            processed_root=resolved_data_root(cfg),
            only=set(synthetic_v3),
            history=int(cfg.get("synthetic", {}).get("history_contract", 48)),
            max_horizon=int(
                cfg.get("synthetic", {}).get("maximum_horizon_contract", 12)
            ),
        )
        done.extend(synthetic_v3)
    synthetic = [
        name
        for name in names
        if name.startswith("spi_synthetic_") and name not in synthetic_v3
    ]
    if synthetic:
        generate_selective_prior_synthetic(
            seed=int(cfg["randomness"]["synthetic_namespace"]),
            T=int(cfg.get("synthetic", {}).get("time_steps", 12000)),
            n=int(cfg.get("synthetic", {}).get("variables", 50)),
            density=float(cfg.get("synthetic", {}).get("density", 0.08)),
            processed_root=resolved_data_root(cfg),
            only=set(synthetic),
        )
        done.extend(synthetic)
    if (
        any(
            not name.startswith("spi_synthetic_")
            and not name.startswith("spi_v3_synthetic_")
            for name in names
        )
        and resolved_data_root(cfg).resolve() != PROCESSED_ROOT.resolve()
    ):
        raise RuntimeError(
            "Non-synthetic preprocessors require config data_root to match RPV_DATA_ROOT/processed"
        )
    if "pems08" in names:
        if download:
            download_dataset("pems08")
        preprocess_pems08(feature_index=0, train_ratio=0.60, val_ratio=0.10)
        done.append("pems08")
    for standard_name in ("pems03", "pems07"):
        if standard_name in names:
            if download:
                download_dataset(standard_name)
            preprocess_pems_standard(name=standard_name, feature_index=0, train_ratio=0.60, val_ratio=0.10)
            done.append(standard_name)
    if "pems04" in names:
        if download:
            download_dataset("pems04")
        preprocess_pems04(feature_index=0, train_ratio=0.60, val_ratio=0.10)
        done.append("pems04")
    if "metr_la" in names:
        preprocess_metr_la(train_ratio=0.60, val_ratio=0.10)
        done.append("metr_la")
    east = [name for name in names if name.startswith("causalrivers_east15m_s50_")]
    if east:
        if download and not (RAW_ROOT / "causalrivers" / "release").exists():
            download_dataset("causalrivers")
        preprocess_causalrivers(
            region="east",
            resolution="15min",
            subgraph_size=50,
            n_subgraphs=4,
            max_missing_fraction=0.20,
            train_ratio=0.60,
            val_ratio=0.10,
            output_prefix="causalrivers_east15m",
        )
        done.extend(east)
    bavaria = [name for name in names if name.startswith("causalrivers_bavaria15m_s50_")]
    if bavaria:
        if download and not (RAW_ROOT / "causalrivers" / "release").exists():
            download_dataset("causalrivers")
        preprocess_causalrivers(
            region="bavaria",
            resolution="15min",
            subgraph_size=50,
            n_subgraphs=4,
            max_missing_fraction=0.20,
            train_ratio=0.60,
            val_ratio=0.10,
            output_prefix="causalrivers_bavaria15m",
        )
        done.extend(bavaria)
    for name in names:
        if not name.startswith("coast_"):
            continue
        preprocess_coast(
            train_ratio=0.60,
            val_ratio=0.10,
            raw_name=name,
            output_name=name,
            relation_status="externally_supplied_frozen_binary_connectivity",
        )
        done.append(name)
    missing_handlers = sorted(set(names) - set(done))
    if missing_handlers:
        raise KeyError(f"No active-protocol data adapter for {missing_handlers}")
    return done


def _prepare_controls(cfg: dict, stage: str, experiments: list[str] | None, datasets: list[str] | None, overwrite: bool) -> list[str]:
    selected = _selected_units(cfg, stage, experiments, datasets)
    data_root = resolved_data_root(cfg)
    done = []
    for name in _unique_datasets(selected):
        bundle = load_bundle(data_root, name, cfg["split"])
        prepare_controls(
            bundle,
            matched_count=int(cfg["method"]["matched_controls"]),
            matched_seed_namespace=int(cfg["randomness"]["matched_control_namespace"]),
            random_seed_namespace=int(cfg["randomness"]["random_control_namespace"]),
            minimum_endpoint_change=float(cfg["method"]["minimum_endpoint_change"]),
            overwrite=overwrite,
        )
        done.append(name)
    return done


def _preflight(
    cfg: dict,
    config_path: Path,
    stage: str,
    experiments: list[str] | None,
    datasets: list[str] | None,
) -> dict:
    if stage == "formal" and cfg.get("schema") in {
        "selective_prior_injection_v3",
        "selective_prior_injection_v4",
    } and str(cfg.get("mode")) != "formal":
        raise RuntimeError(
            "The mechanism config cannot launch a formal matrix; freeze a separate "
            "formal configuration after the Seed42 decision"
        )
    if stage == "formal" and cfg.get("schema") == "selective_prior_injection_v2":
        verify_mechanism_authorization(cfg, config_path)
    if stage == "external":
        verify_external_lock(cfg, config_path)
    selected = _selected_units(cfg, stage, experiments, datasets)
    device = resolve_device(str(cfg["device"]))
    synthetic_identity_protocol = cfg.get("schema") in {
        "selective_prior_injection_v3",
        "selective_prior_injection_v4",
    }
    expected_v3_identity = (
        _expected_v3_generation_identity(cfg) if synthetic_identity_protocol else None
    )
    rows = []
    for unit in selected:
        bundle = load_bundle(resolved_data_root(cfg), str(unit["dataset"]), cfg["split"])
        if synthetic_identity_protocol and bundle.name.startswith("spi_v3_synthetic_"):
            if expected_v3_identity is None:
                raise RuntimeError("Synthetic preflight lacks a generation identity")
            metadata = json.loads(
                (bundle.root / "metadata.json").read_text(encoding="utf-8")
            )
            validate_selective_prior_v3_metadata(
                metadata,
                bundle.name,
                expected_generation_spec=expected_v3_identity["generation_spec"],
                expected_generator_fingerprint=str(expected_v3_identity["generator_fingerprint"]),
            )
        masks = fixed_masks(bundle, device=device)
        matched, matched_provenance = load_matched_controls(
            bundle,
            int(cfg["method"]["matched_controls"]),
            float(cfg["method"]["minimum_endpoint_change"]),
            device=device,
        )
        if bool(unit.get("random_control_ablation", False)):
            random, random_provenance = load_controls(
                bundle,
                int(cfg["method"]["matched_controls"]),
                float(cfg["method"]["minimum_endpoint_change"]),
                kind="random",
                device=device,
            )
        else:
            random, random_provenance = [], []
        expected_resolution = unit.get("expected_resolution")
        if expected_resolution:
            metadata_path = bundle.root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("resolution")) != str(expected_resolution):
                raise ValueError(
                    f"{bundle.name} resolution {metadata.get('resolution')!r} != {expected_resolution!r}"
                )
        rows.append(
            {
                "experiment_id": unit["experiment_id"],
                "dataset": bundle.name,
                "time_steps": bundle.time_steps,
                "variables": bundle.variables,
                "masks": {key: list(value.shape) for key, value in masks.items()},
                "matched_controls": len(matched),
                "random_controls": len(random),
                "minimum_matched_endpoint_change": min(row["endpoint_change"] for row in matched_provenance),
                "minimum_random_endpoint_change": (
                    min(row["endpoint_change"] for row in random_provenance)
                    if random_provenance
                    else None
                ),
            }
        )
    report = {
        "schema": f"selective_prior_injection_preflight_{protocol_version(cfg)}",
        "stage": stage,
        "device": str(device),
        "data_root": str(resolved_data_root(cfg)),
        "output_root": str(resolved_output_root(cfg)),
        "units": rows,
    }
    output = resolved_output_root(cfg) / "preflight" / f"{stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rpv-experiments")
    parser.add_argument("--config", default="configs/experiment_development.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("plan", "preflight", "prepare-data", "prepare-controls", "run"):
        command = sub.add_parser(name)
        command.add_argument("--stage", choices=["seed42", "formal", "external"], default="seed42")
        command.add_argument("--experiments", default="")
        command.add_argument("--datasets", default="")
        if name == "prepare-data":
            command.add_argument("--download", action="store_true")
        if name == "prepare-controls":
            command.add_argument("--overwrite", action="store_true")
        if name == "run":
            command.add_argument("--limit", type=int, default=None)
            command.add_argument(
                "--seed",
                type=int,
                default=None,
                help="Run only this predeclared seed; does not alter the protocol seed list",
            )
            command.add_argument("--resume", action="store_true")
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)

    sub.add_parser("summarize")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--bootstrap", type=int, default=None)
    analyze.add_argument("--seed", type=int, default=20260814)
    sub.add_parser("decision")
    final_stats = sub.add_parser(
        "final-stats",
        help="Summarize locked FINAL_V1 test predictions without retuning",
    )
    final_stats.add_argument("--seeds", default="42,3407,101,2026,777")
    final_stats.add_argument("--bootstrap", type=int, default=5000)
    final_stats.add_argument("--block-length", type=int, default=24)
    final_ablations = sub.add_parser(
        "final-ablations",
        help="Compute frozen-prediction FINAL_V1 ablations without retuning",
    )
    final_ablations.add_argument("--seeds", default="42,3407,101,2026,777")
    final_shift = sub.add_parser(
        "final-shift",
        help="Run frozen-prior reliability-shift inference diagnostics",
    )
    final_shift.add_argument("--datasets", default="pems04,metr_la")
    final_shift.add_argument("--seeds", default="42,3407,101,2026,777")
    final_standard = sub.add_parser(
        "final-standard-reporting",
        help="Write read-only standard 12-step traffic benchmark reporting",
    )
    final_standard.add_argument("--seeds", default="42,3407,101,2026,777")
    final_efficiency = sub.add_parser(
        "final-efficiency",
        help="Measure post-hoc causal aggregation cost on representative runs",
    )
    final_efficiency.add_argument("--datasets", default="pems07,metr_la")
    final_efficiency.add_argument("--seed", type=int, default=42)
    sub.add_parser(
        "mechanism-decision",
        help="Evaluate the frozen E1-E4 Seed42 gate that authorizes formal expansion",
    )
    sub.add_parser("lock", help="Fail closed unless every E1-E9 formal run is complete")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg, config_path = load_config(args.config)
    if args.command == "plan":
        plan = matrix_plan(cfg, config_path, args.stage, _csv(args.experiments), _csv(args.datasets))
        print(
            json.dumps(
                {
                    "unit_seed_runs": len(plan),
                    "heavy_forecasters": sum(len(row["heavy_forecasters"]) for row in plan),
                    "plan": plan,
                },
                indent=2,
            )
        )
    elif args.command == "prepare-data":
        done = _prepare_data(cfg, args.stage, _csv(args.experiments), _csv(args.datasets), args.download)
        print(json.dumps({"prepared": done}, indent=2))
    elif args.command == "prepare-controls":
        done = _prepare_controls(cfg, args.stage, _csv(args.experiments), _csv(args.datasets), args.overwrite)
        print(json.dumps({"controls_prepared": done}, indent=2))
    elif args.command == "preflight":
        print(
            json.dumps(
                _preflight(cfg, config_path, args.stage, _csv(args.experiments), _csv(args.datasets)),
                indent=2,
            )
        )
    elif args.command == "run":
        results = run_matrix(
            cfg,
            config_path,
            args.stage,
            experiment_ids=_csv(args.experiments),
            datasets=_csv(args.datasets),
            limit=args.limit,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            resume=args.resume,
            overwrite=args.overwrite,
            seed=args.seed,
        )
        print(json.dumps({"completed_unit_seed_runs": len(results)}, indent=2))
    elif args.command == "summarize":
        print(summarize_runs(cfg))
    elif args.command == "analyze":
        print(
            analyze_runs(
                cfg,
                config_path,
                replicates=args.bootstrap,
                bootstrap_seed=args.seed,
            )
        )
    elif args.command == "decision":
        print(write_decision(cfg))
    elif args.command == "final-stats":
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        print(
            write_final_v1_stats(
                resolved_output_root(cfg),
                seeds=seeds,
                bootstrap_replicates=args.bootstrap,
                block_length=args.block_length,
            )
        )
    elif args.command == "final-ablations":
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        print(write_final_v1_ablations(resolved_output_root(cfg), seeds=seeds))
    elif args.command == "final-shift":
        datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        print(write_reliability_shift(config_path, datasets=datasets, seeds=seeds))
    elif args.command == "final-standard-reporting":
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        print(write_standard_reporting(resolved_output_root(cfg), seeds=seeds))
    elif args.command == "final-efficiency":
        datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
        print(write_efficiency(resolved_output_root(cfg), datasets=datasets, seed=args.seed))
    elif args.command == "mechanism-decision":
        path = write_mechanism_decision(cfg, config_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        print(path)
        if report.get("proceed_formal") is not True:
            raise SystemExit(
                "Seed42 mechanism validation failed; formal expansion remains locked"
            )
    elif args.command == "lock":
        print(create_study_lock(cfg, config_path))
    else:
        raise AssertionError(args.command)
