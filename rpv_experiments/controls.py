from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rpv_data.graphs import (
    GraphArrays,
    endpoint_change_fraction,
    from_edges,
    load_graph,
    rewire_directed,
    save_graph,
)
from rpv_data.utils import atomic_json, sha256_file

from .data import DatasetBundle, base_graph_path


def _dataset_token(dataset: str) -> int:
    return int.from_bytes(hashlib.sha256(dataset.encode("utf-8")).digest()[:4], "little")


def _seed(namespace: int, dataset: str, index: int) -> int:
    state = np.random.SeedSequence([int(namespace), _dataset_token(dataset), int(index)]).generate_state(1)
    return int(state[0])


def _random_density_control(base: GraphArrays, seed: int) -> GraphArrays:
    rng = np.random.default_rng(seed)
    src, dst = base.edge_index
    nonself = int(np.sum(src != dst))
    candidates = np.asarray(
        [(u, v) for u in range(base.num_nodes) for v in range(base.num_nodes) if u != v],
        dtype=np.int64,
    )
    if nonself > len(candidates):
        raise ValueError("Graph has more non-self edges than possible directed pairs")
    chosen = candidates[rng.choice(len(candidates), size=nonself, replace=False)]
    return from_edges(
        [(int(u), int(v), 1.0) for u, v in chosen],
        n=base.num_nodes,
        add_self=True,
        directed=True,
    )


def prepare_controls(
    bundle: DatasetBundle,
    *,
    matched_count: int,
    matched_seed_namespace: int,
    random_seed_namespace: int,
    minimum_endpoint_change: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    base_path = base_graph_path(bundle)
    base = load_graph(base_path)
    directory = bundle.root / "spi_controls"
    if directory.exists() and not overwrite:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileExistsError(f"Control directory lacks a manifest; rerun with --overwrite: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema": "selective_prior_controls_v1",
            "dataset": bundle.name,
            "base_graph_sha256": sha256_file(base_path),
            "matched_count": int(matched_count),
            "random_count": int(matched_count),
            "minimum_endpoint_change": float(minimum_endpoint_change),
            "matched_seed_namespace": int(matched_seed_namespace),
            "random_seed_namespace": int(random_seed_namespace),
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            records = manifest.get("controls")
            if not isinstance(records, list):
                records = []
            records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
            for record in records:
                if not isinstance(record, dict):
                    continue
                try:
                    key = (str(record.get("kind")), int(record.get("index")))
                except (TypeError, ValueError):
                    continue
                records_by_key[key] = record
            expected_keys = {
                (kind, index)
                for kind in ("matched", "random")
                for index in range(int(matched_count))
            }
            files_match = set(records_by_key) == expected_keys
            for kind, index in sorted(expected_keys):
                record = records_by_key.get((kind, index), {})
                relative = f"spi_controls/{kind}_{index:03d}.npz"
                path = bundle.root / relative
                files_match = bool(
                    files_match
                    and record.get("path") == relative
                    and path.exists()
                    and record.get("sha256") == sha256_file(path)
                )
                if not files_match:
                    break
            if files_match:
                return manifest
        raise FileExistsError(
            "Control graph provenance is stale or its contract changed; "
            f"rerun prepare-controls --overwrite explicitly: {directory}"
        )
    working = Path(tempfile.mkdtemp(prefix=".spi_controls.", dir=str(bundle.root)))
    records: list[dict[str, Any]] = []
    try:
        for kind in ("matched", "random"):
            count = int(matched_count)
            namespace = int(matched_seed_namespace if kind == "matched" else random_seed_namespace)
            for index in range(count):
                path = working / f"{kind}_{index:03d}.npz"
                requested_seed = _seed(namespace, bundle.name, index)
                if kind == "matched":
                    graph = None
                    actual_seed = requested_seed
                    last_error: Exception | None = None
                    for attempt in range(20):
                        actual_seed = requested_seed + attempt * 104729
                        try:
                            graph = rewire_directed(
                                base,
                                seed=actual_seed,
                                matched_weight_bins=True,
                                min_endpoint_change=float(minimum_endpoint_change),
                            )
                            break
                        except (RuntimeError, ValueError) as error:
                            last_error = error
                    if graph is None:
                        raise RuntimeError(f"Could not generate matched control for {bundle.name}: {last_error}")
                else:
                    graph = None
                    actual_seed = requested_seed
                    for attempt in range(20):
                        actual_seed = requested_seed + attempt * 104729
                        candidate = _random_density_control(base, actual_seed)
                        if endpoint_change_fraction(base, candidate) + 1e-12 >= float(minimum_endpoint_change):
                            graph = candidate
                            break
                    if graph is None:
                        raise RuntimeError(f"Could not generate changed random control for {bundle.name}")
                save_graph(path, graph)
                records.append(
                    {
                        "kind": kind,
                        "index": int(index),
                        "path": f"spi_controls/{path.name}",
                        "sha256": sha256_file(path),
                        "requested_seed": requested_seed,
                        "actual_seed": actual_seed,
                        "endpoint_change": float(endpoint_change_fraction(base, graph)),
                    }
                )
        for kind in ("matched", "random"):
            hashes = [record["sha256"] for record in records if record["kind"] == kind]
            if len(set(hashes)) != len(hashes):
                raise RuntimeError(f"Duplicate {kind} controls generated for {bundle.name}")
        manifest = {
            "schema": "selective_prior_controls_v1",
            "dataset": bundle.name,
            "base_graph": str(base_path),
            "base_graph_sha256": sha256_file(base_path),
            "matched_count": int(matched_count),
            "random_count": int(matched_count),
            "minimum_endpoint_change": float(minimum_endpoint_change),
            "matched_seed_namespace": int(matched_seed_namespace),
            "random_seed_namespace": int(random_seed_namespace),
            "controls": records,
        }
        atomic_json(working / "manifest.json", manifest)
        backup = bundle.root / f".spi_controls.backup.{os.getpid()}"
        if directory.exists():
            os.replace(directory, backup)
        try:
            os.replace(working, directory)
        except Exception:
            if backup.exists() and not directory.exists():
                os.replace(backup, directory)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except Exception:
        if working.exists():
            shutil.rmtree(working)
        raise
