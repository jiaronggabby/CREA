from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rpv_data.graphs import (
    GraphArrays,
    all_to_all_graph,
    degree_sequences,
    endpoint_change_fraction,
    identity_graph,
    load_graph,
)
from rpv_data.utils import sha256_file

SplitName = Literal[
    "forecaster_train",
    "forecaster_validation",
    "gate_train",
    "gate_validation",
    "test",
]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    root: Path
    values: np.ndarray
    raw_values: np.ndarray
    observed_mask: np.ndarray
    timestamps: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    boundaries: dict[str, tuple[int, int]]

    @property
    def time_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def variables(self) -> int:
        return int(self.values.shape[1])

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return array * self.std.reshape((1,) * (array.ndim - 1) + (-1,)) + self.mean.reshape(
            (1,) * (array.ndim - 1) + (-1,)
        )


def _five_boundaries(length: int, split: dict[str, Any]) -> dict[str, tuple[int, int]]:
    fractions = [
        float(split["forecaster_train"]),
        float(split["forecaster_validation"]),
        float(split["gate_train"]),
        float(split["gate_validation"]),
        float(split["test"]),
    ]
    cuts = [0]
    cumulative = 0.0
    for fraction in fractions[:-1]:
        cumulative += fraction
        cuts.append(round(length * cumulative))
    cuts.append(length)
    if any(right <= left for left, right in pairwise(cuts)):
        raise ValueError(f"Degenerate five-way split for time series of length {length}")
    return {
        name: (int(cuts[index]), int(cuts[index + 1]))
        for index, name in enumerate(
            ("forecaster_train", "forecaster_validation", "gate_train", "gate_validation", "test")
        )
    }


def _reconstruct_and_impute(raw: np.ndarray, observed: np.ndarray, train_end: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if raw.ndim != 2 or observed.shape != raw.shape:
        raise ValueError("raw_values and observed_mask must have identical [time, node] shapes")
    frame = pd.DataFrame(np.where(observed, raw, np.nan)).ffill()
    medians = pd.DataFrame(np.where(observed[:train_end], raw[:train_end], np.nan)).median(axis=0)
    if medians.isna().any():
        bad = medians[medians.isna()].index.tolist()
        raise ValueError(f"Nodes have no observed forecaster-training values: {bad[:10]}")
    frame = frame.fillna(medians)
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values remain after causal imputation")
    return values


def load_bundle(data_root: str | Path, dataset: str, split: dict[str, Any]) -> DatasetBundle:
    root = Path(data_root) / dataset
    path = root / "series.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "raw_values" not in archive.files or "observed_mask" not in archive.files:
            raise ValueError(
                f"{path} lacks raw_values/observed_mask; the active protocol refuses to reuse old standardized values"
            )
        raw = np.asarray(archive["raw_values"], dtype=np.float64)
        observed = np.asarray(archive["observed_mask"], dtype=np.uint8)
        timestamps = np.asarray(
            archive["timestamps"] if "timestamps" in archive.files else np.arange(raw.shape[0]),
        )
    boundaries = _five_boundaries(len(raw), split)
    train_end = boundaries["forecaster_train"][1]
    imputed = _reconstruct_and_impute(raw, observed, train_end)
    train_observed = observed[:train_end].astype(bool)
    masked_train = np.where(train_observed, imputed[:train_end], np.nan)
    mean = np.nanmean(masked_train, axis=0)
    std = np.nanstd(masked_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, np.mean(imputed[:train_end], axis=0))
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    values = ((imputed - mean) / std).astype(np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid standardized values for {dataset}")
    return DatasetBundle(
        name=dataset,
        root=root,
        values=values,
        raw_values=imputed.astype(np.float32),
        observed_mask=observed,
        timestamps=timestamps,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        boundaries=boundaries,
    )


def window_starts(
    bundle: DatasetBundle,
    split: SplitName,
    history: int,
    horizon: int,
    max_windows: int | None,
    seed: int,
) -> np.ndarray:
    start, end = bundle.boundaries[split]
    first = max(int(history), int(start))
    last_exclusive = int(end) - int(horizon) + 1
    if last_exclusive <= first:
        raise ValueError(f"Split {split} of {bundle.name} has no complete windows")
    starts = np.arange(first, last_exclusive, dtype=np.int64)
    if max_windows is None or max_windows <= 0 or len(starts) <= max_windows:
        return starts
    if split in {"forecaster_train", "gate_train"}:
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(starts, size=int(max_windows), replace=False))
    positions = np.linspace(0, len(starts) - 1, num=int(max_windows), dtype=np.int64)
    return starts[np.unique(positions)]


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, bundle: DatasetBundle, starts: np.ndarray, history: int, horizon: int):
        self.bundle = bundle
        self.starts = np.asarray(starts, dtype=np.int64)
        self.history = int(history)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin = int(self.starts[index])
        x = self.bundle.values[origin - self.history : origin]
        y = self.bundle.values[origin : origin + self.horizon]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.tensor(index, dtype=torch.long)


def base_graph_path(bundle: DatasetBundle) -> Path:
    physical = bundle.root / "graph_physical.npz"
    candidate = bundle.root / "graph_candidate.npz"
    if physical.exists():
        return physical
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No supplied graph for {bundle.name}")


def graph_to_mask(graph: GraphArrays, device: torch.device | None = None) -> torch.Tensor:
    mask = np.zeros((graph.num_nodes, graph.num_nodes), dtype=bool)
    for src, dst in graph.edge_index.T:
        mask[int(dst), int(src)] = True
    np.fill_diagonal(mask, True)
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def fixed_masks(bundle: DatasetBundle, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    return {
        "local": graph_to_mask(identity_graph(bundle.variables), device=device),
        "all": graph_to_mask(all_to_all_graph(bundle.variables), device=device),
        "prior": graph_to_mask(load_graph(base_graph_path(bundle)), device=device),
    }


def load_controls(
    bundle: DatasetBundle,
    count: int,
    minimum_endpoint_change: float,
    kind: str = "matched",
    device: torch.device | None = None,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    if kind not in {"matched", "random"}:
        raise ValueError("kind must be matched or random")
    directory = bundle.root / "spi_controls"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing control manifest {manifest_path}; run prepare-controls first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count_field = "matched_count" if kind == "matched" else "random_count"
    if (
        manifest.get("schema") != "selective_prior_controls_v1"
        or manifest.get("dataset") != bundle.name
        or manifest.get("base_graph_sha256") != sha256_file(base_graph_path(bundle))
        or int(manifest.get(expected_count_field, -1)) < int(count)
    ):
        raise RuntimeError(f"Control manifest provenance does not match current prior: {manifest_path}")
    records = manifest.get("controls")
    if not isinstance(records, list):
        raise TypeError(f"Control manifest lacks records: {manifest_path}")
    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            key = (str(record.get("kind")), int(record.get("index")))
        except (TypeError, ValueError):
            continue
        records_by_key[key] = record
    base = load_graph(base_graph_path(bundle))
    base_out, base_in = degree_sequences(base)
    controls: list[torch.Tensor] = []
    provenance: list[dict[str, Any]] = []
    for index in range(int(count)):
        path = directory / f"{kind}_{index:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing matched control {path}; run prepare-controls first")
        record = records_by_key.get((kind, index))
        expected_relative = f"spi_controls/{kind}_{index:03d}.npz"
        if (
            record is None
            or record.get("path") != expected_relative
            or record.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"Control file hash does not match manifest: {path}")
        control = load_graph(path)
        change = endpoint_change_fraction(base, control)
        out_degree, in_degree = degree_sequences(control)
        if control.num_nodes != base.num_nodes:
            raise ValueError(f"Control node mismatch: {path}")
        if kind == "matched" and (
            not np.array_equal(out_degree, base_out) or not np.array_equal(in_degree, base_in)
        ):
            raise ValueError(f"Control does not preserve node-wise degree: {path}")
        if change + 1e-12 < float(minimum_endpoint_change):
            raise ValueError(f"Control endpoint change {change:.3f} is below the frozen threshold: {path}")
        controls.append(graph_to_mask(control, device=device))
        provenance.append(
            {"kind": kind, "path": str(path), "sha256": sha256_file(path), "endpoint_change": float(change)}
        )
    return controls, provenance


def load_matched_controls(
    bundle: DatasetBundle,
    count: int,
    minimum_endpoint_change: float,
    device: torch.device | None = None,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    return load_controls(bundle, count, minimum_endpoint_change, kind="matched", device=device)


def node_error_scale(bundle: DatasetBundle) -> np.ndarray:
    start, end = bundle.boundaries["forecaster_train"]
    train = np.asarray(bundle.raw_values[start:end], dtype=np.float64)
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    return (0.01 * mad + 1e-6).astype(np.float32)
