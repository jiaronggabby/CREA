from __future__ import annotations

import json
import pickle
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .graphs import (
    all_to_all_graph,
    correlation_graph,
    from_edges,
    identity_graph,
    save_graph,
)
from .utils import (
    PROCESSED_ROOT,
    RAW_ROOT,
    atomic_json,
    chronological_split,
    clean_numeric_frame,
    dataframe_timestamp_ns,
    save_series,
)


def _write_common_graphs(out: Path, n: int) -> None:
    save_graph(out / "graph_identity.npz", identity_graph(n))
    save_graph(out / "graph_all.npz", all_to_all_graph(n))


def preprocess_pems08(feature_index: int = 0, train_ratio: float = 0.80, val_ratio: float = 0.10) -> None:
    raw = RAW_ROOT / "pems08"
    npz_path, edge_path = raw / "pems08.npz", raw / "distance.csv"
    if not npz_path.exists() or not edge_path.exists():
        raise FileNotFoundError("Run download pems08 first")
    z = np.load(npz_path, allow_pickle=False)
    key = "data" if "data" in z.files else z.files[0]
    data = np.asarray(z[key])
    if data.ndim == 3:
        if feature_index < 0 or feature_index >= data.shape[2]:
            raise ValueError(f"feature_index {feature_index} outside shape {data.shape}")
        values = data[:, :, feature_index]
    elif data.ndim == 2:
        values = data
    else:
        raise ValueError(f"Unexpected PEMS08 array shape {data.shape}")
    values = np.asarray(values, dtype=np.float64)
    T, n = values.shape
    train_end, val_end = chronological_split(T, train_ratio, val_ratio)
    df, observed = clean_numeric_frame(pd.DataFrame(values), train_end=train_end)
    values = df.to_numpy()
    out = PROCESSED_ROOT / "pems08"
    save_series(out, values, observed, np.arange(T, dtype=np.int64), train_end, val_end)
    nodes = pd.DataFrame({"node_index": np.arange(n), "node_id": np.arange(n).astype(str)})
    nodes.to_csv(out / "nodes.csv", index=False)
    edges_df = pd.read_csv(edge_path)
    lower = {c.lower(): c for c in edges_df.columns}
    src_col = lower.get("from") or lower.get("src")
    dst_col = lower.get("to") or lower.get("dst")
    weight_col = lower.get("cost") or lower.get("distance") or lower.get("weight")
    if not src_col or not dst_col:
        raise ValueError(f"Cannot identify PEMS08 edge columns: {edges_df.columns.tolist()}")
    if weight_col is None:
        edges_df["_weight"] = 1.0
        weight_col = "_weight"
    edges = [(int(r[src_col]), int(r[dst_col]), float(r[weight_col])) for _, r in edges_df.iterrows()]
    physical = from_edges(edges, n=n, add_self=True, directed=True)
    save_graph(out / "graph_physical.npz", physical)
    _write_common_graphs(out, n)
    atomic_json(out / "metadata.json", {
        "dataset": "pems08",
        "source_array_key": key,
        "raw_shape": list(data.shape),
        "selected_feature_index": feature_index,
        "time_steps": T,
        "num_variables": n,
        "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
        "relation": "supplied directed road-distance candidate",
        "primary_horizons_samples": [3, 6, 12],
        "history_samples": 12,
    })
    print(f"[prepared] pems08: {values.shape}, edges={physical.edge_index.shape[1]}")


def preprocess_pems04(feature_index: int = 0, train_ratio: float = 0.60, val_ratio: float = 0.10) -> None:
    """Prepare the public PEMS04 release under the new five-way experiment contract."""
    raw = RAW_ROOT / "pems04"
    npz_path, edge_path = raw / "pems04.npz", raw / "distance.csv"
    if not npz_path.exists() or not edge_path.exists():
        raise FileNotFoundError("Run download pems04 first")
    archive = np.load(npz_path, allow_pickle=False)
    key = "data" if "data" in archive.files else archive.files[0]
    data = np.asarray(archive[key])
    if data.ndim == 3:
        if not 0 <= feature_index < data.shape[2]:
            raise ValueError(f"feature_index {feature_index} outside PEMS04 shape {data.shape}")
        values = data[:, :, feature_index]
    elif data.ndim == 2:
        values = data
    else:
        raise ValueError(f"Unexpected PEMS04 array shape {data.shape}")
    values = np.asarray(values, dtype=np.float64)
    T, n = values.shape
    train_end, val_end = chronological_split(T, train_ratio, val_ratio)
    clean, observed = clean_numeric_frame(pd.DataFrame(values), train_end=train_end)
    out = PROCESSED_ROOT / "pems04"
    save_series(out, clean.to_numpy(), observed, np.arange(T, dtype=np.int64), train_end, val_end)
    pd.DataFrame({"node_index": np.arange(n), "node_id": np.arange(n).astype(str)}).to_csv(
        out / "nodes.csv", index=False
    )
    edge_frame = pd.read_csv(edge_path)
    lower = {column.lower(): column for column in edge_frame.columns}
    src_col = lower.get("from") or lower.get("src")
    dst_col = lower.get("to") or lower.get("dst")
    weight_col = lower.get("cost") or lower.get("distance") or lower.get("weight")
    if not src_col or not dst_col:
        raise ValueError(f"Cannot identify PEMS04 edge columns: {edge_frame.columns.tolist()}")
    if weight_col is None:
        edge_frame["_weight"] = 1.0
        weight_col = "_weight"
    edges = [
        (int(row[src_col]), int(row[dst_col]), float(row[weight_col]))
        for _, row in edge_frame.iterrows()
    ]
    graph = from_edges(edges, n=n, add_self=True, directed=True)
    save_graph(out / "graph_physical.npz", graph)
    _write_common_graphs(out, n)
    atomic_json(
        out / "metadata.json",
        {
            "dataset": "pems04",
            "source_array_key": key,
            "raw_shape": list(data.shape),
            "selected_feature_index": int(feature_index),
            "time_steps": int(T),
            "num_variables": int(n),
            "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
            "relation": "supplied directed road-distance connectivity",
        },
    )


def preprocess_pems_standard(
    name: str,
    feature_index: int = 0,
    train_ratio: float = 0.60,
    val_ratio: float = 0.10,
) -> None:
    """Prepare the public STSGCN/Zenodo PEMS03 or PEMS07 release.

    The release stores a graph edge list whose node identifiers are not
    necessarily positional (PEMS03 uses detector IDs).  When a detector-ID
    text file is present, the time-series columns are aligned to that file
    before graph edges are converted to positional indices.  This keeps the
    supplied relational prior and the series columns in the same order.
    """
    name = str(name).lower()
    if name not in {"pems03", "pems07"}:
        raise ValueError(f"Unsupported standard PEMS dataset: {name}")
    raw = RAW_ROOT / name
    stem = name.upper()
    npz_path = raw / f"{stem}.npz"
    edge_path = raw / f"{stem}.csv"
    id_path = raw / f"{stem}.txt"
    if not npz_path.exists() or not edge_path.exists():
        raise FileNotFoundError(f"Expected {npz_path} and {edge_path}; run download {name} first")
    archive = np.load(npz_path, allow_pickle=False)
    key = "data" if "data" in archive.files else archive.files[0]
    data = np.asarray(archive[key])
    if data.ndim == 3:
        if not 0 <= int(feature_index) < data.shape[2]:
            raise ValueError(f"feature_index {feature_index} outside {name} shape {data.shape}")
        values = data[:, :, int(feature_index)]
    elif data.ndim == 2:
        values = data
    else:
        raise ValueError(f"Unexpected {name} array shape {data.shape}")
    values = np.asarray(values, dtype=np.float64)
    T, n = values.shape
    def canonical_node_id(value: object) -> str:
        text = str(value).strip()
        try:
            number = float(text)
        except ValueError:
            return text
        return str(int(number)) if number.is_integer() else text

    if id_path.exists():
        node_ids = [
            canonical_node_id(line)
            for line in id_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(node_ids) != n or len(set(node_ids)) != n:
            raise ValueError(f"{id_path} must contain exactly {n} unique detector IDs")
    else:
        node_ids = [str(index) for index in range(n)]
    node_to_index = {value: index for index, value in enumerate(node_ids)}
    train_end, val_end = chronological_split(T, train_ratio, val_ratio)
    clean, observed = clean_numeric_frame(pd.DataFrame(values), train_end=train_end)
    out = PROCESSED_ROOT / name
    save_series(out, clean.to_numpy(dtype=np.float64), observed, np.arange(T, dtype=np.int64), train_end, val_end)
    pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids}).to_csv(out / "nodes.csv", index=False)
    edge_frame = pd.read_csv(edge_path)
    lower = {column.lower(): column for column in edge_frame.columns}
    src_col = lower.get("from") or lower.get("src")
    dst_col = lower.get("to") or lower.get("dst")
    weight_col = lower.get("cost") or lower.get("distance") or lower.get("weight")
    if not src_col or not dst_col:
        raise ValueError(f"Cannot identify {name} edge columns: {edge_frame.columns.tolist()}")
    if weight_col is None:
        edge_frame["_weight"] = 1.0
        weight_col = "_weight"
    edges = []
    for _, row in edge_frame.iterrows():
        source = canonical_node_id(row[src_col])
        target = canonical_node_id(row[dst_col])
        if source not in node_to_index or target not in node_to_index:
            raise ValueError(f"{name} graph references unknown detector: {source}->{target}")
        edges.append((node_to_index[source], node_to_index[target], float(row[weight_col])))
    physical = from_edges(edges, n=n, add_self=True, directed=True)
    save_graph(out / "graph_physical.npz", physical)
    _write_common_graphs(out, n)
    atomic_json(out / "metadata.json", {
        "dataset": name,
        "source_array_key": key,
        "raw_shape": list(data.shape),
        "selected_feature_index": int(feature_index),
        "time_steps": int(T),
        "num_variables": int(n),
        "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
        "relation": "supplied directed road-distance connectivity",
        "node_alignment": {
            "series_order": "detector-id text file" if id_path.exists() else "positional columns",
            "graph_edges_reindexed_to_series_order": True,
            "node_id_file": id_path.name if id_path.exists() else None,
        },
        "source_release": "Zenodo 10.5281/zenodo.7816008",
        "history_samples": 12,
        "primary_horizons_samples": [3, 6, 12],
    })
    print(f"[prepared] {name}: {values.shape}, edges={physical.edge_index.shape[1]}")


def _sensor_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def preprocess_metr_la(train_ratio: float = 0.60, val_ratio: float = 0.10) -> None:
    """Prepare METR-LA from the official DCRNN HDF/adjacency artifacts.

    Expected files are data/raw/metr_la/metr-la.h5 and adj_mx.pkl. CSV input
    must carry sensor IDs as column names. An NPZ fallback must contain both
    ``data`` and ``sensor_ids`` so graph/data alignment can be verified.
    """
    raw = RAW_ROOT / "metr_la"
    hdf = raw / "metr-la.h5"
    npz = raw / "metr-la.npz"
    csv_path = raw / "metr-la.csv"
    graph_path = raw / "adj_mx.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing official DCRNN adjacency: {graph_path}")
    if hdf.exists():
        try:
            frame = pd.read_hdf(hdf)
        except ImportError as error:
            raise RuntimeError("Reading metr-la.h5 requires the optional 'tables' package") from error
        values = frame.to_numpy(dtype=np.float64)
        timestamps = frame.index.astype("int64").to_numpy()
        series_ids = [_sensor_id(value) for value in frame.columns]
        source = hdf
    elif npz.exists():
        with np.load(npz, allow_pickle=False) as archive:
            if "data" not in archive.files or "sensor_ids" not in archive.files:
                raise ValueError(
                    "METR-LA NPZ must contain data and sensor_ids; anonymous numeric columns cannot be aligned to adj_mx.pkl"
                )
            values = np.asarray(archive["data"], dtype=np.float64)
            if values.ndim == 3:
                values = values[..., 0]
            series_ids = [_sensor_id(value) for value in np.asarray(archive["sensor_ids"]).tolist()]
            timestamps = (
                np.asarray(archive["timestamps"], dtype=np.int64)
                if "timestamps" in archive.files
                else np.arange(values.shape[0], dtype=np.int64)
            )
        source = npz
    elif csv_path.exists():
        frame = pd.read_csv(csv_path)
        if frame.columns[0].lower() in {"date", "datetime", "timestamp", "time"}:
            timestamps = pd.to_datetime(frame.iloc[:, 0], utc=True).astype("int64").to_numpy()
            frame = frame.iloc[:, 1:]
        else:
            timestamps = np.arange(len(frame), dtype=np.int64)
        values = frame.to_numpy(dtype=np.float64)
        series_ids = [_sensor_id(value) for value in frame.columns]
        source = csv_path
    else:
        raise FileNotFoundError(f"Place metr-la.h5, metr-la.npz, or metr-la.csv in {raw}")
    if values.ndim != 2:
        raise ValueError(f"METR-LA must be [time,node], got {values.shape}")
    T, n = values.shape
    if len(timestamps) != T:
        raise ValueError(f"METR-LA timestamp length {len(timestamps)} != time dimension {T}")
    if len(series_ids) != n or len(set(series_ids)) != n:
        raise ValueError("METR-LA time-series sensor IDs must be unique and match the data width")
    with graph_path.open("rb") as handle:
        graph_object = pickle.load(handle, encoding="latin1")
    if isinstance(graph_object, tuple) and len(graph_object) == 3:
        sensor_ids, sensor_id_to_ind, adjacency = graph_object
        graph_ids = [_sensor_id(value) for value in sensor_ids]
        adjacency = np.asarray(adjacency, dtype=np.float64)
        if len(graph_ids) != n or len(set(graph_ids)) != n:
            raise ValueError("METR-LA adjacency sensor IDs must be unique and match the data width")
        if set(series_ids) != set(graph_ids):
            missing_in_series = sorted(set(graph_ids) - set(series_ids))
            missing_in_graph = sorted(set(series_ids) - set(graph_ids))
            raise ValueError(
                "METR-LA graph/data sensor-ID mismatch: "
                f"missing_in_series={missing_in_series[:5]}, missing_in_graph={missing_in_graph[:5]}"
            )
        if not isinstance(sensor_id_to_ind, dict):
            raise ValueError("METR-LA adj_mx.pkl must include the official sensor_id_to_ind mapping")
        normalized_mapping = {
            _sensor_id(sensor_id): int(index)
            for sensor_id, index in sensor_id_to_ind.items()
        }
        if normalized_mapping != {sensor_id: index for index, sensor_id in enumerate(graph_ids)}:
            raise ValueError("METR-LA sensor_id_to_ind is inconsistent with adjacency row order")
    else:
        raise ValueError(
            "Unsupported adj_mx.pkl layout; formal METR-LA requires DCRNN (sensor_ids,map,adjacency)"
        )
    if adjacency.shape != (n, n):
        raise ValueError(f"METR-LA graph/data mismatch: {adjacency.shape} != {(n, n)}")
    original_series_ids = list(series_ids)
    series_index = {sensor_id: index for index, sensor_id in enumerate(series_ids)}
    order = np.asarray([series_index[sensor_id] for sensor_id in graph_ids], dtype=np.int64)
    values = values[:, order]
    node_ids = graph_ids
    train_end, val_end = chronological_split(T, train_ratio, val_ratio)
    clean, observed = clean_numeric_frame(
        pd.DataFrame(values, columns=node_ids), train_end=train_end
    )
    edges = [
        (src, dst, float(adjacency[src, dst]))
        for src in range(n)
        for dst in range(n)
        if src != dst and np.isfinite(adjacency[src, dst]) and adjacency[src, dst] > 0
    ]
    out = PROCESSED_ROOT / "metr_la"
    save_series(out, clean.to_numpy(), observed, timestamps, train_end, val_end)
    pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids}).to_csv(out / "nodes.csv", index=False)
    save_graph(out / "graph_physical.npz", from_edges(edges, n=n, add_self=True, directed=True))
    _write_common_graphs(out, n)
    atomic_json(
        out / "metadata.json",
        {
            "dataset": "metr_la",
            "time_series_source": str(source),
            "graph_source": str(graph_path),
            "time_steps": int(T),
            "num_variables": int(n),
            "node_alignment": {
                "policy": "series columns reordered to official adjacency sensor order",
                "series_order_changed": original_series_ids != graph_ids,
                "sensor_id_to_ind_verified": True,
            },
            "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
            "relation": "supplied DCRNN road-network adjacency",
        },
    )


def _read_ts_csv(path: Path, train_ratio: float = 0.80) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    timestamp_col = None
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp", "time"}:
            timestamp_col = c
            break
    if timestamp_col is None:
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() > 0.95:
            timestamp_col = first
    if timestamp_col is not None:
        # Normalize the time axis before selecting variables. Public releases
        # occasionally contain an exact duplicate row (Weather has one at
        # 2020-05-12 06:00). Keep only exact duplicates; reject disagreements
        # instead of silently averaging conflicting observations.
        parsed = pd.to_datetime(df[timestamp_col], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Unparseable timestamps ({int(parsed.isna().sum())}) in {path}")
        order = np.argsort(parsed.to_numpy(dtype="datetime64[ns]"), kind="stable")
        if not np.array_equal(order, np.arange(len(df))):
            df = df.iloc[order].reset_index(drop=True)
            parsed = parsed.iloc[order].reset_index(drop=True)
        duplicate_mask = parsed.duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_rows = df.loc[duplicate_mask].drop(columns=[timestamp_col])
            duplicate_times = parsed.loc[duplicate_mask]
            conflicting = []
            for stamp, group in duplicate_rows.groupby(duplicate_times, sort=False):
                if (group.nunique(dropna=False) > 1).any():
                    conflicting.append(str(stamp))
            if conflicting:
                raise ValueError(
                    f"Conflicting duplicate timestamps in {path}: {conflicting[:5]}"
                )
            keep = ~parsed.duplicated(keep="first")
            df = df.loc[keep].reset_index(drop=True)
            parsed = parsed.loc[keep].reset_index(drop=True)
        df = df.drop(columns=[timestamp_col])
        timestamps = dataframe_timestamp_ns(parsed)
    else:
        timestamps = np.arange(len(df), dtype=np.int64)
    train_end = round(len(df) * train_ratio)
    if train_end < 2 or train_end >= len(df):
        raise ValueError(f"Invalid train ratio {train_ratio} for {path}")
    numeric_cols = []
    converted = {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        # Column eligibility is decided from the training interval only.
        if s.iloc[:train_end].notna().mean() >= 0.80:
            numeric_cols.append(str(c))
            converted[str(c)] = s
    if len(numeric_cols) < 2:
        raise ValueError(f"Found fewer than two numeric variables in {path}")
    numeric = pd.DataFrame(converted)
    observed = numeric.notna().to_numpy(dtype=np.uint8)
    return numeric.to_numpy(dtype=np.float64), observed, numeric_cols, timestamps


def _preprocess_plain_multivariate(name: str, filename: str, train_ratio: float, val_ratio: float, corr_top_k: int) -> None:
    path = RAW_ROOT / name / filename
    if not path.exists():
        raise FileNotFoundError(f"Run download {name} first")
    values, observed, node_ids, timestamps = _read_ts_csv(path, train_ratio=train_ratio)
    T, n = values.shape
    train_end, val_end = chronological_split(T, train_ratio, val_ratio)
    clean, observed = clean_numeric_frame(pd.DataFrame(values, columns=node_ids), train_end=train_end)
    values = clean.to_numpy(dtype=np.float64)
    out = PROCESSED_ROOT / name
    save_series(out, values, observed, timestamps, train_end, val_end)
    pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids}).to_csv(out / "nodes.csv", index=False)
    candidate = correlation_graph(values[:train_end], top_k=corr_top_k)
    save_graph(out / "graph_candidate.npz", candidate)
    _write_common_graphs(out, n)
    atomic_json(out / "metadata.json", {
        "dataset": name,
        "time_steps": T,
        "num_variables": n,
        "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
        "candidate_relation": {"type": "train_only_absolute_correlation_top_k", "top_k": corr_top_k},
        "relation_status": "candidate_not_ground_truth",
        "timestamp_policy": "stable sort; exact duplicate timestamp rows dropped; conflicting duplicates rejected",
    })
    print(f"[prepared] {name}: {values.shape}, candidate_edges={candidate.edge_index.shape[1]}")


def preprocess_weather(train_ratio: float = 0.80, val_ratio: float = 0.10, corr_top_k: int = 3) -> None:
    _preprocess_plain_multivariate("weather", "weather.csv", train_ratio, val_ratio, corr_top_k)


def preprocess_electricity(train_ratio: float = 0.80, val_ratio: float = 0.10, corr_top_k: int = 10) -> None:
    _preprocess_plain_multivariate("electricity", "electricity.csv", train_ratio, val_ratio, corr_top_k)


def preprocess_coast(
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    raw_name: str = "coast",
    output_name: str = "coast",
    relation_status: str = "user_supplied_frozen_candidate",
) -> None:
    raw = RAW_ROOT / raw_name
    series_path, edges_path = raw / "series.csv", raw / "edges.csv"
    if not series_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Place series.csv and edges.csv in {raw}; see data/raw/coast templates")
    values, observed, node_ids, timestamps = _read_ts_csv(series_path, train_ratio=train_ratio)
    T, n = values.shape
    train_end, val_end = chronological_split(T, train_ratio, val_ratio)
    clean, observed = clean_numeric_frame(pd.DataFrame(values, columns=node_ids), train_end=train_end)
    values = clean.to_numpy(dtype=np.float64)
    out = PROCESSED_ROOT / output_name
    save_series(out, values, observed, timestamps, train_end, val_end)
    node_map = {str(v): i for i, v in enumerate(node_ids)}
    pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids}).to_csv(out / "nodes.csv", index=False)
    e = pd.read_csv(edges_path)
    required = {"src", "dst"}
    if not required.issubset(e.columns):
        raise ValueError(f"Coast edges must contain {sorted(required)}")
    weights = e["weight"] if "weight" in e.columns else pd.Series(np.ones(len(e)))
    rows = []
    for idx, row in e.iterrows():
        s, d = str(row["src"]), str(row["dst"])
        if s not in node_map or d not in node_map:
            raise ValueError(f"Coast edge references unknown node: {s}->{d}")
        rows.append((node_map[s], node_map[d], float(weights.iloc[idx])))
    physical = from_edges(rows, n=n, add_self=True, directed=True)
    save_graph(out / "graph_physical.npz", physical)
    _write_common_graphs(out, n)
    e.to_csv(out / "edges_with_attributes.csv", index=False)
    atomic_json(out / "metadata.json", {
        "dataset": output_name,
        "source_raw_directory": raw_name,
        "time_steps": T,
        "num_variables": n,
        "split": {"train_end": train_end, "val_end": val_end},
        "relation": relation_status,
        "edge_attribute_columns": e.columns.tolist(),
        "timestamp_policy": "stable sort; exact duplicate timestamp rows dropped; conflicting duplicates rejected",
    })
    print(f"[prepared] coast: {values.shape}")


def _discover_causalrivers_files(base: Path, region: str) -> tuple[Path, Path, Path | None]:
    files = [p for p in base.rglob("*") if p.is_file()]
    reg_tokens = [region.lower(), "east_germany" if region == "east" else region.lower()]
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    time_candidates = [p for p in csvs if "river" in p.name.lower() and "ts" in p.name.lower() and any(t in p.name.lower() for t in reg_tokens)]
    if not time_candidates:
        time_candidates = [p for p in csvs if any(t in p.name.lower() for t in reg_tokens)]
    graph_ext = {".p", ".pkl", ".pickle", ".gpickle"}
    graph_candidates = [p for p in files if p.suffix.lower() in graph_ext and any(t in p.name.lower() for t in reg_tokens)]
    metadata_candidates = [p for p in csvs if any(k in p.name.lower() for k in ["meta", "node"]) and any(t in p.name.lower() for t in reg_tokens)]
    if not time_candidates or not graph_candidates:
        listing = "\n".join(str(p.relative_to(base)) for p in files[:500])
        raise RuntimeError(
            "Could not uniquely discover CausalRivers files. Candidate listing follows; inspect official release and update adapter:\n" + listing
        )
    if len(time_candidates) != 1 or len(graph_candidates) != 1:
        candidates = {
            "timeseries": [str(p.relative_to(base)) for p in sorted(time_candidates)],
            "graphs": [str(p.relative_to(base)) for p in sorted(graph_candidates)],
        }
        raise RuntimeError("CausalRivers adapter found multiple candidates; explicit mapping is required: " + json.dumps(candidates))
    if len(metadata_candidates) > 1:
        raise RuntimeError("CausalRivers adapter found multiple metadata candidates; explicit mapping is required")
    time_path = time_candidates[0]
    graph_path = graph_candidates[0]
    metadata_path = metadata_candidates[0] if metadata_candidates else None
    return time_path, graph_path, metadata_path


def _load_nx_graph(path: Path) -> nx.DiGraph:
    # CausalRivers publishes trusted NetworkX pickle artifacts in its official release.
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph in {path}, got {type(obj)}")
    if not obj.is_directed():
        raise ValueError(f"CausalRivers graph must be directed; refusing to fabricate direction from {path}")
    if isinstance(obj, (nx.MultiGraph, nx.MultiDiGraph)):
        simple = nx.DiGraph()
        simple.add_nodes_from(obj.nodes(data=True))
        for u, v, attrs in obj.edges(data=True):
            if not simple.has_edge(u, v):
                simple.add_edge(u, v, **attrs)
        obj = simple
    return nx.DiGraph(obj)


def _read_causalrivers_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CausalRivers CSV: {path}")
    # Wide format is expected by the official package. Handle a common long fallback.
    lower = {str(c).lower(): c for c in df.columns}
    time_col = next((lower[k] for k in ["timestamp", "datetime", "date", "time"] if k in lower), df.columns[0])
    if {"id", "value"}.issubset(lower):
        wide = df.pivot(index=time_col, columns=lower["id"], values=lower["value"])
        wide.index = pd.to_datetime(wide.index, errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if idx.notna().mean() < 0.95:
            raise ValueError(f"Could not parse CausalRivers timestamp column {time_col}")
        wide = df.drop(columns=[time_col])
        wide.index = idx
    wide = wide.sort_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide.apply(pd.to_numeric, errors="coerce")


def _deterministic_subgraphs(graph: nx.DiGraph, eligible: set[str], size: int, count: int) -> list[list[str]]:
    g = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes}, copy=True)
    g = g.subgraph([n for n in g.nodes if n in eligible]).copy()
    if len(g) < size:
        raise ValueError(f"Only {len(g)} graph nodes match time series; need {size}")
    weak = g.to_undirected()
    centers = sorted(g.nodes, key=lambda n: (-(g.in_degree(n) + g.out_degree(n)), n))
    chosen: list[list[str]] = []
    used: set[str] = set()
    for center in centers:
        q = deque([center])
        seen = {center}
        nodes = []
        while q and len(nodes) < size:
            u = q.popleft()
            nodes.append(u)
            nbrs = sorted(weak.neighbors(u), key=lambda n: (-(g.in_degree(n) + g.out_degree(n)), n))
            for v in nbrs:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if len(nodes) < size:
            continue
        overlap = len(set(nodes) & used) / size
        if overlap > 0.25 and len(chosen) < count - 1:
            continue
        chosen.append(nodes)
        used.update(nodes)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError(f"Could only construct {len(chosen)} deterministic subgraphs of size {size}")
    return chosen


def preprocess_causalrivers(
    region: str = "east",
    resolution: str = "1h",
    subgraph_size: int = 20,
    n_subgraphs: int = 4,
    max_missing_fraction: float = 0.20,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    output_prefix: str | None = None,
) -> None:
    base = RAW_ROOT / "causalrivers" / "release"
    if not base.exists():
        raise FileNotFoundError("Run download causalrivers first")
    ts_path, graph_path, meta_path = _discover_causalrivers_files(base, region)
    g = _load_nx_graph(graph_path)
    wide = _read_causalrivers_timeseries(ts_path).resample(resolution).mean()
    provisional_train_end, _ = chronological_split(len(wide), train_ratio, val_ratio)
    # Node admission is frozen using training availability only.
    miss = wide.iloc[:provisional_train_end].isna().mean()
    eligible = set(miss[miss <= max_missing_fraction].index.astype(str))
    subgraphs = _deterministic_subgraphs(g, eligible, subgraph_size, n_subgraphs)
    for idx, node_ids in enumerate(subgraphs):
        frame = wide[node_ids]
        T, n = frame.shape
        train_end, val_end = chronological_split(T, train_ratio, val_ratio)
        clean, observed = clean_numeric_frame(frame, train_end=train_end)
        values = clean.to_numpy(dtype=np.float64)
        prefix = output_prefix or f"causalrivers_{region}"
        name = f"{prefix}_s{subgraph_size}_{idx:02d}"
        out = PROCESSED_ROOT / name
        save_series(out, values, observed, clean.index.astype("int64").to_numpy(), train_end, val_end)
        pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids}).to_csv(out / "nodes.csv", index=False)
        node_map = {node: i for i, node in enumerate(node_ids)}
        rows = []
        edge_attrs = []
        for u, v, attrs in nx.relabel_nodes(g, {x: str(x) for x in g.nodes}, copy=False).subgraph(node_ids).edges(data=True):
            weight = attrs.get("geo_distance", attrs.get("weight", 1.0))
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            if not np.isfinite(weight):
                weight = 1.0
            rows.append((node_map[u], node_map[v], weight))
            edge_attrs.append({"src": u, "dst": v, **{str(k): str(val) for k, val in attrs.items()}})
        physical = from_edges(rows, n=n, add_self=True, directed=True)
        save_graph(out / "graph_physical.npz", physical)
        _write_common_graphs(out, n)
        pd.DataFrame(edge_attrs).to_csv(out / "edges_with_attributes.csv", index=False)
        atomic_json(out / "metadata.json", {
            "dataset": name,
            "parent_dataset": "causalrivers",
            "region": region,
            "time_series_source": str(ts_path.relative_to(base)),
            "graph_source": str(graph_path.relative_to(base)),
            "metadata_source": str(meta_path.relative_to(base)) if meta_path else None,
            "resolution": resolution,
            "time_steps": T,
            "num_variables": n,
            "max_missing_fraction": max_missing_fraction,
            "selection": "deterministic high-degree centered weakly connected BFS; no model-result filtering",
            "split": {"train_end": train_end, "val_end": val_end, "train_ratio": train_ratio, "val_ratio": val_ratio},
        })
        print(f"[prepared] {name}: {values.shape}, graph_edges={physical.edge_index.shape[1]}")
