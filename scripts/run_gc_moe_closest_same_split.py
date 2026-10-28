"""Evaluate upstream GC-MoE on the CREA chronological split.

The adapter prepares the upstream loader format and evaluates all test origins.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

if os.name == "posix":
    import fcntl


HISTORY = 12
HORIZON = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def starts(begin: int, end: int, total: int) -> np.ndarray:
    lo = max(0, int(begin))
    hi = min(int(end) - HISTORY - HORIZON, int(total) - HISTORY - HORIZON)
    if hi < lo:
        raise ValueError(f"empty window interval [{lo}, {hi}]")
    return np.arange(lo, hi + 1, dtype=np.int64) + HISTORY - 1


def cap(values: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or values.size <= limit:
        return values
    return values[np.unique(np.linspace(0, values.size - 1, limit, dtype=np.int64))]


@contextlib.contextmanager
def _native_lock(path: Path):
    """Serialize per-dataset native-cache creation across parallel seeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cached_native(native: Path, dataset: str, train_cap: int, val_cap: int, nodes: int) -> dict | None:
    """Reuse a complete materialization without re-reading or re-hashing raw data."""
    required = [
        native / "his.npz",
        native / "idx_train.npy",
        native / "idx_val.npy",
        native / "idx_test.npy",
        native / f"{dataset.lower()}_rn_adj.npy",
        native / "materialization.json",
    ]
    if not all(p.is_file() and p.stat().st_size > 0 for p in required):
        return None
    try:
        spec = json.loads((native / "materialization.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        spec.get("schema") != "w_sdm_gc_moe_native_materialization_v1"
        or spec.get("dataset") != dataset
        or int(spec.get("history", -1)) != HISTORY
        or int(spec.get("horizon", -1)) != HORIZON
        or int(spec.get("train_cap", -1)) != train_cap
        or int(spec.get("validation_cap", -1)) != val_cap
    ):
        return None
    return {"native": native, "adj": native / f"{dataset.lower()}_rn_adj.npy", "nodes": nodes, "spec": spec}


def materialize_native(dataset: str, data_root: Path, native_root: Path, train_cap: int, val_cap: int) -> dict:
    src = data_root / dataset
    meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    total = int(meta["time_steps"])
    nodes = int(meta["num_variables"])
    # FINAL_V1 uses the locked five-way chronological protocol for every
    # dataset.  Dataset-release metadata is not reused because PEMS08 ships
    # an 80/10/10 split rather than the paper's 60/10/10/10/10 contract.
    train_end = int(round(total * 0.60))
    val_end = int(round(total * 0.70))
    test_start = int(round(total * 0.90))
    native = native_root / dataset / "v1"
    train_idx = cap(starts(0, train_end, total), train_cap)
    val_idx = cap(starts(train_end - HISTORY, val_end, total), val_cap)
    test_idx = starts(test_start - HISTORY, total, total)
    with _native_lock(native_root / dataset / ".materialize.lock"):
        cached = _cached_native(native, dataset, train_cap, val_cap, nodes)
        if cached is not None:
            return cached
        z = np.load(src / "series.npz")
        key = "data" if "data" in z.files else z.files[0]
        series = np.asarray(z[key], dtype=np.float32)
        if series.ndim == 3:
            series = series[..., 0]
        if series.shape != (total, nodes):
            raise ValueError(f"unexpected series shape {series.shape} for {dataset}")
        native.mkdir(parents=True, exist_ok=True)
        mean = float(series[:train_end].mean())
        std = float(series[:train_end].std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError("invalid train-only scale")
        data = series[..., None].astype(np.float32)
        adj_z = np.load(src / "graph_physical.npz")
        adj = np.zeros((nodes, nodes), dtype=np.float32)
        edge_index = np.asarray(adj_z["edge_index"])
        edge_weight = np.asarray(adj_z["edge_weight"], dtype=np.float32)
        # The processed contract is source -> destination.  The forecasting
        # implementations represent destination queries attending source keys,
        # so use destination,row / source,column consistently.
        adj[edge_index[1], edge_index[0]] = edge_weight
        np.savez_compressed(native / "his.npz", data=data, mean=np.float32(mean), std=np.float32(std))
        np.save(native / "idx_train.npy", train_idx)
        np.save(native / "idx_val.npy", val_idx)
        np.save(native / "idx_test.npy", test_idx)
        np.save(native / f"{dataset.lower()}_rn_adj.npy", adj)
        spec = {
            "schema": "w_sdm_gc_moe_native_materialization_v1",
            "dataset": dataset,
            "nodes": nodes,
            "history": HISTORY,
            "horizon": HORIZON,
            "train_cap": train_cap,
            "validation_cap": val_cap,
            "test_origins": int(test_idx.size),
            "split": {
                "train_end": train_end,
                "val_end": val_end,
                "test_start": test_start,
                "train_ratio": train_end / total,
                "validation_ratio": (val_end - train_end) / total,
                "excluded_gate_ratio": (test_start - val_end) / total,
                "test_ratio": (total - test_start) / total,
            },
            "data_sha256": sha256(src / "series.npz"),
            "graph_sha256": sha256(src / "graph_physical.npz"),
        }
        (native / "materialization.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return {"native": native, "adj": native / f"{dataset.lower()}_rn_adj.npy", "nodes": nodes, "spec": spec}


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("GC-MoE comparison requires CUDA; refusing CPU fallback")
    threads = max(1, int(os.environ.get("RPV_TORCH_NUM_THREADS", "4")))
    interop = max(1, int(os.environ.get("RPV_TORCH_INTEROP_THREADS", "1")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    source = Path(args.upstream).resolve()
    data_root = Path(args.data_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    native = materialize_native(args.dataset.lower(), data_root, Path(args.native_root), args.train_cap, args.val_cap)
    sys.path.insert(0, str(source))
    os.chdir(source)
    from src.utils.args import get_config  # type: ignore
    import src.__init__ as gc_init  # type: ignore

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    argv = [
        "gc_moe_closest",
        "--device", "cuda",
        "--dataset", args.dataset.upper(),
        "--years", "v1",
        "--model", "moe",
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--seq_length", str(HISTORY),
        "--horizon", str(HORIZON),
        "--input_dim", "1",
        "--output_dim", "1",
        "--max_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--run_id", f"same_split_{args.dataset}_{args.seed}",
        "--expert_models", "gwnet", "stgcn", "agcrn",
        "--router_type", "graph_conditioned",
        "--save", "gc_moe_model.pt",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        cfg = get_config()
    finally:
        sys.argv = old_argv
    cfg.data_path = str(native["native"].parent)
    cfg.adj_path = str(native["adj"])
    cfg.node_num = native["nodes"]
    cfg.log_dir = str(out / "checkpoint")
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    cfg.adj_mx = gc_init.load_adj_from_numpy(cfg.adj_path)
    cfg.supports = [torch.tensor(x, dtype=torch.float32, device=cfg.device) for x in gc_init.normalize_adj_mx(cfg.adj_mx, cfg.adj_type)]
    cfg.loss_fn = gc_init.masked_mae
    cfg.dataloader, cfg.scaler = gc_init.load_dataset(cfg.data_path, cfg)
    engine = gc_init.get_engine(cfg)
    engine.train()
    result = engine.test(save_path=str(out / "predictions"))
    payload = {
        "schema": "w_sdm_gc_moe_closest_same_split_v1",
        "baseline": "gc_moe_closest_official_graph_conditioned_router",
        "dataset": args.dataset,
        "seed": args.seed,
        "upstream_root": str(source),
        "upstream_commit": "unknown",
        "data_spec": native["spec"],
        "model": {"experts": ["gwnet", "stgcn", "agcrn"], "router": "graph_conditioned", "frozen_experts": False},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_selection_use": "none",
        "metrics": result,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=lambda x: float(x)), encoding="utf-8")
    return out / "metrics.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["pems04", "pems07", "metr_la"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--native-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-cap", type=int, default=1024)
    p.add_argument("--val-cap", type=int, default=1024)
    args = p.parse_args()
    print(run(args), flush=True)


if __name__ == "__main__":
    main()
