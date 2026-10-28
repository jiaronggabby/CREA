"""Evaluate upstream STAEformer or STID on the CREA chronological split.

Model implementations are loaded from the supplied upstream directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _configure_runtime_threads() -> None:
    """Bound host threading for concurrent baseline jobs."""
    threads = max(1, int(os.environ.get("RPV_TORCH_NUM_THREADS", "4")))
    interop = max(1, int(os.environ.get("RPV_TORCH_INTEROP_THREADS", "1")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_model(name: str, upstream: Path, dataset: str, nodes: int):
    if name == "staeformer":
        sys_path = upstream / "model"
        import sys

        if "torchinfo" not in sys.modules:
            import types

            shim = types.ModuleType("torchinfo")
            shim.summary = lambda *args, **kwargs: None
            sys.modules["torchinfo"] = shim
        sys.path.insert(0, str(sys_path))
        from STAEformer import STAEformer  # type: ignore

        cfg = {
            "num_nodes": nodes,
            "in_steps": 12,
            "out_steps": 12,
            "steps_per_day": 288,
            "input_dim": 3,
            "output_dim": 1,
            "input_embedding_dim": 24,
            "tod_embedding_dim": 24,
            "dow_embedding_dim": 24,
            "spatial_embedding_dim": 0,
            "adaptive_embedding_dim": 80,
            "feed_forward_dim": 256,
            "num_heads": 4,
            "num_layers": 3,
            "dropout": 0.1,
        }
        return STAEformer(**cfg), {"model_config": cfg, "official_module": "model/STAEformer.py"}
    if name == "stid":
        arch = upstream / "stid" / "arch" / "stid_arch.py"
        spec = importlib.util.spec_from_file_location("stid_arch", arch)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import upstream STID architecture: {arch}")
        module = importlib.util.module_from_spec(spec)
        # stid_arch uses a relative import for mlp; load it as a package module.
        import sys

        if "torchinfo" not in sys.modules:
            import types

            shim = types.ModuleType("torchinfo")
            shim.summary = lambda *args, **kwargs: None
            sys.modules["torchinfo"] = shim
        sys.path.insert(0, str(upstream))
        from stid.arch.stid_arch import STID  # type: ignore

        cfg = {
            "num_nodes": nodes,
            "node_dim": 32,
            "input_len": 12,
            "input_dim": 3,
            "embed_dim": 32,
            "output_len": 12,
            "num_layer": 3,
            "temp_dim_tid": 32,
            "temp_dim_diw": 32,
            "time_of_day_size": 288,
            "day_of_week_size": 7,
            "if_T_i_D": True,
            "if_D_i_W": True,
            "if_node": True,
        }
        return STID(**cfg), {"model_config": cfg, "official_module": "stid/arch/stid_arch.py"}
    raise ValueError(f"Unsupported external baseline: {name}")


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, starts: np.ndarray, *, stid: bool):
        self.data = data
        self.starts = starts.astype(np.int64, copy=False)
        self.stid = stid

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        x = self.data[start : start + 12].copy()
        y = self.data[start + 12 : start + 24, :, :1].copy()
        if self.stid:
            # STID expects normalized day-of-week values and converts them to
            # integer embedding indices internally.
            x[..., 2] = x[..., 2] / 7.0
        return torch.from_numpy(x), torch.from_numpy(y)


def _starts(begin: int, end: int, total: int) -> np.ndarray:
    lo = max(0, int(begin))
    hi = min(int(end) - 24, int(total) - 24)
    if hi < lo:
        raise ValueError(f"Empty window interval [{lo}, {hi}]")
    return np.arange(lo, hi + 1, dtype=np.int64)


def _cap_starts(starts: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or cap <= 0 or starts.size <= cap:
        return starts
    # Preserve the absolute origin values when selecting capped windows.
    # Returning positions directly would leak the training period into the
    # validation split whenever ``starts`` has a non-zero offset.
    positions = np.unique(np.linspace(0, starts.size - 1, int(cap), dtype=np.int64))
    return starts[positions]


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-5)) * 100.0),
    }


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("External baseline requires CUDA; refusing CPU fallback")
    _configure_runtime_threads()
    source = Path(args.upstream).resolve()
    data_root = Path(args.data_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    metadata = json.loads((data_root / args.dataset / "metadata.json").read_text())
    total = int(metadata["time_steps"])
    nodes = int(metadata["num_variables"])
    # Use the single FINAL_V1 external-baseline split for every dataset.
    # PEMS08 metadata contains an older 80/10 split and must not override the
    # locked 60/10/20/10 protocol.
    metadata_split = metadata.get("split", {})
    train_end = int(round(total * 0.60))
    val_end = int(round(total * 0.70))
    test_start = int(round(total * 0.90))
    series_file = data_root / args.dataset / "series.npz"
    z = np.load(series_file)
    key = "data" if "data" in z.files else z.files[0]
    series = np.asarray(z[key], dtype=np.float32)
    if series.ndim == 3:
        series = series[..., 0]
    if series.shape != (total, nodes):
        raise ValueError(f"Unexpected series shape {series.shape}, expected {(total, nodes)}")

    tod = (np.arange(total, dtype=np.float32) % 288) / 288.0
    dow = (np.arange(total, dtype=np.float32) // 288) % 7
    data = np.empty((total, nodes, 3), dtype=np.float32)
    data[..., 0] = series
    data[..., 1] = tod[:, None]
    data[..., 2] = dow[:, None]
    mean = float(series[:train_end].mean())
    std = float(series[:train_end].std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid train-only scale")
    data[..., 0] = (data[..., 0] - mean) / std
    train_starts = _cap_starts(_starts(0, train_end, total), args.max_train_windows)
    val_starts = _cap_starts(_starts(train_end - 12, val_end, total), args.max_val_windows)
    test_starts = _starts(test_start - 12, total, total)
    stid = args.baseline == "stid"
    train_loader = DataLoader(WindowDataset(data, train_starts, stid=stid), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(data, val_starts, stid=stid), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(data, test_starts, stid=stid), batch_size=args.batch_size, shuffle=False)
    model, model_meta = _load_model(args.baseline, source, args.dataset, nodes)
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if stid:
                pred = model(x, y, 0, epoch, True)
            else:
                pred = model(x)
            pred = pred * std + mean
            loss = criterion(pred, y * std + mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, y, 0, epoch, False) if stid else model(x)
                losses.append(float(criterion(pred * std + mean, y * std + mean).item()))
        val = float(np.mean(losses))
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No baseline checkpoint selected")
    model.load_state_dict(best_state)
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, y, 0, epoch, False) if stid else model(x)
            ps.append((pred * std + mean).cpu().numpy())
            ys.append((y * std + mean).cpu().numpy())
    y = np.concatenate(ys, axis=0)
    pred = np.concatenate(ps, axis=0)
    horizons = {}
    for h in (3, 6, 12):
        horizons[str(h)] = _metrics(y[:, h - 1], pred[:, h - 1])
    np.savez_compressed(out / "test_predictions.npz", y_true=y.astype(np.float32), y_pred=pred.astype(np.float32))
    test_start = int(round(total * 0.90))
    payload: dict[str, Any] = {
        "schema": "w_sdm_external_baseline_same_split_v1",
        "baseline": args.baseline,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "upstream_root": str(source),
        "upstream_commit": _git_head(source),
        "source_file_sha256": _sha256(series_file),
        "split": {
            "train_end": train_end,
            "val_end": val_end,
            "test_start": test_start,
            "split_source": "final_v1_locked_ratio_60_10_20_10",
            "metadata_split": metadata_split,
            "forecaster_train_ratio": train_end / total,
            "forecaster_validation_ratio": (val_end - train_end) / total,
            "excluded_gate_ratio": (test_start - val_end) / total,
            "test_ratio": (total - test_start) / total,
        },
        "train_only_scaler": {"mean": mean, "std": std, "train_end": train_end},
        "window_counts": {"train": len(train_starts), "validation": len(val_starts), "test": len(test_starts)},
        "window_caps": {"train": args.max_train_windows, "validation": args.max_val_windows, "test": None},
        "batch_size": int(args.batch_size),
        "optimizer": {"name": "Adam", "lr": args.lr, "weight_decay": args.weight_decay},
        "best_validation_mae": best_val,
        "epochs_run": epoch + 1,
        "horizons": horizons,
        "model": model_meta,
        "test_selection_use": "none",
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out / "metrics.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["staeformer", "stid"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0003)
    p.add_argument("--max-train-windows", type=int, default=1024)
    p.add_argument("--max-val-windows", type=int, default=1024)
    args = p.parse_args()
    print(run(args), flush=True)


if __name__ == "__main__":
    main()
