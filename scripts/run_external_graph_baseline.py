"""Evaluate upstream graph forecasters on the CREA chronological split.

Model implementations are loaded from the supplied upstream directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(path: Path) -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _starts(begin: int, end: int, total: int) -> np.ndarray:
    lo = max(0, int(begin))
    hi = min(int(end) - 24, int(total) - 24)
    if hi < lo:
        raise ValueError(f"Empty window interval [{lo}, {hi}]")
    return np.arange(lo, hi + 1, dtype=np.int64)


def _cap_starts(starts: np.ndarray, cap: int | None) -> np.ndarray:
    """Deterministically cap training/validation origins without touching test."""
    if cap is None or cap <= 0 or starts.size <= cap:
        return starts
    # ``starts`` contains absolute origins.  Select positions in that array,
    # then map them back to absolute origins; returning the positions directly
    # would silently move a capped validation split into the training period.
    positions = np.unique(np.linspace(0, starts.size - 1, int(cap), dtype=np.int64))
    return starts[positions]


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, starts: np.ndarray):
        self.values = values
        self.starts = starts.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        x = self.values[start : start + 12].copy()
        y = self.values[start + 12 : start + 24, :, :1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-5)) * 100.0),
    }


def _normalize_graph(data_root: Path, dataset: str, nodes: int) -> np.ndarray:
    z = np.load(data_root / dataset / "graph_physical.npz")
    edges = np.asarray(z["edge_index"], dtype=np.int64)
    weights = np.asarray(z["edge_weight"], dtype=np.float32)
    a = np.zeros((nodes, nodes), dtype=np.float32)
    # The processed contract is source,destination.  Use destination queries
    # attending source keys, then add self loops and row-normalize.
    a[edges[1], edges[0]] = weights
    a += np.eye(nodes, dtype=np.float32)
    a /= np.maximum(a.sum(axis=1, keepdims=True), 1e-6)
    return a


def _import_model(name: str, upstream: Path, nodes: int, device: torch.device):
    """Import one official model in a fresh process (one baseline per run)."""
    sys.path.insert(0, str(upstream))
    if name == "graph_wavenet":
        sys.path.insert(0, str(upstream))
        model_mod = importlib.import_module("model")
        a = _normalize_graph(Path(os.environ["RPV_EXTERNAL_DATA_ROOT"]), os.environ["RPV_EXTERNAL_DATASET"], nodes)
        supports = [torch.from_numpy(a).to(device), torch.from_numpy(a.T.copy()).to(device)]
        model = model_mod.gwnet(
            device=device,
            num_nodes=nodes,
            dropout=0.3,
            supports=supports,
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=1,
            out_dim=12,
            residual_channels=32,
            dilation_channels=32,
            skip_channels=256,
            end_channels=512,
            kernel_size=2,
            blocks=4,
            layers=2,
        )
        # The audited upstream release declares gate_convs as Conv1d while
        # passing 4-D tensors.  Convert only that layer type at the adapter
        # boundary; the weights, kernel and dilation are unchanged.
        def _conv1d_to_2d(conv: nn.Module) -> nn.Module:
            if not isinstance(conv, nn.Conv1d):
                return conv
            repl = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=(1, conv.kernel_size[-1]),
                dilation=(1, conv.dilation[-1]),
                bias=conv.bias is not None,
            )
            if conv.weight.ndim == 4:
                repl.weight.data.copy_(conv.weight.data)
            else:
                repl.weight.data.copy_(conv.weight.data.unsqueeze(2))
            if conv.bias is not None:
                repl.bias.data.copy_(conv.bias.data)
            return repl

        for i, conv in enumerate(model.gate_convs):
            if isinstance(conv, nn.Conv1d):
                model.gate_convs[i] = _conv1d_to_2d(conv)
        for collection in (model.residual_convs, model.skip_convs):
            for i, conv in enumerate(collection):
                collection[i] = _conv1d_to_2d(conv)
        return model, {"model_config": {"in_dim": 1, "out_dim": 12, "blocks": 4, "layers": 2, "residual_channels": 32}, "official_module": "model.py", "adapter_patches": ["upstream_gate_conv1d_to_conv2d"]}
    if name == "mtgnn":
        layer = importlib.import_module("layer")
        net = importlib.import_module("net")
        a = _normalize_graph(Path(os.environ["RPV_EXTERNAL_DATA_ROOT"]), os.environ["RPV_EXTERNAL_DATASET"], nodes)
        predefined = torch.from_numpy(a).to(device)
        model = net.gtnet(
            gcn_true=True,
            buildA_true=False,
            gcn_depth=2,
            num_nodes=nodes,
            device=device,
            predefined_A=predefined,
            static_feat=None,
            dropout=0.3,
            subgraph_size=min(20, nodes),
            node_dim=40,
            dilation_exponential=1,
            conv_channels=32,
            residual_channels=32,
            skip_channels=64,
            end_channels=128,
            seq_length=12,
            in_dim=1,
            out_dim=12,
            layers=3,
            propalpha=0.05,
            tanhalpha=3,
            layer_norm_affline=True,
        )
        return model, {"model_config": {"in_dim": 1, "out_dim": 12, "layers": 3, "gcn_depth": 2, "buildA_true": False}, "official_module": "net.py"}
    if name == "agcrn":
        module = importlib.import_module("model.AGCRN")
        args = SimpleNamespace(
            num_nodes=nodes,
            input_dim=1,
            rnn_units=64,
            output_dim=1,
            horizon=12,
            num_layers=2,
            default_graph=False,
            cheb_k=2,
            embed_dim=10,
        )
        model = module.AGCRN(args)
        # The audited upstream AGCRN release leaves AVWGCN weight/bias pools
        # as raw FloatTensor parameters and initializes them in its standalone
        # training entrypoint.  This adapter constructs the model directly,
        # so reproduce that published initialization here; otherwise the
        # uninitialized pools can produce a non-finite loss at epoch 0.
        for parameter in model.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)
        return model, {"model_config": vars(args), "official_module": "model/AGCRN.py"}
    if name == "dcrnn":
        # The official TensorFlow reference is not executable in the pinned
        # CUDA/PyTorch environment.  GC-MoE publishes a faithful PyTorch
        # DCRNN reference in the audited checkout; we record that exact
        # upstream identity instead of silently substituting a local model.
        if "wandb" not in sys.modules:
            import types

            wandb_stub = types.ModuleType("wandb")
            wandb_stub.run = None
            sys.modules["wandb"] = wandb_stub
        import types
        from importlib import util as importlib_util

        # Import only the DCRNN file from GC-MoE.  Its package initializer
        # eagerly imports optional torchdiffeq/Lightning models that are not
        # part of this baseline and are intentionally absent from the pinned
        # environment.
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(upstream / "src")]
        base_pkg = types.ModuleType("src.base")
        base_pkg.__path__ = [str(upstream / "src" / "base")]
        utils_pkg = types.ModuleType("src.utils")
        utils_pkg.__path__ = [str(upstream / "src" / "utils")]
        sys.modules["src"] = src_pkg
        sys.modules["src.base"] = base_pkg
        sys.modules["src.utils"] = utils_pkg

        def _load_file(mod_name: str, file_path: Path):
            spec = importlib_util.spec_from_file_location(mod_name, file_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load upstream module {file_path}")
            m = importlib_util.module_from_spec(spec)
            sys.modules[mod_name] = m
            spec.loader.exec_module(m)
            return m

        _load_file("src.base.model", upstream / "src" / "base" / "model.py")
        _load_file("src.utils.graph_algo", upstream / "src" / "utils" / "graph_algo.py")
        module = _load_file("dcrnn_external", upstream / "src" / "models" / "dcrnn.py")
        a = _normalize_graph(Path(os.environ["RPV_EXTERNAL_DATA_ROOT"]), os.environ["RPV_EXTERNAL_DATASET"], nodes)
        model = module.DCRNN(
            device=device,
            adj_mx=a,
            n_filters=64,
            max_diffusion_step=2,
            filter_type="doubletransition",
            num_rnn_layers=2,
            cl_decay_steps=2000,
            use_curriculum_learning=True,
            node_num=nodes,
            input_dim=1,
            output_dim=1,
            seq_len=12,
            horizon=12,
        )
        return model, {"model_config": {"input_dim": 1, "horizon": 12, "n_filters": 64, "max_diffusion_step": 2, "filter_type": "doubletransition", "num_rnn_layers": 2}, "official_module": "src/models/dcrnn.py", "adapter_note": "published_pytorch_reference_for_tf_dcrnn"}
    if name == "megacrn":
        module = importlib.import_module("model.MegaCRN")
        model = module.MegaCRN(
            num_nodes=nodes,
            input_dim=1,
            output_dim=1,
            horizon=12,
            rnn_units=64,
            num_layers=1,
            cheb_k=2,
            ycov_dim=1,
            mem_num=20,
            mem_dim=32,
            cl_decay_steps=2000,
            use_curriculum_learning=True,
        )
        return model, {"model_config": {"input_dim": 1, "horizon": 12, "rnn_units": 64, "mem_num": 20, "mem_dim": 32}, "official_module": "model/MegaCRN.py"}
    raise ValueError(name)


def _forward(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor, epoch: int, train: bool) -> torch.Tensor:
    def canonical(raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 4:
            raise ValueError(f"Unexpected graph baseline output rank: {tuple(raw.shape)}")
        if raw.shape[1] == 12 and raw.shape[-1] == 1:
            return raw
        if raw.shape[1] == 1 and raw.shape[-1] == 12:
            return raw.permute(0, 3, 2, 1).contiguous()
        raise ValueError(f"Cannot canonicalize graph baseline output: {tuple(raw.shape)}")

    if name == "graph_wavenet":
        # Official model uses B,C,N,T.
        return canonical(model(x.permute(0, 3, 2, 1).contiguous()))
    if name == "mtgnn":
        return canonical(model(x.permute(0, 3, 2, 1).contiguous()))
    if name == "agcrn":
        return model(x, y, teacher_forcing_ratio=0.0)
    if name == "dcrnn":
        return model(x, y, iter=epoch if train else None)
    if name == "megacrn":
        cov = torch.zeros((x.shape[0], 12, x.shape[2], 1), dtype=x.dtype, device=x.device)
        pred, *_ = model(x, cov, labels=y, batches_seen=epoch)
        return pred
    raise ValueError(name)


def _configure_runtime_threads() -> None:
    """Bound host threading for concurrent baseline jobs.

    This changes only host scheduling, not the model, split, optimizer, or
    numerical protocol. A bounded default prevents concurrent jobs from
    creating more runnable threads than the shared node has CPUs.
    """
    threads = max(1, int(os.environ.get("RPV_TORCH_NUM_THREADS", "4")))
    interop = max(1, int(os.environ.get("RPV_TORCH_INTEROP_THREADS", "1")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("External baseline requires CUDA; refusing CPU fallback")
    _configure_runtime_threads()
    source = Path(args.upstream).resolve()
    data_root = Path(args.data_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["RPV_EXTERNAL_DATA_ROOT"] = str(data_root)
    os.environ["RPV_EXTERNAL_DATASET"] = args.dataset
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    metadata = json.loads((data_root / args.dataset / "metadata.json").read_text())
    total = int(metadata["time_steps"])
    nodes = int(metadata["num_variables"])
    # FINAL_V1 external baselines use one common chronological split across
    # every dataset, including PEMS08: 60% train, 10% forecaster validation,
    # 20% gate-validation holdout, and 10% locked test.  The processed PEMS08
    # metadata predates this matrix and advertises an 80/10 split, so never
    # inherit its dataset-local endpoints here.
    metadata_split = metadata.get("split", {})
    train_end = int(round(total * 0.60))
    val_end = int(round(total * 0.70))
    test_start = int(round(total * 0.90))
    z = np.load(data_root / args.dataset / "series.npz")
    key = "data" if "data" in z.files else z.files[0]
    series = np.asarray(z[key], dtype=np.float32)
    if series.ndim == 3:
        series = series[..., 0]
    if series.shape != (total, nodes):
        raise ValueError(f"Unexpected series shape {series.shape}, expected {(total, nodes)}")
    mean = float(series[:train_end].mean())
    std = float(series[:train_end].std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid train-only scale")
    values = ((series - mean) / std)[..., None].astype(np.float32)
    train_starts = _cap_starts(_starts(0, train_end, total), args.max_train_windows)
    val_starts = _cap_starts(_starts(train_end - 12, val_end, total), args.max_val_windows)
    test_starts = _starts(test_start - 12, total, total)
    train_loader = DataLoader(WindowDataset(values, train_starts), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(WindowDataset(values, val_starts), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(WindowDataset(values, test_starts), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda")
    model, model_meta = _import_model(args.baseline, source, nodes, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    wait = 0
    epoch = -1
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = _forward(args.baseline, model, x, y, epoch, True)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = _forward(args.baseline, model, x, y, epoch, False)
                val_loss = criterion(pred, y)
                if not torch.isfinite(val_loss):
                    raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
                vals.append(float(val_loss.item()))
        val = float(np.mean(vals))
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
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = _forward(args.baseline, model, x, y, epoch, False)
            ps.append((pred * std + mean).cpu().numpy())
            ys.append((y * std + mean).cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    horizons = {str(h): _metrics(y_true[:, h - 1], y_pred[:, h - 1]) for h in (3, 6, 12)}
    if not np.isfinite(y_pred).all():
        raise FloatingPointError("non-finite test prediction")
    np.savez_compressed(out / "test_predictions.npz", y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32))
    test_start = int(round(total * 0.90))
    payload: dict[str, Any] = {
        "schema": "w_sdm_external_graph_baseline_same_split_v1",
        "baseline": args.baseline,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "upstream_root": str(source),
        "upstream_commit": _git_head(source),
        "source_file_sha256": _sha256(data_root / args.dataset / "series.npz"),
        "graph_file_sha256": _sha256(data_root / args.dataset / "graph_physical.npz"),
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
        "best_validation_mae_scaled": best_val,
        "epochs_run": epoch + 1,
        "horizons": horizons,
        "model": model_meta,
        "test_selection_use": "none",
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out / "metrics.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["dcrnn", "graph_wavenet", "mtgnn", "agcrn", "megacrn"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--max-train-windows", type=int, default=4096)
    p.add_argument("--max-val-windows", type=int, default=2048)
    args = p.parse_args()
    print(run(args), flush=True)


if __name__ == "__main__":
    main()
