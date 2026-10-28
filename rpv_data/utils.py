from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _external_root(name: str, default: Path) -> Path:
    """Resolve an optional externally mounted release root.

    The source checkout can live below a read-only code snapshot while the
    raw/processed data, reports, and manifests live on a separate volume. An
    environment variable is deliberately required for that split; otherwise
    the historical repository-local layout remains the default.
    """
    value = os.environ.get(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else (ROOT / path)).resolve()


DATA_ROOT = _external_root("RPV_DATA_ROOT", ROOT / "data")
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
REPORT_ROOT = _external_root("RPV_REPORT_ROOT", ROOT / "reports")
MANIFEST_ROOT = _external_root("RPV_MANIFEST_ROOT", ROOT / "manifests")


def ensure_dirs() -> None:
    for p in (RAW_ROOT, PROCESSED_ROOT, REPORT_ROOT, MANIFEST_ROOT):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, default=str)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def chronological_split(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n < 10:
        raise ValueError(f"Time series is too short: {n}")
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split ratios")
    train_end = int(round(n * train_ratio))
    val_end = int(round(n * (train_ratio + val_ratio)))
    if train_end < 2 or val_end <= train_end or val_end >= n:
        raise ValueError("Split boundaries are degenerate")
    return train_end, val_end


def clean_numeric_frame(df: pd.DataFrame, train_end: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Causally impute a numeric frame without borrowing future observations.

    Values are forward-filled along time. Leading gaps and columns with no
    earlier observation are filled with a median fitted on the training
    interval only. The original observation mask is returned unchanged.
    """
    numeric = df.apply(pd.to_numeric, errors="coerce")
    observed = numeric.notna().to_numpy(dtype=np.uint8)
    if not (0 < train_end <= len(numeric)):
        raise ValueError("Invalid train_end for imputation")
    filled = numeric.ffill()
    train_median = numeric.iloc[:train_end].median(axis=0, skipna=True)
    if train_median.isna().any():
        bad = train_median[train_median.isna()].index.tolist()
        raise ValueError(f"Columns have no observed training values: {bad[:10]}")
    filled = filled.fillna(train_median)
    if filled.isna().any().any():
        bad = filled.columns[filled.isna().any()].tolist()
        raise ValueError(f"Columns remain missing after causal imputation: {bad[:10]}")
    return filled, observed


def fit_train_standardizer(values: np.ndarray, observed: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"Expected [time, variables], got {values.shape}")
    train = values[:train_end]
    train_obs = observed[:train_end]
    masked = np.where(train_obs, train, np.nan)
    mean = np.nanmean(masked, axis=0)
    std = np.nanstd(masked, axis=0)
    fallback_mean = np.nanmean(train, axis=0)
    mean = np.where(np.isfinite(mean), mean, fallback_mean)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    normalized = (values - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite standardized values")
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def save_series(
    out_dir: Path,
    raw_values: np.ndarray,
    observed_mask: np.ndarray,
    timestamps: np.ndarray,
    train_end: int,
    val_end: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    values, mean, std = fit_train_standardizer(raw_values, observed_mask, train_end)
    np.savez_compressed(
        out_dir / "series.npz",
        values=values,
        raw_values=np.asarray(raw_values, dtype=np.float32),
        observed_mask=np.asarray(observed_mask, dtype=np.uint8),
        timestamps=np.asarray(timestamps),
        train_end=np.int64(train_end),
        val_end=np.int64(val_end),
        mean=mean,
        std=std,
    )


def dataframe_timestamp_ns(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.isna().any():
        raise ValueError("Timestamp column contains unparsable values")
    return dt.astype("int64").to_numpy()


def stamp_metadata(path: Path, extra: dict[str, Any]) -> None:
    """Atomically add immutable protocol/provenance fields to dataset metadata."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    obj.update(extra)
    atomic_json(path, obj)


def is_strictly_increasing(values: np.ndarray) -> bool:
    arr = np.asarray(values)
    return arr.ndim == 1 and len(arr) > 1 and bool(np.all(arr[1:] > arr[:-1]))
