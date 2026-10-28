from __future__ import annotations

import time
import zipfile
from pathlib import Path

import requests

from .registry import DATASETS
from .utils import RAW_ROOT, atomic_json, ensure_dirs, sha256_file


def _download(url: str, dest: Path, retries: int = 8, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    total_size: int | None = None
    chunk_size = 1024 * 1024
    while True:
        offset = tmp.stat().st_size if tmp.exists() else 0
        if total_size is not None and offset > total_size:
            tmp.unlink()
            offset = 0
        if total_size is not None and offset == total_size:
            tmp.replace(dest)
            print(f"[downloaded] {dest} ({dest.stat().st_size:,} bytes)")
            return
        expected = min(chunk_size, total_size - offset) if total_size is not None else chunk_size
        for attempt in range(1, retries + 1):
            try:
                end = offset + expected - 1
                with requests.get(
                    url,
                    headers={"Range": f"bytes={offset}-{end}"},
                    stream=True,
                    timeout=(20, timeout),
                    allow_redirects=True,
                ) as r:
                    r.raise_for_status()
                    content_range = r.headers.get("Content-Range", "")
                    if r.status_code == 206:
                        parts = content_range.removeprefix("bytes ").split("/")
                        bounds = parts[0].split("-") if parts else []
                        if len(bounds) != 2 or int(bounds[0]) != offset:
                            raise OSError(f"Invalid Content-Range for {url}: {content_range!r}")
                        response_end = int(bounds[1])
                        total_size = int(parts[1]) if len(parts) == 2 and parts[1] != "*" else total_size
                        expected = response_end - offset + 1
                        mode = "ab"
                    else:
                        # A mirror that ignores Range returns the complete file.
                        # Restarting from zero is safe and keeps the manifest exact.
                        offset = 0
                        total_size = int(r.headers.get("Content-Length", "0")) or None
                        expected = total_size
                        mode = "wb"
                    received = 0
                    with tmp.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                received += len(chunk)
                if expected is not None and received != expected:
                    raise OSError(
                        f"Truncated response for {url}: received {received}, expected {expected}"
                    )
                if total_size is None:
                    total_size = tmp.stat().st_size
                break
            except (OSError, requests.RequestException) as exc:
                last_error = exc
                if (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code == 416
                ):
                    # The partial file is longer than the remote object (a
                    # failed/restarted proxy transfer).  Discard only this
                    # explicit temporary file and restart from byte zero.
                    tmp.unlink(missing_ok=True)
                    total_size = None
                    break
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 10))
                else:
                    raise RuntimeError(f"Failed to download {url}: {last_error}") from exc


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        root = dest.resolve()
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        zf.extractall(dest)


def download_dataset(name: str) -> None:
    ensure_dirs()
    if name not in DATASETS:
        raise KeyError(f"Unknown downloadable dataset: {name}")
    out = RAW_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in DATASETS[name]["files"].items():
        dest = out / filename
        _download(url, dest)
        records.append({"file": filename, "url": url, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)})
    if name == "causalrivers":
        product_dir = out / "release"
        code_dir = out / "official_code"
        if not product_dir.exists() or not any(product_dir.iterdir()):
            _safe_extract(out / "product.zip", product_dir)
        if not code_dir.exists() or not any(code_dir.iterdir()):
            _safe_extract(out / "official_code.zip", code_dir)
    atomic_json(out / "download_manifest.json", {"dataset": name, "files": records})
