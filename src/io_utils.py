from __future__ import annotations
import os
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import uuid
import xarray as xr


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_exact_path(path: Path) -> None:
    if path.is_dir() and (not path.is_symlink()):
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def atomic_write_zarr(
    dataset: xr.Dataset,
    path: str | os.PathLike,
    *,
    encoding: dict | None = None,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    if destination.suffix != ".zarr":
        raise ValueError(f"Atomic Zarr destination must end in .zarr: {destination}")
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Zarr parent directory does not exist: {parent}")
    if destination.exists() and (not overwrite):
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=parent))
    backup = parent / f".{destination.name}.backup.{uuid.uuid4().hex}"
    try:
        dataset.to_zarr(temporary, mode="w", encoding=encoding or {})
        with xr.open_zarr(temporary) as written:
            if (
                dict(written.sizes) != dict(dataset.sizes)
                or set(written.data_vars) != set(dataset.data_vars)
                or set(written.coords) != set(dataset.coords)
                or (dict(written.attrs) != dict(dataset.attrs))
            ):
                raise RuntimeError(
                    f"Persisted Zarr metadata differs from source dataset: {destination}"
                )
        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup.exists() and (not destination.exists()):
                os.replace(backup, destination)
            raise
        if backup.exists():
            _remove_exact_path(backup)
    finally:
        if temporary.exists():
            _remove_exact_path(temporary)


def atomic_write_json(payload: dict, path: str | os.PathLike) -> None:
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"JSON parent directory does not exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
