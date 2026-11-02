from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from config import CACHE_ROOT, ARTIFACT_SCHEMAS
from io_utils import atomic_write_json, file_sha256

VALIDATED_CACHE_ENV = "TXT_CACHE_VALIDATED_MANIFEST_SHA256"
RAW_SOURCE_IDENTITY_SCHEMA = ARTIFACT_SCHEMAS["raw_data"]
LOCAL_STORE_STAT_SCHEMA = ARTIFACT_SCHEMAS["store_inventory"]


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )


def datetime_inventory_sha256(values) -> str:
    dates = np.asarray(values).astype("datetime64[ns]")
    if dates.ndim != 1 or np.isnat(dates).any():
        raise ValueError("Datetime inventory must be one-dimensional and finite")
    integers = dates.astype(np.int64)
    if len(np.unique(integers)) != len(integers):
        raise ValueError("Datetime inventory contains duplicates")
    if len(integers) > 1 and np.any(np.diff(integers) <= 0):
        raise ValueError("Datetime inventory must be strictly increasing")
    payload = {"count": len(integers), "nanoseconds_since_epoch": integers.tolist()}
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def local_store_stat_identity(stores: dict[str, str | Path]) -> dict:
    records = []
    for label in sorted(stores):
        root = Path(stores[label])
        if not root.is_dir():
            raise RuntimeError(f"Raw source store is missing: {root}")
        entries = []
        total_bytes = 0
        for path in sorted((value for value in root.rglob("*") if value.is_file())):
            stat = path.stat()
            total_bytes += stat.st_size
            entries.append(
                [path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns]
            )
        if not entries:
            raise RuntimeError(f"Raw source store contains no files: {root}")
        digest = hashlib.sha256(
            json.dumps(entries, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "label": label,
                "file_count": len(entries),
                "total_bytes": total_bytes,
                "stat_sha256": digest,
            }
        )
    return {"schema": LOCAL_STORE_STAT_SCHEMA, "stores": records}


def validate_raw_source_identity_record(
    record: dict, *, expected_source_kind: str | None = None
) -> None:
    if not isinstance(record, dict):
        raise RuntimeError("Cache manifest has no raw_source_identity record")
    if record.get("schema") != RAW_SOURCE_IDENTITY_SCHEMA:
        raise RuntimeError(
            f"Cache raw_source_identity schema mismatch: {record.get('schema')!r}"
        )
    source_kind = record.get("source_kind")
    if expected_source_kind is not None and source_kind != expected_source_kind:
        raise RuntimeError(
            f"Cache raw source kind mismatch: stored={source_kind!r}, expected={expected_source_kind!r}"
        )
    count = record.get("init_time_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeError("Cache raw init_time_count must be a positive integer")
    if not _is_sha256(record.get("init_time_sha256")):
        raise RuntimeError("Cache raw init_time_sha256 is not a SHA-256 digest")
    start = record.get("init_time_start")
    end = record.get("init_time_end")
    try:
        start_ns = np.datetime64(start, "ns")
        end_ns = np.datetime64(end, "ns")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Cache raw init-time bounds are invalid") from exc
    if np.isnat(start_ns) or np.isnat(end_ns) or start_ns > end_ns:
        raise RuntimeError("Cache raw init-time bounds are invalid")
    inventory = record.get("store_inventory")
    if not isinstance(inventory, dict):
        raise RuntimeError("Cache raw source has no store inventory")
    if inventory.get("schema") != LOCAL_STORE_STAT_SCHEMA:
        raise RuntimeError(
            f"Cache local-store inventory schema mismatch: {inventory.get('schema')!r}"
        )
    stores = inventory.get("stores")
    if not isinstance(stores, list) or not stores:
        raise RuntimeError("Cache local-store inventory must be non-empty")
    labels = []
    for item in stores:
        if not isinstance(item, dict):
            raise RuntimeError("Cache local-store inventory entry is invalid")
        label = item.get("label")
        file_count = item.get("file_count")
        total_bytes = item.get("total_bytes")
        if not isinstance(label, str) or not label:
            raise RuntimeError("Cache local-store inventory label is invalid")
        if (
            not isinstance(file_count, int)
            or isinstance(file_count, bool)
            or file_count <= 0
        ):
            raise RuntimeError(f"Cache local-store file count is invalid for {label!r}")
        if (
            not isinstance(total_bytes, int)
            or isinstance(total_bytes, bool)
            or total_bytes <= 0
        ):
            raise RuntimeError(f"Cache local-store byte count is invalid for {label!r}")
        if not _is_sha256(item.get("stat_sha256")):
            raise RuntimeError(
                f"Cache local-store stat digest is invalid for {label!r}"
            )
        labels.append(label)
    if labels != sorted(set(labels)):
        raise RuntimeError("Cache local-store labels must be unique and sorted")


def _array_all_finite(array: np.ndarray, *, chunk_size: int = 64) -> bool:
    if array.ndim == 0:
        blocks = (array[None],)
    else:
        blocks = (
            array[start : start + chunk_size]
            for start in range(0, array.shape[0], chunk_size)
        )
    for block in blocks:
        if np.issubdtype(block.dtype, np.datetime64):
            if np.isnat(block).any():
                return False
        elif np.issubdtype(block.dtype, np.number):
            if not np.isfinite(block).all():
                return False
    return True


def require_finite(array, label: str) -> None:
    values = np.asarray(array)
    finite = np.isfinite(values)
    if finite.all():
        return
    bad = int(values.size - finite.sum())
    raise RuntimeError(f"{label} contains {bad} NaN/Inf value(s)")


def cache_manifest_sha256(cache_dir: str | Path) -> str:
    manifest_path = Path(cache_dir) / "manifest.json"
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def archive_cache_manifest(cache_dir: str | Path, archive_root: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_raw_source_identity_record(manifest.get("raw_source_identity"))
    digest = cache_manifest_sha256(cache_dir)
    payload = {
        "schema": ARTIFACT_SCHEMAS["cache_archive"],
        "cache_manifest_sha256": digest,
        "manifest": manifest,
    }
    destination_root = Path(archive_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{digest}.json"
    if destination.is_file():
        existing = json.loads(destination.read_text())
        if existing != payload:
            raise RuntimeError(
                f"Hash-addressed cache manifest archive is inconsistent: {destination}"
            )
        return destination
    atomic_write_json(payload, destination)
    return destination


def validate_cache_path(path: str | Path, expected_basename: str) -> Path:
    cache_root = Path(os.environ.get("TXT_CACHE_ROOT", CACHE_ROOT)).resolve()
    candidate = Path(path).resolve(strict=False)
    if candidate.parent != cache_root or candidate.name != expected_basename:
        raise ValueError(
            f"Unsafe cache path {candidate}; expected {cache_root / expected_basename}"
        )
    return candidate


def build_npy_file_manifest(cache_dir: str | Path) -> dict[str, dict]:
    cache_dir = Path(cache_dir)
    records = {}
    for path in sorted(cache_dir.glob("*.npy")):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        records[path.name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "file_size": path.stat().st_size,
            "sha256": file_sha256(path),
            "all_finite": _array_all_finite(array),
        }
        if not records[path.name]["all_finite"]:
            raise RuntimeError(f"Cache array contains NaN/Inf/NaT: {path}")
        del array
    if not records:
        raise RuntimeError(f"Cache contains no NPY files: {cache_dir}")
    return records


def validate_npy_file_manifest(
    cache_dir: str | Path, records: dict, *, verify_sha256: bool = True
) -> None:
    cache_dir = Path(cache_dir)
    actual_names = {path.name for path in cache_dir.glob("*.npy")}
    expected_names = set(records)
    if actual_names != expected_names:
        raise RuntimeError(
            f"Cache file inventory mismatch: missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )
    mismatches = {}
    for name, expected in records.items():
        path = cache_dir / name
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            actual = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "file_size": path.stat().st_size,
                "all_finite": expected.get("all_finite"),
            }
            if verify_sha256:
                actual["sha256"] = file_sha256(path)
            del array
        except (OSError, ValueError) as exc:
            mismatches[name] = str(exc)
            continue
        expected_comparison = {
            "shape": expected.get("shape"),
            "dtype": expected.get("dtype"),
            "file_size": expected.get("file_size"),
            "all_finite": expected.get("all_finite"),
        }
        if verify_sha256:
            expected_comparison["sha256"] = expected.get("sha256")
        if expected.get("all_finite") is not True or actual != expected_comparison:
            mismatches[name] = {"actual": actual, "expected": expected_comparison}
    if mismatches:
        raise RuntimeError(f"Cache array integrity mismatch: {mismatches}")


def validate_cache_for_consumer(cache_dir: str | Path, manifest: dict) -> str:
    cache_dir = Path(cache_dir)
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise RuntimeError(f"Cache manifest has no file inventory: {cache_dir}")
    validate_raw_source_identity_record(manifest.get("raw_source_identity"))
    manifest_digest = cache_manifest_sha256(cache_dir)
    parent_digest = os.environ.get(VALIDATED_CACHE_ENV)
    validate_npy_file_manifest(
        cache_dir, records, verify_sha256=parent_digest != manifest_digest
    )
    return manifest_digest
