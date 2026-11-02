from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import pandas as pd
import xarray as xr

_TRAINING_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _TRAINING_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))
from config import (
    CHECKPOINT_DIR,
    CONFIG,
    era5_zarr,
    ifs_zarr,
    load_normalizer,
    normalizer_provenance,
    ARTIFACT_SCHEMAS,
)
from time_coordinates import TIME_SCHEMA, canonicalize_ifs
from cache import (
    RAW_SOURCE_IDENTITY_SCHEMA,
    archive_cache_manifest,
    build_npy_file_manifest,
    datetime_inventory_sha256,
    local_store_stat_identity,
    require_finite,
    validate_cache_path,
    validate_npy_file_manifest,
)

VAR_DICT = {"mslp": "mean_sea_level_pressure", "t2m": "2m_temperature"}


def _manifest_matches(
    path: Path, var: str, normalizer: str, raw_source_identity: dict
) -> bool:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return False
    recipe_matches = (
        data.get("schema") == ARTIFACT_SCHEMAS["reference_cache"]
        and data.get("var") == var
        and (data.get("year_range") == list(CONFIG["year_range"]))
        and (data.get("normalizer") == normalizer)
        and (data.get("ifs_members") == [0])
        and (data.get("include_hres") is True)
        and (data.get("input_channels") == 1)
        and (data.get("time_schema") == TIME_SCHEMA)
        and (data.get("normalizer_provenance") == normalizer_provenance(normalizer))
        and (data.get("init_start") == CONFIG["train_init_start"])
        and (data.get("init_end") == CONFIG["val_init_end"])
        and (data.get("raw_source_identity") == raw_source_identity)
    )
    if not recipe_matches or not isinstance(data.get("files"), dict):
        return False
    try:
        validate_npy_file_manifest(path, data["files"])
    except RuntimeError as exc:
        print(f"  Existing cache failed integrity validation: {exc}", flush=True)
        return False
    return True


def _open_ifs_member(var: str, training_var: str, ens_num: int, first_date, last_date):
    ds = xr.open_zarr(
        ifs_zarr(var, ens_num), decode_timedelta=False, consolidated=False
    )
    ifs = ds[training_var]
    ifs = ifs.sortby("latitude", ascending=True).sortby("init_time")
    ifs = ifs.sel(init_time=slice(first_date, last_date))
    ifs = canonicalize_ifs(ifs, training_var)
    return (ds, ifs)


def _reference_data_identity(
    var: str, training_var: str, year_lo: int, year_hi: int, first_date, last_date
) -> dict:
    ds, field = _open_ifs_member(var, training_var, 0, first_date, last_date)
    dates = pd.DatetimeIndex(field.init_time.values)
    ds.close()
    if len(dates) == 0:
        raise RuntimeError("No IFS initialization dates found for control cache")
    stores = {"ifs_member_00": ifs_zarr(var, 0)}
    stores.update(
        {f"era5_hres_{year}": era5_zarr(year) for year in range(year_lo, year_hi + 1)}
    )
    return {
        "schema": RAW_SOURCE_IDENTITY_SCHEMA,
        "source_kind": "control_ifs_hres",
        "init_time_count": len(dates),
        "init_time_start": dates[0].isoformat(),
        "init_time_end": dates[-1].isoformat(),
        "init_time_sha256": datetime_inventory_sha256(dates.values),
        "store_inventory": local_store_stat_identity(stores),
    }


def _write_normalized_ifs_npy(ifs, out_path: Path, norm, chunk_size: int):
    shape = tuple((int(s) for s in ifs.shape))
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=shape)
    n_init = shape[0]
    for start in range(0, n_init, chunk_size):
        stop = min(start + chunk_size, n_init)
        block = ifs.isel(init_time=slice(start, stop))
        normalized = norm.normalize_ifs(block).values.astype(np.float32, copy=False)
        require_finite(
            normalized, f"normalized IFS block for {out_path.name} [{start}:{stop}]"
        )
        out[start:stop] = normalized
        out.flush()
    del out
    return shape


def _load_era5_hres(training_var: str, year_lo: int, year_hi: int, norm):
    parts = []
    for year in range(year_lo, year_hi + 1):
        ds = xr.open_zarr(era5_zarr(year), consolidated=False)
        if "number" in ds.coords:
            ds = ds.drop_vars("number")
        parts.append(ds[training_var].transpose("time", "latitude", "longitude"))
    era5 = (
        xr.concat(parts, dim="time").sortby("time").sortby("latitude", ascending=True)
    )
    times = pd.DatetimeIndex(era5.time.values)
    if not times.is_unique or not times.is_monotonic_increasing:
        raise RuntimeError("ERA5 HRES dates must be unique and increasing")
    normalized = norm.normalize_era5(era5).values.astype(np.float32)
    require_finite(normalized, "normalized ERA5 HRES")
    return (era5, normalized)


def create_reference_cache(
    var: str,
    cache_dir: Path,
    normalizer: str,
    force: bool = False,
    chunk_size: int = 64,
) -> str:
    expected_basename = f"txt_reference_cache_{var}_hres_{normalizer}"
    cache_dir = validate_cache_path(cache_dir, expected_basename)
    manifest = cache_dir / "manifest.json"
    training_var = VAR_DICT[var]
    year_lo, year_hi = CONFIG["year_range"]
    first_date = pd.to_datetime(CONFIG["train_init_start"])
    last_date = pd.to_datetime(CONFIG["val_init_end"])
    raw_source_identity = _reference_data_identity(
        var, training_var, year_lo, year_hi, first_date, last_date
    )
    if force and cache_dir.exists():
        print(f"  Removing cache because --force was set: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir)
    elif _manifest_matches(cache_dir, var, normalizer, raw_source_identity):
        archive_cache_manifest(cache_dir, Path(CHECKPOINT_DIR) / "cache_manifests")
        print(f"  Cache already matches current split: {cache_dir}", flush=True)
        return str(cache_dir)
    elif cache_dir.exists():
        print(f"  Removing stale/incomplete cache: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  Building control cache for {var}: {year_lo}-{year_hi}, normalizer={normalizer}",
        flush=True,
    )
    norm = load_normalizer(training_var, mode=normalizer)
    ref_lat = ref_lon = None
    init_times = None
    ifs_shape = None
    for ens_num in (0,):
        print("  Caching IFS control member 00...", flush=True)
        ds, ifs = _open_ifs_member(var, training_var, ens_num, first_date, last_date)
        if ref_lat is None:
            ref_lat = ifs.latitude.values
            ref_lon = ifs.longitude.values
            init_times = pd.DatetimeIndex(ifs.init_time.values)
            np.save(cache_dir / "init_times.npy", init_times.values)
            np.save(cache_dir / "latitude.npy", ref_lat)
            np.save(cache_dir / "longitude.npy", ref_lon)
        else:
            current_init_times = pd.DatetimeIndex(ifs.init_time.values)
            if not current_init_times.equals(init_times):
                raise RuntimeError(
                    f"IFS member {ens_num} initialization dates differ from member 0"
                )
            if not (
                np.array_equal(ifs.latitude.values, ref_lat)
                and np.array_equal(ifs.longitude.values, ref_lon)
            ):
                raise RuntimeError(f"IFS member {ens_num} grid differs from member 0")
        ifs_shape = _write_normalized_ifs_npy(
            ifs, cache_dir / f"ifs_norm_{ens_num:02d}.npy", norm, chunk_size
        )
        nbytes = int(np.prod(ifs_shape)) * np.dtype(np.float32).itemsize
        print(f"    shape={ifs_shape}, size={nbytes / 1000000000.0:.1f} GB", flush=True)
        try:
            ds.close()
        except Exception:
            pass
        del ifs, ds
    if ifs_shape is None:
        raise RuntimeError("No IFS members were written")
    print("  Caching ERA5 HRES target...", flush=True)
    era5, era5_norm = _load_era5_hres(training_var, year_lo, year_hi, norm)
    if not (
        np.allclose(era5.latitude.values, ref_lat, rtol=0.0, atol=1e-10)
        and np.allclose(era5.longitude.values, ref_lon, rtol=0.0, atol=1e-10)
    ):
        raise RuntimeError("ERA5 HRES grid differs from IFS control grid")
    if era5_norm.shape[1:] != ifs_shape[2:]:
        raise RuntimeError(
            f"ERA5 HRES spatial shape {era5_norm.shape[1:]} differs from IFS {ifs_shape[2:]}"
        )
    np.save(cache_dir / "era5_hres.npy", era5_norm)
    np.save(cache_dir / "era5_times.npy", era5.time.values)
    print(
        f"    shape={era5_norm.shape}, size={era5_norm.nbytes / 1000000000.0:.1f} GB",
        flush=True,
    )
    del era5, era5_norm
    source_identity_after = _reference_data_identity(
        var, training_var, year_lo, year_hi, first_date, last_date
    )
    if source_identity_after != raw_source_identity:
        raise RuntimeError(
            "Raw IFS/ERA5 source stores changed while the control cache was built"
        )
    total_bytes = sum((f.stat().st_size for f in cache_dir.glob("*.npy")))
    manifest_data = {
        "schema": ARTIFACT_SCHEMAS["reference_cache"],
        "var": var,
        "year_range": list(CONFIG["year_range"]),
        "ifs_members": [0],
        "n_ifs": 1,
        "n_eda": 0,
        "include_hres": True,
        "total_gb": round(total_bytes / 1000000000.0, 1),
        "normalizer": normalizer,
        "normalizer_provenance": normalizer_provenance(normalizer),
        "input_channels": 1,
        "time_schema": TIME_SCHEMA,
        "valid_day_offsets": list(range(32)),
        "init_start": CONFIG["train_init_start"],
        "init_end": CONFIG["val_init_end"],
        "raw_source_identity": raw_source_identity,
        "files": build_npy_file_manifest(cache_dir),
    }
    manifest.write_text(json.dumps(manifest_data, indent=2))
    archive_cache_manifest(cache_dir, Path(CHECKPOINT_DIR) / "cache_manifests")
    print(f"  Cache complete: {cache_dir} ({manifest_data['total_gb']} GB)", flush=True)
    return str(cache_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", choices=["mslp", "t2m"], required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    create_reference_cache(
        args.var,
        Path(args.cache_dir),
        "doy_full",
        force=args.force,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
