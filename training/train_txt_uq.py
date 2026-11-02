import subprocess
import argparse
import json
import os
import shlex
import sys
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

_TRAINING_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _TRAINING_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))
from config import (
    CACHE_ROOT,
    CHECKPOINT_DIR,
    CONFIG,
    ERA5_ENS_DIR,
    LOGS_DIR,
    PAIRING_DIR,
    TARGET_LOSS_RELATIVE_TOLERANCE,
    ifs_zarr,
    load_normalizer,
    normalizer_provenance,
    ARTIFACT_SCHEMAS,
)
from time_coordinates import TIME_SCHEMA, canonicalize_ifs
from cache import (
    RAW_SOURCE_IDENTITY_SCHEMA,
    VALIDATED_CACHE_ENV,
    archive_cache_manifest,
    build_npy_file_manifest,
    cache_manifest_sha256,
    datetime_inventory_sha256,
    local_store_stat_identity,
    require_finite,
    validate_cache_path,
    validate_npy_file_manifest,
)
from txt_model import load_txt_checkpoint, resolve_txt_model_bank
from pairing import (
    N_PERTURBED_DATASETS,
    PAIRING_SCHEMA,
    PAIRING_SEED,
    generate_segmented_pairings,
    load_pairing_bank,
    mapping_sha256,
    pairing_manifest,
    split_pairing_dates,
)
from io_utils import atomic_write_json
from gpu_scheduler import GPU_TOKEN, run_fifo


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC_DIR)
    return env


def save_pairing_files(pairings, output_dir, segment_dates):
    output_dir = Path(output_dir)
    manifest = pairing_manifest(pairings, segment_dates)
    if (output_dir / "manifest.json").is_file():
        try:
            existing, _ = load_pairing_bank(output_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Existing pairing bank is invalid: {output_dir}. Remove it only after confirming no checkpoint depends on it."
            ) from exc
        existing_manifest = json.loads((output_dir / "manifest.json").read_text())
        if existing != pairings or existing_manifest != manifest:
            raise RuntimeError(
                f"Existing pairing bank or manifest differs from the live recipe: {output_dir}. Use a new checkpoint version."
            )
        return {
            index: str(output_dir / f"pairing_dataset{index:02d}.json")
            for index in range(N_PERTURBED_DATASETS)
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for dataset_index, mapping in pairings.items():
        path = output_dir / f"pairing_dataset{dataset_index:02d}.json"
        atomic_write_json(mapping, path)
        paths[dataset_index] = str(path)
    atomic_write_json(manifest, output_dir / "manifest.json")
    return paths


def pairing_date_segments(init_dates):
    return split_pairing_dates(
        init_dates,
        training_start=CONFIG["train_init_start"],
        training_end=CONFIG["train_init_end"],
        heldout_start=CONFIG["val_init_start"],
        heldout_end=CONFIG["val_init_end"],
        evaluation_start=CONFIG["test_init_start"],
    )


def _paired_manifest_matches(
    path: Path, var: str, normalizer_mode: str, raw_source_identity: dict
) -> bool:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return False
    recipe_matches = (
        data.get("schema") == ARTIFACT_SCHEMAS["paired_training_cache"]
        and data.get("var") == var
        and (data.get("year_range") == list(CONFIG["year_range"]))
        and (data.get("normalizer") == normalizer_mode)
        and (data.get("ifs_members") == list(range(1, 11)))
        and (data.get("eda_members") == list(range(10)))
        and (data.get("n_ifs") == 10)
        and (data.get("n_eda") == 10)
        and (data.get("input_channels") == 1)
        and (data.get("time_schema") == TIME_SCHEMA)
        and (
            data.get("normalizer_provenance") == normalizer_provenance(normalizer_mode)
        )
        and (data.get("init_start") == CONFIG["train_init_start"])
        and (data.get("init_end") == CONFIG["val_init_end"])
        and (data.get("raw_source_identity") == raw_source_identity)
    )
    if not recipe_matches or not isinstance(data.get("files"), dict):
        return False
    try:
        validate_npy_file_manifest(path, data["files"])
    except RuntimeError as exc:
        print(f"  Existing cache failed integrity validation: {exc}")
        return False
    return True


def _open_ifs_member(var: str, training_var: str, ens_num: int, first_date, last_date):
    ds = xr.open_zarr(
        ifs_zarr(var, ens_num), decode_timedelta=False, consolidated=False
    )
    ifs = ds[training_var].transpose("init_time", "lead_time", "latitude", "longitude")
    ifs = ifs.sortby("latitude", ascending=True).sortby("init_time")
    ifs = ifs.sel(init_time=slice(first_date, last_date))
    ifs = canonicalize_ifs(ifs, training_var)
    return (ds, ifs)


def _paired_data_identity(
    var: str, training_var: str, era5_ens_dir: str, first_date, last_date
) -> dict:
    reference_dates = None
    stores = {}
    for member in range(1, 11):
        ds, field = _open_ifs_member(var, training_var, member, first_date, last_date)
        dates = pd.DatetimeIndex(field.init_time.values)
        if reference_dates is None:
            reference_dates = dates
        elif not dates.equals(reference_dates):
            ds.close()
            raise RuntimeError(
                f"IFS member {member} initialization dates differ from member 1"
            )
        ds.close()
        stores[f"ifs_member_{member:02d}"] = ifs_zarr(var, member)
    for year in range(CONFIG["year_range"][0], CONFIG["year_range"][1] + 1):
        stores[f"era5_eda_{year}"] = Path(era5_ens_dir) / f"{year}.zarr"
    if reference_dates is None:
        raise RuntimeError("No IFS initialization dates found for Source-2 cache")
    return {
        "schema": RAW_SOURCE_IDENTITY_SCHEMA,
        "source_kind": "source2_ifs_eda",
        "init_time_count": len(reference_dates),
        "init_time_start": reference_dates[0].isoformat(),
        "init_time_end": reference_dates[-1].isoformat(),
        "init_time_sha256": datetime_inventory_sha256(reference_dates.values),
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


def create_training_cache(
    var,
    cache_dir,
    era5_ens_dir,
    normalizer_mode="doy_full",
    chunk_size=64,
    expected_cache_name=None,
):
    if expected_cache_name is None:
        raise ValueError("expected_cache_name is required for safe cache creation")
    cache_dir = validate_cache_path(cache_dir, expected_cache_name)
    manifest = cache_dir / "manifest.json"
    var_dict = {"mslp": "mean_sea_level_pressure", "t2m": "2m_temperature"}
    training_var = var_dict[var]
    year_lo, year_hi = CONFIG["year_range"]
    first_date = pd.to_datetime(CONFIG["train_init_start"])
    last_date = pd.to_datetime(CONFIG["val_init_end"])
    raw_source_identity = _paired_data_identity(
        var, training_var, era5_ens_dir, first_date, last_date
    )
    if _paired_manifest_matches(cache_dir, var, normalizer_mode, raw_source_identity):
        archive_cache_manifest(cache_dir, Path(CHECKPOINT_DIR) / "cache_manifests")
        print(f"  Cache already matches current split: {cache_dir}")
        return str(cache_dir)
    if cache_dir.exists():
        print(f"  Removing stale/incomplete cache at {cache_dir}")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Loading normalizer (mode={normalizer_mode})...")
    norm = load_normalizer(training_var, mode=normalizer_mode)
    ref_lat = ref_lon = None
    ref_init_times = None
    ifs_shape = None
    for ens_num in range(1, 11):
        print(f"  Caching IFS ensemble {ens_num}/10...")
        ds, ifs = _open_ifs_member(var, training_var, ens_num, first_date, last_date)
        if ref_lat is None:
            ref_lat = ifs.latitude.values
            ref_lon = ifs.longitude.values
            ref_init_times = pd.DatetimeIndex(ifs.init_time.values)
            np.save(cache_dir / "init_times.npy", ref_init_times.values)
            np.save(cache_dir / "latitude.npy", ref_lat)
            np.save(cache_dir / "longitude.npy", ref_lon)
        else:
            current_init_times = pd.DatetimeIndex(ifs.init_time.values)
            if not current_init_times.equals(ref_init_times):
                raise RuntimeError(
                    f"IFS member {ens_num} initialization dates differ from member 1"
                )
            if not (
                np.array_equal(ifs.latitude.values, ref_lat)
                and np.array_equal(ifs.longitude.values, ref_lon)
            ):
                raise RuntimeError(f"IFS member {ens_num} grid differs from member 1")
        ifs_shape = _write_normalized_ifs_npy(
            ifs, cache_dir / f"ifs_norm_{ens_num:02d}.npy", norm, chunk_size
        )
        nbytes = int(np.prod(ifs_shape)) * np.dtype(np.float32).itemsize
        print(f"    shape={ifs_shape}, size={nbytes / 1000000000.0:.1f} GB")
        try:
            ds.close()
        except Exception:
            pass
        del ifs, ds
    if ifs_shape is None:
        raise RuntimeError("No IFS members were written")
    era5_times_saved = False
    for eda in range(10):
        print(f"  Caching ERA5 EDA member {eda}/9...")
        parts = []
        for year in range(year_lo, year_hi + 1):
            p = xr.open_zarr(f"{era5_ens_dir}/{year}.zarr", consolidated=False)
            part = (
                p[training_var]
                .sel(number=eda)
                .transpose("time", "latitude", "longitude")
            )
            parts.append(part)
        era5 = (
            xr.concat(parts, dim="time")
            .sortby("time")
            .sortby("latitude", ascending=True)
        )
        if not pd.DatetimeIndex(era5.time.values).is_unique:
            raise RuntimeError(f"ERA5 EDA member {eda} contains duplicate dates")
        if not pd.DatetimeIndex(era5.time.values).is_monotonic_increasing:
            raise RuntimeError(f"ERA5 EDA member {eda} dates are not increasing")
        if not (
            np.allclose(era5.latitude.values, ref_lat, rtol=0.0, atol=1e-10)
            and np.allclose(era5.longitude.values, ref_lon, rtol=0.0, atol=1e-10)
        ):
            raise RuntimeError(f"ERA5 EDA member {eda} grid differs from IFS")
        era5_norm = norm.normalize_era5(era5).values.astype(np.float32)
        require_finite(era5_norm, f"normalized ERA5 EDA member {eda}")
        if era5_norm.shape[1:] != ifs_shape[2:]:
            raise RuntimeError(
                f"ERA5 EDA member {eda} spatial shape {era5_norm.shape[1:]} differs from IFS {ifs_shape[2:]}"
            )
        np.save(cache_dir / f"era5_eda_{eda:02d}.npy", era5_norm)
        if not era5_times_saved:
            np.save(cache_dir / "era5_times.npy", era5.time.values)
            era5_times_saved = True
        print(
            f"    shape={era5_norm.shape}, size={era5_norm.nbytes / 1000000000.0:.1f} GB"
        )
        del era5, era5_norm, parts
    source_identity_after = _paired_data_identity(
        var, training_var, era5_ens_dir, first_date, last_date
    )
    if source_identity_after != raw_source_identity:
        raise RuntimeError(
            "Raw IFS/ERA5 source stores changed while the Source-2 cache was built"
        )
    total_bytes = sum((f.stat().st_size for f in cache_dir.glob("*.npy")))
    manifest_data = {
        "schema": ARTIFACT_SCHEMAS["paired_training_cache"],
        "var": var,
        "year_range": list(CONFIG["year_range"]),
        "ifs_members": list(range(1, 11)),
        "eda_members": list(range(10)),
        "n_ifs": 10,
        "n_eda": 10,
        "total_gb": round(total_bytes / 1000000000.0, 1),
        "normalizer": normalizer_mode,
        "normalizer_provenance": normalizer_provenance(normalizer_mode),
        "input_channels": 1,
        "time_schema": TIME_SCHEMA,
        "valid_day_offsets": list(range(32)),
        "init_start": CONFIG["train_init_start"],
        "init_end": CONFIG["val_init_end"],
        "raw_source_identity": raw_source_identity,
        "files": build_npy_file_manifest(cache_dir),
    }
    with open(manifest, "w") as f:
        json.dump(manifest_data, f, indent=2)
    archive_cache_manifest(cache_dir, Path(CHECKPOINT_DIR) / "cache_manifests")
    print(f"  Cache complete: {cache_dir} ({manifest_data['total_gb']} GB)")
    return str(cache_dir)


def compute_target_loss(
    var,
    ckpt_version="txt",
    arch="sp_full",
    era5_ens_dir=None,
    gpu_id=None,
    cache_dir=None,
    normalizer_mode="doy_full",
    pairing_dir=None,
):
    print(f"Computing J0 for {var}...")
    script = str(_TRAINING_DIR / "compute_target_loss.py")
    reference_version = f"{ckpt_version}_reference"
    cmd = ["uv", "run", script, "--var", var, "--ckpt-version", reference_version]
    if not pairing_dir:
        raise ValueError("Source-2 J0 requires pairing_dir")
    cmd += ["--era5-ens-dir", era5_ens_dir, "--pairing-dir", str(pairing_dir)]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    env = _subprocess_env()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_dir = Path(LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"targetloss_{var}_{ckpt_version}_{normalizer_mode}.log"
    with open(log_path, "w", buffering=1) as log_fh:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env=env,
        )
    with open(log_path) as f:
        lines = f.read().strip().split("\n")
    try:
        target_loss = float(lines[-1].strip())
    except ValueError:
        print(f"ERROR: Could not parse J0 from output; see {log_path}")
        print("\n".join(lines[-20:]))
        sys.exit(1)
    print(f"J0 = {target_loss} (log={log_path})")
    return target_loss


def run_jobs(commands, gpus):
    total = len(commands)
    print(f"Queued {total} jobs on GPUs {gpus}...", flush=True)
    if not commands:
        print("All jobs completed successfully.", flush=True)
        return
    jobs_log_dir = Path(LOGS_DIR) / "uqtrain_jobs"
    jobs_log_dir.mkdir(parents=True, exist_ok=True)

    def _job_log_path(cmd):
        try:
            var = cmd[cmd.index("--var") + 1]
            ver = cmd[cmd.index("--ckpt-version") + 1]
            ens = cmd[cmd.index("--ensemble") + 1]
            wt = cmd[cmd.index("--weight") + 1]
            return jobs_log_dir / f"{var}_{ver}_{int(ens):02d}-{int(wt):02d}.log"
        except (ValueError, IndexError):
            return jobs_log_dir / f"job_{os.getpid()}.log"

    templates = []
    log_paths = []
    for command in commands:
        command = [*command, "--gpu", GPU_TOKEN]
        templates.append(shlex.join((str(part) for part in command)))
        log_paths.append(_job_log_path(command))
    env = _subprocess_env()
    env["PYTHONUNBUFFERED"] = "1"
    first = commands[0]
    group = f"{first[first.index('--var') + 1]}_{first[first.index('--ckpt-version') + 1]}_source2"
    return_code = run_fifo(
        templates,
        [str(gpu) for gpu in gpus],
        group,
        jobs_log_dir,
        log_paths=log_paths,
        cwd=_PROJECT_ROOT,
        env=env,
    )
    if return_code:
        raise RuntimeError(f"Source-2 training job failed; see {jobs_log_dir}")
    print("All jobs completed successfully.", flush=True)


def validate_target_loss_models(
    var,
    version,
    target_loss=None,
    *,
    normalizer_mode=None,
    relative_tolerance=TARGET_LOSS_RELATIVE_TOLERANCE,
    architecture=None,
    expected_cache_manifest_identity=None,
    expected_cache_raw_source_identity=None,
):
    ckpt_dir = Path(CHECKPOINT_DIR)
    expected_keys = {(eda, weight) for eda in range(10) for weight in range(10)}
    expected_provenance = (
        normalizer_provenance(normalizer_mode) if normalizer_mode else None
    )
    pairing_digests = None
    pairing_contract = None
    pairings, pairing_contract = load_pairing_bank(PAIRING_DIR / version)
    pairing_digests = {
        dataset_index: mapping_sha256(pairings[dataset_index])
        for dataset_index in range(10)
    }
    resolved_by_key = dict(
        resolve_txt_model_bank(
            ckpt_dir,
            variable=var,
            version=version,
            expected_model_count=100,
            expected_arch=architecture,
            expected_normalizer=normalizer_mode,
            expected_normalizer_provenance=expected_provenance,
            expected_cache_manifest_identity=expected_cache_manifest_identity,
            expected_pairing_digests=pairing_digests,
            expected_pairing_bank_sha256=(
                pairing_contract["pairing_bank_sha256"]
                if pairing_contract is not None
                else None
            ),
            expected_pairing_training_bank_sha256=(
                pairing_contract["segments"]["training"]["pairing_bank_sha256"]
                if pairing_contract is not None
                else None
            ),
            expected_pairing_testing_bank_sha256=(
                pairing_contract["segments"]["testing"]["pairing_bank_sha256"]
                if pairing_contract is not None
                else None
            ),
        )
    )
    ordered = []
    selected_checkpoints = []
    for key in sorted(expected_keys):
        checkpoint = resolved_by_key[key]
        sidecar = checkpoint.with_suffix(".json")
        if not sidecar.exists():
            raise RuntimeError(f"Missing sidecar for selected checkpoint: {checkpoint}")
        row = json.loads(sidecar.read_text())
        recorded_checkpoint = Path(row.get("checkpoint", ""))
        if not recorded_checkpoint.is_absolute():
            recorded_checkpoint = (checkpoint.parent / recorded_checkpoint).resolve()
        else:
            recorded_checkpoint = recorded_checkpoint.resolve()
        if recorded_checkpoint != checkpoint.resolve():
            raise RuntimeError(
                f"Sidecar/checkpoint mismatch: {sidecar} records {recorded_checkpoint}"
            )
        expected_metadata = {
            "var": var,
            "version": version,
            "ensemble": key[0],
            "weight": key[1],
            "time_schema": TIME_SCHEMA,
            "input_mode": "member",
        }
        if normalizer_mode is not None:
            expected_metadata["normalizer"] = normalizer_mode
        expected_metadata["tolerance_definition"] = "abs(J-J0)/J0 <= relative_tolerance"
        if architecture is not None:
            expected_metadata["architecture"] = architecture
        expected_metadata["source2"] = bool(True)
        expected_metadata.update(
            perturbed_dataset_index=key[0],
            weight_level=key[1],
            training_seed=key[1],
            weight_seed_schema=ARTIFACT_SCHEMAS["weight_seed"],
            input_channels=1,
            pairing_schema=PAIRING_SCHEMA,
            pairing_seed=PAIRING_SEED,
            pairing_bank_sha256=pairing_contract["pairing_bank_sha256"],
            pairing_training_bank_sha256=pairing_contract["segments"]["training"][
                "pairing_bank_sha256"
            ],
            pairing_testing_bank_sha256=pairing_contract["segments"]["testing"][
                "pairing_bank_sha256"
            ],
            train_init_start=CONFIG["train_init_start"],
            train_init_end=CONFIG["train_init_end"],
            val_init_start=CONFIG["val_init_start"],
            val_init_end=CONFIG["val_init_end"],
        )
        if expected_cache_manifest_identity is not None:
            expected_metadata["cache_manifest_identity"] = (
                expected_cache_manifest_identity
            )
        if expected_cache_raw_source_identity is not None:
            expected_metadata["cache_raw_source_identity"] = (
                expected_cache_raw_source_identity
            )
        mismatches = {
            name: (row.get(name), expected)
            for name, expected in expected_metadata.items()
            if row.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"Sidecar metadata mismatch in {sidecar}: {mismatches}")
        if (
            expected_provenance is not None
            and row.get("normalizer_provenance") != expected_provenance
        ):
            raise RuntimeError(f"Sidecar normalizer provenance mismatch: {sidecar}")
        pairing_path = PAIRING_DIR / version / f"pairing_dataset{key[0]:02d}.json"
        if not pairing_path.is_file():
            raise RuntimeError(
                f"Missing pairing used by checkpoint {checkpoint}: {pairing_path}"
            )
        pairing = json.loads(pairing_path.read_text())
        expected_mapping_digest = mapping_sha256(pairing)
        if row.get("pairing_mapping_sha256") != expected_mapping_digest:
            raise RuntimeError(f"Checkpoint pairing digest mismatch: {sidecar}")
        if architecture is not None and normalizer_mode is not None:
            expected_training_provenance = {
                "source2": True,
                "perturbed_dataset_index": key[0],
                "weight_level": key[1],
                "training_seed": key[1],
                "weight_seed_schema": ARTIFACT_SCHEMAS["weight_seed"],
                "pairing_schema": PAIRING_SCHEMA,
                "pairing_seed": PAIRING_SEED,
                "pairing_mapping_sha256": expected_mapping_digest,
                "pairing_bank_sha256": pairing_contract["pairing_bank_sha256"],
                "pairing_training_bank_sha256": pairing_contract["segments"][
                    "training"
                ]["pairing_bank_sha256"],
                "pairing_testing_bank_sha256": pairing_contract["segments"]["testing"][
                    "pairing_bank_sha256"
                ],
                "cache_manifest_identity": row.get("cache_manifest_identity"),
                "cache_raw_source_identity": row.get("cache_raw_source_identity"),
                "train_init_start": CONFIG["train_init_start"],
                "train_init_end": CONFIG["train_init_end"],
                "val_init_start": CONFIG["val_init_start"],
                "val_init_end": CONFIG["val_init_end"],
            }
            model = load_txt_checkpoint(
                checkpoint,
                expected_arch=architecture,
                expected_training_variable=var,
                expected_normalizer_provenance=expected_provenance,
                expected_training_provenance=expected_training_provenance,
                map_location="cpu",
            )
            del model
        if not np.isclose(
            float(row.get("relative_tolerance", np.nan)),
            float(relative_tolerance),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(f"Sidecar relative tolerance mismatch: {sidecar}")
        ordered.append(row)
        selected_checkpoints.append(str(checkpoint))
    recorded_targets = np.asarray(
        [row["target_loss"] for row in ordered], dtype=np.float64
    )
    if target_loss is None:
        target_loss = float(recorded_targets[0])
    if not np.allclose(recorded_targets, float(target_loss), rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Inconsistent J0 values in {var}_{version} sidecars: range={recorded_targets.min()}..{recorded_targets.max()}, expected={target_loss}"
        )
    achieved = np.asarray([row["achieved_loss"] for row in ordered], dtype=np.float64)
    effective_tolerance = np.asarray(
        [row["effective_tolerance"] for row in ordered], dtype=np.float64
    )
    expected_effective = float(target_loss) * float(relative_tolerance)
    if not np.allclose(effective_tolerance, expected_effective, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Effective tolerance mismatch: expected {expected_effective}, range={effective_tolerance.min()}..{effective_tolerance.max()}"
        )
    differences = achieved - float(target_loss)
    relative_differences = np.abs(differences) / float(target_loss)
    within = relative_differences <= float(relative_tolerance) + 1e-12
    if not np.all(within):
        raise RuntimeError(
            f"{int((~within).sum())} target-loss models are outside tolerance"
        )
    if not np.all(np.isfinite(achieved)) or np.any(achieved <= 0):
        raise RuntimeError(
            "Target-loss sidecars contain non-positive or non-finite losses"
        )
    audit = {
        "var": var,
        "version": version,
        "model_count": len(ordered),
        "target_loss": float(target_loss),
        "tolerance_definition": "abs(J-J0)/J0 <= relative_tolerance",
        "relative_tolerance": float(relative_tolerance),
        "achieved_loss_min": float(achieved.min()),
        "achieved_loss_max": float(achieved.max()),
        "achieved_loss_mean": float(achieved.mean()),
        "max_absolute_loss_difference": float(np.abs(differences).max()),
        "max_relative_loss_difference": float(relative_differences.max()),
        "all_within_tolerance": True,
        "selected_checkpoints": selected_checkpoints,
        "models": ordered,
    }
    audit_path = Path(LOGS_DIR) / f"targetloss_audit_{var}_{version}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    print(
        f"Target-loss audit: max |J_k-J0|={audit['max_absolute_loss_difference']:.8f}, max relative difference={audit['max_relative_loss_difference']:.6%}, log={audit_path}"
    )
    return audit


def build_weight_training_commands(
    var,
    ckpt_version,
    arch,
    target_loss,
    pairing_paths,
    era5_ens_dir,
    cache_dir=None,
    normalizer_mode="doy_full",
    target_relative_tolerance=TARGET_LOSS_RELATIVE_TOLERANCE,
    cells=None,
):
    script = str(_TRAINING_DIR / "train_txt.py")
    cmds = []
    requested_cells = (
        [(dataset_index, w) for dataset_index in range(10) for w in range(10)]
        if cells is None
        else sorted(cells)
    )
    for dataset_index, w in requested_cells:
        cmd = [
            "uv",
            "run",
            script,
            "--ensemble",
            str(dataset_index),
            "--weight",
            str(w),
            "--target_loss",
            str(target_loss),
            "--var",
            var,
            "--target-relative-tolerance",
            str(target_relative_tolerance),
            "--ckpt-version",
            ckpt_version,
            "--perturbed-dataset-index",
            str(dataset_index),
            "--era5-ens-dir",
            era5_ens_dir,
            "--pairing-file",
            pairing_paths[dataset_index],
        ]
        if cache_dir:
            cmd += ["--cache-dir", cache_dir]
        cmds.append(cmd)
    return cmds


def completed_weight_models(
    var,
    version,
    arch,
    normalizer_mode,
    target_loss,
    relative_tolerance,
    pairing_paths,
    expected_cache_manifest_identity,
    expected_cache_raw_source_identity,
):
    ckpt_dir = Path(CHECKPOINT_DIR)
    pairings, pairing_contract = load_pairing_bank(
        Path(next(iter(pairing_paths.values()))).parent
    )
    available = resolve_txt_model_bank(
        ckpt_dir,
        variable=var,
        version=version,
        expected_model_count=100,
        require_complete=False,
        expected_arch=arch,
        expected_normalizer=normalizer_mode,
        expected_normalizer_provenance=normalizer_provenance(normalizer_mode),
        expected_cache_manifest_identity=expected_cache_manifest_identity,
        expected_pairing_digests={
            index: mapping_sha256(pairings[index]) for index in pairing_paths
        },
        expected_pairing_bank_sha256=pairing_contract["pairing_bank_sha256"],
        expected_pairing_training_bank_sha256=pairing_contract["segments"]["training"][
            "pairing_bank_sha256"
        ],
        expected_pairing_testing_bank_sha256=pairing_contract["segments"]["testing"][
            "pairing_bank_sha256"
        ],
    )
    expected_provenance = normalizer_provenance(normalizer_mode)
    valid = set()
    rejected = []
    for key, checkpoint in available:
        try:
            row = json.loads(checkpoint.with_suffix(".json").read_text())
            pairing = json.loads(Path(pairing_paths[key[0]]).read_text())
            expected = {
                "architecture": arch,
                "normalizer": normalizer_mode,
                "normalizer_provenance": expected_provenance,
                "perturbed_dataset_index": key[0],
                "weight_seed_schema": ARTIFACT_SCHEMAS["weight_seed"],
                "training_seed": key[1],
                "input_channels": 1,
                "pairing_schema": PAIRING_SCHEMA,
                "pairing_seed": PAIRING_SEED,
                "pairing_mapping_sha256": mapping_sha256(pairing),
                "pairing_bank_sha256": pairing_contract["pairing_bank_sha256"],
                "pairing_training_bank_sha256": pairing_contract["segments"][
                    "training"
                ]["pairing_bank_sha256"],
                "pairing_testing_bank_sha256": pairing_contract["segments"]["testing"][
                    "pairing_bank_sha256"
                ],
                "cache_manifest_identity": expected_cache_manifest_identity,
                "cache_raw_source_identity": expected_cache_raw_source_identity,
                "train_init_start": CONFIG["train_init_start"],
                "train_init_end": CONFIG["train_init_end"],
                "val_init_start": CONFIG["val_init_start"],
                "val_init_end": CONFIG["val_init_end"],
            }
            mismatches = {
                name: (row.get(name), value)
                for name, value in expected.items()
                if row.get(name) != value
            }
            if mismatches:
                raise RuntimeError(f"metadata mismatch: {mismatches}")
            if not np.isclose(
                float(row.get("target_loss", np.nan)),
                float(target_loss),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError("J0 differs from the current run")
            if not np.isclose(
                float(row.get("relative_tolerance", np.nan)),
                float(relative_tolerance),
                rtol=0.0,
                atol=1e-15,
            ):
                raise RuntimeError("relative tolerance differs from the current run")
            expected_training_provenance = {
                "source2": True,
                "perturbed_dataset_index": key[0],
                "weight_level": key[1],
                "training_seed": key[1],
                "weight_seed_schema": ARTIFACT_SCHEMAS["weight_seed"],
                "pairing_schema": PAIRING_SCHEMA,
                "pairing_seed": PAIRING_SEED,
                "pairing_mapping_sha256": mapping_sha256(pairing),
                "pairing_bank_sha256": pairing_contract["pairing_bank_sha256"],
                "pairing_training_bank_sha256": pairing_contract["segments"][
                    "training"
                ]["pairing_bank_sha256"],
                "pairing_testing_bank_sha256": pairing_contract["segments"]["testing"][
                    "pairing_bank_sha256"
                ],
                "cache_manifest_identity": expected_cache_manifest_identity,
                "cache_raw_source_identity": expected_cache_raw_source_identity,
                "train_init_start": CONFIG["train_init_start"],
                "train_init_end": CONFIG["train_init_end"],
                "val_init_start": CONFIG["val_init_start"],
                "val_init_end": CONFIG["val_init_end"],
            }
            model = load_txt_checkpoint(
                checkpoint,
                expected_arch=arch,
                expected_training_variable=var,
                expected_normalizer_provenance=expected_provenance,
                expected_training_provenance=expected_training_provenance,
                map_location="cpu",
            )
            del model
            valid.add(key)
        except Exception as exc:
            rejected.append(f"{key}: {exc}")
    if rejected:
        print(
            f"  Rejected {len(rejected)} stale/incompatible completed cell(s): {rejected[:3]}"
        )
    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", type=str, default="mslp", choices=["mslp", "t2m"])
    parser.add_argument("--ckpt-version", type=str, default="txt")
    parser.add_argument(
        "--target-relative-tolerance",
        type=float,
        default=TARGET_LOSS_RELATIVE_TOLERANCE,
        help="Require |J-J0|/J0 to be no larger than this value.",
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate an existing 100-model target-loss bank and exit.",
    )
    parser.add_argument("--era5-ens-dir", type=str, default=ERA5_ENS_DIR)
    args = parser.parse_args()
    gpu_list = [int(g) for g in args.gpus.split(",")]
    if args.audit_only:
        validate_target_loss_models(
            args.var,
            args.ckpt_version,
            normalizer_mode="doy_full",
            relative_tolerance=args.target_relative_tolerance,
            architecture="sp_full",
        )
        raise SystemExit(0)
    cache_dir = None
    current_cache_manifest_identity = None
    current_cache_raw_source_identity = None
    cache_root = os.environ.get("TXT_CACHE_ROOT", CACHE_ROOT)
    cache_dir = f"{cache_root}/txt_cache_{args.var}_{args.ckpt_version}_{'doy_full'}"
    print("=" * 60)
    print(f"Creating data cache at {cache_dir} (normalizer={'doy_full'})...")
    print("=" * 60)
    expected_cache_name = f"txt_cache_{args.var}_{args.ckpt_version}_{'doy_full'}"
    create_training_cache(
        args.var,
        cache_dir,
        args.era5_ens_dir,
        normalizer_mode="doy_full",
        expected_cache_name=expected_cache_name,
    )
    current_cache_manifest_identity = cache_manifest_sha256(cache_dir)
    current_cache_raw_source_identity = json.loads(
        (Path(cache_dir) / "manifest.json").read_text()
    )["raw_source_identity"]
    os.environ[VALIDATED_CACHE_ENV] = current_cache_manifest_identity
    pairing_paths = None
    pairing_dir = None
    init_dates = pd.DatetimeIndex(
        np.load(Path(cache_dir) / "init_times.npy", allow_pickle=True)
    )
    print(
        f"Init dates: {init_dates[0].date()} → {init_dates[-1].date()} ({len(init_dates)} dates)"
    )
    segment_dates = pairing_date_segments(init_dates)
    pairings, segment_dates = generate_segmented_pairings(segment_dates)
    pairing_dir = PAIRING_DIR / args.ckpt_version
    pairing_paths = save_pairing_files(pairings, pairing_dir, segment_dates)
    sample_date = segment_dates["training"][0].strftime("%Y%m%d")
    sample_pairs = [pairings[index][sample_date] for index in range(10)]
    print(
        f"Pairings saved to {pairing_dir}; independent Latin blocks: training={len(segment_dates['training'])}, buffer(unused)={len(segment_dates['buffer'])}, testing={len(segment_dates['testing'])}. First matching {sample_date}: {sample_pairs}"
    )
    target_loss = compute_target_loss(
        args.var,
        ckpt_version=args.ckpt_version,
        arch="sp_full",
        era5_ens_dir=args.era5_ens_dir,
        gpu_id=gpu_list[0],
        cache_dir=cache_dir,
        normalizer_mode="doy_full",
        pairing_dir=pairing_dir,
    )
    print("=" * 40)
    print(f"J0 = {target_loss}")
    print("=" * 40)
    expected_cells = {
        (dataset_index, weight) for dataset_index in range(10) for weight in range(10)
    }
    completed_cells = completed_weight_models(
        args.var,
        args.ckpt_version,
        "sp_full",
        "doy_full",
        target_loss,
        args.target_relative_tolerance,
        pairing_paths,
        current_cache_manifest_identity,
        current_cache_raw_source_identity,
    )
    missing_cells = expected_cells - completed_cells
    print(
        f"Restart audit: {len(completed_cells)} valid cells, {len(missing_cells)} cells to train"
    )
    commands = build_weight_training_commands(
        args.var,
        args.ckpt_version,
        "sp_full",
        target_loss,
        pairing_paths,
        args.era5_ens_dir,
        cache_dir=cache_dir,
        normalizer_mode="doy_full",
        target_relative_tolerance=args.target_relative_tolerance,
        cells=missing_cells,
    )
    run_jobs(commands, gpu_list)
    validate_target_loss_models(
        args.var,
        args.ckpt_version,
        target_loss,
        normalizer_mode="doy_full",
        relative_tolerance=args.target_relative_tolerance,
        architecture="sp_full",
        expected_cache_manifest_identity=current_cache_manifest_identity,
        expected_cache_raw_source_identity=current_cache_raw_source_identity,
    )
    if cache_dir and os.path.exists(cache_dir):
        expected_cache_name = f"txt_cache_{args.var}_{args.ckpt_version}_{'doy_full'}"
        safe_cache_dir = validate_cache_path(cache_dir, expected_cache_name)
        shutil.rmtree(safe_cache_dir)
        print(f"Cache cleaned up: {cache_dir}")
