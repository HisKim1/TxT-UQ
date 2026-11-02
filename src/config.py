import os
from pathlib import Path

ARTIFACT_SCHEMAS = {
    "loss": "publication1_lat_mse_valid_days_18_31_v1",
    "weight_seed": "publication1_weight_index_v1",
    "forecast_time": "publication1_valid_day_v1",
    "pairing": "publication1_block_latin_deranged_pairing_v3",
    "raw_data": "publication1_raw_source_identity_v1",
    "store_inventory": "publication1_local_store_stat_inventory_v1",
    "paired_training_cache": "publication1_source2_cache_v2",
    "reference_cache": "publication1_control_cache_v2",
    "cache_archive": "publication1_cache_manifest_archive_v1",
    "weight_model_completion": "publication1_targetloss_completion_v2",
    "reference_completion": "publication1_control_completion_v2",
}
TARGET_LOSS_RELATIVE_TOLERANCE = 0.01
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("TXT_DATA_ROOT", str(PROJECT_ROOT / "data"))
RUN_ROOT = Path(os.environ.get("TXT_RUN_ROOT", str(PROJECT_ROOT / "runs")))
CACHE_ROOT = os.environ.get("TXT_CACHE_ROOT", str(RUN_ROOT / "cache"))
LOGS_DIR = str(RUN_ROOT / "logs")
PAIRING_DIR = RUN_ROOT / "pairings"
IFS_PATH_TEMPLATE = f"{DATA_ROOT}/hun_{{var}}_ECMWF/{{ensemble}}/{{ensemble}}.zarr"
ERA5_DIR = f"{DATA_ROOT}/era5_1p50"
ERA5_ENS_DIR = f"{DATA_ROOT}/era5_ens_1p50"
OUTPUT_DIR = str(RUN_ROOT / "output")
CHECKPOINT_DIR = str(RUN_ROOT / "checkpoint" / "txt")
ERA5_DOY_CLIM_ZARR = os.environ.get(
    "TXT_ERA5_DOY_CLIM", f"{ERA5_DIR}/era5_doy_clim_1991_2020_w31.zarr"
)


def era5_zarr(year: int) -> str:
    return f"{ERA5_DIR}/{year}.zarr"


def era5_ens_zarr(year: int) -> str:
    return f"{ERA5_ENS_DIR}/{year}.zarr"


def ifs_zarr(var: str, ensemble: int) -> str:
    return IFS_PATH_TEMPLATE.format(var=var, ensemble=ensemble)


def era5_doy_clim(var_training: str):
    import xarray as xr

    ds = xr.open_zarr(ERA5_DOY_CLIM_ZARR)
    expected = {
        "period": "1991-2020",
        "n_years": 30,
        "smoothing_half_width_days": 15,
        "smoothing_window_days": 31,
        "std_ddof": 0,
    }
    mismatches = {
        key: (ds.attrs.get(key), value)
        for key, value in expected.items()
        if ds.attrs.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"DOY climatology provenance mismatch at {ERA5_DOY_CLIM_ZARR}: {mismatches}"
        )
    data_sha256 = ds.attrs.get("data_sha256")
    if not isinstance(data_sha256, str) or len(data_sha256) != 64:
        raise RuntimeError(
            f"DOY climatology is missing a valid data_sha256: {ERA5_DOY_CLIM_ZARR}"
        )
    mean = ds[f"{var_training}_mean"].sortby("latitude", ascending=True).load()
    std = ds[f"{var_training}_std"].sortby("latitude", ascending=True).load()
    return (mean, std)


def climatology_content_sha256(dataset) -> str:
    import hashlib
    import numpy as np

    digest = hashlib.sha256()
    names = ["dayofyear", "latitude", "longitude", *sorted(dataset.data_vars)]
    for name in names:
        values = np.ascontiguousarray(dataset[name].values)
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def normalizer_provenance(mode: str) -> dict:
    if mode != "doy_full":
        raise ValueError("Only doy_full normalization is published")
    import xarray as xr

    with xr.open_zarr(ERA5_DOY_CLIM_ZARR) as ds:
        keys = (
            "period",
            "n_years",
            "calendar_mapping",
            "smoothing_half_width_days",
            "smoothing_window_days",
            "statistic",
            "std_ddof",
            "data_sha256",
        )
        attrs = {key: ds.attrs.get(key) for key in keys}
        variables = sorted(ds.data_vars)
    expected_recipe = {
        "period": "1991-2020",
        "n_years": 30,
        "smoothing_half_width_days": 15,
        "smoothing_window_days": 31,
        "std_ddof": 0,
    }
    mismatches = {
        key: (attrs.get(key), expected)
        for key, expected in expected_recipe.items()
        if attrs.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"DOY normalizer scientific identity mismatch: {mismatches}")
    if not isinstance(attrs["data_sha256"], str) or len(attrs["data_sha256"]) != 64:
        raise RuntimeError("DOY normalizer identity lacks a valid data_sha256")
    return {"mode": mode, "variables": variables, **attrs}


def load_normalizer(var_training: str, mode: str = "doy_full"):
    from datasets import DayOfYearNormalizer

    if mode != "doy_full":
        raise ValueError("Only doy_full normalization is published")
    mu, sig = era5_doy_clim(var_training)
    return DayOfYearNormalizer(mu, sig, sig_min_clip=None)


CONFIG = {
    "n_lon": 240,
    "n_lat": 121,
    "num_workers": 8,
    "prefetch_factor": 2,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "train_init_start": "2005-01-01",
    "train_init_end": "2020-12-31",
    "val_init_start": "2021-02-01",
    "val_init_end": "2024-01-30",
    "test_init_start": "2024-03-01",
    "test_init_end": "2026-02-28",
    "year_range": (2005, 2024),
    "val_years": (2021, 2024),
    "residual_years": (2021, 2024),
    "test_years": (2024, 2026),
    "load_checkpoint": False,
}


def tranche_bounds():
    import pandas as pd

    c = CONFIG
    return tuple(
        (
            pd.to_datetime(c[k])
            for k in (
                "train_init_start",
                "train_init_end",
                "val_init_start",
                "val_init_end",
                "test_init_start",
                "test_init_end",
            )
        )
    )
