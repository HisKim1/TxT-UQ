import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_paths_follow_configured_roots_from_another_directory(tmp_path):
    data = tmp_path / "input data"
    run = tmp_path / "saved run"
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        TXT_DATA_ROOT=str(data),
        TXT_RUN_ROOT=str(run),
    )
    env.pop("TXT_CACHE_ROOT", None)
    env.pop("TXT_ERA5_DOY_CLIM", None)
    script = """
from pathlib import Path
from config import *
from txt_model import PAIRING_DIR as model_pairings
from artifacts import PAIRING_DIR as artifact_pairings
from cache import validate_cache_path
import sys
data, run = map(Path, sys.argv[1:])
assert era5_zarr(2020) == str(data / "era5_1p50/2020.zarr")
assert era5_ens_zarr(2020) == str(data / "era5_ens_1p50/2020.zarr")
assert ifs_zarr("t2m", 1) == str(data / "hun_t2m_ECMWF/1/1.zarr")
assert CHECKPOINT_DIR == str(run / "checkpoint/txt")
assert OUTPUT_DIR == str(run / "output")
assert LOGS_DIR == str(run / "logs")
assert CACHE_ROOT == str(run / "cache")
assert PAIRING_DIR == model_pairings == artifact_pairings == run / "pairings"
assert ERA5_DOY_CLIM_ZARR == str(data / "era5_1p50/era5_doy_clim_1991_2020_w31.zarr")
validate_cache_path(run / "cache/txt_reference_cache_mslp_hres_doy_full",
                   "txt_reference_cache_mslp_hres_doy_full")
"""
    subprocess.run(
        [sys.executable, "-c", script, str(data), str(run)],
        cwd=tmp_path,
        env=env,
        check=True,
    )
