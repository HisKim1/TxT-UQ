import torch
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from config import (
    CHECKPOINT_DIR,
    CONFIG,
    OUTPUT_DIR,
    PAIRING_DIR,
    ifs_zarr,
    load_normalizer,
    normalizer_provenance,
)
from txt_model import (
    checkpoint_bank_identity,
    load_txt_checkpoint,
    resolve_txt_model_bank,
)
from datasets import ForecastInputDataset
from artifacts import (
    ANALOG_DISTANCE_SPACE,
    SAMPLE_PAYLOAD_SHA256_ATTR,
    sample_payload_sha256,
    validate_prediction_dataset,
    validate_sample_dataset,
)
from time_coordinates import (
    TIME_SCHEMA,
    canonicalize_ifs,
    expected_thursday_initializations,
    valid_dates,
    week_index_array,
)
from pairing import (
    PairedMemberArray,
    PAIRING_SCHEMA,
    PAIRING_SEED,
    date_mapping_sha256,
    generate_pairing_lookup,
    build_testing_dataset_member_matrix,
    load_pairing_bank,
    mapping_sha256,
)
from io_utils import atomic_write_zarr
from residuals import (
    RESIDUAL_TRANSFER_SPACE,
    apply_physical_residuals,
    predictions_to_physical,
)

parser = argparse.ArgumentParser()
parser.add_argument("--var", type=str, default="mslp", choices=["mslp", "t2m"])
parser.add_argument(
    "--year",
    type=int,
    default=2025,
    help="New-input/evaluation year to generate UQ samples for",
)
parser.add_argument("--gpu-id", type=int, default=0, help="Single GPU to use")
parser.add_argument("--ens-start", type=int, default=1)
parser.add_argument("--ens-end", type=int, default=10)
parser.add_argument("--date-start", type=str, default=None, help="Start date YYYYMMDD")
parser.add_argument("--date-end", type=str, default=None, help="End date YYYYMMDD")
parser.add_argument(
    "--thursdays-only",
    "--eval-thursdays-only",
    dest="thursdays_only",
    action="store_true",
    help="Generate only available Thursday initialization dates.",
)
parser.add_argument(
    "--require-complete-eval-dates",
    action="store_true",
    help="Fail unless all Thursdays in the configured test interval are present.",
)
parser.add_argument("--overwrite-samples", action="store_true")
parser.add_argument("--overwrite-pred", action="store_true")
parser.add_argument(
    "--repair-incompatible-artifacts",
    action="store_true",
    help="Reuse compatible sample/prediction stores and atomically replace only stores that fail the current TxT-UQ provenance contract.",
)
parser.add_argument(
    "--output-tag",
    type=str,
    default="txt_uq_k5",
    help="Tag appended to output dirs (e.g., 'rmse' → txt_uq_samples_2025_rmse/)",
)
parser.add_argument(
    "--resid-tag",
    type=str,
    default="txt_uq",
    help="Tag for residuals dir (e.g., 'v2' → txt_uq_residuals_2024_v2/)",
)
parser.add_argument(
    "--ckpt-version",
    type=str,
    default="txt",
    help="Checkpoint prefix version (e.g. v2c, v2d)",
)
args = parser.parse_args()


def zarr_float32_encoding(ds: xr.Dataset, var_names: list[str]) -> dict:
    encoding = {}
    for name in var_names:
        dims = ds[name].dims
        chunks = []
        for dim in dims:
            size = int(ds.sizes[dim])
            if dim in {"input", "model", "weight"}:
                chunk = 1
            elif dim == "init_time":
                chunk = min(size, 16)
            elif dim == "sample":
                chunk = min(size, 100)
            else:
                chunk = size
            chunks.append(chunk)
        encoding[name] = {"dtype": "float32", "chunks": tuple(chunks)}
    try:
        import zarr

        zarr_major = int(zarr.__version__.split(".", 1)[0])
        if zarr_major >= 3:
            from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

            codec = BloscCodec(
                cname=BloscCname.zstd, clevel=3, shuffle=BloscShuffle.bitshuffle
            )
            for enc in encoding.values():
                enc["compressors"] = (codec,)
        else:
            from numcodecs import Blosc

            codec = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            for enc in encoding.values():
                enc["compressor"] = codec
    except Exception:
        pass
    return encoding


def write_zarr_dataset(
    ds: xr.Dataset, path: str, var_names: list[str], overwrite: bool
) -> None:
    atomic_write_zarr(
        ds, path, encoding=zarr_float32_encoding(ds, var_names), overwrite=overwrite
    )


def existing_xarray_store(stem: str) -> str | None:
    zarr_path = f"{stem}.zarr"
    if os.path.exists(zarr_path):
        return zarr_path
    return None


def open_xarray_store(path: str) -> xr.Dataset:
    return xr.open_zarr(path)


def model_combo_order() -> str:
    return ",".join(
        (
            f"{eda:02d}-{weight:02d}"
            for eda, weight in getattr(args, "_model_combos", [])
        )
    )


def sample_artifact_attributes(model_count: int) -> dict:
    residual_schema = PAIRING_SCHEMA
    return {
        "space": "physical",
        "week3_valid_day_offsets": "18-24",
        "week4_valid_day_offsets": "25-31",
        "normalizer": "doy_full",
        "normalizer_provenance": normalizer_identity,
        "ckpt_version": args.ckpt_version,
        "architecture": "sp_full",
        "model_count": model_count,
        "model_combo_order": model_combo_order(),
        "checkpoint_bank_identity": args._checkpoint_bank_identity,
        "residual_added": True,
        "residual_target_mode": "eda_paired",
        "residual_pool_axis": "datewise_perturbed_dataset_index",
        "resid_tag": args.resid_tag,
        "output_tag": args.output_tag,
        "neighbor_method": "rmse",
        "k_nearest": int(5),
        "analog_distance": "global",
        "analog_distance_space": ANALOG_DISTANCE_SPACE,
        "residual_source_ifs_members": "1-10",
        "model_batch": MODEL_BATCH,
        "sample_flat_order": "explicit sample_model_index and sample_analog_rank coordinates",
        "residual_pairing_schema": residual_schema,
        "residual_pairing_seed": PAIRING_SEED,
        "residual_pairing_bank_sha256": args._testing_pairing_contract[
            "pairing_bank_sha256"
        ],
        "residual_pairing_date_inventory_sha256": args._testing_pairing_contract[
            "date_inventory_sha256"
        ],
        "residual_pairing_segment": "testing",
        "residual_transfer_space": RESIDUAL_TRANSFER_SPACE,
        "residual_units": "Pa" if variable == "mslp" else "K",
        "time_schema": TIME_SCHEMA,
        "input_channels": 1,
    }


def prediction_artifact_attributes(model_count: int) -> dict:
    return {
        "space": "physical",
        "time_schema": TIME_SCHEMA,
        "normalizer": "doy_full",
        "normalizer_provenance": normalizer_identity,
        "ckpt_version": args.ckpt_version,
        "architecture": "sp_full",
        "model_count": model_count,
        "model_combo_order": model_combo_order(),
        "checkpoint_bank_identity": args._checkpoint_bank_identity,
        "output_tag": args.output_tag,
        "residual_added": True,
        "resid_tag": args.resid_tag,
        "k_nearest": int(5),
        "neighbor_method": "rmse",
        "analog_distance": "global",
        "analog_distance_space": ANALOG_DISTANCE_SPACE,
        "residual_target_mode": "eda_paired",
        "residual_pool_axis": "datewise_perturbed_dataset_index",
        "residual_pairing_bank_sha256": args._testing_pairing_contract[
            "pairing_bank_sha256"
        ],
        "residual_pairing_date_inventory_sha256": args._testing_pairing_contract[
            "date_inventory_sha256"
        ],
        "residual_pairing_segment": "testing",
        "residual_transfer_space": RESIDUAL_TRANSFER_SPACE,
        "residual_units": "Pa" if variable == "mslp" else "K",
        "prediction_kind": "residual_adjusted_uq_mean",
        "input_channels": 1,
    }


def load_existing_sample(
    path: str,
    *,
    input_member: int,
    init_time,
    model_count: int,
    valid_day_offsets,
    raw_lead_start_hour,
    raw_lead_end_hour,
    latitude,
    longitude,
) -> np.ndarray:
    expected_attrs = sample_artifact_attributes(model_count)
    with open_xarray_store(path) as dataset:
        mismatches = {
            key: (dataset.attrs.get(key), value)
            for key, value in expected_attrs.items()
            if dataset.attrs.get(key) != value
        }
        required = {"samples_w1", "samples_w2", "pdf_mean"}
        if not required.issubset(dataset.data_vars):
            mismatches["data_vars"] = (sorted(dataset.data_vars), sorted(required))
        coordinate_ok = (
            np.array_equal(dataset.input.values, [input_member])
            and np.array_equal(
                pd.DatetimeIndex(dataset.init_time.values).values,
                pd.DatetimeIndex([init_time]).values,
            )
            and np.array_equal(dataset.valid_day_offset.values, valid_day_offsets)
            and np.array_equal(
                dataset.valid_date.values.reshape(-1), valid_dates(init_time).values
            )
            and np.array_equal(
                dataset.raw_lead_start_hour.values.reshape(-1), raw_lead_start_hour
            )
            and np.array_equal(
                dataset.raw_lead_end_hour.values.reshape(-1), raw_lead_end_hour
            )
            and np.array_equal(dataset.latitude.values, latitude)
            and np.array_equal(dataset.longitude.values, longitude)
        )
        sample_count = int(dataset.attrs.get("sample_count", -1))
        analog_count = int(dataset.attrs.get("analog_count", -1))
        expected_sample_count = model_count * analog_count
        sample_coordinates_ok = (
            "sample_model_index" in dataset.coords
            and "sample_analog_rank" in dataset.coords
            and (dataset.sample_model_index.sizes.get("sample") == sample_count)
            and (dataset.sample_analog_rank.sizes.get("sample") == sample_count)
            and np.all(
                (dataset.sample_model_index.values.ravel() >= 0)
                & (dataset.sample_model_index.values.ravel() < model_count)
            )
            and np.array_equal(
                np.bincount(
                    dataset.sample_model_index.values.astype(int).ravel(),
                    minlength=model_count,
                ),
                np.full(model_count, analog_count),
            )
        )
        size_ok = (
            dataset.sizes.get("input") == 1
            and dataset.sizes.get("init_time") == 1
            and (dataset.sizes.get("sample") == sample_count)
            and (sample_count == expected_sample_count)
            and (dataset.sizes.get("valid_day_offset") == len(valid_day_offsets))
            and (dataset.sizes.get("analog", 0) == analog_count)
        )
        analog_vars = {
            "analog_test_member",
            "analog_init_time",
            "analog_target_era5_eda_logical_member",
            "analog_target_era5_eda_storage_index",
            "analog_perturbed_dataset_index",
            "analog_rmse",
        }
        analog_metadata_ok = analog_vars.issubset(dataset.data_vars)
        digest_text = dataset.attrs.get("residual_pairing_date_sha256s", "")
        digests = digest_text.split(",") if digest_text else []
        analog_metadata_ok = (
            analog_metadata_ok
            and analog_count > 0
            and (bool(digests) and all((len(value) == 64 for value in digests)))
            and np.all(
                (dataset.sample_analog_rank.values.ravel() >= 0)
                & (dataset.sample_analog_rank.values.ravel() < analog_count)
            )
        )
        finite_ok = all((np.isfinite(dataset[name].values).all() for name in required))
        if (
            mismatches
            or not coordinate_ok
            or (not size_ok)
            or (not sample_coordinates_ok)
            or (not analog_metadata_ok)
            or (not finite_ok)
            or (sample_count <= 0)
        ):
            raise RuntimeError(
                f"Existing sample artifact is incompatible with this run: {path}; attrs={mismatches}, coordinates_match={coordinate_ok}, sizes_match={size_ok}, sample_coordinates_match={sample_coordinates_ok}, finite={finite_ok}. analog_metadata={analog_metadata_ok}. Use --overwrite-samples explicitly."
            )
        residual_pool_indices = sorted(getattr(args, "_residual_pool_indices", []))
        validate_sample_dataset(
            dataset,
            path=path,
            variable=variable,
            expected_attrs=expected_attrs,
            input_member=input_member,
            init_time=init_time,
            model_indices=list(range(model_count)),
            residual_pool_indices=residual_pool_indices,
            k_nearest=int(5),
            residual_pairings=args._testing_pairings,
            latitude=latitude,
            longitude=longitude,
            check_finite=True,
        )
        return (
            dataset["pdf_mean"]
            .isel(input=0, init_time=0, drop=True)
            .transpose("valid_day_offset", "latitude", "longitude")
            .values.astype(np.float32)
        )


if not args.output_tag:
    args.output_tag = "rmse"
variable = args.var
forecast_year = int(args.year)
VAR_DICT = {"mslp": "mean_sea_level_pressure", "t2m": "2m_temperature"}
field_name = VAR_DICT[variable]
resid_suffix = f"_{args.resid_tag}" if args.resid_tag else ""
residual_dir = f"{OUTPUT_DIR}/residuals{resid_suffix}"
tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
sample_dir = f"{OUTPUT_DIR}/samples_{forecast_year}{tag_suffix}"
prediction_dir = f"{OUTPUT_DIR}/predictions_{forecast_year}{tag_suffix}"
MODEL_BATCH = 10
normalizer_identity = normalizer_provenance("doy_full")
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_ifs_inputs(year, ensemble_list, clip_bounds=None):
    print(f"Loading data for year {year}...")
    norm = load_normalizer(field_name, mode="doy_full")
    print(f"  Normalizer: {norm.mode}  sig_scalar={norm.sig_scalar}")
    day_offsets = None
    raw_start_hours = None
    raw_end_hours = None
    lat_vals = None
    lon_vals = None
    f_date = pd.Timestamp(f"{year}-01-01")
    l_date = pd.Timestamp(f"{year}-12-31")
    if clip_bounds is not None:
        _lo, _hi = clip_bounds
        if _lo is not None:
            f_date = max(f_date, pd.to_datetime(_lo))
        if _hi is not None:
            l_date = min(l_date, pd.to_datetime(_hi))
    if f_date > l_date:
        return ([], [], norm, {})
    all_ifs_norm = {}
    reference_dates = None
    for ens in ensemble_list:
        ds = xr.open_zarr(ifs_zarr(variable, ens), decode_timedelta=False)[field_name]
        ds = ds.sel(init_time=slice(f_date, l_date))
        ds = ds.sortby("latitude").sortby("init_time")
        ds = canonicalize_ifs(ds, field_name)
        current_dates = pd.DatetimeIndex(ds.init_time.values)
        if reference_dates is None:
            reference_dates = current_dates
            day_offsets = ds.valid_day_offset.values
            raw_start_hours = ds.raw_lead_start_hour.values
            raw_end_hours = ds.raw_lead_end_hour.values
            lat_vals = ds.latitude.values
            lon_vals = ds.longitude.values
        elif not current_dates.equals(reference_dates):
            raise RuntimeError(
                f"IFS member {ens} dates do not match the first requested member"
            )
        all_ifs_norm[ens] = norm.normalize_ifs(ds).load()
        print(f"  Ens {ens}: {len(current_dates)} dates, shape={tuple(ds.shape)}")
    datasets = []
    dates_list = []
    for ens in ensemble_list:
        ifs_norm = all_ifs_norm[ens]
        datasets.append(ForecastInputDataset(ifs_norm))
        dates_list.append(pd.to_datetime(ifs_norm.init_time.values))
    coords = {
        "valid_day_offset": day_offsets,
        "raw_lead_start_hour": raw_start_hours,
        "raw_lead_end_hour": raw_end_hours,
        "latitude": lat_vals,
        "longitude": lon_vals,
    }
    return (datasets, dates_list, norm, coords)


def compute_latitude_weights(lat_vals):
    lat_rad = np.deg2rad(lat_vals.astype(np.float64))
    w = np.cos(lat_rad)
    return (w / w.mean()).astype(np.float32)


def cache_analog_inputs(datasets_by_year):
    if not datasets_by_year:
        return []
    n_ens = len(next(iter(datasets_by_year.values())))
    per_ens_chunks = [[] for _ in range(n_ens)]
    for yr in sorted(datasets_by_year.keys()):
        ds_list = datasets_by_year[yr]
        for ens_idx in range(len(ds_list)):
            ds = ds_list[ens_idx]
            if hasattr(ds, "ifs"):
                da = ds.ifs
                inputs = np.asarray(da.values, dtype=np.float32)
            else:
                inputs = np.stack([ds[i][0][0].numpy() for i in range(len(ds))])
            per_ens_chunks[ens_idx].append(inputs)
            print(
                f"  Year {yr} Ens {ens_idx + 1}: {inputs.shape[0]} dates, shape {inputs.shape}"
            )
    ifs_residual = [np.concatenate(chunks, axis=0) for chunks in per_ens_chunks]
    total_gb = sum((a.nbytes for a in ifs_residual)) / 1000000000.0
    print(
        f"  Total residual IFS cache: {total_gb:.1f} GB (per-ens dates: {ifs_residual[0].shape[0]})"
    )
    return ifs_residual


def compute_analog_rmse(new_input_array, analog_member_inputs, lat_weights):
    n_dates = analog_member_inputs.shape[0]
    rmse = np.empty(n_dates, dtype=np.float32)
    for i in range(n_dates):
        diff_sq = (analog_member_inputs[i] - new_input_array) ** 2
        weighted = diff_sq * lat_weights[None, :, None]
        rmse[i] = np.sqrt(weighted.mean())
    return rmse


def select_rmse_analogs(new_input_array, analog_member_inputs, lat_weights, k=5):
    rmse = compute_analog_rmse(new_input_array, analog_member_inputs, lat_weights)
    selected = np.argsort(rmse)[:k]
    return (selected, rmse[selected])


def load_model_bank():
    resolved = resolve_txt_model_bank(
        CHECKPOINT_DIR,
        variable=variable,
        version=args.ckpt_version,
        expected_model_count=100,
        expected_arch="sp_full",
        expected_normalizer="doy_full",
        expected_normalizer_provenance=normalizer_identity,
        expected_pairing_digests=args._checkpoint_pairing_digests,
        expected_pairing_bank_sha256=(
            None
            if args._checkpoint_pairing_contract is None
            else args._checkpoint_pairing_contract["pairing_bank_sha256"]
        ),
    )
    model_bank = []
    for _, ckpt_path in resolved:
        model = load_txt_checkpoint(
            ckpt_path,
            expected_arch="sp_full",
            expected_training_variable=variable,
            expected_normalizer_provenance=normalizer_identity,
            map_location=device,
        )
        model.eval()
        model_bank.append(model)
    args._model_combos = [key for key, _ in resolved]
    args._model_checkpoints = [str(path) for _, path in resolved]
    args._checkpoint_bank_identity = checkpoint_bank_identity(resolved)
    print(f"  Loaded {len(model_bank)} models onto {device}")
    return model_bank


def load_validated_residual(
    path,
    *,
    test_member,
    analog_date,
    expected_perturbed_dataset_index,
    model_count,
    coords,
):
    analog_date = pd.Timestamp(analog_date)
    date_key = analog_date.strftime("%Y%m%d")
    pair = args._testing_pairing_lookup[date_key][test_member]
    target_eda_logical_member = int(pair["era5_eda_member"])
    target_eda_storage_index = int(pair["era5_eda_storage_index"])
    perturbed_dataset_index = int(pair["perturbed_dataset_index"])
    pairing_schema = PAIRING_SCHEMA
    pairing_seed = PAIRING_SEED
    if perturbed_dataset_index != int(expected_perturbed_dataset_index):
        raise RuntimeError(
            f"Residual {date_key} member {test_member} belongs to perturbed dataset {perturbed_dataset_index}, expected {expected_perturbed_dataset_index}"
        )
    with open_xarray_store(path) as dataset:
        expected_attrs = {
            "var": variable,
            "ckpt_prefix": f"{variable}_{args.ckpt_version}",
            "architecture": "sp_full",
            "model_count": model_count,
            "model_combo_order": model_combo_order(),
            "checkpoint_bank_identity": args._checkpoint_bank_identity,
            "normalizer": "doy_full",
            "normalizer_provenance": normalizer_identity,
            "space": "physical",
            "units": "Pa" if variable == "mslp" else "K",
            "residual_transfer_space": RESIDUAL_TRANSFER_SPACE,
            "target_era5_eda_logical_member": target_eda_logical_member,
            "target_era5_eda_storage_index": target_eda_storage_index,
            "paired_perturbed_dataset_index": perturbed_dataset_index,
            "pairing_schema": pairing_schema,
            "pairing_seed": pairing_seed,
            "pairing_mapping_sha256": (
                "none"
                if pairing_schema == "none"
                else mapping_sha256(args._testing_pairings[perturbed_dataset_index])
            ),
            "pairing_bank_sha256": (
                "none"
                if pairing_schema == "none"
                else args._testing_pairing_contract["pairing_bank_sha256"]
            ),
            "pairing_date_inventory_sha256": (
                "none"
                if pairing_schema == "none"
                else args._testing_pairing_contract["date_inventory_sha256"]
            ),
            "pairing_segment": "none" if pairing_schema == "none" else "testing",
            "time_schema": TIME_SCHEMA,
            "input_channels": 1,
        }
        mismatches = {
            key: (dataset.attrs.get(key), expected)
            for key, expected in expected_attrs.items()
            if dataset.attrs.get(key) != expected
        }
        expected_mapping_digest = dataset.attrs.get("pairing_mapping_sha256")
        expected_date_mapping_digest = dataset.attrs.get("pairing_date_mapping_sha256")
        mapping_digest_ok = (
            expected_mapping_digest == "none" and expected_date_mapping_digest == "none"
            if pairing_schema == "none"
            else isinstance(expected_mapping_digest, str)
            and len(expected_mapping_digest) == 64
            and (
                expected_date_mapping_digest
                == date_mapping_sha256(analog_date, pairings=args._testing_pairings)
            )
        )
        coordinate_ok = (
            np.array_equal(dataset.test_ens.values, [test_member])
            and np.array_equal(
                pd.DatetimeIndex(dataset.init_time.values).values,
                pd.DatetimeIndex([analog_date]).values,
            )
            and np.array_equal(dataset.model.values, np.arange(model_count))
            and np.array_equal(
                dataset.valid_day_offset.values, coords["valid_day_offset"]
            )
            and np.array_equal(
                dataset.valid_date.values, valid_dates(analog_date).values
            )
            and np.array_equal(dataset.latitude.values, coords["latitude"])
            and np.array_equal(dataset.longitude.values, coords["longitude"])
            and np.array_equal(
                dataset.raw_lead_start_hour.values, coords["raw_lead_start_hour"]
            )
            and np.array_equal(
                dataset.raw_lead_end_hour.values, coords["raw_lead_end_hour"]
            )
        )
        expected_dims = (
            "test_ens",
            "init_time",
            "model",
            "valid_day_offset",
            "latitude",
            "longitude",
        )
        shape_ok = (
            "residual" in dataset.data_vars
            and dataset.residual.dims == expected_dims
            and (dataset.sizes.get("test_ens") == 1)
            and (dataset.sizes.get("init_time") == 1)
            and (dataset.sizes.get("model") == model_count)
            and (dataset.sizes.get("valid_day_offset") == 32)
        )
        if mismatches or not mapping_digest_ok or (not coordinate_ok) or (not shape_ok):
            raise RuntimeError(
                f"Incompatible residual artifact {path}: attrs={mismatches}, mapping_digest={mapping_digest_ok}, coordinates={coordinate_ok}, shape={shape_ok}"
            )
        values = (
            dataset.residual.isel(test_ens=0, init_time=0, drop=True)
            .transpose("model", "valid_day_offset", "latitude", "longitude")
            .values.astype(np.float32)
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Residual contains non-finite values: {path}")
        record = {
            "test_member": int(test_member),
            "init_time": analog_date.to_datetime64(),
            "target_era5_eda_logical_member": target_eda_logical_member,
            "target_era5_eda_storage_index": target_eda_storage_index,
            "perturbed_dataset_index": perturbed_dataset_index,
            "mapping_sha256": expected_date_mapping_digest,
        }
    return (torch.from_numpy(values), record)


def sample_forecast_date(
    x_new,
    current_date,
    analog_dates,
    coords,
    norm,
    w1_idx,
    w2_idx,
    model_bank,
    analog_inputs=None,
    lat_weights=None,
):
    if analog_inputs is None or lat_weights is None:
        raise ValueError(
            "RMSE analog sampling requires the held-out IFS pool and latitude weights"
        )
    t_len, h_len, w_len = x_new.shape[2:]
    sum_samples_phys = np.zeros((t_len, h_len, w_len), dtype=np.float64)
    count_samples = 0
    samples_w1_cpu = []
    samples_w2_cpu = []
    sample_model_indices = []
    sample_analog_ranks = []
    print(f"  Processing {len(model_bank)} models in batches of {MODEL_BATCH}...")
    neighbor_residuals = []
    selected_pairs = []
    selected_rmse = []
    analog_records = []
    for testing_dataset_index in range(len(analog_dates)):
        test_dates = analog_dates[testing_dataset_index]
        new_input_array = x_new[0, 0].cpu().numpy()
        valid_indices, selected_distances = select_rmse_analogs(
            new_input_array, analog_inputs[testing_dataset_index], lat_weights, k=5
        )
        selected_pairs.extend(
            ((testing_dataset_index, int(ni)) for ni in valid_indices)
        )
        selected_rmse.extend(np.asarray(selected_distances, dtype=float))
    for pair_rank, (testing_dataset_index, ni) in enumerate(selected_pairs):
        test_dates = analog_dates[testing_dataset_index]
        test_member = int(args._residual_member_by_pool[testing_dataset_index][ni])
        expected_dataset_index = int(args._residual_pool_indices[testing_dataset_index])
        date_str = pd.to_datetime(test_dates[ni]).strftime("%Y%m%d")
        year_from_date = int(date_str[:4])
        file_stem = f"{residual_dir}/txt_uq_residuals_{variable}_{year_from_date}_ens{test_member:02d}_{date_str}"
        file_path = existing_xarray_store(file_stem)
        if file_path:
            residual, record = load_validated_residual(
                file_path,
                test_member=test_member,
                analog_date=test_dates[ni],
                expected_perturbed_dataset_index=expected_dataset_index,
                model_count=len(model_bank),
                coords=coords,
            )
            record["rmse"] = float(selected_rmse[pair_rank])
            neighbor_residuals.append(residual)
            analog_records.append(record)
        else:
            raise FileNotFoundError(
                f"Selected analog residual is missing: {file_stem}.zarr"
            )
    if len(neighbor_residuals) > 0:
        neighbor_residuals = torch.stack(neighbor_residuals)
        print(f"  Found {len(neighbor_residuals)} neighbor dates (CPU)")
    else:
        raise RuntimeError("No compatible analog residuals were selected")
    for batch_start in range(0, len(model_bank), MODEL_BATCH):
        batch_end = min(batch_start + MODEL_BATCH, len(model_bank))
        batch_slice = model_bank[batch_start:batch_end]
        with torch.no_grad():
            batch_preds = []
            for model in batch_slice:
                pred = model.predict_corrected(x_new)
                batch_preds.append(pred.squeeze(0).squeeze(0))
            batch_preds = torch.stack(batch_preds)
        valid_day_offsets = np.asarray(coords["valid_day_offset"])
        batch_phys = predictions_to_physical(
            norm, batch_preds.cpu().numpy(), current_date, valid_day_offsets
        )
        n_neighbors = neighbor_residuals.shape[0]
        chunk_size = 10
        for i in range(0, n_neighbors, chunk_size):
            end_i = min(i + chunk_size, n_neighbors)
            chunk_residuals = neighbor_residuals[i:end_i, batch_start:batch_end].numpy()
            chunk_phys = apply_physical_residuals(
                batch_phys[np.newaxis, ...], chunk_residuals
            ).reshape(-1, t_len, h_len, w_len)
            sum_samples_phys += chunk_phys.sum(axis=0, dtype=np.float64)
            count_samples += chunk_phys.shape[0]
            w1_mean = chunk_phys[:, w1_idx, :, :].mean(axis=1)
            w2_mean = chunk_phys[:, w2_idx, :, :].mean(axis=1)
            samples_w1_cpu.append(w1_mean.astype(np.float32))
            samples_w2_cpu.append(w2_mean.astype(np.float32))
            n_chunk = end_i - i
            n_batch = batch_end - batch_start
            sample_model_indices.append(
                np.tile(np.arange(batch_start, batch_end), n_chunk)
            )
            sample_analog_ranks.append(np.repeat(np.arange(i, end_i), n_batch))
            del chunk_residuals, chunk_phys, w1_mean, w2_mean
        del batch_preds, batch_phys
        torch.cuda.empty_cache()
    if count_samples == 0:
        pdf_mean_phys = np.zeros((t_len, h_len, w_len), dtype=np.float32)
    else:
        pdf_mean_phys = (sum_samples_phys / float(count_samples)).astype(np.float32)
    if samples_w1_cpu:
        samples_w1_phys = np.concatenate(samples_w1_cpu, axis=0)
        samples_w2_phys = np.concatenate(samples_w2_cpu, axis=0)
    else:
        empty = np.zeros((0, h_len, w_len), dtype=np.float32)
        samples_w1_phys = samples_w2_phys = empty
    model_index = np.concatenate(sample_model_indices).astype(np.int64)
    analog_rank = np.concatenate(sample_analog_ranks).astype(np.int64)
    if not (
        len(model_index) == len(analog_rank) == count_samples
        and len(samples_w1_phys) == count_samples
        and (len(samples_w2_phys) == count_samples)
    ):
        raise RuntimeError("Sample provenance coordinates do not match sample count")
    return (
        pdf_mean_phys,
        samples_w1_phys.astype(np.float32),
        samples_w2_phys.astype(np.float32),
        count_samples,
        model_index,
        analog_rank,
        analog_records,
    )


def generate_uq_samples():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(sample_dir)
    ensure_dir(prediction_dir)
    print("=" * 80)
    print("UQ PIPELINE v2 - Raw Weekly-Mean Samples (no histogram truncation)")
    print("=" * 80)
    print(f"Analog distance selected: {'global'}")
    print(f"Output tag: {'(none)'}")
    from config import tranche_bounds

    _, _, _val_s, _val_e, _test_s, _test_e = tranche_bounds()
    forecast_datasets, forecast_dates, norm, forecast_coordinates = load_ifs_inputs(
        forecast_year,
        range(args.ens_start, args.ens_end + 1),
        clip_bounds=(_test_s, _test_e),
    )
    args._checkpoint_pairing_digests = None
    args._checkpoint_pairing_contract = None
    args._testing_pairings = None
    args._testing_pairing_contract = None
    args._testing_pairing_lookup = None
    checkpoint_pairings, args._checkpoint_pairing_contract = load_pairing_bank(
        PAIRING_DIR / args.ckpt_version
    )
    args._checkpoint_pairing_digests = {
        index: mapping_sha256(checkpoint_pairings[index]) for index in range(10)
    }
    dates_residual = []
    ifs_residual_cache = None
    lat_weights = None
    resid_lo, resid_hi = CONFIG["residual_years"]
    resid_years = list(range(resid_lo, resid_hi + 1))
    print(f"\nLoading residual pool across years {resid_years}...")
    datasets_by_year = {}
    dates_by_year = {}
    residual_ensemble_numbers = list(range(1, 11))
    print(f"Residual source IFS members: {residual_ensemble_numbers}")
    for yr in resid_years:
        ds_yr, dates_yr, _, _ = load_ifs_inputs(
            yr, residual_ensemble_numbers, clip_bounds=(_val_s, _val_e)
        )
        datasets_by_year[yr] = ds_yr
        dates_by_year[yr] = dates_yr
    fixed_member_dates = []
    for member_index in range(len(residual_ensemble_numbers)):
        parts = [dates_by_year[yr][member_index] for yr in sorted(dates_by_year)]
        fixed_member_dates.append(
            pd.DatetimeIndex(np.concatenate([part.values for part in parts]))
        )
    common_residual_dates = fixed_member_dates[0]
    if any(
        (not dates.equals(common_residual_dates) for dates in fixed_member_dates[1:])
    ):
        raise RuntimeError("Residual IFS members do not share a common date axis")
    args._testing_pairings, args._testing_pairing_contract = load_pairing_bank(
        PAIRING_DIR / args.ckpt_version,
        segment="testing",
        expected_dates=common_residual_dates,
    )
    args._testing_pairing_lookup = generate_pairing_lookup(
        common_residual_dates, pairings=args._testing_pairings
    )
    args._residual_pool_indices, args._residual_member_by_pool = (
        build_testing_dataset_member_matrix(
            common_residual_dates,
            available_members=residual_ensemble_numbers,
            pairings=args._testing_pairings,
        )
    )
    dates_residual = [common_residual_dates for _ in args._residual_pool_indices]
    print(
        f"Residual testing dataset rows: {args._residual_pool_indices}; persisted date-wise members are explicit"
    )
    print("\nPrecomputing residual-year IFS inputs for RMSE neighbor selection...")
    fixed_member_cache = cache_analog_inputs(datasets_by_year)
    arrays_by_member = dict(
        zip(residual_ensemble_numbers, fixed_member_cache, strict=True)
    )
    ifs_residual_cache = [
        PairedMemberArray(arrays_by_member, member_by_date)
        for member_by_date in args._residual_member_by_pool
    ]
    lat_weights = compute_latitude_weights(forecast_coordinates["latitude"])
    print(f"Neighbor method: RMSE (k={5}, analog_distance={'global'})")
    print("\nLoading all 100 models (one-time)...")
    model_bank = load_model_bank()
    print(f"Models loaded: {len(model_bank)}")
    day_offsets = np.asarray(forecast_coordinates["valid_day_offset"])
    w1_idx = week_index_array(1)
    w2_idx = week_index_array(2)
    print(f"Week 3 model indices: {w1_idx} (valid-day offsets 18..24)")
    print(f"Week 4 model indices: {w2_idx} (valid-day offsets 25..31)")
    date_start = pd.to_datetime(args.date_start) if args.date_start else None
    date_end = pd.to_datetime(args.date_end) if args.date_end else None
    for ens_idx, forecast_dataset in enumerate(forecast_datasets):
        actual_ens_num = args.ens_start + ens_idx
        dates = forecast_dates[ens_idx]
        if date_start or date_end:
            mask = np.ones(len(dates), dtype=bool)
            if date_start:
                mask &= dates >= date_start
            if date_end:
                mask &= dates <= date_end
            indices = np.where(mask)[0]
            print(
                f"\nFiltered {len(indices)}/{len(dates)} dates for Ens {actual_ens_num}"
            )
        else:
            indices = np.arange(len(dates))
        if args.thursdays_only:
            indices = np.asarray(
                [idx for idx in indices if pd.Timestamp(dates[idx]).weekday() == 3],
                dtype=int,
            )
            print(
                f"Thursday filter: {len(indices)}/{len(dates)} dates for Ens {actual_ens_num}"
            )
        if args.require_complete_eval_dates:
            expected_dates = expected_thursday_initializations(
                _test_s, _test_e, year=forecast_year
            )
            actual_dates = pd.DatetimeIndex(dates[indices])
            if not actual_dates.equals(expected_dates):
                raise RuntimeError(
                    f"Incomplete evaluation dates for {variable} {forecast_year} member {actual_ens_num}: missing={expected_dates.difference(actual_dates).strftime('%Y-%m-%d').tolist()}, extra={actual_dates.difference(expected_dates).strftime('%Y-%m-%d').tolist()}"
                )
        ens_pred_means = []
        ens_ifs_inputs = []
        ens_dates = []
        print(f"\nProcessing Ensemble {actual_ens_num} ({len(indices)} dates)...")
        for i in tqdm(indices, desc=f"Ens {actual_ens_num}"):
            current_date = dates[i]
            date_str = current_date.strftime("%Y%m%d")
            x_new, _ = forecast_dataset[i]
            x_new = x_new.unsqueeze(0).to(device)
            ifs_input_phys = norm.denormalize_grid(
                x_new[0, 0].cpu().numpy(), current_date, day_offsets
            )
            ens_ifs_inputs.append(ifs_input_phys.astype(np.float32))
            samples_stem = f"{sample_dir}/txt_uq_samples_{variable}_{forecast_year}_input{actual_ens_num:02d}_{date_str}"
            samples_path = f"{samples_stem}.zarr"
            existing_samples_path = existing_xarray_store(samples_stem)
            reused_sample = False
            if existing_samples_path and (not args.overwrite_samples):
                try:
                    pdf_mean_phys = load_existing_sample(
                        existing_samples_path,
                        input_member=actual_ens_num,
                        init_time=current_date,
                        model_count=len(model_bank),
                        valid_day_offsets=day_offsets,
                        raw_lead_start_hour=forecast_coordinates["raw_lead_start_hour"],
                        raw_lead_end_hour=forecast_coordinates["raw_lead_end_hour"],
                        latitude=forecast_coordinates["latitude"],
                        longitude=forecast_coordinates["longitude"],
                    )
                except Exception as exc:
                    if not args.repair_incompatible_artifacts:
                        raise
                    print(
                        f"  Rebuilding incompatible sample {existing_samples_path}: {exc}"
                    )
                else:
                    reused_sample = True
                    print(f"  Reusing validated sample artifact for {date_str}")
            if not reused_sample:
                (
                    pdf_mean_phys,
                    samples_w1,
                    samples_w2,
                    sample_count,
                    sample_model_index,
                    sample_analog_rank,
                    analog_records,
                ) = sample_forecast_date(
                    x_new,
                    current_date,
                    dates_residual,
                    forecast_coordinates,
                    norm,
                    w1_idx,
                    w2_idx,
                    model_bank,
                    analog_inputs=ifs_residual_cache,
                    lat_weights=lat_weights,
                )
                attrs = sample_artifact_attributes(len(model_bank))
                attrs["sample_count"] = int(sample_count)
                attrs["analog_count"] = len(analog_records)
                mapping_digests = sorted(
                    {
                        record["mapping_sha256"]
                        for record in analog_records
                        if record["mapping_sha256"] != "none"
                    }
                )
                attrs["residual_pairing_date_sha256s"] = (
                    ",".join(mapping_digests) if mapping_digests else "none"
                )
                ds_samples = xr.Dataset(
                    data_vars={
                        "samples_w1": (("sample", "latitude", "longitude"), samples_w1),
                        "samples_w2": (("sample", "latitude", "longitude"), samples_w2),
                        "pdf_mean": (
                            ("valid_day_offset", "latitude", "longitude"),
                            pdf_mean_phys.astype(np.float32),
                        ),
                        "analog_test_member": (
                            ("analog",),
                            np.asarray(
                                [r["test_member"] for r in analog_records],
                                dtype=np.int16,
                            ),
                        ),
                        "analog_init_time": (
                            ("analog",),
                            np.asarray(
                                [r["init_time"] for r in analog_records],
                                dtype="datetime64[ns]",
                            ),
                        ),
                        "analog_target_era5_eda_logical_member": (
                            ("analog",),
                            np.asarray(
                                [
                                    r["target_era5_eda_logical_member"]
                                    for r in analog_records
                                ],
                                dtype=np.int16,
                            ),
                        ),
                        "analog_target_era5_eda_storage_index": (
                            ("analog",),
                            np.asarray(
                                [
                                    r["target_era5_eda_storage_index"]
                                    for r in analog_records
                                ],
                                dtype=np.int16,
                            ),
                        ),
                        "analog_perturbed_dataset_index": (
                            ("analog",),
                            np.asarray(
                                [r["perturbed_dataset_index"] for r in analog_records],
                                dtype=np.int16,
                            ),
                        ),
                        "analog_rmse": (
                            ("analog",),
                            np.asarray(
                                [r["rmse"] for r in analog_records], dtype=np.float32
                            ),
                        ),
                    },
                    coords={
                        "sample": np.arange(len(samples_w1)),
                        "sample_model_index": ("sample", sample_model_index),
                        "sample_analog_rank": ("sample", sample_analog_rank),
                        "valid_day_offset": day_offsets,
                        "valid_date": (
                            "valid_day_offset",
                            valid_dates(current_date).values,
                        ),
                        "raw_lead_start_hour": (
                            "valid_day_offset",
                            forecast_coordinates["raw_lead_start_hour"],
                        ),
                        "raw_lead_end_hour": (
                            "valid_day_offset",
                            forecast_coordinates["raw_lead_end_hour"],
                        ),
                        "latitude": forecast_coordinates["latitude"],
                        "longitude": forecast_coordinates["longitude"],
                    },
                    attrs=attrs,
                )
                ds_samples = ds_samples.expand_dims(
                    input=[actual_ens_num], init_time=[current_date]
                )
                payload_sha256 = sample_payload_sha256(ds_samples)
                attrs[SAMPLE_PAYLOAD_SHA256_ATTR] = payload_sha256
                ds_samples.attrs[SAMPLE_PAYLOAD_SHA256_ATTR] = payload_sha256
                validate_sample_dataset(
                    ds_samples,
                    path=samples_path,
                    variable=variable,
                    expected_attrs=attrs,
                    input_member=actual_ens_num,
                    init_time=current_date,
                    model_indices=list(range(len(model_bank))),
                    residual_pool_indices=sorted(args._residual_pool_indices),
                    k_nearest=int(5),
                    residual_pairings=args._testing_pairings,
                    latitude=forecast_coordinates["latitude"],
                    longitude=forecast_coordinates["longitude"],
                    check_finite=True,
                )
                write_zarr_dataset(
                    ds_samples,
                    samples_path,
                    ["samples_w1", "samples_w2", "pdf_mean"],
                    overwrite=args.overwrite_samples
                    or existing_samples_path is not None,
                )
            ens_pred_means.append(pdf_mean_phys.astype(np.float32))
            ens_dates.append(current_date)
        if len(ens_pred_means) > 0:
            pred_stem = f"{prediction_dir}/txt_uq_predictions_{variable}_{forecast_year}_input{actual_ens_num:02d}"
            pred_path = f"{pred_stem}.zarr"
            lock_path = pred_path + ".lock"
            y_pred = np.stack(ens_pred_means, axis=0)
            ifs_input = np.stack(ens_ifs_inputs, axis=0)
            prediction_valid_dates = np.stack(
                [valid_dates(date).values for date in ens_dates], axis=0
            )
            ds_new = xr.Dataset(
                data_vars={
                    "y_pred": (
                        ("init_time", "valid_day_offset", "latitude", "longitude"),
                        y_pred,
                    ),
                    "ifs_input": (
                        ("init_time", "valid_day_offset", "latitude", "longitude"),
                        ifs_input,
                    ),
                },
                coords={
                    "init_time": ens_dates,
                    "valid_day_offset": day_offsets,
                    "valid_date": (
                        ("init_time", "valid_day_offset"),
                        prediction_valid_dates,
                    ),
                    "raw_lead_start_hour": (
                        "valid_day_offset",
                        forecast_coordinates["raw_lead_start_hour"],
                    ),
                    "raw_lead_end_hour": (
                        "valid_day_offset",
                        forecast_coordinates["raw_lead_end_hour"],
                    ),
                    "latitude": forecast_coordinates["latitude"],
                    "longitude": forecast_coordinates["longitude"],
                },
                attrs=prediction_artifact_attributes(len(model_bank)),
            )
            import fcntl

            with open(lock_path, "w") as _lf:
                fcntl.flock(_lf, fcntl.LOCK_EX)
                try:
                    existing_pred_path = existing_xarray_store(pred_stem)
                    ds_merged = ds_new
                    if existing_pred_path:
                        expected_attrs = prediction_artifact_attributes(len(model_bank))
                        try:
                            with open_xarray_store(existing_pred_path) as opened:
                                ds_exist = opened.load()
                            ds_exist_sq = validate_prediction_dataset(
                                ds_exist,
                                path=existing_pred_path,
                                variable=variable,
                                expected_attrs=expected_attrs,
                                input_member=actual_ens_num,
                                latitude=forecast_coordinates["latitude"],
                                longitude=forecast_coordinates["longitude"],
                                check_finite=True,
                            )
                        except Exception as exc:
                            if not (
                                args.overwrite_pred
                                or args.repair_incompatible_artifacts
                            ):
                                raise RuntimeError(
                                    f"Existing prediction artifact is stale, corrupt, or incompatible: {existing_pred_path}. Use --overwrite-pred or --repair-incompatible-artifacts explicitly."
                                ) from exc
                            print(
                                f"Replacing incompatible prediction artifact {existing_pred_path}: {exc}"
                            )
                        else:
                            keep = ~pd.DatetimeIndex(ds_exist_sq.init_time.values).isin(
                                pd.DatetimeIndex(ens_dates)
                            )
                            ds_exist_sq = ds_exist_sq.isel(
                                init_time=np.flatnonzero(keep)
                            )
                            ds_merged = xr.concat(
                                [ds_exist_sq, ds_new], dim="init_time"
                            )
                            ds_merged = ds_merged.sortby("init_time")
                    merged_dates = pd.DatetimeIndex(ds_merged.init_time.values)
                    if merged_dates.has_duplicates:
                        raise RuntimeError(
                            f"Prediction merge produced duplicate dates: {pred_path}"
                        )
                    ds_merged = ds_merged.expand_dims(
                        model=[0], weight=[0], input=[actual_ens_num]
                    )
                    ds_merged = ds_merged.transpose(
                        "input",
                        "model",
                        "weight",
                        "init_time",
                        "valid_day_offset",
                        "latitude",
                        "longitude",
                    )
                    validate_prediction_dataset(
                        ds_merged,
                        path=pred_path,
                        variable=variable,
                        expected_attrs=prediction_artifact_attributes(len(model_bank)),
                        input_member=actual_ens_num,
                        latitude=forecast_coordinates["latitude"],
                        longitude=forecast_coordinates["longitude"],
                        expected_init_times=merged_dates,
                        check_finite=True,
                    )
                    write_zarr_dataset(
                        ds_merged, pred_path, ["y_pred", "ifs_input"], overwrite=True
                    )
                finally:
                    fcntl.flock(_lf, fcntl.LOCK_UN)
            print(f"Saved predictions to {pred_path} ({len(ens_dates)} dates)")
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print(
        f"Analog summary: mode={'global'}, samples_dir={sample_dir}, pred_dir={prediction_dir}"
    )
    print("=" * 80)


if __name__ == "__main__":
    generate_uq_samples()
