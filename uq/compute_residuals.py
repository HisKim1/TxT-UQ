import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import torch
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from config import (
    CHECKPOINT_DIR,
    CONFIG,
    ERA5_ENS_DIR,
    OUTPUT_DIR,
    PAIRING_DIR,
    era5_zarr,
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
from artifacts import residual_coordinates_match
from time_coordinates import (
    TIME_SCHEMA,
    canonicalize_ifs,
    canonicalize_spatial_grid,
    valid_dates,
)
from pairing import (
    PAIRING_SCHEMA,
    PAIRING_SEED,
    date_mapping_sha256,
    generate_pairing_lookup,
    load_pairing_bank,
    mapping_sha256,
)
from io_utils import atomic_write_zarr
from residuals import RESIDUAL_TRANSFER_SPACE, compute_physical_residuals

parser = argparse.ArgumentParser()
parser.add_argument("--var", type=str, default="t2m", choices=["mslp", "t2m"])
parser.add_argument("--gpu-id", type=int, default=1, help="CUDA device index")
parser.add_argument(
    "--ens-start", type=int, default=1, help="First IFS ensemble (inclusive)"
)
parser.add_argument(
    "--ens-end", type=int, default=10, help="Last IFS ensemble (inclusive)"
)
parser.add_argument(
    "--overwrite", action="store_true", help="Overwrite existing residual files"
)
parser.add_argument(
    "--repair-incompatible-artifacts",
    action="store_true",
    help="Reuse compatible residual stores, but atomically replace stores whose date inventory, checkpoint bank, or scientific provenance is stale.",
)
parser.add_argument(
    "--resid-tag",
    type=str,
    default="txt_uq",
    help="Suffix for residuals output dir (e.g. 'v2' -> txt_uq_residuals_2024_v2/)",
)
parser.add_argument(
    "--ckpt-version",
    type=str,
    default="txt",
    help="Checkpoint prefix version (e.g. v2c, v2d)",
)
args = parser.parse_args()
variable = args.var
VAR_DICT = {"mslp": "mean_sea_level_pressure", "t2m": "2m_temperature"}
field_name = VAR_DICT[variable]
physical_units = "Pa" if variable == "mslp" else "K"
checkpoint_dir = CHECKPOINT_DIR
resid_suffix = f"_{args.resid_tag}" if args.resid_tag else ""
residual_dir = f"{OUTPUT_DIR}/residuals{resid_suffix}"
os.makedirs(residual_dir, exist_ok=True)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
print(f"Device: {device}")


def zarr_float32_encoding(ds: xr.Dataset, var_names: list[str]) -> dict:
    encoding = {}
    for name in var_names:
        dims = ds[name].dims
        chunks = []
        for dim in dims:
            size = int(ds.sizes[dim])
            if dim in {"test_ens", "input", "model_input", "weight"}:
                chunk = 1
            elif dim == "init_time":
                chunk = min(size, 4)
            elif dim == "model":
                chunk = min(size, 10)
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


def validate_existing_residual(
    path: str,
    *,
    ensemble: int,
    init_time,
    target_dates,
    target_eda_logical_member: int,
    target_eda_storage_index: int,
    perturbed_dataset_index: int,
    pairing_mapping_sha256: str,
    pairing_date_mapping_sha256: str,
    model_count: int,
    normalizer_metadata: dict,
    latitude,
    longitude,
) -> None:
    with xr.open_zarr(path) as dataset:
        expected_attrs = {
            "var": variable,
            "ckpt_prefix": f"{variable}_{args.ckpt_version}",
            "architecture": "sp_full",
            "model_count": model_count,
            "model_combo_order": ",".join(
                (f"{eda:02d}-{weight:02d}" for eda, weight in args._model_combos)
            ),
            "checkpoint_bank_identity": args._checkpoint_bank_identity,
            "normalizer": "doy_full",
            "normalizer_provenance": normalizer_metadata,
            "space": "physical",
            "units": physical_units,
            "residual_transfer_space": RESIDUAL_TRANSFER_SPACE,
            "target_era5_eda_logical_member": target_eda_logical_member,
            "target_era5_eda_storage_index": target_eda_storage_index,
            "paired_perturbed_dataset_index": perturbed_dataset_index,
            "pairing_schema": PAIRING_SCHEMA,
            "pairing_seed": PAIRING_SEED,
            "pairing_mapping_sha256": pairing_mapping_sha256,
            "pairing_date_mapping_sha256": pairing_date_mapping_sha256,
            "pairing_bank_sha256": args._testing_pairing_contract[
                "pairing_bank_sha256"
            ],
            "pairing_date_inventory_sha256": args._testing_pairing_contract[
                "date_inventory_sha256"
            ],
            "pairing_segment": "testing",
            "time_schema": TIME_SCHEMA,
            "input_channels": 1,
        }
        mismatches = {
            key: (dataset.attrs.get(key), value)
            for key, value in expected_attrs.items()
            if dataset.attrs.get(key) != value
        }
        expected_sizes = {
            "test_ens": 1,
            "init_time": 1,
            "model": model_count,
            "valid_day_offset": 32,
        }
        size_mismatches = {
            dim: (dataset.sizes.get(dim), size)
            for dim, size in expected_sizes.items()
            if dataset.sizes.get(dim) != size
        }
        coordinate_ok = residual_coordinates_match(
            dataset,
            variable=variable,
            ensemble=ensemble,
            init_time=init_time,
            target_dates=target_dates,
            model_count=model_count,
            latitude=latitude,
            longitude=longitude,
        )
        residual_ok = (
            "residual" in dataset
            and dataset.residual.dims
            == (
                "test_ens",
                "init_time",
                "model",
                "valid_day_offset",
                "latitude",
                "longitude",
            )
            and np.isfinite(dataset.residual.values).all()
        )
        if mismatches or size_mismatches or (not coordinate_ok) or (not residual_ok):
            raise RuntimeError(
                f"Existing residual is incompatible with this run: {path}; attrs={mismatches}, sizes={size_mismatches}, coordinates_match={coordinate_ok}, residual_valid={residual_ok}. Use --overwrite explicitly."
            )


def load_model_bank():
    resolved = resolve_txt_model_bank(
        checkpoint_dir,
        variable=variable,
        version=args.ckpt_version,
        expected_model_count=100,
        expected_arch="sp_full",
        expected_normalizer="doy_full",
        expected_normalizer_provenance=normalizer_provenance("doy_full"),
        expected_pairing_digests=args._checkpoint_pairing_digests,
        expected_pairing_bank_sha256=args._checkpoint_pairing_contract[
            "pairing_bank_sha256"
        ],
    )
    models = []
    expected_provenance = normalizer_provenance("doy_full")
    for _, path in resolved:
        m = load_txt_checkpoint(
            path,
            expected_arch="sp_full",
            expected_training_variable=variable,
            expected_normalizer_provenance=expected_provenance,
            map_location=device,
        )
        m.eval()
        models.append(m)
    args._model_combos = [key for key, _ in resolved]
    args._model_checkpoints = [str(path) for _, path in resolved]
    args._checkpoint_bank_identity = checkpoint_bank_identity(resolved)
    print(f"Loaded {len(models)} models onto {device}")
    return models


def open_era5_target_years(years, *, eda):
    parts = []
    for year in years:
        path = f"{ERA5_ENS_DIR}/{year}.zarr" if eda else era5_zarr(year)
        part = xr.open_zarr(path)[field_name]
        if not eda and "number" in part.coords:
            part = part.drop_vars("number")
        parts.append(part.transpose(..., "latitude", "longitude"))
    combined = xr.concat(parts, dim="time").sortby("time").sortby("latitude")
    if pd.DatetimeIndex(combined.time.values).has_duplicates:
        raise ValueError(
            f"Duplicate ERA5 dates while concatenating years {list(years)}"
        )
    return combined


def main():
    from config import CONFIG
    from config import tranche_bounds

    _, _, val_s, val_e, _, _ = tranche_bounds()
    resid_lo, resid_hi = CONFIG["residual_years"]
    resid_years = list(range(resid_lo, resid_hi + 1))
    print("=" * 70)
    print(f"TxT Residual Computation — {variable.upper()} — years {resid_years}")
    print(f"IFS ensembles {args.ens_start}–{args.ens_end}, GPU {args.gpu_id}")
    print("=" * 70)
    norm = load_normalizer(field_name, mode="doy_full")
    norm_provenance = normalizer_provenance("doy_full")
    print(f"Normalizer: {norm.mode}  sig_scalar={norm.sig_scalar}")
    init_reference = xr.open_zarr(
        ifs_zarr(variable, 1), decode_timedelta=False, consolidated=False
    )[field_name]
    available_init = pd.DatetimeIndex(init_reference.init_time.values)
    available_init = available_init[
        (available_init >= val_s) & (available_init <= val_e)
    ]
    reference_name = "2m_temperature"
    reference = xr.open_zarr(
        ifs_zarr("t2m", 1), decode_timedelta=False, consolidated=False
    )[reference_name]
    expected_init = pd.DatetimeIndex(reference.init_time.values)
    expected_init = expected_init[(expected_init >= val_s) & (expected_init <= val_e)]
    if not available_init.equals(expected_init):
        missing_init = expected_init.difference(available_init)
        extra_init = available_init.difference(expected_init)
        raise RuntimeError(
            f"MSLP/T2M residual initialization axes differ: missing={missing_init[:10].strftime('%Y-%m-%d').tolist()}, extra={extra_init[:10].strftime('%Y-%m-%d').tolist()}. Finish and validate the raw-data archive before Phase C."
        )
    pairing_dir = PAIRING_DIR / args.ckpt_version
    checkpoint_pairings, args._checkpoint_pairing_contract = load_pairing_bank(
        pairing_dir
    )
    args._checkpoint_pairing_digests = {
        index: mapping_sha256(checkpoint_pairings[index]) for index in range(10)
    }
    args._testing_pairings, args._testing_pairing_contract = load_pairing_bank(
        pairing_dir, segment="testing", expected_dates=available_init
    )
    args._testing_pairing_lookup = generate_pairing_lookup(
        available_init, pairings=args._testing_pairings
    )
    print(
        f"Held-out residual pairing bank: {len(available_init)} dates, sha256={args._testing_pairing_contract['pairing_bank_sha256']}"
    )
    model_bank = load_model_bank()
    n_models = len(model_bank)
    for resid_year in resid_years:
        print(f"\n{'#' * 70}")
        print(f"#  Residual year: {resid_year}")
        print(f"{'#' * 70}")
        first_valid = max(pd.Timestamp(f"{resid_year}-01-01"), val_s)
        last_valid = min(pd.Timestamp(f"{resid_year}-12-31"), val_e)
        if first_valid > last_valid:
            print(f"  {resid_year}: outside residual tranche, skipping")
            continue
        init_times_year = available_init[
            (available_init >= first_valid) & (available_init <= last_valid)
        ]
        target_last = last_valid + pd.Timedelta(days=31)
        target_years = range(first_valid.year, target_last.year + 1)
        era5_eda_raw = canonicalize_spatial_grid(
            open_era5_target_years(target_years, eda=True), label="physical ERA5 EDA"
        )
        era5_eda_norm = canonicalize_spatial_grid(
            norm.normalize_era5(era5_eda_raw), label="normalized ERA5 EDA"
        ).load()
        era5_norm = era5_eda_norm.isel(number=0, drop=True)
        era5_hres_norm = None
        requested_ensembles = list(range(args.ens_start, args.ens_end + 1))
        print(
            f"\nPre-loading requested one-channel IFS {resid_year} ensembles: {requested_ensembles}"
        )
        all_ifs_norm = {}
        for e in requested_ensembles:
            ifs_raw_e = xr.open_zarr(ifs_zarr(variable, e), decode_timedelta=False)[
                field_name
            ]
            ifs_raw_e = ifs_raw_e.sortby("latitude").sortby("init_time")
            ifs_raw_e = ifs_raw_e.sel(init_time=slice(first_valid, last_valid))
            ifs_raw_e = canonicalize_ifs(ifs_raw_e, field_name)
            if not (
                np.array_equal(ifs_raw_e.latitude.values, era5_norm.latitude.values)
                and np.array_equal(
                    ifs_raw_e.longitude.values, era5_norm.longitude.values
                )
            ):
                raise RuntimeError(
                    f"IFS member {e} grid differs from the ERA5/climatology grid"
                )
            member_dates = pd.DatetimeIndex(ifs_raw_e.init_time.values)
            if not member_dates.equals(init_times_year):
                raise RuntimeError(
                    f"IFS member {e} dates differ from member 1 in {resid_year}"
                )
            all_ifs_norm[e] = norm.normalize_ifs(ifs_raw_e).load()
            print(f"  Ens {e}: shape={tuple(all_ifs_norm[e].shape)}")
        first_requested = requested_ensembles[0]
        day_offsets = all_ifs_norm[first_requested].valid_day_offset.values
        raw_start_hours = all_ifs_norm[first_requested].raw_lead_start_hour.values
        raw_end_hours = all_ifs_norm[first_requested].raw_lead_end_hour.values
        for ens in range(args.ens_start, args.ens_end + 1):
            print(f"\n[{resid_year}] Processing IFS ensemble {ens}")
            ifs_norm = all_ifs_norm[ens]
            init_times = pd.to_datetime(ifs_norm.init_time.values)
            dataset = ForecastInputDataset(ifs_norm)
            n_dates = len(dataset)
            print(
                f"  Dates: {n_dates}  ({init_times[0].date()} … {init_times[-1].date()})"
            )
            for i in tqdm(range(n_dates), desc=f"Ens {ens:02d} y{resid_year}"):
                current_date = init_times[i]
                date_str = current_date.strftime("%Y%m%d")
                out_stem = f"{residual_dir}/txt_uq_residuals_{variable}_{resid_year}_ens{ens:02d}_{date_str}"
                out_path = f"{out_stem}.zarr"
                target_dates = valid_dates(current_date)
                pairing_date_digest = date_mapping_sha256(
                    current_date, pairings=args._testing_pairings
                )
                pair = args._testing_pairing_lookup[date_str][ens]
                eda_logical_member = int(pair["era5_eda_member"])
                eda_storage_index = int(pair["era5_eda_storage_index"])
                perturbed_dataset_index = int(pair["perturbed_dataset_index"])
                pairing_mapping_digest = mapping_sha256(
                    args._testing_pairings[perturbed_dataset_index]
                )
                era5_target = era5_eda_norm.sel(number=eda_storage_index).sel(
                    time=target_dates
                )
                era5_target_phys = era5_eda_raw.sel(number=eda_storage_index).sel(
                    time=target_dates
                )
                if not np.array_equal(era5_target.time.values, target_dates.values):
                    raise ValueError(
                        f"ERA5 target dates do not exactly match init+0..31 for {date_str} ens{ens:02d} (ERA5 EDA logical {eda_logical_member}, storage {eda_storage_index})"
                    )
                if not np.array_equal(
                    era5_target_phys.time.values, target_dates.values
                ):
                    raise ValueError(
                        f"Physical ERA5 target dates do not match init+0..31 for {date_str} ens{ens:02d}"
                    )
                replace_existing = bool(args.overwrite)
                if os.path.exists(out_path) and (not args.overwrite):
                    try:
                        validate_existing_residual(
                            out_path,
                            ensemble=ens,
                            init_time=current_date,
                            target_dates=target_dates,
                            target_eda_logical_member=eda_logical_member,
                            target_eda_storage_index=eda_storage_index,
                            perturbed_dataset_index=perturbed_dataset_index,
                            pairing_mapping_sha256=pairing_mapping_digest,
                            pairing_date_mapping_sha256=pairing_date_digest,
                            model_count=n_models,
                            normalizer_metadata=norm_provenance,
                            latitude=era5_norm.latitude.values,
                            longitude=era5_norm.longitude.values,
                        )
                    except Exception as exc:
                        if not args.repair_incompatible_artifacts:
                            raise
                        print(f"  Rebuilding incompatible residual {out_path}: {exc}")
                        replace_existing = True
                    else:
                        continue
                x_tensor, _ = dataset[i]
                x = x_tensor.unsqueeze(0).to(device)
                target_phys = era5_target_phys.values.astype(np.float32, copy=False)
                n_days, height, width = target_phys.shape
                predictions_z = torch.empty(
                    (n_models, n_days, height, width),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    for k, model in enumerate(model_bank):
                        yhat = model.predict_corrected(x)
                        predictions_z[k] = yhat.squeeze(0).squeeze(0)
                residuals_np = compute_physical_residuals(
                    norm,
                    predictions_z.cpu().numpy(),
                    target_phys,
                    current_date,
                    day_offsets,
                )
                ds_out = xr.Dataset(
                    data_vars={
                        "residual": (
                            (
                                "test_ens",
                                "init_time",
                                "model",
                                "valid_day_offset",
                                "latitude",
                                "longitude",
                            ),
                            residuals_np[np.newaxis, np.newaxis],
                        )
                    },
                    coords={
                        "test_ens": [ens],
                        "init_time": [current_date],
                        "model": np.arange(n_models),
                        "valid_day_offset": day_offsets,
                        "valid_date": ("valid_day_offset", target_dates.values),
                        "raw_lead_start_hour": ("valid_day_offset", raw_start_hours),
                        "raw_lead_end_hour": ("valid_day_offset", raw_end_hours),
                        "latitude": era5_norm.latitude.values,
                        "longitude": era5_norm.longitude.values,
                    },
                    attrs={
                        "var": variable,
                        "ckpt_prefix": f"{variable}_{args.ckpt_version}",
                        "architecture": "sp_full",
                        "model_count": n_models,
                        "model_combo_order": ",".join(
                            (
                                f"{eda:02d}-{weight:02d}"
                                for eda, weight in args._model_combos
                            )
                        ),
                        "checkpoint_bank_identity": args._checkpoint_bank_identity,
                        "residual_year": int(resid_year),
                        "normalizer": "doy_full",
                        "normalizer_provenance": norm_provenance,
                        "space": "physical",
                        "units": physical_units,
                        "residual_transfer_space": RESIDUAL_TRANSFER_SPACE,
                        "target_era5_eda_logical_member": eda_logical_member,
                        "target_era5_eda_storage_index": eda_storage_index,
                        "paired_perturbed_dataset_index": perturbed_dataset_index,
                        "pairing_schema": PAIRING_SCHEMA,
                        "pairing_seed": PAIRING_SEED,
                        "pairing_mapping_sha256": pairing_mapping_digest,
                        "pairing_date_mapping_sha256": pairing_date_digest,
                        "pairing_bank_sha256": args._testing_pairing_contract[
                            "pairing_bank_sha256"
                        ],
                        "pairing_date_inventory_sha256": args._testing_pairing_contract[
                            "date_inventory_sha256"
                        ],
                        "pairing_segment": "testing",
                        "time_schema": TIME_SCHEMA,
                        "input_channels": 1,
                    },
                )
                write_zarr_dataset(
                    ds_out, out_path, ["residual"], overwrite=replace_existing
                )
                del x, predictions_z, residuals_np
                torch.cuda.empty_cache()
        del all_ifs_norm, era5_norm, era5_eda_norm, era5_eda_raw
    print("\n" + "=" * 70)
    print("Residual computation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
