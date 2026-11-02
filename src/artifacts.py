from __future__ import annotations
import itertools
import hashlib
import re
from collections.abc import Mapping
import numpy as np
import pandas as pd
from config import PAIRING_DIR
from pairing import (
    PAIRING_SCHEMA,
    PAIRING_SEED,
    date_mapping_sha256,
    date_inventory_sha256,
    load_pairing_bank,
    mapping_sha256,
)
from time_coordinates import raw_interval_hours, valid_dates, valid_dates_2d

ANALOG_DISTANCE_SPACE = "doy_full_gridpoint_standardized_anomaly_v1"
SAMPLE_PAYLOAD_SHA256_ATTR = "sample_payload_sha256"


def sample_payload_sha256(dataset) -> str:
    digest = hashlib.sha256()
    for kind, names in (
        ("data_var", sorted(dataset.data_vars)),
        ("coord", sorted(dataset.coords)),
    ):
        for name in names:
            variable = dataset[name]
            values = np.asarray(variable.values)
            if values.dtype.hasobject:
                raise RuntimeError(f"Cannot content-hash object array {kind} {name!r}")
            header = (
                kind,
                name,
                tuple(variable.dims),
                tuple((int(size) for size in values.shape)),
                values.dtype.str,
            )
            digest.update(repr(header).encode("utf-8"))
            if values.ndim == 0:
                raw = np.ascontiguousarray(values).view(np.uint8)
                digest.update(memoryview(raw))
                continue
            bytes_per_row = max(1, int(values[0:1].nbytes))
            rows_per_chunk = max(1, 8 * 1024 * 1024 // bytes_per_row)
            for start in range(0, len(values), rows_per_chunk):
                block = np.ascontiguousarray(values[start : start + rows_per_chunk])
                digest.update(memoryview(block.view(np.uint8)))
    return digest.hexdigest()


def _exact_integer_array(values, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
    ):
        raise RuntimeError(f"{label} must contain exact integers")
    numeric = array.astype(np.float64, copy=False)
    if (
        not np.isfinite(numeric).all()
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(np.abs(numeric) > np.iinfo(np.int64).max)
    ):
        raise RuntimeError(f"{label} must contain finite exact integers")
    return numeric.astype(np.int64)


def _exact_integer_scalar(value, *, label: str) -> int:
    array = _exact_integer_array(value, label=label)
    if array.size != 1:
        raise RuntimeError(f"{label} must be one exact integer")
    return int(array.reshape(-1)[0])


def residual_coordinates_match(
    dataset,
    *,
    variable: str,
    ensemble: int,
    init_time,
    target_dates,
    model_count: int,
    latitude,
    longitude,
) -> bool:
    required = {
        "test_ens",
        "init_time",
        "model",
        "valid_day_offset",
        "valid_date",
        "raw_lead_start_hour",
        "raw_lead_end_hour",
        "latitude",
        "longitude",
    }
    if not required.issubset(dataset.coords):
        return False
    raw_start, raw_end = raw_interval_hours(variable)
    expected_dims = {
        "test_ens": ("test_ens",),
        "init_time": ("init_time",),
        "model": ("model",),
        "valid_day_offset": ("valid_day_offset",),
        "valid_date": ("valid_day_offset",),
        "raw_lead_start_hour": ("valid_day_offset",),
        "raw_lead_end_hour": ("valid_day_offset",),
        "latitude": ("latitude",),
        "longitude": ("longitude",),
    }
    return bool(
        all((dataset[name].dims == dims for name, dims in expected_dims.items()))
        and np.array_equal(dataset.test_ens.values, [ensemble])
        and np.array_equal(
            pd.DatetimeIndex(dataset.init_time.values).values,
            pd.DatetimeIndex([init_time]).values,
        )
        and np.array_equal(dataset.model.values, np.arange(model_count))
        and np.array_equal(dataset.valid_day_offset.values, np.arange(32))
        and np.array_equal(
            pd.DatetimeIndex(dataset.valid_date.values).values,
            pd.DatetimeIndex(target_dates).values,
        )
        and np.array_equal(dataset.raw_lead_start_hour.values.reshape(-1), raw_start)
        and np.array_equal(dataset.raw_lead_end_hour.values.reshape(-1), raw_end)
        and np.array_equal(dataset.latitude.values, latitude)
        and np.array_equal(dataset.longitude.values, longitude)
    )


def validate_prediction_dataset(
    dataset,
    *,
    path: str,
    variable: str,
    expected_attrs: dict,
    input_member: int,
    latitude,
    longitude,
    expected_init_times=None,
    check_finite: bool = True,
):
    mismatches = {
        key: (dataset.attrs.get(key), expected)
        for key, expected in expected_attrs.items()
        if dataset.attrs.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Prediction provenance mismatch in {path}: {mismatches}")
    required_vars = {"y_pred", "ifs_input"}
    if set(dataset.data_vars) != required_vars:
        raise RuntimeError(
            f"Prediction variables mismatch in {path}: expected {sorted(required_vars)}, got {sorted(dataset.data_vars)}"
        )
    singleton_coords = {"input": input_member, "model": 0, "weight": 0}
    for dim, expected_value in singleton_coords.items():
        actual = (
            np.asarray(dataset[dim].values).reshape(-1)
            if dim in dataset.coords
            else np.asarray([])
        )
        if dataset.sizes.get(dim) != 1 or not np.array_equal(
            actual, np.asarray([expected_value])
        ):
            raise RuntimeError(f"Prediction {dim} coordinate mismatch in {path}")
    init_values = pd.DatetimeIndex(dataset.init_time.values)
    if (
        init_values.hasnans
        or init_values.has_duplicates
        or (not init_values.is_monotonic_increasing)
    ):
        raise RuntimeError(
            f"Prediction init_time must be finite, unique, and sorted in {path}"
        )
    if expected_init_times is not None and (
        not np.array_equal(
            init_values.values, pd.DatetimeIndex(expected_init_times).values
        )
    ):
        raise RuntimeError(f"Prediction init_time inventory mismatch in {path}")
    raw_start, raw_end = raw_interval_hours(variable)
    expected_dims = (
        "input",
        "model",
        "weight",
        "init_time",
        "valid_day_offset",
        "latitude",
        "longitude",
    )
    expected_shape = (1, 1, 1, len(init_values), 32, len(latitude), len(longitude))
    fields_ok = all(
        (
            dataset[name].dims == expected_dims
            and dataset[name].shape == expected_shape
            for name in required_vars
        )
    )
    coords_ok = (
        {"valid_date", "raw_lead_start_hour", "raw_lead_end_hour"}.issubset(
            dataset.coords
        )
        and dataset.valid_date.dims == ("init_time", "valid_day_offset")
        and (dataset.raw_lead_start_hour.dims == ("valid_day_offset",))
        and (dataset.raw_lead_end_hour.dims == ("valid_day_offset",))
        and np.array_equal(dataset.valid_day_offset.values, np.arange(32))
        and np.array_equal(
            dataset.valid_date.values.reshape(-1, 32), valid_dates_2d(init_values)
        )
        and np.array_equal(dataset.raw_lead_start_hour.values.reshape(-1), raw_start)
        and np.array_equal(dataset.raw_lead_end_hour.values.reshape(-1), raw_end)
        and np.array_equal(dataset.latitude.values, latitude)
        and np.array_equal(dataset.longitude.values, longitude)
    )
    if not fields_ok or not coords_ok:
        raise RuntimeError(f"Prediction schema/coordinate mismatch in {path}")
    if check_finite:
        nonfinite = [
            name
            for name in required_vars
            if not np.isfinite(dataset[name].values).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Prediction contains non-finite values in {nonfinite}: {path}"
            )
    return dataset.isel(input=0, model=0, weight=0, drop=True)


def validate_sample_dataset(
    dataset,
    *,
    path: str,
    variable: str,
    expected_attrs: dict,
    input_member: int,
    init_time,
    model_indices: list[int],
    residual_pool_indices: list[int],
    k_nearest: int,
    latitude,
    longitude,
    residual_pairings: Mapping | None = None,
    check_finite: bool = False,
) -> str:
    mismatches = {
        key: (dataset.attrs.get(key), expected)
        for key, expected in expected_attrs.items()
        if dataset.attrs.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Sample provenance mismatch in {path}: {mismatches}")
    bank_identity = dataset.attrs.get("checkpoint_bank_identity")
    if not isinstance(bank_identity, str) or len(bank_identity) != 64:
        raise RuntimeError(f"Sample lacks checkpoint bank identity: {path}")
    init_time = pd.Timestamp(init_time)
    raw_start, raw_end = raw_interval_hours(variable)
    required_coords = {
        "sample_model_index",
        "sample_analog_rank",
        "valid_date",
        "raw_lead_start_hour",
        "raw_lead_end_hour",
    }
    coordinate_ok = (
        required_coords.issubset(dataset.coords)
        and dataset.sizes.get("input") == 1
        and np.array_equal(
            np.asarray(dataset.input.values).reshape(-1), np.asarray([input_member])
        )
        and (dataset.sizes.get("init_time") == 1)
        and np.array_equal(
            pd.DatetimeIndex(dataset.init_time.values).values,
            pd.DatetimeIndex([init_time]).values,
        )
        and np.array_equal(dataset.valid_day_offset.values, np.arange(32))
        and np.array_equal(
            dataset.valid_date.values.reshape(-1), valid_dates(init_time).values
        )
        and np.array_equal(dataset.raw_lead_start_hour.values.reshape(-1), raw_start)
        and np.array_equal(dataset.raw_lead_end_hour.values.reshape(-1), raw_end)
        and np.array_equal(dataset.latitude.values, latitude)
        and np.array_equal(dataset.longitude.values, longitude)
    )
    if not coordinate_ok:
        raise RuntimeError(f"Sample coordinate contract mismatch: {path}")
    analog_count = len(residual_pool_indices) * k_nearest
    samples_per_model = analog_count
    expected_sample_count = len(model_indices) * samples_per_model
    sample_count = _exact_integer_scalar(
        dataset.attrs.get("sample_count", -1), label="sample_count"
    )
    stored_analog_count = _exact_integer_scalar(
        dataset.attrs.get("analog_count", -1), label="analog_count"
    )
    if (
        stored_analog_count != analog_count
        or dataset.sizes.get("analog", 0) != analog_count
        or sample_count != expected_sample_count
        or (dataset.sizes.get("sample") != expected_sample_count)
    ):
        raise RuntimeError(
            f"Sample/analog count mismatch in {path}: sample={sample_count}, analog={dataset.attrs.get('analog_count')}, expected=({expected_sample_count}, {analog_count})"
        )
    model_values = _exact_integer_array(
        dataset.sample_model_index.values.ravel(), label="sample_model_index"
    )
    analog_values = _exact_integer_array(
        dataset.sample_analog_rank.values.ravel(), label="sample_analog_rank"
    )
    actual_pairs = np.column_stack([model_values, analog_values])
    expected_pairs = np.asarray(
        list(itertools.product(model_indices, range(analog_count))), dtype=int
    )
    actual_pairs = actual_pairs[np.lexsort((actual_pairs[:, 1], actual_pairs[:, 0]))]
    expected_pairs = expected_pairs[
        np.lexsort((expected_pairs[:, 1], expected_pairs[:, 0]))
    ]
    if not np.array_equal(actual_pairs, expected_pairs):
        raise RuntimeError(f"Sample model-analog Cartesian identity mismatch: {path}")
    analog_vars = {
        "analog_test_member",
        "analog_init_time",
        "analog_target_era5_eda_logical_member",
        "analog_target_era5_eda_storage_index",
        "analog_perturbed_dataset_index",
        "analog_rmse",
    }
    if not analog_vars.issubset(dataset.data_vars):
        raise RuntimeError(f"Sample lacks analog provenance variables: {path}")
    analog_member = _exact_integer_array(
        dataset.analog_test_member.values.ravel(), label="analog_test_member"
    )
    analog_date = pd.DatetimeIndex(dataset.analog_init_time.values.ravel())
    analog_target_logical = _exact_integer_array(
        dataset.analog_target_era5_eda_logical_member.values.ravel(),
        label="analog_target_era5_eda_logical_member",
    )
    analog_target_storage = _exact_integer_array(
        dataset.analog_target_era5_eda_storage_index.values.ravel(),
        label="analog_target_era5_eda_storage_index",
    )
    analog_dataset_index = _exact_integer_array(
        dataset.analog_perturbed_dataset_index.values.ravel(),
        label="analog_perturbed_dataset_index",
    )
    analog_rmse = dataset.analog_rmse.values.ravel().astype(float)
    expected_schema = PAIRING_SCHEMA
    expected_seed = PAIRING_SEED
    if (
        dataset.attrs.get("residual_pairing_schema") != expected_schema
        or dataset.attrs.get("residual_pairing_seed") != expected_seed
    ):
        raise RuntimeError(f"Sample residual permutation contract mismatch: {path}")
    if residual_pairings is None:
        version = expected_attrs.get("ckpt_version")
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"Live testing pairing bank is required: {path}")
        residual_pairings, _ = load_pairing_bank(
            PAIRING_DIR / version, segment="testing"
        )
    pairing_dates = pd.to_datetime(sorted(residual_pairings[0]), format="%Y%m%d")
    expected_pairing_attrs = {
        "residual_pairing_bank_sha256": mapping_sha256(residual_pairings),
        "residual_pairing_date_inventory_sha256": date_inventory_sha256(pairing_dates),
        "residual_pairing_segment": "testing",
    }
    pairing_attr_mismatch = {
        key: (dataset.attrs.get(key), value)
        for key, value in expected_pairing_attrs.items()
        if dataset.attrs.get(key) != value
    }
    if pairing_attr_mismatch:
        raise RuntimeError(
            f"Sample testing pairing-bank mismatch in {path}: {pairing_attr_mismatch}"
        )
    digest_text = dataset.attrs.get("residual_pairing_date_sha256s", "")
    pool_counts = {
        pool: int(np.sum(analog_dataset_index == pool))
        for pool in residual_pool_indices
    }
    rmse_ordered = all(
        (
            np.all(np.diff(analog_rmse[analog_dataset_index == pool]) >= 0.0)
            for pool in residual_pool_indices
        )
    )
    analog_ok = (
        sorted(np.unique(analog_dataset_index).tolist()) == residual_pool_indices
        and all((count == k_nearest for count in pool_counts.values()))
        and rmse_ordered
        and (not analog_date.hasnans)
        and (
            len(set(zip(analog_dataset_index.tolist(), analog_date.values.tolist())))
            == analog_count
        )
        and np.isfinite(analog_rmse).all()
        and np.all(analog_rmse >= 0)
    )
    digests = sorted(digest_text.split(",")) if digest_text else []
    expected_digests = sorted(
        {date_mapping_sha256(date, pairings=residual_pairings) for date in analog_date}
    )
    analog_ok = (
        analog_ok
        and bool(residual_pool_indices)
        and (residual_pool_indices == sorted(set(residual_pool_indices)))
        and all((pool in range(10) for pool in residual_pool_indices))
        and np.all((analog_member >= 1) & (analog_member <= 10))
        and (digests == expected_digests)
        and np.all((analog_target_logical >= 1) & (analog_target_logical <= 10))
        and np.all((analog_target_storage >= 0) & (analog_target_storage < 10))
        and np.all((analog_dataset_index >= 0) & (analog_dataset_index < 10))
    )
    for member, date, logical, storage, dataset_index in zip(
        analog_member,
        analog_date,
        analog_target_logical,
        analog_target_storage,
        analog_dataset_index,
    ):
        key = date.strftime("%Y%m%d")
        expected_pair = residual_pairings[dataset_index][key]
        analog_ok = (
            analog_ok
            and member == expected_pair["ifs_member"]
            and (logical == expected_pair["era5_eda_member"])
            and (storage == expected_pair["era5_eda_member"] - 1)
        )
    if not analog_ok:
        raise RuntimeError(f"Sample analog identity mismatch: {path}")
    required_samples = {"samples_w1", "samples_w2"}
    expected_sample_dims = ("input", "init_time", "sample", "latitude", "longitude")
    expected_sample_shape = (1, 1, expected_sample_count, len(latitude), len(longitude))
    sample_fields_ok = required_samples.issubset(dataset.data_vars) and all(
        (
            dataset[name].dims == expected_sample_dims
            and dataset[name].shape == expected_sample_shape
            for name in required_samples
        )
    )
    if not sample_fields_ok:
        raise RuntimeError(f"Sample fields have an invalid schema: {path}")
    expected_analog_dims = ("input", "init_time", "analog")
    if any((dataset[name].dims != expected_analog_dims for name in analog_vars)):
        raise RuntimeError(f"Analog fields have an invalid schema: {path}")
    if "pdf_mean" in dataset.data_vars:
        expected_pdf_dims = (
            "input",
            "init_time",
            "valid_day_offset",
            "latitude",
            "longitude",
        )
        expected_pdf_shape = (1, 1, 32, len(latitude), len(longitude))
        if (
            dataset.pdf_mean.dims != expected_pdf_dims
            or dataset.pdf_mean.shape != expected_pdf_shape
        ):
            raise RuntimeError(f"pdf_mean has an invalid schema: {path}")
    if check_finite:
        fields = list(required_samples)
        if "pdf_mean" in dataset.data_vars:
            fields.append("pdf_mean")
        nonfinite = [
            name for name in fields if not np.isfinite(dataset[name].values).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Sample contains non-finite values in {nonfinite}: {path}"
            )
    stored_payload_sha256 = dataset.attrs.get(SAMPLE_PAYLOAD_SHA256_ATTR)
    if (
        not isinstance(stored_payload_sha256, str)
        or re.fullmatch("[0-9a-f]{64}", stored_payload_sha256) is None
    ):
        raise RuntimeError(f"Sample lacks a valid payload digest: {path}")
    actual_payload_sha256 = sample_payload_sha256(dataset)
    if stored_payload_sha256 != actual_payload_sha256:
        raise RuntimeError(
            f"Sample payload digest mismatch in {path}: stored={stored_payload_sha256}, actual={actual_payload_sha256}"
        )
    return bank_identity
