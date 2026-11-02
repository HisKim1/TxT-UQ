from __future__ import annotations
from config import ARTIFACT_SCHEMAS
from collections.abc import Iterable
import numpy as np
import pandas as pd
import xarray as xr

TIME_SCHEMA = ARTIFACT_SCHEMAS["forecast_time"]
N_VALID_DAYS = 32
VALID_DAY_OFFSETS = np.arange(N_VALID_DAYS, dtype=np.int16)
SCORED_DAY_SLICE = slice(18, 32)
WEEK_DAY_SLICES = {1: slice(18, 25), 2: slice(25, 32)}
CANONICAL_LATITUDE = np.linspace(-90.0, 90.0, 121)
CANONICAL_LONGITUDE = np.arange(0.0, 360.0, 1.5)
GRID_ATOL = 1e-10
_VAR_ALIASES = {
    "mslp": "mslp",
    "mean_sea_level_pressure": "mslp",
    "t2m": "t2m",
    "tas": "t2m",
    "2m_temperature": "t2m",
    "2_m_temperature": "t2m",
}


def variable_key(variable: str) -> str:
    try:
        return _VAR_ALIASES[str(variable)]
    except KeyError as exc:
        raise ValueError(f"Unsupported TxT variable: {variable!r}") from exc


def _lead_values_to_hours(values: Iterable) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.timedelta64):
        hours = arr / np.timedelta64(1, "h")
    else:
        hours = arr.astype(np.float64)
    rounded = np.rint(hours).astype(np.int64)
    if not np.allclose(hours, rounded, rtol=0.0, atol=1e-06):
        raise ValueError(f"lead_time contains non-integral hours: {arr!r}")
    return rounded


def expected_raw_lead_end_hours(variable: str) -> np.ndarray:
    key = variable_key(variable)
    if key == "mslp":
        return np.arange(0, 769, 24, dtype=np.int64)
    return np.arange(24, 769, 24, dtype=np.int64)


def raw_interval_hours(variable: str) -> tuple[np.ndarray, np.ndarray]:
    key = variable_key(variable)
    start = np.arange(0, 768, 24, dtype=np.int64)
    if key == "mslp":
        return (start, start.copy())
    return (start, start + 24)


def canonicalize_spatial_grid(data: xr.DataArray, *, label: str) -> xr.DataArray:
    if "latitude" not in data.dims or "longitude" not in data.dims:
        raise ValueError(f"{label} lacks latitude/longitude dimensions: {data.dims}")
    out = data.sortby("latitude").sortby("longitude")
    latitude = np.asarray(out.latitude.values, dtype=np.float64)
    longitude = np.asarray(out.longitude.values, dtype=np.float64)
    if (
        latitude.shape != CANONICAL_LATITUDE.shape
        or longitude.shape != CANONICAL_LONGITUDE.shape
        or (not np.allclose(latitude, CANONICAL_LATITUDE, rtol=0.0, atol=GRID_ATOL))
        or (not np.allclose(longitude, CANONICAL_LONGITUDE, rtol=0.0, atol=GRID_ATOL))
    ):
        raise ValueError(f"{label} is not the canonical 121x240 global 1.5-degree grid")
    return out.assign_coords(latitude=CANONICAL_LATITUDE, longitude=CANONICAL_LONGITUDE)


def canonicalize_ifs(ifs: xr.DataArray, variable: str) -> xr.DataArray:
    ifs = canonicalize_spatial_grid(ifs, label=f"IFS {variable_key(variable)}")
    if "lead_time" not in ifs.dims:
        if "valid_day_offset" in ifs.dims:
            validate_forecast_time_axis(ifs)
            expected_dims = {"init_time", "valid_day_offset", "latitude", "longitude"}
            if set(ifs.dims) != expected_dims:
                raise ValueError(
                    f"Canonical IFS field must have exactly {sorted(expected_dims)}, got {ifs.dims}"
                )
            return ifs.transpose(
                "init_time", "valid_day_offset", "latitude", "longitude"
            )
        raise ValueError(f"IFS data must have a lead_time dimension, got {ifs.dims}")
    expected_raw_dims = {"init_time", "lead_time", "latitude", "longitude"}
    if set(ifs.dims) != expected_raw_dims:
        raise ValueError(
            f"Raw IFS field must have exactly {sorted(expected_raw_dims)}, got {ifs.dims}"
        )
    key = variable_key(variable)
    actual = _lead_values_to_hours(ifs["lead_time"].values)
    expected = expected_raw_lead_end_hours(key)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(
            f"Unexpected {key} raw lead_time axis: expected {expected.tolist()}, got {actual.tolist()}"
        )
    expected_step_type = "instant" if key == "mslp" else "avg"
    step_type = ifs.attrs.get("GRIB_stepType") or ifs.attrs.get("stepType")
    if step_type is None:
        raise ValueError(f"{key} source field lacks GRIB_stepType/stepType provenance")
    if str(step_type) != expected_step_type:
        raise ValueError(
            f"Unexpected {key} stepType {step_type!r}; expected {expected_step_type!r}"
        )
    raw_start, raw_end = raw_interval_hours(key)
    out = ifs.isel(lead_time=slice(0, N_VALID_DAYS)).rename(
        {"lead_time": "valid_day_offset"}
    )
    out = out.transpose("init_time", "valid_day_offset", "latitude", "longitude")
    out = out.assign_coords(
        valid_day_offset=VALID_DAY_OFFSETS,
        raw_lead_start_hour=("valid_day_offset", raw_start.astype(np.int16)),
        raw_lead_end_hour=("valid_day_offset", raw_end.astype(np.int16)),
    )
    out.attrs = dict(out.attrs)
    out.attrs.update(
        time_schema=TIME_SCHEMA,
        raw_step_type=expected_step_type,
        valid_day_definition="calendar day init_time + valid_day_offset",
    )
    validate_forecast_time_axis(out)
    return out


def validate_forecast_time_axis(data: xr.DataArray) -> None:
    if "valid_day_offset" not in data.dims:
        raise ValueError(f"Missing valid_day_offset dimension: {data.dims}")
    offsets = np.asarray(data["valid_day_offset"].values, dtype=np.int64)
    if not np.array_equal(offsets, VALID_DAY_OFFSETS.astype(np.int64)):
        raise ValueError(f"valid_day_offset must be 0..31, got {offsets.tolist()}")


def valid_dates(
    init_time, offsets: Iterable[int] = VALID_DAY_OFFSETS
) -> pd.DatetimeIndex:
    init = pd.Timestamp(init_time)
    offsets_arr = np.asarray(offsets, dtype=np.int64).ravel()
    return pd.DatetimeIndex(init + pd.to_timedelta(offsets_arr, unit="D"))


def valid_dates_2d(
    init_times: Iterable, offsets: Iterable[int] = VALID_DAY_OFFSETS
) -> np.ndarray:
    init = pd.DatetimeIndex(init_times).values.astype("datetime64[ns]")[:, None]
    day = np.asarray(offsets, dtype=np.int64)[None, :].astype("timedelta64[D]")
    return init + day


def climatology_dayofyear(times: Iterable) -> np.ndarray:
    values = np.asarray(times)
    shape = values.shape
    flat = pd.DatetimeIndex(values.reshape(-1))
    anchor = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": np.full(len(flat), 2000, dtype=np.int16),
                "month": flat.month.to_numpy(),
                "day": flat.day.to_numpy(),
            }
        )
    )
    return anchor.dayofyear.to_numpy(dtype=np.int16).reshape(shape)


def build_target_index_matrix(init_times: Iterable, era5_times: Iterable) -> np.ndarray:
    era5_index = pd.DatetimeIndex(era5_times)
    if era5_index.has_duplicates:
        duplicated = era5_index[era5_index.duplicated()].unique()
        raise ValueError(
            f"ERA5 time coordinate contains duplicates: {duplicated[:5].tolist()}"
        )
    expected = valid_dates_2d(init_times)
    flat = pd.DatetimeIndex(expected.ravel())
    positions = era5_index.get_indexer(flat)
    if np.any(positions < 0):
        missing = flat[positions < 0].unique()
        preview = ", ".join((str(pd.Timestamp(v).date()) for v in missing[:10]))
        raise KeyError(
            f"ERA5 is missing {len(missing)} required target date(s): {preview}"
        )
    return positions.reshape(expected.shape).astype(np.int64)


def week_index_array(week: int) -> np.ndarray:
    try:
        day_slice = WEEK_DAY_SLICES[int(week)]
    except KeyError as exc:
        raise ValueError(f"week must be 1 or 2, got {week!r}") from exc
    return np.arange(day_slice.start, day_slice.stop, dtype=np.int64)


def expected_thursday_initializations(
    start, end, year: int | None = None
) -> pd.DatetimeIndex:
    lower = pd.Timestamp(start).normalize()
    upper = pd.Timestamp(end).normalize()
    if year is not None:
        lower = max(lower, pd.Timestamp(year=int(year), month=1, day=1))
        upper = min(upper, pd.Timestamp(year=int(year), month=12, day=31))
    if lower > upper:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.date_range(lower, upper, freq="W-THU")
