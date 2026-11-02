import argparse
import hashlib
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import ERA5_DOY_CLIM_ZARR, climatology_content_sha256, era5_zarr
from io_utils import atomic_write_zarr
from time_coordinates import canonicalize_spatial_grid, climatology_dayofyear

VARS = ("mean_sea_level_pressure", "2m_temperature")


def _load_var_as_numpy(year, var):
    ds = xr.open_zarr(era5_zarr(year))[var]
    if "number" in ds.coords:
        ds = ds.drop_vars("number")
    ds = canonicalize_spatial_grid(ds, label=f"ERA5 {year} {var}")
    ds = ds.transpose("time", "latitude", "longitude")
    ds = ds.load()
    return (
        ds.values.astype(np.float32),
        pd.to_datetime(ds.time.values),
        ds.latitude.values,
        ds.longitude.values,
    )


def _doy_moments_numpy(values, times):
    n_days, height, width = values.shape
    idx = climatology_dayofyear(times).astype(np.int32) - 1
    n_doy = 366
    count = np.bincount(idx, minlength=n_doy).astype(np.int64)
    flat = values.reshape(n_days, -1)
    sum_ = np.zeros((n_doy, height * width), dtype=np.float64)
    sumsq = np.zeros((n_doy, height * width), dtype=np.float64)
    for t in range(n_days):
        d = idx[t]
        sum_[d] += flat[t]
        sumsq[d] += flat[t].astype(np.float64) ** 2
    return (
        sum_.reshape(n_doy, height, width),
        sumsq.reshape(n_doy, height, width),
        count,
    )


def _validate_annual_input(year, values, times, latitude, longitude):
    times = pd.DatetimeIndex(times)
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise RuntimeError(f"ERA5 {year} time coordinate is duplicated or unsorted")
    if not times.equals(expected):
        missing = expected.difference(times)
        extra = times.difference(expected)
        raise RuntimeError(
            f"ERA5 {year} must contain every calendar day exactly once: missing={missing[:10].strftime('%Y-%m-%d').tolist()}, extra={extra[:10].strftime('%Y-%m-%d').tolist()}"
        )
    expected_shape = (len(expected), len(latitude), len(longitude))
    if values.shape != expected_shape:
        raise RuntimeError(
            f"ERA5 {year} shape {values.shape} does not match {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"ERA5 {year} contains non-finite values")


def _circular_window_sum(arr, half_width):
    values = np.asarray(arr, dtype=np.float64)
    if half_width == 0:
        return values.copy()
    padded = np.concatenate([values[-half_width:], values, values[:half_width]], axis=0)
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    zero = np.zeros((1,) + padded.shape[1:], dtype=np.float64)
    cumulative = np.concatenate([zero, cumulative], axis=0)
    window_length = 2 * half_width + 1
    return cumulative[window_length:] - cumulative[:-window_length]


def _circular_window_stats(sum_, sumsq, count, half_width):
    count_window = _circular_window_sum(count.astype(np.float64), half_width)
    sum_window = _circular_window_sum(sum_, half_width)
    sumsq_window = _circular_window_sum(sumsq, half_width)
    denominator = np.maximum(count_window[:, None, None], 1.0)
    mean = sum_window / denominator
    variance = np.clip(sumsq_window / denominator - mean**2, 0.0, None)
    return (mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count_window)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=1991)
    ap.add_argument(
        "--end-year", type=int, default=2020, help="Inclusive TxT-UQ baseline end year."
    )
    ap.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Half-width of the pooled circular calendar window.",
    )
    ap.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Override output zarr path (default: ERA5_DIR/era5_doy_clim_{start}_{end}.zarr)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    required_recipe = (1991, 2020, 15)
    actual_recipe = (args.start_year, args.end_year, args.smooth_window)
    if actual_recipe != required_recipe:
        ap.error(
            f"TxT-UQ climatology is fixed to start-year=1991, end-year=2020, and smooth-window=15. Received {actual_recipe}."
        )
    years = list(range(args.start_year, args.end_year + 1))
    print(f"Period: {years[0]}–{years[-1]} ({len(years)} years)", flush=True)
    print(f"Pooled calendar window: ±{args.smooth_window} days", flush=True)
    window_days = 2 * args.smooth_window + 1
    out_path = args.out_path or ERA5_DOY_CLIM_ZARR
    print(f"Output: {out_path}", flush=True)
    if Path(out_path).exists() and (not args.overwrite):
        raise SystemExit(f"{out_path} exists — pass --overwrite to replace")
    result_vars = {}
    lat = lon = None
    times_by_year = {}
    for var in VARS:
        print(f"\n=== {var} ===", flush=True)
        t0 = time.time()
        year_arrays = []
        year_times = []
        for y in years:
            vals, tms, la, lo = _load_var_as_numpy(y, var)
            _validate_annual_input(y, vals, tms, la, lo)
            if lat is None:
                lat, lon = (la, lo)
            elif not (np.array_equal(la, lat) and np.array_equal(lo, lon)):
                raise RuntimeError(f"ERA5 grid changed for {var} in {y}")
            if y in times_by_year and (
                not pd.DatetimeIndex(tms).equals(times_by_year[y])
            ):
                raise RuntimeError(f"ERA5 variables use different dates in {y}")
            times_by_year[y] = pd.DatetimeIndex(tms)
            year_arrays.append(vals)
            year_times.append(tms)
        values = np.concatenate(year_arrays, axis=0)
        times = pd.DatetimeIndex(np.concatenate(year_times))
        del year_arrays, year_times
        print(
            f"  Loaded {len(years)} years in {time.time() - t0:.1f}s: shape {values.shape}, memory {values.nbytes / 1000000000.0:.2f} GB",
            flush=True,
        )
        t1 = time.time()
        sum_raw, sumsq_raw, counts = _doy_moments_numpy(values, times)
        print(f"  Per-DOY moment reduction in {time.time() - t1:.1f}s", flush=True)
        print(
            f"  Exact month/day counts: min={int(counts.min())}, max={int(counts.max())}",
            flush=True,
        )
        del values, times
        t2 = time.time()
        mean_s, std_s, window_counts = _circular_window_stats(
            sum_raw, sumsq_raw, counts, args.smooth_window
        )
        print(
            f"  Circular pooled-window statistics in {time.time() - t2:.1f}s",
            flush=True,
        )
        print(
            f"  Window sample counts: min={int(window_counts.min())}, max={int(window_counts.max())}",
            flush=True,
        )
        del sum_raw, sumsq_raw
        print(
            f"  Smoothed mean range: [{float(mean_s.min()):.3f}, {float(mean_s.max()):.3f}]",
            flush=True,
        )
        print(
            f"  Smoothed std range:  [{float(std_s.min()):.3f}, {float(std_s.max()):.3f}]",
            flush=True,
        )
        gm = mean_s.mean(axis=(1, 2))
        print(
            f"  Global-mean DOY cycle amplitude (max-min): {float(gm.max() - gm.min()):.3f}",
            flush=True,
        )
        lat_mask = (lat >= 30) & (lat <= 60)
        nh = mean_s[:, lat_mask, :].mean(axis=(1, 2))
        print(
            f"  NH 30–60° DOY cycle amplitude (max-min):  {float(nh.max() - nh.min()):.3f}",
            flush=True,
        )
        result_vars[f"{var}_mean"] = mean_s
        result_vars[f"{var}_std"] = std_s
    doy_coord = np.arange(1, 367)
    data_vars = {
        name: (("dayofyear", "latitude", "longitude"), arr)
        for name, arr in result_vars.items()
    }
    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={"dayofyear": doy_coord, "latitude": lat, "longitude": lon},
    )
    for name in ds_out.data_vars:
        values = ds_out[name].values
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Computed climatology {name} contains non-finite values"
            )
        if name.endswith("_std") and np.any(values <= 0):
            raise RuntimeError(f"Computed climatology {name} is not strictly positive")
    data_sha256 = climatology_content_sha256(ds_out)
    ds_out.attrs.update(
        period=f"{args.start_year}-{args.end_year}",
        years_used=str(years),
        n_years=len(years),
        calendar_mapping="month/day mapped to leap year 2000 (366 coordinates)",
        smoothing_half_width_days=args.smooth_window,
        smoothing_window_days=window_days,
        statistic="population mean and std pooled over the circular calendar window",
        std_ddof=0,
        source="era5_1p50/{year}.zarr beneath TXT_DATA_ROOT",
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        source_script="training/build_climatology.py",
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=data_sha256,
        description="Gridpoint day-of-year climatological mean and standard deviation from ERA5 daily fields. This is a fixed observed-climate statistic, not a forecast-lead-dependent drift correction.",
    )
    encoding = {v: {"dtype": "float32"} for v in ds_out.data_vars}
    print(f"\nWriting {out_path}...", flush=True)
    atomic_write_zarr(ds_out, out_path, encoding=encoding, overwrite=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
