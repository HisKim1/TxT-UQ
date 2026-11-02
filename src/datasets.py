import torch.utils.data as torch_data
import numpy as np
import xarray as xr
import pandas as pd
import torch
from time_coordinates import (
    N_VALID_DAYS,
    validate_forecast_time_axis,
    build_target_index_matrix,
    canonicalize_spatial_grid,
    climatology_dayofyear,
)


class DayOfYearNormalizer:
    mode = "doy_full"
    mu_scalar = None

    def __init__(
        self, mu_doy_da, sig_doy_da, sig_min_clip=None, *, _allow_test_grid=False
    ):
        self._allow_test_grid = bool(_allow_test_grid)
        if self._allow_test_grid:
            self._mu_doy_xr = mu_doy_da.sortby("latitude").sortby("longitude")
            self._sig_doy_xr = sig_doy_da.sortby("latitude").sortby("longitude")
        else:
            self._mu_doy_xr = canonicalize_spatial_grid(
                mu_doy_da, label="DOY climatological mean"
            )
            self._sig_doy_xr = canonicalize_spatial_grid(
                sig_doy_da, label="DOY climatological standard deviation"
            )
        self._mu_doy_xr = self._mu_doy_xr.transpose(
            "dayofyear", "latitude", "longitude"
        )
        self._sig_doy_xr = self._sig_doy_xr.transpose(
            "dayofyear", "latitude", "longitude"
        )
        expected_doy = np.arange(1, 367, dtype=np.int64)
        if not np.array_equal(
            np.asarray(self._mu_doy_xr.dayofyear.values, dtype=np.int64), expected_doy
        ):
            raise ValueError("DOY mean coordinate must be exactly 1..366")
        if not np.array_equal(
            np.asarray(self._sig_doy_xr.dayofyear.values, dtype=np.int64), expected_doy
        ):
            raise ValueError("DOY standard-deviation coordinate must be exactly 1..366")
        for dim in ("latitude", "longitude"):
            if not np.array_equal(
                self._mu_doy_xr[dim].values, self._sig_doy_xr[dim].values
            ):
                raise ValueError(f"DOY mean/std {dim} coordinates differ")
        if sig_min_clip is not None:
            self._sig_doy_xr = self._sig_doy_xr.clip(min=float(sig_min_clip))
        self._mu_np = np.asarray(self._mu_doy_xr.values, dtype=np.float32)
        self._sig_np = np.asarray(self._sig_doy_xr.values, dtype=np.float32)
        if not np.isfinite(self._mu_np).all():
            raise ValueError("DOY climatological mean contains non-finite values")
        if not np.isfinite(self._sig_np).all() or np.any(self._sig_np <= 0):
            raise ValueError(
                "DOY climatological standard deviation must be finite and positive"
            )
        self._sig_min_clip = sig_min_clip

    @property
    def sig_scalar(self):
        return float(self._sig_np.mean())

    @staticmethod
    def _doys_for(init_time, valid_day_offsets):
        init = pd.Timestamp(init_time)
        valid = init + pd.to_timedelta(np.asarray(valid_day_offsets).ravel(), unit="D")
        return climatology_dayofyear(valid).astype(np.int64)

    def _broadcast_time(self, time_index):
        time_values = pd.DatetimeIndex(time_index)
        doy = climatology_dayofyear(time_values)
        doy_da = xr.DataArray(
            np.asarray(doy, dtype=np.int64), dims=["time"], coords={"time": time_values}
        )
        mu_t = self._mu_doy_xr.sel(dayofyear=doy_da)
        sig_t = self._sig_doy_xr.sel(dayofyear=doy_da)
        return (mu_t, sig_t)

    def _broadcast_init_lead(self, init_times, valid_day_offsets):
        init_values = pd.DatetimeIndex(init_times)
        offset_values = np.asarray(valid_day_offsets, dtype=np.int64)
        init_np = init_values.values.astype("datetime64[ns]")[:, None]
        dt = offset_values[None, :].astype("timedelta64[D]").astype("timedelta64[ns]")
        valid = init_np + dt
        doys_2d = climatology_dayofyear(valid)
        doy_da = xr.DataArray(
            doys_2d.astype(np.int64),
            dims=["init_time", "valid_day_offset"],
            coords={"init_time": init_values, "valid_day_offset": offset_values},
        )
        mu_t = self._mu_doy_xr.sel(dayofyear=doy_da)
        sig_t = self._sig_doy_xr.sel(dayofyear=doy_da)
        return (mu_t, sig_t)

    def _assert_grid(self, data, *, label):
        for dim in ("latitude", "longitude"):
            climatology_coord = self._mu_doy_xr[dim].values
            if dim not in data.coords or not np.array_equal(
                data[dim].values, climatology_coord
            ):
                raise ValueError(
                    f"{label} {dim} coordinate does not exactly match the TxT-UQ climatology grid"
                )

    def normalize_era5(self, era5):
        if self._allow_test_grid:
            era5 = era5.sortby("latitude").sortby("longitude")
        else:
            era5 = canonicalize_spatial_grid(era5, label="ERA5")
        self._assert_grid(era5, label="ERA5")
        mu_t, sig_t = self._broadcast_time(era5.time.values)
        return (era5 - mu_t) / sig_t

    def normalize_ifs(self, ifs):
        validate_forecast_time_axis(ifs)
        if self._allow_test_grid:
            ifs = ifs.sortby("latitude").sortby("longitude")
        else:
            ifs = canonicalize_spatial_grid(ifs, label="IFS")
        self._assert_grid(ifs, label="IFS")
        init_times = pd.to_datetime(ifs.init_time.values)
        day_offsets = np.asarray(ifs.valid_day_offset.values, dtype=np.int64)
        mu_t, sig_t = self._broadcast_init_lead(init_times, day_offsets)
        return (ifs - mu_t) / sig_t

    def denormalize_grid(self, z_np, init_time, valid_day_offsets):
        if init_time is None or valid_day_offsets is None:
            raise ValueError(
                "DayOfYearNormalizer.denormalize_grid needs init_time and valid_day_offsets"
            )
        doys = self._doys_for(init_time, valid_day_offsets)
        mu_slice = self._mu_np[doys - 1]
        sig_slice = self._sig_np[doys - 1]
        return z_np * sig_slice + mu_slice


def _slice_to_tensor(vals):
    if isinstance(vals, torch.Tensor):
        return vals.to(dtype=torch.float32)
    arr = np.asarray(vals, dtype=np.float32)
    if not arr.flags.c_contiguous or not arr.flags.writeable:
        arr = np.array(arr, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(arr)


def _ifs_slice(tensor, idx, training_var=None):
    sample = tensor[idx]
    if sample.shape[0] != N_VALID_DAYS:
        raise ValueError(
            f"IFS cache must already use the forecast 32-day axis; got {sample.shape}"
        )
    return sample.unsqueeze(0)


def _ifs_slice_lazy(data, idx, training_var=None):
    if isinstance(data, torch.Tensor):
        return _ifs_slice(data, idx, training_var)
    if hasattr(data, "isel"):
        validate_forecast_time_axis(data)
        vals = data.isel(init_time=idx).values
    else:
        vals = data[idx]
    if vals.shape[0] != N_VALID_DAYS:
        raise ValueError(
            f"IFS cache must already use the forecast 32-day axis; got {vals.shape}"
        )
    return _slice_to_tensor(vals).unsqueeze(0)


def _target_slice_lazy(data, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape != (N_VALID_DAYS,):
        raise ValueError(f"Expected {N_VALID_DAYS} target indices, got {indices.shape}")
    if isinstance(data, torch.Tensor):
        torch_idx = torch.as_tensor(indices, dtype=torch.long, device=data.device)
        return data.index_select(0, torch_idx).to(dtype=torch.float32).unsqueeze(0)
    if hasattr(data, "isel"):
        vals = data.isel(time=indices).values
    else:
        vals = data[indices]
    return _slice_to_tensor(vals).unsqueeze(0)


def _shape0(data):
    return int(data.shape[0])


class ForecastTargetDataset(torch_data.Dataset):

    def __init__(
        self,
        ifs,
        era5,
        training_var="mean_sea_level_pressure",
        ifs_init_times=None,
        era5_times=None,
    ):
        self.ifs = ifs
        self.era5 = era5
        self.training_var = training_var
        if ifs_init_times is None:
            raise ValueError(
                "ifs_init_times is required; positional target alignment is disabled"
            )
        self.init_times = pd.DatetimeIndex(ifs_init_times)
        if era5_times is not None:
            target_times = era5_times
        elif hasattr(era5, "time"):
            target_times = era5.time.values
        else:
            raise ValueError("era5_times is required when era5 has no time coordinate")
        self.target_indices = build_target_index_matrix(self.init_times, target_times)

    def __len__(self):
        return _shape0(self.ifs)

    def __getitem__(self, idx):
        x = _ifs_slice_lazy(self.ifs, idx, self.training_var)
        y = _target_slice_lazy(self.era5, self.target_indices[idx])
        return (x, y)


class ForecastInputDataset(torch_data.Dataset):

    def __init__(self, ifs):
        if hasattr(ifs, "dims"):
            validate_forecast_time_axis(ifs)
        elif ifs.shape[1] != N_VALID_DAYS:
            raise ValueError(
                f"IFS inference cache must have {N_VALID_DAYS} days, got {ifs.shape}"
            )
        self.ifs = ifs
        if hasattr(ifs, "init_time"):
            self.init_times = pd.DatetimeIndex(ifs.init_time.values)
        else:
            self.init_times = None

    def __len__(self):
        return _shape0(self.ifs)

    def __getitem__(self, idx):
        x = _ifs_slice_lazy(self.ifs, idx)
        return (x, torch.empty(0, dtype=torch.float32))


def _validate_paired_dataset_inputs(
    all_ifs, all_era5, pairing_per_date, init_times, era5_times
):
    init_times = pd.DatetimeIndex(init_times)
    if init_times.has_duplicates:
        raise ValueError("init_times contains duplicate dates")
    expected_dates = {date.strftime("%Y%m%d") for date in init_times}
    normalized_pairing = {}
    for key, pair in pairing_per_date.items():
        if not isinstance(pair, dict):
            raise ValueError(f"Pairing for {key} must be a mapping, got {type(pair)}")
        ifs_member = pair.get("ifs_member")
        era5_member = pair.get("era5_eda_member")
        if ifs_member not in range(1, 11) or era5_member not in range(1, 11):
            raise ValueError(f"Invalid logical pairing for {key}: {pair}")
        if ifs_member == era5_member:
            raise ValueError(f"Equal logical members are forbidden for {key}: {pair}")
        normalized_pairing[str(key)] = {
            "ifs_member": int(ifs_member),
            "era5_eda_member": int(era5_member),
        }
    actual_dates = set(normalized_pairing)
    if actual_dates != expected_dates:
        raise ValueError(
            f"Pairing date inventory differs from init_times: missing={sorted(expected_dates - actual_dates)[:10]}, extra={sorted(actual_dates - expected_dates)[:10]}"
        )
    normalized_ifs = {int(key): value for key, value in all_ifs.items()}
    normalized_era5 = {int(key): value for key, value in all_era5.items()}
    used_ifs = {pair["ifs_member"] for pair in normalized_pairing.values()}
    used_era5 = {pair["era5_eda_member"] for pair in normalized_pairing.values()}
    missing_ifs = used_ifs - set(normalized_ifs)
    missing_era5 = used_era5 - set(normalized_era5)
    if missing_ifs or missing_era5:
        raise ValueError(
            f"Pairing references unloaded members: IFS={sorted(missing_ifs)}, ERA5 EDA={sorted(missing_era5)}"
        )
    ref_ifs_shape = next(iter(normalized_ifs.values())).shape
    for member, values in normalized_ifs.items():
        if values.shape != ref_ifs_shape:
            raise ValueError(
                f"IFS member {member} shape {values.shape} != {ref_ifs_shape}"
            )
    ref_era5_shape = next(iter(normalized_era5.values())).shape
    for member, values in normalized_era5.items():
        if values.shape != ref_era5_shape:
            raise ValueError(
                f"ERA5 EDA member {member} shape {values.shape} != {ref_era5_shape}"
            )
    target_indices = build_target_index_matrix(init_times, era5_times)
    return (
        init_times,
        normalized_ifs,
        normalized_era5,
        normalized_pairing,
        target_indices,
    )


class PairedForecastDataset(torch_data.Dataset):

    def __init__(
        self,
        all_ifs,
        all_era5,
        training_var,
        pairing_per_date,
        init_times,
        era5_times=None,
    ):
        if era5_times is None:
            sample = next(iter(all_era5.values()))
            if not hasattr(sample, "time"):
                raise ValueError("era5_times is required without ERA5 time coordinates")
            era5_times = sample.time.values
        (
            self.init_times,
            self.all_ifs,
            self.all_era5,
            self.pairing_per_date,
            self.target_indices,
        ) = _validate_paired_dataset_inputs(
            all_ifs, all_era5, pairing_per_date, init_times, era5_times
        )
        self.training_var = training_var

    def __len__(self):
        return len(self.init_times)

    def __getitem__(self, idx):
        date_key = self.init_times[idx].strftime("%Y%m%d")
        pair = self.pairing_per_date[date_key]
        x_member = _ifs_slice_lazy(
            self.all_ifs[pair["ifs_member"]], idx, self.training_var
        )
        y = _target_slice_lazy(
            self.all_era5[pair["era5_eda_member"]], self.target_indices[idx]
        )
        return (x_member, y)
