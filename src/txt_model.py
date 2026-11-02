ARCHITECTURES = ("sp_full",)
import torch
import torch.nn as nn
import torch.nn.functional as functional
import pytorch_lightning as pl
from collections import deque
from pathlib import Path
import hashlib
import json
import math
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import CONFIG, PAIRING_DIR, TARGET_LOSS_RELATIVE_TOLERANCE, ARTIFACT_SCHEMAS
from cache import validate_raw_source_identity_record
from io_utils import file_sha256
from pairing import PAIRING_SCHEMA, PAIRING_SEED, load_pairing_bank, mapping_sha256
from time_coordinates import N_VALID_DAYS, SCORED_DAY_SLICE, TIME_SCHEMA

TXT_LOSS_SCHEMA = ARTIFACT_SCHEMAS["loss"]
SCORED_DAY_OFFSETS = list(range(18, 32))


def _pad_lon_lat(x, p):
    shift = x.shape[-1] // 2
    south_rows = torch.roll(x[:, :, :, 1 : 1 + p, :].flip(3), shift, dims=-1)
    north_rows = torch.roll(x[:, :, :, -1 - p : -1, :].flip(3), shift, dims=-1)
    x = torch.cat([south_rows, x, north_rows], dim=3)
    x = functional.pad(x, (p, p, 0, 0, 0, 0), mode="circular")
    return x


def geo_pad_3d(x, p):
    x = _pad_lon_lat(x, p)
    x = functional.pad(x, (0, 0, 0, 0, p, p), mode="replicate")
    return x


class GeoConv3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0
        )
        self.padding_size = padding

    def forward(self, x):
        x = geo_pad_3d(x, self.padding_size)
        return self.conv(x)


class ConvNormActivation3D(nn.Module):

    def __init__(
        self, input_channel, output_channel, kernel_size=3, stride=1, padding=1
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        num_groups = 32
        if output_channel % num_groups != 0:
            num_groups = output_channel // 2
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=output_channel)
        self.act = nn.GELU()
        self.p = padding

    def forward(self, x):
        x = geo_pad_3d(x, self.p)
        return self.act(self.norm(self.conv(x)))


def initialize_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class FiLM(nn.Module):

    def __init__(self, num_channels, max_timesteps=32):
        super().__init__()
        self.gamma = nn.Embedding(max_timesteps, num_channels)
        self.beta = nn.Embedding(max_timesteps, num_channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x):
        n_days = x.shape[2]
        t_idx = torch.arange(n_days, device=x.device)
        g = self.gamma(t_idx).permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(t_idx).permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return g * x + b


class AttentionGate3D(nn.Module):

    def __init__(self, gating_channels, skip_channels, intermediate_channels=None):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = max(skip_channels // 4, 8)
        self.gating_projection = nn.Conv3d(
            gating_channels, intermediate_channels, kernel_size=1, bias=True
        )
        self.skip_projection = nn.Conv3d(
            skip_channels, intermediate_channels, kernel_size=1, bias=True
        )
        self.psi = nn.Sequential(
            nn.Conv3d(intermediate_channels, 1, kernel_size=1, bias=True), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.gating_projection(g)
        x1 = self.skip_projection(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = functional.interpolate(
                g1, size=x1.shape[2:], mode="trilinear", align_corners=True
            )
        return x * self.psi(self.relu(g1 + x1))


class TxTBackbone(nn.Module):

    def __init__(self, in_channels, out_channels, unit=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unit = unit
        self.use_coslat = True
        pool_k = (1, 2, 2)
        up_factor = (1, 2, 2)

        def make_block(in_ch, out_ch):
            return nn.Sequential(
                ConvNormActivation3D(in_ch, out_ch),
                ConvNormActivation3D(out_ch, out_ch),
            )

        def make_upconv_cbr(in_ch, out_ch):
            cbr = ConvNormActivation3D(in_ch, out_ch, 3)
            return nn.Sequential(
                nn.Upsample(
                    scale_factor=up_factor, mode="trilinear", align_corners=True
                ),
                cbr,
            )

        actual_in = in_channels + 1
        self.conv1 = make_block(actual_in, self.unit)
        self.pool1 = nn.AvgPool3d(kernel_size=pool_k, stride=pool_k)
        self.conv2 = make_block(self.unit, self.unit * 2)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_k, stride=pool_k)
        self.conv3 = make_block(self.unit * 2, self.unit * 4)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_k, stride=pool_k)
        self.bottleneck = make_block(self.unit * 4, self.unit * 8)
        self.film1 = FiLM(self.unit)
        self.film2 = FiLM(self.unit * 2)
        self.film3 = FiLM(self.unit * 4)
        self.film_bn = FiLM(self.unit * 8)
        self.ag1 = AttentionGate3D(self.unit * 4, self.unit * 4)
        self.ag2 = AttentionGate3D(self.unit * 2, self.unit * 2)
        self.ag3 = AttentionGate3D(self.unit, self.unit)
        self.upconv1 = make_upconv_cbr(self.unit * 8, self.unit * 4)
        self.ex_conv1 = make_block(self.unit * 8, self.unit * 4)
        self.upconv2 = make_upconv_cbr(self.unit * 4, self.unit * 2)
        self.ex_conv2 = make_block(self.unit * 4, self.unit * 2)
        self.upconv3 = make_upconv_cbr(self.unit * 2, self.unit)
        self.ex_conv3 = make_block(self.unit * 2, self.unit)
        self.fc = GeoConv3D(
            self.unit, self.out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        batch_size, channels, n_days, height, width = x.shape
        lat = torch.linspace(-90, 90, height, device=x.device, dtype=x.dtype)
        coslat = torch.cos(lat * (3.141592653589793 / 180.0))
        x = torch.cat(
            [
                x,
                coslat.view(1, 1, 1, height, 1).expand(
                    batch_size, 1, n_days, height, width
                ),
            ],
            dim=1,
        )
        layer1 = self.conv1(x)
        layer1 = self.film1(layer1)
        out = self.pool1(layer1)
        layer2 = self.conv2(out)
        layer2 = self.film2(layer2)
        out = self.pool2(layer2)
        layer3 = self.conv3(out)
        layer3 = self.film3(layer3)
        out = self.pool3(layer3)
        bottleneck = self.bottleneck(out)
        bottleneck = self.film_bn(bottleneck)
        upconv1 = self.upconv1(bottleneck)
        skip3 = self.ag1(g=upconv1, x=layer3)
        cat1 = torch.cat((skip3, upconv1), dim=1)
        ex_layer1 = self.ex_conv1(cat1)
        upconv2 = self.upconv2(ex_layer1)
        skip2 = self.ag2(g=upconv2, x=layer2)
        cat2 = torch.cat((skip2, upconv2), dim=1)
        ex_layer2 = self.ex_conv2(cat2)
        upconv3 = self.upconv3(ex_layer2)
        if upconv3.shape[2:] != layer1.shape[2:]:
            upconv3 = functional.interpolate(
                upconv3, size=layer1.shape[2:], mode="trilinear", align_corners=True
            )
        skip1 = self.ag3(g=upconv3, x=layer1)
        cat3 = torch.cat((skip1, upconv3), dim=1)
        ex_layer3 = self.ex_conv3(cat3)
        out = self.fc(ex_layer3)
        return out


class TxT(pl.LightningModule):

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-06,
        learns_residual: bool = True,
        arch: str = "sp_full",
        normalizer_provenance: dict | None = None,
        time_schema: str | None = None,
        input_mode: str = "member",
        training_variable: str | None = None,
        loss_schema: str = TXT_LOSS_SCHEMA,
        scored_day_offsets: list[int] | None = None,
        training_provenance: dict | None = None,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError(
                f"TxT-UQ accepts exactly one IFS member channel; got input_channels={input_channels}"
            )
        if input_mode != "member":
            raise ValueError(f"TxT-UQ requires input_mode='member', got {input_mode!r}")
        if arch not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture {arch!r}; expected one of {ARCHITECTURES}"
            )
        if training_variable not in (None, "mslp", "t2m"):
            raise ValueError(
                f"training_variable must be 'mslp' or 't2m', got {training_variable!r}"
            )
        if loss_schema != TXT_LOSS_SCHEMA:
            raise ValueError(f"TxT-UQ requires loss_schema={TXT_LOSS_SCHEMA!r}")
        if scored_day_offsets is None:
            scored_day_offsets = SCORED_DAY_OFFSETS.copy()
        if list(scored_day_offsets) != SCORED_DAY_OFFSETS:
            raise ValueError("TxT-UQ scored_day_offsets must be exactly 18..31")
        if training_provenance is not None and (
            not isinstance(training_provenance, dict)
        ):
            raise ValueError("training_provenance must be a dictionary or None")
        self.save_hyperparameters()
        self.model = TxTBackbone(
            in_channels=self.hparams.input_channels,
            out_channels=self.hparams.output_channels,
        )
        self.model.apply(initialize_weights)
        self.register_buffer("_lat_weights", None, persistent=False)

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint["state_dict"]
        for old, new in (
            (".bottleNeck.", ".bottleneck."),
            (".W_g.", ".gating_projection."),
            (".W_x.", ".skip_projection."),
        ):
            for key in list(state):
                if old in key:
                    renamed = key.replace(old, new)
                    if renamed in state:
                        raise ValueError(
                            f"Conflicting checkpoint parameter keys: {key}, {renamed}"
                        )
                    state[renamed] = state.pop(key)

    def forward(self, ifs_forecast):
        if ifs_forecast.ndim != 5 or ifs_forecast.shape[1] != 1:
            raise ValueError(
                f"TxT-UQ model input must have shape (B, 1, 32, H, W); got {tuple(ifs_forecast.shape)}"
            )
        corrected_forecast = self.model(ifs_forecast)
        return corrected_forecast

    def predict_corrected(self, ifs_forecast):
        if self.hparams.learns_residual:
            return ifs_forecast + self(ifs_forecast)
        return self(ifs_forecast)

    def _get_lat_weights(self, height):
        if self._lat_weights is None or self._lat_weights.shape[-2] != height:
            lat_vals = torch.linspace(-90, 90, height, device=self.device)
            weights = torch.cos(torch.deg2rad(lat_vals)).clamp(min=1e-06)
            self._lat_weights = weights.view(1, 1, 1, height, 1)
        return self._lat_weights

    def latitude_weighted_mse(self, pred, target):
        if pred.ndim == 4:
            pred = pred.unsqueeze(1)
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pred.shape[2] != N_VALID_DAYS:
            raise ValueError(
                f"TxT-UQ loss expects {N_VALID_DAYS} model days, got {pred.shape[2]}"
            )
        pred = pred[:, :, SCORED_DAY_SLICE, ...]
        target = target[:, :, SCORED_DAY_SLICE, ...]
        height, width = (pred.shape[-2], pred.shape[-1])
        weights = self._get_lat_weights(height)
        sq = (pred - target) ** 2
        spatial_sum = weights.sum()
        if weights.shape[-1] == 1:
            spatial_sum = spatial_sum * width
        repetitions = sq.shape[0] * sq.shape[1] * sq.shape[2]
        return (weights * sq).sum() / (spatial_sum * repetitions)

    def _common_step(self, batch, batch_idx):
        ifs_forecast, era5_target = batch
        if self.hparams.learns_residual:
            y_hat = ifs_forecast + self(ifs_forecast)
            loss = self.latitude_weighted_mse(y_hat, era5_target)
        else:
            y_hat = self(ifs_forecast)
            loss = self.latitude_weighted_mse(y_hat, era5_target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss_ma"},
        }


def load_txt_checkpoint(
    checkpoint_path,
    *,
    expected_arch: str,
    expected_training_variable: str,
    expected_normalizer_provenance: dict,
    expected_training_provenance: dict | None = None,
    map_location=None,
):
    model = TxT.load_from_checkpoint(
        checkpoint_path, map_location=map_location, strict=True
    )
    expected = {
        "arch": expected_arch,
        "input_channels": 1,
        "output_channels": 1,
        "learns_residual": True,
        "input_mode": "member",
        "time_schema": TIME_SCHEMA,
        "training_variable": expected_training_variable,
        "loss_schema": TXT_LOSS_SCHEMA,
        "scored_day_offsets": SCORED_DAY_OFFSETS,
        "normalizer_provenance": expected_normalizer_provenance,
    }
    mismatches = {
        key: (model.hparams.get(key), value)
        for key, value in expected.items()
        if model.hparams.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"TxT-UQ checkpoint metadata mismatch in {checkpoint_path}: {mismatches}"
        )
    if (
        expected_training_provenance is not None
        and model.hparams.get("training_provenance") != expected_training_provenance
    ):
        raise RuntimeError(
            f"TxT-UQ training provenance mismatch in {checkpoint_path}: checkpoint={model.hparams.get('training_provenance')}, expected={expected_training_provenance}"
        )
    return model


def resolve_txt_model_bank(
    checkpoint_dir,
    *,
    variable: str,
    version: str,
    expected_model_count: int,
    require_complete: bool = True,
    expected_arch: str | None = None,
    expected_normalizer: str | None = None,
    expected_normalizer_provenance: dict | None = None,
    expected_cache_manifest_identity: str | None = None,
    expected_pairing_digests: dict[int, str] | None = None,
    expected_pairing_bank_sha256: str | None = None,
    expected_pairing_training_bank_sha256: str | None = None,
    expected_pairing_testing_bank_sha256: str | None = None,
):
    if expected_model_count == 100:
        expected_keys = {(eda, weight) for eda in range(10) for weight in range(10)}
        live_pairing_path = PAIRING_DIR / version
        if live_pairing_path.joinpath("manifest.json").is_file() or (
            expected_pairing_digests is None or expected_pairing_bank_sha256 is None
        ):
            live_pairings, live_contract = load_pairing_bank(live_pairing_path)
            live_mapping_digests = {
                index: mapping_sha256(live_pairings[index]) for index in range(10)
            }
            live_bank_digest = live_contract["pairing_bank_sha256"]
            live_training_digest = live_contract["segments"]["training"][
                "pairing_bank_sha256"
            ]
            live_testing_digest = live_contract["segments"]["testing"][
                "pairing_bank_sha256"
            ]
            supplied = {
                "pairing mappings": (expected_pairing_digests, live_mapping_digests),
                "full pairing bank": (expected_pairing_bank_sha256, live_bank_digest),
                "training pairing bank": (
                    expected_pairing_training_bank_sha256,
                    live_training_digest,
                ),
                "testing pairing bank": (
                    expected_pairing_testing_bank_sha256,
                    live_testing_digest,
                ),
            }
            conflicts = {
                name: values
                for name, values in supplied.items()
                if values[0] is not None and values[0] != values[1]
            }
            if conflicts:
                raise RuntimeError(
                    f"Expected pairing identities differ from the live bank: {conflicts}"
                )
            if expected_pairing_digests is None:
                expected_pairing_digests = live_mapping_digests
            if expected_pairing_bank_sha256 is None:
                expected_pairing_bank_sha256 = live_bank_digest
            if expected_pairing_training_bank_sha256 is None:
                expected_pairing_training_bank_sha256 = live_training_digest
            if expected_pairing_testing_bank_sha256 is None:
                expected_pairing_testing_bank_sha256 = live_testing_digest
    elif expected_model_count == 10:
        expected_keys = {(0, weight) for weight in range(10)}
    else:
        raise ValueError(
            f"TxT-UQ checkpoint grids contain either 100 Source-2 models or 10 control models, got expected_model_count={expected_model_count}"
        )
    directory = Path(checkpoint_dir)
    prefix = f"{variable}_{version}_"
    suffix = "targetloss=.*\\.ckpt" if expected_model_count == 100 else ".*\\.ckpt"
    pattern = re.compile(f"^{re.escape(prefix)}(\\d{{2}})-(\\d{{2}})-{suffix}$")
    candidates = {}
    for path in directory.glob(f"{prefix}*.ckpt"):
        match = pattern.match(path.name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            candidates.setdefault(key, []).append(path)
    resolved_by_key = {}
    resolved_rows = {}
    invalid = {}
    for key, paths in candidates.items():
        for path in sorted(
            paths, key=lambda item: item.stat().st_mtime_ns, reverse=True
        ):
            sidecar = path.with_suffix(
                ".json" if expected_model_count == 100 else ".complete.json"
            )
            try:
                row = json.loads(sidecar.read_text())
                expected_metadata = {
                    "schema": (
                        ARTIFACT_SCHEMAS["weight_model_completion"]
                        if expected_model_count == 100
                        else ARTIFACT_SCHEMAS["reference_completion"]
                    ),
                    "checkpoint": path.name,
                    "checkpoint_sha256": file_sha256(path),
                    "var": variable,
                    "version": version,
                    "ensemble": key[0],
                    "weight": key[1],
                    "training_variable": variable,
                    "loss_schema": TXT_LOSS_SCHEMA,
                    "scored_day_offsets": SCORED_DAY_OFFSETS,
                    "source2": expected_model_count == 100,
                    "time_schema": TIME_SCHEMA,
                    "input_mode": "member",
                    "input_channels": 1,
                }
                if expected_arch is not None:
                    expected_metadata["architecture"] = expected_arch
                if expected_normalizer is not None:
                    expected_metadata["normalizer"] = expected_normalizer
                if expected_normalizer_provenance is not None:
                    expected_metadata["normalizer_provenance"] = (
                        expected_normalizer_provenance
                    )
                if expected_cache_manifest_identity is not None:
                    expected_metadata["cache_manifest_identity"] = (
                        expected_cache_manifest_identity
                    )
                expected_metadata.update(
                    weight_level=key[1],
                    training_seed=key[1],
                    weight_seed_schema=ARTIFACT_SCHEMAS["weight_seed"],
                    train_init_start=CONFIG["train_init_start"],
                    train_init_end=CONFIG["train_init_end"],
                    val_init_start=CONFIG["val_init_start"],
                    val_init_end=CONFIG["val_init_end"],
                )
                if expected_model_count == 100:
                    expected_metadata.update(
                        perturbed_dataset_index=key[0],
                        pairing_schema=PAIRING_SCHEMA,
                        pairing_seed=PAIRING_SEED,
                        tolerance_definition="abs(J-J0)/J0 <= relative_tolerance",
                    )
                    pairing_digest = row.get("pairing_mapping_sha256")
                    if not isinstance(pairing_digest, str) or len(pairing_digest) != 64:
                        raise RuntimeError(
                            "completion has no valid pairing_mapping_sha256"
                        )
                    if (
                        expected_pairing_digests is not None
                        and pairing_digest != expected_pairing_digests.get(key[0])
                    ):
                        raise RuntimeError(
                            "completion pairing digest differs from the live recipe"
                        )
                    bank_digest = row.get("pairing_bank_sha256")
                    training_bank_digest = row.get("pairing_training_bank_sha256")
                    testing_bank_digest = row.get("pairing_testing_bank_sha256")
                    if any(
                        (
                            not isinstance(value, str) or len(value) != 64
                            for value in (
                                bank_digest,
                                training_bank_digest,
                                testing_bank_digest,
                            )
                        )
                    ):
                        raise RuntimeError(
                            "completion has no valid segmented pairing-bank identity"
                        )
                    if (
                        expected_pairing_bank_sha256 is not None
                        and bank_digest != expected_pairing_bank_sha256
                    ):
                        raise RuntimeError(
                            "completion pairing bank differs from the live recipe"
                        )
                    if (
                        expected_pairing_training_bank_sha256 is not None
                        and training_bank_digest
                        != expected_pairing_training_bank_sha256
                    ):
                        raise RuntimeError(
                            "completion training pairing bank differs from the live recipe"
                        )
                    if (
                        expected_pairing_testing_bank_sha256 is not None
                        and testing_bank_digest != expected_pairing_testing_bank_sha256
                    ):
                        raise RuntimeError(
                            "completion testing pairing bank differs from the live recipe"
                        )
                    try:
                        target_loss = float(row["target_loss"])
                        achieved_loss = float(row["achieved_loss"])
                        relative_tolerance = float(row["relative_tolerance"])
                        effective_tolerance = float(row["effective_tolerance"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "completion has invalid target-loss audit fields"
                        ) from exc
                    if not (
                        math.isfinite(target_loss)
                        and target_loss > 0.0
                        and math.isfinite(achieved_loss)
                        and math.isfinite(relative_tolerance)
                        and (relative_tolerance > 0.0)
                        and math.isfinite(effective_tolerance)
                    ):
                        raise RuntimeError(
                            "completion target-loss audit fields must be finite and positive"
                        )
                    if not math.isclose(
                        relative_tolerance,
                        TARGET_LOSS_RELATIVE_TOLERANCE,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise RuntimeError(
                            "completion relative tolerance differs from the TxT-UQ value"
                        )
                    expected_effective = target_loss * relative_tolerance
                    if not math.isclose(
                        effective_tolerance,
                        expected_effective,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            "completion effective tolerance is inconsistent with J0"
                        )
                    achieved_relative_difference = (
                        abs(achieved_loss - target_loss) / target_loss
                    )
                    if achieved_relative_difference > relative_tolerance + 1e-12:
                        raise RuntimeError(
                            "completion checkpoint did not achieve the relative target-loss tolerance"
                        )
                cache_identity = row.get("cache_manifest_identity")
                if not (
                    isinstance(cache_identity, str)
                    and len(cache_identity) == 64
                    and all(
                        (
                            character in "0123456789abcdef"
                            for character in cache_identity
                        )
                    )
                ):
                    raise RuntimeError(
                        "completion has no valid cache_manifest_identity"
                    )
                cache_raw_source_identity = row.get("cache_raw_source_identity")
                validate_raw_source_identity_record(
                    cache_raw_source_identity,
                    expected_source_kind=(
                        "source2_ifs_eda"
                        if expected_model_count == 100
                        else "control_ifs_hres"
                    ),
                )
                mismatches = {
                    name: (row.get(name), expected)
                    for name, expected in expected_metadata.items()
                    if row.get(name) != expected
                }
                if mismatches:
                    raise RuntimeError(f"completion metadata mismatch: {mismatches}")
                expected_training_provenance = {
                    "source2": expected_model_count == 100,
                    "perturbed_dataset_index": (
                        key[0] if expected_model_count == 100 else -1
                    ),
                    "weight_level": key[1],
                    "training_seed": key[1],
                    "weight_seed_schema": ARTIFACT_SCHEMAS["weight_seed"],
                    "pairing_schema": (
                        PAIRING_SCHEMA if expected_model_count == 100 else "none"
                    ),
                    "pairing_seed": PAIRING_SEED if expected_model_count == 100 else -1,
                    "pairing_mapping_sha256": (
                        row.get("pairing_mapping_sha256")
                        if expected_model_count == 100
                        else "none"
                    ),
                    "pairing_bank_sha256": (
                        row.get("pairing_bank_sha256")
                        if expected_model_count == 100
                        else "none"
                    ),
                    "pairing_training_bank_sha256": (
                        row.get("pairing_training_bank_sha256")
                        if expected_model_count == 100
                        else "none"
                    ),
                    "pairing_testing_bank_sha256": (
                        row.get("pairing_testing_bank_sha256")
                        if expected_model_count == 100
                        else "none"
                    ),
                    "cache_manifest_identity": row.get(
                        "cache_manifest_identity", "none"
                    ),
                    "cache_raw_source_identity": cache_raw_source_identity,
                    "train_init_start": CONFIG["train_init_start"],
                    "train_init_end": CONFIG["train_init_end"],
                    "val_init_start": CONFIG["val_init_start"],
                    "val_init_end": CONFIG["val_init_end"],
                }
                if row.get("training_provenance") != expected_training_provenance:
                    raise RuntimeError("completion training_provenance is inconsistent")
                if (
                    expected_arch is not None
                    and expected_normalizer_provenance is not None
                ):
                    model = load_txt_checkpoint(
                        path,
                        expected_arch=expected_arch,
                        expected_training_variable=variable,
                        expected_normalizer_provenance=expected_normalizer_provenance,
                        expected_training_provenance=expected_training_provenance,
                        map_location="cpu",
                    )
                    del model
            except Exception as exc:
                invalid.setdefault(key, []).append(f"{path.name}: {exc}")
                continue
            resolved_by_key[key] = path
            resolved_rows[key] = row
            break
    actual_keys = set(resolved_by_key)
    unexpected = actual_keys - expected_keys
    missing = expected_keys - actual_keys
    if unexpected or (require_complete and missing):
        raise RuntimeError(
            f"Checkpoint grid mismatch for {variable}_{version}: missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}, invalid={dict(list(invalid.items())[:3])}"
        )
    cache_identities = {
        resolved_rows[key]["cache_manifest_identity"] for key in actual_keys
    }
    if len(cache_identities) > 1:
        raise RuntimeError(
            f"Checkpoint grid mixes cache manifests for {variable}_{version}: {sorted(cache_identities)}"
        )
    raw_source_identities = {
        json.dumps(
            resolved_rows[key]["cache_raw_source_identity"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for key in actual_keys
    }
    if len(raw_source_identities) > 1:
        raise RuntimeError(
            f"Checkpoint grid mixes raw source identities for {variable}_{version}"
        )
    if expected_model_count == 100:
        target_losses = {
            float(resolved_rows[key]["target_loss"]) for key in actual_keys
        }
        relative_tolerances = {
            float(resolved_rows[key]["relative_tolerance"]) for key in actual_keys
        }
        if len(target_losses) > 1:
            raise RuntimeError(
                f"Checkpoint grid mixes J0 targets for {variable}_{version}: {sorted(target_losses)}"
            )
        if len(relative_tolerances) > 1:
            raise RuntimeError(
                f"Checkpoint grid mixes relative target-loss tolerances for {variable}_{version}: {sorted(relative_tolerances)}"
            )
    resolved = [
        (key, resolved_by_key[key]) for key in sorted(expected_keys & actual_keys)
    ]
    return resolved


def checkpoint_bank_identity(resolved_checkpoints) -> str:
    records = []
    for key, path_value in resolved_checkpoints:
        path = Path(path_value)
        record = {"key": list(key), "checkpoint_sha256": file_sha256(path)}
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            completion_sidecar = path.with_suffix(".complete.json")
            if completion_sidecar.is_file():
                sidecar = completion_sidecar
        if sidecar.is_file():
            payload = json.loads(sidecar.read_text())
            payload.pop("checkpoint", None)
            payload.pop("checkpoint_sha256", None)
            canonical_sidecar = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            )
            record["sidecar_payload_sha256"] = hashlib.sha256(
                canonical_sidecar.encode("utf-8")
            ).hexdigest()
        records.append(record)
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TargetLossStopping(pl.Callback):

    def __init__(
        self,
        target_loss: float,
        train_loader,
        monitor: str = "train_loss",
        relative_tolerance: float = TARGET_LOSS_RELATIVE_TOLERANCE,
        max_iter: int = 10,
        min_epochs: int = 0,
    ):
        super().__init__()
        self.target_loss = float(target_loss)
        self.train_loader = train_loader
        self.monitor = monitor
        if self.target_loss <= 0:
            raise ValueError(f"target_loss must be positive, got {target_loss}")
        if relative_tolerance <= 0:
            raise ValueError(
                f"relative_tolerance must be positive, got {relative_tolerance}"
            )
        self.relative_tolerance = float(relative_tolerance)
        self.tolerance = self.target_loss * self.relative_tolerance
        self.max_iter = max_iter
        self.min_epochs = min_epochs
        self.prev_state_dict = None
        self.initial_state_dict = None
        self.high_candidate_history = deque(maxlen=3)
        self.converged = False
        self.achieved_loss = None
        self.failure_reason = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.prev_state_dict = self._clone_state(pl_module)
        if self.initial_state_dict is None:
            self.initial_state_dict = {
                key: value.clone() for key, value in self.prev_state_dict.items()
            }

    def _evaluate_loss(self, pl_module, weights):
        device = pl_module.device
        pl_module.load_state_dict(weights)
        pl_module.eval()
        total_loss, n = (0.0, 0)
        with torch.no_grad():
            for batch in self.train_loader:
                batch = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                ]
                batch_size = int(batch[0].shape[0])
                total_loss += pl_module._common_step(batch, 0).item() * batch_size
                n += batch_size
        pl_module.train()
        return total_loss / n if n > 0 else float("inf")

    @staticmethod
    def _clone_state(pl_module):
        return {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    @staticmethod
    def _midpoint_state(high, low):
        midpoint = {}
        for key in high:
            if torch.is_floating_point(high[key]):
                midpoint[key] = (high[key] + low[key]) / 2.0
            else:
                midpoint[key] = low[key].clone()
        return midpoint

    def _finish(self, trainer, pl_module, weights, loss):
        pl_module.load_state_dict(weights)
        self.converged = True
        self.achieved_loss = float(loss)
        trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics.get(self.monitor)
        if current_loss is None or self.prev_state_dict is None:
            return
        if self.min_epochs and trainer.current_epoch < self.min_epochs:
            return
        current_metric = float(current_loss)
        current_weights = self._clone_state(pl_module)
        if current_metric > self.target_loss + self.tolerance:
            self.high_candidate_history.append(current_weights)
            return
        loss_low = self._evaluate_loss(pl_module, current_weights)
        if loss_low > self.target_loss + self.tolerance:
            self.high_candidate_history.append(current_weights)
            return
        if abs(loss_low - self.target_loss) <= self.tolerance:
            print(f"[TargetLoss] Converged without interpolation: {loss_low:.6f}")
            self._finish(trainer, pl_module, current_weights, loss_low)
            return
        high_candidates = [
            *reversed(self.high_candidate_history),
            self.prev_state_dict,
            self.initial_state_dict,
        ]
        w_high = None
        loss_high = None
        for candidate in high_candidates:
            if candidate is None:
                continue
            candidate_loss = self._evaluate_loss(pl_module, candidate)
            if candidate_loss > self.target_loss:
                w_high = candidate
                loss_high = candidate_loss
                break
        pl_module.load_state_dict(current_weights)
        if w_high is None:
            self.failure_reason = f"No fixed-weight bracket around J0={self.target_loss:.6f}; current fixed loss={loss_low:.6f}"
            print(f"[TargetLoss] ERROR: {self.failure_reason}")
            trainer.should_stop = True
            return
        print(
            f"\n[Bisection] Epoch {trainer.current_epoch}: high={loss_high:.6f}, low={loss_low:.6f}, J0={self.target_loss:.6f}"
        )
        w_low = current_weights
        best_weights = w_low
        best_loss = loss_low
        best_diff = abs(loss_low - self.target_loss)
        for i in range(self.max_iter):
            w_mid = self._midpoint_state(w_high, w_low)
            loss_mid = self._evaluate_loss(pl_module, w_mid)
            diff = loss_mid - self.target_loss
            print(
                f"   Iter {i + 1}/{self.max_iter}: Loss={loss_mid:.6f} (diff={diff:+.6f})"
            )
            if abs(diff) < best_diff:
                best_diff = abs(diff)
                best_loss = loss_mid
                best_weights = self._clone_state(pl_module)
            if abs(diff) <= self.tolerance:
                print(f"[Bisection] Converged: {loss_mid:.6f}")
                self._finish(trainer, pl_module, w_mid, loss_mid)
                return
            if loss_mid > self.target_loss:
                w_high = w_mid
            else:
                w_low = w_mid
        if best_diff <= self.tolerance:
            self._finish(trainer, pl_module, best_weights, best_loss)
            return
        self.failure_reason = f"Bisection did not reach tolerance after {self.max_iter} iterations: best_loss={best_loss:.6f}, abs_diff={best_diff:.6f}, tolerance={self.tolerance:.6f}"
        print(f"[TargetLoss] ERROR: {self.failure_reason}")
        trainer.should_stop = True


class MovingAverageValidationLoss(pl.Callback):

    def __init__(self, window_size: int = 5, monitor: str = "val_loss"):
        super().__init__()
        self.window_size = window_size
        self.monitor = monitor
        self.val_losses = deque(maxlen=self.window_size)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        current_loss_tensor = trainer.callback_metrics.get(self.monitor)
        if current_loss_tensor is None:
            return
        self.val_losses.append(current_loss_tensor.item())
        if len(self.val_losses) > 0:
            moving_avg = sum(self.val_losses) / len(self.val_losses)
            pl_module.log("val_loss_ma", moving_avg, prog_bar=True)
