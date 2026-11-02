import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import json
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd

torch.set_float32_matmul_precision("medium")
from txt_model import (
    ARTIFACT_SCHEMAS,
    CONFIG,
    MovingAverageValidationLoss,
    PAIRING_SCHEMA,
    PAIRING_SEED,
    Path,
    SCORED_DAY_OFFSETS,
    TARGET_LOSS_RELATIVE_TOLERANCE,
    TIME_SCHEMA,
    TXT_LOSS_SCHEMA,
    TargetLossStopping,
    TxT,
    file_sha256,
    json,
    load_pairing_bank,
    load_txt_checkpoint,
    mapping_sha256,
    pl,
    torch,
    validate_raw_source_identity_record,
)
from config import (
    ARTIFACT_SCHEMAS,
    CHECKPOINT_DIR,
    CONFIG,
    ERA5_ENS_DIR,
    LOGS_DIR,
    Path,
    TARGET_LOSS_RELATIVE_TOLERANCE,
    normalizer_provenance,
    os,
)
from datasets import ForecastTargetDataset, PairedForecastDataset, np, pd, torch
from time_coordinates import TIME_SCHEMA
from pairing import PAIRING_SCHEMA, PAIRING_SEED, load_pairing_bank, mapping_sha256
from io_utils import atomic_write_json
from cache import (
    archive_cache_manifest,
    validate_cache_for_consumer,
    validate_raw_source_identity_record,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ensemble", type=int, default=0)
parser.add_argument("--weight", type=int, default=0)
parser.add_argument("--target_loss", type=float, default=None)
parser.add_argument(
    "--target-relative-tolerance",
    type=float,
    default=TARGET_LOSS_RELATIVE_TOLERANCE,
    help="Require |J-J0|/J0 to be no larger than this value.",
)
parser.add_argument("--gpu", type=str, default="0,1,2,3")
parser.add_argument("--var", type=str, default="mslp")
parser.add_argument("--ckpt-version", type=str, default="txt")
parser.add_argument(
    "--perturbed-dataset-index",
    type=int,
    default=-1,
    help="0..9 for a persisted block-Latin TxT-UQ perturbed dataset",
)
parser.add_argument("--era5-ens-dir", type=str, default=ERA5_ENS_DIR)
parser.add_argument(
    "--pairing-file",
    type=str,
    default=None,
    help="JSON file mapping date to logical IFS and ERA5 EDA members",
)
parser.add_argument(
    "--cache-dir",
    type=str,
    default=None,
    help="Pre-computed .npy cache dir (from train_txt_uq). Skips zarr loading.",
)
args = parser.parse_args()
ensemble = args.ensemble
weight = args.weight
target_loss = args.target_loss
gpu = args.gpu
overfit = False
var = args.var
source2 = args.perturbed_dataset_index >= 0
if source2 and args.perturbed_dataset_index not in range(10):
    parser.error("--perturbed-dataset-index must be 0..9")
if source2 and (not args.pairing_file):
    parser.error("--perturbed-dataset-index requires --pairing-file")
if not source2 and args.pairing_file:
    parser.error("--pairing-file requires --perturbed-dataset-index 0..9")
if source2 and (not args.cache_dir):
    parser.error("TxT-UQ perturbed-dataset training requires --cache-dir")
if source2 and ensemble != args.perturbed_dataset_index:
    parser.error(
        "--ensemble is the checkpoint dataset axis and must equal --perturbed-dataset-index"
    )
var_dict = {"mslp": "mean_sea_level_pressure", "t2m": "2m_temperature"}
training_var = var_dict[var]
WEIGHT_SEED_SCHEMA = ARTIFACT_SCHEMAS["weight_seed"]
unique_seed = weight
pl.seed_everything(unique_seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
print("=" * 50)
print(f"TxT Training — {var} ens={ensemble:02d} w={weight} arch={'sp_full'}")
if source2:
    print(f"  Source 2 perturbed dataset: {args.perturbed_dataset_index}")
    print(f"  Persisted block-Latin pairing row: {args.pairing_file}")
print(f"  Weight seed: {unique_seed} ({WEIGHT_SEED_SCHEMA})")
print("=" * 50)


def _dataloader_kwargs(shuffle):
    num_workers = int(CONFIG.get("num_workers", 0))
    kwargs = {
        "batch_size": CONFIG["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(CONFIG.get("prefetch_factor", 2))
    return kwargs


def _validate_pairing(mapping, dates, path):
    try:
        path = Path(path)
        expected_path = (
            path.parent / f"pairing_dataset{args.perturbed_dataset_index:02d}.json"
        )
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"Expected canonical pairing file {expected_path}")
        pairings, contract = load_pairing_bank(path.parent, expected_dates=dates)
        expected = pairings[args.perturbed_dataset_index]
        if mapping != expected:
            raise ValueError("Pairing file differs from its validated bank")
        return (mapping_sha256(expected), contract)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Invalid pairing file {path}: {exc}") from exc


year_range = CONFIG["year_range"]
start_year, end_year = year_range
cache_manifest = None
cache_manifest_identity = "none"
cache_raw_source_identity = "none"
pairing_mapping_digest = "none"
pairing_bank_digest = "none"
pairing_training_bank_digest = "none"
pairing_testing_bank_digest = "none"
if args.cache_dir:
    print(f"Loading from cache: {args.cache_dir}")
    cache = Path(args.cache_dir)
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"TxT-UQ cache is missing its required manifest: {manifest_path}"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    cache_manifest = manifest
    if manifest.get("year_range") != list(CONFIG["year_range"]):
        raise RuntimeError(
            f"Cache year_range {manifest.get('year_range')} does not match CONFIG['year_range']={list(CONFIG['year_range'])}: {cache}"
        )
    if manifest.get("normalizer") != "doy_full":
        raise RuntimeError(
            f"Cache normalizer {manifest.get('normalizer')!r} does not match --normalizer {'doy_full'!r}: {cache}"
        )
    if manifest.get("time_schema") != TIME_SCHEMA:
        raise RuntimeError(
            f"Cache time_schema {manifest.get('time_schema')!r} does not match {TIME_SCHEMA!r}: {cache}"
        )
    if manifest.get("input_channels") != 1:
        raise RuntimeError(f"Cache must be one-channel TxT-UQ data: {cache}")
    expected_provenance = normalizer_provenance("doy_full")
    if manifest.get("normalizer_provenance") != expected_provenance:
        raise RuntimeError(
            f"Cache normalizer provenance does not match the live artifact: {cache}"
        )
    expected_cache_fields = {
        "schema": (
            ARTIFACT_SCHEMAS["paired_training_cache"]
            if source2
            else ARTIFACT_SCHEMAS["reference_cache"]
        ),
        "var": var,
        "year_range": list(CONFIG["year_range"]),
        "valid_day_offsets": list(range(32)),
        "init_start": CONFIG["train_init_start"],
        "init_end": CONFIG["val_init_end"],
    }
    if source2:
        expected_cache_fields.update(
            ifs_members=list(range(1, 11)), eda_members=list(range(10))
        )
    else:
        expected_cache_fields.update(ifs_members=[0], include_hres=True)
    cache_mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected_cache_fields.items()
        if manifest.get(key) != value
    }
    if cache_mismatch:
        raise RuntimeError(
            f"TxT-UQ cache contents do not match this run: {cache_mismatch}"
        )
    validate_raw_source_identity_record(
        manifest.get("raw_source_identity"),
        expected_source_kind="source2_ifs_eda" if source2 else "control_ifs_hres",
    )
    cache_manifest_identity = validate_cache_for_consumer(cache, manifest)
    cache_raw_source_identity = manifest["raw_source_identity"]
    archive_cache_manifest(cache, Path(CHECKPOINT_DIR) / "cache_manifests")
    era5_times = pd.DatetimeIndex(np.load(cache / "era5_times.npy", allow_pickle=True))
    init_times = pd.DatetimeIndex(np.load(cache / "init_times.npy", allow_pickle=True))
    if source2:
        all_era5 = {
            logical_member: np.load(
                cache / f"era5_eda_{logical_member - 1:02d}.npy", mmap_mode="r"
            )
            for logical_member in range(1, 11)
        }
    else:
        era5_hres_path = cache / "era5_hres.npy"
        if not era5_hres_path.exists():
            raise RuntimeError(
                "--cache-dir without --era5-ens-member requires era5_hres.npy (build it with training/build_reference_cache.py)"
            )
        era5 = np.load(era5_hres_path, mmap_mode="r")
    if source2:
        with open(args.pairing_file) as f:
            pairing_per_date = json.load(f)
        pairing_mapping_digest, pairing_contract = _validate_pairing(
            pairing_per_date, init_times, args.pairing_file
        )
        pairing_bank_digest = pairing_contract["pairing_bank_sha256"]
        pairing_training_bank_digest = pairing_contract["segments"]["training"][
            "pairing_bank_sha256"
        ]
        pairing_testing_bank_digest = pairing_contract["segments"]["testing"][
            "pairing_bank_sha256"
        ]
        print("  Opening 10 IFS cache arrays lazily...")
        all_ifs = {}
        for ens_num in range(1, 11):
            all_ifs[ens_num] = np.load(
                cache / f"ifs_norm_{ens_num:02d}.npy", mmap_mode="r"
            )
        dataset = PairedForecastDataset(
            all_ifs=all_ifs,
            all_era5=all_era5,
            training_var=training_var,
            pairing_per_date=pairing_per_date,
            init_times=init_times,
            era5_times=era5_times,
        )
        print("  Lazy cache dataset ready; arrays will be sliced per batch.")
    else:
        ifs = np.load(cache / f"ifs_norm_{ensemble:02d}.npy", mmap_mode="r")
        dataset = ForecastTargetDataset(
            ifs, era5, training_var, ifs_init_times=init_times, era5_times=era5_times
        )
else:
    raise ValueError("Training requires a validated --cache-dir")
training_provenance = {
    "source2": bool(source2),
    "perturbed_dataset_index": int(args.perturbed_dataset_index) if source2 else -1,
    "weight_level": int(weight),
    "training_seed": int(unique_seed),
    "weight_seed_schema": WEIGHT_SEED_SCHEMA,
    "pairing_schema": PAIRING_SCHEMA if source2 else "none",
    "pairing_seed": PAIRING_SEED if source2 else -1,
    "pairing_mapping_sha256": pairing_mapping_digest,
    "pairing_bank_sha256": pairing_bank_digest,
    "pairing_training_bank_sha256": pairing_training_bank_digest,
    "pairing_testing_bank_sha256": pairing_testing_bank_digest,
    "cache_manifest_identity": cache_manifest_identity,
    "cache_raw_source_identity": cache_raw_source_identity,
    "train_init_start": CONFIG["train_init_start"],
    "train_init_end": CONFIG["train_init_end"],
    "val_init_start": CONFIG["val_init_start"],
    "val_init_end": CONFIG["val_init_end"],
}
print(f"Total dataset size: {len(dataset)} samples")
if hasattr(dataset, "init_times") and dataset.init_times is not None:
    ds_init_times = pd.DatetimeIndex(dataset.init_times)
else:
    ds_init_times = pd.DatetimeIndex(init_times)
tr_s = pd.to_datetime(CONFIG["train_init_start"])
tr_e = pd.to_datetime(CONFIG["train_init_end"])
val_s = pd.to_datetime(CONFIG["val_init_start"])
val_e = pd.to_datetime(CONFIG["val_init_end"])
test_s = pd.to_datetime(CONFIG["test_init_start"])
train_mask = (ds_init_times >= tr_s) & (ds_init_times <= tr_e)
val_mask = (ds_init_times >= val_s) & (ds_init_times <= val_e)
train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
n_buffer = len(ds_init_times) - len(train_idx) - len(val_idx)
if len(val_idx) == 0:
    raise RuntimeError(
        f"No validation samples in [{val_s.date()}, {val_e.date()}]. torch_data init_times span {ds_init_times.min()} .. {ds_init_times.max()}"
    )
max_loaded = ds_init_times.max()
if max_loaded >= test_s:
    n_leak = int((ds_init_times >= test_s).sum())
    raise RuntimeError(
        f"Train/val dataset contains {n_leak} init date(s) >= test_init_start {test_s.date()} (max loaded {max_loaded.date()}). Check load bounds / config."
    )
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
print(
    f"Train {ds_init_times[train_idx].min().date()}..{ds_init_times[train_idx].max().date()}: {len(train_dataset)} samples"
)
print(
    f"Val   {ds_init_times[val_idx].min().date()}..{ds_init_times[val_idx].max().date()}: {len(val_dataset)} samples  (= residual/testing pool)"
)
print(f"Buffer init dates dropped (in neither train nor val): {n_buffer}")
train_loader = DataLoader(train_dataset, **_dataloader_kwargs(shuffle=True))
val_loader = DataLoader(val_dataset, **_dataloader_kwargs(shuffle=False))
model = TxT(
    input_channels=1,
    output_channels=1,
    learning_rate=CONFIG["learning_rate"],
    arch="sp_full",
    normalizer_provenance=normalizer_provenance("doy_full"),
    time_schema=TIME_SCHEMA,
    input_mode="member",
    training_variable=var,
    training_provenance=training_provenance,
)
if target_loss is not None:
    early_stopping_callback = TargetLossStopping(
        target_loss=target_loss,
        train_loader=train_loader,
        monitor="train_loss",
        relative_tolerance=args.target_relative_tolerance,
        max_iter=10,
        min_epochs=0,
    )
else:
    early_stopping_callback = EarlyStopping(
        monitor="val_loss_ma", patience=5, mode="min", check_on_train_epoch_end=False
    )


class PrintLossCallback(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        tl, vl, vma = (m.get("train_loss"), m.get("val_loss"), m.get("val_loss_ma"))
        if tl is not None:
            print(
                f"[epoch {trainer.current_epoch:03d}] train={float(tl):.6f} val={float(vl):.6f} val_ma={float(vma):.6f}"
            )


ckpt_prefix = f"{var}_{args.ckpt_version}"
callbacks = [MovingAverageValidationLoss(window_size=5), PrintLossCallback()]
checkpoint_callback = None
if target_loss is None:
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"{ckpt_prefix}_{ensemble:02d}-{weight:02d}-{{epoch:02d}}-{{val_loss_ma:.5f}}",
        monitor="val_loss_ma",
        mode="min",
        save_top_k=1,
    )
    callbacks.insert(1, checkpoint_callback)
callbacks.append(early_stopping_callback)
trainer = pl.Trainer(
    default_root_dir=LOGS_DIR,
    overfit_batches=0,
    limit_val_batches=1.0,
    max_epochs=5000,
    accelerator="auto",
    devices="auto",
    strategy="auto",
    precision="32-true",
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    log_every_n_steps=50,
    callbacks=callbacks,
    enable_checkpointing=target_loss is None,
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
if target_loss is not None:
    if not early_stopping_callback.converged:
        raise RuntimeError("Target-loss training stopped without a verified J0 match")
    achieved = float(early_stopping_callback.achieved_loss)
    diff = achieved - float(target_loss)
    target_ckpt = (
        Path(CHECKPOINT_DIR)
        / f"{ckpt_prefix}_{ensemble:02d}-{weight:02d}-targetloss={achieved:.6f}.ckpt"
    )
    temporary_ckpt = target_ckpt.parent / f".{target_ckpt.name}.tmp.{os.getpid()}.ckpt"
    trainer.save_checkpoint(str(temporary_ckpt))
    os.replace(temporary_ckpt, target_ckpt)
    verified = load_txt_checkpoint(
        target_ckpt,
        expected_arch="sp_full",
        expected_training_variable=var,
        expected_normalizer_provenance=normalizer_provenance("doy_full"),
        expected_training_provenance=training_provenance,
        map_location="cpu",
    )
    del verified
    target_checkpoint_sha256 = file_sha256(target_ckpt)
    sidecar = target_ckpt.with_suffix(".json")
    atomic_write_json(
        {
            "schema": ARTIFACT_SCHEMAS["weight_model_completion"],
            "checkpoint": target_ckpt.name,
            "checkpoint_sha256": target_checkpoint_sha256,
            "var": var,
            "version": args.ckpt_version,
            "ensemble": int(ensemble),
            "weight": int(weight),
            "architecture": "sp_full",
            "training_variable": var,
            "loss_schema": TXT_LOSS_SCHEMA,
            "scored_day_offsets": SCORED_DAY_OFFSETS,
            "source2": bool(source2),
            "perturbed_dataset_index": int(args.perturbed_dataset_index),
            "weight_level": int(weight),
            "training_seed": int(unique_seed),
            "weight_seed_schema": WEIGHT_SEED_SCHEMA,
            "target_loss": float(target_loss),
            "achieved_loss": achieved,
            "difference": diff,
            "absolute_difference": abs(diff),
            "relative_difference": abs(diff) / float(target_loss),
            "tolerance_definition": "abs(J-J0)/J0 <= relative_tolerance",
            "relative_tolerance": args.target_relative_tolerance,
            "effective_tolerance": early_stopping_callback.tolerance,
            "normalizer": "doy_full",
            "normalizer_provenance": normalizer_provenance("doy_full"),
            "time_schema": TIME_SCHEMA,
            "input_mode": "member",
            "input_channels": 1,
            "train_init_start": CONFIG["train_init_start"],
            "train_init_end": CONFIG["train_init_end"],
            "val_init_start": CONFIG["val_init_start"],
            "val_init_end": CONFIG["val_init_end"],
            "cache_manifest_identity": cache_manifest_identity,
            "cache_raw_source_identity": cache_raw_source_identity,
            "training_provenance": training_provenance,
            **(
                {
                    "pairing_schema": PAIRING_SCHEMA,
                    "pairing_seed": PAIRING_SEED,
                    "pairing_mapping_sha256": pairing_mapping_digest,
                    "pairing_bank_sha256": pairing_bank_digest,
                    "pairing_training_bank_sha256": pairing_training_bank_digest,
                    "pairing_testing_bank_sha256": pairing_testing_bank_digest,
                }
                if source2
                else {
                    "pairing_schema": "none",
                    "pairing_seed": -1,
                    "pairing_mapping_sha256": "none",
                    "pairing_bank_sha256": "none",
                    "pairing_training_bank_sha256": "none",
                    "pairing_testing_bank_sha256": "none",
                }
            ),
        },
        sidecar,
    )
    print(f"Verified target-loss model saved at {target_ckpt}")
    print(f"Target loss diff: {diff:+.8f}")
elif checkpoint_callback is not None:
    best_checkpoint = Path(checkpoint_callback.best_model_path)
    if not best_checkpoint.is_file():
        raise RuntimeError(
            f"Training completed without a best checkpoint: {best_checkpoint}"
        )
    verified = load_txt_checkpoint(
        best_checkpoint,
        expected_arch="sp_full",
        expected_training_variable=var,
        expected_normalizer_provenance=normalizer_provenance("doy_full"),
        expected_training_provenance=training_provenance,
        map_location="cpu",
    )
    del verified
    completion = best_checkpoint.with_suffix(".complete.json")
    atomic_write_json(
        {
            "schema": ARTIFACT_SCHEMAS["reference_completion"],
            "checkpoint": best_checkpoint.name,
            "checkpoint_sha256": file_sha256(best_checkpoint),
            "var": var,
            "version": args.ckpt_version,
            "ensemble": int(ensemble),
            "weight": int(weight),
            "architecture": "sp_full",
            "training_variable": var,
            "loss_schema": TXT_LOSS_SCHEMA,
            "scored_day_offsets": SCORED_DAY_OFFSETS,
            "normalizer": "doy_full",
            "normalizer_provenance": normalizer_provenance("doy_full"),
            "time_schema": TIME_SCHEMA,
            "input_mode": "member",
            "input_channels": 1,
            "source2": False,
            "train_init_start": CONFIG["train_init_start"],
            "train_init_end": CONFIG["train_init_end"],
            "val_init_start": CONFIG["val_init_start"],
            "val_init_end": CONFIG["val_init_end"],
            "cache_manifest_identity": cache_manifest_identity,
            "cache_raw_source_identity": cache_raw_source_identity,
            "weight_level": int(weight),
            "training_seed": int(unique_seed),
            "weight_seed_schema": WEIGHT_SEED_SCHEMA,
            "training_provenance": training_provenance,
        },
        completion,
    )
    print(f"Best model saved and completion-audited at {best_checkpoint}")
