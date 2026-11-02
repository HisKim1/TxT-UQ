from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    CHECKPOINT_DIR,
    CONFIG,
    ERA5_ENS_DIR,
    normalizer_provenance,
    ARTIFACT_SCHEMAS,
)
from txt_model import load_txt_checkpoint, resolve_txt_model_bank
from cache import validate_cache_for_consumer, validate_raw_source_identity_record
from datasets import PairedForecastDataset
from pairing import N_PERTURBED_DATASETS, load_pairing_bank
from time_coordinates import TIME_SCHEMA

VAR_NAMES = {"mslp": "mean_sea_level_pressure", "t2m": "2m_temperature"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", choices=sorted(VAR_NAMES), default="mslp")
    parser.add_argument("--ckpt-version", default="txt")
    parser.add_argument("--era5-ens-dir", default=ERA5_ENS_DIR)
    parser.add_argument(
        "--pairing-dir",
        type=Path,
        help="Directory containing pairing_dataset00..09.json and manifest.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Validated normalized NPY cache created by train_txt_uq.",
    )
    parsed = parser.parse_args()
    if parsed.cache_dir is None:
        parser.error("TxT-UQ Source-2 J0 requires --cache-dir")
    if parsed.pairing_dir is None:
        parser.error("TxT-UQ Source-2 J0 requires --pairing-dir")
    return parsed


def _dataloader_kwargs(config):
    num_workers = int(config.get("num_workers", 0))
    kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


def training_date_indices(init_times: pd.DatetimeIndex, config) -> np.ndarray:
    start = pd.Timestamp(config["train_init_start"])
    end = pd.Timestamp(config["train_init_end"])
    indices = np.flatnonzero((init_times >= start) & (init_times <= end))
    if len(indices) == 0:
        raise RuntimeError(
            f"No training dates in [{start.date()}, {end.date()}], archive spans {init_times.min()}..{init_times.max()}"
        )
    return indices


def compute_dataset_loss(model, dataset, train_indices, config, label):
    loader = DataLoader(Subset(dataset, train_indices), **_dataloader_kwargs(config))
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=label):
            x = x.to("cuda", dtype=torch.float32)
            y = y.to("cuda", dtype=torch.float32)
            loss = model.latitude_weighted_mse(model.predict_corrected(x), y)
            batch_size = int(x.shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError(f"torch_data {label} yielded no J0 samples")
    return (loss_sum, sample_count)


def _validate_paired_cache(cache, *, var, normalizer_mode):
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing Source-2 cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "schema": ARTIFACT_SCHEMAS["paired_training_cache"],
        "var": var,
        "year_range": list(CONFIG["year_range"]),
        "ifs_members": list(range(1, 11)),
        "eda_members": list(range(10)),
        "normalizer": normalizer_mode,
        "input_channels": 1,
        "time_schema": TIME_SCHEMA,
        "normalizer_provenance": normalizer_provenance(normalizer_mode),
        "valid_day_offsets": list(range(32)),
        "init_start": CONFIG["train_init_start"],
        "init_end": CONFIG["val_init_end"],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"TxT-UQ cache manifest mismatch: {mismatches}")
    validate_raw_source_identity_record(
        manifest.get("raw_source_identity"), expected_source_kind="source2_ifs_eda"
    )
    validate_cache_for_consumer(cache, manifest)
    return manifest


def _load_pairings(pairing_dir: Path, init_times: pd.DatetimeIndex):
    pairings, _ = load_pairing_bank(pairing_dir, expected_dates=init_times)
    return pairings


def compute_perturbed_target_loss(model, config, args):
    cache = args.cache_dir
    _validate_paired_cache(cache, var=args.var, normalizer_mode="doy_full")
    init_times = pd.DatetimeIndex(np.load(cache / "init_times.npy", allow_pickle=True))
    era5_times = pd.DatetimeIndex(np.load(cache / "era5_times.npy", allow_pickle=True))
    train_indices = training_date_indices(init_times, config)
    pairings = _load_pairings(args.pairing_dir, init_times)
    all_ifs = {
        member: np.load(cache / f"ifs_norm_{member:02d}.npy", mmap_mode="r")
        for member in range(1, 11)
    }
    all_era5 = {
        logical_member: np.load(
            cache / f"era5_eda_{logical_member - 1:02d}.npy", mmap_mode="r"
        )
        for logical_member in range(1, 11)
    }
    total_loss = 0.0
    total_samples = 0
    for dataset_index in range(N_PERTURBED_DATASETS):
        dataset = PairedForecastDataset(
            all_ifs=all_ifs,
            all_era5=all_era5,
            training_var=VAR_NAMES[args.var],
            pairing_per_date=pairings[dataset_index],
            init_times=init_times,
            era5_times=era5_times,
        )
        loss_sum, sample_count = compute_dataset_loss(
            model,
            dataset,
            train_indices,
            config,
            f"PerturbedDataset{dataset_index:02d}",
        )
        average = loss_sum / sample_count
        print(
            f"  PerturbedDataset{dataset_index:02d}: {average:.8f} ({sample_count} dates)"
        )
        total_loss += loss_sum
        total_samples += sample_count
        del dataset
        gc.collect()
    return total_loss / total_samples


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    torch.set_float32_matmul_precision("medium")
    resolved = resolve_txt_model_bank(
        CHECKPOINT_DIR,
        variable=args.var,
        version=args.ckpt_version,
        expected_model_count=10,
        expected_arch="sp_full",
        expected_normalizer="doy_full",
        expected_normalizer_provenance=normalizer_provenance("doy_full"),
        require_complete=False,
    )
    checkpoint = dict(resolved)[0, 0]
    print(f"Loading fixed w0 from {checkpoint}")
    provenance = normalizer_provenance("doy_full")
    model = (
        load_txt_checkpoint(
            checkpoint,
            expected_arch="sp_full",
            expected_training_variable=args.var,
            expected_normalizer_provenance=provenance,
        )
        .eval()
        .to("cuda")
    )
    result = compute_perturbed_target_loss(model, CONFIG, args)
    if not np.isfinite(result) or result <= 0:
        raise RuntimeError(f"J0 must be positive and finite, got {result}")
    print(f"Final Average Loss: {result:.12g}")
    print(result)


if __name__ == "__main__":
    main()
