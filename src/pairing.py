from __future__ import annotations
from config import ARTIFACT_SCHEMAS
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np
import pandas as pd

PAIRING_SCHEMA = ARTIFACT_SCHEMAS["pairing"]
PAIRING_SEED = 42
N_PERTURBED_DATASETS = 10
PAIRING_BLOCK_SIZE = N_PERTURBED_DATASETS
PAIRING_SEGMENTS = ("training", "buffer", "testing")


def _stable_block_seed(date_texts: list[str], seed: int, namespace: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{namespace}:{','.join(date_texts)}".encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little") % 2**31


def _canonical_dates(init_dates: Iterable) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(init_dates).sort_values()
    if dates.hasnans:
        raise ValueError("Perturbed-dataset initialization dates contain NaT")
    if dates.has_duplicates:
        duplicate = dates[dates.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate initialization dates: {duplicate[:10]}")
    return dates


def split_pairing_dates(
    init_dates,
    *,
    training_start,
    training_end,
    heldout_start,
    heldout_end,
    evaluation_start,
) -> dict[str, pd.DatetimeIndex]:
    dates = _canonical_dates(init_dates)
    training_start = pd.Timestamp(training_start)
    training_end = pd.Timestamp(training_end)
    heldout_start = pd.Timestamp(heldout_start)
    heldout_end = pd.Timestamp(heldout_end)
    evaluation_start = pd.Timestamp(evaluation_start)
    if (
        not training_start
        <= training_end
        < heldout_start
        <= heldout_end
        < evaluation_start
    ):
        raise ValueError("Pairing ranges must satisfy training <= heldout < evaluation")
    if len(dates) == 0:
        raise ValueError("Pairing archive axis is empty")
    if dates.min() < training_start or dates.max() > heldout_end:
        raise ValueError(
            "Pairing archive dates must stay within training_start..heldout_end"
        )
    training = dates[(dates >= training_start) & (dates <= training_end)]
    heldout = dates[(dates >= heldout_start) & (dates <= heldout_end)]
    buffer = dates.difference(training).difference(heldout).sort_values()
    if len(training) == 0 or len(heldout) == 0:
        raise ValueError("Training and held-out pairing scopes must be non-empty")
    assigned = training.union(buffer).union(heldout).sort_values()
    if not assigned.equals(dates):
        raise RuntimeError(
            "Pairing partition dropped or duplicated initialization dates"
        )
    return {"training": training, "buffer": buffer, "testing": heldout}


def _latin_block(
    date_texts: list[str], seed: int, namespace: str
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(_stable_block_seed(date_texts, seed, namespace))
    rows = rng.permutation(PAIRING_BLOCK_SIZE)
    columns = rng.permutation(PAIRING_BLOCK_SIZE)[: len(date_texts)]
    symbols = rng.permutation(PAIRING_BLOCK_SIZE)
    eda_slope = int(rng.choice([3, 7, 9]))
    eda_offset = int(rng.choice([1, 3, 5, 7, 9]))
    ifs_indices = (rows[:, None] + columns[None, :]) % PAIRING_BLOCK_SIZE
    eda_indices = (
        rows[:, None] + eda_slope * columns[None, :] + eda_offset
    ) % PAIRING_BLOCK_SIZE
    ifs = symbols[ifs_indices] + 1
    eda = symbols[eda_indices] + 1
    if np.any(ifs == eda):
        raise RuntimeError("Latin construction produced an equal IFS/ERA5 pair")
    return (ifs, eda)


def generate_pairings(
    init_dates, seed: int = PAIRING_SEED, *, namespace: str = "standalone"
) -> dict[int, dict]:
    dates = _canonical_dates(init_dates)
    pairings = {index: {} for index in range(N_PERTURBED_DATASETS)}
    for start in range(0, len(dates), PAIRING_BLOCK_SIZE):
        block = dates[start : start + PAIRING_BLOCK_SIZE]
        date_texts = block.strftime("%Y%m%d").tolist()
        ifs_block, eda_block = _latin_block(date_texts, seed, namespace)
        for dataset_index in range(N_PERTURBED_DATASETS):
            for column, date_text in enumerate(date_texts):
                pairings[dataset_index][date_text] = {
                    "ifs_member": int(ifs_block[dataset_index, column]),
                    "era5_eda_member": int(eda_block[dataset_index, column]),
                }
    validate_pairings(pairings, dates)
    return pairings


def validate_pairings(
    pairings: Mapping, init_dates=None, *, validate_blocks: bool = True
) -> None:
    normalized = {int(key): value for key, value in pairings.items()}
    expected_indices = set(range(N_PERTURBED_DATASETS))
    if set(normalized) != expected_indices:
        raise ValueError(
            f"Perturbed-dataset indices must be exactly 0..9: got {sorted(normalized)}"
        )
    if init_dates is None:
        first_dates = set(normalized[0])
        dates = pd.DatetimeIndex(pd.to_datetime(sorted(first_dates), format="%Y%m%d"))
    else:
        dates = _canonical_dates(init_dates)
    expected_dates = {date.strftime("%Y%m%d") for date in dates}
    for dataset_index, mapping in normalized.items():
        actual_dates = set(mapping)
        if actual_dates != expected_dates:
            raise ValueError(
                f"torch_data {dataset_index} date inventory differs: missing={sorted(expected_dates - actual_dates)[:10]}, extra={sorted(actual_dates - expected_dates)[:10]}"
            )
    logical_members = list(range(1, N_PERTURBED_DATASETS + 1))
    for date_text in sorted(expected_dates):
        ifs_members = []
        eda_members = []
        for dataset_index in range(N_PERTURBED_DATASETS):
            pair = normalized[dataset_index][date_text]
            if not isinstance(pair, Mapping):
                raise ValueError(
                    f"torch_data {dataset_index} {date_text} pairing is not a mapping"
                )
            ifs_member = pair.get("ifs_member")
            eda_member = pair.get("era5_eda_member")
            if (
                not isinstance(ifs_member, int)
                or isinstance(ifs_member, bool)
                or ifs_member not in logical_members
                or (not isinstance(eda_member, int))
                or isinstance(eda_member, bool)
                or (eda_member not in logical_members)
            ):
                raise ValueError(
                    f"Invalid logical members for dataset {dataset_index} on {date_text}: {pair}"
                )
            if ifs_member == eda_member:
                raise ValueError(
                    f"Equal IFS/ERA5 logical member {ifs_member} paired for dataset {dataset_index} on {date_text}"
                )
            ifs_members.append(ifs_member)
            eda_members.append(eda_member)
        if sorted(ifs_members) != logical_members:
            raise ValueError(
                f"IFS members are not bijective on {date_text}: {ifs_members}"
            )
        if sorted(eda_members) != logical_members:
            raise ValueError(
                f"ERA5 EDA members are not bijective on {date_text}: {eda_members}"
            )
    if not validate_blocks:
        return
    ordered_dates = dates.strftime("%Y%m%d").tolist()
    for start in range(0, len(ordered_dates), PAIRING_BLOCK_SIZE):
        block = ordered_dates[start : start + PAIRING_BLOCK_SIZE]
        for dataset_index in range(N_PERTURBED_DATASETS):
            ifs_members = [
                normalized[dataset_index][date]["ifs_member"] for date in block
            ]
            eda_members = [
                normalized[dataset_index][date]["era5_eda_member"] for date in block
            ]
            if len(set(ifs_members)) != len(block):
                raise ValueError(
                    f"torch_data {dataset_index} repeats an IFS member in block {block[0]}..{block[-1]}: {ifs_members}"
                )
            if len(set(eda_members)) != len(block):
                raise ValueError(
                    f"torch_data {dataset_index} repeats an ERA5 member in block {block[0]}..{block[-1]}: {eda_members}"
                )


def validate_pairing_mapping(
    mapping: Mapping,
    init_dates,
    dataset_index: int,
    *,
    seed: int = PAIRING_SEED,
    namespace: str = "standalone",
) -> str:
    dataset_index = int(dataset_index)
    if dataset_index not in range(N_PERTURBED_DATASETS):
        raise ValueError(f"Perturbed-dataset index must be 0..9, got {dataset_index}")
    canonical = {
        str(date): {
            "ifs_member": pair.get("ifs_member") if isinstance(pair, Mapping) else None,
            "era5_eda_member": (
                pair.get("era5_eda_member") if isinstance(pair, Mapping) else None
            ),
        }
        for date, pair in mapping.items()
    }
    expected = generate_pairings(init_dates, seed=seed, namespace=namespace)[
        dataset_index
    ]
    if canonical != expected:
        raise ValueError(
            f"Pairing does not match {PAIRING_SCHEMA} seed {seed} for perturbed dataset {dataset_index}"
        )
    return mapping_sha256(canonical)


def date_inventory_sha256(init_dates) -> str:
    dates = _canonical_dates(init_dates)
    payload = dates.strftime("%Y%m%d").tolist()
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def slice_pairings(pairings: Mapping, init_dates) -> dict[int, dict]:
    dates = _canonical_dates(init_dates)
    date_texts = dates.strftime("%Y%m%d").tolist()
    normalized = {int(key): value for key, value in pairings.items()}
    selected = {
        dataset_index: {
            date_text: normalized[dataset_index][date_text] for date_text in date_texts
        }
        for dataset_index in range(N_PERTURBED_DATASETS)
    }
    validate_pairings(selected, dates, validate_blocks=False)
    return selected


def generate_segmented_pairings(segment_dates: Mapping[str, Iterable]):
    if set(segment_dates) != set(PAIRING_SEGMENTS):
        raise ValueError(
            f"Pairing segments must be exactly {PAIRING_SEGMENTS}, got {tuple(segment_dates)}"
        )
    pairings = {index: {} for index in range(N_PERTURBED_DATASETS)}
    normalized_segments = {}
    all_dates = set()
    for name in PAIRING_SEGMENTS:
        values = segment_dates[name]
        dates = _canonical_dates(values)
        overlap = all_dates.intersection(dates)
        if overlap:
            raise ValueError(f"Pairing segments overlap at {sorted(overlap)[:10]}")
        all_dates.update(dates)
        normalized_segments[name] = dates
        segment_pairings = generate_pairings(dates, namespace=name)
        for dataset_index in range(N_PERTURBED_DATASETS):
            pairings[dataset_index].update(segment_pairings[dataset_index])
    return (pairings, normalized_segments)


def pairing_manifest(pairings: Mapping, segment_dates: Mapping[str, Iterable]) -> dict:
    _, segments = generate_segmented_pairings(segment_dates)
    all_dates = pd.DatetimeIndex(
        np.concatenate([dates.values for dates in segments.values()])
    ).sort_values()
    actual_dates = set(pairings[0])
    expected_dates = set(all_dates.strftime("%Y%m%d"))
    if actual_dates != expected_dates:
        raise ValueError("Pairing bank date inventory differs from its segments")
    validate_pairings(pairings, all_dates, validate_blocks=False)
    segment_metadata = {}
    for name, dates in segments.items():
        selected = slice_pairings(pairings, dates)
        expected = generate_pairings(dates, namespace=name)
        if selected != expected:
            raise ValueError(f"Pairing segment {name!r} differs from the recipe")
        segment_metadata[name] = {
            "n_dates": len(dates),
            "dates": dates.strftime("%Y%m%d").tolist(),
            "date_inventory_sha256": date_inventory_sha256(dates),
            "pairing_bank_sha256": mapping_sha256(selected),
            "n_full_blocks": len(dates) // PAIRING_BLOCK_SIZE,
            "final_partial_block_size": len(dates) % PAIRING_BLOCK_SIZE,
            "n_blocks": (len(dates) + PAIRING_BLOCK_SIZE - 1) // PAIRING_BLOCK_SIZE,
        }
    return {
        "schema": PAIRING_SCHEMA,
        "algorithm": "affine_randomized_paired_latin_square_v1",
        "block_definition": "ten_consecutive_sorted_initialization_records",
        "seed_derivation": "blake2b(seed,segment,ordered_block_dates)",
        "scope_global_no_year_reset": True,
        "seed": PAIRING_SEED,
        "block_size": PAIRING_BLOCK_SIZE,
        "n_perturbed_datasets": N_PERTURBED_DATASETS,
        "member_numbering": "IFS logical 1..10; ERA5 EDA logical 1..10 (storage 0..9)",
        "equal_logical_members_forbidden": True,
        "block_row_member_reuse_forbidden": True,
        "n_dates": len(all_dates),
        "date_inventory_sha256": date_inventory_sha256(all_dates),
        "mapping_sha256": {
            str(dataset_index): mapping_sha256(pairings[dataset_index])
            for dataset_index in range(N_PERTURBED_DATASETS)
        },
        "pairing_bank_sha256": mapping_sha256(pairings),
        "segments": segment_metadata,
    }


def load_pairing_bank(pairing_dir, *, expected_dates=None, segment=None):
    pairing_dir = Path(pairing_dir)
    manifest_path = pairing_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing pairing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_header = {
        "schema": PAIRING_SCHEMA,
        "algorithm": "affine_randomized_paired_latin_square_v1",
        "block_definition": "ten_consecutive_sorted_initialization_records",
        "seed_derivation": "blake2b(seed,segment,ordered_block_dates)",
        "scope_global_no_year_reset": True,
        "seed": PAIRING_SEED,
        "block_size": PAIRING_BLOCK_SIZE,
        "n_perturbed_datasets": N_PERTURBED_DATASETS,
        "member_numbering": "IFS logical 1..10; ERA5 EDA logical 1..10 (storage 0..9)",
        "equal_logical_members_forbidden": True,
        "block_row_member_reuse_forbidden": True,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected_header.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Pairing manifest mismatch: {mismatch}")
    segments = manifest.get("segments")
    if not isinstance(segments, Mapping) or set(segments) != set(PAIRING_SEGMENTS):
        raise RuntimeError("Pairing manifest has an invalid segment inventory")
    pairings = {}
    for dataset_index in range(N_PERTURBED_DATASETS):
        path = pairing_dir / f"pairing_dataset{dataset_index:02d}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing perturbed-dataset pairing: {path}")
        mapping = json.loads(path.read_text())
        actual_digest = mapping_sha256(mapping)
        if manifest.get("mapping_sha256", {}).get(str(dataset_index)) != actual_digest:
            raise RuntimeError(f"Pairing digest mismatch for dataset {dataset_index}")
        pairings[dataset_index] = mapping
    if manifest.get("pairing_bank_sha256") != mapping_sha256(pairings):
        raise RuntimeError("Pairing bank digest differs from its manifest")
    segment_dates = {}
    for name in PAIRING_SEGMENTS:
        metadata = segments[name]
        dates = _canonical_dates(
            pd.to_datetime(metadata.get("dates", []), format="%Y%m%d")
        )
        selected = slice_pairings(pairings, dates)
        if selected != generate_pairings(dates, namespace=name):
            raise RuntimeError(f"Pairing segment {name!r} differs from the recipe")
        expected_metadata = {
            "n_dates": len(dates),
            "date_inventory_sha256": date_inventory_sha256(dates),
            "pairing_bank_sha256": mapping_sha256(selected),
            "n_full_blocks": len(dates) // PAIRING_BLOCK_SIZE,
            "final_partial_block_size": len(dates) % PAIRING_BLOCK_SIZE,
            "n_blocks": (len(dates) + PAIRING_BLOCK_SIZE - 1) // PAIRING_BLOCK_SIZE,
        }
        metadata_mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if metadata_mismatch:
            raise RuntimeError(
                f"Pairing segment {name!r} metadata mismatch: {metadata_mismatch}"
            )
        segment_dates[name] = dates
    combined_dates = pd.DatetimeIndex(
        np.concatenate([dates.values for dates in segment_dates.values()])
    ).sort_values()
    validate_pairings(pairings, combined_dates, validate_blocks=False)
    if manifest.get("n_dates") != len(combined_dates) or manifest.get(
        "date_inventory_sha256"
    ) != date_inventory_sha256(combined_dates):
        raise RuntimeError("Pairing manifest combined date inventory is invalid")
    if segment is not None:
        if segment not in PAIRING_SEGMENTS:
            raise ValueError(f"Unknown pairing segment {segment!r}")
        selected_dates = segment_dates[segment]
        selected = slice_pairings(pairings, selected_dates)
        contract = {
            "schema": PAIRING_SCHEMA,
            "seed": PAIRING_SEED,
            "block_size": PAIRING_BLOCK_SIZE,
            "segment": segment,
            **segments[segment],
        }
    else:
        selected_dates = combined_dates
        selected = pairings
        contract = {
            "schema": PAIRING_SCHEMA,
            "seed": PAIRING_SEED,
            "block_size": PAIRING_BLOCK_SIZE,
            "segment": "all",
            "n_dates": len(combined_dates),
            "date_inventory_sha256": manifest["date_inventory_sha256"],
            "pairing_bank_sha256": manifest["pairing_bank_sha256"],
            "segments": {
                name: {
                    key: value
                    for key, value in segments[name].items()
                    if key != "dates"
                }
                for name in PAIRING_SEGMENTS
            },
        }
    if expected_dates is not None and (
        not _canonical_dates(expected_dates).equals(selected_dates)
    ):
        raise RuntimeError(
            f"Pairing {contract['segment']} date inventory differs from the consumer"
        )
    return (selected, contract)


def generate_pairing_lookup(init_dates, *, pairings: Mapping | None = None) -> dict:
    dates = _canonical_dates(init_dates)
    if pairings is None:
        raise ValueError(
            "A complete persisted pairing bank is required; date subsets must not be regenerated or re-blocked"
        )
    pairings = slice_pairings(pairings, dates)
    lookup = {}
    for date in dates:
        date_text = date.strftime("%Y%m%d")
        by_ifs = {}
        for dataset_index in range(N_PERTURBED_DATASETS):
            pair = pairings[dataset_index][date_text]
            ifs_member = pair["ifs_member"]
            eda_logical = pair["era5_eda_member"]
            by_ifs[ifs_member] = {
                "perturbed_dataset_index": dataset_index,
                "era5_eda_member": eda_logical,
                "era5_eda_storage_index": eda_logical - 1,
            }
        lookup[date_text] = by_ifs
    return lookup


def build_testing_dataset_member_matrix(
    init_dates, *, available_members, pairings: Mapping | None = None
) -> tuple[list[int], list[np.ndarray]]:
    dates = _canonical_dates(init_dates)
    members = sorted((int(value) for value in available_members))
    if members != list(range(1, 11)):
        raise ValueError("Paired testing data require IFS perturbed members 1..10")
    if pairings is None:
        raise ValueError(
            "Paired held-out data require the complete persisted testing bank"
        )
    pairings = slice_pairings(pairings, dates)
    rows = [
        np.asarray(
            [
                pairings[dataset_index][date.strftime("%Y%m%d")]["ifs_member"]
                for date in dates
            ],
            dtype=np.int16,
        )
        for dataset_index in range(N_PERTURBED_DATASETS)
    ]
    return (list(range(N_PERTURBED_DATASETS)), rows)


class PairedMemberArray:

    def __init__(self, arrays_by_member: Mapping, member_by_date):
        arrays = {
            int(member): np.asarray(values)
            for member, values in arrays_by_member.items()
        }
        if not arrays:
            raise ValueError("PairedMemberArray requires at least one member array")
        reference = next(iter(arrays.values()))
        if reference.ndim < 1:
            raise ValueError("Member arrays must have an initialization-date axis")
        for member, values in arrays.items():
            if values.shape != reference.shape:
                raise ValueError(
                    f"Member {member} shape {values.shape} differs from {reference.shape}"
                )
        selectors = np.asarray(member_by_date)
        if selectors.ndim != 1 or len(selectors) != reference.shape[0]:
            raise ValueError(
                "Date-wise member selectors must be one-dimensional and match the member arrays' first axis"
            )
        if not np.issubdtype(selectors.dtype, np.integer):
            raise ValueError("Date-wise member selectors must be integers")
        selectors = selectors.astype(np.int16, copy=False)
        missing = sorted(set((int(value) for value in selectors)) - set(arrays))
        if missing:
            raise ValueError(
                f"Date-wise selectors reference unloaded members: {missing}"
            )
        self.arrays_by_member = arrays
        self.member_by_date = selectors
        self.shape = reference.shape
        self.ndim = reference.ndim
        self.dtype = reference.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            raise TypeError("PairedMemberArray indexes only its date axis")
        indices = np.arange(self.shape[0])[key]
        if np.ndim(indices) == 0:
            index = int(indices)
            return self.arrays_by_member[int(self.member_by_date[index])][index]
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if len(indices) == 0:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        return np.stack(
            [
                self.arrays_by_member[int(self.member_by_date[index])][index]
                for index in indices
            ],
            axis=0,
        )


def mapping_sha256(mapping: Mapping) -> str:
    canonical = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def date_mapping_sha256(init_time, *, pairings: Mapping) -> str:
    date = pd.Timestamp(init_time)
    date_text = date.strftime("%Y%m%d")
    mapping = {
        date_text: {
            str(dataset_index): pairings[dataset_index][date_text]
            for dataset_index in range(N_PERTURBED_DATASETS)
        }
    }
    return mapping_sha256(mapping)
