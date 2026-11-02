from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from txt_model import file_sha256, load_txt_checkpoint
from config import CHECKPOINT_DIR, CONFIG, normalizer_provenance, ARTIFACT_SCHEMAS
from time_coordinates import TIME_SCHEMA
from cache import validate_raw_source_identity_record

COMPLETION_SCHEMA = ARTIFACT_SCHEMAS["reference_completion"]


def resolve_reference_checkpoint(
    variable: str,
    version: str,
    weight: int,
    *,
    architecture: str,
    normalizer: str,
    expected_cache_manifest_identity: str | None = None,
) -> Path:
    prefix = f"{variable}_{version}_00-{weight:02d}-"
    pattern = re.compile(f"^{re.escape(prefix)}.*\\.ckpt$")
    candidates = sorted(
        (
            path
            for path in Path(CHECKPOINT_DIR).glob(f"{prefix}*.ckpt")
            if pattern.match(path.name)
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    errors = []
    expected = {
        "schema": COMPLETION_SCHEMA,
        "var": variable,
        "version": version,
        "ensemble": 0,
        "weight": weight,
        "architecture": architecture,
        "normalizer": normalizer,
        "normalizer_provenance": normalizer_provenance(normalizer),
        "time_schema": TIME_SCHEMA,
        "input_mode": "member",
        "input_channels": 1,
        "source2": False,
        "train_init_start": CONFIG["train_init_start"],
        "train_init_end": CONFIG["train_init_end"],
        "val_init_start": CONFIG["val_init_start"],
        "val_init_end": CONFIG["val_init_end"],
    }
    for checkpoint in candidates:
        sidecar = checkpoint.with_suffix(".complete.json")
        try:
            row = json.loads(sidecar.read_text())
            mismatches = {
                key: (row.get(key), value)
                for key, value in expected.items()
                if row.get(key) != value
            }
            if row.get("checkpoint") != checkpoint.name:
                mismatches["checkpoint"] = (row.get("checkpoint"), checkpoint.name)
            digest = file_sha256(checkpoint)
            if row.get("checkpoint_sha256") != digest:
                mismatches["checkpoint_sha256"] = (row.get("checkpoint_sha256"), digest)
            cache_identity = row.get("cache_manifest_identity")
            if not (
                isinstance(cache_identity, str)
                and len(cache_identity) == 64
                and all(
                    (character in "0123456789abcdef" for character in cache_identity)
                )
            ):
                mismatches["cache_manifest_identity"] = (
                    cache_identity,
                    "64-character SHA-256",
                )
            if (
                expected_cache_manifest_identity is not None
                and cache_identity != expected_cache_manifest_identity
            ):
                mismatches["cache_manifest_identity"] = (
                    cache_identity,
                    expected_cache_manifest_identity,
                )
            cache_raw_source_identity = row.get("cache_raw_source_identity")
            try:
                validate_raw_source_identity_record(
                    cache_raw_source_identity, expected_source_kind="control_ifs_hres"
                )
            except RuntimeError as exc:
                mismatches["cache_raw_source_identity"] = (
                    cache_raw_source_identity,
                    str(exc),
                )
            if mismatches:
                raise RuntimeError(f"completion metadata mismatch: {mismatches}")
            model = load_txt_checkpoint(
                checkpoint,
                expected_arch=architecture,
                expected_training_variable=variable,
                expected_normalizer_provenance=normalizer_provenance(normalizer),
                expected_training_provenance={
                    "source2": False,
                    "perturbed_dataset_index": -1,
                    "weight_level": weight,
                    "training_seed": weight,
                    "weight_seed_schema": ARTIFACT_SCHEMAS["weight_seed"],
                    "pairing_schema": "none",
                    "pairing_seed": -1,
                    "pairing_mapping_sha256": "none",
                    "pairing_bank_sha256": "none",
                    "pairing_training_bank_sha256": "none",
                    "pairing_testing_bank_sha256": "none",
                    "cache_manifest_identity": cache_identity,
                    "cache_raw_source_identity": cache_raw_source_identity,
                    "train_init_start": CONFIG["train_init_start"],
                    "train_init_end": CONFIG["train_init_end"],
                    "val_init_start": CONFIG["val_init_start"],
                    "val_init_end": CONFIG["val_init_end"],
                },
                map_location="cpu",
            )
            del model
            return checkpoint
        except Exception as exc:
            errors.append(f"{checkpoint.name}: {exc}")
    detail = errors[:3] if errors else ["no matching checkpoints"]
    raise RuntimeError(
        f"No completed control checkpoint for {variable} weight {weight}: {detail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", required=True, choices=["mslp", "t2m"])
    parser.add_argument("--version", default="txt_reference")
    parser.add_argument("--weight", type=int, default=None)
    parser.add_argument("--expected-cache-manifest-sha256", default=None)
    args = parser.parse_args()
    weights = [args.weight] if args.weight is not None else list(range(10))
    if any((weight not in range(10) for weight in weights)):
        parser.error("--weight must be in 0..9")
    resolved = [
        resolve_reference_checkpoint(
            args.var,
            args.version,
            weight,
            architecture="sp_full",
            normalizer="doy_full",
            expected_cache_manifest_identity=args.expected_cache_manifest_sha256,
        )
        for weight in weights
    ]
    print("\n".join((str(path) for path in resolved)))


if __name__ == "__main__":
    main()
