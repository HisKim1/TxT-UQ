from __future__ import annotations
import numpy as np

RESIDUAL_TRANSFER_SPACE = "physical_output_error_v1"


def predictions_to_physical(
    normalizer, predictions_z, init_time, valid_day_offsets
) -> np.ndarray:
    values = np.asarray(predictions_z)
    physical = np.asarray(
        normalizer.denormalize_grid(values, init_time, valid_day_offsets),
        dtype=np.float32,
    )
    if physical.shape != values.shape:
        raise ValueError(
            f"Denormalization changed prediction shape {values.shape} -> {physical.shape}"
        )
    if not np.isfinite(physical).all():
        raise ValueError("Denormalized predictions contain NaN or Inf")
    return physical


def compute_physical_residuals(
    normalizer, predictions_z, target_physical, init_time, valid_day_offsets
) -> np.ndarray:
    predictions_physical = predictions_to_physical(
        normalizer, predictions_z, init_time, valid_day_offsets
    )
    target = np.asarray(target_physical, dtype=np.float32)
    if predictions_physical.shape[-target.ndim :] != target.shape:
        raise ValueError(
            f"Target shape {target.shape} is incompatible with predictions {predictions_physical.shape}"
        )
    residual = target - predictions_physical
    if not np.isfinite(residual).all():
        raise ValueError("Physical residuals contain NaN or Inf")
    return residual.astype(np.float32, copy=False)


def apply_physical_residuals(predictions_physical, residuals_physical) -> np.ndarray:
    prediction = np.asarray(predictions_physical, dtype=np.float32)
    residual = np.asarray(residuals_physical, dtype=np.float32)
    samples = prediction + residual
    if not np.isfinite(samples).all():
        raise ValueError("Residual-adjusted physical samples contain NaN or Inf")
    return samples.astype(np.float32, copy=False)
