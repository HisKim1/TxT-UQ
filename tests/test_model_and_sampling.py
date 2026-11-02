import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from txt_model import TxT
from pairing import generate_pairings, validate_pairings
from residuals import apply_physical_residuals, predictions_to_physical


def test_main_model_shape_and_scored_loss():
    torch.set_num_threads(2)
    model = TxT().eval()
    x = torch.zeros(1, 1, 32, 17, 24)
    with torch.inference_mode():
        prediction = model.predict_corrected(x)
    assert prediction.shape == x.shape
    assert torch.isfinite(prediction).all()
    target = x.clone()
    target[:, :, :18] = 100
    assert model.latitude_weighted_mse(x, target) == 0


def test_pairings_are_deterministic_and_deranged():
    dates = pd.date_range("2021-02-01", periods=27)
    pairings = generate_pairings(dates)
    assert pairings == generate_pairings(dates)
    validate_pairings(pairings, dates)
    for row in pairings.values():
        assert all(
            (pair["ifs_member"] != pair["era5_eda_member"] for pair in row.values())
        )


def test_checkpoint_parameter_names_remain_loadable():
    model = TxT()
    current = model.state_dict()
    legacy_names = (
        (".bottleneck.", ".bottleNeck."),
        (".gating_projection.", ".W_g."),
        (".skip_projection.", ".W_x."),
    )
    legacy = {}
    for key, value in current.items():
        for new, old in legacy_names:
            key = key.replace(new, old)
        legacy[key] = value
    checkpoint = {"state_dict": legacy}
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert current.keys() == checkpoint["state_dict"].keys()
    key = next(key for key in current if ".bottleneck." in key)
    duplicate = dict(current)
    duplicate[key.replace(".bottleneck.", ".bottleNeck.")] = current[key]
    with pytest.raises(ValueError, match="Conflicting checkpoint parameter keys"):
        model.on_load_checkpoint({"state_dict": duplicate})


def test_main_samples_match_every_model_with_five_analogs_per_testing_dataset():
    source = Path(__file__).resolve().parents[1] / "uq/sample_txt_uq.py"
    tree = ast.parse(source.read_text())
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = dict(
        np=np,
        pd=pd,
        torch=torch,
        xr=xr,
        MODEL_BATCH=10,
        predictions_to_physical=predictions_to_physical,
        apply_physical_residuals=apply_physical_residuals,
    )
    exec(compile(definitions, str(source), "exec"), namespace)
    dates = pd.date_range("2021-02-01", periods=6)
    namespace["args"] = SimpleNamespace(
        _residual_member_by_pool=[np.full(6, pool + 1) for pool in range(10)],
        _residual_pool_indices=list(range(10)),
    )
    namespace.update(
        residual_dir="unused", variable="mslp", existing_xarray_store=lambda path: path
    )

    def residual(path, *, analog_date, expected_perturbed_dataset_index, **kwargs):
        day = (pd.Timestamp(analog_date) - dates[0]).days
        value = expected_perturbed_dataset_index + day * 0.1
        values = np.arange(100, dtype=np.float32)[:, None, None, None] * 0.01 + value
        return (torch.from_numpy(np.broadcast_to(values, (100, 32, 1, 1)).copy()), {})

    namespace["load_validated_residual"] = residual

    class Normalizer:

        def denormalize_grid(self, values, *args):
            return values * 2 + 10

    models = [
        SimpleNamespace(predict_corrected=lambda x, i=i: x + i * 0.01)
        for i in range(100)
    ]
    call = dict(
        x_new=torch.zeros(1, 1, 32, 1, 1),
        current_date=dates[0],
        analog_dates=[dates] * 10,
        coords={"valid_day_offset": np.arange(32)},
        norm=Normalizer(),
        w1_idx=np.arange(18, 25),
        w2_idx=np.arange(25, 32),
        model_bank=models,
        analog_inputs=[
            np.arange(6, dtype=np.float32)[:, None, None, None]
            * np.ones((1, 32, 1, 1), dtype=np.float32)
        ]
        * 10,
        lat_weights=np.ones(1, dtype=np.float32),
    )
    mean, week3, week4, count, model_index, analog_rank, records = namespace[
        "sample_forecast_date"
    ](**call)
    assert count == 5000 and len(records) == 50
    expected = 10 + model_index * 0.03 + analog_rank // 5 + analog_rank % 5 * 0.1
    np.testing.assert_allclose(week3[:, 0, 0], expected, rtol=1e-06)
    np.testing.assert_array_equal(week3, week4)
    np.testing.assert_allclose(mean, expected.mean(), rtol=1e-06)
    namespace["existing_xarray_store"] = lambda path: None
    with pytest.raises(FileNotFoundError):
        namespace["sample_forecast_date"](**call)
