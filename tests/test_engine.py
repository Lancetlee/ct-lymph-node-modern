"""Engine invariants checked with tiny synthetic data only."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ctln.device import DeviceError, device_report, resolve_device, validate_precision
from ctln.inference import labels_from_probabilities, patch_starts, sliding_window_predict
from ctln.model import (CheckpointError, ModelConfig, build_model, load_checkpoint,
                        make_checkpoint_metadata, save_checkpoint, validate_metadata)


class Pointwise(torch.nn.Module):
    num_classes = 2

    def forward(self, x):
        return torch.cat((-x, x), dim=1)


def test_new_model_odd_dimensions_and_backward():
    model = build_model(ModelConfig(base_channels=2, levels=3))
    x = torch.randn(1, 1, 9, 11, 13)
    output = model(x)
    assert output.shape == (1, 2, 9, 11, 13)
    output.square().mean().backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert not any(isinstance(layer, torch.nn.ConvTranspose3d) for layer in model.modules())


@pytest.mark.parametrize("shape,roi,batch", [((5, 7, 9), (4, 4, 4), 3), ((2, 3, 2), (4, 4, 4), 1)])
def test_gaussian_overlap_matches_pointwise_without_holes(shape, roi, batch):
    source = np.random.default_rng(42).normal(size=(1,) + shape).astype(np.float32)
    original = source.copy()
    model = Pointwise()
    result = sliding_window_predict(model, source, roi, batch_size=batch, device="cpu", allow_untrained=True)
    expected = torch.softmax(model(torch.from_numpy(source)[None]), dim=1)[0].numpy()
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.sum(axis=0), 1.0, atol=2e-6)
    np.testing.assert_array_equal(source, original)
    assert model.training  # Restore caller mode.
    assert labels_from_probabilities(result).shape == shape


def test_patch_starts_covers_the_last_voxel():
    assert patch_starts(11, 4, 0.5) == [0, 2, 4, 6, 7]
    assert patch_starts(2, 4, 0.5) == [0]
    with pytest.raises(ValueError):
        patch_starts(11, 4, 1.0)


def test_random_weights_rejected_and_buffer_cap_checked():
    source = np.zeros((1, 6, 6, 6), dtype=np.float32)
    with pytest.raises(CheckpointError, match="trained"):
        sliding_window_predict(Pointwise(), source, (4, 4, 4), device="cpu")
    with pytest.raises(MemoryError, match="accumulation"):
        sliding_window_predict(Pointwise(), source, (4, 4, 4), device="cpu",
                               max_accumulator_gib=1e-12, allow_untrained=True)
    source[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sliding_window_predict(Pointwise(), source, (4, 4, 4), device="cpu", allow_untrained=True)


def test_checkpoint_roundtrip_and_untrained_guard(tmp_path):
    model = build_model(ModelConfig(base_channels=2, levels=2))
    metadata = make_checkpoint_metadata(model.config)
    path = tmp_path / "self-test.pt"
    save_checkpoint(path, model, metadata)
    with pytest.raises(CheckpointError, match="Untrained"):
        load_checkpoint(path)
    restored, loaded = load_checkpoint(path, allow_untrained=True)
    assert loaded == metadata
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    assert restored.checkpoint_metadata["trained"] is False


def test_trained_requires_provenance_and_metadata_is_strict():
    config = ModelConfig(base_channels=2, levels=2)
    with pytest.raises(CheckpointError, match="training_provenance"):
        make_checkpoint_metadata(config, trained=True)
    metadata = make_checkpoint_metadata(config, trained=True, training_provenance={
        "dataset": "synthetic test only", "training_run": "one explicit fixture run",
        "validation": "not clinically validated", "steps": 1, "loss": 0.5})
    assert metadata["trained"] is True
    invalid = copy.deepcopy(metadata)
    invalid["metastasis_probability"] = 0.9
    with pytest.raises(CheckpointError):
        validate_metadata(invalid)
    invalid = copy.deepcopy(metadata)
    invalid["preprocessing"]["hu_window"] = [240, -160]
    with pytest.raises(CheckpointError):
        validate_metadata(invalid)


def test_legacy_and_wrong_state_are_not_silently_loaded(tmp_path):
    with pytest.raises(CheckpointError, match="not converted"):
        load_checkpoint(tmp_path / "legacy.h5")
    config = ModelConfig(base_channels=2, levels=2)
    path = tmp_path / "bad.pt"
    torch.save({"metadata": make_checkpoint_metadata(config), "state_dict": {"wrong": torch.zeros(1)}}, path)
    with pytest.raises(CheckpointError, match="names/shapes"):
        load_checkpoint(path, allow_untrained=True)


def test_explicit_unavailable_accelerator_has_no_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"
    with pytest.raises(DeviceError):
        resolve_device("cuda")
    with pytest.raises(DeviceError):
        resolve_device("mps")
    with pytest.raises(DeviceError):
        validate_precision(torch.device("mps"), True)
    report = device_report("cpu", run_smoke=False)
    assert report["selected_device"] == "cpu"
    assert report["clinical_validation"] is False


def test_mps_cpu_fallback_is_never_silently_enabled(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(DeviceError, match="hidden CPU"):
        resolve_device("mps")
