import json

import pytest

torch = pytest.importorskip("torch")
from ctln.parity import validate_backend


def test_cpu_parity_exercises_forward_and_all_gradients():
    report = validate_backend("cpu")
    assert report["passed"] is True
    assert report["precision"] == "float32"
    assert report["trained"] is False and report["clinical"] is False
    assert report["accelerator_validated"] is False
    assert report["comparison_scope"] == "cpu_self_check"
    assert report["forward"]["allclose"] and report["input_gradient"]["allclose"]
    assert report["parameter_gradients"]
    assert all(item["allclose"] and item["target_finite"] for item in report["parameter_gradients"].values())
    assert report["forward"]["max_abs_difference"] == 0.0
    assert report["loss"]["both_finite"] is True
    json.dumps(report, allow_nan=False)


def test_same_seed_reproduces_model_state_without_changing_caller_rng():
    before = torch.random.get_rng_state().clone()
    first = validate_backend("cpu", roi_size=(8, 8, 8), seed=123)
    assert torch.equal(before, torch.random.get_rng_state())
    second = validate_backend("cpu", roi_size=(8, 8, 8), seed=123)
    third = validate_backend("cpu", roi_size=(8, 8, 8), seed=124)
    assert first["passed"] and second["passed"] and third["passed"]
    assert first["state_dict_sha256"] == second["state_dict_sha256"]
    assert first["state_dict_sha256"] != third["state_dict_sha256"]


def test_unavailable_mps_is_reported_without_a_cpu_substitution(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    report = validate_backend("mps")
    assert report["passed"] is False
    assert report["device_used"] is None
    assert report["accelerator_validated"] is False
    assert "MPS" in report["error"]
    assert "No alternate-device retry" in report["error"]
    assert "forward" not in report
    json.dumps(report, allow_nan=False)


def test_invalid_tolerances_or_shapes_are_rejected():
    with pytest.raises(ValueError):
        validate_backend("cpu", rtol=float("nan"))
    with pytest.raises(ValueError):
        validate_backend("cpu", roi_size=(9, 9))
