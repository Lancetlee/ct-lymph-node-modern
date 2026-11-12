"""CPU-versus-backend numerical acceptance probe on a synthetic tiny network.

This checks the operators used by the new architecture. It neither validates
clinical accuracy nor converts/validates the legacy Keras model.
"""
from __future__ import annotations

import hashlib
import math
import platform
from typing import Any, Sequence

import torch

from .device import resolve_device, synchronize
from .model import ModelConfig, build_model


def _difference(reference: torch.Tensor, actual: torch.Tensor, *, rtol: float, atol: float) -> dict[str, Any]:
    # Network tensors are FP32; CPU-double statistics avoid overflow while
    # reporting a failed probe that produced very large but finite values.
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    actual = actual.detach().to(device="cpu", dtype=torch.float64)
    finite_reference = bool(torch.isfinite(reference).all().item())
    finite_actual = bool(torch.isfinite(actual).all().item())
    same_shape = reference.shape == actual.shape
    result: dict[str, Any] = {
        "reference_finite": finite_reference, "target_finite": finite_actual,
        "same_shape": same_shape, "allclose": False,
        "max_abs_difference": None, "max_relative_difference": None,
        "relative_l2_error": None,
    }
    # Do not put NaN/Infinity into an ostensibly portable JSON report.
    if not same_shape or not finite_reference or not finite_actual:
        return result
    delta = (reference - actual).abs()
    denominator = reference.abs().clamp_min(1e-8)
    result.update({
        "allclose": bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
        "max_abs_difference": float(delta.max().item()),
        "max_relative_difference": float((delta / denominator).max().item()),
        "relative_l2_error": float((torch.linalg.vector_norm(delta)
                                   / torch.linalg.vector_norm(reference).clamp_min(1e-8)).item()),
    })
    return result


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _forward_backward(model, source: torch.Tensor, selected: torch.device):
    model.zero_grad(set_to_none=True)
    # Explicitly disable any surrounding autocast context. Gradients must remain
    # enabled even when a caller otherwise normally performs inference only.
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(selected.type, enabled=False):
        x = source.detach().clone().to(device=selected, dtype=torch.float32).requires_grad_(True)
        output = model(x)
        if output.dtype != torch.float32:
            raise RuntimeError(f"Expected FP32 network output, received {output.dtype}")
        loss = output.square().mean()
        loss.backward()
    synchronize(selected)
    if x.grad is None:
        raise RuntimeError("The network did not produce an input gradient")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"Missing parameter gradient: {name}")
            gradients[name] = parameter.grad.detach().cpu().clone()
    return output.detach().cpu(), x.grad.detach().cpu(), gradients, float(loss.detach().cpu())


def validate_backend(
    device: str = "mps", *, roi_size: Sequence[int] = (9, 9, 9),
    seed: int = 42, rtol: float = 1e-3, atol: float = 1e-4,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Compare one identical synthetic FP32 network on CPU and ``device``.

    The default odd spatial dimensions also exercise decoder size alignment.
    A CPU request is a utility self-check, never evidence for MPS/CUDA support.
    An unavailable device or failed backend operation yields ``passed=False``;
    no alternate device is substituted and no CPU retry is attempted.
    """
    roi = tuple(roi_size)
    if len(roi) != 3 or any(type(n) is not int or n < 2 for n in roi):
        raise ValueError("roi_size must contain three integers >=2, ordered D,H,W")
    if type(seed) is not int or not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63-1)")
    if (type(rtol) not in (int, float) or type(atol) not in (int, float)
            or not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0):
        raise ValueError("rtol and atol must be finite nonnegative numbers")
    config = config or ModelConfig(base_channels=2, levels=2)
    report: dict[str, Any] = {
        "purpose": "synthetic_backend_numerical_parity", "passed": False,
        "trained": False, "clinical": False, "clinical_validation": False,
        "legacy_weight_conversion_validated": False,
        "reference_device": "cpu", "requested_device": str(device), "device_used": None,
        "accelerator_validated": False, "precision": "float32", "autocast": False,
        "platform": platform.platform(), "machine": platform.machine(),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "torch_config": torch.__config__.show(), "seed": seed,
        "input_shape_ncdhw": [1, config.in_channels, *roi],
        "model_config": config.to_dict(), "rtol": float(rtol), "atol": float(atol),
        "relative_difference_definition": "abs(target-cpu) / max(abs(cpu), 1e-8)",
        "relative_difference_note": "Near-zero reference values can have large relative errors while satisfying the absolute tolerance.",
        "backend_bitwise_determinism_guaranteed": False,
        "no_automatic_device_fallback": True,
    }
    try:
        selected = resolve_device(device)
        report["device_used"] = str(selected)
        report["comparison_scope"] = "cpu_self_check" if selected.type == "cpu" else "cpu_vs_requested_accelerator"
        if selected.type == "cuda":
            report["device_name"] = torch.cuda.get_device_name(selected)
            report["cuda_tf32_matmul_allowed"] = torch.backends.cuda.matmul.allow_tf32
            report["cudnn_tf32_allowed"] = torch.backends.cudnn.allow_tf32
        # Seed only the CPU generator and restore it afterwards. No global
        # accelerator RNG is changed; the architecture has no random layers.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            reference_model = build_model(config).to(device="cpu", dtype=torch.float32).eval()
            target_model = build_model(config)
        state = {key: value.detach().cpu().clone() for key, value in reference_model.state_dict().items()}
        report["state_dict_sha256"] = _state_digest(state)
        report["parameter_count"] = sum(parameter.numel() for parameter in reference_model.parameters())
        target_model.load_state_dict(state, strict=True)
        target_model = target_model.to(device=selected, dtype=torch.float32).eval()
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        source = torch.randn((1, config.in_channels) + roi, generator=generator, dtype=torch.float32)
        reference_output, reference_input_grad, reference_gradients, reference_loss = _forward_backward(
            reference_model, source, torch.device("cpu"))
        target_output, target_input_grad, target_gradients, target_loss = _forward_backward(target_model, source, selected)
        report["loss"] = {
            "reference": reference_loss if math.isfinite(reference_loss) else None,
            "target": target_loss if math.isfinite(target_loss) else None,
            "both_finite": math.isfinite(reference_loss) and math.isfinite(target_loss),
        }
        report["forward"] = _difference(reference_output, target_output, rtol=rtol, atol=atol)
        report["input_gradient"] = _difference(reference_input_grad, target_input_grad, rtol=rtol, atol=atol)
        if reference_gradients.keys() != target_gradients.keys():
            raise RuntimeError("CPU and target parameter-gradient names differ")
        report["parameter_gradients"] = {
            name: _difference(reference_gradients[name], target_gradients[name], rtol=rtol, atol=atol)
            for name in reference_gradients
        }
        comparisons = [report["forward"], report["input_gradient"], *report["parameter_gradients"].values()]
        report["passed"] = report["loss"]["both_finite"] and all(item["allclose"] for item in comparisons)
        report["accelerator_validated"] = bool(report["passed"] and selected.type in ("cuda", "mps"))
        if not report["passed"]:
            report["error"] = "Non-finite outputs/gradients or numerical differences exceeded the specified tolerances."
    except (RuntimeError, ValueError) as exc:
        report["passed"] = False
        report["accelerator_validated"] = False
        report["error"] = f"{type(exc).__name__}: {exc}. No alternate-device retry was made."
    return report
