"""Explicit CPU/CUDA/Metal selection and execution diagnostics."""
from __future__ import annotations

import os
import platform
from typing import Any

import torch


class DeviceError(RuntimeError):
    pass


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    name = str(requested).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if _mps_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise DeviceError(f"Invalid device {requested!r}; choose auto, cpu, mps, or cuda[:index]") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError("CUDA was requested but is unavailable in this PyTorch/runtime")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise DeviceError(f"CUDA index {device.index} is unavailable")
    elif device.type == "mps":
        if not _mps_available():
            raise DeviceError("MPS was requested but is unavailable; run doctor on the target Apple Silicon Mac")
        if device.index not in (None, 0):
            raise DeviceError("Only the default MPS device is supported")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise DeviceError("PYTORCH_ENABLE_MPS_FALLBACK=1 permits hidden CPU execution; unset it explicitly for an MPS-only run")
    elif device.type != "cpu":
        raise DeviceError("Only CPU, CUDA, and MPS devices are supported")
    return device


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def validate_precision(device: torch.device, amp: bool) -> None:
    if type(amp) is not bool:
        raise ValueError("amp must be bool")
    if amp and device.type != "cuda":
        raise DeviceError("Optional FP16 autocast is limited to CUDA; MPS and CPU use FP32")


def device_report(requested: str = "auto", *, run_smoke: bool = True) -> dict[str, Any]:
    """Probe this machine, never infer support from an Apple chip model name."""
    backend = getattr(torch.backends, "mps", None)
    report: dict[str, Any] = {
        "torch_version": str(torch.__version__), "platform": platform.platform(),
        "machine": platform.machine(), "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "mps_built": bool(backend is not None and backend.is_built()),
        "mps_available": _mps_available(), "requested_device": requested,
        "mps_cpu_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "default_precision": "float32", "clinical_validation": False,
    }
    try:
        device = resolve_device(requested)
        report["selected_device"] = str(device)
        if device.type == "cuda":
            report["device_name"] = torch.cuda.get_device_name(device)
        if run_smoke:
            from .model import ModelConfig, build_model
            model = build_model(ModelConfig(base_channels=2, levels=2)).to(device)
            x = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32, device=device)
            output = model(x)
            loss = output.square().mean()
            loss.backward()
            synchronize(device)
            if not torch.isfinite(output).all().item():
                raise DeviceError("3D network smoke test produced non-finite values")
            report["smoke_test"] = {"passed": True, "forward_backward": True,
                                    "shape": list(output.shape), "precision": "float32"}
            del model, output, loss, x
        else:
            report["smoke_test"] = {"passed": None, "reason": "not requested"}
    except (RuntimeError, ValueError) as exc:
        report["smoke_test"] = {"passed": False, "error": str(exc)}
    return report


doctor = device_report
