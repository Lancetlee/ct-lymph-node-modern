"""Synthetic throughput checks; no diagnostic or segmentation-quality claims."""
from __future__ import annotations

from contextlib import nullcontext
import os
import platform
import statistics
import time
from typing import Any, Sequence

import torch

from .device import resolve_device, synchronize, validate_precision
from .model import ModelConfig, build_model


def benchmark_model(
    device: str = "auto", *, roi_size: Sequence[int] = (32, 32, 32),
    batch_size: int = 1, warmup: int = 1, iterations: int = 3,
    amp: bool = False, config: ModelConfig | None = None,
) -> dict[str, Any]:
    roi = tuple(roi_size)
    if len(roi) != 3 or any(type(n) is not int or n < 2 for n in roi):
        raise ValueError("roi_size must contain three integers >=2")
    if any(type(n) is not int for n in (batch_size, warmup, iterations)) or batch_size < 1 or warmup < 0 or iterations < 1:
        raise ValueError("Invalid benchmark batch size, warmup, or iteration count")
    selected = resolve_device(device)
    validate_precision(selected, amp)
    config = config or ModelConfig(base_channels=4, levels=3)
    model = build_model(config).to(device=selected, dtype=torch.float32).eval()
    source = torch.zeros((batch_size, config.in_channels) + roi, dtype=torch.float32, device=selected)
    timings = []
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    with torch.inference_mode():
        for i in range(warmup + iterations):
            synchronize(selected)
            start = time.perf_counter()
            autocast = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
            with autocast:
                output = model(source)
            synchronize(selected)
            elapsed = time.perf_counter() - start
            if not torch.isfinite(output).all().item():
                raise RuntimeError("Synthetic benchmark produced non-finite values")
            if i >= warmup:
                timings.append(elapsed)
    report: dict[str, Any] = {
        "purpose": "synthetic_execution_benchmark", "trained": False,
        "clinical_validation": False, "device": str(selected),
        "torch_version": str(torch.__version__), "architecture": model.architecture,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python_version": platform.python_version(),
        "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "mps_cpu_fallback_env": os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset'),
        "model_config": config.to_dict(), "roi_dhw": list(roi), "batch_size": batch_size,
        "precision": "cuda_fp16_autocast" if amp else "float32",
        "iterations": iterations, "warmup": warmup,
        "latency_seconds": timings, "median_seconds": statistics.median(timings),
        "patches_per_second": batch_size / statistics.median(timings),
        "includes_io_preprocessing_or_sliding_window": False,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if selected.type == "cuda":
        report['device_name'] = torch.cuda.get_device_name(selected)
        report["peak_tensor_bytes"] = torch.cuda.max_memory_allocated(selected)
    elif selected.type == "mps":
        report["current_tensor_bytes"] = torch.mps.current_allocated_memory()
    return report
