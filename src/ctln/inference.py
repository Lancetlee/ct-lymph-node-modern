"""Sliding-window segmentation with FP32 Gaussian fusion on the CPU."""
from __future__ import annotations

from contextlib import nullcontext
from itertools import product
import math
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .device import resolve_device, validate_precision
from .model import CheckpointError, validate_metadata


def patch_starts(length: int, roi: int, overlap: float) -> list[int]:
    if type(length) is not int or type(roi) is not int or length < 1 or roi < 1:
        raise ValueError("length and roi must be positive integers")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0,1)")
    if length <= roi:
        return [0]
    stride = max(1, int(roi * (1.0 - overlap)))
    starts = list(range(0, length - roi + 1, stride))
    if starts[-1] != length - roi:
        starts.append(length - roi)
    return starts


def gaussian_importance(roi_size: Sequence[int], sigma_scale: float = 0.125) -> np.ndarray:
    if not math.isfinite(sigma_scale) or sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive and finite")
    weight = np.ones(tuple(roi_size), dtype=np.float32)
    for axis, length in enumerate(roi_size):
        coordinate = np.arange(length, dtype=np.float32) - (length - 1.0) / 2.0
        values = np.exp(-0.5 * (coordinate / (length * sigma_scale)) ** 2)
        shape = [1, 1, 1]
        shape[axis] = length
        weight *= values.reshape(shape)
    weight /= weight.max()
    return np.maximum(weight, np.float32(1e-4))


def _batches(coordinates: Iterator[tuple[int, int, int]], size: int):
    batch = []
    for coordinate in coordinates:
        batch.append(coordinate)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def sliding_window_predict(
    model: nn.Module, volume: np.ndarray | torch.Tensor,
    roi_size: Sequence[int] = (64, 128, 128), *, overlap: float = 0.5,
    batch_size: int = 1, device: str | torch.device = "auto", amp: bool = False,
    sigma_scale: float = 0.125, max_accumulator_gib: float = 4.0,
    allow_untrained: bool = False,
) -> np.ndarray:
    """Return C,D,H,W segmentation probabilities for a normalized CT tensor.

    Volume remains on the CPU; only current patches occupy accelerator memory.
    The cap covers the two accumulation buffers, not total process memory.
    ``allow_untrained`` is exclusively for synthetic tests/benchmarks; clinical
    CLI inference must never expose or enable it.
    """
    metadata = getattr(model, "checkpoint_metadata", None)
    if not allow_untrained and (not isinstance(metadata, dict) or metadata.get("trained") is not True):
        raise CheckpointError("Inference requires a trained checkpoint; randomly initialized models are refused")
    if not allow_untrained:
        metadata = validate_metadata(metadata)
    roi = tuple(roi_size)
    if len(roi) != 3 or any(type(n) is not int or n < 2 for n in roi):
        raise ValueError("roi_size must contain three integers >=2, ordered D,H,W")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(max_accumulator_gib) or max_accumulator_gib <= 0:
        raise ValueError("max_accumulator_gib must be positive and finite")
    # Validate overlap before allocating source or output buffers.
    patch_starts(1, 1, overlap)
    selected = resolve_device(device)
    validate_precision(selected, amp)
    if isinstance(volume, torch.Tensor):
        if volume.is_complex():
            raise ValueError("volume must contain real intensities")
        source = volume.detach().to(device="cpu", dtype=torch.float32)
    else:
        array = np.asarray(volume)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError("volume must contain real numeric intensities")
        source = torch.as_tensor(np.ascontiguousarray(array, dtype=np.float32))
    if source.ndim == 3:
        source = source.unsqueeze(0)
    if source.ndim != 4 or min(source.shape) < 1:
        raise ValueError("volume must be C,D,H,W (or D,H,W for one channel)")
    if not torch.isfinite(source).all().item():
        raise ValueError("volume contains non-finite values")
    expected_channels = getattr(getattr(model, "config", None), "in_channels", source.shape[0])
    if source.shape[0] != expected_channels:
        raise ValueError("Input channel count does not match model")
    original = tuple(source.shape[1:])
    padded = tuple(max(size, patch) for size, patch in zip(original, roi))
    classes = getattr(model, "num_classes", None)
    if classes is None and isinstance(metadata, dict):
        classes = len(metadata.get("labels", []))
    def check_budget(channel_count: int) -> None:
        required = (channel_count + 1) * math.prod(padded) * np.dtype(np.float32).itemsize
        if required > max_accumulator_gib * 1024 ** 3:
            raise MemoryError(f"CPU accumulation requires {required / 1024**3:.3f} GiB, exceeding max_accumulator_gib={max_accumulator_gib}; crop/resample explicitly or raise the cap")
    if classes:
        check_budget(classes)
    padding = tuple(v for pair in reversed(list(zip(original, padded))) for v in (0, pair[1] - pair[0]))
    if any(padding):
        source = F.pad(source, padding, mode="constant", value=0.0)
    starts = [patch_starts(size, patch, overlap) for size, patch in zip(padded, roi)]
    coordinates = product(*starts)
    importance = gaussian_importance(roi, sigma_scale)
    accumulator = None
    weights = None
    previous_training = model.training
    model.to(device=selected, dtype=torch.float32).eval()
    try:
        with torch.inference_mode():
            for batch in _batches(coordinates, batch_size):
                patches = torch.stack([source[:, z:z+roi[0], y:y+roi[1], x:x+roi[2]] for z, y, x in batch]).to(selected)
                autocast = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
                with autocast:
                    logits = model(patches)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 5:
                    raise ValueError("Model must return one N,C,D,H,W logits tensor")
                if logits.shape[0] != len(batch) or tuple(logits.shape[2:]) != roi or logits.shape[1] < 2:
                    raise ValueError("Model logits have an incompatible batch, class, or spatial shape")
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Model produced non-finite logits")
                if classes is not None and classes != logits.shape[1]:
                    raise ValueError("Model output classes differ from checkpoint metadata")
                classes = int(logits.shape[1])
                if accumulator is None:
                    check_budget(classes)
                    accumulator = np.zeros((classes,) + padded, dtype=np.float32)
                    weights = np.zeros(padded, dtype=np.float32)
                probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
                for output, (z, y, x) in zip(probabilities, batch):
                    region = (slice(z, z+roi[0]), slice(y, y+roi[1]), slice(x, x+roi[2]))
                    accumulator[(slice(None),) + region] += output * importance
                    weights[region] += importance
                del patches, logits, probabilities
    except RuntimeError as exc:
        raise RuntimeError(f"Segmentation execution failed on {selected}. No automatic CPU retry was made. For memory errors reduce ROI/batch size. Original error: {exc}") from exc
    finally:
        model.train(previous_training)
    if accumulator is None or weights is None or np.any(weights <= 0):
        raise RuntimeError("Sliding-window coverage was incomplete")
    accumulator /= weights[None]
    return accumulator[:, :original[0], :original[1], :original[2]]


def labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 4 or probabilities.shape[0] < 2:
        raise ValueError("Expected C,D,H,W segmentation probabilities")
    if not np.isfinite(probabilities).all():
        raise ValueError("Segmentation probabilities must be finite")
    return probabilities.argmax(axis=0).astype(np.uint8)
