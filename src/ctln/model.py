"""A new PyTorch segmentation architecture; not a Keras/H5 conversion.

Tensors use N,C,D,H,W. The interpolation decoder intentionally avoids
ConvTranspose3d. Outputs are segmentation logits, never metastasis risks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "ctln_attention_unet3d_v1"
CHECKPOINT_SCHEMA_VERSION = 1
TASK = "lymph_node_segmentation"


class CheckpointError(ValueError):
    """Invalid, incompatible, or insufficiently documented checkpoint."""


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 2
    base_channels: int = 16
    levels: int = 4
    attention: bool = True

    def __post_init__(self) -> None:
        for name, lo, hi in (("in_channels", 1, 16), ("num_classes", 2, 32),
                             ("base_channels", 2, 128), ("levels", 2, 5)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if type(self.attention) is not bool:
            raise ValueError("attention must be bool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelConfig":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise CheckpointError("model_config must contain exactly the ModelConfig fields")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"Invalid model_config: {exc}") from exc


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels), nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels), nn.SiLU(inplace=True),
        )


class AttentionGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(1, channels // 2)
        self.gate = nn.Conv3d(channels, hidden, 1)
        self.skip = nn.Conv3d(channels, hidden, 1)
        self.score = nn.Conv3d(hidden, 1, 1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        score = torch.sigmoid(self.score(F.silu(self.gate(gate) + self.skip(skip))))
        return skip * score


class AttentionUNet3D(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.num_classes = self.config.num_classes
        widths = [self.config.base_channels * 2 ** i for i in range(self.config.levels)]
        self.encoders = nn.ModuleList()
        previous = self.config.in_channels
        for width in widths:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.pool = nn.MaxPool3d(2, ceil_mode=True)
        self.up_convs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(len(widths) - 1)):
            self.up_convs.append(nn.Conv3d(widths[i + 1], widths[i], 1))
            self.gates.append(AttentionGate(widths[i]) if self.config.attention else nn.Identity())
            self.decoders.append(ConvBlock(2 * widths[i], widths[i]))
        self.head = nn.Conv3d(widths[0], self.num_classes, 1)
        # A newly instantiated model is always untrained. Training code must
        # supply explicit provenance when saving a trained checkpoint.
        self.checkpoint_metadata: dict[str, Any] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.config.in_channels:
            raise ValueError("Model input must be N,C,D,H,W with configured channels")
        if min(x.shape[2:]) < 2:
            raise ValueError("Every spatial input dimension must be at least 2")
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, gate, decoder, skip in zip(self.up_convs, self.gates, self.decoders, reversed(skips)):
            x = up(F.interpolate(x, size=skip.shape[2:], mode="nearest"))
            attended = gate(x, skip) if self.config.attention else skip
            x = decoder(torch.cat((x, attended), dim=1))
        return self.head(x)


def build_model(config: ModelConfig | Mapping[str, Any] | None = None) -> AttentionUNet3D:
    if isinstance(config, Mapping):
        config = ModelConfig.from_dict(dict(config))
    return AttentionUNet3D(config)


_METADATA_KEYS = {"schema_version", "architecture", "model_config", "task", "trained",
                  "labels", "preprocessing", "training_provenance"}
_PREPROCESS_KEYS = {"target_spacing_dhw", "hu_window", "normalization"}


def _numbers(value: Any, length: int, name: str, positive: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CheckpointError(f"{name} must contain {length} numbers")
    if any(type(v) not in (int, float) or not math.isfinite(v) or (positive and v <= 0) for v in value):
        raise CheckpointError(f"{name} contains invalid values")
    return [float(v) for v in value]


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise CheckpointError(f"metadata must contain exactly: {sorted(_METADATA_KEYS)}")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("Unsupported checkpoint schema_version")
    if metadata["architecture"] != ARCHITECTURE or metadata["task"] != TASK:
        raise CheckpointError("Checkpoint architecture/task is incompatible; only LN segmentation is supported")
    config = ModelConfig.from_dict(metadata["model_config"])
    if type(metadata["trained"]) is not bool:
        raise CheckpointError("trained must be an explicit boolean")
    labels = metadata["labels"]
    if (not isinstance(labels, list) or len(labels) != config.num_classes
            or not all(isinstance(v, str) and v.strip() for v in labels)
            or len(set(labels)) != len(labels) or labels[0] != "background"
            or "lymph_node" not in labels):
        raise CheckpointError("labels must match channels and include background first and lymph_node")
    preprocessing = metadata["preprocessing"]
    if not isinstance(preprocessing, dict) or set(preprocessing) != _PREPROCESS_KEYS:
        raise CheckpointError(f"preprocessing must contain exactly: {sorted(_PREPROCESS_KEYS)}")
    _numbers(preprocessing["target_spacing_dhw"], 3, "target_spacing_dhw", positive=True)
    window = _numbers(preprocessing["hu_window"], 2, "hu_window")
    if window[0] >= window[1] or preprocessing["normalization"] != "clip_linear":
        raise CheckpointError("Expected an increasing hu_window and clip_linear normalization")
    provenance = metadata["training_provenance"]
    if not isinstance(provenance, dict):
        raise CheckpointError("training_provenance must be a JSON-compatible object")
    try:
        # Also reject NaN/Infinity and arbitrary objects in provenance.
        clean = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CheckpointError("Metadata must contain finite JSON-compatible values only") from exc
    if metadata["trained"]:
        for key in ("dataset", "training_run", "validation"):
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                raise CheckpointError(f"trained checkpoint requires training_provenance.{key}")
    return clean


def make_checkpoint_metadata(
    config: ModelConfig, *, trained: bool = False,
    labels: list[str] | None = None,
    target_spacing_dhw: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_window: tuple[float, float] = (-160.0, 240.0),
    training_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata. Spacing/window defaults are examples, not validated optima."""
    return validate_metadata({
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "architecture": ARCHITECTURE,
        "model_config": config.to_dict(), "task": TASK, "trained": trained,
        "labels": labels or ["background", "lymph_node"],
        "preprocessing": {"target_spacing_dhw": list(target_spacing_dhw),
                          "hu_window": list(hu_window), "normalization": "clip_linear"},
        "training_provenance": training_provenance or {},
    })


def save_checkpoint(path: str | Path, model: AttentionUNet3D, metadata: dict[str, Any]) -> None:
    metadata = validate_metadata(metadata)
    if not isinstance(model, AttentionUNet3D) or metadata["model_config"] != model.config.to_dict():
        raise CheckpointError("Model and metadata configurations do not match")
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        raise CheckpointError("New PyTorch checkpoints cannot use a Keras H5 filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise CheckpointError("Cannot save non-finite model parameters")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save({"metadata": metadata, "state_dict": state}, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    model.checkpoint_metadata = metadata


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu", *, allow_untrained: bool = False,
) -> tuple[AttentionUNet3D, dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        raise CheckpointError("Legacy Keras H5 weights are not converted. Faithful conversion requires the original architecture and numerical parity tests.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Could not load a restricted PyTorch checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"metadata", "state_dict"}:
        raise CheckpointError("Expected a checkpoint with exactly metadata and state_dict")
    metadata = validate_metadata(checkpoint["metadata"])
    if not metadata["trained"] and not allow_untrained:
        raise CheckpointError("Untrained checkpoint refused for inference; use only for explicit synthetic self-tests")
    model = build_model(metadata["model_config"])
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise CheckpointError("state_dict must contain tensors only")
    if any(not torch.isfinite(v).all().item() for v in state.values()):
        raise CheckpointError("Checkpoint contains non-finite model parameters")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise CheckpointError(f"Checkpoint parameter names/shapes do not match: {exc}") from exc
    from .device import resolve_device
    model.to(device=resolve_device(device), dtype=torch.float32).eval()
    model.checkpoint_metadata = metadata
    return model, metadata
