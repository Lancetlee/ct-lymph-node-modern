"""Original-HU CT -> model grid -> original-grid review candidates."""
from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
import tempfile
import time

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
import torch

from . import __version__
from .device import resolve_device
from .imaging import load_ct, new_image_like, save_labels, save_mask, save_probability
from .inference import sliding_window_predict
from .model import load_checkpoint
from .nodes import extract_candidates


def file_sha256(path: str | Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.ctln-', suffix='.json', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def protect_outputs(paths, inputs=(), *, overwrite=False):
    source_paths = {Path(p).resolve() for p in inputs}
    for path in map(Path, paths):
        if path.resolve() in source_paths:
            raise ValueError(f'Output would overwrite a source file: {path}')
        if path.exists() and not overwrite:
            raise FileExistsError(f'Output exists; select a new output or explicitly use --overwrite: {path}')


def prepare_network_image(ct, preprocessing, *, max_voxels=134217728):
    """Return a separate physical-mm grid and normalized C,D,H,W array."""
    spacing_dhw = preprocessing['target_spacing_dhw']
    spacing_xyz = tuple(reversed(spacing_dhw))
    if type(max_voxels) is not int or max_voxels < 1:
        raise ValueError('max_case_voxels must be a positive integer.')
    if len(spacing_xyz) != 3 or not np.isfinite(spacing_xyz).all() or min(spacing_xyz) <= 0:
        raise ValueError('Target spacing must contain three finite positive physical-mm values.')
    lo, hi = preprocessing['hu_window']
    if preprocessing['normalization'] != 'clip_linear' or not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError('Expected an increasing HU window and clip_linear normalization.')
    # resample_to_output constructs an axis-aligned RAS grid. Rotations/shears
    # can enlarge its bounding box, so column lengths alone underestimate it.
    corners = np.asarray(list(product(*[(0, n - 1) for n in ct.shape])), dtype=float)
    world = nib.affines.apply_affine(ct.affine, corners)
    bounds = np.ptp(world, axis=0)
    estimated = np.prod(np.ceil(bounds / np.asarray(spacing_xyz)) + 2)
    if estimated > max_voxels:
        raise ValueError(f'Resampled grid estimate {int(estimated)} exceeds max_case_voxels={max_voxels}; review spacing/FOV explicitly.')
    grid = resample_to_output(ct, voxel_sizes=spacing_xyz, order=1, cval=-1024)
    if np.prod(grid.shape) > max_voxels:
        raise ValueError('Resampled grid exceeds the configured volume budget.')
    data = grid.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('CT contains non-finite values.')
    normalized = (np.clip(data, lo, hi) - lo) / (hi - lo)
    grid.header.set_xyzt_units('mm')
    return grid, np.ascontiguousarray(normalized.transpose(2, 1, 0)[None], dtype=np.float32)


def infer_case(*, image, checkpoint, output, device='auto', series_uid=None,
               assume_hu=False, assume_mm=False, roi_size=(64, 64, 64),
               overlap=0.5, batch_size=1, amp=False, threshold=0.5,
               scan_id='scan', max_accumulator_gib=4.0,
               max_case_voxels=134217728, overwrite=False):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Segmentation threshold must be in (0,1); it is not a metastasis-risk threshold.')
    output = Path(output)
    paths = {name: output / filename for name, filename in {
        'probability': 'segmentation_probability.nii.gz',
        'mask': 'candidate_mask.nii.gz',
        'instances': 'candidate_instances.nii.gz',
        'review': 'candidate_review.json',
        'manifest': 'run_manifest.json',
    }.items()}
    protect_outputs(paths.values(), [image, checkpoint], overwrite=overwrite)
    selected = resolve_device(device)
    started = time.perf_counter()
    # Loading always requires a trained checkpoint with explicit provenance.
    model, metadata = load_checkpoint(checkpoint, device=selected)
    if model.config.in_channels != 1 or model.config.num_classes != 2:
        raise ValueError('This CT pipeline requires one CT channel and two segmentation classes; APG/multichannel input is not implemented.')
    ct = load_ct(image, series_uid=series_uid, assume_nifti_hu=assume_hu, assume_nifti_mm=assume_mm)
    grid, network_data = prepare_network_image(ct, metadata['preprocessing'], max_voxels=max_case_voxels)
    probabilities = sliding_window_predict(
        model, network_data, roi_size=tuple(roi_size), overlap=overlap,
        batch_size=batch_size, device=selected, amp=amp,
        max_accumulator_gib=max_accumulator_gib,
    )
    node_channel = metadata['labels'].index('lymph_node')
    model_probability = nib.Nifti1Image(probabilities[node_channel].transpose(2, 1, 0), grid.affine)
    restored = resample_from_to(model_probability, ct, order=1, cval=0)
    probability = np.clip(restored.get_fdata(dtype=np.float32), 0, 1)
    binary = (probability >= threshold).astype(np.uint8)
    binary_image = new_image_like(ct, binary, dtype=np.uint8)
    instances, records = extract_candidates(binary_image, ct=ct, scan_id=scan_id)
    output.mkdir(parents=True, exist_ok=True)
    save_probability(probability, ct, paths['probability'])
    save_mask(binary, ct, paths['mask'])
    save_labels(np.asarray(instances.dataobj), ct, paths['instances'])
    write_json(paths['review'], records)
    manifest = {
        'schema_version': 1, 'software': {'ctln': __version__, 'torch': str(torch.__version__), 'numpy': np.__version__, 'nibabel': nib.__version__},
        'task': 'candidate_lymph_node_segmentation_not_metastasis_prediction',
        'device_requested': str(device), 'device_used': str(selected), 'amp': amp,
        'checkpoint_sha256': file_sha256(checkpoint), 'checkpoint_metadata': metadata,
        'source_image': str(Path(image).resolve()),
        'source_file_sha256': file_sha256(image) if Path(image).is_file() else None,
        'series_uid_selected': ct.extra.get('dicom_series_uid', series_uid),
        'nifti_hu_asserted': assume_hu, 'nifti_mm_asserted': assume_mm,
        'original_shape': list(ct.shape), 'original_affine': ct.affine.tolist(),
        'network_shape_dhw': list(network_data.shape[1:]), 'network_affine': grid.affine.tolist(),
        'roi_size_dhw': list(roi_size), 'overlap': overlap, 'batch_size': batch_size,
        'segmentation_threshold': threshold, 'candidate_count': len(records),
        'elapsed_seconds': time.perf_counter() - started,
        'outputs': {k: str(v.resolve()) for k, v in paths.items()},
        'review_status': 'pending', 'station_labels_generated': False, 'metastasis_labels_generated': False,
        'source_data_modified': False,
    }
    write_json(paths['manifest'], manifest)
    return manifest
