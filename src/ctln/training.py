"""Supervised segmentation training with patient-isolated validation.

This trains masks, not metastatic status. Test images are never loaded here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import time
import unicodedata

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from .device import resolve_device
from .imaging import categorical_data, load_ct, load_labelmap, require_same_geometry
from .inference import sliding_window_predict
from .model import ModelConfig, build_model, make_checkpoint_metadata, save_checkpoint
from .pipeline import file_sha256, prepare_network_image, protect_outputs, write_json


def read_cases(manifest):
    manifest = Path(manifest)
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'patient_id', 'scan_id', 'image', 'mask', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires columns {sorted(required)}')
        rows = list(reader)
    groups, scans = {}, set()
    for row in rows:
        if any(not str(row.get(key, '')).strip() for key in required):
            raise ValueError('Required manifest fields cannot be empty.')
        for key in required:
            row[key] = unicodedata.normalize('NFC', str(row[key]).strip())
        split = row['split']
        if split not in {'train', 'val', 'test'}:
            raise ValueError('split must be train, val, or test.')
        patient = row['patient_id']
        if patient in groups and groups[patient] != split:
            raise ValueError(f'Patient leakage across splits: {patient}')
        groups[patient] = split
        if row['scan_id'] in scans:
            raise ValueError(f'Duplicate scan_id: {row["scan_id"]}')
        scans.add(row['scan_id'])
        for field in ('image', 'mask'):
            path = Path(row[field])
            row[field] = str((manifest.parent / path).resolve()) if not path.is_absolute() else str(path.resolve())
    for split in ('train', 'val'):
        if not any(row['split'] == split for row in rows):
            raise ValueError('Both patient-isolated train and val cases are required.')
    # This check includes held-out rows but never opens their image data.
    image_splits = {}
    for row in rows:
        path_key = str(Path(row['image']).resolve())
        if path_key in image_splits and image_splits[path_key] != row['split']:
            raise ValueError('Identical CT path occurs in different splits, including held-out test rows.')
        image_splits[path_key] = row['split']
    # Guard duplicates across different pseudonyms in development data using file hashes.
    hashes = {}
    for row in rows:
        if row['split'] == 'test':
            continue
        fingerprint = file_sha256(row['image'])
        if fingerprint in hashes and hashes[fingerprint] != row['split']:
            raise ValueError('Identical CT file occurs in different development splits.')
        hashes[fingerprint] = row['split']
    return rows


def _load_case(row, preprocessing, settings):
    ct = load_ct(row['image'], assume_nifti_hu=settings.get('assume_nifti_hu', False),
                 assume_nifti_mm=settings.get('assume_nifti_mm', False))
    mask = load_labelmap(row['mask'], assume_nifti_mm=settings.get('assume_nifti_mm', False))
    require_same_geometry(ct, mask)
    binary = categorical_data(mask, binary=True)
    grid, data = prepare_network_image(ct, preprocessing, max_voxels=settings.get('max_case_voxels', 134217728))
    original_mask = nib.Nifti1Image(binary.astype(np.uint8), ct.affine)
    transformed = resample_from_to(original_mask, grid, order=0, cval=0)
    labels = np.ascontiguousarray(np.asarray(transformed.dataobj).transpose(2, 1, 0), dtype=np.int64)
    return data, labels


def _patch(data, labels, roi, generator, foreground_probability):
    padding = [(0, max(0, size - actual)) for size, actual in zip(roi, labels.shape)]
    data = np.pad(data, [(0, 0)] + padding, constant_values=0)
    labels = np.pad(labels, padding, constant_values=0)
    foreground = np.argwhere(labels > 0)
    if len(foreground) and generator.random() < foreground_probability:
        center = foreground[generator.integers(len(foreground))]
        starts = [int(np.clip(c - r // 2, 0, a - r)) for c, r, a in zip(center, roi, labels.shape)]
    else:
        starts = [int(generator.integers(a - r + 1)) for r, a in zip(roi, labels.shape)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, roi))
    return np.ascontiguousarray(data[(slice(None),) + slices]), np.ascontiguousarray(labels[slices])


def _loss(logits, target):
    cross_entropy = F.cross_entropy(logits, target)
    p = torch.softmax(logits, dim=1)[:, 1]
    truth = (target == 1).float()
    dice = (2 * (p * truth).sum() + 1e-5) / (p.sum() + truth.sum() + 1e-5)
    return cross_entropy + 1 - dice


def train_segmentation(manifest, config_path, output, *, device='auto', overwrite=False):
    output, config_path, manifest = Path(output), Path(config_path), Path(manifest)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not all(key in config for key in ('model', 'preprocessing', 'training')):
        raise ValueError('Training config requires model, preprocessing, training mappings.')
    model_config = ModelConfig.from_dict(config['model'])
    if model_config.in_channels != 1 or model_config.num_classes != 2:
        raise ValueError('Training command supports one original CT channel and binary LN segmentation.')
    settings, preprocessing = config['training'], config['preprocessing']
    if not isinstance(settings, dict) or not isinstance(preprocessing, dict):
        raise ValueError('training and preprocessing must be mappings.')
    epochs, patches = int(settings.get('epochs', 0)), int(settings.get('patches_per_case', 0))
    roi = tuple(settings.get('roi_size', []))
    if epochs < 1 or patches < 1 or len(roi) != 3 or any(type(x) is not int or x < 4 for x in roi):
        raise ValueError('Positive epochs/patches_per_case and a 3D integer roi_size >=4 are required.')
    lr = float(settings.get('learning_rate', 0))
    foreground_probability = float(settings.get('foreground_sampling_probability', .5))
    if not np.isfinite(lr) or lr <= 0 or not 0 <= foreground_probability <= 1:
        raise ValueError('Invalid learning rate or foreground sampling probability.')
    if not isinstance(settings.get('dataset_description'), str) or not settings['dataset_description'].strip():
        raise ValueError('Provide training.dataset_description; do not present test fixtures as a clinical cohort.')
    rows = read_cases(manifest)
    protect_outputs([output, output.with_suffix('.training.json')],
                    [manifest, config_path] + [row[key] for row in rows for key in ('image', 'mask')], overwrite=overwrite)
    seed = int(settings.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    selected = resolve_device(device)
    if selected.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if selected.type == 'mps':
        torch.mps.manual_seed(seed)
    model = build_model(model_config).to(selected)
    # Validate the complete checkpoint preprocessing schema before expensive training.
    make_checkpoint_metadata(model_config, target_spacing_dhw=tuple(preprocessing['target_spacing_dhw']),
                             hu_window=tuple(preprocessing['hu_window']))
    if preprocessing.get('normalization') != 'clip_linear':
        raise ValueError('Only explicit clip_linear preprocessing is implemented.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history, best_dice, steps = [], -1., 0
    started = time.perf_counter()
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    for epoch in range(1, epochs + 1):
        model.train()
        train_rows = [row for row in rows if row['split'] == 'train']
        generator.shuffle(train_rows)
        losses = []
        for row in train_rows:
            data, target = _load_case(row, preprocessing, settings)
            for _ in range(patches):
                x, y = _patch(data, target, roi, generator, foreground_probability)
                x = torch.from_numpy(x[None]).to(selected, dtype=torch.float32)
                y = torch.from_numpy(y[None]).to(selected, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss; checkpoint was not promoted.')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                steps += 1
        model.eval()
        model.checkpoint_metadata = make_checkpoint_metadata(
            model_config, trained=True, target_spacing_dhw=tuple(preprocessing['target_spacing_dhw']),
            hu_window=tuple(preprocessing['hu_window']),
            training_provenance={'dataset': settings['dataset_description'], 'training_run': 'in-progress', 'validation': 'in-progress'})
        validations = []
        for row in rows:
            if row['split'] != 'val':
                continue
            data, truth = _load_case(row, preprocessing, settings)
            p = sliding_window_predict(model, data, roi_size=tuple(settings.get('validation_roi_size', roi)),
                                       device=selected, max_accumulator_gib=settings.get('max_accumulator_gib', 4.0))
            predicted = p[1] >= .5
            denominator = int(predicted.sum() + truth.sum())
            dice = 1. if denominator == 0 else float(2 * np.logical_and(predicted, truth).sum() / denominator)
            validations.append({'scan_id': row['scan_id'], 'patient_id': row['patient_id'], 'dice': dice,
                                'truth_foreground_voxels': int(truth.sum()), 'both_empty': denominator == 0})
        mean_dice = float(np.mean([row['dice'] for row in validations]))
        item = {'epoch': epoch, 'mean_training_loss': float(np.mean(losses)),
                'validation_mean_case_dice': mean_dice, 'validation_cases': validations, 'optimizer_steps': steps}
        history.append(item)
        if mean_dice > best_dice:
            best_dice = mean_dice
            provenance = {'dataset': settings['dataset_description'], 'training_run': f'epoch={epoch};optimizer_steps={steps};config_sha256={config_sha}',
                          'validation': f'patient-isolated development val; mean_case_dice={mean_dice:.6g}; threshold=0.5; not clinical validation',
                          'manifest_sha256': file_sha256(manifest), 'config_sha256': config_sha,
                          'selected_epoch': epoch, 'optimizer_steps': steps, 'seed': seed,
                          'train_patient_count': len({r['patient_id'] for r in rows if r['split'] == 'train'}),
                          'val_patient_count': len({r['patient_id'] for r in rows if r['split'] == 'val'}),
                          'held_out_test_rows_not_loaded': sum(r['split'] == 'test' for r in rows)}
            metadata = make_checkpoint_metadata(model_config, trained=True, target_spacing_dhw=tuple(preprocessing['target_spacing_dhw']),
                                                hu_window=tuple(preprocessing['hu_window']), training_provenance=provenance)
            save_checkpoint(output, model, metadata)
        write_json(output.with_suffix('.training.json'), {'task': 'supervised_segmentation_not_metastasis_classification',
                   'device': str(selected), 'history': history, 'best_validation_dice': best_dice,
                   'elapsed_seconds': time.perf_counter() - started, 'test_images_loaded': False,
                   'reproducibility_note': 'Seeded execution is not a guarantee of cross-device bitwise determinism.'})
    return {'checkpoint': str(output), 'training_report': str(output.with_suffix('.training.json')),
            'optimizer_steps': steps, 'best_validation_dice': best_dice, 'device': str(selected)}
