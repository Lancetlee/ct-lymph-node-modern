"""Tiny synthetic end-to-end execution, not evidence of clinical accuracy."""
import csv
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from ctln.cli import main
from ctln.model import ModelConfig, build_model, load_checkpoint
from ctln.pipeline import file_sha256, infer_case, prepare_network_image, protect_outputs, write_json
from ctln.training import read_cases, train_segmentation


def make_cases(tmp_path, unit='mm'):
    affine = np.array([[-1.2, 0, 0, 40], [0, 1.3, 0, -25], [0, 0, 2, 9], [0, 0, 0, 1.]])
    for name, seed in [('train', 1), ('val', 2)]:
        values = np.random.default_rng(seed).integers(-1000, 700, size=(10, 12, 8), dtype=np.int16)
        mask = np.zeros(values.shape, np.uint8)
        mask[3:7, 4:8, 2:6] = 1
        for kind, data in [('ct', values), ('mask', mask)]:
            image = nib.Nifti1Image(data, affine)
            image.header.set_xyzt_units(unit)
            nib.save(image, tmp_path / f'{name}_{kind}.nii.gz')
    manifest = tmp_path / 'cases.csv'
    rows = [dict(patient_id=name, scan_id=name, image=f'{name}_ct.nii.gz', mask=f'{name}_mask.nii.gz', split=name)
            for name in ('train', 'val')]
    # An unreadable test case is deliberate: training must never touch it.
    rows.append(dict(patient_id='heldout', scan_id='test', image='never_load_ct.nii.gz', mask='never_load_mask.nii.gz', split='test'))
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = {'model': ModelConfig(base_channels=2, levels=2).to_dict(),
              'preprocessing': {'target_spacing_dhw': [2., 1.3, 1.2], 'hu_window': [-160, 240], 'normalization': 'clip_linear'},
              'training': {'epochs': 1, 'patches_per_case': 1, 'roi_size': [8, 8, 8], 'validation_roi_size': [8, 12, 12],
                           'learning_rate': .001, 'foreground_sampling_probability': 1., 'seed': 42,
                           'assume_nifti_hu': True, 'assume_nifti_mm': unit == 'unknown',
                           'dataset_description': 'Synthetic software test only; no clinical cohort'}}
    config_path = tmp_path / 'train.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return manifest, config_path


@pytest.mark.parametrize('unit', ['mm', 'unknown'])
def test_real_optimizer_checkpoint_and_original_grid_inference(tmp_path, unit):
    torch.set_num_threads(1)
    manifest, config = make_cases(tmp_path, unit=unit)
    ct_path = tmp_path / 'val_ct.nii.gz'
    initial_hash = file_sha256(ct_path)
    torch.manual_seed(42)
    initial = build_model(ModelConfig(base_channels=2, levels=2)).head.weight.detach().clone()
    checkpoint = tmp_path / 'trained.pt'
    result = train_segmentation(manifest, config, checkpoint, device='cpu')
    assert result['optimizer_steps'] == 1
    trained, metadata = load_checkpoint(checkpoint)
    assert metadata['trained'] is True
    assert metadata['training_provenance']['held_out_test_rows_not_loaded'] == 1
    assert not torch.equal(initial, trained.head.weight)
    training_report = json.loads(checkpoint.with_suffix('.training.json').read_text())
    assert training_report['test_images_loaded'] is False
    assert np.isfinite(training_report['history'][0]['mean_training_loss'])
    report = infer_case(image=ct_path, checkpoint=checkpoint, output=tmp_path / 'result', device='cpu',
                        assume_hu=True, assume_mm=unit == 'unknown', roi_size=(8, 12, 12), scan_id='synthetic-val')
    original = nib.load(ct_path)
    for key in ('probability', 'mask', 'instances'):
        output = nib.load(report['outputs'][key])
        assert output.shape == original.shape
        np.testing.assert_allclose(output.affine, original.affine)
        assert output.header.get_xyzt_units()[0] == 'mm'
    p = nib.load(report['outputs']['probability']).get_fdata()
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    records = json.loads((tmp_path / 'result' / 'candidate_review.json').read_text())
    assert all(record['review']['status'] == 'pending' and record['station'] is None for record in records)
    assert report['metastasis_labels_generated'] is False
    assert file_sha256(ct_path) == initial_hash
    with pytest.raises(FileExistsError):
        infer_case(image=ct_path, checkpoint=checkpoint, output=tmp_path / 'result', device='cpu', assume_hu=True)


def test_patient_split_and_identical_file_guard(tmp_path):
    manifest, _ = make_cases(tmp_path)
    text = manifest.read_text()
    manifest.write_text(text.replace('val,val,', 'train,val,'))
    with pytest.raises(ValueError, match='Patient leakage'):
        read_cases(manifest)
    manifest.write_text(text.replace('val_ct.nii.gz', 'train_ct.nii.gz'))
    with pytest.raises(ValueError, match='Identical CT'):
        read_cases(manifest)
    manifest.write_text(text.replace('val,val,', 'train ,val,'))
    with pytest.raises(ValueError, match='Patient leakage'):
        read_cases(manifest)
    manifest.write_text(text.replace('never_load_ct.nii.gz', 'train_ct.nii.gz'))
    with pytest.raises(ValueError, match='Identical CT path'):
        read_cases(manifest)


def test_training_rejects_float_probability_as_mask(tmp_path):
    manifest, config = make_cases(tmp_path)
    path = tmp_path / 'train_mask.nii.gz'
    original = nib.load(path)
    data = original.get_fdata().astype(np.float32) * .9
    probability = nib.Nifti1Image(data, original.affine, original.header)
    probability.set_data_dtype(np.float32)
    nib.save(probability, path)
    with pytest.raises(ValueError, match='integer|categorical|probability'):
        train_segmentation(manifest, config, tmp_path / 'invalid.pt', device='cpu')
    assert not (tmp_path / 'invalid.pt').exists()


def test_cli_reports_and_protects_inputs(tmp_path, capsys):
    torch.set_num_threads(1)
    assert main(['doctor', '--device', 'cpu', '--output', str(tmp_path / 'doctor.json')]) == 0
    assert json.loads((tmp_path / 'doctor.json').read_text())['smoke_test']['passed'] is True
    args = ['benchmark', '--device', 'cpu', '--roi-size', '8', '8', '8', '--base-channels', '2', '--levels', '2',
            '--warmup', '0', '--iterations', '1', '--output', str(tmp_path / 'benchmark.json')]
    assert main(args) == 0
    benchmark = json.loads((tmp_path / 'benchmark.json').read_text())
    assert benchmark['trained'] is False and benchmark['clinical_validation'] is False
    assert benchmark['model_config']['base_channels'] == 2 and benchmark['median_seconds'] > 0
    assert main(['validate-device', '--device', 'cpu', '--output', str(tmp_path / 'parity.json')]) == 0
    parity = json.loads((tmp_path / 'parity.json').read_text())
    assert parity['passed'] is True and parity['accelerator_validated'] is False
    assert main(args) == 2  # Explicit overwrite required.
    source = tmp_path / 'source.json'
    source.write_text('{}')
    with pytest.raises(ValueError, match='overwrite a source'):
        protect_outputs([source], [source], overwrite=True)
    if not torch.backends.mps.is_available():
        assert main(['doctor', '--device', 'mps', '--output', str(tmp_path / 'mps.json')]) == 1
        assert json.loads((tmp_path / 'mps.json').read_text())['smoke_test']['passed'] is False


def test_temporary_write_preserves_source_and_rotated_budget_is_prechecked(tmp_path, monkeypatch):
    source = tmp_path / 'report.json.tmp'
    source.write_text('original source')
    write_json(tmp_path / 'report.json', {'verified': True})
    assert source.read_text() == 'original source'
    angle = np.pi / 4
    affine = np.eye(4)
    affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ct = nib.Nifti1Image(np.zeros((100, 100, 100), np.int16), affine)
    def allocation_forbidden(*args, **kwargs):
        raise AssertionError('Must reject the RAS bounding-box budget before allocating a resampled image')
    monkeypatch.setattr('ctln.pipeline.resample_to_output', allocation_forbidden)
    with pytest.raises(ValueError, match='estimate'):
        prepare_network_image(ct, {'target_spacing_dhw': [1, 1, 1], 'hu_window': [-160, 240],
                                   'normalization': 'clip_linear'}, max_voxels=1500000)
