import copy
import csv
import json

import nibabel as nib
import numpy as np
import pytest

from ctln import radiomics as bridge
from ctln.imaging import ImagingError
from ctln.nodes import extract_candidates


PARAMS = {"setting": {"binWidth": 25, "normalize": False}, "imageType": {"Original": {}}, "featureClass": {"firstorder": ["Mean"]}}


def inputs():
    ct = nib.Nifti1Image(np.arange(64, dtype=np.float32).reshape(4, 4, 4) - 800, np.diag([2., 3., 4., 1.]))
    ct.header.set_xyzt_units("mm")
    data = np.zeros(ct.shape, dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1
    mask = nib.Nifti1Image(data, ct.affine, header=ct.header.copy())
    mask.header.set_data_dtype(np.uint8)
    labels, records = extract_candidates(mask, ct, scan_id="s1")
    record = records[0]
    record["node_id"] = "human-roi-1"
    record["review"] = {"status": "approved", "reviewer": "reader-A", "reviewed_at": "2026-11-12", "mask_approved": True}
    return ct, labels, records


def test_gates_fail_before_optional_extractor_import(tmp_path, monkeypatch):
    ct, labels, records = inputs()
    monkeypatch.setattr(bridge, "_make_extractor", lambda _: pytest.fail("Gate must precede optional imports"))
    with pytest.raises(ValueError, match="HU provenance"):
        bridge.extract_radiomics(ct, labels, records, tmp_path, params=PARAMS)
    records[0]["review"]["status"] = "pending"
    with pytest.raises(ValueError, match="human mask review"):
        bridge.extract_radiomics(ct, labels, records, tmp_path, params=PARAMS, hu_confirmed=True)
    records[0]["review"]["status"] = "approved"
    probability = nib.Nifti1Image(labels.get_fdata(dtype=np.float32), ct.affine)
    probability.header.set_xyzt_units("mm")
    with pytest.raises(ImagingError, match="probability map"):
        bridge.extract_radiomics(ct, probability, records, tmp_path, params=PARAMS, hu_confirmed=True)


def test_frozen_parameters_have_stable_content_hash():
    frozen, first = bridge.freeze_parameters(PARAMS)
    _, reordered = bridge.freeze_parameters(dict(reversed(list(PARAMS.items()))))
    assert first == reordered
    changed = copy.deepcopy(PARAMS)
    changed["setting"]["binWidth"] = 10
    assert bridge.freeze_parameters(changed)[1] != first
    frozen["setting"]["binWidth"] = 5
    assert PARAMS["setting"]["binWidth"] == 25
    unsafe = copy.deepcopy(PARAMS)
    unsafe["setting"]["correctMask"] = True
    with pytest.raises(ValueError, match="correctMask"):
        bridge.freeze_parameters(unsafe)


def test_manifest_and_csv_keep_manual_identity_and_unverified_pathology(tmp_path, monkeypatch):
    ct, labels, records = inputs()

    class Extractor:
        settings = {"binWidth": 25, "normalize": False, "additionalInfo": True}
        enabledImagetypes = {"Original": {}}
        enabledFeatures = {"firstorder": ["Mean"]}
        def execute(self, source, mask, label):
            assert source.get_fdata().min() == -800
            assert label == 1
            return {"original_firstorder_Mean": np.float64(-768.5), "diagnostics_Versions_Test": "mock"}

    monkeypatch.setattr(bridge, "_make_extractor", lambda p: (Extractor(), "test-version"))
    monkeypatch.setattr(bridge, "_to_sitk", lambda img, **kw: img)
    outputs = bridge.extract_radiomics(ct, labels, records, tmp_path, params=PARAMS, hu_confirmed=True)
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["extraction_status"] == "complete"
    assert manifest["records"][0]["station"] is None
    assert manifest["records"][0]["reference_standard"]["result"] == "unverified"
    assert manifest["roi_results"][0]["training_label_verified"] is False
    assert manifest["ct_sha256"] != manifest["labels_sha256"]
    assert manifest["parameters"] == PARAMS
    with outputs["features_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["node_id"] == "human-roi-1"
    assert rows[0]["candidate_id"] == "s1:candidate-0001"
    assert float(rows[0]["original_firstorder_Mean"]) == -768.5
    assert "diagnostics_Versions_Test" not in rows[0]


def test_missing_label_review_and_extraction_failure_do_not_publish_measurements(tmp_path, monkeypatch):
    ct, labels, records = inputs()
    with pytest.raises(ValueError, match="every nonzero mask label"):
        bridge.extract_radiomics(ct, labels, [], tmp_path / "missing", params=PARAMS, hu_confirmed=True)

    class Broken:
        def execute(self, *args, **kwargs): raise RuntimeError("small ROI")

    monkeypatch.setattr(bridge, "_make_extractor", lambda _: (Broken(), "test"))
    monkeypatch.setattr(bridge, "_to_sitk", lambda img, **kw: img)
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="small ROI"):
        bridge.extract_radiomics(ct, labels, records, destination, params=PARAMS, hu_confirmed=True)
    assert not (destination / "features.csv").exists()
    assert not (destination / "manifest.json").exists()


def test_sitk_bridge_keeps_hu_and_world_geometry():
    sitk = pytest.importorskip("SimpleITK")
    ct, _, _ = inputs()
    volume = bridge._to_sitk(ct)
    assert volume.GetSpacing() == (2., 3., 4.)
    assert volume.GetDirection() == (-1., 0., 0., 0., -1., 0., 0., 0., 1.)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(volume), ct.get_fdata().transpose(2, 1, 0))
