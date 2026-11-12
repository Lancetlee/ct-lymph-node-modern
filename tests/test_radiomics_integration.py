"""Real optional PyRadiomics tests; never substituted with a stub extractor.

Run with the radiomics extra installed. Missing PyRadiomics skips these tests;
an installed but broken PyRadiomics fails them so incompatibility stays visible.
"""
import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from ctln.nodes import extract_candidates
from ctln.radiomics import extract_radiomics


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("radiomics") is None,
    reason="Real optional PyRadiomics dependency is not installed; unit-test stubs are not end-to-end validation.",
)


def synthetic_reviewed_roi():
    grid = np.indices((12, 13, 14))
    data = (-900 + 3 * grid[0] + 7 * grid[1] + 11 * grid[2]).astype(np.float32)
    affine = np.array([[0, -1.5, 0, 17], [2, 0, 0, -24], [0, 0, 2.5, 8], [0, 0, 0, 1]], float)
    ct = nib.Nifti1Image(data, affine)
    ct.header.set_xyzt_units("mm")
    binary = np.zeros(ct.shape, dtype=np.uint8)
    binary[3:9, 3:10, 3:11] = 1
    mask = nib.Nifti1Image(binary, affine)
    mask.header.set_xyzt_units("mm")
    labels, records = extract_candidates(mask, ct, scan_id="synthetic-no-patient")
    records[0]["node_id"] = "synthetic-reviewed-roi"
    records[0]["review"] = {
        "status": "approved", "mask_approved": True,
        "reviewer": "synthetic-test-fixture", "reviewed_at": "synthetic-test",
    }
    return ct, labels, records


def test_real_extractor_preserves_unscaled_hu_mean(tmp_path):
    ct, labels, records = synthetic_reviewed_roi()
    params = {
        "imageType": {"Original": {}}, "featureClass": {"firstorder": ["Mean"]},
        "setting": {"binWidth": 25, "normalize": False, "additionalInfo": True},
    }
    result = extract_radiomics(ct, labels, records, tmp_path / "mean", params=params, hu_confirmed=True)
    with result["features_csv"].open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    expected = ct.get_fdata()[np.asarray(labels.dataobj) == 1].mean()
    assert float(row["original_firstorder_Mean"]) == pytest.approx(expected, abs=1e-6)
    assert float(row["original_firstorder_Mean"]) < -500  # A [0, 1] network array cannot pass as this CT.
    manifest = json.loads(result["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["extraction_status"] == "complete"
    assert manifest["roi_results"][0]["training_label_verified"] is False


def test_real_extractor_accepts_every_example_feature_name(tmp_path):
    ct, labels, records = synthetic_reviewed_roi()
    template = Path(__file__).resolve().parents[1] / "configs" / "radiomics.example.yaml"
    params = yaml.safe_load(template.read_text(encoding="utf-8"))
    result = extract_radiomics(ct, labels, records, tmp_path / "example", params=params, hu_confirmed=True)
    with result["features_csv"].open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    expected = {f"original_{family}_{name}" for family, names in params["featureClass"].items() for name in names}
    assert expected <= set(row)
    assert all(np.isfinite(float(row[name])) for name in expected)
    assert float(row["original_shape_MeshVolume"]) > 0
    manifest = json.loads(result["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["parameters"] == params
    assert manifest["pyradiomics_version"] != "unknown"
    assert manifest["records"][0]["station"] is None
