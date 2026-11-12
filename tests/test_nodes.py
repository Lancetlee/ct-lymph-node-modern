import nibabel as nib
import numpy as np
import pytest

from ctln.imaging import ImagingError
from ctln.nodes import extract_candidates, has_verified_node_label, require_approved_record


def mask_image(data, affine=None, unit="mm"):
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units(unit)
    return image


def test_components_have_physical_measurements_and_pending_review():
    data = np.zeros((6, 6, 6), dtype=np.uint8)
    data[1:3, 1, 1] = 1
    data[5, 5, 5] = 1
    affine = np.array([[0, -3, 0, 12], [2, 0, 0, 10], [0, 0, 4, -20], [0, 0, 0, 1]], float)
    labels, records = extract_candidates(mask_image(data, affine), scan_id="p001-s1")
    assert len(records) == 2
    assert np.asarray(labels.dataobj).dtype == np.uint32
    assert records[0]["voxel_count"] == 2
    assert records[0]["volume_mm3"] == pytest.approx(48)
    np.testing.assert_allclose(records[0]["centroid_world_mm"], [9, 13, -16])
    assert records[0]["short_axis_mm"] is None
    assert records[0]["node_id"] is None
    assert records[0]["station"] is None
    assert records[0]["reference_standard"]["result"] == "unverified"
    assert records[0]["review"]["status"] == "pending"
    assert not has_verified_node_label(records[0])
    with pytest.raises(ValueError, match="human mask review"):
        require_approved_record(records[0])


def test_connectivity_and_volume_filter_are_explicit():
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    data[0, 0, 0] = data[1, 1, 1] = 1
    image = mask_image(data)
    assert len(extract_candidates(image, connectivity=26)[1]) == 1
    assert len(extract_candidates(image, connectivity=6)[1]) == 2
    labels, records = extract_candidates(image, connectivity=26, min_volume_mm3=3)
    assert records == []
    assert not np.asarray(labels.dataobj).any()


def test_unknown_units_and_probability_input_are_blocked():
    data = np.ones((2, 2, 2), dtype=np.uint8)
    image = mask_image(data, unit="unknown")
    with pytest.raises(ImagingError, match="Unknown spatial units"):
        extract_candidates(image)
    _, records = extract_candidates(image, spatial_unit_override="mm")
    assert records[0]["volume_mm3"] == pytest.approx(8)
    with pytest.raises(ImagingError, match="probability map"):
        extract_candidates(mask_image(data.astype(np.float32)))


def test_review_does_not_invent_pathology_or_station():
    _, records = extract_candidates(mask_image(np.ones((2, 2, 2), dtype=np.uint8)))
    record = records[0]
    record["review"] = {"status": "approved", "mask_approved": True, "reviewer": "reader-1", "reviewed_at": "2026-11-12"}
    with pytest.raises(ValueError, match="assign a node_id"):
        require_approved_record(record)
    record["node_id"] = "manual-roi-1"
    require_approved_record(record)
    assert not has_verified_node_label(record)
    record["roi_unit"] = "node"
    record["reference_standard"] = {"result": "positive", "label_unit": "station", "matching_evidence": "station specimen"}
    assert not has_verified_node_label(record)
    record["reference_standard"] = {"result": "positive", "label_unit": "node", "matching_evidence": "specimen A maps to this node", "method": "surgical_pathology"}
    assert has_verified_node_label(record)
    record["reference_standard"]["result"] = "negative"
    assert not has_verified_node_label(record)
    record["reference_standard"]["sampling_adequacy"] = "adequate"
    assert has_verified_node_label(record)
    record["reference_standard"]["method"] = "imaging_report"
    assert not has_verified_node_label(record)
