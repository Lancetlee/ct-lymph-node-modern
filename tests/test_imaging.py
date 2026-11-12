import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from ctln.imaging import (
    ImagingError, geometry_matches, load_ct, load_labelmap, load_nifti, new_image_like,
    resample_mask_to_ct, save_labels, save_mask, save_probability,
    _validate_classic_ct_geometry,
)


def image(data, affine=None, unit="mm"):
    result = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    result.header.set_xyzt_units(unit)
    return result


def test_nifti_requires_hu_and_known_physical_units(tmp_path):
    values = np.array([-1100, -30, 0, 1200], dtype=np.int16).reshape(2, 2, 1)
    path = tmp_path / "original.nii.gz"
    nib.save(image(values, unit="unknown"), path)
    with pytest.raises(ImagingError, match="HU provenance"):
        load_ct(path)
    with pytest.raises(ImagingError, match="Unknown spatial units"):
        load_ct(path, assume_nifti_hu=True)
    original = load_ct(path, assume_nifti_hu=True, assume_nifti_mm=True)
    np.testing.assert_array_equal(original.get_fdata(), values)
    assert original.header.get_xyzt_units()[0] == "mm"
    # IO must never clip or normalize original CT for the network.
    assert original.get_fdata().min() == -1100
    assert original.get_fdata().max() == 1200


def test_ct_converts_full_meter_affine_to_mm(tmp_path):
    affine = np.diag([0.002, 0.003, 0.004, 1.0])
    affine[:3, 3] = [0.01, 0.02, -0.03]
    path = tmp_path / "metres.nii"
    nib.save(image(np.zeros((2, 2, 2), dtype=np.int16), affine, unit="meter"), path)
    loaded = load_ct(path, assume_nifti_hu=True)
    np.testing.assert_allclose(loaded.affine[:3], affine[:3] * 1000, atol=1e-5)


def test_labelmap_unknown_units_need_explicit_assertion(tmp_path):
    values = np.array([0, 1, 300, 4095], dtype=np.uint16).reshape(2, 2, 1)
    path = tmp_path / "unknown-unit-labels.nii.gz"
    nib.save(image(values, unit="unknown"), path)
    with pytest.raises(ImagingError, match="Unknown spatial units"):
        load_labelmap(path)
    labels = load_labelmap(path, assume_nifti_mm=True)
    np.testing.assert_array_equal(np.asarray(labels.dataobj), values)
    assert labels.get_data_dtype() == np.dtype(np.uint16)
    assert labels.header.get_xyzt_units()[0] == "mm"
    assert labels.header.get_slope_inter() == (1., 0.)


@pytest.mark.parametrize("unit,factor", [("meter", 1000.), ("micron", .001)])
def test_labelmap_unit_conversion_preserves_integer_ids_and_physical_grid(tmp_path, unit, factor):
    values = np.array([0, 2, 400, 65535], dtype=np.uint16).reshape(2, 2, 1)
    affine_mm = np.array([[0, -3, 0, 12], [2, 0, 0, -8], [0, 0, 4, 30], [0, 0, 0, 1]], float)
    source_affine = affine_mm.copy()
    source_affine[:3, :] /= factor
    path = tmp_path / f"labels-{unit}.nii.gz"
    nib.save(image(values, source_affine, unit=unit), path)
    labels = load_labelmap(path)
    reference = image(np.zeros(values.shape, dtype=np.int16), affine_mm)
    assert geometry_matches(labels, reference)
    np.testing.assert_array_equal(np.asarray(labels.dataobj), values)
    assert labels.get_data_dtype() == np.dtype(np.uint16)
    assert labels.header.get_slope_inter() == (1., 0.)


def test_labelmap_loader_does_not_threshold_float_probability(tmp_path):
    path = tmp_path / "probability.nii.gz"
    nib.save(image(np.full((2, 2, 2), .5, dtype=np.float32)), path)
    with pytest.raises(ImagingError, match="probability map"):
        load_labelmap(path, assume_nifti_mm=True)


def test_geometry_checks_origin_not_just_shape_or_spacing():
    a = image(np.zeros((4, 4, 4), dtype=np.uint8))
    affine = np.eye(4)
    affine[0, 3] = 20
    b = image(np.zeros(a.shape, dtype=np.uint8), affine)
    assert not geometry_matches(a, b)
    with pytest.raises(ImagingError, match="allow_resample=True"):
        resample_mask_to_ct(b, a)


def test_nearest_mask_resampling_is_explicit_and_categorical():
    ct = image(np.zeros((5, 5, 5), dtype=np.int16))
    data = np.zeros((3, 3, 3), dtype=np.uint16)
    data[1, 1, 1] = 7
    mask = image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    aligned = resample_mask_to_ct(mask, ct, allow_resample=True)
    assert geometry_matches(aligned, ct)
    assert set(np.unique(np.asarray(aligned.dataobj))) == {0, 7}
    assert np.asarray(aligned.dataobj).dtype == np.uint16
    probability = image(data.astype(np.float32) / 7, mask.affine)
    with pytest.raises(ImagingError, match="probability map"):
        resample_mask_to_ct(probability, ct, allow_resample=True)


def test_outputs_preserve_affine_and_intended_dtypes(tmp_path):
    affine = np.array([[0, -2, 0, 12], [3, 0, 0, -4], [0, 0, 4, 80], [0, 0, 0, 1]], dtype=float)
    reference = image(np.zeros((3, 4, 5), dtype=np.int16), affine)
    prob = np.full(reference.shape, 0.25, dtype=np.float64)
    mask = prob > 0.1
    labels = mask.astype(np.uint32) * 1000
    paths = [
        save_probability(prob, reference, tmp_path / "prob.nii.gz"),
        save_mask(mask, reference, tmp_path / "mask.nii.gz"),
        save_labels(labels, reference, tmp_path / "components.nii.gz"),
    ]
    for path, dtype in zip(paths, (np.float32, np.uint8, np.uint32)):
        result = load_nifti(path)
        assert result.get_data_dtype() == np.dtype(dtype)
        np.testing.assert_array_equal(result.affine, affine)
        assert result.header.get_xyzt_units()[0] == "mm"
    with pytest.raises(ImagingError, match="probability map"):
        save_mask(prob, reference, tmp_path / "invalid.nii.gz")
    with pytest.raises(ImagingError, match="lose label"):
        save_labels(labels, reference, tmp_path / "overflow.nii.gz", dtype=np.uint8)
    with pytest.raises(ImagingError, match=r"\[0, 1\]"):
        save_probability(prob * 8, reference, tmp_path / "invalid_probability.nii.gz")


def test_four_dimensional_nifti_is_not_silently_truncated(tmp_path):
    path = tmp_path / "phases.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 2), dtype=np.int16), np.eye(4)), path)
    with pytest.raises(ImagingError, match="3D"):
        load_nifti(path)


def test_dicom_requires_selection_and_converts_lps_to_ras(tmp_path, monkeypatch):
    calls = []

    class Volume:
        def GetDimension(self): return 3
        def GetNumberOfComponentsPerPixel(self): return 1
        def GetDirection(self): return tuple(np.eye(3).ravel())
        def GetSpacing(self): return (2, 3, 4)
        def GetOrigin(self): return (10, 20, 30)
        def GetSize(self): return (4, 3, 2)

    class HeaderReader:
        def SetFileName(self, filename): self.filename = filename
        def ReadImageInformation(self): pass
        def HasMetaDataKey(self, key): return key in self.metadata()
        def GetMetaData(self, key): return self.metadata()[key]
        def metadata(self):
            return {
                "0008|0016": "1.2.840.10008.5.1.4.1.1.2", "0008|0060": "CT",
                "0020|000e": self.filename.split("-slice-")[0], "0020|0052": "frame",
                "0020|0032": "10\\20\\30" if self.filename.endswith("1") else "10\\20\\34",
                "0020|0037": "1\\0\\0\\0\\1\\0", "0028|0030": "3\\2",
                "0028|0010": "3", "0028|0011": "4",
            }

    class Reader:
        @staticmethod
        def GetGDCMSeriesIDs(path): return ["series-a", "series-b"]
        @staticmethod
        def GetGDCMSeriesFileNames(path, uid):
            calls.append(uid)
            return [uid + "-slice-1", uid + "-slice-2"]
        def SetFileNames(self, filenames): self.files = filenames
        def MetaDataDictionaryArrayUpdateOn(self): pass
        def Execute(self): return Volume()
        def HasMetaDataKey(self, i, key): return True
        def GetMetaData(self, i, key): return "CT"

    array = np.arange(24, dtype=np.int16).reshape(2, 3, 4) - 1000
    fake = SimpleNamespace(ImageSeriesReader=Reader, ImageFileReader=HeaderReader, GetArrayFromImage=lambda _: array)
    monkeypatch.setitem(sys.modules, "SimpleITK", fake)
    with pytest.raises(ImagingError, match="Multiple DICOM series"):
        load_ct(tmp_path)
    assert calls == []
    result = load_ct(tmp_path, series_uid="series-b")
    assert calls == ["series-b"]
    np.testing.assert_array_equal(result.get_fdata(), array.transpose(2, 1, 0))
    np.testing.assert_array_equal(result.affine, [[-2, 0, 0, -10], [0, -3, 0, -20], [0, 0, 4, 30], [0, 0, 0, 1]])


def classic_headers():
    return [{
        "0008|0016": "1.2.840.10008.5.1.4.1.1.2", "0008|0060": "CT",
        "0020|000e": "test-series", "0020|0052": "test-frame",
        "0020|0032": f"0\\0\\{z}", "0020|0037": "1\\0\\0\\0\\1\\0",
        "0028|0030": "1.2\\0.7", "0028|0010": "5", "0028|0011": "6",
    } for z in (30, 32.5, 35, 37.5)]


@pytest.mark.parametrize("problem", ["missing_ipp", "duplicate", "nonuniform", "orientation", "pixels", "tilt", "enhanced", "multiframe"])
def test_classic_dicom_geometry_fail_closed(problem):
    headers = copy.deepcopy(classic_headers())
    if problem == "missing_ipp":
        del headers[1]["0020|0032"]
    elif problem == "duplicate":
        headers[1]["0020|0032"] = headers[0]["0020|0032"]
    elif problem == "nonuniform":
        headers[2]["0020|0032"] = "0\\0\\35.4"
    elif problem == "orientation":
        headers[1]["0020|0037"] = "0\\1\\0\\-1\\0\\0"
    elif problem == "pixels":
        headers[1]["0028|0030"] = "1.3\\0.7"
    elif problem == "tilt":
        for index, header in enumerate(headers):
            header["0020|0032"] = f"{index * .2}\\0\\{30 + 2.5 * index}"
    elif problem == "enhanced":
        headers = headers[:1]
        headers[0]["0008|0016"] = "1.2.840.10008.5.1.4.1.1.2.1"
    elif problem == "multiframe":
        headers = headers[:1]
        headers[0]["0028|0008"] = "4"
    with pytest.raises(ImagingError):
        _validate_classic_ct_geometry(headers, "test-series")


def write_synthetic_classic_ct(directory, uid, *, nonuniform=False, offset_hu=0):
    """Write actual classic CT DICOMs, including nontrivial rescale metadata."""
    sitk = pytest.importorskip("SimpleITK")
    directory.mkdir(parents=True, exist_ok=True)
    data = (-1000 + 2 * np.arange(4 * 5 * 6).reshape(4, 5, 6) + offset_hu).astype(np.int16)
    volume = sitk.GetImageFromArray(data)
    volume.SetSpacing((0.7, 1.2, 2.5))
    volume.SetOrigin((12., -8., 30.))
    volume.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for index in range(data.shape[0]):
        slice_image = volume[:, :, index]
        position = list(volume.TransformIndexToPhysicalPoint((0, 0, index)))
        if nonuniform and index >= 2:
            position[2] += .5
        tags = {
            "0008|0016": "1.2.840.10008.5.1.4.1.1.2", "0008|0060": "CT",
            "0008|0018": f"{uid}.{index + 1}",
            "0020|000d": "1.2.826.0.1.3680043.10.543.1", "0020|000e": uid,
            "0020|0052": "1.2.826.0.1.3680043.10.543.2", "0020|0013": str(index + 1),
            "0020|0032": "\\".join(str(v) for v in position),
            "0020|0037": "0\\1\\0\\-1\\0\\0", "0028|0030": "1.2\\0.7",
            "0028|1052": "-1000", "0028|1053": "2", "0018|0050": "2.5",
        }
        for key, value in tags.items():
            slice_image.SetMetaData(key, value)
        writer.SetFileName(str(directory / f"{uid}-{index:03d}.dcm"))
        writer.Execute(slice_image)
    return data, volume


@pytest.fixture
def dicom_tmp_path(tmp_path):
    # GDCM on Windows cannot write this runner's Unicode pytest user directory.
    # Keep real DICOM fixtures under the ASCII workspace instead of mocking IO.
    if sys.platform == "win32" and not str(tmp_path).isascii():
        parent = Path(__file__).resolve().parents[2] / "ctln-dicom-test-data"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="classic-", dir=parent) as directory:
            yield Path(directory)
    else:
        yield tmp_path


def test_real_classic_dicom_preserves_hu_lps_ras_and_selects_series(dicom_tmp_path):
    tmp_path = dicom_tmp_path
    uid_a = "1.2.826.0.1.3680043.10.543.11"
    uid_b = "1.2.826.0.1.3680043.10.543.12"
    original, volume = write_synthetic_classic_ct(tmp_path, uid_a)
    write_synthetic_classic_ct(tmp_path, uid_b, offset_hu=100)
    with pytest.raises(ImagingError, match="Multiple DICOM series"):
        load_ct(tmp_path)
    loaded = load_ct(tmp_path, series_uid=uid_a)
    np.testing.assert_array_equal(loaded.get_fdata(), original.transpose(2, 1, 0))
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = np.asarray(volume.GetDirection()).reshape(3, 3) @ np.diag(volume.GetSpacing())
    affine_lps[:3, 3] = volume.GetOrigin()
    np.testing.assert_allclose(loaded.affine, np.diag([-1., -1., 1., 1.]) @ affine_lps, atol=1e-6)
    assert loaded.extra["dicom_series_uid"] == uid_a
    assert loaded.header.get_xyzt_units()[0] == "mm"


def test_real_nonuniform_classic_dicom_is_rejected(dicom_tmp_path):
    tmp_path = dicom_tmp_path
    uid = "1.2.826.0.1.3680043.10.543.13"
    write_synthetic_classic_ct(tmp_path, uid, nonuniform=True)
    with pytest.raises(ImagingError, match="Nonuniform slice spacing"):
        load_ct(tmp_path, series_uid=uid)


def test_nonfinite_ct_and_singular_grid_are_rejected(tmp_path):
    path = tmp_path / "nan.nii.gz"
    values = np.zeros((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = np.nan
    nib.save(image(values), path)
    with pytest.raises(ImagingError, match="non-finite"):
        load_ct(path, assume_nifti_hu=True)
    bad = image(np.zeros((2, 2, 2), dtype=np.uint8))
    bad._affine[0, 0] = 0
    with pytest.raises(ImagingError, match="singular"):
        new_image_like(bad, values, dtype=np.float32)
