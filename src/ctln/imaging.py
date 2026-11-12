"""CT IO and categorical-mask IO, with explicit intensity and geometry contracts.

Network-normalized arrays must never replace the original HU image.  NIfTI does
not establish HU provenance by itself, so callers must confirm that provenance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


class ImagingError(ValueError):
    """An image cannot safely be interpreted using the requested contract."""


def validate_geometry(image: nib.spatialimages.SpatialImage) -> None:
    if len(image.shape) != 3 or any(n <= 0 for n in image.shape):
        raise ImagingError("Expected one nonempty 3D volume; select the series/phase explicitly.")
    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ImagingError("Image affine must be a finite 4 x 4 matrix.")
    if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ImagingError("Invalid homogeneous image affine.")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ImagingError("Image grid is singular.")


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    path = Path(path)
    if not path.is_file() or not str(path).lower().endswith((".nii", ".nii.gz")):
        raise ImagingError("Expected an existing .nii or .nii.gz file.")
    image = nib.load(str(path))
    validate_geometry(image)
    return image


def spatial_affine_mm(image: nib.Nifti1Image, *, spatial_unit_override: str | None = None) -> np.ndarray:
    """Return the complete affine in mm, including its translation.

    Unknown NIfTI units require an explicit caller assertion instead of silently
    turning voxel counts into purported mm measurements.
    """
    validate_geometry(image)
    unit = image.header.get_xyzt_units()[0]
    if unit == "unknown":
        if spatial_unit_override != "mm":
            raise ImagingError("Unknown spatial units; explicitly confirm mm with spatial_unit_override='mm'.")
        unit = "mm"
    elif spatial_unit_override is not None and spatial_unit_override != unit:
        raise ImagingError("Spatial-unit override conflicts with the image header.")
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors:
        raise ImagingError(f"Unsupported spatial unit: {unit}")
    affine = np.array(image.affine, dtype=float, copy=True)
    affine[:3, :] *= factors[unit]
    return affine


def load_ct(
    path: str | Path, series_uid: str | None = None, *,
    assume_nifti_hu: bool = False, assume_nifti_mm: bool = False,
) -> nib.Nifti1Image:
    """Read original CT, without clipping, normalizing, resizing, or writing it.

    DICOM directories are selected by SeriesInstanceUID.  Multiple series are
    never concatenated or guessed.  SimpleITK applies DICOM pixel rescaling;
    NIfTI HU provenance remains a caller responsibility.
    """
    path = Path(path)
    if path.is_dir():
        return _load_dicom_series(path, series_uid)
    if series_uid is not None:
        raise ImagingError("series_uid is only valid for a DICOM directory.")
    image = load_nifti(path)
    if not assume_nifti_hu:
        raise ImagingError("NIfTI does not prove HU provenance; confirm the original HU conversion with assume_nifti_hu=True.")
    affine_mm = spatial_affine_mm(image, spatial_unit_override="mm" if assume_nifti_mm and image.header.get_xyzt_units()[0] == "unknown" else None)
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ImagingError("CT contains non-finite intensities.")
    header = image.header.copy()
    header.set_xyzt_units("mm")
    header.set_data_dtype(np.float32)
    result = nib.Nifti1Image(data, affine_mm, header=header)
    result.header.set_slope_inter(1.0, 0.0)
    return result


def load_labelmap(path: str | Path, *, assume_nifti_mm: bool = False) -> nib.Nifti1Image:
    """Read categorical NIfTI labels onto their physical-mm grid, without resampling.

    Use the same unit normalization as ``load_ct`` so a caller's explicit mm
    assertion applies to an unknown-unit mask as well as its source CT. Known
    metre/micron units are converted, including the affine translation. Integer
    component values are preserved; probabilities and scaled floating masks are
    rejected instead of implicitly thresholded or rounded.
    """
    image = load_nifti(path)
    data = categorical_data(image)
    override = "mm" if assume_nifti_mm and image.header.get_xyzt_units()[0] == "unknown" else None
    affine_mm = spatial_affine_mm(image, spatial_unit_override=override)
    header = image.header.copy()
    header.set_xyzt_units("mm")
    header.set_data_dtype(data.dtype)
    result = nib.Nifti1Image(data, affine_mm, header=header)
    result.header.set_slope_inter(1.0, 0.0)
    return result


def _load_dicom_series(path: Path, series_uid: str | None) -> nib.Nifti1Image:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("DICOM loading requires the optional SimpleITK dependency.") from exc
    series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or [])
    if not series_ids:
        raise ImagingError("No readable DICOM series found in this directory.")
    if series_uid is None:
        if len(series_ids) != 1:
            raise ImagingError("Multiple DICOM series found; supply series_uid. Available UIDs: " + ", ".join(series_ids))
        series_uid = series_ids[0]
    if series_uid not in series_ids:
        raise ImagingError("Requested series_uid is not present in the DICOM directory.")
    filenames = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_uid))
    if not filenames:
        raise ImagingError("Selected DICOM series is empty.")
    # GDCM can warn about an irregular stack and still return a regular image.
    # Validate source headers before accepting that reconstructed geometry.
    tags = (
        "0008|0016", "0008|0060", "0020|000e", "0020|0052", "0020|0032",
        "0020|0037", "0028|0030", "0028|0010", "0028|0011", "0028|0008",
    )
    metadata = []
    for filename in filenames:
        header_reader = sitk.ImageFileReader()
        header_reader.SetFileName(filename)
        header_reader.ReadImageInformation()
        metadata.append({tag: header_reader.GetMetaData(tag).strip() for tag in tags if header_reader.HasMetaDataKey(tag)})
    geometry = _validate_classic_ct_geometry(metadata, series_uid)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    volume = reader.Execute()
    if volume.GetDimension() != 3 or volume.GetNumberOfComponentsPerPixel() != 1:
        raise ImagingError("Select a scalar 3D CT series, not a multichannel or temporal volume.")
    if tuple(volume.GetSize()) != geometry["shape_xyz"]:
        raise ImagingError("DICOM reader output shape differs from the validated classic CT stack.")
    if not np.allclose(volume.GetSpacing(), geometry["spacing_xyz"], atol=1e-3, rtol=1e-4):
        raise ImagingError("DICOM reader output spacing differs from source slice geometry.")
    if not np.allclose(volume.GetOrigin(), geometry["origin_lps"], atol=1e-3, rtol=0):
        raise ImagingError("DICOM reader output origin differs from source slice geometry.")
    if not np.allclose(np.asarray(volume.GetDirection()).reshape(3, 3), geometry["direction_lps"], atol=1e-4, rtol=0):
        raise ImagingError("DICOM reader output direction differs from source slice geometry.")
    data = np.asarray(sitk.GetArrayFromImage(volume), dtype=np.float32).transpose(2, 1, 0)
    if not np.isfinite(data).all():
        raise ImagingError("CT contains non-finite intensities.")
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = np.asarray(volume.GetDirection()).reshape(3, 3) @ np.diag(volume.GetSpacing())
    affine_lps[:3, 3] = volume.GetOrigin()
    image = nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps)
    image.header.set_xyzt_units("mm")
    image.extra["dicom_series_uid"] = series_uid
    image.extra["dicom_geometry_validation"] = "classic_single_frame_regular_orthogonal_stack"
    validate_geometry(image)
    return image


def _validate_classic_ct_geometry(metadata: list[dict[str, str]], series_uid: str) -> dict[str, Any]:
    """Validate classic single-frame CT headers in the reader's slice order.

    Enhanced/multi-frame CT and gantry-tilted/sheared or irregular stacks require
    an external validated conversion. We do not repair, reorder, or guess them.
    """
    classic_ct_sop = "1.2.840.10008.5.1.4.1.1.2"
    if not metadata:
        raise ImagingError("Selected DICOM series is empty.")

    def required(header: dict[str, str], key: str, name: str) -> str:
        value = header.get(key, "").strip()
        if not value:
            raise ImagingError(f"Classic CT requires {name} ({key}) on every slice; tag missing.")
        return value

    def numbers(header: dict[str, str], key: str, count: int, name: str) -> np.ndarray:
        try:
            values = np.asarray([float(v) for v in required(header, key, name).split("\\")])
        except ValueError as exc:
            raise ImagingError(f"Invalid {name} ({key}) in classic CT slice.") from exc
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ImagingError(f"Invalid {name} ({key}) in classic CT slice.")
        return values

    positions, orientations, spacings, sizes, frame_uids = [], [], [], [], []
    for header in metadata:
        sop = required(header, "0008|0016", "SOPClassUID")
        try:
            frames = int(header.get("0028|0008", "1"))
        except ValueError as exc:
            raise ImagingError("Invalid NumberOfFrames in CT input.") from exc
        if sop != classic_ct_sop or frames != 1:
            raise ImagingError("Only classic single-frame CT series are supported; enhanced/multi-frame or other SOP classes require an external validated conversion.")
        if required(header, "0008|0060", "Modality") != "CT":
            raise ImagingError("Selected DICOM series is not consistently identified as CT.")
        if required(header, "0020|000e", "SeriesInstanceUID") != series_uid:
            raise ImagingError("Selected files contain inconsistent DICOM series UIDs.")
        frame_uids.append(required(header, "0020|0052", "FrameOfReferenceUID"))
        positions.append(numbers(header, "0020|0032", 3, "ImagePositionPatient"))
        orientation = numbers(header, "0020|0037", 6, "ImageOrientationPatient")
        row, column = orientation[:3], orientation[3:]
        if not np.isclose(np.linalg.norm(row), 1, atol=1e-4, rtol=0) or not np.isclose(np.linalg.norm(column), 1, atol=1e-4, rtol=0) or abs(np.dot(row, column)) > 1e-4:
            raise ImagingError("ImageOrientationPatient must define orthonormal in-plane axes.")
        orientations.append(orientation)
        spacing = numbers(header, "0028|0030", 2, "PixelSpacing")
        if np.any(spacing <= 0):
            raise ImagingError("PixelSpacing must be positive.")
        spacings.append(spacing)
        size = numbers(header, "0028|0010", 1, "Rows").tolist() + numbers(header, "0028|0011", 1, "Columns").tolist()
        if any(value <= 0 or value != int(value) for value in size):
            raise ImagingError("Rows and Columns must be positive integers.")
        sizes.append(size)
    if len(metadata) < 2:
        raise ImagingError("At least two classic single-frame slices are required to establish a regular 3D CT grid; convert other input externally.")
    if len(set(frame_uids)) != 1:
        raise ImagingError("Classic CT slices have inconsistent FrameOfReferenceUID.")
    if not np.allclose(orientations, orientations[0], atol=1e-5, rtol=0):
        raise ImagingError("Inconsistent ImageOrientationPatient across CT slices.")
    if not np.allclose(spacings, spacings[0], atol=1e-5, rtol=1e-5):
        raise ImagingError("Inconsistent PixelSpacing across CT slices.")
    if not np.array_equal(sizes, np.broadcast_to(sizes[0], (len(sizes), 2))):
        raise ImagingError("Inconsistent Rows/Columns across CT slices.")
    positions = np.asarray(positions)
    row = orientations[0][:3] / np.linalg.norm(orientations[0][:3])
    column = orientations[0][3:] / np.linalg.norm(orientations[0][3:])
    normal = np.cross(row, column)
    normal /= np.linalg.norm(normal)
    increments = np.diff(positions, axis=0)
    distances = increments @ normal
    if np.any(np.abs(distances) <= 1e-3):
        raise ImagingError("Duplicate slice positions or zero through-plane spacing in classic CT.")
    if not np.all(np.sign(distances) == np.sign(distances[0])):
        raise ImagingError("Slice positions are not monotonic; refusing to guess slice ordering.")
    if not np.allclose(distances, distances[0], atol=1e-3, rtol=1e-4):
        raise ImagingError("Nonuniform slice spacing in classic CT; external validated conversion is required.")
    if np.any(np.linalg.norm(increments - distances[:, None] * normal, axis=1) > 1e-3):
        raise ImagingError("Gantry tilt/in-plane slice offsets cannot be represented by the supported regular orthogonal CT grid; convert externally.")
    return {
        "origin_lps": positions[0],
        "spacing_xyz": np.asarray([spacings[0][1], spacings[0][0], abs(distances[0])]),
        "direction_lps": np.column_stack((row, column, normal * np.sign(distances[0]))),
        "shape_xyz": (int(sizes[0][1]), int(sizes[0][0]), len(metadata)),
    }


def geometry_matches(a: nib.Nifti1Image, b: nib.Nifti1Image, *, tolerance: float = 1e-5) -> bool:
    validate_geometry(a)
    validate_geometry(b)
    # Include unit conversion: identical numeric affines need not mean identical
    # physical grids if one file is encoded in metres and another in millimetres.
    unit_a, unit_b = a.header.get_xyzt_units()[0], b.header.get_xyzt_units()[0]
    if unit_a != unit_b:
        if "unknown" in (unit_a, unit_b):
            return False
        affine_a, affine_b = spatial_affine_mm(a), spatial_affine_mm(b)
    else:
        affine_a, affine_b = a.affine, b.affine
    return a.shape == b.shape and bool(np.allclose(affine_a, affine_b, atol=tolerance, rtol=0))


def require_same_geometry(a: nib.Nifti1Image, b: nib.Nifti1Image, *, tolerance: float = 1e-5) -> None:
    if not geometry_matches(a, b, tolerance=tolerance):
        raise ImagingError("Physical grids differ (shape, affine, or units); explicitly resample the categorical mask to CT.")


def categorical_data(image: nib.Nifti1Image, *, binary: bool = False) -> np.ndarray:
    validate_geometry(image)
    data = np.asanyarray(image.dataobj)
    if data.dtype.kind not in "biu":
        raise ImagingError("Masks must contain integer/bool categorical labels; a float probability map is not a mask.")
    if np.any(data < 0) or (data.size and int(data.max()) > np.iinfo(np.uint32).max):
        raise ImagingError("Mask labels must be nonnegative uint32-compatible integers.")
    if binary and not np.isin(data, [0, 1]).all():
        raise ImagingError("Expected one binary structure mask (values 0 and 1).")
    return data


def new_image_like(reference: nib.Nifti1Image, data: Any, *, dtype: Any) -> nib.Nifti1Image:
    validate_geometry(reference)
    array = np.asarray(data, dtype=dtype)
    if array.dtype.kind == "b":
        array = array.astype(np.uint8)
    if array.shape != reference.shape:
        raise ImagingError("Output shape must match the reference CT grid.")
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    image = nib.Nifti1Image(array, np.array(reference.affine, copy=True), header=header)
    # Avoid inheriting any CT intensity scaling in a categorical/probability file.
    image.header.set_slope_inter(1.0, 0.0)
    return image


def resample_mask_to_ct(mask: nib.Nifti1Image, ct: nib.Nifti1Image, *, allow_resample: bool = False) -> nib.Nifti1Image:
    data = categorical_data(mask)
    validate_geometry(ct)
    if geometry_matches(mask, ct):
        return new_image_like(ct, data, dtype=data.dtype)
    if not allow_resample:
        raise ImagingError("Mask/CT grids differ. Set allow_resample=True to explicitly use nearest-neighbour resampling.")
    unit_mask, unit_ct = mask.header.get_xyzt_units()[0], ct.header.get_xyzt_units()[0]
    if "unknown" in (unit_mask, unit_ct) and unit_mask != unit_ct:
        raise ImagingError("Cannot resample grids with incompatible/unknown spatial units.")
    source = mask
    if unit_mask != unit_ct:
        affine = spatial_affine_mm(mask)
        factor_ct = {"mm": 1, "meter": 1000, "micron": 0.001}[unit_ct]
        affine[:3, :] /= factor_ct
        source = nib.Nifti1Image(data, affine)
    result = resample_from_to(source, (ct.shape, ct.affine), order=0, mode="constant", cval=0)
    return new_image_like(ct, np.asarray(result.dataobj), dtype=data.dtype)


def _save(image: nib.Nifti1Image, path: str | Path) -> Path:
    path = Path(path)
    if not str(path).lower().endswith((".nii", ".nii.gz")):
        raise ImagingError("Output must have a .nii or .nii.gz suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))
    return path


def save_probability(probability: Any, reference: nib.Nifti1Image, path: str | Path) -> Path:
    if isinstance(probability, nib.spatialimages.SpatialImage):
        require_same_geometry(probability, reference)
        probability = probability.get_fdata(dtype=np.float32)
    data = np.asarray(probability, dtype=np.float32)
    if not np.isfinite(data).all() or np.any(data < -1e-6) or np.any(data > 1 + 1e-6):
        raise ImagingError("Segmentation probability values must be finite and in [0, 1].")
    return _save(new_image_like(reference, np.clip(data, 0, 1), dtype=np.float32), path)


def save_labels(labels: Any, reference: nib.Nifti1Image, path: str | Path, *, dtype: Any = np.uint32) -> Path:
    if isinstance(labels, nib.spatialimages.SpatialImage):
        require_same_geometry(labels, reference)
        labels = categorical_data(labels)
    array = np.asarray(labels)
    image = new_image_like(reference, array, dtype=array.dtype)
    categorical_data(image)
    target_dtype = np.dtype(dtype)
    if target_dtype.kind != "u" or (array.size and int(array.max()) > np.iinfo(target_dtype).max):
        raise ImagingError("Selected unsigned label dtype would lose label values.")
    return _save(new_image_like(reference, array, dtype=target_dtype), path)


def save_mask(mask: Any, reference: nib.Nifti1Image, path: str | Path) -> Path:
    if isinstance(mask, nib.spatialimages.SpatialImage):
        require_same_geometry(mask, reference)
        array = categorical_data(mask, binary=True)
    else:
        array = np.asarray(mask)
        categorical_data(new_image_like(reference, array, dtype=array.dtype), binary=True)
    return save_labels(array, reference, path, dtype=np.uint8)
