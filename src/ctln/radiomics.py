"""Optional PyRadiomics bridge for original-HU CT and human-reviewed ROIs.

This module extracts reproducible features. It neither assigns pathology/stations
nor trains a classifier. A segmentation probability is not a categorical ROI.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

from .imaging import categorical_data, load_nifti, require_same_geometry, spatial_affine_mm
from .nodes import has_verified_node_label, require_approved_record


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Non-finite features/configuration cannot be represented as a real measurement.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Configuration/manifest contains a non-JSON value: {type(value).__name__}")


def freeze_parameters(params: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Freeze explicit settings instead of silently inheriting library defaults."""
    if not isinstance(params, Mapping) or not params:
        raise ValueError("Supply an explicit PyRadiomics parameter mapping, including setting and featureClass.")
    frozen = _jsonable(copy.deepcopy(dict(params)))
    settings = frozen.get("setting")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Explicit PyRadiomics 'setting' configuration is required.")
    if ("binWidth" in settings) == ("binCount" in settings):
        raise ValueError("Freeze exactly one discretization setting: binWidth or binCount.")
    if "normalize" not in settings:
        raise ValueError("Freeze the normalize setting explicitly; CT HU preprocessing must not be implicit.")
    if settings.get("correctMask"):
        raise ValueError("Implicit correctMask is disabled; explicitly resample/check the label image upstream.")
    if not isinstance(frozen.get("featureClass"), dict) or not frozen["featureClass"]:
        raise ValueError("Freeze the enabled featureClass mapping explicitly.")
    if not isinstance(frozen.get("imageType"), dict) or not frozen["imageType"]:
        raise ValueError("Freeze imageType explicitly (for example {'Original': {}}).")
    canonical = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return frozen, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coerce_image(image: str | Path | nib.Nifti1Image) -> nib.Nifti1Image:
    return load_nifti(image) if isinstance(image, (str, Path)) else image


def _image_fingerprint(image: nib.Nifti1Image) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(image.affine, dtype="<f8").tobytes())
    digest.update(str(image.header.get_xyzt_units()).encode("ascii"))
    data = np.ascontiguousarray(np.asanyarray(image.dataobj))
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def _to_sitk(image: nib.Nifti1Image, *, spatial_unit_override: str | None = None):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("Radiomics requires optional PyRadiomics and SimpleITK dependencies.") from exc
    affine = spatial_affine_mm(image, spatial_unit_override=spatial_unit_override)
    affine_lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    spacing = np.linalg.norm(affine_lps[:3, :3], axis=0)
    direction = affine_lps[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("SimpleITK cannot represent a sheared grid safely; explicitly resample to an orthogonal CT grid first.")
    data = np.asarray(image.dataobj)
    volume = sitk.GetImageFromArray(np.ascontiguousarray(data.transpose(2, 1, 0)))
    volume.SetSpacing(tuple(float(x) for x in spacing))
    volume.SetOrigin(tuple(float(x) for x in affine_lps[:3, 3]))
    volume.SetDirection(tuple(float(x) for x in direction.ravel()))
    return volume


def _make_extractor(params: dict[str, Any]):
    try:
        package = importlib.import_module("radiomics")
        featureextractor = importlib.import_module("radiomics.featureextractor")
    except ImportError as exc:
        raise ImportError("Feature extraction requires the optional PyRadiomics dependency.") from exc
    return featureextractor.RadiomicsFeatureExtractor(params), str(getattr(package, "__version__", "unknown"))


def extract_radiomics(
    ct: str | Path | nib.Nifti1Image,
    labels: str | Path | nib.Nifti1Image,
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    params: Mapping[str, Any],
    hu_confirmed: bool = False,
    spatial_unit_override: str | None = None,
) -> dict[str, Path]:
    """Extract one row per approved ROI and write features.csv + manifest.json.

    Every nonzero mask label must have one review record. Pending review blocks
    the run; explicitly rejected ROIs are retained in the manifest and skipped.
    Unverified pathology remains unverified, even for an approved image ROI.
    """
    if not hu_confirmed:
        raise ValueError("Confirm original CT HU provenance with hu_confirmed=True; network-normalized CT is not accepted.")
    ct_image, label_image = _coerce_image(ct), _coerce_image(labels)
    require_same_geometry(ct_image, label_image)
    label_data = categorical_data(label_image)
    ct_data = ct_image.get_fdata(dtype=np.float32)
    if not np.isfinite(ct_data).all():
        raise ValueError("Original CT contains non-finite values.")
    # Verify physical units before importing optional extraction dependencies.
    spatial_affine_mm(ct_image, spatial_unit_override=spatial_unit_override)
    frozen, config_hash = freeze_parameters(params)
    record_list = _jsonable([dict(record) for record in records])
    present = set(int(v) for v in np.unique(label_data) if v != 0)
    record_labels: set[int] = set()
    node_ids: set[str] = set()
    approved: list[dict[str, Any]] = []
    for record in record_list:
        label_id = record.get("component_label")
        if isinstance(label_id, bool) or not isinstance(label_id, int) or label_id <= 0:
            raise ValueError("Each record needs a positive integer component_label.")
        if label_id in record_labels:
            raise ValueError("Each component_label must have exactly one review record.")
        record_labels.add(label_id)
        if (record.get("review") or {}).get("status") == "rejected":
            continue
        require_approved_record(record)
        node_id = str(record["node_id"])
        if node_id in node_ids:
            raise ValueError("Reviewed node_id/ROI identifiers must be unique.")
        node_ids.add(node_id)
        approved.append(record)
    if record_labels != present:
        raise ValueError("Review records must match every nonzero mask label exactly; no absent or unreviewed label is allowed.")
    if not approved:
        raise ValueError("No approved ROIs available for feature extraction.")
    extractor, version = _make_extractor(frozen)
    image_sitk = _to_sitk(ct_image, spatial_unit_override=spatial_unit_override)
    labels_sitk = _to_sitk(label_image, spatial_unit_override=spatial_unit_override)
    rows: list[dict[str, Any]] = []
    roi_results: list[dict[str, Any]] = []
    for record in approved:
        label_id = int(record["component_label"])
        try:
            extracted = _jsonable(extractor.execute(image_sitk, labels_sitk, label=label_id))
        except Exception as exc:
            # Never write a zero-valued fake measurement for failed extraction.
            raise ValueError(f"Radiomics failed for node_id={record['node_id']!r}, label={label_id}: {exc}") from exc
        feature_values = {k: v for k, v in extracted.items() if not k.startswith("diagnostics_")}
        if not feature_values or any(not isinstance(v, (float, int)) or isinstance(v, bool) for v in feature_values.values()):
            raise ValueError(f"Expected finite scalar feature values for node_id={record['node_id']!r}.")
        row = {
            "node_id": record["node_id"], "candidate_id": record.get("candidate_id"),
            "scan_id": record.get("scan_id"), "component_label": label_id,
            "roi_unit": record.get("roi_unit", "candidate"), "config_sha256": config_hash,
            **feature_values,
        }
        rows.append(row)
        roi_results.append({
            "node_id": record["node_id"], "component_label": label_id,
            "extraction_status": "complete",
            "training_label_verified": has_verified_node_label(record),
            "diagnostics": {k: v for k, v in extracted.items() if k.startswith("diagnostics_")},
        })
    manifest = {
        "schema_version": "1.0", "purpose": "reviewed_ROI_feature_extraction_not_a_classifier",
        "extraction_status": "complete",
        "hu_provenance_confirmed_by_caller": True,
        "spatial_unit_override": spatial_unit_override,
        "ct_sha256": _image_fingerprint(ct_image), "labels_sha256": _image_fingerprint(label_image),
        "shape": list(ct_image.shape), "affine": ct_image.affine.tolist(),
        "spatial_units": ct_image.header.get_xyzt_units()[0],
        "parameters": frozen, "config_sha256": config_hash, "pyradiomics_version": version,
        "effective_settings": _jsonable(getattr(extractor, "settings", {})),
        "enabled_image_types": _jsonable(getattr(extractor, "enabledImagetypes", {})),
        "enabled_features": _jsonable(getattr(extractor, "enabledFeatures", {})),
        "records": record_list, "roi_results": roi_results,
        "training_warning": "Image review does not verify pathology. Candidate components are not automatically single nodes or IASLC stations. Patient-isolated training/validation remains a separate task.",
    }
    # Complete all extraction/validation before publishing either output.
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path = destination / "features.csv", destination / "manifest.json"
    metadata_fields = ["node_id", "candidate_id", "scan_id", "component_label", "roi_unit", "config_sha256"]
    feature_fields = sorted(set().union(*(row.keys() for row in rows)) - set(metadata_fields))
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=destination,
                                         prefix='.ctln-features-', suffix='.csv', delete=False) as handle:
            csv_temp = Path(handle.name)
            temporary_paths.append(csv_temp)
            writer = csv.DictWriter(handle, fieldnames=metadata_fields + feature_fields)
            writer.writeheader()
            writer.writerows(rows)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination,
                                         prefix='.ctln-manifest-', suffix='.json', delete=False) as handle:
            manifest_temp = Path(handle.name)
            temporary_paths.append(manifest_temp)
            handle.write(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        csv_temp.replace(csv_path)
        manifest_temp.replace(manifest_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return {"features_csv": csv_path, "manifest_json": manifest_path}
