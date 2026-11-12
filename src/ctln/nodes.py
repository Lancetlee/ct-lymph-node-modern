"""Connected components are review candidates, not ground-truth lymph nodes."""
from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from .imaging import categorical_data, new_image_like, require_same_geometry, spatial_affine_mm


def extract_candidates(
    mask: nib.Nifti1Image,
    ct: nib.Nifti1Image | None = None,
    *,
    scan_id: str = "scan",
    connectivity: int = 26,
    min_volume_mm3: float = 0.0,
    spatial_unit_override: str | None = None,
) -> tuple[nib.Nifti1Image, list[dict[str, Any]]]:
    """Label one binary structure mask and create pending-review records.

    A component may merge several true nodes, represent part of one node, or be
    a false positive. Volume and centroid are geometric measurements, not an
    axial clinical short-axis diameter or a station/pathology assignment.
    """
    if not str(scan_id).strip():
        raise ValueError("scan_id must be nonempty.")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26.")
    if not np.isfinite(min_volume_mm3) or min_volume_mm3 < 0:
        raise ValueError("min_volume_mm3 must be finite and nonnegative.")
    binary = categorical_data(mask, binary=True)
    if ct is not None:
        require_same_geometry(mask, ct)
    affine_mm = spatial_affine_mm(mask, spatial_unit_override=spatial_unit_override)
    voxel_volume = float(abs(np.linalg.det(affine_mm[:3, :3])))
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    raw, count = ndimage.label(binary, structure=structure, output=np.uint32)
    labels = np.zeros(raw.shape, dtype=np.uint32)
    records: list[dict[str, Any]] = []
    # find_objects avoids scanning the entire CT once per component.
    for raw_label, bounds in enumerate(ndimage.find_objects(raw, max_label=count), start=1):
        if bounds is None:
            continue
        local = raw[bounds] == raw_label
        voxel_count = int(local.sum())
        volume_mm3 = voxel_count * voxel_volume
        if volume_mm3 < min_volume_mm3:
            continue
        label_id = len(records) + 1
        view = labels[bounds]
        view[local] = label_id
        offset = np.array([b.start for b in bounds])
        centroid_ijk = np.asarray(ndimage.center_of_mass(local)) + offset
        centroid_mm = nib.affines.apply_affine(affine_mm, centroid_ijk)
        records.append({
            "scan_id": str(scan_id),
            "candidate_id": f"{scan_id}:candidate-{label_id:04d}",
            "node_id": None,
            "component_label": label_id,
            "roi_unit": "candidate",
            "voxel_count": voxel_count,
            "volume_mm3": float(volume_mm3),
            "centroid_world_mm": [float(x) for x in centroid_mm],
            "bbox_voxel_ijk": [[int(b.start) for b in bounds], [int(b.stop) for b in bounds]],
            "short_axis_mm": None,
            "station": None,
            "reference_standard": {
                "result": "unverified", "label_unit": None, "matching_evidence": None,
                "method": None, "sampling_adequacy": None,
            },
            "review": {"status": "pending", "reviewer": None, "reviewed_at": None, "mask_approved": False},
            "quality_flags": ["connected_component_is_not_verified_anatomical_node"],
            "component_parameters": {"connectivity": connectivity, "min_volume_mm3": float(min_volume_mm3)},
        })
    return new_image_like(mask, labels, dtype=np.uint32), records


def require_approved_record(record: dict[str, Any]) -> None:
    review = record.get("review") or {}
    if review.get("status") != "approved" or review.get("mask_approved") is not True:
        raise ValueError("Every extracted ROI requires approved human mask review; unreviewed candidates are blocked.")
    if not str(review.get("reviewer") or "").strip() or not str(review.get("reviewed_at") or "").strip():
        raise ValueError("Human review must include reviewer and reviewed_at.")
    if not str(record.get("node_id") or "").strip():
        raise ValueError("The reviewer must assign a node_id/ROI identifier; candidate IDs are not verified node identities.")
    reference = record.get("reference_standard") or {}
    if reference.get("result", "unverified") not in {"positive", "negative", "indeterminate", "unverified"}:
        raise ValueError("Invalid reference-standard result; do not invent pathology labels.")


def has_verified_node_label(record: dict[str, Any]) -> bool:
    """Only indicate explicit, paired node supervision; never infer it from CT."""
    reference = record.get("reference_standard") or {}
    tissue_methods = {"pathology", "cytology", "surgical_pathology", "biopsy_pathology", "ebus_tbna", "eus_fna"}
    return bool(
        record.get("roi_unit") == "node"
        and reference.get("label_unit") == "node"
        and reference.get("result") in {"positive", "negative"}
        and reference.get("matching_evidence")
        and reference.get("method") in tissue_methods
        and (reference.get("result") != "negative" or reference.get("sampling_adequacy") == "adequate")
    )
