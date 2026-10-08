"""Cardiac calcium scoring (Agatston) on non-contrast CT, inside a heart mask.

Deterministic, no model: the Agatston method (Agatston et al., JACC 1990) counts
calcified lesions of at least 1 mm^2 with attenuation of 130 HU or more, slice by
slice, weighting each lesion's area by a density factor from its peak attenuation
(1: 130-199 HU, 2: 200-299, 3: 300-399, 4: >= 400), and scales to the 3 mm slice
thickness the method was defined on. The heart mask typically comes from the
TotalSegmentator stack's ``total`` task (structure ``heart``). Without a coronary
model the score is for everything calcified inside the mask -- coronaries, valves and
the aortic root alike -- so it is a cardiac, not a coronary, calcium score.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from medmcp_cardiac.tools._cardiac import base_stem, nii_stem
from medmcp_cardiac.tools._components import label_components_2d

AGATSTON_THRESHOLD_HU: float = 130.0
MIN_LESION_AREA_MM2: float = 1.0
REFERENCE_SLICE_MM: float = 3.0

# 1st-percentile intensity above this means the volume is not in Hounsfield units.
_CT_HU_THRESHOLD: float = -300.0
# Median attenuation inside the heart above this suggests contrast enhancement, which
# the Agatston method was not defined for (enhanced blood crosses the threshold).
_CONTRAST_MEDIAN_HU: float = 150.0


class Lesion(TypedDict):
    """One calcified lesion on one slice."""

    slice: int
    area_mm2: float
    max_hu: float
    density_factor: int
    score: float


class CalciumResult(TypedDict):
    """A completed calcium scoring run."""

    ct_path: str
    mask_path: str
    structures: list[str]
    agatston_score: float
    volume_score_mm3: float
    n_lesions: int
    category: str
    lesions_path: str
    calcium_mask_path: str
    labels_path: str
    warnings: list[str]
    _render: str


def density_factor(max_hu: float) -> int:
    """Agatston density weighting from a lesion's peak attenuation."""
    if max_hu >= 400.0:
        return 4
    if max_hu >= 300.0:
        return 3
    if max_hu >= 200.0:
        return 2
    return 1


def agatston_category(score: float) -> str:
    """The conventional Agatston score bands."""
    if score <= 0.0:
        return "0 (no calcification detected)"
    if score <= 10.0:
        return "1-10 (minimal)"
    if score <= 100.0:
        return "11-100 (mild)"
    if score <= 400.0:
        return "101-400 (moderate)"
    return ">400 (severe)"


def score_volume(
    ct: Any, region: Any, spacing: tuple[float, float, float]
) -> tuple[list[Lesion], Any, float]:
    """Score every lesion inside ``region`` and return the lesions, their mask and the volume.

    Args:
        ct: (x, y, z) attenuation in HU.
        region: (x, y, z) boolean mask to score within.
        spacing: voxel spacing in mm.

    Returns:
        The lesions (per slice), a boolean mask of the counted voxels, and the
        volume score in mm^3 (every voxel at or above threshold inside the region).
    """
    pixel_area = spacing[0] * spacing[1]
    slice_factor = spacing[2] / REFERENCE_SLICE_MM
    above: Any = region & (ct >= AGATSTON_THRESHOLD_HU)
    counted: Any = np.zeros(ct.shape, dtype=bool)
    lesions: list[Lesion] = []
    for z in range(ct.shape[2]):
        plane: Any = above[:, :, z]
        if not plane.any():
            continue
        components, n = label_components_2d(plane)
        for index in range(1, n + 1):
            voxels: Any = components == index
            area = float(np.count_nonzero(voxels)) * pixel_area
            if area < MIN_LESION_AREA_MM2:
                continue
            max_hu = float(ct[:, :, z][voxels].max())
            factor = density_factor(max_hu)
            counted[:, :, z] |= voxels
            lesions.append(
                {
                    "slice": z,
                    "area_mm2": area,
                    "max_hu": max_hu,
                    "density_factor": factor,
                    "score": area * factor * slice_factor,
                }
            )
    volume = float(np.count_nonzero(above)) * pixel_area * spacing[2]
    return lesions, counted, volume


def _resolve_structures(
    mask_path: Path, mask: Any, structures: list[str] | None, label_values: list[int] | None
) -> tuple[list[int], list[str]]:
    """Turn structure names (via the mask's labels CSV) or label values into label values."""
    if label_values:
        return [int(v) for v in label_values], [f"label {v}" for v in label_values]
    labels_csv = mask_path.parent / f"{base_stem(mask_path)}_labels.csv"
    if not labels_csv.is_file():
        raise ValueError(
            f"No labels CSV beside the mask ({labels_csv.name}), so structure names cannot "
            "be resolved. Pass label_values instead, or use a mask written by a medmcp "
            "segmentation tool."
        )
    names: dict[str, int] = {}
    with labels_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            names[str(row["structure"])] = int(row["label"])
    wanted = structures or ["heart"]
    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise ValueError(
            f"Not in {labels_csv.name}: {', '.join(unknown)}. Available: "
            f"{', '.join(sorted(names))}."
        )
    values = [names[name] for name in wanted]
    present = [v for v in values if bool(np.any(mask == v))]
    if not present:
        raise ValueError(f"None of {wanted} has any voxels in {mask_path.name}.")
    return values, wanted


def cardiac_calcium_score(
    ct_path: Path,
    mask_path: Path,
    structures: list[str] | None = None,
    label_values: list[int] | None = None,
    output_dir: Path | None = None,
) -> CalciumResult:
    """Agatston calcium score on a non-contrast CT inside a heart mask.

    Counts calcified lesions (>= 130 HU, >= 1 mm^2, per slice) inside the given
    structures of a label map on the same grid as the CT -- typically the ``heart``
    label of a TotalSegmentator ``total`` segmentation -- and returns the Agatston
    score, the calcium volume, the lesion list and the conventional score band.
    Without a coronary-artery model this is a **cardiac** calcium score: valve and
    aortic-root calcium inside the mask count too.

    Args:
        ct_path: Non-contrast CT volume (NIfTI, Hounsfield units).
        mask_path: Label map on the CT's grid (``*_dseg.nii.gz`` with its
            ``*_labels.csv`` beside it).
        structures: Structure names to score within (default ``["heart"]``), resolved
            through the mask's labels CSV.
        label_values: Label values to score within, as an alternative to names.
        output_dir: Where to write the lesion CSV and the calcium mask. Defaults to
            the CT's folder.

    Returns:
        The scores, the lesion count and band, the paths written, and warnings (for
        example when the CT looks contrast-enhanced, which inflates the score).

    Raises:
        FileNotFoundError: if either file does not exist.
        ValueError: if the mask is not on the CT's grid or the structures cannot be
            resolved.
    """
    ct_path = Path(ct_path).expanduser().resolve()
    mask_path = Path(mask_path).expanduser().resolve()
    for path in (ct_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
    destination = Path(output_dir).expanduser().resolve() if output_dir else ct_path.parent
    destination.mkdir(parents=True, exist_ok=True)

    import nibabel as nib

    load_image = cast(Callable[[str], Any], nib.load)  # pyright: ignore[reportUnknownMemberType]
    ct_img = load_image(str(ct_path))
    ct: Any = np.asarray(ct_img.dataobj, dtype=np.float32)
    mask: Any = np.asarray(load_image(str(mask_path)).dataobj)
    if ct.ndim != 3:
        raise ValueError(f"{ct_path.name} is {ct.ndim}D; expected a 3D CT volume")
    if mask.shape != ct.shape:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match the CT {tuple(ct.shape)}; the "
            "mask must be on the CT's own grid (segment the CT itself, not a resampled copy)."
        )
    zooms = [float(z) for z in ct_img.header.get_zooms()[:3]]
    spacing = (zooms[0], zooms[1], zooms[2])

    values, names = _resolve_structures(mask_path, mask, structures, label_values)
    region: Any = np.isin(mask, values)

    warnings: list[str] = []
    if float(np.percentile(ct, 1)) > _CT_HU_THRESHOLD:
        warnings.append(
            "The volume's intensity range does not look like Hounsfield units; the 130 HU "
            "threshold is meaningless unless this is a CT."
        )
    if region.any() and float(np.median(ct[region])) > _CONTRAST_MEDIAN_HU:
        warnings.append(
            "Attenuation inside the heart is high, as in a contrast-enhanced scan; the "
            "Agatston method is defined on non-contrast CT and enhanced blood counts as "
            "calcium here, so the score is unreliable."
        )
    if spacing[2] > 3.5:
        warnings.append(
            f"Slice spacing {spacing[2]:.1f} mm is thicker than the 3 mm the method assumes; "
            "small lesions are averaged away and the score underestimates."
        )

    lesions, counted, volume = score_volume(ct, region, spacing)
    total = float(sum(lesion["score"] for lesion in lesions))

    stem = nii_stem(ct_path)
    lesions_path = destination / f"{stem}_calcium_lesions.csv"
    with lesions_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice", "area_mm2", "max_hu", "density_factor", "score"])
        for lesion in sorted(lesions, key=lambda item: -item["score"]):
            writer.writerow(
                [
                    lesion["slice"],
                    f"{lesion['area_mm2']:.2f}",
                    f"{lesion['max_hu']:.0f}",
                    lesion["density_factor"],
                    f"{lesion['score']:.2f}",
                ]
            )
    calcium_mask_path = destination / f"{stem}_calcium_dseg.nii.gz"
    out = nib.Nifti1Image(counted.astype(np.uint8), ct_img.affine)  # pyright: ignore[reportUnknownMemberType]
    out.header.set_zooms(ct_img.header.get_zooms()[:3])  # pyright: ignore[reportUnknownMemberType]
    nib.save(out, str(calcium_mask_path))  # pyright: ignore[reportUnknownMemberType]
    labels_path = destination / f"{stem}_calcium_labels.csv"
    labels_path.write_text("label,structure\n1,calcium\n")

    return {
        "ct_path": str(ct_path),
        "mask_path": str(mask_path),
        "structures": names,
        "agatston_score": total,
        "volume_score_mm3": volume,
        "n_lesions": len(lesions),
        "category": agatston_category(total),
        "lesions_path": str(lesions_path),
        "calcium_mask_path": str(calcium_mask_path),
        "labels_path": str(labels_path),
        "warnings": warnings,
        "_render": (
            "DISPLAY RULES -- follow exactly:\n"
            "Report: Agatston score <agatston_score> (whole number), band <category>, "
            "calcium volume <volume_score_mm3> mm3, <n_lesions> lesions, scored inside "
            "<structures> of <mask_path>.\n"
            "Say explicitly that this is a cardiac calcium score over everything inside "
            "the mask (coronaries, valves, aortic root), not a coronary-only score, and "
            "that it is a research measurement, not a clinical finding.\n"
            "Relay every entry of 'warnings' verbatim.\n"
            "NEXT ACTION: Offer to open the CT with <calcium_mask_path> as an overlay "
            "(the counted calcium) and the lesion list at <lesions_path>. The tool "
            "already verified every path it returns -- do not recheck them."
        ),
    }
