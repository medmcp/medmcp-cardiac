"""Ventricular volumes and function from a cine short-axis segmentation.

Pure numpy over the label map `segment_cine_sax` writes (or any 4D label map using
the same label values): no model, no GPU. Volumes are voxel counts times the voxel
volume from the NIfTI header, so they are only as right as that header.
"""

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

from medmcp_cardiac.tools._cardiac import base_stem
from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL, RV_LABEL

# Myocardial tissue density, the convention for converting myocardial volume to mass.
MYOCARDIUM_DENSITY_G_PER_ML: float = 1.05


class FunctionMetrics(TypedDict):
    """Ventricular function derived from a volume-time curve."""

    n_frames: int
    ed_frame: int
    es_frame: int
    lv_edv_ml: float
    lv_esv_ml: float
    lv_sv_ml: float
    lv_ef_percent: float
    rv_edv_ml: float
    rv_esv_ml: float
    rv_sv_ml: float
    rv_ef_percent: float
    lv_mass_g: float
    bsa_m2: float | None
    lv_edv_index_ml_m2: float | None
    lv_esv_index_ml_m2: float | None
    rv_edv_index_ml_m2: float | None
    rv_esv_index_ml_m2: float | None
    lv_mass_index_g_m2: float | None


class FunctionResult(TypedDict):
    """A completed function analysis."""

    segmentation_path: str
    function: FunctionMetrics
    volumes_path: str
    function_path: str
    warnings: list[str]
    _render: str


class VolumeCurves(TypedDict):
    """Per-frame volumes in ml, one list per structure, indexed by frame."""

    right_ventricle: list[float]
    myocardium: list[float]
    left_ventricle: list[float]


def body_surface_area_m2(height_cm: float, weight_kg: float) -> float:
    """Mosteller body surface area."""
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("height_cm and weight_kg must be positive")
    return math.sqrt(height_cm * weight_kg / 3600.0)


def volume_curves(labels: Any, voxel_volume_ml: float) -> VolumeCurves:
    """Count label voxels per frame and scale to millilitres.

    Args:
        labels: an integer array shaped (x, y, z, frames) -- or (x, y, z) for one frame.
        voxel_volume_ml: the volume of one voxel in ml.
    """
    import numpy as np

    arr: Any = np.asarray(labels)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"expected a 3D or 4D label map, got shape {arr.shape}")

    def curve(value: int) -> list[float]:
        counts: Any = np.count_nonzero(arr == value, axis=(0, 1, 2))
        return [float(c) * voxel_volume_ml for c in counts]

    return {
        "right_ventricle": curve(RV_LABEL),
        "myocardium": curve(MYO_LABEL),
        "left_ventricle": curve(LV_LABEL),
    }


def compute_function(
    curves: VolumeCurves,
    height_cm: float | None = None,
    weight_kg: float | None = None,
) -> FunctionMetrics:
    """Derive ED/ES phases and the standard function metrics from volume curves.

    End-diastole is the frame of maximal LV volume and end-systole the frame of
    minimal LV volume; both ventricles are measured at those two LV-defined phases,
    the usual convention. LV mass is the myocardial volume at ED times 1.05 g/ml.
    With height and weight, volumes and mass are also indexed to body surface area.

    Raises:
        ValueError: with fewer than two frames, or when the LV was not found at all.
    """
    lv = curves["left_ventricle"]
    rv = curves["right_ventricle"]
    myo = curves["myocardium"]
    n_frames = len(lv)
    if n_frames < 2:
        raise ValueError("ventricular function needs a cine series with at least two frames")
    if max(lv) <= 0:
        raise ValueError("no left-ventricle voxels in any frame; the segmentation is empty")

    ed = max(range(n_frames), key=lambda i: lv[i])
    es = min(range(n_frames), key=lambda i: lv[i])
    lv_edv, lv_esv = lv[ed], lv[es]
    rv_edv, rv_esv = rv[ed], rv[es]
    lv_mass = myo[ed] * MYOCARDIUM_DENSITY_G_PER_ML

    bsa: float | None = None
    if height_cm is not None and weight_kg is not None:
        bsa = body_surface_area_m2(height_cm, weight_kg)

    def indexed(value: float) -> float | None:
        return value / bsa if bsa else None

    return {
        "n_frames": n_frames,
        "ed_frame": ed,
        "es_frame": es,
        "lv_edv_ml": lv_edv,
        "lv_esv_ml": lv_esv,
        "lv_sv_ml": lv_edv - lv_esv,
        "lv_ef_percent": 100.0 * (lv_edv - lv_esv) / lv_edv,
        "rv_edv_ml": rv_edv,
        "rv_esv_ml": rv_esv,
        "rv_sv_ml": rv_edv - rv_esv,
        "rv_ef_percent": 100.0 * (rv_edv - rv_esv) / rv_edv if rv_edv > 0 else 0.0,
        "lv_mass_g": lv_mass,
        "bsa_m2": bsa,
        "lv_edv_index_ml_m2": indexed(lv_edv),
        "lv_esv_index_ml_m2": indexed(lv_esv),
        "rv_edv_index_ml_m2": indexed(rv_edv),
        "rv_esv_index_ml_m2": indexed(rv_esv),
        "lv_mass_index_g_m2": indexed(lv_mass),
    }


def function_warnings(curves: VolumeCurves, metrics: FunctionMetrics) -> list[str]:
    """Sanity checks on a volume curve that a reader should hear about."""
    warnings: list[str] = []
    empty = [i for i, v in enumerate(curves["left_ventricle"]) if v <= 0]
    if empty:
        warnings.append(
            f"The left ventricle was not found in {len(empty)} of {metrics['n_frames']} "
            "frames; ED/ES and the volumes may be wrong. Check those frames in the viewer."
        )
    if metrics["rv_edv_ml"] <= 0:
        warnings.append("The right ventricle was not found at end-diastole; RV metrics are 0.")
    if metrics["ed_frame"] == metrics["es_frame"]:
        warnings.append("ED and ES fell on the same frame: the LV volume does not change.")
    if metrics["lv_ef_percent"] < 10.0 or metrics["lv_ef_percent"] > 85.0:
        warnings.append(
            f"LVEF of {metrics['lv_ef_percent']:.0f}% is implausible; the segmentation "
            "likely failed on some frames."
        )
    return warnings


def write_volumes_csv(path: Path, curves: VolumeCurves) -> Path:
    """One row per frame with the three structure volumes in ml."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "left_ventricle_ml", "myocardium_ml", "right_ventricle_ml"])
        for i, (lv, myo, rv) in enumerate(
            zip(
                curves["left_ventricle"],
                curves["myocardium"],
                curves["right_ventricle"],
                strict=True,
            )
        ):
            writer.writerow([i, f"{lv:.2f}", f"{myo:.2f}", f"{rv:.2f}"])
    return path


def write_function_csv(path: Path, metrics: FunctionMetrics) -> Path:
    """The function metrics as ``metric,value,unit`` rows."""
    units = {
        "n_frames": "",
        "ed_frame": "frame index",
        "es_frame": "frame index",
        "lv_edv_ml": "ml",
        "lv_esv_ml": "ml",
        "lv_sv_ml": "ml",
        "lv_ef_percent": "%",
        "rv_edv_ml": "ml",
        "rv_esv_ml": "ml",
        "rv_sv_ml": "ml",
        "rv_ef_percent": "%",
        "lv_mass_g": "g",
        "bsa_m2": "m2",
        "lv_edv_index_ml_m2": "ml/m2",
        "lv_esv_index_ml_m2": "ml/m2",
        "rv_edv_index_ml_m2": "ml/m2",
        "rv_esv_index_ml_m2": "ml/m2",
        "lv_mass_index_g_m2": "g/m2",
    }
    values = cast(dict[str, float | int | None], metrics)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        for key, unit in units.items():
            value = values[key]
            if value is None:
                continue
            text = f"{value:.2f}" if isinstance(value, float) else str(value)
            writer.writerow([key, text, unit])
    return path


def load_label_map(path: Path) -> tuple[Any, float]:
    """Load a label map and the volume of one of its voxels in ml."""
    import nibabel as nib
    import numpy as np

    load_image = cast(Callable[[str], Any], nib.load)  # pyright: ignore[reportUnknownMemberType]
    img = load_image(str(path))
    labels: Any = np.asarray(img.dataobj)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    voxel_volume_ml = float(np.prod(zooms)) / 1000.0
    return labels, voxel_volume_ml


def cardiac_function(
    segmentation_path: Path,
    output_dir: Path | None = None,
    height_cm: float | None = None,
    weight_kg: float | None = None,
) -> FunctionResult:
    """Compute ventricular volumes and function from a cine short-axis segmentation.

    Takes the 4D label map ``segment_cine_sax`` writes (labels 1 = right ventricle,
    2 = myocardium, 3 = left ventricle, one volume per cardiac phase) and derives the
    standard function report: end-diastolic and end-systolic volumes, stroke volume
    and ejection fraction for both ventricles, and LV mass. End-diastole is the frame
    of maximal LV volume, end-systole the frame of minimal LV volume. With
    ``height_cm`` and ``weight_kg`` the volumes and mass are also indexed to body
    surface area (Mosteller).

    ``segment_cine_sax`` already runs this when ``compute_function=True``; call it
    directly to re-analyse an existing segmentation, add height and weight, or analyse
    a label map produced elsewhere with the same label values.

    Args:
        segmentation_path: 4D label map (``*_dseg.nii.gz``) with one frame per phase.
        output_dir: Where to write the CSVs. Defaults to the segmentation's folder.
        height_cm: Patient height, for body-surface-area indexing.
        weight_kg: Patient weight, for body-surface-area indexing.

    Returns:
        The metrics, the paths of a per-frame volume CSV and a metric CSV, and
        warnings about frames where the segmentation looks unreliable.

    Raises:
        FileNotFoundError: if ``segmentation_path`` does not exist.
        ValueError: if the file is not a cine (4D) label map, or contains no ventricle.
    """
    segmentation_path = Path(segmentation_path).expanduser().resolve()
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Segmentation not found: {segmentation_path}")
    if (height_cm is None) != (weight_kg is None):
        raise ValueError("height_cm and weight_kg must be given together")

    destination = (
        Path(output_dir).expanduser().resolve() if output_dir else segmentation_path.parent
    )
    destination.mkdir(parents=True, exist_ok=True)

    labels, voxel_volume_ml = load_label_map(segmentation_path)
    if labels.ndim != 4:
        raise ValueError(
            f"{segmentation_path.name} is {labels.ndim}D; ventricular function needs a 4D "
            "cine label map with one volume per phase."
        )
    curves = volume_curves(labels, voxel_volume_ml)
    metrics = compute_function(curves, height_cm, weight_kg)
    warnings = function_warnings(curves, metrics)

    stem = base_stem(segmentation_path)
    volumes_path = write_volumes_csv(destination / f"{stem}_volumes.csv", curves)
    function_path = write_function_csv(destination / f"{stem}_function.csv", metrics)

    return {
        "segmentation_path": str(segmentation_path),
        "function": metrics,
        "volumes_path": str(volumes_path),
        "function_path": str(function_path),
        "warnings": warnings,
        "_render": render_function_rules("function."),
    }


def render_function_rules(prefix: str) -> str:
    """The display rules for a function result; ``prefix`` locates the metrics dict."""
    return (
        "DISPLAY RULES -- follow exactly:\n"
        "Report the ventricular function as a compact table with one row per metric, "
        "values rounded to whole numbers, using the keys under "
        f"'{prefix.rstrip('.')}':\n"
        "  LV EDV / ESV / SV (ml), LVEF (%), RV EDV / ESV / SV (ml), RVEF (%), "
        "LV mass (g); add the indexed values (ml/m2, g/m2) only when they are not null.\n"
        f"State which frames were taken as ED and ES (<{prefix}ed_frame>, "
        f"<{prefix}es_frame>, zero-based).\n"
        "Relay every entry of 'warnings' verbatim -- they change how the numbers "
        "should be read.\n"
        "Say that these are research measurements from an automatic segmentation, not "
        "a clinical finding, and do not compare them with reference ranges unless "
        "the user asks.\n"
        "NEXT ACTION: Offer the per-frame volume curve at <volumes_path> and the "
        "metric CSV at <function_path>. The tool already verified every path it "
        "returns -- do not recheck them."
    )
