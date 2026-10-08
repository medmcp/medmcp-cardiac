"""Regional left-ventricular wall analysis on the AHA 16-segment model.

Pure numpy over the short-axis label map `segment_cine_sax` writes. Each slice is
divided into angular sectors around the LV cavity centre, oriented on the right
ventricular insertion points (where the RV wall meets the LV myocardium), and the
slices are grouped base to apex into basal, mid and apical thirds -- the standard
American Heart Association segmentation (Cerqueira et al., Circulation 2002).
The apical cap (segment 17) needs a long-axis view and is not reported.

Wall thickness per sector is myocardial area divided by the arc length at the
mid-wall radius, which is robust to the jagged edges of a voxel mask. Systolic
thickening is the ED-to-ES change in that thickness.
"""

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np

from medmcp_cardiac.tools._cardiac import base_stem
from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL, RV_LABEL
from medmcp_cardiac.tools.function import compute_function, volume_curves

SEGMENT_NAMES: dict[int, str] = {
    1: "basal anterior",
    2: "basal anteroseptal",
    3: "basal inferoseptal",
    4: "basal inferior",
    5: "basal inferolateral",
    6: "basal anterolateral",
    7: "mid anterior",
    8: "mid anteroseptal",
    9: "mid inferoseptal",
    10: "mid inferior",
    11: "mid inferolateral",
    12: "mid anterolateral",
    13: "apical anterior",
    14: "apical septal",
    15: "apical inferior",
    16: "apical lateral",
}

# A slice takes part when it holds at least this many cavity and myocardium voxels;
# below that it is the apical cap or a basal slice through the outflow tract.
_MIN_VOXELS: int = 20

# Sector order moving from the anterior RV insertion point through the septum, for
# the six-segment rings (offset into the ring: 0 = basal, 6 = mid).
_RING_ORDER: tuple[int, ...] = (2, 3, 4, 5, 6, 1)
# Apical ring, starting at the septal sector (centred on the septum).
_APICAL_ORDER: tuple[int, ...] = (14, 15, 16, 13)


class SegmentMetrics(TypedDict):
    """One AHA segment."""

    segment: int
    name: str
    region: Literal["basal", "mid", "apical"]
    thickness_ed_mm: float | None
    thickness_es_mm: float | None
    thickening_percent: float | None


class RegionalResult(TypedDict):
    """A completed regional analysis."""

    segmentation_path: str
    ed_frame: int
    es_frame: int | None
    n_slices_used: int
    segments: list[SegmentMetrics]
    mean_thickness_ed_mm: float | None
    mean_thickening_percent: float | None
    segments_path: str
    warnings: list[str]
    _render: str


class _SliceFrame(TypedDict):
    """Orientation of one slice: where the sectors start and which way they run."""

    theta_anterior: float  # angle (radians, voxel space) of the anterior insertion point
    sign: float  # +1 / -1: direction from the anterior insertion through the septum
    phi_inferior: float  # sector angle (degrees) of the inferior insertion point


def _wrap(angle: Any) -> Any:
    """Wrap angles to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _orientation(
    myo: Any, rv: Any, centre: tuple[float, float], affine: Any, z: int
) -> _SliceFrame | None:
    """Find the two RV insertion points on one slice and orient the sectors on them.

    Returns ``None`` when the RV is absent or does not touch the myocardium there.
    """
    if not rv.any():
        return None
    # Myocardium voxels with an RV voxel as a 4-neighbour.
    touching: Any = np.zeros(myo.shape, dtype=bool)
    touching[1:, :] |= rv[:-1, :]
    touching[:-1, :] |= rv[1:, :]
    touching[:, 1:] |= rv[:, :-1]
    touching[:, :-1] |= rv[:, 1:]
    boundary: Any = myo & touching
    if not boundary.any():
        return None
    xs, ys = np.nonzero(boundary)
    rvx, rvy = np.nonzero(rv)
    theta_rv = math.atan2(float(rvy.mean()) - centre[1], float(rvx.mean()) - centre[0])
    relative: Any = _wrap(np.arctan2(ys - centre[1], xs - centre[0]) - theta_rv)
    ends = (int(np.argmin(relative)), int(np.argmax(relative)))

    def world_anterior(index: int) -> float:
        point: Any = affine @ np.array([xs[index], ys[index], z, 1.0])
        return float(point[1])  # RAS: +y is anterior

    anterior, inferior = (
        (ends[0], ends[1])
        if world_anterior(ends[0]) >= world_anterior(ends[1])
        else (ends[1], ends[0])
    )
    theta_anterior = math.atan2(ys[anterior] - centre[1], xs[anterior] - centre[0])
    theta_inferior = math.atan2(ys[inferior] - centre[1], xs[inferior] - centre[0])
    sign = 1.0 if _wrap(theta_rv - theta_anterior) > 0 else -1.0
    phi_inferior = (sign * _wrap(theta_inferior - theta_anterior)) % (2 * math.pi)
    return {
        "theta_anterior": theta_anterior,
        "sign": sign,
        "phi_inferior": math.degrees(phi_inferior),
    }


def _sector_thickness(
    myo: Any,
    centre: tuple[float, float],
    frame: _SliceFrame,
    spacing: tuple[float, float],
    apical: bool,
) -> dict[int, float]:
    """Wall thickness (mm) per AHA ring position for one slice.

    Keys are the segment numbers of the basal ring (1-6) or apical ring (13-16);
    the caller shifts basal numbers to the mid ring.
    """
    xs, ys = np.nonzero(myo)
    if xs.size == 0:
        return {}
    theta: Any = np.arctan2(ys - centre[1], xs - centre[0])
    phi: Any = np.degrees((frame["sign"] * _wrap(theta - frame["theta_anterior"])) % (2 * math.pi))
    radius_mm: Any = np.hypot((xs - centre[0]) * spacing[0], (ys - centre[1]) * spacing[1])
    voxel_area = spacing[0] * spacing[1]

    if apical:
        width = 90.0
        shifted: Any = (phi - frame["phi_inferior"] / 2.0 + 45.0) % 360.0
        order = _APICAL_ORDER
    else:
        width = 60.0
        shifted = phi
        order = _RING_ORDER
    sector: Any = np.minimum((shifted // width).astype(int), len(order) - 1)

    out: dict[int, float] = {}
    for k, segment in enumerate(order):
        inside = sector == k
        count = int(np.count_nonzero(inside))
        if count == 0:
            continue
        arc_mm = math.radians(width) * float(radius_mm[inside].mean())
        if arc_mm <= 0:
            continue
        out[segment] = count * voxel_area / arc_mm
    return out


def _slice_order(labels: Any) -> list[int]:
    """Usable slice indices ordered base to apex (the base has the larger cavity)."""
    usable = [
        z
        for z in range(labels.shape[2])
        if np.count_nonzero(labels[:, :, z] == LV_LABEL) >= _MIN_VOXELS
        and np.count_nonzero(labels[:, :, z] == MYO_LABEL) >= _MIN_VOXELS
    ]
    if len(usable) < 2:
        return usable
    first = int(np.count_nonzero(labels[:, :, usable[0]] == LV_LABEL))
    last = int(np.count_nonzero(labels[:, :, usable[-1]] == LV_LABEL))
    return usable if first >= last else usable[::-1]


def _regions(n: int) -> list[Literal["basal", "mid", "apical"]]:
    """Assign ``n`` base-to-apex slices to thirds; a remainder goes to basal, then mid."""
    base, extra = divmod(n, 3)
    regions: list[Literal["basal", "mid", "apical"]] = []
    for _ in range(base + (1 if extra > 0 else 0)):
        regions.append("basal")
    for _ in range(base + (1 if extra > 1 else 0)):
        regions.append("mid")
    for _ in range(base):
        regions.append("apical")
    return regions


def usable_slices(labels: Any) -> list[int]:
    """Slice indices holding both an LV cavity and myocardium, ordered base to apex."""
    return _slice_order(labels)


def segment_thickness(
    labels: Any,
    affine: Any,
    spacing: tuple[float, float],
    warnings: list[str],
    order: list[int] | None = None,
) -> tuple[dict[int, float], int]:
    """Mean wall thickness per AHA segment (1-16) for one 3D label map.

    ``order`` fixes which slices take part and in what base-to-apex order; when ED and
    ES are compared it must be the same for both, or a slice that drops out at ES
    shifts every segment of the apical third. Defaults to the slices usable in this
    frame.

    Returns the per-segment thickness in mm (segments with no myocardium are absent)
    and the number of slices that contributed.
    """
    order = _slice_order(labels) if order is None else list(order)
    if not order:
        raise ValueError("no slice holds both an LV cavity and myocardium")
    regions = _regions(len(order))

    # Orientation per slice; apical slices often have no RV, so they inherit from the
    # nearest slice toward the base that does.
    frames: list[_SliceFrame | None] = []
    centres: list[tuple[float, float]] = []
    for z in order:
        lv: Any = labels[:, :, z] == LV_LABEL
        myo: Any = labels[:, :, z] == MYO_LABEL
        rv: Any = labels[:, :, z] == RV_LABEL
        xs, ys = np.nonzero(lv if lv.any() else myo)
        centre = (float(xs.mean()), float(ys.mean()))
        centres.append(centre)
        frames.append(_orientation(myo, rv, centre, affine, z))
    if all(frame is None for frame in frames):
        raise ValueError(
            "no slice shows the right ventricle touching the myocardium, so the segments "
            "cannot be oriented on the RV insertion points"
        )
    oriented: list[_SliceFrame] = []
    last: _SliceFrame | None = None
    for frame in frames:
        if frame is not None:
            last = frame
        oriented.append(last if last is not None else next(f for f in frames if f is not None))

    per_segment: dict[int, list[float]] = {}
    for z, region, frame, centre in zip(order, regions, oriented, centres, strict=True):
        myo = labels[:, :, z] == MYO_LABEL
        values = _sector_thickness(myo, centre, frame, spacing, apical=region == "apical")
        for segment, thickness in values.items():
            number = segment + 6 if region == "mid" else segment
            per_segment.setdefault(number, []).append(thickness)
    if any(f["phi_inferior"] < 30.0 or f["phi_inferior"] > 210.0 for f in oriented):
        warnings.append(
            "The RV insertion points span an unusual angle on some slices; the septal "
            "sectors may be misplaced. Check the segmentation where the RV meets the LV."
        )
    return {k: float(np.mean(v)) for k, v in per_segment.items()}, len(order)


def _write_segments_csv(path: Path, segments: list[SegmentMetrics]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment",
                "name",
                "region",
                "thickness_ed_mm",
                "thickness_es_mm",
                "thickening_percent",
            ]
        )
        for row in segments:
            writer.writerow(
                [
                    row["segment"],
                    row["name"],
                    row["region"],
                    "" if row["thickness_ed_mm"] is None else f"{row['thickness_ed_mm']:.2f}",
                    "" if row["thickness_es_mm"] is None else f"{row['thickness_es_mm']:.2f}",
                    "" if row["thickening_percent"] is None else f"{row['thickening_percent']:.1f}",
                ]
            )
    return path


def regional_wall_analysis(
    segmentation_path: Path,
    output_dir: Path | None = None,
    ed_frame: int | None = None,
    es_frame: int | None = None,
) -> RegionalResult:
    """Regional LV wall thickness and systolic thickening on the AHA 16-segment model.

    Takes the short-axis label map ``segment_cine_sax`` writes and reports, for each
    of the 16 American Heart Association segments (basal, mid and apical rings; the
    apical cap needs a long-axis view and is omitted): end-diastolic and end-systolic
    wall thickness in mm and systolic thickening in percent. Segments are oriented on
    the right ventricular insertion points and the slices grouped base to apex. The
    per-segment table is also written as a CSV, the data a bullseye plot is drawn from.

    End-diastole and end-systole default to the frames of maximal and minimal LV
    volume. A 3D (single-frame) label map yields thickness only.

    Args:
        segmentation_path: 4D (or 3D) label map with labels 1 RV, 2 myocardium, 3 LV.
        output_dir: Where to write the CSV. Defaults to the segmentation's folder.
        ed_frame: Override the end-diastolic frame (zero-based).
        es_frame: Override the end-systolic frame (zero-based).

    Returns:
        The per-segment table, means, the frames used, the CSV path, and warnings.

    Raises:
        FileNotFoundError: if ``segmentation_path`` does not exist.
        ValueError: if the label map is not 3D/4D, holds no usable slice, or the RV
            never touches the myocardium (no insertion points to orient on).
    """
    segmentation_path = Path(segmentation_path).expanduser().resolve()
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Segmentation not found: {segmentation_path}")
    destination = (
        Path(output_dir).expanduser().resolve() if output_dir else segmentation_path.parent
    )
    destination.mkdir(parents=True, exist_ok=True)

    labels, affine, spacing = _load(segmentation_path)
    warnings: list[str] = []
    if labels.ndim == 3:
        labels = labels[..., None]
    n_frames = int(labels.shape[3])

    if n_frames > 1 and (ed_frame is None or es_frame is None):
        voxel_ml = float(spacing[0] * spacing[1] * spacing[2]) / 1000.0
        metrics = compute_function(volume_curves(labels, voxel_ml))
        ed = metrics["ed_frame"] if ed_frame is None else ed_frame
        es: int | None = metrics["es_frame"] if es_frame is None else es_frame
    else:
        ed = 0 if ed_frame is None else ed_frame
        es = es_frame
    for name, frame in (("ed_frame", ed), ("es_frame", es)):
        if frame is not None and not 0 <= frame < n_frames:
            raise ValueError(f"{name}={frame} is outside the {n_frames} frames")
    if n_frames == 1:
        warnings.append("Single-frame label map: wall thickness only, no systolic thickening.")

    inplane = (float(spacing[0]), float(spacing[1]))
    order = usable_slices(labels[..., ed])
    if es is not None:
        at_es = set(usable_slices(labels[..., es]))
        dropped = [z for z in order if z not in at_es]
        if dropped:
            warnings.append(
                f"Slice(s) {dropped} hold no LV cavity at end-systole and are left out of "
                "both phases so the segments compare like with like."
            )
        order = [z for z in order if z in at_es]
    thickness_ed, n_slices = segment_thickness(labels[..., ed], affine, inplane, warnings, order)
    thickness_es: dict[int, float] = {}
    if es is not None:
        thickness_es, _ = segment_thickness(labels[..., es], affine, inplane, warnings, order)
    if n_slices < 3:
        warnings.append(
            f"Only {n_slices} usable slice(s): the basal, mid and apical rings cannot all "
            "be populated, so some segments are missing."
        )

    segments: list[SegmentMetrics] = []
    for number in range(1, 17):
        region: Literal["basal", "mid", "apical"] = (
            "basal" if number <= 6 else "mid" if number <= 12 else "apical"
        )
        t_ed = thickness_ed.get(number)
        t_es = thickness_es.get(number)
        thickening = None
        if t_ed is not None and t_es is not None and t_ed > 0:
            thickening = 100.0 * (t_es - t_ed) / t_ed
        segments.append(
            {
                "segment": number,
                "name": SEGMENT_NAMES[number],
                "region": region,
                "thickness_ed_mm": t_ed,
                "thickness_es_mm": t_es,
                "thickening_percent": thickening,
            }
        )
    missing = [s["segment"] for s in segments if s["thickness_ed_mm"] is None]
    if missing:
        warnings.append(
            f"No myocardium found for segment(s) {missing}; they are blank in the table."
        )

    ed_values = [s["thickness_ed_mm"] for s in segments if s["thickness_ed_mm"] is not None]
    thickening_values = [
        s["thickening_percent"] for s in segments if s["thickening_percent"] is not None
    ]
    stem = base_stem(segmentation_path)
    table = _write_segments_csv(destination / f"{stem}_segments.csv", segments)

    return {
        "segmentation_path": str(segmentation_path),
        "ed_frame": ed,
        "es_frame": es,
        "n_slices_used": n_slices,
        "segments": segments,
        "mean_thickness_ed_mm": float(np.mean(ed_values)) if ed_values else None,
        "mean_thickening_percent": (
            float(np.mean(thickening_values)) if thickening_values else None
        ),
        "segments_path": str(table),
        "warnings": warnings,
        "_render": (
            "DISPLAY RULES -- follow exactly:\n"
            "Report the regional analysis as a 16-row table: segment number, name, "
            "ED thickness (mm, one decimal), ES thickness (mm) and thickening (%, whole "
            "number), from 'segments'. Then the means. Name the ED/ES frames used "
            "(<ed_frame>, <es_frame>, zero-based).\n"
            "Relay every entry of 'warnings' verbatim.\n"
            "Say these are research measurements from an automatic segmentation, not a "
            "clinical finding; do not grade segments as normal or abnormal unless asked.\n"
            "NEXT ACTION: Offer the per-segment CSV at <segments_path>. "
            "The tool already verified every path it returns -- do not recheck them."
        ),
    }


def _load(path: Path) -> tuple[Any, Any, tuple[float, float, float]]:
    """Label map, affine and spatial spacing of a NIfTI label file."""
    import nibabel as nib

    load_image = cast(Callable[[str], Any], nib.load)  # pyright: ignore[reportUnknownMemberType]
    img = load_image(str(path))
    labels: Any = np.asarray(img.dataobj)
    if labels.ndim not in (3, 4):
        raise ValueError(f"{path.name} is {labels.ndim}D; expected a 3D or 4D label map")
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    affine: Any = np.asarray(img.affine, dtype=float)
    return labels, affine, (zooms[0], zooms[1], zooms[2])
