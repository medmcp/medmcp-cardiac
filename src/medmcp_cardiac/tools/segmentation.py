"""Cine short-axis cardiac MRI segmentation with CineMA.

The model (the vendored ConvUNetR under :mod:`medmcp_cardiac._cinema`) labels the
right ventricle, the left-ventricular myocardium and the left-ventricular cavity on
every frame of a short-axis cine stack. Weights are baked into the container image,
so a call never reaches the network.
"""

import csv
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

from medmcp_cardiac.tools import checkpoints
from medmcp_cardiac.tools._cardiac import (
    Device,
    cuda_unavailable_note,
    input_warnings,
    nii_stem,
    read_volume_info,
    resolve_device,
)
from medmcp_cardiac.tools.function import (
    FunctionMetrics,
    function_warnings,
    load_label_map,
    render_function_rules,
    volume_curves,
    write_function_csv,
    write_volumes_csv,
)
from medmcp_cardiac.tools.function import compute_function as derive_function

# Tail of the subprocess's stderr quoted back when a run fails. Enough to carry a
# torch OOM or a model trace; short enough not to flood the chat.
_ERROR_TAIL_LINES: int = 25

# Kept in sync with server_config()'s tool_timeout_sec and the Dockerfile label: a
# cine of 30 frames is well under a minute on a GPU, but on CPU each frame is a few
# forward passes of a 100M-parameter network and the whole series can take an hour.
SUBPROCESS_TIMEOUT_SEC: float = 3600.0


class SegmentationResult(TypedDict):
    """A completed cine segmentation run."""

    input_path: str
    segmentation_path: str
    labels_path: str
    n_frames: int
    n_slices: int
    checkpoint: str
    device: str
    ed_frame: int | None
    es_frame: int | None
    ed_image_path: str | None
    ed_segmentation_path: str | None
    es_image_path: str | None
    es_segmentation_path: str | None
    function: FunctionMetrics | None
    volumes_path: str | None
    function_path: str | None
    warnings: list[str]
    _render: str


def write_labels_csv(path: Path) -> Path:
    """Write the label-index -> structure-name lookup the workspace viewer reads."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "structure"])
        for index, name in checkpoints.LABELS.items():
            writer.writerow([index, name])
    return path


def _export_frame(
    image_path: Path, labels: Any, frame: int, destination: Path, stem: str, phase: str
) -> tuple[Path, Path]:
    """Write one phase of the cine and its label map as 3D volumes the viewer can overlay.

    The viewer shows one frame of a 4D volume at a time and cannot yet pair a 4D label
    map with it, so ED and ES are exported as ``<stem>_ed.nii.gz`` +
    ``<stem>_ed_dseg.nii.gz`` (and ``_es``), each with its own labels CSV.
    """
    import nibabel as nib
    import numpy as np

    load_image = cast(Callable[[str], Any], nib.load)  # pyright: ignore[reportUnknownMemberType]
    img = load_image(str(image_path))
    frame_data: Any = np.asarray(img.dataobj[..., frame])
    image_out = destination / f"{stem}_{phase}.nii.gz"
    nib.save(nib.Nifti1Image(frame_data, img.affine), str(image_out))  # pyright: ignore[reportUnknownMemberType]

    label_data: Any = np.asarray(labels[..., frame], dtype=np.uint8)
    label_out = destination / f"{stem}_{phase}_dseg.nii.gz"
    nib.save(nib.Nifti1Image(label_data, img.affine), str(label_out))  # pyright: ignore[reportUnknownMemberType]
    write_labels_csv(destination / f"{stem}_{phase}_labels.csv")
    return image_out, label_out


def segment_cine_sax(
    input_path: Path,
    output_dir: Path | None = None,
    checkpoint: checkpoints.Checkpoint = checkpoints.DEFAULT_CHECKPOINT,
    device: Device = "auto",
    compute_function: bool = True,
    height_cm: float | None = None,
    weight_kg: float | None = None,
) -> SegmentationResult:
    """Segment the ventricles on a short-axis cine cardiac MRI and measure function.

    Labels the right ventricle (1), the left-ventricular myocardium (2) and the
    left-ventricular cavity (3) on every frame of a short-axis (SAX) cine stack with
    CineMA, a cardiac MRI foundation model fine-tuned for segmentation. The input is
    a 4D NIfTI (x, y, slices, frames); convert DICOM with the DICOM stack first.

    Writes a single multilabel volume (``<stem>_dseg.nii.gz``, one label map per frame)
    plus a label->structure CSV, and by default also the ventricular function report:
    end-diastolic and end-systolic volumes, stroke volume and ejection fraction for both
    ventricles, and LV mass (``compute_function``). The ED and ES phases are exported
    as 3D image + label pairs (``<stem>_ed.nii.gz`` / ``<stem>_ed_dseg.nii.gz``, same
    for ``_es``) so the workspace viewer can show them as overlays.

    Three checkpoints are available, all producing the same labels: ``"mnms2"``
    (default; multi-vendor, multi-disease training data), ``"mnms"`` and ``"acdc"``.

    Args:
        input_path: Short-axis cine NIfTI (``.nii`` or ``.nii.gz``), 4D. A 3D volume
            is segmented as a single frame, without function.
        output_dir: Where to write results. Defaults to the input's own folder.
        checkpoint: Which fine-tuned weights to use.
        device: ``"auto"`` (default), ``"cuda"``, ``"mps"`` or ``"cpu"``.
        compute_function: Also derive volumes and ejection fraction from the
            segmentation (needs a 4D input).
        height_cm: Patient height, to index volumes to body surface area.
        weight_kg: Patient weight, to index volumes to body surface area.

    Returns:
        Paths to the segmentation, the label lookup and the ED/ES exports, the
        function metrics (or ``None``), the resolved device, and any warnings worth
        relaying.

    Raises:
        FileNotFoundError: if ``input_path`` does not exist.
        ValueError: if the checkpoint is unknown or the input is not a 3D/4D volume.
        RuntimeError: if the segmentation itself fails.
    """
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input volume not found: {input_path}")
    info = checkpoints.checkpoint_info(checkpoint)
    if (height_cm is None) != (weight_kg is None):
        raise ValueError("height_cm and weight_kg must be given together")

    volume = read_volume_info(input_path)
    destination = Path(output_dir).expanduser().resolve() if output_dir else input_path.parent
    destination.mkdir(parents=True, exist_ok=True)

    resolved = resolve_device(device)
    warnings: list[str] = []
    if resolved == "cpu":
        note = cuda_unavailable_note()
        if note:
            warnings.append(note)
        warnings.append(
            f"Running on CPU: a {volume['n_frames']}-frame series can take many minutes "
            "(seconds on a GPU)."
        )
    warnings.extend(input_warnings(input_path, volume))

    stem = nii_stem(input_path)
    output = destination / f"{stem}_dseg.nii.gz"

    with tempfile.TemporaryDirectory(prefix="medmcp_cardiac_") as scratch:
        scratch_dir = Path(scratch)
        request = {
            "input_path": str(input_path),
            "output": str(output),
            "hf_repo_id": checkpoints.HF_REPO_ID,
            "weights": info["weights"],
            "config": info["config"],
            "device": resolved,
            "target_spacing_mm": list(checkpoints.TARGET_SPACING_MM),
            "patch_size": list(checkpoints.PATCH_SIZE),
        }
        request_file = scratch_dir / "request.json"
        result_file = scratch_dir / "result.json"
        request_file.write_text(json.dumps(request))

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "medmcp_cardiac.tools._run_cinema",
                    str(request_file),
                    str(result_file),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=SUBPROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"CineMA timed out after {SUBPROCESS_TIMEOUT_SEC:.0f} s on {input_path.name} "
                f"(device {resolved})."
            ) from exc
        if completed.returncode != 0 or not result_file.exists():
            tail = "\n".join(completed.stderr.strip().splitlines()[-_ERROR_TAIL_LINES:])
            raise RuntimeError(f"CineMA failed on {input_path.name}:\n{tail}")

    labels_path = write_labels_csv(destination / f"{stem}_labels.csv")

    result: SegmentationResult = {
        "input_path": str(input_path),
        "segmentation_path": str(output),
        "labels_path": str(labels_path),
        "n_frames": volume["n_frames"],
        "n_slices": volume["n_slices"],
        "checkpoint": checkpoint,
        "device": resolved,
        "ed_frame": None,
        "es_frame": None,
        "ed_image_path": None,
        "ed_segmentation_path": None,
        "es_image_path": None,
        "es_segmentation_path": None,
        "function": None,
        "volumes_path": None,
        "function_path": None,
        "warnings": warnings,
        "_render": "",
    }

    if compute_function and volume["n_frames"] > 1:
        labels, voxel_volume_ml = load_label_map(output)
        curves = volume_curves(labels, voxel_volume_ml)
        metrics = derive_function(curves, height_cm, weight_kg)
        warnings.extend(function_warnings(curves, metrics))
        ed_image, ed_seg = _export_frame(
            input_path, labels, metrics["ed_frame"], destination, stem, "ed"
        )
        es_image, es_seg = _export_frame(
            input_path, labels, metrics["es_frame"], destination, stem, "es"
        )
        result["ed_frame"] = metrics["ed_frame"]
        result["es_frame"] = metrics["es_frame"]
        result["ed_image_path"] = str(ed_image)
        result["ed_segmentation_path"] = str(ed_seg)
        result["es_image_path"] = str(es_image)
        result["es_segmentation_path"] = str(es_seg)
        result["function"] = metrics
        result["volumes_path"] = str(write_volumes_csv(destination / f"{stem}_volumes.csv", curves))
        result["function_path"] = str(
            write_function_csv(destination / f"{stem}_function.csv", metrics)
        )

    result["_render"] = _render_rules(with_function=result["function"] is not None)
    return result


def _render_rules(with_function: bool) -> str:
    """Display rules for a segmentation result, with or without a function report."""
    head = (
        "DISPLAY RULES -- follow exactly:\n"
        "Report the segmentation as a compact key-value list:\n"
        "  Input:  <input_path>\n"
        "  Output: <segmentation_path> (labels: right ventricle, myocardium, left ventricle; "
        "<n_frames> frames, <n_slices> slices)\n"
        "  Model:  CineMA, checkpoint <checkpoint>\n"
        "  Device: <device>\n"
        "Substitute values from the result dict. Omit internal keys.\n"
    )
    if not with_function:
        return head + (
            "Relay every entry of 'warnings' verbatim -- they change how the result "
            "should be read.\n"
            "NEXT ACTION: Tell the user the output path and offer to open it in the viewer "
            "as an overlay. No ventricular function was computed; say why if a warning "
            "explains it. The tool already verified every path it returns -- do not "
            "recheck them."
        )
    return (
        head
        + "Then "
        + render_function_rules("function.")
        + "\nFor the viewer, offer the end-diastolic pair: open <ed_image_path> and "
        "overlay <ed_segmentation_path> (the end-systolic pair is <es_image_path> + "
        "<es_segmentation_path>)."
    )
