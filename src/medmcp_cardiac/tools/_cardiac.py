"""Shared helpers for the cardiac tools.

Deliberately free of heavy imports at module scope: the MCP server imports this at
start-up, and the agent gives a stack only a few seconds to answer "what tools do you
have" before dropping it for the whole session. torch is imported lazily inside the
functions that need it, and the model itself only ever inside the
:mod:`_run_cinema` subprocess.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

Device = Literal["auto", "cuda", "mps", "cpu"]

# A cine acquisition with fewer frames than this cannot resolve end-systole well.
_MIN_FRAMES: int = 10
# Fewer slices than this is unlikely to cover the ventricles base to apex.
_MIN_SLICES: int = 5
# Plausible SAX cine spacings. Outside these the input is probably not a SAX stack
# (or its header is wrong), and the resampling to 1 x 1 x 10 mm is doing a lot of work.
_INPLANE_MM: tuple[float, float] = (0.5, 3.0)
_SLICE_MM: tuple[float, float] = (3.0, 15.0)
# 1st-percentile intensity at or below this means Hounsfield units: a CT, not an MR.
_CT_HU_THRESHOLD: float = -300.0


def nii_stem(path: Path) -> str:
    """Return a NIfTI filename stem with both ``.nii`` and ``.nii.gz`` removed.

    Args:
        path: Path to a NIfTI file.

    Returns:
        The bare filename stem (e.g. ``sub-01_cine.nii.gz`` -> ``sub-01_cine``).
    """
    stem = path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


def base_stem(segmentation_path: Path) -> str:
    """The stem shared by a segmentation and the image it was made from.

    ``sub-01_cine_dseg.nii.gz`` -> ``sub-01_cine``, so derived files (volume CSVs,
    ED/ES exports) sit beside the image under its name rather than ``*_dseg_*``.
    """
    stem = nii_stem(segmentation_path)
    return stem[: -len("_dseg")] if stem.endswith("_dseg") else stem


def detect_devices() -> list[str]:
    """Return the compute devices torch reports as usable, best first.

    Returns:
        Some subset of ``["cuda", "mps", "cpu"]``; ``"cpu"`` is always present.
    """
    devices: list[str] = []
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and bool(mps.is_available()):
        devices.append("mps")
    devices.append("cpu")
    return devices


def resolve_device(device: Device) -> str:
    """Resolve a requested compute device to a concrete one (the shared device convention).

    ``"auto"`` selects the best available accelerator -- CUDA, then MPS, then CPU;
    an explicit device is returned unchanged. Tools run on, and report, the
    *resolved* device so an ``"auto"`` -> CPU fallback is never silent.

    Args:
        device: ``"auto"``, or an explicit ``"cuda"`` / ``"mps"`` / ``"cpu"``.

    Returns:
        A concrete device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
    """
    if device != "auto":
        return device
    available = detect_devices()
    if "cuda" in available:
        return "cuda"
    if "mps" in available:
        return "mps"
    return "cpu"


def cuda_unavailable_note() -> str:
    """Return an actionable note if a CUDA-capable torch is installed but CUDA is unavailable.

    Returns:
        Human-readable note, or an empty string when CUDA works or is absent by design.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return ""
    if torch.cuda.is_available():
        return ""
    # torch.version isn't in torch's type stubs; reach it via getattr so strict
    # pyright stays happy (the attribute is the documented way to read the build).
    cuda_ver: str | None = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_ver is None:
        return "CPU-only torch is installed, so GPU inference is not possible."
    return (
        "torch is built for CUDA but no GPU is visible -- the container may be missing "
        "'--device nvidia.com/gpu=all'. Running on CPU instead (much slower)."
    )


class VolumeInfo(TypedDict):
    """What the advisory checks need to know about an input, read from its header."""

    shape: tuple[int, ...]
    spacing_mm: tuple[float, ...]
    n_frames: int
    n_slices: int


def read_volume_info(path: Path) -> VolumeInfo:
    """Read shape and spacing from a NIfTI header (no pixel data loaded).

    Raises:
        ValueError: if the file is not a 3D or 4D volume.
    """
    import nibabel as nib

    # nib.load is annotated as returning a bare FileBasedImage, which declares
    # neither .shape nor .header.
    load_image = cast(Callable[[str], Any], nib.load)  # pyright: ignore[reportUnknownMemberType]
    img = load_image(str(path))
    shape: tuple[int, ...] = tuple(int(dim) for dim in img.shape)
    zooms: tuple[float, ...] = tuple(float(z) for z in img.header.get_zooms())
    if len(shape) not in (3, 4):
        raise ValueError(
            f"Expected a 3D or 4D NIfTI volume (x, y, slices[, frames]), got shape {shape} "
            f"for {path.name}."
        )
    n_frames = shape[3] if len(shape) == 4 else 1
    return {
        "shape": shape,
        "spacing_mm": zooms[:3],
        "n_frames": n_frames,
        "n_slices": shape[2],
    }


def _looks_like_ct(path: Path) -> bool:
    """Whether a volume's intensity range says Hounsfield units (advisory, never raises)."""
    try:
        import nibabel as nib
        import numpy as np

        load_image = cast(Callable[[str], Any], nib.load)  # pyright: ignore[reportUnknownMemberType]
        img = load_image(str(path))
        first = img.dataobj[..., 0] if len(img.shape) == 4 else img.dataobj
        sample: Any = np.asarray(first, dtype="float32")
    except Exception:
        return False
    if sample.size == 0:
        return False
    return float(np.percentile(sample, 1)) <= _CT_HU_THRESHOLD


def input_warnings(path: Path, info: VolumeInfo) -> list[str]:
    """Deterministic checks that an input is plausibly a short-axis cine stack.

    These warn and never block: the model will run on anything shaped like a volume,
    but a long-axis view, a single-frame scan or a CT produces a confident, meaningless
    result, and a header cannot actually prove what a volume is.
    """
    warnings: list[str] = []
    if info["n_frames"] == 1:
        warnings.append(
            "Input is a single 3D frame, not a cine series: the segmentation covers that "
            "frame only and no ventricular function (EF, volumes over time) can be derived."
        )
    elif info["n_frames"] < _MIN_FRAMES:
        warnings.append(
            f"Only {info['n_frames']} frames: end-systole is poorly resolved with fewer "
            f"than {_MIN_FRAMES} phases, so volumes and EF are approximate."
        )
    if info["n_slices"] < _MIN_SLICES:
        warnings.append(
            f"Only {info['n_slices']} slices: a short-axis stack usually has 8-14. Fewer "
            "means incomplete base-to-apex coverage (or a long-axis view), and volumes "
            "derived from it are underestimates."
        )
    sx, sy, sz = info["spacing_mm"]
    if not (_INPLANE_MM[0] <= sx <= _INPLANE_MM[1] and _INPLANE_MM[0] <= sy <= _INPLANE_MM[1]):
        warnings.append(
            f"In-plane spacing {sx:.2f} x {sy:.2f} mm is outside the usual cine range "
            f"({_INPLANE_MM[0]}-{_INPLANE_MM[1]} mm). Check the header before trusting "
            "volumes: every measurement scales with it."
        )
    if not (_SLICE_MM[0] <= sz <= _SLICE_MM[1]):
        warnings.append(
            f"Slice spacing {sz:.2f} mm is outside the usual short-axis range "
            f"({_SLICE_MM[0]}-{_SLICE_MM[1]} mm). The model works at 10 mm; if the "
            "header is wrong, volumes are wrong by the same factor."
        )
    if info["n_slices"] > max(info["shape"][0], info["shape"][1]):
        warnings.append(
            "The third axis is longer than the in-plane axes, which is unusual for a "
            "short-axis stack -- the axes may be in an unexpected order."
        )
    if _looks_like_ct(path):
        warnings.append(
            "Input looks like CT (Hounsfield units) but the model is trained on cine MR, "
            "so the segmentation is likely meaningless."
        )
    return warnings
