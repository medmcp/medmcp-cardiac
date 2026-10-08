"""Typed shims over nibabel for the tests (nibabel ships no type information)."""

from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

_nib: Any = cast(Any, nib)


def save(data: Any, affine: Any, path: Path, zooms: tuple[float, ...] | None = None) -> Path:
    """Write ``data`` as a NIfTI with ``affine`` and, optionally, explicit zooms."""
    img = _nib.Nifti1Image(np.asarray(data), affine)
    if zooms is not None:
        img.header.set_zooms(zooms)
    _nib.save(img, str(path))
    return path


def load_array(path: Path | str) -> Any:
    """The voxel array of a NIfTI file."""
    return np.asarray(_nib.load(str(path)).dataobj)


def shape(path: Path | str) -> tuple[int, ...]:
    """The shape of a NIfTI file, from its header."""
    return tuple(int(d) for d in _nib.load(str(path)).shape)


def load(path: Path | str) -> Any:
    """The nibabel image object, untyped."""
    return _nib.load(str(path))
