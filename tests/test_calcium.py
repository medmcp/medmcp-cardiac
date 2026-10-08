"""Tests for the Agatston calcium score on a synthetic CT."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medmcp_cardiac.tools._components import (
    components_touching_2d,
    label_components_2d,
    largest_component_2d,
)
from medmcp_cardiac.tools.calcium import (
    agatston_category,
    cardiac_calcium_score,
    density_factor,
    score_volume,
)
from tests import _nifti


def test_density_factor_bands() -> None:
    """The four Agatston density bands."""
    assert [density_factor(v) for v in (130, 199, 200, 299, 300, 399, 400, 1000)] == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_categories() -> None:
    """The conventional score bands."""
    assert agatston_category(0).startswith("0")
    assert "minimal" in agatston_category(5)
    assert "mild" in agatston_category(50)
    assert "moderate" in agatston_category(400)
    assert "severe" in agatston_category(401)


def test_components_2d() -> None:
    """Two blobs, one diagonal-touching pair counted as one component."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = mask[2, 2] = True  # diagonal neighbours: one component
    mask[6, 6] = True
    labels, n = label_components_2d(mask)
    assert n == 2
    assert labels[1, 1] == labels[2, 2] != labels[6, 6]
    largest = largest_component_2d(mask)
    assert largest.sum() == 2
    anchor = np.zeros((8, 8), dtype=bool)
    anchor[5, 5] = True
    assert components_touching_2d(mask, anchor).sum() == 1


def _ct_with_lesions() -> tuple[Any, Any]:
    """A soft-tissue CT with two lesions: 3x3 at 250 HU, 2x2 at 450 HU, one at 10 HU."""
    ct = np.full((40, 40, 6), 40.0, dtype=np.float32)
    ct[:, :, 0] = -1000.0  # some air, so the volume reads as CT
    ct[10:13, 10:13, 2] = 250.0
    ct[20:22, 20:22, 3] = 450.0
    ct[30:32, 30:32, 4] = 100.0  # below threshold
    ct[5:8, 30:33, 2] = 500.0  # outside the mask
    mask = np.zeros((40, 40, 6), dtype=np.uint8)
    mask[8:26, 8:26, :] = 1  # "heart"
    mask[0:4, 0:4, :] = 2  # "aorta"
    return ct, mask


def test_score_volume_on_the_phantom() -> None:
    """Scores add up by area x factor x (slice / 3 mm)."""
    ct, mask = _ct_with_lesions()
    lesions, counted, volume = score_volume(ct, mask == 1, (1.0, 1.0, 3.0))
    scores = sorted(lesion["score"] for lesion in lesions)
    assert scores == pytest.approx([4.0 * 4, 9.0 * 2])
    assert counted.sum() == 13
    assert volume == pytest.approx(13 * 3.0)


def test_tool_end_to_end(tmp_path: Path) -> None:
    """Structure names resolve through the mask's labels CSV; outputs land beside the CT."""
    ct, mask = _ct_with_lesions()
    affine: Any = np.diag([1.0, 1.0, 3.0, 1.0])
    ct_path = _nifti.save(ct, affine, tmp_path / "sub-01_ct.nii.gz")
    mask_path = _nifti.save(mask, affine, tmp_path / "sub-01_ct_dseg.nii.gz")
    (tmp_path / "sub-01_ct_labels.csv").write_text("label,structure\n1,heart\n2,aorta\n")

    result = cardiac_calcium_score(ct_path, mask_path)

    assert result["structures"] == ["heart"]
    assert result["agatston_score"] == pytest.approx(34.0)
    assert result["n_lesions"] == 2
    assert "mild" in result["category"]
    assert Path(result["lesions_path"]).name == "sub-01_ct_calcium_lesions.csv"
    assert _nifti.load_array(result["calcium_mask_path"]).sum() == 13
    assert Path(result["labels_path"]).read_text() == "label,structure\n1,calcium\n"
    assert result["warnings"] == []

    with pytest.raises(ValueError, match="Not in"):
        cardiac_calcium_score(ct_path, mask_path, structures=["coronary"])
    assert cardiac_calcium_score(ct_path, mask_path, label_values=[2])["agatston_score"] == 0


def test_contrast_and_non_ct_warnings(tmp_path: Path) -> None:
    """A bright heart warns about contrast; a non-HU volume warns about the threshold."""
    ct, mask = _ct_with_lesions()
    ct[mask == 1] = 300.0  # enhanced blood pool
    affine: Any = np.diag([1.0, 1.0, 3.0, 1.0])
    ct_path = _nifti.save(ct, affine, tmp_path / "ct.nii.gz")
    mask_path = _nifti.save(mask, affine, tmp_path / "ct_dseg.nii.gz")
    result = cardiac_calcium_score(ct_path, mask_path, label_values=[1])
    assert any("contrast" in w for w in result["warnings"])

    mr = np.abs(ct)  # no negative values: not Hounsfield
    mr_path = _nifti.save(mr, affine, tmp_path / "mr.nii.gz")
    result = cardiac_calcium_score(mr_path, mask_path, label_values=[1])
    assert any("Hounsfield" in w for w in result["warnings"])


def test_grid_mismatch_is_refused(tmp_path: Path) -> None:
    """A mask on another grid cannot be used."""
    ct, mask = _ct_with_lesions()
    ct_path = _nifti.save(ct, np.eye(4), tmp_path / "ct.nii.gz")
    mask_path = _nifti.save(mask[:20], np.eye(4), tmp_path / "ct_dseg.nii.gz")
    with pytest.raises(ValueError, match="grid"):
        cardiac_calcium_score(ct_path, mask_path, label_values=[1])
