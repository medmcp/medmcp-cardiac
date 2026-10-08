"""Tests for the AHA 16-segment wall analysis on synthetic hearts.

A synthetic short-axis stack with a known wall thickness, a thicker lateral wall and a
right ventricle attached on one side checks that the sectors land where the anatomy
says they should, and that thickness and thickening come out as built.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL, RV_LABEL
from medmcp_cardiac.tools.regional import (
    SEGMENT_NAMES,
    regional_wall_analysis,
    segment_thickness,
)
from tests import _nifti

SHAPE = (96, 96, 9)


def _heart(inner_r: float, wall: float, lateral_extra: float = 0.0, rv: bool = True) -> Any:
    """One 3D frame: LV ring around the centre, RV crescent on the -x side.

    The affine used in the tests maps +y of the array to anterior (RAS +y), so the
    RV sits on the septal (-x) side and ``lateral_extra`` thickens the +x wall.
    """
    x, y, _z = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    cx, cy = SHAPE[0] / 2, SHAPE[1] / 2
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    labels = np.zeros(SHAPE, dtype=np.uint8)
    outer = inner_r + wall + lateral_extra * (np.cos(angle) > 0)
    labels[(r >= inner_r) & (r < outer)] = MYO_LABEL
    labels[r < inner_r] = LV_LABEL
    if rv:
        rv_centre = cx - (inner_r + wall) - 10
        rv_r = np.hypot(x - rv_centre, dy)
        labels[(rv_r < 16) & (labels == 0) & (dx < 0)] = RV_LABEL
    labels[:, :, 0] = 0  # a basal slice above the heart
    labels[:, :, 8] = 0  # an apical cap slice without a cavity
    return labels


AFFINE: Any = np.diag([1.0, 1.0, 10.0, 1.0])


def test_segments_are_oriented_on_the_rv_and_thickness_matches_the_phantom() -> None:
    """Uniform 6 px wall, the lateral half 6 px thicker: septal 6 mm, lateral 12 mm."""
    labels = _heart(inner_r=18.0, wall=6.0, lateral_extra=6.0)
    warnings: list[str] = []
    thickness, n_slices = segment_thickness(labels, AFFINE, (1.0, 1.0), warnings)

    assert n_slices == 7
    assert set(thickness) == set(range(1, 17))
    # Septal segments (anteroseptal 2/8, inferoseptal 3/9, apical septal 14) ~ 6 mm.
    for segment in (2, 3, 8, 9, 14):
        assert thickness[segment] == pytest.approx(6.0, abs=1.0), SEGMENT_NAMES[segment]
    # Lateral segments (inferolateral 5/11, anterolateral 6/12, apical lateral 16) thicker.
    for segment in (5, 6, 11, 12, 16):
        assert thickness[segment] == pytest.approx(12.0, abs=1.5), SEGMENT_NAMES[segment]
    assert not warnings


def test_anterior_is_decided_from_world_coordinates() -> None:
    """Flipping the y axis in the affine swaps which insertion point is anterior."""
    labels = _heart(inner_r=18.0, wall=6.0)
    # Make the +y side of the array thicker ("anterior" under the default affine).
    x, y, _z = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    dx, dy = x - SHAPE[0] / 2, y - SHAPE[1] / 2
    r = np.hypot(dx, dy)
    labels[(r >= 24) & (r < 28) & (dy > 0) & (labels == 0)] = MYO_LABEL

    up, _ = segment_thickness(labels, AFFINE, (1.0, 1.0), [])
    flipped: Any = np.diag([1.0, -1.0, 10.0, 1.0])
    down, _ = segment_thickness(labels, flipped, (1.0, 1.0), [])

    assert up[1] > up[4], "thick +y wall is anterior under the default affine"
    assert down[4] > down[1], "and inferior once +y maps to posterior"


def test_apical_slices_without_rv_inherit_the_orientation() -> None:
    """The RV is removed from the apical third; segments 13-16 are still produced."""
    labels = _heart(inner_r=18.0, wall=6.0, lateral_extra=6.0)
    labels[:, :, 6:][labels[:, :, 6:] == RV_LABEL] = 0
    thickness, _ = segment_thickness(labels, AFFINE, (1.0, 1.0), [])
    assert {13, 14, 15, 16} <= set(thickness)
    assert thickness[14] < thickness[16]


def test_no_rv_anywhere_is_an_error() -> None:
    """Without insertion points the sectors cannot be oriented."""
    labels = _heart(inner_r=18.0, wall=6.0, rv=False)
    with pytest.raises(ValueError, match="insertion"):
        segment_thickness(labels, AFFINE, (1.0, 1.0), [])


@pytest.fixture
def cine_segmentation(tmp_path: Path) -> Path:
    """Two-frame cine: ED thin wall, ES thicker wall (and smaller cavity)."""
    ed = _heart(inner_r=18.0, wall=6.0)
    es = _heart(inner_r=12.0, wall=9.0)
    labels = np.stack([ed, es], axis=-1)
    return _nifti.save(labels, AFFINE, tmp_path / "sub-01_cine_dseg.nii.gz")


def test_tool_reports_thickening_and_writes_the_table(cine_segmentation: Path) -> None:
    """ES wall 9 mm over ED 6 mm is 50 % thickening in every segment."""
    result = regional_wall_analysis(cine_segmentation)

    assert (result["ed_frame"], result["es_frame"]) == (0, 1)
    assert len(result["segments"]) == 16
    for row in result["segments"]:
        assert row["name"] == SEGMENT_NAMES[row["segment"]]
        assert row["thickness_ed_mm"] == pytest.approx(6.0, abs=1.0)
        assert row["thickening_percent"] == pytest.approx(50.0, abs=15.0)
    assert result["mean_thickening_percent"] == pytest.approx(50.0, abs=10.0)

    table = Path(result["segments_path"]).read_text().splitlines()
    assert table[0] == "segment,name,region,thickness_ed_mm,thickness_es_mm,thickening_percent"
    assert len(table) == 17
    assert "research" in result["_render"]


def test_single_frame_gives_thickness_only(tmp_path: Path) -> None:
    """A 3D label map has no ES, so thickening is null and a warning says so."""
    path = _nifti.save(_heart(18.0, 6.0), AFFINE, tmp_path / "ed_dseg.nii.gz")
    result = regional_wall_analysis(path)
    assert result["es_frame"] is None
    assert all(row["thickening_percent"] is None for row in result["segments"])
    assert any("Single-frame" in w for w in result["warnings"])


def test_frame_override_out_of_range(cine_segmentation: Path) -> None:
    """An ED/ES frame outside the series is an input error."""
    with pytest.raises(ValueError, match="outside"):
        regional_wall_analysis(cine_segmentation, ed_frame=5)
