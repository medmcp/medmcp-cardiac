"""Tests for the ventricular function arithmetic and the cardiac_function tool."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL, RV_LABEL
from medmcp_cardiac.tools.function import (
    MYOCARDIUM_DENSITY_G_PER_ML,
    VolumeCurves,
    body_surface_area_m2,
    cardiac_function,
    compute_function,
    function_warnings,
    volume_curves,
)
from tests import _nifti


def _curves(
    lv: list[float], rv: list[float] | None = None, myo: list[float] | None = None
) -> VolumeCurves:
    n = len(lv)
    return {
        "left_ventricle": lv,
        "right_ventricle": rv if rv is not None else [v * 1.1 for v in lv],
        "myocardium": myo if myo is not None else [100.0] * n,
    }


def test_volume_curves_count_each_label_per_frame() -> None:
    """Each structure's curve is its voxel count per frame times the voxel volume."""
    labels: Any = np.zeros((4, 4, 2, 3), dtype=np.uint8)
    labels[0, 0, 0, 0] = LV_LABEL
    labels[0, 1, 0, 0] = LV_LABEL
    labels[1, 1, 1, 1] = MYO_LABEL
    labels[2, :, :, 2] = RV_LABEL  # 4 * 2 = 8 voxels

    curves = volume_curves(labels, voxel_volume_ml=0.5)

    assert curves["left_ventricle"] == [1.0, 0.0, 0.0]
    assert curves["myocardium"] == [0.0, 0.5, 0.0]
    assert curves["right_ventricle"] == [0.0, 0.0, 4.0]


def test_ed_es_from_lv_extremes_and_standard_metrics() -> None:
    """ED = max LV, ES = min LV; EF, SV and mass follow the textbook definitions."""
    lv = [150.0, 120.0, 80.0, 60.0, 90.0, 140.0]
    rv = [160.0, 130.0, 100.0, 80.0, 110.0, 150.0]
    myo = [110.0, 120.0, 130.0, 135.0, 125.0, 112.0]

    m = compute_function(_curves(lv, rv, myo))

    assert (m["ed_frame"], m["es_frame"]) == (0, 3)
    assert m["lv_edv_ml"] == 150.0 and m["lv_esv_ml"] == 60.0
    assert m["lv_sv_ml"] == 90.0
    assert m["lv_ef_percent"] == pytest.approx(60.0)
    assert m["rv_edv_ml"] == 160.0 and m["rv_esv_ml"] == 80.0
    assert m["rv_ef_percent"] == pytest.approx(50.0)
    assert m["lv_mass_g"] == pytest.approx(110.0 * MYOCARDIUM_DENSITY_G_PER_ML)
    assert m["bsa_m2"] is None and m["lv_edv_index_ml_m2"] is None


def test_body_surface_area_indexing() -> None:
    """Mosteller BSA, and every indexed value is the absolute one divided by it."""
    assert body_surface_area_m2(180.0, 80.0) == pytest.approx(2.0)
    m = compute_function(_curves([150.0, 60.0]), height_cm=180.0, weight_kg=80.0)
    assert m["bsa_m2"] == pytest.approx(2.0)
    assert m["lv_edv_index_ml_m2"] == pytest.approx(75.0)
    assert m["lv_esv_index_ml_m2"] == pytest.approx(30.0)
    assert m["lv_mass_index_g_m2"] == pytest.approx(100.0 * MYOCARDIUM_DENSITY_G_PER_ML / 2)


def test_bsa_rejects_nonsense() -> None:
    """A zero or negative height/weight is an input error, not a NaN."""
    with pytest.raises(ValueError):
        body_surface_area_m2(0.0, 80.0)


def test_single_frame_cannot_give_function() -> None:
    """Function needs the cardiac cycle."""
    with pytest.raises(ValueError, match="at least two frames"):
        compute_function(_curves([100.0]))


def test_empty_segmentation_is_an_error() -> None:
    """No LV anywhere means the segmentation failed outright."""
    with pytest.raises(ValueError, match="empty"):
        compute_function(_curves([0.0, 0.0, 0.0]))


def test_warnings_flag_missing_frames_and_implausible_ef() -> None:
    """A frame without an LV, a missing RV and an implausible EF are each called out."""
    curves = _curves([150.0, 0.0, 10.0], rv=[0.0, 0.0, 0.0])
    m = compute_function(curves)
    text = "\n".join(function_warnings(curves, m))
    assert "not found in 1 of 3 frames" in text
    assert "right ventricle was not found" in text
    assert "implausible" in text


def test_no_warnings_on_a_clean_curve() -> None:
    """A plausible curve produces no noise."""
    curves = _curves([150.0, 120.0, 80.0, 60.0, 90.0, 140.0])
    assert function_warnings(curves, compute_function(curves)) == []


@pytest.fixture
def cine_segmentation(tmp_path: Path) -> Path:
    """A 4D label map with a 2 x 2 x 5 mm voxel, LV shrinking over four frames."""
    labels = np.zeros((10, 10, 4, 4), dtype=np.uint8)
    for t, n in enumerate((6, 4, 2, 5)):
        labels[:n, :n, :, t] = LV_LABEL
        labels[:n, n : n + 2, :, t] = MYO_LABEL
        labels[n + 2 :, :3, :, t] = RV_LABEL
    return _nifti.save(labels, np.diag([2.0, 2.0, 5.0, 1.0]), tmp_path / "sub-01_cine_dseg.nii.gz")


def test_cardiac_function_tool_writes_csvs_beside_the_segmentation(
    cine_segmentation: Path,
) -> None:
    """The tool reads the header's voxel size and names outputs after the image stem."""
    result = cardiac_function(cine_segmentation)

    voxel_ml = 2.0 * 2.0 * 5.0 / 1000.0
    m = result["function"]
    assert m["ed_frame"] == 0 and m["es_frame"] == 2
    assert m["lv_edv_ml"] == pytest.approx(6 * 6 * 4 * voxel_ml)
    assert m["lv_esv_ml"] == pytest.approx(2 * 2 * 4 * voxel_ml)

    assert result["volumes_path"] == str(cine_segmentation.parent / "sub-01_cine_volumes.csv")
    assert result["function_path"] == str(cine_segmentation.parent / "sub-01_cine_function.csv")
    rows = list(csv.DictReader(Path(result["volumes_path"]).read_text().splitlines()))
    assert len(rows) == 4
    assert rows[0].keys() == {"frame", "left_ventricle_ml", "myocardium_ml", "right_ventricle_ml"}
    function_rows = csv.DictReader(Path(result["function_path"]).read_text().splitlines())
    metrics = {r["metric"]: r for r in function_rows}
    assert metrics["lv_ef_percent"]["unit"] == "%"
    assert "bsa_m2" not in metrics, "unset indexed values are left out of the CSV"
    assert "research" in result["_render"]


def test_cardiac_function_rejects_a_3d_label_map(tmp_path: Path) -> None:
    """A single frame has no cycle to measure."""
    path = _nifti.save(
        np.zeros((4, 4, 4), dtype=np.uint8), np.eye(4), tmp_path / "frame_dseg.nii.gz"
    )
    with pytest.raises(ValueError, match="4D"):
        cardiac_function(path)


def test_cardiac_function_missing_file(tmp_path: Path) -> None:
    """A path that does not exist is reported as such."""
    with pytest.raises(FileNotFoundError):
        cardiac_function(tmp_path / "missing_dseg.nii.gz")
