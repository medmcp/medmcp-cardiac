"""Tests for segment_cine_lax4c's output layout and area metrics (inference faked)."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL, RV_LABEL
from medmcp_cardiac.tools.segmentation import area_curves, lax_metrics, segment_cine_lax4c
from tests import _nifti

N_FRAMES = 8


def _lax_labels(shape: tuple[int, ...], n_planes: int, n_frames: int) -> Any:
    """A 2D 'four-chamber' per plane: LV blob shrinking mid-cycle, RV beside it."""
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    labels = np.zeros((*shape, n_planes, n_frames), dtype=np.uint8)
    for plane in range(n_planes):
        scale = 1.0 - 0.3 * plane  # plane 0 has the largest LV
        for t in range(n_frames):
            phase = 0.5 - 0.5 * np.cos(2 * np.pi * t / n_frames)
            lv_r = scale * (10.0 - 4.0 * phase)
            frame = np.zeros(shape, dtype=np.uint8)
            r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
            frame[r < lv_r + 3] = MYO_LABEL
            frame[r < lv_r] = LV_LABEL
            frame[(np.hypot(x - shape[0] / 2 + 20, y - shape[1] / 2) < 8)] = RV_LABEL
            labels[:, :, plane, t] = frame
    return labels


@pytest.fixture
def lax_cine(tmp_path: Path) -> Path:
    """A two-plane long-axis cine NIfTI."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 255, size=(48, 48, 2, N_FRAMES)).astype(np.int16)
    affine: Any = np.diag([1.5, 1.5, 8.0, 1.0])
    return _nifti.save(data, affine, tmp_path / "sub-01_4ch.nii.gz", zooms=(1.5, 1.5, 8.0, 0.04))


def _fake_run(captured: dict[str, Any]) -> Any:
    def run(cmd: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        request_file, result_file = Path(cmd[-2]), Path(cmd[-1])
        request = json.loads(request_file.read_text())
        captured["request"] = request
        src = _nifti.load(request["input_path"])
        shape = tuple(int(d) for d in src.shape)
        labels = _lax_labels(shape[:2], shape[2], shape[3])
        output = Path(request["output"])
        zooms = tuple(float(z) for z in src.header.get_zooms()[:4])
        _nifti.save(labels, src.affine, output, zooms=zooms)
        result_file.write_text(json.dumps({"output": str(output)}))
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    return run


def test_area_metrics_pick_the_largest_plane() -> None:
    """ED is the frame of maximal LV area; the plane with the biggest LV is used."""
    labels = _lax_labels((48, 48), 2, N_FRAMES)
    curves = area_curves(labels, pixel_area_mm2=2.25)
    metrics = lax_metrics(curves)
    assert metrics["plane"] == 0
    assert metrics["ed_frame"] == 0 and metrics["es_frame"] == N_FRAMES // 2
    assert metrics["lv_eda_mm2"] > metrics["lv_esa_mm2"]
    assert 0 < metrics["lv_fac_percent"] < 100
    assert metrics["rv_fac_percent"] == pytest.approx(0.0)  # the RV blob is static


def test_tool_writes_segmentation_exports_and_areas(
    lax_cine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The long-axis run asks for the 2D model and writes the same family of files."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", _fake_run(captured))
    out = tmp_path / "derived"

    result = segment_cine_lax4c(lax_cine, output_dir=out, device="cpu")

    request = captured["request"]
    assert request["view"] == "lax_4c"
    assert request["weights"].endswith("mnms2_lax_4c_0.safetensors")
    assert request["patch_size"] == [256, 256]
    assert request["target_spacing_mm"] == [1.0, 1.0]

    assert result["n_planes"] == 2 and result["n_frames"] == N_FRAMES
    assert result["segmentation_path"] == str(out / "sub-01_4ch_dseg.nii.gz")
    metrics = result["metrics"]
    assert metrics is not None and metrics["plane"] == 0
    assert Path(result["areas_path"] or "").read_text().startswith("plane,frame,")
    assert Path(result["ed_segmentation_path"] or "").is_file()
    assert any("2 planes" in w for w in result["warnings"])
    assert "not volumes" in result["_render"]


def test_metrics_can_be_skipped(lax_cine: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``compute_metrics=False`` writes the label map only."""
    monkeypatch.setattr(subprocess, "run", _fake_run({}))
    result = segment_cine_lax4c(lax_cine, device="cpu", compute_metrics=False)
    assert result["metrics"] is None and result["areas_path"] is None


def test_unknown_lax_checkpoint(lax_cine: Path) -> None:
    """Only the long-axis checkpoints are accepted here."""
    with pytest.raises(ValueError, match="lax_4c"):
        segment_cine_lax4c(lax_cine, checkpoint="acdc")  # type: ignore[arg-type]
