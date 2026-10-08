"""Tests for segment_cine_sax's validation, output layout and failure reporting.

Inference itself is never run here: it needs the weights and, realistically, a GPU.
The subprocess boundary is the seam -- faking it exercises everything this package is
responsible for (what it asks the model to do, what it writes, what it reports back)
without touching torch. The one test that does run the model is opt-in
(``MEDMCP_CARDIAC_E2E``).
"""

import csv
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medmcp_cardiac.tools import segmentation
from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL, RV_LABEL
from medmcp_cardiac.tools.segmentation import segment_cine_sax
from tests import _nifti

N_FRAMES = 12


def _synthetic_labels(shape: tuple[int, ...], n_frames: int) -> Any:
    """A beating 'heart': concentric LV / myocardium / RV blobs whose LV shrinks mid-cycle."""
    x, y, _z = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    cx, cy = shape[0] / 2, shape[1] / 2
    radius = np.hypot(x - cx, y - cy)
    labels = np.zeros((*shape, n_frames), dtype=np.uint8)
    for t in range(n_frames):
        phase = 0.5 - 0.5 * np.cos(2 * np.pi * t / n_frames)  # 0 at ED (frame 0), 1 at ES
        lv_r = 8.0 - 4.0 * phase
        frame = np.zeros(shape, dtype=np.uint8)
        frame[radius < lv_r + 3] = MYO_LABEL
        frame[radius < lv_r] = LV_LABEL
        frame[(x > cx + lv_r + 3) & (radius < lv_r + 9)] = RV_LABEL
        frame[..., 0] = 0  # one empty slice, like an apical slice past the heart
        labels[..., t] = frame
    return labels


@pytest.fixture
def cine(tmp_path: Path) -> Path:
    """A small but real 4D NIfTI, so header checks run on something parseable."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(32, 32, 8, N_FRAMES)).astype(np.int16)
    affine = np.diag([1.5, 1.5, 8.0, 1.0])
    return _nifti.save(data, affine, tmp_path / "sub-01_cine.nii.gz", zooms=(1.5, 1.5, 8.0, 0.04))


def _fake_run(captured: dict[str, Any], *, returncode: int = 0, stderr: str = "") -> Any:
    """Build a subprocess.run stand-in that plays back a CineMA run."""

    def run(cmd: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        request_file, result_file = Path(cmd[-2]), Path(cmd[-1])
        request = json.loads(request_file.read_text())
        captured["request"] = request
        if returncode == 0:
            src = _nifti.load(request["input_path"])
            shape = tuple(int(d) for d in src.shape)
            labels = _synthetic_labels(shape[:3], shape[3]) if len(shape) == 4 else None
            if labels is None:
                labels = _synthetic_labels(shape, 1)[..., 0]
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            zooms = tuple(float(z) for z in src.header.get_zooms()[: labels.ndim])
            _nifti.save(labels, src.affine, output, zooms=zooms)
            result_file.write_text(json.dumps({"output": str(output), "n_frames": shape[-1]}))
        return subprocess.CompletedProcess(list(cmd), returncode, "", stderr)

    return run


def test_missing_input_raises(tmp_path: Path) -> None:
    """A path that does not exist fails before anything is spawned."""
    with pytest.raises(FileNotFoundError):
        segment_cine_sax(tmp_path / "nope.nii.gz")


def test_unknown_checkpoint_is_refused(cine: Path) -> None:
    """A checkpoint name that is not shipped fails fast with the available names."""
    with pytest.raises(ValueError, match="mnms2"):
        segment_cine_sax(cine, checkpoint="ukbb")  # type: ignore[arg-type]


def test_height_without_weight_is_refused(cine: Path) -> None:
    """Body-surface-area indexing needs both measurements."""
    with pytest.raises(ValueError, match="together"):
        segment_cine_sax(cine, height_cm=170.0)


def test_two_dimensional_input_is_refused(tmp_path: Path) -> None:
    """A 2D image cannot be a short-axis stack."""
    path = _nifti.save(np.zeros((32, 32), dtype=np.int16), np.eye(4), tmp_path / "slice.nii.gz")
    with pytest.raises(ValueError, match="3D or 4D"):
        segment_cine_sax(path)


def test_successful_run_writes_segmentation_exports_and_function(
    cine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default run yields the 4D label map, its CSV, the ED/ES pairs and the report."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", _fake_run(captured))
    out = tmp_path / "derivatives"

    result = segment_cine_sax(cine, output_dir=out, device="cpu")

    assert result["segmentation_path"] == str(out / "sub-01_cine_dseg.nii.gz")
    assert result["n_frames"] == N_FRAMES and result["n_slices"] == 8
    assert result["checkpoint"] == "mnms2" and result["device"] == "cpu"

    rows = list(csv.reader(Path(result["labels_path"]).read_text().splitlines()))
    assert rows == [
        ["label", "structure"],
        ["1", "right_ventricle"],
        ["2", "myocardium"],
        ["3", "left_ventricle"],
    ]

    function = result["function"]
    assert function is not None
    assert function["ed_frame"] == 0
    assert function["es_frame"] == N_FRAMES // 2
    assert 0 < function["lv_ef_percent"] < 100
    assert function["lv_edv_ml"] > function["lv_esv_ml"]
    assert function["lv_mass_g"] > 0
    assert function["bsa_m2"] is None

    for key in ("ed_image_path", "ed_segmentation_path", "es_image_path", "es_segmentation_path"):
        path = Path(result[key] or "")
        assert path.is_file(), key
        assert len(_nifti.shape(path)) == 3, f"{key} should be a 3D export"
    assert (out / "sub-01_cine_ed_labels.csv").is_file(), "the viewer needs a labels CSV per dseg"
    ed_labels = _nifti.load_array(result["ed_segmentation_path"] or "")
    assert int(ed_labels.max()) == LV_LABEL

    assert Path(result["volumes_path"] or "").is_file()
    assert Path(result["function_path"] or "").is_file()
    assert "LVEF" in result["_render"]


def test_request_payload_names_the_checkpoint_and_grid(
    cine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subprocess is told exactly which weights, device and grid to use."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(subprocess, "run", _fake_run(captured))

    segment_cine_sax(cine, checkpoint="acdc", device="cpu", compute_function=False)

    request = captured["request"]
    assert request["weights"].endswith("acdc_sax_0.safetensors")
    assert request["config"].endswith("acdc_sax/config.yaml")
    assert request["hf_repo_id"] == "mathpluscode/CineMA"
    assert request["device"] == "cpu"
    assert request["target_spacing_mm"] == [1.0, 1.0, 10.0]
    assert request["patch_size"] == [192, 192, 16]


def test_function_can_be_skipped(cine: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``compute_function=False`` writes the segmentation and nothing derived from it."""
    monkeypatch.setattr(subprocess, "run", _fake_run({}))

    result = segment_cine_sax(cine, device="cpu", compute_function=False)

    assert result["function"] is None
    assert result["volumes_path"] is None and result["ed_image_path"] is None
    assert "No ventricular function" in result["_render"]


def test_bsa_indexing_when_height_and_weight_given(
    cine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Height and weight produce indexed volumes alongside the absolute ones."""
    monkeypatch.setattr(subprocess, "run", _fake_run({}))

    result = segment_cine_sax(cine, device="cpu", height_cm=180.0, weight_kg=75.0)

    function = result["function"]
    assert function is not None
    assert function["bsa_m2"] == pytest.approx((180 * 75 / 3600) ** 0.5)
    assert function["lv_edv_index_ml_m2"] == pytest.approx(
        function["lv_edv_ml"] / (function["bsa_m2"] or 1)
    )


def test_single_frame_input_segments_without_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3D volume is segmented as one frame; function is impossible and is said so."""
    path = _nifti.save(np.zeros((32, 32, 8), dtype=np.int16), np.eye(4), tmp_path / "frame.nii.gz")
    monkeypatch.setattr(subprocess, "run", _fake_run({}))

    result = segment_cine_sax(path, device="cpu")

    assert result["n_frames"] == 1
    assert result["function"] is None
    assert any("single 3D frame" in w for w in result["warnings"])


def test_cpu_run_warns_about_runtime(cine: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Falling back to CPU is never silent -- it changes runtime by orders of magnitude."""
    monkeypatch.setattr(subprocess, "run", _fake_run({}))

    result = segment_cine_sax(cine, device="cpu")

    assert any("CPU" in warning for warning in result["warnings"])


def test_implausible_geometry_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Few frames, few slices and odd spacings each produce a warning, never an error."""
    data = np.zeros((20, 20, 3, 4), dtype=np.int16)
    path = _nifti.save(data, np.diag([5.0, 5.0, 1.0, 1.0]), tmp_path / "odd.nii.gz")
    monkeypatch.setattr(subprocess, "run", _fake_run({}))

    result = segment_cine_sax(path, device="cpu", compute_function=False)

    text = "\n".join(result["warnings"])
    assert "Only 4 frames" in text
    assert "Only 3 slices" in text
    assert "In-plane spacing" in text
    assert "Slice spacing" in text


def test_failure_surfaces_the_subprocess_stderr(
    cine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run reports why, quoting the tail of the child's stderr."""
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({}, returncode=1, stderr="torch.OutOfMemoryError: CUDA out of memory"),
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        segment_cine_sax(cine, device="cpu")


def test_timeouts_stay_in_sync() -> None:
    """The subprocess timeout, server_config and the image label must agree."""
    from medmcp_cardiac.server import server_config

    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
    label = json.loads(dockerfile.split("LABEL org.medmcp.stack='", 1)[1].split("'", 1)[0])
    assert server_config()["tool_timeout_sec"] == segmentation.SUBPROCESS_TIMEOUT_SEC
    assert label["tool_timeout_sec"] == segmentation.SUBPROCESS_TIMEOUT_SEC


def test_render_hint_is_present() -> None:
    """The _render contract drives how the agent reports the result."""
    assert "_render" in segmentation.SegmentationResult.__annotations__
