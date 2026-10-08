"""The checkpoint catalogue is what both the tools and the image build read.

The Dockerfile loads ``checkpoints.py`` on its own to decide which weights to bake,
so the module must stay importable in isolation and the Dockerfile must keep reading
it the same way; otherwise the image silently ships a different set than the tools
offer.
"""

import importlib.util
import re
from pathlib import Path

from medmcp_cardiac.tools import checkpoints

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
MODULE = ROOT / "src" / "medmcp_cardiac" / "tools" / "checkpoints.py"


def test_default_checkpoint_is_shipped() -> None:
    """The default must resolve."""
    assert checkpoints.DEFAULT_CHECKPOINT in checkpoints.CHECKPOINTS


def test_every_checkpoint_has_weights_and_config_on_the_hub() -> None:
    """Paths are Hugging Face repo-relative and name a safetensors file plus a YAML."""
    for name, info in checkpoints.CHECKPOINTS.items():
        assert info["weights"].endswith(".safetensors"), name
        assert info["config"].endswith(".yaml"), name
        assert not info["weights"].startswith("/"), name
        assert info["trained_on"].strip(), name


def test_weight_files_cover_every_checkpoint_once() -> None:
    """The bake list has no duplicates and nothing missing."""
    files = checkpoints.weight_files()
    assert len(files) == len(set(files))
    for info in checkpoints.CHECKPOINTS.values():
        assert info["weights"] in files and info["config"] in files


def test_labels_are_the_upstream_convention() -> None:
    """RV=1, MYO=2, LV=3 is what every CineMA segmentation checkpoint produces."""
    assert checkpoints.LABELS == {1: "right_ventricle", 2: "myocardium", 3: "left_ventricle"}
    assert (checkpoints.RV_LABEL, checkpoints.MYO_LABEL, checkpoints.LV_LABEL) == (1, 2, 3)


def test_module_loads_standalone_as_the_dockerfile_does() -> None:
    """The build imports the file by path, before the package exists in the image."""
    spec = importlib.util.spec_from_file_location("catalog_probe", MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.weight_files() == checkpoints.weight_files()


def test_dockerfile_bakes_from_this_module() -> None:
    """The Dockerfile must probe checkpoints.py and call weight_files(), not a hand list."""
    text = DOCKERFILE.read_text()
    assert "tools/checkpoints.py" in text
    assert "weight_files()" in text
    assert re.search(r"HF_HUB_OFFLINE=1", text), "the container must never fetch at run time"
    assert checkpoints.HF_REPO_ID in text
