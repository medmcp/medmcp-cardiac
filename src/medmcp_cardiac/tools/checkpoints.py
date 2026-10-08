"""The CineMA checkpoints this stack ships, and the labels they produce.

Pure data, stdlib only: the container build loads this file on its own (before the
rest of the package is copied in) to decide which weights to bake, so the image can
never ship a different set than the tools offer. Everything here is MIT-licensed
upstream (code and weights alike); see NOTICE.
"""

from typing import Literal, TypedDict

HF_REPO_ID = "mathpluscode/CineMA"

Checkpoint = Literal["mnms2", "mnms", "acdc"]
LaxCheckpoint = Literal["mnms2"]

# Label values written to the segmentation, in label order. Upstream: RV=1, MYO=2, LV=3,
# the same on short-axis and long-axis checkpoints.
LABELS: dict[int, str] = {
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}

RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3

# The grid every short-axis checkpoint was trained on: 1 x 1 mm in-plane, 10 mm
# between slices, one 192 x 192 x 16 patch (sliding-window for anything larger).
TARGET_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 10.0)
PATCH_SIZE: tuple[int, int, int] = (192, 192, 16)

# The long-axis four-chamber checkpoint is a 2D model: 1 x 1 mm, one 256 x 256 patch.
LAX_TARGET_SPACING_MM: tuple[float, float] = (1.0, 1.0)
LAX_PATCH_SIZE: tuple[int, int] = (256, 256)


class CheckpointInfo(TypedDict):
    """One fine-tuned segmentation checkpoint."""

    weights: str
    config: str
    view: Literal["sax", "lax_4c"]
    trained_on: str


# Upstream publishes three seeds per dataset; seed 0 of each is baked. All three
# share one config.yaml per dataset.
CHECKPOINTS: dict[str, CheckpointInfo] = {
    "mnms2": {
        "weights": "finetuned/segmentation/mnms2_sax/mnms2_sax_0.safetensors",
        "config": "finetuned/segmentation/mnms2_sax/config.yaml",
        "view": "sax",
        "trained_on": (
            "M&Ms-2 (multi-centre, multi-vendor, multi-disease; 360 subjects) -- the "
            "default, trained on the most diverse data"
        ),
    },
    "mnms": {
        "weights": "finetuned/segmentation/mnms_sax/mnms_sax_0.safetensors",
        "config": "finetuned/segmentation/mnms_sax/config.yaml",
        "view": "sax",
        "trained_on": "M&Ms (multi-centre, multi-vendor; 375 subjects)",
    },
    "acdc": {
        "weights": "finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors",
        "config": "finetuned/segmentation/acdc_sax/config.yaml",
        "view": "sax",
        "trained_on": "ACDC (single centre, 150 subjects across five diagnostic groups)",
    },
}

LAX_CHECKPOINTS: dict[str, CheckpointInfo] = {
    "mnms2": {
        "weights": "finetuned/segmentation/mnms2_lax_4c/mnms2_lax_4c_0.safetensors",
        "config": "finetuned/segmentation/mnms2_lax_4c/config.yaml",
        "view": "lax_4c",
        "trained_on": "M&Ms-2 four-chamber long-axis cines (multi-centre, multi-vendor)",
    },
}

DEFAULT_CHECKPOINT: Checkpoint = "mnms2"
DEFAULT_LAX_CHECKPOINT: LaxCheckpoint = "mnms2"


def checkpoint_info(name: str, view: Literal["sax", "lax_4c"] = "sax") -> CheckpointInfo:
    """Look up a checkpoint by name for a view.

    Raises:
        ValueError: if the name is not one of the shipped checkpoints for that view.
    """
    table = CHECKPOINTS if view == "sax" else LAX_CHECKPOINTS
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"Unknown {view} checkpoint {name!r}. Available: {', '.join(table)}."
        ) from None


def weight_files() -> list[str]:
    """Every Hugging Face file the image must bake, in a stable order."""
    files: list[str] = []
    for table in (CHECKPOINTS, LAX_CHECKPOINTS):
        for info in table.values():
            for key in ("weights", "config"):
                if info[key] not in files:
                    files.append(info[key])
    return files
