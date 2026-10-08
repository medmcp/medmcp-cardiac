"""Subprocess entry point that segments one cine series with CineMA.

Inference runs out-of-process for three reasons:

1. **The stdio JSON-RPC stream must stay clean.** The vendored model code logs to
   stdout, and stdout belongs to the MCP framing. In-process, a single log line
   would corrupt the session.
2. **Start-up stays fast.** The MCP server never imports torch, so tool discovery
   answers immediately instead of racing the agent's start-up budget.
3. **A crash or OOM kills one call, not the server.**

Invoked as ``python -m medmcp_cardiac.tools._run_cinema <request> <result>``, both
JSON files. Diagnostics go to stderr, which the parent relays on failure.

What it does, per frame (and per slice for the 2D long-axis model), is what
upstream's preprocessing and inference do together (``cinema/data/*/preprocess.py``
and ``cinema/segmentation/train.py``): resample to the model's grid, clip to the
0.95-99.5 intensity percentiles, z-score, rescale to [0, 1] through the same uint8
quantisation the training data went through, pad to one patch, run sliding-window
inference with half-patch overlap averaging the softmax, then resample the class
probabilities back onto the input's own grid and take the argmax. Resampling
probabilities rather than labels keeps the contour at the input's resolution.

Excluded from pyright: torch and the vendored model code ship no type stubs.
"""

import contextlib
import json
import sys
from pathlib import Path

# Intensity percentiles clipped before normalisation, as in upstream's preprocessing.
_CLIP_PERCENTILES = (0.95, 99.5)


def _resample(volume, size):
    """Linearly resample a (C, *spatial) tensor to ``size`` (2D or 3D).

    ``align_corners=False`` samples at voxel centres, which is what upstream's
    SimpleITK resampling does (it shifts the origin by half the spacing change), and
    it makes a same-size call an exact identity.
    """
    import torch.nn.functional as functional

    mode = "trilinear" if len(size) == 3 else "bilinear"
    return functional.interpolate(volume[None], size=tuple(size), mode=mode, align_corners=False)[0]


def _normalise(frame):
    """Clip, z-score and min-max one frame exactly as upstream did for training.

    Upstream clips to percentiles, applies ``sitk.Normalize`` (zero mean, unit
    variance), rescales to [0, 1] and casts to uint8 for storage; its inference
    transform then min-max scales the stored uint8 back to [0, 1]. The uint8 round
    trip is reproduced so the model sees the quantisation it was trained on.
    """
    import torch

    lo, hi = (torch.quantile(frame.flatten(), p / 100.0).item() for p in _CLIP_PERCENTILES)
    frame = frame.clamp(lo, hi)
    std = frame.std()
    frame = (frame - frame.mean()) / std if std > 0 else frame - frame.mean()
    span = frame.max() - frame.min()
    frame = (frame - frame.min()) / span if span > 0 else torch.zeros_like(frame)
    frame = (frame * 255).to(torch.uint8).to(torch.float32)
    span = frame.max() - frame.min()
    return (frame - frame.min()) / span if span > 0 else torch.zeros_like(frame)


def _predict_probabilities(model, view, frame, patch_size, device, amp_dtype):
    """Softmax class probabilities for one frame on the model's grid.

    Pads at the end to at least one patch, then follows upstream's
    ``segmentation_forward``: a grid of patches with half-patch overlap whose softmax
    outputs are averaged. Returns ``(probabilities, n_patches)`` with the
    probabilities shaped (C, *spatial), cropped back to the frame's size.
    """
    import torch
    import torch.nn.functional as functional

    from medmcp_cardiac._cinema.transform import (
        aggregate_patches,
        get_patch_grid,
        patch_grid_sample,
    )

    size = tuple(frame.shape)
    pad = [max(0, p - s) for s, p in zip(size, patch_size, strict=True)]
    # functional.pad lists the last dimension first: (last before, last after, ...).
    pad_spec = []
    for amount in reversed(pad):
        pad_spec += [0, amount]
    image = functional.pad(frame, pad_spec)[None]  # (1, *spatial)
    starts = get_patch_grid(
        image_size=image.shape[1:],
        patch_size=patch_size,
        patch_overlap=tuple(s // 2 for s in patch_size),
    )
    patches = patch_grid_sample(image, starts, patch_size).to(device=device, dtype=torch.float32)
    probs = []
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"),
    ):
        for i in range(patches.shape[0]):
            logits = model({view: patches[i : i + 1]})[view]
            probs.append(torch.softmax(logits.to(torch.float32), dim=1)[0])
    aggregated = aggregate_patches(torch.stack(probs), starts, image.shape[1:])
    crop = (slice(None), *(slice(0, s) for s in size))
    return aggregated[crop], int(starts.shape[0])


def _keep_one_heart(labels):
    """Long-axis post-processing: one LV, and only the myocardium attached to it.

    The 2D model occasionally labels a second cavity elsewhere in the field of view
    (upstream's example notes the same); keep the largest LV component and the
    myocardium components touching it. The RV is left as predicted.
    """
    import numpy as np

    from medmcp_cardiac.tools._components import components_touching_2d, largest_component_2d
    from medmcp_cardiac.tools.checkpoints import LV_LABEL, MYO_LABEL

    cleaned = np.array(labels, copy=True)
    lv = largest_component_2d(labels == LV_LABEL)
    myo = components_touching_2d(labels == MYO_LABEL, lv)
    cleaned[(labels == LV_LABEL) & ~lv] = 0
    cleaned[(labels == MYO_LABEL) & ~myo] = 0
    return cleaned


def main() -> int:
    """Run one segmentation described by the request file. Returns a process exit code."""
    if len(sys.argv) != 3:
        print("usage: _run_cinema <request.json> <result.json>", file=sys.stderr)
        return 2
    request_path, result_path = Path(sys.argv[1]), Path(sys.argv[2])
    request = json.loads(request_path.read_text())

    # Everything the model code prints goes to stderr, so the parent can surface it
    # in an error message without it ever reaching a stdout the MCP framing owns.
    with contextlib.redirect_stdout(sys.stderr):
        return _run(request, result_path)


def _run(request: dict, result_path: Path) -> int:
    """Load, segment every frame, write the label map and the result file."""
    import nibabel as nib
    import numpy as np
    import torch

    from medmcp_cardiac._cinema.convunetr import ConvUNetR

    input_path = Path(request["input_path"])
    output = Path(request["output"])
    view = request["view"]  # "sax" (3D per frame) or "lax_4c" (2D per slice and frame)
    target_spacing = tuple(float(s) for s in request["target_spacing_mm"])
    patch_size = tuple(int(s) for s in request["patch_size"])
    n_dims = len(patch_size)

    device = torch.device(request["device"])
    amp_dtype = torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    img = nib.load(str(input_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim == 3:
        data = data[..., None]
    if data.ndim != 4:
        print(f"expected a 2D-4D volume, got shape {data.shape}", file=sys.stderr)
        return 1
    spacing = [float(z) for z in img.header.get_zooms()[:3]]
    native_size = tuple(data.shape[:n_dims])
    model_size = [
        max(1, round(n * s / t))
        for n, s, t in zip(native_size, spacing[:n_dims], target_spacing, strict=True)
    ]

    model = ConvUNetR.from_finetuned(
        repo_id=request["hf_repo_id"],
        model_filename=request["weights"],
        config_filename=request["config"],
    )
    model.eval()
    model.to(device)

    n_slices, n_frames = data.shape[2], data.shape[3]
    labels = np.zeros(data.shape, dtype=np.uint8)
    patches = 0
    # The 3D model takes a whole slice stack per frame; the 2D model one slice at a time.
    jobs = (
        [(None, t) for t in range(n_frames)]
        if n_dims == 3
        else [(z, t) for t in range(n_frames) for z in range(n_slices)]
    )
    for i, (z, t) in enumerate(jobs):
        source = data[..., t] if z is None else data[:, :, z, t]
        frame = torch.from_numpy(np.ascontiguousarray(source))
        frame = _resample(frame[None], model_size)[0]
        frame = _normalise(frame)
        probs, patches = _predict_probabilities(model, view, frame, patch_size, device, amp_dtype)
        probs = _resample(probs.cpu(), native_size)
        predicted = torch.argmax(probs, dim=0).numpy().astype(np.uint8)
        if z is None:
            labels[..., t] = predicted
        else:
            labels[:, :, z, t] = _keep_one_heart(predicted)
        if (i + 1) % max(1, len(jobs) // 10) == 0 or i + 1 == len(jobs):
            print(f"{i + 1}/{len(jobs)} done", file=sys.stderr)

    if n_frames == 1:
        labels = labels[..., 0]
    # A fresh header rather than a copy of the input's: the input may carry intensity
    # scaling (scl_slope/inter) or a cal range that must not apply to a label map.
    out_img = nib.Nifti1Image(labels, img.affine)
    out_img.set_data_dtype(np.uint8)
    out_img.header.set_zooms(img.header.get_zooms()[: labels.ndim])
    out_img.header.set_xyzt_units(*img.header.get_xyzt_units())
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output))

    result = {
        "output": str(output),
        "view": view,
        "n_frames": n_frames,
        "n_slices": n_slices,
        "native_shape": list(data.shape[:3]),
        "model_shape": model_size,
        "patches_per_frame": patches,
        "amp_dtype": str(amp_dtype).removeprefix("torch."),
    }
    result_path.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
