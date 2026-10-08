# medmcp-cardiac

Cine cardiac MRI analysis for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio that an LLM agent can invoke to segment the ventricles on short-axis cine series and to measure ventricular volumes, ejection fraction and LV mass. Wraps [CineMA](https://github.com/mathpluscode/CineMA).

<p align="center">
  <a href="https://medmcp.ai"><b>medmcp.ai</b></a> ·
  <a href="https://github.com/medmcp/medmcp">Main repository</a>
</p>

> [!NOTE]
> **This repository is for developers** who build, extend, or run the cardiac stack from source. **If you just want to use MedMCP, you don't need this repo** — install the MedMCP app and add this stack through the workspace UI (one-click install). See [medmcp.ai](https://medmcp.ai) or the [main repository](https://github.com/medmcp/medmcp) to get started.

> [!WARNING]
> MedMCP and its ecosystem are research software under active development and are
> **not licensed for clinical use**. CineMA is not a medical device and its output is
> an estimate, not a clinical finding.

---

## Tool inventory

| Tool name | Description | Inputs | Outputs |
|---|---|---|---|
| `segment_cine_sax` | Segment RV, LV myocardium and LV cavity on every frame of a short-axis cine, and derive ventricular function | `input_path: Path`, `output_dir: Path?`, `checkpoint: "mnms2"\|"mnms"\|"acdc"`, `device: "auto"\|"cuda"\|"mps"\|"cpu"`, `compute_function: bool`, `height_cm: float?`, `weight_kg: float?`, `heart_rate_bpm: float?` | 4D multilabel `*_dseg.nii.gz`, a label→structure CSV, ED/ES image + label pairs for the viewer, the function metrics, a per-frame volume CSV, the resolved device, and warnings |
| `segment_cine_lax4c` | Segment the same structures on a four-chamber long-axis cine (2D model) and measure fractional area change | `input_path: Path`, `output_dir: Path?`, `checkpoint: "mnms2"`, `device`, `compute_metrics: bool` | 4D `*_dseg.nii.gz`, label CSV, ED/ES pairs, LV/RV areas and fractional area change on the plane with the largest LV, a per-frame area CSV, warnings |
| `cardiac_function` | Volumes, stroke volume, ejection fraction, LV mass, and with height/weight/heart rate the indexed values and cardiac output, from an existing cine segmentation | `segmentation_path: Path`, `output_dir: Path?`, `height_cm: float?`, `weight_kg: float?`, `heart_rate_bpm: float?` | The function metrics, a per-frame volume CSV, a metric CSV, and warnings |
| `regional_wall_analysis` | AHA 16-segment end-diastolic and end-systolic wall thickness and systolic thickening (the data of a bullseye) | `segmentation_path: Path`, `output_dir: Path?`, `ed_frame: int?`, `es_frame: int?` | The 16-segment table, means, `*_segments.csv`, warnings |
| `cardiac_calcium_score` | Agatston calcium score on a non-contrast CT inside a heart mask (e.g. TotalSegmentator's `heart`) | `ct_path: Path`, `mask_path: Path`, `structures: list[str]?`, `label_values: list[int]?`, `output_dir: Path?` | Agatston score and band, calcium volume, lesion list CSV, a calcium mask with its label CSV, warnings |

Labels are `1` right ventricle, `2` myocardium, `3` left ventricle. End-diastole is the
frame of maximal LV volume and end-systole the frame of minimal LV volume; both
ventricles are measured at those phases. LV mass is myocardial volume at ED × 1.05 g/ml.
Segments follow the AHA 16-segment model oriented on the RV insertion points; the apical
cap (17) is not reported. The calcium score is a *cardiac* score over everything inside
the mask -- coronary-only scoring would need the licence-gated coronary model.

## Checkpoints

`segment_cine_sax` and `segment_cine_lax4c` take a `checkpoint`. All are CineMA fine-tuned
segmentation models (seed 0 of the three upstream publishes) producing the same labels.

| Checkpoint | View | Trained on |
|---|---|---|
| `mnms2` | short-axis (3D) | M&Ms-2 — multi-centre, multi-vendor, multi-disease (360 subjects). The default. |
| `mnms` | short-axis (3D) | M&Ms — multi-centre, multi-vendor (375 subjects) |
| `acdc` | short-axis (3D) | ACDC — single centre, 150 subjects across five diagnostic groups |
| `mnms2` | four-chamber long-axis (2D) | M&Ms-2 four-chamber cines. The only long-axis checkpoint. |

Inputs are resampled to the grid the models were trained on (1 × 1 × 10 mm short-axis,
1 × 1 mm long-axis) and run through sliding-window inference; the result is resampled
back onto the input's grid.

## Skill inventory

Skills are SKILL.md files the agent loads on demand to follow multi-step workflows. They are bundled under `src/medmcp_cardiac/skills/` and discovered automatically via `server_config()`.

| Skill name | Description |
|---|---|
| `cardiac-function` | Workflow from a cine series to a function report and regional table, the four-chamber view, and calcium scoring on CT: converting DICOM first, picking the right series, reading ED/ES off the volume curve, offering the ED/ES overlays, and the gotchas (short-axis only, header-dependent volumes, areas are not volumes, cardiac not coronary calcium, research-only numbers). |

---

### Bundled tools

| Tool / weights | Used by | Source | License |
|---|---|---|---|
| CineMA — ConvUNetR model code | `segment_cine_sax` | [upstream](https://github.com/mathpluscode/CineMA), vendored under `src/medmcp_cardiac/_cinema/` (see `scripts/vendor-cinema.sh`) | [MIT](https://github.com/mathpluscode/CineMA/blob/main/LICENSE) |
| CineMA — fine-tuned segmentation weights (short-axis `mnms2`, `mnms`, `acdc`; long-axis `mnms2`) | `segment_cine_sax`, `segment_cine_lax4c` | [Hugging Face `mathpluscode/CineMA`](https://huggingface.co/mathpluscode/CineMA), baked into the image | [MIT](https://huggingface.co/mathpluscode/CineMA) |
| timm — `Mlp`, `SwiGLU`, `DropPath`, `LayerScale` layers | model code | [upstream](https://github.com/huggingface/pytorch-image-models), six layers copied into `_cinema/_timm.py` | [Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) |

### Citation

Results produced with this stack should cite the underlying work, not this package:

- **CineMA** — Fu Y, Yi W, Manisty C, Bhuva AN, Treibel TA, Moon JC, Clarkson MJ,
  Davies RH, Hu Y. CineMA: a foundation model for cine cardiac MRI.
  *Communications Medicine* (2026).
  [doi:10.1038/s43856-026-01636-0](https://doi.org/10.1038/s43856-026-01636-0)

The fine-tuned weights derive from the ACDC, M&Ms and M&Ms-2 challenge datasets, which
carry their own citation requirements; see the upstream README.

Full third-party attribution belongs in [`NOTICE`](NOTICE).

### Hardware requirements

- `segment_cine_sax`: CUDA GPU recommended. A 30-frame short-axis cine takes well
  under a minute on a modern GPU; on CPU each frame is a few forward passes of a
  100M-parameter network and a series can take an hour. The tool reports the
  resolved device and warns when it falls back to CPU.
- `segment_cine_lax4c`: as above; a single four-chamber plane takes seconds on a GPU.
- `cardiac_function`, `regional_wall_analysis`, `cardiac_calcium_score`: pure numpy —
  no GPU, no model load.
- Disk: the image is dominated by the CUDA/PyTorch stack; the four checkpoints add
  about 1.9 GB.
- **Runs fully offline.** The weights are baked into the image and Hugging Face Hub
  access is disabled at run time.

---

## Development

### Develop in the dev container (recommended)

This repo ships a dev container (`.devcontainer/`) with the full toolchain
(Python 3.12 + uv, `just`, git, Docker CLI). It derives from the shared
`medmcp-base` image, so build that once from the core repo first (`just docker-base`
in a `medmcp` checkout). Then open the repo with the **Dev Container** action in
PyCharm (2024.2+) or **Reopen in Container** in VS Code — `uv sync` runs on first
start. See the core repo's [CONTRIBUTING](https://github.com/medmcp/medmcp/blob/main/CONTRIBUTING.md)
for IDE specifics.

### Local install (alternative)

```bash
just setup     # install uv, sync dev environment, register pre-commit hooks
just check     # lint + format-check + typecheck + tests
just fix       # auto-fix lint and format
```

For local agent use, install the stack into its own uv tool environment:

```bash
uv tool install --editable .
```

The package registers itself via the `[medmcp.stacks]` entry point. The local
agent autodiscovers it on the next session — no manual config needed. Host-native,
the weights are fetched from Hugging Face on first use into the Hub cache.

### Container image (deployment)

```bash
just docker-build           # build the stack image (FROM medmcp-base)
```

It is a stdio MCP server. The medmcp **core** launches it on demand via a
`stacks.d/<your-package>.toml` manifest (`docker run -i …`; GPU stacks add
`--device nvidia.com/gpu=all`, CDI), so deployment nodes need no host Python
install. Build both architectures — the core refuses to install a foreign-arch
image rather than failing later with "exec format error". Pin any GPU/CUDA build
in `pyproject.toml` against the fleet driver floor (CUDA 12.8 / driver R570).

### Updating the vendored model code

`src/medmcp_cardiac/_cinema/` is a verbatim copy of six upstream modules at the
commit recorded in `_cinema/UPSTREAM`. To move to a newer upstream commit, set
`CINEMA_REV` and run `./scripts/vendor-cinema.sh`; it rewrites only the import
lines and refuses to finish if an upstream import survives.

### Staying in sync with the template

Files shared with [medmcp-template](https://github.com/medmcp/medmcp-template) are
listed in `scripts/shared-files.txt`. The **Template drift** workflow reports when
one of them diverges; `./scripts/sync-from-template.sh` pulls them back. A change
that belongs in every stack goes in the template, not here.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: fork, `just setup`, `just check`, open a PR against `main`.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://pfriedri.github.io"><img src="https://avatars.githubusercontent.com/u/101359393?v=4?s=100" width="100px;" alt="Paul Friedrich"/><br /><sub><b>Paul Friedrich</b></sub></a><br /><a href="https://github.com/medmcp/medmcp-cardiac/commits?author=pfriedri" title="Code">💻</a> <a href="https://github.com/medmcp/medmcp-cardiac/commits?author=pfriedri" title="Documentation">📖</a> <a href="#ideas-pfriedri" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/medmcp/medmcp-cardiac/pulls?q=is%3Apr+reviewed-by%3Apfriedri" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification — contributions of any kind are welcome!

## License

[Apache 2.0](LICENSE). Third-party tools, model weights, and templates bundled by
this stack retain their own licenses and are attributed in [`NOTICE`](NOTICE).
