# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First cut of `medmcp-cardiac` — the cine cardiac MRI stack for MedMCP. **Not licensed
for clinical use.** CineMA is not a medical device.

### Added

- `segment_cine_sax` — CineMA short-axis cine segmentation (RV, myocardium, LV) on every frame; writes a 4D `*_dseg.nii.gz`, a label CSV, and ED/ES image + label pairs the workspace viewer can overlay.
- `cardiac_function` — EDV, ESV, SV, EF for both ventricles and LV mass from a cine label map, optionally indexed to body surface area; per-frame volume CSV.
- Three checkpoints (`mnms2` default, `mnms`, `acdc`), all baked into the image; the build reads the list from `tools/checkpoints.py`.
- CineMA model code vendored from a pinned commit (`scripts/vendor-cinema.sh`) instead of the upstream package, whose exact pins (numpy 1.26) and training-only deps (wandb, plotly, MONAI, SimpleITK) do not fit the image or Python 3.13.
- Shared device convention (`auto` → cuda > mps > cpu, resolved device reported, CPU runs warn) and deterministic input checks (frame/slice counts, spacings, CT-like intensities) that warn, never block.
- Inference in a subprocess so stdout stays clean for MCP framing and the server imports no torch; the tool, `server_config` and the image label share one timeout, enforced by a test.
- Container image: `FROM medmcp-base`, torch 2.7.1 cu128, `org.medmcp.stack` label, `HF_HUB_OFFLINE=1` at run time; arm64 lock walk test.
- `segment_cine_lax4c` — the four-chamber long-axis view with CineMA's 2D model: per-plane label maps, LV/RV areas and fractional area change.
- `regional_wall_analysis` — AHA 16-segment wall thickness at ED/ES and systolic thickening from the short-axis segmentation, oriented on the RV insertion points; per-segment CSV (the data of a bullseye).
- `cardiac_calcium_score` — Agatston score, volume and lesion list on non-contrast CT inside a heart mask from the TotalSegmentator stack; cardiac, not coronary.
- Cardiac output and index from `heart_rate_bpm` in `cardiac_function` and `segment_cine_sax`.
- `cardiac-function` skill.
