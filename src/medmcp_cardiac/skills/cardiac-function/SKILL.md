---
name: cardiac-function
description: Workflow for segmenting the ventricles on a short-axis cine cardiac MRI and reporting volumes, ejection fraction and LV mass with CineMA
---

# Cine cardiac MRI segmentation & ventricular function workflow

`segment_cine_sax` labels the right ventricle, the LV myocardium and the LV cavity on
every frame of a **short-axis cine** stack and, by default, derives the ventricular
function report from the result. `cardiac_function` re-derives that report from an
existing segmentation (for example to add height and weight for indexing).

## When to use

- The user wants an ejection fraction, ventricular volumes, stroke volume or LV mass
  from a cine cardiac MRI.
- The user wants the ventricles labelled on a cine series, or an overlay of the
  segmentation to look at.
- The user has a segmentation already and wants it re-measured or indexed to body size.

## Steps

1. **Start from NIfTI.** The tools take a 4D NIfTI (x, y, slices, frames). If the user
   points at DICOM, convert it with the DICOM stack first and pick the short-axis cine
   series -- not a long-axis view, not a late-enhancement or mapping series.
2. **Run `segment_cine_sax`** with `device="auto"`. Pass `height_cm` and `weight_kg`
   when the user gives them, so the volumes come back indexed to body surface area.
3. **Check the resolved `device`.** A fall-back to `cpu` changes runtime from seconds
   to many minutes; say so.
4. **Report the function table** from `function` and relay every `warnings` entry.
   State which frames were taken as end-diastole and end-systole.
5. **Offer the viewer.** Open `ed_image_path` with `ed_segmentation_path` as the
   overlay (the end-systolic pair is `es_image_path` + `es_segmentation_path`). The
   4D `segmentation_path` holds every frame.

## Gotchas

- **Short-axis only.** The model is trained on short-axis stacks; a long-axis cine
  produces a confident, meaningless label map. The tool warns on unusual shapes and
  spacings but cannot prove the view -- when in doubt, ask.
- **ED/ES come from the volume curve**, not from the DICOM trigger times: ED is the
  frame of maximal LV volume, ES the frame of minimal LV volume. Both ventricles are
  measured at those LV-defined phases.
- **Volumes depend on the header.** Every millilitre is a voxel count times the
  header's voxel size. A wrong slice thickness in the NIfTI gives a wrong volume by
  the same factor.
- **A 3D input gives a segmentation and nothing else.** Function needs the whole
  cardiac cycle.
- **Label values** are 1 = right ventricle, 2 = myocardium, 3 = left ventricle; the
  names are in the CSV at `labels_path`.
- **These are research measurements, not clinical findings.** CineMA is not a medical
  device. Do not compare the numbers with reference ranges unless the user asks, and
  never present them as a diagnosis.
