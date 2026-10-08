---
name: cardiac-function
description: Workflow for cine cardiac MRI with CineMA -- short-axis and four-chamber segmentation, volumes, ejection fraction, LV mass, AHA 16-segment wall thickness -- and cardiac calcium scoring on CT
---

# Cardiac analysis workflow

`segment_cine_sax` labels the right ventricle, the LV myocardium and the LV cavity on
every frame of a **short-axis cine** stack and derives the ventricular function report.
`cardiac_function` re-derives that report from an existing segmentation (to add height,
weight or heart rate). `regional_wall_analysis` turns the same segmentation into the
AHA 16-segment wall-thickness and thickening table.
`segment_cine_lax4c` does the four-chamber long-axis view (areas, fractional area
change). `cardiac_calcium_score` is the CT tool: Agatston score inside a heart mask.

## When to use

- Ejection fraction, ventricular volumes, stroke volume, cardiac output or LV mass
  from a cine cardiac MRI.
- Regional wall motion: which segments thicken, which do not.
- A four-chamber view segmented or its area change measured.
- A calcium score on a non-contrast cardiac CT.

## Steps: cine MRI

1. **Start from NIfTI.** The tools take 4D NIfTI (x, y, slices, frames). If the user
   points at DICOM, convert with the DICOM stack first and pick the **short-axis cine**
   series for `segment_cine_sax` (a 4-chamber cine for `segment_cine_lax4c`) -- not a
   long-axis view for the SAX tool, not late-enhancement or mapping series.
2. **Run `segment_cine_sax`** with `device="auto"`. Pass `height_cm` and `weight_kg`
   when the user gives them (indexed volumes) and `heart_rate_bpm` when known (the
   DICOM header carries it) for cardiac output.
3. **Check the resolved `device`.** A fall-back to `cpu` turns seconds into many
   minutes; say so.
4. **Report the function table** from `function` and relay every `warnings` entry.
   State which frames were taken as end-diastole and end-systole.
5. **Offer the viewer.** Open `ed_image_path` with `ed_segmentation_path` as the
   overlay (the end-systolic pair is `es_image_path` + `es_segmentation_path`).
6. **Regional question?** Run `regional_wall_analysis` on `segmentation_path` and report
   the per-segment table (the CSV is at `segments_path`).

## Steps: calcium score on CT

1. Segment the CT with the TotalSegmentator stack's `total` task first; its
   `*_dseg.nii.gz` and `*_labels.csv` are the mask input.
2. Run `cardiac_calcium_score` with the CT and that mask (`structures=["heart"]` by
   default). Report the Agatston score and its band, and say it is a **cardiac**
   calcium score (valves and aortic root count), not a coronary one.

## Gotchas

- **Short-axis only for `segment_cine_sax`.** A long-axis cine gives a confident,
  meaningless label map. The tool warns on unusual shapes and spacings but cannot
  prove the view -- when in doubt, ask.
- **ED/ES come from the volume curve**, not from DICOM trigger times: ED is the frame
  of maximal LV volume, ES the frame of minimal LV volume.
- **Volumes depend on the header.** Every millilitre is a voxel count times the
  header's voxel size; a wrong slice thickness gives a wrong volume by the same factor.
- **A 3D input gives a segmentation and nothing else.** Function needs the cycle.
- **Long-axis areas are not volumes.** Fractional area change complements EF; it does
  not replace it. A stack of several long-axis planes is segmented plane by plane and
  the metrics come from the plane with the largest LV.
- **The regional analysis needs the RV.** Segments are oriented on the RV insertion points; if
  the RV never touches the myocardium the tool refuses. Segment 17 (apical cap) is not
  reported.
- **Calcium scoring needs non-contrast CT on the CT's own grid.** Contrast inflates the
  score (the tool warns); the mask must be a segmentation of that CT, not of a copy.
- **Label values** are 1 = right ventricle, 2 = myocardium, 3 = left ventricle.
- **Research measurements, not clinical findings.** CineMA is not a medical device. Do
  not compare with reference ranges unless asked; never present a diagnosis.
