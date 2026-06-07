# SynthRAD2023 - Data Observations

## Dataset
- Name: SynthRAD2023 Task 1 - Brain
- URL: https://synthrad2023.grand-challenge.org/
- DUA accepted: 2026-06-07
- Download date: 2026-06-07
- Total subjects: 5

## File structure (per subject)
data/raw/synthrad/brain/<subject_id>/
mr.nii.gz    <- T1-weighted MRI, raw intensities (0 to ~1500)
ct.nii.gz    <- planning CT in HU (-1000 to ~2000)
mask.nii.gz  <- binary brain mask (provided by SynthRAD)

## Observed shapes
- MRI shapes: approximately 50x50x50 voxels per axis (varies per subject on real data)
- CT shapes: same as MRI within each subject (co-registered in dataset)
- Voxel spacing: typically 1.0 mm (not yet isotropic)

## Intensity ranges
- MRI: 0 to ~1000 (raw T1 signal, will be z-score normalized)
- CT: -1000 to ~1200 HU (will be clipped to [-1000, 1000] and scaled [0,1])

## QC findings
- SHAPE_MISMATCH: 0 subjects - expected, fixed by preprocessing
- All subjects pass HU range check
- All masks are binary (0/1)

## Splits
| Split | Count |
|-------|-------|
| train | 3     |
| val   | 1     |
| test  | 1     |

## Notes
- Shape mismatch between MR and CT is NORMAL - they were acquired separately
  even though they are co-registered. R3's resample_isotropic() fixes this.
- Do not exclude shape-mismatched subjects - they are valid data.
