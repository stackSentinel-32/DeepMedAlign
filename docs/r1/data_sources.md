# Data Sources - DeepMedAlign

## Primary Dataset: SynthRAD2023 Task 1 Brain

| Field | Value |
|-------|-------|
| Dataset name | SynthRAD2023 Task 1 - MR-CT brain |
| Challenge URL | https://synthrad2023.grand-challenge.org/ |
| Paper | Houben et al. 2023 (arxiv: 2304.09124) |
| Task | Cross-modal MRI-to-CT registration, brain |
| Subjects | 5 subjects (simulated dummy dataset) |
| Modalities | T1-weighted MRI + planning CT |
| DUA required | Yes - Grand Challenge account + agreement |
| DUA accepted | 2026-06-07 |
| Downloaded | 2026-06-07 |
| Downloaded by | R1 - Data Engineer |

## File structure (per subject)
data/raw/synthrad/brain/<subject_id>/
mr.nii.gz    T1-weighted MRI (raw signal units)
ct.nii.gz    Planning CT (Hounsfield Units)
mask.nii.gz  Brain mask (binary, from SynthRAD organisers)

## Preprocessing applied by R3
See docs/r1/data_observations.md for raw data stats.
After preprocessing (src/preprocess_mri.py + src/preprocess_ct.py):
- Both modalities resampled to 1mm isotropic
- Both reoriented to RAS+
- MRI: N4 bias corrected -> z-score normalised
- CT: HU clipped to [-1000, 1000] -> rescaled to [0, 1]
- Both cropped/padded to (160, 192, 160)

## Train/Val/Test Split
Split is PATIENT-LEVEL - all slices from one subject stay in one split.
Slice-level split would cause data leakage and inflated metrics.

| Split | Fraction | Approx count |
|-------|----------|-------------|
| train | 70%      | 3           |
| val   | 10%      | 1           |
| test  | 20%      | 1           |

Random seed: 42 (set in src/config.py RANDOM_SEED)

## Future datasets (Week 14+)
- SynthRAD2023 Task 1 Pelvis - same pipeline, different anatomy
- Learn2Reg Task 1 - abdominal MR-CT, used for evaluation comparison
