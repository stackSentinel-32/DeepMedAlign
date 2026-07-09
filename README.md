# 🧠 DeepMedAlign

> **Aligning CT and MRI brain scans — pixel by pixel — using classical registration and deep learning.**

Medical imaging generates two fundamentally different views of the same patient: **MRI** captures soft tissue detail, **CT** guides treatment planning. Before clinicians can use them together, these scans must be precisely aligned. DeepMedAlign automates that process — from raw DICOM to a perfectly warped, voxel-registered output — at scale, on 180 real patient brain scans.

---

## 🎯 What It Does

Takes a patient's CT scan and warps it to match their MRI — millimetre by millimetre — so both scans occupy the same coordinate space and can be overlaid perfectly.

```mermaid
flowchart LR
    A["🏥 Raw Patient Scan\n(CT + MRI)"] --> B["🔧 Preprocessing\nNormalise · Skull-strip · Resample to 1mm"]
    B --> C["📐 Rigid Registration\nRotate + Translate\n~3 sec"]
    C --> D["📏 Affine Registration\nScale + Shear\n~3 sec"]
    D --> E["〰️ B-spline Registration\nLocal pixel-level warp\n~3 min"]
    E --> F["✅ Registered CT\nPerfectly overlaid\non MRI space"]

    style A fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style B fill:#1e3a5f,color:#fff,stroke:#4a90d9
    style C fill:#2d5016,color:#fff,stroke:#6abf40
    style D fill:#2d5016,color:#fff,stroke:#6abf40
    style E fill:#2d5016,color:#fff,stroke:#6abf40
    style F fill:#5a1a1a,color:#fff,stroke:#e05252
```

---

## 📊 Baseline Results (Classical Registration)

Evaluated on **125 training subjects** from the [SynthRad 2023](https://synthrad2023.grand-challenge.org/) brain dataset.

| Method | Dice ↑ | HD95 (mm) ↓ |
|--------|--------|-------------|
| Rigid | — | — |
| Affine | — | — |
| **B-spline** | **0.776 ± 0.059** | **19.2 ± 7.6** |

> These are the **floor numbers**. The deep learning model (Week 3) must beat them to be clinically meaningful.

---

## 🗂️ Dataset

- **Source:** SynthRad 2023 — Task 1 (MR → CT synthesis / registration)
- **Subjects:** 180 (125 train / 27 val / 28 test)
- **Resolution:** 160 × 192 × 160 @ 1 mm isotropic
- **Format:** NIfTI (`.nii.gz`), normalised and skull-stripped

> ⚠️ Raw data (~15 GB) is **not tracked in git**. Download from SynthRad and place under `data/raw/synthrad/brain/`.

---

## 🚀 Quick Start

```powershell
# 1. Set up environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run classical registration (train split)
python scripts\run_classical.py --split train

# 3. Compute baseline metrics
python scripts\compute_baseline_metrics.py --method bspline --split train
```

---

## 🗺️ Roadmap

| Phase | Branch | Status |
|-------|--------|--------|
| R1 — Data Pipeline | `r1/data-pipeline` | ✅ Done |
| R2 — Classical Registration | `r2/week2-classical-registration` | ✅ Done |
| R3 — Visualisation | `r3/preprocess-viz` | ✅ Done |
| R4 — Research Docs | `r4/research-docs` | 🔲 Upcoming |
| Week 3 — VoxelMorph (Deep Learning) | `r2/week3-voxelmorph` | 🔲 Upcoming |

```mermaid
flowchart TD
    R1["✅ R1 · Data Pipeline\nDownload · Preprocess · Split\n180 brain scans ready"]
    R2["✅ R2 · Classical Registration\nRigid → Affine → B-spline\nBaseline: Dice=0.776, HD95=19.2mm"]
    R3["✅ R3 · Visualisation\nOverlay checks · QC plots"]
    W3["🔲 Week 3 · VoxelMorph\nTrain neural net to replace\nB-spline in milliseconds"]
    R4["🔲 R4 · Research Docs\nPaper · Analysis · Report"]
    GOAL["🏆 Goal\nDice > 0.776\nHD95 < 19.2 mm\nat inference speed"]

    R1 --> R2
    R2 --> R3
    R3 --> W3
    W3 --> R4
    W3 --> GOAL

    style R1 fill:#1a3a1a,color:#7fff7f,stroke:#4caf50
    style R2 fill:#1a3a1a,color:#7fff7f,stroke:#4caf50
    style R3 fill:#1a3a1a,color:#7fff7f,stroke:#4caf50
    style W3 fill:#2a2a1a,color:#ffff7f,stroke:#cddc39
    style R4 fill:#2a2a1a,color:#ffff7f,stroke:#cddc39
    style GOAL fill:#3a1a1a,color:#ff9f9f,stroke:#f44336
```

---


## 🏗️ Project Structure

```
DeepMedAlign/
├── data/
│   ├── raw/           # Manifests & SynthRad source (data not tracked)
│   └── processed/     # Normalised scans & registration outputs (not tracked)
├── results/           # Metrics CSVs — the numbers that matter
├── scripts/           # Run registration, compute metrics, build manifests
├── src/               # Core library (config, classical_reg, metrics, utils)
├── notebooks/         # Exploration & visualisation
└── tests/             # Unit tests — run with pytest tests/ -v
```

---

## 🤝 Contributing

- **Never commit directly to `main`** — open a PR at the end of each day
- Keep `main` runnable at all times
- Branch naming: `r{id}/short-description`

---

## 📄 License

Research use only. Dataset governed by [SynthRad 2023 terms](https://synthrad2023.grand-challenge.org/).