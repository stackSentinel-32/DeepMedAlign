"""
src/dataset.py
--------------
PyTorch Dataset for MRI-CT registration.

Reads manifest_processed.csv, loads pre-cached .npy volumes,
and returns channel-first float32 tensors ready for VoxelMorph.

Usage
-----
    from src.dataset import MedicalRegistrationDataset

    ds     = MedicalRegistrationDataset(split="train")
    sample = ds[0]
    # sample["mr"]         -> torch.Tensor (1, 160, 192, 160)  float32
    # sample["ct"]         -> torch.Tensor (1, 160, 192, 160)  float32
    # sample["mask"]       -> torch.Tensor (1, 160, 192, 160)  float32  binary
    # sample["subject_id"] -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional

from src.config import DATA_RAW, DATA_PROC, FIXED_SHAPE
from src.utils import get_logger

log = get_logger("dataset")


class MedicalRegistrationDataset(Dataset):
    """PyTorch Dataset for SynthRAD 2023 MRI-CT registration pairs.

    Parameters
    ----------
    split       : one of 'train', 'val', 'test'  (None = all subjects)
    manifest    : path to manifest_processed.csv
                  (defaults to DATA_RAW / manifest_processed.csv)
    transform   : optional callable applied to each sample dict
    require_npy : if True, skip subjects whose .npy files are missing
                  instead of raising (safe for partial datasets)
    """

    # Required columns in the manifest
    _REQUIRED_COLS = {"subject_id", "split"}

    def __init__(
        self,
        split:       Optional[str]      = "train",
        manifest:    Optional[str]      = None,
        transform:   Optional[Callable] = None,
        require_npy: bool               = True,
    ) -> None:
        valid_splits = {"train", "val", "test", None}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got {split!r}")

        self.split       = split
        self.transform   = transform
        self.require_npy = require_npy

        # ── Load manifest ────────────────────────────────────────────────────
        manifest_path = Path(manifest) if manifest else self._find_manifest()
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run: python scripts/update_manifest_preprocessed.py"
            )

        df = pd.read_csv(manifest_path)

        missing_cols = self._REQUIRED_COLS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Manifest is missing columns: {missing_cols}\n"
                "Run: python scripts/update_manifest_preprocessed.py"
            )

        # ── Filter by split ──────────────────────────────────────────────────
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)

        # ── Build sample list from NPY paths ─────────────────────────────────
        self.samples: list[dict] = []
        n_skipped = 0

        for _, row in df.iterrows():
            sid     = str(row["subject_id"])
            out_dir = DATA_PROC / sid

            mr_path   = str(out_dir / f"{sid}_mr.npy")
            ct_path   = str(out_dir / f"{sid}_ct.npy")
            mask_path = str(out_dir / f"{sid}_mask.npy")

            # Fall back to manifest columns if present
            if "mr_npy" in row and row.get("mr_npy", ""):
                mr_path = str(row["mr_npy"])
            if "ct_npy" in row and row.get("ct_npy", ""):
                ct_path = str(row["ct_npy"])
            if "mask_npy" in row and row.get("mask_npy", ""):
                mask_path = str(row["mask_npy"])

            # Validate existence
            mr_ok   = Path(mr_path).exists()
            ct_ok   = Path(ct_path).exists()
            mask_ok = Path(mask_path).exists()

            if require_npy and not (mr_ok and ct_ok):
                log.warning(
                    f"{sid}: skipping — MR OK={mr_ok}, CT OK={ct_ok}"
                )
                n_skipped += 1
                continue

            self.samples.append({
                "subject_id": sid,
                "mr_path":    mr_path   if mr_ok   else None,
                "ct_path":    ct_path   if ct_ok   else None,
                "mask_path":  mask_path if mask_ok else None,
            })

        if n_skipped > 0:
            log.warning(
                f"Skipped {n_skipped} subjects with missing .npy files "
                f"({len(self.samples)} usable)"
            )

        log.info(
            f"MedicalRegistrationDataset  split={split}  "
            f"n={len(self.samples)}"
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_manifest() -> Path:
        """Return the best available manifest path."""
        for name in [
            "manifest_processed.csv",
            "manifest_final.csv",
            "manifest_v2.csv",
            "manifest.csv",
        ]:
            p = DATA_RAW / name
            if p.exists():
                return p
        return DATA_RAW / "manifest_processed.csv"   # will raise later

    @staticmethod
    def _load_npy(path: Optional[str],
                  shape: tuple = FIXED_SHAPE,
                  normalize: bool = False) -> torch.Tensor:
        """Load a .npy file and return a (1, D, H, W) float32 tensor.

        If path is None or missing, returns a zero tensor of the target shape.
        If normalize=True, applies per-volume min-max normalization to [0, 1].
        This is used for MR volumes whose .npy files may contain z-score values.
        """
        if path and Path(path).exists():
            arr = np.load(path).astype("float32")
        else:
            arr = np.zeros(shape, dtype="float32")

        # Ensure 3D
        if arr.ndim == 4:
            arr = arr[0]

        # Crop / pad to expected shape
        if arr.shape != shape:
            out = np.zeros(shape, dtype="float32")
            s   = tuple(min(a, b) for a, b in zip(arr.shape, shape))
            out[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]
            arr = out

        # Per-volume min-max normalization to [0, 1]
        if normalize:
            v_min, v_max = arr.min(), arr.max()
            if v_max > v_min:
                arr = (arr - v_min) / (v_max - v_min)
            else:
                arr = np.zeros_like(arr)

        return torch.from_numpy(arr).unsqueeze(0)   # (1, D, H, W)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]

        sample = {
            "subject_id": info["subject_id"],
            "mr":         self._load_npy(info["mr_path"],   normalize=True),
            "ct":         self._load_npy(info["ct_path"],   normalize=False),
            "mask":       self._load_npy(info["mask_path"], normalize=False),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ── Utility ──────────────────────────────────────────────────────────────

    def subject_ids(self) -> list[str]:
        """Return list of all subject IDs in this split."""
        return [s["subject_id"] for s in self.samples]

    def __repr__(self) -> str:
        return (
            f"MedicalRegistrationDataset("
            f"split={self.split!r}, n={len(self.samples)})"
        )
