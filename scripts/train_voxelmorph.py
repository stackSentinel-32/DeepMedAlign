"""
train_voxelmorph.py
--------------------
Training loop for VoxelMorph MRI-CT deformable registration.

What this script does:
  1. Loads train/val DataLoaders from R1's pipeline
  2. Trains VoxelMorph using Multi-Scale Deep Supervision (MIND at 3 scales)
  3. Logs MIND loss (all 3 scales) + MI loss (for comparison) + NCC every epoch
  4. Saves best model checkpoint (lowest val loss)
  5. Saves training CSV for R3 visualisation

Run:
    python scripts/train_voxelmorph.py
    python scripts/train_voxelmorph.py --epochs 50 --lr 1e-4
    python scripts/train_voxelmorph.py --resume models/voxelmorph_best.pth
    python scripts/train_voxelmorph.py --device cpu
"""

import sys
import time
import argparse
import csv
import platform
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config           import MODELS, RESULTS, EPOCHS, LR, LAMBDA_SMOOTH
from src.voxelmorph_model import VoxelMorph
from src.losses           import multiscale_total_loss, mutual_information_loss
from src.dataloader       import get_dataloaders
from src.metrics          import normalised_cross_correlation as ncc
from src.utils            import get_logger

log = get_logger("train_voxelmorph")


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_reg):
    model.train()
    total = mind_sum = mind_q_sum = mind_h_sum = reg_sum = 0.0
    n = 0

    for batch in loader:
        mr = batch["mr"].to(device)   # (1, 1, D, H, W)
        ct = batch["ct"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            warped_scales, dvf = model(mr, ct)
            loss, losses       = multiscale_total_loss(warped_scales, mr, dvf, lambda_reg)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total      += losses["total"]
        mind_sum   += losses["mind"]
        mind_q_sum += losses["mind_quarter"]
        mind_h_sum += losses["mind_half"]
        reg_sum    += losses["reg"]
        n          += 1

    return {
        "train_loss":         total      / max(n, 1),
        "train_mind":         mind_sum   / max(n, 1),
        "train_mind_quarter": mind_q_sum / max(n, 1),
        "train_mind_half":    mind_h_sum / max(n, 1),
        "train_reg":          reg_sum    / max(n, 1),
    }


@torch.no_grad()
def validate(model, loader, device, lambda_reg):
    model.eval()
    total = mind_sum = mind_q_sum = mind_h_sum = mi_sum = reg_sum = ncc_sum = 0.0
    n = 0

    for batch in loader:
        mr = batch["mr"].to(device)
        ct = batch["ct"].to(device)

        with torch.cuda.amp.autocast():
            warped_scales, dvf = model(mr, ct)
            loss, losses       = multiscale_total_loss(warped_scales, mr, dvf, lambda_reg)

        # The full-resolution warped CT is the last element
        warped_full = warped_scales[-1]

        # --- Also compute MI score for side-by-side comparison ---
        with torch.cuda.amp.autocast(enabled=False):
            mi_val = mutual_information_loss(
                warped_full.float(), mr.float()
            ).item()

        ncc_val = ncc(
            mr[0, 0].cpu().numpy(),
            warped_full[0, 0].cpu().numpy(),
        )

        total      += losses["total"]
        mind_sum   += losses["mind"]
        mind_q_sum += losses["mind_quarter"]
        mind_h_sum += losses["mind_half"]
        mi_sum     += mi_val
        reg_sum    += losses["reg"]
        ncc_sum    += ncc_val
        n          += 1

    return {
        "val_loss":         total      / max(n, 1),
        "val_mind":         mind_sum   / max(n, 1),
        "val_mind_quarter": mind_q_sum / max(n, 1),
        "val_mind_half":    mind_h_sum / max(n, 1),
        "val_mi":           mi_sum     / max(n, 1),
        "val_reg":          reg_sum    / max(n, 1),
        "val_ncc":          ncc_sum    / max(n, 1),
    }


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    log.info(f"Checkpoint saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=EPOCHS)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--lambda-reg", type=float, default=LAMBDA_SMOOTH,
                    dest="lambda_reg")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint to resume from")
    # Auto-detect best worker count: 4 on Linux (Kaggle), 0 on Windows
    default_workers = 4 if platform.system() == "Linux" else 0
    ap.add_argument("--workers",       type=int, default=default_workers)
    ap.add_argument("--diffeomorphic", action="store_true",
                    help="Use diffeomorphic (no-fold) integration in model")
    ap.add_argument("--cosine",        action="store_true",
                    help="Use CosineAnnealingWarmRestarts instead of ReduceLROnPlateau")
    args = ap.parse_args()

    device = get_device(args.device)
    log.info(f"Device: {device}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        batch_size=1,
        num_workers=args.workers,
        augment=True,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    if train_loader is None or len(train_loader.dataset) == 0:
        log.error("Train dataset is empty. Run build_npy_cache.py first.")
        sys.exit(1)

    log.info(f"Train: {len(train_loader.dataset)} subjects")
    log.info(f"Val  : {len(val_loader.dataset)} subjects")
    log.info(f"DataLoader workers: {args.workers}")

    # ── Model ────────────────────────────────────────────────────────────────
    model     = VoxelMorph(diffeomorphic=args.diffeomorphic).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=1e-6)
        log.info("Scheduler: CosineAnnealingWarmRestarts (T_0=100, T_mult=2)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=50, factor=0.5, min_lr=1e-6)
        log.info("Scheduler: ReduceLROnPlateau (patience=50, factor=0.5)")

    # ── AMP Scaler (GPU only) ─────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    log.info(f"AMP (Mixed Precision): {'ENABLED' if use_amp else 'disabled'}")

    # ── torch.compile (PyTorch 2.0+, best on Linux/Kaggle) ────────────────────
    if hasattr(torch, "compile") and device.type == "cuda" and platform.system() == "Linux":
        try:
            model = torch.compile(model)
            log.info("torch.compile: ENABLED (15-30%% extra speedup)")
        except Exception as e:
            log.warning(f"torch.compile skipped: {e}")
    else:
        log.info("torch.compile: skipped (Windows / CPU / older PyTorch)")

    start_epoch = 0
    best_val    = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        log.info(f"Resumed from epoch {start_epoch}, val_loss={best_val:.4f}")

    # ── Output paths ─────────────────────────────────────────────────────────
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    best_path = str(MODELS / "voxelmorph_best.pth")
    last_path = str(MODELS / "voxelmorph_last.pth")
    log_path  = str(RESULTS / "training_log.csv")

    # ── CSV log ──────────────────────────────────────────────────────────────
    csv_fields   = [
        "epoch", "train_loss", "train_mind", "train_mind_quarter",
        "train_mind_half", "train_reg",
        "val_loss", "val_mind", "val_mind_quarter", "val_mind_half",
        "val_mi", "val_reg", "val_ncc", "lr",
    ]
    write_header = not Path(log_path).exists()
    csv_file     = open(log_path, "a", newline="")
    writer       = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"Training for {args.epochs} epochs")
    log.info(f"lambda_reg={args.lambda_reg}  lr={args.lr}")
    log.info("Loss: Multi-Scale MIND (quarter=0.25x, half=0.5x, full=1.0x) + gradient regulariser")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0         = time.time()
        train_logs = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args.lambda_reg)
        val_logs   = validate(
            model, val_loader, device, args.lambda_reg)

        if args.cosine:
            scheduler.step()
        else:
            scheduler.step(val_logs["val_loss"])
        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr":    cur_lr,
            **train_logs,
            **val_logs,
        }
        writer.writerow(row)
        csv_file.flush()

        # Save best model
        if val_logs["val_loss"] < best_val:
            best_val = val_logs["val_loss"]
            save_checkpoint(model, optimizer, epoch, best_val, best_path)

        # Save latest model every 50 epochs
        if epoch % 50 == 0:
            save_checkpoint(model, optimizer, epoch,
                            val_logs["val_loss"], last_path)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:4d} | "
            f"train={train_logs['train_loss']:.6f} "
            f"(mind={train_logs['train_mind']:.6f} "
            f"q={train_logs['train_mind_quarter']:.6f} "
            f"h={train_logs['train_mind_half']:.6f} "
            f"reg={train_logs['train_reg']:.6f}) | "
            f"mind_score={val_logs['val_mind']:.6f}  "
            f"mi_score={val_logs['val_mi']:.6f}  "
            f"ncc={val_logs['val_ncc']:.4f} | "
            f"{elapsed:.1f}s"
        )

    csv_file.close()
    log.info(f"Training complete. Best val loss: {best_val:.4f}")
    log.info(f"Best model: {best_path}")
    log.info(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
