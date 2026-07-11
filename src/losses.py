"""
losses.py
----------
Loss functions for VoxelMorph MRI-CT registration.

Two losses are combined:
  1. MIND-SSC (Modality Independent Neighbourhood Descriptor)
     - Handles MRI/CT intensity difference (no direct MSE possible)
     - Compares local image structure, not raw intensities
     - Typical weight: 1.0

  2. Regularisation -- Gradient smoothness of DVF
     - Penalises folding / unrealistic deformations
     - Typical weight: 1.0 (tune between 0.5 - 2.0)

Total loss = MIND_loss(warped_ct, mr) + lambda * reg_loss(dvf)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MIND-SSC Loss
# ---------------------------------------------------------------------------

def mind_ssc(
    img:         torch.Tensor,
    delta:       int   = 1,
    sigma:       float = 0.5,
    kernel_size: int   = 3,
) -> torch.Tensor:
    """Compute MIND Self-Similarity Context descriptor.

    Parameters
    ----------
    img         : (B, 1, D, H, W) -- input image (normalised [0,1])
    delta       : neighbourhood offset in voxels
    sigma       : Gaussian regularisation std
    kernel_size : local patch size for mean subtraction

    Returns
    -------
    descriptor : (B, 12, D, H, W) -- MIND-SSC feature map
    """
    # 6-neighbourhood offsets (+/-delta in each axis)
    offsets = [
        (delta,  0,     0),
        (-delta, 0,     0),
        (0,      delta, 0),
        (0,     -delta, 0),
        (0,      0,     delta),
        (0,      0,    -delta),
    ]

    mind_vol = []

    # Local mean subtraction (reduce illumination bias)
    padding  = kernel_size // 2
    mean     = F.avg_pool3d(img, kernel_size, stride=1, padding=padding)
    img_sub  = img - mean

    for (dz, dy, dx) in offsets:
        # Shift image by offset using torch.roll
        shifted     = torch.roll(img_sub, shifts=(dz, dy, dx), dims=(2, 3, 4))
        diff        = (img_sub - shifted) ** 2
        # Gaussian-weighted sum in local neighbourhood
        diff_smooth = F.avg_pool3d(diff, kernel_size, stride=1, padding=padding)
        mind_vol.append(diff_smooth)

    mind = torch.cat(mind_vol, dim=1)   # (B, 6, D, H, W)

    # Normalise by local variance + sigma (prevent division by zero)
    mind_var = mind.mean(dim=1, keepdim=True) + sigma ** 2
    mind     = mind / mind_var

    # Exponentiate -> MIND descriptor
    mind = torch.exp(-mind)             # (B, 6, D, H, W)

    # Pairwise differences between 6 neighbours -> 12-dim SSC
    mind_ssc_desc = []
    for i in range(6):
        for j in range(i + 1, 6):
            mind_ssc_desc.append(
                (mind[:, i:i+1] - mind[:, j:j+1]) ** 2
            )

    return torch.cat(mind_ssc_desc[:12], dim=1)   # (B, 12, D, H, W)


def mind_ssc_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    delta:  int   = 1,
    sigma:  float = 0.5,
) -> torch.Tensor:
    """MIND-SSC loss between predicted and target image.

    Parameters
    ----------
    pred   : (B, 1, D, H, W) -- warped CT
    target : (B, 1, D, H, W) -- fixed MRI

    Returns
    -------
    scalar loss (lower = more similar structure)
    """
    mind_pred   = mind_ssc(pred,   delta=delta, sigma=sigma)
    mind_target = mind_ssc(target, delta=delta, sigma=sigma)
    return F.mse_loss(mind_pred, mind_target)


# ---------------------------------------------------------------------------
# Regularisation Loss
# ---------------------------------------------------------------------------

def gradient_loss(dvf: torch.Tensor, penalty: str = 'l2') -> torch.Tensor:
    """Spatial gradient smoothness penalty on DVF.

    Penalises large spatial gradients (folding, discontinuities).

    Parameters
    ----------
    dvf     : (B, 3, D, H, W) -- displacement vector field
    penalty : 'l1' or 'l2'

    Returns
    -------
    scalar loss
    """
    dy = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
    dx = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
    dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]

    if penalty == 'l2':
        return (dy ** 2).mean() + (dx ** 2).mean() + (dz ** 2).mean()
    else:
        return dy.abs().mean() + dx.abs().mean() + dz.abs().mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def total_loss(
    warped_ct:  torch.Tensor,
    mr:         torch.Tensor,
    dvf:        torch.Tensor,
    lambda_reg: float = 1.0,
) -> tuple:
    """Compute total registration loss.

    Returns
    -------
    loss   : scalar tensor (backprop through this)
    losses : dict of individual loss components (for logging)
    """
    sim_loss = mind_ssc_loss(warped_ct, mr)
    reg_loss = gradient_loss(dvf)
    loss     = sim_loss + lambda_reg * reg_loss

    return loss, {
        "total": loss.item(),
        "mind":  sim_loss.item(),
        "reg":   reg_loss.item(),
    }
