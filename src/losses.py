import torch
import torch.nn.functional as F


def mutual_information_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    num_bins: int = 32,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Differentiable MI loss via Parzen-window (soft histogram) estimation.

    Gold standard for multimodal MRI-CT registration. Does not assume any
    intensity relationship between modalities.

    x, y : (B, 1, D, H, W) normalised to [0, 1]
    """
    # Disable autocast so log/exp don't underflow to NaN in float16
    with torch.cuda.amp.autocast(enabled=False):
        x = x.float().clamp(0.0, 1.0)
        y = y.float().clamp(0.0, 1.0)

        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        y_flat = y.reshape(B, -1)

        bins = torch.linspace(0, 1, num_bins, device=x.device)

        dx = x_flat.unsqueeze(2) - bins.view(1, 1, -1)  # (B, N, bins)
        dy = y_flat.unsqueeze(2) - bins.view(1, 1, -1)

        wx = torch.exp(-dx ** 2 / (2 * sigma ** 2))
        wy = torch.exp(-dy ** 2 / (2 * sigma ** 2))

        # Joint histogram (B, bins, bins)
        joint = torch.bmm(wx.permute(0, 2, 1), wy) / x_flat.shape[1]
        joint = joint / (joint.sum(dim=[1, 2], keepdim=True) + 1e-10)

        px = joint.sum(dim=2)
        py = joint.sum(dim=1)
        px_py = px.unsqueeze(2) * py.unsqueeze(1)

        mi = (joint * torch.log(joint / (px_py + 1e-10) + 1e-10)).sum(dim=[1, 2])
        return -mi.mean()  # negate: we minimise loss


def gradient_loss(dvf: torch.Tensor) -> torch.Tensor:
    """L2 smoothness penalty on the DVF."""
    dy = dvf[:, :, 1:, :, :]  - dvf[:, :, :-1, :, :]
    dx = dvf[:, :, :, 1:, :]  - dvf[:, :, :, :-1, :]
    dz = dvf[:, :, :, :, 1:]  - dvf[:, :, :, :, :-1]
    return (dy ** 2).mean() + (dx ** 2).mean() + (dz ** 2).mean()


def total_loss(
    warped_ct:  torch.Tensor,
    mr:         torch.Tensor,
    dvf:        torch.Tensor,
    lambda_reg: float = 0.5,
) -> tuple:
    sim  = mutual_information_loss(warped_ct, mr)
    reg  = gradient_loss(dvf)
    loss = sim + lambda_reg * reg

    return loss, {"total": loss.item(), "mi": sim.item(), "reg": reg.item()}
