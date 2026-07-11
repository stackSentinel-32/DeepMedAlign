"""
voxelmorph_model.py
--------------------
VoxelMorph-style U-Net for deformable MRI-CT registration.

Architecture
------------
Encoder: 4 downsampling blocks (Conv3d + LeakyReLU)
Decoder: 4 upsampling blocks (Upsample + Conv3d + skip connections)
Head   : 1x1x1 Conv -> 3-channel DVF (displacement vector field)
SpatialTransformer: warps CT using predicted DVF

Input:  [MRI, CT] concatenated along channel dim -> (B, 2, D, H, W)
Output: warped_ct (B, 1, D, H, W),  dvf (B, 3, D, H, W)

Usage
-----
    model = VoxelMorph()
    warped_ct, dvf = model(mr, ct)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear',
                                 align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Spatial Transformer (differentiable warp)
# ---------------------------------------------------------------------------

class SpatialTransformer(nn.Module):
    """Warp an image using a dense displacement vector field (DVF).

    Parameters
    ----------
    size : (D, H, W) -- spatial dimensions of the volume
    mode : interpolation mode ('bilinear' works for 3D in PyTorch)
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode

        # Create base sampling grid (identity) and register as buffer
        grid = self._identity_grid(size)
        self.register_buffer('grid', grid)

    @staticmethod
    def _identity_grid(size):
        D, H, W  = size
        vectors  = [torch.arange(0, s) for s in (D, H, W)]
        grids    = torch.meshgrid(vectors, indexing='ij')
        grid     = torch.stack(grids)           # (3, D, H, W)
        grid     = grid.unsqueeze(0).float()    # (1, 3, D, H, W)
        return grid

    def forward(self, src, dvf):
        # dvf: (B, 3, D, H, W) -- displacement in voxel units
        new_locs = self.grid + dvf

        # Normalise to [-1, 1] for grid_sample
        shape = dvf.shape[2:]
        for i, s in enumerate(shape):
            new_locs[:, i] = 2 * (new_locs[:, i] / (s - 1) - 0.5)

        # grid_sample expects (B, D, H, W, 3)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]    # XYZ order

        return F.grid_sample(src, new_locs, mode=self.mode,
                             align_corners=True,
                             padding_mode='border')


# ---------------------------------------------------------------------------
# VoxelMorph U-Net
# ---------------------------------------------------------------------------

class VoxelMorph(nn.Module):
    """U-Net based deformable registration network.

    Parameters
    ----------
    in_ch        : input channels per modality (1 MR + 1 CT = 2)
    enc_features : channels in each encoder stage
    dec_features : channels in each decoder stage
    vol_size     : (D, H, W) for SpatialTransformer
    """

    def __init__(
        self,
        in_ch        = 2,
        enc_features = (16, 32, 32, 32),
        dec_features = (32, 32, 32, 32, 16),
        vol_size     = (160, 192, 160),
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_ch,           enc_features[0], stride=1)
        self.enc2 = ConvBlock(enc_features[0], enc_features[1], stride=2)
        self.enc3 = ConvBlock(enc_features[1], enc_features[2], stride=2)
        self.enc4 = ConvBlock(enc_features[2], enc_features[3], stride=2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec1 = UpConvBlock(enc_features[3] + enc_features[2],
                                dec_features[0])
        self.dec2 = UpConvBlock(dec_features[0] + enc_features[1],
                                dec_features[1])
        self.dec3 = UpConvBlock(dec_features[1] + enc_features[0],
                                dec_features[2])

        # ── DVF head ─────────────────────────────────────────────────────────
        self.dvf_head = nn.Conv3d(dec_features[2], 3,
                                  kernel_size=3, padding=1)
        nn.init.zeros_(self.dvf_head.weight)
        nn.init.zeros_(self.dvf_head.bias)

        # ── Spatial transformer ───────────────────────────────────────────────
        self.spatial_transformer = SpatialTransformer(vol_size)

    def forward(self, mr, ct):
        """
        Parameters
        ----------
        mr : (B, 1, D, H, W)  -- fixed image (MRI)
        ct : (B, 1, D, H, W)  -- moving image (CT)

        Returns
        -------
        warped_ct : (B, 1, D, H, W) -- CT warped to MRI space
        dvf       : (B, 3, D, H, W) -- displacement vector field
        """
        x = torch.cat([mr, ct], dim=1)   # (B, 2, D, H, W)

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decode with skip connections
        d1 = self.dec1(e4, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)

        # DVF + warp
        dvf       = self.dvf_head(d3)
        warped_ct = self.spatial_transformer(ct, dvf)

        return warped_ct, dvf
