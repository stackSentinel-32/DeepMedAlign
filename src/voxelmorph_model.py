"""
VoxelMorph U-Net with multi-resolution pyramid and optional diffeomorphic integration.

Architecture
------------
Encoder  : 4 downsampling ConvBlocks
Decoder  : 4 upsampling blocks with skip connections
DVF Head : Coarse DVF predicted at each decoder scale, accumulated (pyramid)
Integrate: Optional VecInt (scaling-and-squaring) for fold-free deformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class SpatialTransformer(nn.Module):
    """Differentiable warp using a dense DVF."""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        D, H, W  = size
        vectors  = [torch.arange(0, s) for s in (D, H, W)]
        grid     = torch.stack(torch.meshgrid(vectors, indexing='ij')).unsqueeze(0).float()
        self.register_buffer('grid', grid)

    def forward(self, src, dvf):
        new_locs = self.grid + dvf
        shape = dvf.shape[2:]
        for i, s in enumerate(shape):
            new_locs[:, i] = 2 * (new_locs[:, i] / (s - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, mode=self.mode,
                             align_corners=True, padding_mode='border')


class VecInt(nn.Module):
    """Scaling-and-squaring integrator for diffeomorphic (fold-free) DVF."""

    def __init__(self, size, steps=7):
        super().__init__()
        self.steps       = steps
        self.scale       = 1.0 / (2 ** steps)
        self.transformer = SpatialTransformer(size)

    def forward(self, svf):
        flow = svf * self.scale
        for _ in range(self.steps):
            flow = flow + self.transformer(flow, flow)
        return flow


class VoxelMorph(nn.Module):
    """Multi-resolution pyramid VoxelMorph.

    DVF is predicted at three decoder scales and accumulated, giving the
    network explicit coarse-to-fine supervision over the alignment.

    Parameters
    ----------
    diffeomorphic : integrate velocity field for fold-free deformation
    """

    def __init__(
        self,
        in_ch         = 2,
        enc_features  = (16, 32, 32, 32),
        dec_features  = (32, 32, 32, 16),
        vol_size      = (160, 192, 160),
        diffeomorphic = False,
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch,           enc_features[0], stride=1)
        self.enc2 = ConvBlock(enc_features[0], enc_features[1], stride=2)
        self.enc3 = ConvBlock(enc_features[1], enc_features[2], stride=2)
        self.enc4 = ConvBlock(enc_features[2], enc_features[3], stride=2)

        # Decoder
        self.dec1 = UpConvBlock(enc_features[3] + enc_features[2], dec_features[0])
        self.dec2 = UpConvBlock(dec_features[0] + enc_features[1], dec_features[1])
        self.dec3 = UpConvBlock(dec_features[1] + enc_features[0], dec_features[2])

        # Multi-resolution DVF heads (coarse → fine pyramid)
        self.dvf_head1 = nn.Conv3d(dec_features[0], 3, 3, padding=1)  # 1/4 res
        self.dvf_head2 = nn.Conv3d(dec_features[1], 3, 3, padding=1)  # 1/2 res
        self.dvf_head3 = nn.Conv3d(dec_features[2], 3, 3, padding=1)  # full res

        for head in (self.dvf_head1, self.dvf_head2, self.dvf_head3):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Diffeomorphic integrator
        self.diffeomorphic = diffeomorphic
        if diffeomorphic:
            self.vec_int = VecInt(vol_size)

        self.spatial_transformer = SpatialTransformer(vol_size)

    def forward(self, mr, ct):
        x = torch.cat([mr, ct], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)

        # Pyramid: accumulate DVF from coarse to fine
        dvf_coarse = self.dvf_head1(d1)
        dvf_mid    = self.dvf_head2(d2) + F.interpolate(dvf_coarse, scale_factor=2, mode='trilinear', align_corners=False)
        dvf_fine   = self.dvf_head3(d3) + F.interpolate(dvf_mid,    scale_factor=2, mode='trilinear', align_corners=False)

        dvf = dvf_fine
        if self.diffeomorphic:
            dvf = self.vec_int(dvf)

        warped_ct = self.spatial_transformer(ct, dvf)
        return warped_ct, dvf
