import torch
import torch.nn.functional as F


class Augmentations:
    """
    6 signal-level transformations forming the policy search space.
    All accept (B, T, C) tensors and return the same shape.
    Intensity parameter sigma in [0, 1] — scaled inside each method.
    """

    @staticmethod
    def jitter(x, sigma=0.05):
        """Add Gaussian noise — simulates measurement noise"""
        return x + torch.randn_like(x) * sigma

    @staticmethod
    def scaling(x, sigma=0.1):
        """Random amplitude scaling per channel — simulates electrode contact variation"""
        B, T, C = x.shape
        scale = torch.randn(B, 1, C).to(x.device) * sigma + 1.0
        return x * scale

    @staticmethod
    def channel_dropout(x, p=0.15):
        """Zero random channels — simulates electrode dropout"""
        B, T, C = x.shape
        mask = torch.bernoulli(
            torch.ones(B, 1, C).to(x.device) * (1 - p)
        )
        return x * mask

    @staticmethod
    def freq_mask(x, F=20):
        """Mask a random frequency band"""
        X  = torch.fft.rfft(x, dim=1)
        f0 = torch.randint(0, max(1, X.shape[1] - F), (1,)).item()
        X[:, f0:f0 + F, :] = 0
        return torch.fft.irfft(X, n=x.shape[1], dim=1)

    @staticmethod
    def temporal_crop(x, crop_ratio=0.85):
        """Crop sub-window and resize back"""
        B, T, C  = x.shape
        crop_len = int(T * crop_ratio)
        start    = torch.randint(0, T - crop_len, (1,)).item()
        cropped  = x[:, start:start + crop_len, :]
        return F.interpolate(
            cropped.permute(0, 2, 1),
            size=T,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    @staticmethod
    def time_warp(x, sigma=0.2):
        """Randomly warp the time axis"""
        B, T, C = x.shape
        t_orig  = torch.linspace(-1, 1, T).to(x.device)
        warp    = t_orig.unsqueeze(0).expand(B, T)
        warp    = (warp + torch.randn(B, T).to(x.device) * sigma / T)
        warp    = warp.clamp(-1, 1)
        zeros   = torch.zeros(B, T).to(x.device)
        grid    = torch.stack([warp, zeros], dim=-1).unsqueeze(1)
        x_4d    = x.permute(0, 2, 1).unsqueeze(2)
        warped  = F.grid_sample(
            x_4d,
            grid.expand(B, C, T, 2).contiguous()[:, :1],
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        return warped.squeeze(2).permute(0, 2, 1)