import torch
import torch.nn as nn
from src.models.augmentations import Augmentations


class AugmentationPolicy(nn.Module):
    """
    Signal-conditioned learnable augmentation policy.

    Given an input signal window, learns:
      - WHICH augmentation to apply (aug_probs, softmax over 6)
      - HOW MUCH to apply it (intensities, sigmoid in [0,1])

    This is the core CS contribution of the thesis:
    augmentation strategy as a learned component, not a fixed design choice.
    """

    AUG_NAMES = [
        'jitter', 'scaling', 'channel_dropout',
        'freq_mask', 'temporal_crop', 'time_warp'
    ]

    def __init__(self, n_channels=12, n_augmentations=6):
        super().__init__()
        self.n_aug = n_augmentations
        self.aug   = Augmentations()

        # small CNN reads signal properties
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # which augmentation
        self.aug_head = nn.Linear(64, n_augmentations)
        # how intensely
        self.intensity_head = nn.Linear(64, n_augmentations)

    def forward(self, x):
        """
        x: (B, T, C)
        returns:
          aug_probs:   (B, n_aug) — probability over augmentations
          intensities: (B, n_aug) — intensity for each augmentation
        """
        # (B, T, C) → (B, C, T) for Conv1d
        h = self.signal_encoder(
            x.permute(0, 2, 1)
        ).squeeze(-1)                                    # (B, 64)

        aug_probs   = torch.softmax(self.aug_head(h), dim=-1)
        intensities = torch.sigmoid(self.intensity_head(h))
        return aug_probs, intensities

    def apply(self, x, aug_probs, intensities):
        """
        Apply one augmentation per batch item, sampled from the policy.
        x:           (B, T, C)
        aug_probs:   (B, n_aug)
        intensities: (B, n_aug)
        returns:     (B, T, C)
        """
        aug_fns = [
            lambda x, s: self.aug.jitter(x,          sigma=s * 0.2),
            lambda x, s: self.aug.scaling(x,          sigma=s * 0.3),
            lambda x, s: self.aug.channel_dropout(x,  p=s * 0.3),
            lambda x, s: self.aug.freq_mask(x,        F=max(1, int(s * 40))),
            lambda x, s: self.aug.temporal_crop(x,    crop_ratio=1 - s * 0.2),
            lambda x, s: self.aug.time_warp(x,        sigma=s * 0.3),
        ]

        # sample one augmentation per item
        aug_idx = torch.distributions.Categorical(aug_probs).sample()

        results = []
        for b in range(x.shape[0]):
            idx   = aug_idx[b].item()
            sigma = intensities[b, idx].item()
            results.append(aug_fns[idx](x[b:b+1], sigma))

        return torch.cat(results, dim=0)

    def get_preferences(self, x):
        """Return mean aug probabilities for a batch — for analysis"""
        with torch.no_grad():
            aug_probs, _ = self.forward(x)
        return dict(zip(
            self.AUG_NAMES,
            aug_probs.mean(dim=0).cpu().numpy()
        ))