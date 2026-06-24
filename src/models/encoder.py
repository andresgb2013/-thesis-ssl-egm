import torch
import torch.nn as nn


class DilatedCNNEncoder(nn.Module):
    """
    1D Dilated CNN encoder for multi-channel time series.

    Dilated convolutions capture structure at multiple scales:
      dilation=1  → local morphology  (individual beat features)
      dilation=2  → medium range      (beat shape)
      dilation=4  → long range        (rhythm patterns)
      dilation=8  → very long range   (global signal structure)

    GroupNorm instead of BatchNorm — works with small batches.
    Dropout — prevents representation collapse.
    AdaptiveAvgPool1d(8) — preserves more spatial info than pool to 1.
    """

    def __init__(self, in_channels=12, hidden_dim=128, out_dim=64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim,
                      kernel_size=7, padding=3, dilation=1),
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Dropout(0.1),

            nn.Conv1d(hidden_dim, hidden_dim,
                      kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Dropout(0.1),

            nn.Conv1d(hidden_dim, hidden_dim,
                      kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, hidden_dim), nn.GELU(), nn.Dropout(0.1),

            nn.Conv1d(hidden_dim, hidden_dim,
                      kernel_size=3, padding=8, dilation=8),
            nn.GroupNorm(8, hidden_dim), nn.GELU(),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),            # (B, hidden_dim * 8)
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        h = self.conv_layers(x.permute(0, 2, 1))
        h = self.pool(h)
        return self.projector(h)    # (B, out_dim)


def build_encoder(in_channels=12, hidden_dim=128, out_dim=64):
    """Build encoder with proper weight initialisation"""
    encoder = DilatedCNNEncoder(in_channels, hidden_dim, out_dim)

    for m in encoder.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return encoder