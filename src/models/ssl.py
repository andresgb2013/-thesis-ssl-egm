import torch
import torch.nn as nn
from src.models.encoder import build_encoder
from src.models.policy  import AugmentationPolicy


class SSLModel(nn.Module):
    """
    Full SSL model: policy + encoder.
    Wraps both into one clean interface.
    """

    def __init__(self, in_channels=12, hidden_dim=128,
                out_dim=64, n_augmentations=6):
        super().__init__()
        self.policy  = AugmentationPolicy(in_channels, n_augmentations)
        self.encoder = build_encoder(in_channels, hidden_dim, out_dim)

        # decoder for reconstruction loss
        # Replace the decoder in SSLModel.__init__
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim * 8),
            nn.GELU(),
            nn.Unflatten(1, (hidden_dim, 8)),        # (B, hidden, 8)
            nn.ConvTranspose1d(hidden_dim, hidden_dim,
                            kernel_size=4, stride=4),   # → (B, hidden, 32)
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2,
                            kernel_size=4, stride=4),   # → (B, hidden/2, 128)
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 2, in_channels,
                            kernel_size=4, stride=4),   # → (B, C, 512)
            nn.AdaptiveAvgPool1d(500),                     # → (B, C, 500)
)
        

    def forward(self, x):
        """
        Generate two augmented views, encode both.
        Returns z1, z2, x1, x2
        """
        # view 1
        p1, i1 = self.policy(x)
        x1     = self.policy.apply(x, p1, i1)
        z1     = self.encoder(x1)

        # view 2 — independent augmentation decision
        p2, i2 = self.policy(x)
        x2     = self.policy.apply(x, p2, i2)
        z2     = self.encoder(x2)

        return z1, z2, x1, x2

    def encode(self, x):
        """Encode without augmentation — for evaluation"""
        return self.encoder(x)

    def decode(self, z, target_len=500):
        out = self.decoder(z)                          # (B, C, 500)
        return out.permute(0, 2, 1)                    # (B, 500, C)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"✓ Model saved to {path}")

    def load(self, path, device='cpu'):
        self.load_state_dict(
            torch.load(path, map_location=device)
        )
        print(f"✓ Model loaded from {path}")