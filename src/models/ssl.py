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
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, 500 * in_channels),
        )
        self.in_channels = in_channels

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
        """Reconstruct signal from latent vector"""
        B = z.shape[0]
        out = self.decoder(z)
        return out.view(B, target_len, self.in_channels)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"✓ Model saved to {path}")

    def load(self, path, device='cpu'):
        self.load_state_dict(
            torch.load(path, map_location=device)
        )
        print(f"✓ Model loaded from {path}")