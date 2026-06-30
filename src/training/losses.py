import torch
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.07):
    """
    InfoNCE contrastive loss.
    z1, z2: (B, dim) — two augmented views of the same signals.
    Positive pairs: (z1[i], z2[i]) — same signal, different augmentation.
    Negative pairs: all other combinations in the batch.
    """
    B = z1.shape[0]

    # L2 normalise — puts vectors on unit hypersphere
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # concatenate: (2B, dim)
    z = torch.cat([z1, z2], dim=0)

    # similarity matrix: (2B, 2B)
    sim = torch.mm(z, z.T) / temperature

    # positive pair indices
    labels = torch.cat([
        torch.arange(B) + B,
        torch.arange(B)
    ]).to(z.device)

    # mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))

    return F.cross_entropy(sim, labels)


def reconstruction_loss(model, x1, x_original):
    """
    Reconstruct original signal from augmented view encoding.
    Simpler than masked reconstruction — more stable for small datasets.
    x1:         augmented view  (B, T, C)
    x_original: clean original  (B, T, C)
    """
    B, T, C = x1.shape

    # encode augmented view
    z = model.encoder(x1)

    # reconstruct — target is the ORIGINAL clean signal
    x_recon = model.decode(z, target_len=T)

    return F.mse_loss(x_recon, x_original)

def combined_loss(z1, z2, model, x1, x_original,
                  temperature=0.07,
                  recon_weight=0.5):
    c_loss = info_nce_loss(z1, z2, temperature)

    # skip reconstruction if weight is 0
    if recon_weight == 0.0:
        return c_loss, c_loss, torch.tensor(0.0)

    r_loss = reconstruction_loss(model, x1, x_original)
    total  = c_loss + recon_weight * r_loss
    return total, c_loss, r_loss