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


def reconstruction_loss(model, x1, z1, target_len=500):
    """
    Masked reconstruction loss.
    Masks 50% of the signal, reconstructs from latent vector.
    Encourages encoder to capture full temporal structure.
    """
    B, T, C = x1.shape

    # create mask — 50% of timesteps
    mask = torch.rand(B, T, 1).to(x1.device) > 0.5
    mask = mask.expand_as(x1)

    # apply mask
    x_masked = x1 * mask.float()

    # encode masked signal
    z_masked = model.encoder(x_masked)

    # reconstruct
    x_recon = model.decode(z_masked, target_len=T)

    # loss only on masked regions
    loss = F.mse_loss(
        x_recon[~mask],
        x1[~mask]
    )
    return loss


def combined_loss(z1, z2, model, x1,
                temperature=0.07,
                recon_weight=0.5):
    """
    Total SSL loss = contrastive + reconstruction.
    recon_weight controls the balance between the two.
    """
    c_loss = info_nce_loss(z1, z2, temperature)
    r_loss = reconstruction_loss(model, x1, z1)
    total  = c_loss + recon_weight * r_loss
    return total, c_loss, r_loss