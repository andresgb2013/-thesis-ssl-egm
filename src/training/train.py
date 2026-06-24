import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def info_nce_loss(z1, z2, temperature=0.07):
    B  = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    labels = torch.cat([
        torch.arange(B) + B,
        torch.arange(B)
    ]).to(z.device)
    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))
    return F.cross_entropy(sim, labels)


def train_ssl(encoder, policy, dataloader, 
              epochs=100, lr=3e-4, device='cpu'):
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(policy.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    encoder.train()
    policy.train()
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_cos  = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            aug_probs1, intensities1 = policy(x)
            x1 = policy.apply_augmentation(x, aug_probs1, intensities1)

            aug_probs2, intensities2 = policy(x)
            x2 = policy.apply_augmentation(x, aug_probs2, intensities2)

            z1 = encoder(x1)
            z2 = encoder(x2)

            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(policy.parameters()), 1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cos  += F.cosine_similarity(z1, z2).mean().item()

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        avg_cos  = epoch_cos  / len(dataloader)
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'cos_sim': avg_cos
        })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Cos: {avg_cos:.4f}")

    return history


def plot_history(history):
    epochs   = [h['epoch']   for h in history]
    losses   = [h['loss']    for h in history]
    cos_sims = [h['cos_sim'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(epochs, losses, color='steelblue', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('InfoNCE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, cos_sims, color='tomato', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Representation Similarity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.show()