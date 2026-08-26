import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.ptbxl       import load_ptbxl
from src.data.dataset     import make_dataloader, normalise
from src.models.ssl       import SSLModel
from src.models.augmentations import Augmentations
from src.training.losses  import info_nce_loss


# ─── Fixed augmentation wrappers ─────────────────────────
aug = Augmentations()

FIXED_AUGS = {
    'no_augmentation':  lambda x: x,
    'random':           lambda x: aug.jitter(x) if torch.rand(1) > 0.5 else aug.temporal_crop(x),
    'jitter':           lambda x: aug.jitter(x, sigma=0.1),
    'scaling':          lambda x: aug.scaling(x, sigma=0.15),
    'freq_mask':        lambda x: aug.freq_mask(x, F=20),
    'temp_crop':        lambda x: aug.temporal_crop(x, crop_ratio=0.85),
    'time_warp':        lambda x: aug.time_warp(x, sigma=0.15),
}


# ─── Training function ────────────────────────────────────
def train_fixed(aug_name, aug_fn, dataloader, config, device, epochs=50):
    """Train encoder with a fixed augmentation strategy"""
    print(f"\n{'='*50}")
    print(f"  {aug_name}")
    print(f"{'='*50}")

    model = SSLModel(
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        n_augmentations=config['model']['n_augmentations'],
    ).to(device)

    optimizer = optim.Adam(
        model.encoder.parameters(),   # only train encoder, not policy
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_cos  = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            # two fixed augmented views
            x1 = aug_fn(x)
            x2 = aug_fn(x)

            z1 = model.encoder(x1)
            z2 = model.encoder(x2)

            loss = info_nce_loss(z1, z2,
                temperature=config['training']['temperature'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.encoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cos  += F.cosine_similarity(z1, z2).mean().item()

        scheduler.step()
        n = len(dataloader)
        history.append({
            'epoch':   epoch + 1,
            'loss':    epoch_loss / n,
            'cos_sim': epoch_cos  / n,
        })

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {epoch_loss/n:.4f} | "
                  f"Cos: {epoch_cos/n:.4f}")

    return model, history


def evaluate_discrimination(model, signals, device, n_patients=20):
    """
    Measure intra vs inter patient similarity.
    This is the key metric — not just cosine similarity during training.
    """
    model.eval()

    def get_windows(sig, n=5, window_len=200):
        windows = []
        for i in range(n):
            start = i * 100
            windows.append(sig[start:start + window_len])
        return torch.FloatTensor(np.stack(windows))

    intra_sims = []
    inter_sims = []

    with torch.no_grad():
        for i in range(min(n_patients, len(signals) - 10)):
            w_same = get_windows(signals[i]).to(device)
            w_diff = get_windows(signals[i + 10]).to(device)

            z_same = F.normalize(model.encoder(w_same), dim=1)
            z_diff = F.normalize(model.encoder(w_diff), dim=1)

            intra = F.cosine_similarity(z_same[:3], z_same[2:]).mean().item()
            inter = F.cosine_similarity(z_same[:3], z_diff[:3]).mean().item()

            intra_sims.append(intra)
            inter_sims.append(inter)

    return {
        'intra': np.mean(intra_sims),
        'inter': np.mean(inter_sims),
        'gap':   np.mean(intra_sims) - np.mean(inter_sims),
    }


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs  = 50   # enough to compare — not full 150
    print(f"Device: {device} | Epochs per model: {epochs}")

    # load data — use subset for speed
    signals, labels = load_ptbxl(
        config['data']['ptbxl_dir'],
        max_records=3000   # 3k records — fast but meaningful
    )
    signals = normalise(signals, mode=config['data']['norm_mode'])

    loader = make_dataloader(
        signals, labels,
        window_len=config['data']['window_len'],
        overlap=config['data']['overlap'],
        batch_size=config['training']['batch_size'],
        norm_mode=config['data']['norm_mode'],
    )

    results = {}

    # ── 1. Learned policy (load pretrained) ──────────────
    print(f"\n{'='*50}")
    print(f"  learned_policy (pretrained)")
    print(f"{'='*50}")

    learned_model = SSLModel(
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        n_augmentations=config['model']['n_augmentations'],
    )
    learned_model.load(config['paths']['checkpoint'], device=str(device))
    learned_model = learned_model.to(device)

    disc = evaluate_discrimination(learned_model, signals, device)
    results['learned_policy'] = {
        'final_cos': 0.9342,   # from full training run
        'intra': disc['intra'],
        'inter': disc['inter'],
        'gap':   disc['gap'],
    }
    print(f"  Intra: {disc['intra']:.4f} | Inter: {disc['inter']:.4f} | Gap: {disc['gap']:.4f}")

    # ── 2. Fixed augmentations ────────────────────────────
    for aug_name, aug_fn in FIXED_AUGS.items():
        model, history = train_fixed(
            aug_name, aug_fn, loader, config, device, epochs=epochs
        )
        disc = evaluate_discrimination(model, signals, device)
        final_cos = history[-1]['cos_sim']

        results[aug_name] = {
            'final_cos': final_cos,
            'intra':     disc['intra'],
            'inter':     disc['inter'],
            'gap':       disc['gap'],
            'history':   history,
        }
        print(f"  → Intra: {disc['intra']:.4f} | Inter: {disc['inter']:.4f} | Gap: {disc['gap']:.4f}")

    # ── Summary table ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"  3,000 records | {epochs} epochs | PTB-XL")
    print(f"{'='*65}")
    print(f"  {'Method':22s} | {'Cos Sim':>8} | {'Intra':>7} | {'Inter':>7} | {'Gap':>7}")
    print(f"  {'-'*60}")

    sorted_results = sorted(
        results.items(), key=lambda x: x[1]['gap'], reverse=True
    )
    for name, r in sorted_results:
        marker = ' ← best' if name == sorted_results[0][0] else ''
        print(f"  {name:22s} | {r['final_cos']:>8.4f} | "
              f"{r['intra']:>7.4f} | {r['inter']:>7.4f} | "
              f"{r['gap']:>7.4f}{marker}")

    # ── Save results ──────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    save = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
            for k, v in results.items()}
    with open(f'results/ablation_{ts}.json', 'w') as f:
        json.dump(save, f, indent=2)

    # ── Plot ──────────────────────────────────────────────
    names = [n for n, _ in sorted_results]
    gaps  = [r['gap'] for _, r in sorted_results]
    intra = [r['intra'] for _, r in sorted_results]
    inter = [r['inter'] for _, r in sorted_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#4f8ef7' if n == 'learned_policy' else '#8a91a8' for n in names]

    axes[0].barh(names, gaps, color=colors)
    axes[0].set_xlabel('Intra-Inter Gap (higher = more discriminative)')
    axes[0].set_title('Discrimination Gap by Augmentation Strategy')
    axes[0].axvline(x=0, color='black', linewidth=0.5)

    x = np.arange(len(names))
    axes[1].barh(x - 0.2, intra, 0.4, label='Intra-patient', color='#4f8ef7')
    axes[1].barh(x + 0.2, inter, 0.4, label='Inter-patient', color='#e05c7a')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_title('Intra vs Inter-patient Similarity')
    axes[1].legend()

    plt.suptitle('Ablation Study — Learned vs Fixed Augmentation', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/ablation_{ts}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Results saved to results/ablation_{ts}.json")
    print(f"✓ Plot saved to results/ablation_{ts}.png")


if __name__ == '__main__':
    main()