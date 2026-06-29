import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.ptbxl    import load_ptbxl
from src.data.dataset  import make_dataloader, normalise
from src.models.ssl    import SSLModel
from src.training.trainer import Trainer


def run_one(overlap, config, signals, labels):
    loader = make_dataloader(
        signals, labels,
        window_len=config['data']['window_len'],
        overlap=overlap,
        batch_size=config['training']['batch_size'],
    )

    model = SSLModel(
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        n_augmentations=config['model']['n_augmentations'],
    )

    trainer = Trainer(model, config, device=torch.device('cpu'))
    history = trainer.train(loader)
    model.save(f"results/encoder_overlap_{int(overlap*100)}.pt")
    return history, len(loader.dataset)


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # fast settings for comparison
    config['training']['epochs'] = 20
    config['training']['reconstruction_weight'] = 0.0

    print("Loading data...")
    signals, labels = load_ptbxl(
        config['data']['ptbxl_dir'],
        max_records=200    # 200 records is enough for comparison
    )
    signals = normalise(signals)

    overlaps = [0.1, 0.2, 0.3, 0.4, 0.5]
    results  = {}
    n_windows = {}

    for ov in overlaps:
        print(f"\n{'='*50}")
        print(f"  Running overlap = {ov:.0%}")
        print(f"{'='*50}")
        history, nw = run_one(ov, config, signals, labels)
        results[ov]  = history
        n_windows[ov] = nw

    # ── Summary table ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  OVERLAP EXPERIMENT RESULTS")
    print(f"{'='*65}")
    print(f"  {'Overlap':>8} | {'Windows':>8} | "
          f"{'Final Loss':>12} | {'Cos Sim':>10}")
    print(f"  {'-'*55}")
    for ov in overlaps:
        final = results[ov][-1]
        print(f"  {ov:>7.0%} | {n_windows[ov]:>8} | "
              f"{final['c_loss']:>12.4f} | "
              f"{final['cos_sim']:>10.4f}")

    # ── Plot ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#4f8ef7','#50c8a8','#f0a040','#e05c7a','#a060f0']

    for ov, color in zip(overlaps, colors):
        epochs   = [h['epoch']   for h in results[ov]]
        c_losses = [h['c_loss']  for h in results[ov]]
        cos_sims = [h['cos_sim'] for h in results[ov]]

        axes[0].plot(epochs, c_losses,
                     label=f'{ov:.0%} ({n_windows[ov]} windows)',
                     color=color, linewidth=1.5)
        axes[1].plot(epochs, cos_sims,
                     label=f'{ov:.0%} ({n_windows[ov]} windows)',
                     color=color, linewidth=1.5)

    axes[0].set_title('Contrastive Loss by Overlap')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Cosine Similarity by Overlap')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Overlap Experiment — 200 PTB-XL records, 20 epochs',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('results/overlap_comparison.png', dpi=150)
    plt.show()
    print("\n✓ Plot saved to results/overlap_comparison.png")


if __name__ == '__main__':
    main()