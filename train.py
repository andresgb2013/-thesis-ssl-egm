import yaml
import torch
from src.data.ptbxl   import load_ptbxl
from src.data.dataset import make_dataloader
from src.models.ssl   import SSLModel
from src.training.trainer import Trainer


def main():
    # ── load config ──────────────────────────────────
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────
    source = config['data']['source']

    if source == 'ptbxl':
        signals, labels = load_ptbxl(
            config['data']['ptbxl_dir'],
            max_records=config['data']['max_records']
        )
    elif source == 'charite':
        from src.data.charite import load_charite
        signals, labels = load_charite(
            config['data']['charite_dir'],
            max_patients=config['data']['max_records']
        )
    else:
        raise ValueError(f"Unknown data source: {source}")

    loader = make_dataloader(
        signals, labels,
        window_len=config['data']['window_len'],
        overlap=config['data']['overlap'],
        batch_size=config['training']['batch_size'],
        norm_mode=config['data']['norm_mode'],   # ← añade esta línea
)

    # ── model ─────────────────────────────────────────
    model = SSLModel(
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        n_augmentations=config['model']['n_augmentations'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")

    # ── train ─────────────────────────────────────────
    trainer = Trainer(model, config, device=device)
    history = trainer.train(loader)

    print("\n✓ Done. Model saved to", config['paths']['checkpoint'])


if __name__ == '__main__':
    main()