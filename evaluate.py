import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import umap

from src.data.ptbxl   import load_ptbxl
from src.data.dataset import make_dataloader, normalise
from src.models.ssl   import SSLModel


def extract_features(model, dataloader, device):
    model.eval()
    features, indices = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)
            z = model.encode(x)
            features.append(z.cpu().numpy())
            indices.extend([i] * x.shape[0])
    return np.vstack(features), np.array(indices)


def plot_umap(features, indices, title='UMAP'):
    reducer   = umap.UMAP(n_components=2, random_state=42, n_neighbors=10)
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1],
        c=indices, cmap='tab20', alpha=0.6, s=15
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ","_")}.png', dpi=150)
    plt.show()
    print(f"✓ UMAP saved to results/")


def policy_preferences(model, dataloader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            probs, _ = model.policy(x)
            all_probs.append(probs.cpu().numpy())

    mean_probs = np.vstack(all_probs).mean(axis=0)
    aug_names  = ['jitter','scaling','ch_dropout',
                  'freq_mask','temp_crop','time_warp']

    print("\n=== Learned Augmentation Preferences ===\n")
    for name, prob in sorted(
        zip(aug_names, mean_probs), key=lambda x: -x[1]
    ):
        bar = '█' * int(prob * 60)
        print(f"  {name:15s}  {prob:.4f}  {bar}")

    plt.figure(figsize=(8, 4))
    sorted_pairs = sorted(
        zip(aug_names, mean_probs), key=lambda x: -x[1]
    )
    names, probs = zip(*sorted_pairs)
    plt.bar(names, probs, color='steelblue', alpha=0.8)
    plt.ylabel('Mean selection probability')
    plt.title('What the policy learned to prefer')
    plt.tight_layout()
    plt.savefig('results/policy_preferences.png', dpi=150)
    plt.show()


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')

    # load data
    signals, labels = load_ptbxl(
        config['data']['ptbxl_dir'],
        max_records=config['data']['max_records']
    )
    signals = normalise(signals, mode=config["data"]["norm_mode"])
    loader  = make_dataloader(
        signals, labels,
        window_len=config['data']['window_len'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        norm_mode=config["data"]["norm_mode"]
    )

    # load trained model
    model = SSLModel(
        in_channels=config['model']['in_channels'],
        hidden_dim=config['model']['hidden_dim'],
        out_dim=config['model']['out_dim'],
        n_augmentations=config['model']['n_augmentations'],
    )
    model.load(config['paths']['checkpoint'], device=device)

    # 1 — extract features
    print("\nExtracting representations...")
    features, indices = extract_features(model, loader, device)
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Std across samples: {features.std():.4f}")

    # 2 — UMAP
    print("\nRunning UMAP...")
    plot_umap(features, indices, title='SSL Representations')

    # 3 — policy preferences
    print("\nAnalysing policy preferences...")
    policy_preferences(model, loader, device)

    # 4 — PCA analysis
    print("\n=== PCA Analysis ===")
    pca = PCA(n_components=15)
    features_pca = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"✓ 15 components explain {explained[-1]*100:.1f}% of variance")
    print(f"✓ 5 components explain  {explained[4]*100:.1f}% of variance")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 16), explained * 100,
             marker='o', color='steelblue')
    plt.xlabel('Number of PCA components')
    plt.ylabel('Cumulative explained variance (%)')
    plt.title('PCA on learned representations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/pca_variance.png', dpi=150)
    plt.show()
    print("✓ PCA plot saved")


if __name__ == '__main__':
    main()