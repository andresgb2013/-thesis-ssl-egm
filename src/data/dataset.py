import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalise(signals):
    mean = signals.mean(axis=1, keepdims=True)
    std  = signals.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (signals - mean) / std


class SignalDataset(Dataset):
    def __init__(self, windows, labels=None):
        self.windows = torch.FloatTensor(windows)
        self.labels  = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return (x,)


def extract_windows(signals, window_len=500, overlap=0.5):
    """
    Extract all windows from all records.
    Returns array of shape (n_windows, window_len, channels)
    """
    step = int(window_len * (1 - overlap))
    all_windows = []
    all_labels  = []

    for i, sig in enumerate(signals):
        T = sig.shape[0]
        start = 0
        while start + window_len <= T:
            all_windows.append(sig[start:start + window_len])
            all_labels.append(i)
            start += step

    all_windows = np.stack(all_windows, axis=0)
    return all_windows, all_labels


def make_dataloader(signals, labels=None,
                    window_len=500, overlap=0.5,
                    batch_size=32, shuffle=True):
    signals  = normalise(signals)
    windows, win_labels = extract_windows(
        signals, window_len, overlap
    )

    n_windows = len(windows)
    n_batches = n_windows // batch_size

    print(f"✓ Overlap={overlap:.0%} | "
          f"Windows={n_windows} | "
          f"Batches={n_batches} | "
          f"window_len={window_len}")

    dataset = SignalDataset(windows, win_labels)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True
    )
    return loader