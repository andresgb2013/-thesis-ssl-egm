import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalise(signals):
    """Zero-mean unit-variance per window per channel"""
    mean = signals.mean(axis=1, keepdims=True)
    std  = signals.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (signals - mean) / std


class SignalDataset(Dataset):
    """
    Generic dataset for any multi-channel time-series.
    Works for both PTB-XL and Charité data.
    signals: np.array (n_records, timesteps, channels)
    labels:  optional — only used for evaluation, never SSL training
    """
    def __init__(self, signals, labels=None, window_len=500):
        self.signals    = torch.FloatTensor(signals)
        self.labels     = labels
        self.window_len = window_len

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]

        # random window crop for diversity
        if x.shape[0] > self.window_len:
            start = torch.randint(
                0, x.shape[0] - self.window_len, (1,)
            ).item()
            x = x[start:start + self.window_len]

        if self.labels is not None:
            return x, self.labels[idx]
        return (x,)


def make_dataloader(signals, labels=None,
                    window_len=500, batch_size=32,
                    shuffle=True):
    signals = normalise(signals)
    dataset = SignalDataset(signals, labels, window_len)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True     # keeps batch size consistent for InfoNCE
    )
    print(f"✓ Dataset: {len(dataset)} records | "
          f"{len(loader)} batches | "
          f"window={window_len} | batch={batch_size}")
    return loader