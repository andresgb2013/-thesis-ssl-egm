import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalise_by_window(signals):
    """
    Zero-mean unit-variance per window per channel.
    Standard for ECG/PTB-XL — used during development.
    WARNING: destroys absolute amplitude differences between windows.
    NOT recommended for Charité EGM data.
    """
    mean = signals.mean(axis=1, keepdims=True)
    std  = signals.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (signals - mean) / std


def normalise_by_patient(signals):
    """
    Zero-mean unit-variance per patient per channel.
    Preserves intra-patient amplitude variation across windows.
    RECOMMENDED for Charité EGM data — preserves pre/post ablation
    amplitude differences needed for Δ64 computation.
    """
    result = np.zeros_like(signals)
    for i, sig in enumerate(signals):
        mean = sig.mean(axis=0, keepdims=True)   # (1, C)
        std  = sig.std(axis=0, keepdims=True)    # (1, C)
        std[std == 0] = 1
        result[i] = (sig - mean) / std
    return result


def normalise_global(signals):
    """
    Zero-mean unit-variance using dataset-wide statistics per channel.
    Preserves relative amplitude differences between patients and windows.
    Good middle ground between window and patient normalisation.
    """
    mean = signals.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
    std  = signals.std(axis=(0, 1), keepdims=True)   # (1, 1, C)
    std[std == 0] = 1
    return (signals - mean) / std


def normalise(signals, mode='window'):
    """
    Unified normalisation entry point.
    
    mode options:
      'window'  — per window (PTB-XL development, standard SSL)
      'patient' — per patient (Charité EGM, preserves ablation response)
      'global'  — dataset-wide (good middle ground)
    """
    if mode == 'window':
        return normalise_by_window(signals)
    elif mode == 'patient':
        return normalise_by_patient(signals)
    elif mode == 'global':
        return normalise_global(signals)
    else:
        raise ValueError(f"Unknown normalisation mode: {mode}. "
                         f"Choose 'window', 'patient', or 'global'")


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


def extract_windows(signals, window_len=200, overlap=0.5):
    """
    Extract overlapping windows from all records.
    Returns (n_windows, window_len, channels)
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
                    window_len=200, overlap=0.5,
                    batch_size=32, shuffle=True,
                    norm_mode='window'):
    """
    norm_mode: 'window' for PTB-XL, 'patient' for Charité EGM
    """
    signals = normalise(signals, mode=norm_mode)
    windows, win_labels = extract_windows(signals, window_len, overlap)

    n_windows = len(windows)
    n_batches = n_windows // batch_size

    print(f"✓ Normalisation: {norm_mode}")
    print(f"✓ Overlap={overlap:.0%} | Windows={n_windows} | "
          f"Batches={n_batches} | window_len={window_len}")

    dataset = SignalDataset(windows, win_labels)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True
    )
    return loader