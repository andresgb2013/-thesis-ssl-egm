import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalise_by_window(signals):
    mean = signals.mean(axis=1, keepdims=True)
    std  = signals.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (signals - mean) / std


def normalise_by_patient(signals):
    result = np.zeros_like(signals)
    for i, sig in enumerate(signals):
        mean = sig.mean(axis=0, keepdims=True)
        std  = sig.std(axis=0, keepdims=True)
        std[std == 0] = 1
        result[i] = (sig - mean) / std
    return result


def normalise_global(signals):
    mean = signals.mean(axis=(0, 1), keepdims=True)
    std  = signals.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1
    return (signals - mean) / std


def normalise(signals, mode="window"):
    if mode == "window":
        return normalise_by_window(signals)
    elif mode == "patient":
        return normalise_by_patient(signals)
    elif mode == "global":
        return normalise_global(signals)
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
                    norm_mode="window"):
    signals = normalise(signals, mode=norm_mode)
    windows, win_labels = extract_windows(signals, window_len, overlap)
    n_windows = len(windows)
    n_batches = n_windows // batch_size
    print(f"Normalisation: {norm_mode}")
    print(f"Overlap={overlap:.0%} | Windows={n_windows} | Batches={n_batches}")
    dataset = SignalDataset(windows, win_labels)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True, drop_last=True
    )
    return loader
