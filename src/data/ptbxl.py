import os
import numpy as np
import wfdb


def load_ptbxl(data_dir, max_records=None):
    """Load PTB-XL .hea/.dat records from all subfolders"""
    all_hea = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.hea'):
                all_hea.append(os.path.join(root, f))

    all_hea = sorted(all_hea)
    if max_records:
        all_hea = all_hea[:max_records]

    print(f"Loading {len(all_hea)} PTB-XL records...")
    signals, labels = [], []

    for i, hea_path in enumerate(all_hea):
        try:
            record = wfdb.rdrecord(hea_path.replace('.hea', ''))
            sig    = record.p_signal
            if sig is not None and np.isnan(sig).sum() == 0:
                signals.append(sig.astype(np.float32))
                labels.append(0)
        except Exception:
            pass

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(all_hea)}")

    signals = np.stack(signals, axis=0)
    print(f"✓ Loaded {len(signals)} records — shape: {signals.shape}")
    return signals, labels