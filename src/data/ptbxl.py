import os
import numpy as np
import wfdb
import random

TARGET_LENGTH = 1000

def load_ptbxl(data_dir, max_records=None):
    records100_dir = os.path.join(data_dir, "records100")
    all_hea = []
    for root, dirs, files in os.walk(records100_dir):
        for f in files:
            if f.endswith(".hea"):
                all_hea.append(os.path.join(root, f))

    # muestreo aleatorio de todos los folders
    random.seed(42)
    random.shuffle(all_hea)

    if max_records:
        all_hea = all_hea[:max_records]

    print(f"Loading {len(all_hea)} PTB-XL records (random sample)...")
    signals, labels = [], []

    for hea_path in all_hea:
        try:
            record = wfdb.rdrecord(hea_path.replace(".hea", ""))
            sig = record.p_signal
            if sig is None or np.isnan(sig).sum() > 0:
                continue
            if sig.shape[1] != 12:
                continue
            T = sig.shape[0]
            if T < TARGET_LENGTH:
                pad = np.zeros((TARGET_LENGTH - T, 12), dtype=np.float32)
                sig = np.vstack([sig, pad])
            elif T > TARGET_LENGTH:
                sig = sig[:TARGET_LENGTH]
            signals.append(sig.astype(np.float32))
            labels.append(0)
        except Exception:
            pass

    signals = np.stack(signals, axis=0)
    print(f"Loaded {len(signals)} records - shape: {signals.shape}")
    return signals, labels
