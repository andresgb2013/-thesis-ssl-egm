import os
import numpy as np


# ─── Channel layout from PulseSelect document ───────
# 12 ECG + 8 PFA catheter + 5 CS catheter = 25 total
CHARITE_CHANNELS = (
    ['ECG_I','ECG_II','ECG_III',
     'ECG_aVR','ECG_aVL','ECG_aVF',
     'ECG_V1','ECG_V2','ECG_V3',
     'ECG_V4','ECG_V5','ECG_V6']   # 12 ECG
    + [f'PFA_{i+1}' for i in range(8)]  # 8 PFA
    + [f'CS_{i+1}'  for i in range(5)]  # 5 CS
)  # total = 25


def parse_charite_file(filepath):
    """
    Parse one Charité ASCII .txt file.
    Format (from document page 4):
      - Header lines: Label, Range, Low, High,
                      Sample rate, Color, Scale, Channel #
      - [Data] section: comma-separated values,
                        one row per timestep, one col per channel
    Returns: np.array (timesteps, n_channels), float32
    """
    data_lines = []
    in_data    = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Data]':
                in_data = True
                continue
            if in_data and line:
                values = [float(v) for v in line.split(',')]
                data_lines.append(values)

    if not data_lines:
        raise ValueError(f"No data found in {filepath}")

    return np.array(data_lines, dtype=np.float32)


def load_charite(data_dir, max_patients=None):
    """
    Load Charité ASCII files.
    Expected structure:
      data_dir/
        pat1_pfa_seg1.txt
        pat1_pfa_seg2.txt
        ...
    Returns signals (n_files, timesteps, 25), labels
    """
    # ── Update this when you receive the real files ──
    txt_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.txt')
    ])

    if max_patients:
        txt_files = txt_files[:max_patients]

    print(f"Loading {len(txt_files)} Charité files...")
    signals, labels = [], []

    for path in txt_files:
        try:
            sig = parse_charite_file(path)
            if np.isnan(sig).sum() == 0:
                signals.append(sig)
                labels.append(0)    # placeholder — AF recurrence TBD
        except Exception as e:
            print(f"  Skipped {os.path.basename(path)}: {e}")

    signals = np.stack(signals, axis=0)
    print(f"✓ Loaded {len(signals)} files — shape: {signals.shape}")
    return signals, labels