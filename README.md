# thesis-ssl-egm

**Self-Supervised Representation Learning for Intracardiac Electrophysiological Signals**

> Master's Thesis — Work in Progress  
> Andrés · MSc Computer Science (Big Data & AI) · 

---

## What this is

This repository contains the implementation of a self-supervised learning (SSL) framework developed as part of the **PulseSelect EGM project** at Charité. The goal is to learn meaningful representations from intracardiac electrophysiological signals recorded during Pulsed Field Ablation (PFA) procedures — without requiring expert annotations.

The framework follows the two-stage pipeline from the PulseSelect EGM project:

- **Stage 1 (this repo)** — SSL pretraining on unlabelled 5-second signal windows using a learnable augmentation policy + dilated CNN encoder
- **Stage 2 (planned)** — Patient-level AF recurrence prediction using Δ64 representations + clinical variables via XGBoost

---

## Current Status

| Component | Status |
|---|---|
| PTB-XL data loader | ✅ Complete |
| Charité ASCII loader | 🔄 Stub ready — awaiting data |
| 6 signal augmentations | ✅ Complete |
| Learnable augmentation policy | ✅ Complete |
| Dilated CNN encoder | ✅ Complete |
| Contrastive SSL training (InfoNCE) | ✅ Complete |
| Overlap windowing experiment | ✅ Complete — 50% optimal |
| UMAP + policy analysis | ✅ Complete |
| Stage 2 — Δ64 + XGBoost | 🔜 Planned |
| Charité data integration | ⏳ Pending data access |

---

## Key Results So Far (PTB-XL, 500 records)

**Overlap experiment** — 5 windowing strategies compared (10% to 50%):

| Overlap | Windows | Contrastive Loss | Cos Similarity |
|---|---|---|---|
| 10% | 2,500 | 0.0140 | 0.9031 |
| 20% | 3,000 | 0.0106 | 0.9086 |
| 30% | 3,000 | 0.0105 | 0.9063 |
| 40% | 3,500 | 0.0115 | 0.9101 |
| **50%** | **4,500** | **0.0116** | **0.9120 ← best** |

50% overlap aligns with the PulseSelect document's design choice and is confirmed empirically.

**Learned augmentation preferences** (what the policy discovered without supervision):
```
temp_crop    0.179  ← most preferred
time_warp    0.178
jitter       0.174
scaling      0.166
ch_dropout   0.162
freq_mask    0.141  ← least preferred
```

The policy learned to prefer temporal augmentations over frequency masking — consistent with the clinical requirement to preserve morphological signal features.

---

## Repository Structure

```
thesis-ssl-egm/
├── config.yaml              # All hyperparameters and paths
├── train.py                 # Training entry point
├── evaluate.py              # UMAP, policy analysis, PCA
├── overlap_experiment.py    # Windowing overlap comparison
├── src/
│   ├── data/
│   │   ├── dataset.py       # Generic windowed SignalDataset
│   │   ├── ptbxl.py         # PTB-XL loader
│   │   └── charite.py       # Charité ASCII loader (stub)
│   ├── models/
│   │   ├── augmentations.py # 6 signal transformations
│   │   ├── policy.py        # Learnable augmentation policy
│   │   ├── encoder.py       # Dilated CNN encoder
│   │   └── ssl.py           # Full SSL model
│   ├── training/
│   │   ├── losses.py        # InfoNCE contrastive loss
│   │   └── trainer.py       # Training loop
│   └── evaluation/
└── results/                 # Models + plots (gitignored)
```

---

## Setup

```bash
conda create -n thesis python=3.10
conda activate thesis
pip install torch torchvision torchaudio
pip install numpy pandas scipy matplotlib seaborn
pip install scikit-learn tqdm einops pyyaml
pip install wfdb neurokit2 umap-learn
```

---

## Usage

### Configure
Edit `config.yaml` — switch between PTB-XL and Charité:
```yaml
data:
  source: ptbxl        # or charite
  max_records: 500     # null = all records
  window_len: 200
  overlap: 0.5
model:
  in_channels: 12      # 12 PTB-XL, 25 Charité
training:
  reconstruction_weight: 0.0   # contrastive only
```

### Train
```bash
python train.py
```

### Evaluate
```bash
python evaluate.py
```

### Run overlap experiment
```bash
python overlap_experiment.py
```

---

## Architecture

### Learnable Augmentation Policy
A small CNN that reads signal properties and learns which of 6 augmentations to apply and at what intensity — trained jointly with the encoder. This is the core CS contribution: augmentation strategy as a learned component, not a fixed design choice.

### Dilated CNN Encoder
1D CNN with dilated convolutions capturing structure at multiple temporal scales:
- dilation=1 → local morphology
- dilation=2 → beat shape  
- dilation=4 → rhythm patterns
- dilation=8 → global signal structure

Outputs a 64-dimensional latent vector z per window.

### Δ64 Pipeline (Stage 2 — planned)
```
Pre-ablation windows  → Encoder → mean(z_pre)  ┐
                                                 ├→ Δ64 → XGBoost → AF recurrence
Post-ablation windows → Encoder → mean(z_post) ┘
```

---

## Design Choices

| Choice | Rationale |
|---|---|
| Learnable augmentation policy | Augmentation semantics unknown for intracardiac EGMs |
| Dilated CNN | Captures local + long-range temporal patterns on 1D signals |
| GroupNorm over BatchNorm | Works with small batches, prevents collapse |
| Contrastive-only loss | Reconstruction loss unstable with 64-dim bottleneck on normalised ECG |
| Phase-neutral training | Pre/post pairs NOT used as positives — preserves ablation-induced change in Δ64 |
| 50% overlap | Empirically validated — best cosine similarity across 5 strategies |
| LOPO-CV in Stage 2 | Standard for small clinical cohorts (n=65), prevents patient-level leakage |

---

## References

- Chen et al. (2020). SimCLR · [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Wang et al. (2024). AutoTCL · [arXiv:2402.10434](https://arxiv.org/abs/2402.10434)
- Hejč et al. (2024). Intracardiac EGM · [DOI:10.1016/j.bspc.2024.106274](https://doi.org/10.1016/j.bspc.2024.106274)
- Liang et al. (2023). SSL medical time series · [DOI:10.3390/s23094221](https://doi.org/10.3390/s23094221)
- van den Oord et al. (2018). InfoNCE · [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

---

## Contact

**Andrés** · andresgb2013@gmail.com  

