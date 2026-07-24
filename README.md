# thesis-ssl-egm

**Self-Supervised Representation Learning for Intracardiac Electrophysiological Signals**

> Master's Thesis — Work in Progress  
> Andrés · MSc Computer Science (Big Data & AI) · SRH Berlin University of Applied Sciences  
> In collaboration with **Charité – Universitätsmedizin Berlin / DHZC**

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
| Flexible normalisation (window / patient / global) | ✅ Complete |
| Overlap windowing experiment | ✅ Complete — 50% optimal |
| UMAP + policy analysis | ✅ Complete |
| GPU training on Kaggle (T4) | ✅ Running |
| Stage 2 — Δ64 + XGBoost | 🔜 Planned |
| Charité data integration | ⏳ Pending data access |

---

## Experimental Results

### Overlap Experiment (500 records, window_len=200, 150 epochs, CPU)

Five windowing strategies compared — 10% to 50% overlap:

| Overlap | Windows | Contrastive Loss | Cos Similarity |
|---|---|---|---|
| 10% | 2,500 | 0.0140 | 0.9031 |
| 20% | 3,000 | 0.0106 | 0.9086 |
| 30% | 3,000 | 0.0105 | 0.9063 |
| 40% | 3,500 | 0.0115 | 0.9101 |
| **50%** | **4,500** | **0.0116** | **0.9120 ← best** |

50% overlap aligns with the PulseSelect document's design choice and is validated empirically.

### Full Training Run (30% PTB-XL, Kaggle T4 GPU)

| Setting | Value |
|---|---|
| Records | 6,551 (30% of PTB-XL, random sample) |
| Windows | ~59,000 |
| Epochs | 150 |
| Batch size | 256 |
| Device | Tesla T4 (Kaggle) |
| Final contrastive loss | 0.0201 |
| Final cosine similarity | **0.9145** |
| Normalisation | global |

Training curve:
```
Epoch  10 → Cos: 0.8981
Epoch  50 → Cos: 0.9088
Epoch 100 → Cos: 0.9130
Epoch 150 → Cos: 0.9145
```

### Learned Augmentation Preferences

What the policy learned to prefer without supervision:

```
temp_crop    0.205  ← most preferred
freq_mask    0.164
ch_dropout   0.164
jitter       0.162
scaling      0.153
time_warp    0.152  ← least preferred
```

Temporal augmentations are consistently preferred — consistent with the clinical requirement to preserve morphological signal features in EGMs.

---

## Repository Structure

```
thesis-ssl-egm/
├── config.yaml                # All hyperparameters and paths
├── train.py                   # Training entry point
├── evaluate.py                # UMAP, policy analysis, PCA
├── overlap_experiment.py      # Windowing overlap comparison
├── src/
│   ├── data/
│   │   ├── dataset.py         # Windowed SignalDataset + 3 normalisation modes
│   │   ├── ptbxl.py           # PTB-XL loader (records100 only, random sample)
│   │   └── charite.py         # Charité ASCII loader (stub)
│   ├── models/
│   │   ├── augmentations.py   # 6 signal transformations
│   │   ├── policy.py          # Learnable augmentation policy
│   │   ├── encoder.py         # Dilated CNN encoder
│   │   └── ssl.py             # Full SSL model
│   ├── training/
│   │   ├── losses.py          # InfoNCE contrastive loss
│   │   └── trainer.py         # Training loop
│   └── evaluation/
└── results/                   # Models + plots (gitignored)
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

Edit `config.yaml`:

```yaml
data:
  source: ptbxl        # or charite
  max_records: 6551    # null = all records
  window_len: 200
  overlap: 0.5
  norm_mode: global    # window / patient / global

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

### Overlap experiment
```bash
python overlap_experiment.py
```

### Training on Kaggle (recommended)

PTB-XL is available natively on Kaggle — no download needed:

```python
os.symlink(
    '/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1',
    'data/raw/ptb-xl'
)
```

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Learnable augmentation policy | Augmentation semantics unknown for intracardiac EGMs — learning them is the thesis CS contribution |
| Dilated CNN encoder | Captures local morphology and long-range temporal patterns on 1D signals |
| GroupNorm over BatchNorm | Works with small batches, prevents collapse |
| Single GPU training | DataParallel with GroupNorm causes representation collapse across GPUs |
| Contrastive-only loss | Reconstruction loss unstable with 64-dim bottleneck on normalised signals |
| 50% overlap | Empirically validated — best cosine similarity across 5 strategies |
| Per-patient normalisation for Charité | Preserves absolute amplitude differences needed for Δ64 computation |
| Phase-neutral training | Pre/post pairs NOT used as positives — preserves ablation-induced change in Δ64 |
| LOPO-CV in Stage 2 | Standard for small clinical cohorts (n=65), prevents patient-level leakage |

---

## Normalisation Strategy

Three modes are supported — switchable via `config.yaml`:

| Mode | Use case | Behaviour |
|---|---|---|
| `window` | PTB-XL development (standard SSL) | Zero-mean unit-variance per window |
| `patient` | **Charité EGM (recommended)** | Zero-mean unit-variance per patient — preserves pre/post amplitude differences for Δ64 |
| `global` | Middle ground | Dataset-wide statistics per channel |

> ⚠️ Per-window normalisation destroys absolute amplitude differences between windows. For intracardiac EGMs, low-amplitude signals carry clinical significance and the pre/post ablation amplitude change is a key component of Δ64. Per-patient normalisation is strongly recommended for Charité data.

---

## Δ64 Pipeline (Stage 2 — planned)

```
Pre-ablation windows  → Encoder → mean(z_pre)  ┐
                                                 ├→ Δ64 = mean(z_post) - mean(z_pre) → XGBoost → AF recurrence (Yes/No)
Post-ablation windows → Encoder → mean(z_post) ┘
```

---

## Infrastructure Notes

Development and small experiments run locally (Intel i7-1165G7, 16GB RAM, CPU only). Full training runs on **Kaggle Notebooks** (T4 GPU, free tier) where PTB-XL is available as a native dataset. Google Colab was tested but hit GPU usage limits and storage constraints.

Known issue: repository name contains a leading dash (`-thesis-ssl-egm`) which requires using the full path when navigating after cloning on Kaggle: `os.chdir('/kaggle/working/-thesis-ssl-egm')`.

---

## References

### Foundational SSL
- Chen et al. (2020). SimCLR — A Simple Framework for Contrastive Learning · [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- van den Oord et al. (2018). CPC / InfoNCE — Representation Learning with Contrastive Predictive Coding · [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- He et al. (2022). MAE — Masked Autoencoders Are Scalable Vision Learners · [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

### SSL for Time Series & Medical Signals
- Liang et al. (2023). Self-Supervised Contrastive Learning for Medical Time Series: A Systematic Review · [DOI:10.3390/s23094221](https://doi.org/10.3390/s23094221)
- Manimaran et al. (2024). NERULA: Dual-Pathway SSL for ECG · [arXiv:2405.19348](https://arxiv.org/abs/2405.19348)
- Wang et al. (2023). COMET: Contrast Everything — Hierarchical Contrastive for Medical Time Series · [arXiv:2310.14017](https://arxiv.org/abs/2310.14017)
- Tonekaboni et al. (2021). TNC — Unsupervised Representation Learning for Time Series · [arXiv:2106.00750](https://arxiv.org/abs/2106.00750)
- Yue et al. (2022). TS2Vec — Universal Representation of Time Series · [arXiv:2106.10466](https://arxiv.org/abs/2106.10466)
- Eldele et al. (2021). TS-TCC — Self-Supervised Contrastive for Semi-Supervised Time Series · [arXiv:2208.06616](https://arxiv.org/abs/2208.06616)
- Siontis et al. (2024). Foundation Transformer for ECG-Based Cardiac Assessment · [PMC12724683](https://pmc.ncbi.nlm.nih.gov/articles/PMC12724683/)

### Learnable Augmentation
- Cubuk et al. (2019). AutoAugment — Learning Augmentation Strategies from Data · [arXiv:1805.09501](https://arxiv.org/abs/1805.09501)
- Wang et al. (2024). AutoTCL — Parametric Augmentation for Time Series Contrastive Learning · [arXiv:2402.10434](https://arxiv.org/abs/2402.10434)
- Liu et al. (2024). Guidelines for Augmentation Selection in Contrastive Learning for Time Series · [arXiv:2407.09336](https://arxiv.org/abs/2407.09336)
- Gao & Lin (2024). Data Augmentation for Time-Series Classification: Survey · [arXiv:2310.10060](https://arxiv.org/abs/2310.10060)
- Iglesias et al. (2023). Data Augmentation Techniques in Time Series: Survey and Taxonomy · [DOI:10.1007/s00521-023-08459-3](https://doi.org/10.1007/s00521-023-08459-3)

### Intracardiac EGM & Clinical Context
- Hejč et al. (2024). Multi-channel Delineation of Intracardiac Electrograms · [DOI:10.1016/j.bspc.2024.106274](https://doi.org/10.1016/j.bspc.2024.106274)
- Kolk, Tjong et al. (2023). ML of Electrophysiological Signals for Ventricular Arrhythmia Prediction · [DOI:10.1016/j.ebiom.2023.104462](https://doi.org/10.1016/j.ebiom.2023.104462)
- Del Val et al. (2026). ML Prediction of Outcome Following PFA Ablation · [Europace](https://academic.oup.com/europace/article/28/5/euag053/8678359)
- Mulder et al. (2026). ML for Risk Stratification of AF Recurrence After PFA · [BMC Cardiovascular Disorders](https://link.springer.com/article/10.1186/s12872-026-05666-3)
- Liu et al. (2025). Predicting AF Ablation Outcomes with XGBoost · [JMIR Cardio](https://cardio.jmir.org/2025/1/e77380)
Contact

Andrés · andresgb2013@gmail.com
