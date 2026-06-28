# Deep Learning for EEG-Based Motor Imagery BCI
### From Benchmark Architectures to Hardware-Aware Electrode Reduction

> **Graduation Project — E-JUST CSIT / AI & Data Science**
> Supervised by Dr. Reda Albassiouny & Dr. Sameh Sherif

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Architecture Benchmarking (Semester 1)](#architecture-benchmarking-semester-1)
6. [Selected Architecture: DB-ATCNet (Modified)](#selected-architecture-db-atcnet-modified)
7. [Electrode Reduction Methods (Semester 2)](#electrode-reduction-methods-semester-2)
8. [Results Summary](#results-summary)
9. [FPGA Deployment Notes](#fpga-deployment-notes)
10. [Dependencies & Setup](#dependencies--setup)
11. [Citation](#citation)

---

## Project Overview

This module covers the deep learning component of a wearable Brain-Computer Interface (BCI) system for assistive motion control. The system decodes EEG-based **motor imagery (MI)** — specifically **right-hand vs. left-leg** imagined movements — and translates neural intent into servo actuation commands on an FPGA-embedded platform.

The deep learning work spans two semesters:

- **Semester 1:** Systematic benchmarking of nine deep learning architectures on the PhysioNet EEGMMIDB and BCI Competition IV 2a/2b datasets.
- **Semester 2:** Hardware-aware modification of DB-ATCNet and a three-method electrode reduction study on the HALT dataset targeting a 5-electrode dry-electrode wearable headband.

---

## Repository Structure

```
deep-learning/
├── benchmark/                  # Semester 1: all nine architecture implementations
│   ├── eegnet.py
│   ├── shallow_convnet.py
│   ├── deep_convnet.py
│   ├── eeg_tcnet.py
│   ├── eegnex.py
│   ├── atcnet.py
│   ├── ctnet.py
│   ├── graph_cspnet.py
│   └── db_atcnet_original.py
│
├── db_atcnet_modified/         # Semester 2: MHA → Improved CBAM modification
│   ├── model.py                # Full model definition (ADBC + CBAM + TCFN)
│   ├── cbam.py                 # Improved CBAM module (channel + spatial attention)
│   ├── tcfn.py                 # Temporal Convolutional Fusion Network block
│   └── train.py                # Training loop (LOSO cross-validation)
│
├── electrode_reduction/        # Three-method electrode selection pipeline
│   ├── ablation_study.py       # Method 1: Leave-one-channel-out ablation
│   ├── weight_magnitude.py     # Method 2: DepthwiseConv2D weight norm ranking
│   └── gumbel_softmax.py       # Method 3: End-to-end differentiable selection
│
├── mha_analysis/               # Attention entropy analysis (MHA failure evidence)
│   └── entropy_analysis.py
│
├── preprocessing/
│   └── pipeline.py             # Z-score normalization, epoch extraction
│
├── configs/
│   └── gumbel_channel_indices/ # Per-subject JSON channel configs for FPGA LUT
│
├── weights/                    # Exported HDF5 model weights (per subject, per config)
│
├── results/
│   ├── benchmark_table.csv
│   ├── baseline_19ch_loso.csv
│   ├── ablation_5ch_loso.csv
│   ├── weight_guided_5ch_loso.csv
│   └── gumbel_5ch_loso.csv
│
└── README.md
```

---

## Dataset

**HALT Dataset** (Semester 2 primary dataset)

- Binary motor imagery task: **right-hand vs. left-leg**
- 19-channel EEG, 200 Hz sampling rate
- 10 subjects (H and I excluded from electrode reduction — near-chance 19-ch accuracy, extreme noise > 36 µV)
- Evaluation protocol: **Leave-One-Session-Out (LOSO)** cross-validation; Subject J (single session) uses stratified 80/20 split with 5 seeds
- Hardware acquisition: EEG-1200 JE-921A with hardware bandpass 0.53–70 Hz and 50 Hz notch

**Semester 1 Datasets**

- [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) — 109 subjects, 4-class MI
- [BCI Competition IV 2a/2b](https://www.bbci.de/competition/iv/) — standard benchmark

---

## Preprocessing Pipeline

```
Raw EEG (hardware-filtered at acquisition)
    │
    ├── [HALT] No additional bandpass / notch filtering applied
    │         (hardware filter 0.53–70 Hz + 50 Hz notch already applied)
    │
    ├── [BCI IV / PhysioNet] Butterworth bandpass 4–40 Hz (4th order)
    │                        + IIR notch 50 Hz (Q=30)
    │
    ├── Z-score normalization per channel per trial
    │   (fit on training folds only — no leakage)
    │
    └── Epoch segmentation
            HALT / BCI IV 2a:  0.5 – 4.0 s post-cue  →  (C, 600) @ 200 Hz
            PhysioNet:         0.0 – 4.0 s post-cue
```

> **Note:** DB-ATCNet learns its own spectral filters through its temporal convolutional first layer. Manual bandpass filtering before model input is intentionally omitted for the HALT dataset, consistent with the dual-branch ADBC design (D=2 and D=4 branches capture complementary spectral ranges internally).

---

## Architecture Benchmarking (Semester 1)

Nine architectures evaluated. Key FPGA-suitability ratings and findings:

| Model | Params | FPGA Suitability | Key Innovation | Limitation |
|---|---|---|---|---|
| EEGNet | ~2,500 | ✅ Excellent | Depthwise-separable temporal-spatial CNN | Limited temporal context |
| ShallowConvNet | ~60,724 | ✅ Good | Embeds FBCSP as differentiable layers | No hierarchical abstraction |
| DeepConvNet | ~199,829 | ⚠️ Moderate | Hierarchical 4-block CNN | Overfitting risk; large memory |
| EEG-TCNet | ~4,272 | ✅ Very Good | Causal dilated temporal context | No spatial attention |
| EEGNeX | ~63,626 | ✅ Good | Parallel dilated multi-resolution | Routing overhead |
| ATCNet | ~69,900 | ⚠️ Moderate | Sliding-window attention ensemble | O(T²) MHA; variable softmax |
| CTNet | ~25,700 | ⚠️ Low-Moderate | Global Transformer + S&R augmentation | Transformer cost |
| Graph-CSPNet | ~84,285 | ❌ Low | SPD manifold graph neural network | Riemannian ops not FPGA-friendly |
| DB-ATCNet (orig.) | ~150,000 | ⚠️ Moderate | Dual-branch ADBC + sliding MHA | MHA bottleneck |
| **DB-ATCNet (modified)** | **~144,000** | **✅ Good** | **MHA replaced by Improved CBAM** | **S2 base architecture** |

**Selection rationale for DB-ATCNet:** highest accuracy among evaluated architectures; modular design allows targeted MHA → CBAM replacement; fixed-shape tensor flow enables static FPGA memory allocation.

---

## Selected Architecture: DB-ATCNet (Modified)

### Modification: MHA → Improved CBAM

**Empirical evidence for MHA replacement:** attention weight entropy analysis across all 10 subjects showed a global mean normalized Shannon entropy of **H̃ = 0.952** (76.2% of all distributions exceed H̃ = 0.95 threshold). With only T=6 temporal positions per window, MHA was not performing selective temporal attention — it was operating as a near-uniform feature aggregator. The marginal 0.5 pp accuracy advantage of MHA (92.0% vs 91.5%) does not justify its FPGA cost.

### Architecture Flow (5-channel inference)

```
Input (1, 1, 5, 600)
    │
    Permute → (1, 600, 5, 1)
    │
    ┌─── ADBC Block ───────────────────────────────────────────────────┐
    │  Conv2D(16, 64×1) → BN → ECA1(k=3)                              │
    │         │                                                         │
    │    ┌────┴────┐                                                    │
    │  Branch1   Branch2                                                │
    │  D=2        D=4                                                   │
    │  (32 maps)  (64→32 maps)                                          │
    │    └────┬────┘                                                    │
    │      Add → ECA2 → Squeeze                                         │
    └──────────────────────────────────────────────────────────────────┘
    │
    (1, 10, 32)  [10 steps × 50 ms/step]
    │
    Sliding Window × 5  (windows of length 6, 50% overlap)
    │
    ┌─ Per-window (×5, independent weights) ─┐
    │  Improved CBAM (313 params/window)      │
    │  TCFN (dilated causal TCN, d={1,2})     │
    │  Dense(2) → softmax                     │
    └─────────────────────────────────────────┘
    │
    Average across 5 windows → argmax → predicted class
```

### Improved CBAM (per window)

- **Channel attention:** Global avg + max pool → shared MLP (32→4→32) → sigmoid gate (292 params)
- **Spatial attention:** Avg + max + stochastic pool along channel axis → concatenate → Conv2D(1, 7×1) → sigmoid (21 params)
- **Total:** 313 params/window × 5 windows = **1,565 params**; O(C×H×W) linear complexity — FPGA-friendly

### Training Configuration

```python
optimizer  = Adam(lr=0.0009, weight_decay=1e-4)
batch_size = 32
epochs     = 500  # fixed; no early stopping
checkpoint = best validation accuracy across training
```

---

## Electrode Reduction Methods (Semester 2)

Target: reduce from 19 electrodes (full EEG cap) to **5 dry electrodes** (wearable headband).

### Method 1 — Ablation-Based Static Selection (Gold Standard)

Retrain 19 separate 18-channel models per subject; measure accuracy drop per removed channel.

**Final 5-channel set:** `[Cz, P3, T5, F7, T3]`

- **Cz** — primary leg M1 generator (medial wall / paracentral lobule)
- **P3** — right-hand body schema (left parietal sensorimotor integration)
- **T5** — right-limb proprioceptive representation (posterior superior temporal sulcus)
- **F7** — right-hand motor planning (left DLPFC / inferior frontal)
- **T3** — additional right-hemisphere body schema coverage

> **Why not C3/C4?** For right-hand vs. left-leg, C3 ranked negatively (removing it *improves* accuracy) due to EMG contamination from the temporalis muscle and anatomical offset from the true hand knob of M1. The discriminative signal is concentrated in the parietal-temporal-frontal network.

**Computational cost:** 20 full LOSO runs/subject × 10 subjects × avg 2.5 sessions × 45 min ≈ **375 GPU-hours**

### Method 2 — Weight-Magnitude Ranking

Proxy importance from L2 norm of DepthwiseConv2D filter weights per channel. 20× cheaper than ablation (reuses baseline model).

**Selected set:** `[T5, T3, Fp2, Fz, Pz]`

⚠️ **Critical finding:** Fp2 ranked 3rd by weight magnitude but ranked **last (19th)** by ablation with a *negative* accuracy drop. Weight magnitude reflects co-adaptation within the full 19-channel network, not standalone contribution. This demonstrates why single-method analysis is insufficient.

### Method 3 — Gumbel-Softmax Learned Selection

End-to-end differentiable channel selection jointly optimized with network weights. Each subject gets a personalized 5-channel set.

Key hyperparameters:
```
K            = 5 selection neurons
β_start      = 10.0  (diffuse soft mixing)
β_end        = 0.1   (near-hard selection)
T_anneal     = 125 epochs (first 25% of training)
τ            = 3.0 → 1.0  (duplicate penalty annealing)
λ            = 1.0  (regularization weight)
```

At inference: Gumbel noise removed; `argmax(α_nk)` gives hard channel selection stored in a **per-subject JSON LUT** loaded to the FPGA at headband startup.

**Most frequently selected channels across subjects:**
- **F8** — appears in 8/10 subjects (right inferior frontal gyrus; mirror neuron / motor inhibition)
- **Cz** — dominant for 5/10 subjects (consistent with ablation rank 1)

---

## Results Summary

### 19-Channel Baseline (Modified DB-ATCNet, LOSO)

| Subject | Sessions | Mean Accuracy | Mean κ | Std |
|---|---|---|---|---|
| A | 3 | 96.83% | 0.768 | ±3.20% |
| B | 3 | 85.65% | 0.625 | ±4.07% |
| C | 2 | 85.38% | 0.641 | ±10.61% |
| E | 3 | 90.44% | 0.703 | ±6.81% |
| F | 3 | 95.87% | 0.748 | ±2.66% |
| G | 3 | 96.16% | 0.720 | ±10.19% |
| J | 1 | 100.00% | 1.000 | ±0.00% |
| K | 2 | 83.79% | 0.694 | ±2.84% |
| L | 2 | 98.88% | 0.831 | ±0.84% |
| M | 3 | 86.88% | 0.625 | ±6.20% |
| **Overall** | | **91.99%** | **0.839** | **±5.41%** |

> Subject K has anomalously high electrode noise (~155 µV baseline Std vs. 8–20 µV for clean subjects).

### 5-Channel Comparison (All Methods)

| Configuration | Channels | Mean Acc. | Mean κ | vs. 19-ch |
|---|---|---|---|---|
| 19-ch Baseline | 19 | 91.99% | 0.839 | — |
| Weight-Guided `[T5,T3,Fp2,Fz,Pz]` | 5 | 87.08% | 0.741 | −4.91% |
| Ablation-Guided `[Cz,P3,T5,F7,T3]` | 5 | 86.77% | 0.735 | −5.22% |
| **Gumbel Per-Subject** | **5** | **89.94%** | **0.799** | **−2.05%** |

The Gumbel approach closes 60% of the gap between static 5-channel selection and the full 19-channel baseline, while eliminating 74% of electrodes.

---

## FPGA Deployment Notes

This module delivers three artifacts to the hardware team:

1. **Inference specification** — layer-by-layer tensor shapes, weight formats (float32 HDF5), operator definitions for all DB-ATCNet operations.
2. **Trained weights** — HDF5 exports per subject per configuration (19-ch, ablation-5ch, weight-5ch, Gumbel-5ch), organized as per-layer numpy arrays.
3. **Gumbel channel index configs** — per-subject JSON files in `configs/gumbel_channel_indices/` specifying 5 electrode indices per LOSO fold, loaded to FPGA channel-selection register at startup.

**FPGA channel selection modes:**
- Static (ablation / weight) → hardwired 5-of-19 analog multiplexer; no runtime config
- Gumbel (per-subject) → 25-bit config register loaded from flash LUT at startup; downstream RTL is identical

**Target hardware:** Xilinx Zynq UltraScale+ ZCU106 (XCZU7EV)
**Reported inference latency:** 2.31 ms on FPGA

---

## Dependencies & Setup

```bash
pip install tensorflow>=2.10 numpy scipy scikit-learn mne h5py
```

**Training a model (example — LOSO on HALT with Gumbel selection):**

```bash
python electrode_reduction/gumbel_softmax.py \
    --subject A \
    --data_path /path/to/HALT \
    --n_channels 19 \
    --k_select 5 \
    --epochs 500 \
    --beta_start 10.0 \
    --beta_end 0.1
```

**Running the MHA entropy analysis:**

```bash
python mha_analysis/entropy_analysis.py \
    --model_weights weights/subject_A_19ch_fold1.h5 \
    --data_path /path/to/HALT \
    --subject A
```

---

## Citation

If you use this code or findings, please cite the following works that this project builds upon:

```
[1] Lawhern et al., "EEGNet: A compact CNN for EEG-based BCIs," J. Neural Eng., 2018.
[2] Altaheri et al., "ATCNet," IEEE TNSRE, 2022.
[3] Ke et al., "DB-ATCNet," 2023.
[4] Strypsteen & Bertrand, "End-to-end learnable EEG channel selection," J. Neural Eng., 2021.
[5] Woo et al., "CBAM: Convolutional Block Attention Module," ECCV, 2018.
```

---

*E-JUST CSIT Department — AI & Data Science Track*
