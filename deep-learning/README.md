# DB-ATCNet — Edge-Optimized EEG Motor Imagery Classification

> **Deep learning meets embedded systems**: A hardware-aware BCI pipeline that classifies motor imagery from just 5 EEG channels at 89.94% accuracy, engineered for real-time FPGA inference in wearable brain-computer interfaces.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-≥2.9-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Motivation

Standard EEG-based BCI systems rely on 19–128 electrode caps and GPU-class compute — practical for research labs, impractical for consumer wearable devices. This project bridges that gap by systematically optimizing a state-of-the-art deep learning architecture (DB-ATCNet) for deployment on resource-constrained FPGA hardware, reducing the electrode count from 19 to just 5 while retaining competitive accuracy.

Every design decision is guided by a single constraint: **the final model must run in real-time on an FPGA with minimal channel count, fixed compute, and no dynamic memory allocation.**

---

## Key Results

| Configuration | Channels | Accuracy | Hardware Impact |
|:---|:---:|:---:|:---|
| Baseline (Improved CBAM) | 19 | 91.99% | Full EEG cap required |
| Static top-5 (ablation-ranked) `[Cz, P3, T5, F7, T3]` | 5 | 86.77% | Fixed 5-electrode headband, no per-subject config |
| Weight-magnitude top-5 `[T5, T3, Fp2, Fz, Pz]` | 5 | 87.08% | Fixed 5-electrode headband |
| Gumbel-Softmax (per-subject) | 5 | 89.94% | Same headband, per-subject channel LUT at startup |

### Per-Subject Baseline Results (19-Channel, LOSO)

| Subject | Sessions | Mean Accuracy | Mean κ | Std | Protocol |
|---|---|---|---|---|---|
| A | 3 | 96.83% | 0.768 | ±3.20% | LOSO 3-fold |
| B | 3 | 85.65% | 0.625 | ±4.07% | LOSO 3-fold |
| C | 2 | 85.38% | 0.641 | ±10.61% | LOSO 2-fold |
| E | 3 | 90.44% | 0.703 | ±6.81% | LOSO 3-fold |
| F | 3 | 95.87% | 0.748 | ±2.66% | LOSO 3-fold |
| G | 3 | 96.16% | 0.720 | ±10.19% | LOSO 3-fold |
| J | 1 | 100.00% | 1.000 | ±0.00% | 80/20 split |
| K | 2 | 83.79% | 0.694 | ±2.84% | LOSO 2-fold |
| L | 2 | 98.88% | 0.831 | ±0.84% | LOSO 2-fold |
| M | 3 | 86.88% | 0.625 | ±6.20% | LOSO 3-fold |
| **Overall** | | **91.99%** | **0.839** | **±5.41%** | |

> Subjects H and I excluded from electrode reduction experiments (near-chance 19-ch accuracy ~63%, electrode noise > 36 µV). Subject K has anomalously high noise (~155 µV vs. 8–20 µV for clean subjects). Subject J uses 80/20 split due to single session.

---

## Design Evolution

This project follows a three-stage optimization pipeline, each motivated by concrete deployment requirements:

### Stage 1: Attention Mechanism Replacement (MHA → Improved CBAM)

**Problem**: The original DB-ATCNet uses Multi-Head Attention (MHA), which requires dynamic Q/K/V matrix projections and softmax over variable-length sequences — operations that are expensive to implement in RTL and prevent static memory pre-allocation.

**Finding**: Empirical entropy analysis of MHA attention weights across all 10 subjects revealed a global mean normalized Shannon entropy of **H̃ = 0.952**, with 76.2% of all distributions exceeding the H̃ = 0.95 threshold. With only T=6 temporal positions per sliding window, MHA was not performing selective temporal attention — it was operating as a near-uniform feature aggregator, functionally equivalent to a weighted pooling operation.

**Solution**: Replaced MHA with an Improved CBAM (Convolutional Block Attention Module) using only fixed-size global pooling, a small shared MLP, and a 7×1 convolution — all mapping directly to pipelineable FPGA hardware blocks with O(C×H×W) linear complexity.

- **Channel attention**: Global avg + max pool → shared MLP (32→4→32) → sigmoid gate (292 params/window)
- **Spatial attention**: Avg + max + stochastic pool along channel axis → Conv2D(1, 7×1) → sigmoid (21 params/window)
- **Total**: 313 params/window × 5 windows = **1,565 params**

**Result**: ~0.5% accuracy drop (92.0% MHA → 91.5% CBAM) — negligible cost for significantly simpler hardware.

---

### Stage 2: Channel Reduction via Ablation Study

**Problem**: 19 electrodes require a full EEG cap with conductive gel — unsuitable for everyday wearable use. Target: a lightweight headband with ≤5 dry electrodes.

**Method**: Trained the full 19-channel model as baseline (LOSO CV per subject), then retrained with each channel individually removed (18-channel), measuring accuracy drop Δ_c per subject. Channels ranked by mean drop across all subjects.

**Top-5 channels**: `[Cz, P3, T5, F7, T3]`

| Rank | Channel | Mean Δ Drop | Neurophysiological Role |
|---|---|---|---|
| 1 | Cz | +0.0156 | Primary leg M1 generator (medial wall / paracentral lobule) |
| 2 | P3 | +0.0128 | Right-hand body schema (left parietal sensorimotor integration) |
| 3 | T5 | +0.0108 | Right-limb proprioceptive representation (posterior STS) |
| 4 | F7 | +0.0106 | Right-hand motor planning (left DLPFC / inferior frontal) |
| 5 | T3 | +0.0103 | Additional right-hemisphere body schema coverage |

> **Why not C3/C4?** C3 ranked negatively (removing it *improves* accuracy) due to EMG contamination from the temporalis muscle, anatomical offset from the true hand knob of M1, and task asymmetry — the discriminative signal for right-hand vs. left-leg is concentrated in the parietal-temporal-frontal network, not the textbook central electrodes.

**Hardware advantage**: Static channel set = hardwired 5-of-19 analog multiplexer; zero runtime configuration needed.

**Computational cost**: 20 full LOSO runs/subject × 10 subjects × avg 2.5 sessions × 45 min ≈ 375 GPU-hours.

---

### Stage 3: Learned Channel Selection (Gumbel-Softmax)

**Problem**: A single static channel set loses accuracy because different subjects have different optimal electrode locations due to cortical geometry, skull thickness, and mental execution strategy.

**Method**: Prepended a Gumbel-Softmax concrete selector layer ([Strypsteen & Bertrand 2021](https://arxiv.org/abs/2102.09050)) jointly trained with all network weights. During training, each of K=5 selection neurons samples soft channel mixtures via the Concrete distribution; at inference, Gumbel noise is removed and hard selection reduces to `argmax(α_nk)` — a simple per-subject lookup table of 5 channel indices loaded to the FPGA channel-selection register at startup.

**Key hyperparameters**:

```
K              = 5 selection neurons
β_start        = 10.0   (diffuse soft mixing → exploration)
β_end          = 0.1    (near-hard selection → exploitation)
T_anneal       = 125 epochs (first 25% of 500-epoch budget)
τ              = 3.0 → 1.0  (duplicate penalty annealing)
λ_reg          = 1.0
```

**Most frequently selected channels across subjects**:
- **F8** — appears in 8/10 subjects (right inferior frontal gyrus; mirror neuron / motor inhibition system)
- **Cz** — dominant for 5/10 subjects (consistent with ablation rank 1)

**Result**: 89.94% accuracy — recovering most of the gap to the 19-channel baseline while maintaining the 5-channel hardware constraint.

![Architecture](https://github.com/zk-xju/DB-ATCNet/assets/156686159/99f2e790-57f6-43cb-9729-56272b98b027)

---

## Architecture

The inference pipeline operates entirely on fixed tensor shapes with no dynamic memory allocation:

- **ADBC Block** — Dual-branch depthwise separable convolution (D=2 and D=4) with ECA (Efficient Channel Attention) for joint spatial-temporal feature extraction. Two-stage pooling compresses 600 samples → 10 temporal steps (~50 ms/step).
- **Improved CBAM** — Sequential channel attention (dual-pool shared MLP) and spatial attention (tri-pool: average, max, stochastic pooling) applied per sliding window.
- **TCFN Block** — Temporal Convolutional Fusion Network with dilated causal convolutions (dilation rates 1, 2) and multi-level residual connections. Causal padding only — no lookahead buffer required.
- **Sliding Window Fusion** — Five overlapping 6-step windows extracted from the 10-step feature sequence, each independently classified and averaged for the final 2-class softmax prediction.

For the complete layer-by-layer tensor shape flow and operator specification, see **[FPGA_DEPLOYMENT_DOC.md](FPGA_DEPLOYMENT_DOC.md)**.

---

## Preprocessing Pipeline

```
Raw EEG
    │
    ├── [HALT dataset]
    │     No additional filtering applied.
    │     Hardware bandpass 0.53–70 Hz + 50 Hz notch already applied
    │     at acquisition (EEG-1200 JE-921A system).
    │     DB-ATCNet learns its own spectral filters via temporal Conv2D.
    │
    ├── [BCI IV / PhysioNet]
    │     Butterworth bandpass 4–40 Hz (4th order)
    │     + IIR notch 50 Hz (Q=30)
    │
    ├── Z-score normalization per channel per trial
    │   (fit on training folds only — strictly no leakage across folds)
    │
    └── Epoch segmentation
          HALT / BCI IV 2a:  0.5–4.0 s post-cue  →  (C, 600) @ 200 Hz
          PhysioNet:         0.0–4.0 s post-cue
```

---

## FPGA Implementation (In Progress)

Target hardware: **Xilinx Zynq UltraScale+ ZCU106 (XCZU7EV)**

**Milestone Status**:
- [x] Model architecture finalized and frozen for inference
- [x] Inference specification documented — [`FPGA_DEPLOYMENT_DOC.md`](FPGA_DEPLOYMENT_DOC.md)
- [x] Weight export pipeline established (HDF5 → per-layer numpy arrays, organized by subject and configuration)
- [x] Per-subject Gumbel channel index JSON configs ready for FPGA LUT loading
- [ ] RTL design and synthesis
- [ ] On-chip validation and latency benchmarking

**Channel selection hardware modes**:
- Static (ablation / weight-guided) → hardwired 5-of-19 analog MUX; no runtime config
- Gumbel (per-subject) → 25-bit config register loaded from flash LUT at startup; downstream RTL is identical in both cases

---

## Project Structure

```
├── HALT_main.py                  # Training entry point (baseline, ablation, Gumbel modes)
├── models.py                     # Model architectures (DB-ATCNet, Gumbel variant)
├── attention_models.py           # Attention modules (MHA, ECA, CBAM, Improved CBAM)
├── gumbel_channel_selection.py   # Gumbel-Softmax selection layer & annealing callbacks
├── HALT_DataLoad.py              # HaLT dataset loader with session-aware CV splitting
├── channel_Importance.py         # Channel ranking from ablation study results
├── mha_entropy_analysis.py       # MHA attention weight entropy analysis
├── visualize_features.py         # Feature map visualization and attention analysis
├── FPGA_DEPLOYMENT_DOC.md        # Complete FPGA inference specification for RTL team
└── HALT/                         # HaLT dataset directory (not included in repo)
```

---

## Getting Started

### Requirements

```bash
pip install tensorflow>=2.9 numpy scipy scikit-learn matplotlib mne h5py
```

### Usage

```bash
# Baseline: full 19-channel subject-dependent training
python HALT_main.py --single-run

# Ablation: evaluate reduced channel subsets
python HALT_main.py --ablation --ablation-channels 3 5 7 10

# Gumbel-Softmax: learnable per-subject channel selection
python HALT_main.py --gumbel-select --gumbel-k 5

# Gumbel on specific subjects only
python HALT_main.py --gumbel-select --gumbel-k 5 --subjects A B C
```

All modes use leave-one-session-out cross-validation per subject. Results, confusion matrices, and learning curves are saved to `results/`.

---

## Dataset

### HaLT (Hand and Leg Task)

Place the HaLT `.mat` files in the `HALT/` directory. The dataset contains 19-channel EEG recordings at 200 Hz from multiple subjects performing Right Hand and Left Leg motor imagery tasks across multiple recording sessions.

**Channel layout** (19 EEG leads, indices 0–18):
`Fp1(0), Fp2(1), F3(2), F4(3), C3(4), C4(5), P3(6), P4(7), O1(8), O2(9), F7(10), F8(11), T3(12), T4(13), T5(14), T6(15), Fz(16), Cz(17), Pz(18)`

---

## References

1. H. Altaheri et al., "Physics-informed attention temporal convolutional network for EEG-based motor imagery classification," *IEEE Trans. Ind. Inform.*, 2022. [doi:10.1109/TII.2022.3197419](https://doi.org/10.1109/TII.2022.3197419)
2. T. Strypsteen and A. Bertrand, "End-to-end learnable EEG channel selection for deep neural networks with Gumbel-softmax," *arXiv:2102.09050*, 2021.
3. Z. Ke et al., "DB-ATCNet: Dual-Branch Convolution Network with Efficient Channel Attention," [GitHub](https://github.com/zk-xju/DB-ATCNet).
4. V. J. Lawhern et al., "EEGNet: A compact CNN for EEG-based BCIs," *J. Neural Eng.*, 2018.
5. C. Woo et al., "CBAM: Convolutional Block Attention Module," *ECCV*, 2018.

---

## Acknowledgments

This work is built upon the [DB-ATCNet](https://github.com/zk-xju/DB-ATCNet) architecture and the [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) repository by Altaheri et al. We gratefully acknowledge the original authors for making their code and research publicly available.

---

*E-JUST CSIT Department — AI & Data Science | Supervised by Dr. Reda Albassiouny & Dr. Sameh Sherif*
