
# DB-ATCNet — Edge-Optimized EEG Motor Imagery Classification

> **Deep learning meets embedded systems**: A hardware-aware BCI pipeline that classifies motor imagery from just 5 EEG channels at 89.94% accuracy, engineered for real-time FPGA inference in wearable brain-computer interfaces.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-≥2.9-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Motivation

Standard EEG-based BCI systems rely on 19–128 electrode caps and GPU-class compute — practical for research labs, impractical for consumer wearable devices. This project bridges that gap by systematically optimizing a state-of-the-art deep learning architecture (DB-ATCNet) for deployment on resource-constrained FPGA hardware, reducing the electrode count from 19 to just 5 while retaining competitive accuracy.

Every design decision is guided by a single constraint: **the final model must run in real-time on an FPGA with minimal channel count, fixed compute, and no dynamic memory allocation.**

## Key Results

| Configuration | Channels | Accuracy | Hardware Impact |
|:---|:---:|:---:|:---|
| Baseline (Improved CBAM) | 19 | 91.50% | Full EEG cap required |
| Static top-5 (ablation-ranked) | 5 | 87.00% | Fixed 5-electrode headband, no per-subject config |
| Gumbel-Softmax (per-subject) | 5 | 89.94% | Same headband, per-subject channel LUT at startup |

## Design Evolution

This project follows a three-stage optimization pipeline, each motivated by concrete deployment requirements:

### Stage 1: Attention Mechanism Replacement (MHA → Improved CBAM)

**Problem**: The original DB-ATCNet uses Multi-Head Attention (MHA), which requires dynamic Q/K/V matrix projections and softmax over variable-length sequences — operations that are expensive to implement in RTL and difficult to pipeline efficiently.

**Finding**: Empirical analysis revealed the MHA in this architecture was functioning more as a pooling operation than true selective attention — the learned attention weights were near-uniform across time steps.

**Solution**: Replaced MHA with an Improved CBAM (Convolutional Block Attention Module) that adds stochastic pooling alongside average and max pooling in the spatial sub-module. CBAM uses only fixed-size global pooling, small dense layers, and a 7×1 convolution — all of which map directly to efficient, pipelineable hardware blocks.

**Result**: ~0.5% accuracy drop — a negligible cost for a significantly simpler hardware implementation.

### Stage 2: Channel Reduction via Ablation Study

**Problem**: 19 electrodes require a full EEG cap with conductive gel — unsuitable for everyday wearable use. The target form factor is a lightweight headband with ≤5 dry electrodes.

**Method**: Trained the full 19-channel model as baseline (leave-one-session-out CV per subject). Then retrained with each channel individually removed (18 channels), measuring the per-subject accuracy drop. Channels were ranked by their mean drop across all subjects.

**Top-5 channels**: Cz, P3, T5, F7, T3 (indices `[17, 6, 14, 10, 12]`)

**Trade-off**: A fixed channel set is the simplest hardware design (hardwired MUX, no runtime configuration), but does not account for inter-subject variability. Accuracy: 87%.

### Stage 3: Learned Channel Selection (Gumbel-Softmax)

**Problem**: The static top-5 loses 4.5% accuracy because different subjects have different optimal channel subsets.

**Method**: Integrated a Gumbel-Softmax concrete selector layer ([Strypsteen & Bertrand 2021](https://arxiv.org/abs/2102.09050)) that is prepended to the model and jointly trained with all network weights. During training, it samples soft channel mixtures via the Concrete distribution; during inference, it reduces to a deterministic `argmax` — a simple per-subject lookup table of 5 channel indices.

A duplicate-avoidance regularization loss encourages distinct channel selections, though duplicates are permitted when they improve accuracy for a given subject.

**Result**: 89.94% accuracy — recovering most of the gap to the full 19-channel baseline while maintaining the 5-channel hardware constraint.

![1 4](https://github.com/zk-xju/DB-ATCNet/assets/156686159/99f2e790-57f6-43cb-9729-56272b98b027)

## Architecture

The inference pipeline consists of four stages, all operating on fixed tensor shapes with no dynamic allocation:

- **ADBC Block** — Dual-branch depthwise separable convolution with ECA (Efficient Channel Attention) for joint spatial-temporal feature extraction.
- **Improved CBAM** — Sequential channel attention (dual-pool shared MLP) and spatial attention (tri-pool: average, max, and stochastic pooling).
- **TCFN Block** — Temporal Convolutional Fusion Network with dilated causal convolutions (dilation rates 1, 2) and multi-level residual connections.
- **Sliding Window Fusion** — Five overlapping 6-step temporal windows, each independently classified and averaged for the final 2-class softmax prediction.

For a complete layer-by-layer tensor shape flow and operator specification, see the **[FPGA Deployment Specification](FPGA_DEPLOYMENT_DOC.md)**.

## FPGA Implementation (In Progress)

The RTL implementation of the inference pipeline is under active development. The goal is real-time, low-latency motor imagery classification on an FPGA embedded in a wearable BCI headband.

**Milestone Status**:
- [x] Model architecture finalized and frozen for inference
- [x] Inference specification documented — [`FPGA_DEPLOYMENT_DOC.md`](FPGA_DEPLOYMENT_DOC.md)
- [x] Weight export pipeline established (HDF5 → per-layer numpy arrays)
- [ ] RTL design and synthesis
- [ ] On-chip validation and latency benchmarking

Precision, quantization strategy, and FPGA resource optimization are handled by the RTL team based on the inference specification.

## Project Structure

```
├── HALT_main.py                  # Training entry point (baseline, ablation, Gumbel modes)
├── models.py                     # Model architectures (DB-ATCNet, Gumbel variant)
├── attention_models.py           # Attention modules (MHA, ECA, CBAM, Improved CBAM)
├── gumbel_channel_selection.py   # Gumbel-Softmax selection layer & annealing callbacks
├── HALT_DataLoad.py              # HaLT dataset loader with session-aware CV splitting
├── channel_Importance.py         # Channel ranking from ablation study results
├── visualize_features.py         # Feature map visualization and attention analysis
├── FPGA_DEPLOYMENT_DOC.md        # Complete FPGA inference specification for RTL developers
└── HALT/                         # HaLT dataset directory (not included in repo)
```

## Getting Started

### Requirements

- Python ≥ 3.10
- TensorFlow ≥ 2.9
- NumPy, scikit-learn, SciPy, matplotlib, mne

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

All modes use **leave-one-session-out cross-validation** per subject. Results, confusion matrices, and learning curves are saved to `results/`.

## Dataset

### HaLT (Hand and Leg Task)

Place the HaLT `.mat` files in the `HALT/` directory. The dataset contains 19-channel EEG recordings at 200 Hz from multiple subjects performing Right Hand and Left Leg motor imagery tasks across multiple recording sessions.

**Channel layout** (19 EEG leads):
`Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`

## References

1. H. Altaheri, G. Muhammad, and M. Alsulaiman, "Physics-informed attention temporal convolutional network for EEG-based motor imagery classification," *IEEE Trans. Ind. Inform.*, 2022. [doi:10.1109/TII.2022.3197419](https://doi.org/10.1109/TII.2022.3197419)
2. T. Strypsteen and A. Bertrand, "End-to-end learnable EEG channel selection for deep neural networks with Gumbel-softmax," *arXiv:2102.09050*, 2021.
3. Z. Ke et al., "DB-ATCNet: Dual-Branch Convolution Network with Efficient Channel Attention," [GitHub](https://github.com/zk-xju/DB-ATCNet).

## Acknowledgments

This work is built upon the [DB-ATCNet](https://github.com/zk-xju/DB-ATCNet) architecture and the [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) repository by Altaheri et al. We gratefully acknowledge the original authors for making their code and research publicly available.
