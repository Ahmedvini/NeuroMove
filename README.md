
# DB-ATCNet with Improved CBAM and Adaptive Channel Selection

Dual-Branch Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification, extended with Improved CBAM attention and hardware-oriented channel reduction strategies for FPGA deployment.

## Overview

This project targets **real-time motor imagery classification on edge devices** (FPGA-based BCI headsets), not clinical-grade multi-channel EEG systems. Every architectural decision is driven by the need to minimize hardware cost, channel count, and computational complexity while preserving classification accuracy.

The repository builds upon the original [DB-ATCNet](https://github.com/zk-xju/DB-ATCNet) architecture and introduces the following modifications, evaluated on the [HaLT dataset](https://doi.org/10.1016/j.dib.2017.12.032) (2-class: Right Hand vs Left Leg):

1. **Improved CBAM Attention** — The original Multi-Head Attention (MHA) was found to function more as a pooling operation than a true attention mechanism in this architecture — attention weights were near-uniform rather than learning selective focus. It was replaced with an Improved Convolutional Block Attention Module (CBAM) with stochastic pooling. **Why**: MHA requires dynamic Q/K/V projections and variable-length softmax, which are expensive in RTL. CBAM uses only fixed-size pooling and small convolutions — operations that map directly to efficient hardware. Accuracy trade-off: only ~0.5%.

2. **Ablation-Based Static Channel Selection** — A leave-one-channel-out ablation study was conducted across all subjects to rank the 19 EEG channels by importance. **Why**: Fewer EEG channels means fewer physical electrodes, smaller ADC front-end, lower power consumption, and a more practical wearable device. A fixed 5-channel subset avoids any per-subject configuration on the hardware. Accuracy with static top-5: 87%.

3. **Gumbel-Softmax Learnable Channel Selection** — An end-to-end learnable channel selection layer based on [Strypsteen & Bertrand (2021)](https://arxiv.org/abs/2102.09050) was integrated to jointly optimize channel selection with model weights. **Why**: Subject-dependent channel selection recovers most of the accuracy lost by the static approach, while still using only 5 physical channels. At inference, the selection reduces to a hardcoded per-subject lookup table — zero additional hardware cost. Accuracy: 89.94%.

| Configuration | Channels | Accuracy |
|:---|:---:|:---:|
| Baseline (Improved CBAM, all channels) | 19 | 91.50% |
| Static top-5 (ablation-ranked) | 5 | 87.00% |
| Gumbel-Softmax (subject-dependent) | 5 | 89.94% |

![1 4](https://github.com/zk-xju/DB-ATCNet/assets/156686159/99f2e790-57f6-43cb-9729-56272b98b027)

## Architecture

The model pipeline consists of four main stages:

- **ADBC Block** — Dual-branch depthwise separable convolution with ECA attention for spatial-temporal feature extraction.
- **Improved CBAM** — Channel attention (dual-pool shared MLP) followed by spatial attention (tri-pool with average, max, and stochastic pooling).
- **TCFN Block** — Temporal Convolutional Fusion Network with dilated causal convolutions and multi-level residual connections.
- **Sliding Window Fusion** — Five overlapping temporal windows are independently classified and averaged for the final prediction.

For a complete layer-by-layer specification aimed at FPGA/RTL implementation, see [`FPGA_DEPLOYMENT_DOC.md`](FPGA_DEPLOYMENT_DOC.md).

## FPGA Implementation (In Progress)

The RTL implementation of the inference pipeline is currently under active development. The goal is to deploy the frozen 5-channel model on an FPGA for real-time, low-latency motor imagery classification in a wearable BCI device.

**Status**:
- [x] Model architecture finalized and frozen
- [x] Inference specification documented ([`FPGA_DEPLOYMENT_DOC.md`](FPGA_DEPLOYMENT_DOC.md))
- [x] Weight export pipeline established
- [ ] RTL design and synthesis
- [ ] On-chip validation

Precision, quantization, and resource optimization decisions are handled by the RTL team based on the inference specification.

## Project Structure

```
├── HALT_main.py                  # Main training script (baseline, ablation, Gumbel)
├── models.py                     # Model architectures (DB-ATCNet, Gumbel variant, etc.)
├── attention_models.py           # Attention modules (MHA, ECA, CBAM, Improved CBAM)
├── gumbel_channel_selection.py   # Gumbel-Softmax channel selection layer & callbacks
├── HALT_DataLoad.py              # HaLT dataset loader with session-aware splitting
├── channel_Importance.py         # Channel ranking from ablation results
├── visualize_features.py         # Feature visualization and attention analysis
├── FPGA_DEPLOYMENT_DOC.md        # FPGA inference deployment specification
└── HALT/                         # HaLT dataset directory (not included)
```

## Development Environment

Models were trained and evaluated on Ubuntu with a single NVIDIA GPU using Python 3.13 and the TensorFlow/Keras framework.

### Requirements

- TensorFlow ≥ 2.9
- NumPy
- scikit-learn
- SciPy
- matplotlib
- mne

## Dataset

### HaLT (Hand and Leg Task)

The HaLT motor imagery dataset should be placed in the `HALT/` directory. The dataset contains 19-channel EEG recordings at 200 Hz from multiple subjects performing Right Hand and Left Leg motor imagery tasks.

Channel layout (19 EEG leads): `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`

## Usage

```bash
# Baseline: 19-channel subject-dependent training
python HALT_main.py --single-run

# Ablation study: evaluate static channel subsets
python HALT_main.py --ablation
python HALT_main.py --ablation --ablation-channels 3 5 7 10

# Gumbel-Softmax: learnable channel selection
python HALT_main.py --gumbel-select
python HALT_main.py --gumbel-select --gumbel-k 5
python HALT_main.py --gumbel-select --gumbel-k 5 --subjects A B C
```

All modes use leave-one-session-out cross-validation per subject. Results are saved to the `results/` directory.

## References

- H. Altaheri, G. Muhammad, and M. Alsulaiman, "Physics-informed attention temporal convolutional network for EEG-based motor imagery classification," *IEEE Trans. Ind. Inform.*, 2022. [doi:10.1109/TII.2022.3197419](https://doi.org/10.1109/TII.2022.3197419)
- T. Strypsteen and A. Bertrand, "End-to-end learnable EEG channel selection for deep neural networks with Gumbel-softmax," *arXiv:2102.09050*, 2021.

## Acknowledgments

This work is built upon the [DB-ATCNet](https://github.com/zk-xju/DB-ATCNet) codebase and the [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) repository by Altaheri et al. We gratefully acknowledge the original authors for their open-source contributions.
