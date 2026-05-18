# DB-ATCNet: FPGA Inference Deployment Specification

> **Audience**: RTL developers implementing inference-only hardware.
> **Scope**: This document covers the frozen inference graph. All training-only constructs (dropout, Gumbel noise, gradient-based updates) are excluded. Precision and quantization decisions are left to the hardware team.

---

## 1. System Overview

DB-ATCNet is a deep learning model for 2-class EEG motor imagery classification (Right Hand vs Left Leg). It processes raw EEG voltage traces and outputs a 2-element softmax probability vector.

| Parameter | Value |
|---|---|
| Input channels (full model) | 19 EEG electrodes |
| Input channels (reduced model) | 5 EEG electrodes |
| Samples per trial | 600 (3.0 s × 200 Hz) |
| Output classes | 2 (Right Hand, Left Leg) |
| Sampling rate | 200 Hz |

### 1.1 Design Evolution Summary

| Stage | Attention Mechanism | Channels | Accuracy | Notes |
|---|---|---|---|---|
| Baseline | Improved CBAM | 19 (all) | 91.50% | Full-channel reference |
| Static channel selection | Improved CBAM | 5 (fixed set) | 87.00% | Same 5 channels for all subjects |
| Gumbel channel selection | Improved CBAM | 5 (per-subject) | 89.94% | Different 5 channels per subject, duplication allowed |

The original base DB-ATCNet used Multi-Head Attention (MHA). Analysis showed that in this architecture, the MHA layer functioned more as a pooling operation than as true multi-head attention — the attention weights were near-uniform rather than learning meaningful query-key relationships. It was replaced with Improved CBAM (Convolutional Block Attention Module with stochastic pooling), which resulted in a drop of only ~0.5% accuracy while being significantly more hardware-friendly: CBAM uses fixed-size pooling and small convolutions instead of dynamic Q/K/V projections and variable-length softmax.

---

## 2. Dataset: HALT

- **Source**: HaLT (Hand and Leg Task) motor imagery dataset
- **Subjects**: 10 (labeled A–M, excluding H and I)
- **Sessions per subject**: 1–3 recording sessions (`.mat` files)
- **Raw channels**: 22 total in hardware (19 EEG + 2 ground/reference + 1 sync)
- **Used channels**: 19 EEG leads only

### 2.1 Channel Order (Index 0–18)

| Index | Name | Index | Name | Index | Name | Index | Name |
|---|---|---|---|---|---|---|---|
| 0 | Fp1 | 5 | C4 | 10 | F7 | 15 | T6 |
| 1 | Fp2 | 6 | P3 | 11 | F8 | 16 | Fz |
| 2 | F3 | 7 | P4 | 12 | T3 | 17 | Cz |
| 3 | F4 | 8 | O1 | 13 | T4 | 18 | Pz |
| 4 | C3 | 9 | O2 | 14 | T5 | | |

### 2.2 Trial Structure

Each trial is a contiguous block of 600 samples (3 seconds) aligned to the X3 sync pulse. The input tensor shape is `(1, 1, C, 600)` where C is the number of channels (19 or 5).

---

## 3. Channel Selection Strategies

At inference, the model always operates on exactly **5 input channels**. The only difference between the two reduced-channel approaches is *which* 5 channels are routed to the model input.

### 3.1 Static Channel Selection (Ablation-Based)

**Method**: An ablation study was performed — the model was trained on all 19 channels as baseline, then retrained with each channel individually removed (18 channels at a time). The accuracy drop per removed channel was measured across all subjects.

**Ranked channels** (most important first, by mean accuracy drop):

| Rank | Channel | Drop |
|---|---|---|
| 1 | Cz (17) | 0.0156 |
| 2 | P3 (6) | 0.0128 |
| 3 | T5 (14) | 0.0108 |
| 4 | F7 (10) | 0.0106 |
| 5 | T3 (12) | 0.0103 |
| 6 | O2 (9) | 0.0096 |
| 7 | F8 (11) | 0.0090 |
| 8 | P4 (7) | 0.0085 |
| 9 | Fz (16) | 0.0069 |
| 10 | C4 (5) | 0.0059 |

**Static top-5 set**: `[Cz, P3, T5, F7, T3]` → indices `[17, 6, 14, 10, 12]`

**FPGA implementation**: Hardwired 5-to-1 MUX selecting these fixed channel indices from the 19-channel ADC input. No runtime configurability needed.

**Result**: 87.00% average accuracy.

### 3.2 Gumbel-Softmax Channel Selection (Subject-Dependent)

**Method**: A learnable channel selection layer (Gumbel-Softmax, based on Strypsteen & Bertrand 2021) was prepended to the model. During training, it jointly learns which 5 channels are optimal for each subject. At inference, the selection is a deterministic `argmax` over learned logits — effectively a fixed per-subject channel routing table.

**Inference behavior**: The Gumbel layer reduces to:
```
selected_indices = argmax(logits, axis=0)  // shape: (5,)
output[k] = input[selected_indices[k]]     // for k = 0..4
```

The `logits` matrix is shape `(19, 5)`. Each column's argmax gives one selected channel index. **Duplicate indices are allowed** — two selection neurons may converge to the same channel.

**FPGA implementation**: A per-subject lookup table (LUT) of 5 channel indices, loaded at startup. The downstream model is architecturally identical to the static-channel variant.

**Result**: 89.94% average accuracy.

**Example selections from results** (indices into the 19-channel array):

| Subject | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| A | [11,3,5,9,9] | [11,9,12,9,11] | [9,11,12,14,3] |
| B | [8,13,11,14,17] | [5,2,16,15,17] | [15,17,14,4,6] |
| E | [11,11,10,10,10] | [10,11,3,3,10] | [10,10,11,10,11] |
| G | [17,11,13,18,4] | [17,18,11,12,11] | [18,12,4,11,17] |

---

## 4. Model Architecture (Inference Graph)

The model processes input through four sequential stages:

```
Input (1, C, 600) → ADBC → ECA₂ → [Sliding Window × 5] → Average → Softmax → Output (2,)
                                         ↓ per window:
                                    Improved CBAM → TCFN → Dense(2)
```

Where C = 5 for the reduced-channel models.

### 4.1 Complete Tensor Shape Flow (C=5 channels)

```
Layer                              Output Shape         Notes
─────────────────────────────────────────────────────────────────
Input                              (1, 1, 5, 600)       Raw EEG
Permute(3,2,1)                     (1, 600, 5, 1)       Reorder to (T, C, 1)

═══ ADBC Block ═══════════════════════════════════════════════════
Conv2D(16, (64,1), same, no bias)  (1, 600, 5, 16)      Temporal filtering
BatchNorm                          (1, 600, 5, 16)
ECA₁ attention                     (1, 600, 5, 16)      See §5.1

── Branch 1 (depth_multiplier=2) ─────────────────────────────────
DepthwiseConv2D((1,5), D=2)        (1, 600, 1, 32)      Spatial mixing
BatchNorm → ELU                    (1, 600, 1, 32)
AvgPool2D(8,1)                     (1, 75, 1, 32)       Temporal downsampling ×8
Conv2D(32, (16,1), same, no bias)  (1, 75, 1, 32)       Feature refinement
BatchNorm → ELU                    (1, 75, 1, 32)
AvgPool2D(7,1)                     (1, 10, 1, 32)       Temporal downsampling ×7

── Branch 2 (depth_multiplier=4) ─────────────────────────────────
DepthwiseConv2D((1,5), D=4)        (1, 600, 1, 64)      Deeper spatial mixing
BatchNorm → ELU                    (1, 600, 1, 64)
AvgPool2D(8,1)                     (1, 75, 1, 64)
Conv2D(32, (16,1), same, no bias)  (1, 75, 1, 32)       Project back to 32
BatchNorm → ELU                    (1, 75, 1, 32)
AvgPool2D(7,1)                     (1, 10, 1, 32)

── Merge ─────────────────────────────────────────────────────────
Add(Branch1, Branch2)              (1, 10, 1, 32)

═══ ECA₂ Attention ══════════════════════════════════════════════
ECA(in_ch=32)                      (1, 10, 1, 32)       See §5.1
Squeeze: x[:,:,-1,:]               (1, 10, 32)          Remove spatial dim

═══ Sliding Window (×5 windows) ═════════════════════════════════
For window i = 0..4:
  Slice [i : 6+i]                  (1, 6, 32)           6-step subsequence
  Improved CBAM                    (1, 6, 32)           See §5.2
  TCFN (depth=2)                   (1, 6, 32)           See §5.3
  Take last: x[:,-1,:]             (1, 32)
  Dense(2)                         (1, 2)               Per-window logits

Average(5 outputs)                 (1, 2)               Mean of 5 windows
Softmax                            (1, 2)               Final probabilities
```

> **Note for RTL**: Each of the 5 sliding windows has its **own independent weights** (not shared). This means 5 separate Improved CBAM blocks, 5 TCFN blocks, and 5 Dense layers must be instantiated or time-multiplexed.

---

## 5. Layer Specifications (Inference Only)

All dropout layers are **identity (no-op)** at inference. All BatchNorm layers use pre-computed running mean/variance (folded into the preceding convolution if desired).

### 5.1 ECA (Efficient Channel Attention)

Used twice: ECA₁ after the initial Conv2D (in_ch=16), and ECA₂ after the ADBC merge (in_ch=32).

**Adaptive kernel size formula**:
```
k = int(abs((log2(in_channel) + 1) / 2))
if k is even: k = k + 1
```
- For in_ch=16: k = int((4+1)/2) = 2 → odd-ify → **k=3**
- For in_ch=32: k = int((5+1)/2) = 3 → already odd → **k=3**

**Inference dataflow**:
```
input: (T, H, C)                          // e.g. (600, 5, 16)
  → GlobalAveragePool2D → (C,)            // mean over T and H
  → Reshape → (C, 1)
  → Conv1D(filters=1, kernel=3, same, no bias) → (C, 1)
  → Sigmoid → (C, 1)
  → Reshape → (1, 1, C)
  → Multiply(input, broadcast) → (T, H, C)
```

**Weights**: Conv1D kernel of shape `(3, 1, 1)` = **3 parameters** per ECA block.

### 5.2 Improved CBAM (Convolutional Block Attention Module)

Applied once per sliding window (5 instances total). Input shape: `(1, 6, 1, 32)` (expanded from 3D).

#### 5.2.1 Channel Attention Sub-module

```
input: (6, 1, 32)
  ├─ GlobalAvgPool2D → (32,) → Reshape(1,1,32)
  │    → Dense(4, ReLU, bias=yes)           // shared weights
  │    → Dense(32, linear, bias=yes)        // shared weights
  │
  └─ GlobalMaxPool2D → (32,) → Reshape(1,1,32)
       → Dense(4, ReLU, bias=yes)           // same weights as above
       → Dense(32, linear, bias=yes)        // same weights as above

  Add(avg_path, max_path) → Sigmoid → (1,1,32)
  Multiply(input, broadcast) → (6, 1, 32)
```

**Weights per instance**:
- Dense₁: (32×4) + 4 = **132**
- Dense₂: (4×32) + 32 = **160**
- Total: **292 parameters**

#### 5.2.2 Improved Spatial Attention Sub-module

```
input: (6, 1, 32)    // output of channel attention
  ├─ AvgPool across channels (axis=3, keepdims) → (6, 1, 1)
  ├─ MaxPool across channels (axis=3, keepdims) → (6, 1, 1)
  └─ StochasticPool across channels             → (6, 1, 1)

  Concat(axis=3) → (6, 1, 3)
  Conv2D(1, kernel=(7,1), same, sigmoid, no bias) → (6, 1, 1)
  Multiply(input, broadcast) → (6, 1, 32)
```

**Stochastic pooling** (deterministic at inference):
```
abs_x = |input|                           // element-wise absolute value
probs = abs_x / sum(abs_x, axis=channels) // normalize to probabilities
output = sum(probs * input, axis=channels) // weighted sum
```
This is a fixed computation — no learned parameters, no randomness at inference.

**Weights**: Conv2D kernel `(7, 1, 3, 1)` = **21 parameters** per instance.

**Total Improved CBAM per window**: 292 + 21 = **313 parameters**.

### 5.3 TCFN (Temporal Convolutional Fusion Network)

Applied once per sliding window (5 instances total). Input: `(1, 6, 32)`. Parameters: `depth=2, kernel_size=4, filters=32, activation=ELU`.

All Conv1D layers use `padding='causal'` (left-pad with zeros so output length equals input length).

#### Stage 0 (dilation_rate=1):
```
input_layer: (6, 32)
  → Conv1D(32, k=4, d=1, causal, he_uniform) → BN → ELU → (6, 32)    [block]
  → Conv1D(32, k=4, d=1, causal, he_uniform) → BN → ELU → (6, 32)    [block]
  Add(block, input_layer) → ELU → out                                  [Residual 1]
```

#### Stage 1 (dilation_rate=2):
```
out: (6, 32)
  → Conv1D(32, k=4, d=2, causal, he_uniform) → BN → ELU → (6, 32)    [block]
  Add(block, input_layer) → (6, 32)                                    [Residual 2]
  → Conv1D(32, k=4, d=2, causal, he_uniform) → BN → ELU → (6, 32)    [block]
  Add(block, input_layer) → ELU → out                                  [Residual 3]
```

> **Critical detail**: Residuals 2 and 3 add the **original `input_layer`** (input to the entire TCFN), not the output of Stage 0. At inference (no dropout), `input_layer` is constant across all residual connections.

**Weights per TCFN instance**:

| Layer | Weight Shape | Bias | Parameters |
|---|---|---|---|
| Conv1D ×4 | (4, 32, 32) each | (32,) each | 4 × (4096 + 32) = **16,512** |
| BatchNorm ×4 | γ(32), β(32) each | — | 4 × 64 = **256** |
| **Total** | | | **16,768** |

### 5.4 Dense + Softmax (Output Head)

Per sliding window: `Dense(2)` with `max_norm(0.25)` constraint (training only).

- Weights: (32, 2) = 64, Bias: (2,) → **66 parameters per window**
- 5 windows → 330 total
- `Average` layer: element-wise mean of 5 vectors of length 2 (no parameters)
- `Softmax`: `output[i] = exp(x[i]) / sum(exp(x))` (no parameters)

### 5.5 Activation Functions Reference

| Function | Formula | Used In |
|---|---|---|
| ELU | `x if x > 0, else α(exp(x) - 1)`, α=1.0 | ADBC, TCFN |
| Sigmoid | `1 / (1 + exp(-x))` | ECA, CBAM |
| Softmax | `exp(xᵢ) / Σⱼ exp(xⱼ)` | Final output |
| ReLU | `max(0, x)` | CBAM Dense₁ |

### 5.6 BatchNorm at Inference

At inference, BatchNorm is a fixed affine transform per channel:
```
output = γ * (input - μ) / sqrt(σ² + ε) + β
```
Where μ, σ² are the frozen running statistics, γ and β are learned scale/shift, ε = 0.001. This can be **fused** into the preceding Conv/Dense layer as a single multiply-add per channel.

---

## 6. Convolution Specifications Summary

### 6.1 ADBC Block (1 instance)

| # | Type | Kernel | Filters/D | Padding | Bias | Input → Output |
|---|---|---|---|---|---|---|
| 1 | Conv2D | (64, 1) | F=16 | same | No | (600,5,1)→(600,5,16) |
| 2 | DepthwiseConv2D | (1, 5) | D=2 | valid | No | (600,5,16)→(600,1,32) |
| 3 | Conv2D | (16, 1) | F=32 | same | No | (75,1,32)→(75,1,32) |
| 4 | DepthwiseConv2D | (1, 5) | D=4 | valid | No | (600,5,16)→(600,1,64) |
| 5 | Conv2D | (16, 1) | F=32 | same | No | (75,1,64)→(75,1,32) |

### 6.2 Per Sliding Window (×5 instances, independent weights)

| # | Type | Kernel | Filters | Dilation | Padding | Bias |
|---|---|---|---|---|---|---|
| 1 | Conv2D (CBAM spatial) | (7, 1) | 1 | 1 | same | No |
| 2 | Conv1D (TCFN) | 4 | 32 | 1 | causal | Yes |
| 3 | Conv1D (TCFN) | 4 | 32 | 1 | causal | Yes |
| 4 | Conv1D (TCFN) | 4 | 32 | 2 | causal | Yes |
| 5 | Conv1D (TCFN) | 4 | 32 | 2 | causal | Yes |

### 6.3 Pooling Operations

| Type | Size | Stride | Location |
|---|---|---|---|
| GlobalAvgPool2D | full spatial | — | ECA₁, ECA₂, CBAM channel attn |
| GlobalMaxPool2D | full spatial | — | CBAM channel attn |
| AvgPool2D | (8, 1) | (8, 1) | ADBC both branches, post-DepthwiseConv |
| AvgPool2D | (7, 1) | (7, 1) | ADBC both branches, post-Conv2D |

---

## 7. Weight Export and Loading

Trained weights are saved as HDF5 files (`best_model.weights.h5`). For FPGA deployment:

1. **Per-subject weights**: Each subject has independently trained weights (subject-dependent training with leave-one-session-out CV). The best fold's weights should be selected for deployment.

2. **Channel indices** (Gumbel approach only): Stored in `selected_channels.json` per subject/fold. The `selected_indices` array contains the 5 channel indices to use.

3. **Weight extraction**: Use `model.get_weights()` or `model.layers[i].get_weights()` to extract numpy arrays for each layer. BatchNorm layers export `[gamma, beta, running_mean, running_variance]`.

---

## 8. Inference Pipeline Summary

For a single trial classification:

```
1. ACQUIRE 600 samples from 19 EEG channels at 200 Hz (3 seconds)
2. SELECT 5 channels via static MUX or per-subject LUT
3. RESHAPE to (1, 1, 5, 600) and PERMUTE to (1, 600, 5, 1)
4. FORWARD PASS through the frozen model graph (§4.1)
5. READ softmax output: argmax gives predicted class (0=Right Hand, 1=Left Leg)
```

**Latency budget**: The entire forward pass must complete within the 3-second acquisition window of the next trial for real-time operation.

**Memory footprint**: All weights are static after loading. No dynamic memory allocation is required during inference. The intermediate activation buffers can be pre-allocated based on the shapes in §4.1.

---

## Appendix A: Gumbel Layer at Inference (Detail)

The `GumbelChannelSelection` layer at inference is purely a routing operation:

```python
# Learned logits matrix: shape (19, 5), frozen after training
# At inference:
selected_indices = argmax(logits, axis=0)  # → 5 indices, each in [0, 18]
# For each index k in 0..4:
output_channel[k] = input_channel[selected_indices[k]]
```

This is equivalent to a 5-element channel MUX. The logits matrix is **not used at inference** beyond the argmax — only the resulting indices matter. These can be hardcoded.

**Duplication**: If `selected_indices = [10, 10, 11, 10, 11]`, the model receives channel F7 three times and F8 twice. This is by design — the downstream weights were trained expecting this specific (possibly redundant) input arrangement.

## Appendix B: Stochastic Pooling (Deterministic at Inference)

The improved spatial attention in CBAM uses stochastic pooling as a third pooling descriptor alongside average and max pooling. At inference this is a deterministic weighted sum:

```
For input tensor X of shape (T, H, C):
  abs_X = |X|                                    // element-wise
  norm = sum(abs_X, axis=channel_axis)           // (T, H, 1)
  norm = max(norm, 1e-7)                         // avoid division by zero
  probs = abs_X / norm                           // (T, H, C)
  output = sum(probs * X, axis=channel_axis)     // (T, H, 1)
```

No random sampling occurs. The output is deterministic given the input.
