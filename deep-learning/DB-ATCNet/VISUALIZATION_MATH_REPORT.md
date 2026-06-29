# DB-ATCNet Visualization — Mathematical Validation Report (v2 CORRECTED)

> **v2 Changes**: Fixed 3 critical bugs found during self-audit. Layer matching now uses **output tensor shapes** instead of brittle layer name matching. All tensor dimensions verified against the actual model architecture.

---

## 0. Input Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $C$ | 19 | Number of EEG channels |
| $T$ | 600 | Number of time samples (3.0s × 200Hz) |
| $F_1$ | 16 | Temporal filters (first Conv2D) |
| $D_1, D_2$ | 2, 4 | Depth multipliers for ADBC Branch 1/2 |
| $F_2$ | $F_1 \times D = 32$ | Output channels after spatial mixing |
| $K_{temp}$ | 64 | Temporal Conv2D kernel size |
| $P_1$ | 8 | First AveragePooling2D factor |
| $P_2$ | 7 | Second AveragePooling2D factor (`eegn_poolSize`) |
| $N_w$ | 5 | Number of sliding windows |

---

## 1. Temporal Kernels Plot (`temporal_kernels.png`)

### Calculation

```python
W = temporal_layer.get_weights()[0]   # Shape: (64, 1, 1, 16)
axes[i].plot(W[:, 0, 0, i])
```

$$\text{kernel}_i[t] = W_{temp}[t, 0, 0, i], \quad t \in \{0, \ldots, 63\}$$

| Axis | Definition | Range |
|------|-----------|-------|
| **X** | Sample index within the 64-tap filter | 0 – 63 |
| **Y** | Raw learned weight value | Unbounded float |

**No data-dependent computation.** Pure model parameters.

---

## 2. Summary Overview Heatmaps (`summary_overview.png`)

### How layers are selected (CORRECTED in v2)

The old code matched layers by name (`'conv2d'`, `'depthwise'`, `'conv1d'`), which was **broken** — the DepthwiseConv2D name was never found at the top level, and the first Conv1D was from ECA attention (shape `(16,1)`), not TCFN.

**New logic** uses output tensor shape dimensions:

```python
t_dim, f_dim = act_2d.shape
if     low_act is None and t_dim >= 500:              low_act  = ...  # ~600 time steps
elif   mid_act is None and 5 < t_dim < 100 and f_dim >= 16: mid_act = ...  # ~75 pooled steps
elif   t_dim <= 10 and f_dim >= 16:                   high_act = ...  # ~6 TCFN steps
```

---

### 2a. Raw EEG Trace (Top panel)

$$\text{plot}_{ch}(t) = X[ch, t] + ch \times s, \quad s = 1.5 \times \max|X|$$

| Axis | Definition | Range |
|------|-----------|-------|
| **X** | Real time (seconds) | 0.0 – 3.0 |
| **Y** | 19 channels with vertical offset | Fp1 … Pz |

---

### 2b. Low-Level Feature Map

**Source layer**: First `Conv2D` output (before any pooling).

$$X_{perm} \in \mathbb{R}^{600 \times 19 \times 1} \xrightarrow{\text{Conv2D}} A_{low}^{raw} \in \mathbb{R}^{600 \times 19 \times 16}$$

Squeeze removes batch → `(600, 19, 16)`. Average across channels (axis=1):

$$A_{low}[t, f] = \frac{1}{19} \sum_{c=0}^{18} A_{low}^{raw}[t, c, f] \quad \Rightarrow \quad A_{low} \in \mathbb{R}^{600 \times 16}$$

Transpose: $\text{heatmap} = A_{low}^T \in \mathbb{R}^{16 \times 600}$

| Axis | What it means | Value |
|------|--------------|-------|
| **X (columns)** | Time samples at full 200Hz resolution | 0 – 600. Each column = **5 ms** |
| **Y (rows)** | 16 learned frequency filter responses | 0 – 15 |
| **Color** | Activation magnitude | Viridis (yellow = strong match) |

---

### 2c. Mid-Level Feature Map (CORRECTED in v2)

**Source layer**: First layer after AveragePooling2D whose time dimension is between 5 and 100.

The ADBC block applies:
1. `DepthwiseConv2D(1, 19)` → collapses 19 channels to 1 → `(600, 1, 32)`
2. `AveragePooling2D(8, 1)` → divides time by 8 → `(75, 1, 32)`

The first post-pooling layer has output `(batch, 75, 1, 32)`.

Squeeze: `(75, 32)` (the size-1 spatial dim is removed). Already 2D, no averaging needed.

Transpose: $\text{heatmap} = A_{mid}^T \in \mathbb{R}^{32 \times 75}$

| Axis | What it means | Value |
|------|--------------|-------|
| **X (columns)** | Pooled time chunks. $T_{mid} = \lfloor 600/8 \rfloor = 75$ | 0 – 75. Each column ≈ **40 ms** |
| **Y (rows)** | 32 spatial-temporal feature combinations | 0 – 31 |
| **Color** | Activation magnitude | Plasma colormap |

> **v1 bug**: The old code never populated this panel because it searched for a layer named `'depthwise'` but the DepthwiseConv2D output was `(600, 1, 32)` — same as low-level resolution. The new shape-based matching captures the first **post-pooling** layer instead.

---

### 2d. High-Level Feature Map (CORRECTED in v2)

**Source layer**: Last layer with output `(6, 32)` — from the TCFN sliding window blocks.

The path to this shape:
1. ADBC output after both pools: `(10, 1, 32)`
2. `Lambda(x[:,:,-1,:])` squeeze: `(10, 32)`
3. Sliding window: `block1[:, 0:6, :]` → `(6, 32)`
4. MHA + TCFN Conv1D: `(6, 32)`

Squeeze: `(6, 32)`. Already 2D.

Transpose: $\text{heatmap} = A_{high}^T \in \mathbb{R}^{32 \times 6}$

| Axis | What it means | Value |
|------|--------------|-------|
| **X (columns)** | $T_{high} = T_{mid}/P_2 - N_w + 1 = 75/7 - 5 + 1 = 6$ | 0 – 6. Each column ≈ **500 ms** |
| **Y (rows)** | 32 deep temporal sequence features | 0 – 31 |
| **Color** | Activation magnitude | Inferno colormap |

> **v1 bug**: The old code matched the first `conv1d` layer, which was from `eca_attention` (output shape `(16, 1)` → squeeze → 1D → crash). The new code skips 1D tensors and only accepts shapes with `f_dim >= 16`.

---

### X-axis Progression Summary (CORRECTED)

| Level | X-axis | Pooling math | Each column = |
|-------|--------|-------------|---------------|
| Raw EEG | 600 | None | 5 ms |
| Low-Level | 600 | `padding='same'` preserves length | 5 ms |
| Mid-Level | **~75** | $\lfloor 600 / 8 \rfloor = 75$ | ~40 ms |
| High-Level | **6** | $\lfloor 75 / 7 \rfloor - 5 + 1 = 6$ | ~500 ms |

---

## 3. Spatial Channel Importance (`spatial_importance.png`)

### Calculation

**Step 1 — Activation magnitude per channel:**

$$\text{act\_mag}[c] = \frac{1}{T \times F_1} \sum_{t,f} |A_{conv}[t, c, f]|, \quad c \in \{0, \ldots, 18\}$$

**Step 2 — Weight magnitude per channel:**

$$\text{weight\_mag}[c] = \frac{1}{D} \sum_{d} |W_{spatial}[0, c, 0, d]|$$

**Step 3 — Combined importance:**

$$I[c] = \text{act\_mag}[c] \times \text{weight\_mag}[c]$$

| Axis | Definition |
|------|-----------|
| **X** | 19 physical EEG electrodes (Fp1 … Pz) |
| **Y** | Combined importance score $I[c] \geq 0$ |

> **Caveat**: `temporal_act` is the raw Conv2D output (before BatchNorm/ECA). The magnitude scale may differ slightly from what the DepthwiseConv2D actually receives.

---

## 4. Attention Focus Overlay (`attention_overlay.png`)

### Calculation

**Step 1 — Collapse filters:**

$$\text{focus\_raw}[t] = \frac{1}{32} \sum_{f=0}^{31} \text{heatmap}_{high}[f, t], \quad t \in \{0, \ldots, 5\}$$

**Step 2 — Normalize to [0, 1]:**

$$\text{focus}[t] = \frac{\text{focus\_raw}[t] - \min(\text{focus\_raw})}{\max(\text{focus\_raw}) - \min(\text{focus\_raw}) + 10^{-9}}$$

**Step 3 — Map to real time:**

$$\text{time}[t] = \frac{3.0 \times t}{5} \quad \Rightarrow \quad [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]$$

| Axis | Definition |
|------|-----------|
| **X (top)** | Real time 0–3s (raw EEG) |
| **X (bottom)** | Same 0–3s axis (focus curve aligned) |
| **Y (bottom)** | Normalized focus intensity 0–100% |
| **Red line** | $t^* = \arg\max(\text{focus})$ — peak attention timestamp |

> **Important caveat**: This is NOT the actual `MultiHeadAttention` query-key dot-product. It is the **average activation magnitude of the deepest TCN layer**, which reflects where the model's final decision-making neurons fired strongest. The MHA scores feed INTO the TCN, so they are correlated but not identical.

---

## 5. Full Tensor Shape Flow (CORRECTED)

```
Input:                (1, 1, 19, 600)
  ↓ Permute
                      (1, 600, 19, 1)
  ↓ Conv2D(F1=16, k=64, same)
                      (1, 600, 19, 16)        ← LOW-LEVEL heatmap
  ↓ BatchNorm + ECA
  ↓ DepthwiseConv2D(1×19, D=2)
                      (1, 600, 1, 32)
  ↓ BatchNorm + ELU
  ↓ AveragePooling2D(8, 1)
                      (1, 75, 1, 32)          ← MID-LEVEL heatmap
  ↓ Dropout
  ↓ Conv2D(F2=32, k=16, same)
                      (1, 75, 1, 32)
  ↓ BatchNorm + ELU
  ↓ AveragePooling2D(7, 1)
                      (1, 10, 1, 32)
  ↓ Dropout
  ↓ [Branch 2 similar path, then Add]
  ↓ Dropout(drop1)
ADBC output:          (1, 10, 1, 32)
  ↓ ECA attention
  ↓ Lambda(x[:,:,-1,:])
                      (1, 10, 32)
  ↓ Sliding window [0:6]
                      (1, 6, 32)
  ↓ MHA(key_dim=8, heads=2)
                      (1, 6, 32)
  ↓ TCFN Conv1D(k=4, f=32, causal) × depth
                      (1, 6, 32)              ← HIGH-LEVEL heatmap
  ↓ Lambda(x[:,-1,:])
                      (1, 32)
  ↓ Dense(2) + Softmax
                      (1, 2)                  ← Prediction
```
