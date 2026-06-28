#!/usr/bin/env python3
"""
q88_layer4.py — Bit-accurate Q8.8 Python reference for Branch A's depthwise
spatial conv (Layer 4) of HaLT DB-ATCNet, with the following BatchNorm folded
in. This is the FIRST stage of Branch A; the subsequent ELU + AvgPool +
separable conv + BN + ELU + AvgPool are separate references.

Spec (docs/FPGA_DEPLOYMENT_DOC.md §4.1, §6.1):
    DepthwiseConv2D(kernel=(1, NUM_INPUT_CH=5), depth_multiplier=2,
                    padding='valid', data_format='channels_last',
                    use_bias=False)
    BatchNorm(axis=-1)

Input shape (after ECA₁ in channels_last): (T=600, C=5, F1=16)
Kernel shape:                              (1, 5, F1=16, D=2)
Output shape:                              (T=600, 1, F1*D=32)

For each output channel i = f*D + d, with f = i // D and d = i % D:
    y[t, 0, i] = sum_{c=0..C-1} x[t, c, f] * kernel[0, c, f, d]

BN folding (matches Layer 1+2 pattern):
    scale[i] = gamma[i] / sqrt(var[i] + eps)
    bias[i]  = beta[i] - mean[i] * scale[i]
    kernel_folded[0, c, f, d] = kernel[0, c, f, d] * scale[f*D + d]

The Q8.8 sum-of-products is rescaled by >>FRAC_BITS, then bias[i] is added,
then saturated to int16. ELU is NOT applied here (separate stage).

Artifacts written to data/golden_q88/:
    stage_branchA_dwise_input.hex     (input  = stage_eca1_output.hex, T*C*F1 = 48000)
    stage_branchA_dwise_weights.hex   (folded kernel, C*F1*D = 5*16*2 = 160 values)
    stage_branchA_dwise_bias.hex      (folded bias, F1*D = 32 values)
    stage_branchA_dwise_output.hex    (pre-ELU output, T*1*F2 = 600*32 = 19200 values)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
ECA1_OUT_HEX       = "data/golden_q88/stage_eca1_output.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8
ACC_WIDTH  = 48

# Branch A (D=2) parameters
T  = 600
C  = 5
F1 = 16
D  = 2
F2 = F1 * D            # 32 output channels


def quantize_q88(arr: np.ndarray) -> np.ndarray:
    scale = 1 << FRAC_BITS
    hi = (1 << (DATA_WIDTH - 1)) - 1
    lo = -(1 << (DATA_WIDTH - 1))
    q = np.round(arr * scale).astype(np.int64)
    return np.clip(q, lo, hi).astype(np.int16)


def to_hex(q: np.ndarray, width: int = 16) -> str:
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    flat = q.flatten()
    return "\n".join(f"{int(v) & mask:0{digits}X}" for v in flat) + "\n"


def load_q88_hex(path: str, shape: tuple) -> np.ndarray:
    flat = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            v = int(line, 16)
            if v >= (1 << (DATA_WIDTH - 1)):
                v -= (1 << DATA_WIDTH)
            flat.append(v)
    a = np.array(flat, dtype=np.int16)
    assert a.size == np.prod(shape), f"hex size {a.size} != expected {np.prod(shape)} for {path}"
    return a.reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load weights + BN params, fold ----------
    print(f"Loading Branch A depthwise + BN_1 from {args.weights}...")
    with h5py.File(args.weights, "r") as f:
        w     = f["layers/depthwise_conv2d/vars/0"][()].astype(np.float32)     # (1, 5, F1, D)
        bn_g  = f["layers/batch_normalization_1/vars/0"][()].astype(np.float32)  # gamma  (F2,)
        bn_b  = f["layers/batch_normalization_1/vars/1"][()].astype(np.float32)  # beta   (F2,)
        bn_m  = f["layers/batch_normalization_1/vars/2"][()].astype(np.float32)  # mean   (F2,)
        bn_v  = f["layers/batch_normalization_1/vars/3"][()].astype(np.float32)  # var    (F2,)
    eps = 1e-3
    assert w.shape == (1, C, F1, D), f"unexpected depthwise shape {w.shape}"
    print(f"  depthwise weights shape={w.shape}, abs_max={np.abs(w).max():.4f}")
    print(f"  BN_1 gamma abs_max={np.abs(bn_g).max():.4f}, beta abs_max={np.abs(bn_b).max():.4f}")

    # Fold BN into per-output-channel kernel + bias.
    # The 4D kernel (1, C, F1, D) and the BN are indexed by output channel
    # i = f*D + d, so we reshape gamma/beta/mean/var to (F1, D) for clean
    # broadcasting and an explicit dependency on the (f, d) split.
    scale = (bn_g / np.sqrt(bn_v + eps)).reshape(F1, D)             # (F1, D)
    bias  = (bn_b - bn_m * (bn_g / np.sqrt(bn_v + eps))).reshape(F1, D)  # (F1, D)
    w_folded = w[0] * scale[np.newaxis, :, :]                       # (C, F1, D)
    print(f"  folded kernel abs_max={np.abs(w_folded).max():.4f}, folded bias abs_max={np.abs(bias).max():.4f}")

    # ---------- Load input (gated feature map from ECA₁) ----------
    print(f"Loading input from {ECA1_OUT_HEX}...")
    x_q = load_q88_hex(ECA1_OUT_HEX, (T, C, F1))                    # int16
    print(f"  input shape={x_q.shape}, abs_max={int(np.abs(x_q).max())}")

    # ---------- Quantize folded kernel + bias ----------
    qw = quantize_q88(w_folded)                                     # (C, F1, D) int16
    qb = quantize_q88(bias)                                          # (F1, D) int16

    # ---------- Q8.8 depthwise: per output channel i=f*D+d ----------
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))

    # acc[t, f, d] = sum_c x[t,c,f] * w_folded[c,f,d]
    # Use einsum for clarity: t c f, c f d -> t f d
    acc = np.einsum('tcf, cfd -> tfd',
                    x_q.astype(np.int64),
                    qw.astype(np.int64))                            # (T, F1, D)
    shifted   = acc >> FRAC_BITS
    with_bias = shifted + qb[np.newaxis, :, :].astype(np.int64)     # broadcast bias
    q_out = np.clip(with_bias, LO, HI).astype(np.int16)             # (T, F1, D)
    n_sat = int(((with_bias > HI) | (with_bias < LO)).sum())
    # Flatten to (T, F2) with output channel ordering i = f*D + d
    q_out_flat = q_out.reshape(T, F2)                                # (T, F2)
    print(f"  output shape={q_out_flat.shape}, abs_max={int(np.abs(q_out_flat).max())}, sat={n_sat}/{q_out_flat.size}")

    # ---------- HEX files ----------
    # Weights: streaming order is (electrode_idx c, output channel i) where i=f*D+d.
    # Per cycle, the depthwise module sees one electrode worth of input
    # (16-vector of x_in[f]) and needs the corresponding kernel column
    # w_folded[c, f, d] for all i = f*D + d.  Linearize as:
    #     address = c * F2 + i        (c slow, i fast)
    # so the HW reads 32 contiguous weights per (electrode) cycle.
    qw_flat = qw.transpose(0, 1, 2).reshape(C, F2)                  # (C, F2) since i=f*D+d
    bias_flat = qb.reshape(F2)                                       # (F2,)

    in_hex   = OUT_DIR / "stage_branchA_dwise_input.hex"     # symlink-like; matches stage_eca1_output.hex
    w_hex    = OUT_DIR / "stage_branchA_dwise_weights.hex"
    bias_hex = OUT_DIR / "stage_branchA_dwise_bias.hex"
    out_hex  = OUT_DIR / "stage_branchA_dwise_output.hex"
    # Reuse the eca1 output as input (don't duplicate the 48k-line file)
    # but ALSO write a symlink-friendly copy so the TB can $readmemh from a stable path.
    in_hex.write_text(Path(ECA1_OUT_HEX).read_text())
    w_hex.write_text(to_hex(qw_flat, COEF_WIDTH))
    bias_hex.write_text(to_hex(bias_flat, DATA_WIDTH))
    out_hex.write_text(to_hex(q_out_flat, DATA_WIDTH))
    print()
    print(f"wrote {in_hex}     ({x_q.size} values, copy of ECA₁ output)")
    print(f"wrote {w_hex}   ({qw_flat.size} values, folded; layout c-slow / i-fast)")
    print(f"wrote {bias_hex}      ({bias_flat.size} values, folded bias)")
    print(f"wrote {out_hex}    ({q_out_flat.size} values, pre-ELU)")

    # ---------- Meta ----------
    meta = {
        "layer":             "branchA_depthwise + bn_1 (Layer 4 fused)",
        "bn_folded":         True,
        "bn_eps":            eps,
        "weights_file":      args.weights,
        "data_width":        DATA_WIDTH,
        "coef_width":        COEF_WIDTH,
        "frac_bits":         FRAC_BITS,
        "T":                 T,
        "C":                 C,
        "F1":                F1,
        "D":                 D,
        "F2":                F2,
        "input_order":       "(t slow, c, f fast)  same as stage_eca1_output.hex",
        "weight_order":      "(c slow, i=f*D+d fast)  -> address = c*F2 + i",
        "bias_order":        "(i = f*D + d)",
        "output_order":      "(t slow, i fast)     T*F2 = 600*32 = 19200",
        "input_shape":       [T, C, F1],
        "weight_shape":      [C, F2],
        "bias_shape":        [F2],
        "output_shape":      [T, F2],
        "abs_max_q_out":     int(np.abs(q_out_flat).max()),
        "saturation_count":  n_sat,
    }
    (OUT_DIR / "stage_branchA_dwise_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / 'stage_branchA_dwise_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
