#!/usr/bin/env python3
"""
q88_layer1.py — Bit-accurate Q8.8 Python reference for fused Layer 1 + 2
(temporal Conv2D + BatchNorm) of the HaLT-variant DB-ATCNet.

BatchNorm is folded into the Conv at export time. Mathematically:
    y = gamma * (Conv2D(x) - mean) / sqrt(var + eps) + beta
      = scale[f] * Conv2D(x) + bias[f]
where:
    scale[f] = gamma[f] / sqrt(var[f] + eps)        # absorbed into w
    bias[f]  = beta[f]  - mean[f] * scale[f]        # added as per-filter Q8.8 bias

This eliminates a runtime BN module; the RTL just applies the folded
weights and the folded bias as part of conv2d_temporal.sv.

Layer 1 spec (from docs/FPGA_DEPLOYMENT_DOC.md §4.1):
    Conv2D(filters=F1=16, kernel_size=(KE=64, 1),
           padding='same', data_format='channels_last', use_bias=False)
    BatchNorm(axis=-1)

Input shape (Keras channels_last after Permute(3,2,1)):
    (B=1, T=TRIAL_SAMPLES=600, C=NUM_INPUT_CH=5, in_ch=1)
Kernel shape:
    (KE=64, kw=1, in_ch=1, F1=16)   # SAME as the old subject-independent model
Output shape:
    (B=1, T=600, C=5, F1=16)

Padding 'same' with stride 1, kernel 64, Keras convention:
    pad_top = (KE-1)//2 = 31    # zeros prepended  on the time axis
    pad_bot = KE - 1 - pad_top = 32  # zeros appended

Q8.8 dataflow per output element:
    acc = sum_{k=0..KE-1} q_in[t - pad_top + k, c] * q_w[k, f]   (int64)
    out = clip(acc >> 8, [-32768, 32767])                         (int16)

Artifacts written to data/golden_q88/:
    stage_conv2d_input.hex      Q8.8 input   (T*C = 3000 values)
    stage_conv2d_weights.hex    Q8.8 weights (KE*F1 = 1024 values)
    stage_conv2d_output.hex     Q8.8 expected output (T*C*F1 = 48000 values)
    stage_conv2d_meta.json      shapes + scale + saturation stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8
ACC_WIDTH  = 48

KE = 64
F1 = 16
T  = 600
C  = 5


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


def make_synthetic_input(seed: int = 0xA7C) -> np.ndarray:
    """
    Deterministic 5-channel synthetic EEG trial. Returns shape (T, C) of
    float32 values in roughly the same magnitude range as real EEG after
    typical preprocessing (~0.5 max). Matches what the PS would feed the PL
    after Gumbel channel selection.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 200.0   # seconds @ 200 Hz
    x = np.zeros((T, C), dtype=np.float32)
    # Per-channel low-freq sines with phase offsets — different freq per chan
    # so the 5 channels are linearly independent.
    freqs   = [10.0, 12.0, 14.0, 16.0, 18.0]
    phases  = [0.0, 0.7, 1.4, 2.1, 2.8]
    amps    = [0.35, 0.4, 0.3, 0.45, 0.32]
    for c in range(C):
        x[:, c] = amps[c] * np.sin(2 * np.pi * freqs[c] * t + phases[c])
    x += 0.02 * rng.standard_normal(x.shape).astype(np.float32)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    ap.add_argument("--input", default=None,
                    help="optional .npy file of shape (T,C) to use instead of the synthetic input")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load weights + BN params, fold BN into Conv ----------
    print(f"Loading Conv2D + BN weights from {args.weights}...")
    with h5py.File(args.weights, "r") as f:
        w     = f["layers/conv2d/vars/0"][()].astype(np.float32)            # (KE, 1, 1, F1)
        bn_g  = f["layers/batch_normalization/vars/0"][()].astype(np.float32) # gamma  (F1,)
        bn_b  = f["layers/batch_normalization/vars/1"][()].astype(np.float32) # beta   (F1,)
        bn_m  = f["layers/batch_normalization/vars/2"][()].astype(np.float32) # mean   (F1,)
        bn_v  = f["layers/batch_normalization/vars/3"][()].astype(np.float32) # var    (F1,)
    eps = 1e-3   # Keras BatchNormalization default
    print(f"  conv  weights shape={w.shape}, abs_max={np.abs(w).max():.4f}")
    print(f"  BN gamma abs_max={np.abs(bn_g).max():.4f}, beta abs_max={np.abs(bn_b).max():.4f}")
    print(f"  BN mean  abs_max={np.abs(bn_m).max():.4f}, var  abs_max={np.abs(bn_v).max():.4f}")

    # Fold:  y = gamma*(Conv-mean)/sqrt(var+eps) + beta
    #          = scale[f] * Conv + bias[f]
    # where scale[f] = gamma[f]/sqrt(var[f]+eps), bias[f] = beta[f] - mean[f]*scale[f]
    scale = bn_g / np.sqrt(bn_v + eps)                # (F1,)
    bias  = bn_b - bn_m * scale                        # (F1,)
    w     = w * scale[np.newaxis, np.newaxis, np.newaxis, :]   # broadcast over (KE,1,1,F1)
    print(f"  folded weights abs_max={np.abs(w).max():.4f}, folded bias abs_max={np.abs(bias).max():.4f}")

    # ---------- Input ----------
    if args.input:
        x_real = np.load(args.input).astype(np.float32)
        assert x_real.shape == (T, C), f"input shape must be ({T},{C}), got {x_real.shape}"
    else:
        x_real = make_synthetic_input()
    print(f"  input shape={x_real.shape}, abs_max={np.abs(x_real).max():.4f}")

    # ---------- Quantize ----------
    qx = quantize_q88(x_real)                          # (T, C) int16
    qw = quantize_q88(w[:, 0, 0, :])                   # (KE, F1) int16   (BN-folded)
    qb = quantize_q88(bias)                            # (F1,) int16      (BN-folded)
    print(f"  q_in abs_max={int(np.abs(qx).max())}  (Q8.8 limit 32767)")
    print(f"  q_w  abs_max={int(np.abs(qw).max())}")
    print(f"  q_b  abs_max={int(np.abs(qb).max())}")

    # ---------- Q8.8 Conv2D with 'same' padding ----------
    pad_top = (KE - 1) // 2     # 31
    pad_bot = KE - 1 - pad_top  # 32

    qx_padded = np.zeros((T + KE - 1, C), dtype=np.int64)
    qx_padded[pad_top : pad_top + T] = qx.astype(np.int64)

    acc = np.zeros((T, C, F1), dtype=np.int64)
    for k in range(KE):
        # qx_padded[k:k+T]    -> (T, C)
        # qw[k]               -> (F1,)
        acc += qx_padded[k : k + T, :, np.newaxis] * qw[k, np.newaxis, np.newaxis, :].astype(np.int64)

    # Rescale to Q8.8, add folded bias, then saturate to int16.
    # The HW does:  shifted = acc >>> FRAC_BITS; with_bias = shifted + bias; out = sat(with_bias).
    shifted   = acc >> FRAC_BITS
    with_bias = shifted + qb[np.newaxis, np.newaxis, :].astype(np.int64)
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    q_out = np.clip(with_bias, LO, HI).astype(np.int16)
    n_sat = int(((with_bias > HI) | (with_bias < LO)).sum())

    print(f"  output shape={q_out.shape}, abs_max={int(np.abs(q_out).max())}, sat={n_sat}/{q_out.size}")

    # ---------- Write HEX files ----------
    in_hex   = OUT_DIR / "stage_conv2d_input.hex"
    w_hex    = OUT_DIR / "stage_conv2d_weights.hex"
    bias_hex = OUT_DIR / "stage_conv2d_bias.hex"
    out_hex  = OUT_DIR / "stage_conv2d_output.hex"
    in_hex.write_text(to_hex(qx, DATA_WIDTH))
    w_hex.write_text(to_hex(qw, COEF_WIDTH))
    bias_hex.write_text(to_hex(qb, DATA_WIDTH))
    out_hex.write_text(to_hex(q_out, DATA_WIDTH))

    print()
    print(f"wrote {in_hex}     ({qx.size} values)")
    print(f"wrote {w_hex}   ({qw.size} values, BN-folded)")
    print(f"wrote {bias_hex}    ({qb.size} values, BN-folded bias)")
    print(f"wrote {out_hex} ({q_out.size} values)")

    # ---------- Meta ----------
    meta = {
        "layer":         "conv2d + bn (Layer 1+2 fused, F1=16, KE=64, padding='same')",
        "bn_folded":     True,
        "bn_eps":        eps,
        "model_variant": "HaLT subject-dependent",
        "weights_file":  args.weights,
        "data_width":    DATA_WIDTH,
        "coef_width":    COEF_WIDTH,
        "frac_bits":     FRAC_BITS,
        "acc_width":     ACC_WIDTH,
        "T":             T,
        "C":             C,
        "F1":            F1,
        "KE":            KE,
        "pad_top":       pad_top,
        "pad_bot":       pad_bot,
        "input_order":   "time (slowest) -> electrode (fastest)",
        "weight_order":  "tap (slowest) -> filter",
        "bias_order":    "filter",
        "output_order":  "time (slowest) -> electrode -> filter",
        "input_shape":   list(qx.shape),
        "weight_shape":  list(qw.shape),
        "bias_shape":    list(qb.shape),
        "output_shape":  list(q_out.shape),
        "abs_max_real_w_folded":   float(np.abs(w).max()),
        "abs_max_real_bias":       float(np.abs(bias).max()),
        "abs_max_real_in":         float(np.abs(x_real).max()),
        "abs_max_q_out":           int(np.abs(q_out).max()),
        "saturation_count":        n_sat,
    }
    (OUT_DIR / "stage_conv2d_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / 'stage_conv2d_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
