#!/usr/bin/env python3
"""
q88_layer7.py — Bit-accurate Q8.8 Python reference for Branch A's separable
Conv2D (Layer 7) of HaLT DB-ATCNet, with the following BN folded in.

Spec (docs/FPGA_DEPLOYMENT_DOC.md §4.1, §6.1):
    Conv2D(filters=F2=32, kernel_size=(KE=16, 1), padding='same',
           data_format='channels_last', use_bias=False)
    BatchNorm(axis=-1)

Input shape (post pool1):  (T_POOL=75, 1, F2=32)
Kernel shape:              (16, 1, 32, 32)   # (kh, kw, in_ch, out_ch)
Output shape:              (T_POOL=75, 1, F2=32)

Padding 'same' with stride 1, kernel KE=16:
    pad_top = (KE-1)//2 = 7
    pad_bot = KE - 1 - pad_top = 8

Math (BN folded):
    For each output channel f_out, time t:
        y[t, f_out] = sat( ( sum_{k=0..KE-1} sum_{f_in=0..F2-1}
                              x[t - pad_top + k, f_in]
                              * W_folded[k, f_in, f_out] )
                            >>> FRAC_BITS  +  bias_folded[f_out] )

BN fold:
    scale[f_out] = gamma[f_out] / sqrt(var[f_out] + eps)
    bias[f_out]  = beta[f_out] - mean[f_out] * scale[f_out]
    W_folded[k, f_in, f_out] = W[k, 0, f_in, f_out] * scale[f_out]

Artifacts:
    stage_branchA_sep_input.hex     = stage_branchA_pool1_output.hex  (75*32 = 2400)
    stage_branchA_sep_weights.hex   folded kernel (16*32*32 = 16384 values)
    stage_branchA_sep_bias.hex      folded bias   (32)
    stage_branchA_sep_output.hex    pre-ELU output (75*32 = 2400)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
INPUT_HEX          = "data/golden_q88/stage_branchA_pool1_output.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8
ACC_WIDTH  = 48

T_POOL = 75
F2     = 32
KE     = 16


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
    print(f"Loading Branch A separable + BN_3 from {args.weights}...")
    with h5py.File(args.weights, "r") as f:
        w     = f["layers/conv2d_1/vars/0"][()].astype(np.float32)          # (KE, 1, F2, F2)
        bn_g  = f["layers/batch_normalization_3/vars/0"][()].astype(np.float32)
        bn_b  = f["layers/batch_normalization_3/vars/1"][()].astype(np.float32)
        bn_m  = f["layers/batch_normalization_3/vars/2"][()].astype(np.float32)
        bn_v  = f["layers/batch_normalization_3/vars/3"][()].astype(np.float32)
    eps = 1e-3
    assert w.shape == (KE, 1, F2, F2)
    print(f"  conv weights shape={w.shape}, abs_max={np.abs(w).max():.4f}")
    print(f"  BN_3 gamma abs_max={np.abs(bn_g).max():.4f}, beta abs_max={np.abs(bn_b).max():.4f}")
    print(f"  BN_3 mean  abs_max={np.abs(bn_m).max():.4f}, var  abs_max={np.abs(bn_v).max():.4f}")

    scale    = bn_g / np.sqrt(bn_v + eps)        # (F2,)
    bias     = bn_b - bn_m * scale               # (F2,)
    w_folded = w[:, 0, :, :] * scale[np.newaxis, np.newaxis, :]   # (KE, F2_in, F2_out)
    print(f"  folded weights abs_max={np.abs(w_folded).max():.4f}, "
          f"bias abs_max={np.abs(bias).max():.4f}")

    # ---------- Load input ----------
    print(f"Loading input from {INPUT_HEX}...")
    x_q = load_q88_hex(INPUT_HEX, (T_POOL, F2))   # (T_POOL, F2)
    print(f"  input shape={x_q.shape}, abs_max={int(np.abs(x_q).max())}")

    # ---------- Quantize folded weights + bias ----------
    qw = quantize_q88(w_folded)                   # (KE, F2_in, F2_out) int16
    qb = quantize_q88(bias)                       # (F2,)

    # ---------- Q8.8 Conv2D with 'same' padding on the time axis ----------
    pad_top = (KE - 1) // 2     # 7
    pad_bot = KE - 1 - pad_top  # 8

    x_padded = np.zeros((T_POOL + KE - 1, F2), dtype=np.int64)
    x_padded[pad_top : pad_top + T_POOL] = x_q.astype(np.int64)

    acc = np.zeros((T_POOL, F2), dtype=np.int64)
    # acc[t, f_out] = sum_k sum_f_in x_padded[k+t, f_in] * qw[k, f_in, f_out]
    for k in range(KE):
        acc += np.einsum('ti, ij -> tj',
                         x_padded[k : k + T_POOL].astype(np.int64),
                         qw[k].astype(np.int64))

    shifted   = acc >> FRAC_BITS
    with_bias = shifted + qb[np.newaxis, :].astype(np.int64)
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    q_out = np.clip(with_bias, LO, HI).astype(np.int16)
    n_sat = int(((with_bias > HI) | (with_bias < LO)).sum())
    print(f"  output shape={q_out.shape}, abs_max={int(np.abs(q_out).max())}, sat={n_sat}/{q_out.size}")

    # ---------- Write HEX files ----------
    # Weights layout: (k slow, f_in, f_out fast)
    # i.e. address = k * F2 * F2 + f_in * F2 + f_out
    qw_flat = qw.reshape(KE * F2 * F2)
    bias_flat = qb.reshape(F2)

    in_hex   = OUT_DIR / "stage_branchA_sep_input.hex"
    w_hex    = OUT_DIR / "stage_branchA_sep_weights.hex"
    bias_hex = OUT_DIR / "stage_branchA_sep_bias.hex"
    out_hex  = OUT_DIR / "stage_branchA_sep_output.hex"
    in_hex.write_text(Path(INPUT_HEX).read_text())
    w_hex.write_text(to_hex(qw_flat, COEF_WIDTH))
    bias_hex.write_text(to_hex(bias_flat, DATA_WIDTH))
    out_hex.write_text(to_hex(q_out, DATA_WIDTH))
    print()
    print(f"wrote {in_hex}    ({x_q.size} values, copy of pool1 output)")
    print(f"wrote {w_hex}  ({qw_flat.size} values, folded; layout k slow, f_in, f_out fast)")
    print(f"wrote {bias_hex}     ({bias_flat.size} values, folded bias)")
    print(f"wrote {out_hex}   ({q_out.size} values, pre-ELU)")

    meta = {
        "layer":             "branchA_sep + bn_3 (Layer 7 fused)",
        "bn_folded":         True,
        "bn_eps":            eps,
        "weights_file":      args.weights,
        "data_width":        DATA_WIDTH,
        "coef_width":        COEF_WIDTH,
        "frac_bits":         FRAC_BITS,
        "T_POOL":            T_POOL,
        "F2":                F2,
        "KE":                KE,
        "pad_top":           pad_top,
        "pad_bot":           pad_bot,
        "weight_order":      "(k slow, f_in, f_out fast) -> address = k*F2*F2 + f_in*F2 + f_out",
        "bias_order":        "(f_out)",
        "output_order":      "(t slow, f_out fast)",
        "input_shape":       [T_POOL, F2],
        "weight_shape":      [KE, F2, F2],
        "bias_shape":        [F2],
        "output_shape":      [T_POOL, F2],
        "abs_max_q_out":     int(np.abs(q_out).max()),
        "saturation_count":  n_sat,
    }
    (OUT_DIR / "stage_branchA_sep_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / 'stage_branchA_sep_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
