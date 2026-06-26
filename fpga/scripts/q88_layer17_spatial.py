#!/usr/bin/env python3
"""
q88_layer17_spatial.py — Bit-accurate Q8.8 reference for ImpCBAM SPATIAL
attention on sliding-window 0. Per docs/FPGA_DEPLOYMENT_DOC.md §5.2.2 +
the trained weights (conv2d_3 shape (7,7,3,1)):

    input: chan_attn_output (6, 32)        # output of channel attention
      ├─ AvgPool across channels (axis=-1) → (6,)
      ├─ MaxPool across channels (axis=-1) → (6,)
      └─ StochasticPool (deterministic)    → (6,)
            probs[t,c] = |x[t,c]| / sum_c |x[t,c]|   (with eps floor)
            out[t]     = sum_c probs[t,c] * x[t,c]
      Concat → (6, 1, 3)
      Conv2D(kernel=(7,7), in=3, out=1, same, no bias) → (6, 1, 1)
        Note: input W-dim is 1, so kernel columns kw=0..6 are zero-padded
              except kw = (7-1)//2 = 3 (center). Effectively a 7-tap 1D
              conv over T using the center column of the 2D kernel.
      Sigmoid LUT → (6,)
      Multiply input by gate (broadcast over channel) → (6, 32)

For window 0 the conv weights live at layers/conv2d_3/vars/0.

The AvgPool over 32 channels uses reciprocal-multiply:
    INV_C = round(2^24 / 32) = 524288  (clean power of two equivalent of >>5)
For STOCHASTIC POOL the |x| sum may be 0 ; we floor it to 1 in Q8.8 to
mirror Keras' `max(denom, 1e-7)`. This matches the eps behavior of
softmax-like normalisation in deterministic inference.

Artifacts (window 0):
    stage_w0_spat_input.hex    = stage_w0_chan_output.hex (6*32)
    stage_w0_spat_avg.hex      = 6 avg-over-channels descriptors
    stage_w0_spat_max.hex      = 6 max-over-channels descriptors
    stage_w0_spat_stoch.hex    = 6 stochastic-pool descriptors
    stage_w0_spat_concat.hex   = 6*3 concatenated descriptors (T-major)
    stage_w0_spat_conv_w.hex   = 7*7*3 conv weights (kh, kw, ic) row-major
    stage_w0_spat_gate.hex     = 6 sigmoid gate values
    stage_w0_spat_output.hex   = 6*32 gated output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
LUT_FILE           = "data/lut/sigmoid_q88.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8

T_WIN  = 6
NUM_CH = 32
KH     = 7
KW     = 7
IC_CAT = 3
INV_C  = round((1 << 24) / NUM_CH)   # 524288 = exactly 2^19, i.e. >>5 in Q8.8

LUT_RANGE_Q = 1024
LUT_N       = 1024
LUT_SHIFT   = 1


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


def sigmoid_q88_lut(x_q88: np.ndarray, lut: np.ndarray) -> np.ndarray:
    lo = -LUT_RANGE_Q
    hi =  LUT_RANGE_Q - 1
    clipped = np.clip(x_q88.astype(np.int64), lo, hi)
    addr = (clipped + LUT_RANGE_Q) >> LUT_SHIFT
    return lut[addr].astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()
    w = args.window
    assert 0 <= w < 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_hex_path = f"data/golden_q88/stage_w{w}_chan_output.hex"
    print(f"Loading channel-attn output from {in_hex_path}...")
    x_q = load_q88_hex(in_hex_path, (T_WIN, NUM_CH))                  # (6, 32)
    print(f"  input shape={x_q.shape} abs_max={int(np.abs(x_q).max())}")

    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))

    # ---------- Avg pool over channels (32) ----------
    avg_sum = x_q.astype(np.int64).sum(axis=1)                        # (6,)
    avg_pool = (avg_sum * INV_C) >> 24
    avg_pool = np.clip(avg_pool, LO, HI).astype(np.int16)
    print(f"  avg_pool: {avg_pool.tolist()}")

    # ---------- Max pool over channels ----------
    max_pool = x_q.max(axis=1).astype(np.int16)
    print(f"  max_pool: {max_pool.tolist()}")

    # ---------- Stochastic pool over channels (HW-aligned formulation) ----------
    # One division per time step (cheap on FPGA), then per-element multiply.
    #   norm[t]      = sum_c |x[t,c]|                  (Q8.8)
    #   inv_norm[t]  = floor(2^24 / max(norm[t], 1))   (one sequential division per t)
    #   stoch_acc[t] = sum_c (|x[t,c]| * inv_norm[t]) * x[t,c]
    #   stoch_pool[t]= sat( stoch_acc[t] >> 32 )
    #
    # Bit width: |x| (Q8.8, 16b) * inv_norm (~25b) → 41-bit. Multiplied by x_q
    # (16b) → 57-bit. Summed over 32 → 62-bit. Shifted >> 32 → 30-bit, then
    # saturated to Q8.8.
    abs_x = np.abs(x_q.astype(np.int64))
    norm_per_t  = np.maximum(abs_x.sum(axis=1), 1)                     # (6,)
    inv_norm    = (1 << 24) // norm_per_t                              # (6,) one floor-div per t
    probs_q24   = abs_x * inv_norm[:, np.newaxis]                       # (6, 32) Q24-ish
    stoch_acc   = (probs_q24 * x_q.astype(np.int64)).sum(axis=1)        # (6,)
    stoch_pool  = stoch_acc >> 24
    stoch_pool  = np.clip(stoch_pool, LO, HI).astype(np.int16)
    print(f"  stoch_pool: {stoch_pool.tolist()}")
    print(f"  inv_norm:   {inv_norm.tolist()}")

    # ---------- Concat: (6, 1, 3) — order = [avg, max, stoch] ----------
    concat = np.stack([avg_pool, max_pool, stoch_pool], axis=-1)       # (6, 3)
    print(f"  concat shape={concat.shape} abs_max={int(np.abs(concat).max())}")

    # ---------- Load conv weights conv2d_{3+w} ----------
    conv_idx = 3 + w
    with h5py.File(args.weights, "r") as f:
        wt = f[f"layers/conv2d_{conv_idx}/vars/0"][()].astype(np.float32)   # (7, 7, 3, 1)
    assert wt.shape == (KH, KW, IC_CAT, 1), f"unexpected conv weight shape {wt.shape}"
    print(f"  conv2d_{conv_idx} weights abs_max={np.abs(wt).max():.4f}")
    qw = quantize_q88(wt[..., 0])                                       # (7, 7, 3) signed Q8.8

    # ---------- Conv2D (7,7,3,1) same-padding over (6, 1, 3) ----------
    # Padding for H: pad_h = (KH-1)//2 = 3 top, KH-1-pad_h = 3 bottom.
    # Padding for W: pad_w = (KW-1)//2 = 3 left, KW-1-pad_w = 3 right.
    pad_h = (KH - 1) // 2
    pad_w = (KW - 1) // 2

    # We only need the kw=pad_w column of the kernel because input W=1.
    # qw[:, pad_w, :] has shape (7, 3): per-tap weight across 3 input channels.
    qw_slice = qw[:, pad_w, :]                                          # (7, 3) signed
    HI64 = HI; LO64 = LO

    # Pad concat to length (T_WIN + KH - 1) in T-axis with zeros.
    cat_pad = np.zeros((T_WIN + KH - 1, IC_CAT), dtype=np.int64)
    cat_pad[pad_h : pad_h + T_WIN, :] = concat.astype(np.int64)

    # acc[t] = sum_{kh,ic} cat_pad[t+kh, ic] * qw_slice[kh, ic]
    acc = np.zeros(T_WIN, dtype=np.int64)
    for kh in range(KH):
        acc += (cat_pad[kh : kh + T_WIN, :] * qw_slice[kh][np.newaxis, :].astype(np.int64)).sum(axis=1)
    shifted   = acc >> FRAC_BITS
    sig_in    = np.clip(shifted, LO64, HI64).astype(np.int16)            # (6,)
    print(f"  pre-sigmoid abs_max={int(np.abs(sig_in).max())} sat="
          f"{int(((shifted > HI64) | (shifted < LO64)).sum())}/{shifted.size}")

    # ---------- Sigmoid LUT ----------
    lut = load_q88_hex(LUT_FILE, (LUT_N,))
    gate = sigmoid_q88_lut(sig_in, lut)                                  # (6,)
    print(f"  gate values: {gate.tolist()}")

    # ---------- Apply gate per time step (broadcast over channels) ----------
    prod = x_q.astype(np.int64) * gate[:, np.newaxis].astype(np.int64)
    prod_shifted = prod >> FRAC_BITS
    n_sat = int(((prod_shifted > HI) | (prod_shifted < LO)).sum())
    gated = np.clip(prod_shifted, LO, HI).astype(np.int16)               # (6, 32)
    print(f"  gated abs_max={int(np.abs(gated).max())} sat={n_sat}/{gated.size}")

    # ---------- Emit artifacts ----------
    prefix = f"stage_w{w}_spat"
    (OUT_DIR / f"{prefix}_input.hex"  ).write_text(to_hex(x_q,             DATA_WIDTH))
    (OUT_DIR / f"{prefix}_avg.hex"    ).write_text(to_hex(avg_pool,        DATA_WIDTH))
    (OUT_DIR / f"{prefix}_max.hex"    ).write_text(to_hex(max_pool,        DATA_WIDTH))
    (OUT_DIR / f"{prefix}_stoch.hex"  ).write_text(to_hex(stoch_pool,      DATA_WIDTH))
    (OUT_DIR / f"{prefix}_concat.hex" ).write_text(to_hex(concat,          DATA_WIDTH))
    # Conv weights as a flat 7*3 (kh-major, ic-fast) hex file for the RTL.
    (OUT_DIR / f"{prefix}_conv_w.hex" ).write_text(to_hex(qw_slice,        COEF_WIDTH))
    (OUT_DIR / f"{prefix}_gate.hex"   ).write_text(to_hex(gate,            DATA_WIDTH))
    (OUT_DIR / f"{prefix}_output.hex" ).write_text(to_hex(gated,           DATA_WIDTH))
    print(f"wrote {prefix}_{{input,avg,max,stoch,concat,conv_w,gate,output}}.hex")

    (OUT_DIR / f"{prefix}_meta.json").write_text(json.dumps({
        "window":      w,
        "T_WIN":       T_WIN,
        "NUM_CH":      NUM_CH,
        "KH":          KH,
        "KW":          KW,
        "IC_CAT":      IC_CAT,
        "INV_C":       INV_C,
        "conv_layer":  f"conv2d_{conv_idx}",
        "kernel_full_shape": list(wt.shape),
        "stoch_pool_eps_q88": 1,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
