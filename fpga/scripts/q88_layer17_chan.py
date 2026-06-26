#!/usr/bin/env python3
"""
q88_layer17_chan.py — Bit-accurate Q8.8 reference for ImpCBAM channel
attention on sliding-window 0. Per docs/FPGA_DEPLOYMENT_DOC.md §5.2.1:

    input: features[:6, :]                   # (6, 32) slice of ECA₂ output
      ├─ GlobalAvgPool over T=6 → (32,)      # avg_gap
      │   → Dense(4, ReLU)                    # shared MLP
      │   → Dense(32, linear)
      ├─ GlobalMaxPool over T=6 → (32,)      # max_gap
      │   → (same shared MLP)
      Add(avg, max) → Sigmoid → (32,) gate
      Multiply(input, gate broadcast) → (6, 32)

Window i uses Dense layers `dense_i` (32→4 with bias 4) and
`dense_{i+5}` (4→32 with bias 32). We do i=0 here and parameterize the
script by --window.

ReLU is in Q8.8 (negative → 0, positive passes through).

INV_N for the GAP: N = T_win = 6 → INV_N_Q24 = round(2^24/6) = 2796203.

Artifacts (window 0):
    stage_w0_chan_input.hex   = 6*32 features (slice 0..5 of ECA₂ output)
    stage_w0_chan_avg.hex     = 32  GAP avg
    stage_w0_chan_max.hex     = 32  GAP max
    stage_w0_chan_d1w.hex     = 128 Dense₁ kernel (32x4, folded as row-major (i_out, i_in))
    stage_w0_chan_d1b.hex     = 4   Dense₁ bias
    stage_w0_chan_d2w.hex     = 128 Dense₂ kernel (4x32, row-major (i_out, i_in))
    stage_w0_chan_d2b.hex     = 32  Dense₂ bias
    stage_w0_chan_avg_mlp.hex = 32  Dense₂(ReLU(Dense₁(avg)))
    stage_w0_chan_max_mlp.hex = 32  Dense₂(ReLU(Dense₁(max)))
    stage_w0_chan_gate.hex    = 32  sigmoid gate
    stage_w0_chan_output.hex  = 6*32 gated input
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
LUT_FILE           = "data/lut/sigmoid_q88.hex"
ECA2_OUT_HEX       = "data/golden_q88/stage_eca2_output.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8

T_FULL  = 10
NUM_CH  = 32
T_WIN   = 6
D_RED   = 4                  # bottleneck width
INV_N_Q24 = round((1 << 24) / T_WIN)   # 2796203 for N=6

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


def dense_q88(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """x_q (Nin,), w_q (Nin, Nout), b_q (Nout,) — fully-connected layer in Q8.8."""
    acc = x_q.astype(np.int64) @ w_q.astype(np.int64)
    shifted = acc >> FRAC_BITS
    with_bias = shifted + b_q.astype(np.int64)
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    return np.clip(with_bias, LO, HI).astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()
    w = args.window
    assert 0 <= w < 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading channel-attn weights for window {w}...")
    # Keras first layer instance has no suffix; subsequent get _1, _2, ...
    d1_name = "dense"   if w == 0 else f"dense_{w}"
    d2_name = "dense_5" if w == 0 else f"dense_{w + 5}"
    with h5py.File(args.weights, "r") as f:
        d1w = f[f"layers/{d1_name}/vars/0"][()].astype(np.float32)    # (32, 4)
        d1b = f[f"layers/{d1_name}/vars/1"][()].astype(np.float32)    # (4,)
        d2w = f[f"layers/{d2_name}/vars/0"][()].astype(np.float32)    # (4, 32)
        d2b = f[f"layers/{d2_name}/vars/1"][()].astype(np.float32)    # (32,)
    assert d1w.shape == (NUM_CH, D_RED) and d1b.shape == (D_RED,)
    assert d2w.shape == (D_RED, NUM_CH) and d2b.shape == (NUM_CH,)
    qd1w = quantize_q88(d1w); qd1b = quantize_q88(d1b)
    qd2w = quantize_q88(d2w); qd2b = quantize_q88(d2b)
    print(f"  Dense₁ {d1w.shape} abs_max={np.abs(d1w).max():.4f}, "
          f"Dense₂ {d2w.shape} abs_max={np.abs(d2w).max():.4f}")

    # ---------- Load + slice ECA₂ output ----------
    print(f"Loading ECA₂ output from {ECA2_OUT_HEX}...")
    feat_full = load_q88_hex(ECA2_OUT_HEX, (T_FULL, NUM_CH))         # (10, 32)
    win = feat_full[w:w+T_WIN, :]                                     # (6, 32)
    print(f"  window {w} slice [{w}:{w+T_WIN}] shape={win.shape} abs_max={int(np.abs(win).max())}")

    # ---------- Avg / Max GAP over T_WIN ----------
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    avg_sum = win.astype(np.int64).sum(axis=0)                        # (32,)
    avg_gap = (avg_sum * INV_N_Q24) >> 24
    avg_gap = np.clip(avg_gap, LO, HI).astype(np.int16)
    max_gap = win.max(axis=0).astype(np.int16)
    print(f"  avg_gap abs_max={int(np.abs(avg_gap).max())}, "
          f"max_gap abs_max={int(np.abs(max_gap).max())}")

    # ---------- Shared MLP (Dense₁ + ReLU + Dense₂) on both paths ----------
    def mlp(x_q):
        h = dense_q88(x_q, qd1w, qd1b)                                # (4,)
        h = np.maximum(h, np.int16(0))                                # ReLU in Q8.8
        return dense_q88(h, qd2w, qd2b)                               # (32,)

    avg_mlp = mlp(avg_gap)
    max_mlp = mlp(max_gap)
    print(f"  avg_mlp abs_max={int(np.abs(avg_mlp).max())}, "
          f"max_mlp abs_max={int(np.abs(max_mlp).max())}")

    # ---------- Add + Sigmoid ----------
    add_sum = avg_mlp.astype(np.int64) + max_mlp.astype(np.int64)
    add_sat = np.clip(add_sum, LO, HI).astype(np.int16)
    print(f"Loading sigmoid LUT from {LUT_FILE}...")
    lut = load_q88_hex(LUT_FILE, (LUT_N,))
    gate = sigmoid_q88_lut(add_sat, lut)
    print(f"  gate min={int(gate.min())} max={int(gate.max())}")

    # ---------- Apply gate per channel to the WHOLE (6, 32) window ----------
    prod = win.astype(np.int64) * gate[np.newaxis, :].astype(np.int64)
    prod_shifted = prod >> FRAC_BITS
    n_sat = int(((prod_shifted > HI) | (prod_shifted < LO)).sum())
    gated = np.clip(prod_shifted, LO, HI).astype(np.int16)
    print(f"  gated abs_max={int(np.abs(gated).max())} sat={n_sat}/{gated.size}")

    # ---------- Write all artifacts ----------
    prefix = f"stage_w{w}_chan"
    (OUT_DIR / f"{prefix}_input.hex"  ).write_text(to_hex(win,           DATA_WIDTH))
    (OUT_DIR / f"{prefix}_avg.hex"    ).write_text(to_hex(avg_gap,       DATA_WIDTH))
    (OUT_DIR / f"{prefix}_max.hex"    ).write_text(to_hex(max_gap,       DATA_WIDTH))
    # Dense kernels: store as (i_out, i_in) row-major so HW can iterate
    # output-major. Keras stores (in, out); transpose for hex emission.
    (OUT_DIR / f"{prefix}_d1w.hex"    ).write_text(to_hex(qd1w.T,        COEF_WIDTH))
    (OUT_DIR / f"{prefix}_d1b.hex"    ).write_text(to_hex(qd1b,          DATA_WIDTH))
    (OUT_DIR / f"{prefix}_d2w.hex"    ).write_text(to_hex(qd2w.T,        COEF_WIDTH))
    (OUT_DIR / f"{prefix}_d2b.hex"    ).write_text(to_hex(qd2b,          DATA_WIDTH))
    (OUT_DIR / f"{prefix}_avg_mlp.hex").write_text(to_hex(avg_mlp,       DATA_WIDTH))
    (OUT_DIR / f"{prefix}_max_mlp.hex").write_text(to_hex(max_mlp,       DATA_WIDTH))
    (OUT_DIR / f"{prefix}_gate.hex"   ).write_text(to_hex(gate,          DATA_WIDTH))
    (OUT_DIR / f"{prefix}_output.hex" ).write_text(to_hex(gated,         DATA_WIDTH))
    print(f"wrote {prefix}_{{input,avg,max,d1w,d1b,d2w,d2b,avg_mlp,max_mlp,gate,output}}.hex")

    (OUT_DIR / f"{prefix}_meta.json").write_text(json.dumps({
        "window":         w,
        "T_FULL":         T_FULL,
        "T_WIN":          T_WIN,
        "NUM_CH":         NUM_CH,
        "D_RED":          D_RED,
        "INV_N_Q24":      INV_N_Q24,
        "avg_gap_abs_max": int(np.abs(avg_gap).max()),
        "max_gap_abs_max": int(np.abs(max_gap).max()),
        "gate_min":       int(gate.min()),
        "gate_max":       int(gate.max()),
        "gated_abs_max":  int(np.abs(gated).max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
