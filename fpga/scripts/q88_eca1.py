#!/usr/bin/env python3
"""
q88_eca1.py — Bit-accurate Q8.8 Python reference for Layer 3 (ECA₁)
of the HaLT-variant DB-ATCNet.

ECA spec (docs/FPGA_DEPLOYMENT_DOC.md §5.1) — applied to the F1=16 channel
feature map after Layer 1+2 (Conv2D + BN folded):

    input  shape: (T, C, F1) = (600, 5, 16)   # post conv2d+BN
      → GlobalAvgPool2D over (T, C)            → (F1,)        # gap vector
      → Reshape to (F1, 1)
      → Conv1D(filters=1, kernel=3, same, no bias)            # 3-tap MAC over channels
      → Sigmoid                                → (F1, 1)
      → Reshape to (1, 1, F1)                  → gate vector
      → Multiply(input, broadcast)             → (T, C, F1)   # gated output

This script verifies the **gate-compute path** only (GAP-in, gate-out).
The Q8.8 GAP comes from a reciprocal-multiply (no integer divide in HW):
    GAP_q88[f] = (sum_q88[f] * INV_N_Q24) >> 24
with INV_N_Q24 = round(2**24 / (T*C)) = 5592 for T*C = 3000.

The 3-tap Conv1D uses 'same' padding (1 zero on each side).
Sigmoid uses the universal Q8.8 LUT from scripts/gen_sigmoid_lut.py.

Artifacts written to data/golden_q88/:
    stage_eca1_input.hex         16 Q8.8 GAP values
    stage_eca1_weights.hex        3 Q8.8 Conv1D taps (no bias)
    stage_eca1_gate.hex          16 Q8.8 sigmoid-gate values
    stage_eca1_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
LUT_FILE           = "data/lut/sigmoid_q88.hex"
LAYER1_OUT_HEX     = "data/golden_q88/stage_conv2d_output.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8
ACC_WIDTH  = 48

F1 = 16
T  = 600
C  = 5
KECA = 3                   # ECA Conv1D kernel size (k=3 for in_channel=16 per the doc)
N_GAP = T * C              # 3000
INV_N_Q24 = round((1 << 24) / N_GAP)   # 5592 — see HW comment in docstring

# LUT params (must match scripts/gen_sigmoid_lut.py defaults)
LUT_RANGE_Q = 1024         # +/- 4.0 in Q8.8
LUT_N       = 1024
LUT_SHIFT   = 1            # codes_per_entry = 2


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
                v -= (1 << DATA_WIDTH)        # sign-extend
            flat.append(v)
    a = np.array(flat, dtype=np.int16)
    assert a.size == np.prod(shape), f"hex size {a.size} != expected {np.prod(shape)} for {path}"
    return a.reshape(shape)


def sigmoid_q88_lut(x_q88: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Match the RTL LUT lookup exactly: clip + shift + index."""
    lo = -LUT_RANGE_Q
    hi =  LUT_RANGE_Q - 1
    clipped = np.clip(x_q88.astype(np.int64), lo, hi)
    addr = (clipped + LUT_RANGE_Q) >> LUT_SHIFT
    return lut[addr].astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load ECA Conv1D weights ----------
    print(f"Loading ECA₁ Conv1D weights from {args.weights}...")
    with h5py.File(args.weights, "r") as f:
        w_eca = f["layers/conv1d/vars/0"][()].astype(np.float32)   # (K, in_ch=1, out_ch=1) = (3,1,1)
    w_eca_flat = w_eca[:, 0, 0]                                     # (K,)
    print(f"  weights shape={w_eca.shape}, abs_max={np.abs(w_eca).max():.4f}")
    qw = quantize_q88(w_eca_flat)                                   # (K,) int16

    # ---------- Load Layer 1+2 Q8.8 output ----------
    print(f"Loading Layer 1+2 Q8.8 output from {LAYER1_OUT_HEX}...")
    layer12_q = load_q88_hex(LAYER1_OUT_HEX, (T, C, F1))            # (T, C, F1) int16
    print(f"  layer1+2 output abs_max={int(np.abs(layer12_q).max())}, mean={float(layer12_q.mean()):.2f}")

    # ---------- GAP via reciprocal multiply (matches HW exactly) ----------
    # sum over (T, C) -> int64
    gap_sum   = layer12_q.astype(np.int64).sum(axis=(0, 1))         # (F1,)
    # GAP_q88 = (sum * INV_N_Q24) >> 24   (round-toward-zero for negatives in C)
    # Python's >> on negative int rounds toward -inf, which matches Verilog >>> on
    # signed. We use this convention consistently in both Python and RTL.
    gap_q88   = (gap_sum * INV_N_Q24) >> 24
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    gap_q88   = np.clip(gap_q88, LO, HI).astype(np.int16)
    print(f"  GAP q88 abs_max={int(np.abs(gap_q88).max())}")

    # ---------- Conv1D (3-tap, same padding, no bias) over the channel dim ----------
    pad = (KECA - 1) // 2                              # = 1
    gap_padded = np.zeros(F1 + 2*pad, dtype=np.int64)
    gap_padded[pad : pad + F1] = gap_q88.astype(np.int64)

    acc = np.zeros(F1, dtype=np.int64)
    for k in range(KECA):
        acc += gap_padded[k : k + F1] * int(qw[k])
    shifted = acc >> FRAC_BITS                          # Q8.8 result
    # Saturate
    sig_in = np.clip(shifted, LO, HI).astype(np.int16)
    print(f"  Conv1D output abs_max={int(np.abs(sig_in).max())}  (sigmoid input)")

    # ---------- Sigmoid LUT ----------
    print(f"Loading sigmoid LUT from {LUT_FILE}...")
    lut = load_q88_hex(LUT_FILE, (LUT_N,))             # (1024,) int16
    gate = sigmoid_q88_lut(sig_in, lut)                 # (16,) int16
    print(f"  gate values (real): {[round(g/256, 4) for g in gate]}")

    # ---------- Apply gate to the feature map: y[t,c,f] = sat(x[t,c,f] * gate[f] >> 8) ----------
    # This is the per-channel multiplier that downstream Branch A/B consumes.
    prod = layer12_q.astype(np.int64) * gate[np.newaxis, np.newaxis, :].astype(np.int64)
    prod_shifted = prod >> FRAC_BITS
    gated_q = np.clip(prod_shifted, LO, HI).astype(np.int16)   # (T, C, F1)
    n_sat_apply = int(((prod_shifted > HI) | (prod_shifted < LO)).sum())
    print(f"  gated feature map abs_max={int(np.abs(gated_q).max())}  sat={n_sat_apply}/{gated_q.size}")

    # ---------- Write HEX files ----------
    in_hex      = OUT_DIR / "stage_eca1_input.hex"     # GAP vector (16)
    w_hex       = OUT_DIR / "stage_eca1_weights.hex"   # Conv1D taps (3)
    gate_hex    = OUT_DIR / "stage_eca1_gate.hex"      # sigmoid gates (16)
    out_hex     = OUT_DIR / "stage_eca1_output.hex"    # gated feature map (T*C*F1)
    in_hex.write_text(to_hex(gap_q88, DATA_WIDTH))
    w_hex.write_text(to_hex(qw, COEF_WIDTH))
    gate_hex.write_text(to_hex(gate, DATA_WIDTH))
    out_hex.write_text(to_hex(gated_q, DATA_WIDTH))
    print()
    print(f"wrote {in_hex}    ({gap_q88.size} values, GAP vector)")
    print(f"wrote {w_hex}  ({qw.size} values, Conv1D taps)")
    print(f"wrote {gate_hex}     ({gate.size} values, sigmoid gates)")
    print(f"wrote {out_hex}   ({gated_q.size} values, gated feature map)")

    # ---------- Meta ----------
    meta = {
        "layer":          "eca1 (Layer 3) — Conv1D(3-tap) + Sigmoid LUT",
        "F1":             F1,
        "KECA":           KECA,
        "N_GAP":          N_GAP,
        "INV_N_Q24":      INV_N_Q24,
        "lut_file":       LUT_FILE,
        "lut_n":          LUT_N,
        "lut_range_q88":  LUT_RANGE_Q,
        "lut_shift":      LUT_SHIFT,
        "weights_file":   args.weights,
        "gap_abs_max":    int(np.abs(gap_q88).max()),
        "conv1d_out_abs_max": int(np.abs(sig_in).max()),
        "gate_min":       int(gate.min()),
        "gate_max":       int(gate.max()),
    }
    (OUT_DIR / "stage_eca1_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / 'stage_eca1_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
