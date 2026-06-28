#!/usr/bin/env python3
"""
q88_layer16.py — Bit-accurate Q8.8 reference for Layer 16 (ECA₂) of
HaLT DB-ATCNet, applied to the post-Add feature map (T=10, F=32).

Pipeline (mirrors q88_eca1):
    in       : Add(BranchA_pool2, BranchB_pool2)  shape=(10, 32)
      → GlobalAvgPool over T=10                  → (32,)        # gap vector
      → Conv1D(3-tap, same, no bias)              → (32,)        # sigmoid input
      → Sigmoid LUT (universal)                   → (32,)        # gate
      → x * broadcast(gate) / 256 + sat           → (10, 32)     # gated

INV_N for GAP: N = T = 10  →  INV_N_Q24 = round(2^24 / 10) = 1677722

Weights from layers/conv1d_1/vars/0 (the second ECA Conv1D in the model).

Artifacts written to data/golden_q88/:
    stage_eca2_input.hex           = stage_add_output.hex (gated input, 10*32)
    stage_eca2_gap.hex             32 GAP values
    stage_eca2_weights.hex         3 Conv1D taps
    stage_eca2_gate.hex            32 sigmoid-gate values
    stage_eca2_output.hex          (10, 32) gated feature map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
LUT_FILE           = "data/lut/sigmoid_q88.hex"
ADD_OUT_HEX        = "data/golden_q88/stage_add_output.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8

NUM_CH = 32        # post-Add feature channels
T_IN   = 10        # post-pool2 time steps
KECA   = 3
N_GAP  = T_IN
INV_N_Q24 = round((1 << 24) / N_GAP)   # 1677722 for N=10

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
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ECA₂ Conv1D weights from {args.weights}...")
    with h5py.File(args.weights, "r") as f:
        w_eca = f["layers/conv1d_1/vars/0"][()].astype(np.float32)   # (3, 1, 1)
    assert w_eca.shape == (KECA, 1, 1), f"unexpected ECA₂ weight shape {w_eca.shape}"
    w_eca_flat = w_eca[:, 0, 0]
    print(f"  weights shape={w_eca.shape}, abs_max={np.abs(w_eca).max():.4f}")
    qw = quantize_q88(w_eca_flat)

    print(f"Loading Add output from {ADD_OUT_HEX}...")
    x_q = load_q88_hex(ADD_OUT_HEX, (T_IN, NUM_CH))
    print(f"  Add output abs_max={int(np.abs(x_q).max())}")

    # ---------- GAP via reciprocal multiply ----------
    gap_sum = x_q.astype(np.int64).sum(axis=0)          # (NUM_CH,)
    gap_q88 = (gap_sum * INV_N_Q24) >> 24
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    gap_q88 = np.clip(gap_q88, LO, HI).astype(np.int16)
    print(f"  GAP abs_max={int(np.abs(gap_q88).max())}")

    # ---------- Conv1D (3-tap, same padding) over channel dim ----------
    pad = (KECA - 1) // 2
    gap_padded = np.zeros(NUM_CH + 2*pad, dtype=np.int64)
    gap_padded[pad : pad + NUM_CH] = gap_q88.astype(np.int64)

    acc = np.zeros(NUM_CH, dtype=np.int64)
    for k in range(KECA):
        acc += gap_padded[k : k + NUM_CH] * int(qw[k])
    shifted = acc >> FRAC_BITS
    sig_in = np.clip(shifted, LO, HI).astype(np.int16)
    print(f"  Conv1D output abs_max={int(np.abs(sig_in).max())}")

    print(f"Loading sigmoid LUT from {LUT_FILE}...")
    lut = load_q88_hex(LUT_FILE, (LUT_N,))
    gate = sigmoid_q88_lut(sig_in, lut)
    print(f"  gate min={int(gate.min())} max={int(gate.max())} "
          f"(real ~ [{gate.min()/256:.3f}, {gate.max()/256:.3f}])")

    # ---------- Apply gate per channel ----------
    prod         = x_q.astype(np.int64) * gate[np.newaxis, :].astype(np.int64)
    prod_shifted = prod >> FRAC_BITS
    n_sat        = int(((prod_shifted > HI) | (prod_shifted < LO)).sum())
    gated_q      = np.clip(prod_shifted, LO, HI).astype(np.int16)
    print(f"  gated feature map abs_max={int(np.abs(gated_q).max())} sat={n_sat}/{gated_q.size}")

    # ---------- Write HEX files ----------
    (OUT_DIR / "stage_eca2_input.hex"  ).write_text((OUT_DIR / "stage_add_output.hex").read_text())
    (OUT_DIR / "stage_eca2_gap.hex"    ).write_text(to_hex(gap_q88, DATA_WIDTH))
    (OUT_DIR / "stage_eca2_weights.hex").write_text(to_hex(qw, COEF_WIDTH))
    (OUT_DIR / "stage_eca2_gate.hex"   ).write_text(to_hex(gate, DATA_WIDTH))
    (OUT_DIR / "stage_eca2_output.hex" ).write_text(to_hex(gated_q, DATA_WIDTH))
    print(f"wrote stage_eca2_{{input,gap,weights,gate,output}}.hex")

    meta = {
        "layer":          "eca2 (Layer 16) — GAP(T=10) + Conv1D(3-tap) + Sigmoid LUT + per-ch gate",
        "NUM_CH":         NUM_CH,
        "T_IN":           T_IN,
        "KECA":           KECA,
        "N_GAP":          N_GAP,
        "INV_N_Q24":      INV_N_Q24,
        "lut_file":       LUT_FILE,
        "gap_abs_max":    int(np.abs(gap_q88).max()),
        "conv1d_abs_max": int(np.abs(sig_in).max()),
        "gate_min":       int(gate.min()),
        "gate_max":       int(gate.max()),
        "gated_abs_max":  int(np.abs(gated_q).max()),
    }
    (OUT_DIR / "stage_eca2_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
