#!/usr/bin/env python3
"""
q88_layer6.py — Bit-accurate Q8.8 Python reference for AvgPool2D(8, 1)
(Layer 6 of Branch A) — temporal downsampling by factor 8.

Spec (docs/FPGA_DEPLOYMENT_DOC.md §6.3):
    AvgPool2D(pool_size=(8, 1), strides=(8, 1), padding='valid')

Input shape:  (T=600, F2=32)   # ELU output
Output shape: (T_pool=75, F2=32)
    y[t_out, f] = mean over k in 0..POOL-1 of x[t_out * POOL + k, f]

Q8.8 implementation (matches HW exactly):
    sum_q88 = sum over k of x[t_out * POOL + k, f]   (int64)
    y_q88   = (sum_q88 * INV_POOL_Q) >> Q_SHIFT
where INV_POOL_Q = round(2^Q_SHIFT / POOL).
For POOL=8 (power of 2), the cleanest version is just (sum >> 3), with no
reciprocal multiply. We keep the parameterized form for code uniformity.

Artifacts written to data/golden_q88/:
    stage_branchA_pool1_input.hex   = stage_branchA_elu_output.hex   (T*F2 = 19200)
    stage_branchA_pool1_output.hex  pooled output                    (T_pool*F2 = 2400)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ELU_OUT_HEX = "data/golden_q88/stage_branchA_elu_output.hex"
OUT_DIR     = Path("data/golden_q88")

DATA_WIDTH = 16
FRAC_BITS  = 8

T          = 600
F2         = 32
POOL       = 8
T_POOL     = T // POOL          # 75

# Reciprocal: for POOL=8 (power of 2), arithmetic shift right by 3 suffices,
# but keep the generic form so other pool sizes plug in cleanly.
Q_SHIFT     = 24
INV_POOL_Q  = round((1 << Q_SHIFT) / POOL)   # 2097152 for POOL=8 (= 2^24 / 8)


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


def to_hex(q: np.ndarray, width: int = 16) -> str:
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    flat = q.flatten()
    return "\n".join(f"{int(v) & mask:0{digits}X}" for v in flat) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ELU output (Layer 5) from {ELU_OUT_HEX}...")
    x_q = load_q88_hex(ELU_OUT_HEX, (T, F2))
    print(f"  input shape={x_q.shape}, abs_max={int(np.abs(x_q).max())}")

    print(f"AvgPool(POOL={POOL}, 1) → (T_POOL={T_POOL}, F2={F2})...")
    # Reshape (T, F2) -> (T_POOL, POOL, F2) then sum across POOL axis.
    grouped = x_q.astype(np.int64).reshape(T_POOL, POOL, F2)
    sums    = grouped.sum(axis=1)                          # (T_POOL, F2) int64

    # Reciprocal-multiply: y = (sum * INV_POOL_Q) >> Q_SHIFT
    prod    = sums * INV_POOL_Q
    y       = prod >> Q_SHIFT
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    y_q     = np.clip(y, LO, HI).astype(np.int16)
    n_sat   = int(((y > HI) | (y < LO)).sum())
    print(f"  output shape={y_q.shape}, abs_max={int(np.abs(y_q).max())}, sat={n_sat}/{y_q.size}")

    in_hex  = OUT_DIR / "stage_branchA_pool1_input.hex"
    out_hex = OUT_DIR / "stage_branchA_pool1_output.hex"
    in_hex.write_text(Path(ELU_OUT_HEX).read_text())
    out_hex.write_text(to_hex(y_q, DATA_WIDTH))
    print()
    print(f"wrote {in_hex}   ({x_q.size} values, copy of ELU output)")
    print(f"wrote {out_hex}  ({y_q.size} values, AvgPool(8,1) output)")

    meta = {
        "layer":          "branchA_pool1 (Layer 6) — AvgPool2D(8,1)",
        "T":              T,
        "T_POOL":         T_POOL,
        "F2":             F2,
        "POOL":           POOL,
        "INV_POOL_Q":     INV_POOL_Q,
        "Q_SHIFT":        Q_SHIFT,
        "input_shape":    [T, F2],
        "output_shape":   [T_POOL, F2],
        "abs_max_in":     int(np.abs(x_q).max()),
        "abs_max_out":    int(np.abs(y_q).max()),
        "saturation":     n_sat,
    }
    (OUT_DIR / "stage_branchA_pool1_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / 'stage_branchA_pool1_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
