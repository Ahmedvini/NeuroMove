#!/usr/bin/env python3
"""
q88_layer8_9.py — Q8.8 references for Layers 8 and 9 of Branch A:
    Layer 8: ELU (no new weights; uses universal LUT)
    Layer 9: AvgPool2D(7, 1) — temporal downsample 75 → 10

Reuses scripts/gen_elu_lut.py output and the same reciprocal-multiply
pattern as q88_layer6.py.

Inputs/outputs:
    Layer 8 in  : stage_branchA_sep_output.hex  (75 * 32 = 2400)
    Layer 8 out : stage_branchA_sep_elu_output.hex (75 * 32)
    Layer 9 in  : = Layer 8 output
    Layer 9 out : stage_branchA_pool2_output.hex (10 * 32 = 320)

Layer 9 reciprocal-multiply: POOL=7, Q_SHIFT=24, INV_POOL_Q = round(2^24/7) = 2396745
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEP_OUT_HEX     = "data/golden_q88/stage_branchA_sep_output.hex"
ELU_LUT_FILE    = "data/lut/elu_q88.hex"
OUT_DIR         = Path("data/golden_q88")

DATA_WIDTH = 16
FRAC_BITS  = 8

T_POOL   = 75
F2       = 32
POOL2    = 7
T_POOL2  = T_POOL // POOL2          # = 10
N_USED   = T_POOL2 * POOL2          # = 70 (last 5 inputs are dropped)
Q_SHIFT     = 24
INV_POOL_Q  = round((1 << Q_SHIFT) / POOL2)   # 2396745

# ELU LUT params (must match scripts/gen_elu_lut.py defaults)
LUT_RANGE_Q = 2048
LUT_N       = 256
LUT_SHIFT   = 3


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


def elu_q88_lut(x_q88: np.ndarray, lut: np.ndarray) -> np.ndarray:
    out = x_q88.copy().astype(np.int32)
    neg_mask = x_q88 < 0
    neg_vals = x_q88[neg_mask]
    addr = ((-neg_vals.astype(np.int64)) >> LUT_SHIFT).astype(np.int64)
    addr = np.clip(addr, 0, LUT_N - 1)
    out[neg_mask] = lut[addr]
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    return np.clip(out, LO, HI).astype(np.int16)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===== Layer 8: ELU on sep_a output =====
    print("Layer 8 — ELU on Branch A separable output...")
    sep = load_q88_hex(SEP_OUT_HEX, (T_POOL, F2))
    print(f"  input shape={sep.shape}, abs_max={int(np.abs(sep).max())}, "
          f"neg count={int((sep < 0).sum())} / {sep.size}")
    lut = load_q88_hex(ELU_LUT_FILE, (LUT_N,))
    sep_elu = elu_q88_lut(sep, lut)
    print(f"  output abs_max={int(np.abs(sep_elu).max())}")

    l8_in_hex  = OUT_DIR / "stage_branchA_sep_elu_input.hex"
    l8_out_hex = OUT_DIR / "stage_branchA_sep_elu_output.hex"
    l8_in_hex.write_text(Path(SEP_OUT_HEX).read_text())
    l8_out_hex.write_text(to_hex(sep_elu, DATA_WIDTH))
    print(f"  wrote {l8_in_hex}    ({sep.size} values)")
    print(f"  wrote {l8_out_hex}   ({sep_elu.size} values)")
    print()

    # ===== Layer 9: AvgPool(7, 1) =====
    print(f"Layer 9 — AvgPool(POOL={POOL2}, 1) → (T_POOL2={T_POOL2}, F2={F2})...")
    # Drop the last 5 inputs (valid padding with stride 7, only 70 of 75 contribute).
    grouped = sep_elu[:N_USED].astype(np.int64).reshape(T_POOL2, POOL2, F2)
    sums    = grouped.sum(axis=1)
    prod    = sums * INV_POOL_Q
    y       = prod >> Q_SHIFT
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    y_q     = np.clip(y, LO, HI).astype(np.int16)
    n_sat   = int(((y > HI) | (y < LO)).sum())
    print(f"  output shape={y_q.shape}, abs_max={int(np.abs(y_q).max())}, sat={n_sat}/{y_q.size}")

    l9_in_hex  = OUT_DIR / "stage_branchA_pool2_input.hex"
    l9_out_hex = OUT_DIR / "stage_branchA_pool2_output.hex"
    # Drive only N_USED inputs into the pool; write that subset (70*32 = 2240)
    l9_in_hex.write_text(to_hex(sep_elu[:N_USED], DATA_WIDTH))
    l9_out_hex.write_text(to_hex(y_q, DATA_WIDTH))
    print(f"  wrote {l9_in_hex}  ({sep_elu[:N_USED].size} values, first 70 of 75 ELU outputs)")
    print(f"  wrote {l9_out_hex} ({y_q.size} values)")

    meta = {
        "layer8":  {"op": "ELU", "input_shape": [T_POOL, F2], "output_shape": [T_POOL, F2]},
        "layer9":  {"op": "AvgPool(7,1)", "input_shape": [N_USED, F2], "output_shape": [T_POOL2, F2],
                    "POOL": POOL2, "INV_POOL_Q": INV_POOL_Q, "Q_SHIFT": Q_SHIFT},
    }
    (OUT_DIR / "stage_branchA_pool2_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {OUT_DIR / 'stage_branchA_pool2_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
