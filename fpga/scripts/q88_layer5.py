#!/usr/bin/env python3
"""
q88_layer5.py — Bit-accurate Q8.8 Python reference for the ELU activation
(Layer 5 of Branch A) of HaLT DB-ATCNet. Applied per-element to the depthwise
output (T=600, F2=32). The ELU LUT lookup is identical to what the RTL does.

Spec:
    ELU(x) = x                          if x >= 0
           = alpha * (exp(x) - 1)       if x < 0   (alpha = 1.0)

Q8.8 LUT addressing matches scripts/gen_elu_lut.py:
    if x_q88 >= 0:  out = x_q88
    else:            addr = min(((-x_q88) >> SHIFT), N-1)
                     out  = LUT[addr]

Artifacts written to data/golden_q88/:
    stage_branchA_elu_input.hex  — copy of stage_branchA_dwise_output.hex
    stage_branchA_elu_output.hex — ELU(input), same shape, Q8.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ELU_INPUT_HEX = "data/golden_q88/stage_branchA_dwise_output.hex"
LUT_FILE      = "data/lut/elu_q88.hex"
OUT_DIR       = Path("data/golden_q88")

DATA_WIDTH = 16
FRAC_BITS  = 8

T  = 600
F2 = 32

# LUT params (must match scripts/gen_elu_lut.py defaults)
LUT_RANGE_Q = 2048      # ±8.0 in Q8.8 — but we only use the negative half
LUT_N       = 256
LUT_SHIFT   = 3         # codes_per_entry = 8


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


def elu_q88_lut(x_q88: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Match the RTL ELU lookup exactly: bypass for x>=0; LUT for x<0."""
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
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading depthwise output (Layer 4) from {ELU_INPUT_HEX}...")
    x_q = load_q88_hex(ELU_INPUT_HEX, (T, F2))
    print(f"  input shape={x_q.shape}, abs_max={int(np.abs(x_q).max())}, "
          f"neg count={int((x_q < 0).sum())} / {x_q.size}")

    print(f"Loading ELU LUT from {LUT_FILE}...")
    lut = load_q88_hex(LUT_FILE, (LUT_N,))

    print("Applying ELU bit-exactly via LUT...")
    y_q = elu_q88_lut(x_q, lut)
    print(f"  output abs_max={int(np.abs(y_q).max())}")

    in_hex  = OUT_DIR / "stage_branchA_elu_input.hex"
    out_hex = OUT_DIR / "stage_branchA_elu_output.hex"
    in_hex.write_text(to_hex(x_q, DATA_WIDTH))
    out_hex.write_text(to_hex(y_q, DATA_WIDTH))
    print()
    print(f"wrote {in_hex}   ({x_q.size} values, same as depthwise output)")
    print(f"wrote {out_hex}  ({y_q.size} values, ELU applied)")

    meta = {
        "layer":        "branchA_elu (Layer 5) — ELU activation via LUT",
        "lut_file":     LUT_FILE,
        "lut_n":        LUT_N,
        "lut_range_q":  LUT_RANGE_Q,
        "lut_shift":    LUT_SHIFT,
        "alpha":        1.0,
        "input_shape":  list(x_q.shape),
        "output_shape": list(y_q.shape),
        "abs_max_in":   int(np.abs(x_q).max()),
        "abs_max_out":  int(np.abs(y_q).max()),
        "negative_inputs": int((x_q < 0).sum()),
    }
    (OUT_DIR / "stage_branchA_elu_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR / 'stage_branchA_elu_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
