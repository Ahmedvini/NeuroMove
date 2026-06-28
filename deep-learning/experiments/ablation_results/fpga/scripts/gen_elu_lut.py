#!/usr/bin/env python3
"""
gen_elu_lut.py

Generate a Q8.8 ELU lookup table for the negative-input half. ELU is:
    ELU(x) = x                       if x >= 0
           = alpha * (exp(x) - 1)    if x < 0   (alpha = 1.0)

For x < 0, ELU output is in [-alpha, 0] = [-1, 0]. We tabulate the negative
half of the function over x ∈ (-LUT_RANGE, 0], clipping further-negative
inputs to the LUT's most-negative bin (where exp(x) ≈ 0 → ELU ≈ -alpha).

HW address mapping (matches scripts/q88_layer5.py):
    if x >= 0  → bypass LUT, output = x  (no saturation needed; x already in Q8.8)
    else       → addr = min(((-x) >> LUT_SHIFT), N-1)
                  output = LUT[addr]
where (-x) takes the absolute value of the (signed) Q8.8 input.

For ELU_RANGE = 8.0, N = 256:
    LUT_RANGE_Q = 8 * 256 = 2048    (-x is clipped to 0..2047 → addr = (-x>>3))
    LUT_SHIFT   = log2(LUT_RANGE_Q / N) = log2(2048/256) = 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT_DIR  = Path("data/lut")
LUT_NAME = "elu_q88.hex"

DATA_WIDTH = 16
FRAC_BITS  = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=float, default=8.0,
                    help="LUT covers x in real units [-RANGE, 0).  Default 8.0.")
    ap.add_argument("--n",     type=int,   default=256,
                    help="Number of LUT entries.  Default 256.")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="ELU alpha (default 1.0 — Keras default).")
    args = ap.parse_args()

    LUT_RANGE = args.range
    N         = args.n
    alpha     = args.alpha
    scale     = 1 << FRAC_BITS
    LUT_RANGE_Q = int(round(LUT_RANGE * scale))   # 2048 for range=8

    codes_per_entry = LUT_RANGE_Q // N
    assert codes_per_entry > 0 and (codes_per_entry & (codes_per_entry - 1)) == 0, \
        "(range*256) must be a power-of-two multiple of N"
    SHIFT = int(np.log2(codes_per_entry))

    print(f"LUT_RANGE   = {LUT_RANGE} (real) / {LUT_RANGE_Q} (Q8.8)")
    print(f"N entries   = {N}")
    print(f"codes/entry = {codes_per_entry}   (SHIFT = {SHIFT})")
    print(f"alpha       = {alpha}")

    # The k-th entry represents inputs whose addr = (-x_q88) >> SHIFT == k.
    # In Q8.8 input space addr k covers codes -[(k+1)*SHIFT-1 .. k*SHIFT].
    # Pick the center of each bin: avg of (- (k+1)*step) and (- k*step)
    #   = -(k*step + step/2)
    bin_center_q = -(np.arange(N) * codes_per_entry + codes_per_entry // 2).astype(np.int64)
    bin_center_real = bin_center_q.astype(np.float64) / scale          # negative

    elu_real = alpha * (np.exp(bin_center_real) - 1.0)
    elu_q = np.round(elu_real * scale).astype(np.int64)
    hi = (1 << (DATA_WIDTH - 1)) - 1
    lo = -(1 << (DATA_WIDTH - 1))
    elu_q = np.clip(elu_q, lo, hi).astype(np.int16)

    print(f"ELU output range (Q8.8): {elu_q.min()} .. {elu_q.max()}  "
          f"({elu_q.min()/scale:.4f} .. {elu_q.max()/scale:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / LUT_NAME
    mask = (1 << DATA_WIDTH) - 1
    digits = (DATA_WIDTH + 3) // 4
    with out_path.open("w") as f:
        for v in elu_q:
            f.write(f"{int(v) & mask:0{digits}X}\n")
    print(f"wrote {out_path}  ({N} entries)")

    meta_path = OUT_DIR / "elu_q88_meta.txt"
    meta_path.write_text(
        f"# ELU LUT params (negative-half only)\n"
        f"LUT_RANGE_REAL  = {LUT_RANGE}\n"
        f"LUT_RANGE_Q88   = {LUT_RANGE_Q}\n"
        f"N               = {N}\n"
        f"SHIFT           = {SHIFT}\n"
        f"ALPHA           = {alpha}\n"
        f"FRAC_BITS       = {FRAC_BITS}\n"
        f"DATA_WIDTH      = {DATA_WIDTH}\n"
        f"# address calc:\n"
        f"#   if x_q88 >= 0:  out = x_q88\n"
        f"#   else:            addr = min(((-x_q88) >> SHIFT), N-1)\n"
        f"#                    out  = LUT[addr]\n"
    )
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
