#!/usr/bin/env python3
"""
gen_sigmoid_lut.py

Generate a Q8.8 fixed-point sigmoid lookup table for use by ECA / CBAM / softmax-
gate RTL blocks. Universal (does not depend on training); baked into the
bitstream via $readmemh at synthesis time.

LUT shape:
    Input range:  [-LUT_RANGE, +LUT_RANGE]  in Q8.8 (default LUT_RANGE = 4.0)
    Resolution:   N entries equally spaced over that range
                  (default N = 1024 -> step = 8.0 / 1024 = 1/128 in real units
                   = 2 LSBs in Q8.8 input space)
    Outside the range, the input is clipped before lookup.

Output format:
    Each LUT entry is signed Q8.8 (16-bit), where 1.0 = 0x0100.
    Sigmoid range is (0, 1) so entries are always in [0, 256] => fit cleanly.

Address mapping in HW (matches the Python reference exactly):
    clipped_in = clip(in_q88, -LUT_RANGE_Q88, +LUT_RANGE_Q88 - 1)
    addr       = (clipped_in - (-LUT_RANGE_Q88)) >> SHIFT
    out_q88    = LUT[addr]
where SHIFT is chosen so the index covers [0, N).

For LUT_RANGE=4.0, N=1024:
    LUT_RANGE_Q88 = 4 * 256 = 1024  (input fits in -1024..1023 range)
    full range = 2048 input codes / 1024 entries -> SHIFT = 1

The values are stored in `data/lut/sigmoid_q88.hex`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT_DIR    = Path("data/lut")
LUT_NAME   = "sigmoid_q88.hex"

DATA_WIDTH = 16
FRAC_BITS  = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=float, default=4.0,
                    help="LUT covers input in real units [-RANGE, +RANGE].  Default 4.0.")
    ap.add_argument("--n",     type=int,   default=1024,
                    help="Number of LUT entries.  Default 1024.")
    args = ap.parse_args()

    LUT_RANGE = args.range
    N         = args.n
    scale     = 1 << FRAC_BITS
    LUT_RANGE_Q = int(round(LUT_RANGE * scale))    # 1024 for range=4, frac=8

    # The HW address mapping:
    #   addr = (clipped_in_q88 + LUT_RANGE_Q) >> SHIFT
    #   where total input codes = 2 * LUT_RANGE_Q
    # We want addr in [0, N), so SHIFT = log2(2 * LUT_RANGE_Q / N)
    n_codes_per_entry = (2 * LUT_RANGE_Q) // N
    assert n_codes_per_entry > 0 and (n_codes_per_entry & (n_codes_per_entry - 1)) == 0, \
        "(2*range*256) must be a power-of-two multiple of N"
    SHIFT = int(np.log2(n_codes_per_entry))

    print(f"LUT_RANGE         = {LUT_RANGE} (real) / {LUT_RANGE_Q} (Q8.8)")
    print(f"N entries         = {N}")
    print(f"codes per entry   = {n_codes_per_entry}   (SHIFT = {SHIFT})")

    # The k-th entry represents inputs whose addr maps to k.
    # In Q8.8 input space, addr k covers codes [k*shift .. (k+1)*shift - 1].
    # We pick the CENTER of each bin as the representative input.
    bin_centers_q = (np.arange(N) * n_codes_per_entry
                     + n_codes_per_entry // 2 - LUT_RANGE_Q)        # int Q8.8
    bin_centers_real = bin_centers_q.astype(np.float64) / scale     # real

    sig_real = 1.0 / (1.0 + np.exp(-bin_centers_real))
    sig_q = np.round(sig_real * scale).astype(np.int64)
    # Clamp to signed Q8.8 range (sigmoid(4) ~= 0.982 -> Q ~= 251; sigmoid asymptotes
    # to 1.0 -> Q = 256 which still fits in 16-bit signed).
    hi = (1 << (DATA_WIDTH - 1)) - 1
    lo = -(1 << (DATA_WIDTH - 1))
    sig_q = np.clip(sig_q, lo, hi).astype(np.int16)

    # Diagnostic
    print(f"sigmoid output range (Q8.8): {sig_q.min()} .. {sig_q.max()}  "
          f"({sig_q.min()/scale:.4f} .. {sig_q.max()/scale:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / LUT_NAME
    mask = (1 << DATA_WIDTH) - 1
    digits = (DATA_WIDTH + 3) // 4
    with out_path.open("w") as f:
        for v in sig_q:
            f.write(f"{int(v) & mask:0{digits}X}\n")
    print(f"wrote {out_path}  ({N} entries)")

    # Also emit metadata so q88_*.py scripts and RTL agree on the address mapping.
    meta_path = OUT_DIR / "sigmoid_q88_meta.txt"
    meta_path.write_text(
        f"# sigmoid LUT params\n"
        f"LUT_RANGE_REAL  = {LUT_RANGE}\n"
        f"LUT_RANGE_Q88   = {LUT_RANGE_Q}\n"
        f"N               = {N}\n"
        f"SHIFT           = {SHIFT}\n"
        f"FRAC_BITS       = {FRAC_BITS}\n"
        f"DATA_WIDTH      = {DATA_WIDTH}\n"
        f"# address calc:  addr = (clip(in_q88, -LUT_RANGE_Q88, LUT_RANGE_Q88-1) + LUT_RANGE_Q88) >> SHIFT\n"
        f"# returns:       Q{DATA_WIDTH-FRAC_BITS}.{FRAC_BITS} signed sigmoid value\n"
    )
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
