#!/usr/bin/env python3
"""
gen_stimulus.py

Generate paired input / expected-output vectors for the bit-exact RTL
verification flow. Outputs are written as 16-bit hex (one per line),
consumable by `$readmemh` from the SystemVerilog testbench.

Layout:
  data/golden_input.hex     - one signed 16-bit value per line (Q8.8)
  data/golden_expected.hex  - one signed 16-bit value per line (Q8.8)
  data/golden_meta.txt      - human-readable summary

Pattern (selectable via --pattern):
  impulse  - delta at the center
  sine     - 0.5 amplitude, 4 cycles across the window
  ramp     - linear ramp from -0.5 to +0.5
  noise    - deterministic pseudo-random in [-0.5, 0.5]
  combo    - impulse + sine + noise stitched back to back (default)

Defaults match the top.sv / top_tb.sv parameterization:
  WINDOW_SIZE = 32, KERNEL_SIZE = 5, NUM_CH = 1, FRAC_BITS = 8

Coefficients used:
  CONV_COEFFS_A = [1.0, 0.5, 0.25, 0.125, 0.0625]   (matches top_tb)
  CONV_COEFFS_B = [0.5, 0.5,  0.5,  0.5,   0.5  ]
  SCALE_FACTORS = [1.0]
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

from golden_model import (
    Params,
    to_q,
    from_q,
    window_pipeline,
)


def to_hex16(value: int) -> str:
    """Format a signed integer as 4-digit 16-bit hex (two's complement)."""
    return f"{value & 0xFFFF:04X}"


def gen_pattern(name: str, n: int, frac: int, width: int) -> list[int]:
    if name == "impulse":
        x = [0] * n
        x[n // 2] = to_q(0.5, frac, width)
        return x
    if name == "sine":
        return [
            to_q(0.5 * math.sin(2 * math.pi * 4 * i / n), frac, width)
            for i in range(n)
        ]
    if name == "ramp":
        return [
            to_q(-0.5 + (i / max(n - 1, 1)), frac, width) for i in range(n)
        ]
    if name == "noise":
        rng = random.Random(0xA7C)  # deterministic
        return [to_q(rng.uniform(-0.5, 0.5), frac, width) for _ in range(n)]
    if name == "combo":
        parts = ["impulse", "sine", "ramp", "noise"]
        chunk = max(n // len(parts), 1)
        out: list[int] = []
        for p in parts:
            out.extend(gen_pattern(p, chunk, frac, width))
        # Pad to n in case of rounding loss.
        out.extend([0] * (n - len(out)))
        return out[:n]
    raise ValueError(f"unknown pattern: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="combo",
                    choices=["impulse", "sine", "ramp", "noise", "combo"])
    ap.add_argument("--samples", type=int, default=256,
                    help="number of input samples to generate (default: 256)")
    ap.add_argument("--window", type=int, default=32, help="WINDOW_SIZE")
    ap.add_argument("--frac",   type=int, default=8,  help="FRAC_BITS")
    ap.add_argument("--width",  type=int, default=16, help="DATA_WIDTH")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    p = Params(DATA_WIDTH=args.width, COEF_WIDTH=args.width, FRAC_BITS=args.frac)

    # Coefficients (real values, then quantized to Q8.8)
    coef_a_real = [1.0, 0.5, 0.25, 0.125, 0.0625]
    coef_b_real = [0.5, 0.5, 0.5,  0.5,   0.5]
    scale_real  = 1.0

    coef_a = [to_q(v, p.FRAC_BITS, p.COEF_WIDTH) for v in coef_a_real]
    coef_b = [to_q(v, p.FRAC_BITS, p.COEF_WIDTH) for v in coef_b_real]
    scale  = to_q(scale_real, p.FRAC_BITS, p.COEF_WIDTH)

    # Stimulus + golden output
    x = gen_pattern(args.pattern, args.samples, p.FRAC_BITS, p.DATA_WIDTH)
    y = window_pipeline(x, args.window, coef_a, coef_b, scale, p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path  = out_dir / "golden_input.hex"
    exp_path = out_dir / "golden_expected.hex"
    meta_path = out_dir / "golden_meta.txt"

    in_path.write_text("\n".join(to_hex16(v) for v in x) + "\n")
    exp_path.write_text("\n".join(to_hex16(v) for v in y) + "\n")

    meta = [
        f"pattern        = {args.pattern}",
        f"samples_in     = {len(x)}",
        f"samples_out    = {len(y)}",
        f"window_size    = {args.window}",
        f"frac_bits      = {p.FRAC_BITS}",
        f"data_width     = {p.DATA_WIDTH}",
        f"coef_a (real)  = {coef_a_real}",
        f"coef_a (Q8.8)  = {[to_hex16(c) for c in coef_a]}",
        f"coef_b (real)  = {coef_b_real}",
        f"coef_b (Q8.8)  = {[to_hex16(c) for c in coef_b]}",
        f"scale  (real)  = {scale_real}",
        f"scale  (Q8.8)  = {to_hex16(scale)}",
        f"input          = {in_path}",
        f"expected       = {exp_path}",
        "",
        "Notes:",
        f"  - The first {args.window - 1} input samples do NOT produce output",
        "    (window_valid stays low until the buffer is full).",
        "  - From input index window_size-1 onward, each input produces one",
        "    output. samples_out should therefore equal samples_in - (window_size - 1).",
        "",
        "First 8 input/expected (real values):",
    ]
    for i in range(min(8, len(y))):
        meta.append(
            f"  in[{args.window - 1 + i:4d}] = {from_q(x[args.window - 1 + i], p.FRAC_BITS):+.5f}  ->  "
            f"y[{i:4d}] = {from_q(y[i], p.FRAC_BITS):+.5f}"
        )
    meta_path.write_text("\n".join(meta) + "\n")

    print(f"wrote {in_path}  ({len(x)} samples)")
    print(f"wrote {exp_path} ({len(y)} samples)")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
