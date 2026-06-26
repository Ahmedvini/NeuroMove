#!/usr/bin/env python3
"""
hex_to_window.py — Convert a $readmemh-style .hex file (one 16-bit Q8.8 value
per line) into a raw 6000-byte little-endian int16 binary suitable for streaming
to the Zynq via bridge.py --stream.

Example:
  ./hex_to_window.py data/golden_q88/stage_conv2d_input.hex \\
                     -o build/test_window.bin
  ./bridge.py --stream build/test_window.bin
"""

import argparse
import struct
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hexfile", help="Input .hex file ($readmemh format)")
    ap.add_argument("-o", "--out", default="-",
                    help="Output binary path; '-' = stdout (default)")
    ap.add_argument("--samples", type=int, default=3000,
                    help="Expected number of 16-bit samples (default 3000)")
    args = ap.parse_args()

    samples = []
    with open(args.hexfile) as f:
        for line in f:
            tok = line.strip().split("//")[0].strip()
            if not tok:
                continue
            # Treat as hex; convert to signed 16-bit
            v = int(tok, 16)
            if v >= 0x8000:
                v -= 0x10000
            samples.append(v)

    if len(samples) != args.samples:
        print(f"[hex_to_window] WARNING: got {len(samples)} samples, "
              f"expected {args.samples}", file=sys.stderr)

    # Truncate or pad to exact length
    if len(samples) > args.samples:
        samples = samples[:args.samples]
    while len(samples) < args.samples:
        samples.append(0)

    packed = struct.pack(f"<{args.samples}h", *samples)
    if args.out == "-":
        sys.stdout.buffer.write(packed)
    else:
        with open(args.out, "wb") as f:
            f.write(packed)
        print(f"[hex_to_window] wrote {len(packed)} bytes to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
