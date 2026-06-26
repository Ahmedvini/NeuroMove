#!/usr/bin/env python3
"""
cat_window_weights.py — Concatenate the per-window .hex weight files into
single multi-bank files for each block type.

The new RTL modules (channel attn, spatial attn, TCFN, classifier) each
hold an internal weight RAM indexed by [window_idx * inner_size + addr],
so the .hex file contains all 5 windows back-to-back.

Output files (in data/golden_q88/):
    all_chan_d1w.hex     5 * (D_RED  * NUM_CH) = 5 * 128 = 640
    all_chan_d1b.hex     5 * D_RED              = 5 * 4   = 20
    all_chan_d2w.hex     5 * (NUM_CH * D_RED)  = 5 * 128 = 640
    all_chan_d2b.hex     5 * NUM_CH            = 5 * 32  = 160
    all_spat_conv_w.hex  5 * (KH * IC_CAT)     = 5 * 21  = 105
    all_tcfn_w<k>.hex    5 * (K * F * F)       = 5 * 4096 = 20480   (k = 0..3)
    all_tcfn_b<k>.hex    5 * F                  = 5 * 32   = 160     (k = 0..3)
    all_cls_w.hex        5 * (N_CLS * F)       = 5 * 64   = 320
    all_cls_b.hex        5 * N_CLS             = 5 * 2    = 10
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path("data/golden_q88")
N_WIN   = 5


def cat(prefix_per_win: str, out_name: str) -> None:
    """Concatenate stage_w0_{prefix}, stage_w1_{prefix}, … into out_name."""
    out = []
    for w in range(N_WIN):
        p = OUT_DIR / f"stage_w{w}_{prefix_per_win}.hex"
        assert p.exists(), f"missing {p}; run per-window q88_layer scripts first"
        out.extend(line.strip() for line in p.read_text().splitlines() if line.strip())
    (OUT_DIR / out_name).write_text("\n".join(out) + "\n")
    print(f"  wrote {OUT_DIR / out_name}  ({len(out)} entries)")


def main() -> int:
    print("Concatenating channel attention weights...")
    cat("chan_d1w", "all_chan_d1w.hex")
    cat("chan_d1b", "all_chan_d1b.hex")
    cat("chan_d2w", "all_chan_d2w.hex")
    cat("chan_d2b", "all_chan_d2b.hex")
    print("\nConcatenating spatial attention weights...")
    cat("spat_conv_w", "all_spat_conv_w.hex")
    print("\nConcatenating TCFN weights...")
    for k in range(4):
        cat(f"tcfn_w{k}", f"all_tcfn_w{k}.hex")
        cat(f"tcfn_b{k}", f"all_tcfn_b{k}.hex")
    print("\nConcatenating classifier weights...")
    cat("cls_w", "all_cls_w.hex")
    cat("cls_b", "all_cls_b.hex")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
