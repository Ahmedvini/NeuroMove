#!/usr/bin/env python3
"""
q88_layer15.py — Bit-accurate Q8.8 reference for Layer 15 (Add(BranchA, BranchB)).

Inputs:
    data/golden_q88/stage_branchA_pool2_output.hex   (T_POOL2, F_OUT) = (10, 32)
    data/golden_q88/stage_branchB_pool2_output.hex   (T_POOL2, F_OUT) = (10, 32)
Output:
    data/golden_q88/stage_add_output.hex             (10, 32) saturated signed add
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


DATA_WIDTH = 16
OUT_DIR    = Path("data/golden_q88")

T_POOL2 = 10
F_OUT   = 32


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
    a = load_q88_hex(str(OUT_DIR / "stage_branchA_pool2_output.hex"), (T_POOL2, F_OUT))
    b = load_q88_hex(str(OUT_DIR / "stage_branchB_pool2_output.hex"), (T_POOL2, F_OUT))
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    s = a.astype(np.int64) + b.astype(np.int64)
    sat = int(((s > HI) | (s < LO)).sum())
    y = np.clip(s, LO, HI).astype(np.int16)
    print(f"Add(A,B) shape={y.shape} abs_max={int(np.abs(y).max())} sat={sat}/{y.size}")
    (OUT_DIR / "stage_add_input_a.hex").write_text((OUT_DIR / "stage_branchA_pool2_output.hex").read_text())
    (OUT_DIR / "stage_add_input_b.hex").write_text((OUT_DIR / "stage_branchB_pool2_output.hex").read_text())
    (OUT_DIR / "stage_add_output.hex").write_text(to_hex(y, DATA_WIDTH))
    print(f"wrote {OUT_DIR / 'stage_add_output.hex'} ({y.size} values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
