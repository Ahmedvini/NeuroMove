#!/usr/bin/env python3
"""
q88_classifier.py — Bit-accurate Q8.8 reference for the per-window Dense(2)
classifier and the final Average+Argmax output head of HaLT DB-ATCNet.

Per docs/FPGA_DEPLOYMENT_DOC.md §5.4:

  For each sliding window w in 0..4:
    tcfn_out_w : (6, 32)   = TCFN block output for window w
    last_w     : (32,)     = tcfn_out_w[-1, :]
    logits_w   : (2,)      = last_w @ W_w + b_w        (Dense_{10+w})

  Average across windows: mean_logits = mean_w(logits_w) over 5 windows.
  Class                  = argmax(mean_logits) ∈ {0, 1}.

Since softmax and division by 5 are monotonic, argmax(mean) == argmax(sum).
The RTL skips both operations.

For window 0 we have a TCFN output already in
data/golden_q88/stage_w0_tcfn_output.hex. For windows 1..4 we don't yet
have full goldens, so this script only emits window-0 Dense weights and
its logits. Extend to all 5 once the per-window goldens are built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8

T_WIN = 6
F     = 32
N_CLS = 2


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


def dense_q88(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    """x_q (Nin,), w_q (Nin, Nout), b_q (Nout,). Q8.8 fully-connected with saturate."""
    acc = x_q.astype(np.int64) @ w_q.astype(np.int64)
    shifted = acc >> FRAC_BITS
    with_bias = shifted + b_q.astype(np.int64)
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    return np.clip(with_bias, LO, HI).astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()
    w = args.window
    assert 0 <= w < 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tcfn_hex = f"data/golden_q88/stage_w{w}_tcfn_output.hex"
    print(f"Loading TCFN output from {tcfn_hex}...")
    tcfn_out = load_q88_hex(tcfn_hex, (T_WIN, F))
    last_step = tcfn_out[-1, :]                                # (32,)
    print(f"  last step abs_max={int(np.abs(last_step).max())}")

    # ---------- Load Dense classifier weights ----------
    dense_idx = 10 + w
    with h5py.File(args.weights, "r") as f:
        dw = f[f"layers/dense_{dense_idx}/vars/0"][()].astype(np.float32)  # (32, 2)
        db = f[f"layers/dense_{dense_idx}/vars/1"][()].astype(np.float32)  # (2,)
    assert dw.shape == (F, N_CLS) and db.shape == (N_CLS,)
    qdw = quantize_q88(dw)
    qdb = quantize_q88(db)
    print(f"  dense_{dense_idx} weights {dw.shape} abs_max={np.abs(dw).max():.4f}")
    print(f"  bias abs_max={np.abs(db).max():.4f}")

    # ---------- Logits ----------
    logits = dense_q88(last_step, qdw, qdb)                     # (2,)
    print(f"  logits: {logits.tolist()}")

    # ---------- Emit ----------
    prefix = f"stage_w{w}_cls"
    (OUT_DIR / f"{prefix}_input.hex").write_text(to_hex(last_step, DATA_WIDTH))
    # Store kernel as (Nout, Nin) row-major: i_out outer, i_in inner.
    (OUT_DIR / f"{prefix}_w.hex"    ).write_text(to_hex(qdw.T,    COEF_WIDTH))
    (OUT_DIR / f"{prefix}_b.hex"    ).write_text(to_hex(qdb,      DATA_WIDTH))
    (OUT_DIR / f"{prefix}_logits.hex").write_text(to_hex(logits,  DATA_WIDTH))
    (OUT_DIR / f"{prefix}_meta.json").write_text(json.dumps({
        "window": w, "F": F, "N_CLS": N_CLS,
        "dense_layer": f"dense_{dense_idx}",
        "logits": logits.tolist(),
        "argmax": int(np.argmax(logits)),
    }, indent=2))
    print(f"wrote {prefix}_{{input,w,b,logits}}.hex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
