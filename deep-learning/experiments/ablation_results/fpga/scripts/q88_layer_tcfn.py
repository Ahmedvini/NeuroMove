#!/usr/bin/env python3
"""
q88_layer_tcfn.py — Bit-accurate Q8.8 reference for the TCFN block of one
sliding window. Per docs/FPGA_DEPLOYMENT_DOC.md §5.3:

  input_layer: (T_WIN=6, F=32)
  Stage 0 (dilation=1):
    a = ELU(BN(Conv1D_0(input_layer, k=4, d=1, causal)))
    b = ELU(BN(Conv1D_1(a, k=4, d=1, causal)))
    s0 = ELU(b + input_layer)                          # Residual 1
  Stage 1 (dilation=2):
    c = ELU(BN(Conv1D_2(s0, k=4, d=2, causal)))
    r2 = c + input_layer                                # Residual 2 (no ELU)
    d = ELU(BN(Conv1D_3(r2, k=4, d=2, causal)))
    out = ELU(d + input_layer)                          # Residual 3
  return out  # (T_WIN, F)

Per-window weight indexing (assumed window-major in h5):
  conv1d_{2 + 4*w + k}  for k = 0..3
  BN_{5 + 4*w + k}      for k = 0..3      (γ, β, μ, σ²)

Window 0 → conv1d_{2,3,4,5} + BN_{5,6,7,8}.

Artifacts (window 0):
  stage_w0_tcfn_input.hex       (6*32)  copy of stage_w0_spat_output.hex
  stage_w0_tcfn_w_k.hex         (4*32*32 each, k=0..3) folded weights
  stage_w0_tcfn_b_k.hex         (32 each)  folded biases
  stage_w0_tcfn_s0.hex          (6*32)  end of stage 0
  stage_w0_tcfn_output.hex      (6*32)  final TCFN output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
ELU_LUT_FILE       = "data/lut/elu_q88.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8

T_WIN  = 6
F      = 32
K      = 4
D_LIST = [1, 1, 2, 2]

LUT_RANGE_Q = 2048   # ±8.0 in Q8.8 (matches gen_elu_lut.py defaults)
LUT_N       = 256
LUT_SHIFT   = 3


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


def causal_conv1d_q88(x_q: np.ndarray,
                      qw_folded: np.ndarray,
                      qb_folded: np.ndarray,
                      dilation: int) -> np.ndarray:
    """x_q (T, F_in); qw_folded (K, F_in, F_out); qb_folded (F_out,).
       Causal padding: left pad with (K-1)*dilation zeros, no right pad.
       Output shape (T, F_out)."""
    T_in, F_in = x_q.shape
    F_out = qw_folded.shape[2]
    pad_left = (K - 1) * dilation
    xp = np.zeros((T_in + pad_left, F_in), dtype=np.int64)
    xp[pad_left:] = x_q.astype(np.int64)
    acc = np.zeros((T_in, F_out), dtype=np.int64)
    for k in range(K):
        # tap k corresponds to lookback (K-1-k)*dilation from output t
        offset = pad_left - k * dilation   # = (K-1-k)*dilation
        seg = xp[offset : offset + T_in]    # shape (T_in, F_in)
        acc += np.einsum('ti, ij -> tj',
                         seg.astype(np.int64),
                         qw_folded[k].astype(np.int64))
    shifted   = acc >> FRAC_BITS
    with_bias = shifted + qb_folded[np.newaxis, :].astype(np.int64)
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    return np.clip(with_bias, LO, HI).astype(np.int16)


def sat_add_q88(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    s = a.astype(np.int64) + b.astype(np.int64)
    return np.clip(s, LO, HI).astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_H5)
    args = ap.parse_args()
    w = args.window
    assert 0 <= w < 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    in_hex = f"data/golden_q88/stage_w{w}_spat_output.hex"
    print(f"Loading TCFN input from {in_hex}...")
    x_q = load_q88_hex(in_hex, (T_WIN, F))
    print(f"  input shape={x_q.shape} abs_max={int(np.abs(x_q).max())}")

    # ---------- Load 4 Conv1D + 4 BN sets, fold ----------
    print(f"Loading TCFN weights+BN for window {w}...")
    conv_base = 2 + 4 * w
    bn_base   = 5 + 4 * w
    folded_w = []
    folded_b = []
    eps = 1e-3
    with h5py.File(args.weights, "r") as f:
        for k in range(4):
            wf = f[f"layers/conv1d_{conv_base + k}/vars/0"][()].astype(np.float32)  # (K, 32, 32)
            bf = f[f"layers/conv1d_{conv_base + k}/vars/1"][()].astype(np.float32)  # (32,)
            bn_g = f[f"layers/batch_normalization_{bn_base + k}/vars/0"][()].astype(np.float32)
            bn_b = f[f"layers/batch_normalization_{bn_base + k}/vars/1"][()].astype(np.float32)
            bn_m = f[f"layers/batch_normalization_{bn_base + k}/vars/2"][()].astype(np.float32)
            bn_v = f[f"layers/batch_normalization_{bn_base + k}/vars/3"][()].astype(np.float32)
            scale     = bn_g / np.sqrt(bn_v + eps)
            bias_eff  = bn_b - bn_m * scale + bf * scale
            w_folded  = wf * scale[np.newaxis, np.newaxis, :]
            qw = quantize_q88(w_folded)            # (K, F, F)
            qb = quantize_q88(bias_eff)            # (F,)
            folded_w.append(qw)
            folded_b.append(qb)
            print(f"  conv1d_{conv_base + k}: w abs_max={np.abs(wf).max():.4f} "
                  f"bn_g abs_max={np.abs(bn_g).max():.4f} bias_eff abs_max={np.abs(bias_eff).max():.4f}")

    # ---------- ELU LUT ----------
    lut = load_q88_hex(ELU_LUT_FILE, (LUT_N,))

    # ---------- Stage 0 (d=1) ----------
    print("\nStage 0 (d=1)...")
    a_raw = causal_conv1d_q88(x_q,        folded_w[0], folded_b[0], dilation=1)
    a = elu_q88_lut(a_raw, lut)
    b_raw = causal_conv1d_q88(a,          folded_w[1], folded_b[1], dilation=1)
    b = elu_q88_lut(b_raw, lut)
    s0 = elu_q88_lut(sat_add_q88(b, x_q), lut)
    print(f"  a_raw abs_max={int(np.abs(a_raw).max())}, a abs_max={int(np.abs(a).max())}")
    print(f"  s0 abs_max={int(np.abs(s0).max())}")
    (OUT_DIR / f"stage_w{w}_tcfn_a_raw.hex").write_text(to_hex(a_raw, DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_a.hex"    ).write_text(to_hex(a,     DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_b_raw.hex").write_text(to_hex(b_raw, DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_b.hex"    ).write_text(to_hex(b,     DATA_WIDTH))

    # ---------- Stage 1 (d=2) ----------
    print("\nStage 1 (d=2)...")
    c_raw = causal_conv1d_q88(s0,         folded_w[2], folded_b[2], dilation=2)
    c = elu_q88_lut(c_raw, lut)
    r2 = sat_add_q88(c, x_q)                       # NO ELU
    d_raw = causal_conv1d_q88(r2,         folded_w[3], folded_b[3], dilation=2)
    d = elu_q88_lut(d_raw, lut)
    out = elu_q88_lut(sat_add_q88(d, x_q), lut)
    print(f"  c_raw abs_max={int(np.abs(c_raw).max())}, c abs_max={int(np.abs(c).max())}")
    print(f"  r2 abs_max={int(np.abs(r2).max())}, d abs_max={int(np.abs(d).max())}")
    print(f"  out abs_max={int(np.abs(out).max())}")
    print(f"  d_raw[0, 0..7]: {[hex(int(v) & 0xffff) for v in d_raw[0, 0:8]]}")
    print(f"  d    [0, 0..7]: {[hex(int(v) & 0xffff) for v in d    [0, 0:8]]}")
    print(f"  out  [0, 0..7]: {[hex(int(v) & 0xffff) for v in out  [0, 0:8]]}")
    (OUT_DIR / f"stage_w{w}_tcfn_c_raw.hex").write_text(to_hex(c_raw, DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_c.hex"    ).write_text(to_hex(c,     DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_r2.hex"   ).write_text(to_hex(r2,    DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_d_raw.hex").write_text(to_hex(d_raw, DATA_WIDTH))
    (OUT_DIR / f"stage_w{w}_tcfn_d.hex"    ).write_text(to_hex(d,     DATA_WIDTH))

    # ---------- Emit ----------
    prefix = f"stage_w{w}_tcfn"
    (OUT_DIR / f"{prefix}_input.hex"  ).write_text(to_hex(x_q,  DATA_WIDTH))
    for k in range(4):
        (OUT_DIR / f"{prefix}_w{k}.hex").write_text(to_hex(folded_w[k], COEF_WIDTH))
        (OUT_DIR / f"{prefix}_b{k}.hex").write_text(to_hex(folded_b[k], DATA_WIDTH))
    (OUT_DIR / f"{prefix}_s0.hex"     ).write_text(to_hex(s0,    DATA_WIDTH))
    (OUT_DIR / f"{prefix}_output.hex" ).write_text(to_hex(out,   DATA_WIDTH))
    print(f"wrote {prefix}_{{input,w0..3,b0..3,s0,output}}.hex")

    (OUT_DIR / f"{prefix}_meta.json").write_text(json.dumps({
        "window":     w,
        "T_WIN":      T_WIN, "F": F, "K": K,
        "dilations":  D_LIST,
        "conv_base":  conv_base, "bn_base": bn_base,
        "ordering":   "window-major assumption",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
