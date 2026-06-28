#!/usr/bin/env python3
"""
q88_branchB.py — Bit-accurate Q8.8 Python references for ALL of Branch B
(Layers 10-14) of HaLT DB-ATCNet.

Branch B is structurally identical to Branch A; only D and the separable
input-channel count differ:
    Layer 10  DepthwiseConv2D((1,5), D=4) + BN_2 (64 ch)   <-- Branch B depthwise
    Layer 11  ELU                                          <-- reuses elu.sv
    Layer 12  AvgPool2D((8,1))                             <-- reuses avg_pool_time.sv (POOL=8)
    Layer 13  Conv2D(32, (16,1)) F_IN=64 + BN_4 (32 ch)    <-- Branch B separable
    Layer 14  ELU + AvgPool2D((7,1))                       <-- reuses

Pipeline shape: (600, 5, 16) → (600, 1, 64) → ELU → (75, 64) → sep → (75, 32)
                                                  → ELU → AvgPool(7) → (10, 32)

Artifacts written to data/golden_q88/:
    stage_branchB_dwise_input.hex   = stage_eca1_output.hex (48000)
    stage_branchB_dwise_weights.hex C*F2_B = 5*64 = 320     (BN_2 folded)
    stage_branchB_dwise_bias.hex    F2_B = 64
    stage_branchB_dwise_output.hex  T*F2_B = 600*64 = 38400

    stage_branchB_elu1_output.hex   = ELU(dwise_output)     T*F2_B = 38400
    stage_branchB_pool1_output.hex  AvgPool(POOL=8)         T_POOL*F2_B = 75*64 = 4800
    stage_branchB_sep_weights.hex   KE*F_IN*F_OUT = 16*64*32 = 32768 (BN_4 folded)
    stage_branchB_sep_bias.hex      F_OUT = 32
    stage_branchB_sep_output.hex    T_POOL*F_OUT = 75*32 = 2400 (pre-ELU)
    stage_branchB_elu2_output.hex   T_POOL*F_OUT = 2400
    stage_branchB_pool2_output.hex  T_POOL2*F_OUT = 10*32 = 320 (FINAL Branch B output)
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_WEIGHTS_H5 = "weights/subject_A/session_3_heldout/best_model.weights.h5"
ECA1_OUT_HEX       = "data/golden_q88/stage_eca1_output.hex"
ELU_LUT_FILE       = "data/lut/elu_q88.hex"
OUT_DIR            = Path("data/golden_q88")

DATA_WIDTH = 16
COEF_WIDTH = 16
FRAC_BITS  = 8
ACC_WIDTH  = 48

T   = 600
C   = 5
F1  = 16
D   = 4
F2_B = F1 * D                   # 64
KE_SEP = 16
F_SEP_IN = F2_B                 # 64
F_SEP_OUT = 32

POOL1 = 8
POOL2 = 7
T_POOL  = T // POOL1            # 75
T_POOL2 = T_POOL // POOL2       # 10
N_USED  = T_POOL2 * POOL2       # 70

# Reciprocal-multiply constants (must match RTL)
INV_POOL1   = 2097152            # 2^24 / 8
INV_POOL2   = 2396745            # round(2^24 / 7)
Q_SHIFT     = 24

# ELU LUT params
LUT_RANGE_Q = 2048
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


def avg_pool_q88(x_q: np.ndarray, pool: int, inv_pool: int) -> np.ndarray:
    """x_q is shape (T, F). Output is (T // pool, F) via reciprocal-multiply.
    Drops the last (T % pool) inputs (matches Keras valid-stride pooling)."""
    T_full = (x_q.shape[0] // pool) * pool
    grouped = x_q[:T_full].astype(np.int64).reshape(-1, pool, x_q.shape[1])
    sums = grouped.sum(axis=1)
    y = (sums * inv_pool) >> Q_SHIFT
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    return np.clip(y, LO, HI).astype(np.int16)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Branch B weights + BN params...")
    with h5py.File(DEFAULT_WEIGHTS_H5, "r") as f:
        w_dwise   = f["layers/depthwise_conv2d_1/vars/0"][()].astype(np.float32)    # (1, 5, F1, D)
        bn2_g     = f["layers/batch_normalization_2/vars/0"][()].astype(np.float32)  # (F2_B,)
        bn2_b     = f["layers/batch_normalization_2/vars/1"][()].astype(np.float32)
        bn2_m     = f["layers/batch_normalization_2/vars/2"][()].astype(np.float32)
        bn2_v     = f["layers/batch_normalization_2/vars/3"][()].astype(np.float32)
        w_sep     = f["layers/conv2d_2/vars/0"][()].astype(np.float32)               # (KE, 1, F_IN, F_OUT)
        bn4_g     = f["layers/batch_normalization_4/vars/0"][()].astype(np.float32)
        bn4_b     = f["layers/batch_normalization_4/vars/1"][()].astype(np.float32)
        bn4_m     = f["layers/batch_normalization_4/vars/2"][()].astype(np.float32)
        bn4_v     = f["layers/batch_normalization_4/vars/3"][()].astype(np.float32)
    eps = 1e-3
    assert w_dwise.shape == (1, C, F1, D),         f"unexpected dwise shape {w_dwise.shape}"
    assert w_sep.shape   == (KE_SEP, 1, F_SEP_IN, F_SEP_OUT), f"unexpected sep shape {w_sep.shape}"
    print(f"  dwise weights {w_dwise.shape}, abs_max={np.abs(w_dwise).max():.4f}")
    print(f"  BN_2 gamma abs_max={np.abs(bn2_g).max():.4f}, var abs_max={np.abs(bn2_v).max():.4f}")
    print(f"  sep weights   {w_sep.shape}, abs_max={np.abs(w_sep).max():.4f}")
    print(f"  BN_4 gamma abs_max={np.abs(bn4_g).max():.4f}, var abs_max={np.abs(bn4_v).max():.4f}")

    # ===== Layer 10: depthwise + BN_2 folded =====
    print("\n=== Layer 10: Branch B depthwise + BN_2 folded ===")
    scale_d = (bn2_g / np.sqrt(bn2_v + eps)).reshape(F1, D)
    bias_d  = (bn2_b - bn2_m * (bn2_g / np.sqrt(bn2_v + eps))).reshape(F1, D)
    w_dwise_folded = w_dwise[0] * scale_d[np.newaxis, :, :]           # (C, F1, D)
    qw_dwise = quantize_q88(w_dwise_folded)
    qb_dwise = quantize_q88(bias_d)
    x_q = load_q88_hex(ECA1_OUT_HEX, (T, C, F1))
    acc = np.einsum('tcf, cfd -> tfd', x_q.astype(np.int64), qw_dwise.astype(np.int64))
    shifted   = acc >> FRAC_BITS
    with_bias = shifted + qb_dwise[np.newaxis, :, :].astype(np.int64)
    HI = (1 << (DATA_WIDTH - 1)) - 1
    LO = -(1 << (DATA_WIDTH - 1))
    dwise_out = np.clip(with_bias, LO, HI).astype(np.int16).reshape(T, F2_B)
    print(f"  output (T,F2_B)={dwise_out.shape}, abs_max={int(np.abs(dwise_out).max())}, "
          f"sat={int(((with_bias > HI) | (with_bias < LO)).sum())}/{with_bias.size}")
    qw_flat = qw_dwise.reshape(C, F2_B)                                # i = f*D + d
    (OUT_DIR / "stage_branchB_dwise_input.hex").write_text(Path(ECA1_OUT_HEX).read_text())
    (OUT_DIR / "stage_branchB_dwise_weights.hex").write_text(to_hex(qw_flat, COEF_WIDTH))
    (OUT_DIR / "stage_branchB_dwise_bias.hex").write_text(to_hex(qb_dwise.reshape(F2_B), DATA_WIDTH))
    (OUT_DIR / "stage_branchB_dwise_output.hex").write_text(to_hex(dwise_out, DATA_WIDTH))
    print(f"  wrote depthwise weights/bias/output ({qw_flat.size}/{F2_B}/{dwise_out.size} values)")

    # ===== Layer 11: ELU =====
    print("\n=== Layer 11: ELU ===")
    lut = load_q88_hex(ELU_LUT_FILE, (LUT_N,))
    elu1_out = elu_q88_lut(dwise_out, lut)
    print(f"  output abs_max={int(np.abs(elu1_out).max())}")
    (OUT_DIR / "stage_branchB_elu1_output.hex").write_text(to_hex(elu1_out, DATA_WIDTH))

    # ===== Layer 12: AvgPool(8,1) =====
    print("\n=== Layer 12: AvgPool(8,1) → (75, 64) ===")
    pool1_out = avg_pool_q88(elu1_out, POOL1, INV_POOL1)
    print(f"  output shape={pool1_out.shape}, abs_max={int(np.abs(pool1_out).max())}")
    (OUT_DIR / "stage_branchB_pool1_output.hex").write_text(to_hex(pool1_out, DATA_WIDTH))

    # ===== Layer 13: separable Conv2D + BN_4 folded =====
    print("\n=== Layer 13: Branch B separable Conv2D (F_IN=64) + BN_4 folded ===")
    scale_s = bn4_g / np.sqrt(bn4_v + eps)                              # (F_OUT,)
    bias_s  = bn4_b - bn4_m * scale_s                                   # (F_OUT,)
    w_sep_folded = w_sep[:, 0, :, :] * scale_s[np.newaxis, np.newaxis, :]  # (KE, F_IN, F_OUT)
    qw_sep = quantize_q88(w_sep_folded)
    qb_sep = quantize_q88(bias_s)
    pad_top = (KE_SEP - 1) // 2     # 7
    x_padded = np.zeros((T_POOL + KE_SEP - 1, F_SEP_IN), dtype=np.int64)
    x_padded[pad_top : pad_top + T_POOL] = pool1_out.astype(np.int64)
    acc = np.zeros((T_POOL, F_SEP_OUT), dtype=np.int64)
    for k in range(KE_SEP):
        acc += np.einsum('ti, ij -> tj',
                         x_padded[k : k + T_POOL].astype(np.int64),
                         qw_sep[k].astype(np.int64))
    shifted   = acc >> FRAC_BITS
    with_bias = shifted + qb_sep[np.newaxis, :].astype(np.int64)
    sep_out = np.clip(with_bias, LO, HI).astype(np.int16)               # (T_POOL, F_OUT)
    print(f"  output {sep_out.shape}, abs_max={int(np.abs(sep_out).max())}, "
          f"sat={int(((with_bias > HI) | (with_bias < LO)).sum())}/{with_bias.size}")
    (OUT_DIR / "stage_branchB_sep_input.hex").write_text(
        (OUT_DIR / "stage_branchB_pool1_output.hex").read_text())
    (OUT_DIR / "stage_branchB_sep_weights.hex").write_text(to_hex(qw_sep.reshape(-1), COEF_WIDTH))
    (OUT_DIR / "stage_branchB_sep_bias.hex").write_text(to_hex(qb_sep, DATA_WIDTH))
    (OUT_DIR / "stage_branchB_sep_output.hex").write_text(to_hex(sep_out, DATA_WIDTH))
    print(f"  wrote sep weights/bias/output ({qw_sep.size}/{F_SEP_OUT}/{sep_out.size} values)")

    # ===== Layer 14a: ELU + Layer 14b: AvgPool(7,1) =====
    print("\n=== Layer 14: ELU + AvgPool(7,1) → (10, 32) ===")
    elu2_out = elu_q88_lut(sep_out, lut)
    (OUT_DIR / "stage_branchB_elu2_output.hex").write_text(to_hex(elu2_out, DATA_WIDTH))
    pool2_out = avg_pool_q88(elu2_out[:N_USED], POOL2, INV_POOL2)
    print(f"  pool2 output shape={pool2_out.shape}, abs_max={int(np.abs(pool2_out).max())}")
    (OUT_DIR / "stage_branchB_pool2_input.hex").write_text(to_hex(elu2_out[:N_USED], DATA_WIDTH))
    (OUT_DIR / "stage_branchB_pool2_output.hex").write_text(to_hex(pool2_out, DATA_WIDTH))

    # ---------- Meta ----------
    meta = {
        "branchB":  {
            "L10": {"op": "depthwise(1,5) D=4 + BN_2", "out_shape": [T, F2_B]},
            "L11": {"op": "ELU",                       "out_shape": [T, F2_B]},
            "L12": {"op": "AvgPool(8,1)",              "out_shape": [T_POOL, F2_B]},
            "L13": {"op": "Conv2D(32,(16,1)) F_IN=64 + BN_4", "out_shape": [T_POOL, F_SEP_OUT]},
            "L14a":{"op": "ELU",                       "out_shape": [T_POOL, F_SEP_OUT]},
            "L14b":{"op": "AvgPool(7,1)",              "out_shape": [T_POOL2, F_SEP_OUT]},
        }
    }
    (OUT_DIR / "stage_branchB_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT_DIR / 'stage_branchB_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
