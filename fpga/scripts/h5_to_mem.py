#!/usr/bin/env python3
"""
h5_to_mem.py

Extract trained DB-ATCNet weights from a Keras-3 `.h5` checkpoint and emit
fixed-point `.mem` files for `$readmemh` into BRAM/distributed-RAM weight
ROMs on the FPGA.

The Keras 3 checkpoint format keeps each layer's tensors under
`layers/<layer_name>/vars/<idx>`. The mapping from <idx> to (kernel, bias,
gamma, beta, moving_mean, moving_variance) depends on the layer class but
within a single layer the order is stable.

For the PhysioNet DB-ATCNet checkpoint we observed:
  conv2d / conv2d_*     -> vars/0 = kernel, vars/1 = bias (if present)
  conv1d / conv1d_*     -> vars/0 = kernel, vars/1 = bias
  depthwise_conv2d_*    -> vars/0 = kernel  (no bias by default)
  dense / dense_*       -> vars/0 = kernel, vars/1 = bias
  batch_normalization_* -> vars/0 = gamma, vars/1 = beta,
                           vars/2 = moving_mean, vars/3 = moving_variance

This script does NOT fold BN into the preceding conv yet; that depends on
the layer connectivity which the .h5 alone does not encode. Fold once the
architecture is rebuilt and the layer pairing is known.

Outputs (per --out-dir):
  <layer_name>__<var_role>.mem       Q(I).(FRAC) hex, one value per line
  manifest.json                      shape + scale + stats per file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# Order of vars[] inside a Keras-3 batch_normalization layer.
BN_VAR_NAMES = ["gamma", "beta", "moving_mean", "moving_variance"]
# Conv / dense layers usually have just kernel + bias.
KB_VAR_NAMES = ["kernel", "bias"]
# Depthwise/spatial only kernel.
K_ONLY_VAR_NAMES = ["kernel"]


def role_names_for(layer_name: str, n_vars: int) -> list[str]:
    """Return human-readable role names for each `vars/<i>` in a layer."""
    base = layer_name.split("/")[-1]
    if base.startswith("batch_normalization"):
        return BN_VAR_NAMES[:n_vars]
    if base.startswith("dense") or base.startswith("conv1d") or base.startswith("conv2d"):
        return (KB_VAR_NAMES + [f"var{i}" for i in range(2, n_vars)])[:n_vars]
    if base.startswith("depthwise_conv2d"):
        return (K_ONLY_VAR_NAMES + [f"var{i}" for i in range(1, n_vars)])[:n_vars]
    return [f"var{i}" for i in range(n_vars)]


def quantize_q(arr: np.ndarray, frac_bits: int, width: int) -> np.ndarray:
    """Symmetric-saturation quantization to signed Q(I).(frac_bits)."""
    scale = 1 << frac_bits
    hi = (1 << (width - 1)) - 1
    lo = -(1 << (width - 1))
    q = np.round(arr * scale).astype(np.int64)
    q = np.clip(q, lo, hi).astype(np.int32)
    return q


def to_uint16_two_complement(q: np.ndarray, width: int) -> np.ndarray:
    """View signed q-values as unsigned bit patterns (two's complement)."""
    mask = (1 << width) - 1
    return (q.astype(np.int64) & mask).astype(np.uint64)


def emit_mem(out_path: Path, q: np.ndarray, width: int) -> None:
    u = to_uint16_two_complement(q.flatten(), width)
    digits = (width + 3) // 4
    with out_path.open("w") as f:
        for v in u:
            f.write(f"{int(v):0{digits}X}\n")


def collect_layer_vars(group: h5py.Group, prefix: str = "layers"):
    """Yield (full_path, dataset) for every `<prefix>/<layer>/vars/<i>`."""
    if prefix not in group:
        return
    layers = group[prefix]
    for layer_name in sorted(layers.keys()):
        node = layers[layer_name]
        if "vars" not in node:
            continue
        vars_grp = node["vars"]
        # vars are numeric-indexed; sort numerically.
        keys = sorted(vars_grp.keys(), key=lambda k: int(k) if k.isdigit() else 1 << 30)
        for vk in keys:
            ds = vars_grp[vk]
            if isinstance(ds, h5py.Dataset):
                yield (layer_name, vk, ds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5", help="path to Keras .h5 checkpoint")
    ap.add_argument("--out-dir",  default="data/weights")
    ap.add_argument("--frac",     type=int, default=8,  help="FRAC_BITS (Q8.8 default)")
    ap.add_argument("--width",    type=int, default=16, help="DATA_WIDTH (signed)")
    ap.add_argument("--skip-empty", action="store_true", default=True,
                    help="skip 0-element tensors (Keras leaves some empty slots)")
    args = ap.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        print(f"ERROR: not found: {h5_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source":     str(h5_path),
        "frac_bits":  args.frac,
        "width":      args.width,
        "files":      [],
    }

    n_written = 0
    n_skipped = 0
    with h5py.File(h5_path, "r") as f:
        # Cache role names: needs to know how many vars in each layer.
        per_layer_vars: dict[str, list[tuple[str, h5py.Dataset]]] = {}
        for layer_name, vk, ds in collect_layer_vars(f):
            per_layer_vars.setdefault(layer_name, []).append((vk, ds))

        for layer_name, entries in per_layer_vars.items():
            roles = role_names_for(layer_name, len(entries))
            for (vk, ds), role in zip(entries, roles):
                arr = ds[()].astype(np.float32)
                if arr.size == 0:
                    if args.skip_empty:
                        n_skipped += 1
                        continue
                q = quantize_q(arr, args.frac, args.width)
                fname = f"{layer_name}__{role}.mem"
                emit_mem(out_dir / fname, q, args.width)
                stats = {
                    "file":        fname,
                    "layer":       layer_name,
                    "role":        role,
                    "var_idx":     vk,
                    "shape":       list(arr.shape),
                    "n_elements":  int(arr.size),
                    "real_min":    float(arr.min()),
                    "real_max":    float(arr.max()),
                    "real_mean":   float(arr.mean()),
                    "real_abs_max": float(np.abs(arr).max()),
                    "q_min":       int(q.min()),
                    "q_max":       int(q.max()),
                    "saturated":   int(((q == (1 << (args.width - 1)) - 1)
                                        | (q == -(1 << (args.width - 1)))).sum()),
                }
                manifest["files"].append(stats)
                n_written += 1

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {n_written} .mem files to {out_dir}")
    print(f"skipped {n_skipped} empty tensors")
    print(f"manifest: {manifest_path}")

    # Quick sanity: anything that saturated >5% probably needs a bigger
    # word, a per-layer scale, or BN folding before quantization.
    flagged = [s for s in manifest["files"] if s["n_elements"] > 0 and
               s["saturated"] / s["n_elements"] > 0.05]
    if flagged:
        print()
        print("WARNING: layers with >5% saturation under Q{}.{}:".format(
            args.width - args.frac, args.frac))
        for s in flagged:
            print(f"  {s['file']:50s} sat={s['saturated']}/{s['n_elements']} "
                  f"abs_max={s['real_abs_max']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
