#!/usr/bin/env bash
#
# Elaboration / parse-check for the DB-ATCNet RTL (HaLT variant, 5-of-19
# channels, 2-class output, subject-dependent weights).
#
# Primary tool: Vivado (xvlog + xelab). The design uses SystemVerilog
# constructs (unpacked-array parameters, '{default:0} initializers, multi-
# dimensional unpacked arrays) that Yosys 0.33's built-in SV frontend does
# not support without sv2v.
#
# Usage:   ./build.sh [top_module]
# Default top: db_atcnet_axi  (deployment AXI-Stream + AXI-Lite wrapper)
#
# Alternate useful tops (all bit-exact-verified upstream):
#   db_atcnet_top              raw EEG → 1-bit class (no AXI wrappers)
#   db_atcnet_window_pipeline  ECA₂-buf → class (post-ECA₂ only)
#   db_atcnet_post_eca1        ECA₁-gated stream → ECA₂ buf
#   db_atcnet_conv2d_eca1      raw EEG → ECA₁-gated stream

set -euo pipefail

TOP="${1:-db_atcnet_axi}"

# Model RTL (HaLT pipeline). Order matters where one module instantiates another.
MODEL_RTL=(
  # Core compute primitives
  rtl/util/serial_divider.sv
  rtl/conv/conv2d_temporal.sv
  rtl/conv/depthwise_spatial.sv
  rtl/conv/elu.sv
  rtl/conv/avg_pool_time.sv
  rtl/conv/conv1d_temporal.sv
  rtl/conv/conv1d_temporal_tm.sv
  rtl/conv/sat_add.sv
  rtl/conv/tcfn.sv
  rtl/attention/eca_attention.sv
  rtl/attention/gap_accumulator.sv
  rtl/attention/gate_apply.sv
  rtl/attention/cbam_channel_attn.sv
  rtl/attention/cbam_spatial_attn.sv
  rtl/classifier/dense_classifier.sv
  rtl/classifier/output_head.sv
  # Sub-orchestrators
  rtl/attention/eca1_pipeline.sv
  rtl/conv/branch_pipeline.sv
  # Top-level integrators
  rtl/db_atcnet_window_pipeline.sv
  rtl/db_atcnet_post_eca1.sv
  rtl/db_atcnet_conv2d_eca1.sv
  rtl/db_atcnet_top.sv
  # Deployment plumbing
  rtl/weights/axi_lite_weight_loader.sv
  rtl/weights/weight_bank.sv
  rtl/window/gumbel_channel_adapter.sv
  rtl/window/eeg_channel_adapter.sv
  rtl/db_atcnet_axi.sv
)

# Security stack — separate from the AI pipeline, built alongside.
SEC_RTL=(
  rtl/security/aes_256_core.sv
  rtl/security/aes_256_gcm.sv
  rtl/security/eeg_data_encryptor.sv
  rtl/security/eeg_dataset_config.sv
  rtl/security/eeg_security_top.sv
  rtl/security/gcm_ghash.sv
  rtl/security/hmac_demo_top.sv
  rtl/security/hmac_sha256.sv
  rtl/security/rsa2048_core.sv
  rtl/security/secure_boot.sv
  rtl/security/sha256_core.sv
  rtl/security/sha256_hash_chain.sv
)

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(mktemp -d -t atcnet_build.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

ABS_MODEL=( "${MODEL_RTL[@]/#/$REPO_ROOT/}" )
ABS_SEC=(   "${SEC_RTL[@]/#/$REPO_ROOT/}" )

echo "[build] xvlog parse-check (model + security)..."
( cd "$WORK_DIR" && xvlog -sv "${ABS_MODEL[@]}" "${ABS_SEC[@]}" > xvlog.log 2>&1 ) || {
  echo "[build] xvlog FAILED. Log:"; cat "$WORK_DIR/xvlog.log"; exit 1; }
grep -E "ERROR|WARNING" "$WORK_DIR/xvlog.log" || echo "[build] xvlog: clean"

echo "[build] xelab elaborate top='$TOP'..."
( cd "$WORK_DIR" && xelab "work.$TOP" > xelab.log 2>&1 ) || {
  if grep -q "Failed to link" "$WORK_DIR/xelab.log"; then
    echo "[build] note: xelab compiled RTL OK; final link to xsim needs libc-dev installed."
    grep -E "Compiling module|ERROR" "$WORK_DIR/xelab.log" | head -40
  else
    cat "$WORK_DIR/xelab.log"; exit 1
  fi
}

if command -v sv2v >/dev/null 2>&1; then
  echo "[build] sv2v + yosys stat (optional area estimate)..."
  SV2V_OUT="$WORK_DIR/atcnet_v2005.v"
  sv2v "${MODEL_RTL[@]}" > "$SV2V_OUT"
  yosys -q -p "read_verilog '$SV2V_OUT'; hierarchy -check -top $TOP; proc; opt; stat" \
    || echo "[build] yosys stat pass returned non-zero (informational only)"
else
  echo "[build] (sv2v not installed, skipping yosys stat pass)"
fi

echo "[build] done."
