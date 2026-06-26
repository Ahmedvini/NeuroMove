#!/usr/bin/env bash
# =============================================================================
# run_vivado_tests.sh  —  DB-ATCNet (HaLT) Vivado Test Runner (Linux)
#
# Usage:
#   chmod +x sim/run_vivado_tests.sh        # first time only
#   ./sim/run_vivado_tests.sh [test_name]
#
# Available tests:
#   conv2d_temporal  - Bit-exact compare for Layer 1+2 (Conv2D temporal F1=16
#                      KE=64 padding='same' + BN folded into bias) against
#                      data/golden_q88/* — needs `scripts/q88_layer1.py`.
#   eca              - Bit-exact compare for ECA₁ (Conv1D 3-tap + sigmoid LUT)
#                      against data/golden_q88/stage_eca1_* — needs
#                      `scripts/gen_sigmoid_lut.py` and `scripts/q88_eca1.py`.
#
#   security    - Full security stack regression (SHA-256, HMAC, AES-GCM,
#                 hash chain, RSA secure boot, eeg_security_top).
#   aes         - AES-256-GCM standalone (NIST vectors).
#   hmac        - SHA-256 + HMAC + Hash Chain + RSA Secure Boot.
#
#   help        - Show this help.
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST=${1:-help}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    cat <<EOF

${CYAN}DB-ATCNet (HaLT) Vivado Test Runner${NC}
====================================

${YELLOW}Usage:${NC}
  ./sim/run_vivado_tests.sh [test_name]

${YELLOW}Available tests:${NC}
  conv2d_temporal - Layer 1+2 (Conv2D + folded BN) bit-exact regression.
  eca             - Layer 3 ECA₁ gate-compute (Conv1D + σ-LUT) bit-exact.
  gap             - Streaming GAP accumulator (reciprocal-multiply) bit-exact.
  gate_apply      - Per-channel gate multiplier bit-exact.
  dwise_a         - Layer 4: Branch A depthwise (1,5) D=2 + BN folded — bit-exact.
  elu             - Layer 5: ELU activation via LUT — bit-exact.
  pool1_a         - Layer 6: Branch A temporal AvgPool(8,1) — bit-exact.
  sep_a           - Layer 7: Branch A separable Conv2D(32,(16,1)) + BN folded — bit-exact.
  pool2_a         - Layer 9: Branch A temporal AvgPool(7,1) — bit-exact.
  dwise_b         - Layer 10: Branch B depthwise (1,5) D=4 + BN folded — bit-exact.
  sep_b           - Layer 13: Branch B separable Conv2D F_IN=64 + BN folded — bit-exact.
  sat_add         - Layer 15: Add(BranchA_pool2, BranchB_pool2) saturating add — bit-exact.
  eca2            - Layer 16: ECA₂ (GAP + Conv1D + sigmoid + gate apply) on (10,32) — bit-exact.
  chan_w0         - ImpCBAM channel attention, sliding window 0 — bit-exact.
  divider         - Unsigned serial divider sanity test (no h5 refs needed).
  spat_w0         - ImpCBAM spatial attention, sliding window 0 — bit-exact.
  tcfn_w0         - TCFN (4 dilated Conv1D + BN-fold + ELU + 3 residuals), window 0 — bit-exact.
  dense_w0        - Dense(32→2) classifier, window 0 — bit-exact.
  head            - Output head (sum logits + argmax) — sanity test (no h5 needed).
  chan_all        - ImpCBAM channel attn across all 5 sliding windows (N_WIN=5) — bit-exact.
  spat_all        - ImpCBAM spatial attn across all 5 sliding windows (N_WIN=5) — bit-exact.
  tcfn_all        - TCFN across all 5 sliding windows (N_WIN=5) — bit-exact.
  pipe            - Full 5-window pipeline (slice + ImpCBAM + TCFN + Dense + head) → 1-bit class.
  sep_a_tm        - Time-multiplexed conv1d_temporal (F_IN parallel, KE*F_IN cycles) — bit-exact.
  sep_b_tm        - Same time-mux variant, Branch B (F_IN=64) — bit-exact.
  eca1_pipe       - ECA₁ orchestrator (gap_acc + eca_attn + gate_apply + buffer) — bit-exact.
  branchA_pipe    - Full Branch A chain (dwise + ELU + pool8 + sep + ELU + pool7) — bit-exact.
  branchB_pipe    - Same chain, Branch B (D=4, F2=64) — bit-exact.
  eca2_pipe       - ECA₂ orchestrator (eca1_pipeline reused with NUM_CH=32, N=10) — bit-exact.
  post_eca1       - branchA + branchB + sat_add + ECA₂ → (10,32) eca2_buf — bit-exact.
  c2e1            - Conv2D + ECA₁ wrapper (raw EEG → ECA₁-gated stream) — bit-exact.
  top             - Full RTL inference (raw EEG → 1-bit class) — end-to-end.
  axil            - AXI4-Lite weight loader + weight_bank sanity test.
  gca             - Gumbel channel adapter (19→5) with runtime LUT — sanity.
  axi             - Deployment AXIS+AXI-Lite wrapper end-to-end test.
  security        - Full security stack (SHA/HMAC/AES/secure boot).
  aes             - AES-256-GCM standalone test.
  hmac            - SHA/HMAC/HashChain/RSA boot integration.
  help            - Show this message.

${YELLOW}Note:${NC} The Conv2D Layer-1 test reads data/golden_q88/stage_conv2d_*.hex.
       Regenerate those for the new HaLT shape (5x600) with:
           python3 scripts/q88_layer1.py
       (script will be updated for HaLT dims in a follow-up commit)

EOF
}

die() {
    echo -e "${RED}$1${NC}" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Layer 1: bit-exact regression of conv2d_temporal vs Q8.8 reference.
# ---------------------------------------------------------------------------
run_conv2d_temporal() {
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Running Layer 1: Conv2D Temporal Bit-Exact Test${NC}"
    echo -e "${GREEN}================================================${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"

    if [ ! -f data/golden_q88/stage_conv2d_input.hex ] || \
       [ ! -f data/golden_q88/stage_conv2d_weights.hex ] || \
       [ ! -f data/golden_q88/stage_conv2d_output.hex ]; then
        die "Q8.8 reference missing. Run: python3 scripts/q88_layer1.py"
    fi

    echo -e "${YELLOW}[1/3] Compiling...${NC}"
    xvlog --sv \
        rtl/conv/conv2d_temporal.sv \
        sim/conv2d_temporal_tb.sv || die "Compilation failed!"

    echo -e "${YELLOW}[2/3] Elaborating...${NC}"
    xelab conv2d_temporal_tb -debug typical || die "Elaboration failed!"

    echo -e "${YELLOW}[3/3] Simulating...${NC}"
    xsim conv2d_temporal_tb -runall

    echo -e "${GREEN}Layer 1 test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 3: ECA₁ gate-compute (Conv1D + sigmoid LUT).
# ---------------------------------------------------------------------------
run_eca() {
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Running Layer 3: ECA₁ Bit-Exact Test${NC}"
    echo -e "${GREEN}================================================${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"

    [ -f data/lut/sigmoid_q88.hex ] || die "missing data/lut/sigmoid_q88.hex — run: python3 scripts/gen_sigmoid_lut.py"
    [ -f data/golden_q88/stage_eca1_input.hex ] || die "missing ECA₁ refs — run: python3 scripts/q88_eca1.py"

    echo -e "${YELLOW}[1/3] Compiling...${NC}"
    xvlog --sv \
        rtl/attention/eca_attention.sv \
        sim/eca_attention_tb.sv || die "Compilation failed!"

    echo -e "${YELLOW}[2/3] Elaborating...${NC}"
    xelab eca_attention_tb -debug typical || die "Elaboration failed!"

    echo -e "${YELLOW}[3/3] Simulating...${NC}"
    xsim eca_attention_tb -runall

    echo -e "${GREEN}ECA₁ test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Streaming GAP accumulator (reciprocal-multiply divide-by-N).
# ---------------------------------------------------------------------------
run_gap() {
    echo ""
    echo -e "${GREEN}Running GAP accumulator bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_conv2d_output.hex ] || die "missing Layer 1+2 output — run: python3 scripts/q88_layer1.py"
    [ -f data/golden_q88/stage_eca1_input.hex    ] || die "missing GAP reference — run: python3 scripts/q88_eca1.py"
    xvlog --sv rtl/attention/gap_accumulator.sv sim/gap_accumulator_tb.sv || die "Compilation failed!"
    xelab gap_accumulator_tb -debug typical || die "Elaboration failed!"
    xsim gap_accumulator_tb -runall
    echo -e "${GREEN}GAP accumulator test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Per-channel gate-apply multiplier.
# ---------------------------------------------------------------------------
run_gate_apply() {
    echo ""
    echo -e "${GREEN}Running gate_apply bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_conv2d_output.hex ] || die "missing Layer 1+2 output — run: python3 scripts/q88_layer1.py"
    [ -f data/golden_q88/stage_eca1_gate.hex     ] || die "missing ECA₁ gate — run: python3 scripts/q88_eca1.py"
    [ -f data/golden_q88/stage_eca1_output.hex   ] || die "missing gated feature map — run: python3 scripts/q88_eca1.py"
    xvlog --sv rtl/attention/gate_apply.sv sim/gate_apply_tb.sv || die "Compilation failed!"
    xelab gate_apply_tb -debug typical || die "Elaboration failed!"
    xsim gate_apply_tb -runall
    echo -e "${GREEN}gate_apply test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 4: Branch A depthwise (1,5) D=2 + BN folded.
# ---------------------------------------------------------------------------
run_dwise_a() {
    echo ""
    echo -e "${GREEN}Running Layer 4 (Branch A depthwise + BN) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchA_dwise_input.hex   ] || die "missing refs — run: python3 scripts/q88_layer4.py"
    [ -f data/golden_q88/stage_branchA_dwise_weights.hex ] || die "missing refs — run: python3 scripts/q88_layer4.py"
    [ -f data/golden_q88/stage_branchA_dwise_bias.hex    ] || die "missing refs — run: python3 scripts/q88_layer4.py"
    [ -f data/golden_q88/stage_branchA_dwise_output.hex  ] || die "missing refs — run: python3 scripts/q88_layer4.py"
    xvlog --sv rtl/conv/depthwise_spatial.sv sim/depthwise_spatial_tb.sv || die "Compilation failed!"
    xelab depthwise_spatial_tb -debug typical || die "Elaboration failed!"
    xsim depthwise_spatial_tb -runall
    echo -e "${GREEN}Branch A depthwise test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 5: ELU activation (LUT-based).
# ---------------------------------------------------------------------------
run_elu() {
    echo ""
    echo -e "${GREEN}Running Layer 5 (ELU) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/lut/elu_q88.hex ] || die "missing ELU LUT — run: python3 scripts/gen_elu_lut.py"
    [ -f data/golden_q88/stage_branchA_elu_output.hex ] || die "missing refs — run: python3 scripts/q88_layer5.py"
    xvlog --sv rtl/conv/elu.sv sim/elu_tb.sv || die "Compilation failed!"
    xelab elu_tb -debug typical || die "Elaboration failed!"
    xsim elu_tb -runall
    echo -e "${GREEN}ELU test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 6: temporal AvgPool(8,1) — Branch A pool1 (and reusable elsewhere).
# ---------------------------------------------------------------------------
run_pool1_a() {
    echo ""
    echo -e "${GREEN}Running Layer 6 (Branch A AvgPool 8,1) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchA_pool1_output.hex ] || die "missing refs — run: python3 scripts/q88_layer6.py"
    xvlog --sv rtl/conv/avg_pool_time.sv sim/avg_pool_time_tb.sv || die "Compilation failed!"
    xelab avg_pool_time_tb -debug typical || die "Elaboration failed!"
    xsim avg_pool_time_tb -runall
    echo -e "${GREEN}AvgPool test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 7: Branch A separable Conv2D(32,(16,1)) + BN folded.
# ---------------------------------------------------------------------------
run_sep_a() {
    echo ""
    echo -e "${GREEN}Running Layer 7 (Branch A separable + BN) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchA_sep_output.hex ] || die "missing refs — run: python3 scripts/q88_layer7.py"
    xvlog --sv rtl/conv/conv1d_temporal.sv sim/conv1d_temporal_tb.sv || die "Compilation failed!"
    xelab conv1d_temporal_tb -debug typical || die "Elaboration failed!"
    xsim conv1d_temporal_tb -runall
    echo -e "${GREEN}Branch A separable test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 9: Branch A temporal AvgPool(7, 1).
# ---------------------------------------------------------------------------
run_pool2_a() {
    echo ""
    echo -e "${GREEN}Running Layer 9 (Branch A AvgPool 7,1) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchA_pool2_output.hex ] || die "missing refs — run: python3 scripts/q88_layer8_9.py"
    xvlog --sv rtl/conv/avg_pool_time.sv sim/avg_pool_time_7_tb.sv || die "Compilation failed!"
    xelab avg_pool_time_7_tb -debug typical || die "Elaboration failed!"
    xsim avg_pool_time_7_tb -runall
    echo -e "${GREEN}Branch A AvgPool(7,1) test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 10: Branch B depthwise (1,5) D=4 + BN folded.
# ---------------------------------------------------------------------------
run_dwise_b() {
    echo ""
    echo -e "${GREEN}Running Layer 10 (Branch B depthwise + BN) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchB_dwise_output.hex ] || die "missing refs — run: python3 scripts/q88_branchB.py"
    xvlog --sv rtl/conv/depthwise_spatial.sv sim/depthwise_spatial_b_tb.sv || die "Compilation failed!"
    xelab depthwise_spatial_b_tb -debug typical || die "Elaboration failed!"
    xsim depthwise_spatial_b_tb -runall
    echo -e "${GREEN}Branch B depthwise test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 13: Branch B separable Conv2D(32,(16,1)) F_IN=64 + BN folded.
# ---------------------------------------------------------------------------
run_sep_b() {
    echo ""
    echo -e "${GREEN}Running Layer 13 (Branch B separable + BN) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchB_sep_output.hex ] || die "missing refs — run: python3 scripts/q88_branchB.py"
    xvlog --sv rtl/conv/conv1d_temporal.sv sim/conv1d_temporal_b_tb.sv || die "Compilation failed!"
    xelab conv1d_temporal_b_tb -debug typical || die "Elaboration failed!"
    xsim conv1d_temporal_b_tb -runall
    echo -e "${GREEN}Branch B separable test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 15: Add(BranchA_pool2, BranchB_pool2) saturating add.
# ---------------------------------------------------------------------------
run_sat_add() {
    echo ""
    echo -e "${GREEN}Running Layer 15 (Add A+B saturate) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_add_output.hex ] || die "missing refs — run: python3 scripts/q88_layer15.py"
    xvlog --sv rtl/conv/sat_add.sv sim/sat_add_tb.sv || die "Compilation failed!"
    xelab sat_add_tb -debug typical || die "Elaboration failed!"
    xsim sat_add_tb -runall
    echo -e "${GREEN}Add(A,B) test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Layer 16: ECA₂ integrated (GAP + Conv1D + sigmoid + gate apply) on (10, 32).
# ---------------------------------------------------------------------------
run_eca2() {
    echo ""
    echo -e "${GREEN}Running Layer 16 (ECA₂) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_eca2_output.hex ] || die "missing refs — run: python3 scripts/q88_layer16.py"
    xvlog --sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        sim/eca2_tb.sv || die "Compilation failed!"
    xelab eca2_tb -debug typical || die "Elaboration failed!"
    xsim eca2_tb -runall
    echo -e "${GREEN}ECA₂ integrated test complete!${NC}"
}

# ---------------------------------------------------------------------------
# ImpCBAM channel attention (sliding window 0).
# ---------------------------------------------------------------------------
run_chan_w0() {
    echo ""
    echo -e "${GREEN}Running ImpCBAM channel attn (window 0) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_w0_chan_output.hex ] || die "missing refs — run: python3 scripts/q88_layer17_chan.py --window 0"
    xvlog --sv \
        rtl/attention/cbam_channel_attn.sv \
        sim/cbam_channel_attn_tb.sv || die "Compilation failed!"
    xelab cbam_channel_attn_tb -debug typical || die "Elaboration failed!"
    xsim cbam_channel_attn_tb -runall
    echo -e "${GREEN}ImpCBAM channel attn test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Sanity TB for the unsigned serial divider.
# ---------------------------------------------------------------------------
run_divider() {
    echo ""
    echo -e "${GREEN}Running serial divider sanity test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    xvlog --sv rtl/util/serial_divider.sv sim/serial_divider_tb.sv || die "Compilation failed!"
    xelab serial_divider_tb -debug typical || die "Elaboration failed!"
    xsim serial_divider_tb -runall
    echo -e "${GREEN}Divider sanity complete!${NC}"
}

# ---------------------------------------------------------------------------
# ImpCBAM spatial attention (sliding window 0).
# ---------------------------------------------------------------------------
run_spat_w0() {
    echo ""
    echo -e "${GREEN}Running ImpCBAM spatial attn (window 0) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_w0_spat_output.hex ] || die "missing refs — run: python3 scripts/q88_layer17_spatial.py --window 0"
    xvlog --sv \
        rtl/util/serial_divider.sv \
        rtl/attention/cbam_spatial_attn.sv \
        sim/cbam_spatial_attn_tb.sv || die "Compilation failed!"
    xelab cbam_spatial_attn_tb -debug typical || die "Elaboration failed!"
    xsim cbam_spatial_attn_tb -runall
    echo -e "${GREEN}ImpCBAM spatial attn test complete!${NC}"
}

# ---------------------------------------------------------------------------
# TCFN per-window (sliding window 0).
# ---------------------------------------------------------------------------
run_tcfn_w0() {
    echo ""
    echo -e "${GREEN}Running TCFN (window 0) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_w0_tcfn_output.hex ] || die "missing refs — run: python3 scripts/q88_layer_tcfn.py --window 0"
    xvlog --sv \
        rtl/conv/tcfn.sv \
        sim/tcfn_tb.sv || die "Compilation failed!"
    xelab tcfn_tb -debug typical || die "Elaboration failed!"
    xsim tcfn_tb -runall
    echo -e "${GREEN}TCFN test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Dense(32→2) classifier per window.
# ---------------------------------------------------------------------------
run_dense_w0() {
    echo ""
    echo -e "${GREEN}Running Dense(32->2) classifier (window 0) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_w0_cls_logits.hex ] || die "missing refs — run: python3 scripts/q88_classifier.py --window 0"
    xvlog --sv \
        rtl/classifier/dense_classifier.sv \
        sim/dense_classifier_tb.sv || die "Compilation failed!"
    xelab dense_classifier_tb -debug typical || die "Elaboration failed!"
    xsim dense_classifier_tb -runall
    echo -e "${GREEN}Dense classifier test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Output head (sum + argmax) sanity test.
# ---------------------------------------------------------------------------
run_head() {
    echo ""
    echo -e "${GREEN}Running output head sanity test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    xvlog --sv \
        rtl/classifier/output_head.sv \
        sim/output_head_tb.sv || die "Compilation failed!"
    xelab output_head_tb -debug typical || die "Elaboration failed!"
    xsim output_head_tb -runall
    echo -e "${GREEN}Output head sanity complete!${NC}"
}

# ---------------------------------------------------------------------------
# ImpCBAM channel attention — all 5 sliding windows (N_WIN=5).
# ---------------------------------------------------------------------------
run_chan_all() {
    echo ""
    echo -e "${GREEN}Running ImpCBAM channel attn (all 5 windows) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/all_chan_d1w.hex ] || die "missing combined refs — run: python3 scripts/cat_window_weights.py"
    xvlog --sv \
        rtl/attention/cbam_channel_attn.sv \
        sim/cbam_channel_attn_all_tb.sv || die "Compilation failed!"
    xelab cbam_channel_attn_all_tb -debug typical || die "Elaboration failed!"
    xsim cbam_channel_attn_all_tb -runall
    echo -e "${GREEN}ImpCBAM channel attn (all windows) test complete!${NC}"
}

# ---------------------------------------------------------------------------
# ImpCBAM spatial attention — all 5 sliding windows (N_WIN=5).
# ---------------------------------------------------------------------------
run_spat_all() {
    echo ""
    echo -e "${GREEN}Running ImpCBAM spatial attn (all 5 windows) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/all_spat_conv_w.hex ] || die "missing combined refs — run: python3 scripts/cat_window_weights.py"
    xvlog --sv \
        rtl/util/serial_divider.sv \
        rtl/attention/cbam_spatial_attn.sv \
        sim/cbam_spatial_attn_all_tb.sv || die "Compilation failed!"
    xelab cbam_spatial_attn_all_tb -debug typical || die "Elaboration failed!"
    xsim cbam_spatial_attn_all_tb -runall
    echo -e "${GREEN}ImpCBAM spatial attn (all windows) test complete!${NC}"
}

# ---------------------------------------------------------------------------
# TCFN — all 5 sliding windows (N_WIN=5).
# ---------------------------------------------------------------------------
run_tcfn_all() {
    echo ""
    echo -e "${GREEN}Running TCFN (all 5 windows) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/all_tcfn_w0.hex ] || die "missing combined refs — run: python3 scripts/cat_window_weights.py"
    xvlog --sv \
        rtl/conv/tcfn.sv \
        sim/tcfn_all_tb.sv || die "Compilation failed!"
    xelab tcfn_all_tb -debug typical || die "Elaboration failed!"
    xsim tcfn_all_tb -runall
    echo -e "${GREEN}TCFN (all windows) test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Full 5-window pipeline → final argmax class (end-to-end bit-exact).
# ---------------------------------------------------------------------------
run_pipe() {
    echo ""
    echo -e "${GREEN}Running 5-window end-to-end pipeline test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/all_chan_d1w.hex ] || die "missing combined refs — run: python3 scripts/cat_window_weights.py"
    xvlog --sv \
        rtl/util/serial_divider.sv \
        rtl/attention/cbam_channel_attn.sv \
        rtl/attention/cbam_spatial_attn.sv \
        rtl/conv/tcfn.sv \
        rtl/classifier/dense_classifier.sv \
        rtl/classifier/output_head.sv \
        rtl/db_atcnet_window_pipeline.sv \
        sim/db_atcnet_window_pipeline_tb.sv || die "Compilation failed!"
    xelab db_atcnet_window_pipeline_tb -debug typical || die "Elaboration failed!"
    xsim db_atcnet_window_pipeline_tb -runall
    echo -e "${GREEN}Pipeline end-to-end test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Time-multiplexed conv1d_temporal_tm.sv on Branch A separable golden.
# ---------------------------------------------------------------------------
run_sep_a_tm() {
    echo ""
    echo -e "${GREEN}Running time-mux conv1d_temporal (Branch A sep) bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchA_sep_output.hex ] || die "missing refs — run: python3 scripts/q88_layer7.py"
    xvlog --sv rtl/conv/conv1d_temporal_tm.sv sim/conv1d_temporal_tm_tb.sv || die "Compilation failed!"
    xelab conv1d_temporal_tm_tb -debug typical || die "Elaboration failed!"
    xsim conv1d_temporal_tm_tb -runall
    echo -e "${GREEN}Time-mux conv1d test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Time-multiplexed conv1d_temporal_tm.sv on Branch B separable golden.
# ---------------------------------------------------------------------------
run_sep_b_tm() {
    echo ""
    echo -e "${GREEN}Running time-mux conv1d_temporal (Branch B sep, F_IN=64) test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchB_sep_output.hex ] || die "missing refs — run: python3 scripts/q88_branchB.py"
    xvlog --sv rtl/conv/conv1d_temporal_tm.sv sim/conv1d_temporal_tm_b_tb.sv || die "Compilation failed!"
    xelab conv1d_temporal_tm_b_tb -debug typical || die "Elaboration failed!"
    xsim conv1d_temporal_tm_b_tb -runall
    echo -e "${GREEN}Time-mux Branch B conv1d test complete!${NC}"
}

# ---------------------------------------------------------------------------
# ECA₁ orchestrator (3000-frame ingest + gate compute + replay).
# ---------------------------------------------------------------------------
run_eca1_pipe() {
    echo ""
    echo -e "${GREEN}Running ECA₁ orchestrator bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_eca1_output.hex ] || die "missing refs — run: python3 scripts/q88_eca1.py"
    xvlog --sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        rtl/attention/eca1_pipeline.sv \
        sim/eca1_pipeline_tb.sv || die "Compilation failed!"
    xelab eca1_pipeline_tb -debug typical || die "Elaboration failed!"
    xsim eca1_pipeline_tb -runall
    echo -e "${GREEN}ECA₁ orchestrator test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Branch A pipeline (dwise + ELU + pool8 + sep + ELU + pool7).
# ---------------------------------------------------------------------------
run_branchA_pipe() {
    echo ""
    echo -e "${GREEN}Running Branch A pipeline bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchA_pool2_output.hex ] || die "missing refs — run: python3 scripts/q88_layer7.py q88_layer8_9.py"
    xvlog --sv \
        rtl/conv/depthwise_spatial.sv \
        rtl/conv/elu.sv \
        rtl/conv/avg_pool_time.sv \
        rtl/conv/conv1d_temporal_tm.sv \
        rtl/conv/branch_pipeline.sv \
        sim/branch_pipeline_tb.sv || die "Compilation failed!"
    xelab branch_pipeline_tb -debug typical || die "Elaboration failed!"
    xsim branch_pipeline_tb -runall
    echo -e "${GREEN}Branch A pipeline test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Branch B pipeline (D=4, F2=64).
# ---------------------------------------------------------------------------
run_branchB_pipe() {
    echo ""
    echo -e "${GREEN}Running Branch B pipeline bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_branchB_pool2_output.hex ] || die "missing refs — run: python3 scripts/q88_branchB.py"
    xvlog --sv \
        rtl/conv/depthwise_spatial.sv \
        rtl/conv/elu.sv \
        rtl/conv/avg_pool_time.sv \
        rtl/conv/conv1d_temporal_tm.sv \
        rtl/conv/branch_pipeline.sv \
        sim/branch_pipeline_b_tb.sv || die "Compilation failed!"
    xelab branch_pipeline_b_tb -debug typical || die "Elaboration failed!"
    xsim branch_pipeline_b_tb -runall
    echo -e "${GREEN}Branch B pipeline test complete!${NC}"
}

# ---------------------------------------------------------------------------
# ECA₂ orchestrator — reuses the eca1_pipeline module.
# ---------------------------------------------------------------------------
run_eca2_pipe() {
    echo ""
    echo -e "${GREEN}Running ECA₂ orchestrator bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_eca2_output.hex ] || die "missing refs — run: python3 scripts/q88_layer16.py"
    xvlog --sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        rtl/attention/eca1_pipeline.sv \
        sim/eca2_pipeline_tb.sv || die "Compilation failed!"
    xelab eca2_pipeline_tb -debug typical || die "Elaboration failed!"
    xsim eca2_pipeline_tb -runall
    echo -e "${GREEN}ECA₂ orchestrator test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Post-ECA₁ orchestrator: branchA + branchB + sat_add + ECA₂.
# ---------------------------------------------------------------------------
run_post_eca1() {
    echo ""
    echo -e "${GREEN}Running post-ECA₁ orchestrator bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_eca2_output.hex ] || die "missing refs — run: python3 scripts/q88_layer16.py"
    xvlog --sv \
        rtl/conv/depthwise_spatial.sv \
        rtl/conv/elu.sv \
        rtl/conv/avg_pool_time.sv \
        rtl/conv/conv1d_temporal_tm.sv \
        rtl/conv/branch_pipeline.sv \
        rtl/conv/sat_add.sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        rtl/attention/eca1_pipeline.sv \
        rtl/db_atcnet_post_eca1.sv \
        sim/db_atcnet_post_eca1_tb.sv || die "Compilation failed!"
    xelab db_atcnet_post_eca1_tb -debug typical || die "Elaboration failed!"
    xsim db_atcnet_post_eca1_tb -runall
    echo -e "${GREEN}Post-ECA₁ orchestrator test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Conv2D + ECA₁ wrapper (raw EEG → ECA₁-gated stream).
# ---------------------------------------------------------------------------
run_c2e1() {
    echo ""
    echo -e "${GREEN}Running Conv2D + ECA₁ wrapper bit-exact test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/stage_eca1_output.hex ] || die "missing refs — run: python3 scripts/q88_eca1.py"
    xvlog --sv \
        rtl/conv/conv2d_temporal.sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        rtl/attention/eca1_pipeline.sv \
        rtl/db_atcnet_conv2d_eca1.sv \
        sim/db_atcnet_conv2d_eca1_tb.sv || die "Compilation failed!"
    xelab db_atcnet_conv2d_eca1_tb -debug typical || die "Elaboration failed!"
    xsim db_atcnet_conv2d_eca1_tb -runall
    echo -e "${GREEN}Conv2D + ECA₁ wrapper test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Full end-to-end RTL inference (raw EEG → 1-bit class).
# ---------------------------------------------------------------------------
run_top() {
    echo ""
    echo -e "${GREEN}Running full end-to-end RTL inference${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/all_chan_d1w.hex ] || die "missing combined refs — run: python3 scripts/cat_window_weights.py"
    xvlog --sv \
        rtl/conv/conv2d_temporal.sv \
        rtl/conv/depthwise_spatial.sv \
        rtl/conv/elu.sv \
        rtl/conv/avg_pool_time.sv \
        rtl/conv/conv1d_temporal_tm.sv \
        rtl/conv/branch_pipeline.sv \
        rtl/conv/sat_add.sv \
        rtl/conv/tcfn.sv \
        rtl/util/serial_divider.sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        rtl/attention/eca1_pipeline.sv \
        rtl/attention/cbam_channel_attn.sv \
        rtl/attention/cbam_spatial_attn.sv \
        rtl/classifier/dense_classifier.sv \
        rtl/classifier/output_head.sv \
        rtl/db_atcnet_window_pipeline.sv \
        rtl/db_atcnet_post_eca1.sv \
        rtl/db_atcnet_conv2d_eca1.sv \
        rtl/db_atcnet_top.sv \
        sim/db_atcnet_top_tb.sv || die "Compilation failed!"
    xelab db_atcnet_top_tb -debug typical || die "Elaboration failed!"
    xsim db_atcnet_top_tb -runall
    echo -e "${GREEN}Top end-to-end test complete!${NC}"
}

# ---------------------------------------------------------------------------
# AXI4-Lite weight loader sanity test (no h5 refs needed).
# ---------------------------------------------------------------------------
run_axil() {
    echo ""
    echo -e "${GREEN}Running AXI4-Lite weight loader sanity${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    xvlog --sv \
        rtl/weights/axi_lite_weight_loader.sv \
        rtl/weights/weight_bank.sv \
        sim/axil_weight_loader_tb.sv || die "Compilation failed!"
    xelab axil_weight_loader_tb -debug typical || die "Elaboration failed!"
    xsim axil_weight_loader_tb -runall
    echo -e "${GREEN}AXI4-Lite sanity complete!${NC}"
}

# ---------------------------------------------------------------------------
# Gumbel 19→5 channel adapter sanity test.
# ---------------------------------------------------------------------------
run_gca() {
    echo ""
    echo -e "${GREEN}Running Gumbel 19→5 channel adapter sanity${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    xvlog --sv \
        rtl/window/gumbel_channel_adapter.sv \
        sim/gumbel_channel_adapter_tb.sv || die "Compilation failed!"
    xelab gumbel_channel_adapter_tb -debug typical || die "Elaboration failed!"
    xsim gumbel_channel_adapter_tb -runall
    echo -e "${GREEN}Gumbel adapter sanity complete!${NC}"
}

# ---------------------------------------------------------------------------
# Deployment wrapper: AXIS raw EEG + AXI-Lite control/mailbox.
# ---------------------------------------------------------------------------
run_axi() {
    echo ""
    echo -e "${GREEN}Running deployment AXIS+AXI-Lite wrapper test${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    [ -f data/golden_q88/all_chan_d1w.hex ] || die "missing refs — run: python3 scripts/cat_window_weights.py"
    xvlog --sv \
        rtl/conv/conv2d_temporal.sv \
        rtl/conv/depthwise_spatial.sv \
        rtl/conv/elu.sv \
        rtl/conv/avg_pool_time.sv \
        rtl/conv/conv1d_temporal_tm.sv \
        rtl/conv/branch_pipeline.sv \
        rtl/conv/sat_add.sv \
        rtl/conv/tcfn.sv \
        rtl/util/serial_divider.sv \
        rtl/attention/gap_accumulator.sv \
        rtl/attention/eca_attention.sv \
        rtl/attention/gate_apply.sv \
        rtl/attention/eca1_pipeline.sv \
        rtl/attention/cbam_channel_attn.sv \
        rtl/attention/cbam_spatial_attn.sv \
        rtl/classifier/dense_classifier.sv \
        rtl/classifier/output_head.sv \
        rtl/db_atcnet_window_pipeline.sv \
        rtl/db_atcnet_post_eca1.sv \
        rtl/db_atcnet_conv2d_eca1.sv \
        rtl/db_atcnet_top.sv \
        rtl/weights/axi_lite_weight_loader.sv \
        rtl/window/gumbel_channel_adapter.sv \
        rtl/db_atcnet_axi.sv \
        sim/db_atcnet_axi_tb.sv || die "Compilation failed!"
    xelab db_atcnet_axi_tb -debug typical || die "Elaboration failed!"
    xsim db_atcnet_axi_tb -runall
    echo -e "${GREEN}Deployment wrapper test complete!${NC}"
}

# ---------------------------------------------------------------------------
# Security stack — unchanged from original, just kept here.
# ---------------------------------------------------------------------------
run_security() {
    echo ""
    echo -e "${GREEN}Running ALL Security Algorithms${NC}"
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"

    # Part 1: AES-256-GCM standalone
    echo -e "${YELLOW}-- AES-256-GCM standalone --${NC}"
    xvlog --sv \
        rtl/security/gcm_ghash.sv \
        rtl/security/aes_256_core.sv \
        rtl/security/aes_256_gcm.sv \
        sim/aes_256_gcm_tb.sv || die "AES compilation failed!"
    xelab aes_256_gcm_tb -debug typical || die "AES elaboration failed!"
    xsim aes_256_gcm_tb -runall

    # Part 2: full security stack
    echo -e "${YELLOW}-- Full security stack --${NC}"
    xvlog --sv \
        rtl/security/sha256_core.sv \
        rtl/security/sha256_hash_chain.sv \
        rtl/security/hmac_sha256.sv \
        rtl/security/gcm_ghash.sv \
        rtl/security/aes_256_core.sv \
        rtl/security/aes_256_gcm.sv \
        rtl/security/rsa2048_core.sv \
        rtl/security/secure_boot.sv \
        rtl/security/eeg_security_top.sv \
        sim/hmac_sha256_tb.sv || die "Security compilation failed!"
    xelab hmac_sha256_tb -debug typical || die "Security elaboration failed!"
    xsim hmac_sha256_tb -runall

    echo -e "${GREEN}Security stack complete!${NC}"
}

run_hmac() {
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    xvlog --sv \
        rtl/security/sha256_core.sv \
        rtl/security/sha256_hash_chain.sv \
        rtl/security/hmac_sha256.sv \
        rtl/security/gcm_ghash.sv \
        rtl/security/aes_256_core.sv \
        rtl/security/aes_256_gcm.sv \
        rtl/security/rsa2048_core.sv \
        rtl/security/secure_boot.sv \
        rtl/security/eeg_security_top.sv \
        sim/hmac_sha256_tb.sv || die "Compilation failed!"
    xelab hmac_sha256_tb -debug typical || die "Elaboration failed!"
    xsim hmac_sha256_tb -runall
}

run_aes() {
    cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
    xvlog --sv \
        rtl/security/gcm_ghash.sv \
        rtl/security/aes_256_core.sv \
        rtl/security/aes_256_gcm.sv \
        sim/aes_256_gcm_tb.sv || die "Compilation failed!"
    xelab aes_256_gcm_tb -debug typical || die "Elaboration failed!"
    xsim aes_256_gcm_tb -runall
}

case "$TEST" in
    conv2d_temporal) run_conv2d_temporal ;;
    eca)             run_eca ;;
    gap)             run_gap ;;
    gate_apply)      run_gate_apply ;;
    dwise_a)         run_dwise_a ;;
    elu)             run_elu ;;
    pool1_a)         run_pool1_a ;;
    sep_a)           run_sep_a ;;
    pool2_a)         run_pool2_a ;;
    dwise_b)         run_dwise_b ;;
    sep_b)           run_sep_b ;;
    sat_add)         run_sat_add ;;
    eca2)            run_eca2 ;;
    chan_w0)         run_chan_w0 ;;
    divider)         run_divider ;;
    spat_w0)         run_spat_w0 ;;
    tcfn_w0)         run_tcfn_w0 ;;
    dense_w0)        run_dense_w0 ;;
    head)            run_head ;;
    chan_all)        run_chan_all ;;
    spat_all)        run_spat_all ;;
    tcfn_all)        run_tcfn_all ;;
    pipe)            run_pipe ;;
    sep_a_tm)        run_sep_a_tm ;;
    sep_b_tm)        run_sep_b_tm ;;
    eca1_pipe)       run_eca1_pipe ;;
    branchA_pipe)    run_branchA_pipe ;;
    branchB_pipe)    run_branchB_pipe ;;
    eca2_pipe)       run_eca2_pipe ;;
    post_eca1)       run_post_eca1 ;;
    c2e1)            run_c2e1 ;;
    top)             run_top ;;
    axil)            run_axil ;;
    gca)             run_gca ;;
    axi)             run_axi ;;
    security)        run_security ;;
    hmac)            run_hmac ;;
    aes)             run_aes ;;
    help|*)          show_help ;;
esac
