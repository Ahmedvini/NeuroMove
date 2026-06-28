# =============================================================================
# synth_fit.tcl  —  Vivado synthesis + resource-utilization report for
# the DB-ATCNet deployment block.
#
# Usage:  vivado -mode batch -source synth/synth_fit.tcl -tclargs <part>
#         e.g. xczu7ev-ffvc1156-2-e  (ZCU104)
#              xczu3eg-sbva484-1-e   (Ultra96 / KU3P-class)
#              xc7z020clg400-1       (Z-7020 / PYNQ-Z2; tight fit)
#
# Outputs (in build/synth_<part>/):
#   utilization.txt        — LUT/FF/DSP/BRAM totals
#   utilization_hier.txt   — hierarchical breakdown
#   timing.txt             — worst-case-slack report
#
# IMPORTANT — why we bind every *_FILE generic explicitly:
#   The NN modules (conv2d_temporal, depthwise_spatial, conv1d_temporal_tm,
#   tcfn, dense_classifier, eca_attention, cbam_*) store weights in
#   internal arrays initialised at elaboration via $readmemh(WEIGHTS_FILE,…).
#   If WEIGHTS_FILE is the default empty string, $readmemh is silently
#   skipped, the array stays all-zero, and Vivado constant-propagates the
#   entire datapath away (we observed ~300 LUTs / 264 FFs total — i.e. the
#   network was synthesised out of existence). Same applies to the sigmoid
#   and ELU LUTs, which are pure ROM.
#
#   So this script BAKES the trained Subject-A-session-3 weights into the
#   bitstream as ROM. Retraining requires re-running synthesis with new
#   .hex files.
#
#   `rtl/weights/weight_bank.sv` + `axi_lite_weight_loader.sv` exist as
#   scaffolding for future runtime-reload, but the NN modules don't
#   instantiate them yet — that's a follow-up refactor, not a synth fix.
# =============================================================================

if {[llength $argv] != 1} {
    puts "Usage: vivado -mode batch -source synth/synth_fit.tcl -tclargs <part>"
    exit 1
}

set part [lindex $argv 0]
set top  db_atcnet_axi
set repo [file normalize [pwd]]
set out  $repo/build/synth_$part
file mkdir $out

create_project -in_memory -part $part

# Add all RTL.
set rtl_files [list \
    rtl/util/serial_divider.sv \
    rtl/conv/conv2d_temporal.sv \
    rtl/conv/depthwise_spatial.sv \
    rtl/conv/elu.sv \
    rtl/conv/avg_pool_time.sv \
    rtl/conv/conv1d_temporal.sv \
    rtl/conv/conv1d_temporal_tm.sv \
    rtl/conv/sat_add.sv \
    rtl/conv/tcfn.sv \
    rtl/conv/branch_pipeline.sv \
    rtl/attention/eca_attention.sv \
    rtl/attention/gap_accumulator.sv \
    rtl/attention/gate_apply.sv \
    rtl/attention/cbam_channel_attn.sv \
    rtl/attention/cbam_spatial_attn.sv \
    rtl/attention/eca1_pipeline.sv \
    rtl/classifier/dense_classifier.sv \
    rtl/classifier/output_head.sv \
    rtl/window/gumbel_channel_adapter.sv \
    rtl/weights/axi_lite_weight_loader.sv \
    rtl/weights/weight_bank.sv \
    rtl/db_atcnet_window_pipeline.sv \
    rtl/db_atcnet_post_eca1.sv \
    rtl/db_atcnet_conv2d_eca1.sv \
    rtl/db_atcnet_top.sv \
    rtl/db_atcnet_axi.sv \
]
foreach f $rtl_files { add_files -norecurse $f }

# Pin file paths to absolute so $readmemh works during synthesis.
set_property file_type SystemVerilog [get_files *.sv]

# ---------------------------------------------------------------------------
# Register every weight / bias / LUT .hex as a synthesis source so Vivado
# resolves the $readmemh paths and tracks them as project dependencies.
# Paths are kept relative to $repo (the project's working dir) so the
# generic bindings below — which are also relative — line up.
# ---------------------------------------------------------------------------
set mem_files [list \
    data/lut/sigmoid_q88.hex \
    data/lut/elu_q88.hex \
    data/golden_q88/stage_conv2d_weights.hex \
    data/golden_q88/stage_conv2d_bias.hex \
    data/golden_q88/stage_eca1_weights.hex \
    data/golden_q88/stage_branchA_dwise_weights.hex \
    data/golden_q88/stage_branchA_dwise_bias.hex \
    data/golden_q88/stage_branchA_sep_weights.hex \
    data/golden_q88/stage_branchA_sep_bias.hex \
    data/golden_q88/stage_branchB_dwise_weights.hex \
    data/golden_q88/stage_branchB_dwise_bias.hex \
    data/golden_q88/stage_branchB_sep_weights.hex \
    data/golden_q88/stage_branchB_sep_bias.hex \
    data/golden_q88/stage_eca2_weights.hex \
    data/golden_q88/all_chan_d1w.hex \
    data/golden_q88/all_chan_d1b.hex \
    data/golden_q88/all_chan_d2w.hex \
    data/golden_q88/all_chan_d2b.hex \
    data/golden_q88/all_spat_conv_w.hex \
    data/golden_q88/all_tcfn_w0.hex \
    data/golden_q88/all_tcfn_b0.hex \
    data/golden_q88/all_tcfn_w1.hex \
    data/golden_q88/all_tcfn_b1.hex \
    data/golden_q88/all_tcfn_w2.hex \
    data/golden_q88/all_tcfn_b2.hex \
    data/golden_q88/all_tcfn_w3.hex \
    data/golden_q88/all_tcfn_b3.hex \
    data/golden_q88/all_cls_w.hex \
    data/golden_q88/all_cls_b.hex \
]
foreach f $mem_files {
    if {![file exists $repo/$f]} {
        puts "ERROR: missing weight/LUT file: $repo/$f"
        puts "       Regenerate with scripts/h5_to_mem.py + scripts/golden_model.py"
        exit 1
    }
    add_files -norecurse $repo/$f
    set_property file_type {Memory Initialization Files} [get_files $repo/$f]
}

# Pull in the XDC constraints (clock + I/O delays + RAM_STYLE hints).
add_files -fileset constrs_1 -norecurse constraints/db_atcnet_axi.xdc
set_property USED_IN {synthesis implementation out_of_context} \
    [get_files constraints/db_atcnet_axi.xdc]

# ---------------------------------------------------------------------------
# Synthesize. -mode out_of_context skips I/O buffer insertion so we don't
# need to constrain every clk/reset port to a physical pin.
#
# -generic bindings push trained weight/LUT paths into the db_atcnet_axi
# `parameter string` ports. They propagate down through db_atcnet_top to
# every NN module's $readmemh call.
# ---------------------------------------------------------------------------
synth_design -top $top -part $part -include_dirs $repo -mode out_of_context \
    -generic CONV2D_W_FILE=data/golden_q88/stage_conv2d_weights.hex \
    -generic CONV2D_B_FILE=data/golden_q88/stage_conv2d_bias.hex \
    -generic ECA1_W_FILE=data/golden_q88/stage_eca1_weights.hex \
    -generic SIG_LUT_FILE=data/lut/sigmoid_q88.hex \
    -generic ELU_LUT_FILE=data/lut/elu_q88.hex \
    -generic BRANCHA_DWISE_W_FILE=data/golden_q88/stage_branchA_dwise_weights.hex \
    -generic BRANCHA_DWISE_B_FILE=data/golden_q88/stage_branchA_dwise_bias.hex \
    -generic BRANCHA_SEP_W_FILE=data/golden_q88/stage_branchA_sep_weights.hex \
    -generic BRANCHA_SEP_B_FILE=data/golden_q88/stage_branchA_sep_bias.hex \
    -generic BRANCHB_DWISE_W_FILE=data/golden_q88/stage_branchB_dwise_weights.hex \
    -generic BRANCHB_DWISE_B_FILE=data/golden_q88/stage_branchB_dwise_bias.hex \
    -generic BRANCHB_SEP_W_FILE=data/golden_q88/stage_branchB_sep_weights.hex \
    -generic BRANCHB_SEP_B_FILE=data/golden_q88/stage_branchB_sep_bias.hex \
    -generic ECA2_W_FILE=data/golden_q88/stage_eca2_weights.hex \
    -generic CHAN_D1_W_FILE=data/golden_q88/all_chan_d1w.hex \
    -generic CHAN_D1_B_FILE=data/golden_q88/all_chan_d1b.hex \
    -generic CHAN_D2_W_FILE=data/golden_q88/all_chan_d2w.hex \
    -generic CHAN_D2_B_FILE=data/golden_q88/all_chan_d2b.hex \
    -generic SPAT_CONV_W_FILE=data/golden_q88/all_spat_conv_w.hex \
    -generic TCFN_W0_FILE=data/golden_q88/all_tcfn_w0.hex \
    -generic TCFN_B0_FILE=data/golden_q88/all_tcfn_b0.hex \
    -generic TCFN_W1_FILE=data/golden_q88/all_tcfn_w1.hex \
    -generic TCFN_B1_FILE=data/golden_q88/all_tcfn_b1.hex \
    -generic TCFN_W2_FILE=data/golden_q88/all_tcfn_w2.hex \
    -generic TCFN_B2_FILE=data/golden_q88/all_tcfn_b2.hex \
    -generic TCFN_W3_FILE=data/golden_q88/all_tcfn_w3.hex \
    -generic TCFN_B3_FILE=data/golden_q88/all_tcfn_b3.hex \
    -generic CLS_W_FILE=data/golden_q88/all_cls_w.hex \
    -generic CLS_B_FILE=data/golden_q88/all_cls_b.hex

# Reports.
report_utilization                 -file $out/utilization.txt
report_utilization -hierarchical   -file $out/utilization_hier.txt
report_timing_summary -delay_type min_max -report_unconstrained \
                                   -file $out/timing.txt

puts "==== Synthesis fit summary ($part) ===="
puts [exec head -80 $out/utilization.txt]
puts "..."
puts "(full reports in $out/)"

close_project
exit
