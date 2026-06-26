# =============================================================================
# impl_fit.tcl  —  Full Vivado implementation (synth → opt → place → phys_opt
#                  → route) for db_atcnet_axi on ZCU104, after synth has been
#                  proven to close 100 MHz with WNS = +0.253 ns.
#
# Reuses the same -generic *_FILE bindings and constraints as synth_fit.tcl so
# the impl flow sees exactly the design that synth verified.
#
# Outputs (in build/impl_<part>/):
#   utilization_impl.txt        — post-route LUT/FF/DSP/BRAM totals
#   utilization_impl_hier.txt   — hierarchical breakdown
#   timing_impl.txt             — post-route timing summary (the real WNS)
#   drc_impl.txt                — DRC checks after routing
#   power_impl.txt              — power estimate
#   post_route.dcp              — Vivado checkpoint (open this if WNS fails)
#
# Usage:  vivado -mode batch -source synth/impl_fit.tcl -tclargs <part>
#         e.g. xczu7ev-ffvc1156-2-e
# =============================================================================

if {[llength $argv] != 1} {
    puts "Usage: vivado -mode batch -source synth/impl_fit.tcl -tclargs <part>"
    exit 1
}

set part [lindex $argv 0]
set top  db_atcnet_axi
set repo [file normalize [pwd]]
set out  $repo/build/impl_$part
file mkdir $out

create_project -in_memory -part $part

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
set_property file_type SystemVerilog [get_files *.sv]

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
        exit 1
    }
    add_files -norecurse $repo/$f
    set_property file_type {Memory Initialization Files} [get_files $repo/$f]
}

add_files -fileset constrs_1 -norecurse constraints/db_atcnet_axi.xdc
set_property USED_IN {synthesis implementation out_of_context} \
    [get_files constraints/db_atcnet_axi.xdc]

puts "==== Step 1/5: synth_design ===="
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
write_checkpoint -force $out/post_synth.dcp

puts "==== Step 2/5: opt_design ===="
opt_design
write_checkpoint -force $out/post_opt.dcp

puts "==== Step 3/5: place_design ===="
place_design
write_checkpoint -force $out/post_place.dcp

puts "==== Step 4/5: phys_opt_design ===="
phys_opt_design
write_checkpoint -force $out/post_phys_opt.dcp

puts "==== Step 5/5: route_design ===="
route_design
write_checkpoint -force $out/post_route.dcp

puts "==== Final reports ===="
report_utilization                 -file $out/utilization_impl.txt
report_utilization -hierarchical   -file $out/utilization_impl_hier.txt
report_timing_summary -delay_type min_max -report_unconstrained \
                                   -file $out/timing_impl.txt
report_drc                         -file $out/drc_impl.txt
report_power                       -file $out/power_impl.txt

puts "==== Done. Post-route WNS: ===="
puts [exec grep -A 2 "WNS(ns)" $out/timing_impl.txt | head -4]

close_project
exit
