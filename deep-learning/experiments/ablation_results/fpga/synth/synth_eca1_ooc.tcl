# =============================================================================
# synth_eca1_ooc.tcl — fast out-of-context synth of just eca1_pipeline so we
# can verify the buf_mem refactor inferred as BRAM (and FF count collapsed)
# without paying for the full 3-hour db_atcnet_axi synth.
#
# Expected after refactor: FFs drop from ~768k to ~few hundred (only FSM,
# gate_latched, y_out, buf_rd_packed survive); BRAM tile count climbs by
# the buf_mem size (~22 RAMB36 for 3000×16 × 16-bit).
#
# Usage:  vivado -mode batch -source synth/synth_eca1_ooc.tcl
# =============================================================================

set part xczu7ev-ffvc1156-2-e
set repo [file normalize [pwd]]
set out  $repo/build/synth_eca1_ooc
file mkdir $out

create_project -in_memory -part $part

add_files -norecurse [list \
    rtl/attention/gap_accumulator.sv \
    rtl/attention/eca_attention.sv \
    rtl/attention/gate_apply.sv \
    rtl/attention/eca1_pipeline.sv \
]
set_property file_type SystemVerilog [get_files *.sv]

# Register weight/LUT files so $readmemh resolves.
foreach f [list \
    data/golden_q88/stage_eca1_weights.hex \
    data/lut/sigmoid_q88.hex \
] {
    add_files -norecurse $repo/$f
    set_property file_type {Memory Initialization Files} [get_files $repo/$f]
}

synth_design -top eca1_pipeline -part $part -include_dirs $repo -mode out_of_context \
    -generic WEIGHTS_FILE=data/golden_q88/stage_eca1_weights.hex \
    -generic LUT_FILE=data/lut/sigmoid_q88.hex

report_utilization                 -file $out/utilization.txt
report_utilization -hierarchical   -file $out/utilization_hier.txt
puts "==== eca1_pipeline OOC utilization ===="
puts [exec head -60 $out/utilization.txt]
close_project
exit
