# Quick OOC synth of avg_pool_time with POOL=7 to confirm use_dsp=no.
# Expected: DSPs = 0 (was 32 per instance in the full design).
set part xczu7ev-ffvc1156-2-e
set repo [file normalize [pwd]]
set out  $repo/build/synth_pool_ooc
file mkdir $out

create_project -in_memory -part $part
add_files -norecurse rtl/conv/avg_pool_time.sv
set_property file_type SystemVerilog [get_files *.sv]

synth_design -top avg_pool_time -part $part -mode out_of_context \
    -generic POOL=7 -generic INV_POOL=2396745 -generic NUM_CH=32

report_utilization                 -file $out/utilization.txt
puts "==== avg_pool_time(POOL=7) OOC utilization ===="
puts [exec head -55 $out/utilization.txt]
close_project
exit
