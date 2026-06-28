# =============================================================================
# write_bitstream.tcl  —  Open the post-route checkpoint and emit the bitstream.
#
# Usage:  vivado -mode batch -source synth/write_bitstream.tcl -tclargs <part>
# =============================================================================

if {[llength $argv] != 1} {
    puts "Usage: vivado -mode batch -source synth/write_bitstream.tcl -tclargs <part>"
    exit 1
}

set part [lindex $argv 0]
set repo [file normalize [pwd]]
set out  $repo/build/impl_$part

if {![file exists $out/post_route.dcp]} {
    puts "ERROR: missing $out/post_route.dcp — run synth/impl_fit.tcl first."
    exit 1
}

puts "==== Opening post-route checkpoint ===="
open_checkpoint $out/post_route.dcp

puts "==== Writing bitstream ===="
# -bin_file also emits .bin for PetaLinux / FPGA Manager use.
write_bitstream -force -bin_file $out/db_atcnet_axi.bit

# Also write the hardware-handoff XSA for Vivado/Vitis software workflows.
puts "==== Writing hardware handoff (.xsa) ===="
write_hw_platform -fixed -force -file $out/db_atcnet_axi.xsa

puts "==== Done ===="
puts "Bitstream:           $out/db_atcnet_axi.bit"
puts "Binary (FPGA Mgr):   $out/db_atcnet_axi.bin"
puts "Hardware handoff:    $out/db_atcnet_axi.xsa"

close_project
exit
