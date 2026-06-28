# Direct xsdb JTAG load for ZCU106 — bitstream + psu_init + PL reset release + ELF
#
# CRITICAL: Calls psu_ps_pl_reset_config to release the EMIO PL fabric reset.
# Without it, pl_resetn0 stays LOW, proc_sys_reset never deasserts
# peripheral_aresetn, the DMA stays in reset, and any AXI access to PL slaves
# (0xa0000000 DMA, 0x4_0000_0000 db_atcnet) hangs forever, wedging the DAP with
# sticky STKERR (recoverable only by physical PS_POR_B).
puts "=========================================="
puts " ZCU106 JTAG load — db_atcnet"
puts "=========================================="

set BIT_FILE  "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/build/db_atcnet_zcu106_platform/hw/sdt/db_atcnet_zcu106.bit"
set PSU_INIT  "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/build/db_atcnet_zcu106_platform/hw/sdt/psu_init.tcl"
set APP_ELF   "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/build/db_atcnet_ps_app_zcu106/build/db_atcnet_ps_app_zcu106.elf"

puts "STEP 1: connect"
connect

puts "STEP 2: psu_init"
source $PSU_INIT
targets -set -filter {name =~ "PSU"}
rst -system
after 1500
psu_init

puts "STEP 3: program FPGA"
fpga -file $BIT_FILE
after 500

puts "STEP 4: psu_post_config"
psu_post_config

puts "STEP 5: psu_ps_pl_reset_config (release EMIO PL reset)"
psu_ps_pl_reset_config

puts "STEP 6: psu_ps_pl_isolation_removal"
psu_ps_pl_isolation_removal
after 300

puts "STEP 7: sanity probe DMA @ 0xa0000000"
if {[catch {mrd -force -value 0xa0000000} val]} {
    puts "    DMA READ FAILED: $val"
    error "PL unreachable — bailing out"
}
puts "    DMA DMACR = [format 0x%08x $val]"

puts "STEP 8: target Cortex-A53 #0, processor reset, load ELF"
targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor
dow $APP_ELF

puts "STEP 9: run"
con
after 500
puts "STEP 10: A53 #0 running. Banner on /dev/ttyUSB1, test on /dev/ttyUSB3."
disconnect
exit
