# =============================================================================
# retarget_zcu106.tcl — Clone the existing ZCU104 project to a new ZCU106
# project, retarget the device part and board, and swap the UART XDC.
#
# Usage:
#   1. Open Vivado.
#   2. File -> Open Project -> build/db_atcnet_zcu104/db_atcnet_zcu104.xpr
#   3. In the Tcl Console: source synth/retarget_zcu106.tcl
#
# This script:
#   - Saves the open project as build/db_atcnet_zcu106/db_atcnet_zcu106.xpr
#   - Switches the part to xczu9eg-ffvb1156-2-e
#   - Switches the board_part to xilinx.com:zcu106:part0:2.6
#   - Removes constraints/uart1_emio_pmod.xdc (ZCU104 pins)
#   - Adds constraints/uart1_emio_zcu106.xdc (ZCU106 pins)
#
# It does NOT modify the BD's Zynq UltraScale+ PS preset — that has to be
# done manually after this script:
#   1. Open the BD.
#   2. Delete the existing zynq_ultra_ps_e_0 cell.
#   3. Re-add a Zynq UltraScale+ MPSoC IP.
#   4. Click "Run Block Automation" — Vivado will apply the ZCU106 preset
#      automatically because the board_part is now ZCU106.
#   5. Re-wire pl_clk0 -> aclks, pl_resetn0 -> reset block,
#      S_AXI_HP0_FPD <- DMA MM2S via axi_smc_1, M_AXI_HPM0_FPD -> axi_smc.
#   6. Validate BD, regenerate output products.
# =============================================================================

set REPO_ROOT "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main"
set NEW_PROJECT_DIR "$REPO_ROOT/build/db_atcnet_zcu106"
set NEW_PROJECT_NAME "db_atcnet_zcu106"

# ZCU106 production silicon = xczu7ev-ffvc1156-2-e (SAME chip as ZCU104).
# Only the board file changes; the part stays the same.
set NEW_BOARD "xilinx.com:zcu106:part0:2.6"

set OLD_UART_XDC "$REPO_ROOT/constraints/uart1_emio_pmod.xdc"
set NEW_UART_XDC "$REPO_ROOT/constraints/uart1_emio_zcu106.xdc"

puts "==[clock format [clock seconds]] retargeting to ZCU106 / XCZU9EG=="

# Sanity check that a project is open.
if {[catch {current_project}]} {
    error "No project is currently open. Open the ZCU104 .xpr first."
}
set old_proj [current_project]
puts "  source project: $old_proj"

# 1. Save the project under the new name.
puts ">> saving project as $NEW_PROJECT_DIR/$NEW_PROJECT_NAME.xpr ..."
file mkdir $NEW_PROJECT_DIR
save_project_as -force $NEW_PROJECT_NAME $NEW_PROJECT_DIR

# 2. Retarget board only (part stays xczu7ev-ffvc1156-2-e for ZCU106 production).
puts ">> switching board_part to $NEW_BOARD ..."
set_property board_part $NEW_BOARD [current_project]

# 3. Swap the UART XDC: remove the ZCU104 one, add the ZCU106 one.
foreach f [get_files -filter "FILE_TYPE == XDC"] {
    if {[string match "*uart1_emio_pmod.xdc" [get_property FILE_NAME $f]]} {
        puts ">> removing old XDC: $f"
        remove_files $f
    }
}
puts ">> adding new XDC: $NEW_UART_XDC"
add_files -fileset constrs_1 -norecurse $NEW_UART_XDC
set_property USED_IN {synthesis implementation} \
            [get_files [file tail $NEW_UART_XDC]]

puts ""
puts "==[clock format [clock seconds]] retarget done=="
puts ""
puts "NEXT STEPS (do these in the GUI):"
puts "  1. Open the BD (Sources -> db_atcnet_bd -> double-click)."
puts "  2. Delete the zynq_ultra_ps_e_0 cell (right-click -> Delete)."
puts "  3. Drag a new Zynq UltraScale+ MPSoC IP onto the canvas."
puts "  4. Click the green banner: 'Run Block Automation'."
puts "     -> Vivado will apply the ZCU106 preset automatically."
puts "  5. Open the new Zynq cell, set:"
puts "       PSU-PL Configuration -> Fabric Clocks -> pl_clk0 = 60 MHz"
puts "       I/O Configuration -> Low Speed -> UART1 -> Enable -> EMIO"
puts "       PS-PL Interfaces -> S_AXI_HP0_FPD -> Enable, 128-bit"
puts "       PS-PL Interfaces -> M_AXI_HPM0_FPD -> Enable, 32-bit"
puts "       Interrupts -> Enable pl_ps_irq0\[0:0\]"
puts "  6. Re-wire the new Zynq cell to axi_dma_0, axi_smc, axi_smc_1,"
puts "     rst_ps8_0_60M and u_atcnet."
puts "  7. Validate BD (F6). Should report 0 errors."
puts "  8. Right-click db_atcnet_bd in Sources -> Generate Output Products."
puts "  9. Right-click db_atcnet_bd -> Create HDL Wrapper."
puts " 10. Run synthesis, implementation, write_bitstream."
puts " 11. After write_bitstream completes:"
puts "       write_hw_platform -fixed -force -include_bit \\"
puts "         -file $REPO_ROOT/build/db_atcnet_zcu106.xsa"
