# =============================================================================
# apply_zcu106_preset.tcl — Apply the ZCU106 board preset to the existing
# Zynq UltraScale+ MPSoC IP, then re-apply our project-specific overrides:
#   - pl_clk0 = 60 MHz
#   - UART1 enabled, routed to EMIO
#   - S_AXI_HP0_FPD enabled (128-bit, for AXI-DMA MM2S)
#   - M_AXI_HPM0_FPD enabled (32-bit, for AXI-Lite control)
#   - pl_ps_irq0 enabled (1-bit; reserved for DMA done IRQ)
#
# Then validate the BD, regenerate output products, and rebuild the HDL
# wrapper. This is the Tcl-only equivalent of the GUI walkthrough in
# PHASE_B_BRINGUP.md.
#
# Usage:
#   1. ZCU106 project must be open AND have board_part = xilinx.com:zcu106:part0:2.6
#      (already done by retarget_zcu106.tcl).
#   2. In Tcl Console:  source synth/apply_zcu106_preset.tcl
# =============================================================================

set BD_NAME       "db_atcnet_bd"
set ZYNQ_CELL     "zynq_ultra_ps_e_0"
set PL_CLK0_FREQ  60
set REPO_ROOT     "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main"

puts "==[clock format [clock seconds]] applying ZCU106 preset to $ZYNQ_CELL =="

# --- 0. Sanity: project open + board is ZCU106 ---------------------------
if {[catch {current_project}]} { error "no project open" }
set bp [get_property board_part [current_project]]
if {![string match "xilinx.com:zcu106:*" $bp]} {
    error "board_part is '$bp' -- expected xilinx.com:zcu106:*. Run retarget_zcu106.tcl first."
}
puts ">> board_part = $bp"

# --- 1. Open the BD ------------------------------------------------------
if {[catch {current_bd_design}]} {
    puts ">> opening BD $BD_NAME ..."
    open_bd_design [get_files "$BD_NAME.bd"]
} else {
    puts ">> BD already open: [current_bd_design]"
}

# --- 2. Apply ZCU106 board preset to the existing Zynq cell --------------
# This overwrites the Zynq PS configuration with the ZCU106 defaults
# (MIO map, DDR config, clock sources). Existing PL-side connections to
# the Zynq cell are preserved where the pin names still match; any that
# don't match will dangle and be caught by validate_bd_design below.
puts ">> applying ZCU106 board preset (this may take ~30 s) ..."
apply_bd_automation \
    -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config [list apply_board_preset "1"] \
    [get_bd_cells $ZYNQ_CELL]

# --- 3. Re-apply our project-specific PS configuration overrides ---------
puts ">> applying project-specific PS overrides ..."

set zynq [get_bd_cells $ZYNQ_CELL]

# Enable fabric clock 0 at 60 MHz
set_property -dict [list \
    CONFIG.PSU__FPGA_PL0_ENABLE                  {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ   $PL_CLK0_FREQ \
] $zynq

# Enable UART1 on EMIO (UART0 stays on MIO for debug printf)
set_property -dict [list \
    CONFIG.PSU__UART1__PERIPHERAL__ENABLE        {1} \
    CONFIG.PSU__UART1__PERIPHERAL__IO            {EMIO} \
] $zynq

# Enable S_AXI_HP0_FPD (128-bit, used by AXI-DMA MM2S)
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP2                   {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH              {128} \
] $zynq
# Some Vivado versions key this on USE__S_AXI_HP0 instead — set both for safety
catch { set_property CONFIG.PSU__USE__S_AXI_HP0 {1} $zynq }

# Enable M_AXI_HPM0_FPD (32-bit, used for AXI-Lite control)
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0                   {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH              {32} \
] $zynq

# Disable HPM1 (we don't use it; leaving it on triggers an unused-clock error)
catch { set_property CONFIG.PSU__USE__M_AXI_GP1 {0} $zynq }

# Enable pl_ps_irq0[0:0]
set_property -dict [list \
    CONFIG.PSU__USE__IRQ0                        {1} \
] $zynq

# --- 4. Validate BD ------------------------------------------------------
puts ">> validating BD ..."
set v [validate_bd_design]
puts ">> validate result: $v"

# --- 5. Save BD ----------------------------------------------------------
save_bd_design

# --- 6. Regenerate output products + HDL wrapper -------------------------
puts ">> regenerating output products (global) ..."
generate_target -force all [get_files "$BD_NAME.bd"]

puts ">> updating compile order ..."
update_compile_order -fileset sources_1

puts ">> recreating HDL wrapper ..."
set bd_file [get_files "$BD_NAME.bd"]
make_wrapper -files $bd_file -top -import -force

puts ""
puts "==[clock format [clock seconds]] BD preset + overrides applied =="
puts ""
puts "NEXT STEPS:"
puts "  1. Open the BD in the GUI and visually inspect for dangling pins."
puts "  2. If validate reported errors, fix them by re-wiring."
puts "  3. Otherwise run:"
puts "       reset_runs synth_1"
puts "       launch_runs impl_1 -to_step write_bitstream -jobs 4"
puts "  4. Then export the hardware platform:"
puts "       write_hw_platform -fixed -force -include_bit \\"
puts "         -file $REPO_ROOT/build/db_atcnet_zcu106.xsa"
