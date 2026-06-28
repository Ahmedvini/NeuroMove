# =============================================================================
# build_zcu106_clean.tcl — Build the ZCU106 Vivado project from scratch.
#
# Run mode 1 (batch):
#   cd <repo root>
#   vivado -mode batch -source synth/build_zcu106_clean.tcl
#
# Run mode 2 (interactive Tcl Console of an EMPTY Vivado session):
#   source /full/path/to/synth/build_zcu106_clean.tcl
#
# What it does:
#   1. Creates a fresh Vivado project at build/db_atcnet_zcu106_clean/
#   2. Target: ZCU106 production = part xczu7ev-ffvc1156-2-e + board zcu106 v2.6
#   3. Adds every .sv / .v / .svh under rtl/ and the SDV include dir
#   4. Adds constraints/uart1_emio_zcu106.xdc
#   5. Builds the IPI block design db_atcnet_bd from scratch:
#      - Zynq UltraScale+ MPSoC (ZCU106 board preset, then project overrides)
#      - AXI Direct Memory Access (MM2S only, 32-bit, no SG)
#      - 2x AXI SmartConnect (control plane + data plane)
#      - Processor System Reset (synchronous to pl_clk0 at 60 MHz)
#      - db_atcnet_axi (module reference, top = db_atcnet_axi_v)
#   6. Connects every AXI / clock / reset / IRQ as in the ZCU104 design
#   7. Exposes the PS UART1 EMIO interface as top-level ports UART_1_0_*
#   8. Maps db_atcnet_axi to 0x4_0000_0000 in the PS address space
#   9. Validates the BD, generates output products, builds the HDL wrapper
#  10. Optionally launches synth + impl + bitstream + .xsa export
#      (controlled by the LAUNCH_BUILD flag at the top — set to 0 to stop
#       after the BD is built so you can inspect it in the GUI first).
# =============================================================================

# ----- Knobs ---------------------------------------------------------------

set REPO_ROOT     "/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main"
set PROJECT_NAME  "db_atcnet_zcu106_clean"
set PROJECT_DIR   "$REPO_ROOT/build/$PROJECT_NAME"
set PART          "xczu7ev-ffvc1156-2-e"
set BOARD_PART    "xilinx.com:zcu106:part0:2.6"
set BD_NAME       "db_atcnet_bd"
set PL_CLK0_FREQ  60
set LAUNCH_BUILD  0    ;# 1 = run synth+impl+bitstream+xsa export at end; 0 = stop after BD build

# ----- 1. Fresh project ----------------------------------------------------

puts "==[clock format [clock seconds]] creating fresh project=="

# If a previous half-built copy exists, blow it away.
if {[file exists $PROJECT_DIR]} {
    puts ">> removing stale $PROJECT_DIR"
    file delete -force $PROJECT_DIR
}

create_project $PROJECT_NAME $PROJECT_DIR -part $PART -force
set_property board_part $BOARD_PART [current_project]
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# ----- 2. Add RTL ----------------------------------------------------------

puts ">> adding RTL from $REPO_ROOT/rtl (recursive) ..."
# Vivado handles recursion via -fileset; we just point at the rtl/ root.
add_files -fileset sources_1 "$REPO_ROOT/rtl"
puts ">> total RTL files in sources_1: \
    [llength [get_files -filter {FILE_TYPE == \"SystemVerilog\" \
                                  || FILE_TYPE == \"Verilog\" \
                                  || FILE_TYPE == \"Verilog Header\"}]]"

# Make sure the include directory for db_atcnet_params.svh is set
set_property include_dirs "$REPO_ROOT/rtl" [get_filesets sources_1]

# Set file types
foreach f [get_files -filter {FILE_TYPE == ""}] {
    if {[string match "*.svh" $f]} {
        set_property file_type "Verilog Header" [get_files $f]
    } elseif {[string match "*.sv" $f]} {
        set_property file_type "SystemVerilog" [get_files $f]
    }
}

update_compile_order -fileset sources_1

# ----- 3. Add constraints --------------------------------------------------

puts ">> adding constraints/uart1_emio_zcu106.xdc ..."
add_files -fileset constrs_1 -norecurse "$REPO_ROOT/constraints/uart1_emio_zcu106.xdc"

# ----- 4. Create the IPI block design --------------------------------------

puts ">> creating BD $BD_NAME ..."
create_bd_design $BD_NAME

# ---- 4a. Zynq UltraScale+ MPSoC -------------------------------------------
puts ">>   adding zynq_ultra_ps_e_0 (ZCU106 preset)"
set zynq [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e \
              zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config [list apply_board_preset "1"] $zynq

# Project-specific PS configuration (overrides on top of the ZCU106 preset)
set_property -dict [list \
    CONFIG.PSU__FPGA_PL0_ENABLE                  {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ   $PL_CLK0_FREQ \
    CONFIG.PSU__UART1__PERIPHERAL__ENABLE        {1} \
    CONFIG.PSU__UART1__PERIPHERAL__IO            {EMIO} \
    CONFIG.PSU__USE__M_AXI_GP0                   {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH              {32} \
    CONFIG.PSU__USE__M_AXI_GP1                   {0} \
    CONFIG.PSU__USE__S_AXI_GP2                   {1} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH              {128} \
    CONFIG.PSU__USE__IRQ0                        {1} \
] $zynq

# ---- 4b. AXI Direct Memory Access -----------------------------------------
puts ">>   adding axi_dma_0"
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg               {0}  \
    CONFIG.c_sg_include_stscntrl_strm {0}  \
    CONFIG.c_include_mm2s             {1}  \
    CONFIG.c_include_s2mm             {0}  \
    CONFIG.c_mm2s_burst_size          {16} \
    CONFIG.c_m_axi_mm2s_data_width    {32} \
    CONFIG.c_m_axis_mm2s_tdata_width  {32} \
] $dma

# ---- 4c. AXI SmartConnect (control plane) ---------------------------------
puts ">>   adding axi_smc (control plane)"
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $smc

# ---- 4d. AXI SmartConnect (data plane) ------------------------------------
puts ">>   adding axi_smc_1 (data plane)"
set smc1 [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc_1]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc1

# ---- 4e. Processor System Reset -------------------------------------------
puts ">>   adding rst_ps8_0_60M"
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset \
             rst_ps8_0_60M]

# ---- 4f. db_atcnet_axi (module reference) ---------------------------------
# This is a module-ref cell pointing at our Verilog wrapper that hardcodes
# all 29 *_FILE generic paths. Vivado IPI does NOT propagate generics
# through OOC sub-synth of module-reference cells; the wrapper is mandatory.
puts ">>   adding u_atcnet (module reference db_atcnet_axi_v)"
create_bd_cell -type module -reference db_atcnet_axi_v u_atcnet

# ----- 5. Wire everything up -----------------------------------------------

puts ">> connecting AXI interfaces ..."

# Control plane: PS HPM0 → axi_smc → axi_dma_0.S_AXI_LITE + u_atcnet.s_axi
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] \
                    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M01_AXI] \
                    [get_bd_intf_pins u_atcnet/s_axi]

# Data plane: AXI-DMA MM2S → u_atcnet.s_axis (stream) + PS S_AXI_HP0 (memory)
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
                    [get_bd_intf_pins u_atcnet/s_axis]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
                    [get_bd_intf_pins axi_smc_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_1/M00_AXI] \
                    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

puts ">> connecting clocks ..."
# pl_clk0 is the source — do not include it as a sink in the loop.
set src_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
foreach pin [list \
    zynq_ultra_ps_e_0/maxihpm0_fpd_aclk \
    zynq_ultra_ps_e_0/saxihp0_fpd_aclk \
    axi_dma_0/s_axi_lite_aclk \
    axi_dma_0/m_axi_mm2s_aclk \
    axi_smc/aclk \
    axi_smc_1/aclk \
    rst_ps8_0_60M/slowest_sync_clk \
    u_atcnet/clk \
] {
    connect_bd_net $src_clk [get_bd_pins $pin]
}

puts ">> connecting resets ..."
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
               [get_bd_pins rst_ps8_0_60M/ext_reset_in]

foreach pin [list \
    axi_dma_0/axi_resetn \
    axi_smc/aresetn \
    axi_smc_1/aresetn \
] {
    connect_bd_net [get_bd_pins rst_ps8_0_60M/peripheral_aresetn] \
                   [get_bd_pins $pin]
}

# u_atcnet.rst is active-high (per RTL). Use peripheral_reset which is
# the active-high reset from the rst block.
connect_bd_net [get_bd_pins rst_ps8_0_60M/peripheral_reset] \
               [get_bd_pins u_atcnet/rst]

puts ">> connecting AXI-DMA interrupt to PS pl_ps_irq0 ..."
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] \
               [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

# ----- 6. Expose PS UART1 EMIO interface as top-level ports ----------------

puts ">> exposing UART_1 EMIO interface as external ports ..."
make_bd_intf_pins_external [get_bd_intf_pins zynq_ultra_ps_e_0/UART_1]
# After make_bd_intf_pins_external the ports are named UART_1_0_txd/rxd
# automatically (the trailing _0 is appended by Vivado).

# ----- 7. Assign addresses -------------------------------------------------

puts ">> assigning addresses ..."
assign_bd_address

# Override the u_atcnet.s_axi range to 4 GB at 0x4_0000_0000 (project spec)
catch {
    set_property offset 0x0000000400000000 \
        [get_bd_addr_segs zynq_ultra_ps_e_0/Data/SEG_u_atcnet_reg0]
    set_property range  4G \
        [get_bd_addr_segs zynq_ultra_ps_e_0/Data/SEG_u_atcnet_reg0]
}

# ----- 8. Validate, save, generate, wrap -----------------------------------

puts ">> validating BD ..."
set v [validate_bd_design]
puts ">> validate result: $v"

save_bd_design

puts ">> generating output products (global) ..."
generate_target -force all [get_files "$BD_NAME.bd"]
update_compile_order -fileset sources_1

puts ">> building HDL wrapper ..."
make_wrapper -files [get_files "$BD_NAME.bd"] -top -import -force

set_property top "${BD_NAME}_wrapper" [current_fileset]
update_compile_order -fileset sources_1

puts ""
puts "==[clock format [clock seconds]] BD build complete=="
puts ">> Project: $PROJECT_DIR"
puts ">> Top:     ${BD_NAME}_wrapper"

# ----- 9. Optionally launch synth + impl + bitstream + .xsa ----------------

if {$LAUNCH_BUILD} {
    puts ">> launching impl_1 to write_bitstream ..."
    reset_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    puts ">> impl_1 done"
    puts ">> exporting hardware platform ..."
    write_hw_platform -fixed -force -include_bit \
        -file "$REPO_ROOT/build/db_atcnet_zcu106.xsa"
} else {
    puts ""
    puts "LAUNCH_BUILD = 0. To run synth+impl+bitstream:"
    puts "    reset_runs synth_1"
    puts "    launch_runs impl_1 -to_step write_bitstream -jobs 4"
    puts ""
    puts "Then export the .xsa:"
    puts "    write_hw_platform -fixed -force -include_bit \\"
    puts "        -file $REPO_ROOT/build/db_atcnet_zcu106.xsa"
}
