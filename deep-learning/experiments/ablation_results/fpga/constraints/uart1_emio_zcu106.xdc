# =============================================================================
# uart1_emio_zcu106.xdc — Pin constraints for the EMIO PS UART1 routed out
# through the PL to the ZCU106's CP2108 USB-UART (channel 2).
#
# Replaces uart1_emio_pmod.xdc which targeted the ZCU104. On the ZCU106 the
# uart2_PL pins are at AL17 (TX) and AH17 (RX), driven at LVCMOS12 (the
# I/O bank voltage on the ZCU106 is 1.2 V for this region — different from
# the ZCU104's 1.8 V bank).
#
# After bitstream load, this UART appears on the host as one of the
# /dev/ttyUSB* devices exposed by the on-board CP2108 quad-USB-UART bridge
# (typically /dev/ttyUSB1 — the first is the PS debug UART0 on MIO).
#
# Source: Xilinx ZCU106 board file (data/xhub/boards/.../zcu106/2.6/part0_pins.xml).
# =============================================================================

# PL-side UART2 pins on ZCU106
# AL17 = uart2_PL_TX (FPGA -> CP2108)
# AH17 = uart2_PL_RX (CP2108 -> FPGA)
set_property PACKAGE_PIN AL17 [get_ports UART_1_0_txd]
set_property PACKAGE_PIN AH17 [get_ports UART_1_0_rxd]

# I/O bank voltage on the ZCU106 is 1.2 V for these pins.
set_property IOSTANDARD LVCMOS12 [get_ports UART_1_0_txd]
set_property IOSTANDARD LVCMOS12 [get_ports UART_1_0_rxd]

# Drive strength for LVCMOS12 caps at 4 mA. Slew kept SLOW for clean edges
# at 460 800 baud across a few cm of trace into the CP2108.
set_property DRIVE 4    [get_ports UART_1_0_txd]
set_property SLEW SLOW  [get_ports UART_1_0_txd]
