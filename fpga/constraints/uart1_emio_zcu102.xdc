# =============================================================================
# uart1_emio_zcu102.xdc — Pin constraints for the EMIO PS UART1 routed out
# through the PL to the ZCU102's CP2108 USB-UART (channel 2).
#
# Replaces uart1_emio_pmod.xdc which targeted the ZCU104.
#
# Source: Xilinx ZCU102 board file (data/xhub/boards/.../zcu102/3.4/part0_pins.xml):
#   F13 = uart2_PL_TX  (FPGA -> CP2108)
#   E13 = uart2_PL_RX  (CP2108 -> FPGA)
#
# After bitstream load, this UART appears on the host as one of the
# /dev/ttyUSB* devices exposed by the on-board CP2108 quad-USB-UART bridge.
# =============================================================================

set_property PACKAGE_PIN F13 [get_ports UART_1_0_txd]
set_property PACKAGE_PIN E13 [get_ports UART_1_0_rxd]

# I/O bank voltage on the ZCU102 is 1.8 V for these pins.
set_property IOSTANDARD LVCMOS18 [get_ports UART_1_0_txd]
set_property IOSTANDARD LVCMOS18 [get_ports UART_1_0_rxd]

set_property DRIVE 8           [get_ports UART_1_0_txd]
set_property SLEW SLOW         [get_ports UART_1_0_txd]
