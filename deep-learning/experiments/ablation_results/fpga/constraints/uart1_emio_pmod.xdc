# =============================================================================
# uart1_emio_pmod.xdc — Pin constraints for the EMIO PS UART1 routed out
# through the PL to the ZCU104's CP2108 USB-UART (channel 2).
#
# After bitstream load, this UART appears on the host as a second /dev/ttyUSB*
# device (typically /dev/ttyUSB1 — the first is the PS debug UART). The
# Zynq PS sends/receives via UART1, which the PL wires to pins C19/A20, which
# the on-board CP2108 mirrors to USB virtual serial.
#
# For ESP32 wiring you have two options:
#   (a) Plug a second USB cable from the ESP32 to your laptop and run a small
#       Python bridge program forwarding bytes between the two /dev/ttyUSB*
#       devices. No soldering required.
#   (b) Tap the C19 / A20 nets directly if your ZCU104 revision exposes them
#       on a header / test point (check schematic; not all revisions do).
# =============================================================================

# PL-side UART2 pins on ZCU104 — defined in the official board file.
# C19 = uart2_PL_TX (FPGA -> CP2108)
# A20 = uart2_PL_RX (CP2108 -> FPGA)
set_property PACKAGE_PIN C19 [get_ports UART_1_0_txd]
set_property PACKAGE_PIN A20 [get_ports UART_1_0_rxd]

# The bank these pins live in is configured for 1.8V LVCMOS on ZCU104.
set_property IOSTANDARD LVCMOS18 [get_ports UART_1_0_txd]
set_property IOSTANDARD LVCMOS18 [get_ports UART_1_0_rxd]

# Drive strength and slew — defaults are fine; explicit for documentation.
set_property DRIVE 8           [get_ports UART_1_0_txd]
set_property SLEW SLOW         [get_ports UART_1_0_txd]
