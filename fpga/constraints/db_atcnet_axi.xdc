# =============================================================================
# db_atcnet_axi.xdc  —  Constraints for the DB-ATCNet deployment block on
# ZCU104 (XCZU7EV-2FFVC1156).
#
# Target use:
#   * Out-of-context synthesis  (default — used by synth/synth_fit.tcl).
#     IOBs are NOT inserted; the design synthesizes as a block ready for
#     IPI integration. Only the clock constraint applies.
#
#   * In-context bring-up.  When db_atcnet_axi is dropped into a Vivado IPI
#     block design with the Zynq PS, all s_axi / s_axis / irq_done ports are
#     internally connected and NO pin assignments are needed. The clock
#     constraint stays valid (sourced by the PS PL_CLK0).
#
# The 10,520-Bonded-IOB blowup from a naive flat synthesis of a top with
# many AXI ports is avoided by:
#   - keeping the design out-of-context, OR
#   - wrapping it in an IPI block design with the PS for in-context flow.
# Do NOT run "Run Synthesis" from the Vivado GUI against db_atcnet_axi as
# a top-level project — those AXI ports would each pull an IOB.
# =============================================================================

# ---------------------------------------------------------------------------
# Primary clock — 100 MHz on `clk`. For ZCU104, PL_CLK0 from the PS is the
# canonical source; the IPI block design forwards it here.
# ---------------------------------------------------------------------------
create_clock -name clk -period 17.000 [get_ports clk]
# Clock target: 58.82 MHz (17.0 ns period).
# Why not 100 MHz: at 86% LUT / 99.5% DSP utilization the routing is dense;
# the worst path measured after place&route was 16.1 ns (15 ns period →
# WNS = -1.1 ns at 67 MHz; ~64% of the path is wire delay, not logic). At
# 17 ns we get ~1 ns of margin and the design closes cleanly. EEG inference
# needs <5 Hz classification rate; this design at 58.82 MHz runs ~2,000
# inferences/sec — ~400× the application requirement. Pushing higher would
# need either more invasive RTL pipelining (e.g. inside tcfn / cbam_channel)
# or aggressive impl directives, neither of which is worth the engineering
# time given there is no application benefit.

# ---------------------------------------------------------------------------
# Input/output delay defaults — filter by direction so a single AXI-bundle
# wildcard doesn't apply set_input_delay to output ports (and vice versa),
# which Vivado rejects with [Constraints 18-602].
# ---------------------------------------------------------------------------
set_input_delay  -clock [get_clocks clk] -max 2.0 [get_ports -filter {DIRECTION == IN}  s_axi_*]
set_input_delay  -clock [get_clocks clk] -min 0.5 [get_ports -filter {DIRECTION == IN}  s_axi_*]
set_output_delay -clock [get_clocks clk] -max 2.0 [get_ports -filter {DIRECTION == OUT} s_axi_*]
set_output_delay -clock [get_clocks clk] -min 0.5 [get_ports -filter {DIRECTION == OUT} s_axi_*]

set_input_delay  -clock [get_clocks clk] -max 2.0 [get_ports -filter {DIRECTION == IN}  s_axis_*]
set_input_delay  -clock [get_clocks clk] -min 0.5 [get_ports -filter {DIRECTION == IN}  s_axis_*]
set_output_delay -clock [get_clocks clk] -max 2.0 [get_ports -filter {DIRECTION == OUT} s_axis_*]
set_output_delay -clock [get_clocks clk] -min 0.5 [get_ports -filter {DIRECTION == OUT} s_axis_*]

set_output_delay -clock [get_clocks clk] -max 2.0 [get_ports irq_done]
set_output_delay -clock [get_clocks clk] -min 0.5 [get_ports irq_done]

# ---------------------------------------------------------------------------
# Reset is asynchronous — declare to the timing engine.
# Note: avoid `-to [all_registers]` here; on this design that materialises
# ~1M endpoints and balloons synthesis runtime/memory. The unconstrained
# `set_false_path -from` form gives the same async-reset semantics.
# ---------------------------------------------------------------------------
set_input_delay  -clock [get_clocks clk] -max 2.0 [get_ports rst]
set_input_delay  -clock [get_clocks clk] -min 0.5 [get_ports rst]
set_false_path   -from  [get_ports rst]

# ---------------------------------------------------------------------------
# RAM_STYLE hints live in the RTL as `(* ram_style = "..." *)` attributes on
# the array declarations (eca1_pipeline.sv, branch_pipeline.sv, tcfn.sv).
# Applying them via `[get_cells -hierarchical -filter ...]` post-elab forces
# Vivado to scan the entire 1M-cell netlist per filter, which is what was
# wedging synthesis.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# False paths for the AXI-Lite weight loader writes — single-write
# transactions that don't need single-cycle timing closure.
# ---------------------------------------------------------------------------
# (Add here once the post-synth report shows specific cross-domain warnings.)
