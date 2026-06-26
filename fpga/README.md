# FPGA

Hardware design for real-time EEG/fNIRS signal processing and/or
acceleration of inference.

## Layout

```
fpga/
├── rtl/           # Synthesizable HDL source (Verilog / VHDL / SystemVerilog)
├── testbench/     # Simulation testbenches
├── constraints/   # Pin & timing constraints (.xdc / .sdc)
├── ip/            # Vendor / custom IP cores
├── sim/           # Simulation scripts, waveforms, run dirs
├── scripts/       # Tcl build/automation scripts (Vivado/Quartus)
└── docs/          # Block diagrams, register maps, notes
```

## Toolchain
- **Vivado** (Xilinx/AMD) or **Quartus** (Intel) — set in `scripts/`.
- Simulation via the vendor simulator, Verilator, or ModelSim/Questa.

## Typical flow

```bash
# Build (example: Vivado batch mode)
vivado -mode batch -source scripts/build.tcl

# Simulate
cd sim && make
```

## Conventions
- Keep RTL vendor-agnostic where possible; isolate vendor IP in `ip/`.
- Generated/build artifacts (`*.runs/`, `*.cache/`, bitstreams) are git-ignored.
