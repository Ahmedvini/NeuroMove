# `synth/` — synthesis, implementation, bitstream

Vivado Tcl for taking the RTL to a placed-and-routed bitstream on `xczu7ev`.
The binding constraint is **DSP48E2** (the design sits right at the device's
1,728-DSP limit); timing and area tend to move in opposite directions, so
expect to iterate.

## Flow

```bash
# 1. Synthesize (out-of-context, ~30–60 min; peaks ~13–20 GB RAM — have swap)
vivado -mode batch -source synth/synth_fit.tcl -tclargs xczu7ev-ffvc1156-2-e \
  2>&1 | tee build/synth_run.log

# 2. Implement (place & route, timing close)
vivado -mode batch -source synth/impl_fit.tcl

# 3. Bitstream
vivado -mode batch -source synth/write_bitstream.tcl

# Quick single-module out-of-context sanity (BRAM/DSP inference, ~2 min):
vivado -mode batch -source synth/synth_eca1_ooc.tcl
vivado -mode batch -source synth/synth_pool_ooc.tcl
```

Board retarget / preset helpers: `retarget_zcu106.tcl`,
`apply_zcu106_preset.tcl`.

After synth, read the reports:
```
build/synth_xczu7ev-ffvc1156-2-e/utilization.txt        # totals
build/synth_xczu7ev-ffvc1156-2-e/utilization_hier.txt   # per-instance DSP/LUT/FF/BRAM
build/synth_xczu7ev-ffvc1156-2-e/timing.txt             # WNS / TNS / critical path
```

## Where the DSPs go

| Block | DSPs | Notes |
|---|---:|---|
| `conv2d_temporal` | ~732 | irreducible (parallel variable×const MAC) — keep DSPs |
| `cbam_channel_attn` D1+D2 | ~608 | irreducible — keep DSPs |
| `depthwise_spatial`, `eca_attention`, `gap_accumulator`, `avg_pool_time` | swing ±, opportunistic | constant-multiply blocks Vivado promotes/demotes between runs |

**DSP whack-a-mole:** every module-level `(* use_dsp = "no" *)` you add on a
constant-multiplier block makes Vivado re-promote *other* constant-multiplier
blocks into DSPs to "use the budget". Pin them deliberately and re-check totals.
The most reliable area lever is `(* use_dsp = "no" *)` at the **module level**
of `rtl/conv/depthwise_spatial.sv` (costs ~5–10k LUTs, removes its DSP swing).

## Timing levers (known to work)

- **Pipelining `conv2d_temporal`** (split the 64-tap MAC into staged
  sub-sums) was the single biggest WNS win. The current tree has the 2-stage
  version; further splitting (4×16 / 8×8 taps) buys more slack at the cost of
  latency and re-verifying downstream `out_valid` consumers.
- **Pipelining `cbam_channel_attn` Stage A** (register `avg_gap`/`max_gap`
  before Dense1) targets the remaining critical path; the orchestrator's
  `S_CHAN_WAIT_GATE` state absorbs the extra latency, so no upstream change is
  needed. Re-verify with `chan_w0` / `chan_all` after any attention edit.
- Relax the XDC clock period only **after** area/pipeline work, so the achieved
  frequency is honest. Final implementation closes at 50–60 MHz with positive
  WNS.

## Things to avoid (learned the hard way)

- Signal-level `(* use_dsp = "no" *)` — silently ignored; must be module-level.
- 2-D unpacked `mem[D][N]` with per-element for-loops — infers flip-flops, not
  BRAM. Use 1-D flat `logic [W-1:0] mem [0:D-1]` with `(* ram_style="block" *)`.
- `set_false_path -from <port> -to [all_registers]` — materializes ~1M
  endpoints and stalls synthesis. Use `set_false_path -from <port>` (no `-to`).
- `set_property USE_DSP/RAM_STYLE ... [get_cells -hierarchical ...]` in XDC —
  a full netlist scan and unreliable across runs. Keep these hints **in the RTL**
  as attributes instead.

## Constraints
XDC lives in [`../constraints/`](../constraints/): `db_atcnet_axi.xdc` (clock +
false-paths + IO), and `uart1_emio_*.xdc` for routing PS UART1 to PMOD/EMIO
pins on ZCU104/ZCU106/ZCU102.
