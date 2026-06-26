# Zynq PS bare-metal app for db_atcnet_axi

This is the ARM-side bridge between the ESP32 motor controller and the
NN running in the PL. It listens on PS UART1 for a 6000-byte EEG window
from the ESP32, DMAs it into `db_atcnet_axi.s_axis`, polls the NN's
STATUS register until inference completes, reads the classification
from CLASS_REG, and sends a single byte back to the ESP32.

Debug printf goes via PS UART0 (the on-board USB-UART) at 115200 8N1.

## One-time block-design changes

The current BD has Zynq PS UART0 implicitly used by Vitis stdio but does
**not** enable UART1. Before exporting `.xsa`, in Vivado IPI:

1. Double-click `zynq_ultra_ps_e_0` → **I/O Configuration** → **Low Speed**
   → **UART 1**: enable, route to **MIO** (pick free pins) or **EMIO** if
   you'll connect via PMOD/expansion.
2. **PS-PL Configuration** → **General** → **Fabric Clocks** → confirm
   `pl_clk0` is still 60 MHz.
3. Re-validate the BD, regenerate the wrapper, re-export `.xsa`.

If UART1 is on EMIO, also constrain the EMIO ports to the chosen PMOD
pins via an XDC `set_property PACKAGE_PIN`. See the ZCU104 board file
for the PMOD pin LOC names.

## Memory map (matches the BD's address-editor)

| Block | Base | Size |
|---|---|---|
| `db_atcnet_axi.s_axi` | `0x4_0000_0000` | 4 GB region |
| AXI-DMA `S_AXI_LITE` | `0xA000_0000` | 64 KB |
| DDR (for DMA source buffer) | `0x0000_0000` | 2 GB |

`db_atcnet_axi` internal register map (offsets from its base):

| Offset | Name | RW | Bits |
|---|---|---|---|
| `0xF000` | STATUS_REG | R/O | bit 0 = busy, bit 1 = done |
| `0xF004` | CLASS_REG  | R/O | bit 0 = class result (reading clears `done`) |

## Wire protocol (ESP32 ↔ Zynq UART1)

| Direction | Payload | Notes |
|---|---|---|
| ESP32 → Zynq | 6000 bytes raw (3000 × little-endian Q8.8) | No framing; sync via timeout. Sent once per inference. |
| Zynq → ESP32 | 1 byte: `0x00` or `0x01` | Class result. Sent immediately after each successful inference. |

Both sides use 460800 8N1. Mismatch → garbage data → ESP32 watchdog → safe rest.

## Bring-up checklist

1. `write_hw_platform -fixed -include_bit -file db_atcnet_zcu104.xsa` in Vivado.
2. Vitis: New Platform Project from `.xsa` → standalone → A53 #0.
3. Vitis: New Application Project on that platform → **Empty Application**.
4. Drop `main.c` into the application's `src/` folder.
5. Build. Should compile cleanly with the auto-generated BSP.
6. Run/Debug → JTAG → bitstream is auto-programmed → app starts.
7. Open Vitis terminal at 115200 → see `=== db_atcnet PS host bring-up ===`.
8. Connect ESP32 to UART1 pins → flash the ESP32 sketch with matching baud.
9. ESP32 should start receiving 1-byte classifications and driving servos.

## What's NOT in this code yet

| Feature | Why deferred |
|---|---|
| Interrupt-driven DMA / done bit | Polling is simpler and fast enough at 60 MHz inference rate |
| CRC-framed UART protocol | Single byte + 6000-byte payload is fine for bench; add framing for production |
| SD-card dataset replay (no ESP32 needed) | Architecture B path; current code is Architecture A only |
| FreeRTOS / PetaLinux | Bare-metal is the fastest path to demo |

Add these when the bench works.
