# `sw/` — software: firmware, PS bridge, host tools

The software that surrounds the PL accelerator and turns it into a working BCI
loop. Three independent pieces:

```
 sw/esp32/   ── ESP32-S3 firmware (acquire/replay EEG, drive servos)
 sw/zynq_ps/ ── Zynq A53 bare-metal bridge (UART ↔ AXI-DMA ↔ PL mailbox)
 sw/host/    ── Python bench tools (drive the UART path from a PC)
```

## `esp32/` — ESP32-S3 firmware
`db_atcnet_esp32/db_atcnet_esp32.ino` — Arduino sketch that acquires (or
replays) an EEG window, formats it as little-endian Q8.8, streams the
**6000-byte** payload to the Zynq over UART at **460800 8N1**, receives the
1-byte class result, runs a majority voter + safety watchdog, and drives the
servos (PCA9685). On comms mismatch the watchdog returns the actuators to a
safe rest position.

## `zynq_ps/` — Zynq PS bare-metal bridge
`main.c` — the ARM-side bridge. Listens on **PS UART1** for the 6000-byte
window, DMAs it into `db_atcnet_axi.s_axis`, polls `STATUS_REG` until inference
completes, reads `CLASS_REG`, and sends one byte back to the ESP32. Debug
`printf` goes to **PS UART0** (on-board USB-UART, 115200).

See **[`zynq_ps/README.md`](zynq_ps/README.md)** for the memory map, the
`db_atcnet_axi` register layout (`0xF000` STATUS / `0xF004` CLASS), the wire
protocol, and the Vivado/Vitis bring-up checklist.

## `host/` — Python bench tools
For driving the same UART path from a PC (no ESP32 needed):

| File | Purpose |
|---|---|
| `bridge.py` | Host-side UART bridge to the Zynq PS (Path B bench testing). |
| `hex_to_window.py` | Convert a golden/EEG hex window into the 6000-byte Q8.8 wire payload. |
| `test_inference.py` | Send a window, read the class byte back, check it. |

## The loop in one line
`EEG → ESP32 (Q8.8 window) → UART → PS (DMA) → PL (db_atcnet_axi) → class byte
→ ESP32 (vote + servo)`. Both UART ends run 460800 8N1; the PL side is
documented in [`../rtl/README.md`](../rtl/README.md).
