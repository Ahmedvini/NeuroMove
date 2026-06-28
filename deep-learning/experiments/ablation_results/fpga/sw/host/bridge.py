#!/usr/bin/env python3
"""
bridge.py — Bidirectional UART pipe between ESP32 and the ZCU106's PS UART1.

Path A in this thesis is "ESP32 wired direct to the FPGA PMOD". Path B (this
script) keeps the same protocol but inserts a laptop in the middle while the
PMOD route is still un-synthesised:

    ESP32 (USB-Serial) <-> laptop <-> CP2108 ch.2 (/dev/ttyUSBN) <-> PS UART1

The bridge does pure byte pass-through, so the wire protocol is identical to
the eventual direct link:

  ESP32 -> Zynq : 6000 bytes (raw int16 little-endian EEG window)
  Zynq  -> ESP32: 1 byte     (class, 0x00 or 0x01)

Notes:
  * The FPGA side uses raw POSIX I/O — pyserial's Serial.open() toggles
    DTR/RTS on the CP2108 ch.2 and glitches the FPGA RX line. The ESP32
    side uses pyserial (USB CDC doesn't have the same hardware issue).
  * Use `--list` to discover device names, then pass --zynq / --esp
    explicitly. ttyUSB numbering shifts with USB enumeration; see
    /sys/bus/usb-serial/devices/ttyUSB*/uevent for the channel.
  * Ctrl+C exits cleanly.

Usage:
    python3 sw/host/bridge.py --list
    python3 sw/host/bridge.py --zynq /dev/ttyUSB2 --esp /dev/ttyACM0
"""

import argparse
import os
import select
import subprocess
import sys
import time

try:
    from serial import Serial
    from serial.tools.list_ports import comports
    HAVE_PYSERIAL = True
except ImportError:
    HAVE_PYSERIAL = False


def list_serials():
    if not HAVE_PYSERIAL:
        print("pyserial not installed — list mode unavailable. Trying /dev/tty* instead.")
        for path in sorted(os.listdir("/dev")):
            if path.startswith(("ttyUSB", "ttyACM")):
                print(f"  /dev/{path}")
        return
    print("Available serial devices:")
    ports = list(comports())
    if not ports:
        print("  (none — check that both USB cables are plugged in)")
        return
    for p in ports:
        vid_pid = f"{p.vid:04x}:{p.pid:04x}" if p.vid else "?:?"
        print(f"  {p.device:<16s}  {vid_pid:<10s}  {p.description}")
    print()
    print("Heuristic: ZCU106 CP2108 enumerates 4 ports with VID 10c4 PID ea71;")
    print("PS UART1 is CP2108 ch.2 (look for ID_USB_INTERFACE_NUM=02).")
    print("ESP32-S3 native USB usually shows as /dev/ttyACM*.")


def open_fpga(path: str, baud: int) -> int:
    """Configure with stty (no DTR/RTS toggle) and open with raw POSIX."""
    subprocess.run(
        ["stty", "-F", path, str(baud), "cs8", "-cstopb", "-parenb",
         "-ixon", "-ixoff", "-ixany", "-crtscts",
         "raw", "-clocal", "-hupcl", "-echo"],
        check=True, capture_output=True,
    )
    return os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--zynq", default="/dev/ttyUSB2",
                    help="ZCU106 PS UART1 device (default %(default)s)")
    ap.add_argument("--esp", default="/dev/ttyACM0",
                    help="ESP32 USB-Serial device (default %(default)s)")
    ap.add_argument("--zynq-baud", type=int, default=460800,
                    help="ZCU106 baud (default %(default)s)")
    ap.add_argument("--esp-baud", type=int, default=460800,
                    help="ESP32 baud (default %(default)s)")
    ap.add_argument("--list", action="store_true",
                    help="List available serial ports and exit")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-byte logging")
    args = ap.parse_args()

    if args.list:
        list_serials()
        return

    if not HAVE_PYSERIAL:
        print("[bridge] pyserial required for the ESP32 side. `pip install pyserial`.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[bridge] Zynq: {args.zynq} @ {args.zynq_baud} 8N1 (raw POSIX)")
    print(f"[bridge] ESP:  {args.esp}  @ {args.esp_baud}  8N1 (pyserial)")

    try:
        fpga_fd = open_fpga(args.zynq, args.zynq_baud)
    except Exception as e:
        print(f"[bridge] ERROR opening {args.zynq}: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        esp = Serial(args.esp, args.esp_baud, timeout=0)
    except Exception as e:
        print(f"[bridge] ERROR opening {args.esp}: {e}", file=sys.stderr)
        os.close(fpga_fd)
        sys.exit(1)

    # Drain any stale bytes
    while True:
        r, _, _ = select.select([fpga_fd], [], [], 0)
        if not r:
            break
        try:
            os.read(fpga_fd, 4096)
        except BlockingIOError:
            break
    esp.reset_input_buffer()
    time.sleep(0.2)

    print("[bridge] forwarding (Ctrl+C to stop)")
    bytes_e2z = 0   # ESP32 -> FPGA
    bytes_z2e = 0   # FPGA -> ESP32
    classifications = []

    try:
        while True:
            r, _, _ = select.select([fpga_fd, esp.fileno()], [], [], 0.05)
            # FPGA -> ESP32 (class bytes)
            if fpga_fd in r:
                try:
                    chunk = os.read(fpga_fd, 4096)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    esp.write(chunk)
                    bytes_z2e += len(chunk)
                    if not args.quiet:
                        for cls in chunk:
                            classifications.append(cls)
                            print(f"[Z->E] class=0x{cls:02x} "
                                  f"(total Z->E={bytes_z2e}B, E->Z={bytes_e2z}B)")
            # ESP32 -> FPGA (EEG window)
            if esp.fileno() in r:
                chunk = esp.read(4096)
                if chunk:
                    # Write in a loop in case the kernel buffer is small
                    view = memoryview(chunk)
                    while view:
                        try:
                            n = os.write(fpga_fd, view)
                            view = view[n:]
                        except BlockingIOError:
                            time.sleep(0.001)
                    bytes_e2z += len(chunk)
                    if not args.quiet and bytes_e2z % 2000 < len(chunk):
                        print(f"[E->Z] +{len(chunk)}B (total E->Z={bytes_e2z}B)")
    except KeyboardInterrupt:
        print(f"\n[bridge] stopped — totals: E->Z={bytes_e2z}B, Z->E={bytes_z2e}B")
        if classifications:
            c0 = classifications.count(0)
            c1 = classifications.count(1)
            other = len(classifications) - c0 - c1
            print(f"[bridge] classifications: c0={c0} c1={c1} other={other}")
    finally:
        try:
            os.close(fpga_fd)
        except OSError:
            pass
        try:
            esp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
