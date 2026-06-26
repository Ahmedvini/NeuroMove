#!/usr/bin/env python3
"""
test_inference.py — Stream a 6000-byte EEG window to the Zynq PS UART1 and
read the 1-byte classification reply. Use this to verify the FPGA inference
pipeline end-to-end without involving the ESP32.

Usage:
    python3 sw/host/test_inference.py \
        --window build/test_window.bin \
        --port /dev/ttyUSB3 \
        --baud 460800

Default port is /dev/ttyUSB3 (CP2108 channel 2 = PS UART1 EMIO route on the
ZCU106 with the current udev mapping). Adjust if your enumeration differs.
"""
import argparse
import os
import select
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="build/test_window.bin",
                    help="Path to the 6000-byte EEG window binary (default %(default)s)")
    ap.add_argument("--port", default="/dev/ttyUSB3",
                    help="Zynq PS UART1 serial port (default %(default)s)")
    ap.add_argument("--baud", type=int, default=460800,
                    help="Baud rate (default %(default)s)")
    ap.add_argument("--n", type=int, default=1,
                    help="Number of consecutive inferences to run (default 1)")
    args = ap.parse_args()

    # Load the window
    try:
        with open(args.window, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"[test] ERROR reading {args.window}: {e}", file=sys.stderr)
        sys.exit(1)
    if len(data) != 6000:
        print(f"[test] WARN: expected 6000 bytes, got {len(data)} — sending anyway",
              file=sys.stderr)

    # Configure the port with stty, then open via raw POSIX. We avoid pyserial
    # because Serial.open() momentarily asserts DTR/RTS before they can be set
    # to False — on the ZCU106's CP2108 channel 2 this glitches the link and
    # the FPGA UART1 stops receiving bytes. With raw POSIX + stty -hupcl, the
    # control lines stay quiescent. (Discovered during Phase B6 bring-up.)
    stty_args = [
        "stty", "-F", args.port,
        str(args.baud), "cs8", "-cstopb", "-parenb",
        "-ixon", "-ixoff", "-ixany", "-crtscts",
        "raw", "-clocal", "-hupcl", "-echo",
    ]
    try:
        subprocess.run(stty_args, check=True, capture_output=True)
    except Exception as e:
        print(f"[test] ERROR configuring {args.port}: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY)
    except Exception as e:
        print(f"[test] ERROR opening {args.port}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[test] Opened {args.port} @ {args.baud} 8N1 (POSIX, no DTR/RTS toggle)")

    # Drain any pending bytes
    while True:
        r, _, _ = select.select([fd], [], [], 0)
        if not r:
            break
        os.read(fd, 4096)
    time.sleep(0.2)

    def read_with_timeout(timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r, _, _ = select.select([fd], [], [], deadline - time.monotonic())
            if r:
                chunk = os.read(fd, 1)
                if chunk:
                    return chunk
        return b""

    for i in range(args.n):
        t0 = time.monotonic()
        # Write via shell redirection — same path as the manual `cat file > tty`
        # that we know works on this CP2108 channel.
        proc = subprocess.Popen(["cat"], stdin=subprocess.PIPE,
                                stdout=open(args.port, "wb", buffering=0))
        proc.communicate(input=data)
        write_dt = time.monotonic() - t0
        print(f"[test] #{i+1}: sent 6000 bytes in {write_dt*1000:.1f} ms,"
              f" waiting for class byte (5s timeout)...")
        # Longer timeout because we observed bytes can take ~2s to fully
        # propagate via the CP2108 → FPGA path on the first burst.
        reply = read_with_timeout(5.0)
        dt = time.monotonic() - t0
        if len(reply) != 1:
            print(f"[test] #{i+1}: TIMEOUT — no class byte received after"
                  f" {dt*1000:.0f} ms")
            os.close(fd)
            sys.exit(2)
        cls = reply[0]
        print(f"[test] #{i+1}: got class={cls} (0x{cls:02x}) in {dt*1000:.1f} ms"
              f" total round-trip")

    print("[test] all inferences completed cleanly")
    os.close(fd)


if __name__ == "__main__":
    main()
