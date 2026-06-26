#!/usr/bin/env python3
import sys
import os
import glob

try:
    import mne
except ImportError:
    print("ERROR: 'mne' package not found.  Run:  pip install mne")
    sys.exit(1)

CHANNELS_PER_BLOCK = 8
BITS_PER_SAMPLE    = 16
MAX_SAMPLES        = 65535

def to_int16(v, max_val):
    scaled = int(v / max_val * 32767) if max_val != 0 else 0
    return max(-32768, min(32767, scaled)) & 0xFFFF

def process_edf(edf_path, out_file, blocks_written, max_blocks):
    """Read one .edf file and append its blocks to out_file. Returns new block count."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  SKIP {edf_path}: {e}")
        return blocks_written, None

    data, _ = raw.get_data(return_times=True)
    n_channels = len(raw.ch_names)
    sfreq      = int(raw.info['sfreq'])
    n_times    = raw.n_times

    max_val = max(abs(data.max()), abs(data.min()), 1e-12)

    for t in range(n_times):
        if blocks_written >= max_blocks:
            return blocks_written, (n_channels, sfreq)
        word = 0
        for ch in range(CHANNELS_PER_BLOCK):
            sample = to_int16(data[ch, t], max_val) if ch < n_channels else 0
            word   = (word << 16) | sample
        out_file.write(f"{word:032X}\n")
        blocks_written += 1

    return blocks_written, (n_channels, sfreq)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/edf_to_hex.py path/to/dataset_folder/")
        sys.exit(1)

    input_path = sys.argv[1]

    # ── Collect all .edf files (skip .edf.event files) ───────────────────────
    if os.path.isdir(input_path):
        # Recursively find all .edf files, sorted by subject then run
        edf_files = sorted([
            f for f in glob.glob(os.path.join(input_path, "**", "*.edf"), recursive=True)
            if not f.endswith(".event")
        ])
        if not edf_files:
            print(f"ERROR: No .edf files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(edf_files)} .edf files:")
        for f in edf_files:
            print(f"  {f}")
    elif os.path.isfile(input_path) and input_path.endswith(".edf"):
        edf_files = [input_path]
    else:
        print(f"ERROR: {input_path} is not a .edf file or a folder")
        sys.exit(1)

    # ── Process all files ─────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    hex_path  = "data/dataset.hex"
    info_path = "data/dataset_info.txt"

    blocks_written = 0
    last_info      = (64, 250)  # fallback (n_channels, sfreq)

    print(f"\nConverting to {hex_path} ...")
    with open(hex_path, "w") as out:
        for edf_path in edf_files:
            print(f"  Processing: {os.path.basename(edf_path)}", end=" ... ", flush=True)
            prev = blocks_written
            blocks_written, info = process_edf(edf_path, out, blocks_written, MAX_SAMPLES)
            if info:
                last_info = info
            print(f"{blocks_written - prev} blocks")
            if blocks_written >= MAX_SAMPLES:
                print(f"  Reached MAX_SAMPLES ({MAX_SAMPLES}), stopping early.")
                break

    n_channels, sfreq = last_info

    # ── Write info file ───────────────────────────────────────────────────────
    with open(info_path, "w") as f:
        f.write("===== Paste these values into eeg_dataset_config.sv =====\n\n")
        f.write(f"parameter DATASET_DEPTH = {blocks_written};\n\n")
        f.write(f"localparam NUM_CHANNELS    = {n_channels};\n")
        f.write(f"localparam SAMPLE_RATE     = {sfreq};\n")
        f.write(f"localparam BITS_PER_SAMPLE = {BITS_PER_SAMPLE};\n\n")
        f.write(f"// Uncomment in initial block:\n")
        f.write(f'// $readmemh("data/dataset.hex", dataset_memory);\n\n')
        f.write(f"// Source folder: {input_path}\n")
        f.write(f"// Files processed: {len(edf_files)}\n")
        f.write(f"// Total blocks: {blocks_written}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("DONE. Paste these into eeg_dataset_config.sv:")
    print("=" * 60)
    print(f"  parameter DATASET_DEPTH    = {blocks_written};")
    print(f"  localparam NUM_CHANNELS    = {n_channels};")
    print(f"  localparam SAMPLE_RATE     = {sfreq};")
    print(f"  localparam BITS_PER_SAMPLE = {BITS_PER_SAMPLE};")
    print()
    print("And uncomment this line in the initial block:")
    print('  $readmemh("data/dataset.hex", dataset_memory);')
    print()
    print(f"Full details: {info_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

