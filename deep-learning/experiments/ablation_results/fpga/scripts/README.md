# `scripts/` — golden model, weight packing, LUTs, Vitis automation

Python (and a little Tcl) that produces everything the RTL consumes and the
testbenches check against. Outputs land in [`../data/`](../data/README.md) and
[`../weights/`](../weights/README.md), both git-ignored because they are fully
reproducible from here.

## Golden model (bit-exact references)
The `q88_*.py` scripts are the **Q8.8 golden model** — one per layer/stage —
that emit the `data/golden_q88/*.hex` tensors the testbenches assert against:

| Script | Stage |
|---|---|
| `q88_layer1.py` … `q88_layer8_9.py` | per-layer conv / pooling / attention stages |
| `q88_branchB.py` | branch B depthwise-separable path |
| `q88_eca1.py`, `q88_layer17_chan.py`, `q88_layer17_spatial.py` | ECA / CBAM attention |
| `q88_layer_tcfn.py` | TCFN feed-forward |
| `q88_classifier.py` | dense classifier + head |
| `q88_end_to_end.py` | full-network golden (raw EEG → class) |

## Activation LUTs
| Script | Output |
|---|---|
| `gen_elu_lut.py` | `data/lut/elu_q88.hex` |
| `gen_sigmoid_lut.py` | `data/lut/sigmoid_q88.hex` |

## Stimulus & weight packing
| Script | Purpose |
|---|---|
| `gen_stimulus.py` | Generate input EEG stimulus windows for the testbenches. |
| `h5_to_mem.py` | Convert a trained Keras `.h5` model into Q8.8 `.mem`/`.hex` weight files. |
| `pack_weights_axil.py` | Pack weights into the AXI-Lite weight-loader broadcast format. |
| `cat_window_weights.py` | Concatenate/inspect packed window weights. |
| `edf_to_hex.py` | Convert raw EDF EEG recordings into Q8.8 hex windows. |

## Deployment helpers
| File | Purpose |
|---|---|
| `load_zcu106.tcl` | xsdb/JTAG: program bitstream + release PL reset + load the PS ELF. |
| `vitis/create_platform.py`, `vitis/create_app.py`, `vitis/build_app.tcl` | Automate the Vitis platform + bare-metal app build from the exported `.xsa`. |
| `vitis/smoke_test.py` | Post-deploy smoke test. |

> **PL reset note:** JTAG load scripts must call `psu_ps_pl_reset_config` after
> `fpga -file`, otherwise the PL stays held in reset and every AXI access hangs.
