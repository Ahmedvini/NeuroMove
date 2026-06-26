# `rtl/` — SystemVerilog source

All synthesizable RTL for DB-ATCNet. Everything here is Q8.8 fixed-point and
bit-exact against the Python golden model (see [`../sim/`](../sim/README.md)).
The canonical compile order is the `MODEL_RTL`/`SEC_RTL` arrays in
[`../build.sh`](../build.sh) — order matters where one module instantiates
another.

## Top-level integrators (`rtl/*.sv`)

| File | Function |
|---|---|
| `db_atcnet_axi.sv` | **Deployment wrapper.** AXI4-Stream EEG in, AXI4-Lite weight loader + status/class mailbox out. The default synthesis top. |
| `db_atcnet_axi_v.v` | Thin Verilog wrapper that hard-codes the weight/LUT `$readmemh` file paths for Vivado IPI packaging. |
| `db_atcnet_top.sv` | Raw EEG → 1-bit class, no AXI plumbing (clean DUT for `top` tb). |
| `db_atcnet_window_pipeline.sv` | Orchestrates the attention + classifier back half. The `S_CHAN_WAIT_GATE` state absorbs variable CBAM latency. |
| `db_atcnet_post_eca1.sv` | ECA₁-gated stream → ECA₂ buffer. |
| `db_atcnet_conv2d_eca1.sv` | Raw EEG → ECA₁-gated stream (front half). |

## Compute primitives

| Dir | Modules | Role |
|---|---|---|
| `conv/` | `conv2d_temporal`, `conv1d_temporal`, `conv1d_temporal_tm`, `depthwise_spatial`, `branch_pipeline`, `avg_pool_time`, `tcfn`, `elu`, `sat_add` | Temporal/spatial convolutions, the dual depthwise-separable branches, pooling, the TCFN feed-forward, saturating add, ELU. |
| `attention/` | `eca_attention`, `eca1_pipeline`, `gap_accumulator`, `gate_apply`, `cbam_channel_attn`, `cbam_spatial_attn` | ECA + CBAM channel/spatial attention and the gate-application plumbing. |
| `classifier/` | `dense_classifier`, `output_head` | Final dense layer → logits → class bit. |
| `window/` | `gumbel_channel_adapter`, `eeg_channel_adapter` | Learned 5-of-19 channel selection (Gumbel mask) and raw-EEG adaptation. |
| `weights/` | `axi_lite_weight_loader`, `weight_bank` | Self-decoding weight banks loaded over AXI-Lite at boot. |
| `util/` | `serial_divider` | Shared arithmetic helper. |

## Implementation notes (hard-won)

- **Fixed-point**: 16-bit signed Q8.8 everywhere; saturate+round to Q8.8 at
  stage boundaries via `conv/sat_add.sv`. Non-linearities are Q8.8 LUTs.
- **Memory inference**: large buffers use **1-D flat** `logic [W-1:0] mem [0:D-1]`
  with `(* ram_style = "block" *)`, not 2-D unpacked arrays — the latter trip
  Vivado's "3D-RAM/struct" path and fall back to flip-flops. (This is what
  turned ~768k FFs into ~21 RAMB36 in `eca1_pipeline`.)
- **DSP control**: `(* use_dsp = "no" *)` only works at **module level**
  (signal-level is silently ignored). Vivado re-balances DSPs across
  constant-multiplier modules between runs — see `synth/README.md`.
- The genuinely large compute is `conv2d_temporal` (~732 DSPs) and
  `cbam_channel_attn` D1+D2 (~608 DSPs); these must keep their DSPs.

## Security

`security/` is a standalone crypto suite, **not** part of the inference
datapath, built alongside the model (`SEC_RTL` in `build.sh`):

| Module | Function |
|---|---|
| `aes_256_core`, `aes_256_gcm`, `gcm_ghash` | AES-256 + GCM authenticated encryption |
| `sha256_core`, `sha256_hash_chain`, `hmac_sha256` | SHA-256 and HMAC |
| `rsa2048_core` | RSA-2048 |
| `secure_boot`, `eeg_data_encryptor`, `eeg_security_top`, `eeg_dataset_config`, `hmac_demo_top` | Secure-boot skeleton and EEG-path encryption/authentication demos |
