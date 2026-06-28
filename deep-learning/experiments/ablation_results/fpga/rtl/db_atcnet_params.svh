 // =============================================================================
// db_atcnet_params.svh
//
// Central parameter header for the HaLT variant of DB-ATCNet, subject-
// dependent training, 5-of-19 Gumbel channel selection, 2-class output.
//
// Reference: docs/FPGA_DEPLOYMENT_DOC.md
// Reference weights: weights/subject_<X>/session_<N>_heldout/best_model.weights.h5
//
// Architecture summary (per the spec doc, layer-by-layer):
//   Input (1, 1, NUM_INPUT_CH, TRIAL_SAMPLES) - 5 ch x 600 samples @ 200 Hz
//     -> Permute (3,2,1) -> (TRIAL_SAMPLES, NUM_INPUT_CH, 1)
//   ADBC block:
//     Conv2D(F1, (KE,1), same, no bias)        -> (T, C, F1)
//     BatchNorm                                -> (T, C, F1)
//     ECA_1 (kernel = ECA_K_F1)                -> (T, C, F1)
//     Branch A: Depthwise(1,NUM_INPUT_CH) D=2  -> (T, 1, F1*2)
//               BN + ELU
//               AvgPool(POOL1, 1)              -> (T/POOL1, 1, F1*2)
//               Conv2D(F2, (16,1), same)
//               BN + ELU
//               AvgPool(POOL2, 1)              -> (T/POOL1/POOL2, 1, F2)
//     Branch B: same with D=4, final projection back to F2
//     Add(A, B)                                -> (T_pool, 1, F2)
//   ECA_2 (kernel = ECA_K_F2)                  -> (T_pool, 1, F2)
//   Squeeze last spatial dim                   -> (T_pool, F2)
//   x 5 windows (INDEPENDENT weights per window):
//     slice [i:i+WIN_LEN]                      -> (WIN_LEN, F2)
//     improved_cbam_block                      -> (WIN_LEN, F2)
//     TCFN block (depth=2, k=4, dilations 1,2) -> (WIN_LEN, F2)
//     last-step slice                          -> (F2,)
//     Dense(NUM_CLASSES) (with bias)           -> (NUM_CLASSES,)
//   Average(5)                                 -> (NUM_CLASSES,)
//   Softmax -> argmax                          -> class index (1 bit if 2-class)
// =============================================================================

`ifndef DB_ATCNET_PARAMS_SVH
`define DB_ATCNET_PARAMS_SVH

package db_atcnet_pkg;

    // -------------------------------------------------------------------------
    // Fixed-point format (signed Q(I).(FRAC_BITS))
    // -------------------------------------------------------------------------
    localparam int DATA_WIDTH    = 16;
    localparam int COEF_WIDTH    = 16;
    localparam int ACC_WIDTH     = 48;
    localparam int FRAC_BITS     = 8;          // Q8.8 throughout

    // -------------------------------------------------------------------------
    // Dataset / I-O shape (HaLT, subject-dependent)
    // -------------------------------------------------------------------------
    localparam int NUM_EEG_CH    = 19;         // total EEG montage size
    localparam int NUM_INPUT_CH  = 5;          // channels actually fed to the model
    localparam int TRIAL_SAMPLES = 600;        // 3 s @ 200 Hz
    localparam int SAMPLE_RATE   = 200;        // Hz
    localparam int NUM_CLASSES   = 2;          // right-hand fist vs left-leg ankle

    // -------------------------------------------------------------------------
    // ADBC Block (Layer 1..pool_2)
    // -------------------------------------------------------------------------
    localparam int F1            = 16;         // Conv2D filters in Layer 1
    localparam int KE            = 64;         // Conv2D kernel size (temporal axis)
    localparam int F2            = 32;         // channels after branch projection
    localparam int D_BRANCH_A    = 2;          // depthwise multiplier branch A
    localparam int D_BRANCH_B    = 4;          // depthwise multiplier branch B
    localparam int POOL1         = 8;          // first  AvgPool (T -> T/8)
    localparam int POOL2         = 7;          // second AvgPool (T/8 -> T/8/7)
    localparam int SEP_KE        = 16;         // separable-conv temporal kernel
    localparam int T_AFTER_POOL  = TRIAL_SAMPLES / POOL1 / POOL2; // 600/8/7 = 10

    // -------------------------------------------------------------------------
    // ECA (Efficient Channel Attention) — used at two positions
    //   ECA_1 sees F1 channels, ECA_2 sees F2 channels
    //   k = int(abs((log2(C)+1)/2)); odd-ify -> ECA_K_F1 = ECA_K_F2 = 3
    // -------------------------------------------------------------------------
    localparam int ECA_K_F1      = 3;
    localparam int ECA_K_F2      = 3;

    // -------------------------------------------------------------------------
    // Sliding-window ATC branches (5 windows of 6 timesteps each over T_AFTER_POOL=10)
    // -------------------------------------------------------------------------
    localparam int NUM_WINDOWS   = 5;
    localparam int WIN_LEN       = T_AFTER_POOL - NUM_WINDOWS + 1; // 10-5+1 = 6

    // -------------------------------------------------------------------------
    // Improved CBAM (per-window attention) — channel attention + spatial attention
    //   channel_attention: Dense(F2 -> F2/CBAM_RATIO) ReLU -> Dense(-> F2) linear,
    //     shared between avg-pool and max-pool paths, then add + sigmoid + mul.
    //   improved_spatial_attention: 3-pool concat (avg + max + stochastic) ->
    //     Conv2D(1, (SPATIAL_KE,1), same, no bias, sigmoid) -> mul.
    // -------------------------------------------------------------------------
    localparam int CBAM_RATIO    = 8;
    localparam int CBAM_HIDDEN   = F2 / CBAM_RATIO;   // 32 / 8 = 4
    localparam int SPATIAL_KE    = 7;
    localparam int CBAM_POOLS    = 3;                 // avg, max, stochastic

    // -------------------------------------------------------------------------
    // TCFN (Temporal Convolutional Fusion Network) — per-window TCN
    //   depth=2 means: 2x [Conv1D -> BN -> ELU] block + residual,
    //   then dilations doubling each stage (1, 2). 4 Conv1D total per TCFN.
    // -------------------------------------------------------------------------
    localparam int TCN_DEPTH     = 2;
    localparam int TCN_KE        = 4;          // Conv1D kernel size
    localparam int TCN_FILTERS   = F2;         // 32 throughout

    // -------------------------------------------------------------------------
    // Output head
    // -------------------------------------------------------------------------
    localparam int DENSE_OUT     = NUM_CLASSES;
    localparam int CLASS_INDEX_W = (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES);

endpackage : db_atcnet_pkg

`endif // DB_ATCNET_PARAMS_SVH
