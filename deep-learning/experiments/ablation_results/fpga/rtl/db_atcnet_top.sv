// =============================================================================
// db_atcnet_top.sv  —  End-to-end RTL inference for HaLT DB-ATCNet.
//
//   raw EEG stream (1 (sample, electrode) per cycle)
//     → db_atcnet_conv2d_eca1   (Conv2D + ECA₁ gate)
//     → db_atcnet_post_eca1     (Branch A + Branch B + sat_add + ECA₂)
//     → db_atcnet_window_pipeline (5 sliding windows + ImpCBAM + TCFN +
//                                  Dense classifier + sum-argmax head)
//     → 1-bit class output (0 = Right Hand fist, 1 = Left Leg ankle)
//
// Pulses `done` exactly once per inference (after the window pipeline emits
// its argmax). The caller drives the raw input then waits for done; class_out
// is held stable until the next inference.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_top #(
    parameter int    DATA_WIDTH       = 16,
    parameter int    C_IN             = 5,
    parameter int    F1               = 16,
    parameter int    F                = 32,
    parameter int    KE_CONV2D        = 64,
    parameter int    KE_SEP           = 16,
    parameter int    T_IN             = 600,
    parameter int    POOL1            = 8,
    parameter int    POOL2            = 7,
    parameter int    T_FULL           = 10,
    parameter int    T_WIN            = 6,
    parameter int    N_WIN            = 5,
    parameter int    N_CLS            = 2,
    parameter int    ECA1_INV_N       = 5592,
    parameter int    ECA2_INV_N       = 1677722,
    parameter int    CHAN_INV_N       = 2796203,
    parameter int    INV_POOL1        = 2097152,
    parameter int    INV_POOL2        = 2396745,
    parameter int    SIG_LUT_N        = 1024,
    parameter int    SIG_LUT_RANGE_Q  = 1024,
    parameter int    SIG_LUT_SHIFT    = 1,
    parameter int    ELU_LUT_N        = 256,
    parameter int    ELU_LUT_RANGE_Q  = 2048,
    parameter int    ELU_LUT_SHIFT    = 3,
    // Weight files
    parameter string CONV2D_W_FILE        = "",
    parameter string CONV2D_B_FILE        = "",
    parameter string ECA1_W_FILE          = "",
    parameter string SIG_LUT_FILE         = "",
    parameter string ELU_LUT_FILE         = "",
    parameter string BRANCHA_DWISE_W_FILE = "",
    parameter string BRANCHA_DWISE_B_FILE = "",
    parameter string BRANCHA_SEP_W_FILE   = "",
    parameter string BRANCHA_SEP_B_FILE   = "",
    parameter string BRANCHB_DWISE_W_FILE = "",
    parameter string BRANCHB_DWISE_B_FILE = "",
    parameter string BRANCHB_SEP_W_FILE   = "",
    parameter string BRANCHB_SEP_B_FILE   = "",
    parameter string ECA2_W_FILE          = "",
    parameter string CHAN_D1_W_FILE       = "",
    parameter string CHAN_D1_B_FILE       = "",
    parameter string CHAN_D2_W_FILE       = "",
    parameter string CHAN_D2_B_FILE       = "",
    parameter string SPAT_CONV_W_FILE     = "",
    parameter string TCFN_W0_FILE         = "",
    parameter string TCFN_B0_FILE         = "",
    parameter string TCFN_W1_FILE         = "",
    parameter string TCFN_B1_FILE         = "",
    parameter string TCFN_W2_FILE         = "",
    parameter string TCFN_B2_FILE         = "",
    parameter string TCFN_W3_FILE         = "",
    parameter string TCFN_B3_FILE         = "",
    parameter string CLS_W_FILE           = "",
    parameter string CLS_B_FILE           = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    // Streaming raw EEG: (in_sample, in_electrode) per cycle.
    input  logic                            in_valid,
    input  logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0] in_electrode,
    input  logic signed [DATA_WIDTH-1:0]    in_sample,

    // Final 1-bit class output.
    output logic [$clog2(N_CLS)-1:0]        class_out,
    output logic                            done
);

    // -------------------------------------------------------------------------
    // Stage 1: Conv2D + ECA₁ → streaming (F1)-wide gated vectors
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] eca1_y    [0:F1-1];
    logic                         eca1_y_valid;
    logic                         eca1_frame_last;
    logic                         eca1_done;

    db_atcnet_conv2d_eca1 #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .KE(KE_CONV2D), .T_IN(T_IN),
        .ECA1_KECA(3), .ECA1_INV_N(ECA1_INV_N),
        .SIG_LUT_N(SIG_LUT_N), .SIG_LUT_RANGE_Q(SIG_LUT_RANGE_Q), .SIG_LUT_SHIFT(SIG_LUT_SHIFT),
        .CONV2D_W_FILE(CONV2D_W_FILE), .CONV2D_B_FILE(CONV2D_B_FILE),
        .ECA1_W_FILE  (ECA1_W_FILE),   .SIG_LUT_FILE (SIG_LUT_FILE)
    ) u_conv_eca1 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_electrode(in_electrode), .in_sample(in_sample),
        .y_out(eca1_y), .y_valid(eca1_y_valid),
        .frame_last(eca1_frame_last), .done(eca1_done)
    );

    // -------------------------------------------------------------------------
    // Stage 2: Branch A + Branch B + sat_add + ECA₂ → eca2_buf (T_FULL × F)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] eca2_buf [0:T_FULL-1][0:F-1];
    logic                         post_eca1_done;

    db_atcnet_post_eca1 #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .F(F), .KE_SEP(KE_SEP),
        .T_IN(T_IN), .POOL1(POOL1), .POOL2(POOL2), .T_FULL(T_FULL),
        .INV_POOL1(INV_POOL1), .INV_POOL2(INV_POOL2), .ECA2_INV_N(ECA2_INV_N),
        .SIG_LUT_N(SIG_LUT_N), .SIG_LUT_RANGE_Q(SIG_LUT_RANGE_Q), .SIG_LUT_SHIFT(SIG_LUT_SHIFT),
        .ELU_LUT_N(ELU_LUT_N), .ELU_LUT_RANGE_Q(ELU_LUT_RANGE_Q), .ELU_LUT_SHIFT(ELU_LUT_SHIFT),
        .BRANCHA_DWISE_W_FILE(BRANCHA_DWISE_W_FILE), .BRANCHA_DWISE_B_FILE(BRANCHA_DWISE_B_FILE),
        .BRANCHA_SEP_W_FILE  (BRANCHA_SEP_W_FILE),   .BRANCHA_SEP_B_FILE  (BRANCHA_SEP_B_FILE),
        .BRANCHB_DWISE_W_FILE(BRANCHB_DWISE_W_FILE), .BRANCHB_DWISE_B_FILE(BRANCHB_DWISE_B_FILE),
        .BRANCHB_SEP_W_FILE  (BRANCHB_SEP_W_FILE),   .BRANCHB_SEP_B_FILE  (BRANCHB_SEP_B_FILE),
        .ECA2_W_FILE         (ECA2_W_FILE),
        .SIG_LUT_FILE        (SIG_LUT_FILE),
        .ELU_LUT_FILE        (ELU_LUT_FILE)
    ) u_post_eca1 (
        .clk(clk), .rst(rst),
        .x_in(eca1_y), .x_valid(eca1_y_valid), .frame_last(eca1_frame_last),
        .eca2_buf(eca2_buf), .done(post_eca1_done)
    );

    // -------------------------------------------------------------------------
    // Stage 3: 5-window pipeline → 1-bit class
    // -------------------------------------------------------------------------
    logic window_start;
    // post_eca1_done is a 1-cycle pulse; latch into a one-cycle start signal.
    always_ff @(posedge clk) begin
        if (rst)            window_start <= 1'b0;
        else if (post_eca1_done) window_start <= 1'b1;
        else                window_start <= 1'b0;
    end

    db_atcnet_window_pipeline #(
        .DATA_WIDTH(DATA_WIDTH),
        .F(F), .T_FULL(T_FULL), .T_WIN(T_WIN), .D_RED(4),
        .KH_SPAT(7), .IC_CAT(3), .K_TCFN(4), .N_CLS(N_CLS), .N_WIN(N_WIN),
        .SIG_LUT_N(SIG_LUT_N), .SIG_LUT_RANGE_Q(SIG_LUT_RANGE_Q), .SIG_LUT_SHIFT(SIG_LUT_SHIFT),
        .ELU_LUT_N(ELU_LUT_N), .ELU_LUT_RANGE_Q(ELU_LUT_RANGE_Q), .ELU_LUT_SHIFT(ELU_LUT_SHIFT),
        .CHAN_INV_N(CHAN_INV_N),
        .CHAN_D1_W_FILE(CHAN_D1_W_FILE), .CHAN_D1_B_FILE(CHAN_D1_B_FILE),
        .CHAN_D2_W_FILE(CHAN_D2_W_FILE), .CHAN_D2_B_FILE(CHAN_D2_B_FILE),
        .SIG_LUT_FILE  (SIG_LUT_FILE),
        .SPAT_CONV_W_FILE(SPAT_CONV_W_FILE),
        .TCFN_W0_FILE(TCFN_W0_FILE), .TCFN_B0_FILE(TCFN_B0_FILE),
        .TCFN_W1_FILE(TCFN_W1_FILE), .TCFN_B1_FILE(TCFN_B1_FILE),
        .TCFN_W2_FILE(TCFN_W2_FILE), .TCFN_B2_FILE(TCFN_B2_FILE),
        .TCFN_W3_FILE(TCFN_W3_FILE), .TCFN_B3_FILE(TCFN_B3_FILE),
        .ELU_LUT_FILE(ELU_LUT_FILE),
        .CLS_W_FILE(CLS_W_FILE), .CLS_B_FILE(CLS_B_FILE)
    ) u_window (
        .clk(clk), .rst(rst),
        .eca2_buf(eca2_buf), .start(window_start),
        .class_out(class_out), .done(done)
    );

endmodule
