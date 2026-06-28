// =============================================================================
// db_atcnet_conv2d_eca1.sv  —  Conv2D + ECA₁ wrapper for HaLT DB-ATCNet.
//
//   Streaming input  (in_sample, in_electrode) for T_TOTAL × C_IN cycles
//     where T_TOTAL = PAD_TOP + T_IN + PAD_BOT and PAD_TOP=(KE-1)/2.
//   →   conv2d_temporal (1-cycle latency) produces (F1)-wide outputs per cycle.
//   →   Drop the first KE-1=63 warmup outputs per electrode.
//   →   Feed the remaining T_IN × C_IN = 3000 outputs into eca1_pipeline.
//   ←   Stream out: (F1)-wide gated ECA₁ vectors per (t, c) cycle (3000 total)
//       with y_valid and frame_last (asserted on the C_IN-th electrode of
//       each timestep) — the exact format db_atcnet_post_eca1 consumes.
//
// Internally the conv2d_temporal output is filtered by a small counter
// (`out_seq`) and the valid range is gated into the ECA₁ pipeline.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_conv2d_eca1 #(
    parameter int    DATA_WIDTH = 16,
    parameter int    C_IN       = 5,
    parameter int    F1         = 16,
    parameter int    KE         = 64,
    parameter int    T_IN       = 600,
    parameter int    ECA1_KECA  = 3,
    parameter int    ECA1_INV_N = 5592,        // round(2^24 / (T_IN * C_IN))
    parameter int    SIG_LUT_N        = 1024,
    parameter int    SIG_LUT_RANGE_Q  = 1024,
    parameter int    SIG_LUT_SHIFT    = 1,
    parameter string CONV2D_W_FILE    = "",
    parameter string CONV2D_B_FILE    = "",
    parameter string ECA1_W_FILE      = "",
    parameter string SIG_LUT_FILE     = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    // Streaming raw EEG (post-Gumbel-selection): one (in_sample, in_electrode)
    // per cycle. Caller drives T_TOTAL × C_IN cycles.
    input  logic                            in_valid,
    input  logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0] in_electrode,
    input  logic signed [DATA_WIDTH-1:0]    in_sample,

    // Streaming ECA₁-gated output (one (F1)-wide vector per (t, c) cycle).
    output logic signed [DATA_WIDTH-1:0]    y_out [0:F1-1],
    output logic                            y_valid,
    output logic                            frame_last,
    output logic                            done
);

    localparam int PAD_TOP = (KE - 1) / 2;
    localparam int N_FRAMES = T_IN * C_IN;

    // -------------------------------------------------------------------------
    // Conv2D — streaming, 1-cycle latency.
    // -------------------------------------------------------------------------
    logic                                    conv_out_valid;
    logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0]  conv_out_electrode;
    logic signed [DATA_WIDTH-1:0]            conv_out_filter [0:F1-1];

    conv2d_temporal #(
        .NUM_EEG_CH(C_IN), .F1(F1), .KE(KE),
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .ACC_WIDTH(48), .FRAC_BITS(8),
        .WEIGHTS_FILE(CONV2D_W_FILE), .BIAS_FILE(CONV2D_B_FILE)
    ) u_conv2d (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_electrode(in_electrode), .in_sample(in_sample),
        .out_valid(conv_out_valid),
        .out_electrode(conv_out_electrode),
        .out_filter(conv_out_filter)
    );

    // -------------------------------------------------------------------------
    // Filter the warmup outputs. The output sequence index advances per
    // conv_out_valid pulse. Each `t_in` produces C_IN outputs.
    // Valid t_in range: [KE-1, KE-1 + T_IN) → t_out = t_in - (KE-1).
    // -------------------------------------------------------------------------
    logic [$clog2((PAD_TOP + T_IN + PAD_TOP) * C_IN + 1)-1:0] out_seq;
    logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0]                  e_seq;
    logic [$clog2(T_IN + PAD_TOP + 2)-1:0]                   t_in_seq;

    // Combinational view of (t_in, e) for current output.
    wire is_valid_out_window = (t_in_seq >= (KE - 1))
                            && (t_in_seq <  (KE - 1) + T_IN);
    wire eca1_x_valid_in     = conv_out_valid && is_valid_out_window;
    wire eca1_frame_last_in  = eca1_x_valid_in && (e_seq == C_IN - 1);

    always_ff @(posedge clk) begin
        if (rst) begin
            out_seq  <= 0;
            e_seq    <= 0;
            t_in_seq <= 0;
        end else if (conv_out_valid) begin
            if (e_seq == C_IN - 1) begin
                e_seq    <= 0;
                t_in_seq <= t_in_seq + 1;
            end else begin
                e_seq <= e_seq + 1;
            end
            out_seq <= out_seq + 1;
        end
    end

    // -------------------------------------------------------------------------
    // ECA₁ pipeline — N_FRAMES = 3000, NUM_CH = F1 = 16.
    // -------------------------------------------------------------------------
    eca1_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F1), .KECA(ECA1_KECA),
        .N_FRAMES(N_FRAMES),
        .INV_N(ECA1_INV_N), .INV_N_SHIFT(24), .INV_N_WIDTH(24),
        .LUT_N(SIG_LUT_N), .LUT_RANGE_Q(SIG_LUT_RANGE_Q), .LUT_SHIFT(SIG_LUT_SHIFT),
        .WEIGHTS_FILE(ECA1_W_FILE), .LUT_FILE(SIG_LUT_FILE)
    ) u_eca1 (
        .clk(clk), .rst(rst),
        .x_in(conv_out_filter), .x_valid(eca1_x_valid_in),
        .y_out(y_out), .y_valid(y_valid), .done(done)
    );

    // -------------------------------------------------------------------------
    // frame_last: combinational, asserted when y_valid pulses for the C_IN-th
    // electrode of a timestep. A local mod-C_IN counter advances per y_valid.
    // -------------------------------------------------------------------------
    logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0] out_e_cnt;

    always_ff @(posedge clk) begin
        if (rst) out_e_cnt <= 0;
        else if (y_valid) begin
            out_e_cnt <= (out_e_cnt == C_IN - 1) ? '0 : out_e_cnt + 1;
        end
    end

    assign frame_last = y_valid && (out_e_cnt == C_IN - 1);

endmodule
