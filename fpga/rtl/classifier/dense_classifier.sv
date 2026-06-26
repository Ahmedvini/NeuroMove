// =============================================================================
// dense_classifier.sv  —  Per-window Dense(F → N_CLS) classifier.
//
//   logits[o] = sat( (sum_i x[i] * W[o, i]) >>> FRAC_BITS + b[o] )
//
// Weights loaded via $readmemh from WEIGHTS_FILE (row-major (Nout, Nin)),
// bias from BIAS_FILE (Nout entries). 1-cycle combinational MAC array
// (Nin*Nout multiplies, fully parallel — for F=32, N_CLS=2 that's 64
// 16x16-bit DSPs, fine for both Z7020 and US+).
//
// Latency: 1 cycle from `in_valid` to `out_valid`.
// =============================================================================

`timescale 1ns / 1ps

module dense_classifier #(
    parameter int    DATA_WIDTH = 16,
    parameter int    COEF_WIDTH = 16,
    parameter int    ACC_WIDTH  = 40,
    parameter int    FRAC_BITS  = 8,
    parameter int    F          = 32,
    parameter int    N_CLS      = 2,
    parameter int    N_WIN      = 1,
    parameter string WEIGHTS_FILE = "",
    parameter string BIAS_FILE    = ""
) (
    input  logic                              clk,
    input  logic                              rst,

    input  logic [$clog2(N_WIN > 1 ? N_WIN : 2)-1:0] window_idx,

    input  logic signed [DATA_WIDTH-1:0]      x_in [0:F-1],
    input  logic                              in_valid,

    output logic signed [DATA_WIDTH-1:0]      logits   [0:N_CLS-1],
    output logic                              out_valid
);

    localparam int W_STRIDE = N_CLS * F;
    logic signed [COEF_WIDTH-1:0] W [0:N_WIN * W_STRIDE - 1];
    logic signed [DATA_WIDTH-1:0] B [0:N_WIN * N_CLS   - 1];
    initial begin
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, W);
        if (BIAS_FILE    != "") $readmemh(BIAS_FILE,    B);
    end
    wire [31:0] w_base = window_idx * W_STRIDE;
    wire [31:0] b_base = window_idx * N_CLS;

    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    function automatic logic signed [DATA_WIDTH-1:0] sat_q88 (input logic signed [ACC_WIDTH-1:0] v);
        if      (v > SAT_HI) return SAT_HI[DATA_WIDTH-1:0];
        else if (v < SAT_LO) return SAT_LO[DATA_WIDTH-1:0];
        else                 return v[DATA_WIDTH-1:0];
    endfunction

    // -------------------------------------------------------------------------
    // 2-stage pipelined MAC.
    //
    // The combinational F=32-tap dot-product + window_idx-gated weight lookup
    // was the WNS path (~24 ns at 100 MHz). Splitting the dot product into
    // ACC_SEGS parts (8 multiplies each) with a register between caps each
    // combinational stage at ~7-8 ns.
    //
    // Latency: 2 cycles from in_valid → out_valid (was 1). Downstream
    // db_atcnet_window_pipeline reads logits one extra cycle later; verified
    // by the `pipe` regression.
    // -------------------------------------------------------------------------
    localparam int ACC_SEGS    = 4;
    localparam int ACC_SEG_LEN = F / ACC_SEGS;

    logic signed [ACC_WIDTH-1:0] acc_part     [0:N_CLS-1][0:ACC_SEGS-1];
    logic signed [ACC_WIDTH-1:0] acc_part_reg [0:N_CLS-1][0:ACC_SEGS-1];
    logic [31:0]                 b_base_reg;
    logic                        in_valid_d1;

    always_comb begin
        for (int o = 0; o < N_CLS; o++) begin
            for (int s = 0; s < ACC_SEGS; s++) acc_part[o][s] = '0;
            for (int i = 0; i < F; i++) begin
                automatic int s = i / ACC_SEG_LEN;
                acc_part[o][s] = acc_part[o][s]
                               + $signed(x_in[i]) * $signed(W[w_base + o * F + i]);
            end
        end
    end

    // Late, combined view from registered partials.
    logic signed [ACC_WIDTH-1:0] shifted [0:N_CLS-1];
    always_comb begin
        for (int o = 0; o < N_CLS; o++) begin
            automatic logic signed [ACC_WIDTH-1:0] acc_full;
            acc_full = '0;
            for (int s = 0; s < ACC_SEGS; s++)
                acc_full = acc_full + acc_part_reg[o][s];
            shifted[o] = (acc_full >>> FRAC_BITS)
                       + $signed({{(ACC_WIDTH-DATA_WIDTH){B[b_base_reg + o][DATA_WIDTH-1]}}, B[b_base_reg + o]});
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid   <= 1'b0;
            in_valid_d1 <= 1'b0;
            b_base_reg  <= '0;
            for (int o = 0; o < N_CLS; o++) begin
                logits[o] <= '0;
                for (int s = 0; s < ACC_SEGS; s++) acc_part_reg[o][s] <= '0;
            end
        end else begin
            // Stage 0 → stage 1 pipeline: latch partials + bias-base + valid.
            in_valid_d1 <= in_valid;
            b_base_reg  <= b_base;
            for (int o = 0; o < N_CLS; o++)
                for (int s = 0; s < ACC_SEGS; s++)
                    acc_part_reg[o][s] <= acc_part[o][s];

            // Stage 1 → output: sum partials, add bias, saturate, register.
            out_valid <= in_valid_d1;
            if (in_valid_d1) begin
                for (int o = 0; o < N_CLS; o++) logits[o] <= sat_q88(shifted[o]);
            end
        end
    end

endmodule
