// =============================================================================
// elu.sv  —  ELU activation via Q8.8 LUT (negative half only)
//
//     ELU(x) = x                          if x >= 0
//            = LUT[ min((-x) >> LUT_SHIFT, LUT_N-1) ]   if x < 0
//
// The LUT covers x in (-LUT_RANGE_Q88/256.0, 0]  with N entries spaced
// by 2^LUT_SHIFT input codes per bin. Loaded from LUT_FILE at elaboration
// via $readmemh; the LUT itself is universal and shared across all ELU
// instantiations in the design.
//
// Operates on a NUM_CH-wide streaming vector with 1-cycle latency.
//
// Parameters:
//   DATA_WIDTH    width of inputs / outputs (default 16, Q8.8)
//   NUM_CH        channels in the input vector (32 for Branch A post-depthwise)
//   LUT_N         number of LUT entries (default 256)
//   LUT_RANGE_Q   |x| magnitude beyond which output saturates to -alpha
//                 (default 2048 = -8.0 in Q8.8)
//   LUT_SHIFT     log2(LUT_RANGE_Q / LUT_N). With defaults: 3.
//   LUT_FILE      path to .mem with LUT_N signed Q8.8 ELU values
// =============================================================================

`timescale 1ns / 1ps

module elu #(
    parameter int    DATA_WIDTH  = 16,
    parameter int    NUM_CH      = 32,
    parameter int    LUT_N       = 256,
    parameter int    LUT_RANGE_Q = 2048,
    parameter int    LUT_SHIFT   = 3,
    parameter string LUT_FILE    = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    input  logic signed [DATA_WIDTH-1:0]    x_in [0:NUM_CH-1],
    input  logic                            x_valid,

    output logic signed [DATA_WIDTH-1:0]    y_out [0:NUM_CH-1],
    output logic                            y_valid
);

    // -------------------------------------------------------------------------
    // LUT load (negative-half table)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] LUT [0:LUT_N-1];
    initial begin
        if (LUT_FILE != "") $readmemh(LUT_FILE, LUT);
        else                for (int i = 0; i < LUT_N; i++) LUT[i] = '0;
    end

    // -------------------------------------------------------------------------
    // Combinational lookup
    // -------------------------------------------------------------------------
    localparam int ADDR_W = $clog2(LUT_N);

    logic signed [DATA_WIDTH:0]  neg_x   [0:NUM_CH-1];  // +1 bit headroom: -INT16_MIN is INT16_MAX+1
    logic        [ADDR_W:0]      raw_adr [0:NUM_CH-1];  // pre-clip
    logic        [ADDR_W-1:0]    adr     [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] cand   [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            neg_x[c]   = -$signed({{1{x_in[c][DATA_WIDTH-1]}}, x_in[c]});
            // For x < 0: addr = (-x) >> SHIFT; clip to [0, LUT_N-1]
            raw_adr[c] = ({1'b0, neg_x[c][DATA_WIDTH-1:0]}) >>> LUT_SHIFT;
            adr[c]     = (raw_adr[c] >= LUT_N) ? (LUT_N - 1) : raw_adr[c][ADDR_W-1:0];
            cand[c]    = (x_in[c] >= 0) ? x_in[c] : LUT[adr[c]];
        end
    end

    // -------------------------------------------------------------------------
    // Register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) y_out[c] <= '0;
        end else begin
            y_valid <= x_valid;
            if (x_valid) begin
                for (int c = 0; c < NUM_CH; c++) y_out[c] <= cand[c];
            end
        end
    end

endmodule
