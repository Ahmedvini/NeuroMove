// =============================================================================
// gap_accumulator.sv  —  Streaming Global Average Pool over (T, C) per channel
//
// Consumes a feature map streamed as one NUM_CH-wide vector per cycle. On the
// last accepted vector (frame_last asserted with x_valid), it converts each
// per-channel running sum into a Q8.8 GAP value via the reciprocal-multiply
// trick (no integer divider):
//
//     gap_out[f] = sat( (sum[f] * INV_N) >>> INV_N_SHIFT )
//
// where INV_N = round(2^INV_N_SHIFT / N), N = number of vectors accumulated.
// This matches scripts/q88_eca1.py bit-exactly when INV_N=5592, SHIFT=24 for
// N=3000 (ECA₁ position over T=600 × C=5).
//
// Latency: 1 cycle from frame_last → gap_valid.
//
// Parameters:
//   DATA_WIDTH     Q-format width of feature-map elements (default 16).
//   NUM_CH         Number of channels in the input vector (16 for ECA₁).
//   ACC_WIDTH      Accumulator width (default 32). Must hold max |x| * N.
//                  For DATA_WIDTH=16, N≤32k: 31 bits is enough; use 32.
//   INV_N          Reciprocal multiplier; choose so INV_N/(2^INV_N_SHIFT) ≈ 1/N.
//                  Default 5592 = round(2^24/3000) for ECA₁ N=3000.
//   INV_N_SHIFT    Right shift after the multiply (default 24).
//   INV_N_WIDTH    Bit-width of INV_N (default 24 covers up to N≥1).
//   MUL_WIDTH      ACC_WIDTH + INV_N_WIDTH (default 56).
// =============================================================================

`timescale 1ns / 1ps

module gap_accumulator #(
    parameter int DATA_WIDTH  = 16,
    parameter int NUM_CH      = 16,
    parameter int ACC_WIDTH   = 32,
    parameter int INV_N       = 5592,    // round(2^24 / 3000)
    parameter int INV_N_SHIFT = 24,
    parameter int INV_N_WIDTH = 24,
    parameter int MUL_WIDTH   = ACC_WIDTH + INV_N_WIDTH
) (
    input  logic                            clk,
    input  logic                            rst,

    input  logic signed [DATA_WIDTH-1:0]    x_in [0:NUM_CH-1],
    input  logic                            x_valid,
    input  logic                            frame_last,    // assert with last x_valid

    output logic signed [DATA_WIDTH-1:0]    gap_out [0:NUM_CH-1],
    output logic                            gap_valid
);

    // -------------------------------------------------------------------------
    // Per-channel accumulator
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc [0:NUM_CH-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int c = 0; c < NUM_CH; c++) acc[c] <= '0;
        end else if (x_valid && frame_last) begin
            // Reset for the *next* frame: accumulator is fully consumed this
            // cycle, so reload from the just-received vector.
            for (int c = 0; c < NUM_CH; c++) acc[c] <= $signed(x_in[c]);
        end else if (x_valid) begin
            for (int c = 0; c < NUM_CH; c++)
                acc[c] <= acc[c] + $signed(x_in[c]);
        end
    end

    // The accumulator value AT END OF FRAME (i.e. when frame_last pulses with
    // x_valid) is the old acc plus the current input — we compute that view
    // combinationally so the GAP output is ready exactly 1 cycle later.
    logic signed [ACC_WIDTH-1:0] final_sum [0:NUM_CH-1];
    always_comb begin
        for (int c = 0; c < NUM_CH; c++)
            final_sum[c] = acc[c] + $signed(x_in[c]);
    end

    // -------------------------------------------------------------------------
    // Reciprocal-multiply -> rescale -> saturate
    // -------------------------------------------------------------------------
    localparam logic signed [MUL_WIDTH-1:0] SAT_HI =
        (MUL_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [MUL_WIDTH-1:0] SAT_LO =
        -(MUL_WIDTH)'(1 <<< (DATA_WIDTH-1));

    logic signed [MUL_WIDTH-1:0] prod    [0:NUM_CH-1];
    logic signed [MUL_WIDTH-1:0] shifted [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            prod[c]    = final_sum[c] * INV_N_WIDTH'(INV_N);
            shifted[c] = prod[c] >>> INV_N_SHIFT;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            gap_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) gap_out[c] <= '0;
        end else begin
            gap_valid <= (x_valid && frame_last);
            if (x_valid && frame_last) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    if      (shifted[c] > SAT_HI) gap_out[c] <= SAT_HI[DATA_WIDTH-1:0];
                    else if (shifted[c] < SAT_LO) gap_out[c] <= SAT_LO[DATA_WIDTH-1:0];
                    else                          gap_out[c] <= shifted[c][DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule
