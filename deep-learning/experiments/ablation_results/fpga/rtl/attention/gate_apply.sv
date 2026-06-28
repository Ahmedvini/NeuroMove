// =============================================================================
// gate_apply.sv  —  Per-channel multiplier that applies an ECA/CBAM gate to a
// streaming feature map.
//
//     y[c] = sat( x[c] * gate[c] >>> FRAC_BITS )
//
// Used after gap_accumulator -> eca_attention to apply the computed gates
// back to the (re-streamed or buffered) feature map. The gate vector is held
// constant across a frame; only x_in changes per cycle.
//
// Latency: 1 cycle (combinational multiply + registered saturate).
//
// Parameters:
//   DATA_WIDTH   width of input/output samples and gate (default 16)
//   NUM_CH       channel count of the input vector
//   FRAC_BITS    fractional bits in Q-format (default 8 → Q8.8 ; gate is
//                Q0.FRAC_BITS in [0, 1] so the product is at most ~|x|)
//   ACC_WIDTH    intermediate width (default 32 = 16+16; ample headroom)
// =============================================================================

`timescale 1ns / 1ps

module gate_apply #(
    parameter int DATA_WIDTH = 16,
    parameter int NUM_CH     = 16,
    parameter int FRAC_BITS  = 8,
    parameter int ACC_WIDTH  = 32
) (
    input  logic                            clk,
    input  logic                            rst,

    // Streaming feature map (one vector per cycle).
    input  logic signed [DATA_WIDTH-1:0]    x_in     [0:NUM_CH-1],
    input  logic                            x_valid,

    // Per-channel gate vector. Driver should hold it stable across all the
    // cycles of a frame.
    input  logic signed [DATA_WIDTH-1:0]    gate_in  [0:NUM_CH-1],

    output logic signed [DATA_WIDTH-1:0]    y_out    [0:NUM_CH-1],
    output logic                            y_valid
);

    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    logic signed [ACC_WIDTH-1:0] prod    [0:NUM_CH-1];
    logic signed [ACC_WIDTH-1:0] shifted [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            prod[c]    = $signed(x_in[c]) * $signed(gate_in[c]);
            shifted[c] = prod[c] >>> FRAC_BITS;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) y_out[c] <= '0;
        end else begin
            y_valid <= x_valid;
            if (x_valid) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    if      (shifted[c] > SAT_HI) y_out[c] <= SAT_HI[DATA_WIDTH-1:0];
                    else if (shifted[c] < SAT_LO) y_out[c] <= SAT_LO[DATA_WIDTH-1:0];
                    else                          y_out[c] <= shifted[c][DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule
