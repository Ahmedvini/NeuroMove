// =============================================================================
// sat_add.sv  —  Per-channel saturating signed add of two vector streams.
//
// Layer 15 of HaLT DB-ATCNet: Add(BranchA_pool2, BranchB_pool2) over
// NUM_CH-wide Q8.8 vectors. Combinational; one input vector pair in,
// one output vector out, registered.
//
// Latency: 1 cycle (a_valid/b_valid synchronous in → y_valid out).
// =============================================================================

`timescale 1ns / 1ps

module sat_add #(
    parameter int DATA_WIDTH = 16,
    parameter int NUM_CH     = 32
) (
    input  logic                              clk,
    input  logic                              rst,
    input  logic signed [DATA_WIDTH-1:0]      a_in [0:NUM_CH-1],
    input  logic signed [DATA_WIDTH-1:0]      b_in [0:NUM_CH-1],
    input  logic                              in_valid,
    output logic signed [DATA_WIDTH-1:0]      y_out [0:NUM_CH-1],
    output logic                              y_valid
);

    localparam logic signed [DATA_WIDTH:0] HI =  (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH:0] LO = -(1 <<< (DATA_WIDTH-1));

    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int i = 0; i < NUM_CH; i++) y_out[i] <= '0;
        end else begin
            y_valid <= in_valid;
            for (int i = 0; i < NUM_CH; i++) begin
                logic signed [DATA_WIDTH:0] s;
                s = {a_in[i][DATA_WIDTH-1], a_in[i]} + {b_in[i][DATA_WIDTH-1], b_in[i]};
                if (s > HI)      y_out[i] <= HI[DATA_WIDTH-1:0];
                else if (s < LO) y_out[i] <= LO[DATA_WIDTH-1:0];
                else             y_out[i] <= s[DATA_WIDTH-1:0];
            end
        end
    end

endmodule
