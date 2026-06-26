// =============================================================================
// avg_pool_time.sv  —  Temporal AvgPool with stride = pool_size (no overlap).
//
// Used as Branch A pool1 (POOL=8) and pool2 (POOL=7) of HaLT DB-ATCNet,
// and symmetrically in Branch B. Operates on a streaming feature map of
// NUM_CH-wide vectors. Accumulates POOL consecutive vectors, then on the
// POOL-th input cycle emits the averaged output via the reciprocal-multiply:
//
//     out[c] = sat( (sum[c] * INV_POOL) >>> POOL_SHIFT )
//
// where INV_POOL = round(2^POOL_SHIFT / POOL). For POOL=8, POOL_SHIFT=24,
// INV_POOL = 2097152. For POOL=7, INV_POOL = 2396745.
//
// Latency: 1 cycle from the POOL-th input → y_valid.
//
// Parameters:
//   DATA_WIDTH   width of inputs/outputs (default 16, Q8.8)
//   NUM_CH       channels per vector (32 for Branch A; 32 for Branch B too
//                after its separable Conv2D)
//   ACC_WIDTH    accumulator width (default 32; needs DATA_WIDTH + log2(POOL)
//                + headroom)
//   POOL         pool size = stride (default 8)
//   INV_POOL     reciprocal multiplier
//   POOL_SHIFT   right-shift after the multiply (default 24)
//   INV_POOL_WIDTH bit-width of INV_POOL (default 24)
//   MUL_WIDTH    ACC_WIDTH + INV_POOL_WIDTH (default 56)
// =============================================================================

`timescale 1ns / 1ps

// Module-level (* use_dsp = "no" *) forces the per-channel reciprocal
// multiply (final_sum * INV_POOL) into LUTs/CARRY8 instead of DSP48E2.
// INV_POOL is a compile-time constant — for POOL=8 it's 2^21 (a shift),
// for POOL=7 it's 2396745 which constant-folds into ~60 LUTs per lane.
// Saves ~64 DSPs across the two pool2 instances at a cost of ~2-4k LUTs.
(* use_dsp = "no" *)
module avg_pool_time #(
    parameter int DATA_WIDTH     = 16,
    parameter int NUM_CH         = 32,
    parameter int ACC_WIDTH      = 32,
    parameter int POOL           = 8,
    parameter int INV_POOL       = 2097152,    // round(2^24 / 8)
    parameter int POOL_SHIFT     = 24,
    parameter int INV_POOL_WIDTH = 24,
    parameter int MUL_WIDTH      = ACC_WIDTH + INV_POOL_WIDTH
) (
    input  logic                            clk,
    input  logic                            rst,

    input  logic signed [DATA_WIDTH-1:0]    x_in [0:NUM_CH-1],
    input  logic                            x_valid,

    output logic signed [DATA_WIDTH-1:0]    y_out [0:NUM_CH-1],
    output logic                            y_valid
);

    localparam int PIDX_W = (POOL <= 1) ? 1 : $clog2(POOL);

    logic [PIDX_W-1:0]            p_idx;
    logic signed [ACC_WIDTH-1:0]  acc [0:NUM_CH-1];

    logic flush;                   // assert when this is the POOL-th input
    assign flush = x_valid && (p_idx == PIDX_W'(POOL - 1));

    // -------------------------------------------------------------------------
    // Sum-of-window combinational view (acc + current input)
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] final_sum [0:NUM_CH-1];
    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            final_sum[c] = (p_idx == '0)
                           ? $signed(x_in[c])
                           : (acc[c] + $signed(x_in[c]));
        end
    end

    // -------------------------------------------------------------------------
    // Running accumulator + counter
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            p_idx <= '0;
            for (int c = 0; c < NUM_CH; c++) acc[c] <= '0;
        end else if (x_valid) begin
            if (flush) begin
                p_idx <= '0;
                for (int c = 0; c < NUM_CH; c++) acc[c] <= '0;
            end else begin
                p_idx <= p_idx + 1'b1;
                for (int c = 0; c < NUM_CH; c++) acc[c] <= final_sum[c];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Reciprocal-multiply -> rescale -> saturate (registered)
    // -------------------------------------------------------------------------
    localparam logic signed [MUL_WIDTH-1:0] SAT_HI =
        (MUL_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [MUL_WIDTH-1:0] SAT_LO =
        -(MUL_WIDTH)'(1 <<< (DATA_WIDTH-1));

    // The module-level (* use_dsp = "no" *) attribute on `module
    // avg_pool_time` (above) forces these multiplies to LUTs.
    logic signed [MUL_WIDTH-1:0] prod    [0:NUM_CH-1];
    logic signed [MUL_WIDTH-1:0] shifted [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            prod[c]    = final_sum[c] * INV_POOL_WIDTH'(INV_POOL);
            shifted[c] = prod[c] >>> POOL_SHIFT;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) y_out[c] <= '0;
        end else begin
            y_valid <= flush;
            if (flush) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    if      (shifted[c] > SAT_HI) y_out[c] <= SAT_HI[DATA_WIDTH-1:0];
                    else if (shifted[c] < SAT_LO) y_out[c] <= SAT_LO[DATA_WIDTH-1:0];
                    else                          y_out[c] <= shifted[c][DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule
