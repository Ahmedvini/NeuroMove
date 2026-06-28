// =============================================================================
// depthwise_spatial.sv  —  Depthwise Conv2D with kernel (1, C_IN), depth_multiplier=D.
//
// Used as Layer 4 (Branch A, D=2) and Layer 4' (Branch B, D=4) of HaLT
// DB-ATCNet. Operates on a streaming (t, c) feature map where each cycle
// presents the F1-wide input filter vector for one electrode in one time step.
//
// Math (matches scripts/q88_layer4.py bit-exactly, with BN folded into the
// kernel + bias):
//
//   For each output channel  i = f*D + d   (f ∈ [0,F1), d ∈ [0,D)):
//       y[t, i] = sat( ( sum_{c=0..C_IN-1}  x[t, c, f] * W[c, i] )
//                      >>> FRAC_BITS  +  bias[i] )
//
// Pipeline:
//   - The accumulator for each output channel i adds the new product on every
//     accepted (t, c) pair.
//   - On the LAST electrode of a time step (signalled by frame_last), the
//     accumulator's value (acc + current product) is rescaled, biased,
//     saturated and emitted via y_out[i] / y_valid (1 cycle later).
//
// Weights layout (loaded once at elaboration via $readmemh, "c-slow, i-fast"):
//     address = c * F2 + i        where F2 = F1 * D
//
// Latency: 1 cycle from the last-electrode beat → y_valid.
//
// Parameters:
//   DATA_WIDTH    Q-format width of inputs and outputs (default 16).
//   COEF_WIDTH    Q-format width of weights (default 16).
//   ACC_WIDTH     Accumulator width (default 48 — DATA+COEF+log2(C_IN)+headroom).
//   FRAC_BITS     Fractional bits in Q-format (default 8 → Q8.8).
//   C_IN          Kernel spatial width = number of electrodes (default 5).
//   F1            Number of input filters per electrode (default 16).
//   D             Depth multiplier (default 2 for Branch A; 4 for Branch B).
//   F2            Output channel count = F1 * D (default 32).
//   WEIGHTS_FILE  Path to C_IN*F2 signed Q8.8 entries, layout above.
//   BIAS_FILE     Path to F2 signed Q8.8 entries.
// =============================================================================

`timescale 1ns / 1ps

(* use_dsp = "no" *)
module depthwise_spatial #(
    parameter int    DATA_WIDTH   = 16,
    parameter int    COEF_WIDTH   = 16,
    parameter int    ACC_WIDTH    = 48,
    parameter int    FRAC_BITS    = 8,
    parameter int    C_IN         = 5,
    parameter int    F1           = 16,
    parameter int    D            = 2,
    parameter int    F2           = F1 * D,
    parameter string WEIGHTS_FILE = "",
    parameter string BIAS_FILE    = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    // Streaming feature map: one F1-wide vector per (t, c) pair when valid.
    input  logic signed [DATA_WIDTH-1:0]    x_in [0:F1-1],
    input  logic                            x_valid,

    // frame_last == 1 on the cycle that the LAST electrode (c == C_IN-1) of
    // the current time step is presented. The accumulator delivers that
    // time step's F2-vector one cycle later.
    input  logic                            frame_last,

    output logic signed [DATA_WIDTH-1:0]    y_out [0:F2-1],
    output logic                            y_valid
);

    // -------------------------------------------------------------------------
    // Weights + bias load
    // -------------------------------------------------------------------------
    logic signed [COEF_WIDTH-1:0] w_flat [0:C_IN*F2-1];
    logic signed [COEF_WIDTH-1:0] W      [0:C_IN-1][0:F2-1];
    logic signed [DATA_WIDTH-1:0] B      [0:F2-1];

    initial begin
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, w_flat);
        else                    for (int i = 0; i < C_IN*F2; i++) w_flat[i] = '0;
        for (int c = 0; c < C_IN; c++)
            for (int i = 0; i < F2; i++)
                W[c][i] = w_flat[c * F2 + i];
        if (BIAS_FILE != "") $readmemh(BIAS_FILE, B);
        else                 for (int i = 0; i < F2; i++) B[i] = '0;
    end

    // -------------------------------------------------------------------------
    // Per-electrode counter to pick the right kernel column.
    // We DO NOT take this from the caller — it's implicit in the cycle index.
    // c_idx wraps from 0..C_IN-1 across consecutive x_valid pulses, with
    // frame_last asserting on c_idx == C_IN-1.
    // -------------------------------------------------------------------------
    localparam int CIDX_W = (C_IN <= 1) ? 1 : $clog2(C_IN);
    logic [CIDX_W-1:0] c_idx;

    always_ff @(posedge clk) begin
        if (rst) c_idx <= '0;
        else if (x_valid) begin
            if (frame_last) c_idx <= '0;
            else            c_idx <= c_idx + 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Per-output-channel running accumulator. For each i = f*D + d:
    //   acc[i] += x_in[f] * W[c_idx][i]
    //
    // Reset to zero on the FIRST electrode of a new time step (i.e. AFTER a
    // frame_last cycle). On the frame_last cycle itself, we use the
    // combinational "with the current product" value for the output, so the
    // sequential `acc` doesn't need to hold the final sum past one cycle.
    // -------------------------------------------------------------------------
    function automatic int idx_to_f(input int i);
        return i / D;
    endfunction

    logic signed [ACC_WIDTH-1:0] acc       [0:F2-1];
    logic signed [ACC_WIDTH-1:0] acc_next  [0:F2-1];
    logic signed [ACC_WIDTH-1:0] final_sum [0:F2-1];

    always_comb begin
        for (int i = 0; i < F2; i++) begin
            logic signed [ACC_WIDTH-1:0] prod;
            prod = $signed(x_in[idx_to_f(i)]) * $signed(W[c_idx][i]);
            // running accumulator update
            acc_next[i]  = (c_idx == '0) ? prod : (acc[i] + prod);
            // value to ship on frame_last (always includes the current product)
            final_sum[i] = (c_idx == '0) ? prod : (acc[i] + prod);
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < F2; i++) acc[i] <= '0;
        end else if (x_valid && !frame_last) begin
            for (int i = 0; i < F2; i++) acc[i] <= acc_next[i];
        end
        // On frame_last: don't bother updating acc (the next valid is c_idx=0
        // and will overwrite).
    end

    // -------------------------------------------------------------------------
    // Rescale + bias + saturation (latched output)
    // -------------------------------------------------------------------------
    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int i = 0; i < F2; i++) y_out[i] <= '0;
        end else begin
            y_valid <= (x_valid && frame_last);
            if (x_valid && frame_last) begin
                for (int i = 0; i < F2; i++) begin
                    logic signed [ACC_WIDTH-1:0] shifted;
                    logic signed [ACC_WIDTH-1:0] with_bias;
                    shifted   = final_sum[i] >>> FRAC_BITS;
                    with_bias = shifted +
                                $signed({{(ACC_WIDTH-DATA_WIDTH){B[i][DATA_WIDTH-1]}}, B[i]});
                    if      (with_bias > SAT_HI) y_out[i] <= SAT_HI[DATA_WIDTH-1:0];
                    else if (with_bias < SAT_LO) y_out[i] <= SAT_LO[DATA_WIDTH-1:0];
                    else                         y_out[i] <= with_bias[DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule
