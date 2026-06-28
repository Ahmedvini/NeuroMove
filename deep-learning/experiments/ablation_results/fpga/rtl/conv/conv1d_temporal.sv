// =============================================================================
// conv1d_temporal.sv  —  Temporal Conv2D (kernel KE × 1) on a flat F_IN-wide
// streaming feature map. Used for Branch A & B "separable" convolutions
// (Layer 7 and Layer 7'), where the spatial dim has already been collapsed.
//
// Math (matches scripts/q88_layer7.py bit-exactly, with BN folded):
//
//   tap[k]   = (k == KE-1) ? in_sample[*] : shift_reg[in_sample's f_in][k+1]
//             — i.e. tap[KE-1] is the newest fed input, tap[0] is the oldest.
//   acc[fo]  = sum_{k=0..KE-1} sum_{fi=0..F_IN-1} tap[k][fi] * W[k][fi][fo]
//   y[fo]    = sat( ( acc[fo] >>> FRAC_BITS ) + bias[fo] )
//
// Same-padding semantics handled by the caller: feed (KE-1)/2 zero
// pre-padding cycles, then T real cycles, then (KE - 1 - (KE-1)/2) zero
// post-padding cycles. The output for model time t_out = T_in - (KE-1) is
// the registered value at cycle T_in + 1.
//
// Latency: 1 cycle from in_valid → y_valid.
// Resources (fully parallel): F_OUT * F_IN * KE multipliers. For
// F_IN=F_OUT=32, KE=16 → 16,384 multipliers (Ultrascale+ scale).
// Add a TIME_MUX parameter in a follow-up to time-multiplex for Z-7020.
//
// Parameters:
//   DATA_WIDTH   sample width (default 16, Q8.8)
//   COEF_WIDTH   weight width  (default 16)
//   ACC_WIDTH    accumulator width (default 48 — DATA+COEF+log2(KE*F_IN)+headroom)
//   FRAC_BITS    fractional bits in Q-format (default 8)
//   F_IN         number of input channels per cycle (default 32)
//   F_OUT        number of output channels per cycle (default 32)
//   KE           kernel size (default 16)
//   WEIGHTS_FILE path to KE*F_IN*F_OUT signed Q8.8 entries, in (k slow, fi, fo fast) order
//   BIAS_FILE    path to F_OUT signed Q8.8 entries (BN-folded bias)
// =============================================================================

`timescale 1ns / 1ps

module conv1d_temporal #(
    parameter int    DATA_WIDTH   = 16,
    parameter int    COEF_WIDTH   = 16,
    parameter int    ACC_WIDTH    = 48,
    parameter int    FRAC_BITS    = 8,
    parameter int    F_IN         = 32,
    parameter int    F_OUT        = 32,
    parameter int    KE           = 16,
    parameter string WEIGHTS_FILE = "",
    parameter string BIAS_FILE    = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    input  logic signed [DATA_WIDTH-1:0]    in_sample [0:F_IN-1],
    input  logic                            in_valid,

    output logic signed [DATA_WIDTH-1:0]    y_out [0:F_OUT-1],
    output logic                            y_valid
);

    // -------------------------------------------------------------------------
    // Weights + bias load
    // -------------------------------------------------------------------------
    logic signed [COEF_WIDTH-1:0] w_flat [0:KE*F_IN*F_OUT-1];
    logic signed [COEF_WIDTH-1:0] W      [0:KE-1][0:F_IN-1][0:F_OUT-1];
    logic signed [DATA_WIDTH-1:0] B      [0:F_OUT-1];

    initial begin
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, w_flat);
        else                    for (int i = 0; i < KE*F_IN*F_OUT; i++) w_flat[i] = '0;
        for (int k = 0; k < KE; k++)
            for (int fi = 0; fi < F_IN; fi++)
                for (int fo = 0; fo < F_OUT; fo++)
                    W[k][fi][fo] = w_flat[k * F_IN * F_OUT + fi * F_OUT + fo];
        if (BIAS_FILE != "") $readmemh(BIAS_FILE, B);
        else                 for (int fo = 0; fo < F_OUT; fo++) B[fo] = '0;
    end

    // -------------------------------------------------------------------------
    // KE-deep shift register, each slot is an F_IN-wide vector
    // shift_reg[KE-1] = newest cell already in the register; tap[KE-1] is the
    // new in_sample (post-update view).
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] shift_reg [0:KE-1][0:F_IN-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int k = 0; k < KE; k++)
                for (int fi = 0; fi < F_IN; fi++)
                    shift_reg[k][fi] <= '0;
        end else if (in_valid) begin
            for (int fi = 0; fi < F_IN; fi++)
                shift_reg[KE-1][fi] <= in_sample[fi];
            for (int k = 0; k < KE - 1; k++)
                for (int fi = 0; fi < F_IN; fi++)
                    shift_reg[k][fi] <= shift_reg[k+1][fi];
        end
    end

    logic signed [DATA_WIDTH-1:0] tap [0:KE-1][0:F_IN-1];
    always_comb begin
        for (int k = 0; k < KE - 1; k++)
            for (int fi = 0; fi < F_IN; fi++)
                tap[k][fi] = shift_reg[k+1][fi];
        for (int fi = 0; fi < F_IN; fi++)
            tap[KE-1][fi] = in_sample[fi];
    end

    // -------------------------------------------------------------------------
    // F_OUT parallel MACs over KE × F_IN products
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc [0:F_OUT-1];

    always_comb begin
        for (int fo = 0; fo < F_OUT; fo++) begin
            acc[fo] = '0;
            for (int k = 0; k < KE; k++)
                for (int fi = 0; fi < F_IN; fi++)
                    acc[fo] = acc[fo] + $signed(tap[k][fi]) * $signed(W[k][fi][fo]);
        end
    end

    // -------------------------------------------------------------------------
    // Rescale + bias + saturation
    // -------------------------------------------------------------------------
    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int fo = 0; fo < F_OUT; fo++) y_out[fo] <= '0;
        end else begin
            y_valid <= in_valid;
            for (int fo = 0; fo < F_OUT; fo++) begin
                logic signed [ACC_WIDTH-1:0] shifted;
                logic signed [ACC_WIDTH-1:0] with_bias;
                shifted   = acc[fo] >>> FRAC_BITS;
                with_bias = shifted +
                            $signed({{(ACC_WIDTH-DATA_WIDTH){B[fo][DATA_WIDTH-1]}}, B[fo]});
                if      (with_bias > SAT_HI) y_out[fo] <= SAT_HI[DATA_WIDTH-1:0];
                else if (with_bias < SAT_LO) y_out[fo] <= SAT_LO[DATA_WIDTH-1:0];
                else                         y_out[fo] <= with_bias[DATA_WIDTH-1:0];
            end
        end
    end

endmodule
