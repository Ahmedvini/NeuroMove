// =============================================================================
// conv1d_temporal_tm.sv  —  Time-multiplexed Conv1D (kernel KE × 1) on a flat
// F_IN-wide streaming feature map. Functionally identical to the original
// conv1d_temporal.sv (bit-exact) but reuses a single F_OUT-wide MAC bank
// across KE * F_IN cycles per input sample, so DSP count drops from
//   F_OUT * F_IN * KE  →  F_OUT   (e.g. 16,384 → 32 for Branch A sep).
//
// Latency per input sample:
//   1 cycle (accept) + KE * F_IN cycles (MAC) + 1 cycle (emit) = KE*F_IN + 2.
//   For F_IN=32, KE=16 that's 514 cycles per (t, c) input.
//
// Handshake: assert `in_valid` while `in_ready` is high to push a new
// sample. The module deasserts `in_ready` during compute. `y_valid` pulses
// for one cycle when the result for the accepted sample is ready.
//
// Bit-exact match to scripts/q88_layer7.py.
// =============================================================================

`timescale 1ns / 1ps

module conv1d_temporal_tm #(
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
    output logic                            in_ready,

    output logic signed [DATA_WIDTH-1:0]    y_out [0:F_OUT-1],
    output logic                            y_valid
);

    // -------------------------------------------------------------------------
    // Weights + bias load. w_flat layout: (k slow, fi, fo fast).
    //   W[k][fi][fo] = w_flat[k * F_IN * F_OUT + fi * F_OUT + fo]
    // -------------------------------------------------------------------------
    logic signed [COEF_WIDTH-1:0] W_FLAT [0:KE * F_IN * F_OUT - 1];
    logic signed [DATA_WIDTH-1:0] B      [0:F_OUT - 1];

    initial begin
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, W_FLAT);
        if (BIAS_FILE    != "") $readmemh(BIAS_FILE,    B);
    end

    // -------------------------------------------------------------------------
    // KE-deep shift register: each slot is an F_IN-wide vector.
    // After update at cycle T: shift_reg[k] for k=0..KE-1 is the same view as
    // the original module's tap[k] (k=0 oldest, k=KE-1 newest).
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] shift_reg [0:KE-1][0:F_IN-1];

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {S_IDLE, S_COMP, S_EMIT} state_t;
    state_t state;

    logic [$clog2(KE)-1:0]    k_idx;
    logic [$clog2(F_IN)-1:0]  fi_idx;
    logic signed [ACC_WIDTH-1:0] acc [0:F_OUT-1];

    // Combinational F_OUT-wide product for the current (k_idx, fi_idx).
    logic signed [ACC_WIDTH-1:0] prod [0:F_OUT-1];
    logic signed [DATA_WIDTH-1:0] tap_val;
    always_comb begin
        tap_val = shift_reg[k_idx][fi_idx];
        for (int fo = 0; fo < F_OUT; fo++) begin
            prod[fo] = $signed(tap_val)
                     * $signed(W_FLAT[k_idx * F_IN * F_OUT + fi_idx * F_OUT + fo]);
        end
    end

    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    always_ff @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            in_ready <= 1'b1;
            y_valid  <= 1'b0;
            k_idx    <= 0;
            fi_idx   <= 0;
            for (int k = 0; k < KE; k++)
                for (int fi = 0; fi < F_IN; fi++)
                    shift_reg[k][fi] <= '0;
            for (int fo = 0; fo < F_OUT; fo++) begin
                acc[fo]   <= '0;
                y_out[fo] <= '0;
            end
        end else begin
            y_valid <= 1'b0;

            unique case (state)
                S_IDLE: begin
                    if (in_valid && in_ready) begin
                        // Shift register: oldest falls off, new arrives at KE-1.
                        for (int k = 0; k < KE - 1; k++)
                            for (int fi = 0; fi < F_IN; fi++)
                                shift_reg[k][fi] <= shift_reg[k+1][fi];
                        for (int fi = 0; fi < F_IN; fi++)
                            shift_reg[KE-1][fi] <= in_sample[fi];
                        // Clear accumulators.
                        for (int fo = 0; fo < F_OUT; fo++) acc[fo] <= '0;
                        k_idx    <= 0;
                        fi_idx   <= 0;
                        in_ready <= 1'b0;
                        state    <= S_COMP;
                    end
                end
                S_COMP: begin
                    // Add F_OUT-wide product to accumulators.
                    for (int fo = 0; fo < F_OUT; fo++)
                        acc[fo] <= acc[fo] + prod[fo];
                    // Advance counters.
                    if (fi_idx == F_IN - 1) begin
                        fi_idx <= 0;
                        if (k_idx == KE - 1) begin
                            k_idx <= 0;
                            state <= S_EMIT;
                        end else begin
                            k_idx <= k_idx + 1;
                        end
                    end else begin
                        fi_idx <= fi_idx + 1;
                    end
                end
                S_EMIT: begin
                    for (int fo = 0; fo < F_OUT; fo++) begin
                        automatic logic signed [ACC_WIDTH-1:0] with_bias;
                        with_bias = (acc[fo] >>> FRAC_BITS)
                                  + $signed({{(ACC_WIDTH-DATA_WIDTH){B[fo][DATA_WIDTH-1]}}, B[fo]});
                        if      (with_bias > SAT_HI) y_out[fo] <= SAT_HI[DATA_WIDTH-1:0];
                        else if (with_bias < SAT_LO) y_out[fo] <= SAT_LO[DATA_WIDTH-1:0];
                        else                         y_out[fo] <= with_bias[DATA_WIDTH-1:0];
                    end
                    y_valid  <= 1'b1;
                    in_ready <= 1'b1;
                    state    <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
