// =============================================================================
// conv2d_temporal.sv
//
// Layer 1 of DB-ATCNet (PhysioNet variant): per-electrode temporal Conv2D.
//
//     Keras: Conv2D(filters=F1=16, kernel_size=(KE=64, 1),
//                   padding='same', use_bias=False)
//
// Numerical contract (bit-exact match against scripts/q88_layer1.py):
//   tap[k]   = (k==KE-1) ? in_sample : shift_reg[in_electrode][k+1]
//   acc[f]   = sum_{k=0..KE-1} signed(tap[k]) * signed(W[k][f])
//   out[f]   = sat(acc[f] >>> FRAC_BITS)  -> DATA_WIDTH (Q8.8)
//
// Streaming interface: feed (electrode, sample) tuples one per cycle. For
// padding='same' with stride 1, kernel 64, the caller must feed:
//   pad_top = (KE-1)/2 = 31 zero samples first   (warmup; out_valid stays low)
//   then real samples  t=0..TRIAL_SAMPLES-1     (each producing one output)
//   then pad_bot = 32 zero samples              (drains right-side pad)
// Output for input time T_in corresponds to model time t_out = T_in - 32.
//
// Resource note: the combinational MAC is 16 filters × 64 taps = 1024
// multiplies and a 64-input adder tree, evaluated each cycle. Synthesis on
// Z-7020 (220 DSPs) will spill into LUTs; on Ultrascale+ ZU3+ this fits.
// Add a TIME_MUX parameter later to serialize the filter dimension.
// =============================================================================

`timescale 1ns / 1ps

module conv2d_temporal #(
    parameter int NUM_EEG_CH    = 64,
    parameter int F1            = 16,
    parameter int KE            = 64,
    parameter int DATA_WIDTH    = 16,
    parameter int COEF_WIDTH    = 16,
    parameter int ACC_WIDTH     = 48,
    parameter int FRAC_BITS     = 8,
    // Path to .mem with KE*F1 signed Q-format coefficients, tap-major order:
    //   address k*F1 + f -> W[k][f]
    parameter string WEIGHTS_FILE = "",
    // Optional path to .mem with F1 signed Q-format per-filter biases (e.g.
    // BN folded into the Conv at export time). Leave "" for no bias.
    parameter string BIAS_FILE    = ""
) (
    input  logic clk,
    input  logic rst,

    input  logic                                 in_valid,
    input  logic [$clog2(NUM_EEG_CH)-1:0]        in_electrode,
    input  logic signed [DATA_WIDTH-1:0]         in_sample,

    output logic                                 out_valid,
    output logic [$clog2(NUM_EEG_CH)-1:0]        out_electrode,
    output logic signed [DATA_WIDTH-1:0]         out_filter [0:F1-1]
);

    // -------------------------------------------------------------------------
    // Weights: loaded once from WEIGHTS_FILE in tap-major, filter-minor order.
    // Bias:    optional per-filter, loaded from BIAS_FILE (for BN folding).
    // -------------------------------------------------------------------------
    logic signed [COEF_WIDTH-1:0] w_flat [0:KE*F1-1];
    logic signed [COEF_WIDTH-1:0] W      [0:KE-1][0:F1-1];
    logic signed [DATA_WIDTH-1:0] B      [0:F1-1];

    initial begin
        if (WEIGHTS_FILE != "") begin
            $readmemh(WEIGHTS_FILE, w_flat);
        end else begin
            for (int i = 0; i < KE*F1; i++) w_flat[i] = '0;
        end
        for (int k = 0; k < KE; k++)
            for (int f = 0; f < F1; f++)
                W[k][f] = w_flat[k*F1 + f];

        if (BIAS_FILE != "") begin
            $readmemh(BIAS_FILE, B);
        end else begin
            for (int f = 0; f < F1; f++) B[f] = '0;
        end
    end

    // -------------------------------------------------------------------------
    // Per-electrode shift register
    //   shift_reg[e][KE-1] = newest (most recently shifted in for electrode e)
    //   shift_reg[e][0]    = oldest
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] shift_reg [0:NUM_EEG_CH-1][0:KE-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int e = 0; e < NUM_EEG_CH; e++)
                for (int k = 0; k < KE; k++)
                    shift_reg[e][k] <= '0;
        end else if (in_valid) begin
            shift_reg[in_electrode][KE-1] <= in_sample;
            for (int k = 0; k < KE - 1; k++)
                shift_reg[in_electrode][k] <= shift_reg[in_electrode][k+1];
        end
    end

    // -------------------------------------------------------------------------
    // Tap vector (post-update view): includes the new in_sample at index KE-1.
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] tap [0:KE-1];

    always_comb begin
        for (int k = 0; k < KE - 1; k++)
            tap[k] = shift_reg[in_electrode][k+1];
        tap[KE-1] = in_sample;
    end

    // -------------------------------------------------------------------------
    // 16 parallel MACs over 64 taps each — 3-stage pipeline.
    //
    // History: the original combinational version had a 61-deep DSP48E2 cascade
    // (~43 ns logic, WNS = -36.7 ns at 100 MHz). A 2-stage split (32-tap halves)
    // dropped that to ~24 ns (WNS = -14 ns) — still over budget because each
    // half is still a 16-multiplier + 5-level adder tree.
    //
    // This version splits the 64-tap sum into MAC_SEGS = 8 eighth-sums of
    // KE/MAC_SEGS = 8 taps each, computed combinationally in parallel from
    // live tap[]. Pipeline stages:
    //   Stage 0: 8 partials per filter (mul + 3-level adder tree ≈ 7.5 ns)
    //   Stage 1: combine in pairs of four (sum_lo = e0+e1+e2+e3,
    //            sum_hi = e4+e5+e6+e7) — 4-input add tree ≈ 3 ns
    //   Stage 2: sum_lo + sum_hi + bias + sat → out_filter
    // Each stage is well under 10 ns. Latency: 3 cycles (unchanged from the
    // 4-segment version). No tap snapshot needed — all partials use the same
    // cycle's live tap[]. Previous 4-segment version still gave 16-tap MAC at
    // stage 0 (~11 ns), which kept WNS just above 0.
    // -------------------------------------------------------------------------
    localparam int MAC_SEGS    = 8;
    localparam int MAC_SEG_LEN = KE / MAC_SEGS;       // 8 taps per segment

    logic signed [ACC_WIDTH-1:0] sum_q     [0:F1-1][0:MAC_SEGS-1];
    logic signed [ACC_WIDTH-1:0] sum_q_reg [0:F1-1][0:MAC_SEGS-1];

    always_comb begin
        for (int f = 0; f < F1; f++) begin
            for (int s = 0; s < MAC_SEGS; s++) begin
                sum_q[f][s] = '0;
                for (int k = s*MAC_SEG_LEN; k < (s+1)*MAC_SEG_LEN; k++)
                    sum_q[f][s] = sum_q[f][s] + $signed(tap[k]) * $signed(W[k][f]);
            end
        end
    end

    // Stage 1 → 2 pair-add registers (sum_lo = q0+q1, sum_hi = q2+q3).
    logic signed [ACC_WIDTH-1:0] sum_lo_reg [0:F1-1];
    logic signed [ACC_WIDTH-1:0] sum_hi_reg [0:F1-1];

    // Valid + electrode pipeline (3 cycles).
    logic                          in_valid_d1, in_valid_d2;
    logic [$clog2(NUM_EEG_CH)-1:0] in_electrode_d1, in_electrode_d2;

    // -------------------------------------------------------------------------
    // Rescale (>>> FRAC_BITS) + symmetric saturation to DATA_WIDTH
    // -------------------------------------------------------------------------
    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid       <= 1'b0;
            out_electrode   <= '0;
            in_valid_d1     <= 1'b0;
            in_valid_d2     <= 1'b0;
            in_electrode_d1 <= '0;
            in_electrode_d2 <= '0;
            for (int f = 0; f < F1; f++) begin
                sum_lo_reg[f] <= '0;
                sum_hi_reg[f] <= '0;
                out_filter[f] <= '0;
                for (int s = 0; s < MAC_SEGS; s++) sum_q_reg[f][s] <= '0;
            end
        end else begin
            // Stage 0 → 1: latch the four quarter-sums and the valid/electrode.
            in_valid_d1     <= in_valid;
            in_electrode_d1 <= in_electrode;
            for (int f = 0; f < F1; f++)
                for (int s = 0; s < MAC_SEGS; s++)
                    sum_q_reg[f][s] <= sum_q[f][s];

            // Stage 1 → 2: combine the 8 eighth-sums into two halves.
            // 4-input add tree per side (~3 ns combinational).
            in_valid_d2     <= in_valid_d1;
            in_electrode_d2 <= in_electrode_d1;
            for (int f = 0; f < F1; f++) begin
                sum_lo_reg[f] <= sum_q_reg[f][0] + sum_q_reg[f][1]
                               + sum_q_reg[f][2] + sum_q_reg[f][3];
                sum_hi_reg[f] <= sum_q_reg[f][4] + sum_q_reg[f][5]
                               + sum_q_reg[f][6] + sum_q_reg[f][7];
            end

            // Stage 2 → output: final sum, bias, saturate.
            out_valid     <= in_valid_d2;
            out_electrode <= in_electrode_d2;
            for (int f = 0; f < F1; f++) begin
                logic signed [ACC_WIDTH-1:0] acc_full;
                logic signed [ACC_WIDTH-1:0] shifted;
                logic signed [ACC_WIDTH-1:0] with_bias;
                acc_full  = sum_lo_reg[f] + sum_hi_reg[f];
                shifted   = acc_full >>> FRAC_BITS;
                with_bias = shifted + $signed({{(ACC_WIDTH-DATA_WIDTH){B[f][DATA_WIDTH-1]}}, B[f]});
                if (with_bias > SAT_HI)      out_filter[f] <= SAT_HI[DATA_WIDTH-1:0];
                else if (with_bias < SAT_LO) out_filter[f] <= SAT_LO[DATA_WIDTH-1:0];
                else                         out_filter[f] <= with_bias[DATA_WIDTH-1:0];
            end
        end
    end

endmodule
