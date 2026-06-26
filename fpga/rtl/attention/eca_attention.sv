// =============================================================================
// eca_attention.sv  —  Efficient Channel Attention (ECA) gate computation
//
// Math (matches scripts/q88_eca1.py bit-exactly):
//
//     gap_in   : signed Q8.8 vector of length NUM_CH (global-avg-pooled, fed
//                from upstream — see the streaming-GAP wrapper module).
//     z[c]     : 3-tap (or KECA-tap) Conv1D over the channel dim, padding
//                'same' (KECA//2 zeros on each side), no bias.
//     gate[c]  : sigmoid(z[c]) via the universal Q8.8 LUT.
//
// Two memory files are loaded at elaboration via $readmemh:
//   WEIGHTS_FILE  KECA signed Q8.8 entries (Conv1D taps)
//   LUT_FILE      LUT_N signed Q8.8 entries (sigmoid table from
//                 scripts/gen_sigmoid_lut.py — universal, not per-subject)
//
// Latency: 1 cycle from gap_valid → gate_valid.
// Resources: NUM_CH * KECA multiplies + one BRAM18K for the LUT.
//
// Parameters:
//   NUM_CH        Number of channels (16 for ECA₁, 32 for ECA₂).
//   KECA          Conv1D kernel size (3 for both ECA positions per the spec).
//   LUT_N         Sigmoid LUT entry count (default 1024).
//   LUT_RANGE_Q   Q8.8 magnitude that the LUT covers ±range from 0 (default 1024 = ±4.0).
//   LUT_SHIFT     log2(2*LUT_RANGE_Q / LUT_N). With defaults: 1.
// =============================================================================

`timescale 1ns / 1ps

module eca_attention #(
    parameter int    DATA_WIDTH   = 16,
    parameter int    COEF_WIDTH   = 16,
    parameter int    ACC_WIDTH    = 48,
    parameter int    FRAC_BITS    = 8,
    parameter int    NUM_CH       = 16,
    parameter int    KECA         = 3,
    parameter int    LUT_N        = 1024,
    parameter int    LUT_RANGE_Q  = 1024,
    parameter int    LUT_SHIFT    = 1,
    parameter string WEIGHTS_FILE = "",
    parameter string LUT_FILE     = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    input  logic signed [DATA_WIDTH-1:0]    gap_in [0:NUM_CH-1],
    input  logic                            gap_valid,

    output logic signed [DATA_WIDTH-1:0]    gate_out [0:NUM_CH-1],
    output logic                            gate_valid
);

    // -------------------------------------------------------------------------
    // Weights + LUT load (elaboration-time via $readmemh)
    // -------------------------------------------------------------------------
    logic signed [COEF_WIDTH-1:0] W   [0:KECA-1];
    logic signed [DATA_WIDTH-1:0] LUT [0:LUT_N-1];

    initial begin
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, W);
        else                    for (int k = 0; k < KECA; k++)  W[k] = '0;
        if (LUT_FILE     != "") $readmemh(LUT_FILE,     LUT);
        else                    for (int i = 0; i < LUT_N; i++) LUT[i] = '0;
    end

    // -------------------------------------------------------------------------
    // Saturation bounds (signed DATA_WIDTH)
    // -------------------------------------------------------------------------
    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    // -------------------------------------------------------------------------
    // Stage 1 (combinational): Conv1D + rescale + saturate -> sigmoid input
    // -------------------------------------------------------------------------
    localparam int PAD = (KECA - 1) / 2;

    logic signed [DATA_WIDTH-1:0] tap_pad [0:NUM_CH + 2*PAD - 1];
    logic signed [ACC_WIDTH-1:0]  conv_acc  [0:NUM_CH-1];
    logic signed [ACC_WIDTH-1:0]  conv_shft [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] sig_in    [0:NUM_CH-1];

    always_comb begin
        // Build zero-padded view of gap_in
        for (int i = 0;        i < PAD;             i++) tap_pad[i]         = '0;
        for (int i = 0;        i < NUM_CH;          i++) tap_pad[i + PAD]   = gap_in[i];
        for (int i = 0;        i < PAD;             i++) tap_pad[NUM_CH + PAD + i] = '0;

        for (int c = 0; c < NUM_CH; c++) begin
            conv_acc[c] = '0;
            for (int k = 0; k < KECA; k++)
                conv_acc[c] = conv_acc[c] + $signed(tap_pad[c + k]) * $signed(W[k]);
            conv_shft[c] = conv_acc[c] >>> FRAC_BITS;
            if (conv_shft[c] > SAT_HI)      sig_in[c] = SAT_HI[DATA_WIDTH-1:0];
            else if (conv_shft[c] < SAT_LO) sig_in[c] = SAT_LO[DATA_WIDTH-1:0];
            else                            sig_in[c] = conv_shft[c][DATA_WIDTH-1:0];
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2 (combinational, registered): clip → addr → LUT lookup
    //
    //   addr = (clip(sig_in[c], -LUT_RANGE_Q, LUT_RANGE_Q-1) + LUT_RANGE_Q) >>> LUT_SHIFT
    // -------------------------------------------------------------------------
    localparam int                                ADDR_W       = $clog2(LUT_N);
    localparam logic signed [DATA_WIDTH-1:0]      LUT_CLIP_HI  = LUT_RANGE_Q - 1;
    localparam logic signed [DATA_WIDTH-1:0]      LUT_CLIP_LO  = -LUT_RANGE_Q;

    logic signed [DATA_WIDTH-1:0] clipped [0:NUM_CH-1];
    logic        [ADDR_W-1:0]     addr    [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            if      (sig_in[c] >  LUT_CLIP_HI) clipped[c] = LUT_CLIP_HI;
            else if (sig_in[c] <  LUT_CLIP_LO) clipped[c] = LUT_CLIP_LO;
            else                               clipped[c] = sig_in[c];
            addr[c] = ADDR_W'((clipped[c] + LUT_RANGE_Q) >>> LUT_SHIFT);
        end
    end

    // -------------------------------------------------------------------------
    // Output register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            gate_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) gate_out[c] <= '0;
        end else begin
            gate_valid <= gap_valid;
            if (gap_valid) begin
                for (int c = 0; c < NUM_CH; c++) gate_out[c] <= LUT[addr[c]];
            end
        end
    end

endmodule
