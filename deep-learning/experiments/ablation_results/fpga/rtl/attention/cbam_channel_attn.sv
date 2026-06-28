// =============================================================================
// cbam_channel_attn.sv  —  ImpCBAM channel-attention block (one sliding window).
//
// Math (matches scripts/q88_layer17_chan.py bit-exactly):
//
//     in_stream  : NUM_CH-wide vector per cycle, T_WIN of them per window.
//     avg_gap[c] = sat( (sum_t in[t,c] * INV_N) >>> 24 )
//     max_gap[c] = max_t in[t,c]
//     mlp(x)[o]  = Dense₂( ReLU( Dense₁(x) ) )                  // shared between paths
//     gate[c]    = sigmoid_LUT( sat( mlp(avg_gap)[c] + mlp(max_gap)[c] ) )
//     out[t,c]   = sat( in[t,c] * gate[c] >>> 8 )
//
// Dense₁ kernel: (D_RED × NUM_CH) stored output-major, biases length D_RED.
// Dense₂ kernel: (NUM_CH × D_RED) stored output-major, biases length NUM_CH.
//
// Pipeline (single window):
//   Phase 1: stream T_WIN vectors → accumulate avg_sum / running max.
//   Phase 2: on `frame_last` pulse: build avg_gap & max_gap; register them;
//            run shared MLP on both; sigmoid; latch gate.
//   Phase 3: re-stream the T_WIN vectors → multiply by latched gate.
//
// The TB is responsible for streaming the same window twice (phase 1 then
// phase 3). The DUT is purely combinational+register; it does not buffer
// the input itself.
//
// Latency from `gate_valid` is 1 cycle on each x_valid in Phase 3.
//
// Resources (fully-parallel):
//   Dense₁: D_RED * NUM_CH = 128 multiplies (combinational)
//   Dense₂: NUM_CH * D_RED = 128 multiplies (combinational)
//   gate_apply: NUM_CH multiplies (combinational)
// =============================================================================

`timescale 1ns / 1ps

module cbam_channel_attn #(
    parameter int    DATA_WIDTH    = 16,
    parameter int    COEF_WIDTH    = 16,
    parameter int    ACC_WIDTH     = 32,
    parameter int    FRAC_BITS     = 8,
    parameter int    NUM_CH        = 32,
    parameter int    T_WIN         = 6,
    parameter int    D_RED         = 4,
    parameter int    N_WIN         = 1,           // # of per-window weight banks (1 or 5)
    parameter int    INV_N         = 2796203,    // round(2^24 / 6)
    parameter int    INV_N_SHIFT   = 24,
    parameter int    INV_N_WIDTH   = 24,
    parameter int    LUT_N         = 1024,
    parameter int    LUT_RANGE_Q   = 1024,
    parameter int    LUT_SHIFT     = 1,
    parameter string D1_W_FILE     = "",
    parameter string D1_B_FILE     = "",
    parameter string D2_W_FILE     = "",
    parameter string D2_B_FILE     = "",
    parameter string LUT_FILE      = ""
) (
    input  logic                              clk,
    input  logic                              rst,

    // Selects which per-window weight bank to use this inference.
    input  logic [$clog2(N_WIN > 1 ? N_WIN : 2)-1:0] window_idx,

    // Streaming feature map (window of T_WIN vectors).
    input  logic signed [DATA_WIDTH-1:0]      x_in [0:NUM_CH-1],
    input  logic                              x_valid,
    input  logic                              frame_last,   // 1 on the T_WIN-th input

    output logic signed [DATA_WIDTH-1:0]      gate_out [0:NUM_CH-1],
    output logic                              gate_valid,   // pulses 1 cycle when gate updates

    // Phase 3: apply gate to a re-streamed input.
    input  logic signed [DATA_WIDTH-1:0]      apply_x_in [0:NUM_CH-1],
    input  logic                              apply_valid,
    output logic signed [DATA_WIDTH-1:0]      y_out [0:NUM_CH-1],
    output logic                              y_valid
);

    // -------------------------------------------------------------------------
    // Weight memories ($readmemh at elaboration)
    //   D1_W stride per bank = D_RED * NUM_CH ; index = bank*stride + o*NUM_CH + i
    //   D2_W stride per bank = NUM_CH * D_RED ; index = bank*stride + o*D_RED + i
    // -------------------------------------------------------------------------
    localparam int D1W_STRIDE = D_RED * NUM_CH;
    localparam int D2W_STRIDE = NUM_CH * D_RED;
    logic signed [COEF_WIDTH-1:0] D1_W [0:N_WIN * D1W_STRIDE - 1];
    logic signed [DATA_WIDTH-1:0] D1_B [0:N_WIN * D_RED   - 1];
    logic signed [COEF_WIDTH-1:0] D2_W [0:N_WIN * D2W_STRIDE - 1];
    logic signed [DATA_WIDTH-1:0] D2_B [0:N_WIN * NUM_CH  - 1];
    logic signed [DATA_WIDTH-1:0] LUT  [0:LUT_N - 1];

    initial begin
        if (D1_W_FILE != "") $readmemh(D1_W_FILE, D1_W);
        if (D1_B_FILE != "") $readmemh(D1_B_FILE, D1_B);
        if (D2_W_FILE != "") $readmemh(D2_W_FILE, D2_W);
        if (D2_B_FILE != "") $readmemh(D2_B_FILE, D2_B);
        if (LUT_FILE  != "") $readmemh(LUT_FILE,  LUT);
    end

    // window_idx selects per-window weight bank. The orchestrator/TB must hold
    // it stable from the first x_valid of a window through the matching y_valid.
    wire [31:0] d1w_base = window_idx * D1W_STRIDE;
    wire [31:0] d1b_base = window_idx * D_RED;
    wire [31:0] d2w_base = window_idx * D2W_STRIDE;
    wire [31:0] d2b_base = window_idx * NUM_CH;

    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    // -------------------------------------------------------------------------
    // Phase 1 accumulators (avg & max over T_WIN steps)
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0]  avg_acc  [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] max_acc  [0:NUM_CH-1];

    localparam logic signed [DATA_WIDTH-1:0] DATA_MIN = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    // frame_started: 1 once we've seen at least one x_valid of the current
    // frame. On the first x_valid of a new frame we *reset* the accumulator
    // to x_in rather than adding to whatever stale value remains from the
    // previous frame.
    logic frame_started;

    always_ff @(posedge clk) begin
        if (rst) begin
            frame_started <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) begin
                avg_acc[c] <= '0;
                max_acc[c] <= DATA_MIN;
            end
        end else if (x_valid) begin
            if (!frame_started) begin
                // First input of a fresh frame.
                for (int c = 0; c < NUM_CH; c++) begin
                    avg_acc[c] <= $signed({{(ACC_WIDTH-DATA_WIDTH){x_in[c][DATA_WIDTH-1]}}, x_in[c]});
                    max_acc[c] <= $signed(x_in[c]);
                end
                frame_started <= !frame_last;     // stays cleared if single-element frame
            end else if (frame_last) begin
                // Last input of frame; accumulator + this input is the frame's
                // final view (handled combinationally in *_final). Mark frame
                // finished so the next x_valid starts fresh.
                frame_started <= 1'b0;
            end else begin
                for (int c = 0; c < NUM_CH; c++) begin
                    avg_acc[c] <= avg_acc[c]
                                + $signed({{(ACC_WIDTH-DATA_WIDTH){x_in[c][DATA_WIDTH-1]}}, x_in[c]});
                    if ($signed(x_in[c]) > max_acc[c]) max_acc[c] <= $signed(x_in[c]);
                end
            end
        end
    end

    // Combinational "final" view of accumulators at frame_last.
    logic signed [ACC_WIDTH-1:0]  avg_final [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] max_final [0:NUM_CH-1];
    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            avg_final[c] = avg_acc[c] + $signed(x_in[c]);
            max_final[c] = ($signed(x_in[c]) > max_acc[c]) ? $signed(x_in[c]) : max_acc[c];
        end
    end

    // -------------------------------------------------------------------------
    // Phase 2: avg GAP via reciprocal multiply (saturated)
    // -------------------------------------------------------------------------
    localparam int MUL_WIDTH_AVG = ACC_WIDTH + INV_N_WIDTH;
    logic signed [MUL_WIDTH_AVG-1:0] avg_prod    [0:NUM_CH-1];
    logic signed [MUL_WIDTH_AVG-1:0] avg_shifted [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0]    avg_gap     [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0]    max_gap     [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            avg_prod[c]    = avg_final[c] * $signed({1'b0, INV_N_WIDTH'(INV_N)});
            avg_shifted[c] = avg_prod[c] >>> INV_N_SHIFT;
            if      (avg_shifted[c] >  SAT_HI) avg_gap[c] = SAT_HI[DATA_WIDTH-1:0];
            else if (avg_shifted[c] <  SAT_LO) avg_gap[c] = SAT_LO[DATA_WIDTH-1:0];
            else                               avg_gap[c] = avg_shifted[c][DATA_WIDTH-1:0];
            max_gap[c] = max_final[c];
        end
    end

    // -------------------------------------------------------------------------
    // Dense₁: gap (32) -> hidden (4) with ReLU. Two parallel copies, one for
    // avg path and one for max path.
    // -------------------------------------------------------------------------
    localparam int D1_ACC_WIDTH = 40;   // 16-bit * 16-bit * 32 ≤ 32+log2(32)+1 = 38 → 40 safe

    function automatic logic signed [DATA_WIDTH-1:0] dense_step (
        input  logic signed [DATA_WIDTH-1:0]      x_vec  [0:31],
        input  int                                in_cnt,
        input  int                                out_idx,
        input  logic signed [COEF_WIDTH-1:0]      W_flat [],
        input  logic signed [DATA_WIDTH-1:0]      bias
    );
        logic signed [D1_ACC_WIDTH-1:0] acc;
        logic signed [D1_ACC_WIDTH-1:0] shifted;
        acc = '0;
        for (int i = 0; i < in_cnt; i++)
            acc = acc + $signed(x_vec[i]) * $signed(W_flat[out_idx * in_cnt + i]);
        shifted = (acc >>> FRAC_BITS) + bias;
        if      (shifted > SAT_HI) return SAT_HI[DATA_WIDTH-1:0];
        else if (shifted < SAT_LO) return SAT_LO[DATA_WIDTH-1:0];
        else                       return shifted[DATA_WIDTH-1:0];
    endfunction

    logic signed [D1_ACC_WIDTH-1:0] d1_acc_avg [0:D_RED-1];
    logic signed [D1_ACC_WIDTH-1:0] d1_acc_max [0:D_RED-1];
    logic signed [D1_ACC_WIDTH-1:0] d1_sh_avg  [0:D_RED-1];
    logic signed [D1_ACC_WIDTH-1:0] d1_sh_max  [0:D_RED-1];
    logic signed [DATA_WIDTH-1:0]   d1_avg_h   [0:D_RED-1];
    logic signed [DATA_WIDTH-1:0]   d1_max_h   [0:D_RED-1];

    // Pipeline registers split the channel-attention gate computation:
    // frame_last -> GAP regs -> Dense1 partial-sum regs -> Dense1 output regs
    // -> Dense2/sigmoid gate regs. gate_valid pulses 4 cycles after
    // frame_last.
    logic signed [DATA_WIDTH-1:0]   avg_gap_reg [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0]   max_gap_reg [0:NUM_CH-1];
    logic [31:0]                    d1w_base_reg;
    logic [31:0]                    d1b_base_reg;
    logic [31:0]                    d2w_base_reg;
    logic [31:0]                    d2b_base_reg;
    logic                           gap_ready;
    logic                           d1_part_ready;
    logic signed [DATA_WIDTH-1:0]   d1_avg_h_reg [0:D_RED-1];
    logic signed [DATA_WIDTH-1:0]   d1_max_h_reg [0:D_RED-1];
    logic                           d1_ready;

    localparam int D1_SEGS    = 4;
    localparam int D1_SEG_LEN = NUM_CH / D1_SEGS;
    logic signed [D1_ACC_WIDTH-1:0] d1_acc_avg_part     [0:D_RED-1][0:D1_SEGS-1];
    logic signed [D1_ACC_WIDTH-1:0] d1_acc_max_part     [0:D_RED-1][0:D1_SEGS-1];
    logic signed [D1_ACC_WIDTH-1:0] d1_acc_avg_part_reg [0:D_RED-1][0:D1_SEGS-1];
    logic signed [D1_ACC_WIDTH-1:0] d1_acc_max_part_reg [0:D_RED-1][0:D1_SEGS-1];

    always_comb begin
        for (int o = 0; o < D_RED; o++) begin
            d1_acc_avg[o] = '0;
            d1_acc_max[o] = '0;
            for (int s = 0; s < D1_SEGS; s++) begin
                d1_acc_avg[o] = d1_acc_avg[o] + d1_acc_avg_part_reg[o][s];
                d1_acc_max[o] = d1_acc_max[o] + d1_acc_max_part_reg[o][s];
            end
            d1_sh_avg[o] = (d1_acc_avg[o] >>> FRAC_BITS)
                         + $signed({{(D1_ACC_WIDTH-DATA_WIDTH){D1_B[d1b_base_reg + o][DATA_WIDTH-1]}}, D1_B[d1b_base_reg + o]});
            d1_sh_max[o] = (d1_acc_max[o] >>> FRAC_BITS)
                         + $signed({{(D1_ACC_WIDTH-DATA_WIDTH){D1_B[d1b_base_reg + o][DATA_WIDTH-1]}}, D1_B[d1b_base_reg + o]});
            // Saturate to Q8.8 then ReLU
            if      (d1_sh_avg[o] >  SAT_HI) d1_avg_h[o] = SAT_HI[DATA_WIDTH-1:0];
            else if (d1_sh_avg[o] <  SAT_LO) d1_avg_h[o] = '0;          // negative → ReLU 0
            else if (d1_sh_avg[o] <  0)      d1_avg_h[o] = '0;
            else                             d1_avg_h[o] = d1_sh_avg[o][DATA_WIDTH-1:0];
            if      (d1_sh_max[o] >  SAT_HI) d1_max_h[o] = SAT_HI[DATA_WIDTH-1:0];
            else if (d1_sh_max[o] <  SAT_LO) d1_max_h[o] = '0;
            else if (d1_sh_max[o] <  0)      d1_max_h[o] = '0;
            else                             d1_max_h[o] = d1_sh_max[o][DATA_WIDTH-1:0];
        end
    end

    always_comb begin
        for (int o = 0; o < D_RED; o++) begin
            for (int s = 0; s < D1_SEGS; s++) begin
                d1_acc_avg_part[o][s] = '0;
                d1_acc_max_part[o][s] = '0;
                for (int i = 0; i < D1_SEG_LEN; i++) begin
                    int idx;
                    idx = s * D1_SEG_LEN + i;
                    d1_acc_avg_part[o][s] = d1_acc_avg_part[o][s]
                                          + $signed(avg_gap_reg[idx]) * $signed(D1_W[d1w_base_reg + o * NUM_CH + idx]);
                    d1_acc_max_part[o][s] = d1_acc_max_part[o][s]
                                          + $signed(max_gap_reg[idx]) * $signed(D1_W[d1w_base_reg + o * NUM_CH + idx]);
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Dense₂: hidden (4) -> output (32). Combine paths after.
    // -------------------------------------------------------------------------
    logic signed [D1_ACC_WIDTH-1:0] d2_acc_avg [0:NUM_CH-1];
    logic signed [D1_ACC_WIDTH-1:0] d2_acc_max [0:NUM_CH-1];
    logic signed [D1_ACC_WIDTH-1:0] d2_sh_avg  [0:NUM_CH-1];
    logic signed [D1_ACC_WIDTH-1:0] d2_sh_max  [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0]   d2_avg_out [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0]   d2_max_out [0:NUM_CH-1];

    always_comb begin
        for (int o = 0; o < NUM_CH; o++) begin
            d2_acc_avg[o] = '0;
            d2_acc_max[o] = '0;
            for (int i = 0; i < D_RED; i++) begin
                d2_acc_avg[o] = d2_acc_avg[o] + $signed(d1_avg_h_reg[i]) * $signed(D2_W[d2w_base_reg + o * D_RED + i]);
                d2_acc_max[o] = d2_acc_max[o] + $signed(d1_max_h_reg[i]) * $signed(D2_W[d2w_base_reg + o * D_RED + i]);
            end
            d2_sh_avg[o] = (d2_acc_avg[o] >>> FRAC_BITS)
                         + $signed({{(D1_ACC_WIDTH-DATA_WIDTH){D2_B[d2b_base_reg + o][DATA_WIDTH-1]}}, D2_B[d2b_base_reg + o]});
            d2_sh_max[o] = (d2_acc_max[o] >>> FRAC_BITS)
                         + $signed({{(D1_ACC_WIDTH-DATA_WIDTH){D2_B[d2b_base_reg + o][DATA_WIDTH-1]}}, D2_B[d2b_base_reg + o]});
            if      (d2_sh_avg[o] >  SAT_HI) d2_avg_out[o] = SAT_HI[DATA_WIDTH-1:0];
            else if (d2_sh_avg[o] <  SAT_LO) d2_avg_out[o] = SAT_LO[DATA_WIDTH-1:0];
            else                             d2_avg_out[o] = d2_sh_avg[o][DATA_WIDTH-1:0];
            if      (d2_sh_max[o] >  SAT_HI) d2_max_out[o] = SAT_HI[DATA_WIDTH-1:0];
            else if (d2_sh_max[o] <  SAT_LO) d2_max_out[o] = SAT_LO[DATA_WIDTH-1:0];
            else                             d2_max_out[o] = d2_sh_max[o][DATA_WIDTH-1:0];
        end
    end

    // -------------------------------------------------------------------------
    // Add(avg, max) → saturate → sigmoid via LUT → gate (registered)
    // -------------------------------------------------------------------------
    localparam int                                ADDR_W       = $clog2(LUT_N);
    localparam logic signed [DATA_WIDTH-1:0]      LUT_CLIP_HI  = LUT_RANGE_Q - 1;
    localparam logic signed [DATA_WIDTH-1:0]      LUT_CLIP_LO  = -LUT_RANGE_Q;

    logic signed [DATA_WIDTH-1:0] sig_in   [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] clipped  [0:NUM_CH-1];
    logic        [ADDR_W-1:0]     addr     [0:NUM_CH-1];

    localparam logic signed [DATA_WIDTH:0] S17_HI =  (1 <<< (DATA_WIDTH-1)) - 1;
    localparam logic signed [DATA_WIDTH:0] S17_LO = -(1 <<< (DATA_WIDTH-1));

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            logic signed [DATA_WIDTH:0] s;
            s = {d2_avg_out[c][DATA_WIDTH-1], d2_avg_out[c]}
              + {d2_max_out[c][DATA_WIDTH-1], d2_max_out[c]};
            if      (s > S17_HI) sig_in[c] = S17_HI[DATA_WIDTH-1:0];
            else if (s < S17_LO) sig_in[c] = S17_LO[DATA_WIDTH-1:0];
            else                 sig_in[c] = s[DATA_WIDTH-1:0];

            if      (sig_in[c] >  LUT_CLIP_HI) clipped[c] = LUT_CLIP_HI;
            else if (sig_in[c] <  LUT_CLIP_LO) clipped[c] = LUT_CLIP_LO;
            else                               clipped[c] = sig_in[c];
            addr[c] = ADDR_W'((clipped[c] + LUT_RANGE_Q) >>> LUT_SHIFT);
        end
    end

    logic signed [DATA_WIDTH-1:0] gate_reg [0:NUM_CH-1];

    // Pipeline: latch GAP values on frame_last, latch Dense1 outputs one cycle
    // later, then latch the final gate after Dense2/sigmoid.
    always_ff @(posedge clk) begin
        if (rst) begin
            gap_ready <= 1'b0;
            d1_part_ready <= 1'b0;
            d1_ready  <= 1'b0;
            gate_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) begin
                avg_gap_reg[c] <= '0;
                max_gap_reg[c] <= '0;
            end
            d1w_base_reg <= '0;
            d1b_base_reg <= '0;
            d2w_base_reg <= '0;
            d2b_base_reg <= '0;
            for (int o = 0; o < D_RED; o++) begin
                for (int s = 0; s < D1_SEGS; s++) begin
                    d1_acc_avg_part_reg[o][s] <= '0;
                    d1_acc_max_part_reg[o][s] <= '0;
                end
            end
            for (int o = 0; o < D_RED; o++) begin
                d1_avg_h_reg[o] <= '0;
                d1_max_h_reg[o] <= '0;
            end
            for (int c = 0; c < NUM_CH; c++) begin
                gate_reg[c] <= '0;
                gate_out[c] <= '0;
            end
        end else begin
            gap_ready <= (x_valid && frame_last);
            d1_part_ready <= gap_ready;
            d1_ready  <= d1_part_ready;
            gate_valid <= d1_ready;

            if (x_valid && frame_last) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    avg_gap_reg[c] <= avg_gap[c];
                    max_gap_reg[c] <= max_gap[c];
                end
                d1w_base_reg <= d1w_base;
                d1b_base_reg <= d1b_base;
                d2w_base_reg <= d2w_base;
                d2b_base_reg <= d2b_base;
            end

            if (gap_ready) begin
                for (int o = 0; o < D_RED; o++) begin
                    for (int s = 0; s < D1_SEGS; s++) begin
                        d1_acc_avg_part_reg[o][s] <= d1_acc_avg_part[o][s];
                        d1_acc_max_part_reg[o][s] <= d1_acc_max_part[o][s];
                    end
                end
            end

            if (d1_part_ready) begin
                for (int o = 0; o < D_RED; o++) begin
                    d1_avg_h_reg[o] <= d1_avg_h[o];
                    d1_max_h_reg[o] <= d1_max_h[o];
                end
            end

            if (d1_ready) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    gate_reg[c] <= LUT[addr[c]];
                    gate_out[c] <= LUT[addr[c]];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Phase 3: apply latched gate to re-streamed input.
    // -------------------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] apply_prod    [0:NUM_CH-1];
    logic signed [ACC_WIDTH-1:0] apply_shifted [0:NUM_CH-1];

    always_comb begin
        for (int c = 0; c < NUM_CH; c++) begin
            apply_prod[c]    = $signed(apply_x_in[c]) * $signed(gate_reg[c]);
            apply_shifted[c] = apply_prod[c] >>> FRAC_BITS;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            y_valid <= 1'b0;
            for (int c = 0; c < NUM_CH; c++) y_out[c] <= '0;
        end else begin
            y_valid <= apply_valid;
            if (apply_valid) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    if      (apply_shifted[c] > SAT_HI) y_out[c] <= SAT_HI[DATA_WIDTH-1:0];
                    else if (apply_shifted[c] < SAT_LO) y_out[c] <= SAT_LO[DATA_WIDTH-1:0];
                    else                                y_out[c] <= apply_shifted[c][DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule
