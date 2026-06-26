// =============================================================================
// tcfn.sv  —  Temporal Convolutional Fusion Network (one sliding window).
//
// Bit-exact match to scripts/q88_layer_tcfn.py:
//
//   Stage 0 (dilation=1):
//     a  = ELU( BN( Conv1D_0(x_buf) ) )
//     b  = ELU( BN( Conv1D_1(a)     ) )
//     s0 = ELU( b + x_buf )                 [Residual 1]
//   Stage 1 (dilation=2):
//     c  = ELU( BN( Conv1D_2(s0)    ) )
//     r2 = c + x_buf                         [Residual 2 — no ELU]
//     d  = ELU( BN( Conv1D_3(r2)    ) )
//     out= ELU( d + x_buf )                  [Residual 3]
//
// BN is pre-folded into the conv weights/bias by the Python golden, so the
// runtime is just 4 dilated causal Conv1D + ELU LUT + saturating signed adds.
//
// Hardware sequencing: a single F_IN-parallel MAC bank is reused across all
// 4 conv stages, sequentially. Per (t, of_idx) pair we sum 32 input channels
// in parallel and accumulate across the K=4 taps over 4 cycles. So one
// conv = T_WIN * F * K = 6*32*4 = 768 cycles. Four convs = 3072 cycles per
// TCFN; 5 windows fit easily in a 3-second EEG window.
//
// Q8.8 saturation may occur (the trained model has float-range activations
// that overflow Q8.8 in TCFN); the Python golden saturates identically so
// the test remains bit-exact.
//
// Weight layout per stage: kernel = (K, F_IN, F_OUT) flat row-major, i.e.
// W[k * F_IN * F_OUT + ic * F_OUT + of] = w_folded[k, ic, of] in the .hex file.
// Bias is F_OUT entries.
// =============================================================================

`timescale 1ns / 1ps

module tcfn #(
    parameter int    DATA_WIDTH = 16,
    parameter int    COEF_WIDTH = 16,
    parameter int    ACC_WIDTH  = 48,
    parameter int    FRAC_BITS  = 8,
    parameter int    T_WIN      = 6,
    parameter int    F          = 32,            // input == output == feature dim
    parameter int    K          = 4,
    parameter int    LUT_N      = 256,
    parameter int    LUT_RANGE_Q = 2048,
    parameter int    LUT_SHIFT  = 3,
    parameter int    N_WIN      = 1,
    parameter string W0_FILE    = "",
    parameter string B0_FILE    = "",
    parameter string W1_FILE    = "",
    parameter string B1_FILE    = "",
    parameter string W2_FILE    = "",
    parameter string B2_FILE    = "",
    parameter string W3_FILE    = "",
    parameter string B3_FILE    = "",
    parameter string LUT_FILE   = ""
) (
    input  logic                              clk,
    input  logic                              rst,

    // Selects per-window weight bank.
    input  logic [$clog2(N_WIN > 1 ? N_WIN : 2)-1:0] window_idx,

    // Phase 1: streaming input (T_WIN vectors, frame_last on the last).
    input  logic signed [DATA_WIDTH-1:0]      x_in [0:F-1],
    input  logic                              x_valid,
    input  logic                              frame_last,

    // Phase 2: streaming output. y_valid pulses T_WIN times after the
    // entire TCFN computation is done; consecutive cycles t=0..T_WIN-1.
    output logic signed [DATA_WIDTH-1:0]      y_out [0:F-1],
    output logic                              y_valid,
    output logic                              done
);

    // -------------------------------------------------------------------------
    // Weight + bias + LUT memories
    // -------------------------------------------------------------------------
    localparam int W_DEPTH = K * F * F;
    (* ram_style = "ultra" *) logic signed [COEF_WIDTH-1:0] W0 [0:N_WIN * W_DEPTH - 1];
    (* ram_style = "ultra" *) logic signed [COEF_WIDTH-1:0] W1 [0:N_WIN * W_DEPTH - 1];
    (* ram_style = "ultra" *) logic signed [COEF_WIDTH-1:0] W2 [0:N_WIN * W_DEPTH - 1];
    (* ram_style = "ultra" *) logic signed [COEF_WIDTH-1:0] W3 [0:N_WIN * W_DEPTH - 1];
    logic signed [DATA_WIDTH-1:0] B0 [0:N_WIN * F - 1];
    logic signed [DATA_WIDTH-1:0] B1 [0:N_WIN * F - 1];
    logic signed [DATA_WIDTH-1:0] B2 [0:N_WIN * F - 1];
    logic signed [DATA_WIDTH-1:0] B3 [0:N_WIN * F - 1];
    logic signed [DATA_WIDTH-1:0] LUT [0:LUT_N-1];

    wire [31:0] w_base = window_idx * W_DEPTH;
    wire [31:0] b_base = window_idx * F;

    initial begin
        if (W0_FILE != "") $readmemh(W0_FILE, W0);
        if (W1_FILE != "") $readmemh(W1_FILE, W1);
        if (W2_FILE != "") $readmemh(W2_FILE, W2);
        if (W3_FILE != "") $readmemh(W3_FILE, W3);
        if (B0_FILE != "") $readmemh(B0_FILE, B0);
        if (B1_FILE != "") $readmemh(B1_FILE, B1);
        if (B2_FILE != "") $readmemh(B2_FILE, B2);
        if (B3_FILE != "") $readmemh(B3_FILE, B3);
        if (LUT_FILE != "") $readmemh(LUT_FILE, LUT);
    end

    // -------------------------------------------------------------------------
    // Buffers
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] x_buf  [0:T_WIN-1][0:F-1];   // original input
    logic signed [DATA_WIDTH-1:0] src    [0:T_WIN-1][0:F-1];   // conv source
    logic signed [DATA_WIDTH-1:0] dst    [0:T_WIN-1][0:F-1];   // conv destination
    logic signed [DATA_WIDTH-1:0] s0_buf [0:T_WIN-1][0:F-1];   // end of Stage 0

    localparam logic signed [ACC_WIDTH-1:0] SAT_HI =
        (ACC_WIDTH)'((1 <<< (DATA_WIDTH-1)) - 1);
    localparam logic signed [ACC_WIDTH-1:0] SAT_LO =
        -(ACC_WIDTH)'(1 <<< (DATA_WIDTH-1));

    function automatic logic signed [DATA_WIDTH-1:0] sat_q88 (input logic signed [ACC_WIDTH-1:0] v);
        if      (v > SAT_HI) return SAT_HI[DATA_WIDTH-1:0];
        else if (v < SAT_LO) return SAT_LO[DATA_WIDTH-1:0];
        else                 return v[DATA_WIDTH-1:0];
    endfunction

    // ELU LUT addressing.
    localparam int ADDR_W = $clog2(LUT_N);
    function automatic logic signed [DATA_WIDTH-1:0] elu_q88 (
        input logic signed [DATA_WIDTH-1:0] x);
        logic        [DATA_WIDTH:0]   neg_abs;        // 17-bit unsigned to hold +32768
        logic        [DATA_WIDTH:0]   shifted;
        logic        [ADDR_W-1:0]     addr;
        if (x >= 0) return x;
        neg_abs = (DATA_WIDTH+1)'(-$signed({x[DATA_WIDTH-1], x}));  // sign-ext then negate → positive magnitude
        shifted = neg_abs >> LUT_SHIFT;
        addr    = (shifted > (LUT_N - 1)) ? ADDR_W'(LUT_N - 1) : ADDR_W'(shifted);
        return LUT[addr];
    endfunction

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_INGEST,
        S_CONV,
        S_POST,       // ELU + (optional) residual add to produce the next stage
        S_NEXT_STAGE, // bookkeeping between stages
        S_EMIT,
        S_DONE
    } state_t;
    state_t state;

    logic [$clog2(T_WIN+1)-1:0] t_in_idx;      // input ingestion counter
    logic [$clog2(T_WIN+1)-1:0] conv_t;        // current t in conv
    logic [$clog2(F+1)-1:0]     conv_of;       // current output channel
    logic [$clog2(K+1)-1:0]     conv_k;        // current tap
    logic [2:0]                 stage;         // 0..3 conv stage index
    logic [$clog2(T_WIN+1)-1:0] emit_idx;

    logic signed [ACC_WIDTH-1:0] mac_acc;

    // Stage-dependent selectors
    logic                       dil_d2;
    logic                       use_residual;
    logic                       use_relu_final; // ELU after residual? (residual2 is no-ELU)

    always_comb begin
        // Stage 0: conv0 — d=1, no residual at post-conv, ELU only
        // Stage 1: conv1 — d=1, after ELU we add input_layer to form s0 (ELU after)
        // Stage 2: conv2 — d=2, after ELU we add input_layer to form r2 (NO ELU after)
        // Stage 3: conv3 — d=2, after ELU we add input_layer to form out (ELU after)
        unique case (stage)
            3'd0: begin dil_d2 = 1'b0; use_residual = 1'b0; use_relu_final = 1'b0; end
            3'd1: begin dil_d2 = 1'b0; use_residual = 1'b1; use_relu_final = 1'b1; end
            3'd2: begin dil_d2 = 1'b1; use_residual = 1'b1; use_relu_final = 1'b0; end
            3'd3: begin dil_d2 = 1'b1; use_residual = 1'b1; use_relu_final = 1'b1; end
            default: begin dil_d2 = 1'b0; use_residual = 1'b0; use_relu_final = 1'b0; end
        endcase
    end

    // -------------------------------------------------------------------------
    // Combinational MAC tree for one (t, of, k) step.
    //   acc_delta = sum_{ic=0..F-1} src_used[t', ic] * W[stage][k * F*F + ic * F + of]
    // where t' = conv_t - (K-1-k)*dilation  (causal); zero if t' < 0.
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] tap_vec [0:F-1];   // src[t'] or zero
    logic        [$clog2(T_WIN+1)+1:0] back;          // lookback in time
    logic signed [$clog2(T_WIN+2):0]   t_prime;       // signed lookback target
    logic                              tap_in_bounds;

    always_comb begin
        // Python convention: tap k=0 == most recent (no lookback), k=K-1 == oldest.
        // y[t] = sum_k x[t - k*d] * w[k].
        back   = ($clog2(T_WIN+1)+2)'(conv_k) * (dil_d2 ? 2 : 1);
        t_prime = $signed({1'b0, conv_t}) - $signed({1'b0, back[$clog2(T_WIN+1)+1:0]});
        tap_in_bounds = (t_prime >= 0) && (t_prime < T_WIN);
        for (int ic = 0; ic < F; ic++) begin
            if (tap_in_bounds) tap_vec[ic] = src[t_prime[$clog2(T_WIN)-1:0]][ic];
            else               tap_vec[ic] = '0;
        end
    end

    // Compute the partial sum across F input channels (1024-bit-ish wide reduction).
    logic signed [ACC_WIDTH-1:0] tap_partial;
    always_comb begin
        tap_partial = '0;
        if (tap_in_bounds) begin
            for (int ic = 0; ic < F; ic++) begin
                logic signed [COEF_WIDTH-1:0] wv;
                unique case (stage)
                    3'd0: wv = W0[w_base + conv_k * F * F + ic * F + conv_of];
                    3'd1: wv = W1[w_base + conv_k * F * F + ic * F + conv_of];
                    3'd2: wv = W2[w_base + conv_k * F * F + ic * F + conv_of];
                    default: wv = W3[w_base + conv_k * F * F + ic * F + conv_of];
                endcase
                tap_partial = tap_partial + $signed(tap_vec[ic]) * $signed(wv);
            end
        end
    end

    // Bias for current stage / output channel.
    logic signed [DATA_WIDTH-1:0] cur_bias;
    always_comb begin
        unique case (stage)
            3'd0: cur_bias = B0[b_base + conv_of];
            3'd1: cur_bias = B1[b_base + conv_of];
            3'd2: cur_bias = B2[b_base + conv_of];
            default: cur_bias = B3[b_base + conv_of];
        endcase
    end

    // -------------------------------------------------------------------------
    // Main FSM
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            t_in_idx <= 0;
            conv_t   <= 0;
            conv_of  <= 0;
            conv_k   <= 0;
            stage    <= 0;
            mac_acc  <= '0;
            emit_idx <= 0;
            y_valid  <= 1'b0;
            done     <= 1'b0;
            for (int t = 0; t < T_WIN; t++) for (int c = 0; c < F; c++) begin
                x_buf[t][c]  <= '0;
                src[t][c]    <= '0;
                dst[t][c]    <= '0;
                s0_buf[t][c] <= '0;
            end
            for (int c = 0; c < F; c++) y_out[c] <= '0;
        end else begin
            y_valid <= 1'b0;
            done    <= 1'b0;

            unique case (state)
                S_IDLE: begin
                    if (x_valid) begin
                        for (int c = 0; c < F; c++) begin
                            x_buf[0][c] <= x_in[c];
                            src[0][c]   <= x_in[c];
                        end
                        t_in_idx <= 1;
                        state    <= (frame_last) ? S_CONV : S_INGEST;
                        if (frame_last) begin
                            conv_t  <= 0; conv_of <= 0; conv_k <= 0;
                            mac_acc <= '0; stage <= 0;
                        end
                    end
                end
                S_INGEST: begin
                    if (x_valid) begin
                        for (int c = 0; c < F; c++) begin
                            x_buf[t_in_idx][c] <= x_in[c];
                            src[t_in_idx][c]   <= x_in[c];
                        end
                        if (frame_last) begin
                            state   <= S_CONV;
                            conv_t  <= 0; conv_of <= 0; conv_k <= 0;
                            mac_acc <= '0; stage <= 0;
                        end else begin
                            t_in_idx <= t_in_idx + 1;
                        end
                    end
                end
                S_CONV: begin
                    // Accumulate tap_partial for current (t, of, k). At k == K-1
                    // we have the full sum; rescale + bias + saturate + store.
                    if (conv_k == K - 1) begin
                        automatic logic signed [ACC_WIDTH-1:0] sum;
                        automatic logic signed [ACC_WIDTH-1:0] shifted_bias;
                        sum          = mac_acc + tap_partial;
                        shifted_bias = (sum >>> FRAC_BITS)
                                     + $signed({{(ACC_WIDTH-DATA_WIDTH){cur_bias[DATA_WIDTH-1]}}, cur_bias});
                        dst[conv_t][conv_of] <= sat_q88(shifted_bias);
                        mac_acc <= '0;
                        if (conv_of == F - 1) begin
                            conv_of <= 0;
                            conv_k  <= 0;
                            if (conv_t == T_WIN - 1) begin
                                state  <= S_POST;
                                conv_t <= 0;
                            end else begin
                                conv_t <= conv_t + 1;
                            end
                        end else begin
                            conv_of <= conv_of + 1;
                            conv_k  <= 0;
                        end
                    end else begin
                        mac_acc <= mac_acc + tap_partial;
                        conv_k  <= conv_k + 1;
                    end
                end
                S_POST: begin
                    // For one t per cycle: apply ELU and optionally residual+ELU.
                    // dst currently holds the conv output (BN already folded in).
                    for (int c = 0; c < F; c++) begin
                        automatic logic signed [DATA_WIDTH-1:0] elu_val;
                        automatic logic signed [ACC_WIDTH-1:0]  added;
                        automatic logic signed [DATA_WIDTH-1:0] sum_q;
                        elu_val = elu_q88(dst[conv_t][c]);
                        if (use_residual) begin
                            added = $signed({{(ACC_WIDTH-DATA_WIDTH){elu_val[DATA_WIDTH-1]}}, elu_val})
                                  + $signed({{(ACC_WIDTH-DATA_WIDTH){x_buf[conv_t][c][DATA_WIDTH-1]}}, x_buf[conv_t][c]});
                            sum_q = sat_q88(added);
                            dst[conv_t][c] <= use_relu_final ? elu_q88(sum_q) : sum_q;
                        end else begin
                            dst[conv_t][c] <= elu_val;
                        end
                    end
                    if (conv_t == T_WIN - 1) begin
                        state  <= S_NEXT_STAGE;
                        conv_t <= 0;
                    end else begin
                        conv_t <= conv_t + 1;
                    end
                end
                S_NEXT_STAGE: begin
                    // Move dst → src for the next conv; cache s0 at end of stage 1.
                    for (int t = 0; t < T_WIN; t++) for (int c = 0; c < F; c++) begin
                        src[t][c] <= dst[t][c];
                    end
                    if (stage == 1) begin
                        for (int t = 0; t < T_WIN; t++) for (int c = 0; c < F; c++) begin
                            s0_buf[t][c] <= dst[t][c];
                        end
                    end
                    if (stage == 3) begin
                        // Done with stage 3; final output sits in dst.
                        state    <= S_EMIT;
                        emit_idx <= 0;
                    end else begin
                        stage   <= stage + 1;
                        state   <= S_CONV;
                        conv_t  <= 0; conv_of <= 0; conv_k <= 0;
                        mac_acc <= '0;
                    end
                end
                S_EMIT: begin
                    for (int c = 0; c < F; c++) y_out[c] <= dst[emit_idx][c];
                    y_valid <= 1'b1;
                    if (emit_idx == T_WIN - 1) begin
                        state <= S_DONE;
                    end else begin
                        emit_idx <= emit_idx + 1;
                    end
                end
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
