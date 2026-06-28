// =============================================================================
// branch_pipeline.sv  —  One branch of the HaLT DB-ATCNet feature path.
//
// Wraps the full sub-chain depthwise → BN-fold → ELU → AvgPool(8) →
// separable Conv2D + BN-fold → ELU → AvgPool(7) for a single branch.
// Branch A uses D=2 / F2=32 ; Branch B uses D=4 / F2=64. F_SEP_OUT is
// always 32. Bit-exact match to scripts/q88_layer4.py + q88_layer5.py +
// q88_layer6.py + q88_layer7.py + q88_layer8_9.py (or the Branch B
// equivalents in q88_branchB.py).
//
// Input  : streaming (F1)-wide vectors per (t, c). C_IN electrodes per
//          timestep, T_IN timesteps total. frame_last marks the C_IN-th
//          electrode of each timestep.
// Output : F_SEP_OUT-wide vectors, T_POOL2 of them, emitted with y_valid.
//          `done` pulses once when the full chain completes.
//
// Internally uses the synth-fit conv1d_temporal_tm (F_OUT-parallel MACs,
// KE*F_IN cycles per output sample).
// =============================================================================

`timescale 1ns / 1ps

module branch_pipeline #(
    parameter int    DATA_WIDTH    = 16,
    parameter int    C_IN          = 5,
    parameter int    F1            = 16,
    parameter int    D             = 2,
    parameter int    F2            = 32,           // = F1 * D
    parameter int    F_SEP_OUT     = 32,
    parameter int    KE_SEP        = 16,
    parameter int    T_IN          = 600,
    parameter int    POOL1         = 8,            // post-dwise temporal pool
    parameter int    POOL2         = 7,            // post-sep temporal pool
    parameter int    INV_POOL1     = 2097152,      // round(2^24 / 8)
    parameter int    INV_POOL2     = 2396745,      // round(2^24 / 7)
    parameter int    ELU_LUT_N     = 256,
    parameter int    ELU_LUT_RANGE_Q = 2048,
    parameter int    ELU_LUT_SHIFT = 3,
    parameter string DWISE_W_FILE  = "",
    parameter string DWISE_B_FILE  = "",
    parameter string SEP_W_FILE    = "",
    parameter string SEP_B_FILE    = "",
    parameter string ELU_LUT_FILE  = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    // Streaming input (one (F1)-wide vector per cycle).
    input  logic signed [DATA_WIDTH-1:0]    x_in [0:F1-1],
    input  logic                            x_valid,
    input  logic                            frame_last,

    // Streaming output (T_POOL2 vectors of F_SEP_OUT each).
    output logic signed [DATA_WIDTH-1:0]    y_out [0:F_SEP_OUT-1],
    output logic                            y_valid,
    output logic                            done
);

    localparam int T_POOL1 = T_IN / POOL1;          // 600/8 = 75
    localparam int T_POOL2 = T_POOL1 / POOL2;       // 75/7 = 10
    localparam int N_POOL2_USED = T_POOL2 * POOL2;  // 70
    localparam int PAD_TOP = (KE_SEP - 1) / 2;      // 7
    localparam int PAD_BOT = KE_SEP - 1 - PAD_TOP;  // 8
    localparam int T_TOTAL = PAD_TOP + T_POOL1 + PAD_BOT;  // 90

    // -------------------------------------------------------------------------
    // Stage 1: depthwise + BN-fold (F1, D) → (F2)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] dwise_y [0:F2-1];
    logic                         dwise_y_valid;

    depthwise_spatial #(
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .ACC_WIDTH(48), .FRAC_BITS(8),
        .C_IN(C_IN), .F1(F1), .D(D), .F2(F2),
        .WEIGHTS_FILE(DWISE_W_FILE), .BIAS_FILE(DWISE_B_FILE)
    ) u_dwise (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .y_out(dwise_y), .y_valid(dwise_y_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 2: ELU (post-dwise)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] elu1_y [0:F2-1];
    logic                         elu1_y_valid;

    elu #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F2),
        .LUT_N(ELU_LUT_N), .LUT_RANGE_Q(ELU_LUT_RANGE_Q), .LUT_SHIFT(ELU_LUT_SHIFT),
        .LUT_FILE(ELU_LUT_FILE)
    ) u_elu1 (
        .clk(clk), .rst(rst),
        .x_in(dwise_y), .x_valid(dwise_y_valid),
        .y_out(elu1_y), .y_valid(elu1_y_valid)
    );

    // -------------------------------------------------------------------------
    // Stage 3: AvgPool(POOL1) — outputs every POOL1 inputs
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] pool1_y [0:F2-1];
    logic                         pool1_y_valid;

    avg_pool_time #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F2), .ACC_WIDTH(32),
        .POOL(POOL1), .INV_POOL(INV_POOL1), .POOL_SHIFT(24), .INV_POOL_WIDTH(24)
    ) u_pool1 (
        .clk(clk), .rst(rst),
        .x_in(elu1_y), .x_valid(elu1_y_valid),
        .y_out(pool1_y), .y_valid(pool1_y_valid)
    );

    // -------------------------------------------------------------------------
    // Buffer between pool1 and conv1d_tm: T_POOL1 rows × (F2*DATA_WIDTH) bits.
    // Stored as a 1D unpacked array of FLAT bit-vectors so Vivado infers one
    // wide BRAM (the previous 2D-unpacked form `pre_conv_buf[T][F2]` falls
    // back to FFs because the per-channel for-loop access pattern needs F2
    // simultaneous ports).
    // -------------------------------------------------------------------------
    localparam int PRE_W = F2 * DATA_WIDTH;

    (* ram_style = "block" *)
    logic [PRE_W-1:0] pre_conv_buf [0:T_POOL1-1];

    // Pack pool1_y into a flat word for the wide write port.
    logic [PRE_W-1:0] pool1_y_flat;
    for (genvar gi = 0; gi < F2; gi++) begin : g_pack_pool1
        assign pool1_y_flat[gi*DATA_WIDTH +: DATA_WIDTH] = pool1_y[gi];
    end

    // BRAM output register for the conv1d-drive path + pad-mask register.
    logic [PRE_W-1:0] pre_conv_buf_rd_flat;
    logic             pre_pad_reg;

    // -------------------------------------------------------------------------
    // Stage 4: conv1d_temporal_tm — F_IN=F2 → F_SEP_OUT, KE=KE_SEP, same pad.
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] sep_in [0:F2-1];
    logic                         sep_in_valid;
    logic                         sep_in_ready;
    logic signed [DATA_WIDTH-1:0] sep_y  [0:F_SEP_OUT-1];
    logic                         sep_y_valid;

    conv1d_temporal_tm #(
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .ACC_WIDTH(48), .FRAC_BITS(8),
        .F_IN(F2), .F_OUT(F_SEP_OUT), .KE(KE_SEP),
        .WEIGHTS_FILE(SEP_W_FILE), .BIAS_FILE(SEP_B_FILE)
    ) u_sep (
        .clk(clk), .rst(rst),
        .in_sample(sep_in), .in_valid(sep_in_valid), .in_ready(sep_in_ready),
        .y_out(sep_y), .y_valid(sep_y_valid)
    );

    // -------------------------------------------------------------------------
    // Buffer for the 75 valid sep outputs (skip the first KE-1=15 padded
    // ones). Same flat-bit-vector layout as pre_conv_buf so Vivado infers a
    // single wide BRAM instead of F_SEP_OUT parallel FF chains.
    // -------------------------------------------------------------------------
    localparam int POST_W = F_SEP_OUT * DATA_WIDTH;

    (* ram_style = "block" *)
    logic [POST_W-1:0] post_conv_buf [0:T_POOL1-1];

    // Pack sep_y into a flat word for the wide write port.
    logic [POST_W-1:0] sep_y_flat;
    for (genvar gi = 0; gi < F_SEP_OUT; gi++) begin : g_pack_sep_y
        assign sep_y_flat[gi*DATA_WIDTH +: DATA_WIDTH] = sep_y[gi];
    end

    // BRAM output register for the elu2-drive path.
    logic [POST_W-1:0] post_conv_buf_rd_flat;

    // Unpack pre_conv_buf_rd_flat to sep_in combinationally with the pad-zero
    // mask. The matching unpack for elu2_in lives below, after that array is
    // declared (SV requires the target signal to exist before a continuous
    // assignment references it).
    for (genvar gi = 0; gi < F2; gi++) begin : g_unpack_sep_in
        assign sep_in[gi] = pre_pad_reg ? '0
                          : $signed(pre_conv_buf_rd_flat[gi*DATA_WIDTH +: DATA_WIDTH]);
    end

    // -------------------------------------------------------------------------
    // Stage 5/6: ELU + AvgPool(POOL2) — driven during the POST phase
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] elu2_in   [0:F_SEP_OUT-1];
    logic                         elu2_in_valid;
    logic signed [DATA_WIDTH-1:0] elu2_y    [0:F_SEP_OUT-1];
    logic                         elu2_y_valid;

    // Combinational unpack of the post_conv_buf BRAM output register.
    for (genvar gi = 0; gi < F_SEP_OUT; gi++) begin : g_unpack_elu2_in
        assign elu2_in[gi] = $signed(post_conv_buf_rd_flat[gi*DATA_WIDTH +: DATA_WIDTH]);
    end

    elu #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F_SEP_OUT),
        .LUT_N(ELU_LUT_N), .LUT_RANGE_Q(ELU_LUT_RANGE_Q), .LUT_SHIFT(ELU_LUT_SHIFT),
        .LUT_FILE(ELU_LUT_FILE)
    ) u_elu2 (
        .clk(clk), .rst(rst),
        .x_in(elu2_in), .x_valid(elu2_in_valid),
        .y_out(elu2_y), .y_valid(elu2_y_valid)
    );

    logic signed [DATA_WIDTH-1:0] pool2_y [0:F_SEP_OUT-1];
    logic                         pool2_y_valid;

    avg_pool_time #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F_SEP_OUT), .ACC_WIDTH(32),
        .POOL(POOL2), .INV_POOL(INV_POOL2), .POOL_SHIFT(24), .INV_POOL_WIDTH(24)
    ) u_pool2 (
        .clk(clk), .rst(rst),
        .x_in(elu2_y), .x_valid(elu2_y_valid),
        .y_out(pool2_y), .y_valid(pool2_y_valid)
    );

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_INGEST,       // pass x through dwise+elu+pool1; capture pool1 outputs.
        S_CONV_DRIVE,   // drive the T_TOTAL inputs into conv1d_tm with handshake.
        S_CONV_WAIT,    // wait for the last sep output (15 cycles after last input).
        S_POST,         // stream post_conv_buf[0..N_POOL2_USED-1] through elu+pool2.
        S_DRAIN,
        S_DONE
    } state_t;
    state_t state;

    logic [$clog2(T_POOL1+1)-1:0] pool1_capture_cnt;     // 0..T_POOL1 inputs captured to pre_conv_buf
    logic [$clog2(T_TOTAL+1)-1:0] sep_drive_idx;         // next input to send to conv1d_tm (0..T_TOTAL-1)
    logic [$clog2(T_TOTAL+2)-1:0] sep_recv_idx;          // count of sep_y_valid pulses received so far
    logic [$clog2(T_POOL1+1)-1:0] post_drive_cnt;        // input index into elu2/pool2 during POST
    logic [$clog2(T_POOL2+1)-1:0] pool2_out_cnt;         // # of pool2 outputs received this inference

    // sep_sample_at_idx() removed: the pad-zero mask now lives on pre_pad_reg
    // and the per-channel slicing is done by the g_unpack_sep_in continuous
    // assigns above. This keeps a single addressable read port on the BRAM.

    always_ff @(posedge clk) begin
        if (rst) begin
            state             <= S_IDLE;
            pool1_capture_cnt <= 0;
            sep_drive_idx     <= 0;
            sep_recv_idx      <= 0;
            post_drive_cnt    <= 0;
            pool2_out_cnt     <= 0;
            sep_in_valid      <= 1'b0;
            elu2_in_valid     <= 1'b0;
            y_valid           <= 1'b0;
            done              <= 1'b0;
            pre_conv_buf_rd_flat  <= '0;
            post_conv_buf_rd_flat <= '0;
            pre_pad_reg           <= 1'b1;
            for (int i = 0; i < F_SEP_OUT; i++) y_out[i]   <= '0;
        end else begin
            done    <= 1'b0;
            y_valid <= 1'b0;

            // Pool1 output capture during ingest — single wide BRAM write.
            if (pool1_y_valid && state == S_INGEST) begin
                pre_conv_buf[pool1_capture_cnt] <= pool1_y_flat;
                pool1_capture_cnt <= pool1_capture_cnt + 1;
            end

            // Sep output capture: count y_valid pulses, store the (KE-1..KE-1+T_POOL1-1)
            // range (the in-bounds outputs of the same-padded convolution).
            if (sep_y_valid && (state == S_CONV_DRIVE || state == S_CONV_WAIT)) begin
                if (sep_recv_idx >= (KE_SEP - 1)
                    && sep_recv_idx < (KE_SEP - 1) + T_POOL1) begin
                    post_conv_buf[sep_recv_idx - (KE_SEP - 1)] <= sep_y_flat;
                end
                sep_recv_idx <= sep_recv_idx + 1;
            end

            unique case (state)
                S_IDLE: begin
                    if (x_valid) begin
                        pool1_capture_cnt <= 0;
                        state             <= S_INGEST;
                    end
                end
                S_INGEST: begin
                    if (pool1_capture_cnt == T_POOL1 - 1 && pool1_y_valid) begin
                        sep_drive_idx <= 0;
                        sep_recv_idx  <= 0;
                        state         <= S_CONV_DRIVE;
                    end
                end
                S_CONV_DRIVE: begin
                    // Standard valid/ready: drive valid+data; the cycle the DUT
                    // also reports ready, the transaction completes (advance idx).
                    // The data path is now a single wide BRAM read into
                    // pre_conv_buf_rd_flat; sep_in is unpacked combinationally
                    // with the pad-zero mask via pre_pad_reg.
                    sep_in_valid           <= 1'b1;
                    pre_conv_buf_rd_flat   <= pre_conv_buf[sep_drive_idx - PAD_TOP];
                    pre_pad_reg            <= (sep_drive_idx < PAD_TOP)
                                            || (sep_drive_idx >= PAD_TOP + T_POOL1);
                    if (sep_in_valid && sep_in_ready) begin
                        // Just accepted; advance to the next input on the next cycle.
                        sep_in_valid <= 1'b0;
                        if (sep_drive_idx == T_TOTAL - 1) begin
                            state <= S_CONV_WAIT;
                        end else begin
                            sep_drive_idx <= sep_drive_idx + 1;
                        end
                    end
                end
                S_CONV_WAIT: begin
                    sep_in_valid <= 1'b0;
                    // Wait for the last valid sep output (recv_idx == KE-1+T_POOL1).
                    if (sep_recv_idx == (KE_SEP - 1) + T_POOL1) begin
                        post_drive_cnt <= 0;
                        state          <= S_POST;
                    end
                end
                S_POST: begin
                    // Wide BRAM read into post_conv_buf_rd_flat; elu2_in is
                    // unpacked combinationally via g_unpack_elu2_in.
                    elu2_in_valid         <= 1'b1;
                    post_conv_buf_rd_flat <= post_conv_buf[post_drive_cnt];
                    if (post_drive_cnt == N_POOL2_USED - 1) begin
                        state <= S_DRAIN;
                    end
                    post_drive_cnt <= post_drive_cnt + 1;
                end
                S_DRAIN: begin
                    elu2_in_valid <= 1'b0;
                    // Wait until all T_POOL2 pool2 outputs have been emitted.
                    if (pool2_out_cnt == T_POOL2) state <= S_DONE;
                end
                S_DONE: begin
                    done          <= 1'b1;
                    pool2_out_cnt <= 0;
                    state         <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase

            // pool2 outputs are the branch output.
            if (pool2_y_valid) begin
                for (int i = 0; i < F_SEP_OUT; i++) y_out[i] <= pool2_y[i];
                y_valid       <= 1'b1;
                pool2_out_cnt <= pool2_out_cnt + 1;
            end
        end
    end

endmodule
