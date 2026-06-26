// =============================================================================
// db_atcnet_window_pipeline.sv  —  5-window sequencer for HaLT DB-ATCNet
// post-ECA₂ pipeline.
//
// Input  : ECA₂ output buffer (T_FULL × F) = (10, 32) of Q8.8.
// Output : 1-bit class (0 = Right Hand fist, 1 = Left Leg ankle).
//
// For each sliding window w in 0..N_WIN-1:
//   1. Slice eca2_buf[w..w+T_WIN-1] → chan_attn (phase 1 ingest + phase 2
//      apply). Capture (T_WIN × F) output.
//   2. Stream that into spat_attn (phase 1 + phase 2). Capture output.
//   3. Stream that into tcfn. Wait for `done`, capture last time step.
//   4. Drive dense_classifier with the (F,) last-step input → logits (2,).
//   5. Push logits into output_head with in_last on window N_WIN-1.
// output_head pulses `done` with the final argmax class.
//
// All four learnable blocks share single instances with N_WIN-banked weights;
// `window_idx` is held stable from a window's first push to the matching
// downstream completion.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_window_pipeline #(
    parameter int    DATA_WIDTH = 16,
    parameter int    F          = 32,
    parameter int    T_FULL     = 10,
    parameter int    T_WIN      = 6,
    parameter int    D_RED      = 4,
    parameter int    KH_SPAT    = 7,
    parameter int    IC_CAT     = 3,
    parameter int    K_TCFN     = 4,
    parameter int    N_CLS      = 2,
    parameter int    N_WIN      = 5,
    // sigmoid LUT params
    parameter int    SIG_LUT_N  = 1024,
    parameter int    SIG_LUT_RANGE_Q = 1024,
    parameter int    SIG_LUT_SHIFT   = 1,
    // ELU LUT params
    parameter int    ELU_LUT_N  = 256,
    parameter int    ELU_LUT_RANGE_Q = 2048,
    parameter int    ELU_LUT_SHIFT   = 3,
    // GAP reciprocal-multiply constants
    parameter int    CHAN_INV_N = 2796203,   // 2^24/T_WIN=6
    // .hex paths (all combined multi-bank files)
    parameter string CHAN_D1_W_FILE = "",
    parameter string CHAN_D1_B_FILE = "",
    parameter string CHAN_D2_W_FILE = "",
    parameter string CHAN_D2_B_FILE = "",
    parameter string SIG_LUT_FILE   = "",
    parameter string SPAT_CONV_W_FILE = "",
    parameter string TCFN_W0_FILE = "",
    parameter string TCFN_B0_FILE = "",
    parameter string TCFN_W1_FILE = "",
    parameter string TCFN_B1_FILE = "",
    parameter string TCFN_W2_FILE = "",
    parameter string TCFN_B2_FILE = "",
    parameter string TCFN_W3_FILE = "",
    parameter string TCFN_B3_FILE = "",
    parameter string ELU_LUT_FILE   = "",
    parameter string CLS_W_FILE = "",
    parameter string CLS_B_FILE = ""
) (
    input  logic                              clk,
    input  logic                              rst,

    // ECA₂ output buffer (TB or upstream pipeline drives this once before
    // pulsing `start`).
    input  logic signed [DATA_WIDTH-1:0]      eca2_buf [0:T_FULL-1][0:F-1],
    input  logic                              start,

    // Final output (one inference per `start` pulse).
    output logic [$clog2(N_CLS)-1:0]          class_out,
    output logic                              done
);

    localparam int WI_W = $clog2(N_WIN > 1 ? N_WIN : 2);
    logic [WI_W-1:0]                          window_idx;

    // -------------------------------------------------------------------------
    // Intermediate buffers
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] chan_in_buf  [0:T_WIN-1][0:F-1];  // sliced from eca2
    logic signed [DATA_WIDTH-1:0] chan_out_buf [0:T_WIN-1][0:F-1];
    logic signed [DATA_WIDTH-1:0] spat_out_buf [0:T_WIN-1][0:F-1];
    logic signed [DATA_WIDTH-1:0] tcfn_last_buf[0:F-1];

    // -------------------------------------------------------------------------
    // Block 1: cbam_channel_attn
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] chan_x_in       [0:F-1];
    logic                         chan_x_valid;
    logic                         chan_frame_last;
    logic signed [DATA_WIDTH-1:0] chan_gate_out   [0:F-1];
    logic                         chan_gate_valid;
    logic signed [DATA_WIDTH-1:0] chan_apply_x_in [0:F-1];
    logic                         chan_apply_valid;
    logic signed [DATA_WIDTH-1:0] chan_y_out      [0:F-1];
    logic                         chan_y_valid;

    cbam_channel_attn #(
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .ACC_WIDTH(32), .FRAC_BITS(8),
        .NUM_CH(F), .T_WIN(T_WIN), .D_RED(D_RED), .N_WIN(N_WIN),
        .INV_N(CHAN_INV_N), .INV_N_SHIFT(24), .INV_N_WIDTH(24),
        .LUT_N(SIG_LUT_N), .LUT_RANGE_Q(SIG_LUT_RANGE_Q), .LUT_SHIFT(SIG_LUT_SHIFT),
        .D1_W_FILE(CHAN_D1_W_FILE), .D1_B_FILE(CHAN_D1_B_FILE),
        .D2_W_FILE(CHAN_D2_W_FILE), .D2_B_FILE(CHAN_D2_B_FILE),
        .LUT_FILE (SIG_LUT_FILE)
    ) u_chan (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(chan_x_in), .x_valid(chan_x_valid), .frame_last(chan_frame_last),
        .gate_out(chan_gate_out), .gate_valid(chan_gate_valid),
        .apply_x_in(chan_apply_x_in), .apply_valid(chan_apply_valid),
        .y_out(chan_y_out), .y_valid(chan_y_valid)
    );

    // -------------------------------------------------------------------------
    // Block 2: cbam_spatial_attn
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] spat_x_in       [0:F-1];
    logic                         spat_x_valid;
    logic                         spat_frame_last;
    logic                         spat_gate_valid;
    logic signed [DATA_WIDTH-1:0] spat_gate_out   [0:T_WIN-1];
    logic signed [DATA_WIDTH-1:0] spat_apply_x_in [0:F-1];
    logic                         spat_apply_valid;
    logic signed [DATA_WIDTH-1:0] spat_y_out      [0:F-1];
    logic                         spat_y_valid;

    cbam_spatial_attn #(
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .FRAC_BITS(8),
        .NUM_CH(F), .LOG2_NUM_CH(5), .T_WIN(T_WIN), .KH(KH_SPAT), .IC_CAT(IC_CAT),
        .NORM_WIDTH(25), .QSUM_WIDTH(40), .N_WIN(N_WIN),
        .LUT_N(SIG_LUT_N), .LUT_RANGE_Q(SIG_LUT_RANGE_Q), .LUT_SHIFT(SIG_LUT_SHIFT),
        .CONV_W_FILE(SPAT_CONV_W_FILE), .LUT_FILE(SIG_LUT_FILE)
    ) u_spat (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(spat_x_in), .x_valid(spat_x_valid), .frame_last(spat_frame_last),
        .gate_valid(spat_gate_valid), .gate_out(spat_gate_out),
        .apply_x_in(spat_apply_x_in), .apply_valid(spat_apply_valid),
        .y_out(spat_y_out), .y_valid(spat_y_valid)
    );

    // -------------------------------------------------------------------------
    // Block 3: TCFN
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] tcfn_x_in     [0:F-1];
    logic                         tcfn_x_valid;
    logic                         tcfn_frame_last;
    logic signed [DATA_WIDTH-1:0] tcfn_y_out    [0:F-1];
    logic                         tcfn_y_valid;
    logic                         tcfn_done;

    tcfn #(
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .ACC_WIDTH(48), .FRAC_BITS(8),
        .T_WIN(T_WIN), .F(F), .K(K_TCFN), .N_WIN(N_WIN),
        .LUT_N(ELU_LUT_N), .LUT_RANGE_Q(ELU_LUT_RANGE_Q), .LUT_SHIFT(ELU_LUT_SHIFT),
        .W0_FILE(TCFN_W0_FILE), .B0_FILE(TCFN_B0_FILE),
        .W1_FILE(TCFN_W1_FILE), .B1_FILE(TCFN_B1_FILE),
        .W2_FILE(TCFN_W2_FILE), .B2_FILE(TCFN_B2_FILE),
        .W3_FILE(TCFN_W3_FILE), .B3_FILE(TCFN_B3_FILE),
        .LUT_FILE(ELU_LUT_FILE)
    ) u_tcfn (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(tcfn_x_in), .x_valid(tcfn_x_valid), .frame_last(tcfn_frame_last),
        .y_out(tcfn_y_out), .y_valid(tcfn_y_valid), .done(tcfn_done)
    );

    // -------------------------------------------------------------------------
    // Block 4: dense_classifier
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] cls_x_in   [0:F-1];
    logic                         cls_in_valid;
    logic signed [DATA_WIDTH-1:0] cls_logits [0:N_CLS-1];
    logic                         cls_out_valid;

    dense_classifier #(
        .DATA_WIDTH(DATA_WIDTH), .COEF_WIDTH(16), .ACC_WIDTH(40), .FRAC_BITS(8),
        .F(F), .N_CLS(N_CLS), .N_WIN(N_WIN),
        .WEIGHTS_FILE(CLS_W_FILE), .BIAS_FILE(CLS_B_FILE)
    ) u_cls (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(cls_x_in), .in_valid(cls_in_valid),
        .logits(cls_logits), .out_valid(cls_out_valid)
    );

    // -------------------------------------------------------------------------
    // Block 5: output_head
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] head_logits  [0:N_CLS-1];
    logic                         head_in_valid;
    logic                         head_in_last;

    output_head #(
        .DATA_WIDTH(DATA_WIDTH), .N_CLS(N_CLS), .N_WIN(N_WIN)
    ) u_head (
        .clk(clk), .rst(rst),
        .logits_in(head_logits), .in_valid(head_in_valid), .in_last(head_in_last),
        .class_out(class_out), .done(done)
    );

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE,
        S_SLICE,           // copy eca2_buf[w..w+T_WIN-1] to chan_in_buf
        S_CHAN_PHASE1,     // drive chan_x_valid for T_WIN cycles
        S_CHAN_WAIT_GATE,
        S_CHAN_PHASE2,     // drive chan_apply_valid + capture y_out
        S_SPAT_PHASE1,
        S_SPAT_WAIT_GATE,
        S_SPAT_PHASE2,
        S_TCFN_PUSH,
        S_TCFN_WAIT,       // wait for tcfn done; capture last step
        S_DENSE,
        S_HEAD_WAIT,       // wait for head done if last window
        S_NEXT_WIN
    } state_t;
    state_t state;

    logic [$clog2(T_WIN+1)-1:0] phase_t;     // streaming counter
    logic [$clog2(T_WIN+1)-1:0] capture_t;   // y_valid count
    logic [$clog2(T_WIN+1)-1:0] tcfn_emit_t; // tcfn y_valid count for last-step grab

    // ---- combinational helpers ----
    wire last_window = (window_idx == (N_WIN-1));

    always_ff @(posedge clk) begin
        if (rst) begin
            state            <= S_IDLE;
            window_idx       <= '0;
            phase_t          <= 0;
            capture_t        <= 0;
            tcfn_emit_t      <= 0;
            chan_x_valid     <= 1'b0;
            chan_frame_last  <= 1'b0;
            chan_apply_valid <= 1'b0;
            spat_x_valid     <= 1'b0;
            spat_frame_last  <= 1'b0;
            spat_apply_valid <= 1'b0;
            tcfn_x_valid     <= 1'b0;
            tcfn_frame_last  <= 1'b0;
            cls_in_valid     <= 1'b0;
            head_in_valid    <= 1'b0;
            head_in_last     <= 1'b0;
            for (int c = 0; c < F; c++) begin
                chan_x_in[c]       <= '0;
                chan_apply_x_in[c] <= '0;
                spat_x_in[c]       <= '0;
                spat_apply_x_in[c] <= '0;
                tcfn_x_in[c]       <= '0;
                cls_x_in[c]        <= '0;
                tcfn_last_buf[c]   <= '0;
            end
            for (int t = 0; t < T_WIN; t++) for (int c = 0; c < F; c++) begin
                chan_in_buf[t][c]  <= '0;
                chan_out_buf[t][c] <= '0;
                spat_out_buf[t][c] <= '0;
            end
            for (int o = 0; o < N_CLS; o++) head_logits[o] <= '0;
        end else begin
            // Default deassertions; specific states re-assert as needed.
            chan_x_valid     <= 1'b0;
            chan_frame_last  <= 1'b0;
            chan_apply_valid <= 1'b0;
            spat_x_valid     <= 1'b0;
            spat_frame_last  <= 1'b0;
            spat_apply_valid <= 1'b0;
            tcfn_x_valid     <= 1'b0;
            tcfn_frame_last  <= 1'b0;
            cls_in_valid     <= 1'b0;
            head_in_valid    <= 1'b0;
            head_in_last     <= 1'b0;

            // y_valid capture for chan/spat (these run in parallel with phase 2).
            if (chan_y_valid) begin
                for (int c = 0; c < F; c++) chan_out_buf[capture_t][c] <= chan_y_out[c];
                if (capture_t == T_WIN - 1) capture_t <= 0;
                else                        capture_t <= capture_t + 1;
            end
            if (spat_y_valid) begin
                for (int c = 0; c < F; c++) spat_out_buf[capture_t][c] <= spat_y_out[c];
                if (capture_t == T_WIN - 1) capture_t <= 0;
                else                        capture_t <= capture_t + 1;
            end
            if (tcfn_y_valid) begin
                if (tcfn_emit_t == T_WIN - 1) begin
                    for (int c = 0; c < F; c++) tcfn_last_buf[c] <= tcfn_y_out[c];
                    tcfn_emit_t <= 0;
                end else begin
                    tcfn_emit_t <= tcfn_emit_t + 1;
                end
            end

            unique case (state)
                S_IDLE: begin
                    if (start) begin
                        window_idx <= '0;
                        state      <= S_SLICE;
                    end
                end
                S_SLICE: begin
                    for (int t = 0; t < T_WIN; t++)
                        for (int c = 0; c < F; c++)
                            chan_in_buf[t][c] <= eca2_buf[window_idx + t][c];
                    phase_t <= 0;
                    state   <= S_CHAN_PHASE1;
                end
                S_CHAN_PHASE1: begin
                    chan_x_valid    <= 1'b1;
                    chan_frame_last <= (phase_t == T_WIN - 1);
                    for (int c = 0; c < F; c++) chan_x_in[c] <= chan_in_buf[phase_t][c];
                    if (phase_t == T_WIN - 1) begin
                        phase_t <= 0;
                        state   <= S_CHAN_WAIT_GATE;
                    end else begin
                        phase_t <= phase_t + 1;
                    end
                end
                S_CHAN_WAIT_GATE: begin
                    if (chan_gate_valid) begin
                        phase_t   <= 0;
                        capture_t <= 0;
                        state     <= S_CHAN_PHASE2;
                    end
                end
                S_CHAN_PHASE2: begin
                    chan_apply_valid <= 1'b1;
                    for (int c = 0; c < F; c++) chan_apply_x_in[c] <= chan_in_buf[phase_t][c];
                    if (phase_t == T_WIN - 1) begin
                        phase_t <= 0;
                        state   <= S_SPAT_PHASE1;
                        // We have to wait for the LAST chan_y_valid to arrive,
                        // but that fires while spat is in S_SPAT_PHASE1's first
                        // ingest cycle and the capture above still runs.
                    end else begin
                        phase_t <= phase_t + 1;
                    end
                end
                S_SPAT_PHASE1: begin
                    spat_x_valid    <= 1'b1;
                    spat_frame_last <= (phase_t == T_WIN - 1);
                    for (int c = 0; c < F; c++) spat_x_in[c] <= chan_out_buf[phase_t][c];
                    if (phase_t == T_WIN - 1) begin
                        phase_t <= 0;
                        state   <= S_SPAT_WAIT_GATE;
                    end else begin
                        phase_t <= phase_t + 1;
                    end
                end
                S_SPAT_WAIT_GATE: begin
                    if (spat_gate_valid) begin
                        phase_t   <= 0;
                        capture_t <= 0;
                        state     <= S_SPAT_PHASE2;
                    end
                end
                S_SPAT_PHASE2: begin
                    spat_apply_valid <= 1'b1;
                    for (int c = 0; c < F; c++) spat_apply_x_in[c] <= chan_out_buf[phase_t][c];
                    if (phase_t == T_WIN - 1) begin
                        phase_t <= 0;
                        state   <= S_TCFN_PUSH;
                    end else begin
                        phase_t <= phase_t + 1;
                    end
                end
                S_TCFN_PUSH: begin
                    tcfn_x_valid    <= 1'b1;
                    tcfn_frame_last <= (phase_t == T_WIN - 1);
                    for (int c = 0; c < F; c++) tcfn_x_in[c] <= spat_out_buf[phase_t][c];
                    if (phase_t == T_WIN - 1) begin
                        phase_t <= 0;
                        state   <= S_TCFN_WAIT;
                        tcfn_emit_t <= 0;
                    end else begin
                        phase_t <= phase_t + 1;
                    end
                end
                S_TCFN_WAIT: begin
                    if (tcfn_done) begin
                        state <= S_DENSE;
                    end
                end
                S_DENSE: begin
                    cls_in_valid <= 1'b1;
                    for (int c = 0; c < F; c++) cls_x_in[c] <= tcfn_last_buf[c];
                    state <= S_HEAD_WAIT;
                end
                S_HEAD_WAIT: begin
                    if (cls_out_valid) begin
                        for (int o = 0; o < N_CLS; o++) head_logits[o] <= cls_logits[o];
                        head_in_valid <= 1'b1;
                        head_in_last  <= last_window;
                        state         <= S_NEXT_WIN;
                    end
                end
                S_NEXT_WIN: begin
                    // The pulse above lasted one cycle; head is now accumulating.
                    if (last_window) begin
                        state <= S_IDLE;     // wait for done pulse from head
                    end else begin
                        window_idx <= window_idx + 1;
                        state      <= S_SLICE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
