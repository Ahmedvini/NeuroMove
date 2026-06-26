// =============================================================================
// db_atcnet_post_eca1.sv  —  Post-ECA₁ orchestrator for HaLT DB-ATCNet.
//
// Wires the two parallel feature branches + the merge + ECA₂ + buffer into
// one module that takes the ECA₁ gated stream and produces the (T_FULL × F)
// = (10 × 32) feature map the window pipeline consumes.
//
//   Streaming input  (F1)-wide per (t,c) for T_IN × C_IN cycles
//     ├─→ branch_pipeline(D=2)  →  (T_FULL × F) Branch A out
//     └─→ branch_pipeline(D=4)  →  (T_FULL × F) Branch B out
//
//   For t in 0..T_FULL-1:
//     summed[t]  = sat( branchA[t] + branchB[t] )       (sat_add, 1 cycle)
//     summed[t]  → eca1_pipeline (parameterised NUM_CH=F, N_FRAMES=T_FULL,
//                  INV_N for 1/T_FULL)
//   eca2_pipeline produces 10 (F)-wide vectors which are captured into
//   the output buffer eca2_buf[T_FULL][F].
//
// Reuses the verified components:
//   rtl/conv/branch_pipeline.sv (×2) — Branch A (D=2) and B (D=4)
//   rtl/conv/sat_add.sv
//   rtl/attention/eca1_pipeline.sv (reparameterised for ECA₂)
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_post_eca1 #(
    parameter int    DATA_WIDTH    = 16,
    parameter int    C_IN          = 5,
    parameter int    F1            = 16,
    parameter int    F             = 32,    // post-branch feature dim
    parameter int    KE_SEP        = 16,
    parameter int    T_IN          = 600,
    parameter int    POOL1         = 8,
    parameter int    POOL2         = 7,
    parameter int    T_FULL        = 10,    // T_IN / (POOL1 * POOL2)
    parameter int    INV_POOL1     = 2097152,
    parameter int    INV_POOL2     = 2396745,
    parameter int    ECA2_INV_N    = 1677722,   // round(2^24 / 10)
    parameter int    SIG_LUT_N     = 1024,
    parameter int    SIG_LUT_RANGE_Q = 1024,
    parameter int    SIG_LUT_SHIFT   = 1,
    parameter int    ELU_LUT_N     = 256,
    parameter int    ELU_LUT_RANGE_Q = 2048,
    parameter int    ELU_LUT_SHIFT = 3,
    // Branch A weight files
    parameter string BRANCHA_DWISE_W_FILE = "",
    parameter string BRANCHA_DWISE_B_FILE = "",
    parameter string BRANCHA_SEP_W_FILE   = "",
    parameter string BRANCHA_SEP_B_FILE   = "",
    // Branch B weight files
    parameter string BRANCHB_DWISE_W_FILE = "",
    parameter string BRANCHB_DWISE_B_FILE = "",
    parameter string BRANCHB_SEP_W_FILE   = "",
    parameter string BRANCHB_SEP_B_FILE   = "",
    // ECA₂ weights + LUT files
    parameter string ECA2_W_FILE          = "",
    parameter string SIG_LUT_FILE         = "",
    parameter string ELU_LUT_FILE         = ""
) (
    input  logic                            clk,
    input  logic                            rst,

    // Streaming ECA₁-gated input.
    input  logic signed [DATA_WIDTH-1:0]    x_in [0:F1-1],
    input  logic                            x_valid,
    input  logic                            frame_last,   // 1 on the C_IN-th electrode

    // Final (T_FULL × F) feature map ready for the window pipeline.
    output logic signed [DATA_WIDTH-1:0]    eca2_buf [0:T_FULL-1][0:F-1],
    output logic                            done
);

    // -------------------------------------------------------------------------
    // Branch A (D=2, F2=32)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] branchA_y [0:F-1];
    logic                         branchA_y_valid;
    logic                         branchA_done;

    branch_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .D(2), .F2(32),
        .F_SEP_OUT(F), .KE_SEP(KE_SEP),
        .T_IN(T_IN), .POOL1(POOL1), .POOL2(POOL2),
        .INV_POOL1(INV_POOL1), .INV_POOL2(INV_POOL2),
        .ELU_LUT_N(ELU_LUT_N), .ELU_LUT_RANGE_Q(ELU_LUT_RANGE_Q), .ELU_LUT_SHIFT(ELU_LUT_SHIFT),
        .DWISE_W_FILE(BRANCHA_DWISE_W_FILE), .DWISE_B_FILE(BRANCHA_DWISE_B_FILE),
        .SEP_W_FILE  (BRANCHA_SEP_W_FILE),   .SEP_B_FILE  (BRANCHA_SEP_B_FILE),
        .ELU_LUT_FILE(ELU_LUT_FILE)
    ) u_branchA (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .y_out(branchA_y), .y_valid(branchA_y_valid), .done(branchA_done)
    );

    // -------------------------------------------------------------------------
    // Branch B (D=4, F2=64)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] branchB_y [0:F-1];
    logic                         branchB_y_valid;
    logic                         branchB_done;

    branch_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .D(4), .F2(64),
        .F_SEP_OUT(F), .KE_SEP(KE_SEP),
        .T_IN(T_IN), .POOL1(POOL1), .POOL2(POOL2),
        .INV_POOL1(INV_POOL1), .INV_POOL2(INV_POOL2),
        .ELU_LUT_N(ELU_LUT_N), .ELU_LUT_RANGE_Q(ELU_LUT_RANGE_Q), .ELU_LUT_SHIFT(ELU_LUT_SHIFT),
        .DWISE_W_FILE(BRANCHB_DWISE_W_FILE), .DWISE_B_FILE(BRANCHB_DWISE_B_FILE),
        .SEP_W_FILE  (BRANCHB_SEP_W_FILE),   .SEP_B_FILE  (BRANCHB_SEP_B_FILE),
        .ELU_LUT_FILE(ELU_LUT_FILE)
    ) u_branchB (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .y_out(branchB_y), .y_valid(branchB_y_valid), .done(branchB_done)
    );

    // -------------------------------------------------------------------------
    // Branch output buffers (T_FULL × F each)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] branchA_buf [0:T_FULL-1][0:F-1];
    logic signed [DATA_WIDTH-1:0] branchB_buf [0:T_FULL-1][0:F-1];
    logic [$clog2(T_FULL+1)-1:0]  branchA_capture_cnt;
    logic [$clog2(T_FULL+1)-1:0]  branchB_capture_cnt;

    // -------------------------------------------------------------------------
    // sat_add (1-cycle latency) — drives ECA₂ pipeline
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] add_a   [0:F-1];
    logic signed [DATA_WIDTH-1:0] add_b   [0:F-1];
    logic                         add_in_valid;
    logic signed [DATA_WIDTH-1:0] add_y   [0:F-1];
    logic                         add_y_valid;

    sat_add #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F)
    ) u_add (
        .clk(clk), .rst(rst),
        .a_in(add_a), .b_in(add_b), .in_valid(add_in_valid),
        .y_out(add_y), .y_valid(add_y_valid)
    );

    // -------------------------------------------------------------------------
    // ECA₂ pipeline (eca1_pipeline reused with NUM_CH=F, N_FRAMES=T_FULL)
    // -------------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] eca2_y      [0:F-1];
    logic                         eca2_y_valid;
    logic                         eca2_done;

    eca1_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F), .KECA(3),
        .N_FRAMES(T_FULL),
        .INV_N(ECA2_INV_N), .INV_N_SHIFT(24), .INV_N_WIDTH(24),
        .LUT_N(SIG_LUT_N), .LUT_RANGE_Q(SIG_LUT_RANGE_Q), .LUT_SHIFT(SIG_LUT_SHIFT),
        .WEIGHTS_FILE(ECA2_W_FILE), .LUT_FILE(SIG_LUT_FILE)
    ) u_eca2 (
        .clk(clk), .rst(rst),
        .x_in(add_y), .x_valid(add_y_valid),
        .y_out(eca2_y), .y_valid(eca2_y_valid), .done(eca2_done)
    );

    // -------------------------------------------------------------------------
    // FSM: capture branch outputs → drive sat_add → wait for ECA₂ → done
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_RUN_BRANCHES,
        S_WAIT_BOTH_DONE,
        S_ADD_DRIVE,
        S_ADD_DRAIN,
        S_WAIT_ECA2,
        S_DONE
    } state_t;
    state_t state;

    logic [$clog2(T_FULL+1)-1:0] add_drive_cnt;
    logic [$clog2(T_FULL+1)-1:0] eca2_capture_cnt;

    always_ff @(posedge clk) begin
        if (rst) begin
            state               <= S_RUN_BRANCHES;
            branchA_capture_cnt <= 0;
            branchB_capture_cnt <= 0;
            add_drive_cnt       <= 0;
            eca2_capture_cnt    <= 0;
            add_in_valid        <= 1'b0;
            done                <= 1'b0;
            for (int t = 0; t < T_FULL; t++) for (int c = 0; c < F; c++) begin
                branchA_buf[t][c] <= '0;
                branchB_buf[t][c] <= '0;
                eca2_buf[t][c]    <= '0;
            end
            for (int c = 0; c < F; c++) begin
                add_a[c] <= '0;
                add_b[c] <= '0;
            end
        end else begin
            done <= 1'b0;

            // Continuous captures for the branches.
            if (branchA_y_valid && branchA_capture_cnt < T_FULL) begin
                for (int c = 0; c < F; c++) branchA_buf[branchA_capture_cnt][c] <= branchA_y[c];
                branchA_capture_cnt <= branchA_capture_cnt + 1;
            end
            if (branchB_y_valid && branchB_capture_cnt < T_FULL) begin
                for (int c = 0; c < F; c++) branchB_buf[branchB_capture_cnt][c] <= branchB_y[c];
                branchB_capture_cnt <= branchB_capture_cnt + 1;
            end

            // ECA₂ output capture into the final buffer.
            if (eca2_y_valid && eca2_capture_cnt < T_FULL) begin
                for (int c = 0; c < F; c++) eca2_buf[eca2_capture_cnt][c] <= eca2_y[c];
                eca2_capture_cnt <= eca2_capture_cnt + 1;
            end

            unique case (state)
                S_RUN_BRANCHES: begin
                    if (branchA_done && branchB_done) begin
                        // (Both done at slightly different times; FSM waits for
                        //  the slower one.)
                        add_drive_cnt <= 0;
                        state         <= S_ADD_DRIVE;
                    end else if (branchA_done || branchB_done) begin
                        state <= S_WAIT_BOTH_DONE;
                    end
                end
                S_WAIT_BOTH_DONE: begin
                    if (branchA_done || branchB_done || (branchA_capture_cnt == T_FULL && branchB_capture_cnt == T_FULL)) begin
                        add_drive_cnt <= 0;
                        state         <= S_ADD_DRIVE;
                    end
                end
                S_ADD_DRIVE: begin
                    add_in_valid <= 1'b1;
                    for (int c = 0; c < F; c++) begin
                        add_a[c] <= branchA_buf[add_drive_cnt][c];
                        add_b[c] <= branchB_buf[add_drive_cnt][c];
                    end
                    if (add_drive_cnt == T_FULL - 1) begin
                        state <= S_ADD_DRAIN;
                    end
                    add_drive_cnt <= add_drive_cnt + 1;
                end
                S_ADD_DRAIN: begin
                    add_in_valid <= 1'b0;
                    state        <= S_WAIT_ECA2;
                end
                S_WAIT_ECA2: begin
                    if (eca2_done) begin
                        state <= S_DONE;
                    end
                end
                S_DONE: begin
                    done                <= 1'b1;
                    state               <= S_RUN_BRANCHES;
                    branchA_capture_cnt <= 0;
                    branchB_capture_cnt <= 0;
                    eca2_capture_cnt    <= 0;
                end
                default: state <= S_RUN_BRANCHES;
            endcase
        end
    end

endmodule
