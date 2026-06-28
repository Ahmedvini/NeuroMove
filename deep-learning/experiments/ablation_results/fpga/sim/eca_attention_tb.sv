// =============================================================================
// eca_attention_tb.sv  —  Bit-exact regression for the ECA gate-compute module
//
// Drives the GAP vector from data/golden_q88/stage_eca1_input.hex, loads the
// trained Conv1D weights and the universal sigmoid LUT, and compares each
// gate output to data/golden_q88/stage_eca1_gate.hex.
//
// Requires scripts/q88_eca1.py and scripts/gen_sigmoid_lut.py to have run.
// =============================================================================

`timescale 1ns / 1ps

module eca_attention_tb;

    // --- Parameters (match q88_eca1.py for ECA₁ on the F1=16 stream) ---
    localparam int    NUM_CH       = 16;
    localparam int    KECA         = 3;
    localparam int    DATA_WIDTH   = 16;
    localparam int    COEF_WIDTH   = 16;
    localparam int    ACC_WIDTH    = 48;
    localparam int    FRAC_BITS    = 8;
    localparam int    LUT_N        = 1024;
    localparam int    LUT_RANGE_Q  = 1024;
    localparam int    LUT_SHIFT    = 1;
    localparam int    CLK_PERIOD   = 10;

    // --- Memories ---
    logic signed [DATA_WIDTH-1:0] gap_mem  [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] exp_mem  [0:NUM_CH-1];

    initial begin
        $readmemh("data/golden_q88/stage_eca1_input.hex", gap_mem);
        $readmemh("data/golden_q88/stage_eca1_gate.hex",  exp_mem);
    end

    // --- DUT ---
    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] gap_in    [0:NUM_CH-1];
    logic                         gap_valid;
    logic signed [DATA_WIDTH-1:0] gate_out  [0:NUM_CH-1];
    logic                         gate_valid;

    eca_attention #(
        .DATA_WIDTH    (DATA_WIDTH),
        .COEF_WIDTH    (COEF_WIDTH),
        .ACC_WIDTH     (ACC_WIDTH),
        .FRAC_BITS     (FRAC_BITS),
        .NUM_CH        (NUM_CH),
        .KECA          (KECA),
        .LUT_N         (LUT_N),
        .LUT_RANGE_Q   (LUT_RANGE_Q),
        .LUT_SHIFT     (LUT_SHIFT),
        .WEIGHTS_FILE  ("data/golden_q88/stage_eca1_weights.hex"),
        .LUT_FILE      ("data/lut/sigmoid_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .gap_in(gap_in),
        .gap_valid(gap_valid),
        .gate_out(gate_out),
        .gate_valid(gate_valid)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // --- Mismatch tracker (single always_ff owner) ---
    int outputs_received;
    int mismatches;
    int first_mismatch_idx;
    logic signed [DATA_WIDTH-1:0] first_mismatch_got;
    logic signed [DATA_WIDTH-1:0] first_mismatch_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received   <= 0;
            mismatches         <= 0;
            first_mismatch_idx <= -1;
            first_mismatch_got <= '0;
            first_mismatch_exp <= '0;
        end else if (gate_valid) begin
            outputs_received <= outputs_received + 1;
            for (int c = 0; c < NUM_CH; c++) begin
                if (gate_out[c] !== exp_mem[c]) begin
                    if (mismatches < 8) begin
                        $display("[eca_tb] mismatch c=%0d  got=%h  exp=%h", c, gate_out[c], exp_mem[c]);
                    end
                    if (mismatches == 0) begin
                        first_mismatch_idx <= c;
                        first_mismatch_got <= gate_out[c];
                        first_mismatch_exp <= exp_mem[c];
                    end
                    mismatches <= mismatches + 1;
                end
            end
        end
    end

    // --- Drive ---
    initial begin
        rst = 1'b1;
        gap_valid = 1'b0;
        for (int c = 0; c < NUM_CH; c++) gap_in[c] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        @(posedge clk);
        for (int c = 0; c < NUM_CH; c++) gap_in[c] <= gap_mem[c];
        gap_valid <= 1'b1;

        @(posedge clk);
        gap_valid <= 1'b0;
        for (int c = 0; c < NUM_CH; c++) gap_in[c] <= '0;

        repeat (4) @(posedge clk);

        $display("[eca_tb] outputs received: %0d", outputs_received);
        $display("[eca_tb] mismatches      : %0d", mismatches);
        if (outputs_received != 1) $fatal(1, "FAIL: expected exactly 1 gate_valid pulse");
        if (mismatches != 0) begin
            $display("[eca_tb] first mismatch c=%0d  got=%h  exp=%h",
                     first_mismatch_idx, first_mismatch_got, first_mismatch_exp);
            $fatal(1, "FAIL");
        end
        $display("[eca_tb] PASS");
        $finish;
    end

endmodule
