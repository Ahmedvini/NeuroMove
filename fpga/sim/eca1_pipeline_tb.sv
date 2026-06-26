// =============================================================================
// eca1_pipeline_tb.sv  —  Bit-exact regression for the ECA₁ orchestrator.
//
// Streams stage_conv2d_output.hex (T*C*F1 = 3000*16 = 48000 Q8.8 values) into
// eca1_pipeline.sv as (16)-wide vectors over 3000 cycles, captures the 3000
// gated output vectors, and compares against stage_eca1_output.hex.
//
// Setup: python3 scripts/q88_eca1.py
// =============================================================================

`timescale 1ns / 1ps

module eca1_pipeline_tb;

    localparam int T          = 600;
    localparam int C          = 5;
    localparam int F1         = 16;
    localparam int N_FRAMES   = T * C;            // 3000
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:N_FRAMES * F1 - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:N_FRAMES * F1 - 1];

    initial begin
        $readmemh("data/golden_q88/stage_conv2d_output.hex", in_mem);
        $readmemh("data/golden_q88/stage_eca1_output.hex",   exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic signed [DATA_WIDTH-1:0] x_in   [0:F1-1];
    logic                         x_valid;
    logic signed [DATA_WIDTH-1:0] y_out  [0:F1-1];
    logic                         y_valid;
    logic                         done;

    eca1_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F1), .KECA(3),
        .N_FRAMES(N_FRAMES),
        .INV_N(5592), .INV_N_SHIFT(24), .INV_N_WIDTH(24),
        .LUT_N(1024), .LUT_RANGE_Q(1024), .LUT_SHIFT(1),
        .WEIGHTS_FILE("data/golden_q88/stage_eca1_weights.hex"),
        .LUT_FILE    ("data/lut/sigmoid_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid),
        .y_out(y_out), .y_valid(y_valid), .done(done)
    );

    int outputs_received;
    int mismatches;
    int first_idx, first_ch;
    logic signed [DATA_WIDTH-1:0] first_got, first_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received <= 0;
            mismatches       <= 0;
            first_idx <= -1; first_ch <= -1;
            first_got <= '0; first_exp <= '0;
        end else if (y_valid) begin
            int idx;
            idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (idx < N_FRAMES) begin
                for (int i = 0; i < F1; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[idx * F1 + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 4)
                            $display("[eca1_pipe_tb] mismatch idx=%0d ch=%0d got=%h exp=%h",
                                     idx, i, y_out[i], exp_v);
                        if (mismatches == 0) begin
                            first_idx <= idx; first_ch <= i;
                            first_got <= y_out[i]; first_exp <= exp_v;
                        end
                        mismatches <= mismatches + 1;
                    end
                end
            end
        end
    end

    initial begin
        rst = 1'b1;
        x_valid = 1'b0;
        for (int i = 0; i < F1; i++) x_in[i] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < N_FRAMES; t++) begin
            @(posedge clk);
            x_valid <= 1'b1;
            for (int i = 0; i < F1; i++) x_in[i] <= in_mem[t * F1 + i];
        end
        @(posedge clk);
        x_valid <= 1'b0;
        for (int i = 0; i < F1; i++) x_in[i] <= '0;

        wait (done == 1'b1);
        repeat (4) @(posedge clk);

        $display("[eca1_pipe_tb] outputs received: %0d (expected %0d)", outputs_received, N_FRAMES);
        $display("[eca1_pipe_tb] mismatches      : %0d", mismatches);
        if (outputs_received != N_FRAMES) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[eca1_pipe_tb] first mismatch idx=%0d ch=%0d got=%h exp=%h",
                     first_idx, first_ch, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[eca1_pipe_tb] PASS");
        $finish;
    end

endmodule
