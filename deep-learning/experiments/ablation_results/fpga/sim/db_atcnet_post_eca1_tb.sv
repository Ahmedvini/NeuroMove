// =============================================================================
// db_atcnet_post_eca1_tb.sv  —  Drives the ECA₁-gated stream into the
// post-ECA₁ orchestrator and checks the resulting (T_FULL × F) buffer
// against stage_eca2_output.hex.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_post_eca1_tb;

    localparam int T_IN     = 600;
    localparam int C_IN     = 5;
    localparam int F1       = 16;
    localparam int F        = 32;
    localparam int T_FULL   = 10;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_IN * C_IN * F1 - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_FULL * F - 1];

    initial begin
        $readmemh("data/golden_q88/stage_eca1_output.hex", in_mem);
        $readmemh("data/golden_q88/stage_eca2_output.hex", exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic signed [DATA_WIDTH-1:0] x_in [0:F1-1];
    logic                         x_valid;
    logic                         frame_last;
    logic signed [DATA_WIDTH-1:0] eca2_buf [0:T_FULL-1][0:F-1];
    logic                         done;

    db_atcnet_post_eca1 #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .F(F), .KE_SEP(16),
        .T_IN(T_IN), .POOL1(8), .POOL2(7), .T_FULL(T_FULL),
        .INV_POOL1(2097152), .INV_POOL2(2396745), .ECA2_INV_N(1677722),
        .SIG_LUT_N(1024), .SIG_LUT_RANGE_Q(1024), .SIG_LUT_SHIFT(1),
        .ELU_LUT_N(256), .ELU_LUT_RANGE_Q(2048), .ELU_LUT_SHIFT(3),
        .BRANCHA_DWISE_W_FILE("data/golden_q88/stage_branchA_dwise_weights.hex"),
        .BRANCHA_DWISE_B_FILE("data/golden_q88/stage_branchA_dwise_bias.hex"),
        .BRANCHA_SEP_W_FILE  ("data/golden_q88/stage_branchA_sep_weights.hex"),
        .BRANCHA_SEP_B_FILE  ("data/golden_q88/stage_branchA_sep_bias.hex"),
        .BRANCHB_DWISE_W_FILE("data/golden_q88/stage_branchB_dwise_weights.hex"),
        .BRANCHB_DWISE_B_FILE("data/golden_q88/stage_branchB_dwise_bias.hex"),
        .BRANCHB_SEP_W_FILE  ("data/golden_q88/stage_branchB_sep_weights.hex"),
        .BRANCHB_SEP_B_FILE  ("data/golden_q88/stage_branchB_sep_bias.hex"),
        .ECA2_W_FILE         ("data/golden_q88/stage_eca2_weights.hex"),
        .SIG_LUT_FILE        ("data/lut/sigmoid_q88.hex"),
        .ELU_LUT_FILE        ("data/lut/elu_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .eca2_buf(eca2_buf), .done(done)
    );

    initial begin
        rst = 1'b1;
        x_valid = 1'b0; frame_last = 1'b0;
        for (int i = 0; i < F1; i++) x_in[i] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        // Stream T_IN timesteps × C_IN electrodes = 3000 (F1)-wide vectors.
        for (int t = 0; t < T_IN; t++) begin
            for (int c = 0; c < C_IN; c++) begin
                @(posedge clk);
                x_valid    <= 1'b1;
                frame_last <= (c == C_IN - 1);
                for (int i = 0; i < F1; i++) x_in[i] <= in_mem[(t * C_IN + c) * F1 + i];
            end
        end
        @(posedge clk);
        x_valid    <= 1'b0;
        frame_last <= 1'b0;
        for (int i = 0; i < F1; i++) x_in[i] <= '0;

        // Wait for done (Branch B is the slow path at ~95k cycles).
        repeat (200000) begin
            @(posedge clk);
            if (done) break;
        end
        if (!done) $fatal(1, "[post_eca1_tb] timeout waiting for done");

        // Verify eca2_buf against expected.
        begin
            int mismatches;
            mismatches = 0;
            for (int t = 0; t < T_FULL; t++) begin
                for (int i = 0; i < F; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[t * F + i];
                    if (eca2_buf[t][i] !== exp_v) begin
                        if (mismatches < 4)
                            $display("[post_eca1_tb] mismatch t=%0d i=%0d got=%h exp=%h",
                                     t, i, eca2_buf[t][i], exp_v);
                        mismatches++;
                    end
                end
            end
            $display("[post_eca1_tb] mismatches: %0d", mismatches);
            if (mismatches != 0) $fatal(1, "[post_eca1_tb] FAIL");
        end
        $display("[post_eca1_tb] PASS");
        $finish;
    end

endmodule
