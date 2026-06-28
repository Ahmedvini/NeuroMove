// =============================================================================
// branch_pipeline_tb.sv  —  Bit-exact regression for the full single-branch
// chain (dwise + ELU + AvgPool(8) + sep + ELU + AvgPool(7)) using Branch A.
//
// Setup: python3 scripts/q88_layer4.py … q88_layer8_9.py
// =============================================================================

`timescale 1ns / 1ps

module branch_pipeline_tb;

    localparam int T_IN      = 600;
    localparam int C_IN      = 5;
    localparam int F1        = 16;
    localparam int D         = 2;
    localparam int F2        = F1 * D;     // 32
    localparam int F_SEP_OUT = 32;
    localparam int T_POOL2   = 10;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_IN * C_IN * F1 - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_POOL2 * F_SEP_OUT - 1];

    initial begin
        $readmemh("data/golden_q88/stage_branchA_dwise_input.hex",   in_mem);
        $readmemh("data/golden_q88/stage_branchA_pool2_output.hex",  exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic signed [DATA_WIDTH-1:0] x_in [0:F1-1];
    logic                         x_valid;
    logic                         frame_last;
    logic signed [DATA_WIDTH-1:0] y_out [0:F_SEP_OUT-1];
    logic                         y_valid;
    logic                         done;

    branch_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .D(D), .F2(F2),
        .F_SEP_OUT(F_SEP_OUT), .KE_SEP(16),
        .T_IN(T_IN), .POOL1(8), .POOL2(7),
        .INV_POOL1(2097152), .INV_POOL2(2396745),
        .ELU_LUT_N(256), .ELU_LUT_RANGE_Q(2048), .ELU_LUT_SHIFT(3),
        .DWISE_W_FILE("data/golden_q88/stage_branchA_dwise_weights.hex"),
        .DWISE_B_FILE("data/golden_q88/stage_branchA_dwise_bias.hex"),
        .SEP_W_FILE  ("data/golden_q88/stage_branchA_sep_weights.hex"),
        .SEP_B_FILE  ("data/golden_q88/stage_branchA_sep_bias.hex"),
        .ELU_LUT_FILE("data/lut/elu_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .y_out(y_out), .y_valid(y_valid), .done(done)
    );

    int outputs_received;
    int mismatches;
    int first_t, first_i;
    logic signed [DATA_WIDTH-1:0] first_got, first_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received <= 0;
            mismatches       <= 0;
            first_t <= -1; first_i <= -1;
            first_got <= '0; first_exp <= '0;
        end else if (y_valid) begin
            int idx;
            idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (idx < T_POOL2) begin
                for (int i = 0; i < F_SEP_OUT; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[idx * F_SEP_OUT + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 4)
                            $display("[branchA_tb] mismatch t=%0d i=%0d got=%h exp=%h",
                                     idx, i, y_out[i], exp_v);
                        if (mismatches == 0) begin
                            first_t <= idx; first_i <= i;
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

        // Wait for done with generous slack: ingest + conv_drive + post + ELU/pool drains
        repeat (200000) begin
            @(posedge clk);
            if (done) break;
        end

        if (!done) $fatal(1, "[branchA_tb] timeout waiting for done");
        $display("[branchA_tb] outputs received: %0d (expected %0d)", outputs_received, T_POOL2);
        $display("[branchA_tb] mismatches      : %0d", mismatches);
        if (outputs_received != T_POOL2) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[branchA_tb] first mismatch t=%0d i=%0d got=%h exp=%h",
                     first_t, first_i, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[branchA_tb] PASS");
        $finish;
    end

endmodule
