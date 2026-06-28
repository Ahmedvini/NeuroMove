// =============================================================================
// dense_classifier_tb.sv  —  Bit-exact regression for Dense(32→2) classifier
// using window 0's last-step TCFN output as input.
//
// Setup:  python3 scripts/q88_classifier.py --window 0
// =============================================================================

`timescale 1ns / 1ps

module dense_classifier_tb;

    localparam int F     = 32;
    localparam int N_CLS = 2;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:F-1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:N_CLS-1];

    initial begin
        $readmemh("data/golden_q88/stage_w0_cls_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_w0_cls_logits.hex", exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                         window_idx;
    logic signed [DATA_WIDTH-1:0] x_in   [0:F-1];
    logic                         in_valid;
    logic signed [DATA_WIDTH-1:0] logits [0:N_CLS-1];
    logic                         out_valid;

    dense_classifier #(
        .DATA_WIDTH(DATA_WIDTH),
        .COEF_WIDTH(16),
        .ACC_WIDTH (40),
        .FRAC_BITS (8),
        .F         (F),
        .N_CLS     (N_CLS),
        .WEIGHTS_FILE("data/golden_q88/stage_w0_cls_w.hex"),
        .BIAS_FILE   ("data/golden_q88/stage_w0_cls_b.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(x_in), .in_valid(in_valid),
        .logits(logits), .out_valid(out_valid)
    );

    int mismatches = 0;

    initial begin
        rst = 1'b1;
        in_valid = 1'b0;
        window_idx = 1'b0;
        for (int i = 0; i < F; i++) x_in[i] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        @(posedge clk);
        in_valid <= 1'b1;
        for (int i = 0; i < F; i++) x_in[i] <= in_mem[i];
        @(posedge clk);
        in_valid <= 1'b0;
        for (int i = 0; i < F; i++) x_in[i] <= '0;
        wait (out_valid == 1'b1);
        @(posedge clk);

        for (int o = 0; o < N_CLS; o++) begin
            if (logits[o] !== exp_mem[o]) begin
                $display("[dense_tb] mismatch o=%0d got=%h exp=%h",
                         o, logits[o], exp_mem[o]);
                mismatches++;
            end
        end
        $display("[dense_tb] logits = %0d %0d (expected %0d %0d)",
                 $signed(logits[0]), $signed(logits[1]),
                 $signed(exp_mem[0]), $signed(exp_mem[1]));
        if (mismatches != 0) $fatal(1, "FAIL");
        $display("[dense_tb] PASS");
        $finish;
    end

endmodule
