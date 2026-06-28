// =============================================================================
// tcfn_tb.sv  —  Bit-exact regression for TCFN (window 0) of HaLT DB-ATCNet.
// Setup:  python3 scripts/q88_layer_tcfn.py --window 0
// =============================================================================

`timescale 1ns / 1ps

module tcfn_tb;

    localparam int T_WIN      = 6;
    localparam int F          = 32;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_WIN * F - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_WIN * F - 1];

    initial begin
        $readmemh("data/golden_q88/stage_w0_tcfn_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_w0_tcfn_output.hex", exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                         window_idx;
    logic signed [DATA_WIDTH-1:0] x_in       [0:F-1];
    logic                         x_valid;
    logic                         frame_last;
    logic signed [DATA_WIDTH-1:0] y_out      [0:F-1];
    logic                         y_valid;
    logic                         done;

    tcfn #(
        .DATA_WIDTH (DATA_WIDTH),
        .COEF_WIDTH (16),
        .ACC_WIDTH  (48),
        .FRAC_BITS  (8),
        .T_WIN      (T_WIN),
        .F          (F),
        .K          (4),
        .LUT_N      (256),
        .LUT_RANGE_Q(2048),
        .LUT_SHIFT  (3),
        .W0_FILE("data/golden_q88/stage_w0_tcfn_w0.hex"),
        .B0_FILE("data/golden_q88/stage_w0_tcfn_b0.hex"),
        .W1_FILE("data/golden_q88/stage_w0_tcfn_w1.hex"),
        .B1_FILE("data/golden_q88/stage_w0_tcfn_b1.hex"),
        .W2_FILE("data/golden_q88/stage_w0_tcfn_w2.hex"),
        .B2_FILE("data/golden_q88/stage_w0_tcfn_b2.hex"),
        .W3_FILE("data/golden_q88/stage_w0_tcfn_w3.hex"),
        .B3_FILE("data/golden_q88/stage_w0_tcfn_b3.hex"),
        .LUT_FILE("data/lut/elu_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
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
            int vec_idx;
            vec_idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (vec_idx < T_WIN) begin
                for (int i = 0; i < F; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * F + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 4)
                            $display("[tcfn_tb] mismatch t=%0d i=%0d got=%h exp=%h",
                                     vec_idx, i, y_out[i], exp_v);
                        if (mismatches == 0) begin
                            first_t <= vec_idx; first_i <= i;
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
        window_idx = 1'b0;
        for (int i = 0; i < F; i++) x_in[i] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < T_WIN; t++) begin
            @(posedge clk);
            x_valid    <= 1'b1;
            frame_last <= (t == T_WIN - 1);
            for (int i = 0; i < F; i++) x_in[i] <= in_mem[t * F + i];
        end
        @(posedge clk);
        x_valid    <= 1'b0;
        frame_last <= 1'b0;
        for (int i = 0; i < F; i++) x_in[i] <= '0;

        wait (done == 1'b1);
        repeat (4) @(posedge clk);

        $display("[tcfn_tb] outputs received: %0d (expected %0d)", outputs_received, T_WIN);
        $display("[tcfn_tb] mismatches      : %0d", mismatches);
        if (outputs_received != T_WIN) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[tcfn_tb] first mismatch t=%0d i=%0d got=%h exp=%h",
                     first_t, first_i, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[tcfn_tb] PASS");
        $finish;
    end

endmodule
