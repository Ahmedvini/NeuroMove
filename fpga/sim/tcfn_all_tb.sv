// =============================================================================
// tcfn_all_tb.sv  —  Bit-exact regression for TCFN across all 5 sliding
// windows (N_WIN=5).
// =============================================================================

`timescale 1ns / 1ps

module tcfn_all_tb;

    localparam int T_WIN      = 6;
    localparam int F          = 32;
    localparam int N_WIN      = 5;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem [0:N_WIN-1][0:T_WIN * F - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem[0:N_WIN-1][0:T_WIN * F - 1];

    initial begin
        $readmemh("data/golden_q88/stage_w0_tcfn_input.hex",  in_mem[0]);
        $readmemh("data/golden_q88/stage_w1_tcfn_input.hex",  in_mem[1]);
        $readmemh("data/golden_q88/stage_w2_tcfn_input.hex",  in_mem[2]);
        $readmemh("data/golden_q88/stage_w3_tcfn_input.hex",  in_mem[3]);
        $readmemh("data/golden_q88/stage_w4_tcfn_input.hex",  in_mem[4]);
        $readmemh("data/golden_q88/stage_w0_tcfn_output.hex", exp_mem[0]);
        $readmemh("data/golden_q88/stage_w1_tcfn_output.hex", exp_mem[1]);
        $readmemh("data/golden_q88/stage_w2_tcfn_output.hex", exp_mem[2]);
        $readmemh("data/golden_q88/stage_w3_tcfn_output.hex", exp_mem[3]);
        $readmemh("data/golden_q88/stage_w4_tcfn_output.hex", exp_mem[4]);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic [$clog2(N_WIN)-1:0]     window_idx;
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
        .N_WIN      (N_WIN),
        .W0_FILE("data/golden_q88/all_tcfn_w0.hex"),
        .B0_FILE("data/golden_q88/all_tcfn_b0.hex"),
        .W1_FILE("data/golden_q88/all_tcfn_w1.hex"),
        .B1_FILE("data/golden_q88/all_tcfn_b1.hex"),
        .W2_FILE("data/golden_q88/all_tcfn_w2.hex"),
        .B2_FILE("data/golden_q88/all_tcfn_b2.hex"),
        .W3_FILE("data/golden_q88/all_tcfn_w3.hex"),
        .B3_FILE("data/golden_q88/all_tcfn_b3.hex"),
        .LUT_FILE("data/lut/elu_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .y_out(y_out), .y_valid(y_valid), .done(done)
    );

    int mismatches = 0;
    int total_outputs = 0;
    logic [$clog2(T_WIN+1)-1:0] check_t;

    always_ff @(posedge clk) begin
        if (rst) check_t <= 0;
        else if (y_valid) begin
            for (int i = 0; i < F; i++) begin
                if (y_out[i] !== exp_mem[window_idx][check_t * F + i]) begin
                    if (mismatches < 4)
                        $display("[tcfn_all_tb] mismatch w=%0d t=%0d i=%0d got=%h exp=%h",
                                 window_idx, check_t, i, y_out[i],
                                 exp_mem[window_idx][check_t * F + i]);
                    mismatches <= mismatches + 1;
                end
            end
            total_outputs <= total_outputs + 1;
            if (check_t == T_WIN - 1) check_t <= 0;
            else                      check_t <= check_t + 1;
        end
    end

    task automatic run_window(input int w);
        for (int t = 0; t < T_WIN; t++) begin
            @(posedge clk);
            x_valid    <= 1'b1;
            frame_last <= (t == T_WIN - 1);
            for (int i = 0; i < F; i++) x_in[i] <= in_mem[w][t * F + i];
        end
        @(posedge clk);
        x_valid    <= 1'b0;
        frame_last <= 1'b0;
        for (int i = 0; i < F; i++) x_in[i] <= '0;
        wait (done == 1'b1);
        @(posedge clk);
    endtask

    initial begin
        rst = 1'b1;
        window_idx = 0;
        x_valid = 0; frame_last = 0;
        for (int i = 0; i < F; i++) x_in[i] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int w = 0; w < N_WIN; w++) begin
            window_idx <= w[$clog2(N_WIN)-1:0];
            run_window(w);
            $display("[tcfn_all_tb] window %0d done", w);
        end

        repeat (4) @(posedge clk);
        $display("[tcfn_all_tb] total outputs: %0d (expected %0d)", total_outputs, N_WIN * T_WIN);
        $display("[tcfn_all_tb] mismatches   : %0d", mismatches);
        if (mismatches != 0) $fatal(1, "FAIL");
        if (total_outputs != N_WIN * T_WIN) $fatal(1, "output count off");
        $display("[tcfn_all_tb] PASS");
        $finish;
    end

endmodule
