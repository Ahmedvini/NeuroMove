// =============================================================================
// cbam_channel_attn_all_tb.sv  —  Bit-exact regression for ImpCBAM channel
// attention across ALL 5 sliding windows. Loads the combined multi-bank
// weight files (all_chan_*.hex) and drives `window_idx` from 0..4, comparing
// each window's (6, 32) output to its Python golden.
//
// Setup:  python3 scripts/q88_layer17_chan.py --window {0,1,2,3,4}
//         python3 scripts/cat_window_weights.py
// =============================================================================

`timescale 1ns / 1ps

module cbam_channel_attn_all_tb;

    localparam int T_WIN      = 6;
    localparam int NUM_CH     = 32;
    localparam int D_RED      = 4;
    localparam int N_WIN      = 5;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam int INV_N      = 2796203;

    logic signed [DATA_WIDTH-1:0] in_mem [0:N_WIN-1][0:T_WIN * NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem[0:N_WIN-1][0:T_WIN * NUM_CH - 1];

    initial begin
        $readmemh("data/golden_q88/stage_w0_chan_input.hex",  in_mem[0]);
        $readmemh("data/golden_q88/stage_w1_chan_input.hex",  in_mem[1]);
        $readmemh("data/golden_q88/stage_w2_chan_input.hex",  in_mem[2]);
        $readmemh("data/golden_q88/stage_w3_chan_input.hex",  in_mem[3]);
        $readmemh("data/golden_q88/stage_w4_chan_input.hex",  in_mem[4]);
        $readmemh("data/golden_q88/stage_w0_chan_output.hex", exp_mem[0]);
        $readmemh("data/golden_q88/stage_w1_chan_output.hex", exp_mem[1]);
        $readmemh("data/golden_q88/stage_w2_chan_output.hex", exp_mem[2]);
        $readmemh("data/golden_q88/stage_w3_chan_output.hex", exp_mem[3]);
        $readmemh("data/golden_q88/stage_w4_chan_output.hex", exp_mem[4]);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic [$clog2(N_WIN)-1:0]     window_idx;
    logic signed [DATA_WIDTH-1:0] x_in       [0:NUM_CH-1];
    logic                         x_valid;
    logic                         frame_last;
    logic signed [DATA_WIDTH-1:0] gate_out   [0:NUM_CH-1];
    logic                         gate_valid;
    logic signed [DATA_WIDTH-1:0] apply_x_in [0:NUM_CH-1];
    logic                         apply_valid;
    logic signed [DATA_WIDTH-1:0] y_out      [0:NUM_CH-1];
    logic                         y_valid;

    cbam_channel_attn #(
        .DATA_WIDTH  (DATA_WIDTH),
        .COEF_WIDTH  (16),
        .ACC_WIDTH   (32),
        .FRAC_BITS   (8),
        .NUM_CH      (NUM_CH),
        .T_WIN       (T_WIN),
        .D_RED       (D_RED),
        .N_WIN       (N_WIN),
        .INV_N       (INV_N),
        .INV_N_SHIFT (24),
        .INV_N_WIDTH (24),
        .LUT_N       (1024),
        .LUT_RANGE_Q (1024),
        .LUT_SHIFT   (1),
        .D1_W_FILE   ("data/golden_q88/all_chan_d1w.hex"),
        .D1_B_FILE   ("data/golden_q88/all_chan_d1b.hex"),
        .D2_W_FILE   ("data/golden_q88/all_chan_d2w.hex"),
        .D2_B_FILE   ("data/golden_q88/all_chan_d2b.hex"),
        .LUT_FILE    ("data/lut/sigmoid_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .gate_out(gate_out), .gate_valid(gate_valid),
        .apply_x_in(apply_x_in), .apply_valid(apply_valid),
        .y_out(y_out), .y_valid(y_valid)
    );

    int mismatches = 0;
    int total_outputs = 0;

    task automatic run_window(input int w);
        int got_count;
        int local_mm;
        got_count = 0;
        local_mm  = 0;

        // Phase 1: stream T_WIN inputs.
        for (int t = 0; t < T_WIN; t++) begin
            @(posedge clk);
            x_valid    <= 1'b1;
            frame_last <= (t == T_WIN - 1);
            for (int i = 0; i < NUM_CH; i++) x_in[i] <= in_mem[w][t * NUM_CH + i];
        end
        @(posedge clk);
        x_valid    <= 1'b0;
        frame_last <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) x_in[i] <= '0;

        wait (gate_valid == 1'b1);
        @(posedge clk);

        // Phase 2: stream apply inputs.
        for (int t = 0; t < T_WIN; t++) begin
            @(posedge clk);
            apply_valid <= 1'b1;
            for (int i = 0; i < NUM_CH; i++) apply_x_in[i] <= in_mem[w][t * NUM_CH + i];
        end
        @(posedge clk);
        apply_valid <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) apply_x_in[i] <= '0;
        repeat (2) @(posedge clk);

        // Outputs were sampled by the checker thread; report via globals.
    endtask

    // Checker: every cycle when y_valid is high, compare against the current
    // expected slice indexed by `window_idx` and an internal counter that
    // resets when window_idx changes.
    logic [$clog2(T_WIN+1)-1:0] check_t;
    logic [$clog2(N_WIN)-1:0]   check_wi;

    always_ff @(posedge clk) begin
        if (rst) begin
            check_t  <= 0;
            check_wi <= 0;
        end else if (y_valid) begin
            for (int i = 0; i < NUM_CH; i++) begin
                if (y_out[i] !== exp_mem[window_idx][check_t * NUM_CH + i]) begin
                    if (mismatches < 4)
                        $display("[chan_all_tb] mismatch w=%0d t=%0d i=%0d got=%h exp=%h",
                                 window_idx, check_t, i, y_out[i],
                                 exp_mem[window_idx][check_t * NUM_CH + i]);
                    mismatches <= mismatches + 1;
                end
            end
            total_outputs <= total_outputs + 1;
            if (check_t == T_WIN - 1) check_t <= 0;
            else                      check_t <= check_t + 1;
        end
    end

    initial begin
        rst = 1'b1;
        window_idx = 0;
        x_valid = 0; frame_last = 0; apply_valid = 0;
        for (int i = 0; i < NUM_CH; i++) begin x_in[i] = '0; apply_x_in[i] = '0; end
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int w = 0; w < N_WIN; w++) begin
            window_idx <= w[$clog2(N_WIN)-1:0];
            run_window(w);
            $display("[chan_all_tb] window %0d done", w);
        end

        repeat (4) @(posedge clk);
        $display("[chan_all_tb] total outputs: %0d (expected %0d)", total_outputs, N_WIN * T_WIN);
        $display("[chan_all_tb] mismatches   : %0d", mismatches);
        if (mismatches != 0) $fatal(1, "FAIL");
        if (total_outputs != N_WIN * T_WIN) $fatal(1, "output count off");
        $display("[chan_all_tb] PASS");
        $finish;
    end

endmodule
