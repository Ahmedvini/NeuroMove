// =============================================================================
// cbam_channel_attn_tb.sv  —  Bit-exact regression for ImpCBAM channel
// attention on sliding-window 0 (HaLT DB-ATCNet).
//
// Setup:  python3 scripts/q88_layer17_chan.py --window 0
// =============================================================================

`timescale 1ns / 1ps

module cbam_channel_attn_tb;

    localparam int T_WIN      = 6;
    localparam int NUM_CH     = 32;
    localparam int D_RED      = 4;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam int INV_N      = 2796203;     // round(2^24 / 6)

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_WIN * NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_WIN * NUM_CH - 1];

    initial begin
        $readmemh("data/golden_q88/stage_w0_chan_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_w0_chan_output.hex", exp_mem);
    end

    logic clk;
    logic rst;
    initial begin clk = 1'b0; forever #(CLK_PERIOD/2) clk = ~clk; end

    logic signed [DATA_WIDTH-1:0] x_in       [0:NUM_CH-1];
    logic                         x_valid;
    logic                         frame_last;
    logic                         window_idx;
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
        .INV_N       (INV_N),
        .INV_N_SHIFT (24),
        .INV_N_WIDTH (24),
        .LUT_N       (1024),
        .LUT_RANGE_Q (1024),
        .LUT_SHIFT   (1),
        .D1_W_FILE   ("data/golden_q88/stage_w0_chan_d1w.hex"),
        .D1_B_FILE   ("data/golden_q88/stage_w0_chan_d1b.hex"),
        .D2_W_FILE   ("data/golden_q88/stage_w0_chan_d2w.hex"),
        .D2_B_FILE   ("data/golden_q88/stage_w0_chan_d2b.hex"),
        .LUT_FILE    ("data/lut/sigmoid_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .window_idx(window_idx),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
        .gate_out(gate_out), .gate_valid(gate_valid),
        .apply_x_in(apply_x_in), .apply_valid(apply_valid),
        .y_out(y_out), .y_valid(y_valid)
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
                for (int i = 0; i < NUM_CH; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * NUM_CH + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 8)
                            $display("[chan_tb] mismatch t=%0d i=%0d got=%h exp=%h",
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
        x_valid = 1'b0; frame_last = 1'b0; apply_valid = 1'b0;
        window_idx = 1'b0;
        for (int i = 0; i < NUM_CH; i++) begin x_in[i] = '0; apply_x_in[i] = '0; end
        repeat (4) @(posedge clk);
        rst = 1'b0;

        // --- Phase 1: stream T_WIN vectors to accumulate avg/max ---
        for (int t = 0; t < T_WIN; t++) begin
            @(posedge clk);
            x_valid    <= 1'b1;
            frame_last <= (t == T_WIN - 1);
            for (int i = 0; i < NUM_CH; i++) x_in[i] <= in_mem[t * NUM_CH + i];
        end
        @(posedge clk);
        x_valid    <= 1'b0;
        frame_last <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) x_in[i] <= '0;

        wait (gate_valid == 1'b1);
        @(posedge clk);

        // --- Phase 3: re-stream T_WIN vectors through gate apply ---
        for (int t = 0; t < T_WIN; t++) begin
            @(posedge clk);
            apply_valid <= 1'b1;
            for (int i = 0; i < NUM_CH; i++) apply_x_in[i] <= in_mem[t * NUM_CH + i];
        end
        @(posedge clk);
        apply_valid <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) apply_x_in[i] <= '0;
        repeat (4) @(posedge clk);

        $display("[chan_tb] outputs received: %0d (expected %0d)", outputs_received, T_WIN);
        $display("[chan_tb] mismatches      : %0d", mismatches);
        if (outputs_received != T_WIN) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[chan_tb] first mismatch t=%0d i=%0d got=%h exp=%h",
                     first_t, first_i, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[chan_tb] PASS");
        $finish;
    end

endmodule
