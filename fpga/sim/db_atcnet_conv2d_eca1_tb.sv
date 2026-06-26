// =============================================================================
// db_atcnet_conv2d_eca1_tb.sv  —  Bit-exact regression for the Conv2D + ECA₁
// wrapper. Drives raw EEG (stage_conv2d_input.hex) and checks the resulting
// 3000 ECA₁-gated vectors against stage_eca1_output.hex.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_conv2d_eca1_tb;

    localparam int C_IN       = 5;
    localparam int F1         = 16;
    localparam int KE         = 64;
    localparam int T_IN       = 600;
    localparam int N_FRAMES   = T_IN * C_IN;     // 3000
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam int PAD_TOP    = (KE - 1) / 2;    // 31
    localparam int PAD_BOT    = KE - 1 - PAD_TOP;// 32
    localparam int T_TOTAL    = PAD_TOP + T_IN + PAD_BOT;  // 663

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_IN * C_IN - 1];          // 3000 raw samples
    logic signed [DATA_WIDTH-1:0] exp_mem [0:N_FRAMES * F1 - 1];        // 48000 expected

    initial begin
        $readmemh("data/golden_q88/stage_conv2d_input.hex", in_mem);
        $readmemh("data/golden_q88/stage_eca1_output.hex",  exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                                    in_valid;
    logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0]  in_electrode;
    logic signed [DATA_WIDTH-1:0]            in_sample;
    logic signed [DATA_WIDTH-1:0]            y_out [0:F1-1];
    logic                                    y_valid;
    logic                                    frame_last;
    logic                                    done;

    db_atcnet_conv2d_eca1 #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .KE(KE), .T_IN(T_IN),
        .ECA1_KECA(3), .ECA1_INV_N(5592),
        .SIG_LUT_N(1024), .SIG_LUT_RANGE_Q(1024), .SIG_LUT_SHIFT(1),
        .CONV2D_W_FILE("data/golden_q88/stage_conv2d_weights.hex"),
        .CONV2D_B_FILE("data/golden_q88/stage_conv2d_bias.hex"),
        .ECA1_W_FILE  ("data/golden_q88/stage_eca1_weights.hex"),
        .SIG_LUT_FILE ("data/lut/sigmoid_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_electrode(in_electrode), .in_sample(in_sample),
        .y_out(y_out), .y_valid(y_valid), .frame_last(frame_last), .done(done)
    );

    function automatic logic signed [DATA_WIDTH-1:0] raw_sample_at(int t_in, int e);
        int t_real;
        if (t_in < PAD_TOP || t_in >= PAD_TOP + T_IN) return '0;
        t_real = t_in - PAD_TOP;
        return in_mem[t_real * C_IN + e];
    endfunction

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
                            $display("[c2e1_tb] mismatch idx=%0d ch=%0d got=%h exp=%h",
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
        in_valid = 1'b0; in_electrode = '0; in_sample = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        // Drive T_TOTAL × C_IN raw-sample cycles (pad_top + T_IN + pad_bot)
        // × electrodes 0..C_IN-1.
        for (int t = 0; t < T_TOTAL; t++) begin
            for (int e = 0; e < C_IN; e++) begin
                @(posedge clk);
                in_valid     <= 1'b1;
                in_electrode <= e[$clog2(C_IN > 1 ? C_IN : 2)-1:0];
                in_sample    <= raw_sample_at(t, e);
            end
        end
        @(posedge clk);
        in_valid <= 1'b0;
        in_sample <= '0;

        // Wait for done. Total cycles: T_TOTAL*C_IN ingest + ECA₁ pipeline
        // (~3000 replay cycles) — within 20k cycles total.
        repeat (50000) begin
            @(posedge clk);
            if (done) break;
        end
        if (!done) $fatal(1, "[c2e1_tb] timeout waiting for done");

        $display("[c2e1_tb] outputs received: %0d (expected %0d)", outputs_received, N_FRAMES);
        $display("[c2e1_tb] mismatches      : %0d", mismatches);
        if (outputs_received != N_FRAMES) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[c2e1_tb] first mismatch idx=%0d ch=%0d got=%h exp=%h",
                     first_idx, first_ch, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[c2e1_tb] PASS");
        $finish;
    end

endmodule
