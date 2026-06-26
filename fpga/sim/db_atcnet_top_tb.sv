// =============================================================================
// db_atcnet_top_tb.sv  —  End-to-end RTL inference test for HaLT DB-ATCNet.
//
// Drives raw EEG samples (stage_conv2d_input.hex, 600×5 = 3000 Q8.8 values)
// into db_atcnet_top, waits for done, and checks class_out == 0 (Right Hand
// fist), the answer the full Python golden produced for Subject A session 3.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_top_tb;

    localparam int C_IN       = 5;
    localparam int F1         = 16;
    localparam int KE         = 64;
    localparam int T_IN       = 600;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam int PAD_TOP    = (KE - 1) / 2;
    localparam int PAD_BOT    = KE - 1 - PAD_TOP;
    localparam int T_TOTAL    = PAD_TOP + T_IN + PAD_BOT;

    logic signed [DATA_WIDTH-1:0] in_mem [0:T_IN * C_IN - 1];

    initial $readmemh("data/golden_q88/stage_conv2d_input.hex", in_mem);

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                                    in_valid;
    logic [$clog2(C_IN > 1 ? C_IN : 2)-1:0]  in_electrode;
    logic signed [DATA_WIDTH-1:0]            in_sample;
    logic [0:0]                              class_out;   // $clog2(2) = 1
    logic                                    done;

    db_atcnet_top #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(C_IN), .F1(F1), .F(32),
        .KE_CONV2D(KE), .KE_SEP(16),
        .T_IN(T_IN), .POOL1(8), .POOL2(7), .T_FULL(10), .T_WIN(6),
        .N_WIN(5), .N_CLS(2),
        .ECA1_INV_N(5592), .ECA2_INV_N(1677722), .CHAN_INV_N(2796203),
        .INV_POOL1(2097152), .INV_POOL2(2396745),
        .SIG_LUT_N(1024), .SIG_LUT_RANGE_Q(1024), .SIG_LUT_SHIFT(1),
        .ELU_LUT_N(256), .ELU_LUT_RANGE_Q(2048), .ELU_LUT_SHIFT(3),
        .CONV2D_W_FILE("data/golden_q88/stage_conv2d_weights.hex"),
        .CONV2D_B_FILE("data/golden_q88/stage_conv2d_bias.hex"),
        .ECA1_W_FILE  ("data/golden_q88/stage_eca1_weights.hex"),
        .SIG_LUT_FILE ("data/lut/sigmoid_q88.hex"),
        .ELU_LUT_FILE ("data/lut/elu_q88.hex"),
        .BRANCHA_DWISE_W_FILE("data/golden_q88/stage_branchA_dwise_weights.hex"),
        .BRANCHA_DWISE_B_FILE("data/golden_q88/stage_branchA_dwise_bias.hex"),
        .BRANCHA_SEP_W_FILE  ("data/golden_q88/stage_branchA_sep_weights.hex"),
        .BRANCHA_SEP_B_FILE  ("data/golden_q88/stage_branchA_sep_bias.hex"),
        .BRANCHB_DWISE_W_FILE("data/golden_q88/stage_branchB_dwise_weights.hex"),
        .BRANCHB_DWISE_B_FILE("data/golden_q88/stage_branchB_dwise_bias.hex"),
        .BRANCHB_SEP_W_FILE  ("data/golden_q88/stage_branchB_sep_weights.hex"),
        .BRANCHB_SEP_B_FILE  ("data/golden_q88/stage_branchB_sep_bias.hex"),
        .ECA2_W_FILE         ("data/golden_q88/stage_eca2_weights.hex"),
        .CHAN_D1_W_FILE("data/golden_q88/all_chan_d1w.hex"),
        .CHAN_D1_B_FILE("data/golden_q88/all_chan_d1b.hex"),
        .CHAN_D2_W_FILE("data/golden_q88/all_chan_d2w.hex"),
        .CHAN_D2_B_FILE("data/golden_q88/all_chan_d2b.hex"),
        .SPAT_CONV_W_FILE("data/golden_q88/all_spat_conv_w.hex"),
        .TCFN_W0_FILE("data/golden_q88/all_tcfn_w0.hex"),
        .TCFN_B0_FILE("data/golden_q88/all_tcfn_b0.hex"),
        .TCFN_W1_FILE("data/golden_q88/all_tcfn_w1.hex"),
        .TCFN_B1_FILE("data/golden_q88/all_tcfn_b1.hex"),
        .TCFN_W2_FILE("data/golden_q88/all_tcfn_w2.hex"),
        .TCFN_B2_FILE("data/golden_q88/all_tcfn_b2.hex"),
        .TCFN_W3_FILE("data/golden_q88/all_tcfn_w3.hex"),
        .TCFN_B3_FILE("data/golden_q88/all_tcfn_b3.hex"),
        .CLS_W_FILE("data/golden_q88/all_cls_w.hex"),
        .CLS_B_FILE("data/golden_q88/all_cls_b.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_electrode(in_electrode), .in_sample(in_sample),
        .class_out(class_out), .done(done)
    );

    function automatic logic signed [DATA_WIDTH-1:0] raw_sample_at(int t_in, int e);
        int t_real;
        if (t_in < PAD_TOP || t_in >= PAD_TOP + T_IN) return '0;
        t_real = t_in - PAD_TOP;
        return in_mem[t_real * C_IN + e];
    endfunction

    initial begin
        rst = 1'b1;
        in_valid = 1'b0;
        in_electrode = '0;
        in_sample = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

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

        // Inference budget: ~200k cycles (Branch B's conv1d_tm dominates).
        // Use 400k for headroom.
        repeat (400000) begin
            @(posedge clk);
            if (done) break;
        end
        if (!done) $fatal(1, "[top_tb] timeout waiting for done");

        $display("[top_tb] class_out = %0d (expected 0)", class_out);
        if (class_out !== 1'b0) $fatal(1, "[top_tb] FAIL class mismatch");
        $display("[top_tb] PASS");
        $finish;
    end

endmodule
