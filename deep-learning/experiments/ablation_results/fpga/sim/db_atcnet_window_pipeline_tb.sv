// =============================================================================
// db_atcnet_window_pipeline_tb.sv  —  End-to-end bit-exact regression for the
// post-ECA₂ pipeline of HaLT DB-ATCNet (5 sliding windows → final class).
//
// Setup:
//   python3 scripts/q88_layer17_chan.py    --window {0..4}
//   python3 scripts/q88_layer17_spatial.py --window {0..4}
//   python3 scripts/q88_layer_tcfn.py      --window {0..4}
//   python3 scripts/q88_classifier.py      --window {0..4}
//   python3 scripts/cat_window_weights.py
//   python3 scripts/q88_end_to_end.py     (for the expected class)
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_window_pipeline_tb;

    localparam int F          = 32;
    localparam int T_FULL     = 10;
    localparam int T_WIN      = 6;
    localparam int N_WIN      = 5;
    localparam int N_CLS      = 2;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] eca2_buf [0:T_FULL-1][0:F-1];
    logic signed [DATA_WIDTH-1:0] eca2_flat[0:T_FULL * F - 1];

    initial begin
        $readmemh("data/golden_q88/stage_eca2_output.hex", eca2_flat);
        for (int t = 0; t < T_FULL; t++)
            for (int c = 0; c < F; c++)
                eca2_buf[t][c] = eca2_flat[t * F + c];
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                          start;
    logic [$clog2(N_CLS)-1:0]      class_out;
    logic                          done;

    db_atcnet_window_pipeline #(
        .DATA_WIDTH(DATA_WIDTH),
        .F(F), .T_FULL(T_FULL), .T_WIN(T_WIN), .D_RED(4),
        .KH_SPAT(7), .IC_CAT(3), .K_TCFN(4), .N_CLS(N_CLS), .N_WIN(N_WIN),
        .SIG_LUT_N(1024), .SIG_LUT_RANGE_Q(1024), .SIG_LUT_SHIFT(1),
        .ELU_LUT_N(256), .ELU_LUT_RANGE_Q(2048), .ELU_LUT_SHIFT(3),
        .CHAN_INV_N(2796203),
        .CHAN_D1_W_FILE("data/golden_q88/all_chan_d1w.hex"),
        .CHAN_D1_B_FILE("data/golden_q88/all_chan_d1b.hex"),
        .CHAN_D2_W_FILE("data/golden_q88/all_chan_d2w.hex"),
        .CHAN_D2_B_FILE("data/golden_q88/all_chan_d2b.hex"),
        .SIG_LUT_FILE  ("data/lut/sigmoid_q88.hex"),
        .SPAT_CONV_W_FILE("data/golden_q88/all_spat_conv_w.hex"),
        .TCFN_W0_FILE("data/golden_q88/all_tcfn_w0.hex"),
        .TCFN_B0_FILE("data/golden_q88/all_tcfn_b0.hex"),
        .TCFN_W1_FILE("data/golden_q88/all_tcfn_w1.hex"),
        .TCFN_B1_FILE("data/golden_q88/all_tcfn_b1.hex"),
        .TCFN_W2_FILE("data/golden_q88/all_tcfn_w2.hex"),
        .TCFN_B2_FILE("data/golden_q88/all_tcfn_b2.hex"),
        .TCFN_W3_FILE("data/golden_q88/all_tcfn_w3.hex"),
        .TCFN_B3_FILE("data/golden_q88/all_tcfn_b3.hex"),
        .ELU_LUT_FILE("data/lut/elu_q88.hex"),
        .CLS_W_FILE("data/golden_q88/all_cls_w.hex"),
        .CLS_B_FILE("data/golden_q88/all_cls_b.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .eca2_buf(eca2_buf), .start(start),
        .class_out(class_out), .done(done)
    );

    initial begin
        rst = 1'b1;
        start = 1'b0;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        repeat (4) @(posedge clk);
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        // Upper bound: each window takes < 4000 cycles (TCFN is the dominant
        // cost at ~3072 cycles). 5 windows = ~16k cycles. Add slack.
        repeat (40000) begin
            @(posedge clk);
            if (done) break;
        end

        if (!done) $fatal(1, "[pipe_tb] timeout waiting for done");
        $display("[pipe_tb] class_out = %0d (expected 0)", class_out);
        if (class_out !== 1'b0) $fatal(1, "[pipe_tb] FAIL class mismatch");
        $display("[pipe_tb] PASS");
        $finish;
    end

endmodule
