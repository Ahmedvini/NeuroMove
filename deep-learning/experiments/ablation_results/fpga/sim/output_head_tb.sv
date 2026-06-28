// =============================================================================
// output_head_tb.sv  —  Sanity TB for the average+argmax output head.
// Synthetic test (no .h5 refs needed): feeds 5 known logit pairs and checks
// the argmax of their sum.
// =============================================================================

`timescale 1ns / 1ps

module output_head_tb;

    localparam int DATA_WIDTH = 16;
    localparam int N_CLS      = 2;
    localparam int N_WIN      = 5;
    localparam int CLK_PERIOD = 10;

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic signed [DATA_WIDTH-1:0] logits_in [0:N_CLS-1];
    logic                         in_valid;
    logic                         in_last;
    logic [$clog2(N_CLS)-1:0]     class_out;
    logic                         done;

    output_head #(
        .DATA_WIDTH(DATA_WIDTH),
        .N_CLS     (N_CLS),
        .N_WIN     (N_WIN)
    ) dut (
        .clk(clk), .rst(rst),
        .logits_in(logits_in), .in_valid(in_valid), .in_last(in_last),
        .class_out(class_out), .done(done)
    );

    // Case A: per-window logits sum to (+5, -5). argmax = 0.
    int test_a[5][2] = '{
        '{ 100,  -50},
        '{ -10,  -100},
        '{ 0,    50},
        '{ -50,  -100},
        '{ -35,  -100}
    };

    initial begin
        rst = 1'b1;
        in_valid = 1'b0;
        in_last  = 1'b0;
        for (int o = 0; o < N_CLS; o++) logits_in[o] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int w = 0; w < N_WIN; w++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_last  <= (w == N_WIN - 1);
            for (int o = 0; o < N_CLS; o++) logits_in[o] <= DATA_WIDTH'(test_a[w][o]);
        end
        @(posedge clk);
        in_valid <= 1'b0;
        in_last  <= 1'b0;
        for (int o = 0; o < N_CLS; o++) logits_in[o] <= '0;

        wait (done == 1'b1);
        @(posedge clk);

        // Sum: (100 + -10 + 0 + -50 + -35, -50 + -100 + 50 + -100 + -100)
        //     = (5, -300). argmax = 0.
        if (class_out !== 0) $fatal(1, "[head_tb] FAIL: expected class=0 got=%0d", class_out);
        $display("[head_tb] case A PASS: class=%0d", class_out);
        $finish;
    end

endmodule
