// =============================================================================
// eca2_pipeline_tb.sv  —  Reuses rtl/attention/eca1_pipeline.sv as the
// general ECA orchestrator; here NUM_CH=32, N_FRAMES=10, INV_N=1677722 to
// match Layer 16 (ECA₂). Bit-exact against stage_eca2_output.hex.
// =============================================================================

`timescale 1ns / 1ps

module eca2_pipeline_tb;

    localparam int T          = 10;
    localparam int F          = 32;
    localparam int N_FRAMES   = T;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:N_FRAMES * F - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:N_FRAMES * F - 1];

    initial begin
        $readmemh("data/golden_q88/stage_eca2_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_eca2_output.hex", exp_mem);
    end

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic signed [DATA_WIDTH-1:0] x_in   [0:F-1];
    logic                         x_valid;
    logic signed [DATA_WIDTH-1:0] y_out  [0:F-1];
    logic                         y_valid;
    logic                         done;

    eca1_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_CH(F), .KECA(3),
        .N_FRAMES(N_FRAMES),
        .INV_N(1677722), .INV_N_SHIFT(24), .INV_N_WIDTH(24),
        .LUT_N(1024), .LUT_RANGE_Q(1024), .LUT_SHIFT(1),
        .WEIGHTS_FILE("data/golden_q88/stage_eca2_weights.hex"),
        .LUT_FILE    ("data/lut/sigmoid_q88.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid),
        .y_out(y_out), .y_valid(y_valid), .done(done)
    );

    int outputs_received;
    int mismatches;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received <= 0;
            mismatches       <= 0;
        end else if (y_valid) begin
            int idx;
            idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (idx < N_FRAMES) begin
                for (int i = 0; i < F; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[idx * F + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 4)
                            $display("[eca2_pipe_tb] mismatch idx=%0d ch=%0d got=%h exp=%h",
                                     idx, i, y_out[i], exp_v);
                        mismatches <= mismatches + 1;
                    end
                end
            end
        end
    end

    initial begin
        rst = 1'b1;
        x_valid = 1'b0;
        for (int i = 0; i < F; i++) x_in[i] = '0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < N_FRAMES; t++) begin
            @(posedge clk);
            x_valid <= 1'b1;
            for (int i = 0; i < F; i++) x_in[i] <= in_mem[t * F + i];
        end
        @(posedge clk);
        x_valid <= 1'b0;
        for (int i = 0; i < F; i++) x_in[i] <= '0;

        wait (done == 1'b1);
        repeat (4) @(posedge clk);

        $display("[eca2_pipe_tb] outputs received: %0d (expected %0d)", outputs_received, N_FRAMES);
        $display("[eca2_pipe_tb] mismatches      : %0d", mismatches);
        if (outputs_received != N_FRAMES) $fatal(1, "output count off");
        if (mismatches != 0) $fatal(1, "FAIL");
        $display("[eca2_pipe_tb] PASS");
        $finish;
    end

endmodule
