// =============================================================================
// avg_pool_time_tb.sv  —  Bit-exact regression for the temporal AvgPool block
//
// Streams the Branch A ELU output (T=600, F2=32) into avg_pool_time.sv with
// POOL=8 and verifies the 75 emitted vectors against
// stage_branchA_pool1_output.hex.
// =============================================================================

`timescale 1ns / 1ps

module avg_pool_time_tb;

    localparam int T          = 600;
    localparam int NUM_CH     = 32;
    localparam int POOL       = 8;
    localparam int T_POOL     = T / POOL;       // 75
    localparam int INV_POOL   = 2097152;        // round(2^24 / 8)
    localparam int POOL_SHIFT = 24;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T*NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_POOL*NUM_CH - 1];

    initial begin
        $readmemh("data/golden_q88/stage_branchA_pool1_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_branchA_pool1_output.hex", exp_mem);
    end

    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] x_in  [0:NUM_CH-1];
    logic                         x_valid;
    logic signed [DATA_WIDTH-1:0] y_out [0:NUM_CH-1];
    logic                         y_valid;

    avg_pool_time #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_CH     (NUM_CH),
        .POOL       (POOL),
        .INV_POOL   (INV_POOL),
        .POOL_SHIFT (POOL_SHIFT)
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in),
        .x_valid(x_valid),
        .y_out(y_out),
        .y_valid(y_valid)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    int outputs_received;
    int mismatches;
    int first_mismatch_t, first_mismatch_c;
    logic signed [DATA_WIDTH-1:0] first_mismatch_got;
    logic signed [DATA_WIDTH-1:0] first_mismatch_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received   <= 0;
            mismatches         <= 0;
            first_mismatch_t   <= -1;
            first_mismatch_c   <= -1;
            first_mismatch_got <= '0;
            first_mismatch_exp <= '0;
        end else if (y_valid) begin
            int vec_idx;
            vec_idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (vec_idx < T_POOL) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * NUM_CH + c];
                    if (y_out[c] !== exp_v) begin
                        if (mismatches < 8)
                            $display("[pool_tb] mismatch t=%0d c=%0d got=%h exp=%h",
                                     vec_idx, c, y_out[c], exp_v);
                        if (mismatches == 0) begin
                            first_mismatch_t   <= vec_idx;
                            first_mismatch_c   <= c;
                            first_mismatch_got <= y_out[c];
                            first_mismatch_exp <= exp_v;
                        end
                        mismatches <= mismatches + 1;
                    end
                end
            end
        end
    end

    int n_drive;
    initial begin
        rst = 1'b1;
        x_valid = 1'b0;
        for (int c = 0; c < NUM_CH; c++) x_in[c] = '0;
        n_drive = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < T; t++) begin
            @(posedge clk);
            x_valid <= 1'b1;
            for (int c = 0; c < NUM_CH; c++)
                x_in[c] <= in_mem[t * NUM_CH + c];
            n_drive++;
        end
        @(posedge clk);
        x_valid <= 1'b0;
        for (int c = 0; c < NUM_CH; c++) x_in[c] <= '0;

        repeat (4) @(posedge clk);

        $display("[pool_tb] vectors driven  : %0d (expected %0d)", n_drive, T);
        $display("[pool_tb] outputs received: %0d (expected %0d)", outputs_received, T_POOL);
        $display("[pool_tb] mismatches      : %0d", mismatches);
        if (outputs_received != T_POOL)  $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[pool_tb] first mismatch t=%0d c=%0d got=%h exp=%h",
                     first_mismatch_t, first_mismatch_c,
                     first_mismatch_got, first_mismatch_exp);
            $fatal(1, "FAIL");
        end
        $display("[pool_tb] PASS");
        $finish;
    end

endmodule
