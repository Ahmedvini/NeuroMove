// =============================================================================
// sat_add_tb.sv  —  Bit-exact regression for Layer 15 (Add(A,B)) of
// HaLT DB-ATCNet. Drives stage_add_input_{a,b}.hex; checks stage_add_output.hex.
//
// Setup:  python3 scripts/q88_layer15.py
// =============================================================================

`timescale 1ns / 1ps

module sat_add_tb;

    localparam int T_POOL2   = 10;
    localparam int NUM_CH    = 32;
    localparam int DATA_WIDTH= 16;
    localparam int CLK_PERIOD= 10;
    localparam int N_VEC     = T_POOL2;

    logic signed [DATA_WIDTH-1:0] a_mem   [0:N_VEC * NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] b_mem   [0:N_VEC * NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:N_VEC * NUM_CH - 1];

    initial begin
        $readmemh("data/golden_q88/stage_add_input_a.hex", a_mem);
        $readmemh("data/golden_q88/stage_add_input_b.hex", b_mem);
        $readmemh("data/golden_q88/stage_add_output.hex",  exp_mem);
    end

    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] a_in  [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] b_in  [0:NUM_CH-1];
    logic                         in_valid;
    logic signed [DATA_WIDTH-1:0] y_out [0:NUM_CH-1];
    logic                         y_valid;

    sat_add #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_CH    (NUM_CH)
    ) dut (
        .clk(clk), .rst(rst),
        .a_in(a_in), .b_in(b_in), .in_valid(in_valid),
        .y_out(y_out), .y_valid(y_valid)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

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
            if (vec_idx < N_VEC) begin
                for (int i = 0; i < NUM_CH; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * NUM_CH + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 8)
                            $display("[satAdd_tb] mismatch t=%0d i=%0d got=%h exp=%h",
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
        in_valid = 1'b0;
        for (int i = 0; i < NUM_CH; i++) begin a_in[i] = '0; b_in[i] = '0; end
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < N_VEC; t++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            for (int i = 0; i < NUM_CH; i++) begin
                a_in[i] <= a_mem[t * NUM_CH + i];
                b_in[i] <= b_mem[t * NUM_CH + i];
            end
        end
        @(posedge clk);
        in_valid <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) begin a_in[i] <= '0; b_in[i] <= '0; end
        repeat (4) @(posedge clk);

        $display("[satAdd_tb] outputs received: %0d (expected %0d)", outputs_received, N_VEC);
        $display("[satAdd_tb] mismatches      : %0d", mismatches);
        if (outputs_received != N_VEC) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[satAdd_tb] first mismatch t=%0d i=%0d got=%h exp=%h",
                     first_t, first_i, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[satAdd_tb] PASS");
        $finish;
    end

endmodule
