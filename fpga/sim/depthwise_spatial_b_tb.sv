// =============================================================================
// depthwise_spatial_b_tb.sv  —  Bit-exact regression for Branch B depthwise
// (Layer 10 + BN folded) of HaLT DB-ATCNet — D=4, F2 = F1*D = 64.
//
// Same RTL module as Branch A (depthwise_spatial.sv), only D is changed.
// Setup:  python3 scripts/q88_branchB.py
// =============================================================================

`timescale 1ns / 1ps

module depthwise_spatial_b_tb;

    localparam int T          = 600;
    localparam int C_IN       = 5;
    localparam int F1         = 16;
    localparam int D          = 4;
    localparam int F2         = F1 * D;     // 64
    localparam int DATA_WIDTH = 16;
    localparam int COEF_WIDTH = 16;
    localparam int ACC_WIDTH  = 48;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;

    localparam int N_INPUT_VEC  = T * C_IN;
    localparam int N_OUTPUT_VEC = T;

    logic signed [DATA_WIDTH-1:0] fmap_mem [0:N_INPUT_VEC  * F1 - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem  [0:N_OUTPUT_VEC * F2 - 1];

    initial begin
        $readmemh("data/golden_q88/stage_branchB_dwise_input.hex",  fmap_mem);
        $readmemh("data/golden_q88/stage_branchB_dwise_output.hex", exp_mem);
    end

    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] x_in       [0:F1-1];
    logic                         x_valid;
    logic                         frame_last;
    logic signed [DATA_WIDTH-1:0] y_out      [0:F2-1];
    logic                         y_valid;

    depthwise_spatial #(
        .DATA_WIDTH   (DATA_WIDTH),
        .COEF_WIDTH   (COEF_WIDTH),
        .ACC_WIDTH    (ACC_WIDTH),
        .FRAC_BITS    (FRAC_BITS),
        .C_IN         (C_IN),
        .F1           (F1),
        .D            (D),
        .F2           (F2),
        .WEIGHTS_FILE ("data/golden_q88/stage_branchB_dwise_weights.hex"),
        .BIAS_FILE    ("data/golden_q88/stage_branchB_dwise_bias.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in), .x_valid(x_valid), .frame_last(frame_last),
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
            if (vec_idx < N_OUTPUT_VEC) begin
                for (int i = 0; i < F2; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * F2 + i];
                    if (y_out[i] !== exp_v) begin
                        if (mismatches < 8)
                            $display("[dwiseB_tb] mismatch t=%0d i=%0d got=%h exp=%h",
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

    int n_drive;
    initial begin
        rst = 1'b1;
        x_valid = 1'b0;
        frame_last = 1'b0;
        for (int f = 0; f < F1; f++) x_in[f] = '0;
        n_drive = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < T; t++) begin
            for (int c = 0; c < C_IN; c++) begin
                @(posedge clk);
                x_valid    <= 1'b1;
                frame_last <= (c == C_IN - 1);
                for (int f = 0; f < F1; f++)
                    x_in[f] <= fmap_mem[(t * C_IN + c) * F1 + f];
                n_drive++;
            end
        end
        @(posedge clk);
        x_valid <= 1'b0;
        frame_last <= 1'b0;
        for (int f = 0; f < F1; f++) x_in[f] <= '0;
        repeat (4) @(posedge clk);

        $display("[dwiseB_tb] vectors driven  : %0d (expected %0d)", n_drive, N_INPUT_VEC);
        $display("[dwiseB_tb] outputs received: %0d (expected %0d)", outputs_received, N_OUTPUT_VEC);
        $display("[dwiseB_tb] mismatches      : %0d", mismatches);
        if (outputs_received != N_OUTPUT_VEC) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[dwiseB_tb] first mismatch t=%0d i=%0d got=%h exp=%h",
                     first_t, first_i, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[dwiseB_tb] PASS");
        $finish;
    end

endmodule
