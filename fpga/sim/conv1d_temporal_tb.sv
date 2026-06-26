// =============================================================================
// conv1d_temporal_tb.sv  —  Bit-exact regression for Branch A separable
// Conv2D + BN-fold (Layer 7) of HaLT DB-ATCNet.
//
// Streams stage_branchA_sep_input.hex (T_POOL=75 × F_IN=32) into
// conv1d_temporal.sv with KE=16 same-padding (7 zeros pre, 8 zeros post),
// then checks 75 output vectors against stage_branchA_sep_output.hex.
//
// Setup:
//   python3 scripts/q88_layer7.py
// =============================================================================

`timescale 1ns / 1ps

module conv1d_temporal_tb;

    localparam int T_POOL     = 75;
    localparam int F_IN       = 32;
    localparam int F_OUT      = 32;
    localparam int KE         = 16;
    localparam int DATA_WIDTH = 16;
    localparam int COEF_WIDTH = 16;
    localparam int ACC_WIDTH  = 48;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;

    localparam int PAD_TOP = (KE - 1) / 2;            // 7
    localparam int PAD_BOT = KE - 1 - PAD_TOP;        // 8
    localparam int T_TOTAL = PAD_TOP + T_POOL + PAD_BOT;  // 90

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_POOL*F_IN - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_POOL*F_OUT - 1];

    initial begin
        $readmemh("data/golden_q88/stage_branchA_sep_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_branchA_sep_output.hex", exp_mem);
    end

    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] in_sample [0:F_IN-1];
    logic                         in_valid;
    logic signed [DATA_WIDTH-1:0] y_out     [0:F_OUT-1];
    logic                         y_valid;

    conv1d_temporal #(
        .DATA_WIDTH   (DATA_WIDTH),
        .COEF_WIDTH   (COEF_WIDTH),
        .ACC_WIDTH    (ACC_WIDTH),
        .FRAC_BITS    (FRAC_BITS),
        .F_IN         (F_IN),
        .F_OUT        (F_OUT),
        .KE           (KE),
        .WEIGHTS_FILE ("data/golden_q88/stage_branchA_sep_weights.hex"),
        .BIAS_FILE    ("data/golden_q88/stage_branchA_sep_bias.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .in_sample(in_sample),
        .in_valid(in_valid),
        .y_out(y_out),
        .y_valid(y_valid)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    function automatic logic signed [DATA_WIDTH-1:0] sample_at(int t_in, int fi);
        int t_real;
        if (t_in < PAD_TOP || t_in >= PAD_TOP + T_POOL) return '0;
        t_real = t_in - PAD_TOP;
        return in_mem[t_real * F_IN + fi];
    endfunction

    // --- Counter + mismatch tracker (single owner) ---
    int out_seq;
    int outputs_received;
    int mismatches;
    int first_mismatch_t, first_mismatch_f;
    logic signed [DATA_WIDTH-1:0] first_mismatch_got;
    logic signed [DATA_WIDTH-1:0] first_mismatch_exp;
    int n_check;

    always_ff @(posedge clk) begin
        if (rst) begin
            out_seq            <= 0;
            outputs_received   <= 0;
            mismatches         <= 0;
            n_check            <= 0;
            first_mismatch_t   <= -1;
            first_mismatch_f   <= -1;
            first_mismatch_got <= '0;
            first_mismatch_exp <= '0;
        end else if (y_valid) begin
            int t_in;
            int t_out;
            t_in = out_seq;
            outputs_received <= outputs_received + 1;
            // Output for T_in is for t_out = T_in - (KE - 1) = T_in - 15
            if (t_in >= (KE-1) && t_in < (KE-1) + T_POOL) begin
                t_out = t_in - (KE - 1);
                for (int fo = 0; fo < F_OUT; fo++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[t_out * F_OUT + fo];
                    if (y_out[fo] !== exp_v) begin
                        if (mismatches < 8)
                            $display("[sepA_tb] mismatch t=%0d fo=%0d got=%h exp=%h",
                                     t_out, fo, y_out[fo], exp_v);
                        if (mismatches == 0) begin
                            first_mismatch_t   <= t_out;
                            first_mismatch_f   <= fo;
                            first_mismatch_got <= y_out[fo];
                            first_mismatch_exp <= exp_v;
                        end
                        mismatches <= mismatches + 1;
                    end
                end
                n_check <= n_check + 1;
            end
            out_seq <= out_seq + 1;
        end
    end

    int n_drive;

    initial begin
        rst = 1'b1;
        in_valid = 1'b0;
        for (int fi = 0; fi < F_IN; fi++) in_sample[fi] = '0;
        n_drive = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < T_TOTAL; t++) begin
            @(posedge clk);
            in_valid <= 1'b1;
            for (int fi = 0; fi < F_IN; fi++)
                in_sample[fi] <= sample_at(t, fi);
            n_drive++;
        end
        @(posedge clk);
        in_valid <= 1'b0;
        for (int fi = 0; fi < F_IN; fi++) in_sample[fi] <= '0;

        repeat (4) @(posedge clk);

        $display("[sepA_tb] cycles driven   : %0d (expected %0d)", n_drive, T_TOTAL);
        $display("[sepA_tb] outputs received: %0d", outputs_received);
        $display("[sepA_tb] outputs checked : %0d (expected %0d)", n_check, T_POOL);
        $display("[sepA_tb] mismatches      : %0d", mismatches);
        if (n_check != T_POOL)         $fatal(1, "check count off");
        if (mismatches != 0) begin
            $display("[sepA_tb] first mismatch t=%0d fo=%0d got=%h exp=%h",
                     first_mismatch_t, first_mismatch_f,
                     first_mismatch_got, first_mismatch_exp);
            $fatal(1, "FAIL");
        end
        $display("[sepA_tb] PASS");
        $finish;
    end

endmodule
