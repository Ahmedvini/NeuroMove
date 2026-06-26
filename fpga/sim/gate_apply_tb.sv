// =============================================================================
// gate_apply_tb.sv  —  Bit-exact regression for the per-channel gate multiplier
//
// Streams the Layer 1+2 feature map (stage_conv2d_output.hex) plus the
// precomputed ECA₁ gate vector (stage_eca1_gate.hex) into gate_apply.sv,
// then compares every output vector against the gated reference
// (stage_eca1_output.hex).
//
// Setup:
//   python3 scripts/q88_layer1.py
//   python3 scripts/q88_eca1.py    # writes stage_eca1_output.hex
// =============================================================================

`timescale 1ns / 1ps

module gate_apply_tb;

    // --- Parameters (match q88_eca1.py for ECA₁ position) ---
    localparam int T          = 600;
    localparam int C          = 5;
    localparam int NUM_CH     = 16;
    localparam int N_VECTORS  = T * C;
    localparam int DATA_WIDTH = 16;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;

    localparam int TOTAL_SCALARS = N_VECTORS * NUM_CH;   // 48000

    // --- Memories ---
    logic signed [DATA_WIDTH-1:0] fmap_mem [0:TOTAL_SCALARS - 1];
    logic signed [DATA_WIDTH-1:0] gate_mem [0:NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem  [0:TOTAL_SCALARS - 1];

    initial begin
        $readmemh("data/golden_q88/stage_conv2d_output.hex", fmap_mem);
        $readmemh("data/golden_q88/stage_eca1_gate.hex",     gate_mem);
        $readmemh("data/golden_q88/stage_eca1_output.hex",   exp_mem);
    end

    // --- DUT ---
    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] x_in    [0:NUM_CH-1];
    logic                         x_valid;
    logic signed [DATA_WIDTH-1:0] gate_in [0:NUM_CH-1];
    logic signed [DATA_WIDTH-1:0] y_out   [0:NUM_CH-1];
    logic                         y_valid;

    gate_apply #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_CH     (NUM_CH),
        .FRAC_BITS  (FRAC_BITS)
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in),
        .x_valid(x_valid),
        .gate_in(gate_in),
        .y_out(y_out),
        .y_valid(y_valid)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // --- Counters & mismatch tracker (single always_ff owner) ---
    int outputs_received;
    int mismatches;
    int first_mismatch_vec, first_mismatch_c;
    logic signed [DATA_WIDTH-1:0] first_mismatch_got;
    logic signed [DATA_WIDTH-1:0] first_mismatch_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received   <= 0;
            mismatches         <= 0;
            first_mismatch_vec <= -1;
            first_mismatch_c   <= -1;
            first_mismatch_got <= '0;
            first_mismatch_exp <= '0;
        end else if (y_valid) begin
            int vec_idx;
            vec_idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (vec_idx < N_VECTORS) begin
                for (int c = 0; c < NUM_CH; c++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * NUM_CH + c];
                    if (y_out[c] !== exp_v) begin
                        if (mismatches < 8) begin
                            $display("[gate_tb] mismatch vec=%0d c=%0d  got=%h  exp=%h",
                                     vec_idx, c, y_out[c], exp_v);
                        end
                        if (mismatches == 0) begin
                            first_mismatch_vec <= vec_idx;
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

    // --- Drive ---
    int n_drive;

    initial begin
        rst = 1'b1;
        x_valid = 1'b0;
        for (int c = 0; c < NUM_CH; c++) begin
            x_in[c]    = '0;
            gate_in[c] = '0;
        end
        n_drive = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        // Hold the gate steady across the entire frame.
        @(posedge clk);
        for (int c = 0; c < NUM_CH; c++) gate_in[c] <= gate_mem[c];

        for (int v = 0; v < N_VECTORS; v++) begin
            @(posedge clk);
            x_valid <= 1'b1;
            for (int c = 0; c < NUM_CH; c++) begin
                x_in[c] <= fmap_mem[v * NUM_CH + c];
            end
            n_drive++;
        end
        @(posedge clk);
        x_valid <= 1'b0;
        for (int c = 0; c < NUM_CH; c++) x_in[c] <= '0;

        repeat (4) @(posedge clk);

        $display("[gate_tb] vectors driven : %0d (expected %0d)", n_drive, N_VECTORS);
        $display("[gate_tb] outputs received: %0d (expected %0d)", outputs_received, N_VECTORS);
        $display("[gate_tb] mismatches      : %0d", mismatches);
        if (n_drive != N_VECTORS)              $fatal(1, "drive count off");
        if (outputs_received != N_VECTORS)     $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[gate_tb] first mismatch vec=%0d c=%0d got=%h exp=%h",
                     first_mismatch_vec, first_mismatch_c,
                     first_mismatch_got, first_mismatch_exp);
            $fatal(1, "FAIL");
        end
        $display("[gate_tb] PASS");
        $finish;
    end

endmodule
