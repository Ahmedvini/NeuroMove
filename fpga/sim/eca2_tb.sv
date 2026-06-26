// =============================================================================
// eca2_tb.sv  —  Integrated bit-exact regression for Layer 16 (ECA₂) of
// HaLT DB-ATCNet. Wires gap_accumulator + eca_attention + gate_apply with
// NUM_CH=32 and N=T_IN=10. Drives stage_eca2_input.hex (10*32 Q8.8),
// compares against stage_eca2_output.hex.
//
// Setup:  python3 scripts/q88_layer16.py
// =============================================================================

`timescale 1ns / 1ps

module eca2_tb;

    localparam int T_IN       = 10;
    localparam int NUM_CH     = 32;
    localparam int KECA       = 3;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam int INV_N      = 1677722;     // round(2^24 / 10)

    logic signed [DATA_WIDTH-1:0] in_mem  [0:T_IN * NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:T_IN * NUM_CH - 1];

    initial begin
        $readmemh("data/golden_q88/stage_eca2_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_eca2_output.hex", exp_mem);
    end

    logic clk;
    logic rst;
    initial begin clk = 1'b0; forever #(CLK_PERIOD/2) clk = ~clk; end

    // ------------- GAP -------------
    logic signed [DATA_WIDTH-1:0] gap_in_vec   [0:NUM_CH-1];
    logic                         gap_valid_in;
    logic                         gap_frame_last;
    logic signed [DATA_WIDTH-1:0] gap_out_vec  [0:NUM_CH-1];
    logic                         gap_valid_out;

    gap_accumulator #(
        .DATA_WIDTH (DATA_WIDTH),
        .NUM_CH     (NUM_CH),
        .ACC_WIDTH  (32),
        .INV_N      (INV_N),
        .INV_N_SHIFT(24),
        .INV_N_WIDTH(24)
    ) u_gap (
        .clk(clk), .rst(rst),
        .x_in(gap_in_vec), .x_valid(gap_valid_in), .frame_last(gap_frame_last),
        .gap_out(gap_out_vec), .gap_valid(gap_valid_out)
    );

    // ------------- ECA Conv1D + sigmoid -------------
    logic signed [DATA_WIDTH-1:0] gate_vec   [0:NUM_CH-1];
    logic                         gate_valid;

    eca_attention #(
        .DATA_WIDTH  (DATA_WIDTH),
        .COEF_WIDTH  (16),
        .ACC_WIDTH   (48),
        .FRAC_BITS   (8),
        .NUM_CH      (NUM_CH),
        .KECA        (KECA),
        .LUT_N       (1024),
        .LUT_RANGE_Q (1024),
        .LUT_SHIFT   (1),
        .WEIGHTS_FILE("data/golden_q88/stage_eca2_weights.hex"),
        .LUT_FILE    ("data/lut/sigmoid_q88.hex")
    ) u_eca (
        .clk(clk), .rst(rst),
        .gap_in(gap_out_vec), .gap_valid(gap_valid_out),
        .gate_out(gate_vec), .gate_valid(gate_valid)
    );

    // Latch gate when valid (so we can hold it constant during apply phase).
    logic signed [DATA_WIDTH-1:0] gate_held [0:NUM_CH-1];
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < NUM_CH; i++) gate_held[i] <= '0;
        end else if (gate_valid) begin
            for (int i = 0; i < NUM_CH; i++) gate_held[i] <= gate_vec[i];
        end
    end

    // ------------- Gate apply -------------
    logic signed [DATA_WIDTH-1:0] apply_in_vec [0:NUM_CH-1];
    logic                         apply_valid_in;
    logic signed [DATA_WIDTH-1:0] apply_out    [0:NUM_CH-1];
    logic                         apply_valid_out;

    gate_apply #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_CH    (NUM_CH),
        .FRAC_BITS (8),
        .ACC_WIDTH (32)
    ) u_apply (
        .clk(clk), .rst(rst),
        .x_in(apply_in_vec), .x_valid(apply_valid_in),
        .gate_in(gate_held),
        .y_out(apply_out), .y_valid(apply_valid_out)
    );

    // ------------- Checker -------------
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
        end else if (apply_valid_out) begin
            int vec_idx;
            vec_idx = outputs_received;
            outputs_received <= outputs_received + 1;
            if (vec_idx < T_IN) begin
                for (int i = 0; i < NUM_CH; i++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[vec_idx * NUM_CH + i];
                    if (apply_out[i] !== exp_v) begin
                        if (mismatches < 8)
                            $display("[eca2_tb] mismatch t=%0d i=%0d got=%h exp=%h",
                                     vec_idx, i, apply_out[i], exp_v);
                        if (mismatches == 0) begin
                            first_t <= vec_idx; first_i <= i;
                            first_got <= apply_out[i]; first_exp <= exp_v;
                        end
                        mismatches <= mismatches + 1;
                    end
                end
            end
        end
    end

    initial begin
        rst = 1'b1;
        gap_valid_in   = 1'b0;
        gap_frame_last = 1'b0;
        apply_valid_in = 1'b0;
        for (int i = 0; i < NUM_CH; i++) begin
            gap_in_vec[i]   = '0;
            apply_in_vec[i] = '0;
        end
        repeat (4) @(posedge clk);
        rst = 1'b0;

        // --- Phase 1: drive 10 vectors into GAP ---
        for (int t = 0; t < T_IN; t++) begin
            @(posedge clk);
            gap_valid_in   <= 1'b1;
            gap_frame_last <= (t == T_IN - 1);
            for (int i = 0; i < NUM_CH; i++)
                gap_in_vec[i] <= in_mem[t * NUM_CH + i];
        end
        @(posedge clk);
        gap_valid_in   <= 1'b0;
        gap_frame_last <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) gap_in_vec[i] <= '0;

        // Wait for gate_valid + 1 (so gate_held is latched).
        wait (gate_valid == 1'b1);
        @(posedge clk);

        // --- Phase 2: re-drive 10 vectors into gate_apply ---
        for (int t = 0; t < T_IN; t++) begin
            @(posedge clk);
            apply_valid_in <= 1'b1;
            for (int i = 0; i < NUM_CH; i++)
                apply_in_vec[i] <= in_mem[t * NUM_CH + i];
        end
        @(posedge clk);
        apply_valid_in <= 1'b0;
        for (int i = 0; i < NUM_CH; i++) apply_in_vec[i] <= '0;
        repeat (4) @(posedge clk);

        $display("[eca2_tb] outputs received: %0d (expected %0d)", outputs_received, T_IN);
        $display("[eca2_tb] mismatches      : %0d", mismatches);
        if (outputs_received != T_IN) $fatal(1, "output count off");
        if (mismatches != 0) begin
            $display("[eca2_tb] first mismatch t=%0d i=%0d got=%h exp=%h",
                     first_t, first_i, first_got, first_exp);
            $fatal(1, "FAIL");
        end
        $display("[eca2_tb] PASS");
        $finish;
    end

endmodule
