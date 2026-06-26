// =============================================================================
// gap_accumulator_tb.sv  —  Bit-exact regression for the streaming GAP block
//
// Streams the Layer 1+2 output (data/golden_q88/stage_conv2d_output.hex) into
// gap_accumulator.sv as 3000 sixteen-wide vectors (one per (t, c) pair), with
// frame_last asserted on the final beat. Compares the registered gap_out
// against stage_eca1_input.hex (which scripts/q88_eca1.py wrote as the
// GAP values fed into ECA₁).
//
// Setup:
//   python3 scripts/q88_layer1.py       # produces stage_conv2d_output.hex
//   python3 scripts/q88_eca1.py         # produces stage_eca1_input.hex (=GAP)
// =============================================================================

`timescale 1ns / 1ps

module gap_accumulator_tb;

    // --- Parameters (match q88_eca1.py for ECA₁ position) ---
    localparam int T            = 600;
    localparam int C            = 5;
    localparam int NUM_CH       = 16;       // F1 in db_atcnet_params
    localparam int N_GAP        = T * C;    // 3000
    localparam int DATA_WIDTH   = 16;
    localparam int ACC_WIDTH    = 32;
    localparam int INV_N        = 5592;     // round(2^24/3000)
    localparam int INV_N_SHIFT  = 24;
    localparam int INV_N_WIDTH  = 24;
    localparam int CLK_PERIOD   = 10;

    // --- Memories ---
    logic signed [DATA_WIDTH-1:0] fmap_mem [0:T*C*NUM_CH - 1];
    logic signed [DATA_WIDTH-1:0] exp_gap  [0:NUM_CH-1];

    initial begin
        $readmemh("data/golden_q88/stage_conv2d_output.hex", fmap_mem);
        $readmemh("data/golden_q88/stage_eca1_input.hex",    exp_gap);
    end

    // --- DUT ---
    logic clk;
    logic rst;
    logic signed [DATA_WIDTH-1:0] x_in    [0:NUM_CH-1];
    logic                         x_valid;
    logic                         frame_last;
    logic signed [DATA_WIDTH-1:0] gap_out [0:NUM_CH-1];
    logic                         gap_valid;

    gap_accumulator #(
        .DATA_WIDTH  (DATA_WIDTH),
        .NUM_CH      (NUM_CH),
        .ACC_WIDTH   (ACC_WIDTH),
        .INV_N       (INV_N),
        .INV_N_SHIFT (INV_N_SHIFT),
        .INV_N_WIDTH (INV_N_WIDTH)
    ) dut (
        .clk(clk), .rst(rst),
        .x_in(x_in),
        .x_valid(x_valid),
        .frame_last(frame_last),
        .gap_out(gap_out),
        .gap_valid(gap_valid)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // --- Counters & mismatch tracker (single always_ff owner) ---
    int outputs_received;
    int mismatches;
    int first_mismatch_idx;
    logic signed [DATA_WIDTH-1:0] first_mismatch_got;
    logic signed [DATA_WIDTH-1:0] first_mismatch_exp;

    always_ff @(posedge clk) begin
        if (rst) begin
            outputs_received   <= 0;
            mismatches         <= 0;
            first_mismatch_idx <= -1;
            first_mismatch_got <= '0;
            first_mismatch_exp <= '0;
        end else if (gap_valid) begin
            outputs_received <= outputs_received + 1;
            for (int c = 0; c < NUM_CH; c++) begin
                if (gap_out[c] !== exp_gap[c]) begin
                    if (mismatches < 8) begin
                        $display("[gap_tb] mismatch c=%0d  got=%h  exp=%h", c, gap_out[c], exp_gap[c]);
                    end
                    if (mismatches == 0) begin
                        first_mismatch_idx <= c;
                        first_mismatch_got <= gap_out[c];
                        first_mismatch_exp <= exp_gap[c];
                    end
                    mismatches <= mismatches + 1;
                end
            end
        end
    end

    // --- Drive ---
    int n_drive;

    initial begin
        rst = 1'b1;
        x_valid = 1'b0;
        frame_last = 1'b0;
        for (int c = 0; c < NUM_CH; c++) x_in[c] = '0;
        n_drive = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t = 0; t < T; t++) begin
            for (int e = 0; e < C; e++) begin
                @(posedge clk);
                x_valid <= 1'b1;
                frame_last <= ((t == T-1) && (e == C-1));
                for (int c = 0; c < NUM_CH; c++) begin
                    x_in[c] <= fmap_mem[(t * C + e) * NUM_CH + c];
                end
                n_drive++;
            end
        end
        @(posedge clk);
        x_valid    <= 1'b0;
        frame_last <= 1'b0;
        for (int c = 0; c < NUM_CH; c++) x_in[c] <= '0;

        repeat (4) @(posedge clk);

        $display("[gap_tb] vectors driven : %0d (expected %0d)", n_drive, N_GAP);
        $display("[gap_tb] outputs        : %0d (expected 1)", outputs_received);
        $display("[gap_tb] mismatches     : %0d", mismatches);
        if (n_drive != N_GAP)            $fatal(1, "drive count off");
        if (outputs_received != 1)       $fatal(1, "FAIL: expected exactly 1 gap_valid");
        if (mismatches != 0) begin
            $display("[gap_tb] first mismatch c=%0d got=%h exp=%h",
                     first_mismatch_idx, first_mismatch_got, first_mismatch_exp);
            $fatal(1, "FAIL");
        end
        $display("[gap_tb] PASS");
        $finish;
    end

endmodule
