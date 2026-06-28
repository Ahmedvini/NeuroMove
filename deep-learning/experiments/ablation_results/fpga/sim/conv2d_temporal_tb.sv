// =============================================================================
// conv2d_temporal_tb.sv
//
// Bit-exact regression for rtl/conv/conv2d_temporal.sv against the Q8.8
// Python golden produced by scripts/q88_layer1.py.
//
// Data files (loaded via $readmemh, paths relative to the simulator cwd):
//   data/golden_q88/stage_conv2d_input.hex     T*C   Q8.8 (time-major, electrode)
//   data/golden_q88/stage_conv2d_weights.hex   KE*F1 Q8.8 (tap-major, filter)
//   data/golden_q88/stage_conv2d_output.hex    T*C*F1 Q8.8 expected
//
// Test loop:
//   - Feed pad_top zero samples per electrode (warmup left padding).
//   - Feed real samples for t = 0..TRIAL_SAMPLES-1, electrodes 0..C-1.
//   - Feed pad_bot zero samples to drain the right padding.
//   - Compare each registered output to expected[t_out, electrode, filter].
//   - PASS if all match exactly.
// =============================================================================

`timescale 1ns / 1ps

module conv2d_temporal_tb;

    // --- Parameters (match scripts/q88_layer1.py for HaLT) ---
    localparam int NUM_EEG_CH    = 5;     // selected channels after Gumbel
    localparam int F1            = 16;
    localparam int KE            = 64;
    localparam int TRIAL_SAMPLES = 600;   // 3 s @ 200 Hz
    localparam int DATA_WIDTH    = 16;
    localparam int COEF_WIDTH    = 16;
    localparam int ACC_WIDTH     = 48;
    localparam int FRAC_BITS     = 8;

    localparam int PAD_TOP = (KE - 1) / 2;   // 31
    localparam int PAD_BOT = KE - 1 - PAD_TOP;   // 32
    localparam int T_TOTAL = PAD_TOP + TRIAL_SAMPLES + PAD_BOT;  // 703 cycles per electrode

    localparam int CLK_PERIOD = 10;

    // --- Memories ---
    logic signed [DATA_WIDTH-1:0] in_mem  [0:TRIAL_SAMPLES*NUM_EEG_CH-1];
    logic signed [DATA_WIDTH-1:0] exp_mem [0:TRIAL_SAMPLES*NUM_EEG_CH*F1-1];

    initial begin
        $readmemh("data/golden_q88/stage_conv2d_input.hex",  in_mem);
        $readmemh("data/golden_q88/stage_conv2d_output.hex", exp_mem);
    end

    // --- DUT ---
    logic clk;
    logic rst;
    logic                                in_valid;
    logic [$clog2(NUM_EEG_CH)-1:0]       in_electrode;
    logic signed [DATA_WIDTH-1:0]        in_sample;
    logic                                out_valid;
    logic [$clog2(NUM_EEG_CH)-1:0]       out_electrode;
    logic signed [DATA_WIDTH-1:0]        out_filter [0:F1-1];

    conv2d_temporal #(
        .NUM_EEG_CH   (NUM_EEG_CH),
        .F1           (F1),
        .KE           (KE),
        .DATA_WIDTH   (DATA_WIDTH),
        .COEF_WIDTH   (COEF_WIDTH),
        .ACC_WIDTH    (ACC_WIDTH),
        .FRAC_BITS    (FRAC_BITS),
        .WEIGHTS_FILE ("data/golden_q88/stage_conv2d_weights.hex"),
        .BIAS_FILE    ("data/golden_q88/stage_conv2d_bias.hex")
    ) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid),
        .in_electrode(in_electrode),
        .in_sample(in_sample),
        .out_valid(out_valid),
        .out_electrode(out_electrode),
        .out_filter(out_filter)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // --- Stim / drive (drive at posedge with non-blocking) ---
    // Helper: t_in is the input time index in [0, T_TOTAL). Returns the
    // electrode's Q8.8 sample, or 0 during pad_top / pad_bot.
    function automatic logic signed [DATA_WIDTH-1:0] sample_at(int t_in, int e);
        int t_real;
        if (t_in < PAD_TOP || t_in >= PAD_TOP + TRIAL_SAMPLES) begin
            return '0;
        end else begin
            t_real = t_in - PAD_TOP;
            return in_mem[t_real * NUM_EEG_CH + e];
        end
    endfunction

    // --- Counters & mismatch trackers (driven only in the verify always_ff) ---
    int outputs_received;
    int mismatches;
    int first_mismatch_t, first_mismatch_e, first_mismatch_f;
    logic signed [DATA_WIDTH-1:0] first_mismatch_got, first_mismatch_exp;
    int n_drive;
    int n_check;
    int out_seq;

    int local_n_drive;

    initial begin
        rst = 1'b1;
        in_valid = 1'b0;
        in_electrode = '0;
        in_sample = '0;
        local_n_drive = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        for (int t_in = 0; t_in < T_TOTAL; t_in++) begin
            for (int e = 0; e < NUM_EEG_CH; e++) begin
                @(posedge clk);
                in_valid     <= 1'b1;
                in_electrode <= e[$clog2(NUM_EEG_CH)-1:0];
                in_sample    <= sample_at(t_in, e);
                local_n_drive++;
            end
        end
        @(posedge clk);
        in_valid <= 1'b0;
        in_sample <= '0;

        // Drain the pipeline
        repeat (4) @(posedge clk);

        // Report
        $display("[conv2d_temporal_tb] inputs driven : %0d", local_n_drive);
        $display("[conv2d_temporal_tb] outputs seen  : %0d", outputs_received);
        $display("[conv2d_temporal_tb] outputs checked: %0d", n_check);
        $display("[conv2d_temporal_tb] mismatches    : %0d", mismatches);
        if (mismatches != 0) begin
            $display("[conv2d_temporal_tb] first mismatch t_out=%0d e=%0d f=%0d  got=%h  exp=%h",
                     first_mismatch_t, first_mismatch_e, first_mismatch_f,
                     first_mismatch_got, first_mismatch_exp);
            $fatal(1, "FAIL");
        end
        if (n_check == 0) $fatal(1, "no outputs checked");
        $display("[conv2d_temporal_tb] PASS");
        $finish;
    end

    // -------------------------------------------------------------------------
    // Output check (single always_ff that owns all the counter variables).
    // Inputs are driven in order (t_in=0,e=0..63), (t_in=1,e=0..63), ... and
    // outputs follow with fixed 1-cycle latency, so the k-th out_valid pulse
    // corresponds to the k-th input.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            out_seq            <= 0;
            outputs_received   <= 0;
            mismatches         <= 0;
            n_check            <= 0;
            first_mismatch_t   <= -1;
            first_mismatch_e   <= -1;
            first_mismatch_f   <= -1;
            first_mismatch_got <= '0;
            first_mismatch_exp <= '0;
        end else if (out_valid) begin
            int t_in;
            int e_seen;
            int t_out;
            t_in   = out_seq / NUM_EEG_CH;
            e_seen = out_seq % NUM_EEG_CH;
            outputs_received <= outputs_received + 1;
            // Output at drive cycle T_in corresponds to Keras output time
            //   t_out = T_in - (KE-1).
            // The shift register needs KE-1 inputs of warmup before the
            // first valid window is in place.
            if (t_in >= (KE-1) && t_in < (KE-1) + TRIAL_SAMPLES) begin
                t_out = t_in - (KE-1);
                for (int f = 0; f < F1; f++) begin
                    logic signed [DATA_WIDTH-1:0] exp_v;
                    exp_v = exp_mem[t_out * NUM_EEG_CH * F1 + e_seen * F1 + f];
                    if (out_filter[f] !== exp_v) begin
                        if (mismatches < 8) begin
                            $display("[conv2d_temporal_tb] mismatch t_out=%0d e=%0d f=%0d  got=%h  exp=%h",
                                     t_out, e_seen, f, out_filter[f], exp_v);
                        end
                        if (mismatches == 0) begin
                            first_mismatch_t   <= t_out;
                            first_mismatch_e   <= e_seen;
                            first_mismatch_f   <= f;
                            first_mismatch_got <= out_filter[f];
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

endmodule
