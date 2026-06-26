// =============================================================================
// gumbel_channel_adapter_tb.sv  —  Sanity tests for the 19→5 adapter.
//   Case A : Subject A session 3 LUT [9, 11, 12, 14, 3] (no duplicates).
//            Drive 19 distinct electrodes; expect 5 outputs in LUT order
//            of incidence (i.e. position 4 first since channel 3, then
//            position 0 for channel 9, etc.).
//   Case B : Duplicate LUT [9, 9, 12, 9, 3]: channel-9 input should fan
//            out to positions 0, 1, 3 over 3 cycles.
// =============================================================================

`timescale 1ns / 1ps

module gumbel_channel_adapter_tb;

    localparam int DATA_WIDTH = 16;
    localparam int NUM_RAW    = 19;
    localparam int NUM_SEL    = 5;
    localparam int CLK_PERIOD = 10;

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                       wr_en;
    logic [31:0]                wr_addr;
    logic [31:0]                wr_data;
    logic                       in_valid;
    logic                       in_ready;
    logic [$clog2(NUM_RAW)-1:0] in_electrode;
    logic signed [DATA_WIDTH-1:0] in_sample;
    logic                       out_valid;
    logic [$clog2(NUM_SEL)-1:0] out_electrode;
    logic signed [DATA_WIDTH-1:0] out_sample;

    gumbel_channel_adapter #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_RAW(NUM_RAW), .NUM_SEL(NUM_SEL),
        .AXI_ADDR_WIDTH(32), .AXI_DATA_WIDTH(32),
        .LUT_BASE_ADDR(32'h0000_2000)
    ) dut (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .in_valid(in_valid), .in_ready(in_ready),
        .in_electrode(in_electrode), .in_sample(in_sample),
        .out_valid(out_valid), .out_electrode(out_electrode), .out_sample(out_sample)
    );

    int seen_out_a;       // outputs observed in Case A
    int seen_out_b;       // outputs observed in Case B
    int mismatches_ff;    // driven only in always_ff
    int mismatches_init;  // driven only in initial block

    // Expected Case A (deploy LUT [9, 11, 12, 14, 3]): driving electrodes
    // 0..18, only electrodes 3, 9, 11, 12, 14 produce outputs.
    // Order of emission = order of input. Expected (in driver order):
    //   in elec=3  → out pos=4
    //   in elec=9  → out pos=0
    //   in elec=11 → out pos=1
    //   in elec=12 → out pos=2
    //   in elec=14 → out pos=3
    int unsigned exp_pos_a [0:4] = '{4, 0, 1, 2, 3};
    int unsigned exp_src_a [0:4] = '{3, 9, 11, 12, 14};

    // Capture outputs for Case A only (i.e. while seen_out_a < 5).
    int unsigned out_log_pos_a [0:4];
    int unsigned out_log_src_a [0:4];

    // For Case B: LUT will be [9, 9, 12, 9, 3]. Driving electrode=9 with
    // sample=0x0099 should emit three outputs at positions {0, 1, 3} all
    // with sample 0x0099.
    int seen_b_outs;
    int unsigned b_pos_seen [0:4];

    int phase;     // 0=case A, 1=case B
    int b_send_idx;

    always_ff @(posedge clk) begin
        if (rst) begin
            seen_out_a    <= 0;
            seen_out_b    <= 0;
            mismatches_ff <= 0;
        end else if (out_valid) begin
            if (phase == 0 && seen_out_a < 5) begin
                out_log_pos_a[seen_out_a] <= out_electrode;
                out_log_src_a[seen_out_a] <= out_sample[7:0];
                seen_out_a <= seen_out_a + 1;
            end else if (phase == 1 && seen_out_b < NUM_SEL) begin
                b_pos_seen[seen_out_b] <= out_electrode;
                seen_out_b <= seen_out_b + 1;
                if (out_sample !== 16'h0099) mismatches_ff <= mismatches_ff + 1;
            end
        end
    end

    task automatic drive_one(input int unsigned e, input logic [DATA_WIDTH-1:0] s);
        // Wait for adapter to be ready, then drive one (electrode, sample).
        while (1) begin
            @(posedge clk);
            if (in_ready) break;
        end
        in_valid     <= 1'b1;
        in_electrode <= e[$clog2(NUM_RAW)-1:0];
        in_sample    <= s;
        @(posedge clk);
        in_valid <= 1'b0;
    endtask

    task automatic axil_write_lut_word(input int unsigned word_idx,
                                       input logic [31:0] data);
        wr_addr <= 32'h0000_2000 + word_idx * 4;
        wr_data <= data;
        wr_en   <= 1'b1;
        @(posedge clk);
        wr_en   <= 1'b0;
    endtask

    initial begin
        rst = 1'b1;
        wr_en = 1'b0; wr_addr = '0; wr_data = '0;
        in_valid = 1'b0; in_electrode = '0; in_sample = '0;
        phase = 0; b_send_idx = 0;
        mismatches_init = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        // ---------- Case A: default LUT, drive electrodes 0..18 ----------
        for (int e = 0; e < NUM_RAW; e++) begin
            drive_one(e, 16'(e));   // encode electrode in low byte
        end
        // Wait for all outputs to drain.
        repeat (10) @(posedge clk);
        $display("[gca_tb] Case A: seen %0d outputs (expected 5)", seen_out_a);
        if (seen_out_a !== 5) $fatal(1, "case A output count off");
        for (int k = 0; k < 5; k++) begin
            if (out_log_pos_a[k] !== exp_pos_a[k] || out_log_src_a[k] !== exp_src_a[k]) begin
                $display("[gca_tb] mismatch k=%0d got(pos=%0d src=%0d) exp(pos=%0d src=%0d)",
                         k, out_log_pos_a[k], out_log_src_a[k], exp_pos_a[k], exp_src_a[k]);
                mismatches_init++;
            end
        end

        // ---------- Case B: rewrite LUT to [9, 9, 12, 9, 3] then drive elec=9 ----------
        phase = 1;
        // Pack: word0 = {16'd9, 16'd9}, word1 = {16'd9, 16'd12}, word2 = {?, 16'd3}.
        axil_write_lut_word(0, {16'd9, 16'd9});
        axil_write_lut_word(1, {16'd9, 16'd12});
        axil_write_lut_word(2, {16'd0, 16'd3});
        @(posedge clk);
        drive_one(9, 16'h0099);
        // Wait for 3 outputs to drain.
        repeat (10) @(posedge clk);
        $display("[gca_tb] Case B: seen %0d outputs (expected 3)", seen_out_b);
        if (seen_out_b !== 3) $fatal(1, "case B output count off");
        // Expect positions 0, 1, 3 (LUT positions 0..3 that hold 9).
        if (!(b_pos_seen[0] == 0 && b_pos_seen[1] == 1 && b_pos_seen[2] == 3)) begin
            $display("[gca_tb] case B positions wrong: %0d %0d %0d",
                     b_pos_seen[0], b_pos_seen[1], b_pos_seen[2]);
            mismatches_init++;
        end

        if (mismatches_ff + mismatches_init != 0)
            $fatal(1, "[gca_tb] %0d ff + %0d init mismatches", mismatches_ff, mismatches_init);
        $display("[gca_tb] PASS");
        $finish;
    end

endmodule
