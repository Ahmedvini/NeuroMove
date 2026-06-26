// =============================================================================
// gumbel_channel_adapter.sv  —  19 → 5 channel selector for HaLT DB-ATCNet.
//
// The trained Gumbel layer at inference is argmax(logits, axis=0) over a
// (NUM_RAW=19, NUM_SEL=5) matrix → 5 electrode indices in [0, 18]. This
// module captures that selection as a tiny LUT (5 entries × 5 bits) and
// translates a streaming 19-channel raw EEG feed into the 5-channel
// stream db_atcnet_top expects. Duplicate LUT entries are handled.
//
// Per-subject `selected_channels.json` files in weights/ list the
// indices. For Subject A session 3: [9, 11, 12, 14, 3] (no duplicates).
// For other subjects/folds duplicates appear (e.g. [11, 3, 5, 9, 9]),
// so the model expects the channel-9 sample to be presented in both
// positions 3 and 4 of the 5-channel stream.
//
// Protocol:
//   - Caller drives one (in_sample, in_electrode) per cycle when
//     `in_ready` is high.
//   - For each accepted input the adapter emits N outputs over the next
//     N cycles, where N = number of LUT positions whose entry matches
//     `in_electrode`. While emitting, `in_ready` is low.
//   - Output: (out_sample, out_electrode = LUT position) one per cycle.
//
// LUT is loadable at runtime via the same broadcast (wr_en, wr_addr,
// wr_data) interface that weight_bank uses (PACK_2 = 1, 2 entries/word).
// Default LUT is Subject A session 3.
// =============================================================================

`timescale 1ns / 1ps

module gumbel_channel_adapter #(
    parameter int DATA_WIDTH     = 16,
    parameter int NUM_RAW        = 19,
    parameter int NUM_SEL        = 5,
    parameter int AXI_ADDR_WIDTH = 32,
    parameter int AXI_DATA_WIDTH = 32,
    parameter int LUT_BASE_ADDR  = 32'h0000_2000
) (
    input  logic                            clk,
    input  logic                            rst,

    // AXI-Lite broadcast write (snoop the LUT range only).
    input  logic                            wr_en,
    input  logic [AXI_ADDR_WIDTH-1:0]       wr_addr,
    input  logic [AXI_DATA_WIDTH-1:0]       wr_data,

    // Streaming raw 19-channel input.
    input  logic                            in_valid,
    output logic                            in_ready,
    input  logic [$clog2(NUM_RAW > 1 ? NUM_RAW : 2)-1:0] in_electrode,
    input  logic signed [DATA_WIDTH-1:0]    in_sample,

    // Streaming selected 5-channel output.
    output logic                            out_valid,
    output logic [$clog2(NUM_SEL > 1 ? NUM_SEL : 2)-1:0] out_electrode,
    output logic signed [DATA_WIDTH-1:0]    out_sample
);

    localparam int RAW_W = $clog2(NUM_RAW > 1 ? NUM_RAW : 2);
    localparam int SEL_W = $clog2(NUM_SEL > 1 ? NUM_SEL : 2);

    // -------------------------------------------------------------------------
    // 5-entry electrode-index LUT (default = Subject A session 3).
    // -------------------------------------------------------------------------
    logic [RAW_W-1:0] lut [0:NUM_SEL-1];

    initial begin
        lut[0] = RAW_W'(9);
        lut[1] = RAW_W'(11);
        lut[2] = RAW_W'(12);
        lut[3] = RAW_W'(14);
        lut[4] = RAW_W'(3);
    end

    // 3 × 32-bit AXI words cover the 5 (× 5-bit) LUT entries.
    localparam int LUT_END = LUT_BASE_ADDR + 3 * 4;

    always_ff @(posedge clk) begin
        if (!rst && wr_en && wr_addr >= LUT_BASE_ADDR && wr_addr < LUT_END) begin
            automatic int idx;
            idx = ((wr_addr - LUT_BASE_ADDR) / 4) * 2;
            if (idx     < NUM_SEL) lut[idx]     <= wr_data[RAW_W-1:0];
            if (idx + 1 < NUM_SEL) lut[idx + 1] <= wr_data[16 + RAW_W - 1:16];
        end
    end

    // -------------------------------------------------------------------------
    // Match mask + sample latch on each accepted input.
    // -------------------------------------------------------------------------
    logic [NUM_SEL-1:0]              match_mask;
    logic signed [DATA_WIDTH-1:0]    captured_sample;
    logic [SEL_W-1:0]                emit_pos;

    // Combinational match for the incoming input.
    logic [NUM_SEL-1:0]              cur_match;
    always_comb begin
        for (int j = 0; j < NUM_SEL; j++)
            cur_match[j] = (lut[j] == in_electrode);
    end

    wire is_busy = |match_mask;
    assign in_ready = !is_busy;

    always_ff @(posedge clk) begin
        if (rst) begin
            match_mask      <= '0;
            captured_sample <= '0;
            emit_pos        <= '0;
            out_valid       <= 1'b0;
            out_electrode   <= '0;
            out_sample      <= '0;
        end else begin
            out_valid <= 1'b0;

            if (!is_busy && in_valid && in_ready) begin
                // Capture the input and the full match mask.
                match_mask      <= cur_match;
                captured_sample <= in_sample;
            end

            // Drain one bit of match_mask per cycle (lowest-index first).
            if (is_busy) begin
                for (int j = 0; j < NUM_SEL; j++) begin
                    if (match_mask[j]) begin
                        out_valid     <= 1'b1;
                        out_electrode <= j[SEL_W-1:0];
                        out_sample    <= captured_sample;
                        match_mask[j] <= 1'b0;
                        break;
                    end
                end
            end
        end
    end

endmodule
