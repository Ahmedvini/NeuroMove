// =============================================================================
// eeg_channel_adapter.sv
//
// Maps NUM_INPUT_CH (default 3) parallel EEG samples into NUM_EEG_CH (default
// 64) slots of the trained-model's input array, with unmapped slots tied to
// constant 0. The mapping is given by the INPUT_INDICES parameter:
//
//   out[ INPUT_INDICES[k] ] = in[k]    for k in [0, NUM_INPUT_CH)
//   out[other indices]      = 0
//
// Default mapping (PhysioNet 10-10 ordering):
//   in[0] (C3) -> out[8]
//   in[1] (Cz) -> out[10]
//   in[2] (C4) -> out[12]
//
// Latency: 1 cycle (registered output).
//
// Notes:
//   - Vivado synthesis will constant-propagate the 61 hardwired-0 channels
//     and remove their downstream multiplications. No DSP cost.
//   - For Z-7020 vs Ultrascale+ this module is identical (no DSPs, just
//     wires + 64 flip-flops).
// =============================================================================

`timescale 1ns / 1ps

module eeg_channel_adapter #(
    parameter int DATA_WIDTH    = 16,
    parameter int NUM_INPUT_CH  = 3,
    parameter int NUM_EEG_CH    = 64,
    // PhysioNet 10-10 indices for C3 / Cz / C4 by default.
    parameter int INPUT_INDICES [0:NUM_INPUT_CH-1] = '{8, 10, 12}
) (
    input  logic clk,
    input  logic rst,

    input  logic signed [DATA_WIDTH-1:0] in_sample [0:NUM_INPUT_CH-1],
    input  logic                         in_valid,

    output logic signed [DATA_WIDTH-1:0] out_sample [0:NUM_EEG_CH-1],
    output logic                         out_valid
);

    // Elaboration-time check: every input index must fall inside [0, NUM_EEG_CH).
    generate
        for (genvar gi = 0; gi < NUM_INPUT_CH; gi++) begin : gen_idx_check
            if (INPUT_INDICES[gi] < 0 || INPUT_INDICES[gi] >= NUM_EEG_CH) begin
                $error("INPUT_INDICES[%0d]=%0d is out of range [0, %0d).",
                       gi, INPUT_INDICES[gi], NUM_EEG_CH);
            end
        end
    endgenerate

    // Build the unpacked-array next-state combinationally.
    logic signed [DATA_WIDTH-1:0] next_out [0:NUM_EEG_CH-1];

    always_comb begin
        // Default: zero everywhere.
        for (int i = 0; i < NUM_EEG_CH; i++) next_out[i] = '0;
        // Overlay the active inputs at their mapped indices.
        for (int k = 0; k < NUM_INPUT_CH; k++) next_out[INPUT_INDICES[k]] = in_sample[k];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < NUM_EEG_CH; i++) out_sample[i] <= '0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                for (int i = 0; i < NUM_EEG_CH; i++) out_sample[i] <= next_out[i];
            end
        end
    end

endmodule
