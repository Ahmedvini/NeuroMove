// =============================================================================
// output_head.sv  —  Accumulate logits from N_WIN windows and argmax.
//
//   For w in 0..N_WIN-1:
//     accept (logits_w[0], logits_w[1], ...) on `in_valid`
//     sum_logits[o] += logits_w[o]
//   When `in_last && in_valid`:
//     class_out = argmax_o(sum_logits[o])
//     done = 1 (1-cycle pulse)
//
// Since average = sum/N_WIN with N_WIN constant > 0, argmax(mean) ==
// argmax(sum). We skip the division.
//
// For ties between equal logits, the lower index wins (Numpy convention).
// =============================================================================

`timescale 1ns / 1ps

module output_head #(
    parameter int DATA_WIDTH = 16,
    parameter int ACC_WIDTH  = 24,             // 16 + log2(5) + headroom
    parameter int N_CLS      = 2,
    parameter int N_WIN      = 5
) (
    input  logic                              clk,
    input  logic                              rst,

    input  logic signed [DATA_WIDTH-1:0]      logits_in [0:N_CLS-1],
    input  logic                              in_valid,
    input  logic                              in_last,     // pulse with last window's logits

    output logic [$clog2(N_CLS)-1:0]          class_out,
    output logic                              done
);

    logic signed [ACC_WIDTH-1:0] sum_logits [0:N_CLS-1];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int o = 0; o < N_CLS; o++) sum_logits[o] <= '0;
            class_out <= '0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;
            if (in_valid && in_last) begin
                // Compute argmax over sum_logits + this last window's logits.
                // For ties, lower index wins.
                automatic logic signed [ACC_WIDTH-1:0] best_val;
                automatic logic [$clog2(N_CLS)-1:0]    best_idx;
                automatic logic signed [ACC_WIDTH-1:0] this_val;
                best_idx = '0;
                this_val = sum_logits[0]
                         + $signed({{(ACC_WIDTH-DATA_WIDTH){logits_in[0][DATA_WIDTH-1]}}, logits_in[0]});
                best_val = this_val;
                for (int o = 1; o < N_CLS; o++) begin
                    this_val = sum_logits[o]
                             + $signed({{(ACC_WIDTH-DATA_WIDTH){logits_in[o][DATA_WIDTH-1]}}, logits_in[o]});
                    if (this_val > best_val) begin
                        best_val = this_val;
                        best_idx = ($clog2(N_CLS))'(o);
                    end
                end
                class_out <= best_idx;
                done      <= 1'b1;
                // Reset accumulator for next inference.
                for (int o = 0; o < N_CLS; o++) sum_logits[o] <= '0;
            end else if (in_valid) begin
                for (int o = 0; o < N_CLS; o++)
                    sum_logits[o] <= sum_logits[o]
                                   + $signed({{(ACC_WIDTH-DATA_WIDTH){logits_in[o][DATA_WIDTH-1]}}, logits_in[o]});
            end
        end
    end

endmodule
