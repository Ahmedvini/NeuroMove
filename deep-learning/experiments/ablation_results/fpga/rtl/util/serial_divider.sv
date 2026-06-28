// =============================================================================
// serial_divider.sv  —  Unsigned restoring divider (radix-2). Computes
//
//     quotient  = floor(numerator / denominator)
//     remainder = numerator - quotient * denominator
//
// Latency: N_WIDTH+2 cycles between `start` pulse and `done` pulse.
// One operation at a time (`busy` is high during the work). When `done`
// pulses, `quotient` and `remainder` are valid for one cycle.
//
// numerator is N_WIDTH bits. denominator is D_WIDTH bits. Internally we
// use an (N_WIDTH+1)-bit partial remainder.
//
// Division by zero: caller must guarantee denominator != 0. For the
// stochastic-pool use case we floor the denominator to 1 in Q8.8.
// =============================================================================

`timescale 1ns / 1ps

module serial_divider #(
    parameter int N_WIDTH = 32,
    parameter int D_WIDTH = 16,
    parameter int Q_WIDTH = N_WIDTH    // quotient width = N_WIDTH bits
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic [N_WIDTH-1:0]     numerator,
    input  logic [D_WIDTH-1:0]     denominator,
    output logic                   busy,
    output logic                   done,
    output logic [Q_WIDTH-1:0]     quotient,
    output logic [D_WIDTH-1:0]     remainder
);

    typedef enum logic [1:0] {S_IDLE, S_WORK, S_DONE} state_t;
    state_t state;

    // Partial remainder is wide enough to absorb a left-shifted bit of the
    // dividend without losing it: (D_WIDTH+1) bits.
    localparam int R_WIDTH = D_WIDTH + 1;

    logic [N_WIDTH-1:0]     n_reg;
    logic [R_WIDTH-1:0]     r_reg;
    logic [Q_WIDTH-1:0]     q_reg;
    logic [$clog2(Q_WIDTH+1)-1:0] bit_idx;
    logic [D_WIDTH-1:0]     d_reg;

    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            quotient  <= '0;
            remainder <= '0;
            n_reg     <= '0;
            r_reg     <= '0;
            q_reg     <= '0;
            d_reg     <= '0;
            bit_idx   <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        n_reg   <= numerator;
                        d_reg   <= denominator;
                        r_reg   <= '0;
                        q_reg   <= '0;
                        bit_idx <= Q_WIDTH;
                        busy    <= 1'b1;
                        state   <= S_WORK;
                    end
                end
                S_WORK: begin
                    // Shift numerator bit into r.
                    logic [R_WIDTH-1:0] r_shift;
                    logic [R_WIDTH-1:0] r_sub;
                    r_shift = {r_reg[R_WIDTH-2:0], n_reg[N_WIDTH-1]};
                    r_sub   = r_shift - {1'b0, d_reg};
                    if (!r_sub[R_WIDTH-1]) begin     // r_shift >= d
                        r_reg <= r_sub;
                        q_reg <= {q_reg[Q_WIDTH-2:0], 1'b1};
                    end else begin
                        r_reg <= r_shift;
                        q_reg <= {q_reg[Q_WIDTH-2:0], 1'b0};
                    end
                    n_reg   <= {n_reg[N_WIDTH-2:0], 1'b0};
                    bit_idx <= bit_idx - 1;
                    if (bit_idx == 1) state <= S_DONE;
                end
                S_DONE: begin
                    quotient  <= q_reg;
                    remainder <= r_reg[D_WIDTH-1:0];
                    busy      <= 1'b0;
                    done      <= 1'b1;
                    state     <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
