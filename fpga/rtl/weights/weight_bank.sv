// =============================================================================
// weight_bank.sv  —  Generic single-bank weight RAM with optional AXI-Lite
// write port and $readmemh init for simulation.
//
// Each block in the DB-ATCNet design (Conv2D, ECA, Branches, ImpCBAM, TCFN,
// Dense) hosts one or more instances of this module to store its weights.
// In simulation the array initialises from INIT_FILE via $readmemh; in
// synthesis the PS writes the same values via AXI-Lite at boot. Both paths
// yield bit-exact reads.
//
// Address decoding: a write with `wr_en && wr_addr ∈ [BASE_ADDR, BASE_ADDR
// + DEPTH*BYTES_PER_ENTRY)` lands in this bank. Address is byte-granular
// (matches AXI-Lite naming) and step is BYTES_PER_ENTRY (default 4 for
// 32-bit AXI-Lite words — two Q8.8 weights per write).
//
// PACK_2 (default 1): each AXI-Lite write programs TWO consecutive Q8.8
// entries: [15:0] → array[idx], [31:16] → array[idx+1]. Caller's hex file
// is naturally one Q8.8 entry per line, so the .h2 → .hex packing for
// AXI-Lite needs an extra step (see scripts/pack_weights_axil.py).
//
// Read port: read[0..DEPTH-1] is a simple registered array (single-port
// register file). Consumers tap any subset combinationally. Synthesis maps
// to distributed RAM or BRAM depending on size.
// =============================================================================

`timescale 1ns / 1ps

module weight_bank #(
    parameter int    DATA_WIDTH       = 16,
    parameter int    DEPTH            = 1024,
    parameter int    AXI_DATA_WIDTH   = 32,
    parameter int    AXI_ADDR_WIDTH   = 32,
    parameter int    BASE_ADDR        = 32'h0000_0000,
    parameter int    BYTES_PER_ENTRY  = 4,          // PACK_2 writes 4B / 2 entries
    parameter bit    PACK_2           = 1,
    parameter string INIT_FILE        = ""          // $readmemh source for sim
) (
    input  logic                       clk,
    input  logic                       rst,

    // Generic weight-write fan-out from axi_lite_weight_loader.
    input  logic                       wr_en,
    input  logic [AXI_ADDR_WIDTH-1:0]  wr_addr,
    input  logic [AXI_DATA_WIDTH-1:0]  wr_data,

    // Parallel read access — consumers iterate over `mem`.
    output logic signed [DATA_WIDTH-1:0] mem [0:DEPTH-1]
);

    // -------------------------------------------------------------------------
    // Storage
    // -------------------------------------------------------------------------
    initial begin
        if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
        else for (int i = 0; i < DEPTH; i++) mem[i] = '0;
    end

    // -------------------------------------------------------------------------
    // AXI-Lite write decode (byte-address granularity, 4B/entry-pair).
    // -------------------------------------------------------------------------
    localparam int END_ADDR = BASE_ADDR + (PACK_2 ? (DEPTH / 2) : DEPTH) * BYTES_PER_ENTRY;

    always_ff @(posedge clk) begin
        if (rst) begin
            // Don't clobber INIT_FILE; rst only resets non-array state (none here).
        end else if (wr_en && wr_addr >= BASE_ADDR && wr_addr < END_ADDR) begin
            automatic int idx;
            if (PACK_2) begin
                idx = ((wr_addr - BASE_ADDR) / BYTES_PER_ENTRY) * 2;
                if (idx     < DEPTH) mem[idx]     <= $signed(wr_data[DATA_WIDTH-1:0]);
                if (idx + 1 < DEPTH) mem[idx + 1] <= $signed(wr_data[2*DATA_WIDTH-1:DATA_WIDTH]);
            end else begin
                idx = (wr_addr - BASE_ADDR) / BYTES_PER_ENTRY;
                if (idx < DEPTH) mem[idx] <= $signed(wr_data[DATA_WIDTH-1:0]);
            end
        end
    end

endmodule
