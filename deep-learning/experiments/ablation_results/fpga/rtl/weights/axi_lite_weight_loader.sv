// =============================================================================
// axi_lite_weight_loader.sv  —  AXI4-Lite slave for runtime weight programming.
//
// Exposes a 32-bit AXI4-Lite write port (the read channel returns 0; PS only
// writes). Each AXI-Lite write becomes a one-cycle pulse on a generic
// "weight write" interface:
//
//   wr_en   : 1-cycle pulse when a write completes
//   wr_addr : 32-bit byte address (the same address PS wrote)
//   wr_data : 32-bit data (two Q8.8 weights packed: [15:0] = even-addr,
//             [31:16] = odd-addr — caller decides packing)
//
// Downstream "weight banks" (the model's per-block weight RAMs) snoop this
// interface and self-decode their own address ranges. The slave doesn't
// hold any weight memory itself — it's a thin protocol shim.
//
// AXI-Lite specifics implemented:
//   - Independent AW + W channel handshake (accepts when both VALID),
//   - Write response (BVALID/BREADY, BRESP = OKAY),
//   - Read channel returns zeros so the PS can probe without hanging.
//
// All output registers initialise to 0 on rst — no $readmemh used here.
// =============================================================================

`timescale 1ns / 1ps

module axi_lite_weight_loader #(
    parameter int ADDR_WIDTH = 32,
    parameter int DATA_WIDTH = 32
) (
    input  logic                   clk,
    input  logic                   rst,    // active high

    // AXI4-Lite slave write address channel
    input  logic [ADDR_WIDTH-1:0]  s_awaddr,
    input  logic                   s_awvalid,
    output logic                   s_awready,

    // AXI4-Lite slave write data channel
    input  logic [DATA_WIDTH-1:0]  s_wdata,
    input  logic [DATA_WIDTH/8-1:0] s_wstrb,
    input  logic                   s_wvalid,
    output logic                   s_wready,

    // AXI4-Lite slave write response channel
    output logic [1:0]             s_bresp,
    output logic                   s_bvalid,
    input  logic                   s_bready,

    // AXI4-Lite slave read address channel (no-op, returns 0)
    input  logic [ADDR_WIDTH-1:0]  s_araddr,
    input  logic                   s_arvalid,
    output logic                   s_arready,

    // AXI4-Lite slave read data channel
    output logic [DATA_WIDTH-1:0]  s_rdata,
    output logic [1:0]             s_rresp,
    output logic                   s_rvalid,
    input  logic                   s_rready,

    // Generic weight write fan-out: one-cycle pulse + addr/data.
    output logic                   wr_en,
    output logic [ADDR_WIDTH-1:0]  wr_addr,
    output logic [DATA_WIDTH-1:0]  wr_data
);

    // -------------------------------------------------------------------------
    // Write address & data capture. We assert AWREADY and WREADY together
    // when both AWVALID and WVALID are high (paired-accept policy).
    // -------------------------------------------------------------------------
    logic                   aw_taken;
    logic [ADDR_WIDTH-1:0]  awaddr_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            aw_taken  <= 1'b0;
            awaddr_q  <= '0;
            s_awready <= 1'b0;
            s_wready  <= 1'b0;
            s_bvalid  <= 1'b0;
            s_bresp   <= 2'b00;
            wr_en     <= 1'b0;
            wr_addr   <= '0;
            wr_data   <= '0;
        end else begin
            wr_en   <= 1'b0;     // default: only pulses for one cycle

            // Combined AW + W handshake.
            s_awready <= !aw_taken && !s_bvalid;
            s_wready  <= !aw_taken && !s_bvalid;

            if (s_awvalid && s_wvalid && !aw_taken && !s_bvalid) begin
                aw_taken  <= 1'b1;
                awaddr_q  <= s_awaddr;
                s_awready <= 1'b0;
                s_wready  <= 1'b0;
                // Pulse the generic write interface.
                wr_en   <= 1'b1;
                wr_addr <= s_awaddr;
                wr_data <= s_wdata;
                // Drive the response.
                s_bvalid <= 1'b1;
                s_bresp  <= 2'b00;
            end

            // BVALID/BREADY handshake.
            if (s_bvalid && s_bready) begin
                s_bvalid <= 1'b0;
                aw_taken <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Read channel — no registers, all zero.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            s_arready <= 1'b0;
            s_rvalid  <= 1'b0;
            s_rdata   <= '0;
            s_rresp   <= 2'b00;
        end else begin
            // Drop into a simple two-state read: accept AR, raise RVALID, drop on RREADY.
            if (!s_rvalid && s_arvalid) begin
                s_arready <= 1'b1;
                s_rvalid  <= 1'b1;
                s_rdata   <= '0;
                s_rresp   <= 2'b00;
            end else if (s_rvalid && s_rready) begin
                s_arready <= 1'b0;
                s_rvalid  <= 1'b0;
            end else begin
                s_arready <= 1'b0;
            end
        end
    end

endmodule
