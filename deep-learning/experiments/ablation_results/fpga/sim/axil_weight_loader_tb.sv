// =============================================================================
// axil_weight_loader_tb.sv  —  Sanity TB for axi_lite_weight_loader.sv +
// weight_bank.sv. Drives a few AXI-Lite writes and confirms the values land
// in the bank's mem[] array at the expected indices.
// =============================================================================

`timescale 1ns / 1ps

module axil_weight_loader_tb;

    localparam int DATA_WIDTH = 16;
    localparam int AXI_DATA_WIDTH = 32;
    localparam int AXI_ADDR_WIDTH = 32;
    localparam int DEPTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam logic [AXI_ADDR_WIDTH-1:0] BASE = 32'h0000_1000;

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    // AXI-Lite signals
    logic [AXI_ADDR_WIDTH-1:0]  s_awaddr;
    logic                       s_awvalid;
    logic                       s_awready;
    logic [AXI_DATA_WIDTH-1:0]  s_wdata;
    logic [AXI_DATA_WIDTH/8-1:0] s_wstrb;
    logic                       s_wvalid;
    logic                       s_wready;
    logic [1:0]                 s_bresp;
    logic                       s_bvalid;
    logic                       s_bready;
    logic [AXI_ADDR_WIDTH-1:0]  s_araddr;
    logic                       s_arvalid;
    logic                       s_arready;
    logic [AXI_DATA_WIDTH-1:0]  s_rdata;
    logic [1:0]                 s_rresp;
    logic                       s_rvalid;
    logic                       s_rready;

    // Internal write fan-out
    logic                       wr_en;
    logic [AXI_ADDR_WIDTH-1:0]  wr_addr;
    logic [AXI_DATA_WIDTH-1:0]  wr_data;

    axi_lite_weight_loader #(
        .ADDR_WIDTH(AXI_ADDR_WIDTH), .DATA_WIDTH(AXI_DATA_WIDTH)
    ) u_loader (
        .clk(clk), .rst(rst),
        .s_awaddr(s_awaddr), .s_awvalid(s_awvalid), .s_awready(s_awready),
        .s_wdata (s_wdata),  .s_wstrb(s_wstrb), .s_wvalid(s_wvalid), .s_wready(s_wready),
        .s_bresp(s_bresp), .s_bvalid(s_bvalid), .s_bready(s_bready),
        .s_araddr(s_araddr), .s_arvalid(s_arvalid), .s_arready(s_arready),
        .s_rdata(s_rdata), .s_rresp(s_rresp), .s_rvalid(s_rvalid), .s_rready(s_rready),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data)
    );

    // Weight bank under test
    logic signed [DATA_WIDTH-1:0] bank_mem [0:DEPTH-1];

    weight_bank #(
        .DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH), .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .BASE_ADDR(BASE), .BYTES_PER_ENTRY(4), .PACK_2(1),
        .INIT_FILE("")
    ) u_bank (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .mem(bank_mem)
    );

    // AXI-Lite write task.
    task automatic axil_write(input logic [AXI_ADDR_WIDTH-1:0] addr,
                              input logic [AXI_DATA_WIDTH-1:0] data);
        // Drive AW + W simultaneously.
        @(posedge clk);
        s_awaddr  <= addr;
        s_awvalid <= 1'b1;
        s_wdata   <= data;
        s_wstrb   <= 4'b1111;
        s_wvalid  <= 1'b1;
        s_bready  <= 1'b1;
        // Wait for the slave to take both AW and W (s_awready && s_wready).
        do @(posedge clk); while (!(s_awready && s_wready));
        s_awvalid <= 1'b0;
        s_wvalid  <= 1'b0;
        // Wait for BVALID then drop BREADY.
        do @(posedge clk); while (!s_bvalid);
        @(posedge clk);
        s_bready <= 1'b0;
    endtask

    int mismatches;

    initial begin
        rst = 1'b1;
        s_awaddr = '0; s_awvalid = 1'b0;
        s_wdata  = '0; s_wstrb   = 4'b0; s_wvalid = 1'b0;
        s_bready = 1'b0;
        s_araddr = '0; s_arvalid = 1'b0; s_rready = 1'b0;
        mismatches = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        // Write 0xABCD at index 0, 0x1234 at index 1 → both in 32-bit word at BASE+0
        axil_write(BASE + 32'h0, 32'h1234_ABCD);

        // Write 0xDEAD at index 4, 0xBEEF at index 5 → 32-bit word at BASE+8
        axil_write(BASE + 32'h8, 32'hBEEF_DEAD);

        // Write 0xCAFE at index 14, 0xF00D at index 15 → 32-bit word at BASE+28
        axil_write(BASE + 32'h1C, 32'hF00D_CAFE);

        // Read-modify-check.
        @(posedge clk);

        if (bank_mem[0]  !== 16'hABCD) begin $display("FAIL idx 0  got %h", bank_mem[0]);  mismatches++; end
        if (bank_mem[1]  !== 16'h1234) begin $display("FAIL idx 1  got %h", bank_mem[1]);  mismatches++; end
        if (bank_mem[4]  !== 16'hDEAD) begin $display("FAIL idx 4  got %h", bank_mem[4]);  mismatches++; end
        if (bank_mem[5]  !== 16'hBEEF) begin $display("FAIL idx 5  got %h", bank_mem[5]);  mismatches++; end
        if (bank_mem[14] !== 16'hCAFE) begin $display("FAIL idx 14 got %h", bank_mem[14]); mismatches++; end
        if (bank_mem[15] !== 16'hF00D) begin $display("FAIL idx 15 got %h", bank_mem[15]); mismatches++; end

        if (mismatches != 0) $fatal(1, "[axil_tb] %0d mismatches", mismatches);
        $display("[axil_tb] PASS");
        $finish;
    end

endmodule
