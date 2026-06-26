// =============================================================================
// db_atcnet_axi.sv  —  Deployment wrapper around db_atcnet_top for Zynq /
// Zynq Ultrascale+.
//
// PL exposure:
//   * AXI4-Stream slave (`s_axis_*`) — PS streams raw 19-channel EEG samples
//     one (sample, electrode) per beat. Beat layout:
//       tdata[15:0]  = signed Q8.8 sample
//       tdata[20:16] = electrode index in [0, 18]
//       tdata[31:21] = reserved (0)
//       tlast        = asserted on the LAST beat of an inference (after the
//                      pad_bot zero samples). Internally we ignore tlast and
//                      rely on done coming from db_atcnet_top, but PS can use
//                      tlast for framing.
//   * AXI4-Lite slave (`s_axi_*`) — used for two purposes:
//       (a) Weight broadcast writes (low half of address space →
//           addressed weight banks self-decode their slots).
//       (b) Status/class mailbox at the high addresses:
//             0xF000 = status   (R/O):  {30'b0, done, busy}
//             0xF004 = class    (R/O):  {31'b0, class_bit}
//
// Build flow:
//   PS boot:
//     1. Reset PL.
//     2. AXI-Lite-write all weight banks + Gumbel LUT.
//     3. Stream raw EEG via AXI-Stream.
//     4. Poll status until done=1, read class.
//     5. Issue rst (or wait one cycle and continue) for the next inference.
//
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_axi #(
    parameter int DATA_WIDTH    = 16,
    parameter int NUM_RAW       = 19,
    parameter int NUM_SEL       = 5,
    parameter int AXIS_TDATA_W  = 32,
    parameter int AXI_DATA_W    = 32,
    parameter int AXI_ADDR_W    = 32,
    // AXI-Lite address map
    parameter int STATUS_REG_ADDR = 32'h0000_F000,
    parameter int CLASS_REG_ADDR  = 32'h0000_F004,
    parameter int GUMBEL_LUT_BASE = 32'h0000_2000,
    // db_atcnet_top forwarded parameters (all $readmemh file paths can stay
    // empty if PS is going to AXI-Lite-load every bank; for sim we keep them
    // populated to mirror the existing db_atcnet_top tb).
    parameter string CONV2D_W_FILE        = "",
    parameter string CONV2D_B_FILE        = "",
    parameter string ECA1_W_FILE          = "",
    parameter string SIG_LUT_FILE         = "",
    parameter string ELU_LUT_FILE         = "",
    parameter string BRANCHA_DWISE_W_FILE = "",
    parameter string BRANCHA_DWISE_B_FILE = "",
    parameter string BRANCHA_SEP_W_FILE   = "",
    parameter string BRANCHA_SEP_B_FILE   = "",
    parameter string BRANCHB_DWISE_W_FILE = "",
    parameter string BRANCHB_DWISE_B_FILE = "",
    parameter string BRANCHB_SEP_W_FILE   = "",
    parameter string BRANCHB_SEP_B_FILE   = "",
    parameter string ECA2_W_FILE          = "",
    parameter string CHAN_D1_W_FILE       = "",
    parameter string CHAN_D1_B_FILE       = "",
    parameter string CHAN_D2_W_FILE       = "",
    parameter string CHAN_D2_B_FILE       = "",
    parameter string SPAT_CONV_W_FILE     = "",
    parameter string TCFN_W0_FILE         = "",
    parameter string TCFN_B0_FILE         = "",
    parameter string TCFN_W1_FILE         = "",
    parameter string TCFN_B1_FILE         = "",
    parameter string TCFN_W2_FILE         = "",
    parameter string TCFN_B2_FILE         = "",
    parameter string TCFN_W3_FILE         = "",
    parameter string TCFN_B3_FILE         = "",
    parameter string CLS_W_FILE           = "",
    parameter string CLS_B_FILE           = ""
) (
    input  logic                      clk,
    input  logic                      rst,    // active high, sync

    // AXI4-Stream slave (raw EEG)
    input  logic [AXIS_TDATA_W-1:0]   s_axis_tdata,
    input  logic                      s_axis_tvalid,
    output logic                      s_axis_tready,
    input  logic                      s_axis_tlast,

    // AXI4-Lite slave (control + weights)
    input  logic [AXI_ADDR_W-1:0]     s_axi_awaddr,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,
    input  logic [AXI_DATA_W-1:0]     s_axi_wdata,
    input  logic [AXI_DATA_W/8-1:0]   s_axi_wstrb,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,
    input  logic [AXI_ADDR_W-1:0]     s_axi_araddr,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,
    output logic [AXI_DATA_W-1:0]     s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    // Interrupt output: pulses when an inference completes.
    output logic                      irq_done
);

    // -------------------------------------------------------------------------
    // AXI-Lite weight loader — drives the broadcast (wr_en, wr_addr, wr_data)
    // that all weight banks + the Gumbel LUT snoop.
    // -------------------------------------------------------------------------
    logic                       wr_en;
    logic [AXI_ADDR_W-1:0]      wr_addr;
    logic [AXI_DATA_W-1:0]      wr_data_bus;

    // Internal AXI-Lite signals (we replicate the loader logic here so we
    // can also intercept reads at the status/class regs).
    logic                       loader_awready, loader_wready, loader_bvalid;
    logic [1:0]                 loader_bresp;
    logic                       loader_arready, loader_rvalid;
    logic [AXI_DATA_W-1:0]      loader_rdata;
    logic [1:0]                 loader_rresp;

    axi_lite_weight_loader #(
        .ADDR_WIDTH(AXI_ADDR_W), .DATA_WIDTH(AXI_DATA_W)
    ) u_loader (
        .clk(clk), .rst(rst),
        .s_awaddr(s_axi_awaddr), .s_awvalid(s_axi_awvalid), .s_awready(loader_awready),
        .s_wdata(s_axi_wdata),   .s_wstrb(s_axi_wstrb), .s_wvalid(s_axi_wvalid), .s_wready(loader_wready),
        .s_bresp(loader_bresp),  .s_bvalid(loader_bvalid), .s_bready(s_axi_bready),
        // Reads handled below; we feed the loader's read channel with zeroes
        // and never assert s_arvalid into it (the address decoder picks).
        .s_araddr('0), .s_arvalid(1'b0), .s_arready(loader_arready),
        .s_rdata(loader_rdata),  .s_rresp(loader_rresp), .s_rvalid(loader_rvalid),
        .s_rready(1'b1),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data_bus)
    );

    assign s_axi_awready = loader_awready;
    assign s_axi_wready  = loader_wready;
    assign s_axi_bvalid  = loader_bvalid;
    assign s_axi_bresp   = loader_bresp;

    // -------------------------------------------------------------------------
    // AXI-Lite READ channel: intercept STATUS_REG_ADDR and CLASS_REG_ADDR.
    // -------------------------------------------------------------------------
    logic                  busy_reg;
    logic                  done_reg;
    logic [0:0]            class_reg;

    always_ff @(posedge clk) begin
        if (rst) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= '0;
            s_axi_rresp   <= 2'b00;
        end else begin
            if (!s_axi_rvalid && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                case (s_axi_araddr)
                    STATUS_REG_ADDR: s_axi_rdata <= {30'b0, done_reg, busy_reg};
                    CLASS_REG_ADDR:  s_axi_rdata <= {31'b0, class_reg};
                    default:         s_axi_rdata <= '0;
                endcase
                s_axi_rresp <= 2'b00;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_arready <= 1'b0;
                s_axi_rvalid  <= 1'b0;
            end else begin
                s_axi_arready <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Gumbel 19 → 5 channel adapter (AXIS-driven).
    // -------------------------------------------------------------------------
    logic                                  ad_in_valid;
    logic                                  ad_in_ready;
    logic [$clog2(NUM_RAW)-1:0]            ad_in_electrode;
    logic signed [DATA_WIDTH-1:0]          ad_in_sample;
    logic                                  ad_out_valid;
    logic [$clog2(NUM_SEL)-1:0]            ad_out_electrode;
    logic signed [DATA_WIDTH-1:0]          ad_out_sample;

    gumbel_channel_adapter #(
        .DATA_WIDTH(DATA_WIDTH), .NUM_RAW(NUM_RAW), .NUM_SEL(NUM_SEL),
        .AXI_ADDR_WIDTH(AXI_ADDR_W), .AXI_DATA_WIDTH(AXI_DATA_W),
        .LUT_BASE_ADDR(GUMBEL_LUT_BASE)
    ) u_gca (
        .clk(clk), .rst(rst),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data_bus),
        .in_valid(ad_in_valid), .in_ready(ad_in_ready),
        .in_electrode(ad_in_electrode), .in_sample(ad_in_sample),
        .out_valid(ad_out_valid),
        .out_electrode(ad_out_electrode), .out_sample(ad_out_sample)
    );

    // AXIS tdata unpack: electrode in [20:16], sample in [15:0].
    assign ad_in_valid     = s_axis_tvalid;
    assign ad_in_electrode = s_axis_tdata[16 + $clog2(NUM_RAW) - 1 : 16];
    assign ad_in_sample    = s_axis_tdata[DATA_WIDTH-1:0];
    assign s_axis_tready   = ad_in_ready;

    // -------------------------------------------------------------------------
    // db_atcnet_top — full inference pipeline.
    // -------------------------------------------------------------------------
    logic [0:0] top_class_out;
    logic       top_done;

    db_atcnet_top #(
        .DATA_WIDTH(DATA_WIDTH), .C_IN(NUM_SEL), .F1(16), .F(32),
        .KE_CONV2D(64), .KE_SEP(16),
        .T_IN(600), .POOL1(8), .POOL2(7), .T_FULL(10), .T_WIN(6),
        .N_WIN(5), .N_CLS(2),
        .ECA1_INV_N(5592), .ECA2_INV_N(1677722), .CHAN_INV_N(2796203),
        .INV_POOL1(2097152), .INV_POOL2(2396745),
        .SIG_LUT_N(1024), .SIG_LUT_RANGE_Q(1024), .SIG_LUT_SHIFT(1),
        .ELU_LUT_N(256), .ELU_LUT_RANGE_Q(2048), .ELU_LUT_SHIFT(3),
        .CONV2D_W_FILE(CONV2D_W_FILE), .CONV2D_B_FILE(CONV2D_B_FILE),
        .ECA1_W_FILE  (ECA1_W_FILE),
        .SIG_LUT_FILE (SIG_LUT_FILE), .ELU_LUT_FILE (ELU_LUT_FILE),
        .BRANCHA_DWISE_W_FILE(BRANCHA_DWISE_W_FILE), .BRANCHA_DWISE_B_FILE(BRANCHA_DWISE_B_FILE),
        .BRANCHA_SEP_W_FILE  (BRANCHA_SEP_W_FILE),   .BRANCHA_SEP_B_FILE  (BRANCHA_SEP_B_FILE),
        .BRANCHB_DWISE_W_FILE(BRANCHB_DWISE_W_FILE), .BRANCHB_DWISE_B_FILE(BRANCHB_DWISE_B_FILE),
        .BRANCHB_SEP_W_FILE  (BRANCHB_SEP_W_FILE),   .BRANCHB_SEP_B_FILE  (BRANCHB_SEP_B_FILE),
        .ECA2_W_FILE         (ECA2_W_FILE),
        .CHAN_D1_W_FILE(CHAN_D1_W_FILE), .CHAN_D1_B_FILE(CHAN_D1_B_FILE),
        .CHAN_D2_W_FILE(CHAN_D2_W_FILE), .CHAN_D2_B_FILE(CHAN_D2_B_FILE),
        .SPAT_CONV_W_FILE(SPAT_CONV_W_FILE),
        .TCFN_W0_FILE(TCFN_W0_FILE), .TCFN_B0_FILE(TCFN_B0_FILE),
        .TCFN_W1_FILE(TCFN_W1_FILE), .TCFN_B1_FILE(TCFN_B1_FILE),
        .TCFN_W2_FILE(TCFN_W2_FILE), .TCFN_B2_FILE(TCFN_B2_FILE),
        .TCFN_W3_FILE(TCFN_W3_FILE), .TCFN_B3_FILE(TCFN_B3_FILE),
        .CLS_W_FILE(CLS_W_FILE), .CLS_B_FILE(CLS_B_FILE)
    ) u_top (
        .clk(clk), .rst(rst),
        .in_valid(ad_out_valid),
        .in_electrode(ad_out_electrode), .in_sample(ad_out_sample),
        .class_out(top_class_out), .done(top_done)
    );

    // -------------------------------------------------------------------------
    // Status / class mailbox register state.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            busy_reg  <= 1'b0;
            done_reg  <= 1'b0;
            class_reg <= '0;
            irq_done  <= 1'b0;
        end else begin
            irq_done <= 1'b0;
            if (s_axis_tvalid && s_axis_tready) busy_reg <= 1'b1;
            if (top_done) begin
                busy_reg  <= 1'b0;
                done_reg  <= 1'b1;
                class_reg <= top_class_out;
                irq_done  <= 1'b1;
            end
            // Clear done when PS reads the class register.
            if (s_axi_rvalid && s_axi_rready && s_axi_araddr == CLASS_REG_ADDR) begin
                done_reg <= 1'b0;
            end
        end
    end

endmodule
