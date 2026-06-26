// =============================================================================
// db_atcnet_axi_v.v — Plain Verilog wrapper around the SystemVerilog
// db_atcnet_axi module so it can be added to a Vivado IPI block design as a
// Module Reference (IPI Module Reference rejects .sv files as the top).
//
// All ports pass through 1:1. The 29 *_FILE parameters that point at the
// trained weight/LUT .hex files are HARDCODED here with absolute paths so
// that the inner SV modules get the right values even when this wrapper is
// instantiated from a BD that can't pass synth_design -generic flags through.
// =============================================================================

`timescale 1ns / 1ps

module db_atcnet_axi_v #(
    parameter AXIS_TDATA_W = 32,
    parameter AXI_ADDR_W   = 32,
    parameter AXI_DATA_W   = 32
) (
    input  wire                       clk,
    input  wire                       rst,

    input  wire [AXIS_TDATA_W-1:0]    s_axis_tdata,
    input  wire                       s_axis_tvalid,
    output wire                       s_axis_tready,
    input  wire                       s_axis_tlast,

    input  wire [AXI_ADDR_W-1:0]      s_axi_awaddr,
    input  wire                       s_axi_awvalid,
    output wire                       s_axi_awready,
    input  wire [AXI_DATA_W-1:0]      s_axi_wdata,
    input  wire [AXI_DATA_W/8-1:0]    s_axi_wstrb,
    input  wire                       s_axi_wvalid,
    output wire                       s_axi_wready,
    output wire [1:0]                 s_axi_bresp,
    output wire                       s_axi_bvalid,
    input  wire                       s_axi_bready,
    input  wire [AXI_ADDR_W-1:0]      s_axi_araddr,
    input  wire                       s_axi_arvalid,
    output wire                       s_axi_arready,
    output wire [AXI_DATA_W-1:0]      s_axi_rdata,
    output wire [1:0]                 s_axi_rresp,
    output wire                       s_axi_rvalid,
    input  wire                       s_axi_rready,

    output wire                       irq_done
);

    db_atcnet_axi #(
        .AXIS_TDATA_W (AXIS_TDATA_W),
        .AXI_ADDR_W   (AXI_ADDR_W),
        .AXI_DATA_W   (AXI_DATA_W),

        // Trained weight / LUT files — must match data/golden_q88/ and data/lut/.
        // Paths are absolute so they resolve regardless of how Vivado launches
        // the OOC sub-synthesis for this module reference.
        .CONV2D_W_FILE        ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_conv2d_weights.hex"),
        .CONV2D_B_FILE        ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_conv2d_bias.hex"),
        .ECA1_W_FILE          ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_eca1_weights.hex"),
        .SIG_LUT_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/lut/sigmoid_q88.hex"),
        .ELU_LUT_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/lut/elu_q88.hex"),
        .BRANCHA_DWISE_W_FILE ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchA_dwise_weights.hex"),
        .BRANCHA_DWISE_B_FILE ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchA_dwise_bias.hex"),
        .BRANCHA_SEP_W_FILE   ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchA_sep_weights.hex"),
        .BRANCHA_SEP_B_FILE   ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchA_sep_bias.hex"),
        .BRANCHB_DWISE_W_FILE ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchB_dwise_weights.hex"),
        .BRANCHB_DWISE_B_FILE ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchB_dwise_bias.hex"),
        .BRANCHB_SEP_W_FILE   ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchB_sep_weights.hex"),
        .BRANCHB_SEP_B_FILE   ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_branchB_sep_bias.hex"),
        .ECA2_W_FILE          ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/stage_eca2_weights.hex"),
        .CHAN_D1_W_FILE       ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_chan_d1w.hex"),
        .CHAN_D1_B_FILE       ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_chan_d1b.hex"),
        .CHAN_D2_W_FILE       ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_chan_d2w.hex"),
        .CHAN_D2_B_FILE       ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_chan_d2b.hex"),
        .SPAT_CONV_W_FILE     ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_spat_conv_w.hex"),
        .TCFN_W0_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_w0.hex"),
        .TCFN_B0_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_b0.hex"),
        .TCFN_W1_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_w1.hex"),
        .TCFN_B1_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_b1.hex"),
        .TCFN_W2_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_w2.hex"),
        .TCFN_B2_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_b2.hex"),
        .TCFN_W3_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_w3.hex"),
        .TCFN_B3_FILE         ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_tcfn_b3.hex"),
        .CLS_W_FILE           ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_cls_w.hex"),
        .CLS_B_FILE           ("/home/ahmedelsheikh/Documents/GitHub/fpga-atcnet-main/data/golden_q88/all_cls_b.hex")
    ) u_inst (
        .clk            (clk),
        .rst            (rst),

        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tlast   (s_axis_tlast),

        .s_axi_awaddr   (s_axi_awaddr),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),
        .s_axi_araddr   (s_axi_araddr),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),

        .irq_done       (irq_done)
    );

endmodule
