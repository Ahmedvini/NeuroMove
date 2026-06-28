// =============================================================================
// serial_divider_tb.sv  —  Sanity TB for rtl/util/serial_divider.sv.
// Runs a handful of pinned cases and ensures floor division matches.
// =============================================================================

`timescale 1ns / 1ps

module serial_divider_tb;

    localparam int N_WIDTH = 32;
    localparam int D_WIDTH = 16;
    localparam int Q_WIDTH = N_WIDTH;
    localparam int CLK_PERIOD = 10;

    logic clk = 0;
    initial forever #(CLK_PERIOD/2) clk = ~clk;
    logic rst;

    logic                  start;
    logic [N_WIDTH-1:0]    numerator;
    logic [D_WIDTH-1:0]    denominator;
    logic                  busy;
    logic                  done;
    logic [Q_WIDTH-1:0]    quotient;
    logic [D_WIDTH-1:0]    remainder;

    serial_divider #(.N_WIDTH(N_WIDTH), .D_WIDTH(D_WIDTH), .Q_WIDTH(Q_WIDTH)) dut (
        .clk(clk), .rst(rst),
        .start(start), .numerator(numerator), .denominator(denominator),
        .busy(busy), .done(done),
        .quotient(quotient), .remainder(remainder)
    );

    int mismatches;

    task automatic check_div(input int unsigned num, input int unsigned den,
                             input int unsigned exp_q, input int unsigned exp_r);
        @(posedge clk);
        start       <= 1'b1;
        numerator   <= num;
        denominator <= den;
        @(posedge clk);
        start       <= 1'b0;
        wait (done == 1'b1);
        @(posedge clk);
        if (quotient !== exp_q || remainder !== exp_r) begin
            $display("[divtb] FAIL: %0d/%0d -> got q=%0d r=%0d, exp q=%0d r=%0d",
                     num, den, quotient, remainder, exp_q, exp_r);
            mismatches++;
        end else begin
            $display("[divtb] OK  : %0d/%0d -> q=%0d r=%0d",
                     num, den, quotient, remainder);
        end
    endtask

    initial begin
        rst = 1'b1;
        start = 1'b0;
        numerator = '0;
        denominator = '0;
        mismatches = 0;
        repeat (4) @(posedge clk);
        rst = 1'b0;

        check_div(    16777216,        564, 29746,         472);
        check_div(    16777216,        644, 26051,         372);
        check_div(    16777216,         32, 524288,         0);
        check_div(           0,         5,      0,          0);
        check_div(    16777215,         1, 16777215,        0);
        check_div(     1000000,      1000,    1000,         0);
        check_div(     1000001,      1000,    1000,         1);
        check_div(  4294967295, 16'hFFFF,  65537,           0);

        if (mismatches == 0) $display("[divtb] PASS");
        else                 $fatal(1, "[divtb] %0d mismatches", mismatches);
        $finish;
    end

endmodule
