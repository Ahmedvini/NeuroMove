/**
 * Module: hmac_demo_top
 *
 * Description:
 *   Minimal FPGA top to demonstrate HMAC-SHA256 on hardware without
 *   exposing wide data buses as I/O pins.
 *
 *   A hardcoded test vector (HMAC-SHA256 of "abc" with key "key") is
 *   run once on reset release. The result is compared against a known-
 *   good golden value. Three LEDs report status.
 *
 * Ports:
 *   clk      - Board clock (assumed 100 MHz; adjust CLKDIV if different)
 *   rst_n    - Active-low reset (push-button)
 *   btn      - Push-button: re-run the test
 *   led_busy - ON while HMAC is computing
 *   led_pass - ON when computed HMAC matches expected golden value
 *   led_fail - ON when computed HMAC does NOT match expected golden value
 */

`timescale 1ns / 1ps

module hmac_demo_top (
    input  logic clk,
    input  logic rst_n,   // active-low reset (board button)
    input  logic btn,     // re-trigger button

    output logic led_busy,
    output logic led_pass,
    output logic led_fail
);

    // -------------------------------------------------------------------------
    // Internal reset (synchronous, active-high)
    // -------------------------------------------------------------------------
    logic rst;
    always_ff @(posedge clk)
        rst <= ~rst_n;

    // -------------------------------------------------------------------------
    // Test vector (HMAC-SHA256 RFC 4231 Test Case 1)
    //   Key  = 0x0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b  (20 bytes)
    //   Data = "Hi There"  (8 bytes)
    //   Expected HMAC = b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7
    // -------------------------------------------------------------------------
    localparam logic [511:0] TV_KEY =
        {{20{8'h0b}}, {44{8'h00}}};  // 20 bytes of 0x0b, zero-padded to 64 bytes

    localparam logic [511:0] TV_MSG =
        {8'h48, 8'h69, 8'h20, 8'h54, 8'h68, 8'h65, 8'h72, 8'h65,  // "Hi There"
         {504/8{8'h00}}};                                            // zero-pad

    localparam logic [63:0] TV_MSG_LEN = 64'd8; // 8 bytes

    localparam logic [255:0] TV_EXPECTED =
        256'hb0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7;

    // -------------------------------------------------------------------------
    // Button edge-detect + auto-start on reset release
    // -------------------------------------------------------------------------
    logic btn_sync0, btn_sync1, btn_prev;
    logic btn_rise;

    always_ff @(posedge clk) begin
        btn_sync0 <= btn;
        btn_sync1 <= btn_sync0;
        btn_prev  <= btn_sync1;
    end
    assign btn_rise = btn_sync1 & ~btn_prev;

    // -------------------------------------------------------------------------
    // HMAC signals
    // -------------------------------------------------------------------------
    logic        start;
    logic [255:0] hmac_out;
    logic         hmac_valid;
    logic         busy;

    // Auto-start: pulse start for one cycle after rst de-assertion
    logic started;
    always_ff @(posedge clk) begin
        if (rst) begin
            started <= 1'b0;
            start   <= 1'b0;
        end else begin
            start <= 1'b0;
            if (!started) begin
                start   <= 1'b1;
                started <= 1'b1;
            end else if (btn_rise) begin
                start   <= 1'b1;  // re-run on button press
            end
        end
    end

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    hmac_sha256 dut (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .key          (TV_KEY),
        .key_len      (8'd20),
        .key_is_hashed(1'b0),
        .msg_data     (TV_MSG),
        .msg_len      (TV_MSG_LEN),
        .msg_valid    (1'b1),
        .msg_last     (1'b1),
        .hmac_out     (hmac_out),
        .hmac_valid   (hmac_valid),
        .busy         (busy)
    );

    // -------------------------------------------------------------------------
    // LED output
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            led_busy <= 1'b0;
            led_pass <= 1'b0;
            led_fail <= 1'b0;
        end else begin
            led_busy <= busy;
            if (hmac_valid) begin
                led_pass <= (hmac_out == TV_EXPECTED);
                led_fail <= (hmac_out != TV_EXPECTED);
            end
        end
    end

endmodule
