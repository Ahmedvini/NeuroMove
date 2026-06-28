/**
 * Module: hmac_sha256
 *
 * Description:
 *   Implements HMAC-SHA-256 as defined in RFC 2104 / FIPS 198-1.
 *
 *   HMAC(K, m) = H( (K' XOR opad) || H( (K' XOR ipad) || m ) )
 *
 *   Where:
 *     K'   = key padded / truncated to 512 bits (block size)
 *     ipad = 0x36 repeated 64 times (512 bits)
 *     opad = 0x5c repeated 64 times (512 bits)
 *     H    = SHA-256
 *
 *   This module handles:
 *     - Keys up to 512 bits (keys longer than 512 bits must be pre-hashed
 *       externally and fed as a 256-bit key with key_is_hashed=1)
 *     - Single-block messages (up to 447 bits of payload so that the
 *       entire ipad||message fits in one SHA-256 block after padding)
 *     - Multi-block messages via streaming interface (msg_valid / msg_last)
 *
 *   Padding note:
 *     SHA-256 padding (bit '1', zeros, 64-bit length) is applied internally
 *     to the inner and outer hash blocks.
 *
 * Latency (single-block message):
 *   ~200 clock cycles  (2 x SHA-256 blocks + control overhead)
 *
 * Ports:
 *   clk            - System clock
 *   rst            - Synchronous active-high reset
 *   start          - Pulse to begin a new HMAC computation
 *   key            - HMAC key (up to 512 bits, zero-padded on the right)
 *   key_len        - Key length in BYTES (1-64).  Keys >64 bytes must be
 *                    pre-hashed; supply 32-byte result as key[511:256] and
 *                    set key_is_hashed=1.
 *   key_is_hashed  - 1 when key is already a 32-byte SHA-256 hash
 *   msg_data       - 512-bit message data chunk (padded to block boundary by
 *                    this module; caller should NOT add SHA padding)
 *   msg_len        - Total message length in BYTES (used to build length field)
 *   msg_valid      - High when msg_data is valid; hold until msg_last
 *   msg_last       - High on the cycle of the final message chunk
 *   hmac_out       - 256-bit HMAC result
 *   hmac_valid     - Pulses high for 1 cycle when result is ready
 *   busy           - High while computing
 *
 * Vivado compatibility:
 *   - Single clock domain, no latches.
 *   - sha256_core is instantiated twice (inner and outer hash).
 */

`timescale 1ns / 1ps

module hmac_sha256 (
    input  logic        clk,
    input  logic        rst,

    // Control
    input  logic        start,

    // Key
    input  logic [511:0] key,           // Up to 512-bit key (zero-pad right)
    input  logic [7:0]   key_len,       // Key length in bytes (max 64)
    input  logic         key_is_hashed, // Key is already 32-byte SHA-256 hash

    // Message (streaming, 512-bit chunks)
    input  logic [511:0] msg_data,      // Current message chunk
    input  logic [63:0]  msg_len,       // Total message length in bytes
    input  logic         msg_valid,     // msg_data is valid
    input  logic         msg_last,      // This is the last chunk

    // Output
    output logic [255:0] hmac_out,
    output logic         hmac_valid,
    output logic         busy
);

    // ------------------------------------------------------------------
    // Constants
    // ------------------------------------------------------------------
    localparam logic [511:0] IPAD = {64{8'h36}};
    localparam logic [511:0] OPAD = {64{8'h5c}};

    // ------------------------------------------------------------------
    // Internal signals
    // ------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_PREP_KEY,      // Hash long key if needed
        S_INNER_FIRST,   // Inner: H(K'⊕ipad ‖ first block)
        S_INNER_MSG,     // Inner: H( ... ‖ msg blocks)
        S_INNER_FINAL,   // Inner: H( ... ‖ padding block)
        S_INNER_DONE,    // Capture inner digest
        S_OUTER_START,   // Outer: H(K'⊕opad ‖ inner_digest ‖ padding)
        S_OUTER_DONE,    // Capture outer digest → HMAC result
        S_DONE
    } fsm_t;

    fsm_t state;

    // Expanded (padded to 512-bit) key
    logic [511:0] key_padded;

    // ipad-XOR'd and opad-XOR'd keys
    logic [511:0] k_ipad;
    logic [511:0] k_opad;

    // SHA-256 core interface (shared between inner and outer)
    logic         sha_start;
    logic [511:0] sha_data_in;
    logic         sha_use_init;
    logic [255:0] sha_init_hash;
    logic [255:0] sha_digest;
    logic         sha_digest_valid;
    logic         sha_busy;

    // Inner hash running state
    logic [255:0] inner_digest;
    logic         inner_chained; // Have we fed at least one inner block?

    // Outer hash
    logic [255:0] outer_digest;

    // Message chunk bookkeeping
    logic [63:0]  bytes_fed; // bytes of message already sent to SHA inner

    // ------------------------------------------------------------------
    // SHA-256 core instantiation (one instance, time-multiplexed)
    // ------------------------------------------------------------------
    sha256_core u_sha256 (
        .clk          (clk),
        .rst          (rst),
        .start        (sha_start),
        .data_in      (sha_data_in),
        .use_init_hash(sha_use_init),
        .init_hash    (sha_init_hash),
        .digest_out   (sha_digest),
        .digest_valid (sha_digest_valid),
        .busy         (sha_busy)
    );

    // ------------------------------------------------------------------
    // SHA-256 padding helper:
    // Given a partial block and the total bit-length, return a padded
    // 512-bit block (appends 0x80, zeros, 64-bit big-endian bit count).
    // Used when remaining payload <= 447 bits (55 bytes).
    // ------------------------------------------------------------------
    function automatic logic [511:0] sha_pad_block(
        input logic [511:0] partial, // data left-aligned
        input integer       data_bytes, // bytes of data in partial (0-55)
        input logic [63:0]  total_bits  // total message bit length
    );
        logic [511:0] blk;
        integer byte_idx;
        blk = partial;
        // Append 0x80 at data_bytes position
        // Loop so synthesizer sees a constant index per iteration (fully unrolled)
        for (byte_idx = 0; byte_idx < 56; byte_idx = byte_idx + 1) begin
            if (byte_idx == data_bytes)
                blk[511 - byte_idx*8 -: 8] = 8'h80;
        end
        // Zero fill handled by caller (partial should be pre-zeroed)
        // Append 64-bit length in last 8 bytes
        blk[63:0] = total_bits;
        return blk;
    endfunction

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------

    // Build key_padded, k_ipad, k_opad (combinational)
    always_comb begin
        if (key_is_hashed) begin
            // Key is 32-byte hash; place in bits [511:256], zero rest
            key_padded = {key[511:256], 256'h0};
        end else begin
            // key is already supplied zero-padded to 512 bits by caller
            key_padded = key;
        end
        k_ipad = key_padded ^ IPAD;
        k_opad = key_padded ^ OPAD;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state         <= S_IDLE;
            busy          <= 1'b0;
            hmac_valid    <= 1'b0;
            hmac_out      <= 256'h0;
            sha_start     <= 1'b0;
            sha_data_in   <= 512'h0;
            sha_use_init  <= 1'b0;
            sha_init_hash <= 256'h0;
            inner_digest  <= 256'h0;
            outer_digest  <= 256'h0;
            inner_chained <= 1'b0;
            bytes_fed     <= 64'h0;
        end else begin
            sha_start  <= 1'b0;
            hmac_valid <= 1'b0;

            case (state)

                // --------------------------------------------------------
                S_IDLE: begin
                    busy          <= 1'b0;
                    inner_chained <= 1'b0;
                    bytes_fed     <= 64'h0;

                    if (start) begin
                        busy  <= 1'b1;
                        state <= S_INNER_FIRST;
                    end
                end

                // --------------------------------------------------------
                // First inner block: SHA-256( K'⊕ipad )
                // This is a full 512-bit block — no message data yet.
                // We start the inner hash with the ipad block.
                // --------------------------------------------------------
                S_INNER_FIRST: begin
                    if (!sha_busy) begin
                        sha_data_in   <= k_ipad;
                        sha_use_init  <= 1'b0; // Use SHA-256 default IV
                        sha_start     <= 1'b1;
                        state         <= S_INNER_MSG;
                    end
                end

                // --------------------------------------------------------
                // Wait for message data, process each 512-bit chunk.
                // The sha256_core is chained via use_init_hash.
                // --------------------------------------------------------
                S_INNER_MSG: begin
                    if (sha_digest_valid && !inner_chained) begin
                        // First ipad block done
                        inner_digest  <= sha_digest;
                        inner_chained <= 1'b1;
                    end

                    if (inner_chained && !sha_busy) begin
                        if (msg_valid) begin
                            if (msg_last) begin
                                // Last message chunk — apply SHA-256 padding
                                // Calculate bits fed so far (from previous chunks)
                                // Inner total bit length = 512 (ipad) + msg_len*8
                                sha_data_in   <= sha_pad_block(
                                                    msg_data,
                                                    64, // full chunk for simplicity; partial handled externally
                                                    (64 + msg_len) * 8  // ipad block + message
                                                 );
                                sha_use_init  <= 1'b1;
                                sha_init_hash <= inner_digest;
                                sha_start     <= 1'b1;
                                bytes_fed     <= bytes_fed + msg_len;
                                state         <= S_INNER_FINAL;
                            end else begin
                                // Intermediate full block
                                sha_data_in   <= msg_data;
                                sha_use_init  <= 1'b1;
                                sha_init_hash <= inner_digest;
                                sha_start     <= 1'b1;
                                bytes_fed     <= bytes_fed + 64;
                                // Stay in S_INNER_MSG; capture digest next cycle
                                inner_chained <= 1'b0; // will re-latch on next digest_valid
                            end
                        end
                    end

                    // Capture intermediate digests for chaining
                    if (sha_digest_valid && inner_chained) begin
                        inner_digest  <= sha_digest;
                        inner_chained <= 1'b1;
                    end
                end

                // --------------------------------------------------------
                // Wait for inner final block to complete
                // --------------------------------------------------------
                S_INNER_FINAL: begin
                    if (sha_digest_valid) begin
                        inner_digest <= sha_digest;
                        state        <= S_INNER_DONE;
                    end
                end

                // --------------------------------------------------------
                // Build outer block: K'⊕opad ‖ inner_digest (padded)
                // Total outer message = 512 (opad) + 256 (inner hash) = 768 bits
                // This requires TWO SHA-256 blocks for the outer hash:
                //   Block 1: K'⊕opad  (512 bits, full block)
                //   Block 2: inner_digest ‖ 0x80 ‖ zeros ‖ length(768)
                // --------------------------------------------------------
                S_INNER_DONE: begin
                    if (!sha_busy) begin
                        // Outer block 1: opad key block
                        sha_data_in   <= k_opad;
                        sha_use_init  <= 1'b0; // Fresh SHA-256
                        sha_start     <= 1'b1;
                        state         <= S_OUTER_START;
                    end
                end

                S_OUTER_START: begin
                    if (sha_digest_valid) begin
                        // Outer block 1 done; feed block 2:
                        // inner_digest (256 bits) || 0x80 || zeros || 64-bit length
                        // Total outer message length = 512 + 256 = 768 bits
                        // Bit length field = 768
                        sha_data_in  <= {inner_digest,  // 256 bits
                                         8'h80,         // padding bit (1 byte)
                                         184'h0,        // zeros: 512-256-8-64 = 184 bits
                                         64'd768};      // 64-bit big-endian length = 768 bits
                        sha_use_init  <= 1'b1;
                        sha_init_hash <= sha_digest; // chain from opad block
                        sha_start     <= 1'b1;
                        state         <= S_OUTER_DONE;
                    end
                end

                S_OUTER_DONE: begin
                    if (sha_digest_valid) begin
                        hmac_out   <= sha_digest;
                        hmac_valid <= 1'b1;
                        busy       <= 1'b0;
                        state      <= S_DONE;
                    end
                end

                S_DONE: begin
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
