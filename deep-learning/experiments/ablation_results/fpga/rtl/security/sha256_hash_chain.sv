/**
 * Module: sha256_hash_chain
 *
 * Description:
 *   Implements a SHA-256 based tamper-evident hash chain for audit logging.
 *
 *   Each log entry is linked to the previous entry by including the previous
 *   hash in the computation:
 *
 *     hash_0 = genesis_hash               (anchor / seed)
 *     hash_i = SHA-256( hash_{i-1} || log_data_i )
 *
 *   If ANY stored entry is modified, all subsequent hashes become invalid.
 *   Verification computes the expected hash and compares — mismatch = tamper.
 *
 *   Use Cases (ATCNet EEG Project):
 *     - Safety log  : noise events, emergency stops, actuator faults
 *     - Audit log   : encryption operations, key uses, access attempts
 *     - History log : any stored record that must be trusted on playback
 *
 *   Message Construction:
 *     Input message = prev_hash (256 bits) || log_data (256 bits) = 512 bits
 *     SHA-256 of a 512-bit message requires two SHA-256 compression blocks:
 *       Block 1 : the 512-bit message  (data_in = prev_hash || log_data)
 *       Block 2 : SHA-256 padding      (0x80 || zeros[440] || 64'h200)
 *     The sha256_core handles one 512-bit block per invocation; both blocks
 *     are chained via use_init_hash / init_hash.
 *
 * Ports:
 *   log_valid          - Pulse to append a new log entry
 *   log_data           - 256-bit log record (caller packs fields)
 *   genesis_hash       - Chain anchor hash (set before first entry)
 *   init_chain         - Synchronous reset of chain to genesis_hash
 *   chain_hash         - Current head of chain (hash of last entry)
 *   entry_count        - Total entries appended (wraps at 2^16)
 *   hash_valid         - Pulsed for 1 cycle when a new hash is ready
 *   busy               - High while computing
 *   verify_start       - Pulse to verify a stored entry
 *   verify_prev_hash   - Hash BEFORE the entry being verified
 *   verify_log_data    - The stored log entry data to verify
 *   verify_expected    - The expected resulting hash
 *   verify_ok          - Hash matches → chain intact
 *   verify_fail        - Hash mismatch → tampering detected!
 *
 * Latency:  ~134 clock cycles per entry (2 x sha256_core blocks)
 *
 * Vivado compatibility:
 *   - Single clock domain, no latches.
 *   - sha256_core instantiated internally.
 */

`timescale 1ns / 1ps

module sha256_hash_chain (
    input  logic         clk,
    input  logic         rst,

    // ------------------------------------------------------------------
    // Append interface
    // ------------------------------------------------------------------
    input  logic         log_valid,          // Pulse: new entry to append
    input  logic [255:0] log_data,           // 256-bit log record

    // ------------------------------------------------------------------
    // Chain anchor
    // ------------------------------------------------------------------
    input  logic [255:0] genesis_hash,       // Chain start hash
    input  logic         init_chain,         // Sync reset to genesis

    // ------------------------------------------------------------------
    // Chain state output
    // ------------------------------------------------------------------
    output logic [255:0] chain_hash,         // Current head hash
    output logic [15:0]  entry_count,        // Entries appended
    output logic         hash_valid,         // Pulsed when new hash ready
    output logic         busy,

    // ------------------------------------------------------------------
    // Verification interface
    // ------------------------------------------------------------------
    input  logic         verify_start,       // Trigger verification
    input  logic [255:0] verify_prev_hash,   // Hash BEFORE the entry
    input  logic [255:0] verify_log_data,    // Stored log entry to check
    input  logic [255:0] verify_expected,    // Expected resulting hash
    output logic         verify_ok,          // Chain intact
    output logic         verify_fail         // Tampering detected!
);

    // ------------------------------------------------------------------
    // SHA-256 padding for a 512-bit message:
    //   Message length = 512 bits = 0x200
    //   Block 2 = 0x80 || zeros[440] || 64'h0000_0000_0000_0200
    //   8 + 440 + 64 = 512 bits ✓
    // ------------------------------------------------------------------
    localparam logic [511:0] PAD_BLOCK_512 =
        {8'h80, 440'h0, 64'h0000_0000_0000_0200};

    // ------------------------------------------------------------------
    // FSM states
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_HASH1,        // Feed block 1 (prev_hash || log_data) to SHA-256
        S_HASH1_WAIT,   // Wait for block 1 digest
        S_HASH2,        // Feed padding block (chained from block 1)
        S_HASH2_WAIT,   // Wait for final digest
        S_OUTPUT        // Latch result, one cycle
    } fsm_t;

    fsm_t state;

    // ------------------------------------------------------------------
    // Registered capture (hold inputs during multi-cycle computation)
    // ------------------------------------------------------------------
    logic [255:0] prev_hash_r;     // prev chain hash for this operation
    logic [255:0] log_data_r;      // log entry data for this operation
    logic [255:0] expected_r;      // expected hash (verify mode)
    logic         is_verify_r;     // 1 = verify mode, 0 = append mode

    // ------------------------------------------------------------------
    // SHA-256 core interface
    // ------------------------------------------------------------------
    logic         sha_start;
    logic [511:0] sha_data;
    logic         sha_use_init;
    logic [255:0] sha_init;
    logic [255:0] sha_digest;
    logic         sha_digest_valid;
    logic         sha_core_busy;

    logic [255:0] block1_digest;   // Intermediate digest after block 1

    sha256_core u_sha256 (
        .clk          (clk),
        .rst          (rst),
        .start        (sha_start),
        .data_in      (sha_data),
        .use_init_hash(sha_use_init),
        .init_hash    (sha_init),
        .digest_out   (sha_digest),
        .digest_valid (sha_digest_valid),
        .busy         (sha_core_busy)
    );

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            hash_valid     <= 1'b0;
            verify_ok      <= 1'b0;
            verify_fail    <= 1'b0;
            chain_hash     <= 256'h0;
            entry_count    <= 16'h0;
            sha_start      <= 1'b0;
            sha_use_init   <= 1'b0;
            sha_init       <= 256'h0;
            sha_data       <= 512'h0;
            prev_hash_r    <= 256'h0;
            log_data_r     <= 256'h0;
            expected_r     <= 256'h0;
            block1_digest  <= 256'h0;
            is_verify_r    <= 1'b0;
        end else begin
            sha_start   <= 1'b0;    // default: no new block
            hash_valid  <= 1'b0;    // default: no new hash
            verify_ok   <= 1'b0;    // default: no result
            verify_fail <= 1'b0;

            // Synchronous chain reset (takes effect immediately in IDLE)
            if (init_chain) begin
                chain_hash  <= genesis_hash;
                entry_count <= 16'h0;
            end

            case (state)

                // --------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;

                    if (log_valid && !init_chain) begin
                        // Append mode
                        prev_hash_r <= chain_hash;
                        log_data_r  <= log_data;
                        is_verify_r <= 1'b0;
                        busy        <= 1'b1;
                        state       <= S_HASH1;

                    end else if (verify_start && !init_chain) begin
                        // Verify mode
                        prev_hash_r <= verify_prev_hash;
                        log_data_r  <= verify_log_data;
                        expected_r  <= verify_expected;
                        is_verify_r <= 1'b1;
                        busy        <= 1'b1;
                        state       <= S_HASH1;
                    end
                end

                // --------------------------------------------------------
                // Block 1: SHA-256 compression of (prev_hash || log_data)
                // Uses the default SHA-256 IV (not chained)
                // --------------------------------------------------------
                S_HASH1: begin
                    if (!sha_core_busy) begin
                        sha_data     <= {prev_hash_r, log_data_r}; // 256+256=512b
                        sha_use_init <= 1'b0;
                        sha_start    <= 1'b1;
                        state        <= S_HASH1_WAIT;
                    end
                end

                S_HASH1_WAIT: begin
                    if (sha_digest_valid) begin
                        block1_digest <= sha_digest;
                        state         <= S_HASH2;
                    end
                end

                // --------------------------------------------------------
                // Block 2: SHA-256 compression of the padding block,
                // chained from block 1 result.
                // This completes the full SHA-256 of the 512-bit message.
                // --------------------------------------------------------
                S_HASH2: begin
                    if (!sha_core_busy) begin
                        sha_data     <= PAD_BLOCK_512;
                        sha_use_init <= 1'b1;
                        sha_init     <= block1_digest;
                        sha_start    <= 1'b1;
                        state        <= S_HASH2_WAIT;
                    end
                end

                S_HASH2_WAIT: begin
                    if (sha_digest_valid) begin
                        state <= S_OUTPUT;

                        if (is_verify_r) begin
                            // Verification: compare with expected hash
                            if (sha_digest == expected_r)
                                verify_ok   <= 1'b1;
                            else
                                verify_fail <= 1'b1;
                        end else begin
                            // Append: update chain head
                            chain_hash  <= sha_digest;
                            entry_count <= entry_count + 16'h1;
                            hash_valid  <= 1'b1;
                        end
                    end
                end

                // --------------------------------------------------------
                S_OUTPUT: begin
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
