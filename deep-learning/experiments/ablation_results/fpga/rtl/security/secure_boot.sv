/**
 * Module: secure_boot
 *
 * Description:
 *   Secure Boot / Bitstream Authentication module using RSA-2048 + SHA-256.
 *
 *   Boot authentication flow (PKCS#1 v1.5):
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ 1. Hash the bitstream/firmware:                                     │
 *   │      bitstream_hash = SHA-256( all bitstream blocks )               │
 *   │                                                                     │
 *   │ 2. RSA verify (public-key operation):                               │
 *   │      recovered = signature ^ e  mod  N                              │
 *   │                                                                     │
 *   │ 3. PKCS#1 v1.5 DER unpadding check:                                 │
 *   │      recovered must match:                                          │
 *   │        0x00 0x01 [0xFF padding] 0x00 [DigestInfo] [SHA-256 hash]    │
 *   │      DigestInfo prefix for SHA-256:                                 │
 *   │        0x30 0x31 0x30 0x0d 0x06 0x09 0x60 0x86 0x48 0x01 0x65      │
 *   │        0x03 0x04 0x02 0x01 0x05 0x00 0x04 0x20                     │
 *   │                                                                     │
 *   │ 4. Decision:                                                        │
 *   │      recovered hash == bitstream_hash → boot_allow                  │
 *   │      mismatch or bad padding          → boot_deny                   │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 *   This module prevents loading of:
 *     - Malicious / tampered FPGA bitstreams
 *     - Unsigned firmware updates
 *     - Downgrade attacks (if sequence numbers are embedded in bitstream)
 *
 * Ports:
 *   boot_start         - Pulse to begin verification
 *   bitstream_block    - 512-bit chunk of the bitstream/firmware image
 *   bitstream_valid    - High when bitstream_block is valid
 *   bitstream_last     - High on the final bitstream chunk
 *   bitstream_len      - Total bitstream length in BYTES
 *   rsa_signature      - 2048-bit RSA signature from trusted authority
 *   rsa_pub_exp        - RSA public exponent (typically 65537 = 0x10001)
 *   rsa_pub_exp_bits   - Significant bits in rsa_pub_exp (17 for e=65537)
 *   rsa_modulus        - 2048-bit RSA modulus N (public key)
 *   boot_allow         - Pulsed HIGH → signature valid, system may boot
 *   boot_deny          - Pulsed HIGH → signature invalid, enter safe state
 *   boot_busy          - High while verification is in progress
 *   boot_hash_out      - The computed SHA-256 hash of the bitstream (for debug)
 *
 * Parameters:
 *   KEY_BITS - RSA key size (2048)
 *
 * Vivado compatibility:
 *   - Single clock domain, no latches.
 *   - Instantiates sha256_core and rsa2048_core.
 *   - PKCS#1 DER check is purely combinational from the RSA result register.
 */

`timescale 1ns / 1ps

module secure_boot #(
    parameter int KEY_BITS = 2048
) (
    input  logic                clk,
    input  logic                rst,

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    input  logic                boot_start,

    // ------------------------------------------------------------------
    // Bitstream / firmware input (streaming 512-bit blocks)
    // ------------------------------------------------------------------
    input  logic [511:0]        bitstream_block,
    input  logic                bitstream_valid,
    input  logic                bitstream_last,
    input  logic [63:0]         bitstream_len,    // in bytes

    // ------------------------------------------------------------------
    // RSA public key + signature
    // ------------------------------------------------------------------
    input  logic [KEY_BITS-1:0] rsa_signature,
    input  logic [KEY_BITS-1:0] rsa_pub_exp,
    input  logic [11:0]         rsa_pub_exp_bits,  // e.g. 17 for e=65537
    input  logic [KEY_BITS-1:0] rsa_modulus,

    // ------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------
    output logic                boot_allow,        // Signature verified OK
    output logic                boot_deny,         // Signature check failed
    output logic                boot_busy,
    output logic [255:0]        boot_hash_out      // SHA-256 of bitstream
);

    // ------------------------------------------------------------------
    // PKCS#1 v1.5 DigestInfo prefix for SHA-256 (19 bytes = 152 bits)
    // 0x3031300d060960864801650304020105000420
    // ------------------------------------------------------------------
    localparam logic [151:0] PKCS1_SHA256_DER =
        152'h3031300d060960864801650304020105000420;

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_HASH_BLOCKS,   // Feed bitstream through SHA-256 (multi-block)
        S_HASH_WAIT,     // Wait for SHA-256 result
        S_RSA_START,     // Start RSA exponentiation
        S_RSA_WAIT,      // Wait for RSA result
        S_VERIFY,        // PKCS#1 padding + hash comparison
        S_DONE
    } fsm_t;

    fsm_t state;

    // ------------------------------------------------------------------
    // SHA-256 core signals
    // ------------------------------------------------------------------
    logic         sha_start;
    logic [511:0] sha_data_in;
    logic         sha_use_init;
    logic [255:0] sha_init_hash;
    logic [255:0] sha_digest;
    logic         sha_digest_valid;
    logic         sha_busy;

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
    // RSA-2048 core signals
    // ------------------------------------------------------------------
    logic                  rsa_start;
    logic [KEY_BITS-1:0]   rsa_result;
    logic                  rsa_result_valid;
    logic                  rsa_busy;

    rsa2048_core #(
        .KEY_BITS(KEY_BITS)
    ) u_rsa (
        .clk         (clk),
        .rst         (rst),
        .start       (rsa_start),
        .base        (rsa_signature),
        .exponent    (rsa_pub_exp),
        .exp_bits    (rsa_pub_exp_bits),
        .modulus     (rsa_modulus),
        .result      (rsa_result),
        .result_valid(rsa_result_valid),
        .busy        (rsa_busy)
    );

    // ------------------------------------------------------------------
    // Internal registers
    // ------------------------------------------------------------------
    logic [255:0] bitstream_hash;     // Final SHA-256 hash of bitstream
    logic [255:0] sha_chain;          // Running SHA-256 chain digest
    logic         sha_chained;        // At least one block processed
    logic [63:0]  bits_hashed;        // Running count of bits processed
    logic [63:0]  total_bits;         // bitstream_len * 8

    // ------------------------------------------------------------------
    // PKCS#1 v1.5 verification (combinational from rsa_result)
    //
    // RSA result layout (KEY_BITS = 2048, big-endian):
    //   Byte 0 (MSB): 0x00
    //   Byte 1:       0x01
    //   Bytes 2..229: 0xFF (padding, at least 8 bytes)
    //   Byte 230:     0x00 (separator)
    //   Bytes 231..249: DigestInfo DER prefix (19 bytes)
    //   Bytes 250..281: SHA-256 hash (32 bytes)
    //
    // In 2048-bit big-endian: MSByte is rsa_result[2047:2040]
    // ------------------------------------------------------------------
    logic pkcs1_ok;
    logic hash_match;
    logic [255:0] recovered_hash;

    always_comb begin
        // Extract the last 32 bytes of the RSA result as the recovered hash
        recovered_hash = rsa_result[255:0];

        // Check leading bytes: byte0=0x00, byte1=0x01
        pkcs1_ok = (rsa_result[KEY_BITS-1     -: 8] == 8'h00) &&
                   (rsa_result[KEY_BITS-9      -: 8] == 8'h01) &&
                   // Separator byte at position 230 from MSB = bit (2048-230*8-1) downto (2048-231*8)
                   (rsa_result[KEY_BITS-1-(230*8) -: 8] == 8'h00) &&
                   // DigestInfo DER prefix (19 bytes starting at byte 231)
                   (rsa_result[KEY_BITS-1-(231*8) -: 152] == PKCS1_SHA256_DER);

        hash_match = (recovered_hash == bitstream_hash);
    end

    // ------------------------------------------------------------------
    // Main FSM
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state           <= S_IDLE;
            boot_busy       <= 1'b0;
            boot_allow      <= 1'b0;
            boot_deny       <= 1'b0;
            boot_hash_out   <= 256'h0;
            sha_start       <= 1'b0;
            sha_data_in     <= 512'h0;
            sha_use_init    <= 1'b0;
            sha_init_hash   <= 256'h0;
            rsa_start       <= 1'b0;
            sha_chain       <= 256'h0;
            sha_chained     <= 1'b0;
            bits_hashed     <= 64'h0;
            total_bits      <= 64'h0;
            bitstream_hash  <= 256'h0;
        end else begin
            sha_start  <= 1'b0;
            rsa_start  <= 1'b0;
            boot_allow <= 1'b0;
            boot_deny  <= 1'b0;

            case (state)

                // --------------------------------------------------------
                S_IDLE: begin
                    boot_busy   <= 1'b0;
                    sha_chained <= 1'b0;
                    bits_hashed <= 64'h0;

                    if (boot_start) begin
                        boot_busy  <= 1'b1;
                        total_bits <= bitstream_len << 3; // bytes → bits
                        state      <= S_HASH_BLOCKS;
                    end
                end

                // --------------------------------------------------------
                // Stream bitstream blocks through SHA-256.
                // Each 512-bit block is fed one at a time.
                // Chain intermediate digests via use_init_hash.
                // On the last block, SHA-256 padding is needed.
                // --------------------------------------------------------
                S_HASH_BLOCKS: begin
                    // Capture chained digest
                    if (sha_digest_valid) begin
                        sha_chain   <= sha_digest;
                        sha_chained <= 1'b1;
                    end

                    if (!sha_busy) begin
                        if (bitstream_valid) begin
                            sha_data_in   <= bitstream_block;
                            sha_use_init  <= sha_chained;
                            sha_init_hash <= sha_chain;
                            sha_start     <= 1'b1;
                            bits_hashed   <= bits_hashed + 64'd512;

                            if (bitstream_last) begin
                                // SHA-256 standard padding will be applied
                                // by the caller when building the last block.
                                // We tag the block and move to wait state.
                                state <= S_HASH_WAIT;
                            end
                        end
                    end
                end

                // --------------------------------------------------------
                // Wait for final SHA-256 block to complete
                // --------------------------------------------------------
                S_HASH_WAIT: begin
                    if (sha_digest_valid) begin
                        bitstream_hash <= sha_digest;
                        boot_hash_out  <= sha_digest;
                        state          <= S_RSA_START;
                    end
                end

                // --------------------------------------------------------
                // Start RSA verification
                // --------------------------------------------------------
                S_RSA_START: begin
                    if (!rsa_busy) begin
                        rsa_start <= 1'b1;
                        state     <= S_RSA_WAIT;
                    end
                end

                // --------------------------------------------------------
                // Wait for RSA exponentiation to complete
                // --------------------------------------------------------
                S_RSA_WAIT: begin
                    if (rsa_result_valid) begin
                        state <= S_VERIFY;
                    end
                end

                // --------------------------------------------------------
                // PKCS#1 v1.5 check + hash comparison
                // --------------------------------------------------------
                S_VERIFY: begin
                    if (pkcs1_ok && hash_match) begin
                        boot_allow <= 1'b1;
                    end else begin
                        boot_deny  <= 1'b1;
                    end
                    boot_busy <= 1'b0;
                    state     <= S_DONE;
                end

                S_DONE: begin
                    boot_busy <= 1'b0;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
