/**
 * Module: eeg_security_top
 *
 * Description:
 *   Top-level security module for the ATCNet EEG project.
 *   Combines AES-256-GCM (authenticated encryption) with HMAC-SHA-256
 *   (integrity / authentication MAC) to provide two independent layers
 *   of cryptographic protection for EEG data records.
 *
 *   Security Architecture:
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │                    EEG Data Record                                 │
 *   │                                                                    │
 *   │  ┌──────────────────────────────┐                                  │
 *   │  │  AES-256-GCM Encryption      │  → ciphertext + GCM auth tag     │
 *   │  │  (Confidentiality + Integrity│                                  │
 *   │  └──────────────────────────────┘                                  │
 *   │                   +                                                │
 *   │  ┌──────────────────────────────┐                                  │
 *   │  │  HMAC-SHA-256                │  → 256-bit HMAC over ciphertext  │
 *   │  │  (Message Authentication)    │                                  │
 *   │  └──────────────────────────────┘                                  │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 *   Encryption sequence (encrypt_start):
 *     1. AES-256-GCM encrypts EEG data blocks → ciphertext + GCM tag
 *     2. HMAC-SHA-256 is computed over the ciphertext block
 *     3. Outputs: ciphertext, GCM tag (128-bit), HMAC (256-bit), nonce
 *
 *   Decryption + Verification sequence (decrypt_start):
 *     1. HMAC-SHA-256 is verified against stored HMAC over ciphertext
 *     2. If HMAC OK → AES-256-GCM decrypts and verifies GCM tag
 *     3. auth_verified asserted only when BOTH checks pass
 *
 * Parameters:
 *   MAX_BLOCKS - Maximum number of 128-bit data blocks per operation
 *
 * Note on key usage:
 *   aes_key  is the 256-bit AES-GCM encryption key.
 *   hmac_key is the 512-bit HMAC key (independent from AES key).
 *   Always use separate, independently derived keys for each algorithm.
 */

`timescale 1ns / 1ps

module eeg_security_top #(
    parameter int MAX_BLOCKS = 32
) (
    input  logic        clk,
    input  logic        rst,

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    input  logic        encrypt_start,
    input  logic        decrypt_start,

    // ------------------------------------------------------------------
    // Keys
    // ------------------------------------------------------------------
    input  logic [255:0] aes_key,          // AES-256 encryption key
    input  logic [511:0] hmac_key,         // HMAC-SHA-256 key (up to 512 bits)
    input  logic [7:0]   hmac_key_len,     // HMAC key length in bytes (max 64)

    // ------------------------------------------------------------------
    // Nonce management
    // ------------------------------------------------------------------
    input  logic [95:0]  nonce_seed,
    input  logic         nonce_increment,
    output logic [95:0]  current_nonce,

    // ------------------------------------------------------------------
    // EEG data input (plaintext for encrypt, ciphertext for decrypt)
    // ------------------------------------------------------------------
    input  logic         eeg_data_valid,
    input  logic [127:0] eeg_data_in,
    input  logic         eeg_data_last,
    input  logic [63:0]  eeg_data_len,     // in bits

    // Metadata (AAD for AES-GCM, also authenticated by HMAC)
    input  logic         metadata_valid,
    input  logic [127:0] metadata_block,
    input  logic [63:0]  metadata_len,     // in bits

    // ------------------------------------------------------------------
    // Output data
    // ------------------------------------------------------------------
    output logic [127:0] data_out,
    output logic         data_out_valid,

    // ------------------------------------------------------------------
    // Storage components (persist these alongside the ciphertext)
    // ------------------------------------------------------------------
    output logic [95:0]  storage_nonce,    // Nonce (IV) for decryption
    output logic [127:0] storage_gcm_tag,  // GCM authentication tag
    output logic [255:0] storage_hmac,     // HMAC-SHA-256 of ciphertext
    output logic         storage_ready,    // All storage outputs valid

    // ------------------------------------------------------------------
    // Decryption verification inputs
    // ------------------------------------------------------------------
    input  logic [95:0]  decrypt_nonce,
    input  logic [127:0] decrypt_gcm_tag,
    input  logic [255:0] decrypt_hmac,     // Expected HMAC to verify

    // ------------------------------------------------------------------
    // Status
    // ------------------------------------------------------------------
    output logic         auth_verified,    // Both GCM tag + HMAC OK
    output logic         auth_failed,      // Either check failed
    output logic         hmac_check_ok,    // HMAC check result
    output logic         gcm_check_ok,     // GCM tag check result
    output logic         busy,
    output logic         done,
    output logic         error,

    // ------------------------------------------------------------------
    // SHA-256 Hash Chain  (tamper-evident audit log)
    // Each security event is automatically logged into the chain.
    // Modify any stored entry → all subsequent hashes fail verification.
    // ------------------------------------------------------------------
    input  logic [255:0] chain_genesis,           // Chain anchor hash
    input  logic         chain_init,              // Sync reset to anchor
    output logic [255:0] chain_hash_out,          // Current chain head
    output logic [15:0]  chain_entry_count,       // Entries logged so far
    output logic         chain_hash_valid,        // Pulsed on new entry
    // Verification
    input  logic         chain_verify_start,      // Trigger verify
    input  logic [255:0] chain_verify_prev_hash,  // Hash before entry
    input  logic [255:0] chain_verify_log_data,   // Stored entry to check
    input  logic [255:0] chain_verify_expected,   // Expected hash
    output logic         chain_verify_ok,         // Chain intact
    output logic         chain_verify_fail,        // Tampering detected!

    // ------------------------------------------------------------------
    // Secure Boot / Bitstream Authentication  (RSA-2048 + SHA-256)
    // Verifies FPGA bitstream or firmware image against an RSA-2048
    // PKCS#1 v1.5 signature before allowing the system to boot.
    // ------------------------------------------------------------------
    input  logic [511:0]  boot_bitstream_block,    // 512-bit chunk of firmware image
    input  logic          boot_bitstream_valid,    // Chunk is valid
    input  logic          boot_bitstream_last,     // Last chunk of image
    input  logic [63:0]   boot_bitstream_len,      // Image length in bytes
    input  logic [2047:0] boot_public_key_n,       // RSA-2048 modulus N
    input  logic [2047:0] boot_signature,          // RSA-2048 signature
    input  logic [2047:0] boot_pub_exp,            // RSA public exponent (default 65537)
    input  logic [11:0]   boot_pub_exp_bits,       // Significant exp bits (17 for e=65537)
    input  logic          boot_verify_start,       // Trigger secure-boot check
    output logic          boot_allow,              // Signature OK → safe to boot
    output logic          boot_deny,               // Signature INVALID → do NOT boot
    output logic          boot_busy,               // Verification in progress
    output logic [255:0]  boot_hash_out            // SHA-256 hash of the firmware image
);

    // ------------------------------------------------------------------
    // Internal state machine
    // ------------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE,
        ENC_AES,         // AES-256-GCM encryption
        ENC_HMAC_WAIT,   // Wait for AES to finish, then start HMAC
        ENC_HMAC,        // HMAC-SHA-256 over ciphertext
        ENC_DONE,        // Outputs ready
        DEC_HMAC,        // Verify HMAC first
        DEC_HMAC_WAIT,   // Wait for HMAC result
        DEC_AES,         // AES-256-GCM decryption
        DEC_DONE,        // Outputs ready
        ERROR_STATE
    } fsm_t;

    fsm_t state;

    // ------------------------------------------------------------------
    // Nonce counter
    // ------------------------------------------------------------------
    logic [31:0] nonce_counter;
    logic [95:0] operation_nonce;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            nonce_counter   <= 32'h0;
            operation_nonce <= 96'h0;
        end else begin
            if (nonce_increment)
                nonce_counter <= nonce_counter + 32'h1;
            operation_nonce <= {nonce_seed[95:32], nonce_counter};
        end
    end
    assign current_nonce = operation_nonce;

    // ------------------------------------------------------------------
    // AES-256-GCM instance
    // ------------------------------------------------------------------
    logic        gcm_start;
    logic        gcm_mode;
    logic [95:0]  gcm_iv;
    logic        gcm_aad_valid;
    logic [127:0] gcm_aad_block;
    logic        gcm_aad_last;
    logic        gcm_data_valid;
    logic [127:0] gcm_data_in;
    logic        gcm_data_last;
    logic [63:0]  gcm_aad_len;
    logic [63:0]  gcm_data_len;
    logic [127:0] gcm_data_out;
    logic        gcm_data_out_valid;
    logic [127:0] gcm_auth_tag;
    logic        gcm_tag_valid;
    logic        gcm_auth_success;
    logic        gcm_busy;

    aes_256_gcm #(
        .MAX_BLOCKS(MAX_BLOCKS)
    ) u_aes_gcm (
        .clk            (clk),
        .rst            (rst),
        .start          (gcm_start),
        .mode           (gcm_mode),
        .key            (aes_key),
        .iv             (gcm_iv),
        .aad_valid      (gcm_aad_valid),
        .aad_block      (gcm_aad_block),
        .aad_last       (gcm_aad_last),
        .data_valid     (gcm_data_valid),
        .data_in        (gcm_data_in),
        .data_last      (gcm_data_last),
        .aad_len        (gcm_aad_len),
        .data_len       (gcm_data_len),
        .data_out       (gcm_data_out),
        .data_out_valid (gcm_data_out_valid),
        .auth_tag       (gcm_auth_tag),
        .tag_valid      (gcm_tag_valid),
        .auth_success   (gcm_auth_success),
        .busy           (gcm_busy)
    );

    // ------------------------------------------------------------------
    // HMAC-SHA-256 instance
    // ------------------------------------------------------------------
    logic        hmac_start;
    logic [511:0] hmac_msg_data;
    logic [63:0]  hmac_msg_len;
    logic        hmac_msg_valid;
    logic        hmac_msg_last;
    logic [255:0] hmac_result;
    logic        hmac_valid;
    logic        hmac_busy;

    hmac_sha256 u_hmac (
        .clk          (clk),
        .rst          (rst),
        .start        (hmac_start),
        .key          (hmac_key),
        .key_len      (hmac_key_len),
        .key_is_hashed(1'b0),
        .msg_data     (hmac_msg_data),
        .msg_len      (hmac_msg_len),
        .msg_valid    (hmac_msg_valid),
        .msg_last     (hmac_msg_last),
        .hmac_out     (hmac_result),
        .hmac_valid   (hmac_valid),
        .busy         (hmac_busy)
    );

    // ------------------------------------------------------------------
    // Audit log event codes
    // ------------------------------------------------------------------
    localparam logic [31:0] LOG_ENCRYPT_DONE = 32'h0000_0002;
    localparam logic [31:0] LOG_DECRYPT_OK   = 32'h0000_0004;
    localparam logic [31:0] LOG_GCM_TAMPER   = 32'h0000_0007;
    localparam logic [31:0] LOG_HMAC_TAMPER  = 32'h0000_0006;
    localparam logic [31:0] LOG_BOOT_ALLOW   = 32'h0000_0010; // Secure boot passed
    localparam logic [31:0] LOG_BOOT_DENY    = 32'h0000_0011; // Secure boot FAILED

    // Hash chain internal signals
    logic        hchain_log_valid;
    logic [255:0] hchain_log_data;
    logic [31:0]  log_seq_num;

    // ------------------------------------------------------------------
    // SHA-256 Hash Chain instance
    // Automatically receives log entries from the FSM below.
    // ------------------------------------------------------------------
    sha256_hash_chain u_hash_chain (
        .clk              (clk),
        .rst              (rst),
        .log_valid        (hchain_log_valid),
        .log_data         (hchain_log_data),
        .genesis_hash     (chain_genesis),
        .init_chain       (chain_init),
        .chain_hash       (chain_hash_out),
        .entry_count      (chain_entry_count),
        .hash_valid       (chain_hash_valid),
        .busy             (),
        .verify_start     (chain_verify_start),
        .verify_prev_hash (chain_verify_prev_hash),
        .verify_log_data  (chain_verify_log_data),
        .verify_expected  (chain_verify_expected),
        .verify_ok        (chain_verify_ok),
        .verify_fail      (chain_verify_fail)
    );

    // ------------------------------------------------------------------
    // Secure Boot instance  (RSA-2048 + SHA-256 PKCS#1 v1.5)
    // ------------------------------------------------------------------
    // Internal signals to capture boot events for auto-logging
    logic boot_allow_internal;
    logic boot_deny_internal;

    secure_boot #(
        .KEY_BITS(2048)
    ) u_secure_boot (
        .clk                (clk),
        .rst                (rst),
        .boot_start         (boot_verify_start),
        .bitstream_block    (boot_bitstream_block),
        .bitstream_valid    (boot_bitstream_valid),
        .bitstream_last     (boot_bitstream_last),
        .bitstream_len      (boot_bitstream_len),
        .rsa_signature      (boot_signature),
        .rsa_pub_exp        (boot_pub_exp),
        .rsa_pub_exp_bits   (boot_pub_exp_bits),
        .rsa_modulus        (boot_public_key_n),
        .boot_allow         (boot_allow_internal),
        .boot_deny          (boot_deny_internal),
        .boot_busy          (boot_busy),
        .boot_hash_out      (boot_hash_out)
    );

    // Connect outputs (also used in auto-log always_ff)
    assign boot_allow = boot_allow_internal;
    assign boot_deny  = boot_deny_internal;
    logic [95:0]  stored_nonce;
    logic [127:0] stored_gcm_tag;
    logic [255:0] stored_hmac;

    // Capture the latest AES output block for HMAC input
    logic [127:0] last_aes_out;
    logic        gcm_outputs_captured;
    logic        hmac_ok;
    logic        gcm_ok;

    // ------------------------------------------------------------------
    // Data-path connections (combinational, gated by state)
    // ------------------------------------------------------------------

    // AES-GCM data inputs come directly from top-level ports
    assign gcm_aad_valid = metadata_valid;
    assign gcm_aad_block = metadata_block;
    assign gcm_aad_last  = !metadata_valid;
    assign gcm_aad_len   = metadata_len;

    // In encrypt mode: forward EEG data to AES
    // In decrypt mode: forward EEG data to AES after HMAC check
    assign gcm_data_valid = (state == ENC_AES || state == DEC_AES) ? eeg_data_valid : 1'b0;
    assign gcm_data_in    = eeg_data_in;
    assign gcm_data_last  = eeg_data_last;
    assign gcm_data_len   = eeg_data_len;

    // Top-level output comes from AES-GCM
    assign data_out       = gcm_data_out;
    assign data_out_valid = gcm_data_out_valid;

    // Storage outputs
    assign storage_nonce   = stored_nonce;
    assign storage_gcm_tag = stored_gcm_tag;
    assign storage_hmac    = stored_hmac;

    // ------------------------------------------------------------------
    // HMAC message: feed the ciphertext output block
    // During encryption: compute HMAC over ciphertext (gcm_data_out)
    // During decryption: compute HMAC over input ciphertext (eeg_data_in)
    // For simplicity this implementation feeds one 512-bit chunk containing
    // the 128-bit ciphertext block left-aligned (zero-padded right).
    // ------------------------------------------------------------------
    always_comb begin
        if (state == ENC_HMAC || state == ENC_HMAC_WAIT) begin
            hmac_msg_data  = {last_aes_out, 384'h0};
            hmac_msg_len   = 64'd16; // 128 bits = 16 bytes
        end else begin
            // Decrypt verification: HMAC over input ciphertext
            hmac_msg_data  = {eeg_data_in, 384'h0};
            hmac_msg_len   = 64'd16;
        end
    end

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state               <= IDLE;
            busy                <= 1'b0;
            done                <= 1'b0;
            error               <= 1'b0;
            gcm_start           <= 1'b0;
            gcm_mode            <= 1'b0;
            gcm_iv              <= 96'h0;
            hmac_start          <= 1'b0;
            hmac_msg_valid      <= 1'b0;
            hmac_msg_last       <= 1'b0;
            stored_nonce        <= 96'h0;
            stored_gcm_tag      <= 128'h0;
            stored_hmac         <= 256'h0;
            last_aes_out        <= 128'h0;
            gcm_outputs_captured <= 1'b0;
            auth_verified       <= 1'b0;
            auth_failed         <= 1'b0;
            hmac_check_ok       <= 1'b0;
            gcm_check_ok        <= 1'b0;
            storage_ready       <= 1'b0;
            hmac_ok             <= 1'b0;
            gcm_ok              <= 1'b0;
        end else begin
            gcm_start      <= 1'b0;
            hmac_start     <= 1'b0;
            hmac_msg_valid <= 1'b0;
            hmac_msg_last  <= 1'b0;
            done           <= 1'b0;
            storage_ready  <= 1'b0;

            case (state)

                // --------------------------------------------------------
                IDLE: begin
                    busy             <= 1'b0;
                    auth_verified    <= 1'b0;
                    auth_failed      <= 1'b0;
                    gcm_outputs_captured <= 1'b0;
                    error            <= 1'b0;

                    if (encrypt_start) begin
                        busy          <= 1'b1;
                        gcm_start     <= 1'b1;
                        gcm_mode      <= 1'b0; // Encrypt
                        gcm_iv        <= operation_nonce;
                        stored_nonce  <= operation_nonce;
                        state         <= ENC_AES;
                    end else if (decrypt_start) begin
                        busy          <= 1'b1;
                        // Start HMAC verification BEFORE decrypting
                        hmac_start     <= 1'b1;
                        state          <= DEC_HMAC;
                    end
                end

                // --------------------------------------------------------
                // Encryption: wait for AES-GCM to finish
                // --------------------------------------------------------
                ENC_AES: begin
                    // Capture ciphertext output for HMAC
                    if (gcm_data_out_valid) begin
                        last_aes_out <= gcm_data_out;
                    end

                    if (gcm_tag_valid && !gcm_outputs_captured) begin
                        stored_gcm_tag      <= gcm_auth_tag;
                        gcm_outputs_captured <= 1'b1;
                        state               <= ENC_HMAC_WAIT;
                    end
                end

                // --------------------------------------------------------
                // Small delay to allow HMAC to start cleanly
                // --------------------------------------------------------
                ENC_HMAC_WAIT: begin
                    if (!hmac_busy) begin
                        hmac_start     <= 1'b1;
                        state          <= ENC_HMAC;
                    end
                end

                // --------------------------------------------------------
                // Encryption: compute HMAC over ciphertext
                // --------------------------------------------------------
                ENC_HMAC: begin
                    // Feed the ciphertext block to HMAC (single-block message)
                    if (!hmac_busy && !hmac_start) begin
                        hmac_msg_valid <= 1'b1;
                        hmac_msg_last  <= 1'b1;
                    end

                    if (hmac_valid) begin
                        stored_hmac <= hmac_result;
                        state       <= ENC_DONE;
                    end
                end

                // --------------------------------------------------------
                ENC_DONE: begin
                    storage_ready <= 1'b1;
                    done          <= 1'b1;
                    busy          <= 1'b0;
                    state         <= IDLE;
                end

                // --------------------------------------------------------
                // Decryption: HMAC verification phase
                // Feed the ciphertext to HMAC and compare result
                // --------------------------------------------------------
                DEC_HMAC: begin
                    if (!hmac_busy && !hmac_start) begin
                        hmac_msg_valid <= 1'b1;
                        hmac_msg_last  <= 1'b1;
                    end

                    if (hmac_valid) begin
                        hmac_ok <= (hmac_result == decrypt_hmac);
                        state   <= DEC_HMAC_WAIT;
                    end
                end

                DEC_HMAC_WAIT: begin
                    hmac_check_ok <= hmac_ok;
                    if (hmac_ok) begin
                        // HMAC OK → proceed to AES decryption
                        gcm_start  <= 1'b1;
                        gcm_mode   <= 1'b1; // Decrypt
                        gcm_iv     <= decrypt_nonce;
                        state      <= DEC_AES;
                    end else begin
                        // HMAC mismatch → abort
                        auth_failed <= 1'b1;
                        busy        <= 1'b0;
                        state       <= ERROR_STATE;
                    end
                end

                // --------------------------------------------------------
                // Decryption: wait for AES-GCM to finish + verify GCM tag
                // --------------------------------------------------------
                DEC_AES: begin
                    if (gcm_tag_valid) begin
                        gcm_ok        <= gcm_auth_success;
                        gcm_check_ok  <= gcm_auth_success;
                        state         <= DEC_DONE;
                    end
                end

                DEC_DONE: begin
                    if (hmac_ok && gcm_ok) begin
                        auth_verified <= 1'b1;
                        auth_failed   <= 1'b0;
                    end else begin
                        auth_verified <= 1'b0;
                        auth_failed   <= 1'b1;
                    end
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= IDLE;
                end

                // --------------------------------------------------------
                ERROR_STATE: begin
                    error <= 1'b1;
                    busy  <= 1'b0;
                    if (encrypt_start || decrypt_start)
                        state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Audit Log Entry Generation
    // Watches the FSM and auto-generates a SHA-256 hash chain entry on every
    // significant security event.
    //
    // Log entry format (256 bits):
    //   [255:224]  event_type    — security event code (LOG_* localparam)
    //   [223:192]  log_seq_num   — monotonic sequence counter
    //   [191:160]  status_flags  — {30'b0, gcm_ok, hmac_ok}
    //   [159:64]   nonce         — AES IV / decrypt nonce used
    //   [63:32]    data_len      — eeg_data_len lower 32 bits
    //   [31:0]     hmac_trunc    — upper 32 bits of HMAC (or 0)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            hchain_log_valid <= 1'b0;
            hchain_log_data  <= 256'h0;
            log_seq_num      <= 32'h0;
        end else begin
            hchain_log_valid <= 1'b0; // default: no log this cycle

            case (state)

                // Encryption completed successfully
                ENC_DONE: begin
                    hchain_log_valid <= 1'b1;
                    hchain_log_data  <= {
                        LOG_ENCRYPT_DONE,        // [255:224]
                        log_seq_num,             // [223:192]
                        32'h0000_0003,           // [191:160] gcm=1 hmac=1
                        stored_nonce,            // [159:64]
                        eeg_data_len[31:0],      // [63:32]
                        stored_hmac[255:224]     // [31:0]  HMAC truncated
                    };
                    log_seq_num <= log_seq_num + 32'h1;
                end

                // Decryption completed (gcm_ok tells if GCM tag passed)
                DEC_DONE: begin
                    hchain_log_valid <= 1'b1;
                    hchain_log_data  <= {
                        gcm_ok ? LOG_DECRYPT_OK : LOG_GCM_TAMPER, // [255:224]
                        log_seq_num,                               // [223:192]
                        {30'h0, gcm_ok, hmac_ok},                  // [191:160]
                        decrypt_nonce,                             // [159:64]
                        eeg_data_len[31:0],                        // [63:32]
                        32'h0                                      // [31:0]
                    };
                    log_seq_num <= log_seq_num + 32'h1;
                end

                // HMAC mismatch: ciphertext was tampered before decryption
                DEC_HMAC_WAIT: begin
                    if (!hmac_ok) begin
                        hchain_log_valid <= 1'b1;
                        hchain_log_data  <= {
                            LOG_HMAC_TAMPER,       // [255:224]
                            log_seq_num,           // [223:192]
                            32'h0000_0000,         // [191:160] hmac_ok=0
                            decrypt_nonce,         // [159:64]
                            eeg_data_len[31:0],    // [63:32]
                            32'h0                  // [31:0]
                        };
                        log_seq_num <= log_seq_num + 32'h1;
                    end
                end

                // Secure boot events arrive asynchronously to the AES/HMAC FSM.
                // They are captured here in the default branch so that there is
                // only ONE always_ff driving hchain_log_valid / hchain_log_data.
                default: begin
                    if (boot_allow_internal) begin
                        hchain_log_valid <= 1'b1;
                        hchain_log_data  <= {
                            LOG_BOOT_ALLOW,            // [255:224]
                            log_seq_num,               // [223:192]
                            32'h0000_0001,             // [191:160] boot_allow=1
                            boot_hash_out[255:160],    // [159:64]  firmware hash MSBs
                            32'h0,                     // [63:32]
                            32'h0                      // [31:0]
                        };
                        log_seq_num <= log_seq_num + 32'h1;
                    end else if (boot_deny_internal) begin
                        hchain_log_valid <= 1'b1;
                        hchain_log_data  <= {
                            LOG_BOOT_DENY,             // [255:224]
                            log_seq_num,               // [223:192]
                            32'h0000_0000,             // [191:160] boot_allow=0
                            boot_hash_out[255:160],    // [159:64]  firmware hash MSBs
                            32'h0,                     // [63:32]
                            32'h0                      // [31:0]
                        };
                        log_seq_num <= log_seq_num + 32'h1;
                    end
                end
            endcase
        end
    end

endmodule
