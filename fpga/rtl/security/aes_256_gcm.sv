module aes_256_gcm #(
    parameter int MAX_BLOCKS = 16
) (
    input  logic        clk,
    input  logic        rst,
    
    // Control signals
    input  logic        start,           // Start operation
    input  logic        mode,            // 0=encrypt, 1=decrypt
    
    // Key and IV
    input  logic [255:0] key,            // 256-bit key
    input  logic [95:0]  iv,             // 96-bit IV (nonce)
    
    // Additional Authenticated Data (AAD)
    input  logic        aad_valid,       // AAD block valid
    input  logic [127:0] aad_block,      // AAD data block
    input  logic        aad_last,        // Last AAD block
    
    // Data input/output
    input  logic        data_valid,      // Data block valid
    input  logic [127:0] data_in,        // Plaintext/ciphertext block
    input  logic        data_last,       // Last data block
    
    // Length information (in bits)
    input  logic [63:0]  aad_len,        // AAD length in bits
    input  logic [63:0]  data_len,       // Data length in bits
    
    // Output
    output logic [127:0] data_out,       // Ciphertext/plaintext block
    output logic        data_out_valid,  // Output valid
    output logic [127:0] auth_tag,       // Authentication tag
    output logic        tag_valid,       // Tag valid
    output logic        auth_success,    // Authentication success (decrypt only)
    output logic        busy
);

    // -------------------------------------------------------------------------
    // Internal Signals
    // -------------------------------------------------------------------------
    
    typedef enum logic [3:0] {
        IDLE,
        GEN_HASH_KEY,      // Generate H = AES_K(0)
        GEN_COUNTER,       // Generate initial counter
        PROCESS_AAD,       // Process AAD blocks
        PROCESS_DATA,      // Process data blocks
        PROCESS_LENGTHS,   // Process length block
        GEN_TAG,           // Generate authentication tag
        VERIFY_TAG,        // Verify tag (decrypt mode)
        DONE
    } state_t;
    
    state_t state, next_state;
    
    // AES core signals
    logic        aes_start;
    logic        aes_mode;
    logic [127:0] aes_data_in;
    logic [255:0] aes_key;
    logic [127:0] aes_data_out;
    logic        aes_data_valid;
    logic        aes_busy;
    
    // GHASH signals
    logic        ghash_start;
    logic        ghash_aad_valid;
    logic        ghash_data_valid;
    logic        ghash_is_final;
    logic [127:0] ghash_hash_key;
    logic [127:0] ghash_block_in;
    logic [127:0] ghash_tag_out;
    logic        ghash_tag_valid;
    logic        ghash_busy;
    
    // Internal state
    logic [127:0] hash_key_h;           // H = AES_K(0^128)
    logic [31:0]  counter;              // CTR mode counter
    logic [127:0] counter_block;        // Current counter block
    logic [127:0] encrypted_counter;    // Encrypted counter for XOR
    logic [127:0] tag_mask;             // E(K, Y_0) for tag encryption
    logic [127:0] computed_tag;         // Computed authentication tag
    logic [127:0] received_tag;         // Received tag (for verification)
    
    logic [7:0]   block_count;          // Number of blocks processed
    logic [127:0] aad_buffer;           // AAD accumulator
    logic [127:0] data_buffer;          // Data accumulator
    
    // -------------------------------------------------------------------------
    // Module Instantiations
    // -------------------------------------------------------------------------
    
    aes_256_core u_aes_core (
        .clk        (clk),
        .rst        (rst),
        .start      (aes_start),
        .mode       (aes_mode),
        .data_in    (aes_data_in),
        .key_in     (aes_key),
        .data_out   (aes_data_out),
        .data_valid (aes_data_valid),
        .busy       (aes_busy)
    );
    
    gcm_ghash u_ghash (
        .clk            (clk),
        .rst            (rst),
        .start          (ghash_start),
        .aad_valid      (ghash_aad_valid),
        .data_valid_in  (ghash_data_valid),
        .is_final       (ghash_is_final),
        .hash_key       (ghash_hash_key),
        .block_in       (ghash_block_in),
        .tag_out        (ghash_tag_out),
        .tag_valid      (ghash_tag_valid),
        .busy           (ghash_busy)
    );
    
    // -------------------------------------------------------------------------
    // Counter Block Generation
    // -------------------------------------------------------------------------
    
    function automatic logic [127:0] build_counter_block(
        input logic [95:0] nonce,
        input logic [31:0] count
    );
        return {nonce, count};
    endfunction
    
    // -------------------------------------------------------------------------
    // State Machine
    // -------------------------------------------------------------------------
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            counter <= 32'h1;
            block_count <= 0;
            hash_key_h <= 128'h0;
            tag_mask <= 128'h0;
            computed_tag <= 128'h0;
            received_tag <= 128'h0;
            aes_start <= 1'b0;
            ghash_start <= 1'b0;
        end else begin
            state <= next_state;
            aes_start <= 1'b0;
            ghash_start <= 1'b0;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        counter <= 32'h1;
                        block_count <= 0;
                        aes_key <= key;
                    end
                end
                
                GEN_HASH_KEY: begin
                    if (!aes_busy && !aes_start) begin
                        // Encrypt all-zeros block to generate H
                        aes_start <= 1'b1;
                        aes_mode <= 1'b0;  // Encrypt
                        aes_data_in <= 128'h0;
                    end
                    
                    if (aes_data_valid) begin
                        hash_key_h <= aes_data_out;
                    end
                end
                
                GEN_COUNTER: begin
                    if (!aes_busy && !aes_start) begin
                        // Generate E(K, Y_0) for tag masking
                        counter_block <= build_counter_block(iv, 32'h0);
                        aes_start <= 1'b1;
                        aes_mode <= 1'b0;  // Encrypt
                        aes_data_in <= counter_block;
                    end
                    
                    if (aes_data_valid) begin
                        tag_mask <= aes_data_out;
                        ghash_start <= 1'b1;
                        counter <= 32'h1;
                    end
                end
                
                PROCESS_AAD: begin
                    if (aad_valid) begin
                        ghash_aad_valid <= 1'b1;
                        ghash_block_in <= aad_block;
                    end else begin
                        ghash_aad_valid <= 1'b0;
                    end
                end
                
                PROCESS_DATA: begin
                    if (data_valid && !aes_busy && !aes_start) begin
                        // Generate encrypted counter block
                        counter_block <= build_counter_block(iv, counter);
                        aes_start <= 1'b1;
                        aes_mode <= 1'b0;  // Always encrypt counter
                        aes_data_in <= counter_block;
                        counter <= counter + 1;
                        block_count <= block_count + 1;
                    end
                    
                    if (aes_data_valid) begin
                        encrypted_counter <= aes_data_out;
                        
                        // XOR with plaintext/ciphertext
                        if (mode == 0) begin
                            // Encryption: C = P XOR E(K, counter)
                            data_buffer <= data_in ^ aes_data_out;
                            // Feed ciphertext to GHASH
                            ghash_data_valid <= 1'b1;
                            ghash_block_in <= data_in ^ aes_data_out;
                        end else begin
                            // Decryption: P = C XOR E(K, counter)
                            data_buffer <= data_in ^ aes_data_out;
                            // Feed ciphertext to GHASH
                            ghash_data_valid <= 1'b1;
                            ghash_block_in <= data_in;
                        end
                    end else begin
                        ghash_data_valid <= 1'b0;
                    end
                end
                
                PROCESS_LENGTHS: begin
                    // Send length block to GHASH: len(A) || len(C)
                    ghash_is_final <= 1'b1;
                    ghash_block_in <= {aad_len, data_len};
                end
                
                GEN_TAG: begin
                    if (ghash_tag_valid) begin
                        // T = GHASH(H, A, C) XOR E(K, Y_0)
                        computed_tag <= ghash_tag_out ^ tag_mask;
                    end
                end
                
                VERIFY_TAG: begin
                    // In decrypt mode, verify received tag matches computed tag
                    // Note: received_tag should be captured from input
                end
                
                DONE: begin
                    // Operation complete
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start)
                    next_state = GEN_HASH_KEY;
            end
            
            GEN_HASH_KEY: begin
                if (aes_data_valid)
                    next_state = GEN_COUNTER;
            end
            
            GEN_COUNTER: begin
                if (aes_data_valid)
                    next_state = PROCESS_AAD;
            end
            
            PROCESS_AAD: begin
                if (aad_last)
                    next_state = PROCESS_DATA;
            end
            
            PROCESS_DATA: begin
                if (data_last)
                    next_state = PROCESS_LENGTHS;
            end
            
            PROCESS_LENGTHS: begin
                next_state = GEN_TAG;
            end
            
            GEN_TAG: begin
                if (ghash_tag_valid) begin
                    if (mode == 1)
                        next_state = VERIFY_TAG;
                    else
                        next_state = DONE;
                end
            end
            
            VERIFY_TAG: begin
                next_state = DONE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // -------------------------------------------------------------------------
    // Output Assignments
    // -------------------------------------------------------------------------
    
    assign data_out = data_buffer;
    assign data_out_valid = (state == PROCESS_DATA) && aes_data_valid;
    assign auth_tag = computed_tag;
    assign tag_valid = (state == DONE);
    assign auth_success = (mode == 1) ? (computed_tag == received_tag) : 1'b1;
    assign busy = (state != IDLE) && (state != DONE);
    
    // Connect GHASH hash key
    assign ghash_hash_key = hash_key_h;
    
endmodule
