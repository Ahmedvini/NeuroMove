/**
 * Module: gcm_ghash
 * 
 * Description:
 *   Implements GHASH function for Galois/Counter Mode (GCM) authentication.
 *   GHASH is used to compute authentication tags for GCM.
 *   
 *   GHASH(H, A, C) computes authentication over AAD (A) and ciphertext (C)
 *   using hash subkey H derived from encrypting all-zeros block.
 *   
 * Operation:
 *   1. Process Additional Authenticated Data (AAD) blocks
 *   2. Process ciphertext/plaintext blocks
 *   3. Process length block (len(A) || len(C))
 *   4. Output authentication tag
 * 
 * Latency:
 *   Variable depending on data length
 *   ~2 cycles per 128-bit block processed
 *   
 * Parameters:
 *   None (fixed 128-bit blocks)
 */

module gcm_ghash (
    input  logic        clk,
    input  logic        rst,
    
    // Control signals
    input  logic        start,        // Start GHASH computation
    input  logic        aad_valid,    // AAD block valid
    input  logic        data_valid_in,// Data block valid
    input  logic        is_final,     // Final block (length block)
    
    // Hash subkey H (from AES encryption of zero block)
    input  logic [127:0] hash_key,
    
    // Input blocks
    input  logic [127:0] block_in,
    
    // Authentication tag output
    output logic [127:0] tag_out,
    output logic        tag_valid,
    output logic        busy
);

    // -------------------------------------------------------------------------
    // Internal Signals
    // -------------------------------------------------------------------------
    
    typedef enum logic [2:0] {
        IDLE,
        PROCESS_AAD,
        PROCESS_DATA,
        PROCESS_LEN,
        DONE
    } state_t;
    
    state_t state, next_state;
    
    logic [127:0] ghash_state;     // Running GHASH accumulator (Y)
    logic [127:0] mult_result;     // Result of GF multiplication
    logic [127:0] next_ghash_state;
    
    // -------------------------------------------------------------------------
    // GF(2^128) Multiplication
    // Multiplies two 128-bit values in GF(2^128)
    // Uses reduction polynomial: x^128 + x^7 + x^2 + x + 1
    // -------------------------------------------------------------------------
    
    function automatic logic [127:0] gf128_mult(input logic [127:0] x, input logic [127:0] y);
        logic [127:0] z;
        logic [127:0] v;
        logic [127:0] r; // Reduction polynomial: 11100001 || 0^120
        logic lsb;
        integer i;
        
        z = 128'h0;
        v = y;
        r = 128'he1000000000000000000000000000000; // Reverse bit order for standard
        
        for (i = 0; i < 128; i = i + 1) begin
            // If bit i of x is 1, XOR z with v
            if (x[127-i])
                z = z ^ v;
            
            // Check if LSB of v is 1
            lsb = v[0];
            
            // Right shift v
            v = v >> 1;
            
            // If LSB was 1, XOR with R
            if (lsb)
                v = v ^ r;
        end
        
        return z;
    endfunction
    
    // -------------------------------------------------------------------------
    // State Machine
    // -------------------------------------------------------------------------
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            ghash_state <= 128'h0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        ghash_state <= 128'h0;
                    end
                end
                
                PROCESS_AAD: begin
                    if (aad_valid) begin
                        // Y_{i} = (Y_{i-1} XOR X_i) * H
                        next_ghash_state = ghash_state ^ block_in;
                        ghash_state <= gf128_mult(next_ghash_state, hash_key);
                    end
                end
                
                PROCESS_DATA: begin
                    if (data_valid_in) begin
                        // Y_{i} = (Y_{i-1} XOR C_i) * H
                        next_ghash_state = ghash_state ^ block_in;
                        ghash_state <= gf128_mult(next_ghash_state, hash_key);
                    end
                end
                
                PROCESS_LEN: begin
                    if (is_final) begin
                        // Process length block: len(A) || len(C)
                        next_ghash_state = ghash_state ^ block_in;
                        ghash_state <= gf128_mult(next_ghash_state, hash_key);
                    end
                end
                
                DONE: begin
                    // Tag ready
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
                    next_state = PROCESS_AAD;
            end
            
            PROCESS_AAD: begin
                if (!aad_valid && data_valid_in)
                    next_state = PROCESS_DATA;
                else if (!aad_valid && is_final)
                    next_state = PROCESS_LEN;
            end
            
            PROCESS_DATA: begin
                if (!data_valid_in && is_final)
                    next_state = PROCESS_LEN;
            end
            
            PROCESS_LEN: begin
                if (is_final)
                    next_state = DONE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output assignments
    assign tag_out = ghash_state;
    assign tag_valid = (state == DONE);
    assign busy = (state != IDLE) && (state != DONE);
    
endmodule
