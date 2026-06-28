/**
 * Module: sha256_core
 *
 * Description:
 *   Implements the SHA-256 cryptographic hash function (FIPS PUB 180-4).
 *   Processes 512-bit (16 x 32-bit word) message blocks and produces a
 *   256-bit digest.
 *
 *   Operation:
 *     1. Assert start=1 for one clock, load the 512-bit block on data_in.
 *     2. Wait for digest_valid to go high (64 compression rounds + overhead).
 *     3. Read the 256-bit digest on digest_out.
 *     4. For a multi-block message keep track of the running hash state by
 *        feeding it back on init_hash and asserting use_init_hash=1 for every
 *        block after the first.  The padding / length appending must be done
 *        by the caller (HMAC wrapper handles this).
 *
 * Latency:   ~66 clock cycles per 512-bit block
 * Throughput: 1 block / 66 cycles (no pipelining between blocks)
 *
 * Vivado compatibility:
 *   - Pure synchronous design, no latches, no generate with function calls.
 *   - All constants are localparams or parameters.
 *   - Simulation-safe (no $random etc. in synthesisable code).
 */

`timescale 1ns / 1ps

module sha256_core (
    input  logic        clk,
    input  logic        rst,

    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    input  logic        start,          // Pulse high for 1 cycle to begin
    input  logic [511:0] data_in,       // 512-bit padded message block

    // ------------------------------------------------------------------
    // Chaining (multi-block messages)
    // ------------------------------------------------------------------
    input  logic        use_init_hash,  // 1 = use init_hash as IV
    input  logic [255:0] init_hash,     // Previous block digest (chaining)

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------
    output logic [255:0] digest_out,    // 256-bit SHA-256 digest
    output logic         digest_valid,  // High for 1 cycle when done
    output logic         busy           // High while processing
);

    // ------------------------------------------------------------------
    // SHA-256 initial hash values H0..H7 (first 32 bits of fractional
    // parts of the square roots of the first 8 primes)
    // ------------------------------------------------------------------
    localparam logic [31:0] H0_INIT = 32'h6a09e667;
    localparam logic [31:0] H1_INIT = 32'hbb67ae85;
    localparam logic [31:0] H2_INIT = 32'h3c6ef372;
    localparam logic [31:0] H3_INIT = 32'ha54ff53a;
    localparam logic [31:0] H4_INIT = 32'h510e527f;
    localparam logic [31:0] H5_INIT = 32'h9b05688c;
    localparam logic [31:0] H6_INIT = 32'h1f83d9ab;
    localparam logic [31:0] H7_INIT = 32'h5be0cd19;

    // ------------------------------------------------------------------
    // SHA-256 round constants K[0..63]
    // ------------------------------------------------------------------
    logic [31:0] K [0:63];
    always_comb begin
        K[ 0] = 32'h428a2f98; K[ 1] = 32'h71374491; K[ 2] = 32'hb5c0fbcf; K[ 3] = 32'he9b5dba5;
        K[ 4] = 32'h3956c25b; K[ 5] = 32'h59f111f1; K[ 6] = 32'h923f82a4; K[ 7] = 32'hab1c5ed5;
        K[ 8] = 32'hd807aa98; K[ 9] = 32'h12835b01; K[10] = 32'h243185be; K[11] = 32'h550c7dc3;
        K[12] = 32'h72be5d74; K[13] = 32'h80deb1fe; K[14] = 32'h9bdc06a7; K[15] = 32'hc19bf174;
        K[16] = 32'he49b69c1; K[17] = 32'hefbe4786; K[18] = 32'h0fc19dc6; K[19] = 32'h240ca1cc;
        K[20] = 32'h2de92c6f; K[21] = 32'h4a7484aa; K[22] = 32'h5cb0a9dc; K[23] = 32'h76f988da;
        K[24] = 32'h983e5152; K[25] = 32'ha831c66d; K[26] = 32'hb00327c8; K[27] = 32'hbf597fc7;
        K[28] = 32'hc6e00bf3; K[29] = 32'hd5a79147; K[30] = 32'h06ca6351; K[31] = 32'h14292967;
        K[32] = 32'h27b70a85; K[33] = 32'h2e1b2138; K[34] = 32'h4d2c6dfc; K[35] = 32'h53380d13;
        K[36] = 32'h650a7354; K[37] = 32'h766a0abb; K[38] = 32'h81c2c92e; K[39] = 32'h92722c85;
        K[40] = 32'ha2bfe8a1; K[41] = 32'ha81a664b; K[42] = 32'hc24b8b70; K[43] = 32'hc76c51a3;
        K[44] = 32'hd192e819; K[45] = 32'hd6990624; K[46] = 32'hf40e3585; K[47] = 32'h106aa070;
        K[48] = 32'h19a4c116; K[49] = 32'h1e376c08; K[50] = 32'h2748774c; K[51] = 32'h34b0bcb5;
        K[52] = 32'h391c0cb3; K[53] = 32'h4ed8aa4a; K[54] = 32'h5b9cca4f; K[55] = 32'h682e6ff3;
        K[56] = 32'h748f82ee; K[57] = 32'h78a5636f; K[58] = 32'h84c87814; K[59] = 32'h8cc70208;
        K[60] = 32'h90befffa; K[61] = 32'ha4506ceb; K[62] = 32'hbef9a3f7; K[63] = 32'hc67178f2;
    end

    // ------------------------------------------------------------------
    // Internal state
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE,
        S_EXPAND,   // Message schedule expansion (rounds 16-63 prep)
        S_COMPRESS, // 64 compression rounds
        S_DONE
    } fsm_t;

    fsm_t state;

    // Working variables
    logic [31:0] a, b, c, d, e, f, g, h;

    // Hash state (running)
    logic [31:0] hh [0:7]; // hh[0]=H0 ... hh[7]=H7

    // Message schedule W[0..63]
    logic [31:0] W [0:63];

    // Round counter
    logic [6:0] round; // 0..63

    // ------------------------------------------------------------------
    // Rotation / shift helper functions (combinational)
    // ------------------------------------------------------------------
    function automatic logic [31:0] rotr32(input logic [31:0] x, input integer n);
        return (x >> n) | (x << (32 - n));
    endfunction

    function automatic logic [31:0] shr32(input logic [31:0] x, input integer n);
        return x >> n;
    endfunction

    // SHA-256 sigma functions
    function automatic logic [31:0] sigma0(input logic [31:0] x);
        return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
    endfunction

    function automatic logic [31:0] sigma1(input logic [31:0] x);
        return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
    endfunction

    function automatic logic [31:0] gamma0(input logic [31:0] x);
        return rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3);
    endfunction

    function automatic logic [31:0] gamma1(input logic [31:0] x);
        return rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10);
    endfunction

    function automatic logic [31:0] ch(input logic [31:0] x, y, z);
        return (x & y) ^ (~x & z);
    endfunction

    function automatic logic [31:0] maj(input logic [31:0] x, y, z);
        return (x & y) ^ (x & z) ^ (y & z);
    endfunction

    // ------------------------------------------------------------------
    // Combinational: one compression round outputs
    // ------------------------------------------------------------------
    logic [31:0] T1, T2;
    always_comb begin
        T1 = h + sigma1(e) + ch(e, f, g) + K[round] + W[round];
        T2 = sigma0(a) + maj(a, b, c);
    end

    // ------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            digest_valid <= 1'b0;
            digest_out   <= 256'h0;
            round        <= 7'h0;
            a <= 32'h0; b <= 32'h0; c <= 32'h0; d <= 32'h0;
            e <= 32'h0; f <= 32'h0; g <= 32'h0; h <= 32'h0;
            for (int i = 0; i < 8;  i++) hh[i] <= 32'h0;
            for (int i = 0; i < 64; i++) W[i]  <= 32'h0;
        end else begin
            digest_valid <= 1'b0; // default

            case (state)
                // --------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy  <= 1'b1;
                        round <= 7'h0;

                        // Load initial hash values (IV or chained)
                        if (use_init_hash) begin
                            hh[0] <= init_hash[255:224];
                            hh[1] <= init_hash[223:192];
                            hh[2] <= init_hash[191:160];
                            hh[3] <= init_hash[159:128];
                            hh[4] <= init_hash[127: 96];
                            hh[5] <= init_hash[ 95: 64];
                            hh[6] <= init_hash[ 63: 32];
                            hh[7] <= init_hash[ 31:  0];
                        end else begin
                            hh[0] <= H0_INIT; hh[1] <= H1_INIT;
                            hh[2] <= H2_INIT; hh[3] <= H3_INIT;
                            hh[4] <= H4_INIT; hh[5] <= H5_INIT;
                            hh[6] <= H6_INIT; hh[7] <= H7_INIT;
                        end

                        // Load first 16 words of message schedule from data_in
                        // data_in[511:480] = W[0], data_in[479:448] = W[1], ...
                        for (int j = 0; j < 16; j++) begin
                            W[j] <= data_in[511 - j*32 -: 32];
                        end

                        // Initialise remaining schedule slots to 0 (computed in S_EXPAND)
                        for (int j = 16; j < 64; j++) W[j] <= 32'h0;

                        state <= S_EXPAND;
                    end
                end

                // --------------------------------------------------------
                // Expand message schedule W[16..63] one word per cycle
                // --------------------------------------------------------
                S_EXPAND: begin
                    // round used as expansion index (16..63)
                    W[round + 16] <= gamma1(W[round + 14]) + W[round + 9] +
                                     gamma0(W[round +  1]) + W[round];

                    if (round == 7'd47) begin // 48 expansion steps (16..63)
                        round <= 7'h0;
                        // Initialise working variables
                        a <= hh[0]; b <= hh[1]; c <= hh[2]; d <= hh[3];
                        e <= hh[4]; f <= hh[5]; g <= hh[6]; h <= hh[7];
                        state <= S_COMPRESS;
                    end else begin
                        round <= round + 7'h1;
                    end
                end

                // --------------------------------------------------------
                // 64 compression rounds
                // --------------------------------------------------------
                S_COMPRESS: begin
                    h <= g;
                    g <= f;
                    f <= e;
                    e <= d + T1;
                    d <= c;
                    c <= b;
                    b <= a;
                    a <= T1 + T2;

                    if (round == 7'd63) begin
                        round <= 7'h0;
                        state <= S_DONE;
                    end else begin
                        round <= round + 7'h1;
                    end
                end

                // --------------------------------------------------------
                S_DONE: begin
                    // Add compressed chunk to current hash value
                    hh[0] <= hh[0] + a; hh[1] <= hh[1] + b;
                    hh[2] <= hh[2] + c; hh[3] <= hh[3] + d;
                    hh[4] <= hh[4] + e; hh[5] <= hh[5] + f;
                    hh[6] <= hh[6] + g; hh[7] <= hh[7] + h;

                    digest_out   <= {hh[0]+a, hh[1]+b, hh[2]+c, hh[3]+d,
                                     hh[4]+e, hh[5]+f, hh[6]+g, hh[7]+h};
                    digest_valid <= 1'b1;
                    busy         <= 1'b0;
                    state        <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
