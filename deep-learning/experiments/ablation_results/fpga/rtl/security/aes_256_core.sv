

module aes_256_core (
    input  logic        clk,
    input  logic        rst,
    
    // Control signals
    input  logic        start,        // Start encryption/decryption
    input  logic        mode,         // 0=encrypt, 1=decrypt
    
    // Data input (128-bit block)
    input  logic [127:0] data_in,
    
    // Key input (256-bit key)
    input  logic [255:0] key_in,
    
    // Data output
    output logic [127:0] data_out,
    output logic        data_valid,
    output logic        busy
);

    // -------------------------------------------------------------------------
    // Constants and Types
    // -------------------------------------------------------------------------
    
    localparam int NUM_ROUNDS = 14;
    localparam int KEY_WORDS = 8;   // 256-bit key = 8 words
    localparam int ROUND_KEYS = 15; // 15 round keys for 14 rounds + initial
    
    typedef enum logic [3:0] {
        IDLE,
        KEY_EXPANSION,
        INIT_ROUND,
        MAIN_ROUNDS,
        FINAL_ROUND,
        DONE
    } state_t;
    
    // -------------------------------------------------------------------------
    // Internal Signals
    // -------------------------------------------------------------------------
    
    state_t state, next_state;
    logic [3:0] round_cnt;
    logic [127:0] state_reg, next_state_reg;
    logic [127:0] round_key;
    
    // Key expansion
    logic [31:0] round_keys [0:59]; // 60 words for 15 round keys
    logic key_expansion_done;
    logic [5:0] key_exp_cnt;

    // Temporary variables for always_ff (cannot declare inside case/if in Vivado)
    logic [127:0] fsm_temp;
    logic [31:0]  key_temp;
    integer       key_idx;
    
    // -------------------------------------------------------------------------
    // S-Box (SubBytes transformation)
    // -------------------------------------------------------------------------
    
    function automatic logic [7:0] sbox(input logic [7:0] in);
        logic [7:0] out;
        case(in)
            8'h00: out = 8'h63; 8'h01: out = 8'h7c; 8'h02: out = 8'h77; 8'h03: out = 8'h7b;
            8'h04: out = 8'hf2; 8'h05: out = 8'h6b; 8'h06: out = 8'h6f; 8'h07: out = 8'hc5;
            8'h08: out = 8'h30; 8'h09: out = 8'h01; 8'h0a: out = 8'h67; 8'h0b: out = 8'h2b;
            8'h0c: out = 8'hfe; 8'h0d: out = 8'hd7; 8'h0e: out = 8'hab; 8'h0f: out = 8'h76;
            8'h10: out = 8'hca; 8'h11: out = 8'h82; 8'h12: out = 8'hc9; 8'h13: out = 8'h7d;
            8'h14: out = 8'hfa; 8'h15: out = 8'h59; 8'h16: out = 8'h47; 8'h17: out = 8'hf0;
            8'h18: out = 8'had; 8'h19: out = 8'hd4; 8'h1a: out = 8'ha2; 8'h1b: out = 8'haf;
            8'h1c: out = 8'h9c; 8'h1d: out = 8'ha4; 8'h1e: out = 8'h72; 8'h1f: out = 8'hc0;
            8'h20: out = 8'hb7; 8'h21: out = 8'hfd; 8'h22: out = 8'h93; 8'h23: out = 8'h26;
            8'h24: out = 8'h36; 8'h25: out = 8'h3f; 8'h26: out = 8'hf7; 8'h27: out = 8'hcc;
            8'h28: out = 8'h34; 8'h29: out = 8'ha5; 8'h2a: out = 8'he5; 8'h2b: out = 8'hf1;
            8'h2c: out = 8'h71; 8'h2d: out = 8'hd8; 8'h2e: out = 8'h31; 8'h2f: out = 8'h15;
            8'h30: out = 8'h04; 8'h31: out = 8'hc7; 8'h32: out = 8'h23; 8'h33: out = 8'hc3;
            8'h34: out = 8'h18; 8'h35: out = 8'h96; 8'h36: out = 8'h05; 8'h37: out = 8'h9a;
            8'h38: out = 8'h07; 8'h39: out = 8'h12; 8'h3a: out = 8'h80; 8'h3b: out = 8'he2;
            8'h3c: out = 8'heb; 8'h3d: out = 8'h27; 8'h3e: out = 8'hb2; 8'h3f: out = 8'h75;
            8'h40: out = 8'h09; 8'h41: out = 8'h83; 8'h42: out = 8'h2c; 8'h43: out = 8'h1a;
            8'h44: out = 8'h1b; 8'h45: out = 8'h6e; 8'h46: out = 8'h5a; 8'h47: out = 8'ha0;
            8'h48: out = 8'h52; 8'h49: out = 8'h3b; 8'h4a: out = 8'hd6; 8'h4b: out = 8'hb3;
            8'h4c: out = 8'h29; 8'h4d: out = 8'he3; 8'h4e: out = 8'h2f; 8'h4f: out = 8'h84;
            8'h50: out = 8'h53; 8'h51: out = 8'hd1; 8'h52: out = 8'h00; 8'h53: out = 8'hed;
            8'h54: out = 8'h20; 8'h55: out = 8'hfc; 8'h56: out = 8'hb1; 8'h57: out = 8'h5b;
            8'h58: out = 8'h6a; 8'h59: out = 8'hcb; 8'h5a: out = 8'hbe; 8'h5b: out = 8'h39;
            8'h5c: out = 8'h4a; 8'h5d: out = 8'h4c; 8'h5e: out = 8'h58; 8'h5f: out = 8'hcf;
            8'h60: out = 8'hd0; 8'h61: out = 8'hef; 8'h62: out = 8'haa; 8'h63: out = 8'hfb;
            8'h64: out = 8'h43; 8'h65: out = 8'h4d; 8'h66: out = 8'h33; 8'h67: out = 8'h85;
            8'h68: out = 8'h45; 8'h69: out = 8'hf9; 8'h6a: out = 8'h02; 8'h6b: out = 8'h7f;
            8'h6c: out = 8'h50; 8'h6d: out = 8'h3c; 8'h6e: out = 8'h9f; 8'h6f: out = 8'ha8;
            8'h70: out = 8'h51; 8'h71: out = 8'ha3; 8'h72: out = 8'h40; 8'h73: out = 8'h8f;
            8'h74: out = 8'h92; 8'h75: out = 8'h9d; 8'h76: out = 8'h38; 8'h77: out = 8'hf5;
            8'h78: out = 8'hbc; 8'h79: out = 8'hb6; 8'h7a: out = 8'hda; 8'h7b: out = 8'h21;
            8'h7c: out = 8'h10; 8'h7d: out = 8'hff; 8'h7e: out = 8'hf3; 8'h7f: out = 8'hd2;
            8'h80: out = 8'hcd; 8'h81: out = 8'h0c; 8'h82: out = 8'h13; 8'h83: out = 8'hec;
            8'h84: out = 8'h5f; 8'h85: out = 8'h97; 8'h86: out = 8'h44; 8'h87: out = 8'h17;
            8'h88: out = 8'hc4; 8'h89: out = 8'ha7; 8'h8a: out = 8'h7e; 8'h8b: out = 8'h3d;
            8'h8c: out = 8'h64; 8'h8d: out = 8'h5d; 8'h8e: out = 8'h19; 8'h8f: out = 8'h73;
            8'h90: out = 8'h60; 8'h91: out = 8'h81; 8'h92: out = 8'h4f; 8'h93: out = 8'hdc;
            8'h94: out = 8'h22; 8'h95: out = 8'h2a; 8'h96: out = 8'h90; 8'h97: out = 8'h88;
            8'h98: out = 8'h46; 8'h99: out = 8'hee; 8'h9a: out = 8'hb8; 8'h9b: out = 8'h14;
            8'h9c: out = 8'hde; 8'h9d: out = 8'h5e; 8'h9e: out = 8'h0b; 8'h9f: out = 8'hdb;
            8'ha0: out = 8'he0; 8'ha1: out = 8'h32; 8'ha2: out = 8'h3a; 8'ha3: out = 8'h0a;
            8'ha4: out = 8'h49; 8'ha5: out = 8'h06; 8'ha6: out = 8'h24; 8'ha7: out = 8'h5c;
            8'ha8: out = 8'hc2; 8'ha9: out = 8'hd3; 8'haa: out = 8'hac; 8'hab: out = 8'h62;
            8'hac: out = 8'h91; 8'had: out = 8'h95; 8'hae: out = 8'he4; 8'haf: out = 8'h79;
            8'hb0: out = 8'he7; 8'hb1: out = 8'hc8; 8'hb2: out = 8'h37; 8'hb3: out = 8'h6d;
            8'hb4: out = 8'h8d; 8'hb5: out = 8'hd5; 8'hb6: out = 8'h4e; 8'hb7: out = 8'ha9;
            8'hb8: out = 8'h6c; 8'hb9: out = 8'h56; 8'hba: out = 8'hf4; 8'hbb: out = 8'hea;
            8'hbc: out = 8'h65; 8'hbd: out = 8'h7a; 8'hbe: out = 8'hae; 8'hbf: out = 8'h08;
            8'hc0: out = 8'hba; 8'hc1: out = 8'h78; 8'hc2: out = 8'h25; 8'hc3: out = 8'h2e;
            8'hc4: out = 8'h1c; 8'hc5: out = 8'ha6; 8'hc6: out = 8'hb4; 8'hc7: out = 8'hc6;
            8'hc8: out = 8'he8; 8'hc9: out = 8'hdd; 8'hca: out = 8'h74; 8'hcb: out = 8'h1f;
            8'hcc: out = 8'h4b; 8'hcd: out = 8'hbd; 8'hce: out = 8'h8b; 8'hcf: out = 8'h8a;
            8'hd0: out = 8'h70; 8'hd1: out = 8'h3e; 8'hd2: out = 8'hb5; 8'hd3: out = 8'h66;
            8'hd4: out = 8'h48; 8'hd5: out = 8'h03; 8'hd6: out = 8'hf6; 8'hd7: out = 8'h0e;
            8'hd8: out = 8'h61; 8'hd9: out = 8'h35; 8'hda: out = 8'h57; 8'hdb: out = 8'hb9;
            8'hdc: out = 8'h86; 8'hdd: out = 8'hc1; 8'hde: out = 8'h1d; 8'hdf: out = 8'h9e;
            8'he0: out = 8'he1; 8'he1: out = 8'hf8; 8'he2: out = 8'h98; 8'he3: out = 8'h11;
            8'he4: out = 8'h69; 8'he5: out = 8'hd9; 8'he6: out = 8'h8e; 8'he7: out = 8'h94;
            8'he8: out = 8'h9b; 8'he9: out = 8'h1e; 8'hea: out = 8'h87; 8'heb: out = 8'he9;
            8'hec: out = 8'hce; 8'hed: out = 8'h55; 8'hee: out = 8'h28; 8'hef: out = 8'hdf;
            8'hf0: out = 8'h8c; 8'hf1: out = 8'ha1; 8'hf2: out = 8'h89; 8'hf3: out = 8'h0d;
            8'hf4: out = 8'hbf; 8'hf5: out = 8'he6; 8'hf6: out = 8'h42; 8'hf7: out = 8'h68;
            8'hf8: out = 8'h41; 8'hf9: out = 8'h99; 8'hfa: out = 8'h2d; 8'hfb: out = 8'h0f;
            8'hfc: out = 8'hb0; 8'hfd: out = 8'h54; 8'hfe: out = 8'hbb; 8'hff: out = 8'h16;
        endcase
        return out;
    endfunction
    
    // Inverse S-Box for decryption
    function automatic logic [7:0] inv_sbox(input logic [7:0] in);
        logic [7:0] out;
        case(in)
            8'h00: out = 8'h52; 8'h01: out = 8'h09; 8'h02: out = 8'h6a; 8'h03: out = 8'hd5;
            8'h04: out = 8'h30; 8'h05: out = 8'h36; 8'h06: out = 8'ha5; 8'h07: out = 8'h38;
            8'h08: out = 8'hbf; 8'h09: out = 8'h40; 8'h0a: out = 8'ha3; 8'h0b: out = 8'h9e;
            8'h0c: out = 8'h81; 8'h0d: out = 8'hf3; 8'h0e: out = 8'hd7; 8'h0f: out = 8'hfb;
            8'h10: out = 8'h7c; 8'h11: out = 8'he3; 8'h12: out = 8'h39; 8'h13: out = 8'h82;
            8'h14: out = 8'h9b; 8'h15: out = 8'h2f; 8'h16: out = 8'hff; 8'h17: out = 8'h87;
            8'h18: out = 8'h34; 8'h19: out = 8'h8e; 8'h1a: out = 8'h43; 8'h1b: out = 8'h44;
            8'h1c: out = 8'hc4; 8'h1d: out = 8'hde; 8'h1e: out = 8'he9; 8'h1f: out = 8'hcb;
            8'h20: out = 8'h54; 8'h21: out = 8'h7b; 8'h22: out = 8'h94; 8'h23: out = 8'h32;
            8'h24: out = 8'ha6; 8'h25: out = 8'hc2; 8'h26: out = 8'h23; 8'h27: out = 8'h3d;
            8'h28: out = 8'hee; 8'h29: out = 8'h4c; 8'h2a: out = 8'h95; 8'h2b: out = 8'h0b;
            8'h2c: out = 8'h42; 8'h2d: out = 8'hfa; 8'h2e: out = 8'hc3; 8'h2f: out = 8'h4e;
            8'h30: out = 8'h08; 8'h31: out = 8'h2e; 8'h32: out = 8'ha1; 8'h33: out = 8'h66;
            8'h34: out = 8'h28; 8'h35: out = 8'hd9; 8'h36: out = 8'h24; 8'h37: out = 8'hb2;
            8'h38: out = 8'h76; 8'h39: out = 8'h5b; 8'h3a: out = 8'ha2; 8'h3b: out = 8'h49;
            8'h3c: out = 8'h6d; 8'h3d: out = 8'h8b; 8'h3e: out = 8'hd1; 8'h3f: out = 8'h25;
            8'h40: out = 8'h72; 8'h41: out = 8'hf8; 8'h42: out = 8'hf6; 8'h43: out = 8'h64;
            8'h44: out = 8'h86; 8'h45: out = 8'h68; 8'h46: out = 8'h98; 8'h47: out = 8'h16;
            8'h48: out = 8'hd4; 8'h49: out = 8'ha4; 8'h4a: out = 8'h5c; 8'h4b: out = 8'hcc;
            8'h4c: out = 8'h5d; 8'h4d: out = 8'h65; 8'h4e: out = 8'hb6; 8'h4f: out = 8'h92;
            8'h50: out = 8'h6c; 8'h51: out = 8'h70; 8'h52: out = 8'h48; 8'h53: out = 8'h50;
            8'h54: out = 8'hfd; 8'h55: out = 8'hed; 8'h56: out = 8'hb9; 8'h57: out = 8'hda;
            8'h58: out = 8'h5e; 8'h59: out = 8'h15; 8'h5a: out = 8'h46; 8'h5b: out = 8'h57;
            8'h5c: out = 8'ha7; 8'h5d: out = 8'h8d; 8'h5e: out = 8'h9d; 8'h5f: out = 8'h84;
            8'h60: out = 8'h90; 8'h61: out = 8'hd8; 8'h62: out = 8'hab; 8'h63: out = 8'h00;
            8'h64: out = 8'h8c; 8'h65: out = 8'hbc; 8'h66: out = 8'hd3; 8'h67: out = 8'h0a;
            8'h68: out = 8'hf7; 8'h69: out = 8'he4; 8'h6a: out = 8'h58; 8'h6b: out = 8'h05;
            8'h6c: out = 8'hb8; 8'h6d: out = 8'hb3; 8'h6e: out = 8'h45; 8'h6f: out = 8'h06;
            8'h70: out = 8'hd0; 8'h71: out = 8'h2c; 8'h72: out = 8'h1e; 8'h73: out = 8'h8f;
            8'h74: out = 8'hca; 8'h75: out = 8'h3f; 8'h76: out = 8'h0f; 8'h77: out = 8'h02;
            8'h78: out = 8'hc1; 8'h79: out = 8'haf; 8'h7a: out = 8'hbd; 8'h7b: out = 8'h03;
            8'h7c: out = 8'h01; 8'h7d: out = 8'h13; 8'h7e: out = 8'h8a; 8'h7f: out = 8'h6b;
            8'h80: out = 8'h3a; 8'h81: out = 8'h91; 8'h82: out = 8'h11; 8'h83: out = 8'h41;
            8'h84: out = 8'h4f; 8'h85: out = 8'h67; 8'h86: out = 8'hdc; 8'h87: out = 8'hea;
            8'h88: out = 8'h97; 8'h89: out = 8'hf2; 8'h8a: out = 8'hcf; 8'h8b: out = 8'hce;
            8'h8c: out = 8'hf0; 8'h8d: out = 8'hb4; 8'h8e: out = 8'he6; 8'h8f: out = 8'h73;
            8'h90: out = 8'h96; 8'h91: out = 8'hac; 8'h92: out = 8'h74; 8'h93: out = 8'h22;
            8'h94: out = 8'he7; 8'h95: out = 8'had; 8'h96: out = 8'h35; 8'h97: out = 8'h85;
            8'h98: out = 8'he2; 8'h99: out = 8'hf9; 8'h9a: out = 8'h37; 8'h9b: out = 8'he8;
            8'h9c: out = 8'h1c; 8'h9d: out = 8'h75; 8'h9e: out = 8'hdf; 8'h9f: out = 8'h6e;
            8'ha0: out = 8'h47; 8'ha1: out = 8'hf1; 8'ha2: out = 8'h1a; 8'ha3: out = 8'h71;
            8'ha4: out = 8'h1d; 8'ha5: out = 8'h29; 8'ha6: out = 8'hc5; 8'ha7: out = 8'h89;
            8'ha8: out = 8'h6f; 8'ha9: out = 8'hb7; 8'haa: out = 8'h62; 8'hab: out = 8'h0e;
            8'hac: out = 8'haa; 8'had: out = 8'h18; 8'hae: out = 8'hbe; 8'haf: out = 8'h1b;
            8'hb0: out = 8'hfc; 8'hb1: out = 8'h56; 8'hb2: out = 8'h3e; 8'hb3: out = 8'h4b;
            8'hb4: out = 8'hc6; 8'hb5: out = 8'hd2; 8'hb6: out = 8'h79; 8'hb7: out = 8'h20;
            8'hb8: out = 8'h9a; 8'hb9: out = 8'hdb; 8'hba: out = 8'hc0; 8'hbb: out = 8'hfe;
            8'hbc: out = 8'h78; 8'hbd: out = 8'hcd; 8'hbe: out = 8'h5a; 8'hbf: out = 8'hf4;
            8'hc0: out = 8'h1f; 8'hc1: out = 8'hdd; 8'hc2: out = 8'ha8; 8'hc3: out = 8'h33;
            8'hc4: out = 8'h88; 8'hc5: out = 8'h07; 8'hc6: out = 8'hc7; 8'hc7: out = 8'h31;
            8'hc8: out = 8'hb1; 8'hc9: out = 8'h12; 8'hca: out = 8'h10; 8'hcb: out = 8'h59;
            8'hcc: out = 8'h27; 8'hcd: out = 8'h80; 8'hce: out = 8'hec; 8'hcf: out = 8'h5f;
            8'hd0: out = 8'h60; 8'hd1: out = 8'h51; 8'hd2: out = 8'h7f; 8'hd3: out = 8'ha9;
            8'hd4: out = 8'h19; 8'hd5: out = 8'hb5; 8'hd6: out = 8'h4a; 8'hd7: out = 8'h0d;
            8'hd8: out = 8'h2d; 8'hd9: out = 8'he5; 8'hda: out = 8'h7a; 8'hdb: out = 8'h9f;
            8'hdc: out = 8'h93; 8'hdd: out = 8'hc9; 8'hde: out = 8'h9c; 8'hdf: out = 8'hef;
            8'he0: out = 8'ha0; 8'he1: out = 8'he0; 8'he2: out = 8'h3b; 8'he3: out = 8'h4d;
            8'he4: out = 8'hae; 8'he5: out = 8'h2a; 8'he6: out = 8'hf5; 8'he7: out = 8'hb0;
            8'he8: out = 8'hc8; 8'he9: out = 8'heb; 8'hea: out = 8'hbb; 8'heb: out = 8'h3c;
            8'hec: out = 8'h83; 8'hed: out = 8'h53; 8'hee: out = 8'h99; 8'hef: out = 8'h61;
            8'hf0: out = 8'h17; 8'hf1: out = 8'h2b; 8'hf2: out = 8'h04; 8'hf3: out = 8'h7e;
            8'hf4: out = 8'hba; 8'hf5: out = 8'h77; 8'hf6: out = 8'hd6; 8'hf7: out = 8'h26;
            8'hf8: out = 8'he1; 8'hf9: out = 8'h69; 8'hfa: out = 8'h14; 8'hfb: out = 8'h63;
            8'hfc: out = 8'h55; 8'hfd: out = 8'h21; 8'hfe: out = 8'h0c; 8'hff: out = 8'h7d;
        endcase
        return out;
    endfunction
    
    // -------------------------------------------------------------------------
    // ShiftRows transformation
    // -------------------------------------------------------------------------
    
    function automatic logic [127:0] shift_rows(input logic [127:0] state_in);
        logic [127:0] state_out;
        // Row 0: no shift
        state_out[127:120] = state_in[127:120];
        state_out[95:88]   = state_in[95:88];
        state_out[63:56]   = state_in[63:56];
        state_out[31:24]   = state_in[31:24];
        // Row 1: shift left by 1
        state_out[119:112] = state_in[87:80];
        state_out[87:80]   = state_in[55:48];
        state_out[55:48]   = state_in[23:16];
        state_out[23:16]   = state_in[119:112];
        // Row 2: shift left by 2
        state_out[111:104] = state_in[47:40];
        state_out[79:72]   = state_in[15:8];
        state_out[47:40]   = state_in[111:104];
        state_out[15:8]    = state_in[79:72];
        // Row 3: shift left by 3
        state_out[103:96]  = state_in[7:0];
        state_out[71:64]   = state_in[103:96];
        state_out[39:32]   = state_in[71:64];
        state_out[7:0]     = state_in[39:32];
        return state_out;
    endfunction
    
    // Inverse ShiftRows
    function automatic logic [127:0] inv_shift_rows(input logic [127:0] state_in);
        logic [127:0] state_out;
        // Row 0: no shift
        state_out[127:120] = state_in[127:120];
        state_out[95:88]   = state_in[95:88];
        state_out[63:56]   = state_in[63:56];
        state_out[31:24]   = state_in[31:24];
        // Row 1: shift right by 1 (left by 3)
        state_out[119:112] = state_in[23:16];
        state_out[87:80]   = state_in[119:112];
        state_out[55:48]   = state_in[87:80];
        state_out[23:16]   = state_in[55:48];
        // Row 2: shift right by 2 (left by 2)
        state_out[111:104] = state_in[47:40];
        state_out[79:72]   = state_in[15:8];
        state_out[47:40]   = state_in[111:104];
        state_out[15:8]    = state_in[79:72];
        // Row 3: shift right by 3 (left by 1)
        state_out[103:96]  = state_in[71:64];
        state_out[71:64]   = state_in[39:32];
        state_out[39:32]   = state_in[7:0];
        state_out[7:0]     = state_in[103:96];
        return state_out;
    endfunction
    
    // -------------------------------------------------------------------------
    // GF(2^8) multiplication for MixColumns
    // -------------------------------------------------------------------------
    
        function automatic [7:0] gf_mult;
            input [7:0] a;
            input [7:0] b;
            reg [7:0] p;
            reg [7:0] temp_a;
            reg [7:0] temp_b;
            reg hi_bit_set;
            integer i;
            begin
                p = 8'h00;
                temp_a = a;
                temp_b = b;
            
                for (i = 0; i < 8; i = i + 1) begin
                    if (temp_b[0])
                        p = p ^ temp_a;
                
                    hi_bit_set = temp_a[7];
                    temp_a = temp_a << 1;
                
                    if (hi_bit_set)
                        temp_a = temp_a ^ 8'h1b; // AES irreducible polynomial
                
                    temp_b = temp_b >> 1;
                end
            
                gf_mult = p;
            end
        endfunction
    
    // -------------------------------------------------------------------------
    // MixColumns transformation
    // -------------------------------------------------------------------------
    
    function automatic logic [31:0] mix_column(input logic [31:0] col);
        logic [7:0] a0, a1, a2, a3;
        logic [7:0] b0, b1, b2, b3;
        
        a0 = col[31:24];
        a1 = col[23:16];
        a2 = col[15:8];
        a3 = col[7:0];
        
        b0 = gf_mult(8'h02, a0) ^ gf_mult(8'h03, a1) ^ a2 ^ a3;
        b1 = a0 ^ gf_mult(8'h02, a1) ^ gf_mult(8'h03, a2) ^ a3;
        b2 = a0 ^ a1 ^ gf_mult(8'h02, a2) ^ gf_mult(8'h03, a3);
        b3 = gf_mult(8'h03, a0) ^ a1 ^ a2 ^ gf_mult(8'h02, a3);
        
        return {b0, b1, b2, b3};
    endfunction
    
    function automatic logic [127:0] mix_columns(input logic [127:0] state_in);
        logic [127:0] state_out;
        state_out[127:96] = mix_column(state_in[127:96]);
        state_out[95:64]  = mix_column(state_in[95:64]);
        state_out[63:32]  = mix_column(state_in[63:32]);
        state_out[31:0]   = mix_column(state_in[31:0]);
        return state_out;
    endfunction
    
    // Inverse MixColumns
    function automatic logic [31:0] inv_mix_column(input logic [31:0] col);
        logic [7:0] a0, a1, a2, a3;
        logic [7:0] b0, b1, b2, b3;
        
        a0 = col[31:24];
        a1 = col[23:16];
        a2 = col[15:8];
        a3 = col[7:0];
        
        b0 = gf_mult(8'h0e, a0) ^ gf_mult(8'h0b, a1) ^ gf_mult(8'h0d, a2) ^ gf_mult(8'h09, a3);
        b1 = gf_mult(8'h09, a0) ^ gf_mult(8'h0e, a1) ^ gf_mult(8'h0b, a2) ^ gf_mult(8'h0d, a3);
        b2 = gf_mult(8'h0d, a0) ^ gf_mult(8'h09, a1) ^ gf_mult(8'h0e, a2) ^ gf_mult(8'h0b, a3);
        b3 = gf_mult(8'h0b, a0) ^ gf_mult(8'h0d, a1) ^ gf_mult(8'h09, a2) ^ gf_mult(8'h0e, a3);
        
        return {b0, b1, b2, b3};
    endfunction
    
    function automatic logic [127:0] inv_mix_columns(input logic [127:0] state_in);
        logic [127:0] state_out;
        state_out[127:96] = inv_mix_column(state_in[127:96]);
        state_out[95:64]  = inv_mix_column(state_in[95:64]);
        state_out[63:32]  = inv_mix_column(state_in[63:32]);
        state_out[31:0]   = inv_mix_column(state_in[31:0]);
        return state_out;
    endfunction
    
    // -------------------------------------------------------------------------
    // SubBytes transformation
    // -------------------------------------------------------------------------
    
    function automatic logic [127:0] sub_bytes(input logic [127:0] state_in);
        logic [127:0] state_out;
        integer i;
        for (i = 0; i < 16; i = i + 1) begin
            state_out[i*8 +: 8] = sbox(state_in[i*8 +: 8]);
        end
        return state_out;
    endfunction
    
    function automatic logic [127:0] inv_sub_bytes(input logic [127:0] state_in);
        logic [127:0] state_out;
        integer i;
        for (i = 0; i < 16; i = i + 1) begin
            state_out[i*8 +: 8] = inv_sbox(state_in[i*8 +: 8]);
        end
        return state_out;
    endfunction
    
    // -------------------------------------------------------------------------
    // Key Expansion
    // -------------------------------------------------------------------------
    
    logic [31:0] rcon [0:14];
    initial begin
        rcon[0]  = 32'h01000000;
        rcon[1]  = 32'h02000000;
        rcon[2]  = 32'h04000000;
        rcon[3]  = 32'h08000000;
        rcon[4]  = 32'h10000000;
        rcon[5]  = 32'h20000000;
        rcon[6]  = 32'h40000000;
        rcon[7]  = 32'h80000000;
        rcon[8]  = 32'h1b000000;
        rcon[9]  = 32'h36000000;
        rcon[10] = 32'h6c000000;
        rcon[11] = 32'hd8000000;
        rcon[12] = 32'hab000000;
        rcon[13] = 32'h4d000000;
        rcon[14] = 32'h9a000000;
    end
    
    function automatic logic [31:0] rot_word(input logic [31:0] word);
        return {word[23:0], word[31:24]};
    endfunction
    
    function automatic logic [31:0] sub_word(input logic [31:0] word);
        logic [31:0] result;
        result[31:24] = sbox(word[31:24]);
        result[23:16] = sbox(word[23:16]);
        result[15:8]  = sbox(word[15:8]);
        result[7:0]   = sbox(word[7:0]);
        return result;
    endfunction
    
    // -------------------------------------------------------------------------
    // State Machine
    // -------------------------------------------------------------------------
    
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            round_cnt <= 0;
            state_reg <= 128'h0;
            key_exp_cnt <= 0;
            for (int i = 0; i < 60; i = i + 1)
                round_keys[i] <= 32'h0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        state_reg <= data_in;
                        round_cnt <= 0;
                        key_exp_cnt <= 0;
                        
                        // Initialize first 8 words from key
                        round_keys[0] <= key_in[255:224];
                        round_keys[1] <= key_in[223:192];
                        round_keys[2] <= key_in[191:160];
                        round_keys[3] <= key_in[159:128];
                        round_keys[4] <= key_in[127:96];
                        round_keys[5] <= key_in[95:64];
                        round_keys[6] <= key_in[63:32];
                        round_keys[7] <= key_in[31:0];
                    end
                end
                
                KEY_EXPANSION: begin
                    // Generate round keys
                    if (key_exp_cnt < 52) begin
                        key_idx = key_exp_cnt + 8;
                        
                        if ((key_idx % 8) == 0) begin
                            key_temp = sub_word(rot_word(round_keys[key_idx-1])) ^ rcon[key_idx/8 - 1];
                            round_keys[key_idx] <= round_keys[key_idx-8] ^ key_temp;
                        end else if ((key_idx % 8) == 4) begin
                            key_temp = sub_word(round_keys[key_idx-1]);
                            round_keys[key_idx] <= round_keys[key_idx-8] ^ key_temp;
                        end else begin
                            round_keys[key_idx] <= round_keys[key_idx-8] ^ round_keys[key_idx-1];
                        end
                        
                        key_exp_cnt <= key_exp_cnt + 1;
                    end
                end
                
                INIT_ROUND: begin
                    // AddRoundKey with first round key
                    state_reg <= state_reg ^ {round_keys[0], round_keys[1], round_keys[2], round_keys[3]};
                end
                
                MAIN_ROUNDS: begin
                    if (mode == 0) begin // Encrypt
                        // SubBytes -> ShiftRows -> MixColumns -> AddRoundKey
                        fsm_temp = sub_bytes(state_reg);
                        fsm_temp = shift_rows(fsm_temp);
                        fsm_temp = mix_columns(fsm_temp);
                        state_reg <= fsm_temp ^ {round_keys[round_cnt*4+4], round_keys[round_cnt*4+5], 
                                            round_keys[round_cnt*4+6], round_keys[round_cnt*4+7]};
                    end else begin // Decrypt
                        // InvShiftRows -> InvSubBytes -> AddRoundKey -> InvMixColumns
                        fsm_temp = inv_shift_rows(state_reg);
                        fsm_temp = inv_sub_bytes(fsm_temp);
                        fsm_temp = fsm_temp ^ {round_keys[(13-round_cnt)*4+4], round_keys[(13-round_cnt)*4+5],
                                       round_keys[(13-round_cnt)*4+6], round_keys[(13-round_cnt)*4+7]};
                        state_reg <= inv_mix_columns(fsm_temp);
                    end
                    
                    if (round_cnt < 12)
                        round_cnt <= round_cnt + 1;
                end
                
                FINAL_ROUND: begin
                    if (mode == 0) begin // Encrypt
                        // SubBytes -> ShiftRows -> AddRoundKey
                        fsm_temp = sub_bytes(state_reg);
                        fsm_temp = shift_rows(fsm_temp);
                        state_reg <= fsm_temp ^ {round_keys[56], round_keys[57], round_keys[58], round_keys[59]};
                    end else begin // Decrypt
                        // InvShiftRows -> InvSubBytes -> AddRoundKey
                        fsm_temp = inv_shift_rows(state_reg);
                        fsm_temp = inv_sub_bytes(fsm_temp);
                        state_reg <= fsm_temp ^ {round_keys[0], round_keys[1], round_keys[2], round_keys[3]};
                    end
                end
                
                DONE: begin
                    // Output is ready
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
                    next_state = KEY_EXPANSION;
            end
            
            KEY_EXPANSION: begin
                if (key_exp_cnt >= 52)
                    next_state = INIT_ROUND;
            end
            
            INIT_ROUND: begin
                next_state = MAIN_ROUNDS;
            end
            
            MAIN_ROUNDS: begin
                if (round_cnt >= 12)
                    next_state = FINAL_ROUND;
            end
            
            FINAL_ROUND: begin
                next_state = DONE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Output assignments
    assign data_out = state_reg;
    assign data_valid = (state == DONE);
    assign busy = (state != IDLE) && (state != DONE);
    
endmodule
