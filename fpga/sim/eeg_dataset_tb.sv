

`timescale 1ns / 1ps

module eeg_dataset_tb;

    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Encryption key (256-bit) - REPLACE WITH YOUR SECURE KEY
    reg [255:0] encryption_key = 256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f;
    
    // Nonce seed (96-bit) - Can be derived from timestamp
    reg [95:0] nonce_seed = 96'h000000000000000000000001;
    
    // Dataset reader signals
    reg start_read;
    wire data_valid;
    wire read_complete;
    wire [127:0] eeg_data;
    wire [31:0] data_length;
    wire [31:0] samples_read;
    
    // Encryption signals
    reg encrypt_enable;
    wire encrypt_ready;
    wire encrypt_done;
    wire [95:0] storage_nonce;
    wire [127:0] encrypted_data;
    wire [127:0] auth_tag;
    wire tag_valid;
    
    // ==========================================================================
    // DATASET CONFIGURATION
    // ==========================================================================
    // TODO: When you attach your dataset, uncomment and configure these:
    // Dataset path: [YOUR_DATASET_LINK_HERE]
    // Dataset format: CSV/EDF/Binary
    // Number of samples: [CONFIGURE_HERE]
    // Number of channels: [CONFIGURE_HERE]
    // Sampling rate: [CONFIGURE_HERE] Hz
    // ==========================================================================
    
    // Clock generation (100 MHz)
    always #5 clk = ~clk;
    
    // Dataset reader instance
    eeg_dataset_reader #(
        .DATA_WIDTH(128),
        .DATASET_DEPTH(1024)
    ) dataset_reader (
        .clk(clk),
        .rst_n(rst_n),
        .start_read(start_read),
        .data_valid(data_valid),
        .read_complete(read_complete),
        .eeg_data(eeg_data),
        .data_length(data_length),
        .samples_read(samples_read)
    );
    
    // EEG data encryptor instance
    eeg_data_encryptor #(
        .DATA_WIDTH(128)
    ) encryptor (
        .clk(clk),
        .rst_n(rst_n),
        .key(encryption_key),
        .nonce_seed(nonce_seed),
        .eeg_data(eeg_data),
        .data_length(data_length),
        .data_valid(data_valid & encrypt_enable),
        .encrypt_enable(data_valid & encrypt_enable),
        .ready(encrypt_ready),
        .done(encrypt_done),
        .storage_nonce(storage_nonce),
        .data_out(encrypted_data),
        .storage_tag(auth_tag),
        .tag_valid(tag_valid)
    );
    
    // Storage for encrypted data
    reg [95:0] stored_nonces [0:1023];
    reg [127:0] stored_data [0:1023];
    reg [127:0] stored_tags [0:1023];
    integer encrypted_count;
    
    // Test procedure
    initial begin
        // Initialize
        clk = 0;
        rst_n = 0;
        start_read = 0;
        encrypt_enable = 0;
        encrypted_count = 0;
        
        // Display configuration
        $display("==========================================================================");
        $display("EEG Dataset Encryption Test");
        $display("==========================================================================");
        $display("Waiting for dataset configuration...");
        $display("TODO: Configure dataset path in eeg_dataset_config.sv");
        $display("==========================================================================");
        
        // Reset
        #100;
        rst_n = 1;
        #100;
        
        // Wait for encryption module to be ready
        wait(encrypt_ready);
        $display("[%0t] Encryption module ready", $time);
        
        // Start reading and encrypting dataset
        $display("[%0t] Starting dataset encryption...", $time);
        start_read = 1;
        encrypt_enable = 1;
        
        // Monitor encryption process
        fork
            // Monitor data reading
            begin
                while (!read_complete) begin
                    @(posedge clk);
                    if (data_valid) begin
                        $display("[%0t] Reading sample %0d: 0x%h", 
                                $time, samples_read, eeg_data);
                    end
                end
                $display("[%0t] Dataset reading complete. Total samples: %0d", 
                        $time, samples_read);
            end
            
            // Monitor encryption
            begin
                while (!read_complete) begin
                    @(posedge clk);
                    if (tag_valid) begin
                        // Store encrypted data
                        stored_nonces[encrypted_count] = storage_nonce;
                        stored_data[encrypted_count] = encrypted_data;
                        stored_tags[encrypted_count] = auth_tag;
                        
                        $display("[%0t] Encrypted sample %0d:", $time, encrypted_count);
                        $display("  Nonce: 0x%h", storage_nonce);
                        $display("  Data:  0x%h", encrypted_data);
                        $display("  Tag:   0x%h", auth_tag);
                        
                        encrypted_count = encrypted_count + 1;
                    end
                end
            end
        join
        
        // Summary
        #1000;
        $display("==========================================================================");
        $display("Encryption Complete");
        $display("==========================================================================");
        $display("Total samples encrypted: %0d", encrypted_count);
        $display("Storage overhead per sample: 28 bytes (12B nonce + 16B tag)");
        $display("Total storage needed: %0d bytes", encrypted_count * (16 + 28));
        $display("==========================================================================");
        

        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #10_000_000; // 10ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
