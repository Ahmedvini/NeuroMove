// EEG Dataset Configuration and Reader
// This module provides configuration and interface for reading EEG datasets
// that will be encrypted using the AES-256-GCM security module

`timescale 1ns / 1ps

module eeg_dataset_reader #(
    parameter DATA_WIDTH = 128,  // Width of data blocks to encrypt
    parameter DATASET_DEPTH = 65535  // EEG_MotorMovement-Imagery_Dataset (64ch @ 160Hz)
)(
    input  wire clk,
    input  wire rst_n,
    
    // Control signals
    input  wire start_read,
    output reg  data_valid,
    output reg  read_complete,
    
    // Data output to encryption module
    output reg [DATA_WIDTH-1:0] eeg_data,
    output reg [31:0] data_length,
    
    // Status
    output reg [31:0] samples_read
);

    // ==========================================================================
    // DATASET CONFIGURATION
    // ==========================================================================
    // TODO: Add your dataset path here when available
    // Example dataset link (commented out - replace with your actual dataset):
    // Dataset: [YOUR_DATASET_LINK_HERE]
    // Format: CSV, EDF, or binary format
    // Expected columns: timestamp, channel1, channel2, ..., channelN
    // ==========================================================================
    
    // Dataset parameters (configure these based on your dataset)
    localparam NUM_CHANNELS    = 64;   // EEG channels
    localparam SAMPLE_RATE     = 160;  // Hz — PhysioNet Motor Movement/Imagery
    localparam BITS_PER_SAMPLE = 16;   // Bits per sample per channel
    
    // Memory for dataset (will be loaded from file)
    reg [DATA_WIDTH-1:0] dataset_memory [0:DATASET_DEPTH-1];
    
    // State machine
    typedef enum logic [1:0] {
        IDLE,
        READING,
        DONE
    } state_t;
    
    state_t current_state, next_state;
    
    // Internal counters
    reg [31:0] read_address;
    
    // ==========================================================================
    // State Machine
    // ==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE: begin
                if (start_read)
                    next_state = READING;
            end
            
            READING: begin
                if (read_address >= DATASET_DEPTH - 1)
                    next_state = DONE;
            end
            
            DONE: begin
                if (!start_read)
                    next_state = IDLE;
            end
        endcase
    end
    
    // ==========================================================================
    // Read Logic
    // ==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_address <= 0;
            data_valid <= 0;
            read_complete <= 0;
            eeg_data <= 0;
            data_length <= 0;
            samples_read <= 0;
        end else begin
            case (current_state)
                IDLE: begin
                    read_address <= 0;
                    data_valid <= 0;
                    read_complete <= 0;
                    samples_read <= 0;
                    data_length <= DATA_WIDTH / 8;  // Length in bytes
                end
                
                READING: begin
                    // Read data from memory
                    eeg_data <= dataset_memory[read_address];
                    data_valid <= 1;
                    samples_read <= read_address + 1;
                    read_address <= read_address + 1;
                end
                
                DONE: begin
                    data_valid <= 0;
                    read_complete <= 1;
                end
            endcase
        end
    end
    
    // ==========================================================================
    // Dataset Loading (simulation only)
    // ==========================================================================
    // *** THIS IS WHERE YOU LOAD YOUR EEG DATASET ***
    //
    // STEP 1: Convert your EDF dataset to hex format (run on Linux from project root):
    //   cd ~/Documents/WITHSEC/fpga-atcnet       (adjust to your path)
    //   python3 scripts/edf_to_hex.py EEG_MotorMovement-Imagery_Dataset/
    //
    //   Output will be written to: data/dataset.hex
    //   Update DATASET_DEPTH and localparam values from data/dataset_info.txt
    //
    // STEP 2: Uncomment the $readmemh line below.
    //
    // FORMAT REQUIREMENTS:
    //   - Each line = one 128-bit (32 hex char) EEG sample
    //   - Big-endian byte order
    //   - No 0x prefix, no spaces within a line
    //   - Up to DATASET_DEPTH (1024) lines
    //
    // EXAMPLE line in dataset.hex:
    //   00FF12A3B4C5D6E7F801234567890ABC
    // ==========================================================================
    initial begin
        // ---------------------------------------------------------------
        // Dataset: EEG_MotorMovement-Imagery_Dataset (64ch, 160Hz, 109 subjects)
        $readmemh("data/dataset.hex", dataset_memory);
        // ---------------------------------------------------------------

        // Default: synthetic sine-wave-like test pattern
        // Remove these lines once you load your real dataset
        for (int i = 0; i < DATASET_DEPTH; i++) begin
            // 8 channels x 16-bit samples packed into 128 bits
            // Values sweep from 0x0000 to 0x7FFF simulating EEG amplitude
            dataset_memory[i] = {
                16'(i * 7),   16'(i * 13),
                16'(i * 17),  16'(i * 23),
                16'(i * 29),  16'(i * 31),
                16'(i * 37),  16'(i * 41)
            };
        end

        $display("EEG Dataset Reader: Loaded %0d synthetic samples", DATASET_DEPTH);
        $display("  Replace with: $readmemh(\"data/dataset.hex\", dataset_memory)");
    end

endmodule
