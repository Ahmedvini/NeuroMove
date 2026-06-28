// =============================================================================
// db_atcnet_ps_demo.c  —  PS-side reference driver for the DB-ATCNet
// deployment block (rtl/db_atcnet_axi.sv).
//
// Build context: Vitis bare-metal application running on Zynq PS7 / Zynq
// Ultrascale+ APU.  Assumes the IP is mapped at PL_BASE in the PS memory
// map and connected to a high-performance AXI-Stream master (for raw EEG
// data) plus an AXI-Lite slave port (for weight load + status mailbox).
//
// Address map (matches rtl/db_atcnet_axi.sv defaults):
//   0x0000_0000 .. 0x0000_1FFF   weight bank writes (broadcast)
//   0x0000_2000 .. 0x0000_200B   Gumbel 19→5 LUT (3 × 32-bit words)
//   0x0000_F000                  STATUS    [1]=done, [0]=busy
//   0x0000_F004                  CLASS     [0]=class (0=Right Hand, 1=Left Leg)
//
// Workflow per inference:
//   1. After power-up: load weights + LUT once (see load_weights_from_mem).
//   2. For each trial:
//      - Push 19 × 663 raw EEG samples over AXI-Stream
//        (PAD_TOP=31 + T_IN=600 + PAD_BOT=32 = 663 timesteps × 19 electrodes).
//      - Poll STATUS until done=1 (or wait on irq_done — preferred).
//      - Read CLASS register → 1-bit prediction.
//      - Drive the servo / GPIO accordingly.
//
// On Z-7020 with PL clock at 100 MHz, a single inference takes ~1.15 ms of
// compute + AXIS streaming overhead (typically <2 ms total for a 600-sample
// 19-electrode trial).
// =============================================================================

#include <stdint.h>
#include <stddef.h>
#include "xparameters.h"
#include "xil_io.h"

// IP base address — point at the AXI-Lite slave of db_atcnet_axi.
#define DB_ATCNET_BASE        XPAR_DB_ATCNET_AXI_0_S_AXI_BASEADDR
#define DB_ATCNET_AXIS_DMA    XPAR_AXI_DMA_0_BASEADDR    // separate AXI-DMA streams to S_AXIS

// Register offsets (within the AXI-Lite slave window).
#define REG_STATUS            0xF000
#define REG_CLASS             0xF004

// Weight bank base addresses (must match the BASE_ADDR parameter of each
// weight_bank instantiated inside db_atcnet_top — TODO: declare these
// uniformly when the per-block AXI-Lite refactor lands).
#define WB_CONV2D             0x00000000   // 1040 entries × 2B = 2080 B
#define WB_ECA1               0x00000840   // 3 entries (one 32-bit word)
#define WB_BRANCHA_DWISE      0x00000850   // 192 entries
#define WB_BRANCHA_SEP        0x00000B50   // 16416 entries  (≈ 32 KB)
#define WB_BRANCHB_DWISE      0x00010B58   // 384 entries
#define WB_BRANCHB_SEP        0x00010E58   // 32800 entries  (≈ 64 KB)
#define WB_ECA2               0x00020F58   // 3 entries
#define WB_CHAN_D1W           0x00020F70   // 640 entries
#define WB_CHAN_D1B           0x00021470   // 20  entries
#define WB_CHAN_D2W           0x00021498   // 640 entries
#define WB_CHAN_D2B           0x00021998   // 160 entries
#define WB_SPAT_CONV_W        0x00021A38   // 105 entries
#define WB_TCFN_W0            0x00021AD8   // 20480 entries (× 4 banks)
#define WB_TCFN_B0            0x00031AE0   //   160 entries (× 4 banks)
#define WB_DENSE_W            0x00031C00   // 320 entries
#define WB_DENSE_B            0x00031D40   // 10  entries
#define LUT_SIG               0x00031D60   // 1024 entries
#define LUT_ELU               0x00032560   // 256  entries
#define GUMBEL_LUT            0x00002000   // 5  electrode indices

// Pre-flashed weight & LUT image bundled with the application binary.
// Build script `scripts/pack_weights_axil.py` converts the .hex files in
// data/golden_q88/ into a single packed image, with each section aligned to
// the offsets above.
extern const uint32_t kModelImageStart;
extern const uint32_t kModelImageEnd;
extern const uint32_t kModelImageOffsets[];  // [base, len] pairs

static inline uint32_t db_read(uint32_t off) {
    return Xil_In32(DB_ATCNET_BASE + off);
}
static inline void db_write(uint32_t off, uint32_t val) {
    Xil_Out32(DB_ATCNET_BASE + off, val);
}

// ---------------------------------------------------------------------------
// Bulk weight load: walk the embedded image and broadcast each 32-bit word
// at its assigned base address. The IP's axi_lite_weight_loader fan-out
// reaches every weight_bank / gumbel-LUT in a single pass.
// ---------------------------------------------------------------------------
void db_atcnet_load_weights(void) {
    const uint32_t* img = (const uint32_t*)&kModelImageStart;
    const uint32_t* end = (const uint32_t*)&kModelImageEnd;
    // The packing layout matches kModelImageOffsets[] entries:
    //   pair i:  { base_addr, len_words } — `len_words` 32-bit words follow.
    size_t i = 0;
    while (kModelImageOffsets[i] != 0xFFFFFFFFu) {
        uint32_t base = kModelImageOffsets[i++];
        uint32_t len  = kModelImageOffsets[i++];
        for (uint32_t w = 0; w < len; w++) {
            db_write(base + w * 4, *img++);
        }
    }
    (void)end;
}

// ---------------------------------------------------------------------------
// Per-trial inference: stream raw EEG via AXI-DMA, poll STATUS, return class.
// `eeg_samples` is a contiguous array of 19 × 663 packed (electrode | sample)
// 32-bit beats (see rtl/db_atcnet_axi.sv tdata layout).
// Returns 0 on success, -1 on timeout.
// ---------------------------------------------------------------------------
int db_atcnet_infer(const uint32_t* eeg_samples, size_t n_beats, uint8_t* class_out) {
    // Kick the AXI-DMA to push n_beats × 4 bytes of eeg_samples to S_AXIS.
    // (PS-driver call — abbreviated, see Xilinx XAxiDma API.)
    extern int xaxidma_simple_transfer(uint32_t base, const void* src, size_t bytes);
    int rc = xaxidma_simple_transfer(DB_ATCNET_AXIS_DMA, eeg_samples, n_beats * 4);
    if (rc != 0) return rc;

    // Poll STATUS until done bit is set, with a generous timeout (~10 ms at
    // 100 MHz APU = 1 M reads).
    for (int t = 0; t < 1000000; t++) {
        uint32_t st = db_read(REG_STATUS);
        if (st & 0x2) {
            *class_out = (uint8_t)(db_read(REG_CLASS) & 0x1);
            return 0;
        }
    }
    return -1;  // timeout
}

// ---------------------------------------------------------------------------
// Servo helper (PYNQ-style PWM via TTC). Drives a hobby servo to one of two
// preset angles based on the predicted class.
// ---------------------------------------------------------------------------
void db_atcnet_drive_servo(uint8_t pred_class) {
    extern void servo_set_angle_deg(int deg);
    if (pred_class == 0) {
        servo_set_angle_deg(90);   // Right Hand fist → mid (or whatever the
                                   // demo's "grip" pose is)
    } else {
        servo_set_angle_deg(0);    // Left Leg ankle  → release
    }
}

// ---------------------------------------------------------------------------
// Reference main() — single inference followed by servo drive.
// ---------------------------------------------------------------------------
int main(void) {
    db_atcnet_load_weights();

    // EEG buffer prepared by an upstream ADC driver (OpenBCI / ADS1115 / etc).
    extern uint32_t g_eeg_buffer[];
    extern size_t   g_eeg_buffer_beats;

    uint8_t cls = 0;
    int rc = db_atcnet_infer(g_eeg_buffer, g_eeg_buffer_beats, &cls);
    if (rc != 0) {
        return rc;
    }

    db_atcnet_drive_servo(cls);
    return 0;
}
