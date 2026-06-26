// =============================================================================
// db_atcnet_esp32.ino  —  ESP32-S3 motor controller for the DB-ATCNet BCI.
//
// Target board: ESP32-S3-DevKitC-1 (or any ESP32-S3 module). Arduino IDE
// board target = "ESP32S3 Dev Module". Pin choices below are S3-safe
// (no strapping pins, no USB-JTAG pins, no internal SPI/PSRAM pins).
//
// Receives one-byte classification results from the Zynq UltraScale+ PS over
// UART2 and drives two servos through a PCA9685:
//   - Class 0 ("HAND" task)  -> close/open hand fingers (CH 0, "hand" servo)
//   - Class 1 ("FOOT" task)  -> dorsiflex / plantarflex ankle (CH 1, "leg" servo)
//
// Safety features (all configurable in the constants below):
//   * Per-servo angle clamps (won't drive past mechanical limits)
//   * Rate limiter (smooth motion, max degrees/second)
//   * Soft start on power-up (rests at neutral before accepting commands)
//   * Majority-vote debouncing (won't act on a single noisy classification)
//   * UART watchdog (returns to safe rest if no FPGA data for N ms)
//   * Test mode: type "T0\n" / "T1\n" on USB Serial to simulate classifications
//   * Heartbeat LED (slow blink = ok, fast = fault, off = boot)
//
// Wiring (ESP32-S3 dev kit):
//   ESP32-S3 GPIO8  (SDA)      <-> PCA9685 SDA, ADS1115 SDA  (I2C bus)
//   ESP32-S3 GPIO9  (SCL)      <-> PCA9685 SCL, ADS1115 SCL  (I2C bus)
//   ESP32-S3 GPIO17 (UART2 TX) <-> Zynq PS UART1 RX (PL pin A20 / CP2108 ch.2)
//   ESP32-S3 GPIO18 (UART2 RX) <-> Zynq PS UART1 TX (PL pin C19 / CP2108 ch.2)
//   ESP32-S3 GPIO48 (LED)      built-in WS2812 / heartbeat on most S3 dev kits
//   PCA9685 V+ <- 6 V (servo buck), CH0 -> hand, CH1 -> leg
//
// NOTE for porting back to classic ESP32 (NOT recommended; thesis targets S3):
//   I2C SDA/SCL = 21/22, UART2 RX/TX = 16/17, LED = 2.
// =============================================================================

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>     // fabsf() — explicit to keep us portable beyond Arduino-ESP32

// -----------------------------------------------------------------------------
// User-tunable constants
// -----------------------------------------------------------------------------

// I2C / PCA9685
constexpr uint8_t  PCA9685_ADDR     = 0x40;
constexpr uint8_t  CH_HAND          = 0;
constexpr uint8_t  CH_LEG           = 1;
constexpr uint32_t SERVO_FREQ_HZ    = 50;     // standard analog servo

// UART2 to Zynq PS. 460800 is a comfortable middle ground:
//   - Plenty of headroom for the 1-byte/inference return path.
//   - Enough throughput (~45 KB/s effective) for the 24 KB/s EEG-streaming
//     path if Architecture A (real-time bio-amp) is ever enabled.
//   - More tolerant of cheap jumper wires than 921600.
constexpr long     UART2_BAUD       = 460800;
constexpr int      UART2_RX_PIN     = 18;     // ESP32-S3 safe GPIO (was 16 on classic)
constexpr int      UART2_TX_PIN     = 17;     // ESP32-S3 safe GPIO (was 17 on classic)

// I2C bus shared by PCA9685 (servo PWM) and ADS1115 (analog front-end).
// Pin choices avoid every S3 strapping / USB / internal-SPI pin, so they
// work on the stock ESP32-S3-DevKitC-1 without re-soldering. 4.7 kΩ pull-ups
// to 3V3 on each line, populated on the PCA9685 / ADS1115 break-out boards.
constexpr int      I2C_SDA_PIN      = 8;      // ESP32-S3 safe GPIO (was 21 on classic)
constexpr int      I2C_SCL_PIN      = 9;      // ESP32-S3 safe GPIO (was 22 on classic)
constexpr uint32_t I2C_CLOCK_HZ     = 400000; // Fast-mode I2C

// Periodic re-assertion of the current committed target while in RUNNING.
// Cheap defense against PCA9685 channel state being out of sync after a
// transient I2C glitch. Set to 0 to disable.
constexpr uint32_t REASSERT_MS      = 500;

// Heartbeat / status LED. GPIO 48 is the on-board WS2812 RGB on most
// ESP32-S3-DevKitC-1 boards; if your S3 board has a plain LED on a
// different pin, change here. Setting LED_PIN = -1 disables the heartbeat.
constexpr int      LED_PIN          = 48;     // was 2 on classic ESP32 dev kits

// Mechanical angle limits per servo (degrees). Both prosthetic actuators move
// +45° from their rest position when the corresponding class is decided —
// keep the *_MAX_DEG at REST + 45 so the rate-limited servo lands cleanly on
// that target without overshoot.
constexpr int      HAND_REST_DEG    = 180;     // rest at max positive
constexpr int      HAND_MIN_DEG     = 0;       // full -180° stroke available
constexpr int      HAND_MAX_DEG     = 180;     // upper bound = rest
constexpr int      HAND_ACT_DEG     = 0;       // active at min — full -180° from rest

constexpr int      LEG_REST_DEG     = 70;      // ankle plantarflex (rest)
constexpr int      LEG_MIN_DEG      = 70;
constexpr int      LEG_MAX_DEG      = LEG_REST_DEG  + 45;  // ankle dorsiflex (+45° lift)

// Rate limiter: how fast a servo can slew. Lower = safer, smoother, slower.
// At 90°/s the hand takes ~0.5 s to reach +45°; at 60°/s the leg takes ~0.75 s.
constexpr float    HAND_RATE_DPS    = 90.0f;
constexpr float    LEG_RATE_DPS     = 60.0f;

// Classifier debouncing: hold a sliding window of the last N classifications,
// only commit a new target when ≥ MAJORITY_THRESHOLD match.
constexpr int      VOTE_WINDOW      = 5;
constexpr int      VOTE_THRESHOLD   = 4;      // need 4-of-5 same class

// Watchdog: if no FPGA byte arrives for this many ms, return to rest.
constexpr uint32_t UART_TIMEOUT_MS  = 1500;

// -----------------------------------------------------------------------------
// PCA9685 pulse timing helpers (4096 ticks per 20 ms period at 50 Hz)
// -----------------------------------------------------------------------------
//   0.5 ms ->  102 ticks  -> 0°
//   1.5 ms ->  307 ticks  -> 90°
//   2.5 ms ->  512 ticks  -> 180°
static inline uint16_t degToTicks(int deg) {
    if (deg < 0)   deg = 0;
    if (deg > 180) deg = 180;
    return (uint16_t)map(deg, 0, 180, 102, 512);
}

// -----------------------------------------------------------------------------
// SafeServo — clamped, rate-limited servo with a remembered current position
// -----------------------------------------------------------------------------
class SafeServo {
public:
    SafeServo(uint8_t channel, int min_deg, int max_deg,
              int rest_deg, float rate_dps)
        : ch_(channel), min_(min_deg), max_(max_deg),
          rest_(rest_deg), rate_(rate_dps),
          current_(rest_deg), target_(rest_deg),
          last_update_us_(0), enabled_(true) {}
        // last_update_us_ is initialised to 0 (NOT micros()) because the
        // constructor runs during global static-init, before the Arduino
        // runtime has necessarily set up the hardware timer. begin() sets
        // a real value once it's safe.

    void begin(Adafruit_PWMServoDriver& drv) {
        drv_ = &drv;
        // Force the servo to the rest position immediately at boot.
        current_ = rest_;
        target_  = rest_;
        drv_->setPWM(ch_, 0, degToTicks(current_));
        last_update_us_ = micros();
    }

    void setTarget(int deg) {
        if (deg < min_) deg = min_;
        if (deg > max_) deg = max_;
        target_ = deg;
    }

    void emergencyStop() {
        target_ = rest_;
    }

    void disable() { enabled_ = false; }
    void enable()  { enabled_ = true;  }

    // Call from loop(); steps `current_` toward `target_` no faster than rate_.
    void update() {
        if (!enabled_ || drv_ == nullptr) return;
        uint32_t now_us = micros();
        float dt = (now_us - last_update_us_) * 1e-6f;
        last_update_us_ = now_us;

        float step = rate_ * dt;             // max degrees this tick
        float delta = (float)target_ - (float)current_;
        if (fabsf(delta) <= step) {
            current_ = target_;              // arrived
        } else {
            current_ += (delta > 0 ? step : -step);
        }
        drv_->setPWM(ch_, 0, degToTicks((int)current_));
    }

    int currentDeg() const { return (int)current_; }
    int targetDeg()  const { return target_;       }

    // Tolerance-band "have I arrived?" check. Avoids exact float equality on
    // the tracked current_ position (dt jitter, float drift across many small
    // steps can leave current_ within < 1 deg of target_ but never bit-equal).
    bool atTarget(int tol_deg = 1) const {
        int diff = (int)current_ - target_;
        if (diff < 0) diff = -diff;
        return diff <= tol_deg;
    }

private:
    Adafruit_PWMServoDriver* drv_ = nullptr;
    uint8_t   ch_;
    int       min_, max_, rest_, target_;
    float     rate_;       // degrees per second
    float     current_;    // tracked actual angle (we don't have feedback)
    uint32_t  last_update_us_;
    bool      enabled_;
};

// -----------------------------------------------------------------------------
// Classifier — sliding-window majority vote over recent FPGA outputs
// -----------------------------------------------------------------------------
class MajorityVoter {
public:
    void push(uint8_t cls) {
        buf_[idx_] = cls;
        idx_ = (idx_ + 1) % VOTE_WINDOW;
        if (count_ < VOTE_WINDOW) count_++;
    }

    // Returns the dominant class (0 or 1), or 0xFF if no majority yet.
    uint8_t decision() const {
        if (count_ < VOTE_WINDOW) return 0xFF;
        int n0 = 0, n1 = 0;
        for (int i = 0; i < VOTE_WINDOW; i++) {
            if (buf_[i] == 0) n0++;
            else if (buf_[i] == 1) n1++;
        }
        if (n0 >= VOTE_THRESHOLD) return 0;
        if (n1 >= VOTE_THRESHOLD) return 1;
        return 0xFF;
    }

    void clear() { idx_ = 0; count_ = 0; }

private:
    uint8_t buf_[VOTE_WINDOW] = {0};
    int     idx_   = 0;
    int     count_ = 0;
};

// -----------------------------------------------------------------------------
// Global state
// -----------------------------------------------------------------------------
Adafruit_PWMServoDriver pwm(PCA9685_ADDR);

SafeServo handServo(CH_HAND,
                    HAND_MIN_DEG, HAND_MAX_DEG,
                    HAND_REST_DEG, HAND_RATE_DPS);
SafeServo legServo (CH_LEG,
                    LEG_MIN_DEG, LEG_MAX_DEG,
                    LEG_REST_DEG, LEG_RATE_DPS);

MajorityVoter voter;

enum class State : uint8_t {
    BOOT,
    HOMING,
    IDLE,
    RUNNING,
    FAULTING,   // transitional: slew to rest with servos still enabled
    FAULT       // terminal: servos disabled (no PWM updates)
};
State state = State::BOOT;

uint32_t last_fpga_byte_ms = 0;
uint32_t state_entered_ms  = 0;
uint32_t last_reassert_ms  = 0;
uint8_t  last_committed    = 0xFF;

// Forward decls
void enterState(State s);
void handleByte(uint8_t b);
void applyDecision(uint8_t cls);
void heartbeat();

// =============================================================================
// Arduino entry points
// =============================================================================
void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.begin(115200);
    delay(300);
    Serial.println(F("\n=== db_atcnet ESP32 controller boot ==="));

    // UART2 to Zynq PS
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);
    Serial.print(F("Serial2 @ ")); Serial.print(UART2_BAUD);
    Serial.println(F(" baud — listening for FPGA class bytes"));

    // I2C / PCA9685
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_HZ);
    pwm.begin();
    pwm.setPWMFreq(SERVO_FREQ_HZ);
    handServo.begin(pwm);
    legServo.begin(pwm);
    Serial.println(F("PCA9685 ready; servos forced to rest"));

    // Seed the watchdog/reassert timestamps so any future change that lets
    // the watchdog fire in IDLE (instead of RUNNING only) doesn't fault out
    // at boot before the FPGA's first byte arrives.
    last_fpga_byte_ms = millis();
    last_reassert_ms  = millis();

    enterState(State::HOMING);
}

void loop() {
    uint32_t now_ms = millis();

    // ---- Read FPGA classifications ----
    while (Serial2.available() > 0) {
        uint8_t b = (uint8_t)Serial2.read();
        last_fpga_byte_ms = now_ms;
        handleByte(b);
    }

    // ---- USB Serial: binary class bytes (from host bridge) + text commands ----
    // The host bridge in raw pass-through mode delivers FPGA class bytes
    // verbatim (0x00 or 0x01) over the same USB CDC channel that carries the
    // text command parser. 0x00 / 0x01 are not valid bytes in any text
    // command, so we route them straight to handleByte() and keep the
    // existing T0 / T1 / STOP / RESET / STATUS parser intact for everything
    // else. Overlong text lines latch cmd_overflow so we drop the whole line
    // at the next newline rather than silently truncating into a prefix that
    // could accidentally fire a motor command.
    static char    cmd_buf[16];
    static uint8_t cmd_len = 0;
    static bool    cmd_overflow = false;
    while (Serial.available() > 0) {
        int rb = Serial.read();
        if (rb < 0) break;
        uint8_t ub = (uint8_t)rb;
        if (ub == 0x00 || ub == 0x01) {
            handleByte(ub);
            last_fpga_byte_ms = now_ms;
            continue;
        }
        char c = (char)ub;
        if (c == '\n' || c == '\r') {
            if (cmd_overflow) {
                Serial.println(F("[cmd] line too long — ignored"));
                cmd_overflow = false;
                cmd_len = 0;
                continue;
            }
            if (cmd_len == 0) continue;
            cmd_buf[cmd_len] = '\0';
            cmd_len = 0;
            if      (strcasecmp(cmd_buf, "T0") == 0)     { handleByte(0); last_fpga_byte_ms = now_ms; }
            else if (strcasecmp(cmd_buf, "T1") == 0)     { handleByte(1); last_fpga_byte_ms = now_ms; }
            else if (strcasecmp(cmd_buf, "STOP") == 0)   { enterState(State::FAULTING); }
            else if (strcasecmp(cmd_buf, "RESET") == 0)  { enterState(State::HOMING); }
            else if (strcasecmp(cmd_buf, "STATUS") == 0) {
                Serial.printf("state=%u  hand=%d°/%d°  leg=%d°/%d°  last_class=%u  ms_since_uart=%lu\n",
                              (unsigned)state,
                              handServo.currentDeg(), handServo.targetDeg(),
                              legServo.currentDeg(),  legServo.targetDeg(),
                              (unsigned)last_committed,
                              (unsigned long)(now_ms - last_fpga_byte_ms));
            } else {
                Serial.print(F("[cmd] unknown: ")); Serial.println(cmd_buf);
            }
        } else if (cmd_len + 1 < sizeof(cmd_buf)) {
            cmd_buf[cmd_len++] = c;
        } else {
            cmd_overflow = true;     // latch; ignore remainder of this line
        }
    }

    // ---- State machine ----
    switch (state) {
        case State::BOOT:
            // Should never linger here; setup() transitions to HOMING.
            enterState(State::HOMING);
            break;

        case State::HOMING:
            // Wait for both servos to reach their rest positions (tolerance
            // band; see SafeServo::atTarget()).
            if (handServo.atTarget() && legServo.atTarget()) {
                enterState(State::IDLE);
            }
            break;

        case State::IDLE:
            // Listening for first valid majority decision.
            if (voter.decision() != 0xFF) enterState(State::RUNNING);
            break;

        case State::RUNNING: {
            // Watchdog: lost the FPGA stream -> require operator-ack fault.
            // Going to FAULTING (rather than HOMING) is the safer policy for a
            // BCI-driven prosthetic: a glitchy UART that briefly recovers
            // shouldn't silently resume motor commands without the user being
            // ready. The user must send "RESET\n" to leave FAULT.
            // enterState(FAULTING) issues emergencyStop() on both servos for us.
            if (now_ms - last_fpga_byte_ms > UART_TIMEOUT_MS) {
                Serial.println(F("[WDOG] UART timeout — FAULTING"));
                voter.clear();
                enterState(State::FAULTING);
                break;
            }
            // Periodic re-assertion of current target (defense against
            // PCA9685 channel state drift after transient I2C glitches).
            if (REASSERT_MS > 0 && last_committed != 0xFF &&
                now_ms - last_reassert_ms >= REASSERT_MS) {
                applyDecision(last_committed);
                last_reassert_ms = now_ms;
            }
            break;
        }

        case State::FAULTING:
            // Slew to rest while servos are still enabled; once both arrive
            // (within tol), transition to FAULT which disables PWM updates.
            if (handServo.atTarget() && legServo.atTarget()) {
                enterState(State::FAULT);
            }
            break;

        case State::FAULT:
            // Servos disabled; stay here until user sends "RESET\n".
            break;
    }

    // ---- Servo rate-limited updates (always) ----
    handServo.update();
    legServo.update();

    // ---- Heartbeat ----
    heartbeat();
}

// =============================================================================
// Helpers
// =============================================================================
void enterState(State s) {
    state = s;
    state_entered_ms = millis();
    switch (s) {
        case State::BOOT:    Serial.println(F("[state] BOOT")); break;
        case State::HOMING:  Serial.println(F("[state] HOMING (servos -> rest)"));
                             handServo.setTarget(HAND_REST_DEG);
                             legServo.setTarget(LEG_REST_DEG);
                             handServo.enable(); legServo.enable();
                             break;
        case State::IDLE:    Serial.println(F("[state] IDLE (waiting for FPGA class)"));
                             voter.clear();
                             last_committed = 0xFF;
                             break;
        case State::RUNNING: Serial.println(F("[state] RUNNING (acting on classifications)"));
                             last_reassert_ms = millis();
                             break;
        case State::FAULTING: Serial.println(F("[state] FAULTING (slewing to rest before disable)"));
                              // Keep servos ENABLED so update() actually drives
                              // them toward rest; once both arrive we move on
                              // to FAULT (handled in the loop's case body).
                              handServo.enable(); legServo.enable();
                              handServo.emergencyStop();
                              legServo.emergencyStop();
                              break;
        case State::FAULT:   Serial.println(F("[state] FAULT (motors disabled; send RESET to recover)"));
                             handServo.disable();
                             legServo.disable();
                             break;
    }
}

void handleByte(uint8_t b) {
    // Drop everything while faulted. applyDecision() already blocks motor
    // moves in FAULT, but letting the voter keep accumulating bytes during
    // FAULTING/FAULT muddies the internal state — the moment we re-enter
    // IDLE via RESET, voter.clear() runs anyway, but it's cleaner to never
    // touch the voter at all while the system is supposed to be inert.
    if (state == State::FAULT || state == State::FAULTING) return;

    // Also drop while HOMING — accepting class bytes here would let
    // applyDecision() retarget the servos before they finish slewing to
    // rest, and HOMING's atTarget() check would then "succeed" against
    // the new target instead of the REST position. That confused the
    // demo trace (state slips into IDLE while last_committed is stale).
    if (state == State::HOMING) return;

    // Accept only valid class codes; treat anything else as noise.
    if (b != 0 && b != 1) {
        Serial.print(F("[rx] ignored byte 0x")); Serial.println(b, HEX);
        return;
    }
    voter.push(b);
    uint8_t d = voter.decision();
    if (d != 0xFF && d != last_committed) {
        applyDecision(d);
        last_committed = d;
        if (state == State::IDLE) enterState(State::RUNNING);
    }
}

void applyDecision(uint8_t cls) {
    Serial.print(F("[decide] class=")); Serial.println(cls);
    if (state == State::FAULT) return;
    if (cls == 0) {
        // HAND task -> close fingers, ankle stays at rest.
        handServo.setTarget(HAND_ACT_DEG);
        legServo.setTarget(LEG_REST_DEG);
    } else {
        // FOOT task -> ankle dorsiflex, hand stays at rest.
        handServo.setTarget(HAND_REST_DEG);
        legServo.setTarget(LEG_MAX_DEG);
    }
}

void heartbeat() {
    static uint32_t last = 0;
    uint32_t now = millis();
    uint32_t period;
    switch (state) {
        case State::FAULT:    period = 100;  break;  // very fast = locked
        case State::FAULTING: period = 150;  break;  // fast while slewing to rest
        case State::HOMING:   period = 250;  break;  // medium
        case State::IDLE:     period = 1000; break;  // slow
        case State::RUNNING:  period = 500;  break;
        default:              period = 50;   break;
    }
    if (now - last >= period) {
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        last = now;
    }
}
