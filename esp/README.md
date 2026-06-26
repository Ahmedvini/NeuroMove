# ESP Firmware

Firmware for ESP32 / ESP8266 microcontrollers — signal acquisition,
streaming, and edge inference.

## Layout

```
esp/
├── src/             # Application source (main.cpp / main.c)
├── include/         # Project headers
├── lib/             # Project-specific libraries
├── test/            # Unit tests
├── data/            # Files for SPIFFS/LittleFS (web UI, configs)
└── platformio.ini   # Build configuration (boards, framework, deps)
```

## Toolchain
- **PlatformIO** (recommended) or **Arduino IDE** or **ESP-IDF**.

## Usage (PlatformIO)

```bash
pio run                 # build
pio run --target upload # flash
pio device monitor      # serial monitor
```

## Notes
- Configure Wi-Fi/sensor pins in `include/` or via build flags in `platformio.ini`.
- Build output (`.pio/`, `build/`) is git-ignored.
