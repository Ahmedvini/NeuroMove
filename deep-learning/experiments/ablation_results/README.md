# EEG–fNIRS BCI Platform

A multidisciplinary research platform combining **AI models**, **embedded hardware
(FPGA + ESP)**, and **physics/robotics simulation** around a central goal:
the **cross-modal synthesis of fNIRS hemodynamic signals from EEG**.

---

## Repository structure

```
eeg-fnirs-bci/
├── ai-models/                 # General AI/ML models
│   ├── deep-learning/         #   Neural-network models (PyTorch/TF)
│   └── machine-learning/      #   Classical ML (scikit-learn, XGBoost, …)
│
├── cross-modal-synthesis/     # EEG → fNIRS hemodynamics synthesis (core task)
│
├── fpga/                      # RTL design, testbenches, constraints, IP
│
├── esp/                       # ESP32/ESP8266 firmware (PlatformIO/Arduino)
│
├── simulation/
│   ├── comsol/                # COMSOL Multiphysics models (.mph)
│   └── coppeliasim/           # CoppeliaSim scenes (.ttt) and scripts
│
├── data/                      # Datasets (raw → interim → processed)
├── docs/                      # Documentation, architecture, references
├── scripts/                   # Cross-component utility scripts
├── results/                   # Figures, reports, logs (shared outputs)
└── tests/                     # Integration tests
```

Each top-level component has its own `README.md` describing its layout,
tooling, and how to run it.

---

## Components at a glance

| Component | Purpose | Primary tooling |
|-----------|---------|-----------------|
| `ai-models/deep-learning` | DL models for EEG/fNIRS classification & regression | PyTorch / TensorFlow |
| `ai-models/machine-learning` | Classical ML baselines & feature pipelines | scikit-learn, XGBoost |
| `cross-modal-synthesis` | Generate fNIRS hemodynamics from EEG | Deep generative models |
| `fpga` | Hardware acceleration / real-time signal processing | Verilog/VHDL, Vivado |
| `esp` | Embedded acquisition & edge inference | ESP-IDF / Arduino / PlatformIO |
| `simulation/comsol` | Physical/biophysical modeling (e.g. hemodynamics, optics) | COMSOL Multiphysics |
| `simulation/coppeliasim` | Robotics / experimental setup simulation | CoppeliaSim (Lua/Python) |

---

## Getting started

```bash
# 1. Clone
git clone <repo-url> && cd eeg-fnirs-bci

# 2. Create the Python environment
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate eeg-fnirs-bci

# 3. Explore a component
cd cross-modal-synthesis && cat README.md
```

## Data

Large datasets and model checkpoints are **not** committed to git.
See `data/README.md` for the expected layout and download instructions.

## License

See [LICENSE](LICENSE).
