<h1 align="center">Design &amp; Development of a Brain–Computer Interface for Assistive Motion Control, Rehabilitation &amp; Mobility Applications</h1>

<h3 align="center">NeuroMove — control a prosthetic hand &amp; leg with your mind</h3>

<p align="center">
  <img src="https://img.shields.io/badge/BCI-EEG%20motor%20imagery-blue.svg" alt="BCI">
  <img src="https://img.shields.io/badge/model-DB--ATCNet-orange.svg" alt="Model">
  <img src="https://img.shields.io/badge/accelerator-Zynq%20UltraScale%2B-red.svg" alt="FPGA">
  <img src="https://img.shields.io/badge/security-5--layer%20crypto-green.svg" alt="Security">
  <img src="https://img.shields.io/badge/status-graduation%20project-lightgrey.svg" alt="Status">
</p>

<p align="center">
  <em>A non-invasive EEG brain–computer interface that reads motor-imagery brainwaves and drives a
  3D-printed hand and leg in real time — decoded by the <strong>DB-ATCNet</strong> neural network,
  accelerated on a <strong>Xilinx Zynq UltraScale+</strong> FPGA, and secured with a five-layer
  cryptographic stack. Built for people who keep the intention to move, but have lost the means.</em>
</p>

<p align="center">
  <strong>Egypt-Japan University of Science and Technology (E-JUST)</strong><br>
  Computer Science &amp; Information Technology — Graduation Project 2025 / 2026
</p>

---

## Vision

Millions of people living with **ALS, stroke, or spinal-cord injury** keep the *intention* to move
but have lost the muscular means to act on it. **NeuroMove** restores that link: a user imagines a
movement, non-invasive EEG captures the motor-imagery signature, a hardware-accelerated neural network
decodes the intent, and a powered 3D-printed limb executes it — with **no muscle activity required**.

This repository is the complete, multidisciplinary research platform behind that system, spanning
**AI models, embedded hardware (FPGA + ESP32), cryptographic security, physics/FEA analysis, and
robotics simulation**.

| Headline metric | Value | Context |
|---|:---:|---|
| **Brain-to-motion latency** | **~134 ms** | Below the 200 ms human reaction threshold |
| **Live decoding accuracy** | **~90 %** | DB-ATCNet · HaLT dataset · 5 electrodes |
| **Electrode count** | **5 / 19** | Learned channel selection (Gumbel-Softmax) |
| **On-chip inference** | **sub-millisecond** | DB-ATCNet on Zynq UltraScale+ (`xczu7ev`) |
| **Actuation** | **20 kg·cm** | 5-finger tendon-driven hand + powered ankle/leg |

---

## End-to-end system

```
   ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────┐
   │  SENSE   │──▶│  DECODE   │──▶│ ACCELERATE │──▶│    SECURE    │──▶│   ACT    │
   │  EEG cap │   │ DB-ATCNet │   │ Zynq FPGA  │   │ AES/HMAC/RSA │   │  limb    │
   └──────────┘   └───────────┘   └────────────┘   └──────────────┘   └──────────┘
   motor imagery   5-of-19 ch      Q8.8 pipeline    encrypt + auth     hand & leg
   μ/β rhythm      classifier      bit-exact RTL     the data path      servos
        │                                                                   ▲
        └───────────────── ESP32-S3 acquisition / servo control ───────────┘
```

Two research tracks feed this pipeline: a **classical ML** baseline that establishes the signal-
processing and feature-engineering foundation, and a **cross-modal synthesis** track that generates
fNIRS hemodynamics from EEG to reach hybrid-BCI accuracy from an EEG headset alone.

---

## Components at a glance

| Component | What it does | Key result | Primary tooling |
|---|---|---|---|
| [`machine-learning/`](machine-learning/README.md) | Classical MI classification pipeline (preprocessing, CSP/PSD features, augmentation, LDA/SVM/RF/XGB) | 89.1 % subject-specific (CSP+SVM); 93.8 % ensemble | scikit-learn, MNE |
| [`deep-learning/`](deep-learning/README.md) | **DB-ATCNet** — edge-optimized DL classifier with FPGA-aware attention and learned channel reduction | 89.94 % @ 5 ch (Gumbel); 91.99 % @ 19 ch | TensorFlow |
| [`cross-modal-synthesis/`](cross-modal-synthesis/README.md) | **SCDM** — diffusion model synthesizing fNIRS hemodynamics from EEG for hybrid BCI | PCC 0.437; recovers 96 % of the real-fNIRS accuracy gain | PyTorch |
| [`Secure-EEG-Based-Person-Identification-and-Authentication-Systems-main/`](Secure-EEG-Based-Person-Identification-and-Authentication-Systems-main/README.md) | EEG biometric **identification** (109-class CNN) and **authentication** (CNN + Siamese) | 96.7 % ID accuracy; EER ~3.8 % | TensorFlow |
| [`fpga/`](fpga/README.md) | SystemVerilog **DB-ATCNet accelerator** (bit-exact Q8.8) + AES/SHA/HMAC/RSA security cores + host/PS/ESP32 stack | Fits `xczu7ev`; sub-ms inference; ≈5.1 W | Vivado, SystemVerilog |
| [`fpga/sw/esp32/`](fpga/sw/README.md) | **ESP32-S3 motor controller** — decodes FPGA class bytes over UART2 and drives the PCA9685 hand/leg servos, safety-first | Class 0 → hand, Class 1 → ankle; clamps, rate limit, 4-of-5 vote, UART watchdog, fault states | Arduino / ESP32-S3 |
| [`FEA-Biomaterial-Selection-COMSOL-main/`](FEA-Biomaterial-Selection-COMSOL-main/README.md) | **FEA + biomaterial selection** — structural & thermal safety of the powered lower-limb prosthesis | PETG selected; FOS 1.6 @ 900 N; T_max safe @ baseline | COMSOL 6.1 |
| [`simulation/coppeliasim/`](simulation/coppeliasim/PROJECT_DOCUMENTATION.md) | **Online BCI→exoskeleton simulation** — FastAPI + Docker pipeline mapping decoded motor imagery to a 3-joint arm in CoppeliaSim over ZMQ (decode → kinematics → control → sim) | Real-time ~20 Hz joint control; `scenes/bci_exo_scene.ttt` | CoppeliaSim, FastAPI, ZMQ |

---

## Repository structure

```
NeuroMove/
├── machine-learning/          # Classical ML MI pipeline (CSP+PSD → LDA/SVM/RF/XGB)
│   ├── notebooks/             #   full pipeline: preprocessing → features → models → eval
│   ├── src/                   #   package scaffold (code currently lives in notebooks/)
│   └── saved-models/          #   (git-ignored)
│
├── deep-learning/             # DB-ATCNet — edge-optimized DL classifier
│   ├── DB-ATCNet/             #   model, attention, Gumbel channel selection, training
│   ├── configs/  checkpoints/ experiments/  notebooks/
│
├── cross-modal-synthesis/     # SCDM — EEG → fNIRS hemodynamic synthesis (diffusion)
│   ├── src/                   #   data · models · training · evaluation · utils
│   ├── configs/  notebooks/  scripts/  tests/  assets/  figures/
│
├── Secure-EEG-Based-...-main/ # EEG biometric identification & authentication
│   ├── identification/        #   1D CNN 109-class classifier + real-time GUI
│   └── authentication/        #   CNN + Siamese (triplet loss) + enroll/verify GUI
│
├── fpga/                      # SystemVerilog accelerator + security + embedded stack
│   ├── rtl/                   #   model datapath · attention · classifier · security · util
│   ├── sim/                   #   per-module + end-to-end bit-exact testbenches
│   ├── sw/                    #   esp32/ firmware · host/ Python tools · zynq_ps/ PS bridge
│   ├── scripts/  synth/  constraints/  weights/  data/  ps_demo/
│
├── FEA-Biomaterial-Selection-COMSOL-main/  # COMSOL FEA of the prosthesis
│   ├── models/  plots/  docs/  reports/
│
├── simulation/
│   └── coppeliasim/           # Online BCI→exoskeleton sim (FastAPI + Docker + CoppeliaSim)
│       ├── bci_exo/           #   decode → kinematics → control-signal mapping
│       ├── sim_controller.py  #   CoppeliaSim ZMQ real-time joint control
│       ├── pipeline.py · api.py  #   orchestrator + REST API
│       └── scenes/bci_exo_scene.ttt
│
├── data/                      # Dataset layout + download instructions (data git-ignored)
│   └── raw · interim · processed · external
├── scripts/                   # Cross-component orchestration utilities
├── results/                   # Shared figures / reports / logs (git-ignored contents)
├── tests/                     # Cross-component integration tests
│
├── environment.yml            # Conda environment (name: neuromove)
├── requirements.txt           # pip fallback
├── NeuroMove_poster.pdf       # Project poster
└── NeuroMove_flyer.pdf        # Project flyer
```

Each top-level component carries its own `README.md` describing its layout, tooling, and run
instructions. Large datasets, model checkpoints, and generated hardware artifacts are **not**
committed to git — see [`data/README.md`](data/README.md) and each component's README for
reproduction steps.

---

## Datasets

The platform is validated across five public EEG / EEG–fNIRS datasets (full download and layout
instructions in [`data/README.md`](data/README.md)):

| Dataset | Subjects | Channels | Classes | Rate | Role |
|---|:---:|:---:|:---:|:---:|---|
| **HaLT** (Kaya et al., 2018) | 13 | 19 EEG | 6 (2 used) | 200 Hz | **Primary** — all deployment experiments |
| PhysioNet EEGMMIDB | 109 | 64 EEG | 4 | 160 Hz | Architecture & cross-subject benchmarking; biometrics |
| BCI Competition IV 2a | 9 | 22 EEG | 4 | 250 Hz | 4-class benchmark |
| BCI Competition IV 2b | 9 | 3 EEG | 2 | 250 Hz | Sparse-channel evaluation |
| Shin et al. (2016) EEG+fNIRS | 26 | 30 EEG + 30 fNIRS | 2 | 200 / 10.4 Hz | Cross-modal synthesis & neurovascular coupling |

---

## Getting started

```bash
# 1. Clone
git clone <repo-url> NeuroMove && cd NeuroMove

# 2. Create the Python environment (shared across the AI components)
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate neuromove

# 3. Pick a component and follow its README
cd deep-learning        && cat README.md   # DB-ATCNet training & channel selection
cd ../cross-modal-synthesis && cat README.md   # SCDM EEG→fNIRS synthesis
cd ../fpga              && cat README.md   # RTL accelerator: build.sh, sim, synth
```

The AI tracks (`machine-learning`, `deep-learning`, `cross-modal-synthesis`, biometrics) run on the
shared Python stack. The hardware track (`fpga`) additionally needs **Vivado 2025.2** for
simulation/synthesis; the physics track needs **COMSOL Multiphysics 6.1** and **CoppeliaSim**. See
each component README for exact prerequisites.

---

## System architecture notes

- **Decoder:** DB-ATCNet — a dual-branch, attention-augmented temporal convolutional network — was
  chosen for its accuracy/latency balance and, critically, its mapping to static-shape RTL. MHA was
  replaced with an Improved CBAM and channels reduced from 19 → 5 via learned Gumbel-Softmax
  selection, all driven by the FPGA deployment constraint (no dynamic memory, fixed compute).
- **Accelerator:** A fully streaming **Q8.8 fixed-point** pipeline on `xczu7ev` (ZCU104 / ZCU106),
  **bit-exact** against a Python golden model, with subject-dependent weights loaded over AXI-Lite.
- **Security:** An independent RTL crypto stack (AES-256-GCM, SHA-256, HMAC-SHA256, RSA-2048, secure
  boot) authenticates and encrypts the EEG data and control path; EEG biometrics gate who may drive
  the limb.
- **Actuation:** The Zynq PS returns a 1-bit class per inference; the ESP32-S3 firmware
  (`fpga/sw/esp32/`) reads that class over UART2 and drives the PCA9685 servos that flex the
  tendon-driven hand (class 0) and powered ankle/leg (class 1). It is safety-first — angle clamps,
  rate limiting, 4-of-5 majority-vote debouncing, a UART watchdog, and HOMING/IDLE/RUNNING/FAULT
  states keep the limb inert unless a stable decision stream is present. (Streaming raw EEG windows
  from the ESP32 is scaffolded as an optional future bio-amp path.)
- **Physics &amp; robotics:** COMSOL FEA verifies the printed **PETG** components are structurally and
  thermally safe under stance/actuation loads. A CoppeliaSim testbed closes the loop in software — a
  FastAPI + Docker pipeline (`simulation/coppeliasim/`) decodes motor imagery, maps it through
  `bci_exo` (class → kinematics → control signal), and drives a 3-joint exoskeleton arm over the ZMQ
  remote API in real time, with an optional feedback-corrected control loop.

---

## Team &amp; affiliation

Graduation project (B.Sc.), **Egypt-Japan University of Science and Technology (E-JUST)** —
Computer Science &amp; Information Technology, AI &amp; Data Science, 2025 / 2026.

**Team:** Ahmed Elsheikh · Karim Hossam · Miral Farghaly · Noran Morad · Moustafa Abdullah ·
Ezaldeen Ahmed · Rasha Gamel · Mariam Ihab · Noureen Tamer · Jomana George

**Supervisors:** Dr. Reda Albassiouny &amp; Dr. Sameh Sherif

---

## License

See [LICENSE](LICENSE). Individual components may carry their own license terms; refer to each
component's README.
