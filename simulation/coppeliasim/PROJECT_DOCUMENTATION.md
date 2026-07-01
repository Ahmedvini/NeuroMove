# BCI Assistive Motion Graduation Project — Complete Documentation

**Version:** 1.2.0 (Milestones 1-7 Complete, Milestone 8 Optional Loop Enabled)  
**Date:** April 15, 2026  
**Status:** Fully Operational (API + Pipeline + Docker)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Milestones Completed](#milestones-completed)
4. [File Structure & Descriptions](#file-structure--descriptions)
5. [Required Libraries & Dependencies](#required-libraries--dependencies)
6. [Installation & Setup](#installation--setup)
7. [Running the Project](#running-the-project)
8. [API Endpoints Reference](#api-endpoints-reference)
9. [Workflow Examples](#workflow-examples)
10. [Deployment (Docker)](#deployment-docker)
11. [Feedback Loop (Optional)](#feedback-loop-optional)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

**BCI Assistive Motion** is a Brain-Computer Interface (BCI) system that:

1. **Reads EEG signals** from the Physionet BCI Motor Imagery dataset
2. **Performs AI inference** using deep learning models (DB_ATCNet, ATCNet, EEGNet)
3. **Decodes motor imagery** into 4 classes:
   - Class 0: Both Feet (Rest)
   - Class 1: Left Fist
   - Class 2: Both Fists
   - Class 3: Right Fist
4. **Maps predictions** to robot kinematics (joint angles, Cartesian positions)
5. **Simulates** the exoskeleton arm in CoppeliaSim in real-time
6. **Exposes results** via a FastAPI web service

**Users can:**

- Train models on EEG data
- Run inference on new signals
- Map outputs to simulation-ready control signals
- Execute real-time simulation
- Access all stages via REST API

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HTTP Clients (Postman / Web)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI (api.py) — Port 8000             │
│  /run  /infer  /map  /simulate  /models  /health           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                Pipeline Controller (pipeline.py)            │
│         Orchestrates: Inference → Mapping → Simulation      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐  ┌─────▼────┐  ┌─────▼──────┐
    │Decoder │  │  Mapper  │  │Sim Control │
    │(bci_exo)  │(bci_exo) │  │(sim_ctrl)  │
    └────┬───┘  └─────┬────┘  └─────┬──────┘
         │            │             │
    ┌────▼─────────┬──▼──────┬──────▼────┐
    │ BCIFrame     │Control  │CoppeliaSim│
    │ Decoder      │Signal   │ ZMQ API   │
    └──────────────┴─────────┴───────────┘
```

---

## Milestones Completed

### ✅ Milestone 1: Data Loading & Preprocessing

**Files:** `Physionet_DataLoad.py`, `BCI2A_preprocess.py`

- Load BCI Motor Imagery dataset from Physionet
- Preprocess EEG signals (filtering, bandpass, normalization)
- Stratified k-fold splitting for training

### ✅ Milestone 2: Model Training

**Files:** `attention_models.py`, `models.py`, `BCI_2A_main.py`, `Physionet_main.py`

- Implemented 3 deep learning models: DB_ATCNet, ATCNet, EEGNet
- Attention mechanisms for spatial-temporal feature extraction
- k-fold cross-validation, hyperparameter tuning
- Save best models per subject

### ✅ Milestone 3: Structured Output Interface

**Files:** `bci_exo/` (package)

- `decoder.py`: Convert model predictions to BCIFrame objects
- `structured_output.py`: Define BCIFrame, JointAngles, CartesianPosition
- `kinematics.py`: Class-to-angle mappings (biomechanical)
- `mapper.py`: Convert BCIFrame → ControlSignal (simulation-ready)
- `results_loader.py`: Load predictions from CSV/JSON/NPZ
- `stream.py`: Frame streaming interface

### ✅ Milestone 4: CoppeliaSim Integration

**Files:** `sim_controller.py`, `build_scene.lua`, `build_scene.py`

- ZMQ remote API connection to CoppeliaSim
- Real-time joint control (shoulder, elbow, wrist)
- End-effector positioning (Cartesian)
- Custom wall-clock synchronizer (20 Hz default)
- Scene builder Lua script for 3-joint exoskeleton

### ✅ Milestone 5: Automation Pipeline

**Files:** `pipeline.py`

- Single orchestrator: inference → mapping → simulation
- Support for 3 input modes: dataset, image, parameters
- Error handling & recovery (continue_on_frame_error)
- Async job queueing support
- PipelineConfig + PipelineRunResult data models

### ✅ Milestone 6: Online REST API

**Files:** `api.py`

- FastAPI framework (uvicorn server)
- 5 core endpoints: `/run`, `/infer`, `/map`, `/simulate`, `/models`
- Request validation (Pydantic models)
- Async job tracking (`/status/{job_id}`, `/result/{job_id}`)
- Swagger UI at `/docs`

### ✅ Milestone 7: Deployment (Docker)

**Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`

- Containerized API deployment using Python 3.11 slim image
- Dependency installation from `requirements.txt`
- CoppeliaSim ZMQ client installation inside container
- Port mapping to expose FastAPI on `8000`

### ✅ Milestone 8: Feedback Loop (Optional, Advanced)

**Files:** `pipeline.py`, `api.py`

- Optional closed-loop control path in simulation stage
- Runtime correction using observed simulator joint states
- Configurable correction gain, tolerance, and max cycles
- Feedback telemetry samples returned in pipeline result

---

## File Structure & Descriptions

### Root-Level Files

| File                     | Purpose                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `pipeline.py`            | **Milestone 5** — Orchestrator. Chains inference → mapping → simulation without manual intervention. |
| `api.py`                 | **Milestone 6** — FastAPI server. Exposes all pipeline stages via REST endpoints.                    |
| `sim_controller.py`      | **Milestone 4** — CoppeliaSim integration. Connects via ZMQ, controls joints, reads state.           |
| `main_interface.py`      | Integration hook for model.predict() output. Converts softmax → structured frames.                   |
| `models.py`              | Model architectures (custom layers, loss functions).                                                 |
| `attention_models.py`    | **Milestone 2** — DB_ATCNet, ATCNet implementations with attention.                                  |
| `BCI_2A_main.py`         | Training script for 2A dataset. Handles k-fold CV, model checkpointing.                              |
| `Physionet_main.py`      | Training script for Physionet Motor Imagery dataset.                                                 |
| `BCI2A_preprocess.py`    | **Milestone 1** — Bandpass filtering, normalization, windowing.                                      |
| `Physionet_DataLoad.py`  | **Milestone 1** — Dataset loading, subject enumeration, trial segmentation.                          |
| `build_scene.py`         | Lua/Python wrapper for CoppeliaSim scene building.                                                   |
| `build_scene.lua`        | **Milestone 4** — Lua script executed in CoppeliaSim to create 3-joint arm.                          |
| `Dockerfile`             | **Milestone 7** — Container image definition for API service deployment.                             |
| `docker-compose.yml`     | **Milestone 7** — Compose service for local/cloud deployment orchestration.                          |
| `.dockerignore`          | **Milestone 7** — Excludes development artifacts from Docker build context.                          |
| `test_sim_controller.py` | Unit tests for sim_controller.py (mocked ZMQ).                                                       |
| `test_step.py`           | Test script for CoppeliaSim stepping.                                                                |

### Package: `bci_exo/`

| File                   | Purpose                                    | Key Classes/Functions                                                              |
| ---------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| `__init__.py`          | Package exports                            | All public APIs                                                                    |
| `decoder.py`           | **Milestone 3** — Softmax → BCIFrame       | `BCIDecoder`, `decode_batch()`, `stream()`                                         |
| `structured_output.py` | **Milestone 3** — Data models              | `BCIFrame`, `JointAngles`, `CartesianPosition`, `CLASS_LABELS`                     |
| `kinematics.py`        | **Milestone 3** — Motor imagery angles     | `map_class_to_kinematics()`, `L1`, `L2` (limb lengths)                             |
| `mapper.py`            | **Milestone 3** — BCIFrame → ControlSignal | `map_to_control_signal()`, `ControlSignal`, `EulerRotation`, `NormalisedRotations` |
| `results_loader.py`    | **Milestone 3** — Load predictions         | `load_results()`, `load_npz_results()`, `load_csv_predictions()`                   |
| `stream.py`            | **Milestone 3** — Streaming interface      | `stream_from_file()`, `stream_from_model()`, `FrameSink`                           |

### Results Directories

| Directory                      | Purpose                                        |
| ------------------------------ | ---------------------------------------------- |
| `BCI_2A_Results/`              | 2A dataset results (models, logs, performance) |
| `Physionet_FourClass_Results/` | Physionet Motor Imagery results                |
| `scenes/`                      | CoppeliaSim scene files (`.ttt`)               |

### Environment

| Item       | Purpose                           |
| ---------- | --------------------------------- |
| `bci_env/` | Python virtual environment (venv) |

---

## Required Libraries & Dependencies

### Core ML/Data

```
numpy              >=1.21.0      # Numerical computing
tensorflow         >=2.10.0      # Deep learning (Keras API)
scikit-learn       >=1.0.0       # Preprocessing, metrics
scipy              >=1.7.0       # Signal processing
mne                >=0.24.0      # EEG processing utilities
```

### Structured Output & Serialization

```
pydantic           >=1.8.0       # Data validation, JSON serialization
```

### API & Server

```
fastapi            >=0.95.0      # REST framework
uvicorn            >=0.21.0      # ASGI server
pydantic           >=1.8.0       # Request validation
```

### Simulation

```
coppeliasim-zmqremoteapi-client  >=1.0.0  # CoppeliaSim ZMQ interface
```

### Development & Testing

```
pytest             >=6.0.0       # Unit testing
pytest-cov         >=2.12.0      # Coverage reporting
black              >=21.0        # Code formatting
```

### Complete `requirements.txt`

Create this file in project root:

```txt
numpy>=1.21.0
tensorflow>=2.10.0
scikit-learn>=1.0.0
scipy>=1.7.0
mne>=0.24.0
pydantic>=1.8.0
fastapi>=0.95.0
uvicorn>=0.21.0
coppeliasim-zmqremoteapi-client>=1.0.0
pytest>=6.0.0
pytest-cov>=2.12.0
black>=21.0
```

---

## Installation & Setup

### Step 1: Clone Repository

```bash
cd /path/to/workspace
git clone https://github.com/YOUR_REPO/BCI-Assistive-Motion-Grad-Project-.git
cd BCI-Assistive-Motion-Grad-Project-
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv bci_env
.\bci_env\bin\Activate.ps1
```

**Linux/macOS (Bash):**

```bash
python3 -m venv bci_env
source bci_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Or minimal install for API only (no simulation):**

```bash
pip install fastapi uvicorn pydantic numpy
```

**For full functionality including simulation:**

```bash
pip install fastapi uvicorn pydantic numpy coppeliasim-zmqremoteapi-client
```

### Step 4: Verify Installation

```bash
python -c "from bci_exo import BCIDecoder; print('✓ bci_exo OK')"
python -c "import fastapi; print('✓ fastapi OK')"
python -c "import pipeline; print('✓ pipeline OK')"
python api.py
# Should see: "Uvicorn running on http://0.0.0.0:8000"
```

---

## Running the Project

### Option A: Full Pipeline (No API)

**Single inference → mapping → simulation run:**

```bash
python pipeline.py
```

Uses synthetic data by default, dry-run mode (no simulator needed).

**From your own CSV predictions:**

```python
from pipeline import PipelineController, PipelineConfig

cfg = PipelineConfig(
    model_name="DB_ATCNet",
    subject_id=1,
    dry_run=True  # Set to False for real simulation
)
pipeline = PipelineController(cfg)
result = pipeline.run_from_results_file("path/to/predictions.csv")
print(result)
```

### Option B: REST API Server

**Start API:**

```bash
python api.py
```

**View interactive docs:**

```
http://localhost:8000/docs
```

**Test endpoints with curl:**

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Inference (POST)
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "parameters",
    "parameters": {"num_samples": 10, "seed": 7}
  }'

# Full pipeline (POST)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "parameters",
    "parameters": {"labels": [1, 2, 3, 1]},
    "simulation": {"enabled": false}
  }'
```

### Option C: Training New Models

**BCI 2A Dataset:**

```bash
python BCI_2A_main.py
```

**Physionet Motor Imagery:**

```bash
python Physionet_main.py
```

### Option D: Real-Time Simulation

**Prerequisites:**

1. CoppeliaSim installed (https://www.coppeliarobotics.com/)
2. Scene file: `scenes/bci_exo_scene.ttt`
3. ZMQ API enabled in CoppeliaSim

**Steps:**

1. Open CoppeliaSim
2. Load scene: `File → Open Scene → scenes/bci_exo_scene.ttt`
3. Start API:

```bash
python api.py
```

4. POST to `/run` with `simulation.enabled = true`:

```json
{
  "input_type": "parameters",
  "parameters": { "labels": [1, 2, 3] },
  "simulation": {
    "enabled": true,
    "host": "localhost",
    "port": 23000
  }
}
```

### Option E: One-Command Visual Simulation (Recommended)

Use the helper script to validate full visible motion in CoppeliaSim:

```bash
scripts/run_full_simulation.sh full
```

Quick shorter run:

```bash
scripts/run_full_simulation.sh quick
```

Save full run response JSON for reporting:

```bash
scripts/run_full_simulation_and_save.sh full
```

This writes timestamped files under `runs/`, e.g.:

```text
runs/simulation_run_YYYYMMDD_HHMMSS.json
```

What this script does automatically:

1. Checks that CoppeliaSim ZMQ API is listening on port `23000`
2. Starts API server if not already running
3. Sends `/run` request with simulation + feedback enabled
4. Prints summary (`signals_applied`, `feedback_iterations`, elapsed time)

---

## API Endpoints Reference

### Summary Table

| Method | Endpoint           | Purpose               | Input                    | Output              |
| ------ | ------------------ | --------------------- | ------------------------ | ------------------- |
| GET    | `/`                | Root info             | None                     | Service metadata    |
| GET    | `/health`          | Health check          | None                     | `{ status: ok }`    |
| GET    | `/models`          | List available models | None                     | `{ models: [...] }` |
| POST   | `/infer`           | Inference only        | Dataset/image/parameters | Probabilities       |
| POST   | `/map`             | Mapping only          | Probabilities or labels  | Control values      |
| POST   | `/run`             | Full pipeline         | Dataset/image/parameters | Pipeline result     |
| POST   | `/simulate`        | Simulation only       | Control values           | Simulation result   |
| GET    | `/status/{job_id}` | Async job status      | job_id                   | Job state           |
| GET    | `/result/{job_id}` | Async job result      | job_id                   | Job result          |

### Detailed Endpoint Descriptions

#### POST `/infer`

**Purpose:** Run AI inference only.  
**Request:**

```json
{
  "input_type": "parameters|dataset|image",
  "parameters": { "num_samples": 20, "seed": 7 },
  "dataset_path": "optional/path/to/file.csv"
}
```

**Response:**

```json
{
  "status": "completed",
  "input_type": "parameters",
  "predictions": [[0.1, 0.7, 0.1, 0.1], ...],
  "shape": [20, 4]
}
```

#### POST `/map`

**Purpose:** Convert probabilities/labels to control signals.  
**Request:**

```json
{
  "model_name": "DB_ATCNet",
  "subject_id": 1,
  "probabilities": [[0.1, 0.7, 0.1, 0.1], ...]
}
```

**Response:**

```json
{
  "status": "completed",
  "count": 2,
  "mapped_control_values": [
    {
      "trial_id": 0,
      "action": "Left Fist",
      "class_index": 1,
      "confidence": 0.7,
      "rotations": {
        "shoulder": {"yaw_deg": 0, "pitch_deg": 45, "roll_deg": 13.5},
        ...
      }
    }
  ]
}
```

#### POST `/run`

**Purpose:** Full pipeline: inference → mapping → simulation.  
**Request:**

```json
{
  "input_type": "parameters",
  "parameters": {
    "labels": [1, 2, 3, 1],
    "confidences": [0.9, 0.8, 0.95, 0.88]
  },
  "simulation": { "enabled": false },
  "feedback": {
    "enabled": false,
    "max_cycles_per_signal": 3,
    "correction_gain": 0.5,
    "error_tolerance_deg": 3.0,
    "collect_samples": true
  },
  "async_mode": false
}
```

**Sync Response (async_mode=false):**

```json
{
  "status": "completed",
  "input_type": "parameters",
  "simulation_enabled": false,
  "result": {
    "frames_inferred": 4,
    "frames_mapped": 4,
    "signals_applied": 4,
    "feedback_enabled": false,
    "feedback_iterations": 0,
    "feedback_samples": [],
    "elapsed_seconds": 0.0234
  }
}
```

**Async Response (async_mode=true):**

```json
{
  "status": "accepted",
  "job_id": "8a3c1e2b-4f9e-11eb-ae93-0242ac120008",
  "status_endpoint": "/status/8a3c1e2b-...",
  "result_endpoint": "/result/8a3c1e2b-..."
}
```

#### POST `/simulate`

**Purpose:** Apply pre-prepared control values to simulator.  
**Request:**

```json
{
  "simulation": {
    "enabled": true,
    "host": "localhost",
    "port": 23000
  },
  "control_values": [
    {
      "shoulder_pitch_deg": 25.0,
      "elbow_pitch_deg": 40.0,
      "wrist_roll_deg": -10.0,
      "pos_x": 0.2,
      "pos_y": 0.15,
      "pos_z": -0.1
    }
  ]
}
```

**Response:**

```json
{
  "status": "completed",
  "simulation_enabled": true,
  "signals_applied": 1
}
```

---

## Workflow Examples

### Example 1: Dry-Run Local Test

```bash
# Start API
python api.py &

# In another terminal, test inference
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"input_type": "parameters", "parameters": {"num_samples": 5}}'
```

### Example 2: Full Pipeline from Dataset

```python
from pipeline import PipelineController, PipelineConfig

cfg = PipelineConfig(
    model_name="DB_ATCNet",
    subject_id=1,
    dry_run=True  # No simulator needed
)

pipeline = PipelineController(cfg)
result = pipeline.run_from_results_file("BCI_2A_Results/perf_allRuns.npz")

print(f"Inferred: {result.frames_inferred}")
print(f"Mapped: {result.frames_mapped}")
print(f"Time: {result.elapsed_seconds}s")
```

### Example 3: Real-Time Simulation

```python
from pipeline import PipelineController, PipelineConfig
from sim_controller import SimConfig

sim_cfg = SimConfig(host="localhost", port=23000, loop_hz=20.0)

cfg = PipelineConfig(
    model_name="DB_ATCNet",
    subject_id=1,
    dry_run=False,
    sim=sim_cfg
)

pipeline = PipelineController(cfg)

# Requires CoppeliaSim running with scene loaded
labels = [1, 2, 3, 1, 2, 3]  # Motor imagery classes
result = pipeline.run_from_labels(labels)
```

### Example 4: Async API Job

```bash
# Submit async job
JOB_ID=$(curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "parameters",
    "parameters": {"num_samples": 50},
    "async_mode": true
  }' | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Poll status
while true; do
  curl http://localhost:8000/status/$JOB_ID | jq .
  sleep 2
done

# Get result
curl http://localhost:8000/result/$JOB_ID | jq .
```

### Example 5: Closed-Loop Feedback Run (Optional)

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "parameters",
    "parameters": {"labels": [1, 2, 3, 1]},
    "simulation": {"enabled": true, "host": "localhost", "port": 23000},
    "feedback": {
      "enabled": true,
      "max_cycles_per_signal": 3,
      "correction_gain": 0.5,
      "error_tolerance_deg": 3.0,
      "collect_samples": true
    }
  }'
```

---

## Deployment (Docker)

### Files

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

### Build and Run

```bash
docker build -t bci-online-simulation .
docker run --rm -p 8000:8000 bci-online-simulation
```

### Run with Compose

```bash
docker compose up --build
```

### Notes

- API will be available at `http://localhost:8000`
- If simulation is disabled (`simulation.enabled=false`), container can run without CoppeliaSim
- For real simulation control, CoppeliaSim must be reachable from the container network

---

## Feedback Loop (Optional)

When enabled in `/run`, the pipeline performs:

1. Apply mapped control signal.
2. Read simulator joint positions.
3. Compute error vs target angles.
4. Apply bounded corrective update.
5. Repeat until tolerance or max cycles reached.

### Feedback Configuration

- `enabled`: turn loop on/off
- `max_cycles_per_signal`: max correction attempts per signal
- `correction_gain`: proportional correction factor in `[0, 1]`
- `error_tolerance_deg`: stop condition in degrees
- `collect_samples`: include telemetry snapshots in response

### Returned Feedback Fields

- `feedback_enabled`
- `feedback_iterations`
- `feedback_samples`

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**

```bash
pip install fastapi uvicorn
```

### Issue: `ModuleNotFoundError: No module named 'coppeliasim_zmqremoteapi_client'`

**Solution (optional for simulation only):**

```bash
pip install coppeliasim-zmqremoteapi-client
```

Or run with `simulation.enabled = false` to skip.

### Issue: `TimeoutError: Simulation did not start within timeout`

**Causes & Solutions:**

1. CoppeliaSim not running → Start CoppeliaSim application
2. Scene not loaded → File → Open Scene → bci_exo_scene.ttt
3. ZMQ API disabled → Tools → ZMQ remote API → Enable
4. Wrong port → Check sim_controller.SimConfig port (default 23000)

### Issue: API returns 404 on GET requests

**Solution:**

- Only `/health`, `/models`, `/status/{job_id}`, `/result/{job_id}` accept GET
- `/run`, `/infer`, `/map`, `/simulate` require POST
- Use Swagger UI (`/docs`) or Postman, not browser URL bar

### Issue: Environment activation fails (PowerShell)

**Solution:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\bci_env\bin\Activate.ps1
```

### Issue: Model training out of memory

**Solution:**
Reduce batch size in training script:

```python
batch_size = 16  # Reduce from 32
```

### Issue: Docker build fails on dependency compilation

**Solution:**

```bash
docker build --no-cache -t bci-online-simulation .
```

If host resources are limited, close heavy applications and retry.

### Issue: Container cannot reach CoppeliaSim

**Solution:**

- Ensure CoppeliaSim is running and ZMQ API is enabled.
- Ensure host/port in API request matches reachable address from container.
- If needed, run container with host networking on Linux:

```bash
docker run --rm --network host bci-online-simulation
```

---

## Development Notes

### Code Style

All code follows PEP 8. Format with:

```bash
black pipeline.py api.py sim_controller.py
```

### Testing

Run unit tests:

```bash
pytest test_sim_controller.py -v
pytest test_step.py -v
```

### Logging

All modules use Python's `logging` module. Configure verbosity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # More verbose
```

### Adding New Models

1. Add model class to `models.py`
2. Update `AVAILABLE_MODELS` in `api.py`
3. Retrain and save checkpoint
4. Endpoint `/models` will automatically include it

---

## Summary

| Component                | Status      | Location                           |
| ------------------------ | ----------- | ---------------------------------- |
| Data Loading             | ✅ Complete | `Physionet_DataLoad.py`            |
| Preprocessing            | ✅ Complete | `BCI2A_preprocess.py`              |
| Model Training           | ✅ Complete | `attention_models.py`, `models.py` |
| Structured Output        | ✅ Complete | `bci_exo/` package                 |
| Simulation               | ✅ Complete | `sim_controller.py`                |
| Pipeline Orchestration   | ✅ Complete | `pipeline.py`                      |
| REST API                 | ✅ Complete | `api.py`                           |
| Deployment (Docker)      | ✅ Complete | `Dockerfile`, `docker-compose.yml` |
| Feedback Loop (Optional) | ✅ Complete | `pipeline.py`, `api.py`            |
| Documentation            | ✅ Complete | This file                          |

**Project is production-ready for:**

- ✅ Offline batch inference
- ✅ Real-time simulation control
- ✅ Online REST API access
- ✅ Async job management
- ✅ Multi-model support
- ✅ Docker deployment
- ✅ Optional closed-loop feedback

---

**For questions or issues, refer to endpoint documentation at `/docs` when API is running, or review code docstrings in each module.**
