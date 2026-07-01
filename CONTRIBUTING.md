# Contributing to NeuroMove

Thanks for your interest in **NeuroMove** — *Design & Development of a Brain–Computer Interface for
Assistive Motion Control, Rehabilitation & Mobility Applications* (E-JUST graduation project,
2025/2026).

This is a research monorepo with several independent components (AI models, FPGA RTL, ESP32
firmware, COMSOL FEA, and a CoppeliaSim simulation). Please read the section for the component you're
touching — each has its own toolchain.

---

## Ground rules

- **Open an issue first** for anything non-trivial (a bug, a proposed feature, a refactor) so we can
  agree on the approach before code is written.
- **One component per pull request** where possible — don't mix an FPGA change with an ML change.
- **Never commit large or generated artifacts.** Datasets, model checkpoints, Vivado runs, results,
  and virtual environments are git-ignored on purpose (see the root [`.gitignore`](.gitignore) and
  each component's `data/`/`results/` README). If `git status` shows a `.pt`, `.h5`, `.gdf`, `.edf`,
  `.mat`, `.bit`, `.dcp`, or a `venv/`, stop and check `.gitignore`.
- **Respect dataset licenses.** Raw EEG/fNIRS datasets are downloaded, never redistributed here —
  see [`data/README.md`](data/README.md).

---

## Repository layout

| Path | Component | Toolchain |
|---|---|---|
| `machine-learning/` | Classical ML motor-imagery pipeline | Python (scikit-learn, MNE), notebooks |
| `deep-learning/` | DB-ATCNet edge-optimized classifier | Python (TensorFlow) |
| `cross-modal-synthesis/` | SCDM EEG→fNIRS diffusion model | Python (PyTorch) |
| `Secure-EEG-…-main/` | EEG biometric ID / authentication | Python (TensorFlow) |
| `fpga/` | SystemVerilog accelerator + crypto | Vivado 2025.2 |
| `fpga/sw/esp32/` | ESP32-S3 motor-controller firmware | Arduino / ESP32-S3 |
| `FEA-Biomaterial-…-main/` | COMSOL FEA of the prosthesis | COMSOL 6.1 |
| `simulation/coppeliasim/` | Online BCI→exoskeleton simulation | CoppeliaSim, FastAPI, Docker |

---

## Python environment (shared AI stack)

Most Python components share one environment:

```bash
conda env create -f environment.yml   # creates the `neuromove` env
conda activate neuromove
# or, without conda:
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

> The `simulation/coppeliasim/` component has its **own** `requirements.txt` (FastAPI/uvicorn/ZMQ) —
> install that separately when working there.

---

## Working on each component

**`deep-learning/` (DB-ATCNet)** — run from `deep-learning/DB-ATCNet/`:
```bash
python HALT_main.py --single-run                 # 19-ch baseline (LOSO)
python HALT_main.py --gumbel-select --gumbel-k 5 # learned 5-channel selection
```
Hyperparameters live in [`deep-learning/configs/config.yaml`](deep-learning/configs/config.yaml).
The frozen inference graph is specified in
[`deep-learning/FPGA_DEPLOYMENT_DOC.md`](deep-learning/FPGA_DEPLOYMENT_DOC.md) — **if you change the
model, update that doc and re-export weights/goldens for the FPGA.**

**`cross-modal-synthesis/` (SCDM)** — run from that folder:
```bash
PYTHONPATH=. python tests/test_shapes.py         # fast shape/integration check (no data)
PYTHONPATH=. python scripts/train.py --config configs/config.yaml
```

**`machine-learning/`** — the pipeline lives in `notebooks/`; keep new work in notebooks or promote
shared code into the `src/` package (currently a scaffold).

**`fpga/`** — needs Vivado 2025.2:
```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
./build.sh                          # parse + elaborate
./sim/run_vivado_tests.sh top       # bit-exact vs. Python golden
```
RTL is **Q8.8 fixed-point and must stay bit-exact** against the golden model — every module has a
testbench under `fpga/sim/`. Add/extend a testbench for any RTL change and keep the goldens in sync.

**`simulation/coppeliasim/`** — with CoppeliaSim running and `scenes/bci_exo_scene.ttt` loaded:
```bash
pip install -r requirements.txt
python api.py                        # FastAPI at http://localhost:8000/docs
scripts/run_full_simulation.sh full
```

**Cross-component integration tests:**
```bash
pytest tests/
```

---

## Coding conventions

- **Python** — PEP 8, 4-space indent, type hints on new public functions, docstrings on modules and
  non-trivial functions. Match the style of the file you're editing.
- **SystemVerilog** — follow the existing `fpga/rtl/` conventions: Q8.8 everywhere, saturate+round
  at stage boundaries, module-level `(* use_dsp = ... *)`, flat 1-D `block`-RAM buffers (see
  `fpga/rtl/README.md`).
- **Firmware** — keep the ESP32 controller's safety invariants (angle clamps, rate limiting,
  majority-vote debounce, watchdog/fault states). Don't add code paths that can drive a servo
  outside its configured limits.
- Keep comments at the density of the surrounding code; explain *why*, not *what*.

---

## Commits & pull requests

- **Commits:** use short, imperative, scoped messages — the repo already uses a
  Conventional-Commits style, e.g. `feat: add gumbel channel selector`, `fix: correct HRF kernel
  scaling`, `docs: update FPGA deployment spec`.
- **Branch** off `main`; don't push directly to `main`.
- **Before opening a PR:** run the relevant component's tests, make sure `git status` is free of
  generated artifacts, and update any README/doc your change affects.
- **PR description:** state which component(s) it touches, what changed and why, and how you tested
  it. Link the issue it closes.

---

## Questions

Open a GitHub issue, or contact the maintainers — see [SECURITY.md](SECURITY.md) for the security
contact. This project is supervised by **Dr. Reda Albassiouny** & **Dr. Sameh Sherif** at
Egypt-Japan University of Science and Technology (E-JUST).
