#!/usr/bin/env bash
# Bootstrap the Python environment for the EEG–fNIRS BCI project.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1; then
    echo ">> Creating conda environment from environment.yml"
    conda env create -f environment.yml || conda env update -f environment.yml
    echo ">> Done. Activate with: conda activate eeg-fnirs-bci"
else
    echo ">> conda not found; using venv + pip"
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo ">> Done. Activate with: source .venv/bin/activate"
fi
