#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SAVE_JSON=1 RUNS_DIR="${ROOT_DIR}/runs" "${ROOT_DIR}/scripts/run_full_simulation.sh" "${1:-full}"
