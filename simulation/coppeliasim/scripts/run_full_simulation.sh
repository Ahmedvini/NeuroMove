#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
API_URL="${API_URL:-http://localhost:8000}"
HOST="${SIM_HOST:-localhost}"
PORT="${SIM_PORT:-23000}"
MODE="${1:-full}"
SAVE_JSON="${SAVE_JSON:-0}"
RUNS_DIR="${RUNS_DIR:-${ROOT_DIR}/runs}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python environment not found at ${PYTHON_BIN}" >&2
  echo "Create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl is required but not installed." >&2
  exit 1
fi

if ! command -v ss >/dev/null 2>&1; then
  echo "[ERROR] ss command is required to check CoppeliaSim port." >&2
  exit 1
fi

if [[ "${MODE}" != "full" && "${MODE}" != "quick" ]]; then
  echo "Usage: scripts/run_full_simulation.sh [full|quick]" >&2
  exit 1
fi

if ! ss -ltn | grep -q ":${PORT} "; then
  echo "[ERROR] CoppeliaSim ZMQ API not detected on port ${PORT}." >&2
  echo "Start CoppeliaSim, load scenes/bci_exo_scene.ttt, and enable ZMQ Remote API." >&2
  exit 1
fi

echo "[INFO] CoppeliaSim is listening on ${HOST}:${PORT}"

API_PID=""
cleanup() {
  if [[ -n "${API_PID}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
  echo "[INFO] API not running, starting ${PYTHON_BIN} api.py"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" api.py >/tmp/bci_api.log 2>&1
  ) &
  API_PID=$!

  for _ in {1..30}; do
    if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
      echo "[INFO] API is ready at ${API_URL}"
      break
    fi
    sleep 1
  done

  if ! curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
    echo "[ERROR] API failed to start. Check /tmp/bci_api.log" >&2
    exit 1
  fi
else
  echo "[INFO] API already running at ${API_URL}"
fi

if [[ "${MODE}" == "quick" ]]; then
  LABELS='[1,2,3,1,2,3]'
  MAX_CYCLES=1
else
  LABELS='[1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]'
  MAX_CYCLES=2
fi

PAYLOAD=$(cat <<JSON
{
  "input_type": "parameters",
  "parameters": {
    "labels": ${LABELS}
  },
  "simulation": {
    "enabled": true,
    "host": "${HOST}",
    "port": ${PORT}
  },
  "feedback": {
    "enabled": true,
    "max_cycles_per_signal": ${MAX_CYCLES},
    "correction_gain": 0.5,
    "error_tolerance_deg": 3.0,
    "collect_samples": true
  },
  "async_mode": false
}
JSON
)

echo "[INFO] Triggering simulation run (${MODE} mode). Watch CoppeliaSim window for motion..."
RESPONSE=$(curl -fsS -X POST "${API_URL}/run" -H "Content-Type: application/json" -d "${PAYLOAD}")

echo "[INFO] API response received."

if [[ "${SAVE_JSON}" == "1" ]]; then
  mkdir -p "${RUNS_DIR}"
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_FILE="${RUNS_DIR}/simulation_run_${TS}.json"
  printf '%s\n' "${RESPONSE}" > "${OUT_FILE}"
  echo "[INFO] Saved full response to ${OUT_FILE}"
fi

RESPONSE_JSON="${RESPONSE}" "${PYTHON_BIN}" - <<'PY'
import json
import os

raw = os.environ.get("RESPONSE_JSON", "")
data = json.loads(raw)
res = data.get("result", {})
print("status:", data.get("status"))
print("simulation_enabled:", data.get("simulation_enabled"))
print("frames_inferred:", res.get("frames_inferred"))
print("frames_mapped:", res.get("frames_mapped"))
print("signals_applied:", res.get("signals_applied"))
print("feedback_iterations:", res.get("feedback_iterations"))
print("elapsed_seconds:", res.get("elapsed_seconds"))
PY

echo "[INFO] Done."
