#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${ROOT_DIR}/joint_angle_3d_palm_rl/runs"

find_latest_model() {
  find "${RUNS_DIR}" -name "best_model.zip" -print0 2>/dev/null \
    | xargs -0 ls -t 2>/dev/null \
    | head -n 1
}

MODEL_PATH=""
if [[ $# -ge 1 && -f "$1" ]]; then
  MODEL_PATH="$1"
  shift
else
  MODEL_PATH="$(find_latest_model || true)"
fi

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
  echo "[error] best_model.zip not found."
  echo "Usage:"
  echo "  ./run_simple_drag_render.sh /absolute/path/to/best_model.zip [extra args]"
  exit 1
fi

echo "[model] ${MODEL_PATH}"

CMD=(
  python -m joint_angle_3d_palm_rl.render_simple_reach
  --model "${MODEL_PATH}"
  --hide-background
  --no-project-targets-to-reachable
  --no-terminate-on-success
  --max-steps 200000
  --control-hz 30
  --interactive-render
)

# Allow user overrides, e.g.:
# ./run_simple_drag_render.sh /path/best_model.zip --success-steps-required 15
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

cd "${ROOT_DIR}"
if command -v conda >/dev/null 2>&1; then
  ENV_NAME="${CONDA_ENV_NAME:-vibe}"
  echo "[run] conda env=${ENV_NAME}"
  conda run -n "${ENV_NAME}" "${CMD[@]}"
else
  echo "[run] system python"
  "${CMD[@]}"
fi
