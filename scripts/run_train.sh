#!/usr/bin/env bash
# Quick trigger script for training sessions
#
# Usage:
#   scripts/run_train.sh textual_inversion
#   scripts/run_train.sh dreambooth
#
# Point at a different YAML by exporting CONFIG_FILE=... before running the script
set -euo pipefail

CHOSEN_METHOD="${1:-textual_inversion}"
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

case "${CHOSEN_METHOD}" in
  textual_inversion)
    DEFAULT_CONFIG_PATH="${REPOSITORY_ROOT}/configs/textual_inv_cfg.yaml"
    ;;
  dreambooth)
    DEFAULT_CONFIG_PATH="${REPOSITORY_ROOT}/configs/dreambooth_cfg.yaml"
    ;;
  *)
    echo "Unknown method: ${CHOSEN_METHOD} (expected textual_inversion|dreambooth)" >&2
    exit 1
    ;;
esac

CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_PATH}}"

echo "Running ${CHOSEN_METHOD} with config ${CONFIG_FILE}"
python "${REPOSITORY_ROOT}/scripts/train.py" \
  --method "${CHOSEN_METHOD}" \
  --config "${CONFIG_FILE}"
