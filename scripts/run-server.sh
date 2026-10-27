#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/common.sh"
. "$(dirname "$0")/model-registry.sh"

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo -e "${RED}Please specify protocol and network.${NC}"
  echo "Usage: run-server.sh [jaguar] [resnet18|resnet50|mobilenet_v2] [st|mt]"
  exit 1
fi

protocol="$1"
model="$2"
mode="${3:-${JAGUAR_DEFAULT_MODE}}"

jaguar_validate_protocol "${protocol}"
jaguar_validate_model "${model}"

if [ ! -x "${JAGUAR_BIN}" ]; then
  echo -e "${RED}${JAGUAR_BIN}${NC} is missing. Run bash scripts/build.sh first."
  exit 1
fi

mkdir -p "${JAGUAR_LOG_DIR}"
log_file="${JAGUAR_LOG_DIR}/${protocol}-${model}_server.log"
cmd="$(jaguar_emit_command cloud "${model}" "${mode}" "${SERVER_IP}" "${SERVER_PORT}")"

echo -e "Running ${GREEN}${model}${NC} on ${GREEN}${protocol}${NC} as server/cloud"
echo "Model note: $(jaguar_model_note "${model}")"
echo "Command: ${cmd}"
echo "Log: ${log_file}"

mkdir -p "${WORK_DIR}/run_cloud"
cd "${WORK_DIR}/run_cloud"
eval "${cmd}" 2>&1 | tee "${log_file}"
