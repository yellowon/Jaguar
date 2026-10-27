#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/common.sh"
. "$(dirname "$0")/model-registry.sh"

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
  echo "Usage: jaguar_run_model_pair.sh [model] [run_id] [port] [st|mt]"
  exit 1
fi

model="$1"
run_id="${2:-${model}_$(date +%Y%m%d_%H%M%S)}"
port="${3:-${SERVER_PORT}}"
mode="${4:-${JAGUAR_DEFAULT_MODE}}"

jaguar_validate_model "${model}"

run_dir="${JAGUAR_RUN_DIR}/${run_id}"
summary_dir="${run_dir}/summary"
mkdir -p "${run_dir}" "${summary_dir}"

client_log="${run_dir}/client.log"
cloud_log="${run_dir}/cloud.log"
client_status="${run_dir}/client.status"
cloud_status="${run_dir}/cloud.status"
client_pid_file="${run_dir}/client.pid"
cloud_pid_file="${run_dir}/cloud.pid"

rm -f "${client_status}" "${cloud_status}"

client_cmd="$(jaguar_emit_command client "${model}" "${mode}" "127.0.0.1" "${port}")"
cloud_cmd="$(jaguar_emit_command cloud "${model}" "${mode}" "127.0.0.1" "${port}")"
mkdir -p "${WORK_DIR}/run_client" "${WORK_DIR}/run_cloud"

(
  cd "${WORK_DIR}/run_client"
  export JAGUAR_LOG_DIR="${summary_dir}"
  export JAGUAR_LOG_FILE="${summary_dir}/jaguar_protocol_summary.log"
  export JAGUAR_LAYER_CSV="${summary_dir}/jaguar_layer_summary.csv"
  exit_code=0
  bash -c "${client_cmd}" >"${client_log}" 2>&1 || exit_code=$?
  echo "${exit_code}" >"${client_status}"
) &
echo $! >"${client_pid_file}"
sleep 1
(
  cd "${WORK_DIR}/run_cloud"
  export JAGUAR_LOG_DIR="${summary_dir}"
  export JAGUAR_LOG_FILE="${summary_dir}/jaguar_protocol_summary.log"
  export JAGUAR_LAYER_CSV="${summary_dir}/jaguar_layer_summary.csv"
  exit_code=0
  bash -c "${cloud_cmd}" >"${cloud_log}" 2>&1 || exit_code=$?
  echo "${exit_code}" >"${cloud_status}"
) &
echo $! >"${cloud_pid_file}"

echo "run_id=${run_id}"
echo "model=${model}"
echo "client_pid=$(cat "${client_pid_file}")"
echo "cloud_pid=$(cat "${cloud_pid_file}")"
echo "run_dir=${run_dir}"
