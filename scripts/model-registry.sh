#!/usr/bin/env bash
set -euo pipefail

function jaguar_validate_protocol {
  local protocol="$1"
  case "${protocol}" in
    jaguar) ;;
    *)
      echo "Unsupported protocol: ${protocol}"
      return 1
      ;;
  esac
}

function jaguar_validate_model {
  local model="$1"
  case "${model}" in
    resnet18|resnet50|mobilenet_v2) ;;
    *)
      echo "Unsupported model: ${model}"
      return 1
      ;;
  esac
}

function jaguar_model_args {
  local model="$1"
  case "${model}" in
    resnet18)
      echo "2048 15 224 3 3 64"
      ;;
    resnet50)
      echo "2048 15 224 3 3 64"
      ;;
    mobilenet_v2)
      echo "2048 15 224 3 3 32"
      ;;
  esac
}

function jaguar_model_note {
  local model="$1"
  case "${model}" in
    resnet18)
      echo "ImageNet-style 224x224 path"
      ;;
    resnet50)
      echo "ImageNet-style 224x224 path"
      ;;
    mobilenet_v2)
      echo "ImageNet-style 224x224 MobileNetV2 path"
      ;;
    *)
      echo ""
      ;;
  esac
}

function jaguar_emit_command {
  local party="$1"
  local model="$2"
  local mode="${3:-${JAGUAR_DEFAULT_MODE}}"
  local peer_ip="${4:-${SERVER_IP}}"
  local port="${5:-${SERVER_PORT}}"
  local args
  args="$(jaguar_model_args "${model}")"
  local -a model_args
  read -r -a model_args <<< "${args}"
  printf '%q ' "${JAGUAR_BIN}" "${model_args[@]}" "${model}" "${mode}" 1 1 0 0 0 0 0 --jaguar-party "${party}" --jaguar-peer "${peer_ip}" --jaguar-port "${port}" --USE_FUSED_BN 1
  printf '\n'
}
