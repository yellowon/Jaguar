export PATH="${HOME}/.local/bin:${PATH}"
RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

case ":${PATH}:" in
  *:/usr/local/bin:*) ;;
  *) export PATH="${PATH}:/usr/local/bin" ;;
esac

function has_tool {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo -e "No ${RED}${tool}${NC} is found."
    exit 1
  fi
  echo -e "${GREEN}${tool}${NC} found."
}

function check_tools {
  has_tool g++
  has_tool make
  has_tool cmake
}

function contains {
  local haystack="$1"
  local needle="$2"
  case " ${haystack} " in
    *" ${needle} "*) return 0 ;;
    *) return 1 ;;
  esac
}

WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-${WORK_DIR}/build_avxon}"
DEPS_DIR="${DEPS_DIR:-${WORK_DIR}/deps}"

case "${BUILD_DIR}" in
  /*) ;;
  *) BUILD_DIR="${WORK_DIR}/${BUILD_DIR#./}" ;;
esac
case "${DEPS_DIR}" in
  /*) ;;
  *) DEPS_DIR="${WORK_DIR}/${DEPS_DIR#./}" ;;
esac

PREFIX_DIR="${PREFIX_DIR:-${BUILD_DIR}/deps-prefix}"
case "${PREFIX_DIR}" in
  /*) ;;
  *) PREFIX_DIR="${WORK_DIR}/${PREFIX_DIR#./}" ;;
esac

if [ -n "${JAGUAR_BIN:-}" ]; then
  case "${JAGUAR_BIN}" in
    /*) ;;
    *) JAGUAR_BIN="${WORK_DIR}/${JAGUAR_BIN#./}" ;;
  esac
fi
if [ -n "${JAGUAR_BIN:-}" ] && [ -x "${JAGUAR_BIN}" ]; then
  :
else
  JAGUAR_BIN="${BUILD_DIR}/jaguar_main"
fi
export JAGUAR_EXACT_LIBRARY="${JAGUAR_EXACT_LIBRARY:-${WORK_DIR}/exact/libjaguar_exact_v51.so}"
case "${JAGUAR_EXACT_LIBRARY}" in
  /*) ;;
  *) JAGUAR_EXACT_LIBRARY="${WORK_DIR}/${JAGUAR_EXACT_LIBRARY#./}" ;;
esac
export LD_LIBRARY_PATH="${PREFIX_DIR}/lib:${PREFIX_DIR}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
JAGUAR_LOG_DIR="${WORK_DIR}/logs"
JAGUAR_RUN_DIR="${JAGUAR_LOG_DIR}/pair_runs"

SERVER_IP="${SERVER_IP:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-19001}"
NUM_THREADS="${NUM_THREADS:-1}"
JAGUAR_DEFAULT_MODE="${JAGUAR_DEFAULT_MODE:-st}"

mkdir -p "${BUILD_DIR}" "${DEPS_DIR}" "${PREFIX_DIR}" "${JAGUAR_LOG_DIR}" "${JAGUAR_RUN_DIR}"
