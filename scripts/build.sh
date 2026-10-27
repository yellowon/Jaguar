#!/usr/bin/env bash
set -euo pipefail

if command -v scl >/dev/null 2>&1 && [ "${Jaguar_SCL_ENABLED:-0}" != "1" ]; then
  gcc_major="$(g++ -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
  if [ "${gcc_major}" -lt 9 ] && scl -l 2>/dev/null | grep -q '^devtoolset-11$'; then
    exec scl enable devtoolset-11 "env Jaguar_SCL_ENABLED=1 bash \"$0\""
  fi
fi

. "$(dirname "$0")/common.sh"

"${WORK_DIR}/scripts/build-deps.sh"
cmake -S "${WORK_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${PREFIX_DIR}" -DUSE_AVX512="${USE_AVX512:-ON}"
cmake --build "${BUILD_DIR}" --target client -j"${BUILD_JOBS:-$(nproc)}"
