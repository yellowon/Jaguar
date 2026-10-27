#!/usr/bin/env bash
set -euo pipefail

if command -v scl >/dev/null 2>&1 && [ "${Jaguar_SCL_ENABLED:-0}" != "1" ]; then
  gcc_major="$(g++ -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)"
  if [ "${gcc_major}" -lt 9 ] && scl -l 2>/dev/null | grep -q '^devtoolset-11$'; then
    exec scl enable devtoolset-11 "env Jaguar_SCL_ENABLED=1 bash \"$0\""
  fi
fi

. "$(dirname "$0")/common.sh"

check_tools

function build_dep {
  local target="$1"
  local source_dir="$2"
  shift 2

  mkdir -p "${BUILD_DIR}/${target}"
  cd "${BUILD_DIR}/${target}"
  cmake "${source_dir}" -DCMAKE_INSTALL_PREFIX="${PREFIX_DIR}" "$@"
  cmake --build . -j"${BUILD_JOBS:-$(nproc)}"
  cmake --install .
}

build_dep emp-tool "${DEPS_DIR}/emp-tool" -DCMAKE_CXX_STANDARD=17 -DBUILD_TESTING=OFF
build_dep emp-ot "${DEPS_DIR}/emp-ot" -DCMAKE_PREFIX_PATH="${PREFIX_DIR}" -DCMAKE_CXX_STANDARD=17 -DBUILD_TESTING=OFF
build_dep eigen "${DEPS_DIR}/eigen" \
  -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_TESTING=OFF \
  -DEIGEN_BUILD_DOC=OFF \
  -DEIGEN_BUILD_PKGCONFIG=OFF
# Build the pinned cpu_features source locally so HEXL need not download it.
cpu_features_source="${CPU_FEATURES_SOURCE_DIR:-${DEPS_DIR}/cpu_features}"
if [ -f "${cpu_features_source}/CMakeLists.txt" ]; then
  build_dep cpu_features "${cpu_features_source}" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_PIC=ON -DBUILD_TESTING=OFF
fi

seal_source_args=()
hexl_source="${HEXL_SOURCE_DIR:-${DEPS_DIR}/hexl-1.2.5}"
if [ -f "${hexl_source}/CMakeLists.txt" ]; then
  # SEAL 4.1.1 builds HEXL beside its source. Stage source under BUILD_DIR
  # so all generated files stay in the disposable build directory.
  staged_hexl="${BUILD_DIR}/SEAL/thirdparty/hexl-src"
  mkdir -p "${staged_hexl}"
  cmake -E copy_directory "${hexl_source}" "${staged_hexl}"
  seal_source_args+=("-DFETCHCONTENT_SOURCE_DIR_HEXL=${staged_hexl}")
fi

build_dep SEAL "${DEPS_DIR}/SEAL" \
  -DCMAKE_PREFIX_PATH="${PREFIX_DIR}" \
  -DCMAKE_CXX_STANDARD=17 \
  -DSEAL_BUILD_DEPS=ON \
  -DSEAL_USE_MSGSL=OFF \
  -DSEAL_USE_INTEL_HEXL=ON \
  -DSEAL_USE_ZLIB=OFF \
  -DSEAL_USE_ZSTD=OFF \
  -DSEAL_BUILD_EXAMPLES=OFF \
  -DSEAL_BUILD_TESTS=OFF \
  -DSEAL_BUILD_BENCH=OFF \
  -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON \
  "${seal_source_args[@]}"
