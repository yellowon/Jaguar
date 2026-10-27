cmake_minimum_required(VERSION 3.16)
project(Jaguar LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(JAGUAR_PREFIX "${CMAKE_BINARY_DIR}/deps-prefix")
list(APPEND CMAKE_PREFIX_PATH "${JAGUAR_PREFIX}")

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Eigen3 3.3.3 REQUIRED PATHS "${JAGUAR_PREFIX}" NO_DEFAULT_PATH)
find_package(SEAL 4.1.1 EXACT REQUIRED PATHS "${JAGUAR_PREFIX}" NO_DEFAULT_PATH)
find_package(CpuFeatures QUIET PATHS "${JAGUAR_PREFIX}" NO_DEFAULT_PATH)
find_package(emp-tool REQUIRED PATHS "${JAGUAR_PREFIX}" NO_DEFAULT_PATH)
find_package(emp-ot REQUIRED PATHS "${JAGUAR_PREFIX}" NO_DEFAULT_PATH)

# CMake drops /usr/include as an implicit include path, which can let headers
# from another OpenSSL in /usr/local override the selected library. Expose only
# the selected openssl directory so C++ standard header ordering stays intact.
set(JAGUAR_OPENSSL_HEADERS "${CMAKE_BINARY_DIR}/openssl-include")
file(MAKE_DIRECTORY "${JAGUAR_OPENSSL_HEADERS}")
file(CREATE_LINK "${OPENSSL_INCLUDE_DIR}/openssl"
  "${JAGUAR_OPENSSL_HEADERS}/openssl" SYMBOLIC)

add_library(jaguar_sci_support
  Jaguar_src/jaguar_profiler.cpp
  Jaguar_src/jaguar_relu.cpp
  SCI/src/globals.cpp
  SCI/src/library_fixed.cpp
  SCI/src/library_fixed_uniform.cpp
  SCI/src/library_fixed_uniform_cheetah.cpp
  SCI/src/cleartext_library_fixed.cpp
  SCI/src/cheetah/cheetah-api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/cheetah/hom_bn_ss.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/cheetah/hom_conv2d_ss.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/cheetah/hom_fc_ss.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/cheetah/shape_inference.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/cheetah/tensor_encoder.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/cheetah/tensor_shape.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/core/logging.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/core/types.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/core/util/math.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gemini/core/util/seal.cc
  SCI/src/LinearOT/linear-ot.cpp
  SCI/src/Math/math-functions.cpp
  SCI/src/BuildingBlocks/value-extension.cpp
  SCI/src/BuildingBlocks/aux-protocols.cpp
  SCI/src/BuildingBlocks/truncation.cpp
)

target_include_directories(jaguar_sci_support
  PUBLIC
    ${JAGUAR_OPENSSL_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/SEAL/native/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/SCI/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENSSL_INCLUDE_DIR}
    ${EMP-TOOL_INCLUDE_DIRS}
    ${EMP-OT_INCLUDE_DIRS}
)

option(USE_AVX512 "Enable AVX-512 optimized Jaguar quant kernels" OFF)

set(JAGUAR_COMMON_COMPILE_FLAGS
  -pthread -maes -mrdseed -msse4.1 -mavx -mavx2 -faligned-new -O3
)
if(USE_AVX512)
  list(APPEND JAGUAR_COMMON_COMPILE_FLAGS -mavx512f -mavx512dq)
endif()

target_compile_options(jaguar_sci_support
  PUBLIC ${JAGUAR_COMMON_COMPILE_FLAGS}
)

target_compile_definitions(jaguar_sci_support PUBLIC _GLIBCXX_USE_CXX11_ABI=0 SCI_OT=1 USE_CHEETAH=1)
if(USE_AVX512)
  target_compile_definitions(jaguar_sci_support PUBLIC USE_AVX512=1)
endif()

target_link_libraries(jaguar_sci_support
  PUBLIC
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
    ${EMP-TOOL_LIBRARIES}
    ${EMP-OT_LIBRARIES}
    Eigen3::Eigen
    SEAL::seal
)

# HEXL leaves cpu_features external when it finds an installed package.
# SEAL's combined static archive does not export this dependency.
if(TARGET CpuFeatures::cpu_features)
  target_link_libraries(jaguar_sci_support PUBLIC CpuFeatures::cpu_features)
endif()

add_library(jaguar_rerandomization Jaguar_src/jaguar_rerandomize.cpp)
target_include_directories(jaguar_rerandomization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Jaguar_src)
target_compile_definitions(jaguar_rerandomization PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(jaguar_rerandomization PUBLIC SEAL::seal Threads::Threads)
if(TARGET CpuFeatures::cpu_features)
  target_link_libraries(jaguar_rerandomization PUBLIC CpuFeatures::cpu_features)
endif()

add_executable(client
  Jaguar_src/jaguar_main.cpp
  Jaguar_src/client_lwe_decrypt.cpp
  Jaguar_src/client_lwe_export.cpp
  Jaguar_src/jaguar_transport.cpp
  Jaguar_src/client_layer_helpers.cpp
)
set_target_properties(client PROPERTIES OUTPUT_NAME jaguar_main)
target_include_directories(client PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/SEAL/native/src
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/SCI/src
  ${CMAKE_CURRENT_SOURCE_DIR}/networks
)
target_compile_definitions(client PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
if(USE_AVX512)
  target_compile_options(client PRIVATE -mavx512f -mavx512dq)
  target_compile_definitions(client PRIVATE USE_AVX512=1)
endif()
target_link_libraries(client PRIVATE jaguar_sci_support jaguar_rerandomization SEAL::seal OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})  # Jaguar_52: dlopen of the exact ReLU backend
