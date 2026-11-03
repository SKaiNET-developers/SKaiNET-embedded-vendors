#!/bin/bash
# build-stablehlo-opt.sh <src> <build>: stablehlo-opt (StableHLO's TOSA legalization) from a torq-compiler checkout,
# with stock IREE CMake (no Torq plugin), so it uses the LLVM the Torq compiler is built from. Run in the builder image.
set -euo pipefail
SRC=$1; B=$2
cmake -S "$SRC/third_party/iree" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIREE_ENABLE_LLD=ON \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_BUILD_BUNDLED_LLVM=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_TOSA=OFF -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
cmake --build "$B" --target stablehlo-opt -- -j "${JOBS:-11}"
ls -la "$B/llvm-project/bin/stablehlo-opt"
