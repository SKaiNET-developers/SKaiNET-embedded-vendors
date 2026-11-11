#!/usr/bin/env bash
# Runs while the image builds: fetches the public sources at the pins and cross-builds the Torq-fork IREE runtime
# (local-sync, local-task, vulkan and torq HAL drivers) into one static archive for the board.
#   build-runtime.sh <torq tag> <work dir> <dist dir> <directory of toolchain.aarch64.cmake>
set -euo pipefail
TAG=$1; W=$2; OUT=$3; HERE=$4
mkdir -p "$W" "$OUT/lib" "$OUT/include" "$OUT/licenses"

fetch() { # <dir> <url> <commit>
  mkdir -p "$1" && git -C "$1" init -q && git -C "$1" remote add origin "$2"
  git -C "$1" fetch -q --depth 1 origin "$3" && git -C "$1" checkout -q FETCH_HEAD
}
git clone -q --depth 1 --branch "$TAG" https://github.com/synaptics-torq/torq-compiler "$W/torq"
# The IREE fork commit the tag pins.
IREE_PIN=$(git -C "$W/torq" ls-tree HEAD third_party/iree | awk '{print $3}')
fetch "$W/iree" https://github.com/synaptics-torq/iree.git "$IREE_PIN"
for sm in flatcc benchmark vulkan_headers; do
  url=$(git -C "$W/iree" config -f .gitmodules "submodule.third_party/$sm.url")
  sha=$(git -C "$W/iree" ls-tree HEAD "third_party/$sm" | awk '{print $3}')
  rmdir "$W/iree/third_party/$sm" 2>/dev/null || true
  fetch "$W/iree/third_party/$sm" "$url" "$sha"
done
rmdir "$W/torq/third_party/iree" 2>/dev/null || true
ln -sfn "$W/iree" "$W/torq/third_party/iree"

# Host tools (flatcc compiler, embed-data) from the same fork.
cmake -S "$W/iree" -B "$W/build-host" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_ENABLE_CPUINFO=OFF
cmake --build "$W/build-host" --target iree-flatcc-cli iree-c-embed-data
# The aarch64 runtime and Torq driver.
cmake -S "$W/torq" -B "$W/build-a64" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$HERE/toolchain.aarch64.cmake" \
  -DIREE_HOST_BIN_DIR="$W/build-host/tools" \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DTORQ_ENABLE_AWS_FPGA=OFF -DTORQ_ENABLE_SOC_FPGA=OFF -DTORQ_ENABLE_ASTRA_MACHINA=ON \
  -DTORQ_ENABLE_SIMULATOR=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_VULKAN=ON \
  -DIREE_ENABLE_CPUINFO=OFF
cmake --build "$W/build-a64" --target iree_runtime_impl torq-run-module iree-run-module

# One fat archive of every runtime and driver archive (not --whole-archive: it holds duplicate objects).
B=$W/build-a64
(cd "$B" && find . -name '*.a' ! -path "*/benchmark/*" ! -name libiree_runtime_unified.a | sort > "$W/archives.txt")
(cd "$B" && "$KONAN_LLVM/bin/llvm-ar" qcL "$OUT/lib/libiree_torq_runtime.a" $(cat "$W/archives.txt"))
"$KONAN_LLVM/bin/llvm-ar" s "$OUT/lib/libiree_torq_runtime.a"

# The headers a shim compiles against: the IREE runtime API and its generated flatbuffer headers.
for root in "$W/iree/runtime/src" "$B/third_party/iree/runtime/src"; do
  (cd "$root" && find . -name '*.h' -print0 | cpio -0 -pdmu --quiet "$OUT/include")
done
for src in torq iree iree/third_party/flatcc iree/third_party/vulkan_headers; do
  name=$(basename "$src"); mkdir -p "$OUT/licenses/$name"
  find "$W/$src" -maxdepth 1 -iname 'LICENSE*' -exec cp {} "$OUT/licenses/$name/" \;
done
{
  echo "torq: $TAG $(git -C "$W/torq" rev-parse HEAD)"
  echo "iree: $(git -C "$W/iree" rev-parse HEAD)"
} > "$OUT/BUILD-INFO"
rm -rf "$W"
ls -la "$OUT/lib"; cat "$OUT/BUILD-INFO"
