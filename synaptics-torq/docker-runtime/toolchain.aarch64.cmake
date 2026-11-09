# Cross toolchain for SL26xx boards (aarch64, glibc >= 2.25): the Kotlin/Native clang-21/lld + gcc-8.3/glibc-2.25
# sysroot, the sysroot Kotlin/Native links linuxArm64 executables against, so the archive is cinterop-ready.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(KONAN_LLVM "$ENV{KONAN_LLVM}")
set(KONAN_GCC  "$ENV{KONAN_GCC}")
set(CMAKE_SYSROOT "${KONAN_GCC}/aarch64-unknown-linux-gnu/sysroot")

set(CMAKE_C_COMPILER   "${KONAN_LLVM}/bin/clang")
set(CMAKE_CXX_COMPILER "${KONAN_LLVM}/bin/clang++")
set(CMAKE_AR     "${KONAN_LLVM}/bin/llvm-ar" CACHE FILEPATH "")
set(CMAKE_RANLIB "${KONAN_GCC}/bin/aarch64-unknown-linux-gnu-ranlib" CACHE FILEPATH "")
set(CMAKE_C_COMPILER_TARGET   aarch64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-unknown-linux-gnu)

# The kernel-4.9 sysroot headers lack the dma-buf/dma-heap uapi: add copies.
set(_common "--gcc-toolchain=${KONAN_GCC} -mcpu=cortex-a55 -isystem ${CMAKE_CURRENT_LIST_DIR}/compat-include")
set(CMAKE_C_FLAGS_INIT   "${_common}")
set(CMAKE_CXX_FLAGS_INIT "${_common}")
set(CMAKE_EXE_LINKER_FLAGS_INIT    "--gcc-toolchain=${KONAN_GCC} -fuse-ld=lld -static-libstdc++ -static-libgcc")
# gcc-8 libstdc++ keeps std::filesystem (used by the Torq driver) in a separate archive.
set(CMAKE_CXX_STANDARD_LIBRARIES "-lstdc++fs" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "--gcc-toolchain=${KONAN_GCC} -fuse-ld=lld -static-libstdc++ -static-libgcc")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
