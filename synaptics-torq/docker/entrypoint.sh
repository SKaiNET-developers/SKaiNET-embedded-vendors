#!/usr/bin/env bash
# Subcommand entrypoint for skainet/iree-compiler-torq. Same interface as skainet/iree-compiler where it overlaps,
# so blueprint recipes can switch images per target.
set -euo pipefail

cmd="${1:-help}"
[ $# -gt 0 ] && shift

# --out <f> [--target <t>] [--cpu <c>] [--input <type>] parsing shared by the compile subcommands.
parse() {
  OUT=""; TARGET="${DEFAULT_TARGET:-}"; CPU=""; INPUT="stablehlo"; EXTRA=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --out) OUT="$2"; shift 2 ;;
      --target) TARGET="$2"; shift 2 ;;
      --cpu) CPU="$2"; shift 2 ;;
      --input) INPUT="$2"; shift 2 ;;
      --) shift; EXTRA+=("$@"); break ;;
      *) EXTRA+=("$1"); shift ;;
    esac
  done
}

# A Torq vmfb links aarch64 host code; IREE calls `ld` for it, so `ld` must be the bundled iree-lld.
ld_shim() {
  local d; d="$(mktemp -d)"
  printf '#!/bin/sh\nexec "%s" -flavor gnu "$@"\n' "$MLIRLIBS/iree-lld" > "$d/ld"
  chmod +x "$d/ld"; echo "$d"
}

case "$cmd" in

  compile-cpu)
    # compile-cpu <in.mlir> [--out f.vmfb] [--target linux-arm64|host] [--cpu <name>] [-- extra iree-compile args]
    #   linux-arm64 : SL26xx application cores (Cortex-A55, v8.2a + dotprod), embedded ELF — no system linker,
    #                 loads on the board's Torq-fork runtime with the local-task / local-sync drivers.
    #   host        : x86_64 host, embedded ELF.
    IN="${1:?Usage: compile-cpu <in.mlir> [--out f.vmfb] [--target linux-arm64|host] [--cpu <name>]}"; shift
    DEFAULT_TARGET=linux-arm64 parse "$@"
    [ -z "$OUT" ] && OUT="${IN%.mlir}_cpu_${TARGET}.vmfb"
    case "$TARGET" in
      linux-arm64) FLAGS=(
          --iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu
          --iree-llvmcpu-target-cpu="${CPU:-cortex-a55}"
          --iree-llvmcpu-target-cpu-features=+v8.2a,+neon,+dotprod ) ;;
      host) FLAGS=() ;;
      *) echo "unknown --target '$TARGET' (linux-arm64|host)"; exit 2 ;;
    esac
    echo "==> $IN → $OUT (torq-fork llvm-cpu $TARGET)"
    iree-compile "$IN" --iree-input-type=stablehlo \
      --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
      "${FLAGS[@]}" "${EXTRA[@]}" -o "$OUT"
    echo "==> Done: $OUT ($(du -h "$OUT" | cut -f1))"
    ;;

  compile-torq)
    # compile-torq <in.mlir> [--out f.vmfb] [--target SL2610] [--input stablehlo|tosa-torq|onnx-torq|…]
    #              [-- extra iree-compile args, e.g. --torq-disable-slices]
    #   --input tosa-torq: Torq's TOSA pipeline, which makes each function ONE dispatch (as for the vendor's models)
    IN="${1:?Usage: compile-torq <in.mlir> [--out f.vmfb] [--target SL2610]}"; shift
    DEFAULT_TARGET=SL2610 parse "$@"
    [ -z "$OUT" ] && OUT="${IN%.mlir}_torq.vmfb"
    SHIM="$(ld_shim)"
    echo "==> $IN → $OUT (torq $TARGET, input $INPUT)"
    PATH="$SHIM:$PATH" iree-compile "$IN" --iree-input-type="$INPUT" \
      --iree-hal-target-device=torq --torq-hw="$TARGET" --torq-fallback-f32-to-host \
      "${EXTRA[@]}" -o "$OUT"
    rm -rf "$SHIM"
    echo "==> Done: $OUT ($(du -h "$OUT" | cut -f1))"
    ;;

  compile-vulkan)
    echo "compile-vulkan: not provided by the Torq image (use skainet/iree-compiler for GPU targets)"; exit 2 ;;

  compile)  [ "${1:-}" = "--" ] && shift; exec iree-compile "$@" ;;
  version)  echo "torq ${TORQ_VERSION}"; exec iree-compile --version ;;
  shell)    exec /bin/bash "$@" ;;

  *)
    echo "skainet/iree-compiler-torq ${TORQ_VERSION:-} — Synaptics Torq-fork IREE compiler"
    echo ""
    echo "Commands:"
    echo "  compile-cpu <in.mlir> [--out f.vmfb] [--target linux-arm64|host]   MLIR → CPU VMFB (A55, embedded ELF)"
    echo "  compile-torq <in.mlir> [--out f.vmfb] [--target SL2610] [--input tosa-torq]   MLIR → Torq NPU VMFB"
    echo "  compile -- <raw iree-compile args>                                escape hatch"
    echo "  version | shell"
    echo ""
    echo "Parameter archives (.irpa): use skainet/iree-compiler convert-parameters; the format is shared."
    [ "$cmd" = "help" ] || exit 1
    ;;
esac
