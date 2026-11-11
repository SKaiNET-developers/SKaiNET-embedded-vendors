#!/usr/bin/env bash
# skainet/iree-runtime-torq: hands out the runtime built into the image.
set -euo pipefail
cmd="${1:-help}"; shift || true
case "$cmd" in
  dist)
    # dist <out dir>: copies lib/, include/, licenses/ and BUILD-INFO, owned like <out dir>
    OUT="${1:?Usage: dist <out dir>}"
    mkdir -p "$OUT"
    cp -r "$TORQ_RUNTIME_DIST"/. "$OUT"/
    chown -R "$(stat -c %u:%g "$OUT")" "$OUT"
    ;;
  version) cat "$TORQ_RUNTIME_DIST/BUILD-INFO" ;;
  shell) exec bash "$@" ;;
  *) echo "commands: dist <out dir> | version | shell" ;;
esac
