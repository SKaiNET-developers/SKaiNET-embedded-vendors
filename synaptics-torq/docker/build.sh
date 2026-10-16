#!/usr/bin/env bash
# Builds skainet/iree-compiler-torq:<TORQ_VERSION> from torq.env (override any pin through the environment).
#   synaptics-torq/docker/build.sh
#   TORQ_VERSION=2.2.1 TORQ_MANYLINUX=2_39 TORQ_BASE_IMAGE=python:3.12-slim-trixie synaptics-torq/docker/build.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
pins=$(env -i bash -c "set -a; . '$D/torq.env'; env" | grep '^TORQ_')
for p in $pins; do k=${p%%=*}; [ -n "${!k:-}" ] || export "$p"; done
REGISTRY=${REGISTRY:-skainet}
docker build \
  --build-arg TORQ_VERSION="$TORQ_VERSION" \
  --build-arg TORQ_MANYLINUX="$TORQ_MANYLINUX" \
  --build-arg TORQ_BASE_IMAGE="$TORQ_BASE_IMAGE" \
  -t "$REGISTRY/iree-compiler-torq:$TORQ_VERSION" "$D"
docker run --rm "$REGISTRY/iree-compiler-torq:$TORQ_VERSION" version
