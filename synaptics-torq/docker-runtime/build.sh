#!/usr/bin/env bash
# Builds skainet/iree-runtime-torq:<TORQ_VERSION>, the version from ../docker/torq.env (TORQ_VERSION overrides it).
#   synaptics-torq/docker-runtime/build.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
pin=$(env -i bash -c "set -a; . '$D/../docker/torq.env'; echo \$TORQ_VERSION")
TORQ_VERSION=${TORQ_VERSION:-$pin}
REGISTRY=${REGISTRY:-skainet}
docker build --build-arg TORQ_VERSION="$TORQ_VERSION" -t "$REGISTRY/iree-runtime-torq:$TORQ_VERSION" "$D"
docker run --rm "$REGISTRY/iree-runtime-torq:$TORQ_VERSION" version
