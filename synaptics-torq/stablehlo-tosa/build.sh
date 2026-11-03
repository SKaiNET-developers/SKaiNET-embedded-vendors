#!/usr/bin/env bash
# Builds skainet/stablehlo-tosa:<TORQ_VERSION> (pins: ../docker/torq.env, override through the environment).
#   synaptics-torq/stablehlo-tosa/build.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
pins=$(env -i bash -c "set -a; . '$D/../docker/torq.env'; env" | grep '^TORQ_')
for p in $pins; do k=${p%%=*}; [ -n "${!k:-}" ] || export "$p"; done
REGISTRY=${REGISTRY:-skainet}
docker build --build-arg TORQ_VERSION="$TORQ_VERSION" -t "$REGISTRY/stablehlo-tosa:$TORQ_VERSION" "$D"
docker run --rm "$REGISTRY/stablehlo-tosa:$TORQ_VERSION" --help | grep -E "stablehlo-(prepare-for|legalize-to)-tosa"
