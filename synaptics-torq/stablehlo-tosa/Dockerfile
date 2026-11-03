# skainet/stablehlo-tosa — StableHLO's TOSA legalization (`stablehlo-opt --stablehlo-prepare-for-tosa
# --stablehlo-legalize-to-tosa`) built from the Torq compiler's own sources, so the TOSA it writes parses in the
# Torq-fork iree-compile of skainet/iree-compiler-torq at the same TORQ_VERSION. The Torq wheel ships no stablehlo-opt.
#
# Why from the Torq tree: standalone StableHLO pins its own LLVM, whose TOSA dialect may differ from the
# synaptics-torq/llvm-project fork the Torq compiler uses. Building stock IREE (no Torq plugin) from
# torq-compiler v${TORQ_VERSION} with only the `stablehlo-opt` target gives exactly the compiler's LLVM.
#
# Build: synaptics-torq/stablehlo-tosa/build.sh (~35 min on 12 cores, ~10 GB of build tree inside the build stage).
ARG TORQ_BUILDER_IMAGE=ghcr.io/synaptics-torq/torq-compiler/builder@sha256:06f66e143b5c109ca0b0477220120a1426f6cde309079f94a81648a329476dd4
FROM ${TORQ_BUILDER_IMAGE} AS build
ARG TORQ_VERSION
RUN test -n "${TORQ_VERSION}" || { echo "TORQ_VERSION build-arg required"; exit 1; }
RUN git clone --depth 1 --branch "v${TORQ_VERSION}" https://github.com/synaptics-torq/torq-compiler.git /src \
 && cd /src && scripts/checkout_submodules.sh
COPY build-stablehlo-opt.sh /usr/local/bin/build-stablehlo-opt.sh
RUN build-stablehlo-opt.sh /src /build && strip /build/llvm-project/bin/stablehlo-opt

FROM debian:trixie-slim
ARG TORQ_VERSION
LABEL org.opencontainers.image.title="skainet/stablehlo-tosa" \
      org.opencontainers.image.description="stablehlo-opt (StableHLO → TOSA legalization) from the Torq ${TORQ_VERSION} compiler's LLVM" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET-embedded-vendors" \
      torq.version="${TORQ_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/llvm-project/bin/stablehlo-opt /usr/local/bin/stablehlo-opt
WORKDIR /work
ENTRYPOINT ["stablehlo-opt"]
CMD ["--help"]
