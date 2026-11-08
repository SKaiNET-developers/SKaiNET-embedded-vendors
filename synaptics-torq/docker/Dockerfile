# skainet/iree-compiler-torq — the Synaptics Torq-fork IREE compiler with the skainet/iree-compiler entrypoint
# conventions (compile-cpu, compile-torq, compile, shell), for SL26xx boards (Cortex-A55 + Torq NPU).
#
# Why a separate image: the board runtime is the Torq fork of IREE (Torq 2.0.x = IREE 3.10.0.dev), and it rejects
# modules compiled by upstream IREE 3.11 ("required module features [Ch]"). Every graph for the board — CPU ones
# too — must come from this compiler, at the same Torq version as the board runtime.
#
# Build: synaptics-torq/docker/build.sh (pins in synaptics-torq/docker/torq.env).
ARG TORQ_BASE_IMAGE=python:3.12-slim
FROM ${TORQ_BASE_IMAGE}

ARG TORQ_VERSION
ARG TORQ_MANYLINUX=2_28
RUN test -n "${TORQ_VERSION}" || { echo "TORQ_VERSION build-arg required (use make)"; exit 1; }
ARG TORQ_WHEEL_URL=https://github.com/synaptics-torq/torq-compiler/releases/download/v${TORQ_VERSION}/torq_compiler-${TORQ_VERSION}-cp312-cp312-manylinux_${TORQ_MANYLINUX}_x86_64.whl

LABEL org.opencontainers.image.title="skainet/iree-compiler-torq" \
      org.opencontainers.image.description="Torq-fork iree-compile (llvm-cpu + Torq NPU) with skainet/iree-compiler subcommands" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET-embedded-vendors" \
      torq.version="${TORQ_VERSION}"
ENV TORQ_VERSION=${TORQ_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "${TORQ_WHEEL_URL}"

# iree-compile, torq-compile (the Torq driver, used by compile-torq), iree-lld (embedded ELF linker for llvm-cpu,
# and the Torq host-code linker) live in the wheel.
ENV MLIRLIBS=/usr/local/lib/python3.12/site-packages/iree/compiler/_mlir_libs
ENV PATH=${MLIRLIBS}:${PATH}
RUN iree-compile --version && torq-compile --version

WORKDIR /work
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
CMD ["help"]
