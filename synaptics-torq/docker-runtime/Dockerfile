# skainet/iree-runtime-torq: the Torq-fork IREE runtime for SL26xx boards (aarch64), source-built from the public
# synaptics-torq/torq-compiler tag and the IREE fork commit it pins. The counterpart of skainet/iree-compiler-torq:
# a module only loads on a runtime of the same Torq version.
#
# Build: synaptics-torq/docker-runtime/build.sh (the version is in synaptics-torq/docker/torq.env).
FROM --platform=linux/amd64 debian:trixie-slim

ARG TORQ_VERSION
RUN test -n "${TORQ_VERSION}" || { echo "TORQ_VERSION build-arg required (use build.sh)"; exit 1; }

LABEL org.opencontainers.image.title="skainet/iree-runtime-torq" \
      org.opencontainers.image.description="Torq-fork IREE runtime static archive for SL26xx (aarch64), built from public sources" \
      org.opencontainers.image.source="https://github.com/SKaiNET-developers/SKaiNET-embedded-vendors" \
      torq.version="${TORQ_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates clang cmake cpio curl git libncurses6 lld ninja-build python3 \
 && rm -rf /var/lib/apt/lists/*

# The Kotlin/Native bundled toolchain, so the archive matches what Kotlin/Native links for linuxArm64.
ENV KONAN_LLVM=/opt/konan/llvm-21-x86_64-linux-essentials-116
ENV KONAN_GCC=/opt/konan/aarch64-unknown-linux-gnu-gcc-8.3.0-glibc-2.25-kernel-4.9-2
RUN mkdir -p /opt/konan && cd /opt/konan \
 && curl -fsSL -o llvm.tgz https://download.jetbrains.com/kotlin/native/resources/llvm/21-x86_64-linux/llvm-21-x86_64-linux-essentials-116.tar.gz \
 && echo "0b1f290214dd7ec20e2506210967e0ebd735cdefc3a3b4485bbccb439fa78e5b  llvm.tgz" | sha256sum -c - \
 && curl -fsSL -o gcc.tgz https://download.jetbrains.com/kotlin/native/aarch64-unknown-linux-gnu-gcc-8.3.0-glibc-2.25-kernel-4.9-2.tar.gz \
 && echo "a026ede10b5dc2f2dcba91115338040ba62558ca14cf1a18cfba50768b245385  gcc.tgz" | sha256sum -c - \
 && tar xzf llvm.tgz && tar xzf gcc.tgz && rm llvm.tgz gcc.tgz

ENV TORQ_RUNTIME_DIST=/opt/torq-runtime/dist
COPY toolchain.aarch64.cmake build-runtime.sh entrypoint.sh /opt/torq-runtime/
COPY compat-include /opt/torq-runtime/compat-include
RUN chmod +x /opt/torq-runtime/build-runtime.sh /opt/torq-runtime/entrypoint.sh \
 && /opt/torq-runtime/build-runtime.sh "v${TORQ_VERSION}" /tmp/torq-work "${TORQ_RUNTIME_DIST}" /opt/torq-runtime

ENTRYPOINT ["/opt/torq-runtime/entrypoint.sh"]
CMD ["help"]
