# syntax=docker/dockerfile:1
# hipfire container — host-ROCm multi-stage build (docker + podman).
#
# Architecture contract: ROCm is NOT installed in any image. The host's
# /opt/rocm is bind-mounted into containers at run time, and into the builder
# at image-build time (BuildKit named context / podman build --volume) so
# scripts/compile-kernels.sh can drive the host's own hipcc. This works
# because hipfire dlopen's libamdhip64/libhsa-runtime64 — nothing links HIP
# at build time.
#
#   base ──► builder ──► gate-runner   (PR / dev-build validation)
#    │           ├─────► runtime        (deliverable; host hipcc JIT works)
#    │           │          └─────► quantizer  (model-build image)
#    │           └─────► runtime-slim   (precompiled routes only, no JIT deps)
#    └────────────────► (nothing else uses base directly)
#
# Requirements:
#   * The image ubuntu:26.10 base MUST match (or exceed) the glibc the host's
#     /opt/rocm was built for — amdclang/hipcc run from the mount.
#   * Build needs the rocm build context: scripts/container-build.sh wraps it.
#     Manual:  docker build --build-context rocm=/opt/rocm \
#                 -f Containerfile --target runtime -t hipfire:dev .
#   * Models are runtime-only downloads — never baked in; mount as a volume.
#
# Build:
#   scripts/container-build.sh runtime hipfire:dev
#   scripts/container-build.sh runtime-slim hipfire:dev-slim
#   scripts/container-build.sh quantizer hipfire-quantize:dev
#   scripts/container-build.sh gate-runner hipfire-gate:dev
#
# Run (deliverable, needs GPU passthrough + host ROCm mount):
#   docker run --rm -it \
#     --device /dev/kfd --device /dev/dri \
#     --security-opt seccomp=unconfined \
#     -v /opt/rocm:/opt/rocm:ro \
#     -v hipfire-models:/root/.hipfire/models \
#     -v hipfire-kcache:/var/cache/hipfire \
#     -p 127.0.0.1:11435:11435 \
#     hipfire:dev serve qwen3.5:4b 0.0.0.0:11435
#   (rootless podman adds --group-add keep-groups)

# ─────────────────────────────────────────────────────────────────────────────
# base — Ubuntu matching the host + user-space deps the mounted ROCm libs and
# hipfire need. No ROCm, no toolchain, no python — shared by every runtime
# image so they store these layers once.
# ─────────────────────────────────────────────────────────────────────────────
FROM docker.io/library/ubuntu:26.10 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libdrm-amdgpu1 \
        libnuma1 \
        libstdc++6 \
        zlib1g \
        libzstd1 \
        libelf1 \
    && rm -rf /var/lib/apt/lists/*

# /opt/rocm is empty in the image; it arrives as a read-only bind mount at run
# time (and inside the builder via the rocm build context). ENV makes the
# dlopen'd runtime and the ROCm resolver find it without ld.so.conf.
ENV HIP_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    PATH=/opt/rocm/bin:$PATH \
    HIPFIRE_KERNEL_CACHE=/var/cache/hipfire
RUN mkdir -p /var/cache/hipfire && chmod 1777 /var/cache/hipfire

# ─────────────────────────────────────────────────────────────────────────────
# builder — Rust toolchain + cargo builds. Kernel packages compile against the
# HOST's hipcc, mounted at /opt/rocm via the `rocm` build context.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        unzip \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /hipfire
COPY . .

# Daemon is a standalone binary crate (hipfire-daemon) that bundles every
# architecture unconditionally — no --features or --example needed.
RUN cargo build --release --locked -p hipfire-daemon
RUN cargo build --release --locked -p hipfire-cli
RUN cargo build --release --locked -p hipfire-quantize

# Exact Rust-registry kernel packages for all five admitted arches, compiled
# by the host's own hipcc. Without the `rocm` build context this RUN fails —
# use scripts/container-build.sh.
RUN --mount=type=bind,from=rocm,target=/opt/rocm,ro \
    bash scripts/compile-kernels.sh

# ─────────────────────────────────────────────────────────────────────────────
# runtime — TARGET A, deliverable. No Rust, no source tree. python3 stays so
# the mounted hipcc's helper scripts and hipfire's harnesses run.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-yaml \
        libstdc++-15-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /hipfire/target/release/daemon /opt/hipfire/bin/daemon
COPY --from=builder /hipfire/kernels/compiled /opt/hipfire/bin/kernels/compiled
COPY --from=builder /hipfire/target/release/hipfire /usr/local/bin/hipfire

# rocprofv3 baked into the image under its own prefix. The runtime /opt/rocm
# bind mount does NOT provide tools on the usual packaged install: real files
# live under /opt/rocm/core-<ver>/ while /opt/rocm/{bin,lib} are symlinks into
# /etc/alternatives, which dangle inside the container (this is also why COPY
# --from=rocm can't reach them — the context root has the same dangling
# links). rocprofv3's Python launcher resolves ROCM_DIR as
# dirname(dirname(realpath($0))), so a $P/bin script finds its libs at
# $P/lib; the SDK .so's carry $ORIGIN RPATHs into ../lib, ../lib/llvm/lib,
# ../lib/rocm_sysdeps/lib and the SDK finds counter data under ../share — the
# copy preserves that directory shape so the profiler is self-contained and
# does not depend on the mount layout.
# Transitive closure (~330 MB): sdk tool/list-avail/kokkosp/gotcha libs, sdk +
# attach/rocattach/roctx/rocpd/register/trace-decoder, aqlprofile + the HSA
# runtime (PMC counters), amd_comgr + libLLVM/libclang-cpp (code objects),
# and the rocm_sysdeps subtree. cp -d keeps the SONAME symlink chains.
# SRC resolves whichever root layout the rocm context presents: classic
# <root>/lib directly, or the packaged <root>/core-<ver>/ split tree.
RUN --mount=type=bind,from=rocm,target=/opt/rocm,ro \
    set -eu; \
    SRC="/opt/rocm"; \
    if [ ! -f "$SRC/bin/rocprofv3" ]; then SRC="$(dirname "$(dirname "$(ls -d /opt/rocm/core-*/bin/rocprofv3 | sort -V | tail -n1)")")"; fi; \
    [ -f "$SRC/bin/rocprofv3" ] || { echo "rocprofv3 not found under $SRC" >&2; exit 1; }; \
    mkdir -p /opt/rocprofiler/bin /opt/rocprofiler/lib /opt/rocprofiler/share /opt/rocprofiler/libexec; \
    cp -d "$SRC"/bin/rocprofv3 "$SRC"/bin/rocprofv3-avail "$SRC"/bin/rocprof-attach /opt/rocprofiler/bin/; \
    cp -d "$SRC"/lib/librocprofiler*.so* "$SRC"/lib/librocprof-trace-decoder.so* /opt/rocprofiler/lib/; \
    cp -d "$SRC"/lib/libhsa-amd-aqlprofile64.so* "$SRC"/lib/libhsa-runtime64.so* "$SRC"/lib/libamd_comgr.so* /opt/rocprofiler/lib/; \
    cp -ad "$SRC"/lib/rocprofiler-sdk /opt/rocprofiler/lib/; \
    cp -ad "$SRC"/lib/rocm_sysdeps /opt/rocprofiler/lib/; \
    mkdir -p /opt/rocprofiler/lib/llvm; cp -ad "$SRC"/lib/llvm/lib /opt/rocprofiler/lib/llvm/; \
    rm -f /opt/rocprofiler/lib/llvm/lib/*.a; \
    cp -ad "$SRC"/share/rocprofiler-sdk "$SRC"/share/rocprofiler-sdk-rocpd /opt/rocprofiler/share/; \
    mkdir -p /opt/rocprofiler/lib/python3; cp -ad "$SRC"/lib/python3/site-packages /opt/rocprofiler/lib/python3/; \
    cp -ad "$SRC"/libexec/rocprofiler-sdk /opt/rocprofiler/libexec/

# Prepend the baked profiler prefix so it wins over the (possibly dangling)
# /opt/rocm/bin that base put first on PATH.
ENV PATH=/opt/rocprofiler/bin:$PATH

ENV HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon \
    HIPFIRE_HOME=/root/.hipfire

EXPOSE 11435

ENTRYPOINT ["hipfire"]
CMD ["--help"]

# ─────────────────────────────────────────────────────────────────────────────
# runtime-slim — TARGET B, deliverable for pinned registered models. No
# python3: every route must resolve to an indexed kernel pack; an unregistered
# route fails fast instead of JITing. ~150 MB lighter than runtime.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS runtime-slim

COPY --from=builder /hipfire/target/release/daemon /opt/hipfire/bin/daemon
COPY --from=builder /hipfire/kernels/compiled /opt/hipfire/bin/kernels/compiled
COPY --from=builder /hipfire/target/release/hipfire /usr/local/bin/hipfire

ENV HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon \
    HIPFIRE_HOME=/root/.hipfire

EXPOSE 11435

ENTRYPOINT ["hipfire"]
CMD ["--help"]

# ─────────────────────────────────────────────────────────────────────────────
# quantizer — TARGET C, model-build image. hipfire-quantize on the runtime
# base; writes .hfq/.mq* artifacts into the mounted models volume. Optional
# librocblas/librocsolver dlopens resolve from the host's /opt/rocm mount.
# ─────────────────────────────────────────────────────────────────────────────
FROM runtime AS quantizer

COPY --from=builder /hipfire/target/release/hipfire-quantize /usr/local/bin/hipfire-quantize

ENTRYPOINT ["hipfire-quantize"]

# ─────────────────────────────────────────────────────────────────────────────
# gate-runner — TARGET D, PR/dev-build validation. Full toolchain + source +
# in-repo GPU gate scripts. Runs with GPU passthrough + host ROCm mount
# against mounted models.
# ─────────────────────────────────────────────────────────────────────────────
FROM builder AS gate-runner

# pytest/numpy for the Python-backed detector arms in the gate scripts.
RUN pip install --no-cache-dir --break-system-packages pytest numpy

COPY --from=builder /hipfire/target/release/hipfire /usr/local/bin/hipfire

ENV HIPFIRE_DAEMON_BIN=/hipfire/target/release/daemon \
    HIPFIRE_HOME=/root/.hipfire \
    MODELS_DIR=/root/.hipfire/models

WORKDIR /hipfire
# No default gate: there is no universal correctness battery. The old default
# (scripts/coherence-gate.sh) is retired AND absent from the checkout, so it
# execed as "no such file" (127). Fail closed with guidance instead; pick a
# route from docs/VALIDATION.md and pass it explicitly, e.g.
#   docker run ... hipfire-gate:dev python3 scripts/serve_harness.py battery --model <model>
ENTRYPOINT []
CMD ["/bin/bash", "-c", "echo '[hipfire-gate] no gate command given.' >&2; echo 'There is no universal gate. Select a validation route from docs/VALIDATION.md and pass it explicitly, e.g.:' >&2; echo '  python3 scripts/serve_harness.py battery --model <model>' >&2; echo '  python3 scripts/redline_daemon_harness.py   # kernel/dispatch/graph/Redline changes' >&2; exit 2"]
