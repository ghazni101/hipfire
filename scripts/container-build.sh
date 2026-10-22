#!/usr/bin/env bash
# Build a hipfire container image against the HOST's ROCm.
#
# The Containerfile carries no ROCm: the builder stage mounts the host's
# /opt/rocm so scripts/compile-kernels.sh runs the host's own hipcc, and
# runtime containers mount it read-only at run time. This wrapper supplies
# that mount during `docker build` (BuildKit named context) or
# `podman build` (--volume).
#
# Usage:
#   scripts/container-build.sh <target> [image-tag]
#
# Targets: runtime | runtime-slim | quantizer | gate-runner
#
# Environment:
#   HIPFIRE_CONTAINER=docker|podman   container runtime (default: docker)
#   HIPFIRE_ROCM=<path>             host ROCm root   (default: readlink /opt/rocm)
#
# Examples:
#   scripts/container-build.sh runtime hipfire:dev
#   scripts/container-build.sh gate-runner hipfire-gate:dev
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${HIPFIRE_CONTAINER:-docker}"

TARGET="${1:-}"
TAG="${2:-}"
case "$TARGET" in
    runtime)        TAG="${TAG:-hipfire:dev}" ;;
    runtime-slim)   TAG="${TAG:-hipfire:dev-slim}" ;;
    quantizer)      TAG="${TAG:-hipfire-quantize:dev}" ;;
    gate-runner)    TAG="${TAG:-hipfire-gate:dev}" ;;
    *)
        echo "usage: $0 <runtime|runtime-slim|quantizer|gate-runner> [image-tag]" >&2
        exit 2
        ;;
esac

ROCM_ROOT="$(readlink -f "${HIPFIRE_ROCM:-/opt/rocm}")"
if [ ! -x "$ROCM_ROOT/bin/hipcc" ]; then
    echo "[container-build] no hipcc at $ROCM_ROOT/bin/hipcc" >&2
    echo "  set HIPFIRE_ROCM to the host ROCm root." >&2
    exit 2
fi
echo "[container-build] $RUNTIME build target=$TARGET tag=$TAG rocm=$ROCM_ROOT"

if [ "$RUNTIME" = "docker" ]; then
    # BuildKit named context: mounts the host ROCm into the builder stage's
    # RUN --mount=type=bind,from=rocm without copying it into the context.
    exec docker build --build-context "rocm=$ROCM_ROOT" \
        -f "$ROOT/Containerfile" --target "$TARGET" -t "$TAG" "$ROOT"
else
    # Podman: build-time volume mount gives the builder the same view.
    exec podman build --volume "$ROCM_ROOT:/opt/rocm:ro" \
        -f "$ROOT/Containerfile" --target "$TARGET" -t "$TAG" "$ROOT"
fi
