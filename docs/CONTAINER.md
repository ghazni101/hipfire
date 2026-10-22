# Containerized hipfire

Repo-root multi-stage `Containerfile`. **No image carries ROCm**: the host's
`/opt/rocm` is mounted into the builder at image-build time (BuildKit named
context / `podman build --volume`) and into every container read-only at run
time. This works because hipfire `dlopen`s `libamdhip64.so`/
`libhsa-runtime64.so` — nothing links HIP at build time, and containers share
the host's KMD, so user-space/KMD versions always match.

| Target | Purpose | Image contents |
|---|---|---|
| `runtime` | Deliverable inference image | daemon + `hipfire` CLI + five-arch indexed kernel packs + python3 (mounted hipcc JIT works) |
| `runtime-slim` | Deliverable for pinned registered models | same minus python3; unregistered routes fail fast |
| `quantizer` | Model-build image | `runtime` + `hipfire-quantize` |
| `gate-runner` | Local GPU harness / dev-build validation | full builder tree + source + gate scripts |

| Field | Value |
|---|---|
| Base image | `docker.io/library/ubuntu:26.10` (must match the host Ubuntu the mounted `/opt/rocm` was built for) |
| ROCm | host `/opt/rocm`, mounted `ro` — never an image layer |
| Control plane | native `hipfire-cli` Rust binary |
| Published port | `11435` (`EXPOSE`) |
| JIT kernel cache env | `HIPFIRE_KERNEL_CACHE=/var/cache/hipfire` |
| Daemon path in runtime | `HIPFIRE_DAEMON_BIN=/opt/hipfire/bin/daemon` |

## Design constraints

- **ROCm is dlopen'd at runtime** — `cargo build` needs no GPU and no ROCm.
  Only `scripts/compile-kernels.sh` needs `hipcc`, and it gets the host's.
- **Base tag MUST match the host OS** that produced `/opt/rocm`. The mounted
  amdclang/hipcc run inside the container against its glibc; a base older
  than the host breaks on glibc symbols. Host Ubuntu 26.10 → `ubuntu:26.10`.
- **Exact-source registry kernels are packaged at build time** for
  `gfx1201/1100/1151/906/942` by the host's hipcc. One image serves all
  arches — never build per-arch images.
- **Models are never baked in** — the named volume `hipfire-models` is
  shared by the pull, quantize, and serve containers so weights exist on
  disk exactly once.
- **No `HSA_OVERRIDE_GFX_VERSION`** is set; arch comes from HIP
  `gcnArchName` at runtime.
- **Host-ROCm caveat:** images are host-bound — they run only where an
  equivalent `/opt/rocm` exists. A host ROCm upgrade propagates to all
  containers instantly; kernel-cache entries are content-keyed so a new
  ROCm just writes new entries.

`.dockerignore` keeps the build context small (excludes `target/`, models,
worktrees, `kernels/compiled/`).

## Build

`scripts/container-build.sh` supplies the host-ROCm mount for the build:

```bash
scripts/container-build.sh runtime      hipfire:dev
scripts/container-build.sh runtime-slim hipfire:dev-slim
scripts/container-build.sh quantizer    hipfire-quantize:dev
scripts/container-build.sh gate-runner  hipfire-gate:dev
```

Environment: `HIPFIRE_CONTAINER=docker|podman` (default `docker`),
`HIPFIRE_ROCM=<path>` (default `readlink -f /opt/rocm`). Under docker the
wrapper passes `--build-context rocm=<path>`; the builder stage mounts it
inside the kernel-pack `RUN` step. Under podman it passes
`--volume <path>:/opt/rocm:ro` to `podman build`.

Equivalent manual docker command:

```bash
docker build --build-context rocm=/opt/rocm \
  -f Containerfile --target runtime -t hipfire:dev .
```

## Run (GPU required)

GPU device nodes plus the host ROCm mount are required. Docker does not
implement podman's `--group-add keep-groups`; if the daemon user can't open
`/dev/kfd`, pass host GIDs with `--group-add $(stat -c %g /dev/kfd)` (or run
rootless podman, which honors `keep-groups`).

**Unauthenticated HTTP.** Serve has no auth and no TLS ([SERVE.md](SERVE.md)).
Publish only on loopback (`-p 127.0.0.1:11435:11435`) unless a trusted
network or authenticated TLS reverse proxy protects it.

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  --security-opt seccomp=unconfined \
  -v /opt/rocm:/opt/rocm:ro \
  -v hipfire-models:/root/.hipfire/models \
  -v hipfire-kcache:/var/cache/hipfire \
  -p 127.0.0.1:11435:11435 \
  hipfire:dev serve qwen3.5:4b 0.0.0.0:11435
```

Bind `0.0.0.0:11435` inside the container so published-port DNAT reaches the
listener; keep the host publication on loopback.

Do **not** pass `-d`/`--detach` as the container command — that forks a
background child and lets PID 1 exit. Detach with the container runtime
(`docker run -d …`) for a long-lived container.

`docker/compose.yaml` wires the same layout (`hipfire` service + `tools`
profile with one-shot `pull` and `quantize` services):

```bash
docker compose -f docker/compose.yaml up hipfire          # serve
docker compose -f docker/compose.yaml --profile tools \
    run --rm pull qwen3.8:27b-mq4-xts                     # fetch model + draft sidecar
```

Volumes:

| Volume / path | Role |
|---|---|
| `hipfire-models` → `/root/.hipfire/models` | Pulled / quantized model files |
| `hipfire-kcache` → `/var/cache/hipfire` | Persistent JIT kernel cache |
| `/opt/rocm` → `/opt/rocm` (ro) | Host ROCm — mount the real dir, not a broken symlink |


### Profiling (`rocprofv3`)

`hipfire:dev` (the `runtime` target) bakes the ROCm profiler toolchain into
the image at `/opt/rocprofiler` — `rocprofv3`, `rocprofv3-avail` and
`rocprof-attach` are on `PATH`. This is a deliberate exception to the
"no ROCm in the image" rule: packaged host installs keep `/opt/rocm/{bin,lib}`
as `/etc/alternatives` symlinks that dangle inside containers, so the bind
mount alone cannot provide working tools. The profiler's own closure (SDK,
comgr/LLVM, rocm_sysdeps, rocpd schemas) is self-contained under
`/opt/rocprofiler`; the app under test still uses the mounted ROCm for HIP.

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined --entrypoint bash \
    -v /opt/rocm:/opt/rocm:ro -v "$PWD/traces:/out" hipfire:dev -c \
    'rocprofv3 --hip-trace --kernel-trace -d /out -o run -- hipfire bench <model> ...'
```

Output is `rocpd` (SQLite) by default; convert on the host with
`rocpd convert` or pass `--output-format csv`/`json`/`pftrace`.

## Local containerized GPU runs (`scripts/container-gate.sh`)

Builds `gate-runner` and runs a command inside it with GPU passthrough, the
host ROCm mount, and the host models directory bind-mounted. Defaults to
docker (`HIPFIRE_CONTAINER=podman` for podman).

The `gate-runner` image has no default gate — select a route from
[VALIDATION.md](VALIDATION.md) and pass it explicitly:

```bash
scripts/container-gate.sh python3 scripts/serve_harness.py --mode chain \
    --model /root/.hipfire/models/<file> --tag qwen3.5:9b
scripts/container-gate.sh python3 scripts/redline_daemon_harness.py
```

**Scope limits (fail closed):** a successful image build proves the
multi-stage compile path only — not GPU runtime correctness, serve
semantics, or perf. Retired `coherence-gate-*.sh` batteries are not
acceptance evidence. Current user-facing serve smoke is
`scripts/serve_harness.py` per [VALIDATION.md](VALIDATION.md).

## Publishing

Not applicable under the host-ROCm design — images only run where an
equivalent `/opt/rocm` exists. Distributing a self-contained image would
mean returning to a `FROM rocm/…` base stage; that is the deliberate
tradeoff this layout avoids.

## Related paths

| Path | Role |
|---|---|
| `Containerfile` | Multi-stage definition (base → builder → 4 targets) |
| `scripts/container-build.sh` | Build wrapper — supplies the host-ROCm build mount |
| `scripts/container-gate.sh` | Local GPU container runner |
| `docker/compose.yaml` | Serve/pull/quantize service layout |
| [SERVE.md](SERVE.md) | HTTP API |
| [NIXOS.md](NIXOS.md) | NixOS alternative install |
