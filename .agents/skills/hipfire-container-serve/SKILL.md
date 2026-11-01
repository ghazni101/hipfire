---
name: hipfire-container-serve
description: Use when serving a hipfire model with Docker or Podman compose, Containerfile images, container-gate, the Strix Halo NPU overlay, or the VMM multi-slot recipe. Use for HIPFIRE_MODEL/port/bind/models volumes, rootless podman, and compose health/smoke — not native `hipfire serve` on the host. Default host port 9001, HIPFIRE_UI=1, --idle-timeout 0; one model at a time.
---

# hipfire-container-serve

Operator playbook for **containerized** `hipfire serve` on AMD GPUs.
Host-native serve (`hipfire serve` on the machine) is
[`docs/SERVE.md`](/home/ghazni/github/hipfire/docs/SERVE.md). This skill is the compose /
Containerfile / gate path.

Canonical operator page: [`docs/CONTAINER.md`](/home/ghazni/github/hipfire/docs/CONTAINER.md).

This copy lives in `~/.agents/skills/` so omp loads it in **any** cwd. Canonical files stay in `/home/ghazni/github/hipfire`.
Re-read it when a flag, image target, or env name looks new.

## Agent defaults (always)

When *you* bring a container up, apply these unless the user names a different value:

| Setting | Value | How |
|---|---|---|
| Host port | **9001** | `HIPFIRE_PORT=9001` |
| Chat UI | **on** | `HIPFIRE_UI=1` (→ `/ui`; same as `hipfire serve --ui`) |
| Idle unload | **never** | `HIPFIRE_SERVE_ARGS=--idle-timeout 0` |

One model / one serve container at a time. Reusing **9001** when a previous container still holds it is OK — stop/replace the old one. Do not hunt a free port for a second concurrent serve.

Compose.yml still maps `${HIPFIRE_BIND}:${HIPFIRE_PORT}` → container **11435**. The process inside always listens on 11435; **9001 is the host publish**. Health and harness hit **9001**.

If the user asks for another host port, use that instead of 9001. Do not change the container-side 11435 listen.

Pass extras in `HIPFIRE_SERVE_ARGS` whitespace-split (no quoting). Idle timeout stays in that string:

```bash
HIPFIRE_PORT=9001 HIPFIRE_UI=1 HIPFIRE_SERVE_ARGS='--idle-timeout 0' docker compose up -d
```

LAN + API key: append to the same var, still including idle timeout:
`HIPFIRE_SERVE_ARGS='--idle-timeout 0 --api-key-file /home/hipfire/.hipfire/api-key'`.


## Operator home (this host)

Bring serve containers up from **`~/docker-containers/hipfire/`**, not the
source checkout. Compose `name: hipfire` so Docker tags the project to that
folder (`com.docker.compose.project=hipfire`, working_dir that path). One
model served at a time; `container_name` is the served model. The identity,
listing and port rules below are mandatory on this host.

| Setting | Value |
|---|---|
| Working dir | `~/docker-containers/hipfire` |
| Compose project | `hipfire` (`name: hipfire` in the compose file) |
| `container_name` | short model id, e.g. `qwen36-35b-a3b-mq4r` |
| Service name | model id (one per stack); the *only* thing that may differ between files |
| `/v1/models` id | registry tag, e.g. `qwen3.6:35b-a3b-mq4r` |

Pass `--model <registry-tag>` (no slash). Bind-mount the trunk onto
`/home/hipfire/.hipfire/models/<registry-file>` so the tag resolves. A path
`--model /models/….mq4r` makes `/v1/models` advertise that path — do not.
A vision sidecar mounts as the trunk's sibling (`….vl`) and is passed with
`--vision` plus `HIPFIRE_VISION_MODE=on` (required on the VMM slot route).

Current Qwen3.6-35B-A3B MQ4R recipe:

```bash
cd ~/docker-containers/hipfire
docker compose -f compose.qwen36-35b-a3b-mq4r.yaml up -d
curl -s http://10.0.101.101:9001/health   # the bound host IP, not loopback
curl -s http://10.0.101.101:9001/v1/models   # id: qwen3.6:35b-a3b-mq4r
```

Replace the occupant of 9001 rather than picking another port.

### Container identity, listing and ports (mandatory here)

- **Every stack keeps compose project `hipfire`.** Arcane groups containers by
  `com.docker.compose.project`; giving a stack its own project
  (`name: hipfire-<model>`) splits the tag and drops the model out of the
  `hipfire` group. Never rename the project per model.
- **Distinctness comes from the service name.** Within project `hipfire`, each
  stack declares its own model-named service (`services: qwen38-27b-iu4p:`)
  and `container_name` equal to the model. Two files with the same project
  **and** the same service are one compose slot: `up -f fileB` recreates
  (removes) the container `fileA` started.
- **All stacks publish host `9001`.** The conflict is the enforcement, not an
  accident: the second `up` must fail
  `Bind for 0.0.0.0:9001 failed: port is already allocated` and its container
  stays `Created`. Never pick a second port; never remove the occupant to
  make room — `stop` it first when the user wants the other model.
- **Keep containers, stop them.** Arcane lists stopped containers and cannot
  list a removed one. Use `docker compose -f <file> stop` / `start` (or
  `up -d` on the file you want). `up -f <file>` leaves sibling stacks alone
  and prints an `orphan containers` warning — expected. `down -f <file>`
  removes only that file's service but also tries to delete the shared
  `hipfire_default` network, printing `Resource is still in use` (harmless
  while a sibling runs). Never run `down --remove-orphans` or a project-wide
  `docker compose -p hipfire down`.
- **Pre-list a stack without binding 9001**: `docker compose -f <file> create`
  (image must exist locally; `create` does not bind host ports).
- **After any compose edit**: `docker compose -f <file> config >/dev/null`,
  then `up -d` (a config change recreates the container, ~30 s model load).

### LAN exposure (this host)

- Publish on the host's LAN address, not every interface:
  `ports: - "${HIPFIRE_BIND:-10.0.101.101}:9001:9001"` (`ens18` is
  `10.0.101.101/24`). `HIPFIRE_BIND` overrides per start.
- **The publish target must be a host IP.** A container address
  (`172.20.0.x` on `hipfire_default`) or any non-local address fails at start:
  `failed to bind host port <ip>:9001/tcp: cannot assign requested address`.
  There is no "forward on the container IP"; a per-container LAN address is
  macvlan on `ens18` with the `ports:` rule removed, a different design.
- `192.168.1.200:9001` reaches this host through the router's DNAT to
  `10.0.101.101:9001`; verify with
  `curl -s http://192.168.1.200:9001/health`.
- Binding a named host IP means `127.0.0.1:9001` is **not** published — curl
  the bound address, not loopback. Confirm with `ss -ltn | grep 9001` and
  `docker port <container>`.
- **API key:** required only when publishing `0.0.0.0`. This host publishes a
  named LAN IP and runs keyless by operator decision, so every route —
  `/v1/*`, `/ui`, full `/health` — is open to whoever can reach the address.
  To re-enable: mount `./api-key:/home/hipfire/.hipfire/api-key:ro` (0600,
  ≥32 printable ASCII) and add
  `--api-key-file /home/hipfire/.hipfire/api-key` to the command. The embedded
  UI cannot send a bearer and will 401 while a key is set (`docs/SERVE.md`).

### Long context, volumes, restores

- `--max-seq` is bounded by `memory.max_seq` (schema default **32768**): a
  200k/262k serve needs the bound raised in that stack's mounted
  `config.toml` (`[memory] max_seq = 262144`), then `--max-seq 200000`.
  Budget: Qwen3.8-27B q8 KV logs `kv_bytes_per_token=34816`, so 200k is
  ~6.5 GiB on top of the 16.5 GB weight file; the load log prints
  `max_seq … card_cap=…`.
- Project/service renames re-prefix named volumes. Pin a warm kernel cache
  before renaming: `<key>: { external: true, name: hipfire_<old>_<key> }`.
- A model file with no registry tag is served by its **bare filename**
  (`--model qwen3.8-27b.iu4p`), never a container path; do not invent a tag.
- Restoring a stack whose image tag no longer exists: rebuild that exact image
  from its commit (`git worktree add --detach /tmp/b <sha>` then
  `docker build -f /tmp/b/Containerfile --target full -t <tag> /tmp/b`), never
  retag a different build in its place.
- Stale mounts/configs in a restored stack are the failure mode worth fixing at
  restore time: a `-UNIFORM`/`-MQ4` model directory that no longer exists, or a
  `[prompt] chat_template` path to a deleted file, must be repointed or dropped
  (the HFQ carries an inline template) before the stack is listed.

## Reach for this when

- `docker compose up` / `podman compose up` to serve a registry tag
- Building `Containerfile` targets `full` / `runtime` / `gate-runner`
- Reusing `~/.hipfire/models` inside a container
- Strix Halo NPU (`compose.halo.yml`) or the VMM 4-slot recipe
- Running `scripts/container-gate.sh` against a VALIDATION route
- Operator compose under `~/docker-containers/hipfire/` (project `hipfire`, model-named container)

Not this skill: writing quants ([hipfire-quantize](/home/ghazni/.agents/skills/hipfire-quantize/SKILL.md)),
host-only `hipfire serve` without containers, or NixOS service install
([`docs/NIXOS.md`](/home/ghazni/github/hipfire/docs/NIXOS.md)).

## Canonical owners (read these; do not fork)

| Concern | Owner |
|---|---|
| Operator stack | [`docs/CONTAINER.md`](/home/ghazni/github/hipfire/docs/CONTAINER.md) |
| Compose | [`compose.yml`](/home/ghazni/github/hipfire/compose.yml), [`compose.halo.yml`](/home/ghazni/github/hipfire/compose.halo.yml), [`.env.example`](/home/ghazni/github/hipfire/.env.example) |
| Image | [`Containerfile`](/home/ghazni/github/hipfire/Containerfile) (ROCm 10.1) |
| Entrypoint | [`docker/hipfire-entrypoint.sh`](/home/ghazni/github/hipfire/docker/hipfire-entrypoint.sh) |
| HTTP / API key | [`docs/SERVE.md`](/home/ghazni/github/hipfire/docs/SERVE.md) |
| VMM slot recipe | [`docker/compose.vmm-9001.yaml`](/home/ghazni/github/hipfire/docker/compose.vmm-9001.yaml) |
| Gate runner | [`scripts/container-gate.sh`](/home/ghazni/github/hipfire/scripts/container-gate.sh) |
| Claim → route | [`docs/VALIDATION.md`](/home/ghazni/github/hipfire/docs/VALIDATION.md) |
| Operator home compose | [`~/docker-containers/hipfire/`](/home/ghazni/docker-containers/hipfire) | this host; `name: hipfire`, model `container_name` |
| Publishing | [`.github/workflows/container.yml`](/home/ghazni/github/hipfire/.github/workflows/container.yml), README container section |

`docker/quantizer.Dockerfile` and `docker/rocm7-builder.Dockerfile` are
**not** serve images.

## First-class stack

One supported serve compose: root `compose.yml`.

```
host amdgpu (/dev/kfd, /dev/dri, optional /dev/accel)
  compose.yml x-hipfire-engine
  ├─ hipfire-init  → hipfire pull --yes $HIPFIRE_MODEL   (idempotent, exits)
  └─ hipfire       → entrypoint → hipfire serve
                     publish $HIPFIRE_BIND:$HIPFIRE_PORT → container :11435
                     skill default: HIPFIRE_PORT=9001, HIPFIRE_UI=1, --idle-timeout 0
```

Models are **never** baked into the image. They live in the models volume
(or a host dir) at `/home/hipfire/.hipfire/models`. Config:
`hipfire-config` → `/home/hipfire/.hipfire`. Kernel cache:
`hipfire-kernels` → `/var/cache/hipfire`.

`docs/CONTAINER.md` "Publishing status" still says images are unpublished.
Current truth: `v*` tags push `ghcr.io/warpfront/hipfire:<version>` and
`:latest` (full target). README pins `ghcr.io/warpfront/hipfire:0.4.4`.
Compose still builds locally when the tag is missing.

## Workflow

1. **Host GPU.** `/dev/kfd` + `/dev/dri` present; user in `render` (and
   `video`). No host ROCm install required. Confirm with `hipfire diag` on
   the host if the binary exists, else `ls /dev/kfd /dev/dri`.
2. **Settings.** From repo root: `cp .env.example .env` (optional). Then
   override skill defaults in the shell (or `.env`): `HIPFIRE_PORT=9001`,
   `HIPFIRE_UI=1`, `HIPFIRE_SERVE_ARGS=--idle-timeout 0`. Compose file
   defaults remain `11435` / UI off / idle 300s if you omit these.
3. **Up.** `docker compose up -d` or `podman compose up -d` with the env
   above. First start builds unless `HIPFIRE_IMAGE` pulls. `hipfire-init`
   pulls the tag into the models volume; `hipfire` starts after that succeeds.
   If 9001 is already published, replace the existing container — do not pick
   another port.
4. **Health.** `curl http://127.0.0.1:9001/health` — wait until
   `loading_model` is absent (healthcheck `start_period` 15 m). Then
   `/v1/models`, `/ui`, and a chat completion.
5. **Smoke the user path.** Against a running engine use
   `python3 scripts/serve_harness.py --no-spawn --port 9001 …` and pick the
   VALIDATION route. Image build ≠ GPU correctness.

## Quickstart

```bash
cp .env.example .env          # optional
HIPFIRE_PORT=9001 HIPFIRE_UI=1 HIPFIRE_SERVE_ARGS='--idle-timeout 0' \
  docker compose up -d          # or: podman compose up -d
docker compose logs -f hipfire-init hipfire

curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9001/v1/models
# UI: http://127.0.0.1:9001/ui
curl http://127.0.0.1:9001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8:27b-mq4-xts","messages":[{"role":"user","content":"Hello"}]}'
```

Pin a release image:

```bash
HIPFIRE_PORT=9001 HIPFIRE_UI=1 HIPFIRE_SERVE_ARGS='--idle-timeout 0' \
  HIPFIRE_IMAGE=ghcr.io/warpfront/hipfire:0.4.4 docker compose up -d
```

Reuse host models (do **not** bind-mount all of `~/.hipfire` — that shadows
the config volume):

```bash
HIPFIRE_PORT=9001 HIPFIRE_UI=1 HIPFIRE_SERVE_ARGS='--idle-timeout 0' \
  HIPFIRE_MODELS_PATH=/home/you/.hipfire/models docker compose up -d
```

| Variable | Compose file default | This skill |
|---|---|---|
| `HIPFIRE_MODEL` | `qwen3.8:27b-mq4-xts` | Init pull + serve pre-warm (~17.4 GB with MTP/DFlash/vision). Halo flagship: `qwen3.8:flash-next` (~125 GB) |
| `HIPFIRE_PORT` | `11435` | **9001** host publish; container still listens on 11435 |
| `HIPFIRE_BIND` | `127.0.0.1` | Host publish address (API **and** monitoring) |
| `HIPFIRE_UI` | `0` | **1** → `/ui`. Env switch, **not** a compose profile |
| `HIPFIRE_SERVE_ARGS` | empty | **`--idle-timeout 0`** (never unload). Extra `serve` args, whitespace-split, **no quoting** |
| `HIPFIRE_MODELS_PATH` | `hipfire-models` | Named volume or absolute host dir |
| `HIPFIRE_IMAGE` / `HIPFIRE_IMAGE_TARGET` | `ghcr.io/warpfront/hipfire:latest` / `full` | Image tag and build target |
| `HF_TOKEN` | unset | Gated Hugging Face pulls |
| `COMPOSE_FILE` | unset | `compose.yml:compose.halo.yml` for gfx1151 NPU |
| `HIPFIRE_DEVICES` / `ROCR_VISIBLE_DEVICES` | unset | Export in the **shell** before `up` |

## Images

| Target | When | Failure mode |
|---|---|---|
| `full` (**compose default**) | hipcc present; missing pack JIT-compiles once into `hipfire-kernels` | Use this unless you have measured the route on `runtime` |
| `runtime` | Slim, no compiler | Missing packaged kernel is a **hard error** (`no usable cached kernel image exists, but hipcc is unavailable`) |
| `gate-runner` | `scripts/container-gate.sh` only | Empty ENTRYPOINT; not a serve stack |

Toolchain is **ROCm 10.1**, same as release kernel packs. Baked arches:
`HIPFIRE_GPU_ARCHES` default `gfx1100 gfx1151 gfx1201`.

```bash
docker build -f Containerfile --target full    -t hipfire:full .
docker build -f Containerfile --target runtime -t hipfire .
HIPFIRE_IMAGE_TARGET=runtime docker compose up -d --build
docker build -f Containerfile --build-arg HIPFIRE_GPU_ARCHES="gfx1030" --target full -t hipfire:full .
```

## Rootless podman vs rootful docker

- Compose sets `run.oci.keep_original_groups: "1"`. Podman honors it; docker
  must **not** be given Podman's `keep-groups` token (it becomes a literal
  group name). `container-gate.sh` adds `--group-add keep-groups` only for
  podman.
- Entrypoint starts as root, grants device GIDs, then `setpriv` drops to
  uid 1000 `hipfire`, except: in a rootless user namespace, if the models
  dir is not writable by uid 1000, it **keeps namespace root** (your host
  user) and logs that. No `chown` / `chmod` / `podman unshare` under podman.
- Rootful docker + host models dir: `chown 1000:1000` the directory, or use
  the named volume.
- `HIPFIRE_RUN_AS_ROOT=1` skips the drop — last resort.
- Halo memlock: container cannot raise `ulimit -Hl` above the invoking
  user's hard limit (`≥ 1318060032`).
- Rootless cannot write host sysfs: run `sudo docker/hipfire-halo-tune.sh apply`
  on the host instead of the `halo-tune` profile.

## Halo, monitoring, VMM recipe

**gfx1151 NPU** — opt-in overlay, not a second stack:

```bash
echo 'COMPOSE_FILE=compose.yml:compose.halo.yml' >> .env
HIPFIRE_PORT=9001 HIPFIRE_UI=1 HIPFIRE_SERVE_ARGS='--idle-timeout 0' docker compose up -d
docker compose stop hipfire
docker compose run --rm --no-deps hipfire doctor --npu
```

`doctor --npu` fails `gpu_free` while serve holds the GPU — stop first.

**Halo host power** (privileged, never started by plain `up`):

```bash
docker compose run --rm halo-tune apply --dry-run
docker compose run --rm halo-tune apply
```

**Monitoring:** `docker compose --profile monitoring up -d` → Prometheus
`127.0.0.1:9090`, Grafana `127.0.0.1:3000`. Change
`GRAFANA_ADMIN_PASSWORD` before exposing 3000.

**VMM 4-slot manual test** (already host **9001**; **not** the product stack;
publishes `0.0.0.0` by default — set `HIPFIRE_BIND=127.0.0.1`; no TLS).
Trunk + `.mtp` + `.vl` must be siblings in `HIPFIRE_MODEL_DIR`:

```bash
docker build -f Containerfile --target full -t hipfire:dev .
docker compose -f docker/compose.vmm-9001.yaml up
curl -s http://127.0.0.1:9001/health
```

## Smoke

Compose healthcheck: `GET /health` until `"loading_model"` is absent.

Harness against a running container (do not spawn a second serve):

```bash
python3 scripts/serve_harness.py --no-spawn --port 9001 \
  --model /path/on/host/<file> --tag qwen3.8:27b-mq4-xts --mode battery
```

Gate runner — **no default gate** (no-arg exits 2). Name `bash` explicitly
(`gate-runner` ENTRYPOINT is empty; a bare `-lc` fails with
`exec: "-lc": executable file not found`):

```bash
scripts/container-gate.sh bash -lc 'exec python3 scripts/serve_harness.py --mode chain --model /root/.hipfire/models/<file> --tag qwen3.5:9b'
```

Env: `HIPFIRE_CONTAINER=podman|docker` (default podman),
`HIPFIRE_MODELS_DIR` (default `~/.hipfire/models`, mounted at
`/root/.hipfire/models`), `HIPFIRE_SKIP_BUILD=1`, `HIPFIRE_IMAGE`
(default `hipfire-gate`). Pick the route from VALIDATION.md.

GPU-free image smoke (`hipfire --version`, packed kernels under
`/opt/hipfire/bin/kernels/compiled`) is **not** GPU proof.

LAN bind: prefer a named host IP (`HIPFIRE_BIND=10.0.101.101`, see "LAN
exposure" above). Only `0.0.0.0` demands a key: keep idle timeout in the same
args and append `--api-key-file /home/hipfire/.hipfire/api-key` (file inside
the container, chmod 600, ≥32 printable ASCII). Serve has no TLS.

## Do not

- Omit `HIPFIRE_PORT=9001`, `HIPFIRE_UI=1`, or `--idle-timeout 0` unless the user asked
- Pick a second host port to avoid a conflict — one model at a time; replace the occupant of 9001
- Bake models into the image
- Use `runtime` for an unmeasured route
- Treat image build, SCS suite (`docs/SCS-SUITE.md` defaults port **8420**,
  not 9001), or Astrea KLD as serve acceptance
- Run retired `scripts/coherence-gate-*.sh` as a gate
- Enable PFlash or CASK (deprecated 0.4.0, removal 0.5.0)
- `chown` a rootless-podman models dir
- Auto-run `halo-tune`
- Run the operator serve from the git checkout (`…/hipfire/docker`) — use `~/docker-containers/hipfire/`
- Name the container after the compose service (`hipfire-vmm-1`) instead of the model
- Advertise a container path (`/models/….mq4r`) as the `/v1/models` id — pass the registry tag
- Publish `0.0.0.0` without an API key
- Rename the compose project per model (it must stay `hipfire`; make the
  service model-named instead)
- `down --remove-orphans` / project-wide `down` on this shared project
- Publish a container IP or any non-local IP (`cannot assign requested
  address`)
- Swap an image tag for a different build when restoring a stack

## Related skills

- [hipfire-quantize](/home/ghazni/.agents/skills/hipfire-quantize/SKILL.md) — write the HFQM file first
- [hipfire-autoheal](/home/ghazni/github/hipfire/.agents/skills/hipfire-autoheal/SKILL.md) — serve/runtime repair on the host
- [hipfire-tester](/home/ghazni/github/hipfire/.agents/skills/hipfire-tester/SKILL.md) — claim-scoped harness routes
- [astrea](/home/ghazni/github/hipfire/.agents/skills/astrea/SKILL.md) — quality, not serve
