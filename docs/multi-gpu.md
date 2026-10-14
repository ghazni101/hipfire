# Multi-GPU operator guide

Operator reference for hipfire multi-device modes. Mutable env inventory lives in
[`env-vars.md`](env-vars.md). Validation route selection lives **only** in
[`VALIDATION.md`](VALIDATION.md) — this page does not invent minimum routes.
Bring-up narrative is historical:
[`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md).

| Field | Value |
|---|---|
| Inventory date | 2026-10-13 |
| Audited source ref | `c7a732ad5b921706519cdba30fdb43a7e62aa2d4` (`origin/beta`) |
| Page state | **shipped / ref-pinned** for source-wired multi-device behavior (see [`INDEX.md`](INDEX.md)); the [Measured](#measured-flash-next-ep-on-2x-r9700-2026-10-13) tables are **measured** (fixture-bound, dated, not promises); the appendix tables are **historical** only |
| Orchestration | `crates/hipfire-runtime/src/multi_gpu.rs` |
| PP load path | `crates/hipfire-loader/src/carriers.rs` (`load_qwen35_pp`) |
| Daemon load / refusals | `crates/hipfire-daemon/src/main.rs` |
| Supporting multi-GPU script | [`scripts/pp-gate.sh`](../scripts/pp-gate.sh) (not a VALIDATION selector minimum route) |
| Admissions | [`admissions.yml`](admissions.yml) — schema v2, exactly one single-GPU retained-PM4 record; no multi-GPU records (fail closed; none inferred) |

## Modes

Three multi-device routes exist. PP and `tp` are **mutually exclusive** at load
(`tp > 1 && pp > 1` → error); Flash-Next expert parallelism (`topology.ep`)
cannot be combined with either (`pp` or `tp` ≠ 1 is refused when `ep > 1`).

| Mode | Load knob | What it does | Source-wired runtime surface (audited ref) |
|---|---|---|---|
| **Pipeline parallel (PP)** | daemon load `params.pp` (`N`, default `1`) | Contiguous **layer bands** across `N` devices; residual stream crosses bands via `boundary_copy` | Qwen3.5 / 3.6 **HFQ** only (`arch_id` 5 dense, 6 MoE/A3B) via `load_qwen35_pp` |
| **Expert parallel (EP / `tp`)** | `params.tp`, or CLI `hipfire serve --tp N` → `HIPFIRE_TP` | Within-layer expert sharding + all-reduce; every rank runs every layer (`Gpus::init_tp`) | MiniMax-M2 (`arch_id` 10) and DeepSeek V4 Flash (`arch_id` 9) via `load_model_ep` |
| **Flash-Next expert parallel (EP2 / EP4)** | config `topology.ep` (`hipfire config set topology.ep N`; env `HIPFIRE_TOPOLOGY_EP`; default `1`, allowed `1`/`2`/`4`; `HIPFIRE_QWEN4_EP=0` forces `1`). **Not `--tp`.** | Routed experts shard by id (`e % N`); every rank holds the whole trunk and runs the unchanged single-device program on its own worker thread; a copy-only owner exchange feeds the unchanged slot-order combine. **Opt-in, experimental.** | Qwen3.8 Flash-Next (Qwen4, `arch_id` 16) on exact **gfx1201** only, 2 or 4 ranks, every directed card pair peer-accessible (Qwen4 carrier, `load_qwen4_rank`; admission in `crates/hipfire-loader/src/admission.rs`) |

`pp = 1`, `tp = 1` and `ep = 1` are single-GPU. Behavior matches the pre-multi-GPU paths.

**None of the listed PP/EP routes is an admission or product default.**
[`admissions.yml`](admissions.yml) has no multi-GPU records at schema v2 (the sole earned row is single-GPU `pp=tp=1`).
Source-wired means the load path exists in runtime at the audited ref — not that
it is promoted. Flash-Next EP is opt-in (`topology.ep` defaults to `1`) and
carries the `experimental` lifecycle status in [`CONFIG.md`](CONFIG.md); the
defaults listed in [Flash-Next EP: enabling and status](#flash-next-ep-enabling-and-status)
apply only once `topology.ep` is set to `2` or `4`.

**PP is not tensor-parallel serving.** It does not give multi-user throughput.
It is a **capacity** tool: fit larger context / larger HFQ weights by splitting
layers. **No speedup is promised** for models that already fit on one card;
current historical measurements on one 2× gfx1100 station were slower under
sequential PP=2 (see [Appendix A](#appendix-a-historical-pp-evidence-immutable)).

## Which mode for which model

| Model / situation | Cards | Use | Why |
|---|---|---|---|
| Qwen3.8 Flash-Next (Qwen4) | 2 or 4 gfx1201 | **EP** (`topology.ep` = 2 or 4) | Experts are what does not fit one card: EP2 keeps 30 of 49 routed layers in VRAM per rank (19 host-mapped), EP4 keeps all 49, and EP2 prefill is measured faster than one card ([Measured](#measured-flash-next-ep-on-2x-r9700-2026-10-13)). EP4 prefill has no owner-routed path yet (see status). |
| MiniMax-M2, DeepSeek V4 Flash | 2+ | **EP via `--tp N`** | The `load_model_ep` route above; no admission, no measured numbers on this page. |
| Dense Qwen 27B-class that fits one card (registry weights: 11.78 GB for `qwen3.8:27b-mq3-xt`, [`MODELS.md`](MODELS.md); the historical probe in Appendix A shows about 15.5 GB for the mq4 27B proxy) | 2+ | **One model per card** (one daemon per card, each pinned by `hardware.devices`; the per-card lock makes two daemons asking for `gfx1201` get different cards) or **batched serving** on one card ([`SERVE.md`](SERVE.md) continuous batching) rather than splitting | Splitting a model that already fits adds cross-card traffic and, under PP, a sequential decode penalty; it does not buy multi-user throughput. |
| Qwen3.5 / 3.6 HFQ that does not fit one card at the wanted context | 2+ | **PP** (`params.pp`) | PP is a **capacity** tool, not a speedup: historical PP=2 decode measured about two thirds of single-card decode on short prompts (Appendix A). |

**CLI note:** the shipped CLI forwards **`tp`** (`HIPFIRE_TP` / `--tp`) on serve
load messages; that is the knob for MiniMax-M2 / DeepSeek V4 Flash. **Do not
pass `--tp` for Flash-Next:** with `topology.ep > 1` a load with `tp` ≠ 1 is
refused (`cannot combine with pipeline/tensor parallelism`); use
`topology.ep`. **`params.pp` is not a first-class CLI flag today** — set it on a
raw daemon JSONL `load` (or any client that builds that message). Examples and
`scripts/pp-gate.sh` do this directly.

## Topology (PP)

Source of truth: `Gpus` in `multi_gpu.rs`.

1. **Device pick** — `hardware.devices` lists physical cards in logical order
   (see [Device selection](#device-selection)). Startup resolves it from the
   KFD topology, reserves the cards, installs ROCr selectors plus HIP logical
   `0..N-1`, and after HIP init aborts unless every logical device has the
   resolved card's arch and PCI bus ID. `Gpus` construction repeats that check.
2. **Layer map**
   - Default: `Gpus::init_uniform(pp, n_layers)` — contiguous bands,
     `base = n_layers / N`, remainder distributed so max−min ≤ 1 layer.
   - Escape hatch: `HIPFIRE_PP_LAYERS=a,b,…` → `Gpus::init_layers` (length must
     equal `pp`, sum must equal `n_layers`). Skips the uniform free-VRAM delta
     check; still enforces arch match unless overridden.
3. **Placement convention (Variant 2)** — `output_device = last` device holds
   `output_norm + lm_head`. Device 0 holds the embedding side of the split.
4. **Boundary traffic** — at each band edge, `boundary_copy` moves the residual
   (`hipMemcpyPeerAsync` when peer access is up; otherwise HIP host-staging).
   Caller waits with `wait_boundary`.
5. **Peer access** — `enable_peer_all` must run **after** weights/KV/scratch that
   need peer maps are allocated. Incomplete peer matrix → host-staging fallback
   (slower, still correct). Partial pair failure does not abort capable pairs.
6. **Preflight**
   - Default: **exact `Gpu.arch` string match** across devices
     (`d.arch != arch0` in `preflight_vram` / `multi_gpu.rs`). This is not a
     loose “family” compare.
   - Free-VRAM delta ≤ `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` (default **2.0**) for
     `init_uniform` / `init_tp` only.
   - `HIPFIRE_ALLOW_MIXED_ARCH=1` opts into mixed-arch pairs (JIT per arch;
     peer may host-stage).
7. **Threading** — HIP work is **single-threaded** for the daemon lifetime.
   `bind_thread` before peer enable and before device-bound work. No rayon/tokio
   HIP callers in v1.

EP topology is different: `init_tp` sets every device’s layer map as “all layers
on rank 0” for PP helpers, while the EP forward ignores bands and shards experts.
RCCL all-reduce is used unless `HIPFIRE_TP_USE_RCCL=0` (host fallback not
implemented — that opt-out errors). Non-standard ROCm layouts (e.g. nixpkgs
splitting `librccl` out of the ROCm root) set `HIPFIRE_RCCL_LIB` to the full
`librccl.so` path; the loader tries that before the ROCm root candidates.

### Peer / fabric checks (host)

```sh
rocm-smi --showtopo
rocm-smi --showtoponuma
rocm-smi --showtopoaccess
```

A full `True` peer-access matrix is ideal. Missing peer access does not block
load; copies fall back to host staging.

### Device selection

`hardware.devices` (`HIPFIRE_DEVICES`, singular alias `HIPFIRE_DEVICE`) is a
comma-separated list; its order is the logical device order for PP/TP/EP.
Entries:

| Entry | Meaning |
|---|---|
| `N` | Index into the GPUs sorted by PCI address — the `rocm-smi` order. Not a ROCr or HIP ordinal. |
| `gfxNNNN` | First card of that arch, in PCI order, that is free (its daemon lock is not held) and not named elsewhere in the list. `gfx1201,gfx1201` gives two distinct cards. |
| `GPU-<hex>` | Exact UUID (`rocminfo` / `rocm-smi --showuniqueid`). |
| `DDDD:BB:DD.F` or `BB:DD.F` | Exact PCI address — the only exact form for cards whose UUID reads `GPU-XX`/`0x0`. |

Why a new index: ROCr numbers GPUs in KFD topology node order (`rocminfo`),
HIP numbers whatever ROCr exposed, and `rocm-smi` sorts by PCI bus. On a host
with a 7900 XTX (66:00.0), 6950 XT (6e:00.0), 5700 XT (99:00.0) and Strix Halo
iGPU (bf:00.0) the ROCr order is gfx1100, gfx1151, gfx1030, gfx1010 while the
PCI order is gfx1100, gfx1030, gfx1010, gfx1151. `HIPFIRE_DEVICES=2` is the
5700 XT there regardless of what ROCr calls it.

Resolution reads `/sys/class/kfd/kfd/topology/nodes/*/properties` only (no GPU
contact): nodes with SIMDs are GPUs, `gfx_target_version` gives the arch,
`unique_id` the UUID (0 = none), `domain` + `location_id` the PCI address. The
KFD GPU-node order equals ROCr's agent order (checked against `rocminfo` on a
5× gfx1201 host and the mixed host above). Each card lowers to
`ROCR_VISIBLE_DEVICES` as its UUID, or as its ROCr ordinal when it has none;
HIP gets `0..N-1`. Exact entries are claimed first, then each `gfxNNNN` entry in
list order. Unknown entries, misses, duplicates, a busy exact card, or no free
card of an arch fail closed and print the full device table.

Raw inherited `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` remain the expert
override when `hardware.devices` is unset: compatible pairs are normalized,
ambiguous pairs fail closed, and the daemon then identifies the HIP-visible
cards by PCI address to lock them.

Each daemon holds a non-blocking flock per card in `HIPFIRE_LOCK_DIR` (else
`/run/lock/hipfire`, else `/tmp/hipfire-locks`): `gpu-GPU-<uuid>.lock`, or
`gpu-pci-<dddd:bb:dd.f>.lock` for a card without a UUID. The `gfxNNNN` choice
and the reservation are one step, so two daemons asking for `gfx1201` get
different cards. `scripts/gpu-lock.sh` uses the same keys.

## Logical GPU emulation (developer-only)

Logical GPU emulation is a developer-only diagnostic route for exercising
expert-parallel rank, ownership, and combine behavior on one physical device.
It is not a multi-GPU product mode and does not change ordinary physical
admission.

`HIPFIRE_EMULATE_GPUS` is captured in the startup snapshot before GPU
initialization and parsed as `usize`. Only a successfully parsed integer `>=2`
enables emulation; an unset or malformed value, `0`, or `1` disables it. When
enabled, each requested logical device ID is aliased modulo the loaded
physical-device count:

| Physical devices loaded | Requested logical IDs | Resolved physical IDs |
|---|---|---|
| 1 | `[0, 1, 2, 3]` | `[0, 0, 0, 0]` |
| 2 | `[0, 1, 2, 3]` | `[0, 1, 0, 1]` |

The switch is only an enable switch; it does not choose the logical EP width.
`Gpus::init_ep(2, ...)` owns the two-rank layout and `Gpus::init_ep(4, ...)`
owns the four-rank layout. Duplicate physical IDs remain refused in ordinary
mode; the snapshotted emulation state is the explicit diagnostic exception for
sealed ownership, load, and batch policy.

### Single-device proof workflow

Use one physical device and keep the selectors explicit:

```sh
unset HIPFIRE_DEVICES HIPFIRE_TP_USE_RCCL
export HIP_VISIBLE_DEVICES=0
export HIPFIRE_EMULATE_GPUS=4       # use 2 for the logical EP2 route
export HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1
export HIPFIRE_EP_PEER_ALLREDUCE=1
```

Use `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` for decode and
`HIPFIRE_EP_PEER_ALLREDUCE=1` for batched prefill/tick. These selectors must be
chosen explicitly for the corresponding route. Never set either selector to
`0`, and never set `HIPFIRE_TP_USE_RCCL=0`: those values select/request RCCL
behavior and are not a host all-reduce fallback.

Aliased logical ranks retain separate rank state, streams, buffers, and sealed
ownership. Same-device peer setup is fail-soft: when two logical ranks resolve
to the same physical device, peer capability/enable calls are skipped for that
pair and the route continues with peer access disabled rather than aborting.

### Evidence boundary

The logical EP route-oracle suite is designed as byte-exact checks on one
gfx1151 device. A successful run validates sealed ownership, route broadcast,
canonical slot-order combine, result broadcast, inactive rows, and lifecycle
sequencing. The setup documented here is not itself evidence of a completed
lifecycle pass.

A passing logical-rank oracle does **not** prove RCCL, PCIe/xGMI or other
interconnect behavior, physical-device synchronization, throughput, product
gfx1201 admission, or G5 acceptance. Do not describe this evidence as physical
EP2/EP4, interconnect, or performance validation.

## Launch and config

### PP load (daemon JSONL)

```json
{"type":"load","model":"/path/to/qwen3.5-9b.mq4","params":{"max_seq":16384,"pp":2}}
```

Example process:

```sh
# Persist one device list for both HIP and ROCr (PCI-order indices).
hipfire config set hardware.devices 0,1

# Optional: asymmetric bands (must sum to n_layers, length == pp)
# export HIPFIRE_PP_LAYERS=16,16

# Bit-stable k-split reduction for pp=1 vs pp=2 parity work
export HIPFIRE_DETERMINISTIC=1

cargo run --release -p hipfire-daemon
# then send the load JSON above on stdin
```

### EP load (CLI): MiniMax-M2 / DeepSeek V4 Flash (`--tp`)

```sh
HIPFIRE_DEVICES=gfx1201,gfx1201 hipfire serve <minimax-or-deepseek4-tag> --tp 2
# equivalent: HIPFIRE_TP=2 …
```

EP does not honor a non-default `--kv-mode` the same way single-GPU load does
today — the CLI warns when both are set. Reload without a DFlash draft for EP.

### Flash-Next EP: enabling and status

Flash-Next (Qwen3.8 Flash-Next, Qwen4, registry tag `qwen3.8:flash-next`) uses
`topology.ep`, not `--tp`:

```sh
# two (or four) distinct gfx1201 cards, in logical order
hipfire config set hardware.devices gfx1201,gfx1201     # or HIPFIRE_DEVICES=…
hipfire config set topology.ep 2                         # or HIPFIRE_TOPOLOGY_EP=2; 4 for EP4
hipfire serve qwen3.8:flash-next
# optional: native MTP under EP is opt-in, `auto` stays AR
hipfire serve qwen3.8:flash-next --spec mtp
```

`HIPFIRE_QWEN4_EP=0` forces `topology.ep = 1` whatever the config says. The
first `ep` visible gfx1201 cards become the ranks. The load is refused (naming
`topology.ep`) when `pp` or `tp` is not `1`, when device 0 is not gfx1201, when
fewer than `ep` devices are visible, or when a directed card pair is not
peer-accessible; EP also refuses DFlash, DSpark, n-gram drafting, DDTree,
adaptive KV, drafters, CASK, PFlash, lane batching and the retained Redline
default, and `HIPFIRE_MTP_SAMPLED_MODE=naive`. EP decode is eager (no graph
capture).

| | EP2 | EP4 |
|---|---|---|
| Load / decode / verify / native MTP (`--spec mtp`) | Supported (opt-in, experimental) | Supported (opt-in, experimental) |
| Balanced per-call expert ownership on host-mapped layers ("W1") | **On** by default under EP; kill switch `HIPFIRE_EP_BALANCED_OWNERS=0` | On by default under EP (same switch); the decode numbers on this page are EP2 |
| Owner-routed prefill chunks ("S1") | **On** by default when every admission condition holds; kill switch `HIPFIRE_EP_OWNER_ROUTED_PREFILL=0` | **Refused at admission** with `EP4 owner-routed prefill not yet gated` until its hardware cells have run; EP4 keeps the full-table chunk route (replicated prefill, peer experts read over P2P) |
| Long-prefill expert staging ("S2") | **On** by default when S1 is admitted; kill switch `HIPFIRE_EP_EXPERT_STAGE=0` (pipeline-wide kill switch `HIPFIRE_QWEN4_EXPERT_STAGE`) | Never created (needs S1) |
| Routed layers resident per rank at 32K context | 30 of 49 (19 host-mapped), +0.34 GiB headroom | 49 of 49, +3.77 GiB headroom |

For the three levers, exactly `0` is off and unset or any other value is on
(they follow one rule, `ep_lever_on`). Single-card (`ep = 1`) behavior is
unchanged by all of them. The per-rank load log states what ran (`qwen4 EP
rank r/n: balanced ownership on N host-mapped MoE layers`; `owner-routed
prefill on …` or `owner-routed prefill not admitted (reason)`).

**Exactness.** Under EP the output is bit-identical to the same build on one
card, not merely close. Evidence (`fn_ep_oracle` and the serve harnesses, 2×/4×
R9700, pinned Flash-Next XTS sha256 prefix `8b15b6fe`; fixture-bound):

- EP2 and EP4 AR decode: logits and per-request state match one card at every
  step over 256 greedy tokens (ctx 512, 8192, 32768).
- Native MTP: greedy tokens, per-window results and trunk/head/drafter state
  match one card on EP2 (34/34 compared) and EP4 (52/52); greedy MTP equals AR.
- Sampled MTP (T 0.6, top-p 0.95, top-k 20, seed 7, two fresh processes) on
  EP2 with the defaults above: tokens, windows and state identical to one card,
  and per-seed stable. **No sampled-MTP oracle was run on EP4**, and no EP4
  hardware cell exists for S1/S2 (it is refused).
- S1 + S2 on EP2: 25 of 25 compares against one card pass (greedy AR and MTP at
  p8k and p16k, a 4,396-token prompt whose last chunk is 300 rows, bitwise chunk
  digests at layers 0/2/3/31/47); flag-off EP2 equals the previous EP2.

### Flash-Next EP: balanced expert ownership (default on; `HIPFIRE_EP_BALANCED_OWNERS=0` disables)

Under Flash-Next (Qwen4) expert parallelism rank `r` owns routed expert `e`
when `e % ranks == r`. On the trunk layers whose experts live in pinned host
memory (the host-mapped layers, read zero-copy over PCIe) the number of routed
experts a token draws from each rank differs call to call, so the rank with
fewer host experts finishes early and waits for the other at every such layer.

Balanced ownership (default on under EP; `HIPFIRE_EP_BALANCED_OWNERS=0` keeps
the static split) balances that per call. After the router wrote the call's
routed ids, one small device kernel deals the distinct routed experts to the
ranks alternately in ascending expert id, identically on every rank, and
rewrites the rank's live pointer tables for the layer: an expert the rank was
dealt keeps its real address (its own pinned tensor, or the owner peer's pinned
tensor, which every GPU can read), every other entry names the rank's zero
dummy expert. The routed kernels run unchanged on the unchanged ids, and the
row exchange then copies each expanded row bit-for-bit from the rank that
computed it. Every routed expert is computed from its own bytes by the same
kernel on exactly one rank, and the canonical combine is untouched, so the
output is byte-identical to the static split and to one card.

Scope: host-mapped trunk MoE layers on one-row decode and few-row forwards
(verify, replay and takeover, 2 to 8 rows). VRAM-resident layers, chunk
forwards (more than 8 rows) and the MTP head keep the static split. With
`HIPFIRE_EP_BALANCED_OWNERS=0`, no table, buffer, stage or kernel module is
added. Each rank logs `qwen4 EP rank r/n: balanced ownership on N host-mapped
MoE layers` at attach when it is on.

### Flash-Next EP2/EP4: owner-routed prefill chunks (default on under EP2; `HIPFIRE_EP_OWNER_ROUTED_PREFILL=0` disables)

Without the lever every EP rank runs the whole 512-expert grouped MoE of every
prefill chunk, reading its peers' experts through P2P weight tables.
`HIPFIRE_EP_OWNER_ROUTED_PREFILL` (default ON under EP2 on gfx1201; `=0` is the kill switch, any other value or unset is on; single card unchanged; EP4 is refused at admission with `EP4 owner-routed prefill not yet gated` until its hardware cells have run, the code is rank-generic)
splits that work. All ranks still run the router, top-k and stable grouping
(a deterministic function of replicated bytes, so the grouped layout is
identical everywhere). Each rank runs the grouped symmetric IU4 gate/up and
down on its own experts' tiles only (`e % ranks`):
a PeaceMaker kernel (`ep_tile_mask_pm_gfx1201`) writes
a copy of the tile ids with every tile of a peer expert set to the `-1`
sentinel the symmetric kernels already return on before any load, and the
kernels read the copy (the scatter's ids are never modified). The rank then
pulls every peer-owned tile's sixteen BF16 grouped rows of `y_down_grouped`
(the PeaceMaker tile pull: predicated 16-byte copies, approximately 105 MB per
layer/rank on EP2 or 157 MB on EP4 before grouping padding), and the unchanged
BF16-row combine reads exactly the bytes a single card stores. Per layer:
record ready after down, cross a host barrier, wait every peer's ready, pull,
record done and cross a second host barrier. Right before the next routed
down, wait every peer's previous done (one layer of trunk work later). Two
alternating event pairs make buffer reuse safe for either rank count.
Without an installed launch stream the waits are host
`hipEventSynchronize` calls, as in the decode row gather.

Scope: chunks of at least 512 rows when every MoE layer verified symmetric on
every rank. A tail chunk of 9 to 511 rows, few-row forwards (2 to 8 rows),
decode and the MTP head keep today's route. Per rank one `qwen4 EP rank r/n:
owner-routed prefill not admitted (reason)` line explains a refusal (ranks
other than 2 or 4, not gfx1201, the sym-IU4 route not requested, the compact
scratch plan, mismatching per-layer verdicts). The rotation of the grouped
activations still runs over every slot (it has no tile input);
non-owned tiles' activation rows are never consumed. Diagnostics:
`fn_ep_oracle --chunk-dump` (`HIPFIRE_CHUNK_DUMP_DIR`) digests `y_down_grouped`
tile by tile, the FFN output and the HC streams of selected layers for
bitwise comparison with a single card (`fn-ep/coal/s1/compare_chunk.py`).

### Flash-Next EP: long-prefill expert staging

`HIPFIRE_EP_EXPERT_STAGE` (default ON under EP2 when the owner-routed prefill is admitted; `=0` is the kill switch, any other value or unset is on) admits the existing
`HIPFIRE_QWEN4_EXPERT_STAGE` pipeline under expert parallelism. Each rank
borrows its last two resident compact expert layers as alternating DMA stages,
then restores their original bytes before the forward returns. Residency is
unchanged; each host layer copies 637.5 MiB/rank on EP2 or 318.75 MiB/rank
on EP4. Two donor mirrors/rank are charged for all ranks against one host's
RAM/GTT budgets; an all-resident load needs no host-layer stage.
Decode and verify continue to use mapped experts and host-wait row exchange,
independent of the staging launch stream. Staging stays outside capture/tapes.
EP staging requires every trunk MoE in the forward to bind the compact
owner-routed route: full-table forwards read peer donor VRAM and never stage.
`begin_forward(ep_compact_all_layers)` is the explicit per-forward admission
input. Before returning to any full-table EP chunk, every rank must call
`ep_quiesce(gpu)` and then cross a host barrier before any rank launches it,
so all donors' restore waits have completed before peer reads.

Stage tables preserve the bound runtime's non-owned entries: peer-mapped
experts for the full-table route, zero dummy experts for owner-routed compact
prefill. Owned expert `e` points at donor slot `e / ranks`. Internally
`Qwen4ExpertStage::prepare_runtime(gpu, layer, runtime)` prepares each full
table once before binding; compact tables are built at load. The compact-route
binder may call `tables_compact(layer)`; `tables(layer)` also selects the last
prepared runtime kind. Single-card tables retain their original parity layout.

### Rehostable owner-routed tile exchange

The PeaceMaker tile mask and pull kernels and runtime `EpTileExchange` are
model-independent building blocks for two or four ranks on gfx1201. The mask
reads `owner_table[expert]`; the pull uses the peer source on EP2 or selects
one of four registered bases by that logical owner on EP4. Every rank waits all peers' ready
events before pulling, and all peers' done events before overwriting its
grouped-row buffer. Two alternating event pairs and two host barriers per
exchange preserve the buffer lifetime without another row bank.

The owner table can be derived from the generic runtime's
`ExpertExecutionPlan::rank_ownership()` (`ExpertRankOwnership`), retaining
logical ranks rather than physical device IDs. This transport does not change
the generic EP schedule's numerical combine contract or attach itself to a
model. Flash-Next binds these pieces through its rank-generic S1/S2 adapter
for 0.4.2; moving that adapter onto the generic engine is separate work.
The EP2 pull retains its 48-byte kernarg specialization behind the same API.


### Environment (operator-facing)

Canonical table: [`env-vars.md`](env-vars.md) (`MULTI-GPU` group). Short map:

| Variable | Role |
|---|---|
| `hardware.devices` | Persistent device list (index, `gfxNNNN`, `GPU-<uuid>`, PCI address); resolved to physical cards, reserved, lowered to ROCr selectors plus HIP logical `0..N-1` |
| `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` | Expert one-shot filters when `hardware.devices` is unset; compatible pairs are normalized and ambiguous pairs fail closed |
| `HIPFIRE_DEVICES` / `HIPFIRE_DEVICE` | Compatibility aliases for `hardware.devices` |
| `HIPFIRE_PP_LAYERS` | Explicit per-device layer counts for PP |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB` | `init_uniform` / `init_tp` free-VRAM delta |
| `HIPFIRE_ALLOW_MIXED_ARCH` | Opt into mixed-arch device sets |
| `HIPFIRE_DETERMINISTIC` | Deterministic WMMA reduction path (parity / bisect) |
| `HIPFIRE_TP` | EP degree (CLI `--tp` sets this) |
| `HIPFIRE_TOPOLOGY_EP` | Flash-Next EP degree (`topology.ep`, default `1`, allowed `1`/`2`/`4`); **not** `--tp` |
| `HIPFIRE_QWEN4_EP` | `0` forces Flash-Next `topology.ep = 1` (developer kill switch) |
| `HIPFIRE_EP_BALANCED_OWNERS` | Flash-Next EP: balanced per-call expert ownership on host-mapped layers; on under EP, exact `0` disables |
| `HIPFIRE_EP_OWNER_ROUTED_PREFILL` | Flash-Next EP2: owner-routed prefill chunks; on under EP2, exact `0` disables; EP4 refused at admission |
| `HIPFIRE_EP_EXPERT_STAGE` | Flash-Next EP2: long-prefill expert staging; on when owner-routed prefill is admitted, exact `0` disables |
| `HIPFIRE_TP_USE_RCCL` | `0` opts out of RCCL (errors; no host AR yet) |
| `HIPFIRE_RCCL_LIB` | Explicit `librccl.so` path tried before the ROCm root (nixpkgs / split RCCL installs) |
| `HIPFIRE_PP_PFLASH=1` | **Experimental** — accept PFlash compose with `pp>1` (not a product default; not route-certified) |
| `HIPFIRE_PP_DFLASH=1` | **Experimental** — accept DFlash draft field with `pp>1` (cross-card spec generate is **not** fully implemented; see daemon refusal text) |

Test-only: `HIPFIRE_HAVE_2_GPU`, `HIPFIRE_PP_PARITY_MODEL` (pp_parity harness).

## Refusals, silent paths, and experimental exceptions (`pp > 1`)

Many non-support cases are enforced at **load** (daemon and/or carrier) and fail
closed — do not expect silent degrade to single-GPU for those. **Exceptions and
carrier-specific wording matter**; do not treat the table as one universal
string or a blanket “always refuse at load.”

| Condition | Behavior |
|---|---|
| `tp > 1` and `pp > 1` | Error: mutually exclusive |
| DFlash `draft` set and `HIPFIRE_PP_DFLASH` unset | Error: DFlash requires `pp=1` |
| DFlash `draft` set and `HIPFIRE_PP_DFLASH=1` | **Experimental exception:** load may accept; cross-card speculative generate is **not** fully implemented (daemon message states PR2–4 of the hetero PFlash/DFlash plan are incomplete). Not an admission. |
| CASK / TriAttention sidecar set | Error: requires `pp=1` |
| PFlash drafter / mode on and `HIPFIRE_PP_PFLASH` unset | Error: PFlash requires `pp=1` |
| PFlash on and `HIPFIRE_PP_PFLASH=1` | **Experimental exception:** opt-in only; not a product default and not route-certified performance. |
| Non-PP carriers at `pp>1` | Carrier-specific error strings (not one universal quote). Examples at the audited ref: `qwen2: pipeline-parallel (pp>1) unsupported`; `llama:` / `dots_ocr:` / `deepseek4:` / `minimax:` / `lfm2moe:` `pipeline-parallel (pp>1) unsupported` on HFQ (and distinct `safetensors + pp>1 unsupported` on dirs); **`cohere2moe: pp>1 unsupported via registry`** (different wording). |
| Qwen3.5 **safetensors directory** + `pp>1` | Error: `qwen35: safetensors + pp>1 unsupported` |
| Qwen3.5 / 3.6 **VL HFQ** + `pp>1` | **Not a hard refuse.** `load_qwen35_pp` is the text HFQ loader only — vision weights are **not** loaded on that path, so VL artifacts **silently become text-only** under PP. Serve real VL at `pp=1`. |
| `bench_prefill` / multi-GPU EP | Daemon refuses bench_prefill when `pp>1` or EP is active |

EP-only: `tp>1` with a DFlash draft → refused; non-EP arch → `load_model_ep` error.
Flash-Next `topology.ep > 1` has its own refusals (pp/tp ≠ 1, non-gfx1201, too few
devices, missing peer access, DFlash/DSpark/n-gram/DDTree/adaptive KV/drafters/CASK/PFlash/lanes/Redline,
`HIPFIRE_MTP_SAMPLED_MODE=naive`; EP4 owner-routed prefill): see
[Flash-Next EP: enabling and status](#flash-next-ep-enabling-and-status).

Dense Qwen3.5/3.6 at `tp>1` can speculate with MTP, **opt-in**: only an explicit
`speculation = "mtp"` (`speculation.mtp = on`, `--spec mtp`) takes the spec route.
The loader then replicates the MTP head on rank 0 (bundled `.mq4-mtp` trailer, the
CLI-resolved `.mtp` sidecar, or the trunk's sibling `.mtp`); the request drafts on
rank 0 and verifies across ranks. A missing head is a load error, as `mtp=on` is
on one GPU. `auto` (the default) and `off` keep the AR mesh loop even when a head
sits beside the trunk, so existing `tp>1` serves are unchanged. Block-verify
drafters (n-gram, DFlash, DSpark) stay refused on dense TP, and the n-gram-mod
composition inside MTP (`HIPFIRE_MTP_NGRAM`) drafts natively there.

Dense-TP MTP is not a byte-identical replacement for TP AR: speculative
verification runs a multi-row trunk geometry, whereas AR advances one token at
a time. The H2 greedy gate observed different output in some workloads, so a
throughput improvement alone does not qualify this route for automatic use.

### Dense TP2 reduction protocol (developer)

Dense Qwen3.5/3.6 TP decode reduces each layer's attention and FFN partials
into the residual through `Gpus::all_reduce_sum_f32_peer_add_routed`
(`hipfire-runtime`). Every route evaluates the same rank-ordered
`x + (p0 + p1)`, so the output is the same bytes on each route. With two
peer-access ranks the default is a symmetric direct-peer read; each reduction
waits on the peer's producer event and, before the next layer overwrites a
partial, on the peer's consumer-complete event.

Two developer controls, both default off, change only the synchronization:

- `HIPFIRE_TP_PEER_ALTERNATING=1`: the attention partial and the FFN partial
  live in different buffers. Once a peer has produced the next partial, it
  has finished reading the previous one, so the reverse wait is dropped.
- `HIPFIRE_TP_FULL_GRAPH=1`: exact gfx1100, TP=2, peer access, plain AR
  decode. Each rank's whole decode token is captured as one hipGraph. Every
  reduction becomes one graph-resident kernel: a device-resident signal
  barrier plus the residual add, which reads the peer's partial through a
  host-mapped, system-coherent exchange. Captured peer-VRAM reads can be stale
  on gfx1100. The loader prepares the signal tape and the exchange (about
  80 KiB for Qwen3.8-27B). This control implies the alternating protocol
  unless `HIPFIRE_TP_PEER_ALTERNATING=0`.

Neither control is a product default. The whole-token graph needs two gfx1100
cards, and its evidence so far is the author's 2x W7900 measurement in #830.
Validate it with the lab oracle `tp2_cross_device_graph_reduce`
(`--features lab`, `HIPFIRE_TP2_GRAPH_REPLAYS`) and with greedy transcript
identity against the default route.

### Architectural limits (current)

- Homogeneous **exact arch string** by default (`ALLOW_MIXED_ARCH` is opt-in).
- No automatic VRAM-weighted split.
- PP decode is sequential across bands (no async multi-band pipeline / per-band
  graph capture as a documented product path).
- Experimental `HIPFIRE_PP_*` flags are **not** admissions and are not
  route-certified performance features.

## Memory budget

Weights, KV, and scratch land on the devices that own each band. Last device
also carries `output_norm + lm_head`. Per-card headroom depends on quant, KV
mode, `max_seq`, and split.

**Reproduce on your hardware** (2+ visible GPUs, deltanet feature):

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
  -p hipfire-runtime --example pp2_vram_probe -- \
  ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

Historical VRAM and throughput tables from the original multi-GPU PP doc are
preserved **verbatim** in [Appendix A](#appendix-a-historical-pp-evidence-immutable).
They are **historical** only: they do not carry a complete measured-identity
manifest (measurement date + binary md5 + model identity on the same report),
so they must not be labeled **measured**, must not be treated as floors, and
must not be edited in place. Re-run the probe / perf protocol before claiming
fit or speed on other SKUs.

PP KV policy for Qwen3.5 multi-GPU defaults through `QWEN35_PP_POLICY` (see
`kv_mode.rs`) — do not assume single-GPU `auto` KV defaults apply unchanged:
PP and dense TP / MoE EP have no native constructor, so `auto` is `q8/q8` on
every arch there (gfx1201 included), and PP accepts only `q8`, `fwht2`,
`fwht3` and legacy `legacy-asym3` K.

## Measured: Flash-Next EP on 2× R9700 (2026-10-13)

> **Truth state: measured, fixture-bound** ([`INDEX.md`](INDEX.md)). These are
> observations on the named fixture, binaries and date below — **not** promises,
> floors, defaults or admissions. They were taken on the pre-merge branch build
> that became the Flash-Next S1/S2 entries of the 0.4.2 changelog, not re-run on
> the audited ref, and become **historical** the moment the code, the fixture
> or the cards change. Any other model, prompt length or card pair is
> unmeasured here.

| Field | Value |
|---|---|
| Date | 2026-10-13 |
| Hardware | 2× Radeon R9700 (gfx1201), EP2, peer access on |
| Model | Qwen3.8 Flash-Next MQ4 XTS, sha256 prefix `8b15b6fe` |
| Binaries (md5 prefixes) | prefill: `ade91299` daemon, `83c40bb1` CLI (S1+S2 build, flags forced on); default-on re-run: `6eeeed5f` daemon, `ac01b942` CLI; decode no-trade: `f7970c02` daemon, `2a9a2091` CLI |
| Single-card arm | same binary, `topology.ep = 1` |
| EP2 arm | S1 + S2 + balanced owners on (S1/S2 default on in the default-on re-run) |

### Prefill (`hipfire bench --spec off`)

Method: committed prompts (p8k 9,319 tokens, md5 prefix `c3ccdc43`; p16k 16,343
tokens, md5 prefix `68b768d0`), warm (`--runs 1 --warmups 1` per process), **3
fresh processes per arm**, arms interleaved, median of the three, prefill
tok/s.

| Prompt | Single card (3 runs) | Median | EP2 (3 runs) | Median | EP2 / single |
|---|---|---|---|---|---|
| p8k | 3030.1 / 3034.0 / 3032.5 | 3032.5 | 3942.3 / 3946.4 / 3942.6 | 3942.6 | 1.300 |
| p16k | 3754.7 / 3752.1 / 3753.9 | 3753.9 | 4301.6 / 4307.6 / 4308.4 | 4307.6 | 1.147 |

Every EP2 run was above the single-card median at both lengths. A
default-flags re-run at p8k (no `HIPFIRE_EP_*` set, three fresh processes per
arm, interleaved) gave single 3039.8 / 3037.4 / 3035.2 (median 3037.4) and EP2
3948.1 / 3941.6 / 3936.7 (median 3941.6), ratio 1.298. **EP4 prefill was not
measured** (S1 is refused on EP4); the changelog's earlier EP4 run measured a
42-token prompt at 408 ms to first token against 286 ms on one card.

### Decode: no trade for the prefill levers (EP2)

Method: `hipfire bench --runs 3 --warmups 2 --max-tokens 256`, greedy, 3 fresh
processes per arm, one round, EP2 with balanced owners only ("W1 only") against
S1 + S2 on ("combined"). Short outputs; decode tok/s, not end-to-end.

| Workload | W1 only (3 runs) | Median | Combined (3 runs) | Median | Change |
|---|---|---|---|---|---|
| MTP code | 84.0 / 84.0 / 85.3 | 84.0 | 85.1 / 84.1 / 83.6 | 84.1 | +0.1 % |
| MTP prose | 61.7 / 62.2 / 65.6 | 62.2 | 61.4 / 65.9 / 65.9 | 65.9 | +5.9 % (noise: the W1-only arm itself spans 61.7–65.6) |
| AR code | 50.4 / 50.4 / 50.5 | 50.4 | 50.2 / 50.3 / 50.3 | 50.3 | −0.2 % |

Resident routed layers per rank are identical in both arms (29 of 49 under MTP,
30 of 49 under AR). Read this as "S1/S2 did not slow decode on this fixture",
not as a speedup.

### Decode against one card (changelog-reported, EP work window 2026-10-11 to 2026-10-13)

Same pinned model, R9700s, greedy, committed prompts, 3 fresh processes per
arm, `hipfire bench`; the per-run receipts are held by the maintainers.

| Workload | One card | EP2 | EP4 |
|---|---|---|---|
| MTP code (`humaneval_0`, 134 tokens), tok/s | 44.4 | 81.7 | 121.6 |
| MTP prose (`fiction_lighthouse`, 44 tokens), tok/s | 34.5 | 63.4 | 87.1 |
| AR decode, tg128 `--spec off`, tok/s | 36.1 | — | 61.2 |

These are short outputs and decode tok/s, not end-to-end. The changelog line
does not record which `HIPFIRE_EP_*` defaults were in force for the EP2 rows,
and the AR row is one card against EP4 only.

## Validation routes

There is **no universal GPU gate** ([`VALIDATION.md`](VALIDATION.md)). Multi-GPU
claims are **manual** and path-specific. **`scripts/pp-gate.sh` is not named in
the VALIDATION claim→route selector** and therefore is **not** a canonical
minimum route ([`VALIDATION.md`](VALIDATION.md) § retired / unnamed gate
scripts). Treat it only as **supporting manual evidence**.

Map claims through the selector’s classes:

| Claim class (selector) | What applies for multi-GPU | Notes |
|---|---|---|
| Forward / fusion / KV **numerical or state parity** | Path-specific parity/state oracle when one exists for the surface | Supporting tool today: `pp_parity_chatml` example (also invoked by `pp-gate.sh`). If no oracle exists for a surface → **blocked**. Not `serve_harness.py`. |
| Forward / serve **user-facing semantics** | `scripts/serve_harness.py` with the exact model (after parity if numbers/state can break) | Semantics only |
| Perf improvement under PP or EP | [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md) + stationary matched runs; `speed-gate.sh` / `gates.sh` perf arm when applicable | Bench numbers without protocol/identity are not promotion evidence. Historical appendix rows are not floors. |
| Arch port (new device family behavior) | [`methodology/arch-port-validation.md`](methodology/arch-port-validation.md) | Channel + speed; no retired coherence battery as acceptance |
| Model/route **admission** | Row in [`admissions.yml`](admissions.yml) | Schema v2 exact-row only — multi-GPU PP/EP are **not** admitted |
| Docs-only edits | No-GPU CI / `scripts/no-gpu-ci.sh` | Never substitutes for GPU parity |
| Unknown multi-device surface | **Blocked** until VALIDATION grows a row | Fail closed |

Supporting manual commands (not selector minimums):

```sh
# Supporting multi-GPU battery (parity + daemon e2e + refusals). Skips cleanly
# with <2 usable devices. Not automatic merge proof; not a VALIDATION minimum.
./scripts/pp-gate.sh

# Faster: parity example only
./scripts/pp-gate.sh --skip-end-to-end

# Topology filter dry-run (no GPU work)
./scripts/pp-gate.sh --dry-run

# Direct parity example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
  -p hipfire-runtime --example pp_parity_chatml -- \
  ~/.hipfire/models/qwen3.5-0.8b.mq4
```

**pp-gate knobs** (see script header): `PP_GATE_DEVICES`,
`HIPFIRE_PP_GATE_INCLUDE_IGPU`, `HIPFIRE_PP_GATE_HETEROGENEOUS`,
`HIPFIRE_PP_GATE_MODEL`, `HIPFIRE_PP_GATE_REQUIRE_SYSFS`. Filters drop known APU
iGPUs and skip heterogeneous ISA families unless overridden.

Pre-commit (when hooks installed) runs pp-gate if staged paths match the
multi-GPU hotspot regex in [`.githooks/pre-commit`](../.githooks/pre-commit).
That is a **path-gated local guard**, not full product admission.

Retired `scripts/coherence-gate-*.sh` batteries are **not** acceptance for PP
([`VALIDATION.md`](VALIDATION.md) § retired).

## Related

- Env inventory: [`env-vars.md`](env-vars.md)
- Serve / EP CLI: [`SERVE.md`](SERVE.md), [`CLI.md`](CLI.md)
- Validation selector: [`VALIDATION.md`](VALIDATION.md)
- Historical bring-up: [`multi-gpu-bringup-lessons.md`](multi-gpu-bringup-lessons.md)
- Issue tracker context: [#58](https://github.com/warpfront/hipfire/issues/58)

---

## Appendix A — Historical PP evidence (immutable)

> **Lifecycle — historical only.** The block below is the prior multi-GPU PP
> document body retained for provenance. It is **not** current procedure, **not**
> a product floor, and **not** an admission. Truth state: **historical** (not
> **measured** — the retained tables lack a complete same-report measurement
> date + binary identity + model identity manifest required by
> [`INDEX.md`](INDEX.md)). Do not edit the evidence body; amend only by adding
> new dated sections outside this appendix. Warnings in the active sections
> above supersede any stronger claim language inside the retained body
> (including “measured”, “Status: v1 feature-complete”, absolute speedup
> wording, and gate-as-acceptance framing).
>
> **What the appendix tables are.** The July 2026 tables below (PP=2 VRAM
> probe and PP=2 throughput baseline) were taken on 2× Radeon RX 7900 XTX
> (gfx1100) for Qwen3.5-family models, with **short prompts only** (15 to 322
> tokens), before expert parallelism for Flash-Next existed. They are
> **historical**, not evidence for any current route, and are superseded for
> Flash-Next by the dated [Measured](#measured-flash-next-ep-on-2x-r9700-2026-10-13)
> section, which is a different model, different cards and 8k to 16k prompts.

# Multi-GPU Pipeline-Parallel

**Status:** v1 feature-complete on `feat/multi-gpu-pp` branch — tracking
issue [#58](https://github.com/warpfront/hipfire/issues/58). Stages
0–9 of the v2 plan are merged; refusal contracts (DFlash / VL / CASK +
pp>1) are wired and validated. This doc is the source of truth for
memory budget, deployment recipes, throughput, and known limitations.

## Why PP

hipfire on a single 24 GB card hits VRAM walls on:

- 27B at `--max-ctx ≥ 16K` with `kv_mode=asym3` (`AGENTS.md:356`)
- 35B-A3B at `--max-ctx ≥ 4K` with FP32 KV
- hypothetical 80B-A3B at any context

Pipeline-Parallel (PP) shards layers across N devices. Each device owns a contiguous "band"
of consecutive layers. The residual stream `s.x` flows through the bands sequentially: dev_0
runs layers `0..k1`, copies `s.x` to dev_1, dev_1 runs layers `k1..k2`, and so on. Final
`output_norm + lm_head` run on the last device (dev_last) — its `s.logits` is read by the
sampler in place.

**What PP gives you on 2× 24 GB:**
- Run 27B / 35B-A3B that don't fit on one card with extended context
- Unlock max_ctx on 27B beyond single-GPU OOM limits
- ~50-70% of single-GPU throughput on already-fitting models (sequential PP=2 is slower per token)

**What PP does NOT give you:**
- Faster multi-user serving — that's TP (tensor parallel), separate roadmap
- Speedup on models that already fit on one card

## Memory budget (per-card, PP=2)

Numbers below are **measured** on 2× Radeon RX 7900 XTX (gfx1100, 25.8 GiB VRAM
each) via `crates/hipfire-runtime/examples/pp2_vram_probe.rs` —
`hipMemGetInfo` deltas captured at each allocation stage
(`load_weights_multi`, `Qwen35ScratchSet`,
`KvCache::new_gpu_asym3_capped_multi`, `DeltaNetState`). Per-card columns
report the worst-of-two (the device that holds more — typically dev_last,
which carries `output_norm + lm_head`). `total` is the sum across both cards.

| Model | quant | n_layers | dim | KV mode | ctx | weights | KV/card | scratch+DN/card | total | per-card max | fits 24 GiB? |
|-------|-------|----------|-----|---------|-----|---------|---------|-----------------|-------|--------------|--------------|
| qwen3.5:0.8b | mq4 | 24 | 1024 | asym3 | 4096 | 1.3 GB | 50 MB | 8 MB | 1.3 GB | 0.7 GB | yes |
| qwen3.5:4b | mq4 | 32 | 2560 | asym3 | 4096 | 4.0 GB | 134 MB | 15 MB | 4.0 GB | 2.0 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 4096 | 5.6 GB | 134 MB | 19 MB | 5.6 GB | 2.8 GB | yes |
| qwen3.5:9b | mq4 | 32 | 4096 | asym3 | 16K | 6.2 GB | 436 MB | 46 MB | 6.2 GB | 3.1 GB | yes |
| qwen3.5:9b | mq3 | 32 | 4096 | asym3 | 4096 | 4.4 GB | 134 MB | 19 MB | 4.4 GB | 2.2 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 4096 | 15.5 GB | 268 MB | 42 MB | 15.5 GB | 7.8 GB | yes |
| qwen3.5:27b *(via 3.6 proxy)* | mq4 | 64 | 5120 | asym3 | 16K | 16.8 GB | 872 MB | 80 MB | 16.8 GB | 8.4 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 4096 | 12.6 GB | 268 MB | 40 MB | 12.6 GB | 6.3 GB | yes |
| qwen3.5:27b | mq3 | 64 | 5120 | asym3 | 16K | 13.8 GB | 872 MB | 78 MB | 13.8 GB | 6.9 GB | yes |
| qwen3.6:35b-a3b | mq4 | 40 | 2048 | asym3 | 4096 | 23.5 GB | 103 MB | 21 MB | 23.5 GB | 11.8 GB | yes |
| (hypothetical) 80B-a3b | mq4 | 80 | 8192 | asym3 | 4096 | ~42 GB | ~250 MB | ~2 GB | ~44 GB | ~22 GB | estimate, tight |

Notes:
- `qwen3.5:27b mq4` rows use `qwen3.6:27b mq4` as a measurement proxy (same
  `n_layers=64`, `dim=5120`, `head_dim=256`, `n_kv_heads=4` — VRAM-equivalent).
  When `qwen3.5:27b mq4` lands as a downloadable artifact the rows can be
  re-measured directly.
- `qwen3.6:35b-a3b mq4` is the current-generation A3B; `qwen3.5:35b-a3b mq4`
  ships local-only with the same MoE shape — measurement carries over.
- The 80B row stays an estimate — no public artifact exists.
- "scratch+DN/card" combines `Qwen35ScratchSet::per_device[i]` (residual
  stream, attention/FFN scratch, flash partials, logits) and the
  `DeltaNetState` slice owned by that device's LA-layer band.

**Asymmetry under Variant 2 (lm_head on dev_last):**
- dev_0 carries `token_embd`
- dev_last carries `output_norm + lm_head`
- Layer count shifts by ±1 for `n_layers % 2 != 0` (uniform split formula
  `base + (i < rem ? 1 : 0)`)

Probed on this hardware: per-card max < 24 GiB at every measured shape. The
A3B model at 11.8 GB/card has the largest headroom consumer; everything
else stays under 8.5 GB/card with 4 K context, under 9 GB/card at 16 K.

To reproduce on your hardware:

```sh
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp2_vram_probe -- \
    ~/.hipfire/models/qwen3.5-9b.mq4 4096
```

## Deployment recipes

The daemon takes a `pp` field in the load message (default `1` =
single-GPU, identical behavior to pre-PP code paths):

```sh
# Filter to the two 7900 XTX (drop the iGPU)
HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:27b --max-ctx 16384 "Hi"

# Bypass the inherited `gemm_..._wmma_ksplit` non-determinism (k-split
# atomicAdd reduction varies by warp scheduling — see commit f54ca71
# and kernels/src/gemm_hfq4g256_residual_wmma_ksplit.hip:22). Required
# for byte-equivalent output across processes / pp configurations.
HIPFIRE_DETERMINISTIC=1 HIP_VISIBLE_DEVICES=0,1 hipfire run qwen3.5:9b "Hi"

# Override uniform-VRAM tolerance (default 2 GiB; arches must match)
HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=4 hipfire run qwen3.5:27b "Hi"
```

Direct daemon JSON (driving without the CLI):

```json
{"type":"load","model":".../qwen3.5-27b.mq4","params":{"max_seq":16384,"pp":2}}
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `hardware.devices = "3,1"` | PCI-order cards 3 and 1 become logical devices 0 and 1; ROCr receives their UUIDs (or ROCr ordinals for no-UUID cards) |
| `HIPFIRE_DETERMINISTIC=1` | Force k2 WMMA reduction (no atomicAdd) — bit-identical across processes/pp configs at ~33% perf cost on small-batch decode |
| `HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB=N` | Pre-flight VRAM-asymmetry tolerance for `Gpus::init_uniform` (default 2.0) |
| `HIPFIRE_PREFILL_BATCHED=0` | Disable batched WMMA prefill (per-token fallback). Diagnostic for ksplit non-det isolation |
| `HIPFIRE_PREFILL_MAX_BATCH=N` | Override per-chunk prefill batch. When unset/invalid: arch defaults are **512** on exact `gfx1100`, **384** on exact `gfx1201`, else **256** (`PREFILL_MAX_BATCH`). Under TP, `prefill_max_batch_tp` uses default×`tp` (cap **2048**). An explicit `HIPFIRE_PREFILL_MAX_BATCH` wins over both the arch default and the TP scale. Chunks > N split with peer-copy at the boundary |
| `HIPFIRE_WO_WMMA_VARIANT={k2,ksplit,k4,…}` | Manual override of the wo-residual GEMM variant — see `dispatch.rs` auto-dispatch |

### Refusal matrix at load (`pp > 1`)

| Feature | Behavior | Why |
|---------|----------|-----|
| arch_id ∈ {5, 6} (Qwen3.5 dense + MoE/A3B) | Accepted | Validated end-to-end |
| arch_id = others (LLaMA / Qwen3) | Refused | Single-GPU only in v1 |
| VL models (vision_config + vision tensors) | Refused | v1.1 |
| DFlash draft (`draft` field set) | Refused | v1.1 — see `feedback_cask_mfold_dflash_broken.md` for the v1 ship-blocker |
| CASK / TriAttention sidecar | Refused | Eviction context is single-device — v1.1 |

## Throughput baseline (gfx1100 × 2)

Measured on 2× Radeon RX 7900 XTX, ROCm 6.4.3, with
`HIPFIRE_DETERMINISTIC=1` (bit-equivalent pp=1 ↔ pp=2 output).

| Model | Prompt | pp=1 prefill | pp=2 prefill | pp=1 decode | pp=2 decode | pp=2/pp=1 decode |
|-------|--------|--------------|--------------|-------------|-------------|------------------|
| 0.8B mq4 | 22 tok | 838 tok/s | 588 tok/s | 332 tok/s | 227 tok/s | 68% |
| 0.8B mq4 | 322 tok (chunked) | 6493 tok/s | 5490 tok/s | 315 tok/s | 212 tok/s | 67% |
| 35B-A3B mq4 (MoE) | 15 tok | 331 tok/s | 258 tok/s | 142 tok/s | 97 tok/s | 68% |

The pp=2 decode penalty is inherent to v1: per-token
`forward_scratch_multi` pays one HIP launch per kernel per layer with
no graph capture (vs pp=1 which captures + replays the AR-step graph
after warmup). Pipelined decode + per-band graph capture lift this in
v1.1.

## Limitations (v1)

Refused at load time:
- `pp > 1` + DFlash speculative decode
- `pp > 1` + CASK/TriAttention sidecar (eviction is single-device)
- `pp > 1` + VL models (vision encoder is single-device)
- `pp > 1` + arch_id ∉ {5, 6} (LLaMA / Qwen3 dense are pp=1 only)

Architectural limits in v1:
- Homogeneous arch only (`init_uniform` hard-fails on arch mismatch)
- Uniform layer split — `init_layers(per_device)` is the manual escape hatch
- Per-token decode (no async stream pipeline / per-band graph capture) — v1.1
- Pipelined prefill (chunk N+1 on dev_0 while chunk N processes on dev_1) — v1.1

## Validation (Stage 9)

```sh
# Multi-GPU gate. Skips silently when fewer than 2 GPU visible.
./scripts/pp-gate.sh

# Just the parity smoke (no daemon end-to-end), faster
./scripts/pp-gate.sh --skip-end-to-end

# Underlying byte-equivalence example
HIP_VISIBLE_DEVICES=0,1 cargo run --release --features deltanet \
    -p hipfire-runtime --example pp_parity_chatml -- \
    ~/.hipfire/models/qwen3.5-0.8b.mq4
```

The `pp-gate.sh` battery checks:
1. Per-token `forward_scratch_multi` ≡ `forward_scratch` bit-exact (pp_parity_chatml)
2. Daemon `pp=1` ≡ `pp=2` byte-identical with `HIPFIRE_DETERMINISTIC=1` (greedy ChatML)
3. DFlash + pp=2 refusal at load
4. CASK + pp=2 refusal at load

A pre-commit hook calls `pp-gate.sh` automatically when staged files
match the `multi_gpu|pp_|peer_access|pipeline|stages` hotspot regex.

## Verifying peer access on your hardware

```sh
rocm-smi --showtopo            # weights, hops, link types
rocm-smi --showtoponuma        # NUMA placement
rocm-smi --showtopoaccess      # peer accessibility matrix
```

A `True` in every cell of `--showtopoaccess` for the cards you plan to use means peer-access
should work. If not, hipfire falls back to host-staging via pinned buffers (slower but correct).

## Open questions (will be filled in as Stages land)

- DFlash + PP integration scope — pending maintainer guidance on issue #58
- Whether mixed-arch should be soft-warn or hard-fail — currently hard-fail
- `hardware.devices` is the physical device source of truth; see [Device selection](#device-selection)
