<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving cache scheduler — work summary and remaining gaps

- **Date:** 2026-09-05 (review pass appended same day)
- **Branch:** `feat/serving-cache-scheduler` (pushed to `origin` at `ghazni101/hipfire`)
- **Tip:** `aeeda2259` `fix(loader): probe stem .mtp sidecars like the slot engine`
- **Spec:** [2026-09-05-serving-cache-scheduler-spec.md](2026-09-05-serving-cache-scheduler-spec.md)
- **Plan:** [2026-09-05-serving-cache-scheduler-plan.md](2026-09-05-serving-cache-scheduler-plan.md)
- **GPU:** gfx1101 (AMD Radeon RX 7700 XT), HIP 7.15, ROCm 10 container (`local/rocm-base:10.0.0`)
- **Model:** `qwen3.5-4b.mq4v2.hfq` + `qwen3.5-4b.mtp` sidecar

---

## Review pass (2026-09-05, wave 7)

Two parallel code-review sweeps (structured-generation wave; daemon/admission/bench wave) plus a first-party audit of the ownership/scheduler core. Every finding below was verified against source, fixed, and covered by a test or an oracle cell.

### Correctness bugs found and fixed

| # | Area | Bug | Fix |
|---|---|---|---|
| 1 | `page_pool.rs` | `release_cache_ref` with in-flight refs left the page in limbo — neither free nor in `reclaim_pending` — leaking it permanently | Mirror `dec_table_ref`: enter `ReclaimPending`, freed by `drain_completed`; `add_{cache,inflight}_ref`/`refcount_inc` refuse `ReclaimPending` pages; `drain_completed` also requires zero table/cache refs |
| 2 | `prefix_index.rs` | Resumability compared a checkpoint boundary against a cumulative count of valid pages (gaps below a boundary could be compensated by valid pages above) | Walk tracks the gap-free contiguous resident prefix; a checkpoint is resumable only while every page below it is resident (A8) |
| 3 | `prefix_index.rs` | A chain insert that hit the CPU-node bound partway left its already-inserted nodes in the tree without adding them to `total_nodes` — the bound silently stopped binding | `create_chain` rolls its partial chain back on bound failure; accounting stays exact |
| 4 | `prefix_index.rs` | Eviction found a leaf's parent by an O(nodes) scan per leaf | Nodes carry a parent pointer; eviction unlinks in O(1) |
| 5 | `serve_fairness.rs` | A prefill whose remaining tail was smaller than `prefill_min_tokens` could never be granted — the slot livelocked in prefill forever (hidden at default min=1) | A grant that completes the prompt (`rows == uncached`) is always allowed; sliver control still applies to non-completing grants |
| 6 | `serve_fairness.rs` | `aged` was never cleared — one skipped round promoted a request ahead of the whole queue forever | Age recomputed per round: served requests leave the aged band and can be re-aged |
| 7 | `checkpoint.rs` | Insert "break"ed out of the eviction loop when everything was pinned and inserted anyway — the byte ceiling was soft | Refuse the capture (drop the blob, return `CheckpointId::NONE`); the boundary stays honestly unresumable |
| 8 | `serve_engine.rs` | Checkpoints captured the live DeltaNet state (`S_next_pos`) but were labeled at the page-aligned boundary below — any chunk ending mid-page poisoned every resume from that boundary (spec §4.5 relabeling ban) | Capture only when the published boundary equals the state boundary; mid-page publications are KV-only |
| 9 | `serve_engine.rs` | DN checkpoint restore was fire-and-forget (`let _ = restore_to(..)`) — a failed restore left shared KV paired with zeroed recurrent state | Restore first, fail-closed to the cold path on error |
| 10 | `serve_engine.rs` | Publish failure orphaned the cache refs taken before `publish_sealed_pages` | Refs rolled back on failure (tree-unchanged guarantee from fix 3 makes this exact) |
| 11 | `serve_engine.rs` | `reset` replaced the radix without releasing its cache leases — every published page stranded CacheOnly; the pool drained to 0 free pages within a few reset cycles (caught by the new A20 re-warm cell) | `PrefixIndex::release_all` + `drain_completed` wired into Reset |
| 12 | `serve_engine.rs` | Admit had no page-pressure reclaim (spec §4.4: reclaim cache-only pages before rejecting work) | Cold admit evicts oldest unpinned radix leaves when free pages cannot back suffix + generation budget |
| 13 | `serve_engine.rs` | The step-reservation shrink path trimmed phantom rows (accounting said fit, batch unchanged) | After draft dropping, a remaining overflow is an accounting fault: fail closed per S4 |
| 14 | `serve_engine.rs` | Continuation-admit compiled the JSON schema with `.ok()` — a compile failure admitted the request silently unconstrained after `Accepted` went out | Compile hoisted before any session mutation; failure rejects before `Accepted` |
| 15 | `serve_engine.rs` | Jump-forward greedy gate (`> 1e-6`) disagreed with the sampler's argmax gate (`== 0.0`) — tiny temperatures took the RNG-consuming path, breaking G3.4 draw accounting | Gate aligned to `temperature == 0.0`; `DrafterDecision::Reseed` now declared truthfully |
| 16 | `grammar.rs` | Unknown keywords fell through `_ => {}` — `if`/`then`/`else`, `additionalItems`, and arbitrary unknowns compiled with constraints ignored | Default-deny; annotations remain the only silent accepts |
| 17 | `grammar.rs` | `{"type":["integer","null"]}` fell into the inference path and compiled to unconstrained `Any`; schema-form `additionalProperties` flattened to `true` | Both rejected before generation; CLI validator aligned (same refusal set, doc parity) |
| 18 | `grammar.rs` | `json_equal` was serde `PartialEq` — variant-strict (`1 != 1.0`), stranding matchers mid-generation on numeric enums | Semantic numeric equality (integer-exact when possible, f64 otherwise, recursive on composites); `number_could_grow_to_match` accepts semantically-equal spellings |
| 19 | `grammar.rs` | Duplicate object keys (including escaped spellings) merged silently via serde | Raw byte scan decodes keys and rejects duplicates at accept time; same-key-in-sibling-objects stays legal |
| 20 | `grammar.rs` | Required name without a `properties` entry rejected as unsatisfiable even when the object was open | Compiles as an `Any` property when `additionalProperties: true`; genuinely unsatisfiable only when closed |
| 21 | `grammar.rs` | Recursion heuristic rejected finite schemas on repeated property names | Dropped — `$ref` (already rejected) is the only recursion vector |
| 22 | `admission.rs` | `materialize_growth` = release-then-charge: over-materialize discarded other requests' credits and charged anyway; releases saturated on underflow, inflating free capacity | Atomic materialize (verify-then-mutate); releases are checked errors; checked arithmetic in `AdmissionController` |
| 23 | `http.rs` | Stream backpressure byte bound computed `forwarded/forwarded == 1` byte per pending slot — the 16 MiB bound could never trip, and a Full channel blocked the forwarder forever with no deadline | Real pending-byte counter shared producer↔consumer (saturating, never wraps); stall = retry loop within `stream_stall_timeout_ms`, then typed abort; regression tests for trip/resume/deadline |
| 24 | `daemon` | Slot worker threads spawned before the concurrency check (spec §5.3 anti-pattern); `queue_bytes` hardcoded 0 so the wait-queue byte cap could not bind; `json_object` accepted then silently unconstrained (front end dropped it) | Pre-spawn bound at the dispatch site; `canonical_prompt_bytes` wired into `SubmitRequest.queue_bytes`; `json_object` rejected typed at both front end and daemon |

Perf hardening: `is_token_allowed` gained bounded fast paths (structural first-byte set + inert string content) so the vocabulary scan stops cloning+reparsing for inert tokens; the scanner also feeds duplicate detection.

### Verification (all in the rocm10 container, gfx1101)

- `cargo test --workspace --lib`: all suites green (the only failure, `oversized_pool_is_refused_not_allocated`, is the documented pre-existing environmental case — passes with `HIPFIRE_OOM_GUARD=1`; the 7 `tests::update_*` self-installer tests also fail on the base commit in this container).
- GPU oracle `test_serve_prefix_cache --mtp-k 4`: PASS — greedy/sampled/grammar reuse cells, plus new A10 (96-token generation crossing page boundaries replays identically), A13 (long cold prefill + short warm request both complete), A20 (soak, reset → cold, **re-warm after reset reuses again** — the cell that caught fix 11), and A19 `--fault-publish` (injected publish failure → honest miss with identical output).
- End-to-end HTTP (multi-slot + prefix cache serve): `serve_harness.py` chain `cached_tokens 0→89→192→265→346` (matches the pre-change reference) and battery complete; streaming strict-schema requests get typed A17 rejections instead of false successes; `json_object` and unsupported schema keywords are typed rejections.

### Remaining gaps (unchanged or newly precise)

- **Early per-token schema pruning (new, precise):** the incremental matcher is syntax-conservative — a schema-invalid *value start* (e.g. `null` under a `"type":"string"` property) is only caught at terminal validation as a typed A17 failure, never a false success. Spec-conformant but late; schema-aware incremental matching (the dormant `Frame` machinery) is the follow-up.
- **Vision + prefix reuse (X2/A18)** — still off; needs the pixel/embedding/position oracle.
- **Slots tools / stop / logprobs** — refusals stay (spec: remove only with complete behavior).
- **`admissions.yml` / ARCHITECTURE.md / default-on** — still gated on the full §12 evidence tuple.
- **P6 overlap** — flag only, stays off.

---

## What was done

### Waves 0–4 (prior commits, `6b07ef415`–`f3222d63d`)

| Wave | What landed |
|---|---|
| 0 merge | Folded `feat/multislot-vision-mtp` ViT layer-chunking (no D2H/H2D roundtrip) into the branch base |
| 1 P0 | `serve_contract` types (CacheDomain, PrefixLookup, DrafterDecision); multi-inflight bench client API; daemon slots benchmark scaffold |
| 2 P1 | `PagePool` generations, sealing, COW, leases, deferred reclaim; `SlotPool` integration; bounded admission with physical/future budget accounting |
| 3 P2 | Token-keyed radix `PrefixIndex` with domain isolation; Qwen `DeltaNetSnapshot` checkpoint pool + resume planner; prefix-cache lookup → pin → resume-plan in the admit path; sealed-page publication at commit boundaries |
| 4 P3 | Global `max_batch_tokens` budget; prefill quantum; per-slot COW isolation (S4); `FairQueue` age-aware scheduling with deficit ring; bounded `WaitQueue` (replaces R-A4 reject); `forced_token_run` planner for JSON Schema (default off) |

**Defaults:** `serve.prefix_cache=false`, `serve.structured_jump_forward=false`, `serve.scheduler_overlap=false`. All off.

### Wave 5 — P5 composition (this session, `fd3364798`–`aeeda2259`)

#### C1 cache-domain identity (`fd3364798`)

Replaced zeroed tokenizer/template stubs and file-size proxy with real SHA-256 digests:

- `Tokenizer::vocab_digest()` — SHA-256 of decoded vocabulary in id order
- `Tokenizer::config_digest()` — SHA-256 of merge ranks, special tokens, BOS/EOS/EOT, BPE/SentencePiece flags
- `HfqFile::content_digest()` — SHA-256 of arch_id, metadata JSON, ordered tensor manifest, file length (path/inode/mtime excluded)
- `serve_contract::sha256_len_prefixed()` / `sha256_file()` — shared digest helpers
- Sidecar digests (MTP, VL) hashed independently via `find_mtp_sidecar` + `HfqFile::content_digest` or `sha256_file`

#### Token-keyed page handles (`fd3364798`)

- `PrefixIndex::lookup_with_pages()` returns the actual sealed `Handle` list along the matched prefix, not a length-keyed side map
- Two equal-length different prefixes return distinct physical pages (`lookup_with_pages_isolates_equal_length_prefixes` host test)
- Removed `published_handles: HashMap<u64, Vec<Handle>>` side map from `Rig`
- Admit path shares `hit_handles` directly into the destination slot via `share_published_pages`

#### Exact-match reuse (`fd3364798`)

- `reused_tokens_from_plan(boundary, prompt_len)` now returns `boundary.min(prompt_len)` — a boundary covering the entire prompt is a full hit, not zero

#### MTP sidecar discovery (`3803a0573`, `aeeda2259`)

- `mtp_sidecar_candidates()` / `find_mtp_sidecar()` in `crates/hipfire-arch-qwen35/src/mtp_head.rs` — probes both `with_extension("mtp")` (last extension) and the stem after stripping `.hfq` + quant suffixes (`.mq4v2`, `.mq4`, …), so `qwen3.5-4b.mq4v2.hfq` finds `qwen3.5-4b.mtp`
- Slot engine (`serve_engine.rs`): both bundled-miss and bundled-error fallbacks use `find_mtp_sidecar`; C1 sidecar identity uses `find_mtp_sidecar`; miss message lists all probed paths
- Single-slot loader (`hipfire-loader/src/lib.rs`): both fallbacks use `find_mtp_sidecar`; error message lists probed paths

#### GPU oracle (`3803a0573`)

`crates/hipfire-runtime/examples/test_serve_prefix_cache.rs` — compose oracle with `--mtp-k N`:

| Cell | Result |
|---|---|
| Greedy MTP cold | reused=0, generated >0 |
| Greedy MTP warm identical | reused=256, tokens match cold |
| Greedy MTP branch (divergent suffix) | reused=256, tokens differ from Italy |
| Sampled AR (temp 0.8, seed 7) | reuse 256/256, deterministic (same seed → same tokens) |
| JSON-Schema AR | reuse 256, output parses as `{"city": "Rome"}` |
| A20 soak (4× warm) | reuse 256 each, tokens match cold |
| A20 reset | `SlotEngine::reset` → reused=0 (allocation epoch bump drops radix) |
| Stats | admitted=11, reused_tokens=2304, prefix_hits=0 |

#### serve_harness chain (`fd3364798`)

`serve_harness.py --mode chain --no-spawn --port 11524 --sampling greedy --max-tokens 48 --max-think-tokens 0`:

```
cached_tokens: 0 → 89 → 192 → 265 → 346
```

#### Single-slot MTP load (`aeeda2259`)

`hipfire run --spec mtp` on gfx1101 logs:

```
MTP head loaded (sidecar /home/ghazni/.hipfire/models/qwen3.5-4b.mtp): n_embd=2560 vocab=248320
qwen35 MTP speculator enabled (compressed-serial, K=3)
```

The subsequent generate fails with an open-think-span validator (too few tokens for the think contract) — unrelated to sidecar discovery.

### Host tests

| Suite | Count | Status |
|---|---|---|
| `sidecar_probe_tests` (mtp_head) | 3 | ok |
| `mtp_sidecar_probe_tests` (loader) | 1 | ok |
| `cache_identity_digest_tests` (tokenizer) | 2 | ok |
| `lookup_with_pages_isolates_equal_length_prefixes` (prefix_index) | 1 | ok |
| `should_lookup_prefix` policy (serve_engine) | 3 | ok |
| `reused_tokens_from_plan` policy (serve_engine) | 4 | ok |

### Documentation

- `docs/CONFIG.md` — 7 new `serve.*` keys documented (prefix_cache, prefix_cache_max_bytes, structured_jump_forward, scheduler_overlap, max_batch_tokens, prefill_min_tokens, max_queue_bytes)
- `docs/SERVE.md` — experimental prefix-cache section (opt-in only, not admitted)
- `docs/specs/2026-09-05-serving-cache-scheduler-plan.md` — §0 progress table updated with wave 5 evidence

### P6 overlap

`serve.scheduler_overlap` is a registered config flag only (`hipfire-config`, default false). No host-prep overlap path exists in the engine — there is no CPU gap to measure. Per spec §10: leave off.

---

## What still exists as gaps

### Not admitted

- **No `admissions.yml` row.** Spec truth state remains **planned**. Completion of code ≠ admission, default-on, or product claim.
- **Defaults stay off:** `serve.prefix_cache`, `serve.structured_jump_forward`, `serve.scheduler_overlap` all `false`.

### Vision + prefix reuse (X2 / A18)

- `should_lookup_prefix(prefix_cache, has_visual) = prefix_cache && !has_visual` — vision requests skip the radix entirely.
- Needs a pixel/embedding/position identity oracle before vision+prefix can be enabled.
- No VL model oracle has been run on the slot engine with prefix cache.

### Slots tools / stop / logprobs

- `validate_generate_caps` in `crates/hipfire-daemon/src/slots.rs` still refuses:
  - `tools` (non-empty array) → rejected
  - `stop` (non-null) → rejected
  - `logprobs` / `top_logprobs` → rejected
- Spec: remove refusals only when complete behavior lands in the same change.

### A19 — HIP fault injection / fail-closed reuse

- Pool/engine-seam fault path landed: `HIPFIRE_FAULT_PREFIX_PUBLISH=1` fails the first publication; `test_serve_prefix_cache --fault-publish` proves honest miss + unchanged output + no cache-ref leak.
- Still open: fault injection below the HIP bridge itself (upload/launch/sync failure injection).

### A20 — Full lifecycle soak

- Landed: 4-hit warm soak, reset → cold (reused=0), **re-warm after reset reuses again** (allocation epoch drop + cache-lease release), post-reset output matches the original greedy run.
- Still open: repeated model *swap* with cache on/off (unload/reload path), idle spill/restore cycle on the slot engine, long soak with a page-level monotonic-leak counter.

### A10 — MTP page-crossing / partial-accept cache-visibility

- Landed: 96-token MTP generation (crossing generated-page boundaries with verify/repair writes) replays identically from cache on the warm run — candidate rows are provably not cache-visible; non-aligned prompts exercise SuffixRecompute against the capture-boundary guard.
- Still open: an explicit forced full-reject sweep (prose prompt driving τ→1.2) asserted against cache visibility.

### A13 — Mixed-load progress bound

- Landed: long cold prefill submitted before a short warm request — both complete; cold miss stays a miss (no false reuse).
- Still open: adversarial multi-domain mixed samplers run with per-request wait-bound assertions on GPU.

### P6 — Scheduler overlap

- Flag only, no implementation. Stays off until a measured host-gap exceeds fixture noise.

### `admissions.yml` / `ARCHITECTURE.md` / changelog / default-on

- Blocked until every advertised cell (including vision) has its oracle and product evidence.

---

## Commit history (this branch, newest first)

```
aeeda2259 fix(loader): probe stem .mtp sidecars like the slot engine
3803a0573 feat(slots): probe stem .mtp sidecars and compose prefix reuse
fd3364798 feat(slots): C1 content digests and GPU-verified prefix reuse
bffd2d9a9 docs: record wave 0–4 landing on serving-cache-scheduler
f3222d63d feat(slots): bounded wait on busy slots and optional jump-forward
e94572339 feat(slots): enable strict JSON Schema on the multi-slot path
6679deddf feat(runtime): bounded waiting-room policy for slot admission
9edd7e4c3 feat(grammar): conservative forced-token-run planner for JSON Schema
802dfca02 feat(slots): consult FairQueue for per-tick eligibility
df1efcf9c feat(slots): global row budget, prefill quantum, and per-slot COW isolation
4e2ed2423 feat(runtime): age-aware fair queue policy for slot scheduling
21f202f3d feat(slots): wire prefix cache lookup, COW barrier, and publication
d4faba38b feat(qwen35): hybrid-state checkpoint pool and resume planner
0f648d091 feat(runtime): token-keyed prefix radix index with domain isolation
3f024ec4c feat(slots): response_format contract, lossless token bytes, pre-sample mask seam
9d7110d46 feat(bench): multi-inflight client API and daemon slots benchmark
f6904c85b feat(serve): freeze cache/scheduler contracts and bound admission
b38e56b65 docs: serving cache scheduler spec and implementation plan
b6b4a6a30 feat(pages): generations, sealing, COW, leases, deferred reclaim in PagePool/SlotPool
6b07ef415 merge: fold feat/multislot-vision-mtp ViT layer-chunking into serving-cache-scheduler base
```

---

## Key files touched (this session)

| File | Change |
|---|---|
| `crates/hipfire-runtime/src/serve_contract.rs` | `sha256_len_prefixed`, `sha256_file`, `Digest` type |
| `crates/hipfire-runtime/src/tokenizer.rs` | `vocab_digest()`, `config_digest()`, digest tests |
| `crates/hipfire-runtime/src/hfq.rs` | `content_digest()` |
| `crates/hipfire-runtime/src/prefix_index.rs` | `lookup_with_pages()`, `Handle` collection in walk, equal-length isolation test |
| `crates/hipfire-runtime/Cargo.toml` | `sha2 = "0.10"` dep, `test_serve_prefix_cache` example registration |
| `crates/hipfire-runtime/examples/test_serve_prefix_cache.rs` | GPU compose oracle (greedy/sampled/grammar/soak/reset) |
| `crates/hipfire-arch-qwen35/src/mtp_head.rs` | `mtp_sidecar_candidates()`, `find_mtp_sidecar()`, sidecar probe tests |
| `crates/hipfire-arch-qwen35/src/serve_engine.rs` | C1 real digests, `lookup_with_pages` admit path, `find_mtp_sidecar` for slot engine + C1 sidecar identity, `reused_tokens_from_plan` exact-match fix, `published_handles` removal |
| `crates/hipfire-loader/src/lib.rs` | `find_mtp_sidecar` for single-slot loader fallbacks, probed-path error message, loader stem probe test |
| `docs/CONFIG.md` | 7 new `serve.*` keys |
| `docs/SERVE.md` | Experimental prefix-cache section |
| `docs/specs/2026-09-05-serving-cache-scheduler-plan.md` | §0 progress table with wave 5 evidence |
