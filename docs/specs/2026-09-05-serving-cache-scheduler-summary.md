<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving cache scheduler — work summary and remaining gaps

- **Date:** 2026-09-05
- **Branch:** `feat/serving-cache-scheduler` (pushed to `origin` at `ghazni101/hipfire`)
- **Tip:** `aeeda2259` `fix(loader): probe stem .mtp sidecars like the slot engine`
- **Spec:** [2026-09-05-serving-cache-scheduler-spec.md](2026-09-05-serving-cache-scheduler-spec.md)
- **Plan:** [2026-09-05-serving-cache-scheduler-plan.md](2026-09-05-serving-cache-scheduler-plan.md)
- **GPU:** gfx1101 (AMD Radeon RX 7700 XT), HIP 7.15, ROCm 10 container (`local/rocm-base:10.0.0`)
- **Model:** `qwen3.5-4b.mq4v2.hfq` + `qwen3.5-4b.mtp` sidecar

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

- No fault-injection path for HIP upload/launch/sync failure.
- No poisoned-page reuse guard tested.

### A20 — Full lifecycle soak

- Only reset + 4-hit soak landed. Still missing:
  - Repeated model swap with cache on/off
  - Idle spill/restore cycle
  - Long soak with monotonic resource leak check

### A10 — MTP page-crossing / partial-accept cache-visibility

- MTP full reject / partial / full accept not tested for cache-visibility of candidate rows.
- Page-crossing accept not tested on GPU.

### A13 — Mixed-load progress bound

- Long cold prompt + continuous short warm requests + mixed samplers not run on GPU.

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
