<!-- SPDX-License-Identifier: Apache-2.0 -->

# SCS suite — serving-cache-scheduler black-box test suite

`scripts/scs_suite.py` is a 69-cell black-box test suite for the
`feat/serving-cache-scheduler` branch, driven entirely over the
OpenAI-compatible HTTP surface of a running `hipfire serve` container. It was
built to exercise every externally observable feature of the branch against
the live `qwen35-4b-mq4v1-multislot` container (4 slots, ctx 50000, prefix
cache on, `json-schema-strict-v1`, jump-forward on, `.mtp` + `.vl` sidecars
present, queue timeout 300 s).

It complements, and does not replace, the other verification layers:

| Layer | Tool | What it sees |
|---|---|---|
| Host unit tests | `cargo test --workspace --lib` | pool/index/fairness/grammar internals |
| GPU compose oracle | `test_serve_prefix_cache --mtp-k N` | engine-level reuse + fault cells (A19/A20) |
| Product harness | `scripts/serve_harness.py` | generic serve semantics, chain/battery/session |
| **This suite** | `scripts/scs_suite.py` | **the branch's user-facing contracts end to end**, incl. areas no other layer covers over HTTP (capability advertisement, 429/cancel/stall behavior, schema×cache composition, strict-JSON framing, error-status mapping) |

## Usage

```bash
# against the default container (127.0.0.1:8420)
python3 scripts/scs_suite.py                       # full suite, ~5-8 min
python3 scripts/scs_suite.py --list                # list all cells
python3 scripts/scs_suite.py --area C              # one area (A B C D E F G V H)
python3 scripts/scs_suite.py --cell E2 --cell D3   # specific cells
python3 scripts/scs_suite.py --json-out report.json --docker-name <container>
```

Exit code 0 iff no FAIL. A companion JSON report (per-cell status, evidence,
prompt md5s, health snapshot, container binary md5s) is written after every
cell, so a killed run still leaves a partial report. All request timeouts
scale with `--timeout-scale`; the default non-stream timeout is 150 s (the
daemon's queue timeout on this container is 300 s, so an admission park fails
the cell before the daemon would time it out).

**Exclusivity:** the suite assumes it is the only client of the serve
endpoint. Concurrent external traffic produces spurious failures in the
timing-sensitive cells (D1/D3/D6 parallelism and overload, B7/B8/C2/C5
determinism) and skews the VRAM headroom the crash-sensitive areas probe.
The runner waits for a resident model before starting, but not for a quiet
endpoint.

## Coverage map

| Area | Branch feature | Cells |
|---|---|---|
| A | capability advertisement (`/health` capabilities, per-model projection on `/v1/models`, `/stats`, Prometheus `/metrics`, 404/OPTIONS) | A1-A6 |
| B | generation + sampling contract (shapes, SSE frames, usage chunk, reasoning separation, coherence battery, greedy/seeded determinism, sampling params, finish=length) | B1-B10 |
| C | prefix cache correctness: cold/warm reuse, warm replay identity, A10 long-gen replay, branch + mid-page fork (W10-4), multi-turn chain, false-reuse guard, concurrent shared-prefix isolation (COW), soak stability, gen-param independence | C1-C9 |
| E | strict JSON Schema: valid output + determinism, streaming schema, framing-aware cursor (W10-2), enum/const, min/maxItems, required/closed, nested, unsupported keywords (P0-2), malformed keyword values, union type, bare object, structural errors, integer dead-end (P1-7), `json_object`/`text` refusal, schema×cache composition | E1-E16 |
| F | typed refusals on the multi-slot route: tools, stop, logprobs, tool-role messages, reasoning_effort/think-cap/budget, min_p range, `n`, `logit_bias` | F1-F9 |
| G | HTTP limits + error mapping: invalid JSON, unknown model, max_tokens bounds, 413 Content-Length, seed validation, empty messages, ctx overshoot | G1-G7 |
| V | vision×radix isolation (A18: repeated image request must show `cached_tokens 0`) | V1 |
| D | scheduler/admission: two-slot parallel generation + progressive service (wave-12), four-slot saturation, queue-full 429 + Retry-After, post-burst drain (pop_ready regression), cancel storm, long/short prefill interleave, queue timeout (env-dependent) | D1-D7 |
| H | lifecycle canaries: post-gauntlet health + long-prompt liveness, cold-cycle stability, idle eviction (env-dependent), model swap (out of scope for a single-model container) | H1-H4 |

Design invariants enforced across cells:

- **Reuse never changes output.** Every warm/replayed generation is
  byte-compared against its cold baseline under `temperature 0`
  (C1/C2/C3/C4/C5/C7/C8; B7/B8 for sampling determinism).
- **Eyeball-equivalent coherence.** Every long generation goes through an
  n-gram attractor detector (uniq / maxfreq / trigram thresholds mirroring
  `serve_harness.py`); genre prompts additionally assert on-topic substrings.
- **Prompt md5s recorded** for every fixture (in the JSON report), per the
  byte-identical-prompt rule.

## Results on qwen35-4b-mq4v1-multislot (2026-09-13)

Five full runs on freshly restarted engines (run2 `16:05`, run3 `16:28`,
run4 `16:40`, run5 `16:54`), plus one aborted run against a degraded engine.
Container image `local/hipfire-serve:serving-cache-scheduler`; binary md5s in
each JSON report (`.codeinsight+research/scs_suite_report*.json`).
Run-to-run stable results are separated from flaky ones; flaky ≠ less
important — several flaky failures are the most serious findings.

**Environment caveat discovered during the campaign:** this container's GPU
(12.87 GB) sits at **99% VRAM use on a freshly restarted, idle engine** — the
4×50000-ctx slot pool + weights fill the card by config. Any extra allocation
(vision tower activations, generation bursts) hits `hipMalloc: out of
memory`, the daemon poisons, and the container exits cleanly (code 0). Two of
five runs ended this way (run3 during the schema area, run5 at the vision
request). This is a deployment-sizing issue (the branch's own wave-9 notes
flag the same class) but the daemon's failure mode — poison + exit rather
than degrade/reject — is engine behavior worth fixing.

### Stable failures (reproduce every run)

| Cell | Symptom | Severity / class |
|---|---|---|
| E2 | Streaming strict-schema generation emits **all content tokens, then no terminal chunk, no `finish_reason`, no `data: [DONE]`** — the stream stalls and the server closes it 30-120 s later. Non-schema streams (B2/B3) are fine. OpenAI clients will hang until their own timeout. | **P1** — SSE contract break on the schema path |
| E3 | Valid schema + default thinking → typed **400 `grammar constraint reached an unsatisfiable state`** (fail-closed, no corruption). The framing-aware cursor (W10-2) is not effective for this model/route. | **P1** — advertised feature fails closed |
| E4 | Valid enum/const schema → same 400 after a 30-137 s generation burn. | **P1** — grammar dead-end on satisfiable schemas |
| E12 | `minItems > maxItems` → **500** (server_error), message proves the CLI validator detected it ("contradictory schema") | P2 — error-status mapping |
| E15 | `response_format {"type":"text"}` → **500**. `text` is the OpenAI *default* type; clients that send it explicitly get a 5xx | P2 — compat + status mapping |
| F8 | `n: 2` → **200** (silently accepted, single choice). `validate_generate_caps` (slots.rs) documents `n` as refused | P2 — refusal gate not reached on the serve path |
| F9 | `logit_bias` → **200**, silently ignored | P2 — same class |
| G3 | `max_tokens: -5` → **200** (silently treated as absent → uncapped); `0` and `400000` correctly 400 | P2 — validation gap |
| G5 | `seed: -1` → **500**; SERVE.md documents "400-style error" for negative seeds (fractional seeds correctly 400) | P2 — contract drift |
| G6 | `messages: []` → **500** server_error | P2 — client-input error as 5xx |
| G7 | ctx overshoot (`prompt + max_tokens > 50000`) → **500**; request-specific, server stays healthy; should be 4xx | P2 — status mapping |

### Flaky / recurring failures

| Cell | Symptom | Runs | Severity / class |
|---|---|---|---|
| B8 | Same seed + identical request → **different outputs** (sampled determinism is not reliable) | failed run3+run4+run5, passed run2 | **P1** — determinism contract |
| C5 | Multi-turn final-turn replay differs from the original in-context answer — hybrid-state resume is not bit-faithful | failed run3+run4+run5, passed run2 | **P1** — same class, checkpoint-resume path |
| C2 | Warm long-gen replay (A10) differs from cold — candidate rows or MTP retire decisions leak into the visible output | failed run4, passed run2/run3/run5 | **P1** — "reuse never changes output" |
| V1 | Vision request → `hipMalloc: out of memory` → daemon poisoned (all later requests 500) — VRAM overcommit (see env caveat) | run5; passed run2-4 | **P1** on this deployment — vision + full KV pool does not fit |
| C2/C6/H2 phantom reuse | Cold, never-seen prompts report `cached_tokens 640-768` (full pages of *another* same-shape passage's chain). Outputs stay keyword-correct ⇒ count/accounting lie rather than foreign-KV use, but it defeats `cached_tokens` semantics | run2 (C2, C6, H2); run1 C1 | **P1** — radix walk / reused accounting; could not be isolated to a single trigger (15 isolated probes came back clean; reproduces only mid-suite) |
| D1 | WARN: batched concurrent outputs differ from solo baselines (coherence + topical isolation held) | run4 | P3 — reduction-order sensitivity, observational |
| H2 | Post-flood warm miss: identical prompt re-run gets `cached_tokens 0` after the D3 overload | run4 | P2 — reuse reliability after admission pressure; needs `pool_free_pages` visibility to distinguish eviction from publication failure |
| serve exit | **Serve/daemon dies mid-traffic under VRAM exhaustion** — observed twice: run3 (clean exit 0 during schema area, after E2's stalled-stream abort, during E4's enum burn) and run5 (hipMalloc OOM at vision forward → daemon poison → exit 0). Neither trigger reproduces alone on a fresh engine (verified) — it needs the VRAM-pressure environment above. | run3, run5 | **P0** on this deployment — availability |

### The run1 degradation episode (first run, aborted)

The very first full run (against an engine with ~80 requests of prior
foreign traffic) showed a *progressive admission wedge*: first seeded/sampled
~700-token requests parked in the daemon wait queue for exactly
`queue_timeout_ms` (300 s) while greedy requests kept being served; within
minutes **every** ~700-token prompt parked, including previously-successful
ones, with `queue_depth 1` frozen and no admission/rejection log line.
Restarting the container cleared it completely; the identical requests then
passed. A fresh-engine full run (run2) did not reproduce the parking, but did
reproduce the phantom `cached_tokens` (above). Interpretation: a slow
state-corruption class in the admission/radix path that several hours of
traffic (including vision + schema requests from before the suite) can
trigger. The H1 long-prompt liveness canary exists to catch this class.

### What passed everywhere (stable green, worth knowing)

Capability advertisement and telemetry (A1-A6), non-stream/stream shapes and
usage chunks (B1-B3, B5), coherence battery (B6), greedy determinism on
short/medium generations (B7), sampling-param acceptance (B9), finish=length
(B10), the entire cold/warm/branch/fork/COW/soak cache block in 3 of 4 runs
(C1, C3, C4, C6-C9), strict-schema valid output + streaming-assembly +
min/maxItems + nested + all typed schema rejections (E1, E5-E11, E13, E14,
E16), the multi-slot refusal set (F1-F7), HTTP limits (G1, G2, G4), vision
skips the radix (V1), the whole scheduler block (D2-D6: flood 429s with
Retry-After, drain after burst, cancel storm, progressive service), and the
post-gauntlet canaries (H1, H2 cold-side).

## Notes for triage

- The JSON reports contain full per-cell evidence, all fixture prompt md5s,
  the `/health` snapshot (config actually in force), and container binary
  md5s — attach one to any issue.
- The determinism cluster (B8/C2/C5, D1-WARN) and the phantom-`cached_tokens`
  cluster point at the same subsurface: greedy/sampled bit-exactness under
  changing batch composition, and the radix walk's verification of matched
  pages. Engine-side, `EngineStats.pool_free_pages` / `reused_tokens` are not
  exposed over HTTP — adding a stats surface would let black-box runs
  distinguish eviction from publication failure (H2) and honest reuse from
  accounting lies (phantom).
- The run3/run5 daemon deaths are VRAM-exhaustion events on an overcommitted
  GPU (see env caveat): `rocm-smi --showmeminfo vram` shows 12.76/12.87 GB
  used on a freshly restarted idle engine. Shrinking `multi_slot_slots` ×
  `multi_slot_ctx`, disabling the vision sidecar, or enabling the OOM guard
  for this discrete-GPU deployment are the deployment-side mitigations; the
  engine-side gap is that hipMalloc failure poisons the daemon and exits
  instead of failing the offending request and continuing.
- The run1 degradation episode (admission parking) was only observed on an
  engine that had served hours of foreign traffic first; if it recurs,
  capture `/stats` (`queue_depth` frozen at 1) plus `docker logs` — the
  signature is requests parked in the daemon wait queue with no admission
  and no rejection until the queue timeout.
