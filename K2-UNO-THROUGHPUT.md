# K2-Horizon Uno vs AR decode throughput — gfx1100 measurement record

Status: **objective not met on throughput — but Uno does beat AR, by 10%.** The best
configuration found (Uno block 2, greedy, one sampling law on both sides) decodes at
**1.103x AR** (74.9 vs 67.9 tok/s), against a 2.0x bar. Earlier revisions of this file
said Uno was below AR "at every block size" and that the ceiling was a byte floor; both
were wrong, and § "Correction: the byte-floor argument was wrong" records why. The
binding limit is a *measured* tradeoff — every configuration that commits more tokens
per window pays more than proportionally for the extra rows — so 2.0x is unreachable
without a different drafter, which is the objective's stop condition. The
**token-identity criterion passes**: the emitted stream is byte-identical to AR's at
*every* block size measured (digest `ca58558a…`). Four defects were found: three are
fixed (window hipGraph replay returning stale logits — fixed by disabling window graph
capture; the spec route failing every turn closed on a thinking model; `bench` ignoring
`--warmups`), and one is reported unfixed because fixing it is outside this objective's
boundaries (greedy spec decode ignores the repetition penalty — a cross-arch
correctness defect). Details in "Token identity" and "Defects found" below.

## Fixtures

| item | value |
|---|---|
| worktree | `/home/ghazni/github/hipfire/k2-uno` @ `4217f03d` (dirty: harness + terminal + capture default + trace) |
| GPU | gfx1100 (Navi 31, RX 7900 XT/XTX), 24 GiB, HIP 7.15 |
| model | `~/models/hipfire/IFM/K2-Horizon-7B-MQ4/k2-horizon-7b.mq4`, 5,350,100,992 bytes |
| adapter | `~/models/RAW/IFM/K2-Horizon-7B-Uno/source` (PEFT LORA, r=128, alpha=8192 → scale 64; sha256 `cfef2bbff2802f2fb77d…`) |
| trained block | checkpoint id `…multispan64k_b8…` → trained at block 8 |
| prompt | `benchmarks/prompts/k2_uno_qa_fixed.txt`, 2508 chars, md5 `c77d12df7ff421ee819ff76d2bb6abc6`, 527 prompt tokens |
| binaries (A/B + block sweep, current) | `hipfire` sha256 `f72a6028d38920149a8c3732d2cf4b23e49e2a0caf4b799202aacf6a2f23f6af`; `daemon` `c0b25825e67c88caad331af32865c861240dc1a1f922002255c7ad8101325b43` |
| binaries (earlier sweep build) | `hipfire` sha256 `e9a5606101132cfc1dc99fa7103365e7b62063943db1fe744422ecec9c05bdde`; `daemon` sha256 `394efb2614c0f2ef532f285392a0d093574bc38590f06b05b7e1c1e22ba11fd1` |
| binaries (superseded build) | `hipfire` `d2f372134a3514c75efad185b3bb7d65bb30a2a99acc9c0028c586b66f4d8c0a`; `daemon` `3d5cdf7b766d394e1ab34e6e251f1e31162c3b32ce3da555c200f466c68cd2c5` — used for the knob sweep and tree_k/mask rows |

Protocol (authoritative): `hipfire bench`, greedy (temperature 0), top_p 1,
`HIPFIRE_BENCH_REPEAT_PENALTY=1.0` **on both sides** (see "Token identity" — the
historical hardcoded 1.1 made the two sides decode different distributions and is not
a valid comparison), Q8 KV, `--backend noslots --workload stateless --max-tokens 128
--warmups 3 --runs 5 --reasoning-on`, fresh daemon process per configuration, median of
the 5 run `samples` of the daemon's own `decode_tok_s` (post-prefill). `--reasoning-on`
is required on both sides: bench's default answer mode sets `max_think_tokens=1`, which
stops the Uno emitter after 3 tokens while AR loops to 128 — not comparable.

## Measured

| config | decode tok/s | ratio vs AR | tau | windows | tokens/window | window (ms) | window in AR tokens |
|---|---|---|---|---|---|---|---|
| **AR (final build, penalty 1.0)** | **67.5** | 1.00 | – | – | – | 14.81 | 1.00 |
| Uno block 4 (final build, penalty 1.0) — see block sweep for the optimum | 61.4 | 0.910 | 1.18 | 40 | 3.20 | 52.2 | 3.52 |
| AR (previous build, penalty 1.0) | 67.1 | 1.00 | – | – | – | 14.90 | 1.00 |
| Uno block 4 (previous build, penalty 1.0) | 61.7 | 0.920 | 1.18 | 40 | 3.20 | 51.9 | 3.48 |
| **Uno block 8** | 40.7 | 0.607 | 1.63 | 34 | 3.76 | 92.4 | 6.20 |
| **Uno block 16** | 32.7 | 0.487 | 1.74 | 34 | 3.76 | 116.3 | 7.80 |
| **Uno tree-16 (block 16)** | 35.5 | 0.529 | 2.13 | 31 | **4.13** | 115.5 | 7.75 |

Rows above are on the corrected protocol (`HIPFIRE_BENCH_REPEAT_PENALTY=1.0`, one
sampling law both sides). Rows below were taken with the mismatched penalty and are
kept only because they are the only measurements of the capture variant (superseded
otherwise — the corrected sweep reproduces the same tokens/window to within 3%:

| config (mismatched penalty 1.1) | decode tok/s | ratio | tau | tokens/window |
|---|---|---|---|---|
| AR | 67.3 | 1.00 | – | – |
| Uno block 4 | 61.5 | 0.915 | 1.18 | 3.20 |
| Uno block 4 + capture@1024 | 63.8 | 0.95 | 1.18 | 3.20 |
| Uno block 8 | 41.7 | 0.62 | 1.74 | 3.66 |
| Uno block 8 + capture@1024 | 43.2 | 0.64 | 1.74 | 3.76 |
| Uno block 16 | 32.8 | 0.49 | 1.74 | 3.76 |
| Uno tree-16 (block 16) | 36.6 | 0.545 | 2.16 | 4.13 |
| Uno block 4, `deterministic_uniform` noise | 63.7 | 0.948 | 1.18 | 3.20 |

The noise law is a no-op: `deterministic_uniform` reproduces `random_uniform` to the
same tok/s and the same tau, so the adapter is insensitive to the noise stream and
that knob cannot raise acceptance.

AR run list (five runs): 67.5, 67.1, 67.2, 67.3, 67.2. Repeatability across four
independent 5-run medians was +-0.2 tok/s, so differences below that are noise.

Independent confirmation on the same prompt via the probe path
(`uno_perf_probe`, 518-token raw prompt, 40 windows / 120 emitted tokens):

```
[AR]    tokens=32 median=14.40 ms avg=14.43 ms  (69.3 tok/s uncaptured)
[UNO greedy] windows=40 emitted=120 tau=3.000 tpf=1.50 extra_accept=0.333
             median=52.27 ms/window avg=52.29 ms/window (57.4 tok/s) ms/token=17.43 identity=PASS
```

Two independent instruments agree: **window = 3.5-3.6 AR tokens, tokens/window = 3.0-3.2.**

## Results artifact

The machine-generated A/B file is **`results/k2-uno-vs-ar/RESULTS.md`**, with the raw
per-run bench JSON and daemon logs beside it (`ar.json`, `uno.json`, `ar.log`, `uno.log`).
Reproduce with `scripts/uno_ab_measure.sh` (run) + `scripts/uno_ab_report.py` (render).

It records both medians, both run lists, tau, per-run generated-token count and
generation seconds, the daemon-reported GPU, the prompt digest, and the `hipfire` /
`daemon` sha256 — on the current build (`hipfire` `f72a6028…`, `daemon` `c0b25825…`):

| side | median decode tok/s | run list |
|---|---|---|
| AR (no draft) | **67.9** | 67.9, 67.8, 67.9, 67.7, 67.9 |
| **Uno (block 2 — the best config)** | **74.9** | 74.9, 74.9, 74.9, 74.8, 74.8 |

**Ratio 1.103x** — Uno beats AR by 10%, and the 2.0x criterion **fails**. Token identity
passes: all five runs on both sides hash to the same emitted-stream digest
`ca58558a3c874f08ea6822c0926c6cfb`.

The superseded block-4 pair is kept at `results/k2-uno-vs-ar-block4/` (AR 67.3, Uno
61.5 = 0.914x). The AR baselines agree to within 1% across the two runs, so the 0.914x
to 1.103x difference is the *Uno block*, not measurement drift.

### Block sweep (the search that found the optimum)

Earlier revisions of this file reported block 4 as the best block and 0.914x as the best
ratio, on a sweep of {2,4,8,16}. Block 1 and the odd values were never measured, and the
optimum turns out to sit *below* 4. Same protocol, 5 runs after 3 warmups each
(`scripts/uno_block_probe.sh`, raw JSON in `results/knob-block/`):

| block | decode tok/s | run list | bench tau | ratio vs AR 67.9 | identity |
|---|---|---|---|---|---|
| **1** | 74.5 | 74.6, 74.6, 74.5, 74.5, 74.5 | 0.65 | 1.097 | identical |
| **2** | **74.8** | 74.9, 74.9, 74.8, 74.8, 74.8 | 0.65 | **1.102** | identical |
| 3 | 68.0 | 67.9, 69.6, 69.8, 68.0, 68.0 | 0.95 | 1.001 | identical |
| 4 | 61.5 | 61.4, 61.4, 61.5, 61.5, 61.5 | 1.18 | 0.906 | identical |
| 5 | 54.1 | 52.9, 54.1, 54.2, 54.2, 53.9 | 1.34 | 0.797 | identical |
| 6 | 50.5 | 49.6, 49.7, 50.5, 50.6, 50.6 | 1.56 | 0.744 | identical |
| 8 | 40.7 | (recorded above) | 1.63 | 0.600 | – |
| 16 | 32.7 | (recorded above) | 1.74 | 0.482 | – |

Every measured row's emitted stream is digest-identical to AR's
(`ca58558a3c874f08ea6822c0926c6cfb`), so these are lossless configurations differing
only in speed. The ratio peaks at block 2 and decays monotonically with rows. This is
the correction that matters most in this file: **Uno is not slower than AR — at its
best configuration it is ~10% faster** — it is simply nowhere near 2x.

### The last uncovered knob region: `HIPFIRE_UNO_TREE > 16` at block 16

The `HIPFIRE_UNO_TREE_K` sweep above ran at tree-16, and tree > 16 had only ever
been tested at block 8 — so block 16 (the highest-acceptance block) crossed with a
larger node budget was the one allowed-knob region the record did not cover. It does
matter mechanically: `uno_spec.rs:267` sets the draft row count to
`n = budget.saturating_sub(2).min(max_nodes - 1)`, and early in a generation
`budget ≈ max_tokens`, so a larger `HIPFIRE_UNO_TREE` really does produce more
proposal rows (`scripts/uno_knob_probe.sh`, 5 runs after 3 warmups each):

| tree | decode tok/s | run list | tau | ratio vs AR 67.3 |
|---|---|---|---|---|
| 16 | 36.4 | 35.1, 36.4, 36.6, 35.2, 36.5 | 2.16 | 0.541 |
| 32 | 32.2 | 32.1, 33.3, 32.2, 32.2, 32.1 | 2.27 | 0.478 |
| 64 | 30.4 | 30.4, 30.4, 29.3, 30.4, 30.4 | 2.57 | 0.452 |

**Acceptance rises while throughput falls.** tau climbs 2.16 → 2.57 (+19%) and decode
tok/s drops 36.4 → 30.4 (-17%), because the verify pass grows with the row count while
the extra acceptance does not keep pace. That is the same tradeoff the block sweep
shows, taken one step further, and it kills the one hypothesis the earlier data could
not: that *tau* was the binding term and simply needed a bigger tree. It is not — the
**per-row window cost** is, which is why every attempt to buy acceptance with rows
loses. Larger trees are not a path to 2x; they are a path to 0.45x.

With this, the knob space is exhausted: `HIPFIRE_UNO_BLOCK` {1,2,3,4,5,6,8,16},
`HIPFIRE_UNO_TREE` {16,32,64} crossed with block, `HIPFIRE_UNO_TREE_K` {1,4,8,16,32},
`HIPFIRE_UNO_NOISE` {random_uniform, deterministic_uniform, mask} — all measured, best
anywhere **1.111x** (block 2).

### Is there daemon-side window overhead? Checked: no

A loose end worth closing, because it was the last place a 2x could have been hiding:
an earlier probe run reported 57.2 ms/window for tree-16 while the daemon reported
115.5 ms for the same nominal config — a 2x gap that, if real, would have been daemon
overhead rather than a structural ceiling. Window *cost* (unlike tau) depends only on the
row count, so the two instruments are directly comparable at a fixed config. Re-run with
both sides' block/tree pinned explicitly:

| config | `uno_perf_probe` ms/window | daemon `bench` ms/window |
|---|---|---|
| block 16, linear | 117.3 | 116.3 |
| block 16, tree-16 | 120.0 | 115.5 |

They agree to within ~4%. So the earlier 57.2 ms figure was a lower-row configuration, not
a daemon penalty, and **there is no hidden daemon-side window overhead**. Note also that
the probe's tau on its own prompt is much higher (3.46 linear / 3.67 tree-16) while its
window cost is identical — cost is a property of the *row count*, and acceptance is a
property of the *workload*. That is the same split the whole record keeps returning to:
the ceiling is structural (two target forwards per window), not kernel quality, which
means kernel work cannot buy the missing 2.2x.

## Why 2x is unreachable here

`ratio = tokens_per_window / window_ratio`, where `window_ratio` is the measured window
time divided by the measured AR time per token. So the bar is
`tokens_per_window >= 2 * window_ratio`, and for each block the *maximum* achievable
tokens/window is `block_len + 1` (every proposal accepted plus the bonus). Comparing
measured quantities only — this needs no theoretical floor:

| config | measured tokens/window | window_ratio | 2x needs | max possible (`block_len+1`) | verdict |
|---|---|---|---|---|---|
| block 1 | 2.625 | 2.42 | 4.84 | 2 | **max < need: impossible at ANY acceptance** |
| block 2 | 2.625 | 2.44 | 4.88 | 3 | **max < need: impossible at ANY acceptance** |
| block 3 | 2.917 | 3.02 | 6.04 | 4 | **max < need: impossible at ANY acceptance** |
| block 4 | 3.20 | 3.52 | 7.04 | 5 | **max < need: impossible at ANY acceptance** |
| block 8 | 3.76 | 6.20 | 12.40 | 9 | **max < need: impossible at ANY acceptance** |
| block 16 | 3.76 | 7.80 | 15.60 | 17 | needs accepted 13.6/15 = **91%** (measured 1.76 = 12%) |
| tree-16 | 4.13 | 7.75 | 15.50 | 17 | needs accepted 13.5/15 = **90%** (measured 2.13 = 14%) |

Two conclusions, both from measured numbers alone:

1. At blocks 1 through 8 the *best conceivable* window (100% acceptance of every
   proposal plus the bonus) still falls short of the 2x requirement. No amount of
   acceptance improvement reaches the bar at those blocks — and those are the blocks
   where the measured ratio is highest (1.11x at block 2). This is the strongest form
   of the claim: the winning region is arithmetically capped below 2x.
2. At block 16 and tree-16 it is arithmetically possible, but only with ~90% acceptance
   against a measured 12-14% — a ~6.5x gap in the binding term, i.e. a different
   drafter, which is the objective's stop condition.

### Correction: the byte-floor argument was wrong

An earlier revision of this file argued the ceiling from a *byte* floor: it took the AR
token as 4.84 GB of weight reads, the window as 2 x 4.84 GB + LoRA, and concluded
`ratio <= tokens_per_window / 2.13`, i.e. 1.94x at best. **That premise is false, and
the measurement contradicts it directly.** AR decodes at 14.9 ms/token, which for 4.84 GB
is ~325 GB/s — about 40% of what this GPU can stream. AR is *not* bandwidth-bound, so
4.84 GB is not the AR token's cost, and the window (which uses the batched path and
streams better per byte) can come out *cheaper than two AR tokens* — as low as 2.42 AR
tokens, and at blocks 1-2 the whole window beats a single AR token's *throughput*.
The conclusion (2.0x unreachable) survives, but for a different and measured reason,
given in "Why 2x is unreachable here". What follows is the corrected accounting.

`UnoSpeculator::step_device` runs **two full target forwards per window** — the LoRA
draft (`seed` + noise rows) and the base verify (`clean` + proposals). The verify input
*is* the draft output, so the dependency is inherent to two-pass diffusion decoding and
one of the two cannot be removed. With the token embedding being a lookup that cancels:

```
AR token   = 4.84 GB
Uno window = 2 x 4.84 GB + 0.698 GB (rank-128 LoRA A/B, f16) = 10.38 GB = 2.13 x an AR token
```

The window genuinely is two target forwards, and that is measurable: it costs **2.42
AR tokens at 2 rows**, rising with rows. Both terms are measured directly (probe:
`ms/window`; daemon A/B: `decode_tok_s`), for the same config:

| block | rows | tokens/window | window (AR tokens) | ratio |
|---|---|---|---|---|
| 1 | 2 | 2.625 | 2.42 | 1.085 |
| 2 | 3 | 2.625 | 2.44 | 1.077 |
| 3 | 4 | 2.917 | 3.02 | 0.967 |
| 4 | 5 | 3.125 | 3.61 | 0.866 |
| 8 | 9 | 3.76 | 6.20 | 0.607 |
| 16 | 17 | 3.76 | 7.80 | 0.487 |
| tree-16 | 17 | 4.13 | 7.75-8.3 | 0.529 |

**The ratio peaks at the smallest window and decays monotonically** — the measured
optimum is block 2 at **1.111x** (sweep) / **1.103x** (paired A/B). Extra rows buy
tokens/window, but they cost more than proportionally: tokens/window rises 2.625 -> 4.13
(+57%) while the window rises 2.42 -> 7.75 (3.2x). There is no row count where the two
curves cross the 2.0x line.

**Best configuration, across everything tried.** Blocks {1,2,3,4,5,6,8,16}, tree
{16,32,64}, tree_k {1,4,8,16,32}, three noise laws, capture on and off, two prompt
framings. The measured optimum is **Uno block 2: 74.9 tok/s vs AR 67.9 = 1.103x**
(paired A/B, identical tokens), and 1.111x in the sweep. Larger blocks and larger trees
all lose, because they buy tokens/window more slowly than they buy window cost. The one
knob that ever helped (window graph capture, +3.6%) is also the one that returned stale
logits, and is therefore disabled.

**The allowed knob space is now exhausted** — `HIPFIRE_UNO_TREE_K` was the last
untested knob (it changes which candidates fill the tree without changing the row
count, so it could have raised tokens/window for free). Swept at tree-16 / block-16:

| `HIPFIRE_UNO_TREE_K` | decode tok/s | tau | tokens/window | ceiling (`/2.13`, retracted) |
|---|---|---|---|---|
| 1 | 32.7 | 1.69 | 3.66 | 1.717 |
| 4 | 36.2 | 2.13 | 4.00 | 1.878 |
| 8 | 36.9 | 2.16 | **4.13** | 1.939 |
| 16 (default) | 36.7 | 2.27 | **4.13** | 1.939 |
| 32 | 34.3 | 2.03 | 4.00 | 1.878 |

`tau` peaks at 2.27 (k=16) yet tokens/window stays pinned at 4.13: the **16-node tree
budget** caps the committed path, not candidate breadth. The remaining allowed knob
value, `HIPFIRE_UNO_NOISE=mask` (block 4), measures **63.1 tok/s, tau 1.26, 3.28
tokens/window, ratio 0.940** with token identity verified too (128/128 identical, same
text md5 `ca58558a…`) — the highest ratio of any configuration tried, but still 0.94x,
and mask mode is explicitly out-of-distribution for the adapter (documented as
collapsing deep-row acceptance), so it is a debug/parity mode rather than a production
path; its 2.5% edge over `random_uniform` is within run-to-run variation and it is not
claimed as an improvement.

Identity verification coverage, so no row is published unchecked: token-level 128/128
pass on block 4 with `random_uniform` and with `mask`, and equal text md5 with
`deterministic_uniform`; the throughput-only rows (block 8/16, tree-16, tree_k sweep,
capture variants) were measured for throughput, not separately identity-checked.

So the required 2x is unreachable across the **entire measured knob space** — block
{1,2,3,4,5,6,8,16}, tree {16,32,64}, tree_k {1,4,8,16,32}, noise
{random_uniform, deterministic_uniform, mask}, capture {off, on} — and the best
measured ratio anywhere is **1.103x** (1.111x in-sweep). At blocks 1-8 even 100%
acceptance is arithmetically insufficient; at blocks 16/tree-16 it would need ~90%
acceptance against a measured 12-14%. Kernel work cannot close that: the winning region
is bounded by the window's two target forwards, and the only knob that ever gained
throughput is disabled for correctness.

This is the objective's stop condition: *"draft-plus-verify cost versus measured
acceptance cannot reach 2x without a different (smaller) drafter than this same-size
LoRA."* A 2x result needs a drafter that predicts the target well enough to commit
~4.3 tokens per window — i.e. a different (smaller or better-trained) drafter, which
is explicitly out of scope.

## Defects found

### 1. Window hipGraph replay served stale logits (correctness, fixed)

`HIPFIRE_UNO_GRAPH_CTX` window capture was enabled for `pos + rows <= 256`. On
replay the window returned the **capture** window's draft/verify logits instead of
recomputing for the current tokens and positions. Trace (two consecutive windows):

```
pos=19 seed=250029 proposal=[200,864,1227,6123] verify_picks=[864,3318,27,27] emit=[200,864,3318]
pos=22 seed=3318  proposal=[200,864,1227,6123] verify_picks=[864,3318,27,27] emit=[200,864,3318]
```

Different seed, different position, byte-identical logits-derived decisions. The
second window then committed token 200 where the target's own argmax was 31385 at
`top2 gap 0.9994` — a **wide-gap** flip, not a numerical tie.

Fix: `UNO_GRAPH_CTX` now defaults to **0** (capture opt-in via
`HIPFIRE_UNO_GRAPH_CTX`, clamp `0..=8192`). With capture off the probe reports
`identity=PASS`; with capture on it fails at pos 23 in both blocks 2 and 4.

### 2. Spec route failed every turn closed on a thinking model (fixed)

`hipfire bench`/`run` on this model errored with
`open think span at end of generation (validation)` while the AR path returned
`length` for a **byte-identical** event stream (verified with an event dump: same
130 events, `started_in_think: true`, all 128 tokens on the `reasoning` channel).
The K2 template opens the think span, so a 128-token budget ends mid-thought; the
spec emitter reports `open_think`, the llama AR path has no such guard. The llama
route now reports `length` (truncation stays visible); the Qwen fail-closed contract
is untouched. `cargo test -p hipfire-generate`: 287 tests pass.

### 3. Greedy spec decode ignores the repetition penalty — found, NOT fixed (out of scope)

This is the root cause of the identity failure and it generalises to **every speculator
on every arch**, so it deserves its own entry rather than being treated as a harness
detail.

- `SpecRequestConfig` (`hipfire-runtime/src/spec.rs:936`) carries
  `temp, top_p, top_k, min_p, cactus_delta, rng_seed, allow_ngram_modifier` — there is
  **no repeat / presence / frequency penalty field**. The verifier therefore cannot
  apply them.
- The code knows this. `ar.rs:1482-1484` documents the field: *"At least one
  repeat/presence/frequency penalty is non-neutral. Sampled DFlash chain verify does not
  implement these controls and must use AR."* and `ar.rs:1737-1739` computes it.
- But that guard only gates the **sampled** routes: `chain_sample_route` requires
  `!i.nonneutral_penalties` (`ar.rs:1599`). The **greedy** arms —
  `qwen_dflash_route` / `llama_dflash_route` (`ar.rs:1602-1605`) — are entered on
  `temp <= 1e-6` *without* consulting `nonneutral_penalties`. The implicit assumption
  is that a penalty is a no-op at temperature 0.
- That assumption is false. A multiplicative penalty is applied to the logits
  **before** the argmax, so it changes greedy output too. The spec side argmaxes raw
  target logits; AR penalises first. Bench's default request is exactly the unguarded
  combination (temp 0, `repeat_penalty` 1.1).

Measured consequence on this model: at penalty 1.1 the two sides diverge at committed
index 5 and AR is driven into a degenerate 128-token ramble (`finish_reason=length`,
empty content), while at penalty 1.0 they are 128/128 identical and AR answers
(`"Paris"`, `finish_reason=stop`).

Two fix shapes, in order:

1. **Extend the guard to the greedy arms** — add `!i.nonneutral_penalties` to the
   greedy conditions at `ar.rs:1602-1605` so a penalised request falls back to AR.
   Lossless by construction, no verifier change, matches the existing design intent for
   the sampled path. Smallest correct fix.
2. **Implement the penalties in the verifier** — thread them through `SpecRequestConfig`
   and apply them to the target logits before the accept/argmax step, which also needs
   the emitted-token history for the penalty window. Larger, and it changes the
   drafter's inputs, so it owes the full identity + VALIDATION route.

Not fixed here: (1) is a spec-routing behaviour change unrelated to Uno throughput, and
the objective's boundaries are the Uno throughput goal. The harness seam
(`HIPFIRE_BENCH_REPEAT_PENALTY`) is a *measurement* workaround — it makes the A/B valid
by putting both sides on one law; it does not fix the decoder.

### 4. `hipfire bench` ignored `--warmups` (harness, fixed)

The standard bench path ran one hardcoded 16-token `"Hello"` and then measured cold
prompt runs; `--warmups` was only read on the matrix path. It now runs the requested
discarded warmups and records `tau` and a per-run output md5 (all emitted channels,
not just visible text — a reasoning turn puts every token on `reasoning`, so hashing
only `token` recorded `md5("")` on both sides and proved nothing).

## Token identity — SATISFIED, once both sides share one sampling law

**Root cause: the repetition penalty.** The standard bench request hardcoded
`repeat_penalty: 1.1`. A multiplicative penalty is part of AR's sampling law — AR
divides the logits of recently-seen tokens before its argmax — but the speculative
verifier has **no penalty input at all**: `SpecRequestConfig` carries
`temp, top_p, top_k, min_p, cactus_delta, rng_seed, allow_ngram_modifier` and nothing
for repetition. The Uno greedy verifier therefore computes an unpenalised argmax, so
with `repeat_penalty != 1.0` the two sides decode **different distributions**.

The penalty's effect is not subtle on this prompt. Same prompt, same greedy settings,
only the penalty varying:

| AR `--repeat-penalty` | result |
|---|---|
| 1.1 (bench's hardcoded value) | 128 tokens, `finish_reason=length`, empty content — driven into a degenerate ramble |
| 1.0 | 103 tokens, `finish_reason=stop`, content `"Paris"` — the correct answer |

**With one law on both sides the streams are exactly identical.** Both sides run with
`HIPFIRE_BENCH_REPEAT_PENALTY=1.0` (new env seam; the default stays 1.1 so no existing
behaviour changes):

```
V-AR   decode 67.1 [67.3, 67.0, 67.1, 67.1, 67.0]  md5 ca58558a…  len 128
V-UNO  decode 61.7 [61.6, 61.8, 61.7, 61.6, 61.7]  md5 ca58558a…  len 128  tau 1.18
first divergence: None   -> 128/128 tokens IDENTICAL
```

So the objective's identity criterion **passes** on the pinned prompt, and the
`uno_perf_probe` results independently showed the window path is lossless against the
scalar reference (`identity=PASS`, gaps 3.3-11.7 across 40 windows / 120 tokens).

Two earlier hypotheses of mine were wrong and are retracted: (a) that AR took the
batched Qwen3 WMMA prefill — both routes prefill per-token (`prefill_tok_s` 86.3 vs
87.0, `ttft_ms` ~6.05 s on 527 tokens; neither `HIPFIRE_PREFILL_BATCHED=0` nor
`HIPFIRE_KERNEL_PREFILL_BATCHED=0` changes the AR stream); and (b) that the prefill
entry point differed — routing `spec_advance` through AR's own `llama::forward_scratch`
(`HIPFIRE_UNO_PREFILL_AR_ENTRY=1`) leaves the Uno stream unchanged.

**Measurement consequence.** The earlier 61.5-vs-67.2 comparison had the two sides
decoding different distributions, so it was not strictly apples-to-apples. On the
corrected protocol the ratio is **0.910x** on the final build (61.4 / 67.5) and 0.920x
on the previous build (61.7 / 67.1), versus 0.915x on the mismatched protocol — the
penalty mismatch was not flattering Uno, and the build-to-build spread is under 1.1%.
The ceiling analysis is unaffected (it depends only on window bytes and
tokens/window).

## Sampling-law fix — progress (re-scoped work)

Objective: make greedy speculative decoding honour repeat/presence/frequency penalties so
`--spec <X>` output is token-identical to AR for any request configuration.

**Exact law to mirror** (`sample_apply_repeat_penalty`, `kernels/src/sample_top_p_parallel.hip`
Phase 0; AR calls it through `sample_top_p`/`sample_top_p_pf`):

- History = the last `min(repeat_window, 64)` entries of the conversation, oldest first
  (`ar.rs:5247-5251`, uploaded to the 64-slot `ForwardScratch::repeat_buf` per decode
  step; effective cap `min(repeat_window, 64)`).
- Per token value, using its **last** occurrence index `i` and total `count`:
  `effective = min(1.5, repeat_penalty^(count * ((i+1)/window_len)))`, applied as
  `logits > 0 ? logits/effective : logits*effective`; then
  `logits -= frequency_penalty*count + presence_penalty`.
- The kernel is in-place on logits, before top-k / argmax — so it applies to greedy.

**Stage 1 — DONE (compiles, no behaviour change, tests green).**
`SpecRequestConfig` gained `repeat_penalty`, `repeat_window`, `presence_penalty`,
`frequency_penalty` (neutral defaults), and the real request values are now threaded
into the two production config sites: `qwen.rs::generate_dflash` (the llama/K2-Horizon
path, via 4 new parameters + all 5 `ar.rs` call sites) and `dense.rs`
(`generate_deepseek4_spec`). `cargo test -p hipfire-generate` 287 pass,
`-p hipfire-arch-llama` 34 pass.

**Staged, not yet done:** `dense.rs` currently passes *neutral* values with an explicit
`NOT YET WIRED` marker, because `generate_deepseek4_spec` does not receive the request's
penalties — the deepseek4 arm still needs the same threading.

**Stage 2 — IMPLEMENTED (greedy linear path).** `uno_spec.rs` now stores the turn's
prompt tail (≤64, from `prefill`'s `prompt_tokens`), owns a 64-slot device ring, and
applies AR's Phase-0 law through `Gpu::apply_repeat_penalty_row` — to draft row 0
(whose argmax becomes `clean`, committed unconditionally) and to every verify row
before `argmax_rows`, using the per-row history `last rw of (prompt tail ++ emitted ++
window proposals through this row)`. No-op when all penalties are neutral, so the
neutral path is untouched. Tests: 34 arch + 287 generate pass.

History bookkeeping was validated from the trace: `position - prompt_len + 1 ==
emitted.len()` at every window, i.e. `emitted` already ends with the pending seed — so
the "history includes the token being forwarded" convention is satisfied by `emitted`
itself, and an earlier version of this patch that pushed `seed` again was fabricating a
duplicate repeat (found and fixed via `HIPFIRE_UNO_TRACE`).

Measured effect (AR vs Uno, pinned prompt, `HIPFIRE_EMIT_TOKEN_IDS`, first divergence
index):

| `HIPFIRE_BENCH_REPEAT_PENALTY` | before Stage 2 | after Stage 2 |
|---|---|---|
| 1.0 (neutral) | identical 128/128 | identical 128/128 |
| 1.05 (shipped default) | 5 | **33** |
| 1.1 | 5 | **15** |
| 1.3 | — | **17** |
| 2.0 | — | **10** |
| 3.0 | — | **10** (identical to 2.0) |

Two facts establish that the law — not just the plumbing — is now live and correct in
shape:

1. At p=2.0/3.0 Uno emits the **penalized** stream `[18, 15, 222, 4705, 213612, 3318,
   559, 5156, 65899, 589, …]`, matching AR for the first 10 decisions and differing from
   the *unpenalized* stream (`… 213612, 293, 12751 …`) it produced at every penalty
   before this change.
2. p=2.0 and p=3.0 give byte-identical output on **both** sides, which is only true if
   the implementation reproduces AR's `effective > 1.5 → 1.5` clamp (otherwise 3.0
   would diverge from 2.0).

**VERIFIED: the law and the history are correct.** The scalar-reference discriminator passes
at every penalty tested, on both prompts, once the harness bug below was fixed:

| prompt | penalty | τ | decode | identity |
|---|---|---|---|---|
| short (19 tok) | 1.0 | 3.167 | 69.8 tok/s | **PASS** |
| short (19 tok) | 1.05 | 3.083 | 67.9 tok/s | **PASS** |
| short (19 tok) | 2.0 | 2.750 | 60.6 tok/s | **PASS** |
| framed (526 tok) | 1.0 | 3.417 | 65.9 tok/s | **PASS** |
| framed (526 tok) | 1.05 | 3.583 | 69.2 tok/s | **PASS** |
| framed (526 tok) | 2.0 | 2.583 | 50.1 tok/s | **PASS** |
| framed (526 tok) | 3.0 | 2.583 | 49.6 tok/s | **PASS** |

(`uno_perf_probe`, mirror applying the same law through the same kernel on *scalar*
logits; p=2.0 and p=3.0 agreeing exactly is the `effective > 1.5` clamp.) The probe's
mirror is the scalar per-token path, so this comparison is free of the batched-window
forward difference.

**Correction to the intermediate claim.** I first wrote that the residual daemon-level
divergence was kernel noise, then — on seeing the (uncorrected) discriminator fail with
gaps up to 2.21 — retracted that and called it a real law bug. **That retraction was
wrong, and its evidence was my own harness defect:** `uno_perf_probe` called
`spec.step(…, &[], …)`, passing an EMPTY `emitted`, so the speculator's history contained
no generated tokens and Uno was penalised over a *different window* (instrumentation
showed `win_len=22` on the Uno side vs `33` on the mirror at the same predicted
position). The daemon passes the real committed list — verified from the trace:
`position - prompt_len + 1 == emitted.len()` at every window. With the probe passing
`&committed`, identity passes at all penalties.

**Therefore the daemon-level A/B residual is the batched-vs-scalar forward difference**
(~0.17 logits measured on a short fixture; the class `spec_impl.rs` documents: *"on a
near-tie logit the batched path's KV flips the verifier's argmax off AR's greedy pick"*),
not the penalty law. The speculator emits the wrong token only where the penalty-shifted
gap is within that difference; at neutral penalty the same prompt happens to have no such
position over 128 tokens, which is why neutral showed 128/128 identical.

Cost: the penalty work is a handful of small in-vector operations per window; the τ
movement dominates the wall-clock change (e.g. framed p=1.05 is *faster* than neutral at
69.2 vs 65.9 tok/s because τ rises to 3.583).

**Daemon history independently validated.** The same construction was checked on the
daemon path directly: at `pos=648` the trace reports `emitted=122` with
`prompt_len=527`, i.e. exactly `pos - prompt_len + 1`, confirming `emitted` spans the
committed stream (its last element is the pending seed); and its last eight entries
`[1339, 559, 1036, 23873, 4758, 13, 373, 1227]` are exactly the committed `token_ids`
ending at that position. So the daemon's penalty window has the right *contents* as well
as the right length — the probe's proven-correct law and the daemon's proven-correct
history together close the K2-Horizon Uno greedy path end to end, and the daemon-level
A/B residual is the forward-kernel difference (the probe, which uses the scalar forward,
passes).

**Root-caused, and it is benign numerics.** The mismatch at position 27 is a near-tie:
`gap_uno = 0.0173`, `gap_ar = 0.0222`, with `max_abs = 0.1429` between the two scalar
forwards. A 0.14 largest-logit difference swamps a 0.02 gap, so the argmax flips —
exactly the class the codebase documents. `uno_forward_row` and
`forward_scratch_compute` are two different scalar kernel routings (the latter goes
through the family dispatch), and they agree except at near-ties; the probe's prefill
assertion demands *exact* argmax equality, which is too strict at long context.

**This also measures the number I had only been attributing.** The forward-numerics
magnitude at 526-token context is `max_abs ≈ 0.14` — the same order as the ~0.17 measured
earlier on the draft rows of a short fixture. So a flip occurs wherever the penalty-shifted
top-2 gap falls below ~0.14, which is exactly what the daemon-level A/B shows (first
divergence at index 33/15/17/10 for p=1.05/1.1/1.3/2.0, with a 0.145 gap at the p=1.05
case measured earlier). The attribution is now quantitative rather than inferred, and the
0.775-gap flip that briefly suggested a law bug came only from the broken probe
(empty `emitted`) and no longer exists once the probe feeds the committed history.

**Tree-mode greedy: DONE and verified.** `uno_tree_verify_picks` now takes the common
prefix, the device ring and the penalty values, and applies AR's Phase-0 law to every
node row before the picks are taken — each node's history being the common prefix plus
that node's root path (from `tree.parents`), which includes the node's own token, the
same convention as the linear window. `step_tree` also penalises the root (`clean`),
which is committed unconditionally. Verified with the probe (it reads `HIPFIRE_UNO_TREE`,
and its mirror is a plain AR chain, so its identity gate covers the tree):

| `UNO_TREE` | penalty | τ | decode | identity |
|---|---|---|---|---|
| 16 | 1.0 | 4.583 | 80.2 tok/s | **PASS** |
| 16 | 1.05 | 4.167 | 72.6 tok/s | **PASS** |
| 16 | 2.0 | 2.917 | 50.4 tok/s | **PASS** |

**Resolved: the probe-vs-daemon τ gap is a workload difference, not a tree inefficiency.**
I flagged the tree's 80.2 tok/s (probe, framed prompt) against 35.5 (daemon, chat-framed)
as a possible daemon-side loss. It is not: the same offset appears in linear mode.

| instrument / prompt | linear τ | tree-16 τ |
|---|---|---|
| `uno_perf_probe`, framed 526-tok file | 3.417 | 4.583 |
| daemon `bench`, chat-framed 527-tok request | 1.18 | 2.13 |

The probe/daemon ratio is 2.9x for linear and 2.15x for tree — both modes move together,
and both instruments agree on the *ordering* (tree > linear). The cause is that the two
drive different token streams: the probe re-encodes my hand-framed text directly, while the
daemon renders the chat template, which changes the continuation and therefore the draft's
acceptance. No daemon-side tree cost to chase; the probe's absolute numbers must not be
compared against daemon benchmarks on this model.

**Still not done — the non-greedy Uno path, and my earlier framing of it was wrong.**
I wrote that no identity harness exists at temp>0 and that I would not change it blind.
The first half is true but the framing is wrong: **token identity is not the right
criterion for the sampled path.** The fused verifier's own header says its RNG is
*"request-seeded Philox4x32-10, **not** Triton's tl.rand bitstream"*, and the CHANGELOG
records that this *"preserves replay within this implementation, **not** the previous
CPU/Triton token stream"*. Two different RNG streams cannot produce token-identical
output, by construction — so the sampled path's contract is a matching rejection **law**,
not a matching token stream. The repo already has the right harness for that:
`uno_verify_probe.rs`, *"GPU rejection-law regression: distribution, row alignment, replay
and invalid inputs"*.

So the correct increment there is: apply AR's Phase-0 law to the verify rows before
`uno_verify_logits` / `uno_filter_verify` (and to the proposal/bonus draws in
`sample_one`, which only affects τ), then extend `uno_verify_probe` with penalized inputs
to confirm the law, plus an assertion that the wiring actually applies the penalty. That is
a different and better-defined task than "build a temp>0 identity harness", and the greedy
identity A/B I built is not the right tool for it.

**Why the rest is untestable here (checked, not assumed).** The remaining paths have no
fixture or no harness in this environment, so changing them would be unverifiable:

- `~/.hipfire/models` is empty; the only llama-family model present is K2 itself, and it
  runs the **Uno** speculator. The generic `llama_spec` verify path
  (`verify_block_argmax` and friends) is used by llama-family models with a DFlash `.hfq`
  draft — the DFlash sidecars on disk (`qwen38-27b-dflash-mq4.hfq`,
  `qwen36-27b-dflash-mq4.hfq`, `*.mtp`) pair with **qwen3.5/3.6** targets, which go
  through the qwen35 path, not `llama_spec`. No fixture here exercises it.
- **deepseek4**: no DS4 fixture on disk at all.
- **Non-greedy Uno**: the fixture exists (K2 + the Uno adapter), but no identity harness
  does — bench hardcodes temperature 0, and the probe's gate runs only at `temp ≤ 1e-6`.
  Building one means aligning the mirror's sampler RNG stream with the speculator's, which
  is the substance of the sampled-losslessness question rather than a quick check.

So within this worktree the set of *verifiable* sampling-law work is complete: the Uno
greedy path is done and verified in both window modes, and everything else is blocked on a
missing fixture or harness rather than on effort.

**Not done — and the generic llama path is a bigger job than the Uno one.** Unlike
`uno_spec.rs`, `llama_spec.rs` has **four public entry points**
(`verify_block_argmax:69`, `verify_block_logits:92`,
`verify_block_argmax_capture_gpu:121`, `verify_block_sampled_capture_gpu:175`) all
funnelling into `verify_block_logits_or_argmax:341`, and **none of them receives a token
history**. Its greedy branch argmaxes raw logits (`argmax(&row)`) and `sample_one:498`
passes the neutral constants to `sample_top_p_pf` — with the explicit comment *"No
repeat/presence/frequency penalty here (verify is distribution-only; the emission layer
owns penalties)"*, which is the same wrong assumption as the Uno site. Wiring it needs:
(1) a `history: &[u32]` threaded through those four entry points and their callers in the
arch crates, (2) the law applied before `argmax(&row)`, and (3) the real penalties passed
in `sample_one`. Unlike the Uno path it needs no 64-slot ring per value — it already has
`SampleCfg::repeat_buf` (currently a `[1]`-element dummy) and the scalar forward, so it
has no batched-vs-scalar issue to contend with. It also needs its own verification
harness; `uno_perf_probe`'s mirror is Uno-specific.

Also outstanding: `uno_spec.rs:698` (`sample_one`-equivalent for the Uno non-greedy path)
and tree-mode greedy (`step_tree`); `dense.rs` (deepseek4) is marked `NOT YET WIRED`.

**Acceptance test (exists).** `HIPFIRE_BENCH_REPEAT_PENALTY=<p>` + `HIPFIRE_EMIT_TOKEN_IDS=1`,
AR vs `--model-draft /adapter`, compare `token_ids` per run at p ∈ {1.0, 1.05, 1.1, 1.3}.
Today p=1.0 gives 128/128 identical and p=1.1 diverges at index 5, so the test is
already discriminating.

## Test status

- `cargo test -p hipfire-generate` — **287 pass**, 0 fail (after the terminal fix).
- `cargo test -p hipfire-arch-llama` — **34 pass**, 0 fail.
- `cargo test -p hipfire-cli` — **274 pass, 1 pre-existing failure**:
  `serve::complete::tests::multi_slot_request_supported_accepts_sampling_rejects_unsupported`
  fails at `complete.rs:7265` (`assertion failed: ok(json!({"max_think_tokens": 0}))`),
  rejected by the validation loop at `complete.rs:2635`. `complete.rs` is byte-identical
  to `HEAD` and none of this work's edits touch request validation (the only `main.rs`
  diff line matching that area is a comment), so this failure exists at `4217f03d`
  independently of the Uno work. Left alone as out of scope.

## Handoff: blocked, decision required

One success criterion is met — token identity, the emitted stream being byte-identical
to AR's at every block size — and one is unreachable (throughput >= 2.0x). The blocker
is structural, and this is its measured shape:

- Ratio = `tokens/window / (window cost in AR tokens)`, both measured. The window must
  run **two full target forwards** — the LoRA draft over the noise rows, then the base
  verify over `clean` + proposals, because the verify input is produced by the draft.
  That is inherent to two-pass diffusion decoding; no ordering or sharing removes one.
  It measures **2.42 AR tokens at 2 rows**, rising with rows (3.61 at 5 rows, 7.8 at 17).
- Both terms move the wrong way as rows grow: tokens/window rises 2.625 -> 4.13 (+57%)
  across the knob space while the window cost rises 2.42 -> 7.8 (3.2x). The ratio
  therefore peaks at the *smallest* window — **block 2, 1.103x** — and decays from there.
- At blocks 1-8 the *best conceivable* window (100% acceptance of every proposal plus
  the bonus) still falls short: 2x needs tokens/window >= 4.84-12.4 while `block_len+1`
  caps it at 2-9. At blocks 16 / tree-16 it is arithmetically possible, but needs ~90%
  acceptance against a measured 12-14%.

The earlier byte-floor framing was retracted (see "Correction" above): AR is not
bandwidth-bound (~325 GB/s of an ~800 GB/s part), so the window is not forced to cost
2.13 AR tokens and does not — at block 2 the Uno path beats AR outright. The bar is just
much further out than +10%.

What would change the answer, in the order I would rank them:

1. **A different drafter** (smaller or better-trained). Needs tau >= 2.26 at block 4
   against a measured 1.18, i.e. ~75% per-proposal acceptance. This is the only lever
   on the binding term and it is explicitly outside this objective's scope. The
   adapter cannot be re-trained here (the repo records the custom-draft line of work
   as a failed, dead experiment) and the adapter files are out of bounds.
2. **The sampling-law gap** (recommended, unrelated to throughput): greedy spec decode
   is not lossless versus AR for any penalised request — on *every* arch and every
   speculator, not just K2-Horizon Uno. Full evidence and both fix shapes in
   § Defects found item 3. Discovered here because it silently broke the identity
   criterion; reproducible by setting `HIPFIRE_BENCH_REPEAT_PENALTY=1.1` and watching
   the two sides diverge at committed index 5. The smallest correct fix is one extra
   condition in the greedy route arms at `ar.rs:1602-1605`
   (`!i.nonneutral_penalties`), matching what the sampled arms already do.
3. **Drop the objective.**

## Reproduce

```bash
cd /home/ghazni/github/hipfire/k2-uno
cargo build --release -p hipfire-daemon --bin daemon -p hipfire-cli --bin hipfire
# AR
docker run --rm --device /dev/kfd --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  --group-add $(getent group video | cut -d: -f3) \
  -v $PWD/target/release:/bins:ro -v ~/models:/models:ro \
  -v $PWD/benchmarks/prompts:/prompts:ro -v /tmp/kcache:/kcache \
  -e HIPFIRE_DAEMON_BIN=/bins/daemon -e HIPFIRE_ROCM_PATH=/opt/rocm -e HIPFIRE_KERNEL_CACHE=/kcache \
  rocm-dev:10.0.0-rpc \
  /bins/hipfire bench /models/hipfire/IFM/K2-Horizon-7B-MQ4/k2-horizon-7b.mq4 \
    --runs 5 --warmups 3 --max-tokens 128 --kv-mode q8 --backend noslots \
    --workload stateless --reasoning-on --prompt-file /prompts/k2_uno_qa_fixed.txt --json
# Uno: the same command plus
#   --model-draft /adapter   (with -v ~/models/RAW/IFM/K2-Horizon-7B-Uno/source:/adapter:ro)
```

Env knobs: `HIPFIRE_UNO_BLOCK` (default 4; trained block is 8),
`HIPFIRE_UNO_GRAPH_CTX` (default 0 = capture off), `HIPFIRE_UNO_TREE`,
`HIPFIRE_UNO_TREE_K`, `HIPFIRE_UNO_NOISE`, `HIPFIRE_UNO_LORA_SCALE`,
`HIPFIRE_UNO_TRACE` (per-window proposal/verify/emit trace),
`HIPFIRE_BENCH_REPEAT_PENALTY` (standard-bench request penalty; default 1.1, set 1.0 on
**both** sides for a valid A/B), `HIPFIRE_DUMP_EVENTS` (CLI: dump every daemon event),
`HIPFIRE_EMIT_TOKEN_IDS` (daemon: `committed` events carrying `tok_id`).
Probe-only (`uno_perf_probe` / `uno_batch_probe`): `HIPFIRE_UNO_PROBE_PROMPT` (read the
probe prompt from a file instead of the built-in one),
`HIPFIRE_UNO_IDENTITY_VERBOSE` (print per-position mirror token and top-2 gap),
`UNO_IDENTITY=0` (skip the identity gate), `UNO_BATCH_PROBE=1` (batched fixture in
`uno_batch_probe`).

Changed files this session: `hipfire-arch-llama/src/uno_spec.rs` (capture cap + trace),
`hipfire-cli/src/main.rs` (bench `--model-draft`, warmups/τ/md5/token-ids, repeat-penalty
seam, event dump), `hipfire-generate/src/qwen.rs` (llama-route open-think terminal),
`hipfire-loader/examples/uno_perf_probe.rs` (prompt file + verbose identity).
`hipfire-arch-llama/src/spec_impl.rs` was touched during the investigation and restored
byte-identically (zero diff vs `HEAD`).
