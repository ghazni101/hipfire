# qkvza fusednorm negative oracle — measured on gfx1101 (gate widened for diagnostic A/B)

**Date:** 2026-10-09 · **Lifecycle:** historical
**Parent:** [2026-10-09-gfx1101-campaign-verify.md](2026-10-09-gfx1101-campaign-verify.md)
(verdict 5 — this record supersedes its "not measurable here by design"
statement) and the gfx1100 fixture record
[2026-08-23-qwen35-4b-mq4-campaign-amendment.md](2026-08-23-qwen35-4b-mq4-campaign-amendment.md)
amendment 3.

**Change:** `qkvza_fusednorm_fold_enabled` now also admits gfx1101 under the
experimental `HIPFIRE_GFX1101_GFX1100_CAMPAIGN=1` gate (double opt-in with
`HIPFIRE_QKVZA_FUSEDNORM=1`), so the negative oracle is reproducible on the
second RDNA3 dGPU instead of being an unreachable gfx1100-only code path.

**Why the gate widening is sound:** the fold's numerics are arch-portable —
`probe_qkvza_fusednorm` is BITEXACT on gfx1101 — and e2e greedy output stays
byte-identical to master here (Fibonacci prompt, 96 tokens, thinking off).
Only the perf verdict was gfx1100-fixture-bound; it is now measured on both
parts.

## Measurement (same rig as the parent record: gfx1101 RX 7700 XT,
fresh container per process, interleaved rounds, qwen3.5-4b.mq4
`712b69f8…`, AR, q8 KV)

| arm | tg128@64 | tg128@2048 |
|---|---|---|
| B1 (fusions on, oracle off) | 115.12 | 113.84 |
| B3W (B1 + oracle engaged) | **87.54** | **86.72** |

**Oracle cost on gfx1101: −24.0% decode** — same direction and nearly the
same magnitude as gfx1100's recorded −26%. The failure mode (per-row-block
norm+rotate recompute across the ~12k-row GEMV grid overwhelming the saved
producer launch) is structural across RDNA3 dGPU, not a one-part artifact.

daemon md5 `b7da7aef…` (branch + gate widening). Raw JSON:
`/tmp/hipfire-ab/out/B3W_*.json` (discovery pointer; medians above).

**Standing conclusion (now two-arch):** the oracle must stay default-OFF.
No perf claim attaches to engaging it on any part.
