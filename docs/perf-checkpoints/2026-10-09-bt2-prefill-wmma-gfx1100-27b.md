# BT2 batch-tiled WMMA prefill GEMMs — Qwen3.8-27B deep A/B (gfx1100)

**Date:** 2026-10-09 (recording; measurements taken 2026-08-23 on the
`tune/iter3-gate-up-bt2` campaign branch, commit `b7a8f9a1a2`) ·
**Lifecycle:** historical
**Branch:** `perf/gfx1100-decode-campaign` (BT2 commits `124c5d7c6`,
`a571a014e`, `579277fa1`, `aa39f6c07`)
**Fixture:** `qwen3.8-27b.mq4` md5 `2fb2edc27865b697c5209c104de91f32`,
`bench_qwen35_mq4` md5 `3c5040e5a490dbc479696141c238bb76`, pp32 prefill /
tg128 decode, KV q8, plain AR.
**GPU:** gfx1100 (RX 7900 XTX class), `HIPFIRE_VERIFY_GRAPH=0`,
`HIPFIRE_DPM_WARMUP_SECS=20`.

## What BT2 is

Batch-tiled B=2 WMMA variants of the four dense prefill GEMMs
(`gemm_{gate_up,qkvza,qkv}_hfq4g256_wmma*`, residual
`ksplit_det`): two independent accumulator chains per block reuse the same
weight tile across 2 batch tiles, halving the N-grid. Dispatched by default
for `batch_size >= 32` on RDNA3 dGPU (gfx1100/1101/1102);
`HIPFIRE_BT2_DISABLE=1` is the kill switch, `HIPFIRE_{QKVZA,QKV,KSPLIT_DET}_BT2_FORCE=1`
force arms for probing.

## Result (deep A/B: alternating arms, 20 prefill runs per session, fresh
processes; 4 baseline sessions / 3 bt2 sessions)

| arm | n | prefill median tok/s | range | decode median |
|---|---|---|---|---|
| baseline (bt2 off) | 80 | 452.55 | 436.0–457.3 | 48.50 |
| bt2 | 60 | 583.95 | 548.4–599.5 | 48.50 |

**Prefill +29.0% median, every bt2 sample above every baseline sample**
(commit message reports p<0.01). Decode unchanged (BT2 arms only engage at
batch ≥ 32; decode runs batch 1).

Raw sample arrays originally lived in the campaign worktree under
`.codeinsight+research/kernel-tune/deep_ab_qwen38_27b.json` (gitignored
scratch); medians and ranges above are computed from that file's samples,
retained here as the durable record.

## 4B note

The Qwen3.5-4B speed-gate floor re-capture on this branch
(`4b_mq4_pp32_prefill_tok_s` 1014.0 → 1843.5) was taken against the
campaign tree but the previous floor dated from 2026-04-12 — the delta
conflates six months of master-side prefill work with this branch's BT2
arms. See the gfx1101 container A/B recorded in
[2026-10-09-gfx1101-campaign-verify.md](2026-10-09-gfx1101-campaign-verify.md)
for a same-day master-vs-branch isolation of the branch's own prefill delta.
