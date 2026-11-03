---
title: qt59 W8A8 vs f16-twin crossover is 48 rows; DFlash verify arithmetic follows it
date: 2026-11-03
tags: [iu8g32,qt59,w8a8,dflash,gfx1100]
---

## Crossover (gfx1100, `iu8g32_route_crossover`)

qt59 (IU8G32 = IU8G128) has two batched GEMMs: W8A8 (`gemm_iu8g32_w8a8_min`,
128x128 LDS tile, int8 activations) and the f16-WMMA twin
(`gemm_iu8g32_residual_wmma` + mb4/mb4m2). On 27B `wo` (M=5120, K=6144) and
`w_down` (M=5120, K=17408), activation conversion charged per call, two passes:

- n <= 32: twin 1.3-2.4x faster (W8A8 idles most of its 128-row tile)
- n = 40-48: parity
- n = 56-63: W8A8 1.26-1.36x
- n >= 64: W8A8 1.9-2.8x (the twin's mb4 fanout starts at 64 and is itself
  1.5-1.9x slower than the twin's n=63 tile on these shapes; worth a look if
  K%128-only matrices or `HIPFIRE_IU8G32_W8A8=0` ever matter)

`IU8G32_W8A8_MIN_ROWS = 48`; `Gpu::iu8g32_gemm_route` is the single selector for
the singleton batched body and the multi-slot body. Before this, the singleton
body ran W8A8 at every n and the slots body only at n >= 128.

## DFlash consequence

DFlash verify (16 rows) on the singleton body used W8A8 for the qt59 `wo`;
the serve path (multi-slot body) already used the twin. With one selector the
two agree. Measured on 27B IU4P + `qwen38-27b-dflash-mq4.hfq` (md5 0133955...),
pep8 LRU prompt (md5 df5dedc8...), `hipfire bench --spec dflash`, fresh process,
ABBA: tau 9.17 -> 10.75, decode 72.4 -> 92.2 tok/s, BUT the generated text
changed (61 -> 188 tokens), so this is an arithmetic change, not a like-for-like
speed win. Through `serve_harness` (greedy, think cap 256) the base and new
binaries are byte-identical for both AR and DFlash (tau 7.11).

## Gotcha

`iu8g32_wmma_matches_reference` probed `iu8g32_wmma_eligible(32)`; the compact
row refuses K=32, so the twin's numerics gate silently SKIPPED. Probe with an
admitted width.
