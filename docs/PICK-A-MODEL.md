# Pick a model for your GPU

A starting point, not a guarantee. VRAM and size come from the registry
(`registry/models.json`, `min_vram_gb` / `size_gb`); `hipfire list -r` is the
authoritative list. Every tok/s figure is copied from
[BENCHMARKS.md](BENCHMARKS.md), whose tables are **historical**: each was
measured once under the fixture named in its column, not on your card or the
current binary. Use them to compare sizes, not as a promise.

## By VRAM

| Your VRAM | Tag | Size | Speculation setting | Expected decode (historical) |
|---|---|---:|---|---|
| 2–4 GB | `qwen3.5:0.8b` | 0.55 GB | defaults (`speculation=auto`, `dflash_mode=off`) | 391 tok/s on 7900 XTX [A]; 250 RX 6800 XT, 190 RX 5700 XT, 231 MI50 [C] |
| 4–6 GB | `qwen3.5:4b` | 2.59 GB | defaults | 180 tok/s on 7900 XTX [A]; 61 RX 5700 XT, 61 MI50 [C] |
| 8–12 GB | `qwen3.5:9b` | 5.31 GB | defaults; for code, `dflash_mode auto` with `qwen3.5:9b-draft` | AR 132 tok/s on 7900 XTX [A]; 65 RX 6800 XT, 59 MI50 [C]. DFlash on code: 329–373 tok/s vs 122–124 AR; on prose it lost (98–99 vs 123–125) [B] |
| 16–24 GB | `qwen3.5:27b` (or `qwen3.6:27b`) | 15.0 GB | defaults; for code, `dflash_mode auto` with `qwen3.5:27b-draft` / `qwen3.6:27b-draft` | AR 47 tok/s on 7900 XTX [A]; 22 RX 6800 XT, 21 MI50 [C]. DFlash on code: 196.0 (3.5) / 185.5 (3.6) tok/s vs ≈44 AR; prose/instruct ≈1.0–1.1× [B] |

`min_vram_gb` in the registry: 0.8B 2, 4B 4.1, 9B 6.8, 27B 16. 27B does not
fit on 8 GB cards (OOM on RX 5700 XT and BC-250 in [C]). Long context needs
more room than the weights alone; see the KV notes in
[GETTING_STARTED.md](GETTING_STARTED.md#long-context-optional).

## Fixtures

- **[A]** [Historical AR decode, 7900 XTX (gfx1100)](BENCHMARKS.md#historical-autoregressive-decode-no-spec--7900-xtx-gfx1100):
  MQ4, then-default asym3 KV, median of 5 runs, no speculation.
- **[B]** [Historical DFlash by genre, 7900 XTX](BENCHMARKS.md#historical-dflash-speculative-decode-by-genre--7900-xtx):
  asym3 KV, `--no-chatml`, `max_tokens=120`, DFlash forced on, median of 5.
  Superseded methodology; the speedup depends on the prompt genre.
- **[C]** [Historical other arches](BENCHMARKS.md#historical-other-arches-decode-toks):
  then-default configuration, decode tok/s.

## Speculation settings

- **Defaults** (`speculation=auto`, `dflash_mode=off`): eligible MTP paths may
  still engage; DFlash stays off. Force plain AR with
  `hipfire config set speculation off`.
- **DFlash** is opt-in: `hipfire pull <tag>-draft`, then
  `hipfire config <tag> set dflash_mode auto`. It helped code prompts and hurt
  prose in fixture [B]; measure your own workload before leaving it on.

Larger and newer families (Qwen 3.6 35B-A3B, Qwen 3.8 27B, Flash-Next) have no
rows in BENCHMARKS.md; see [MODELS.md](MODELS.md) and the dated
[perf checkpoints](perf-checkpoints/).
