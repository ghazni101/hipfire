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

`min_vram_gb` in the registry: 0.8B 2, 4B 4.1, 9B 6.8, 27B (Qwen 3.5 / 3.6 MQ4) 16.
A 27B MQ4 trunk does not fit on 8 GB cards (OOM on RX 5700 XT and BC-250 in
[C]); the lowest 27B floors in the registry are the 1-bit and ternary Bonsai
tags (5 and 8 GB, below). Long context needs more room than the weights
alone; see the KV notes in
[GETTING_STARTED.md](GETTING_STARTED.md#long-context-optional).

## Larger and newer families (registry floors only)

These tags have no rows in [BENCHMARKS.md](BENCHMARKS.md), so only size and the
registry VRAM floor are given; no speed is claimed. Pick the tier whose
`min_vram_gb` fits with headroom for KV, and read [MODELS.md](MODELS.md) for
the family notes.

| Your VRAM | Tag (`min_vram_gb`) | Size | Note |
|---|---|---:|---|
| 5–8 GB | `bonsai:27b-bq1` (5) / `bonsai:27b-tq2` (8) | 3.8 / 7.16 GB | 27B at 1.14 / 2.125 bpw; heavy quantization, see the registry `desc` for its KLD |
| 12 GB | `qwen3.5:27b-mq3` (12), `qwen3.6:27b-mq3` (12) | 10.7 GB | MQ3 alpha, gfx11 / gfx12 |
| 13–15 GB | `qwen3.8:27b-mq3-xt` (13), `qwen3.8:27b-mq3` (14), `qwen3.8:27b-mq3-pro` (15) | 11.78–13.18 GB | MQ3V2 tiers |
| 16–18 GB | `qwen3.8:27b-mq4-xt` (16, speed), `qwen3.8:27b` (17, registry default), `qwen3.8:27b-mq4-pro` (18) | 14.99–16.46 GB | MQ4V2 tiers; `qwen3.8:27b-mq4-xts` (16) is the canonical dense fixture |
| 20–24 GB | `qwen3.8:27b-mq5*` (20–21), `qwen3.8:27b-mq6*` (23–24) | 18.18–22.17 GB | higher-quality V2 tiers |
| 22 GB | `qwen3.6:35b-a3b` (22), `ornith-1.5:35b-a3b` (22), `nex-n2:mini` (22) | 19–19.8 GB | 35B / 3B-active MoE |
| 24 GB | `muse-glimmer:fast` (24) / `muse-glimmer` (26) | 16.26 / 18.61 GB | 30B dense, thinking model |
| 96 GB | `deepseek-v4-flash` (96), `minimax-m2.7` (96) | 82 / 79.2 GB | MQ2-class MoE |
| 128 GB | `qwen3.8:flash` → `qwen3.8:flash-next` (128) | 125.3 GB | Flash-Next (arch 16) |

Qwen 3.8 27B tiers also pull the shared vision tower (`vision_mode`, default
`off`) and the shared MTP head. Image generation (`flux.schnell:1`, 24 GB) is a
separate surface: [IMAGEGEN.md](IMAGEGEN.md).

## Fixtures

- **[A]** [Historical AR decode, 7900 XTX (gfx1100)](BENCHMARKS.md#historical-autoregressive-decode-no-spec--7900-xtx-gfx1100):
  MQ4, then-default asym3 KV (deprecated since 0.4.0, removal in 0.5.0; on
  Qwen `asym3` is now an alias of `fwht3`), median of 5 runs, no speculation.
- **[B]** [Historical DFlash by genre, 7900 XTX](BENCHMARKS.md#historical-dflash-speculative-decode-by-genre--7900-xtx):
  asym3 KV (same deprecation), `--no-chatml`, `max_tokens=120`, DFlash forced on, median of 5.
  Superseded methodology; the speedup depends on the prompt genre.
- **[C]** [Historical other arches](BENCHMARKS.md#historical-other-arches-decode-toks):
  then-default configuration, decode tok/s.

## Speculation settings

- **Defaults** (`speculation=auto`, `dflash_mode=off`): eligible MTP paths may
  still engage; DFlash stays off. Force plain AR with
  `hipfire config set speculation off`. Exception: `qwen3.8:27b` and its
  `mq3*` / `mq4*` tiers resolve `dflash_mode=auto` from the registry, so a
  paired draft file that is present is used (absent: MTP `auto` as before);
  `hipfire config <tag> set dflash_mode off` opts out
  ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)).
- **DFlash** elsewhere is opt-in: `hipfire config <tag> set dflash_mode auto`.
  `hipfire pull <tag>` already fetches a draft the registry declares for that
  tag (for example `qwen3.5:9b`, `qwen3.5:27b`, `qwen3.6:27b`), and the
  separate `-draft` tags are listed in [MODELS.md](MODELS.md). It helped code
  prompts and hurt prose in fixture [B]; measure your own workload before
  leaving it on.

## Not recommended

PFlash prefill compression and CASK / TriAttention KV eviction are deprecated
since 0.4.0 and scheduled for removal in 0.5.0; they are unsupported and no
row here depends on them. The legacy contiguous-DFlash knobs
(`HIPFIRE_DFLASH_WINDOW=0`, `HIPFIRE_DFLASH_CTX_CAP`) and the Givens `legacy-asymN`
/ `asymN` / `turboN` KV names are deprecated on the same schedule; prefer a
`kv_cache` of `auto`, `q8` or `fwhtN`. See the dated
[perf checkpoints](perf-checkpoints/) for newer-family measurements.
