# CLI reference

Audience: operators driving the native Rust `hipfire` binary
(`crates/hipfire-cli`).
Run `hipfire <cmd> --help` (or `hipfire help <cmd>`) for the live flag text. This page indexes subcommands and file locations; it does **not** duplicate the full model registry or every config key.

Bare `hipfire` (no subcommand) is equivalent to `hipfire tui`: it launches the terminal UI and **requires** the `hipfire-tui` binary (`HIPFIRE_TUI_BIN`, `~/.hipfire/bin/hipfire-tui`, the workspace `target/release`, or `PATH`). When that binary is missing the command exits with an error naming the build command; it does not print a command list (use `hipfire help`). Config keys: [CONFIG.md](CONFIG.md). Env vars: [env-vars.md](env-vars.md). Models: [MODELS.md](MODELS.md).

## Model lifecycle

| Command | Purpose |
|---|---|
| `hipfire pull <tag>` | Download a registry model into `~/.hipfire/models/` and fetch its declared DFlash (`dflash.file`), vision (`vision.file`) and MTP (`mtp.file`) sidecars. The schema `dflash_mode` default is **`off`**; Qwen3.8-27B tiers with a declared draft (`qwen3.8:27b`, `-mq3*`, `-mq4*`) carry registry-layer `auto` ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)). A present draft enables eligible DFlash; an absent file leaves native MTP governed by its separate `auto` policy. `dflash_mode=on` / `speculation.dflash=on` refuses without a DFlash draft, even on an MTP-only tier; it never aliases to MTP. Use `--spec mtp` / `speculation.mtp=on` for explicit native MTP. Native MTP `auto` declines a resolved host-mapped routed-expert spill on either execution seam. Uniform MQ6G256 V1 (qt15) Qwen35 MoE on gfx1200/gfx1201 with FP16 WMMA enabled refuses an attaching native MTP head, including resident `auto`; use `--spec off` / `speculation.mtp=off` for AR or `HIPFIRE_FP16=0` for explicit MTP. Other resident attachment is unchanged. `dflash_mode=off` beats registry `auto`. Draft path override: `developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` / `run --model-draft`; vision override: `run --vision` / `serve --vision` / `HIPFIRE_VISION_SIDECAR`. See [MODELS.md](MODELS.md#dflash-draft-artifacts-registry). |
| `hipfire pull <tag> --force` | Replace an existing target after downloading and verifying a new copy. |
| `hipfire pull <tag>` (on a terminal) | Asks only what the registry leaves open, then confirms: (1) the **tier**, when the registry has other tags `<tag>-<suffix>` of the same `arch_id` (aliases resolve first, so `qwen3.8:flash` offers the `qwen3.8:flash-next` tiers; same-artifact tags, a split entry's monolithic sibling, drafts and local-only entries are not tiers), each with its size, the first sentence of its `desc` and whether `min_vram_gb` fits the detected VRAM or VRAM+RAM; (2) the **PLE rows** of a split entry (published q4/q8, both, or none; preselected: what `--ple auto` fetches, or q4 when host RAM cannot hold the base plus the q8 rows); (3) the registry's **draft/MTP sidecars** (MTP, DSpark, DFlash; default yes); (4) a **summary** with every file, the total download and the free disk space (default no when it does not fit). Enter takes the default. A flag skips its question: name the tier's tag, `--ple`, `--draft` / `--no-draft`. A single-variant model with nothing to ask pulls without a prompt. `-y` / `--yes`, or a stdin that is not a terminal, asks nothing and takes the registry defaults (the tag as given, `--ple auto`, every declared sidecar), so scripts and CI are unchanged. |
| `hipfire pull <tag> --no-draft` | Skip the registry's draft/MTP sidecars (MTP, DSpark, DFlash); `--draft` fetches them without asking (the default). Other declared sidecars are always fetched. |
| `hipfire pull <tag> --ple auto\|q4\|q8\|all\|none` | For a split Flash-Next entry, choose which PLE row sidecar is downloaded next to the base (default `auto`: `memory.ple_format`, else q4 on gfx1151 / q8 elsewhere). See [Flash-Next split PLE layout](#flash-next-split-ple-layout). |
| `hipfire pull <flash-next tag>` (trunk overlay) | The four stock Flash-Next MQ4 XTS tags also fetch the `qwen3.8-flash-next.mq4-xts.trunk-overlay` sidecar (~305.5 MB) next to the base, from a pin compiled into the client. See [Flash-Next trunk overlay](#flash-next-trunk-overlay). |
| `hipfire pull <flash-next tag>` (expert hotset) | The four stock Flash-Next MQ4 XTS tags also fetch the `qwen3.8-flash-next.mq4-xts.expert-hotset` sidecar (217676 B, ~213 KiB) next to the base, selected by default in the CLI and as the default-on **Expert hot set (R9700 decode, 213 KB)** yes/no choice in the TUI, from a pin compiled into the client. See [Flash-Next expert hotset](#flash-next-expert-hotset). |
| `hipfire ple derive [--format q4] [--rank PATH] <model>` / `hipfire ple split [--out PATH] [--keep-original] <model>` | Derive `<model>.ple-q4` from the Q8 rows, or split a monolithic model into a PLE-less base + `<base>.ple-q8`. `<model>` is a path or registry tag. |
| `hipfire ple rebind --old-base PATH --new-base PATH --source PATH` | Copy an existing q4 PLE sidecar to `<new-base>.ple-q4` with a new base binding, only after proving both monolithic bases have identical embedded PLE descriptors and Q8 bytes. The source sidecar is never modified. |
| `hipfire list [-r\|--remote] [-j\|--json]` | Local models; `-r` also lists pullable registry tags; user aliases from `quantize --register` appear separately. |
| `hipfire rm <tag\|path> [-y\|--yes]` | Delete the weight file and sibling sidecars (`.triattn*.bin`, `*.mtp`, matching DSpark). A declared DFlash draft is **shared**: several tags can name the same `dflash.file` (e.g. `qwen3.8:27b` / `qwen3.8:27b-mq4-pro` / `qwen3.8:27b-mq4-xt` → one `qwen38-27b-dflash-mq4.hfq`). If any *other* registry entry that declares the same draft still has its own target file on disk, `rm` **keeps** the sidecar and prints one stderr line; otherwise the draft is removed with the target. The `vision` tower sidecar and the `mtp` head follow the same shared-keeper rule (every `qwen3.8:27b*` tier declares the one `qwen3.8-27b-vision.hfq`, and every trunk tier the one `qwen3.8-27b.mtp`). Confirms unless `-y`. |
| `hipfire ps [-j\|--json]` | Running daemon / quantize / upload processes and whether the configured serve port is busy (process scan is Linux-oriented). |
| `hipfire remove <tag\|path>` | Alias of `rm`. |
| `hipfire registry status\|list\|update [-j\|--json]`, `hipfire registry show <tag> [-j]`, `hipfire registry verify [PATH] [-j]` | Inspect the registry migration surface. `status` prints the source in use (bundled, cache or network), schema version, revision, cache path and warnings; `list` prints the registered tags; `show` prints one canonical entry; `update` loads the registry and prints the same report as `status` (the cache is refetched only when older than 24 h); `verify` validates the bundled registry or an explicit v1 JSON file. |

Tags resolve through the dynamic registry + aliases. Authoritative live list:
`hipfire list -r`. The native binary embeds `registry/v1.json`; the curated
editing surface is `registry/models.json`. At startup it may refresh the v1
payload into `~/.hipfire/registry.cache.json` (24h). Offline or invalid remote
payloads fall back to cache then the embedded registry. Pin the bundle with
`HIPFIRE_NO_REGISTRY_FETCH=1`.

Flash-Next MQ4 XTS tags (short name `qwen3.8:flash`, alias for
`qwen3.8:flash-next`) automatically migrate the exact SHA-verified installed
`qwen3.8-flash-next-gptq3.mq4` to `qwen3.8-flash-next.mq4-xts` by a no-clobber
hardlink, preserving the old name. External-storage symlinks are preserved;
when linking is unsupported or read-only, the verified old path is used.
This does not download or copy the payload. Conflicting canonical files fail
clearly; the historical RTN `.mq4` artifact is never migrated. See
[MODELS.md](MODELS.md#qwen-38-flash-next-mq4-xts) for the compatibility window.

### Flash-Next split PLE layout

A split registry entry (`file` = `<name>-base.<ext>`, e.g.
`qwen3.8-flash-next-base.mq4-xts`) ships the model without its 128 PLE n-gram
shards plus swappable row files next to it: `<base>.ple-q8` (exact Q8F16 rows,
~54.4 GB) and `<base>.ple-q4` (4-bit Lloyd-Max rows, ~28.8 GB, bit-identical in
decode to the q4 the loader derives locally from the q8 rows).

- **Pull.** `hipfire pull <tag> [--ple auto|q4|q8|all|none]` downloads the base
  plus the chosen sidecar(s) after printing the plan (files, sizes). `auto`
  (default) follows `memory.ple_format` (`HIPFIRE_PLE_FORMAT`), whose own `auto`
  is q4 on gfx1151 and q8 elsewhere; an unknown GPU arch resolves to q8. `none`
  fetches the base only. All files are SHA-256 verified; a failed sidecar fails
  the pull. Without `--ple`, a pull on a terminal asks which rows to fetch
  (preselecting what `auto` fetches, or q4 when host RAM cannot hold the base
  plus the q8 rows); `--yes` or a non-terminal stdin takes `auto`.
- **On-demand.** `run`, `bench` and `serve` download a missing sidecar before
  loading (`ple: downloading <file> (<GB>) for memory.ple_format=<..>`): q8 when
  q8 is requested; q4 only when neither `<base>.ple-q4` nor `<base>.ple-q8` is on
  disk (the loader derives q4 from q8 locally). `auto` uses whichever of the two
  is present and downloads the arch's preferred format only when neither is. A
  serve *request* naming such a model downloads only when
  `serve.allow_request_pull` is on; otherwise the load is refused with the pull
  command.
- **No forced migration.** A user who already has the monolithic
  `qwen3.8-flash-next.mq4-xts` (or its legacy `…-gptq3.mq4` name) keeps serving
  it under `qwen3.8:flash-next`: while the split base is absent the tag resolves
  to the local monolithic (SHA-verified as before) and one note says
  `hipfire ple split <tag>` or `hipfire pull <tag>` gets the split layout.
- **`hipfire ple derive [--format q4] [--rank PATH] <model>`** writes
  `<model>.ple-q4` from the model's Q8 rows (`<model>.ple-q8` when present, else
  the embedded rows), ordered by `--rank` (default `<model>.plehot` when present,
  else row order). `--format q8` is an error: q8 rows are the model's own rows or
  the file `hipfire ple split` writes. Progress (percent, GB, MB/s) prints at
  most every ~2 s; the summary lists output, sizes, elapsed time and the
  `payload_sha256` / `q8_rows_sha256` digests.
- **`hipfire ple split [--out PATH] [--keep-original] <model>`** converts a
  monolithic model into `<base>` (default: `-base` inserted before the
  extension) + `<base>.ple-q8`, verifies both outputs, then removes the original
  (and its hardlinked legacy spelling) unless `--keep-original`. It refuses to
  overwrite existing outputs and a non-monolithic input. `<model>.plehot*` files
  belong to the monolithic layout and are left in place with a note. The summary
  prints `base_index_sha256` and `q8_rows_sha256`.
- **`hipfire ple rebind --old-base PATH --new-base PATH --source PATH`** is for a repacked monolithic base whose non-PLE weights changed but whose embedded PLE rows did not. It validates the source q4 sidecar against the old base, compares both bases' PLE descriptors and Q8 payloads, verifies every q4 segment digest, and copies the unchanged q4 payload with only the new base-size/index binding. After readback and attachment validation it atomically replaces `<new-base>.ple-q4` with a separate inode, including when that destination was a hardlink to the source. The source file is never rewritten. Changed PLE bytes/layouts or corrupt digests are refused; split bases are not supported. A stock sidecar must not simply be hardlinked to a smaller requantized base: its binding will correctly fail the loader.
- **`rm` and listings.** `rm` also removes `<model>.ple-q4`, `<model>.ple-q8`,
  their `.plehotrows` packs and `<model>.plehot`. `*.ple-q4`, `*.ple-q8`,
  `*.plehot` and `*.plehotrows` files are sidecars: never listed or served as
  models.

### Flash-Next trunk overlay

The four stock Flash-Next MQ4 XTS tags (`qwen3.8:flash-next`, `-mq4`, `-gptq3`,
`-mq4-xts`) share one base artifact (sha256 `8b15b6fe…`, 125288544792 B). The
client carries a compiled-in pin for that base, naming the trunk overlay
`qwen3.8-flash-next.mq4-xts.trunk-overlay` (the canonical base file name plus
`.trunk-overlay`; 305543168 B, sha256 `df49bc0a…`) in the
`hipfire-models/qwen3.8-flash-next` repo. There is no registry schema change, so
older clients keep parsing the published registry. The overlay holds the 108 GDN
`in_proj_{z,a,b}` projections as pre-encoded symmetric QT44, bound to the base by
size and index SHA-256. The Qwen4 loader substitutes them for the base's MQ6
tensors on gfx1151 by default; `HIPFIRE_QWEN4_TRUNK_OVERLAY=0` restores the base
weights and `=1` applies the overlay on any architecture.

- **Pull.** `hipfire pull <tag>` downloads and SHA-256 verifies the pinned
  overlay with the other sidecars; a valid file already on disk is not
  downloaded again, a stale one is refreshed. As for other sidecars, a failed
  overlay download prints a warning and the pull still succeeds (the base loads
  unchanged). Any other registry entry (e.g. the RTN-asym file or the local-only
  zab tier) has no overlay.
- **`rm` and listings.** `rm <tag>` removes the overlay with the base
  (`<file>.trunk-overlay`). `*.trunk-overlay` files are sidecars: never listed
  or served as models.

### Flash-Next expert hotset

The same four stock Flash-Next MQ4 XTS tags (one base artifact, sha256
`8b15b6fe…`, 125288544792 B) also have an expert-hotset sidecar. As for the
trunk overlay, the client carries the pin, not the registry: it names
`qwen3.8-flash-next.mq4-xts.expert-hotset` (the canonical base file name plus
`.expert-hotset`; 217676 B, ~213 KiB, sha256 `7f6760e1…`) in the
`hipfire-models/qwen3.8-flash-next` repo, matched by the base's exact size and
SHA-256. `registry/v1.json` is unchanged, so older clients keep parsing the
published registry.

- **Pull.** `hipfire pull <tag>` selects the hotset by default. The TUI fetch
  screen has an independent default-on yes/no choice, **Expert hot set (R9700
  decode, 213 KB)**, shown in the summary only when selected; the "213 KB" is
  the rounded display label, the pinned file is 217676 B (~213 KiB). The hotset
  is downloaded and SHA-256 verified with the other sidecars to the sidecar path
  of the local base file, which may be a renamed local file as long as it matches
  the pinned base identity; a valid file already on disk is not downloaded again,
  a stale one is refreshed. As for other sidecars, a failed hotset download
  prints a warning and the pull still succeeds. Any other registry entry (e.g.
  the RTN-asym file or the local-only zab tier) has no hotset.
- **Runtime.** The hotset applies by default on a single R9700 (gfx1201) when
  the sidecar is present and bound to the base by size and index SHA-256.
- **`rm` and listings.** `rm <tag>` removes the hotset with the base
  (`<file>.expert-hotset`). `*.expert-hotset` files are sidecars: never listed
  or served as models.

## Inference

| Command | Purpose |
|---|---|
| `hipfire run <model> [flags] [prompt...]` | One-shot generate. Model = registry tag, alias, or path. Uses a healthy `serve` over HTTP when present; otherwise spawns a one-shot daemon. Forces local spawn when `HIPFIRE_LOCAL=1` or a load-time override such as `--kv-mode`/`--image` cannot safely reuse the resident model. Recognized missing registry tags auto-pull. |
| `hipfire img <model> [flags] <prompt...>` | One-shot txt2img on a diffusion checkpoint (arch 40 FLUX.1, arch 45 FLUX.2 Klein). Model = an HFQ trunk pack (`<base>-transformer.hfq`, sidecars next to it) or a registry tag that resolves to one (`flux.schnell:1`); build packs with `hipfire-quantize --flux-pipe` below. Writes `hipfire-<seed>.png` in the working directory (or the `-o` path) and prints the path. |
| `hipfire embed <model> [flags] [text...]` | Encode text and/or images with an embedding checkpoint (arch 17, the Qwen3.5 multi-vector encoder — e.g. `pplx-embed-v2-late-0.6b`; or arch 18, EmbeddingGemma-2, one pooled vector). Prints tokens×dims; `--query`/`--doc` score the pair by late-interaction MaxSim; `--image`/`--audio`/`--video` add media (arch 18: `--video` takes one clip per flag, as its decoded frames); `--json` emits the token-embedding matrix. See [`hipfire embed` flags](#hipfire-embed-flags). |
| `hipfire chat [model] [-t <f>] [--top-p <f>] [-n <n>] [--system <text>] [--no-color]` | Interactive multi-turn line client over the HTTP service (not the TUI; for the TUI use `hipfire tui`). The model defaults to `serve.default_model`. `-t` is `0..=2`, `--top-p` is `(0,1]`, `-n` is `1..=393216`. See [CHAT.md](CHAT.md). |
| `hipfire serve [model] [host] [port] [flags]` | OpenAI-compatible HTTP server. See [SERVE.md](SERVE.md). |
| `hipfire restart [serve flags...]` | `stop --force` semantics then start with the same flags. |
| `hipfire stop [port] [--force] [--all]` | Stop the tracked background serve (`~/.hipfire/serve.pid`). `--force` reaps orphan daemons and frees the port; `--all` also reaps orphan quantize jobs. |
| `hipfire tui [flags...]` | Launch `hipfire-tui` (Home / Dashboard / Chat / Models / Settings / System). Trailing arguments are forwarded to `hipfire-tui` (e.g. `--check`). |

### `hipfire run` flags
Flags may appear before or after the model. CLI help and the native typed schema list stored globals; **effective `run` sampling** is not “global default unless overridden.” The Rust request resolver transmits only explicit CLI flags, per-model overlay values, or registry `recommended_settings`. Otherwise the field is **omitted** so daemon/HFQ/arch fallback applies. Global-only `temperature` / `top_p` / `repeat_penalty` are not sent. **Chat** is the exception (global session snapshot — [CHAT.md](CHAT.md)).

| Flag | Purpose |
|---|---|
| `-t, --temp <float>` | Temperature when set on the CLI (`0` = greedy). Stored global default is `0.3` but is not auto-sent. |
| `--top-p <float>` | Nucleus sampling when set. Stored global default `0.8` is not auto-sent. |
| `--repeat-penalty <float>` | Repeat penalty when set. Stored global default `1.05` is not auto-sent. |
| `-n, --max-tokens <int>` | Generation cap (config default `4096`). |
| `--kv-mode <m>` | Whole-cache KV preset for this load: `auto`, `q8`, `fp8`, `bf16`, `fwht4`/`3`/`2`; legacy spellings `asym4`/`3`/`2`, `turbo`… (see [KV load flags](#kv-load-flags-run--serve--bench)). |
| `--kv-backend <legacy\|vmm>` | KV allocation backend. Omitted = automatic (prefer VMM). `contiguous` is **rejected** (renamed to `legacy`). |
| `--kv-k <name>` | Qwen-family K-axis override (`q8`, `fwhtN`; deprecated since 0.4.0, removal in 0.5.0: legacy `asymN`/`turboN` → `fwhtN`; legacy Givens `legacy-asymN`). |
| `--kv-v <name>` | Qwen-family V-axis override (`q8`, `lloyd2`/`3`/`4`). |
| `--head <name>` | Select a published `lm_head` variant from the registry's `heads` (e.g. `--head q4k`). The overlay shadows the model's own head at load time; omitted = the head baked into the model file. |
| `--tp <N>` | Tensor/expert-parallel degree for this load, `1..=64` (same admission as `serve --tp` / `bench --tp`). Only values above 1 are forwarded and force a local daemon. |
| `--max-seq <n>` | KV context for this load, within the `memory.max_seq` bounds (512–1048576); sent as the load's `max_seq` and replaces any configured `memory.max_seq`, as `bench --max-seq` does. Unset: eligible growing Qwen VMM KV defaults to min(model trained context, measured card capacity); legacy and other owners retain their bounds. Forces a local daemon (see below). |
| `--spec <m>` / `--speculation <m>` | Spec mechanism: `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` (config default `auto`). |
| `-md, --model-draft <path>` | DFlash draft path; implies `--spec dflash` when `--spec` is not given. |
| `--draft-max <N>` (alias `--draft`) | Draft window, `1..=32`. With `--spec ngram` it sets the n-gram window, with `--spec mtp` the MTP window, otherwise both. |
| `--dspark-conf-threshold <f>` | DSpark confidence cutoff in `[0,1]` (qwen3 + deepseek4). |
| `--system <text>` | System prompt. |
| `--image <path>` | Vision input (when the model supports it). |
| `--vision <path>` | Vision-tower sidecar for this load; wins over the registry `vision` slot and `HIPFIRE_VISION_SIDECAR`. Skipped while `vision_mode=off` (default); required when `on`. Also accepted by `serve` (where `serve -d` does not forward it to the detached child). |
| `-j, --json` | Machine-readable output. |
| `--no-stream` | Buffer full response. |
| `--stats` | Emit one statistics footer on stderr after successful generation (local or HTTP, streaming or `--no-stream`); generated stdout is unchanged. Suppressed with `--json` and on failed/aborted generation. |

Resolution ladder for speculation: **CLI flag > env > per-model > global > registry card > built-in default**. On `run`, `--spec` replaces the selector the config and environment layers resolved (`hipfire config explain speculation.mode` prints the winning source).
`dflash_mode` itself defaults to **`off`** in the schema — see [CONFIG.md](CONFIG.md#speculative-decode) — except that Qwen3.8-27B tiers with a registry-declared draft resolve `auto` through the registry layer ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)); an explicit `dflash_mode=off` (per-model, global or env) wins. Pulling a draft for any other model does not enable DFlash.

`--stats` illustrative output (values are placeholders, not a benchmark): `[stats] tokens=256 decode=123.30 tok/s ttft=42.1 ms tau=7.15`.
`tokens` counts committed daemon completion IDs, including reasoning, not
printed characters. `decode` is decode throughput, not wall-inclusive `tok_s`;
`ttft` is time to first token. `tau` is included only for speculation when
available. Missing, invalid or nonfinite metrics show `n/a`, never a substituted
wall rate or zero. A continuous-batch response may legitimately have `decode=n/a`.

```bash
hipfire run qwen3.5:9b "What's 2+2?"
hipfire run qwen3.5:9b "What's 2+2?" --stats
hipfire run qwen3.5:9b --spec ngram "Repeat verbatim: ..."
hipfire run qwen3.5:27b -md ~/.hipfire/models/qwen35-27b-dflash-mq4.hfq "..."
HIPFIRE_LOCAL=1 hipfire run qwen3.5:4b "..."   # skip HTTP; always local spawn
```

Local-forcing (skip a healthy serve): `HIPFIRE_LOCAL` truthy, `--image`, `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--max-seq`, `--head`, `--tp` above 1, `--spec`/`--speculation`, `--model-draft`, `--vision`, `--draft-max`, or `--dspark-conf-threshold` (exact list: `run_should_force_local` in `crates/hipfire-cli/src/main.rs`). JSON and non-streaming responses are supported by the native HTTP service and do not by themselves force a local daemon.

### KV load flags (`run` / `serve` / `bench`)

Shared by `hipfire run`, `hipfire serve`, and `hipfire bench` (including `--matrix`). Full config surface: [CONFIG.md](CONFIG.md#kv-cache).

| Flag | Values / default | Notes |
|---|---|---|
| `--kv-backend` | `legacy` \| `vmm`; **default automatic prefer VMM** | Accepts only those two spellings. Old `contiguous` is rejected with a migration error that names `legacy` (e.g. use `--kv-backend legacy` or `memory.kv_backend = "legacy"`). Selecting legacy (explicit or automatic fallback) prints one stderr warning containing the stable token `HIPFIRE_KV_BACKEND=legacy`. Explicit `vmm` on an unsupported combination fails closed before teardown. |
| `--kv-mode` | `auto` (default when unset), `q8`, `fp8`, `bf16`, `fwht2`/`3`/`4`; legacy spellings `asym2`/`3`/`4`, `turbo`/`turbo2`/`3`/`4`; … | Whole-cache preset. Native `fp8`/`bf16` encode K and V together, without a separate V mode. `fwhtN` is an optional headroom mode, not a default. |
| `--kv-k` / `--kv-v` | Qwen-only axis overrides; omitted when unset | Orthogonal to mode. On supported Qwen sites the legacy spellings `asymN` and `turboN` (bare `turbo` = `turbo3`) mean **`fwhtN`**; `legacy-asymN` selects the legacy Givens asym K. `asymN`, `turboN` and `legacy-asymN` are deprecated since 0.4.0 (removal in 0.5.0) and warn at load. V names: `q8`, `lloyd2`/`3`/`4`. Non-Qwen carriers refuse these axes before teardown. |
| `--max-seq` | int (512–1048576) when set; else automatic | On eligible growing Qwen VMM KV, default = **min(model trained context, measured card capacity)** after weights load; legacy and other owners retain their existing bounds. An explicit `--max-seq` replaces any configured `memory.max_seq` for that load (per process for `serve`). `bench` sizes its own load instead ([below](#bench-context-max_seq)). |

**Qwen `auto` / unset mode:** native `fp8` (both K and V) on exact `gfx1201` when the load is native-eligible — H24/Hkv4/D256 attention, single GPU (no PP/TP/EP), no `kv_adaptive`, no CASK sidecar; `q8`/`q8` everywhere else, including gfx1100 and gfx1151. Native modes report `kv_mode=fp8` or `bf16` in loaded/diag/bench JSON; no synthetic `V=q8` axis is reported. Non-Qwen family defaults are unchanged (e.g. Maple BF16, DeepSeek compressor F32, Gemma layered policy).

**K/V precedence** (split modes only): CLI flag **>** per-model config **>** global config **>** registry / arch default. Mode seeds the pair; `--kv-k` overrides K and `--kv-v` overrides V. An authored `--kv-mode fp8`/`bf16` refuses every axis override, including `--kv-v q8`; an auto-selected native pair can be replaced only when both axes are supplied. Full refusal rules: [CONFIG.md](CONFIG.md#kv-cache).

**Backend precedence** (forced choices): CLI `--kv-backend` **>** per-model `memory.kv_backend` **>** global `memory.kv_backend`. Omitted backend stays automatic VMM-preferring policy, not an authored pin.

### `hipfire serve` flags

| Flag / arg | Purpose |
|---|---|
| up to three positionals: `[model]` `[host]` `[port]` or `host:port` | Optional model, then bind (config defaults `127.0.0.1` and `11435`). **No TLS; authentication is optional** (`--api-key-file`) — `0.0.0.0` exposes the API on every interface; a key is recommended off loopback, and use only a trusted/firewalled network or a **TLS-terminating reverse proxy**. |
| `--api-key-file <absolute-path>` | Optional UTF-8 bearer key file (overrides `serve.api_key_file`); see [SERVE.md](SERVE.md#security-bearer-authentication-and-https-proxies). |
| `--model <tag>` | Model tag/path to pre-warm for this process. |
| `-d`, `--detach`, `--background` | Background; log `~/.hipfire/serve.log`, pid `~/.hipfire/serve.pid`. |
| `--kv-mode <m>` | KV mode preset for this process (same contract as `run`). |
| `--kv-backend <legacy\|vmm>` | KV allocation backend (same as `run`; `contiguous` rejected → use `legacy`). |
| `--kv-k <name>` / `--kv-v <name>` | Qwen-family K/V axis overrides (same as `run`). |
| `--max-seq <n>` | KV context for every ordinary model load of this service (512–1048576); replaces the configured/automatic `memory.max_seq` (same contract as `run`). With `serve.multi_slot` it sets the per-slot context (overriding `serve.multi_slot_ctx`), so total KV scales with the slot count (`serve.multi_slot_slots`). Forwarded by `serve -d`. |
| `--spec <m>` / `--speculation <m>` | Speculation mode for every model this service loads: `off` \| `auto` \| `ngram` \| `dflash` \| `mtp` \| `dspark` (other values fail at parse). Applied per load after the config selector, so it overrides `speculation.mode` and per-model speculation; `dflash` / `auto` also resolve the registry draft. `--spec off` turns MTP and DFlash off (plain AR), e.g. for an A/B. Unset: config (default `auto`). Forwarded by `serve -d`. |
| `--idle-timeout <s>` | Unload after idle seconds (`0` = never; max `86400`). |
| `--no-prewarm` | Do not load the configured default model before accepting requests (lazy-load on first request). |
| `--tp N` | Tensor/expert-parallel degree, `1..=64` (supported paths only). |
| `--continuous-batch-size N` | Number of eligible concurrent batched lanes, `1..=256` (`1` preserves sequential behavior), subject to memory admission; same range as the `serve.continuous_batch_size` config key. The runtime batch width is dynamic, and the 128/63 chunk sizes are not lane limits. Forwarded by `serve -d`. |
| `--ui` | Serve the embedded chat UI at `/ui` on the serve listener. |
| `--open` | Open the chat UI in the default browser once serving; implies `--ui`. With `-d` the browser launch happens in the launching process and the detached child only enables the route. |
| `--vision <path>` | Vision-tower sidecar wired into every model load of this process. Skipped while `vision_mode=off` (default). **Foreground serve only:** `serve -d` does not forward it to the detached child, so set `HIPFIRE_VISION_SIDECAR` or the registry `vision` slot for a detached service. |

### `hipfire img` flags

Run in a fresh daemon process; the prompt is the positional text (a built-in sample prompt when omitted). `--steps`
defaults to the model's own default (4 for step-distilled `flux.schnell:1`,
28 for guidance-distilled `flux.dev:1`).

| Flag | Purpose |
|---|---|
| `-o, --out <path>` | Output PNG path (default `hipfire-<seed>.png` in the working directory). |
| `--width` / `--height` | Pixel size, `1..=8192` and divisible by the VAE compression factor. Default 1024×1024; with `--image` the default is the reference image's own size. |
| `--steps <n>` | Denoise steps, `1..=128`; omit for the architecture default. |
| `--seed <n>` | Noise seed (default `0`). Same model + seed → byte-identical PNG. |
| `--backend <cpu\|gpu>` | `gpu` = the HIP transformer forward, `cpu` = the f32 reference oracle; defaults to `gpu` when available, else `cpu`. Other values are rejected. |
| `--image <path>` | Reference image for FLUX.2 Klein edit (arch 45 only; a FLUX.1 pipe refuses it). Repeatable, up to 4. Give every `--image` **before** the prompt words: the prompt is a greedy trailing positional. The CLI reads each file and sends its bytes; the daemon never opens a client-named path. |
| `-j, --json` | Print one JSON result object (`path`, `bytes`, `width`, `height`, `steps`, `seed`, `ms`, `model`) instead of the progress lines and the PNG path. |

Model resolution accepts an HFQ **component pack** (trunk file) — the
`t5`/`clip`/`vae` sidecar packs are discovered next to it by sibling name
(`<stem>-t5.hfq` etc., or shared `t5-xxl.hfq` / `clip-l.hfq` / `vae.hfq`).
`hipfire pull flux.schnell:1` fetches the trunk + all three sidecars at once.

### `hipfire embed` flags

Runs in a fresh daemon process. The production target is an embedding
checkpoint loaded as an **HF safetensors directory** (e.g. a
`pplx-embed-v2-late-0.6b` snapshot), or a raw-dtype `.hfq` built from one
with `hipfire-quantize` (the quantizer keeps the norms, the 128-d Dense head
and the vision tower raw, folds `1_Dense/model.safetensors` into the trunk,
and auto-includes the tower).

EmbeddingGemma-2 (arch 18) snapshots — e.g. a `google/embeddinggemma-2`
directory, whose quantized form writes a `.vl` tower sidecar — are served by
the same command. They answer with **one pooled vector** (`n_tokens: 1`)
instead of arch 17's token matrix, and `--image` / `--audio` / `--video` reach
their vision and audio towers (video drives the vision tower frame by frame)
when the sidecar is resident.

| Flag | Purpose |
|---|---|
| `--query <text>` | Query-side text (its marker is `"[Q] "`). |
| `--doc <text>` | Document-side text (marker `"[D] "`); with `--query`, prints the MaxSim score. |
| `--kind <query\|document>` | Side for the positional text; default `document`. Sets the marker and truncation limit. |
| `--image <path>` | Encode an image (PNG/JPEG) with the checkpoint's vision tower. Repeatable — the images are appended after the text, in flag order; text and images may be combined. An arch-18 checkpoint without its `.vl` sidecar, or a host where `vision_mode` is off, refuses the image instead of dropping it. |
| `--audio <path>` | Encode an audio clip with the checkpoint's audio tower (arch 18). Repeatable — clips are appended after the text and images, in flag order. Takes 16 kHz mono/stereo **WAV/RIFF** (PCM16 or float32); any other container is refused with the `ffmpeg` conversion to run. |
| `--video <frame>[,<frame>…]` | Encode a video with the checkpoint's vision tower (arch 18), which runs once per frame. **One clip per flag**: give that clip's decoded **frames**, comma separated, in display order — the daemon applies the video processor's own sampler (`max_frames` 32: ≤32 frames are all used, more are strided uniformly) itself, so pass a clip at whatever rate you extracted it. Repeat the flag for several clips; each clip is spliced at its own `<|video|>` marker (one marker per video, never per frame). Frames are PNG/JPEG; a container (MP4/MKV/…) is refused with the `ffmpeg` extraction to run. |
| `-j, --json` | Print the full JSON result — the `tokens×dims` matrix for a single encode, or `{score, query_tokens, doc_tokens, dim}` for a pair. |

Token limits are per side (`query_length` 1024, `document_length` 4096 on the
0.6B); text is truncated from the end with the marker always kept. The
document side drops punctuation token vectors (the ColBERT skiplist); the
query side keeps them. Scores are late interaction (`Σ max dot`), so they are
not cosine similarities and are only meaningfully comparable between the same
query across documents.

```bash
hipfire embed ~/.hipfire/models/pplx-embed-v2-late-0.6b \
  --query "what is the capital of France" --doc "Paris is the capital of France"
hipfire embed <model> "encode this document" --kind document --json
hipfire embed <model> --image chart.png "the caption" --json
```

The same surface is exposed over HTTP as `POST /v1/embeddings` (OpenAI-shaped;
the `embedding` array is the flat `n_tokens × dim` matrix and each data object
carries `n_tokens`, with an optional `"kind": "query" | "document"`). `input`
takes a string, a one-element list, the object form (`{text, image_base64}`, or
`{text, images_base64: […]}` for several — the same spelling the JSONL `embed`
verb takes), or the typed content array
(`[{"content": [{"type": "text", "text": …}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}]}]`)
— one content object per request, text parts concatenated in order, image
parts decoded at the route. A bare content-part array without the
`{"content": …}` wrapper is refused with the wrapper it should have used.
Interleave media by putting one `<|image|>` marker per image in the text
(arch 18); a text with no marker gets its image runs appended before EOS, and
a marker count that disagrees with the image count is refused. Audio takes the
OpenAI `input_audio` part
(`{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,…", "format": "wav"}}`,
WAV/RIFF only, `format` optional) and the JSONL `audios_base64` list, with one
`<|audio|>` marker per clip under the same count rule; a text with no marker
gets its audio runs appended after the image runs. Audio needs the
checkpoint's audio tower (the same `.vl` sidecar as vision): a vision-only load
refuses a clip by name rather than encoding the text alone.

Video takes the llama.cpp-shaped `input_video` part
(`{"type": "input_video", "input_video": {"data": ["data:image/png;base64,…", …]}}`,
one entry per frame in display order, `format` optional and PNG/JPEG only) or
the JSONL `videos_base64` field — **one frame list per video**, e.g.
`{"text": "… <|video|> …", "videos_base64": [[frame0, frame1, …]]}`. One
`<|video|>` marker per video under the same count rule, and the daemon samples
the frames itself (the video processor's `max_frames` 32 budget, uniform
stride) before running the vision tower once per frame; a frame that is not a
PNG/JPEG image is refused with the `ffmpeg` extraction to run.

## Configuration

| Command | Purpose |
|---|---|
| `hipfire config` | Global interactive TUI → `~/.hipfire/config.toml`. |
| `hipfire config <tag>` | Print resolved per-model policy and override provenance. |
| `hipfire config list\|get\|set\|reset ...` | Scriptable global ops (`-j/--json` on list/get; `reset` with no key removes every override). |
| `hipfire config explain <key> [-j]` | Explain a key's type, scope, default, effective value and provenance. |
| `hipfire config schema [-j]` | Print the authoritative typed configuration schema. |
| `hipfire config migrate` | Convert legacy `config.json` to sparse `config.toml` without deleting the JSON. |
| `hipfire config profile [set <name>\|create <name>]` | `set` replaces the global sparse config with a built-in (`default`, `dev`, `hip`, `redline`) or custom profile; `create` snapshots the current global config as a new custom profile; with no action it opens the profile wizard in `hipfire-tui`. `migrate`, `schema` and `profile` are global-only. |
| `hipfire config <tag> list\|get\|set\|reset ...` | Same, scoped to per-model keys. |

Do not inventory every key here — [CONFIG.md](CONFIG.md) owns defaults and ranges. Notable defaults from source: `dflash_mode=off` (schema default; Qwen3.8-27B tiers with a paired draft resolve `auto` through the registry layer), `speculation=auto`, `thinking=on`, `reasoning_effort=auto` (no preference: an explicit non-`auto` config/CLI/HTTP value wins; when every layer is unset/`auto`, Qwen3.8 and Ornith 1.5 use hipfire's `medium` default, so `reasoning_effort = "xhigh"` restores the official-template default), schema default `thinking_budget=med` (legacy named-cap route only; dropped+warned on effort-native models), `max_tokens=4096`, `idle_timeout=300`. Qwen Jinja contracts additionally accept an explicit integer think cap; other families do not synthesize a force-close mechanism. Reasoning axes and family examples: [CONFIG.md](CONFIG.md), [SERVE.md](SERVE.md).

## Quantization and calibration

| Command | Purpose |
|---|---|
| `hipfire quantize <hf-id\|dir\|file.gguf> [flags]` | CPU quantize via `hipfire-quantize`. GGUF weight input is deprecated since 0.4.0 (removal in 0.5.0): GGUF→mqN is lossy double quantization; quantize from the original safetensors checkpoint instead. GGUF input accepts only `hf4`, `hf6`, `mq4` and `mq6`. |
| `hipfire-quantize --flux-pipe <pipe_dir> --output <base.hfq>` | Pack a FLUX.1 or FLUX.2 Klein diffusers pipe into per-component HFQ files (`<base>-transformer.hfq` plus `-t5.hfq`, `-clip.hfq`, `-vae.hfq` for FLUX.1, or `-qwen3.hfq`, `-vae.hfq` for Klein; arch ids 40–46). `--flux-component` packs one. The packs are the only form the daemon loads. |
| `hipfire sidecar-gen <model> [flags]` | **Deprecated since 0.4.0 (CASK), removal in 0.5.0.** Build a `.triattn.bin` next to the model (does not pull); prints a deprecation warning. |
| `hipfire calibrate-npu-trunk --from <measurements.json> --out <cal.json>` | Write a keyed NPU trunk calibration from offline measurement inputs. Every measured cost is mandatory; the record's `key.kernel.uname_release` must equal the current `uname -r`, otherwise the command fails. |

### `quantize` (summary)

| Flag | Purpose |
|---|---|
| `--format <fmt>` | Repeatable: `mq4`, `mq6`, `q8`, `q8f16`, `hf4`, `hf6` (`hfq4`/`hfq4g256` map to `hf4`, `hfq6`/`hfq6g256` to `hf6`). With no `--format` and no `--both` the default is **`mq4`** for safetensors input and **`hf4`** for a `.gguf` file. |
| `--both` | Add `mq4` and `mq6` to the requested formats (suppresses the default format). |
| `-o` / `--output` | Exact output path; valid with exactly one format. |
| `--output-dir`, `--stem` | Output directory (default: working directory) and filename stem (default: the input's file/directory name, `.gguf` removed); files are `<stem>.<format>`. |
| `--install` | Copy into `~/.hipfire/models/`. |
| `--register <tag>` | Local alias in `~/.hipfire/models.toml`, pointing at the `mq4` artifact when produced, else the first one. |
| `--upload <owner/repo>`, `--create-repo` | Optional HF publish through the `hf` CLI (`--create-repo` creates the model repo first). |

Supported CLI formats include `mq4`, `mq6`, `q8`/`q8f16`, `hf4`/`hf6` and hfq aliases. Graded MoE recipes need the quantizer binary directly — [QUANTIZE.md](QUANTIZE.md).

### `sidecar-gen` (summary; deprecated since 0.4.0, removal in 0.5.0)

| Flag | Purpose |
|---|---|
| `--corpus PATH` | Calibration text (else builtin seeds). |
| `--max-tokens N` | Default `4000`. |
| `--chunk-len N` | Default `256`. |
| `--gpu-calib` / `--cpu-calib` | Device (GPU is the usual default path). |
| `-o PATH` | Output path (default `<model-path>.triattn.bin`). |
| `--skip-validation` | Skip post-gen validation. |

## Benchmarks and diagnostics

| Command | Purpose |
|---|---|
| `hipfire bench <model> [opts] [prompt]` | Prefill/decode timing. `--runs N` (default 5), `--json`, `--exp` (RDNA2 variant sweep), `--max-seq N` ([context](#bench-context-max_seq)). `--prompt-file PATH` reads the prompt verbatim; JSON records `prompt_tokens`/`prompt_md5`/`prompt_chars`/`warnings` (short prompts warn that `prefill_tok_s` is launch overhead), the load's `max_seq`, and the GPU power and memory state of every measured run ([below](#bench-power-and-memory-state)). |
| `hipfire bench <model> --matrix ...` | Synthetic PP/context/TG matrix (`--pp`, `--ctx`, `--tg`, `--sustained-tg`, `--sustained-ctx`, `--warmups`, `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--max-seq`, `--redline`). Same KV contract as `run`/`serve`, except that the load's `max_seq` is sized to the matrix ([context](#bench-context-max_seq)); `--json` surfaces the effective loaded `kv_mode` and backend fields, and each row carries a `power` entry per run. |
| `hipfire bench <model> --spec <m> ...` | Every bench generate sends greedy sampling with the neutral `repeat_penalty 1.0` (any other value routes the daemon to the plain AR decoder), so `--spec mtp\|dflash\|ngram\|dspark\|auto` measures the speculative decoder and `--spec off` the AR decoder on the same workload. JSON records `spec_requested`, a per-run `spec_route` (`"mtp"`, `"mtp+ngram"`, `"dflash"`, a daemon `spec` name, or `null` when AR ran) and per-run `spec_tau`; when `--spec` is not `off` but no run speculated, the bench prints a `warning:` and adds it to `warnings`. |
| `hipfire bench <model> --concurrency 1,2,4,8 ...` | Closed-loop serving sweep over the daemon wire: at each K, K lanes drain the same N requests, after a discarded warmup at that K. Reports output tok/s, steady-window tok/s, TTFT/TPOT p50/p99, achieved concurrency, τ, finish reasons and routes ([below](#bench-concurrency---concurrency)). `--json` is not supported on this path. |
| `hipfire bench <model> --emu <id\|off> ...` | SKU application-capacity profile for **every** spawned bench daemon arm (standard, `--matrix`, `--ttft`, `--concurrency`, `--exp`): sets `developer.emu` in each child's process snapshot, overrides `HIPFIRE_EMU` including `--emu off`, validates the id before anything spawns, and logs the effective value on stderr (`emu: effective profile ...`; silent when no profile is requested). The host still runs at full speed; only application memory capacity is capped. With a profile active the bench JSON `gpu` object (the daemon `diag`) gains an `emu` object (budgeted free/total, ledger); the `vram_free_mb`/`vram_total_mb` beside it stay physical. See [EMU.md](EMU.md). |
| `hipfire emu list [-j\|--json]` | Print the SKU emulation registry: id, host gfx, target gfx, target CUs, memory, peak bandwidth, IC/L2, VGPR file, fidelity class. CPU-only. |
| `hipfire emu validate <id> [-j\|--json] [--mib N ...]` | Start a fresh model-free daemon with `developer.emu=<id>` and run the `emu_validate` verb (finite eager/graph traffic and topology checks, ledger checks; `--mib` buffer sizes, 1..=2048, repeatable). Prints a summary plus the pretty receipt, or one raw JSON line with `--json`. Exits nonzero when the capacity, host-topology or graph section fails (or the receipt status is `fail`); `unavailable` is reported, not failed. The daemon refuses a request whose id differs from the installed profile, or a loaded model. Requires a compatible host (see [EMU.md](EMU.md)); GPU runs belong in the fabric scheduler. |
| `hipfire emu estimate <id> --input FILE [-j\|--json]` | CPU-only derating **estimate** of the profile's target from a measured `DeratingInput` JSON (schema v1: run identity, host/target resource rates, per-kernel observations). Never starts a daemon, never touches HIP. The report keeps `measured` (input), `estimate` (model output; `estimated_wall_ns` is `null` with reasons when time closure, rates or identity are missing) and `target_profile` (registry metadata) apart. Exit 0 also when the estimate is `unavailable`; nonzero for unreadable or malformed input. |
| `hipfire profile [model] [--kernel substr] [--json]` | Live daemon roofline and compiled-kernel VGPR/SGPR/LDS/occupancy report. Use `hipfire-atlas` for measured ISA-fit and workload analysis. |
| `hipfire diag [-j\|--json]` | Static device/runtime checks plus a live HIP arch, version, and VRAM probe when the daemon is available. |
| `hipfire --version` | Concise semver + build commit + source ref identity. |
| `hipfire version [--json]` | Detailed build/source match, checkout state, target, config schema, and installed-daemon SHA-256. |
| `hipfire update [REF]` | **Linux managed installs:** advance the current branch or auto-detect and install a branch/tag/commit; a leading `@` on `REF` is optional. |
| `hipfire update --branch NAME` | Switch to and track a remote branch such as `master` or `beta`. |
| `hipfire update --tag TAG` / `--commit SHA` | Install an immutable detached revision. A later explicit selector moves away from the pin. |
| `hipfire setup --source PATH [flags]` | Install or repair this machine's runtime from a source checkout (normally run by `scripts/install.sh`; `--source` is required and must contain `Cargo.toml`). Flags: `--rocm-root PATH`, `--hipcc PATH` (also `HIPFIRE_HIPCC`), `--strict-rocm` (also `HIPFIRE_ROCM_STRICT=1`), `--gpu-arch ARCH`, `--profile auto\|hip\|redline` (`auto` leaves `replay.backend=auto` so `.mq4r` models select Redline), `-y`/`--yes` (alias `--non-interactive`), `--compile-kernels` (compile locally with `hipcc` even when a release kernel pack exists), `--kernel-pack-url URL` (directory or URL holding the kernel-pack assets; default: the tag's GitHub release). A release tag selects that release's prebuilt kernel pack, which needs no device compiler. |
| `hipfire kernel-pack install --tag TAG --arch ARCH --source PATH [--dest DIR] [--url URL] [--rocm-version V]` | Download, verify and install the prebuilt kernel pack for `TAG` and `ARCH`. `--source` is the checkout the installed daemon was built from; its `HEAD` must be the pack's commit. `--dest` defaults to `~/.hipfire/bin/kernels/compiled/<ARCH>`; `--rocm-version` (default: the resolved ROCm root's `.info/version`) is checked against the pack's supported range. |

Perf claim protocol (warmup, fresh-process, noise): [methodology/perf-benchmarking.md](methodology/perf-benchmarking.md). Published tables are measured/historical: [BENCHMARKS.md](BENCHMARKS.md).

`hipfire bench` flags (summary; `hipfire bench --help` is authoritative):

| Flag | Purpose |
|---|---|
| `--runs N` | Measured runs; default `5`, must be positive. |
| `--tp N` | Tensor/expert-parallel degree for the bench daemon, `1..=64`. |
| `-j`, `--json` | JSON report; not supported with `--exp` or `--concurrency`. |
| `--exp` | RDNA2 kernel-variant sweep in isolated daemon processes; cannot combine with `--matrix`/`--redline`, `--json` or `--reasoning-on`. |
| `--matrix` / `--redline` | Synthetic token-id rows; neither takes a prompt. `--redline` additionally selects the Redline replay backend (PM4 transport, graph AR/forward). |
| `--pp`, `--ctx` | Comma lists for the matrix; default `128,512,2048,4096,8192,20000`. |
| `--tg N` | Matrix decode length; default `128`. |
| `--sustained-tg N`, `--sustained-ctx` | Sustained-decode rows run when `--sustained-tg` is set; `--sustained-ctx` defaults to `128,8192`. |
| `--warmups N` | Warmup iterations before the measured runs of `--ttft` and `--matrix`; default `10`. |
| `--ttft` | Client-side time to first token: per run, generate one token on the standard prompt and time request-send to first streamed token; reports `prompt_tokens` and TTFT. Standard path only (not with `--matrix`). |
| `--max-tokens N` | Generated tokens per measured request: default `128` on the standard bench, `256` on the `--concurrency` `steady` shape. |
| `--reasoning-on` | Let a reasoning model think during the bench (off by default, because a think span usually cannot close inside the benchmark's token budget and the daemon then fails that turn closed); pair with a large enough `--max-tokens`. |
| `--spec <m>`, `--kv-mode`, `--kv-backend`, `--kv-k`, `--kv-v`, `--max-seq` | Same contracts as on `run`/`serve`. |
| `--prompt-file PATH` | Read the standard-bench prompt verbatim (raw bytes); mutually exclusive with positional prompt words. |
| `--emu <id\|off>` | SKU capacity-emulation profile for every spawned bench daemon; overrides `HIPFIRE_EMU`, including `off`. Unknown ids and UMA-knob conflicts fail before any daemon starts. See [EMU.md](EMU.md). |
| `--concurrency`, `--backend`, `--workload`, `--bench-shape`, `--num-prompts`, `--temperature` | Concurrency sweep controls ([below](#bench-concurrency---concurrency)). |

### Bench concurrency (`--concurrency`)

`hipfire bench <model> --concurrency 1,2,4,8 [--backend batch|noslots|slots|both] [--workload stateless|multiturn|both] [--bench-shape <shape>] [--num-prompts N] [--temperature T] [--reasoning-on] [--runs N]` measures serving throughput with a closed-loop sweep at each concurrency level K (`--bench-shape` defaults to `steady`; `hipfire bench --help` lists the other shapes and the flags they need):

- **Closed loop at K.** K client lanes drain one FIFO of N requests. A lane sends its next request the moment its previous one ends, so there is no burst barrier and no lane idles until the queue is empty. Request i is prompt `i mod len` of the shape's list, so every K sends the same multiset in the same order.
- **Warmup discarded.** Before every sample, 8 separate warmup prompts run closed-loop at the same K with 64-token outputs. No measured prompt is ever a warmup prompt.
- **No prefix-cache hits.** Each request's first message starts with a `Q<run>-<i>.` tag, unique per `--runs` round and request. Every K of a round therefore sends the same tokens, and a later round's repeat never hits the daemon's prompt cache. `cached` reports any prompt tokens the daemon served from cache.
- **Chat turns, neutral sampling.** Requests carry `messages` (plus `prompt.system`, as `hipfire serve` injects it), thinking off unless `--reasoning-on`, and neutral penalties (`repeat_penalty` 1.0, `presence_penalty` 0, `min_p` 0).
- **Levels are interleaved** across `--runs` (`1,2,4,8,1,2,4,8,…`), and the table reports each metric's median over the runs.

| Backend | What it loads and sends |
|---|---|
| `batch` | The VMM continuous-batching route `hipfire serve` runs with `serve.vmm_batch` on: `continuous_batch_size` = max K (at least 2) plus the `serve_vmm_batch` load object built from `serve.batch_spec`, `serve.batch_nonexact`, `serve.max_batch_tokens` and `serve.prefill_min_tokens`. Requests are the chat requests serve forwards, with `serve_continuous_batch`. The driver refuses a daemon that did not stage `continuous_batch_route: "vmm"`. A lone request takes the singleton route, as under serve. |
| `noslots` | No batching. The daemon serves one request at a time and the other lanes wait client-side, as serve queues them with batching off. The wait counts in TTFT. |
| `slots` | The experimental multi-slot engine (`experimental_multi_slot`). |
| `both` | All three, one after another. Only one daemon is resident at a time. |

The `multiturn` workload makes each request a two-turn chat. The follow-up ("Explain that again in one sentence.", with the first answer as the assistant turn) is sent on the same lane as soon as the first turn completes, and both turns count. On `slots`, a sample whose follow-ups recorded no prefix-cache hit is reported `[invalid]` and dropped.

| Shape | Requests | Output length | Sampling |
|---|---|---|---|
| `steady` (default) | 64 long-answer code prompts built into the binary (`--num-prompts N` cycles them), measured-set md5 `f0b077a766edc166a0c3f18cf81c169c` | `--max-tokens`, default 256 | greedy |

`--temperature` overrides the shape's temperature (default `0` for `steady`). The daemon has no `ignore_eos`, so `steady` fixes the output length by prompt choice: its prompts ask for answers longer than 256 tokens. Check that the `finish` column reads `length` only. Sampled requests (temperature > 0) are not batched: the VMM route serves greedy requests only, so a sampled sweep measures the singleton route queued K deep.

Each sample prints one line on stderr: output tok/s, steady tok/s, TTFT and TPOT p50/p99, in-flight, τ, ok/total, tokens and duration, finish reasons, routes (`execution_mode/batched=`), cached prompt tokens, mean prompt tokens and the md5 of the decoded texts. The final table has one row per (backend, workload, K):

- **out tok/s**: output throughput, output tokens ÷ (last completion − first send). The median of the runs, with min–max in brackets.
- **steady tok/s**: tokens whose arrival falls in the steady window ÷ the window. The window runs from the moment all of the first K requests have a token to the first completion after the last send.
- **TTFT**: first token − send. **TPOT**: (completion − first token) ÷ (tokens − 1). Both are the medians of the per-sample p50/p99.
- **in-flight**: achieved concurrency, Σ request latency ÷ duration.
- **τ**: output tokens per verify window, token-weighted over the sample.
- **tokens**: median output tokens per sample.
- **finish**, **batched**, **cached**, **errors**: summed over all samples. A failed request counts no tokens.

A daemon that sends no frame for 300 s fails the sweep. Its in-flight requests are cancelled first.

### Bench context (`max_seq`)

`--max-seq N` sets the benchmark load's KV context, within the `memory.max_seq` bounds (512–1048576), and replaces any configured `memory.max_seq`.

Without it, `--matrix` and `--redline` size the load to the run instead of using the configured or automatic context. The run needs the largest `--pp`, or the largest `--ctx` plus `--tg` (and `--sustained-ctx` plus `--sustained-tg` when sustained rows run), whichever is larger, plus the 32 positions of headroom the daemon's probes keep. That is raised to the `memory.max_seq` minimum of 512 and capped at the model's declared context (`max_position_embeddings` in its metadata). `--pp 8192 --ctx 128 --tg 32` loads `max_seq` 8224.

- A shape past the model's declared context fails before the daemon starts, naming the run's need and the model's limit. This applies with or without `--max-seq`.
- A `--max-seq` below a matrix run's need fails the same way, naming both numbers.
- Past the declared context, a `--max-seq` that holds the run is sent as given, and the daemon's admission is the bound. For example, Qwen4 refuses anything above its admitted 262144.

The standard, `--ttft`, `--exp` and `--concurrency` paths have no fixed shape, because the prompt is tokenized by the daemon. Without `--max-seq` they keep the configured or automatic context.

Every JSON report has a `max_seq` object:

- `effective`: what the daemon loaded.
- `requested`: the value sent, or `null`.
- `source`: `flag`, `run_shape`, `config` (an authored `memory.max_seq`) or `automatic`.
- `run_need`: the matrix need, or `null`.
- `model_limit`: the declared context, or `null`.

The bench also prints a `max_seq:` line.

### Bench power and memory state

`hipfire bench` samples the GPU every 0.5 s during each measured run (`--ttft` and `--matrix` too; not `--exp` or `--concurrency`). It reads only world-readable sysfs, so it needs no root, and it never fails a bench: whatever it cannot read is reported as the string `"unavailable"`.

- **`power_device`** (once per report): the card sampled, `card` (`/sys/class/drm/cardN`), `pci`, `arch`, `selected_by` and `sample_period_ms`. It is the daemon's logical device 0: the card named by `hardware.devices` (`HIPFIRE_DEVICES`), else the first entry of an inherited `ROCR_VISIBLE_DEVICES`, else the first ROCr agent of the daemon's arch. With `--tp` above 1 only that first card is sampled. If no card resolves, it holds `status: "unavailable"` and a `reason`.
- **`power`**: one object per measured run, in the order of the `samples` arrays (top level for the standard and `--ttft` paths, per row for `--matrix`):

| Field | Source | Meaning |
|---|---|---|
| `samples`, `duration_s` | — | Samples taken (one at run start, every 0.5 s, one at run end) and the span they cover. |
| `gpu_metrics_format` | `gpu_metrics` header | `"3.0"` is decoded (Strix Halo and other `gpu_metrics_v3_0` APUs). Any other `format.content` (e.g. `"1.3"` on RDNA dGPUs), `"malformed"`, or `"unavailable"` (no file) leaves every `gpu_metrics` field below `"unavailable"`. |
| `gfxclk_mhz` `{avg,min,max}` | `gpu_metrics` | `average_gfxclk_frequency`. |
| `gfx_maxfreq_mhz_min` | `gpu_metrics` | Lowest `current_gfx_maxfreq` the SMU allowed during the run. |
| `socket_power_w`, `gfx_power_w` `{avg,max}` | `gpu_metrics` | `average_socket_power`, `average_gfx_power`. |
| `gfx_temp_c_max` | `gpu_metrics` | Highest `temperature_gfx`. |
| `fclk_mhz_avg` | `gpu_metrics` | Mean `average_fclk_frequency`. |
| `throttle_pct` | `gpu_metrics` | Percent of the run's wall time at each limiter: `fast_ppt`, `slow_ppt`, `stapm` (the SMU's SPL counter), `thm_gfx`, `thm_core`, `thm_soc`, `prochot`. The SMU counters are cumulative milliseconds; the percentage is their increase between the first and last sample over the time between them, capped at 100. |
| `tctl_c_max` | k10temp hwmon | Highest CPU package `Tctl` (meaningful on APUs, where the GPU shares the package). |
| `perf_level` | `power_dpm_force_performance_level` | Read at run start. |
| `od_sclk` `{min_mhz,max_mhz,range_min_mhz,range_max_mhz}` | `pp_od_clk_voltage` | The `OD_SCLK` levels and the `OD_RANGE` SCLK bounds, at run start. |
| `hwmon` | the card's amdgpu hwmon | Every labelled input, keyed by its label: `temp_c` (max; e.g. `edge`, `junction`, `mem`), `power_w` (`powerN_average`, else `powerN_input`; e.g. `ppt`) and `freq_mhz` (e.g. `sclk`, `mclk`), with avg/min/max. This is all that remains on a card whose `gpu_metrics` format is not decoded. |
| `memory` | `mem_info_{vram,gtt}_{total,used}`, `/proc/meminfo` | At run start, in MiB: `vram_total_mib` (the UMA carve-out on an APU), `vram_used_mib`, `vram_free_mib`, the same three for `gtt`, and the host's `mem_total_mib` and `mem_available_mib`. Shrinking the BIOS carve-out and serving from GTT changes these, so tag baselines with them. |

`gpu_metrics` values are the SMU's time-filtered averages, so the first second of a run lags the load.

When any limiter's residency exceeds 10 % of a measured run, the bench prints one `warning:` line naming the worst residency per limiter, and adds it to the JSON `warnings`. The number is then power- or thermal-limited and is comparable only with runs whose power fields match.

**High-TDP APU thermal note.** On a high-TDP Strix Halo box at stock power limits, long prefill (e.g. pp8192) can be limited first by fast PPT and then by GPU temperature, which reaches the thermal point within seconds of prefill start. Capping the four SMU limits (STAPM, fast PPT, slow PPT, APU slow PPT) to a common value around 130 W (`ryzenadj --stapm-limit=130000 --fast-limit=130000 --slow-limit=130000 --apu-slow-limit=130000`, or the BIOS power mode) can keep the GPU below its thermal limit at some cost in burst prefill throughput; lower caps trade more throughput for energy efficiency. The effect depends on chassis, cooling and firmware, so measure the trade-off on your own machine. The power limits need root and are not visible to the bench, so record them with the result.

## Where files live

| Path | Role |
|---|---|
| `~/.hipfire/bin/` | native `hipfire`, `daemon`, optional `hipfire-tui`, tools |
| `~/.hipfire/bin/kernels/compiled/<arch>/` | Precompiled / JIT kernels |
| `~/.hipfire/models/` | Weight files and co-located sidecars |
| `~/.hipfire/config.toml` | Sparse typed global config (`config.json` is migration input) |
| `~/.hipfire/models.toml` | Local aliases, paths, registry identities, and per-model overrides |
| `~/.hipfire/models.json` | Legacy catalog migration input |
| `~/.hipfire/per_model_config.json` | Legacy overlay; folded into the catalog on refresh |
| `~/.hipfire/registry.cache.json` | Dynamic registry cache |
| `~/.hipfire/serve.pid` / `serve.log` | Detached serve pid record and log |
| `~/.hipfire/hf-cache/` | HF download cache used by quantize |
| `~/.hipfire/src/` | Source tree used by `update` / some installs |

Override catalog path with `HIPFIRE_MODELS_CATALOG_PATH` if needed.

## Dynamic registry

1. Bundled `registry/v1.json` (embedded in the binary).
2. Optional network fetch of `registry/v1.json` → 24h cache.
3. Invalid remote payloads are rejected wholesale.
`hipfire diag` reports which source the process used. Entry metadata may include size, `arch_id` ([architecture-ids.md](architecture-ids.md)), quant, and hashes — registry presence is not a runtime admission ([INDEX.md](INDEX.md), [admissions.yml](admissions.yml)).

## Useful environment overrides

Single-invocation knobs (non-exhaustive; full list in [env-vars.md](env-vars.md)):

| Variable | Effect |
|---|---|
| `HIPFIRE_LOCAL=1` | `run` skips HTTP serve and spawns a local daemon (also forced by load-time overrides such as `--kv-mode` or `--image`). |
| `HIPFIRE_HOME=...` | Override the state/config root (default `~/.hipfire`). |
| `HIPFIRE_MODELS_DIR=...` | Override model discovery, pull, list, and TUI model paths. |
| `HIPFIRE_KV_MODE=...` | Override KV layout preset (`auto` → Qwen fp8 on native-eligible exact gfx1201, else q8/q8; non-Qwen unchanged). |
| `HIPFIRE_KV_BACKEND=...` | Not a second default plane for ordinary loads; the stable **warning token** `HIPFIRE_KV_BACKEND=legacy` appears on stderr when legacy storage is selected. Config/CLI use `legacy`\|`vmm` only. |
| `HIPFIRE_SPECULATION=...` | Speculation selector from the environment: overrides per-model and global config; a CLI `--spec` still wins. |
| `HIPFIRE_DFLASH_DRAFT=...` | Explicit draft path. |
| `HIPFIRE_VISION_SIDECAR=...` | Explicit vision-tower sidecar path; empty opts out. Skipped while `vision_mode=off`. |
| `HIPFIRE_VISION_MODE=...` | Tower sidecar gate: `off` (default) / `auto` / `on`. |
| `HIPFIRE_DFLASH_MODE=...` | Daemon-side mode (CLI default config is still `off`). |
| `HIPFIRE_NO_REGISTRY_FETCH=1` | Pin bundled registry. |
| `HIPFIRE_REGISTRY_URL=...` | Alternate registry URL. |
| `HIPFIRE_NORMALIZE_PROMPT=0` | Opt out of `\n{3,}` collapse (config `prompt_normalize` default on). |
| `HIPFIRE_DEBUG=1` | Full stacks on CLI failure. |
| `--no-color` (chat) | Accepted by `hipfire chat` for compatibility; the line client never emits ANSI color. |

## Errors and next steps

| Problem | Action |
|---|---|
| Unknown command | `hipfire help` |
| Unknown config key | `hipfire config set` prints valid keys; see [CONFIG.md](CONFIG.md) |
| Model missing | `hipfire list` / `pull` |
| Serve already running | `hipfire stop` or `restart` |
| Stale pid / port busy | `hipfire stop --force`; inspect `serve.log` |
| `update` on non-Linux | Re-run `install.ps1 -Branch/-Tag/-Commit` or use a source build |
| `update` from a detached pin | Choose the next target explicitly, e.g. `hipfire update @beta` |
| Need HTTP API detail | [SERVE.md](SERVE.md) |
| Need chat UX | [CHAT.md](CHAT.md) |
| Need onboarding | [GETTING_STARTED.md](GETTING_STARTED.md) |
