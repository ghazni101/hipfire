---
name: hipfire-quantize
description: Use when converting HuggingFace safetensors or a local checkpoint into hipfire HFQM weights with `hipfire quantize` or `hipfire-quantize` (MQ4/MQ6/MQ3/HFQ/Q8/HFP/MFP, product tiers, PLE split, Qwen4 Flash-Next, vision sidecars). Use for writing `.mq4`/`.hfq` artifacts — not Astrea KLD/PPL calibration.
---

# hipfire-quantize

Operator playbook for **writing** HFQM artifacts. The thin CLI is
`hipfire quantize`; the CPU binary it wraps is `hipfire-quantize`.
Quality judgment, imatrix join, KLD/PPL, and promotion policy live in
the [astrea](../astrea/SKILL.md) skill (`python3 scripts/astrea.py`).

**Use installed bins from `~/binaries/hipfire-bin/`**, not a fresh
`cargo build` and not a random `target/release/` unless that directory is
empty. Prefix `PATH` (or invoke the absolute path) so `hipfire`,
`hipfire-quantize`, `hfq`, `mtp_extract`, and `dflash_convert` all resolve
there:

```bash
export PATH="$HOME/binaries/hipfire-bin:$PATH"
# or: ~/binaries/hipfire-bin/hipfire-quantize --help
```

Mutable format math and the full flag inventory live in canonical owners —
link them; do not fork tables here. Executable truth for advanced flags:
`hipfire-quantize --help`.

## Reach for this when

- Quantizing a HF snapshot or local `config.json` + `.safetensors` dir
- Choosing thin-CLI vs direct-binary (MQ3, V2 ladder, HFP/MFP, Lloyd, Qwen4)
- Writing vision / PLE / trunk-overlay / FLUX component sidecars
- Installing or registering a just-written file under `~/.hipfire/models/`

Do **not** use this skill to claim quality, promote a default, or run
Astrea `calibrate`/`eval`/`metrics`. Hand those to astrea.

## Canonical owners (read these; do not fork)

| Concern | Owner |
|---|---|
| Operator surface | [`docs/QUANTIZE.md`](../../../docs/QUANTIZE.md), [`docs/CLI.md`](../../../docs/CLI.md) (`quantize`, `ple`) |
| Format math / QuantType IDs | [`docs/QUANTIZATION.md`](../../../docs/QUANTIZATION.md) |
| Magnum V2 / ladder | [`docs/quant-formats/mq4-v2.md`](../../../docs/quant-formats/mq4-v2.md), [`mq-v2-family.md`](../../../docs/quant-formats/mq-v2-family.md), [`ladder.md`](../../../docs/quant-formats/ladder.md) |
| QuantType register | [`docs/quant-formats/qt-register.txt`](../../../docs/quant-formats/qt-register.txt) |
| HFP4 | [`docs/quant-formats/hfp4.md`](../../../docs/quant-formats/hfp4.md) |
| Arch IDs / loaders | [`docs/architecture-ids.md`](../../../docs/architecture-ids.md) |
| Quality / promotion | [`.agents/skills/astrea/`](../astrea/SKILL.md) |
| Validation routes | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |

## Two surfaces

| Surface | Produces | Allowlist |
|---|---|---|
| `hipfire quantize <src>` | Thin wrapper: one `hipfire-quantize --input --output --format` per format, then optional `--install` / `--register` / `--upload` | **Exactly** `mq4`, `mq6`, `q8`, `q8f16`, `hf4`, `hf6` (`hfq4`/`hfq4g256`→`hf4`, `hfq6`/`hfq6g256`→`hf6`). `--format mq4` writes **qt 44 `MQ4G256V2`**, not legacy qt 13. Default: `mq4` (safetensors) / `hf4` (`.gguf`). `-o` requires exactly one format. |
| `hipfire-quantize --input … --output …` | Full encoder + sidecars. **Default `--format` is `q8f16`** — always pass `--format` | `mq3`, `mq{2,3,5,6}v2`, `mq4v1`/`mq4c`, HFP/MFP/Lloyd, `--tier`/`--fixed-tier`, `--qwen4-flash-next`, vision, FLUX, PLE hot-row, trunk-overlay. Imatrix/AWQ/Hessian never reach the thin wrapper. |

Missing binary — lookup order (stop at the first hit):

1. `~/binaries/hipfire-bin/hipfire-quantize` (and sibling tools in that dir)
2. `$HIPFIRE_HOME/bin/hipfire-quantize`
3. `$PATH`

```bash
ls -l ~/binaries/hipfire-bin/hipfire-quantize ~/binaries/hipfire-bin/hipfire
# only if those are absent:
# cargo build --release -p hipfire-quantize
```

Quantization is CPU- and memory-bandwidth bound (`HIPFIRE_QUANT_THREADS`,
default ~80% cores). No GPU required to write; a loader/carrier must exist
for the `arch_id` or inference will refuse the file.

## Format routing

Pick the **source** first (safetensors, never GGUF weights), then the
**family**, then the **surface**.

| Goal | Command shape | Notes |
|---|---|---|
| Qwen 3.5+ default 4-bit | `hipfire quantize DIR --format mq4 -o out.mq4` | qt 44 V2 |
| Qwen 3.5+ higher quality | `hipfire quantize DIR --format mq6` or `--both` (`mq4`+`mq6`) | `--both` is thin-CLI only |
| Llama / Mistral / Gemma / old Qwen | `hipfire quantize DIR --format hf4` (or `hf6`) | MQ rotate cost with little quality win on Llama-style dense |
| Q8 reference | `hipfire quantize DIR --format q8` | Thin CLI; safetensors path |
| Qwen3.8 product cell | `hipfire-quantize --input DIR --output out.hfq --format mq4v2 --tier base` | `xt` / `base` / `pro` — [ladder.md](../../../docs/quant-formats/ladder.md) |
| MQ3 v1 | `hipfire-quantize --input DIR --output out.mq3 --format mq3` | Thin CLI **rejects** `mq3`. Load: production on gfx11/gfx12 dense; MoE/A3B **refused at load** |
| Dense V2 ladder body | `hipfire-quantize … --format mq3v2` (or mq5v2/mq6v2) | `mq{2,3,5,6}v2` **refuse MoE** `arch_id`s |
| HFP / MFP | `hipfire-quantize --format hfp4` / `--format mfp4` | Direct only. E8/GPTQ needs `--hessian-dir` |
| Lloyd | `--allow-mq{2,3,4}-lloyd` | Research; not a product default |
| Uniform MQ2 v1 | `--format mq2 --allow-mq2` or `HIPFIRE_ALLOW_MQ2=1` | Reserved — codebook collapses. Do not serve. |
| MQ2V2 | wire-legal, **not** a Qwen3.8 product body | Prefer `mq3v2` and above to serve |
| Bonsai `tq2`/`bq1` | GGUF-input **only** | Safetensors path: `only supported for GGUF weight input` |
| PARO (`paro4`) | **Not** a quantizer `--format` | Astrea `paro-import` / `paro-oracle` (qt 28/29 are loader-only) |

Family rule of thumb: **FWHT (MQ)** for Qwen 3.5+; **HFQ** for Llama-style
dense. Confirm against QUANTIZE.md, not this sentence.

## Workflow

1. **Source.** Local dir with `config.json` + `.safetensors`, or a HF repo id
   (`hipfire quantize org/name`). Do **not** start from GGUF weights
   (deprecated since 0.4.0, removal 0.5.0 — lossy double quant).
   `imatrix.gguf` via `--imatrix` stays supported.
2. **Surface.** Thin CLI if the format is in the allowlist; otherwise
   `hipfire-quantize` direct. Graded MoE, E8, `--tier`, HFP/MFP, MQ3, V2
   ladder cells, Qwen4 Flash-Next are **direct only**.
3. **Write.** One `--output` per artifact. `--qwen4-flash-next` refuses an
   existing `--output` (no overwrite). Default output ext follows the format
   (`.mq4`, `.hf4`, `.mq6`, …); magic is always `HFQM`.
4. **Census, not quality.** Encoder prints `Done: N MB written` plus param
   accounting (unaccounted params **refuse** the write). Then `hfq_dump` for
   `arch_id` / tensor count / qt histogram. A written file is not a calibrated
   candidate. sha256/size is a **pin** check (`scripts/check_fixture.sh`,
   AGENTS.md §5), not a gate on a new artifact.
5. **Install (optional).** Thin CLI `--install --register <tag>` copies the
   trunk plus any derived `.vl` pack into `~/.hipfire/models/` and aliases
   `hipfire run <tag>`. Direct binary writes wherever `--output` points.
6. **Quality (optional, other skill).** Astrea `inspect` → `plan` → `eval` →
   `metrics`. Do not invent a KLD number from the quantize log. PARO stays
   in astrea.
7. **Load smoke (optional).** `hipfire fit <model>` then `hipfire run` or
   container serve. MoE+MQ3 and MQ2-without-allow are load refusals, not
   encoder bugs.

## Recipes (copy from docs; re-read QUANTIZE.md if a flag looks new)

HuggingFace → mq4+mq6, install, register:

```bash
hipfire quantize Jackrong/Qwopus3.5-4B-v3 \
    --both --install --register qwopus:4b
```

Local safetensors → default MQ4V2:

```bash
hipfire quantize ./my-finetune/ --format mq4 -o my-finetune.mq4
```

Qwen3.8 ladder `base` (direct):

```bash
hipfire-quantize --input ./qwen3.8-27b --output qwen3.8-27b.mq4v2.base.hfq \
  --format mq4v2 --tier base
```

MQ3 v1 (direct; dense only at load for MoE/A3B):

```bash
hipfire-quantize --input ./model --output model.mq3 --format mq3
```

AWQ / imatrix (direct only). `--awq` **requires** `--imatrix`. `--imatrix`
alone does **not** enable AWQ and errors on a format that does not consume
it. `--awq-a4-aware` is MQ4V2-only and **conflicts with** `--awq-alpha`.

```bash
hipfire-quantize --input ./model --output model.mq4 --format mq4v2 \
  --imatrix ./imatrix.gguf --awq-alpha 0.55
hipfire-quantize --input ./model --output model.mq4 --format mq4v2 \
  --imatrix ./imatrix.gguf --awq-a4-aware
```

`--q8-router` / `--no-q8-router` (reproduce `.mq4r`) / `--no-q8-conv1d`.
K-map is on by default for MoE (`--no-kmap` / `--uniform` to disable; dense
needs `--kmap-dense`). Read QUANTIZE.md before claiming Promote6 qt IDs.

HFP / MFP (direct):

```bash
hipfire-quantize --input ./model --output model.hfp4 --format hfp4
hipfire-quantize --input ./model --output model.mfp4 --format mfp4
```

Qwen4 Flash-Next streaming producer (direct, **no** `--format` / `--tier` /
`--imatrix` — extra flags are refused, not ignored):

```bash
hipfire-quantize --qwen4-flash-next --input ./source --output out.hfq
# remote pin must be hf://OWNER/REPO@<40-hex-commit>
```

This route does **not** reproduce the shipped XTS pin (GPTQ-solved experts).
Identify Flash-Next artifacts by size+SHA-256 in AGENTS.md §5, not filename.

Vision tower sidecar (format ignored; F16 matrices + F32 norms):

```bash
hipfire-quantize --input ./qwen3.8-27b-tower --vision-only \
    --output ~/.hipfire/models/qwen3.8-27b-vision.hfq
```

PLE split / derive (CLI, not the quantize binary):

```bash
hipfire ple split [--out PATH] [--keep-original] <model>
hipfire ple derive [--format q4] <model>    # writes <model>.ple-q4 from Q8 rows
```

FLUX pipe re-container (no quantization; `--output` is a **base** name):

```bash
hipfire-quantize --flux-pipe <pipe_dir> --output <base>.hfq
# FLUX.1: <base>-transformer.hfq (40) / -t5.hfq (41) / -clip.hfq (42) / -vae.hfq (43)
```

Draft sidecars are **separate bins** in `~/binaries/hipfire-bin/`, not
`--format` values: `mtp_extract` (arch 21 `.mtp`), `dflash_convert` (arch 20).
There is **no** `hipfire-quantize` builder for `<model>.expert-hotset` — do
not invent one.

CPU-only quantizer image (`docker/quantizer.Dockerfile`, `FROM hipfire-gate`;
no published tag, no compose service):

```bash
docker build -f Containerfile --target gate-runner -t hipfire-gate .
docker build -f docker/quantizer.Dockerfile -t hipfire-quantize .
docker run --rm -v "$PWD:/work" -w /work hipfire-quantize \
  --input /work/parent --output /work/out.mq4 --format mq4
```

Census (`hfq` in `~/binaries/hipfire-bin/`):

```bash
~/binaries/hipfire-bin/hfq list OUT.hfq | head
~/binaries/hipfire-bin/hfq meta-get OUT.hfq | head
```

## Guardrails

- GGUF **weight** input: warn + do not recommend. `imatrix.gguf` is the
  exception. Thin GGUF formats narrow further to `hf4|hf6|mq4|mq6`.
- Thin CLI will not grow V2/MQ3/Lloyd/HFP flags. `hipfire quantize --format
  mq4v2` is **rejected** (MODELS.md is wrong on this); use `mq4` or the
  direct binary. If `--format mq3` fails, that is the allowlist.
- Direct `--format` defaults to `q8f16`. Omit it and you do not get mq4.
- `--awq` requires `--imatrix` (exit 1). `--imatrix` with no consumer
  (`--awq` or an imatrix recipe) is exit 2, not silent ignore.
- `--allow-mq2` / Lloyd flags are research gates. MQ2V2 is not a product body.
- `mq{2,3,5,6}v2` refuse MoE at encode. MoE/A3B + MQ3 is refused **at
  daemon load**. Use `mq4`/`mq4v2`/`mq4c` or legacy `mq{2,3,5,6}` for MoE
  encode.
- `tq2`/`bq1` are GGUF-input only. PARO is Astrea, not `--format paro4`.
- Param-accounting miss or MQ4V2-Lloyd gain below the 12% floor **refuses
  the write**.
- CASK / `hipfire sidecar-gen` is deprecated (removal 0.5.0) and is **not** a
  post-quantize step.
- A green write is not an admission. Admissions live in `docs/admissions.yml`.
- Do not compare artifacts across different encoder recipes as if they were
  the same pin.

## Related skills

- [astrea](../astrea/SKILL.md) — calibration, KLD/PPL, mixed-format policy
- [hipfire-container-serve](../hipfire-container-serve/SKILL.md) — serve the
  artifact in compose / container-gate
- [hipfire-tester](../hipfire-tester/SKILL.md) — GPU bring-up / smoke after load
