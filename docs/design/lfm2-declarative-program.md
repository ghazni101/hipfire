# LFM2 declarative layer program (#666 G10)

Status: single-lane decode cut over (dense LFM2.5 and LFM2.5-8B-A1B MoE).
The continuous-batching decode (`forward_batch.rs`) is not cut over yet.

## Goal

LFM2 declares each decoder layer as typed `hipfire_dispatch` steps, as Qwen4,
Qwen3.5 (#795) and Gemma 4 (#813) do. The architecture crate binds resident
weights, KV and convolution state and scratch; dispatch owns every executor
body and route choice.

`crates/hipfire-arch-lfm2moe/src/program.rs` builds three step lists:

| Block | Steps |
|---|---|
| Conv mixer (LIV) | `RmsnormAutomatic` → `Gemv` (in_proj) → `GatedShortConv` → `GemvResidual` (out_proj) |
| Attention mixer | `RmsnormAutomatic` → `Gemv` ×3 → `QkNorm` ×2 → `Rope` → `Attend` → `GemvResidual` |
| Dense FFN | `RmsnormAutomatic` → `Gemv` (w1, w3) → `SiluMul` → `GemvResidual` (w2) |
| MoE FFN | `Moe` (sealed call: norm, router projection, biased sigmoid top-k, experts, combine) |
| Head | `RmsnormAutomatic` → `Gemv` (tied lm_head) |

`forward.rs` keeps only the lifecycle around the program: token/position
staging, hipGraph capture (`HIPFIRE_LFM2_GRAPH`), Redline retained replay
(dense only), and the oracle capture of per-layer hidden states.

Removed: the hand decode loop, the #397 super-op path (`Lfm2MoeBindings`,
`lfm2_lower_variant`, `HIPFIRE_FORWARD_LOWERED` for LFM) and the hand MoE
block.

## New shared pieces

- **`Step::GatedShortConv` (`GatedShortConvOp`)**: the double-gated causal
  short convolution with its rolling `[rows, channels, K-1]` state, over any
  number of independent lanes (`conv1d_gated_decode_f32`). Model-neutral: the
  batched decode can bind the same op with `rows = B`.
- **`Step::SiluMul`**: the one-row SwiGLU activation between the gate/up and
  down projections.
- **Biased sigmoid router in the sealed MoE (`MoeParams::selection_bias`,
  `MoeSelectionBias`)**: experts are selected on `sigmoid(logits) + bias`,
  weighted by the unbiased scores, normalized, and scaled by `route_scale`
  (`deepseek4_moe_topk_bias_aware_f32`). With a bias the sigmoid recipe takes
  any width `1..=8` and routes on the GPU; only the uniform FWHT `MQ4G256` and
  `MQ6G256` expert formats are admitted, because their indexed gate/up,
  activation, down and combine kernels read the width at run time. The
  unbiased sigmoid route (Cohere) stays top-8. DeepSeek4 (G9) uses the same
  selection.

LFM2 MoE layers seal their resident experts at load (`MoeFfn::seal`: one
`ExpertTable` row per expert, live pointer tables bound once). A layer with
mixed expert formats, or with the layer-wide expert AWQ sidecars the old hand
path read (no hipfire quantizer writes them), is refused at load instead of
running on a path that would drop them.

## Numerics contract

Every projection reads the plain RMS-normed input (`GemvInput::Raw`) and
rotates it for its own format, as the pre-port route did. This keeps every
kernel and reduction order of beta's route: the fusion table (fused QKV,
fused gate/up, fused rmsnorm+FWHT) is not admitted. An earlier revision that
fed prerotated inputs took the Q8 fused QKV and gate/up kernels; its logits
moved by up to 0.16 (350M Q8) with the same greedy tokens. Admitting those
fusions is a separate, measured change.

## Evidence (gfx1151, halo)

Base: beta `901e44fdd`. Route oracle: a throwaway dump driver
(`~/mesh-validation/HIPF-14/lfm2_route_dump.rs`, built unchanged in both
trees) runs `decode_step` over a fixed 24-token prompt plus 48 greedy tokens
and records at every committed position the fed token, argmax, the full logits
and digests of all convolution state and all KV bytes, plus the final state
bytes. Beta is deterministic run to run (two beta runs identical).

| Fixture | Positions | Tokens | Logits | Conv state | KV |
|---|---|---|---|---|---|
| `lfm2.5-350m.q8` | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-350m.mq4` | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-1.2b.mq4` | 72 | identical | bitwise | bitwise | bitwise |
| `lfm2.5-8b-a1b.mq4` (MoE, sealed route) | 72 | identical | bitwise | bitwise | bitwise |

The same parity holds on the hipGraph route (`HIPFIRE_LFM2_GRAPH=1`, beta vs
head, and both equal to the default route) on `lfm2.5-350m.q8` and
`lfm2.5-8b-a1b.mq4`.

Serve (`serve_harness.py --thinking off --sampling greedy
--compare-transcript`, beta daemon vs head daemon): `transcript_byte_identical=true`
for battery and chain on `lfm2.5-1.2b.mq4` and `lfm2.5-8b-a1b.mq4`. The 8B-A1B
text is degenerate on both beta and head (empty turns, whitespace attractors);
that is a pre-existing defect tracked separately, not a change of this port.

Redline (`redline_daemon_harness.py --pm4 --skip-prefill`, `lfm2.5-350m.mq4`):
decode launch sequence unchanged (280 launches, 11 kernels, hash
`de9b61e3ebd1de33` on both), PM4 shadow exact on both.

Decode, A/B alternating under the halo lock, no other cargo/rustc running:

| Model | Measure | beta | head | ratio |
|---|---|---|---|---|
| `lfm2.5-1.2b.mq4` | `hipfire bench --matrix` tg128 @ ctx 128, 2 runs | 232.5 / 231.6 | 231.2 / 231.5 | 0.997 |
| `lfm2.5-1.2b.mq4` | same @ ctx 2048 | 125.78 / 125.77 | 125.73 / 125.70 | 0.9995 |
| `lfm2.5-8b-a1b.mq4` | 256 greedy tokens through `decode_step`, 4 rounds | 166.6 / 166.6 / 167.7 / 167.5 | 166.8 / 166.0 / 166.9 / 167.3 | 0.999 |

The sealed MoE first ran the 8B-A1B gate/up through the top-8 launcher
(`gemv_hfq4g256_moe_gate_up_k8_indexed`): bitwise identical, but 1.2 % slower
at k = 4 on gfx1151. Below top-8 (only the biased route reaches those arms)
the executor now takes the batched HFQ4 and HFQ6 gate/up launchers the hand
path used.

Not covered on halo: LFM2-VL (`prefill_embed_step` runs the same program; no
VL fixture on halo) and the `.mq6e` MQ6-expert artifact (no fixture; the
sealed route calls the same MQ6 gate/up, activation, down and combine kernels
with the same arguments as the hand path, but it was not run). The gfx1201
retained-PM4 admission row in `docs/admissions.yml` was screened on gfx1201;
the gfx1151 launch sequence is unchanged, but the gfx1201 sequence hash is not
re-measured.
