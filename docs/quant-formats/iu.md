# IU4 / IU8G128 (qt 60 / qt 61)

MQ4G256V2 codes stored so an integer matrix instruction can load them.
The product is `iu4`, with the three rungs in
[ladder.md](ladder.md). `--format iu4` on
`hipfire-quantize` uses the MQ4V2 tensor map for the body. A role the rung
lifts is stored as IU8G128 (qt 61), a contiguous signed-int8 operand, instead
of interleaved Q8F16. Norms stay F16. K-map Promote6 does not rewrite a
body row to MQ6. Those rows stay on this layout, which is the operand
`v_wmma_i32_16x16x16_iu4` and `v_dot4_i32_iu8` can load. Q8 and F16 K-map
decisions are unchanged. `--uniform` still turns the K-map off entirely. `--moe-expert-down` is refused:
there is no G128 IU layout.

## Products

The bare name is the normal rung. `--tier` omitted selects it. `--tier normal`
is the same rung as `--tier base` and is stored as `base`. The file records
`hipfire_product` (`iu4`, `iu4-xt`, `iu4 pro`) plus
`hipfire_product_tier`.

| product | flag | body | embed | lm_head | ssm_out |
|---|---|---|---|---|---|
| `iu4-xt` | `--format iu4 --tier xt` | IU4G128 | IU8G128 | body | body |
| `iu4` | `--format iu4` | IU4G128 | IU8G128 | IU8G128 | body |
| `iu4 pro` | `--format iu4 --tier pro` | IU4G128 | IU8G128 | IU8G128 | IU8G128 |

`linear_attn.conv1d` is a 3-D tensor, so it stays interleaved Q8F16 (qt 3) at
every rung; it does not take the IU8G128 row layout. On a MoE model the
attention and router Q8 pin is unchanged: lifting attention is a different
product, the same rule as the MQ ladder. These names are not `mq4`. The bytes
are a different wire type, so a loader that expects qt 44 will not accept them.

The 4-bit math is unchanged. Weights are FWHT-256 rotated (seeds 42 / 1042), then
each 128-wide half is an unsigned code `0..=15` with

```text
w = f32(scale) * code + f32(zero)
```

`scale` and `zero` are the fp16 values MQ4V2 would have stored.
`--mq4v2-symmetric` still applies, so `zero = f16(-8 * scale)` and code 8 is
the zero bin. A kernel that wants the integer seed `sum((q - 8) * x)` may use
that identity only after checking the stored zero.

IU4 packs the codes the way
`v_wmma_i32_16x16x16_iu4` wants them: low nibble = even K, high nibble = odd K,
weight side unsigned.

## Row layout

`K % 256 == 0`. One row of `K` weights:

```text
[codes][f16 scale, f16 zero] * (K / 128)[zero pad to a 16-byte multiple]
```

Both IU layouts pad every row with zero bytes to a multiple of 16
(`iu_quant::ROW_ALIGN`), so each row starts 16-byte aligned and the decode
kernels can use 16-byte loads at any `K`. Padding is not counted in stored
bpw. For IU4 it is non-zero only when `K % 512 != 0`; for IU8G128 only when
`K % 1024 != 0` (the `K = 2560` lm_head row is 2608 B instead of 2600 B). A
file written before padding fails the loader's size check at those shapes
and has to be re-quantized.

| | code bytes | scale bytes | row bytes at K=256 | stored bpw |
|---|---:|---:|---:|---:|
| IU4, qt 60 | `K/2` | `K/128 * 4` | 136 | 4.25 |

Scales are little-endian and in half order: for each 256-weight group, half 0
(weights 0..127) then half 1 (128..255), which is the 8-byte MQ4V2 header
moved to the tail of the row instead of sitting in front of every 128 nibble
bytes. The group field written in the file stays 256.

There is no lane swizzle in the file. Storage is row-major and K-linear. The
kernel applies the WMMA lane map. On the gfx11 probe, nibble `m` of a dword
is already K element `m`, so IU4 needs no permute before
`v_wmma_i32_16x16x16_iu4`.

## What a kernel binds

`hipfire_runtime::iu_quant` is the oracle: `repack_mq4v2`, `validate`,
`dequant`, `gemv_iu4_dot8`, `row_bytes`, `code_bytes`. Dequant is the
rotated-basis f32 value above.

Decode GEMV for qt 60 is `gemv_iu4g128` with f32 activations, on every arch.
It is a wide-load kernel: one row per 32-lane wave, four rows per 128-thread
block, and each lane reads 16 code bytes (32 weights) at a time, so it needs
the 16-byte row alignment above. Per 32 weights it folds

```text
acc += scale * sum(code * x) + zero * sum(x)
```

with `x` the FWHT-256 activation. One opt-in route needs gfx11/gfx12 IU
hardware: `HIPFIRE_IU4_DECODE=dot4` (W4A8, `v_dot4_i32_iu8`). The W4A4
`v_dot8_i32_iu4` decode route was removed; `iu_quant::gemv_iu4_dot8` stays as
the CPU int4-activation reference.

Prefill GEMM for qt 60 is W4A8 by default (`gemm_iu4g128_w4a8_*`, int8
activation per 128, `v_wmma_i32_16x16x16_iu8`); `HIPFIRE_IU4_PREFILL_ACT=a4`
selects the faster W4A4 (`gemm_iu4g128_w4a4_*`, `v_wmma_i32_16x16x16_iu4`).
Qwen3.5-4B on gfx1100, pp512: W4A8 about 7,700 tok/s and prefill KL vs
BF16 0.0467 (equal to decode); W4A4 about 9,470 tok/s and KL 0.0982.
Both run on exact gfx1100 and gfx1151 only, gated by
`rdna_compute::arch_caps::supports_iu4g128_gemm`; every other arch takes the
per-token decode loop. DFlash verify runs the target's batched body, so it
takes the same route.

## Lifted roles (qt 61)

A lifted row is not put on the 4-bit grid. Q4-grade error there is the reason it was Q8F16, and a byte that only holds `0..=15` does not use the int8 multiplier: `v_wmma_i32_16x16x16_iu8` would be multiplying a nibble stored in a byte, at twice the bandwidth of IU4 for the same values.

They stay on the Q8 grid and change layout only: signed int8 codes with one fp16
scale per 128 weights, zero point 0, no FWHT. The scale is not Q8F16's per-32
scale; quantization is `d = amax/127` over each 128-wide group and every code is
`round(w/d)` clamped to `-127..=127`:

```text
w = f32(scale) * f32(i8(code))
```

IU8G128 (qt 61) stores the codes contiguously and the scales at the row tail:

```text
IU8G128, one row, K = 256                          260 B/row = 8.125 bpw

  ┌────────────────────────────────────────────┬────┬────┐
  │ 256 B signed codes, K-linear, -128..=127   │ sc │ sc │
  │                                            │ 0  │ 1  │
  └────────────────────────────────────────────┴────┴────┘
        code plane (offset 0)                    4 B of f16 scales
```

`K % 128 == 0`; a row that is not is left as interleaved Q8F16. The row is
`K + (K/128)*2` bytes (8.125 bpw), six bytes per 128 smaller than the old
one-scale-per-32 layout (`K + K/16`, 8.5 bpw), then zero-padded to a 16-byte
multiple (see Row layout).

The wire name is `IU8G128` (`QuantType::IU8G128`, qt 61). The runtime types
(`rdna_compute::DType::IU8G128`, `EmbeddingFormat::IU8G128`), `iu_quant`, and
`DType::iu_quant_name()` use the same name. The row carries one f16 scale per 128.

Decode GEMV is `gemv_iu8g128` v2 with f32 activations, on every arch;
`HIPFIRE_IU8_GEMV=v1` selects the original kernel (A/B). The int8×int8 `_a8`
decode variants were removed: on 4B they were speed-neutral (−0.4%) and
cost 7.3e-4 KL (IU4_NOTES.md C8).

Prefill is two kernels behind one selector, `Gpu::iu8g128_gemm_route`, used by
both the singleton batched prefill and the multi-slot prefill. W8A8
(`gemm_iu8g128_w8a8_min`, one int8 activation block per 128, one f32 activation
scale) is chosen when `K % 256 == 0` and there are at least 48 token rows
(`IU8G128_W8A8_MIN_ROWS`); otherwise the f16-WMMA twin
(`gemm_iu8g128_residual_wmma` + mb4/mb4m2, needs `K % 128 == 0`). Both run on
exact gfx1100/gfx1151 only (`arch_caps::supports_iu8g128_gemm`).
`HIPFIRE_IU8G128_W8A8=0` forces the twin everywhere (A/B). The 48-row crossover
was measured on gfx1100: the twin is 1.3–2.4× faster at n≤32, parity at 40–48,
and W8A8 is 1.26–1.36× faster at 56–63 and 1.9–2.8× from 64.

Norms stay F16. The embedding table is a gather, not a matrix instruction. On the normal rung it is stored this way so a tied `lm_head` is the same buffer the int8 unit can load. Attention and the router are not rungs; on a MoE model the Q8 pin still holds them.

Q8 and F16 tensors outside an `iu4` file are unchanged. The generic
GEMM dispatcher refuses qt 60 and 61; DFlash admits qt 61 on exact
gfx1100/gfx1151 (its verify head runs `gemm_iu8g128_residual_wmma` and its
noise embedding runs `embedding_lookup_iu8g128`).

## What runs

| reader | qt 60 IU4 | qt 61 IU8G128 |
|---|---|---|
| decode GEMV | `gemv_iu4g128` (f32 activations, every arch; `HIPFIRE_IU4_DECODE=dot4` opt-in, gfx11/gfx12) | `gemv_iu8g128` v2 (f32 activations, every arch) |
| fused `y += W·x` | `gemv_iu4g128_residual` | `gemv_iu8g128_residual` |
| embedding gather | — (the table is qt 61 on every rung) | `embedding_iu8g128` |
| prefill GEMM | `gemm_iu4g128_w4a8_*` (`v_wmma_i32_16x16x16_iu8`; `HIPFIRE_IU4_PREFILL_ACT=a4` = W4A4 `gemm_iu4g128_w4a4_*`; gfx1100/gfx1151 only) | one selector, `Gpu::iu8g128_gemm_route` (gfx1100/gfx1151 only): `gemm_iu8g128_w8a8_min` (IU8 WMMA, W8A8) when `K % 256 == 0` and n ≥ 48 rows; the f16-WMMA twin (`gemm_iu8g128_residual_wmma` + mb4/mb4m2, `K % 128 == 0`) otherwise |
| DFlash verify | — | qt 61 head via `gemm_iu8g128_residual_wmma` (gfx1100/gfx1151) |

`hipfire_runtime::iu_quant` is the CPU contract: `dequant` for qt 60
and `gemv_q8` (`dequant_q8` · `x`) for qt 61. The loaders admit both dtypes
(`RAW_CODECS`), and qt 61 embedding tables are admitted raw (one contiguous
plane + tail scales), so an `iu4` artifact loads and decodes on the
per-token path. Batched prefill runs qt 60 (W4A8, or W4A4 opt-in) and qt 61 (W8A8 or its
f16-WMMA twin, per `Gpu::iu8g128_gemm_route`) on exact gfx1100/gfx1151 only;
the generic GEMM dispatcher still refuses both. A rung's `ssm_out` is a body row on `xt`/base, so that rung takes the
fused residual twin; there is no `_batched` gather.
