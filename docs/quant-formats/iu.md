# IU4 / IU8 (qt 57 / qt 58 / qt 59)

MQ4G256V2 codes stored so an integer matrix instruction can load them.
The products are `iu4` and `iu8`, each with the three rungs in
[ladder.md](ladder.md). `--format iu4` and `--format iu8` on
`hipfire-quantize` use the MQ4V2 tensor map for the body. A role the rung
lifts is stored as IU8G32 (qt 59), a contiguous signed-int8 operand, instead
of interleaved Q8F16. Norms stay F16. K-map Promote6 does not rewrite a
body row to MQ6. Those rows stay on this layout, which is the operand
`v_wmma_i32_16x16x16_iu4` and `v_dot4_i32_iu8` can load. Q8 and F16 K-map
decisions are unchanged. `--uniform` still turns the K-map off entirely. `--moe-expert-down` is refused:
there is no G128 IU layout.

## Products

The bare name is the normal rung. `--tier` omitted selects it. `--tier normal`
is the same rung as `--tier base` and is stored as `base`. The file records
`hipfire_product` (`iu4`, `iu4-xt`, `iu4 pro`, and the `iu8` twins) plus
`hipfire_product_tier`.

| product | flag | body | embed | lm_head | ssm_out |
|---|---|---|---|---|---|
| `iu4-xt` | `--format iu4 --tier xt` | IU4G256 | IU8G32 | body | body |
| `iu4` | `--format iu4` | IU4G256 | IU8G32 | IU8G32 | body |
| `iu4 pro` | `--format iu4 --tier pro` | IU4G256 | IU8G32 | IU8G32 | IU8G32 |
| `iu8-xt` | `--format iu8 --tier xt` | IU8G256 | IU8G32 | body | body |
| `iu8` | `--format iu8` | IU8G256 | IU8G32 | IU8G32 | body |
| `iu8 pro` | `--format iu8 --tier pro` | IU8G256 | IU8G32 | IU8G32 | IU8G32 |

`linear_attn.conv1d` is a 3-D tensor, so it stays interleaved Q8F16 (qt 3) at
every rung; it does not take the IU8G32 row layout. On a MoE model the
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

IU8 is not an 8-bit quant. Each code is widened to a byte whose value is
still `0..=15`, which is what `v_wmma_i32_16x16x16_iu8` takes after the
nibble expand today's W4A8 kernel does in VALU. The extra bytes buy a load
with no expand; no prefill GEMM uses that operand yet — qt 58's kernels are
the decode GEMV and its fused residual. IU4 is the same codes packed the way
`v_wmma_i32_16x16x16_iu4` wants them: low nibble = even K, high nibble = odd K,
weight side unsigned.

## Row layout

`K % 256 == 0`. One row of `K` weights:

```text
[codes][f16 scale, f16 zero] * (K / 128)
```

| | code bytes | scale bytes | row bytes at K=256 | stored bpw |
|---|---:|---:|---:|---:|
| IU4, qt 57 | `K/2` | `K/128 * 4` | 136 | 4.25 |
| IU8, qt 58 | `K` | `K/128 * 4` | 264 | 8.25 |

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

Decode GEMV for qt 57 is `gemv_iu4g256` with f32 activations by default, on
every arch, and is bit-exact with the MQ4G256V2 twin. Two opt-in routes need
gfx11/gfx12 IU WMMA hardware: `HIPFIRE_IU4_DECODE=dot4` (W4A8,
`v_dot4_i32_iu8`) and `=dot8` (W4A4, `v_dot8_i32_iu4`). The dot8 route loads
one wave per row and issues the code plane unsigned to `v_dot8_i32_iu4`; the
activation is signed int4, one scale per 32 elements (`d = max(|x|) / 7`), and
each eight weights fold

```text
acc += (d * scale) * dot(q, xq) + zero * (d * sum(xq))
```

That integer form is the dot8 contract only, not the default GEMV. `x` is the
FWHT-256 activation. The integer dot is exact. The activation quant is the
approximation.

Prefill GEMM for qt 57 is W4A4 (`gemm_iu4g256_w4a4_*`,
`v_wmma_i32_16x16x16_iu4`) on exact gfx1100 and gfx1151 only, gated by
`rdna_compute::arch_caps::supports_iu4g256_gemm`; every other arch takes the
per-token decode loop. DFlash verify runs the target's batched body, so it
takes the same route.

## Lifted roles (qt 59)

A lifted row is not put on the 4-bit grid. Q4-grade error there is the reason it was Q8F16, and a byte that only holds `0..=15` does not use the int8 multiplier: `v_wmma_i32_16x16x16_iu8` would be multiplying a nibble stored in a byte, at twice the bandwidth of IU4 for the same values.

They stay on the Q8 grid and change layout only: signed int8 codes with one fp16
scale per 128 weights, zero point 0, no FWHT. The scale is not Q8F16's per-32
scale; quantization is `d = amax/127` over each 128-wide group and every code is
`round(w/d)` clamped to `-127..=127`:

```text
w = f32(scale) * f32(i8(code))
```

IU8G32 (qt 59) stores the codes contiguously and the scales at the row tail:

```text
IU8G32, one row, K = 256                          260 B/row = 8.125 bpw

  ┌────────────────────────────────────────────┬────┬────┐
  │ 256 B signed codes, K-linear, -128..=127   │ sc │ sc │
  │                                            │ 0  │ 1  │
  └────────────────────────────────────────────┴────┴────┘
        code plane (offset 0)                    4 B of f16 scales
```

`K % 128 == 0`; a row that is not is left as interleaved Q8F16. The row is
`K + (K/128)*2` bytes (8.125 bpw), six bytes per 128 smaller than the old
one-scale-per-32 layout (`K + K/16`, 8.5 bpw).

The wire name is `IU8G32` (`QuantType::IU8G32`, qt 59); the runtime types
(`rdna_compute::DType::IU8G128`, `EmbeddingFormat::IU8G128`), `iu_quant`, and
`DType::iu_quant_name()` spell it `IU8G128`. The “G32” in the wire name, the
kernel symbols (`gemv_iu8g32*`, `gemm_iu8g32*`, `embedding_iu8g32*`), and the
`HIPFIRE_IU8G32_*` dev knobs is historical — the row carries one f16 scale per
128.

Decode GEMV is `gemv_iu8g32` v2 with f32 activations, on every arch;
`HIPFIRE_IU8_GEMV=v1` selects the original kernel (A/B). The int8×int8 `_a8`
decode variants were removed: on 4B they were speed-neutral (−0.4%) and
cost 7.3e-4 KL (IU4_NOTES.md C8).

Prefill is two kernels behind one selector, `Gpu::iu8g32_gemm_route`, used by
both the singleton batched prefill and the multi-slot prefill. W8A8
(`gemm_iu8g32_w8a8_min`, one int8 activation block per 128, one f32 activation
scale) is chosen when `K % 256 == 0` and there are at least 48 token rows
(`IU8G32_W8A8_MIN_ROWS`); otherwise the f16-WMMA twin
(`gemm_iu8g32_residual_wmma` + mb4/mb4m2, needs `K % 128 == 0`). Both run on
exact gfx1100/gfx1151 only (`arch_caps::supports_iu8g32_gemm`).
`HIPFIRE_IU8G32_W8A8=0` forces the twin everywhere (A/B). The 48-row crossover
was measured on gfx1100: the twin is 1.3–2.4× faster at n≤32, parity at 40–48,
and W8A8 is 1.26–1.36× faster at 56–63 and 1.9–2.8× from 64.

Norms stay F16. The embedding table is a gather, not a matrix instruction. On the normal rung it is stored this way so a tied `lm_head` is the same buffer the int8 unit can load. Attention and the router are not rungs; on a MoE model the Q8 pin still holds them.

Q8 and F16 tensors outside an `iu4` / `iu8` file are unchanged. The generic
GEMM dispatcher refuses qt 57, 58, and 59; DFlash admits qt 59 on exact
gfx1100/gfx1151 (its verify head runs `gemm_iu8g32_residual_wmma` and its
noise embedding runs `embedding_lookup_iu8g32`).

## What runs

| reader | qt 57 IU4 | qt 58 IU8 | qt 59 IU8G32 |
|---|---|---|---|
| decode GEMV | `gemv_iu4g256` (f32 activations, every arch; `HIPFIRE_IU4_DECODE=dot4`/`dot8` opt-in, gfx11/gfx12) | `gemv_iu8g256` (byte plane, `(scale·code + zero)·x`, every arch) | `gemv_iu8g32` v2 (f32 activations, every arch) |
| fused `y += W·x` | `gemv_iu4g256_residual` | `gemv_iu8g256_residual` | `gemv_iu8g32_residual` |
| embedding gather | — (the table is qt 59 on every rung) | — | `embedding_iu8g32` |
| prefill GEMM | `gemm_iu4g256_w4a4_*` (`v_wmma_i32_16x16x16_iu4`, gfx1100/gfx1151 only) | — (per-token decode loop) | one selector, `Gpu::iu8g32_gemm_route` (gfx1100/gfx1151 only): `gemm_iu8g32_w8a8_min` (IU8 WMMA, W8A8) when `K % 256 == 0` and n ≥ 48 rows; the f16-WMMA twin (`gemm_iu8g32_residual_wmma` + mb4/mb4m2, `K % 128 == 0`) otherwise |
| DFlash verify | — | — | qt 59 head via `gemm_iu8g32_residual_wmma` (gfx1100/gfx1151) |

`hipfire_runtime::iu_quant` is the CPU contract: `gemv_iu4_dot8` for qt 57,
`gemv_iu8_dot` (`dequant(Layout::Iu8)` · `x`) for qt 58, and `gemv_q8`
(`dequant_q8` · `x`) for qt 59. The loaders admit all three dtypes
(`RAW_CODECS`), and qt 59 embedding tables are admitted raw (one contiguous
plane + tail scales), so an `iu4` / `iu8` artifact loads and decodes on the
per-token path. Batched prefill runs qt 57 (W4A4) and qt 59 (W8A8 or its
f16-WMMA twin, per `Gpu::iu8g32_gemm_route`) on exact gfx1100/gfx1151 only;
the generic GEMM dispatcher still refuses all three. A rung's `ssm_out` is a body row on `xt`/base, so that rung takes the
fused residual twin; there is no `_batched` gather.
