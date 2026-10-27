# IU4 / IU8 (qt 57 / qt 58 / qt 59)

MQ4G256V2 codes stored so an integer matrix instruction can load them.
The products are `iu4` and `iu8`, each with the three rungs in
[ladder.md](ladder.md). `--format iu4` and `--format iu8` on
`hipfire-quantize` use the MQ4V2 tensor map for the body. A role the rung
lifts is stored as IU8G32 (qt 59), a contiguous signed-int8 operand, instead
of interleaved Q8F16. Norms stay F16. A K-map promotion can still rewrite a
body row that the rung does not own to MQ6. Pass `--uniform` when every
eligible body row must stay on this layout. `--moe-expert-down` is refused:
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

`linear_attn.conv1d` stays on the Q8 grid at every rung. On a MoE model the
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
with no expand. IU4 is the same codes packed the way
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
`dequant`, `row_bytes`, `code_bytes`. Dequant is the rotated-basis f32 value
above. Generic GEMV, GEMM, and DFlash refuse both dtypes. A new kernel should
load `row_bytes(K)` per row, take the code plane at offset 0, and fold
`scale * acc + zero * sum(x)` per 128 K after the integer dot.

## Lifted roles (qt 59)

A lifted row is not put on the 4-bit grid. Q4-grade error there is the reason it was Q8F16, and a byte that only holds `0..=15` does not use the int8 multiplier: `v_wmma_i32_16x16x16_iu8` would be multiplying a nibble stored in a byte, at twice the bandwidth of IU4 for the same values.

They stay on the Q8F16 grid and change layout only. Signed int8, one fp16 scale per 32 weights, zero point 0, no FWHT:

```text
w = f32(scale) * f32(i8(code))
```

Q8F16 stores that scale in front of every 32 codes (34 B per block). IU8G32 (qt 59) moves the scales to the tail of the row so the codes are one contiguous signed operand:

```text
IU8G32, one row, K = 256                         272 B/row = 8.5 bpw

  ┌────────────────────────────────────────────┬────┬────┬────┬────┐
  │ 256 B signed codes, K-linear, -128..=127   │ sc │ sc │ sc │ sc │
  │                                            │ 0  │ 1  │ …  │ 7  │
  └────────────────────────────────────────────┴────┴────┴────┴────┘
        code plane (offset 0)                    16 B of f16 scales
```

`K % 32 == 0`. A row that is not is left as interleaved Q8F16. Two WMMA K=16 tiles share one scale. The kernel sets the weight operand signed and folds `scale * float(C)` per 32 K. There is no zero term.

Norms stay F16. The embedding table is a gather, not a matrix instruction. On the normal rung it is stored this way so a tied `lm_head` is the same buffer the int8 unit can load. Attention and the router are not rungs; on a MoE model the Q8 pin still holds them.

Q8 and F16 tensors outside an `iu4` / `iu8` file are unchanged. Generic GEMV, GEMM, and DFlash refuse qt 59 along with qt 57 and 58.
