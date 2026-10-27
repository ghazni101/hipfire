# IU4 / IU8 (qt 57 / qt 58)

MQ4G256V2 codes stored so an integer matrix instruction can load them.
`--format iu4` and `--format iu8` on `hipfire-quantize` use the MQ4V2 tensor
map: the same rows are 4-bit, embeddings and the protected tier stay Q8F16,
norms stay F16, and a K-map promotion still rewrites a row to MQ6. Pass
`--uniform` when every eligible body row must stay on this layout.
`--moe-expert-down` is refused: there is no G128 IU layout.

The math is unchanged. Weights are FWHT-256 rotated (seeds 42 / 1042), then
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

Q8 and F16 tensors in the same file are unchanged and do not use these units.
