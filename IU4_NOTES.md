# IU4 review — feat/iu4-iu8-quant-family @ 474514e16 (worktree ~/work/hipfire-iu4, branch review/iu4)

Merge base vs master: 3e2a6f0d8. Times UTC (Dubai = UTC+4).

## Inventory (~/models/hipfire)
- Qwen/Qwen3.5-4B-IU4P/qwen3.5-4b.iu4p  (iu4 pro rung) — twin Qwen/Qwen3.5-4B-MQ4P/qwen3.5-4b.mq4p
- Qwen/Qwen3.8-27B-IU4P/qwen3.8-27b.iu4p — twin Qwen/Qwen3.8-27B-MQ4P/qwen3.8-27b.mq4p

## Twin proof (examples/iu4_twin_check.rs), 23:26 UTC
Every tensor of each IU4P file is a byte-exact relayout of its MQ4P twin:
- 4B: qt57<-qt44 224/224 equal (repack_mq4v2), qt59<-qt3 25/25 equal (repack_q8f16), qt1 153/153, qt3 24/24 identical. TOTAL_BAD=0
- 27B: qt57<-qt44 448/448, qt59<-qt3 50/50, qt1 305/305, qt3 48/48. TOTAL_BAD=0
=> an IU kernel that runs MQ4V2's algorithm must reproduce the MQ4P model bit-for-bit. MQ4P is the oracle.

## Baseline (pre-change build 474514e16, daemon md5 a8795a7a..., hipfire 222a1f37...)
`hipfire bench <m> --matrix --pp 512,2048 --ctx 128,4096 --tg 64 --runs 2 --warmups 2 --spec off`
| model | pp512 | pp2048 | tg64@128 | tg64@4096 |
|---|---|---|---|---|
| 4B IU4P | 126.45 | 123.75 | 129.61 | 124.48 |
| 4B MQ4P | 8117.70 | 7475.80 | 201.52 | 189.73 |

## Gaps (code review)
- G1 prefill: no IU GEMM; is_batchable_la has no IU arm -> per-token loop (64x slower than twin at pp512)
- G2 decode fused gate_up/qkv/qkvza keys panic for IU -> unfused launches
- G3 gemv_iu4g256: requantizes x to int4 inside every row (M-fold redundant), 1 dword/lane/group, no unroll/multirow
- G4 accuracy: decode is W4A4 (int4 activation per 32) vs twin W4 x f32
- G5 DFlash/MTP/multi-slot batched: refused/untested for IU
- G6 MoE refuses IU (no MoE iu4 models present)
- G7 IU8 (qt58) / IU8G32 scalar kernels
- G8 route plan does not predict IU kernels ("unplanned kernel ... bound lazily" x18 warnings at bench)

## C2 progress (00:12 UTC = 04:12 Dubai)
- C1 (uncommitted): f32 gemv, 2-row residual, fused keys, swiglu arm. C1 binaries saved at ~/work/iu4/c1/.
- chainA (C1 build, 4B): dot8 == pre on all 6 prompts (hash 3778622fa4d6 ...). new IU4P == new MQ4P on code/math, differs elsewhere (prefill paths differ pre-C2). unstable 0, conc_mismatch 0, klog 0 for all four runs.
- C2: IU4 MMQ kernel (HIPFIRE_MMQ_IU4L variant of gemm_mq4g256v2_residual_mmq.hip), gemm_iu4.rs, dispatch arms, is_batchable_la admits IU4G256 on gfx1100|gfx1151 unless dot8.
- iu4_mmq_gemm_gpu: 548 checks, bitdiff vs MQ4V2 MMQ = 0 everywhere (random 128 cases, + real 4B 18 cases); worst mmq rel vs f64 2.305e-2 (Q8_1 activation quant, same as MQ4V2); pertoken rel 2.0e-7.
- Test fix: rel normalized by max|Wx| (add-seed cancellation made rel 0.385 at m=1 n=1 while abs equalled set).
- next: build6 (release C2) + t_c2 (unit tests), then chainB gates.

## chainB (C2 build, daemon md5 ad2b3032.., 00:12-00:35 UTC) and STOP
- c2_iu4p4b: all 6 shas == C1 (chainA new_iu4p4b), unstable 0 conc 0 klog 0, run time 210s (C1 201s) -> C2 MMQ prefill NOT ACTIVE.
- c2_mq4p4b == chainA new_mq4p4b on all 6 (no MQ4V2 regression from the ifdef'd .hip). c2dot8 == pre on all 6.
- 27B: c2_iu4p27b solo 6/6 STABLE, concurrent phase cut by the 600s gate timeout (no GATE line, no json). c2_mq4p27b GATE 0/0/0.
- 27B IU4P vs MQ4P: hash/code/math equal; story/ctx1k/ctx4k differ.
- Unit tests: dispatch 464 pass, qwen35 638 pass.
- Hypothesis (untested): the 25 qt59 (IU8G32, relaid Q8_0) tensors, probably the DeltaNet wo/ssm_out, fail is_batchable_la, so the whole prefill stays per-token. Next: HIPFIRE_DEBUG_BATCH=1 + log the admissible Err; add an IU8G32 batched GEMM (Q8_0 path over the relayout).
- STOP: disk free 2.7 GB (<3 GB rule). target-perf = 19G (debug 13G protected, release 6.3G).

## C3/C4/C5 (01:00-01:45 UTC = 05:00-05:45 Dubai)
- ROOT CAUSE CONFIRMED (HIPFIRE_DEBUG_BATCH): prefill per-token because DeltaNet `wo` (ssm_out, qt59 IU8G32) failed is_batchable_la ("DeltaNet wo dtype IU8G32 not batchable on gfx1100"). After C3: all_layers_ok=true, iu4_gemm mmq_eligible=true.
- C3: IU8G32 WMMA residual GEMM = gemm_q8_0_residual_wmma.hip under `HIPFIRE_IU8G32L` (planar codes/scales addressing), gemm_iu8g32.rs, batched.rs IU8G32 arm, prefill.rs admits wo IU8G32 on gfx1100/1151.
  - test iu8g32_wmma_gemm_gpu: 115/115 random cases bit-exact vs Q8_0 WMMA over the Q8_0 twin; worst rel vs f64 1.478e-4 (f16 W rounding); real 4B out_proj m=2560 k=4096 n=1/77/512 bit-exact.
  - bench (01:17 UTC): pp512 126.45 -> 5628.45, pp2048 123.75 -> 5149.75, tg64@128 165.34 (MQ4P 8117.70/7475.80/201.52).
- C4: batched IU8G32 embedding kernel + lm_head/embedding allow-lists (batch.rs, prefill.rs, batch_staging.rs). gate c4_iu4p4b (serve default = sequential): unstable 0 / conc 0 / errors 0 / klog 0.
- GAP: the gate's "concurrent" phase was serialized (serve defaults to continuous_batch_size 1). With `--continuous-batch-size 4` every request failed: `forward_decode_batch: IU8G32 ... no GPU kernel reads it yet` (lm_head_batched IU8G32 arm called llama::weight_gemv, which refuses qt58/59).
- C5: lm_head_batched IU8G32 arm runs the solo Step::Gemv per lane. gate c5cb_iu4p4b (cb=4): unstable 0 / conc_mismatch 0 / errors 0 / klog 0. c5cb_mq4p4b: 0/0/0.
- Batched-route speed (rocprofv3 under serve cb=4, 256 tok solo + 2x256 concurrent): IU4P solo 3.47 s, conc2 4.07 s. Top kernels: fused_iu4g256_rows 29%, gemv_iu8g32 (lm_head 248320x2560) 1.51 ms/call 26%, gemv_iu8g32_residual 14%.

## C6 IU8G32 decode GEMV v2 (01:46-, UTC)
- Why: rocprof (cb=4 serve) gemv_iu8g32 lm_head 248320x2560 = 1.511 ms/call vs roofline ~0.66 ms (635 MB @ ~960 GB/s); MQ4P's gemv_q8_0 lm_head max 1.266 ms. IU8G32 total 1.80 s vs Q8_0 0.91 s.
- v2: wave/row(s), lane strides 16-byte code chunks (uint4), 4x float4 x loads, R rows/wave (R=4 m>=32768, 2 m>=4096, else 1), block 128, fallback v1 if HIPFIRE_IU8_GEMV=v1 or not 16B-aligned (K%16!=0 rows).
- test iu8g32_v2_gemv_gpu (01:47 UTC): 42 cases (random M 1..40000 x K 32..9216 + real 4B embed 248320x2560 and LA out_proj 2560x4096): worst rel v1 6.319e-8, v2 3.892e-8; v2 bit-deterministic over 2 runs. PASS.
- ps_i2 (01:50 UTC) ran on the PRE-C6 binary (build14 not yet run): fresh v1 baseline reproduces gemv_iu8g32 1.511 ms/call, solo 4.40 s, conc2 4.10 s.
- build14 (C6) EXIT=0 01:59 UTC. ISA (hipcc --genco gfx1100 -O3, clang-offload-bundler unbundle, llvm-objdump -d; ~/work/iu4/isa/iu8v2.s): v1 gemv_iu8g32 = 1x global_load_b32 + global_load_i8 per lane-iter; v2_r1 = 5x global_load_b128 (1 code uint4 + 4 x float4), v2_r4 = 8x global_load_b128, decode v_bfe_i32 + v_cvt_f32_i32 + v_fmac_f32 (f32 activation; no int dot - see DOT4 variant below).
- A/B pserve 4B IU4P cb=4 interleaved v2/v1/v2 (02:05-02:13 UTC) rocprof per-call:
  | kernel | v2 run a | v1 (HIPFIRE_IU8_GEMV=v1) | v2 run b |
  |---|---|---|---|
  | lm_head 248320x2560 | 0.7724 ms (v2_r4) | 1.5108 ms | 0.7722 ms |
  | out_proj residual 2560x4096 | 17.28 us (v2_residual_r1) | 34.25 us | 17.19 us |
  | wall solo 256 tok | 4.17 s | 3.42 s | 3.13 s |
  | wall conc2 2x256 | 3.40 s | 4.06 s | 3.52 s |
  lm_head = 1.96x faster, 85% of the 0.66 ms roofline; wall times are noisy under rocprof (first-request warmup), kernel times are the measurement.
- C6 gates (02:06-02:10 UTC, 4B IU4P): c6 (serve sequential) unstable 0 / conc_mismatch 0 / errors 0 / klog 0; shas == C4 on hash/code/math/ctx1k/ctx4k, story differs from char 1660 of 3705 (C6 907 tok vs C4 855 tok, both coherent to the end; v2 reorders the out_proj/lm_head reduction -> greedy drift, hypothesis = near-tie, KL below). c6cb (cb=4) 0/0/0/klog 0, all 6 shas == C5 cb (batched route runs out_proj through the WMMA GEMM; only lm_head uses v2 there).

## C7/C8 integer-activation decode (02:15- UTC = 06:15- Dubai)

### Where the rotation lives (quantizer -> runtime), and the fwht4 KV mapping
- Quantizer: `quantize_mq4g256_v2` (hipfire-quantize quant_fwht.rs:357-376) applies
  `cpu_fwht_256(group, signs1, signs2)` per 256-group *before* 4-bit min/max quant;
  signs = `gen_fwht_signs(42,256)` / `gen_fwht_signs(1042,256)` (pipeline.rs:967;
  LCG `s = s*1103515245+12345 & 0x7fffffff`, sign = bit16, dispatch.rs:2086).
  `repack_mq4v2` -> IU4 keeps W already rotated, so W x = (H W)(H x).
- Runtime: `rotate_x_mq` (rdna-compute gemv.rs:6683) rotates x with the same signs
  (`ensure_mq_signs`) into `x_rot` once per layer input; every IU4 GEMV consumes
  `x_rot`. So for IU4 tensors "int8/int4 activations" are *already* "int8/int4+FWHT":
  the integer quant happens on the rotated x inside the GEMV (per 32 lanes), no
  second rotation and no requant needed.
- fwht4 KV mode (model for the knobs): turbo_common.h:81 `fwht_shfl_forward`
  (4 values/lane, shfl butterflies), :168 `fwht_shfl_forward_256` (256-wide, 32 lanes x 8);
  signs from KvCache gen_fwht_signs (llama.rs:7627); knob parsed once in kv_mode.rs
  (~199-269: `"fwht4"|"asym4"|"turbo4" => Fwht4`). Mapping used here:
  | fwht4 KV | IU4 decode |
  |---|---|
  | FWHT-256, seeded +-1 signs, block 256 | identical FWHT-256, seeds 42/1042 (the weight's own), `rotate_x_mq` |
  | 4-bit K per block, scale from amax | int8 (dot4) or int4 (dot8) x per **32**, d = amax/127 or amax/7 |
  | `HIPFIRE_KV_MODE=fwht4` parsed once | `HIPFIRE_IU4_DECODE=f32|dot4|dot8`, `HIPFIRE_IU8_GEMV=v2|v1|dot4`, OnceLock |
  Existing prior art: `rotate_x_mq_i4_batched` (gemv.rs:6880) = FWHT + in-register
  `block_i4_128` sidecar for the MQ4 int4 MMQ prefill.
- IU8G32 (qt59: embed/lm_head, LA out_proj) is an UNROTATED Q8_0 relayout
  (pipeline.rs:263-275). int8 activations there = plain W8A8 (C8 below).
  "int8+FWHT" for qt59 would need rotated requantized weights (new files alongside,
  not done; would ask before replacing anything) - UNTESTED.

### C7: `gemv_iu4g256_dot4` (W4A8, `HIPFIRE_IU4_DECODE=dot4`)
- Lane owns 8 rotated x; amax over 4 lanes (32 elems, shfl_xor 1,2); q = clamp(rint(x*127/amax));
  dot = V_DOT4_I32_IU8(w_hi, a_hi, V_DOT4_I32_IU8(w_lo, a_lo, 0)) with unsigned nibbles
  (w & 0x0F0F0F0F, (w>>4) & 0x0F0F0F0F); sumq the same with 0x01010101;
  y += (d*sc)*dot + zp*(d*sumq).
- ISA (llvm-objdump of hipcc --genco gfx1100, ~/work/iu4/isa/gemv_iu4.s):
  `v_dot4_i32_iu8 ... neg_lo:[0,1,0]` encoding `CC1640xx 5A..` -> VOP3P opcode 0x16=22
  (RDNA3 ISA §16.10, V_DOT4_I32_IU8), NEG_LO[1]=1 => src1 (activation) signed, src0 (weight) unsigned
  (§7.9/VOP3P NEG semantics for IU8/IU4 dots). Per kernel: 4x v_dot4_i32_iu8, 8 v_rndne_f32, 8 v_med3_i32.
  dot8: 2x `v_dot8_i32_iu4 neg_lo:[0,1,0]` `CC18..` -> opcode 0x18=24 (§16.10 V_DOT8_I32_IU4).
  f32 `gemv_iu4g256`: 49 v_fmac_f32, no integer dot.
- Numerics (t_dot4.out, 45 cases: M {1,7,129,4097} x K {256..9216}, uniform + heavy-tailed x, 7 real 4B tensors):
  dot4 vs CPU contract worst 3.38e-6 (gate <1e-5) PASS; deterministic; residual == plain+seed (<6e-8).
  vs f64 dequant ref (rel = max|err| / max|ref|), worst: **f32 1.50e-7, dot4 4.02e-2, dot8 4.12e-1**;
  real tensors: f32 ~1e-7, dot4 3.4-4.3e-3, dot8 5.2-8.0e-2 (dot8 ~15-20x worse than dot4).
  (First test run failed on the residual assertion: metric bug - rel denominator shrank when the
  seed was added for m=1; replaced by direct d4r == d4+seed check.)

### C8: `gemv_iu8g32_a8_r{1,2,4}` (+residual) W8A8 for qt59 (`HIPFIRE_IU8_GEMV=dot4`)
- v2 lane map; x per 16-chunk quantized once (32-group = lanes c, c^1, shfl_xor 1), reused over R rows;
  4x V_DOT4_I32_IU8 per row-chunk with both sign bits; y += (d_w*d_x)*dot.
- First compile failed: `__builtin_amdgcn_sdot4` needs dot1-insts (not on gfx11). Fixed with
  `__builtin_amdgcn_sudot4(true,w,true,a,c,false)`; objdump prints the alias `v_dot4_i32_i8`
  but encoding `CC16400F 7A..` = opcode 22 with NEG_LO=[1,1] (signed x signed). r1: 4 dots, r4: 16.

### C8 numerics (t_a8.out, 42 cases incl. real embed 248320x2560 + LA out_proj 2560x4096, uniform + heavy x)
- a8 vs CPU W8A8 contract worst 1.13e-7 (gate 1e-5) PASS, deterministic, residual ok; vs f64 ref: v2 (f32 x) 1.70e-7, a8 2.30e-2.
- v1/v2 regression re-run in same binary: 42 cases, worst v1 6.32e-8, v2 3.89e-8 PASS.

### Gate 2: decode-path KL/PPL vs MQ4P (iu4_kld, per-token forward_scratch), 4B, 02:21-02:27 UTC
ref = MQ4P 4B, same build. text = first 1024 tokens of benchmarks/calib/bartowski_v3.txt (1023 scored);
gen = 256 greedy tokens of an ISO-8601 python prompt. KL = mean KL(ref||cand) over ref top-64 + remainder.
| variant (IU4P 4B) | text ident | top1 /1023 | first div | KL mean | KL max | PPL (ref 4.6923) | gen top1 /256 | gen KL mean | gen PPL (ref 1.2462) |
|---|---|---|---|---|---|---|---|---|---|
| MQ4P self (sanity) | 1023 | 1023 | - | 6e-20 | 3e-16 | 4.6923 | | | |
| f32 IU4 + IU8 v1 | **1023 (bit-exact)** | 1023 | - | 6e-20 | 3e-16 | 4.6923 | 256 (bit-exact) | 5e-19 | 1.2462 |
| f32 IU4 + IU8 v2 (default) | 0 | 1020 | 574 | 3.81e-5 | 2.47e-3 | 4.6945 | 255 | 2.35e-5 | 1.2468 |
| IU8 dot4 (W8A8, unrotated) | 0 | 1016 | 340 | 7.34e-4 | 2.01e-1 | 4.6893 | 255 | 1.43e-4 | 1.2472 |
| IU4 dot4 (W4A8 + FWHT) | 0 | 1017 | 91 | 3.58e-4 | 7.91e-2 | 4.6917 | 254 | 1.25e-4 | 1.2457 |
| IU4 dot4 + IU8 dot4 (all int8) | 0 | 1016 | 176 | 8.38e-4 | 2.23e-1 | 4.6951 | 255 | 2.23e-4 | 1.2475 |
| IU4 dot8 (W4A4 + FWHT, branch original) | 0 | 921 | 28 | **4.19e-2** | 4.41 | **4.8937 (+4.3%)** | 235 | 2.34e-2 | 1.2856 |
| IU4 dot8 + IU8 dot4 (int4 mix) | 0 | 940 | 20 | 3.64e-2 | 1.78 | 4.8474 | 241 | 2.62e-2 | 1.2971 |
Findings:
- C1 f32 IU4 GEMV + fused rows reproduce MQ4P bit-for-bit on the decode path (with IU8 v1). The only
  default-path deviation is C6 v2's different f32 summation order (KL 3.8e-5, PPL +0.05%).
  Hypothesis (untested): an ordering-matched v2 would restore bit-exactness at the same speed.
- int4 activations (dot8) are ~100x worse KL than int8 (dot4) and ~1100x worse than f32; PPL +4.3%.
  => keep f32 as default; dot4 is the integer option worth keeping; dot8 only as A/B.
- int8 on the unrotated IU8G32 tensors (lm_head) costs more KL (7.3e-4) than int8 on the rotated IU4
  tensors (3.6e-4) despite 8-bit weights: lm_head's activation outliers are not spread by FWHT. This is
  the motivation for int8+FWHT on qt59 (needs rotated requant, not done).

## Checkpoint 2026-10-30 ~06:0x Dubai: prefill gating fix (C7 follow-up)
- Bug found: `is_batchable_la` (prefill.rs:4132) and its unit test (prefill.rs:7883) used
  `!iu4_decode_dot8()`, but that function returns true for every integer decode mode, so
  `HIPFIRE_IU4_DECODE=dot4` also turned off the IU4 MMQ prefill and fell back to per-token W4A4 prefill.
  Fix: gate on `iu4_decode_mode() != Iu4DecodeMode::Dot8`. Only dot8 keeps the original per-token prefill.
  The fused-decode-key skips (gated_mlp.rs:706, steps.rs:377) stay on for every integer mode,
  because those fused kernels are f32-activation only.
- The 4B KL table above is not affected: iu4_kld scores token by token, which is the decode path.
- The A/B (ab.sh) was stopped before it started (it was still waiting for done27) and relaunched after rebuild b18.
- gate.py: added opt-in `GATE_KEYS`/`GATE_ROUNDS` so the 27B gate runs as two parts under 10 min each.
  Part A = hash,code,math; part B = ctx4k,story,ctx1k. Both use 4-lane concurrent rounds at cb=4.
  Default behaviour is unchanged. The chain is chainG.sh (runs after ab DONE).

## Gate 2: decode-path KL/PPL vs MQ4P (27B), 06:27-06:33 Dubai
Ref: Qwen3.8-27B-MQ4P, 512 tokens of bartowski_v3 (511 scored), iu4_kld per-token. MQ4P ref PPL row: iu8v1 is bit-identical (511/511), PPL 5.8400.

| variant | bit-ident | top1 /511 | first div | KL mean | KL max | PPL |
|---|---|---|---|---|---|---|
| f32 + IU8 v1 | 511 | 511 | none | ~0 | 1.9e-16 | 5.8400 |
| f32 default (IU8 v2) | 0 | 510 | 448 | 2.06e-5 | 3.91e-3 | 5.8412 |
| iu8dot4 | 0 | 510 | 448 | 1.03e-4 | 1.15e-2 | 5.8386 |
| iu4dot4 | 0 | 509 | 240 | 8.27e-5 | 1.70e-3 | 5.8394 |
| allint8 (iu4dot4+iu8dot4) | 0 | 510 | 448 | 1.59e-4 | 1.14e-2 | 5.8395 |
| iu4dot8 (W4A4) | 0 | 484 | 28 | 1.68e-2 | 6.02e-1 | 5.9258 (+1.5%) |

The ordering matches 4B: f32 < int8 variants (~1e-4) << dot8 (~1.7e-2). 27B dot8 PPL +1.5% (4B +4.3%).

## Decode-mode speed A/B (4B IU4P), 3 interleaved rounds, 06:33-06:35 Dubai
Command: `hipfire bench <4B IU4P> --pp 512 --ctx 128,4096 --tg 64 --runs 2 --warmups 2 --spec off`.
Without `--matrix`, bench IGNORES --pp/--ctx/--tg and runs the standard 24-token prompt / 128-token decode,
so these rows are short-context decode (decode median tok/s, per round):

| mode | r1 | r2 | r3 | vs f32 |
|---|---|---|---|---|
| f32 (default) | 195.55 | 194.85 | 195.70 | - |
| iu4dot4 | 151.90 | 151.95 | 152.05 | -22% |
| iu4dot8 | 150.20 | 150.25 | 149.75 | -23% (and prefill 150 tok/s: per-token) |
| iu8dot4 | 194.50 | 195.10 | 194.45 | -0.4% (noise level) |
| allint8 | 151.30 | 150.95 | 151.30 | -23% |

Finding: the IU4 integer decode modes are ~22% SLOWER than f32 on 4B. Hypothesis (untested): integer modes skip the fused
`fused_iu4g256_rows` decode keys (gated_mlp.rs:706, steps.rs:377), so they launch more kernels and add an extra x-quantize pass.
A fused dot4 rows kernel would be the fix. Not done.

## Matrix A/B (ab.sh phase 2): m_pre_* INVALID
`~/work/iu4/pre/hipfire bench` spawned the NEW daemon from target/release: its logs bind `gemm_iu4g256_mmq_*`, which only
exists in the new tree. ab.sh did not set HIPFIRE_DAEMON_BIN. The m_new and m_mq4p rows are valid:
new pp512 5798/5796/5815, pp2048 5490/5476/5477, tg64@128 204.6/205.0/204.3, tg64@4096 192.4/192.6/192.4;
MQ4P pp512 8070/8101/8087, pp2048 7442/7444/7446, tg64@128 201.1/201.0/200.2, tg64@4096 189.7/189.3/188.7.
Corrected rerun: ab2.sh (pre pinned to pre/daemon, plus dot4/allint8 matrix rows, which now use MMQ prefill), queued after chainG.

## Gates 3/4 on the C8 build (g8 = current default), 06:37 Dubai
g8_iu4p4b solo: 0 unstable, 0/12 concurrent mismatch, 0 errors, klog 0. All 6 hashes are identical to c6_iu4p4b, so C7/C8 and
the debug-print removal are no-ops on the default path. dot8_iu4p4b (earlier) == pre_iu4p4b on all 6 hashes.

## ab2: corrected 3-round interleaved matrix (pre pinned via HIPFIRE_DAEMON_BIN), 4B, 03:00–03:28 UTC

`hipfire bench <model> --matrix --pp 512,2048 --ctx 128,4096 --tg 64 --runs 2 --warmups 2 --spec off`; tok/s, rounds 1/2/3 interleaved (median):

| row | pp512 | pp2048 | tg64@128 | tg64@4096 |
|---|---|---|---|---|
| new IU4P (f32 decode) | 5784/5794/5793 (5793) | 5451/5456/5446 (5451) | 204.1/204.4/203.9 (204.1) | 191.8/192.3/192.2 (192.2) |
| pre IU4P (branch tip) | 125.4/125.6/125.3 (125.4) | 124.6/124.6/124.6 (124.6) | 128.9/128.9/129.1 (128.9) | 124.4/123.9/124.4 (124.4) |
| new, IU4 dot4 | 5727/5755/5765 (5755) | 5425/5428/5439 (5428) | 155.7/155.5/155.4 (155.5) | 149.2/149.0/149.2 (149.2) |
| new, dot4 + IU8 dot4 | 5776/5736/5776 (5776) | 5464/5457/5463 (5463) | 155.1/155.5/155.3 (155.3) | 148.7/148.6/148.4 (148.6) |
| MQ4P (reference) | 8054/8084/8027 (8054) | 7432/7408/7395 (7408) | 200.9/201.0/200.9 (200.9) | 189.9/189.3/189.1 (189.3) |

New vs pre: prefill 46.2x (pp512), decode +58% (tg64@128). IU4P prefill = 72% of MQ4P (item 2). IU4P f32 decode = MQ4P +1.6%.
dot4 decode −24% vs f32 (item 1, C9 below). All rows rc=0, klog=0.

## 27B gates parts C/D (chain27c, cb4, new build C8)

- g27newC (GATE_KEYS=ctx4k,ctx1k, 2 rounds): solo ctx4k 3304d572ccb9 STABLE, ctx1k fe0b307cf1ec STABLE; concurrent 0/8 mismatch; 0 errors; klog 0; 122 s.
- g27newD (story,ctx4k, 1 round, solo once): story c9864c21e22b (1024 tok) STABLE, ctx4k STABLE; concurrent 0 mismatch; klog 0; 310 s.
- With g27newA (hash/code/math, 0/8), all six 27B prompts are now covered solo + concurrent on the new build: 0 unstable, 0 concurrent mismatch.
- Solo hashes equal the g27newB solo hashes (same build), so they are reproducible across server restarts.
- 27B greedy vs pre: hashes differ on every prompt (pre = W4A4 numerics, new = f32/MQ4V2-exact). Expected; KL on 27B is the quality gate (f32 KL 2.06e-5 vs MQ4P).

## C9 — IU4 W4A8 decode v2 (`HIPFIRE_IU4_DECODE=dot4`)

**Root cause of the C8 dot4 slowdown** (156.7 vs 204.9 tok/s f32, 4B tg64@128):
1. Each row re-quantized x per 256-group (2 ds_bpermute, 8 v_rndne, 8 v_med3 per group per row), so the work was repeated M times.
2. Integer modes skipped the fused qkv/qkvza/gate_up keys (`iu4g256_fusion_ok`) and the IU4 down arm (gated_mlp.rs). The generic fallback allocated and freed per token.

**Design:**
- `iu4_quant_x_a8` (grid K/256, block 32) quantizes x once into `int4 xq[g*32+t] = {a_lo, a_hi, sumq, float_as_int(d)}`. The per-lane math is identical to C8 `iu4_row_dot_a8`.
- `iu4_row_dot_a8q` uses the same accumulation expression, so results are bit-identical.
- Kernels: `gemv_iu4g256_dot4q`, `_dot4q_residual`, `fused_iu4g256_dot4q_rows` (up to 4 matrices). Each is one row per block with `__launch_bounds__(32,16)`.
- Scratch is a fixed 256 KiB `iu4_a8_x_scratch`, allocated once and never regrown, so graph capture stays valid. It errors if K*2 is over the limit.
- `iu4_decode_w4a4()` is true only for dot8. dot4 now uses the fused keys and the down arm; dot8 keeps the per-token route.

**ISA** (llvm-objdump, gfx1100, `isa/gemv_iu4.s`): `gemv_iu4g256_dot4q` has 10× `v_dot4_i32_iu8` (VOP3P op 22, RDNA3 ISA §16.10; unroll 4 + remainder), 5× `global_load_b128` and 10× `global_load_b32`. The inner loop has no ds_bpermute, v_rndne or v_med3.

**Numerics** (`tests/iu4_dot4_gemv_gpu.rs`, dot4q arm): 45 cases (38 random including odd M, M=1, tails, large K, plus 7 real 4B tensors).
- dot4q plain and residual are bit-identical to the C8 dot4 kernel.
- A fused 3-matrix launch is bit-identical to per-matrix dot4q.
- Two runs are bit-identical.
- Worst error vs the CPU W4A8 contract is 3.379e-6. dot4 vs the f64 ref is 4.02e-2 (f32 path: 1.5e-7). klog 0.

**A/B** (ab3, 4B, 3 interleaved rounds, medians, tok/s):

| build/mode | pp512 | pp2048 | tg64@128 | tg64@4096 |
|---|---|---|---|---|
| C8 dot4 | 5836 | 5502 | 156.67 | 149.84 |
| C9 dot4 | 5809 | 5490 | **192.42** | **181.26** |
| C9 f32 (default) | 5818 | 5493 | 204.91 | 192.88 |
| C9 dot4 + IU8 dot4 (all-int8) | 5841 | 5492 | 192.25 | 181.36 |
| MQ4P | 8073 | 7465 | 200.59 | 189.48 |

Per-round tg64@128: C8 dot4 156.99/156.67/156.53; C9 dot4 192.74/192.42/191.74. dot4 decode is +22.8% (tg128) and +21.0% (tg4096), now 6% below f32. Prefill is unchanged (within noise).

**KL / PPL vs MQ4P** (unchanged from C8 to every digit, as expected from bit-identity):
- 4B dot4: text KL 3.575e-4, top1 1017/1023, PPL 4.6917; gen 1.245e-4, top1 254/256.
- 4B dot4 + IU8 dot4 (all-int8): 8.383e-4, PPL 4.6951.
- 4B f32: 3.813e-5, PPL 4.6945.
- MQ4P self-PPL: 4.6923.
- 27B dot4: 8.271e-5, top1 509/511, PPL 5.8394. 27B f32: 2.055e-5, PPL 5.8412.

f32 has ~4–10× lower KL and is also faster, so it **stays the default**.

**Gates** (6 prompts: hash/code/math/story/ctx1k/ctx4k):
- 4B g9d4 and g9d4cb: hashes identical to g8d4 and g8d4cb, 0 unstable, 0/12 concurrent mismatch.
- 4B g9 (default f32): identical to g8, 0 unstable, 0/12.
- klog 0 throughout.
- 27B g27c9d4A (hash/code/math, solo + cb4 concurrent, dot4): 0 unstable, 0 concurrent mismatch, klog 0; hashes identical to g27newA (C8 f32 default).

## Item 2 — prefill profile, IU4P vs MQ4P (4B, pp2048)

Method: `ppdrive.py` runs the daemon directly under `rocprofv3 --kernel-trace --stats`. It sends a load, then 4× `bench_prefill 2048`, then EOF so the daemon exits cleanly and the trace flushes. (The `hipfire bench` + HIPFIRE_DAEMON_BIN wrapper route wrote no CSV because the daemon is killed rather than exiting.) Under the tracer, IU4P runs ~391 ms/pass (5234 tok/s) and MQ4P ~302 ms/pass (6788 tok/s).

GPU time summed over all 4 passes:

| role | IU4P kernel | µs | MQ4P kernel | µs | Δ µs |
|---|---|---|---|---|---|
| qkv/gate_up GEMM (set) | gemm_iu4g256_mmq_full_set_x128 | 580658 | gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151 | 343541 | +237117 |
| down/out GEMM (add) | gemm_iu4g256_mmq_full_add_x128 | 212283 | …mmq_iu4_full_add_lf16_col_gfx1151 | 141126 | +71157 |
| small-M GEMM | gemm_iu4g256_mmq_x128 (n=768) | 81667 | gemm_mq4g256v2_residual_wmma_gfx11_mw4_lds | 11408 | +70259 |
| IU8G32 / Q8_0 wo | gemm_iu8g32_residual_wmma | 177217 | gemm_q8_0_residual_wmma | 149040 | +28177 |
| act quant | quantize_q8_1_mmq_ds4_x128 + rmsnorm/silu rotate | 40233 | rotate_i4 fused (rmsnorm/silu/mq_rotate) | 44937 | −4704 |
| attention, GDN, conv | — | ~327k | — | ~333k | ≈0 |

Total 1469 ms vs 1076 ms (+393 ms, ≈98 ms/pass, which matches the measured 391 vs 302 ms).

Reading (measured): ~95% of the gap is the three IU4 MMQ GEMMs. MQ4P's prefill kernels are W4A4 integer MMQ: activations are quantized to int4 by the fused `*_rotate_i4` kernels and the GEMM uses the lf16_col layout. IU4P's MMQ is W4A8 (q8_1 activations).

Hypotheses (untested):
- (a) Most of the set/add gap is the A8-vs-A4 WMMA work plus layout/tile efficiency.
- (b) The small-M gap (+70 ms) is a tile shape that is poor for small N. MQ4P routes those matrices to a WMMA mw4_lds kernel instead.
- (c) IU8G32 wo is 19% slower than Q8_0 WMMA for the same math.

## C10 — IU4 W4A4 prefill + small-M WMMA (6d9af850d, test 3e2e7c795, iu4_kld pscore f5d172365)

What: MQ4P's int4-activation MMQ kernels (`..._mmq_iu4_full_{set,add}_lf16_col`, generic tail) compiled with
`HIPFIRE_IU4_PLANAR` planar qt57 loads -> `gemm_iu4g256_w4a4*`; m<128 projections -> `gemm_iu4g256_residual_wmma_gfx11_mw4_lds`
(f16 X converted once per multi-GEMM, uncached convert + `invalidate_fp16_cache`, fixes a stale pointer-keyed cache).
Knobs: `HIPFIRE_IU4_PREFILL_ACT=a4` (default) | `a8` (C7 W4A8 q8_1 path); `HIPFIRE_IU4_SMALLM=wmma` (default) | `mmq`.

### Numerics (t1.out, 10:20–10:22 UTC = 14:20–14:22 Dubai)
`cargo test -p hipfire-runtime --release --test iu4_w4a4_gemm_gpu -- --ignored` with IU4_REAL/MQ_REAL = 4B IU4P/MQ4P:
490 cases, all `a4_bitdiff=0`, `mwl_bitdiff=0`, `det_diff=0` vs the MQ4G256V2 kernels on twin weights (random + real
in_proj_a m=32, qkv m=8192, z m=4096, down k=9216, gate m=9216, k_proj m=1024, o_proj k=4096; N in {1,77,512}; odd/tail M).
vs f32 reference: a4 worst rel 0.208 (random), 0.045–0.113 real (int4 activation noise, same as MQ4P); mwl worst rel 9.1e-4.

### ISA (llvm-objdump / llvm-readelf --notes on ~/.hipfire_kernels/gfx1100, unbundled)
- First version: `gemm_iu4g256_w4a4_full_{set,add}_lf16_col` 96 VGPR, **vgpr_spill_count=2, private 12 B** (MQ twin: 0).
  Fix: uniform column base `A + kb*128 - 8` + 32-bit row offset `row*gpr*136` (header: `A + gpr*128 + kb*8` + 32-bit) ->
  96 VGPR, 0 spills, 0 private. Test re-run after fix: 490/490 bitdiff=0.
- `gemm_iu4g256_w4a4` (generic) 193 VGPR, 0 spills; `gemm_iu4g256_residual_wmma_gfx11_mw{4,8}_lds` 90 VGPR, 0 spills, 8 KiB LDS.
- MQ code objects unchanged by the `#ifdef` patches: disassembly sha identical pre/post for
  `gemm_mq4g256v2_residual_mmq_iu4` (0d0b9bd1bf21), `_gridspec` (a4f9db208282), `gemm_mqv2_wmma_gfx1100_mw_lds` (42ea4d2268d2).

### Prefill-path KL/PPL, 4B (iu4_kld pref/pscore, 1023 tokens, 10:35–10:37 UTC = 14:35–14:37 Dubai)
Refs: `ref4` = MQ4P per-token decode (PPL 4.6923); `refP4` = MQ4P prefill (W4A4, PPL 4.7699).

| run | vs ref | mean KL top64 | max KL | top1 agree | PPL |
|---|---|---|---|---|---|
| MQ4P prefill | MQ4P decode | 0.0388 | 2.16 | 937/1023 | 4.7699 |
| IU4P a4 (C10 default) | MQ4P decode | 0.0345 | 1.35 | 938/1023 | 4.8722 |
| IU4P a8 (W4A8) | MQ4P decode | **0.0016** | 0.149 | **1007/1023** | **4.7058** |
| IU4P a4 | MQ4P prefill | 0.0650 | 7.55 | 910/1023 | 4.8722 |
| IU4P a8 | MQ4P prefill | 0.0471 | 5.91 | 933/1023 | 4.7058 |
| IU4P a4 + SMALLM=mmq | MQ4P prefill | 0.0731 | 4.42 | 907/1023 | 4.9541 |
| IU4P a8 + SMALLM=mmq | MQ4P prefill | 0.0461 | 5.83 | 939/1023 | 4.6923 |

Determinism: a4 repeat run identical (same KL/PPL/top1/first_div). Reading: W4A4 activation quantization costs
PPL (+0.10 vs decode-equivalent a8, +2.2%); W4A8 prefill is ~decode-exact (KL 0.0016 vs MQ4P decode), better than MQ4P's own
W4A4 prefill (0.0388). Small-M WMMA (f16 X) beats small-M MMQ int4 on a4 (0.065 vs 0.073, PPL 4.87 vs 4.95).

### Speed A/B, 4B, 3 interleaved rounds (chW4b, 10:41–10:45 UTC = 14:41–14:45 Dubai)
`hipfire bench <model> --matrix --pp 512,2048 --ctx 128,4096 --tg 64 --runs 2 --warmups 2 --spec off`, HIPFIRE_DAEMON_BIN pinned,
GPU idle-gated, klog=0 every run. Median of 3 rounds (tok/s); logs ~/work/logs/abW/<label>_<r>.log.

| config | pp512 | pp2048 | tg64@128 | tg64@4096 |
|---|---|---|---|---|
| C9 (h1b6, W4A8 prefill, small-M MMQ) | 5864 | 5518 | 204.9 | 192.7 |
| C10 a4 (default) | **7612** | **7025** | 205.5 | 193.2 |
| C10 a8 | 6088 | 5718 | 205.3 | 192.9 |
| C10 a4 + SMALLM=mmq | 7376 | 6815 | 204.4 | 192.4 |
| MQ4P | 8102 | 7479 | 201.0 | 189.8 |

a4 vs C9: +29.8% pp512, +27.3% pp2048; vs MQ4P −6.0% / −6.1% (remaining gap, unprofiled). Small-M WMMA vs MMQ: +3.2% / +3.1% (a4).
a8 (W4A8 at C10) vs C9: +3.8% / +3.6% (small-M WMMA only). Decode unchanged (±0.3%), still +2% over MQ4P.

### Gates 3/4, 4B (gs.sh, 10:45–10:50 UTC = 14:45–14:50 Dubai)
w4 (C10 a4 solo serve), w4cb (`--continuous-batch-size 4`), w4a8 (a8): each `unstable 0 conc_mismatch 0 errors 0`, klog 0
(solo twice-identical per key = determinism; solo vs 4-way concurrent identical = EQ4K-style check). ctx4k needle
answered correctly (BLUE-4471 / Dr. Okafor) in w4 and w4cb; code/story text coherent. Greedy hashes differ from C9 for
hash/math/story/ctx1k/ctx4k (prefill numerics changed: W4A8 -> W4A4); `code` equal to C9 in solo. The cb4 server's
greedy text differs from the solo server's (code diverges at "Assumes both input lists" vs "Assumes list1 and list2");
C9 also diverged solo-vs-cb4 on `story`, so a cb4-server path difference predates C10. Root cause not investigated (untested).

### C10 follow-ups (2026-11-02 10:51-11:20 UTC = 14:51-15:20 Dubai)
- 27B prefill-path KL (iu4_kld pscore, 511 steps):
  | run | ref | mean KL top64 | top-1 | PPL |
  |---|---|---|---|---|
  | MQ4P prefill | MQ4P decode | 0.0179 | 477 | 6.0610 |
  | C10 a4 | MQ4P prefill | 0.0344 | 459 | 5.9183 |
  | C10 a8 | MQ4P prefill | 0.0175 | 478 | 5.8249 |
  | C10 a4 | MQ4P decode (PPL 5.8400) | 0.0204 | 480 | 5.9183 |
  | C10 a8 | MQ4P decode | 0.00061 | 506 | 5.8249 |
  (first a8_vsdec attempt OOMed at 10:56 UTC, VRAM of the previous run not yet freed; rerun 11:17 UTC with a VRAM<2000 MB gate.)
- 27B A/B in chW27 is INVALID except round 1 (p27_1 pp128 29.00 / tg 29.33; h1_27_1 pp512 1028.7, pp2048 1028.0, tg@128 46.77, tg@4096 45.19).
  From 10:58:50 UTC every bench died at once: "FATAL: GPU ... already reserved by holder PID 686978" (holder gone by 11:20; most likely the
  h1_27_1 daemon still tearing down - idle() gated only on GFX%, not VRAM; hypothesis). Then chX.sh, which waited on grep "DONE",
  matched "ABDONE" and started at 10:59:14, overlapping the w4cb27 gate (errors 8) with dv_mq_solo (errors 12). Both are my scheduling bugs.
  w4s27 (11:05, alone) = unstable 0 / conc 0 / errors 0. Clean reruns queued in chR.sh, now gated on GFX<10% AND VRAM<2500 MB.
- Prefill gap profile (rocprofv3, ppdrive.py, 4B, 5x pp2048 incl. 1 warm, 11:17 UTC, GPU alone): C10 1145.0 ms vs MQ4P 1077.1 ms kernel time (+68 ms, 6.3%).
  | bucket | C10 ms | MQ4P ms | delta |
  |---|---|---|---|
  | act rotate+int4 quant | 92.8 (quantize_int4_mmq_ds128 68.3 + fused_silu_mul_mq_rotate 14.3 + fused_rmsnorm_mq_rotate 10.2) | 41.8 (fused_*_rotate_i4{,_b8} 22.3+19.5) | +51.0 |
  | LA wo | gemm_iu8g32_residual_wmma 180.6 | gemm_q8_0_residual_wmma 146.1 | +34.5 |
  | W4A4 MMQ set+add | 470.5 | 487.0 | -16.5 |
  | everything else | ~equal (attention 154, GDN 132, conv 48.6) | | ~0 |
  => gap fully explained: (1) IU4 path quantizes x in a separate kernel instead of MQ4P's fused rotate->int4 kernels (same FWHT seeds,
  so the _i4 fused kernels should be reusable: candidate C11, untested); (2) IU8G32 wo layout 24% slower than Q8_0 WMMA (as seen before).
- batch-4 vs solo divergence (4B, gate.py solo phase; shas per key code/ctx1k/ctx4k/hash/math/story):
  | label | server | shas |
  |---|---|---|
  | dv_c10_cb1 | C10 IU4P cb1 | 166f783c 8e3f23e5 a8398834 6dbe5cb4 c101d4fb d4eed6f4 (= w4_iu4p4b) |
  | dv_c10_cb4_hip | C10 IU4P cb4, REPLAY_BACKEND=hip | b4408528 a7e12605 0ca372b6 1fe51a95 2ca2ea0f fe1d7775 (= w4cb_iu4p4b) |
  | dv_c10_cb4_auto | cb4, REPLAY_BACKEND=auto | identical to cb4_hip |
  | dv_c10_cb4_ch4k | cb4, PREFILL_CHUNK=4096 | identical to cb4_hip |
  | dv_mq_solo | MQ4P cb1 | 166f783c 23210b84 65ce263d 507cb95b dce45d88 5f291a9a |
  | dv_mq_cb4 | MQ4P cb4 | 166f783c 23210b84 65ce263d d5deb4c5 dce45d88 56f72465 |
  Each config deterministic (unstable 0). Serve log: KV backend=legacy (not VMM), so replay backend does not change anything here.
  => NOT the VMM chunk-boundary class: prompts are <1024 tok except ctx4k, and pinning chunk 4096 changes nothing. It is the cb>1 server
  taking a different (multi-slot) kernel path even for a lone request. MQ4P has the same effect on 2/6 keys; IU4P on 6/6. Which kernels
  differ (slot GEMM vs solo GEMV, f16 X small-M) is untested.

## C11a — IU4 W4A8 on the iu8 WMMA (checkpoint 2026-11-02 15:35 Dubai)

- New: `HIPFIRE_IU4_PREFILL_ACT=a8` = `block_i8_128` X (ensure_int8_mmq_x) + `v_wmma_i32_16x16x16_iu8`
  W4A8 (gemm_iu4g256_w4a8_*). Old q8_1 path renamed `a8q`. `HIPFIRE_IU4_I8_TILE=128|256` (auto: 256 when
  ceil(m/256)*ceil(n/256) >= CUs).
- Kernel = MQ4P's gfx11 i8 source + `I8_ASYM` (asym fold, IU4_FOLD_RN DAG: t1=RN(sc*d) p=RN(t1*C) t2=RN(zp*d)
  sum+=fma(t2,s,p); zp plane in LDS, +1024 B / +2048 B) + `HIPFIRE_IU4_PLANAR` (codes at 0, stride 128, header
  plane at gpr*128). Needed because MQ4P's A8 kernel is symmetric-only and both iu4p and mq4p files are asym.
- MQ4V2 symmetric code objects unchanged (measured): clang-offload-bundler --unbundle, llvm-objdump -d sha:
  t128 b8a765c9a764 = b8a765c9a764, t256 bf6cdeafe461 = bf6cdeafe461; v_wmma_i32_16x16x16_iu8 count 96 / 192.
  (First attempt differed: `hdr_sp + I8_W_HDR()` with outer parens re-associated the pointer adds; fixed.)
- Resources (hipcc -O3 gfx1151, before the paren fix): iu t128 118 VGPR, t256 187/188, 0 spill / 0 scratch.
- Build: daemon release BRC=0, snapshot ~/work/iu4/c11a/. Test crates/hipfire-runtime/tests/iu4_w4a8_i8_gemm_gpu.rs
  built (TRC=0); queued in chZ.sh after chR (GPU busy). UNTESTED until then: twin bit-exact, CPU fold emulation,
  f64 rel, determinism, both tiles, model-level KL/speed.
- Next (C11b, not started): route fused_rmsnorm/silu_hin/gated_norm_rotate_mq_i8 producers for IU4G256 (no
  symmetric gate; i8 blocks are weight-independent) into gemm_iu4g256_w4a8_prequant; a4 producers for IU4G256.
- C11a numerics (measured 11:40-11:41 UTC = 15:40-15:41 Dubai), test iu4_w4a8_i8_gemm_gpu, 4B real tensors:
  tile 128 and tile 256 each: 260/260 calls twin_bitdiff=0 (IU4 planar == MQ4V2 asym interleaved twin),
  244/244 emulated calls emu_bitdiff=0 (CPU fold emulation over the downloaded block_i8_128 X: bit-exact
  contract; the 16 unemulated are m4096xk2560xn2048 and real n=512), 260/260 det_diff=0, worst rel vs f64
  9.508e-3 (int8 X), m in {1,7,127,128,129,384,4096}, k in {256,768,1280,2560}, n in {1,63,64,65,130,257,512,2048}.
- chR (27B, 3 interleaved rounds, medians, pp512 / pp2048 / tg64@128 / tg64@4096):
  a4 (C10)   1460.1 / 1456.3 / 46.70 / 45.12
  a8q (q8_1) 1053.2 / 1051.6 / 46.78 / 45.18
  MQ4P       1578.1 / 1574.7 / 46.78 / 45.22
  h1 (pre-C10 build) 1026.8 / 1026.8 / 46.75 / 45.17;  dot4 decode 1028.8 / 1026.7 / 46.16 / 44.61 (-1.3% tg)
  pre (branch base) pp128 28.90, tg64@128 29.22
  Verdict: no decode regression (C10 46.70 vs h1 46.75, within run spread 46.65-46.76); a4 prefill +42% vs h1,
  -7.5% vs MQ4P; a8q -33% vs MQ4P. GATE w4cb27 (cb4 serve) unstable 0 conc_mismatch 0 errors 0.
- chZ: FAILED to measure. `bench --backend slots` on the IU4P model: "forward_batch_slots: embedding format
  IU8G32 has no batched lookup in the slots path (admitted: Q8_0, F32, HFQ4G256, HFQ4G128)". Gap: slots-path
  IU8G32 embedding lookup. (Serve --continuous-batch-size works: w4cb27 gate passed.)

## C11b fused producer sidecars (11:50-12:16 UTC = 15:50-16:16 Dubai)
- Design: the rmsnorm / silu*up / rotate producers emit the MMQ X sidecar next to the f32 x_rot and hand it to `gemm_iu4g256_multi` through `gpu.iu4_pending_x` (consumed only when ptr/n/k/kind match, generation-checked; otherwise falls back to ensure_int*_mmq_x). New `fused_silu_mul_mq_rotate_i8_gfx11` (block_i8_128 quant+emit prepended to the IU4_SIDECAR silu producer). Knob `HIPFIRE_IU4_FUSED_X`: 0 off; unset/1 = a8 fused (default); `all` = also a4.
- a4 fusion is opt-in because the i4 producer emit uses the reduced MSE candidate set, so the sidecar is not bit-exact to `ensure_int4_mmq_x` (8 candidates); ~20% of sidecar bytes differ in t11b. Exact a4 would need the RUNTIME_CAND cand_mode=8 variant (not done).
- RMSNorm producers that keep x_rot in LDS need (k+256)*4 <= 65536; larger k (17408) falls back (the f32 reference itself fails "invalid argument" there).
- t11b (iu4_fused_x_producers_gpu, k {256,768,2560,9728,17408} x n {1,7,64,333,2048} x Norm/Silu/Rot x i4/i8 x awq): SUMMARY cases=224 bad=0 (x_rot bit-exact + deterministic everywhere, i8 sidecars bit-exact, i4 = candset only).
- Model-level bitwise (iu4_kld pref with FUSED_X=0 vs pscore fused, twice): 4B a8 logit_rows_bit_identical 1023/1023 both runs (PPL 4.69521); 27B a8 511/511 both runs (PPL 5.84497). Fused == standalone, deterministic.
- KL (mean top64; refP = MQ4P prefill W4A4, ref = MQ4P per-token decode):
  | run | vs | mean KL | top1 | PPL |
  |---|---|---|---|---|
  | 4B a8 | refP | 0.04496 | 939/1023 | 4.6952 |
  | 4B a8 tile128 | refP | 0.04496 (identical) | 939 | |
  | 4B a8 | ref (decode) | 3.90e-4 | 1013/1023 | |
  | 4B a4all | refP | 3.3e-13 | 1023 | |
  | 4B a4all | ref | 0.03876 | 937 | |
  | 4B a8q | ref | 1.64e-3 | 1007 | 4.7058 |
  | 27B a8 | refP | 0.01724 | 477/511 | 5.8450 |
  | 27B a8 | ref | 1.11e-4 | 511/511 | |
  | 27B a4all | refP | 3.6e-13 | 511 | 6.0610 |
- Speed (hipfire bench --pp 512,2048 --tg 64, 3 interleaved rounds, median of medians, 12:04-12:15 UTC, GPU alone, klog 0):
  | 4B | pp512 | pp2048 | tg64 |
  |---|---|---|---|
  | a4 (C11b default) | 7607.85 | 7033.60 | 204.8 |
  | a4all (fused a4) | 8004.30 (+5.2%) | 7365.15 (+4.7%) | 204.8 |
  | a8 fused (C11b) | 6615.95 | 6169.95 | 204.5 |
  | a8 C11a | 6568.15 | 6151.55 | 205.0 |
  | MQ4P | 8116.15 | 7500.85 | 201.0 |
  | 27B | pp512 | pp2048 | tg64 |
  | a4 | 1468.60 | 1459.65 | 46.8 |
  | a8 fused | 1165.40 | 1162.40 | 46.7 |
  | a8 C11a | 1156.30 | 1154.60 | 46.7 |
  | MQ4P | 1575.10 | 1572.65 | 46.8 |
  - a8 fusion: +0.7%/+0.3% (4B), +0.8%/+0.7% (27B): small, the a8 gap is the matmul (step 3). a8 vs a4: -13% (4B), -21% (27B). a4all closes 4B to -1.4%/-1.8% of MQ4P.
- chZ (batched decode): MQ4P 4B slots r1-3 rc=0; every IU4 slots run failed: embedding IU8G32 had no slots lookup. Step 5 finding: the embedding is not the only gap. forward_slots.rs admits no IU4G256/IU8G32 projection (slots_proj_admissible, slots_plain_gemm_key, slots_residual_gemm_key, lm_head_slots_admissible), so batched slots decode for IU4 needs a real port (KernelKey entries for the IU4 multi GEMM + IU8G32 residual, graph-capture-safe X prep). Embedding arm added (embed_lookup_slots + admit list, uses existing embedding_lookup_iu8g32_batched, device token ids => capture safe); slots lib tests 22/22 pass. Serve --continuous-batch-size (non-slots batched route) already works (w4cb27 gate).
- Step 4 (IU8G32 wo gap) in progress: same ISA mix as Q8_0 (6 b128 + 8 b32 + 1 d16_b16, llvm-objdump/clang -S gfx1100); hypothesis (untested): the planar scale stream (K bytes past the codes) costs an extra scattered cache line per block. Change: hoist scales to one b128 load per 8 blocks when K%256==0 (IU8 ISA now 7 b128, 77 VGPR; Q8_0 ISA unchanged byte-for-byte except cuid). Needs bit-exact test + rocprof A/B.

### C11b gates (chB, finished 16:21 Dubai)
b4a4, b4a8, b4a8cb, b27a8, b27a4: unstable 0, conc_mismatch 0, errors 0 for every gate.

### Step 4: IU8G32 wo WMMA scale hoist (C12)
- Correctness: iu8g32_wmma_gemm_gpu passes 112 cases, worst rel vs f64 1.478e-4.
- A/B: 4B IU4P bench_prefill 2048 x4 under rocprof kernel-trace, c11b vs c12, interleaved 3 rounds.
  gemm_iu8g32_residual_wmma total over 384 calls: c11b 180.57 / 179.97 / 178.77 ms vs c12 141.22 / 140.07 / 140.58 ms, so -22%.
  All kernels total: 1142.4 / 1139.2 / 1133.4 vs 1105.5 / 1096.8 / 1101.3 ms.
- Q8_0 path ISA is identical apart from the cuid.

### a4all KL vs MQ4P decode (refs from the same runs)
| model | a4 (standalone quantizer) | a4all (fused, MQ4P math) |
|---|---|---|
| 4B mean KL | 0.03453 (top1 938/1023, ppl 4.8722) | 0.03876 (top1 937, ppl 4.7699); repeat 0.0387564 |
| 27B mean KL | 0.02038 (top1 480/511, ppl 5.9183) | 0.01794 (top1 477, ppl 6.0610) |

### C12 model-level checks (c12 build, chD, 16:30-16:31 Dubai)
- 4B IU4P a4 default: KL vs MQ4P decode 0.03453435172173479, top1 938. That is bit-for-bit the same as c11b, and a repeat gives the same value.
- Gate c12a4: unstable 0, conc 0, errors 0. All 6 greedy shas match b4a4 (hash 6dbe5cb43ee4, code 166f783cdca7, math c101d4fb84d0, story d4eed6f4386f, ctx1k 8e3f23e5a295, ctx4k a8398834af83).

### a4all decision evidence
- Gates: c4all (solo), c4allcb (cb4) and c27all all show unstable 0, conc 0, errors 0.
- The c4all shas differ from b4a4, as expected because it uses a different quantizer. The c4all story ran to the 1024-token cap where b4a4 stopped at 924. Its text is coherent, with a max 8-gram repeat of 1.
- 27B kernel-trace totals (bench_prefill 2048 x2, one run each): a4 2587.8 ms vs a4all 2480.1 ms (-4.2%; the quantize_int4_mmq_ds128 130 ms disappears).
- 4B: pp512 +5.2%, pp2048 +4.7% (from C11b).
- KL vs decode is mixed: 4B 0.0345 -> 0.0388, 27B 0.0204 -> 0.0179. a4all is bit-exact to MQ4P prefill on both models (KL 3e-13).
- Decision: a4all becomes the default (C13). `HIPFIRE_IU4_FUSED_X=1` or `a8` keeps the old behaviour.

### Step 3: a8 vs a4 prefill matmul, 4B IU4P, bench_prefill 2048 x4, rocprof kernel-trace (c11b)
| kernel | calls | ms | VGPR | WG |
|---|---|---|---|---|
| gemm_iu4g256_w4a8_full_set (t128) | 2176 | 489.11 | 120 | 512 |
| gemm_iu4g256_w4a4_full_set_lf16_col | 2176 | 331.52 | 96 | 512 |
| MQ4P gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col | 2176 | 343.91 | 96 | 512 |

- CORRECTION: the runtime a4 kernel is gemm_iu4g256_w4a4_* (the mmq_iu4 gfx11 file, v_wmma_i32_16x16x16_iu4), not GEMM_IU4G256_MMQ_SRC (iu8 WMMA). The earlier static-ISA "a4 also uses iu8 WMMA" note was about the wrong kernel. So a4 vs a8 is iu4 WMMA vs iu8 WMMA, plus data movement.
- `HIPFIRE_GFX11_A8_PREFILL=1` on MQ4P did not engage MQ4P's a8 kernel: the trace shows the same mmq_iu4 kernels. a8_prefill_active also requires mq4v2_symmetric, and this MQ4P model apparently is not (untested inference). So there is no runtime comparison against MQ4P's iu8 kernel. The static ISA comparison stands: IU4 t128 has 1062 instructions vs MQ4P's 917, with 32 WMMAs each.
- PMC (rocprofv3 --pmc, 4 counters per pass) is unreliable here. SQ_INSTS_LDS, SQ_INSTS_VALU, SQ_WAVE_CYCLES and SQ_INST_CYCLES_VMEM read 0. The same MQ4P kernel gives SQC_LDS_BANK_CONFLICT 5.49e8 in one pass and 2.74e8 in another, with IDX_ACTIVE 0 vs 8.2e8. Only SQ_WAVES (3554 per dispatch for both) and SQ_BUSY_CYCLES (a8/a4 = 1.39) look plausible. No bank-conflict claim is made.
- Tile 128 vs 256 (c12, x2):
  - 4B: full_set 246.7 vs 324.0 ms, full_add 97.1 vs 173.0 ms; t128 wins, and the default already picks it.
  - 27B: full_set 1752.8 vs 1767.7 ms, full_add 764.4 vs 703.2 ms; total 3491.6 vs 3346.6 ms, but gemm_iu8g32 also moved 482.6 vs 428.4 ms, so clocks/power differed. One run only, inconclusive.
- Global-load pattern (from source):
  - Per K128 per t128 CTA, a8 stages X 128 tok x 128 B = 16 KB plus W 8 KB; the a4 kernel stages X 8 KB plus W 8 KB. That is 1.5x the bytes, close to the 1.475x kernel-time ratio.
  - X rows are block_i8_128 at a 136 B stride with qs at +8. Each thread's b128 covers part of a 64 B token segment that is not 128 B aligned (8 + 136t + 64s), so a good share of the segments straddle two cache lines.
  - HYPOTHESIS, untested: a planar i8 sidecar (qs at 128 B-aligned rows, d/s in a separate plane) would cut the TA/L2 requests per byte. The fused producer owns the layout now (C11b), so this is cheap to try next.
- C13 verification (c13 build, 16:35-16:37 Dubai):
  - Default KL is 0.03875639474581316, the same as the FUSED_X=all value.
  - Gate c13def: all 6 shas match c4all.
  - Gate c13x1 (FUSED_X=1): all 6 shas match b4a4.
  - Both gates: unstable 0, conc 0, errors 0.
- Slots step 5: only the IU8G32 embedding arm has been committed (74b59ff5a).
  - Still missing: IU4G256/IU8G32 in slots_proj_admissible, slots_plain_gemm_key, slots_residual_gemm_key and lm_head_slots_admissible, plus the slots-side producer and pending-x wiring.
  - chZ (slots) does not run on IU4 until that port lands. NOT STARTED.
- qt59 rotation: last item, untouched.

## Chunk 4:38 PM — items (3) 27B C13 gate, (2) per-role a8 (C14)

### (3) 27B gate on C13 (chF, 12:38–12:47 UTC)
- c13_27 default (a4all): hash=57040570c7bf code=c0425548b9e4 math=80e54e58e52e ctx4k=1c373245e97a — identical to c27all (a4all on c11b). unstable 0, conc_mismatch 0, errors 0.
- c13_27a8 (PREFILL_ACT=a8): 15c387bfc1cf / c0425548b9e4 / a21ee78300f1 / a60387026fc6 — identical to b27a8.
- c13_27cb (cb4, 2 rounds): 4aae6a25ebc9 / 689db6c41ce1 / 54c324ce92c9 / 4897a51473a0, stable, conc_mismatch 0 (no prior cb4 ref to compare).
- Note: C14 cargo build ran (nice -19) during the c13_27 gate's 2nd half; gate is hash-only, so not a correctness concern.

### (2) C14 commit 82672f514: per-role/layer a8 override
- Gpu fields iu4_role (u8, MAX=none) and iu4_layer (idx, n_layers); role set at entry of the 6 projection fns
  (full-attn in/out, deltanet in/out, ffn gate_up/down); layer set (+role reset) at the top of both prefill.rs layer loops.
- Gpu::iu4_act_eff() replaces the global act in iu4_fused_x_act (producer) and gemm_iu4g256_multi (consumer), so producer+consumer agree.
- Env: HIPFIRE_IU4_A8_ROLES=aqkv,ao,dqkv,do,gu,dn,all ; HIPFIRE_IU4_A8_LAYERS=firstN,lastN,a-b,a. Only when base act is a4. Both unset = off.
- Sanity: default KL bitwise = C13 (0.03875639474581316); ROLES=all KL = a8 bitwise (4B 3.897451151503087e-4).
- `do` is a no-op: KL bitwise = default on both models (DeltaNet wo in these IU4P models is IU8G32 (C12), not IU4 — inference from C12, consistent with the result).
- KL vs decode ref (mean_kl_top64; 4B 1023 steps / 27B 511 steps, one text each), and pp2048 tok/s (sp.py: plain daemon bench_prefill 2048, 1 warm + 2, interleaved rounds; 4B n=4-10, 27B n=6, medians; spreads <1.5%):

| config | 4B KL | 4B pp2048 | 27B KL | 27B pp2048 |
|---|---|---|---|---|
| default a4all | 0.03876 | ~7610 | 0.01794 | ~1614 |
| aqkv | 0.04005 | – | 0.01749 | – |
| ao | 0.03486 | – | 0.02564 | – |
| dqkv | 0.03329 | ~7250 (−4.7%) | 0.01335 | ~1530 (−5.2%) |
| do | = default (no-op) | – | = default | – |
| gu | 0.02365 | ~6952 (−8.6%) | 0.01091 | ~1389 (−14.0%) |
| dn | 0.03500 | – | 0.01497 | – |
| first4 (all roles) | 0.04071 | – | 0.01644 | – |
| last4 (all roles) | 0.03116 | – | 0.01483 | – |
| gu,dn | 0.01979 | ~6692 (−12.1%) | 0.00630 | ~1295 (−19.8%) |
| gu,dqkv | 0.01908 | – | 0.00663 | – |
| gu,dqkv,dn | 0.00686 | ~6360 (−16.4%) | 0.00255 | ~1237.5 (−23.3%) |
| aqkv,ao,dqkv | 0.02709 | – | 0.01522 | – |
| all (=a8) | 3.897e-4 | ~6328 (−16.8%) | 1.108e-4 | ~1219 (−24.5%) |

- Finding: the a4 KL is NOT concentrated in a few roles. Single switches are non-additive (some raise KL: 4B aqkv/first4, 27B ao), and recovering most of it needs gu+dqkv+dn, which costs ~all of a8's speed. Cheapest useful point: gu only (KL −39% on both, −8.6%/−14% pp2048). Knob kept as C14 (default off); no default change.
- Caveats: one eval text per model; single-role deltas of ~±10% are within what non-additivity/noise could produce (untested with more text). r1 of the 4B chH speed sweep was invalid (sp.py key bug, fixed) and excluded.
- Not started this chunk: (1) aligned int8 layout, (4) slots port, (5) qt59.

## Chunk 5:06 PM — item (1) aligned int8 layout (C15, 8ddece8d7)

Layout `I8_XPLANAR` (opt-in `HIPFIRE_IU4_I8_XPLANAR=1`, IU4 W4A8 only; MQ4 A8 route untouched):
the same N*(K/128)*136 B, per K128 block kb one chunk at kb*N*136 = [N x 128 B qs plane][N x 8 B (d,s) plane].
Every token's qs starts 128 B aligned (legacy layout: 136*T+8). The consumer's fetch_slab uses a qs-plane
descriptor (bound N*128), and fetch_hdr uses a header-plane descriptor (bound N*8, offset 8*T). Twins were added for the
standalone quantizer (quantize_int8_mmq_ds128_xp), the 3 gfx11 i8 producers (*_i8x_gfx11) and the
w4a8 t128/t256 GEMMs (gemm_iu4g256_w4a8x_*).

Bit-exactness (test `iu4_i8_xplanar_gpu`, run with TILE auto/128/256: bad=0 each):
- standalone XP bytes == relayout(legacy bytes): 25 (K,N) cases, K∈{256..17408}, N∈{1,7,64,333,2048}
- producers XP == relayout(legacy producer): 90 cases incl. AWQ rmsnorm, deterministic
- GEMM legacy vs XP: bitdiff 0 on 7 shapes up to 17408x5120x2048, set+add, deterministic
- model: a8 KL JSON (all fields except secs) identical xp0 vs xp1; 4B 3.897451151503087e-4 (== C14 all),
  27B 1.1079767487187929e-4 (== C14 all)

rocprof A/B (PREFILL_ACT=a8, pp2048, NB=2, 3 rounds interleaved, clocks NOT pinned):

| model | w4a8 ms xp0 (r1,r2,r3) | xp1 | Δ | pp tok/s under rocprof xp0 → xp1 |
|---|---|---|---|---|
| 4B (t128) | 337.1 338.0 337.7 (337.6) | 336.3 336.7 337.2 (336.7) | -0.26% | 6306 → 6328 (+0.35%) |
| 27B | 2401.7 2408.5 2409.8 (2406.7) | 2392.7 2396.3 2400.0 (2396.3) | -0.43% | 1223 → 1229 (+0.48%) |

xp1 wins in all 6 pairs, but the gain is under 0.5%. So X misalignment is not what limits the W4A8 GEMM; the 8 B
offset b128 loads cost almost nothing here. The default stays off: the default prefill is a4, so this only matters for the a8 path or role overrides.
27B gate with a8 + XPLANAR=1 (hash,code,math,ctx4k x2): 15c387bfc1cf / c0425548b9e4 / a21ee78300f1 / a60387026fc6 == c13_27a8; unstable 0, conc_mismatch 0, errors 0.
Not measured: an ISA diff of the xp kernel (llvm-objdump), a 4B gate on the xp path, and clock-pinned timing.

## Chunk 5:30 PM — C16 slots port + batched decode IU4P vs MQ4P (13:30-13:45 UTC = 17:30-17:45 Dubai)
Steer: int8 work deprioritised. a8 GEMM-gap investigation and WMMA iu8 microbench DROPPED (wm.hip compiled+ISA-checked, never run).
Future work (untested hypotheses): a8 gap = weight-unpack ALU + 2x X LDS bytes, not WMMA rate (static ISA: a4 lf16_col 64 iu4 WMMA/955 instr/96 VGPR; a8x 32 iu8 WMMA/1072 instr/118 VGPR, not per-tile normalised). qt59 (IU8G32) int8 rotation: not started.

C16 (89d74ab2f): forward_slots admits IU4G256 (proj + rotated), IU8G32 lm_head + residual (gfx1100/1151, k%32==0); Uniform IU4 groups (qkvza x2, qkv x2, gate_up) + residual/w_down via gemm_iu4g256_multi on x_rot. Builds clean.
**C16 is compiled but NOT exercised**: the only daemon consumer (serve.vmm_batch, HIPFIRE_SERVE_VMM_BATCH=1) refuses Qwen3.5-4B for BOTH formats:
"VMM continuous batch unsupported for this model: exact VMM route: layer 0 is not a dense uniform-MQ4G256V2 layer". Legacy SlotEngine (serve.multi_slot) not tried. vmm_iu4 c15 vs c16 hashes identical (aeff3a104d...) = singleton route, nothing changed. saddle-lab test_forward_slots_golden on IU4P not run -> C16 numerics/coherence UNTESTED.

Batched decode that actually runs today = default `--continuous-batch-size 8` route (qwen35::forward_decode_batch), already IU4-capable. ct.py, 8 prompts, 128 tok, T=0, 3 rounds alternating c-order, median agg tok/s:
| route / model (4B) | c1 | c4 | c8 |
|---|---|---|---|
| singleton (vmm_batch=1 falls back) IU4P | 186.1 | 186.4 | 186.6 |
| singleton MQ4P | 187.7 | 188.1 | 188.2 |
| cb IU4P a4 (default) | 28.5 | 110.9 | 217.7 |
| cb IU4P a8 (HIPFIRE_IU4_PREFILL_ACT=a8) | 35.7 | 138.8 | 272.3 |
| cb IU4P HIPFIRE_IU4_PREFILL=0 | 28.5 | 110.9 | 217.7 |
| cb MQ4P | 48.3 | 188.3 | 372.4 |
- cb step time is ~flat in lanes for both models (IU4 a4 ~36 ms/step, a8 ~29, MQ4P ~21 ms/step) vs ~5.4 ms singleton: cb route has a large fixed per-step cost (both formats; untested hypothesis: un-graphed launch overhead / fixed lane padding). IU4 adds ~15 ms/step over MQ4P (a4). Not investigated (time).
- IU4P cb c8 = 0.58x MQ4P cb c8; cb beats singleton only at c8 for IU4P (217.7 vs 186.6); MQ4P ties at c4 and wins at c8 (372.4 vs 188.2).
- Determinism: all hashes identical across 3 rounds and across c1/c4/c8 for the same prompt (both routes, all configs).
- Byte-identity to singleton: MQ4P cb == MQ4P singleton (0b3f35bf4a...). IU4P cb a4 (5cacd738a8) != IU4P singleton (aeff3a104d): batched decode uses the a4 W4A4 GEMM (quantised activations) instead of the f32 decode GEMV. HIPFIRE_IU4_PREFILL=0 gives the same hash as a4 -> that knob does not reach forward_decode_batch (untested why). Open gap: IU4P batched decode is not the f32-decode numerics; KL of the cb path vs MQ4P not measured.
- fl_iu4_c16 (C16 daemon, cb route) == c15 a4 numbers and hashes (28.5/110.9/217.7, 5cacd738a8...): C16 is a no-op on the default route, as expected.
- Gate c16_vmm4b (HIPFIRE_SERVE_VMM_BATCH=1, --continuous-batch-size 4, keys hash/code/math/ctx4k, 2 rounds): unstable 0, conc_mismatch 0, errors 0, klog 0. NB this exercised the singleton fallback, not forward_slots.

## C17 — 27B gfx1100 prefill/TTFT/TG pass (2026-11-02, feat/iu4-iu8-quant-family)

Host/container: RX 7900 XTX (gfx1100), ROCm 10.1 / HIP 7.16 / clang 24, model
`qwen3.8-27b.iu4p` (qt57 body, qt59 embed/lm_head/LA out_proj, 64 layers, 16
full-attention, 5120 hidden, 248320 vocab). All numbers are fresh-process,
`--kv-mode q8`, `--spec off` unless stated.

### C17a — IU8G32 prefill GEMM fanout (mb4, mb4m2)

`gemm_iu8g32_residual_wmma` repeated its weight dequantization and global
re-read once per 16-token tile (128x at N=2048). Two bit-exact fanouts:

- `gemm_iu8g32_residual_wmma_mb4`: one 16-row wave covers four 16-token batch
  tiles (64 tokens), one dequant/read per tile.
- `gemm_iu8g32_residual_wmma_mb4m2`: two 16-row M tiles share the same four X
  fragments (32 rows x 64 tokens per wave), halving X re-reads. 184 VGPR,
  0 spills.

Both are bit-identical to the base kernel on the twin test
(`iu8g32_wmma_gemm_gpu`: 112 cases, `bitdiff_vs_q8_0_wmma=0`, worst rel vs f64
1.478e-4; n=77/512 exercise the fanouts). Per-pass kernel time at pp2048:
base 214 ms -> mb4 193 ms -> mb4m2 180 ms.

`hipfire bench --matrix --pp 512,2048 --ctx 128,4096 --tg 64`, 2 rounds
interleaved (tok/s medians):

| build | pp512 | pp2048 | tg@128 | tg@4096 |
|---|---|---|---|---|
| branch tip (base, 512-row prefill) | 1628 | 1626 | 46.3 | 44.7 |
| + mb4 | 1699 | 1697 | 46.3 | 44.8 |
| + mb4 + mb4m2 | 1822 | 1822 | 45.2 | — |

`gemm_iu8g32_residual_wmma_mb4m2` is default for M>=32, N>=64; `HIPFIRE_IU8G32_M2=0`
and `HIPFIRE_IU8G32_MB4=0` are A/B controls.

### C17b — IU4 dense widened prefill (opt-in `HIPFIRE_IU4_WIDE_PREFILL=1`)

`ordinary_prefill_static_ceiling` admitted only uniform MQ4G256V2 / native
MQ4G256 dense models to the widened chunks; IU4 dense stayed on the 512-row
legacy cadence. The IU4 family (all projections IU4G256, or IU8G32 in the LA
out role) is now admitted when the env opt-in is set. The route keeps its
512-row commit seams, so it is the same per-row arithmetic grouped into larger
chunks.

`hipfire bench --matrix --pp 2048,8192 --ctx 128 --tg 16`, 2 interleaved
rounds (pp tok/s): legacy 1690/1686 and 1629/1626; wide 1774/1772 and
1747/1742. (+5.0% pp2048, +7.2% pp8192.) Decode is unchanged (45.2 vs 45.3).

Serve-path cold TTFT (`HIPFIRE_QWEN_PROMPT_CACHE=0`, 200k ctx, 3 runs,
medians): 8140-token prompt 5.170 s wide vs 5.173 s legacy (tie); 21580-token
prompt 14.021 s wide vs 14.526 s legacy (-3.5%). Greedy output on the 8140-token
prompt is byte-identical between the two routes (reasoning text, 96 tokens).
The container enables the opt-in; it stays opt-in in code until more artifacts
are pinned.

### C17c — DFlash on qt59 (IU8G32) targets

DFlash was refused for qt59 (`batched lm_head admission ... unsupported`).
The IQ8G32 lm_head is the qt59 relayout of Q8_0, and C17a gives a batched
gfx11 WMMA GEMM for it, so:

- loader admission admits qt59 on exact gfx1100/gfx1151
  (`dflash_lm_head_quant_supported`), unit-tested for both accept and
  gfx1200/gfx1201/gfx1030/gfx942 refusal;
- `dflash_batched_lm_head_supported` includes IU8G32; the verify head and the
  draft head zero the logits slice and run `gemm_iu8g32_residual_wmma`
  (natural-basis X, no rotation);
- the DFlash noise-embedding paths (planned and scalar) read the qt59
  embedding via `embedding_lookup_iu8g32`;
- the duplicated head-dtype allowlists in `draft_dflash_block_rank` and the
  sampled draft path now call the shared predicate.

Measured with the registry 27B MQ4 DFlash draft (md5
`013395583cd04206c8aa68f4d061983d`) on the IU4P trunk, serve 200k ctx, greedy,
128-token decode: 1129-token prompt 44.6 tok/s (τ 4.29) vs AR 44.9; 8140-token
prose 47.9 vs AR 43.6 (+10%); 21580-token code 29.2 vs AR 40.2 (-27%). The
long-context loss is the 16-row verify re-reading the KV span per cycle; it is
inherent to this draft/verify shape, not a kernel gap. DFlash therefore stays
**off** in the container default; it is an opt-in (`speculation.dflash=on` plus
`developer.dflash_draft`) for short-context code turns.

### Final container numbers (serve, 200k ctx, q8, as deployed)

`/v1/chat/completions`, greedy, no prompt cache, medians of 3:

| prompt tokens | TTFT | prefill tok/s | decode tok/s (128 generated) |
|---|---|---|---|
| 1129 | 0.44 s (warm cache) / 0.83 s cold | ~1390-1500 | 44.9 |
| 8140 | 5.17 s | 1574 | 43.6 |
| 21580 | 14.02 s | 1539 | 40.2 |

Decode at ctx128 is 45.2 tok/s under `hipfire bench` (46.3 before C17; within
the ±3% session spread). Remaining prefill split at pp8192 (rocprof, wide):
w4a4 set/add 61%, IU8 wo 16%, GDN 8%, attention 5%, producers 7%.

## C18 — accelerator audit and gap-closure attempts (2026-11-02)

Read `rdna3-isa-condensed.md` against the shipped kernels, unbundled the cached
gfx1100 code objects (`clang-offload-bundler` + `llvm-objdump -d --mcpu=gfx1100`)
and used the radiowave inspection metadata for VGPR/spill/instruction counts.
Method: fresh-process `hipfire bench --matrix --spec off`, rocprofv3 kernel
traces, FILE header FLOP accounting.

### What the hardware can offer, and what we use

- RDNA3 WMMA is 16x16x16 for F16/BF16/IU8/IU4; IU4/IU8 issue the same 8192
  FLOP/instr as F16, so their peak is the model's 122.9 TFLOPS (7900 XTX).
- 27B IU4P = 26.899 B params, 25.622 B matmul params (qt57 22.841 B, qt59
  4.053 B, qt1/qt3 0.006 B). 2048-token prefill = 104.9 TFLOP.
- IU4 W4A4 tile: 64 WMMA, 95 VOPD, 1,131 static instructions, 96 VGPR, 0
  spills, LDS 30,720 B/CTA (2 CTAs/CU of 64 KiB). 80.8 TFLOPS on the qt57
  matmul = **66% of F16 WMMA peak**. MQ4V2 twin: 64 WMMA, 94 VOPD, 1,151
  static, same 96 VGPR = 75.4 TFLOPS (61%).
- qt59 wo GEMM: F16 WMMA fed by int8->f16 dequant; mb4m2 body 16 WMMA, 182
  VGPR, 0 spills. 15.3 TFLOPS = **12% of peak**. `V_WMMA_I32_16X16X16_IU8`
  is not used in production; per-32 scales would need int32 accumulators
  scaled per K32, which trades the weight dequant for an equal count of
  accumulator converts, and the existing opt-in W4A8 route measured slower.
- Decode: f32 FMAC/FMA (VOPD present but limited by pairing rules), 14.82 GB
  of weights per token / 23.03 ms = **643 GB/s = 67% of 960 GB/s** (76% of a
  realistic ~850 GB/s). `fused_iu4g256_rows` is the best stream (~874 GB/s);
  qt59 wo ~623 GB/s.
- No hot kernel sets `SLC`/`DLC`; every load is default LRU.

### IU4 vs MQ4 (why speeds match)

Same quant math and bpw; layout differs. qt44 group = 8 B header + 128 B codes
interleaved (136 B/256); qt57 = 128 B code plane + 8 B header plane per group,
so code reads are 128 B-aligned. qt59 = `[K int8 codes][K/32 f16 scales]`
(planar Q8_0 relayout). The twin proof is byte-exact; the prefill kernels are
the same source under `HIPFIRE_MMQ_IU4L` and differ by 20 static instructions
of addressing. Identical-settings bench:

| model | pp512 | pp2048 | pp8192 | tg@128 | tg@4096 |
|---|---|---|---|---|---|
| IU4P | 1,672 | 1,669 | 1,607 | 45.21 | 44.70 |
| MQ4P | 1,580 | 1,574 | 1,519 | 45.69 | 45.22 |

### Attempts that lost (kept out of the tree)

- qt59 mb8 fanout (16 rows x 128 tokens): pp2048 1,560 vs 1,669 (-6.5%);
  X traffic doubles and the 157-VGPR body gained nothing.
- qt59 mb8m2 fanout (32 rows x 128 tokens): pp2048 1,687 vs 1,665 (+1.3%)
  and pp8192 1,627 vs 1,607, but it needs 256 VGPR and spills 8 registers;
  not worth defaulting.
- Non-temporal (`__builtin_nontemporal_load` -> `global_load_b128 ...
  slc dlc`) weight loads in the qt59 decode GEMV: 46.38/46.27 tok/s vs
  46.38/46.49, i.e. noise. Removed.
- DFlash block 4 / 8 at 21,580-token context (IU4P, registry draft): 23.4 /
  26.7 tok/s, tau 1.7 / 2.1, vs AR 40.3. Block 16 was 29.2. Long-context
  verification is KV-traffic bound, not block-size bound.

### Remaining ceilings

- qt59 GEMM is format-bound at this tile level (12% peak). The next real
  step is an MMQ-style 128x128 LDS-staged Q8 GEMM that reuses dequantized
  weights across 128 tokens; the fanout sweeps above show the current
  register-only geometry is within ~1% of its tradeoff frontier.
- W4A4 sits at 66% because its 128-K `block_i4_128` tile fixes a 30,720 B
  LDS footprint (2 CTAs/CU). Beating it needs a new activation layout
  (K64), not a dispatch knob.
- GDN (7.8% of prefill) has no chunk-scan route on gfx1100; the WMMA chunk
  scan is admitted on gfx1200/gfx1201 only.
- Decode is at the HBM wall for this weight set; the only large lever left
  is speculation, whose long-context arm is measured dead.

## C19 — design-3 prototype: int8-per-128 + LDS + WMMA-IU8 (2026-11-02)

The W4A8 LDS MMQ (`gemm_mq4g256v2_residual_mmq_i8`) run on the 27B LA
out-proj shape (M=5120, K=6144, N=512) is the int-core upper bound:
**0.507 ms / 63.6 TFLOPS**. The shipped qt59 f16-register kernel on the same
shape is **1.320 ms / 24.4 TFLOPS**. So the int matrix core plus 128x128 LDS
reuse has ~2.6x of headroom before any format change; the work is to give it
int8 weights with a scale cadence it can accumulate across.

Prototype (`kernels/src/gemm_iu8g32_w8a8_min.hip` +
`Gpu::gemm_iu8g32_w8a8_min`):

- Weight format: `[K signed int8 in block_i8_128 qs order][K/128 f16 scale]`
  per row (8.125 bpw, zero = 0). For this prototype the 128 codes are the
  nibble-pair permutation of `block_i8_128`, so the weight side pairs with the
  existing activation plane without a K-loop shuffle.
- Kernel: 128x128 tile, 512 threads / 16 waves, X and W staged in LDS
  (16 KiB each), one WMMA-IU8 accumulation over the whole K128, one
  `sc_w * d_x` fold. No double buffer/raster yet.
- Correctness: a synthetic m=64 k=512 n=32 case matches a CPU emulation of
  the same fold **bit for bit** (worst rel 0.0e0).
- Speed at the real wo shape: **0.847 ms / 38.0 TFLOPS** vs 1.320 ms / 24.4
  for the shipped f16 path, **1.56x**, with no pipeline/raster.

Projected model effect: the qt59 prefill bucket is 338 ms/pass at the model's
N=512 chunks (192 calls); 0.847 ms/call gives ~163 ms, so pp2048 1,822 ->
~2,010 tok/s (+10%) before the production kernel's double buffering. The
remaining gap to the 0.507 ms W4A8 bound is the int8 weight plane (2x bytes)
and the prototype's lack of a slot pipeline.

Not yet wired (no route change): the loader still needs the qt59 -> int8-per-128
requantization (per-32 scales folded to one scale per 128, lossy), a decode
GEMV for the same format (f32 x, one scale per 128), and the claim-scoped KL
gate. Activation cost for prefill is the existing W8A8 KL (~1.0e-4 on 27B
decode); the weight requant is the new quality term.

## C20 — long-context decode: GQA-shared Q8 tile + parallel reducer max (2026-11-02, feat/iu4-iu8-quant-family)

Goal: make growing context cost decode (tg) and prefill (pp) as little as
possible on the 27B IU4P container. This section covers the decode arm; the
prefill arm is measured and bounded, not yet moved (see "Remaining ceilings").

Method. Fresh-process `bench_decode` / `bench_prefill` against one resident
`target/release/daemon` (`HIPFIRE_GRAPH=0`, `HIPFIRE_AR_GRAPH=0`,
`HIPFIRE_CASK_OFF=1`, `HIPFIRE_IU4_WIDE_PREFILL=1`), 12 steps per round, 2
rounds; rocprofv3 `--kernel-trace --stats` on the same driver; and the
container serve (`/v1/chat/completions`, greedy, `reasoning_effort=none`, no
prompt cache, unique prefix per request).

### Attribution (rocprofv3, decode at ctx 131072, 8 steps)

Per step, before any change: `attention_flash_q8_0_tile` + its reduce = 11.8 ms
of a 36.9 ms step. The tile kernel ran one 32-thread workgroup per **q head**
(grid `[n_heads, max_tiles, 1]`), so each K/V tile was read six times — once
per q head sharing a kv head — and the reducer found the global max with a
single thread walking `n_tiles` records strided 1,032 B (25.4 MiB per layer at
tile 128; 24 workgroups = 24 CUs).

Every other decode kernel is context-independent: the weight stream
(14.82 GB/step: `gemm_iu4g256_w4a4_*`, `gemm_iu8g32_residual_wmma_mb4m2`,
`gemv_*`), GDN (`gated_delta_net_q8_fast`), the FA prep/norm/rope chain and the
q8 KV row write.

### Change

- The **ungated** gfx1100 decode route (`output_gate = None`) is the one a qt59
  `IU8G32` `wo` takes: the fused gate epilogue is admitted only for MQ4 `wo`, so
  the model's `attn_output_gate` is applied outside attention. That route now
  runs the existing GQA-shared `attention_flash_q8_0_tile_gqa_gfx1100` (one
  256-thread workgroup per (kv head, tile) serves all six q heads; byte-identical
  partials) with the plain shared reducer; the gated non-AWQ route uses the same
  tile with the gated MQ-rotate reducer. `HIPFIRE_GFX1100_DECODE_ATTN_GQA=0`
  restores the per-head tile.
- Both Q8 reducers (`attention_flash_q8_0_reduce`,
  `attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100`) now scan the global max
  across the block's tile stripes and reduce through LDS. `fmaxf` is exact and
  order-independent, so the published maximum is bit-identical to the
  single-thread increasing-tile scan.

Output identity: the same 12,795-token prompt, greedy, emits the same 14 tokens
with the GQA route on and off (`ab-gqa` container on :9002 vs the product
container).

### bench_decode (tok/s, medians of 2 rounds × 12 steps)

| ctx | before | after | delta |
|---|---|---|---|
| 128 | 43.6 | 44.7 | +2.5% |
| 8,192 | 42.2 | 43.2 | +2.4% |
| 16,384 | 40.4 | 42.1 | +4.2% |
| 32,768 | 36.8 | 40.5 | +10.1% |
| 65,536 | 31.6 | 37.8 | +19.6% |
| 131,072 | 24.7 | 32.4 | +31.2% |

The KV read is now one pass per kv head per step: 4.56 GiB/step at 131k. The
GQA tile streams it at ~660 GB/s (6.9 ms/step); the reducer is ~0.5 ms. A step
at 131k is 30.5 ms against 22.3 ms at ctx 128, i.e. within ~5% of reading
weights + KV once at the observed weight-stream bandwidth (~664 GB/s). Further
decode gains need faster KV streaming or a smaller KV format, not a route
change.

### Container serve (200k ctx, q8, greedy)

TTFT, prefill and decode (medians of 2, unique prefix, no prompt cache,
128 generated tokens so the decode rate covers 127 intervals):

| prompt tokens | TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|
| 951 | 0.75 s | 1,255 | 45.9 |
| 6,534 | 3.92 s | 1,667 | 45.0 |
| 17,240 | 10.84 s | 1,591 | 43.8 |
| 26,150 | 18.19 s | 1,438 | 42.8 |
| 52,248 | 41.88 s | 1,248 | 40.2 |
| 104,446 | 104.86 s | 996 | 35.4 |

C17 (previous container build) measured 44.9 tok/s at 1,129 prompt tokens, 43.6
at 8,140 and 40.2 at 21,580. Decode at 104k is 35.4 tok/s against a
weights+KV-once floor of ~36.0 tok/s at the 664 GB/s weight stream: the KV read
itself is what remains.

### Prefill attribution and remaining ceiling

pp falls from ~1.67k tok/s at 6.5k to ~1.0k at 104k. rocprofv3 on pp32768
(fresh, 4,096-row chunks) attributes 15.6% to `attention_q8_0_fa2_gqa_gfx1100`
(3.19 s of 20.56 s), 67% to the IU4/IU8 projection GEMMs, 7.5% to GDN. The
attention term is the only quadratic one, so it is ~32% of a 104k prefill.

The FA2 body is at 54% of F16 WMMA peak (66 TFLOPS) and register-bound:
255 VGPR / wave32 with 0 spills (`radiowave` inspection), i.e. 2 waves/SIMD =
**one** workgroup per CU regardless of the 32 KiB LDS, so the 2-barrier
fill/compute cadence cannot be hidden by a second resident workgroup. Its memory
demand at peak compute (675 GB/s) is under HBM, so this is not a KV-traffic
limit — a bigger q tile (Q32) would not help. `HIPFIRE_FA2_FILL=0` (all-wave
fill) measured 1,139/1,229 tok/s against 1,312/1,350 with the fill path at a
41.7k prompt, so the warp-specialized fill stays. Moving pp further needs an
FA2 body under 128 VGPR (2 workgroups/CU) or a KT64 rebuild now that occupancy
is register- rather than LDS-bound; both are kernel redesigns, not knobs.

## C20 — design 3 routed: W8A8 int8-per-128 on the qt59 roles (2026-11-03)

The qt59 weight rows are now requantized **in place** at load when
`HIPFIRE_IU8_W8A8=1`:

- one scale per 128: `sc = max|w| / 127`, codes `round(w/sc)`, written
  linearly; the four per-32 scale slots of each K128 block receive the same
  f16 `sc`. Row size is unchanged (K + K/16), so **every existing qt59 reader
  stays correct** (decode GEMV and the f16 prefill see four equal per-32
  scales) and the mark is per weight buffer.
- prefill dispatch: the existing `IU8G32` arm checks the mark, quantizes the
  unrotated input with the new linear-order K128 quantizer
  (`quantize_int8_mmq_linear128`), and runs `gemm_iu8g32_w8a8_min` (128x128
  LDS, WMMA-IU8, one scale fold per K128). Decode and every other route are
  untouched.

Speed (`hipfire bench --matrix --spec off`, 2 rounds, fresh process):

| prefill route | pp2048 base | pp2048 W8A8 | pp8192 base | pp8192 W8A8 |
|---|---:|---:|---:|---:|
| 512-row legacy | 1,665 | **1,910 (+14.7%)** | 1,607 | **1,835 (+14.2%)** |
| wide (container) | 1,821 | **2,022 (+11.0%)** | 1,817 | **1,993 (+9.7%)** |

Decode is unchanged (46.4-46.5 tok/s at ctx128). The synthetic kernel is
bit-exact against a CPU fold (worst rel 0.0) at m=4/128 and the real
M=5120/K=6144 shape.

Quality (`iu4_kld`, 511 teacher-forced tokens, 27B, same build):

| comparison | top1 | mean KL top64 | PPL |
|---|---:|---:|---:|
| W8A8 off vs itself (sanity) | 511/511 bit-identical | 2e-18 | 6.0168 |
| weight requant, decode path vs qt59 | 509/511 | 3.09e-5 | 5.8585 vs 5.8634 |
| W8A8 prefill vs f16 prefill | 469/511 | 2.90e-2 | 5.8552 vs 6.0168 |
| W8A8 prefill vs decode ref | 485/511 | 1.85e-2 | 5.8552 |
| f16 prefill vs decode ref (baseline) | 473/511 | 1.72e-2 | 6.0168 |

The weight requant is negligible on the decode path. The prefill delta is
dominated by the W8A8 activation quantizer; it is slightly above the f16
prefill KL (0.029 vs the body's existing 0.017 vs decode) while top-1 and PPL
improve. Enabled per-container; the code gate is `HIPFIRE_IU8_W8A8=1`.

## C9 compact qt59 (2026-11-03): one scale per 128, no in-place requant

- Change: qt59 row is `[K signed i8][K/128 f16]` (260 B at K=256) instead of
  Q8F16's `[K codes][K/32 scales]` (272 B). `d = amax/127` is computed per 128
  codes at quantization time; the row size is the kernel's stride.
- Removed the load-time shrink (`requant_iu8g32_per128`) and the
  `HIPFIRE_IU8_W8A8` request. Every reader (`gemv_iu8g32` v1/v2/a8,
  `embedding_iu8g32{,_batched}`, `gemm_iu8g32_residual_wmma` + mb4/mb4m2,
  `gemm_iu8g32_w8a8_min`) reads the compact row; W8A8 is the unconditional
  prefill path. Old files are refused (qt59 blob length), not misread.
- `convert_iu8g128` performs the offline per32 -> per128 migration.
- Model (27B IU4-Pro, gfx1100, q8 KV): file 16,274,209,792 B; pp2048 2,030,
  pp8192 1,991 tok/s, tg16 46.7 tok/s (3-run medians). Decode vs the old per32 ref: mean KLD
  6.0e-5 (top-1 508/511); prefill (W8A8) 0.0160 (top-1 483/511).
- Contract tests updated: `iu8g32_wmma_gemm_gpu` (f64 reference, 100 cases,
  worst rel 9.9e-5 incl. real lm_head/wo) and `iu8g32_v2_gemv_gpu` (v2/a8 vs
  the compact contract).

## C21 — multi-slot decode: one weight pass for every live slot (2026-11-03)

**Problem.** The paged 4-slot decode ran the IU4 projections through
`gemm_iu4g256_multi`. Under retained PM4 recording that route is not
MMQ-eligible (`replay.is_recording()`), so it fell to
`gemm_iu4g256_pertoken`: one f32 GEMV launch per projection per live slot.
Every decode step re-read the whole IU4 weight set once per slot. The engine
is weight-bandwidth bound, and the numbers said exactly that: four lanes at
61.8-65.5 tok/s aggregate = 14.82 GB x 64/s ≈ 950 GB/s, i.e. the HBM wall
for four independent passes.

**Change.** `gemv_iu4g256_xbatch` / `gemv_iu4g256_xbatch_residual`
(`kernels/src/gemv_iu4g256_xbatch.hip`), the IU4 twin of the MQ4V2 xbatch
GEMV: each weight row is decoded once per launch and dotted with up to four
activation columns. `slots_iu4_multi` routes decode-width rows
(`2..=FEW_ROW_ROUTE_ROWS`) through it, so qkvza / qkv / gate+up / w_down / wo
each become one launch per step; `n = 1` and prefill keep the incumbent
route. `gemv_iu4g256_xbatch` is bit-identical to `gemv_iu4g256` on every
tested shape (random tails and the model's 2560/5120/6144/9216/17408-K
projections). The residual variant follows the same expression order but is
compiled in a different module, so the FMA contraction differs: measured
<= 1.3e-7 relative against the per-token residual GEMV (same order as the
accepted IU8 v2 reorder), and `iu4_xbatch_gemv_gpu` gates it at < 1e-6 plus
determinism.

**Measured** (gfx1100, Qwen3.8-27B IU4P, 4-slot paged VMM, q8 KV, retained
PM4, prefill chunk 1024, 737-token prompts, 128 greedy tokens per stream,
fresh process):

| concurrent streams | decode tok/s before | after |
|---|---:|---:|
| 1 | 40.5 | 41.9-42.2 |
| 2 | 55.0-55.1 | 49.0-69.9 (step-composition noise) |
| 4 | 61.8 / 65.5 | **87.9 / 88.8** |

Prefill/aggregate PP and TTFT are unchanged (decode-only change).

**Remaining aggregate ceiling.** With the projections batched, the step is no
longer purely HBM bound. Per-pass microbench (same kernel, model shapes):
B=1 reaches 656-784 GB/s, B=4 only 336-477 GB/s, and B=4 buys 1.25x (M=5120
K=5120) to 1.98x (lm_head shape) over four separate passes. The x columns are
re-read per output row; amortizing them (R output rows per block) needs the
quad-unrolled accumulator layout, not the naive R-loop prototype (which lost
to B=1 everywhere). The other per-slot costs left: the qt59 lm_head runs
`per_slot_head` (4 x 0.64 GB per step; an IU8G32 xbatch twin is the next
~5-15%) and the qt59 `wo` runs the M=4 WMMA residual GEMM.

### Paged-slot prefill/TTFT ceiling (measured, not changed)

Same build, single stream, unique prompt per run:

| prompt tokens | singleton PP tok/s | paged 4-slot PP tok/s |
|---:|---:|---:|
| 2064 | 1463 | 1490 |
| 4107 | 1917 | 1480 |
| 8193 | 1936 | 1360 |
| 16362 | 1776 | 1108 |

Fit `t = a*n + b*n^2`: the paged route's quadratic term is 3.6x the
singleton's (20.4 vs 5.7 ns/token^2) with a ~15% higher linear term. The
quadratic term is `attention_q8_0_flash_prefill_wmma_slots_vmm_paged`; the
singleton runs `attention_q8_0_fa2_gqa_gfx1100`. `HIPFIRE_FLASH_PREFILL=0`
(fallback) drops paged PP 1357 -> 748 at 8193, so the slots WMMA kernel is
the fast available slots path; closing the rest needs an FA2-style
descriptor-addressed prefill kernel. `multi_slot_prefill_chunk` 1024 -> 4096
bought only +7% PP at 8193 (1263 -> 1357, TTFT 6.49 -> 6.04 s) while costing
38% of the KV pool (1380 -> 855 pages, scratch +2.3 GiB), so the container
stays at 1024.

### KV pool vs the advertised per-slot context (observed)

`--max-seq 262144` is not physically available on the 24 GB card: the auto
pool is 1380 pages x 128 = 176,640 tokens for all slots together (one slot at
262144 alone needs 2048 pages). A ~180k-token request is refused with a
retryable `page demand exceeds pool: need 1409 pages`, and on the deployed
container the daemon was observed spinning at ~99% GPU/CPU afterwards with
later (small) requests unanswered until the container was restarted. The
honest ceilings are ~176k tokens for one request and ~44k each for four.

## C22 — IU4/IU8 accelerator utilization on the multi-slot engine (2026-11-03)

Audit of the RDNA3 IU4/IU8 units on the paged 4-slot route, and the one
place they were being left idle.

### What runs where

| stage | route | unit | state |
|---|---|---|---|
| prefill, qt57 body | `gemm_iu4g256_multi` -> `gemm_iu4g256_w4a4_*` | `v_wmma_i32_16x16x16_iu4` | used, 66% of F16 peak (C18) |
| prefill, qt57 A8 opt-in | `gemm_iu4g256_w4a8_*` | `v_wmma_i32_16x16x16_iu8` | opt-in, slower (C14/C15) |
| prefill, qt59 wo | was `gemm_iu8g32_residual_wmma` | F16 WMMA + int8->f16 dequant | **idle IU8 core** |
| prefill, qt59 wo (now) | `gemm_iu8g32_w8a8_min` (128x128 LDS) | `v_wmma_i32_16x16x16_iu8` | used |
| decode, qt57 | `gemv_iu4g256_xbatch` (new) / `gemv_iu4g256` at n=1 | f32 FMAC | by design; see below |
| decode, qt59 lm_head/wo | `gemv_iu8g32_v2_r*` | f32 FMAC | by design |
| decode, integer opt-ins | `HIPFIRE_IU4_DECODE=dot4/dot8`, `HIPFIRE_IU8_GEMV=dot4` | `v_dot4_i32_iu8` / `v_dot8_i32_iu4` | off, measured slower/bad |

Decode integer dots are not a lever here: the step is weight-bandwidth bound
(one pass per token, or one pass per 4 tokens with the xbatch). C7/C8 measured
W4A8 -22..-24% (loses the fused rows keys and adds an x quantize pass) and
W4A4 dot8 costs +1.5% (27B) to +4.3% (4B) PPL. IU8's a8 dot was within noise
on 4B. The xbatch (`C21`) is the correct decode utilization: one weight pass
for every live slot, f32 math.

### The gap fixed here: qt59 prefill never reached the IU8 core

C16 admitted IU8G32 residuals to the slots body with the f16 twin before C9
made the qt59 row compact (`[K i8][K/128 f16]`) and the singleton batched body
unconditionally W8A8. The multi-slot body kept calling
`gemm_iu8g32_residual_wmma` (f16 WMMA fed by int8->f16 dequant: 20 TFLOPS at
the wo shape, ~12% of peak), so the IU8 matrix core never ran on this engine.

`slots_residual_proj` now quantizes the activation once
(`ensure_int8_mmq_linear_x`) and calls `gemm_iu8g32_w8a8_min(..., add=true)`
for prompt chunks of at least one 128-row tile with `K % 256 == 0`; sub-tile
chunks and decode rows keep the f16 twin. `HIPFIRE_SLOTS_IU8_W8A8=0` restores
the twin for A/B.

Kernel microbench (wo shape M=5120, K=6144, weights 2.78 GB):

| rows | W8A8 | f16 twin | speedup |
|---:|---:|---:|---:|
| 128 | 0.430 ms (18.7 TF) | 0.477 ms (16.9 TF) | 1.11x |
| 1024 | 1.824 ms (35.3 TF) | 3.208 ms (20.1 TF) | 1.76x |
| 4096 | 3.422 ms (75.3 TF) | 12.633 ms (20.4 TF) | 3.69x |

End-to-end, host multi-slot serve, one lane, two fresh-process interleaved
rounds per arm:

| prompt | f16 twin PP / TTFT | W8A8 PP / TTFT | delta |
|---|---:|---:|---:|
| 4107 | 1391 tok/s / 2.95 s | 1607 / 2.56 s | +15.5% / -13% |
| 8193 | 1259 tok/s / 6.51 s | 1435 / 5.71 s | +14.0% / -12% |

4-lane xbatch decode (C21) is untouched; n=1 decode unchanged.

### Remaining accelerator / prefill ceiling

- qt59 **decode** (lm_head 4 x 0.64 GB, wo) still runs f32 GEMVs per slot; the
  lm_head is the last per-slot weight pass (~5-15% of the step). An IU8G32
  xbatch twin would close it.
- qt59 prefill **plain** roles (if a mixed tier ever puts one outside the
  residual role) have no plain W8A8 entry; only residuals exist today.
- The paged prefill attention remains `attention_q8_0_flash_prefill_wmma_slots`
  (~3.6x the singleton FA2's quadratic coefficient, C21); that is the remaining
  long-context TTFT/PP gap and needs an FA2-style descriptor-addressed kernel.

## C23 — qt59 (IU8G32) row-shape audit, 32 -> 128 everywhere (2026-11-03)

The qt59 format is **one f16 scale per 128**, not per 32 and not per 256:
`[K signed i8 codes][K/128 f16 scales]`, row `K + K/64` bytes (260 B at
K=256, 8.125 bpw). `iu_quant` calls it `IU8G128` (`Q8_SCALE_GROUP = 128`);
the `IU8G32` DType name is historical. `convert_iu8g128` migrates the legacy
per-32 rows.

Audited every writer/reader/guard:

| site | expected | found |
|---|---|---|
| loader `validate_q8` / `q8_tensor_bytes` | M*(K + K/64) | correct |
| `DType::IU8G32::row_bytes` | K + K/64 | correct (unit-tested: 130 @128, 260 @256, None @32) |
| quantizer `emit_q8_or_iu8` / `convert_iu8g128` | per-128 | correct |
| `gemv_iu8g32` v1/v2/a8 | scale `bi>>2` / `c>>3` | correct; **guard was K%32** -> fixed to K%128 |
| `embedding_iu8g32{,_batched}` | `dim/128` scales | correct; **no dim guard** -> K%128 added; comments fixed |
| `gemm_q8_0_residual_wmma` (IU8G32L), mb4, mb4m2 | stride K + K/64, scale `bi>>2` | correct; comments fixed |
| `gemm_iu8g32_w8a8_min` + `block_i8_128` | per-128 | correct |
| `gemm_iu8g32_residual_wmma` eligibility | K%128 | **was K%32** -> fixed; profiler bytes were `m*(K/32)*34` -> fixed |
| multi-slot `slots_iu8g32_residual_ok` | K%128 | **was K%32** -> fixed |
| `iu4_w4a8_i8_gemm_gpu` qt59 fixture | per-128 | **built per-32** -> fixed |
| `iu8g32_v2_gemv_gpu` / `iu8g32_wmma_gemm_gpu` | per-128 | correct (the `k/32` there is the activation block / code loop) |
| docs (`quant-formats/iu.md`, `QUANTIZE.md`) | per-128 + current kernels | corrected (prefill/DFlash capability rows were stale) |

No loaded model was misread: the loader already refused a per-32-length qt59
blob, so every reachable shape satisfied K%128. The fixed guards close the
latent synthetic/future-shape hole (a K%96 call would have computed
`row_stride = K` and read scales from code bytes).

### Container settings applied from the findings

- `multi_slot_prefill_chunk` 1024 -> **2048**. Measured with the W8A8 route,
  one lane and four concurrent 8k prompts, two fresh-process rounds each:
  PP +3.7-4.8% single / +4.1% aggregate, TTFT -3.6-4.6% single / -3.8%
  aggregate. Cost: the auto KV pool drops 1380 -> **1200 pages**
  (176,640 -> 153,600 tokens of shared KV, chunk 4096 was 855).
- `--max-seq 262144` stays (it bounds requests, not reservations; admission
  charges the request's real prompt+max_tokens). The physical ceilings are
  documented in the config: 1200 pages = 153.6k tokens total, so one request
  fits up to prompt+max_tokens <= 153.4k (e.g. 120k prompt + 32k output).
- `prefix_cache` left off: enabling it produced no reuse across independent
  requests and the chain harness reported `cached=0` in both arms; no
  measured benefit on this route.
