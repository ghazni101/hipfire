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
| C9 dot8 | 5841 | 5492 | 192.25 | 181.36 |
| MQ4P | 8073 | 7465 | 200.59 | 189.48 |

Per-round tg64@128: C8 dot4 156.99/156.67/156.53; C9 dot4 192.74/192.42/191.74. dot4 decode is +22.8% (tg128) and +21.0% (tg4096), now 6% below f32. Prefill is unchanged (within noise).

**KL / PPL vs MQ4P** (unchanged from C8 to every digit, as expected from bit-identity):
- 4B dot4: text KL 3.575e-4, top1 1017/1023, PPL 4.6917; gen 1.245e-4, top1 254/256.
- 4B dot8: 8.383e-4, PPL 4.6951.
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
