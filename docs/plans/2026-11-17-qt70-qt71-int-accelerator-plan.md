# qt70 / qt71 integer-accelerator critique and test plan

Date: 2026-11-17. Host: gfx1101 (RX 7700 XT), confirmed with `rocminfo`.

Scope: IU4G128 (qt 70) and IU8G128 (qt 71) as implemented on
`feat/iu4-iu8-quant-family` @ `83fb107a8`, rebased onto `master` `d435a753d`
(IU is not on `master`). The same numbers are the `SharedWeightFormat` tags.
Read against the local `rdna3-isa-condensed.md`, `docs/quant-formats/iu.md`,
`IU4_NOTES.md` (C1–C27) and the kernels under `kernels/src/`.

Wire-id note: the IU layouts were renumbered from qt 60/61 to qt 70/71 after
the rebase, and the shared-weight tags followed them. No kernel or byte layout
changed with the renumber, but older notes (including `IU4_NOTES.md`) say
qt60/qt61 for what the tree now calls qt 70/71; the mapping is also recorded
at the top of `IU4_NOTES.md`.

Status of the evidence: every speed and KL number below is quoted from the
branch's own notes, measured on gfx1100 or gfx1151. Nothing here was run on a
GPU. None of those numbers may be reused for gfx1101 (CLAUDE.md rule 4); the
plan re-measures them.

## 1. Where the integer units run today

| path | qt70 (IU4G128) | qt71 (IU8G128) |
|---|---|---|
| decode, 1 slot | f32 FMAC (`gemv_iu4g128`; `HIPFIRE_IU4_DECODE=dot4` = W4A8 `gemv_iu4g128_dot4q`) | f32 FMAC (`gemv_iu8g128_v2`) |
| decode, multi-slot | f32 FMAC (`gemv_iu4g128_xbatch`) | f32 FMAC (`gemv_iu8g128_xbatch`) |
| prefill, n >= 48 rows | IU8 WMMA W4A8 (default) or IU4 WMMA W4A4 | IU8 WMMA (`gemm_iu8g128_w8a8_min`) |
| prefill / verify, n < 48 | same 128x128 tile, mostly idle | F16 WMMA fed by int8 -> f16 dequant |
| any GEMM on gfx1101 | not admitted | not admitted |

So the integer matrix core is used in one regime (large-N prefill) on two
architectures. The dot instructions (`v_dot4_i32_iu8`, `v_dot8_i32_iu4`) are
used nowhere by default.

## 2. Critique

Ordered by expected served impact. "Certain" means read directly from the
code; "inferred" means reasoned from the code plus the notes' numbers.

### F1. On gfx1101 neither format reaches an integer unit at all (certain)

`arch_caps::supports_iu4g128_gemm` and `supports_iu8g128_gemm` are
`matches!(arch, "gfx1100" | "gfx1151")`. gfx1101 has the same ISA (WMMA
IU4/IU8, both dots, VOPD), but both formats fall to the per-token decode loop
for prefill. The notes show what that costs on a 4B model on gfx1100: about
150 tok/s per-token against about 5,800 batched. The exact-arch gate is
correct as a *tuning* gate (LDS footprint, occupancy, tile shape) and wrong as
a *capability* gate. This is the largest gap on this host and needs no new
kernel design.

### F2. Small-N is on no integer core, and it is the regime that sets served speed (inferred)

Both GEMMs are built for the n >= 128 asymptote. Notes, wo shape
(M=5120, K=6144, a 31.5 MB int8 matrix):

- n=128: 0.430 ms. That is 73 GB/s of weight stream, against roughly 620
  GB/s for the GEMV on the same bytes and a compute bound near 0.08 ms. Both
  walls say about 5x headroom.
- n < 48: the selector leaves the IU8 core for the F16 twin (12% of WMMA peak).

DFlash verify is 16 rows, multi-slot decode is 2–4 rows, and short prompt
chunks and chunk tails sit below 128. A 128-token-wide LDS tile cannot be
efficient there: LDS staging pays for reuse across token tiles, and at
N <= 16 each weight byte is used exactly once. WMMA's N dimension is 16,
which is exactly one DFlash block.

### F3. The experiment that ruled out integer decode did not test the right design (inferred)

C7–C9 concluded "decode integer dots are not a lever". The losing kernel
(`gemv_iu4g128_dot4q`) differs from the f32 winner in four ways that have
nothing to do with the dot instruction:

- 4-byte code loads per lane (f32 kernel: 16-byte), so 4x the load
  instructions per weight byte;
- `xq` is 16 B per 8 activations, re-read by every row: 4x the weight bytes;
- one row per 32-thread block instead of four waves per block;
- a per-32 activation scale, which forces an int -> f32 fold every 8 weights
  per lane instead of one per 128.

It lost by 6% with all four handicaps. For one slot on a card whose fused
stream already runs at about 874 GB/s there is probably nothing for an integer
unit to win, because the step is bandwidth bound. That part of the conclusion
is likely right. It was not shown for the multi-slot case (F4), and the
quality rejection compared KL against MQ4P (3.6e-4 vs 3.8e-5) rather than
against the weight-quantization budget (KL vs BF16 about 4.7e-2), where the
A8 term is under 1%.

### F4. Multi-slot qt70 decode hit a register wall that integer activations remove (inferred)

C25: the B=4 xbatch reads projections at about 400 GB/s, against 622 for the
singleton and about 870 for the best stream. Row blocking lost because f32
activations for four slots cost 32 VGPR words per 8 weights per lane. Int8
activations cost 8 words and int4 cost 4, and one `v_dot4_i32_iu8` does four
multiply-accumulates. VALU per 8 weights per lane at B=4: f32 about 48 ops
(8 extract, 8 convert, 32 FMAC), W4A8 about 11 (nibble split plus 8 dots).
Integer xbatch was never tried; dot4 was only measured at B=1.

### F5. `iu4_row_dot_wide` repeats row-independent work in every row (certain)

- `xs = sum(x)` over each 32-chunk is recomputed by every row's wave: 32 adds
  per 16 weight bytes. It is a property of `x` only. A `K/128` vector `XS`
  computed once per GEMV turns the zero-point term into `K/128` FMACs per row.
- 128 B of f32 `x` are re-loaded per 16 B of weights: 8 `global_load_b128` of
  activation for every 1 of weights, one row per wave. `gemv_iu8g128_v2`
  already amortizes this over R = 2 or 4 rows; qt70 has no R > 1 variant.

Whether either matters depends on whether the kernel is load-issue bound or
bandwidth bound on the target card, which P1 measures before anything is
changed.

### F6. `gemm_iu8g128_w8a8_min` is a prototype serving as the production route (certain)

It says so in its header and in C19 ("no double buffer/raster yet"). Against
its W4A8 sibling `gemm_mq4g256v2_residual_mmq_i8.gfx11.hip`:

- two `__syncthreads` per K128 with load -> LDS store -> barrier in series; the
  sibling runs a register-prefetched ring with one barrier per K64 slab;
- LDS is 33,792 B. The `dx` and `sw` planes are exactly the 1,024 B that put it
  over 32 KiB, so it runs one CTA per CU. W4A4 reaches two per CU at 30,720 B;
- no 256 tile, which the sibling added because gfx1151 is bound by
  memory-system power rather than WMMA issue;
- 75 TFLOPS at 4,096 rows, but 35 at 1,024 and 18.7 at 128. Real prefill
  chunks are N=512, so qt71 runs at roughly 20–29% of WMMA peak.

### F7. qt70's default prefill gives up 19% for quality without a measured Pareto (inferred)

4B, gfx1100: W4A8 7,700 tok/s at KL 0.0467; W4A4 9,470 at 0.0982. IU4 and IU8
WMMA are both 16x16x16 with the same FLOP per instruction, so A8 costs bytes
(LDS footprint, traffic, occupancy), not multiplier time. The untested middle:
per-role mixes (C14's override was removed by the cleanup commit, so a new one
would have to be re-added), a better A4 quantizer (two-candidate scale search,
MSE clip), and the K64 activation layout that C18 names as the only way past
66% of peak.

### F8. Format choices that cost the integer path

- **qt70 stores unsigned codes plus an f16 zero per 128**, while the only
  integer-friendly identity is the symmetric one (`zero = -8 * scale`). Every
  integer kernel must check the stored zero, then either XOR `0x88888888` or
  carry a `-8 * sum(x)` seed. Storing signed nibbles with a scale-only header
  is 4.125 bpw instead of 4.25 and deletes the zero-point path. This breaks
  the byte-exact MQ4V2 twin and asymmetric artifacts, so it is a candidate to
  measure, not a recommendation.
- **qt71 is unrotated**, so its int8 activation scale is `amax/127` over 128
  raw activations, which are heavy tailed. Kernel-level A8 error is 2.3e-2
  relative. FWHT-rotated qt71 was noted as untested.
- **Per-128 f16 scales on both operands** force one int -> f32 fold per C
  element per K128. This is the fixed overhead that small tiles cannot hide.
- **WMMA A/B operands are duplicated across lanes 0–15 and 16–31**, so half of
  every operand load is redundant. Whether the hardware reads the upper lanes
  is not stated in the ISA guide and has not been probed.

### What is already right, and should not be re-litigated

Planar code plane with tail scales and 16-byte row padding; nibble order
matching the WMMA K order with no permute; the magic-seed int -> f32 fold; the
x-batched lm_head (at the wall, 815–873 GB/s); one selector with a measured
crossover; the removed losers (non-temporal loads, mb8 fanouts, row-blocked
f32 xbatch, int8 qt71 GEMV at B=1, and the C14/C15 role/planar knobs). The
plan does not repeat these.

## 3. Test plan

Rules that apply throughout (from `docs/methodology/lessons-0.4.5.md` and
CLAUDE.md): decompose the served gap before choosing a lever; kill experiments
before kernels; a microbench win is a candidate, not a result; warm every
(config x shape) cell; throwaway arm first; at least 4 fresh processes per
arm, ABBA; byte-identical prompts with md5 recorded; distinct prompts per lane
at C > 1; bootstrap 95% CI of the difference; measure the A/A band on the
fixture; hold the GPU lock; commit every arm including the losers.

### P0. Baseline and attribution on gfx1101

1. Worktree of the branch; `codegraph init .`; build; quantize or fetch one 4B
   `iu4` artifact and the largest that fits 12 GiB.
2. A/A pair on decode and prefill to get this card's noise band.
3. `rocprofv3 --kernel-trace` on a decode step and on a prefill run
   (`hipfire bench --matrix`; `prefill_microbench` for a single shape), both
   through the daemon so the trace flushes. Produce one table: per kernel,
   time share, GB/s of weight stream, TFLOPS.
4. Record the per-token prefill baseline, which is what F1 replaces.

Exit: a decomposed gap. Every later phase is ranked by this table, not by §2.

### P1. Ceilings and hardware truth (no production code)

Each probe is a throwaway kernel in an example binary.

| id | probe | answers | kills |
|---|---|---|---|
| K1 | Null-math stream: wide loads + XOR reduce, swept over load width, rows per wave, waves per block, matrix size | this card's bandwidth wall and the launch-overhead floor for small matrices | F5 and all B=1 decode work if the incumbent is within the A/A band of the wall |
| K2 | Register-only instruction loops: `v_dot4_i32_iu8`, `v_dot8_i32_iu4`, WMMA IU4/IU8/F16, VOPD FMAC pair, extract+convert | measured MAC/s per CU: the gfx1101 cost table | any design whose arithmetic bound is below K1's stream bound |
| K3 | WMMA with garbage in lanes 16–31 of A and B | whether upper-lane operand loads can be skipped or masked | the lane-dedup idea in F8 |
| K4 | "x-free" twins: each candidate decode kernel with activations held constant in registers | upper bound for any activation-resident design | F4 if the bound is not above the incumbent by A/A + 5% |
| K5 | LDS footprint sweep on the existing tiles (CTAs per CU vs throughput) | gfx1101's occupancy optimum | borrowing the gfx1100 shapes |
| K6 | CPU emulation in `iu_quant` on captured activations: A8 per-128 in decode, A4 variants, rotated vs unrotated qt71, symmetric vs asymmetric qt70 | the KL price of each numeric change, against BF16 | any format or activation change that costs more KL than it can buy |

K6 needs no GPU and runs in parallel with the rest.

### P2. Candidates

Built only if P1 leaves them alive. Each new kernel gets a PeaceMaker twin, a
CPU oracle in `iu_quant` (independent, never the kernel against itself), a
VGPR/LDS/spill check with the `hipfire-kernel-tuning` skill, and NaN-explicit
comparisons.

| id | candidate | addresses | kill criterion |
|---|---|---|---|
| A | Admit gfx1101 to the existing qt70/qt71 GEMMs; re-tune tile, LDS and occupancy from K5; add gfx1101 to the selector receipts | F1 | none expected; bit-identity against the CPU oracle is the gate |
| B | qt70 f32 decode: hoisted `XS` vector, R rows per wave | F5 | served decode CI spans zero |
| C | Integer xbatch decode for qt70: activations quantized once to `block_i8_128` (the prefill contract, so decode and prefill share numerics), held in LDS or registers, `v_dot4_i32_iu8`, one fold per 128, B = 2..4 | F3, F4 | below f32 xbatch + 5% at B=4, or KL vs BF16 moves by more than the K6 budget |
| D | Narrow-N streaming WMMA (N tile 16 and 32): weights global -> operand registers, no LDS staging of weights, IU8 for qt71 and IU4/IU8 for qt70; replaces the F16 twin below the crossover and serves DFlash verify | F2 | not above the F16 twin (qt71) or the 128-tile (qt70) at n = 16 and n = 32 |
| E | Production qt71 W8A8: prefetch ring, LDS <= 32,768 B (scales out of LDS), optional 256 tile | F6 | no gain at N = 512 and N = 1024 |
| F | qt70 prefill Pareto: A8, A4, A4 with scale search, per-role mixes (re-add the C14 selector), K64 activation groups | F7 | n/a: output is a frontier, the default is picked from it |
| G | Format variants: signed-symmetric qt70, rotated qt71 | F8 | K6 shows a KL cost, or A–F show under 5% served gain from it |

Order: A first (largest, cheapest, unblocks everything else on this card),
then D and C (the served-speed regimes), then E, F, B, and G last because it
is the only one that invalidates artifacts.

### P3. Acceptance

- Kernel and dispatch changes: `scripts/redline_daemon_harness.py`, with stable
  capture, valid contracts and multi-position parity; JSON report recorded.
- Generation: `scripts/serve_harness.py` `battery`, `chain` and `session` on
  the exact model and settings; include a bare factual prompt.
- Speed: the fresh-process ABBA protocol above, on the daemon, not a demo.
  Spec-decode arms report tau and tok/s, with q8 KV.
- Numeric changes: paired KL per prompt-length band against BF16 and against
  the incumbent; served C1 replies compared against the incumbent.
- `scripts/pre-commit-ci.sh origin/<base>` on the final commit and
  `cargo test -p railgun-cert`.

### Decision rule

For each (arch x regime) cell, where the regimes are N=1, N=2..47 and N>=48,
keep the candidate on the speed/KL frontier with a CI that excludes zero, and
record the crossover in `receipts/selectors/`. A default changes only on a
served win. A format change (G) needs a served win of at least 5% that no
kernel-only candidate reaches.

## 4. Expected outcome (hypothesis, to be confirmed or killed by the above)

- N = 1 decode stays f32. The stream is bandwidth bound and an integer dot
  cannot beat a wall; the gains there are fusion and hoists.
- N = 2..47 moves to streaming integer kernels: dot4 for the 2–4 slot decode,
  narrow-N WMMA for verify blocks and short chunks. This is where both formats
  currently leave the integer units idle and where the notes show 2–5x between
  measured and either wall.
- N >= 48 keeps the LDS-tiled WMMA, with the qt71 kernel brought up to the
  qt70 kernels' standard and every shape re-tuned per arch.
- The wire formats stay unless G clears its bar.
