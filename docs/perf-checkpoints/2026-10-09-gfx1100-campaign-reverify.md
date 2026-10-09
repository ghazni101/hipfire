# gfx1100 decode-campaign — independent re-verification vs current master

- **Date:** 2026-10-09
- **Lifecycle:** historical
- **Branch under test:** `perf/gfx1100-decode-campaign` @ `c382347e7822`
  (binary md5 `0f6345abb0ce92eb9b68803d166900db`,
  daemon md5 `f931d578a9b8361145c53cb7f90a559b`)
- **Baseline:** `master` @ `0f999cb4dc` (binary md5 `879d8643ae946c2be6dc05e4e2d0b408`,
  daemon md5 `022719b51d0a62f04d3b4bbe000fbedf`)
- **GPU:** gfx1100 (Radeon RX 7900 XTX, 24 GiB), exclusive.
- **Env:** containerized, `buun-llama:git-9be465c6c-rocm10.0.0`
  (ROCm 7.15 / HIP 7.15), shared `HIPFIRE_KERNEL_CACHE` volume.
- **Method:** `hipfire bench <model> --matrix --json --kv-mode q8 --spec off`
  (native daemon protocol), 5 runs / 3 warmups per cell, fresh process per
  arm, arms interleaved A/B/A/B within a window. Identical image, model
  bytes and env for both arms; only the binary differs.

> This record is a **re-verification** of the 2026-08 campaign numbers
> against the *current* master (1957 commits ahead of the campaign's
> base). It is not a re-measurement of the campaign's original
> base-to-tip delta, and it must not be read as one.

## Fixtures

| Artifact | Identity | Notes |
|---|---|---|
| `qwen3.5-4b.mq4` (2,588,006,400 B) | md5 `712b69f8cf1016081cfa507c4d50e33d` | matches the campaign's fixture exactly |
| `qwen3.8-27b.mq4-xt` (14,987,185,152 B) | sha256 `80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3` | current HF artifact. **The AGENTS.md §5 pin (`9f91556f…` / 14,980,361,216 B) is stale** — HF re-uploaded the file; our download byte-matches the live HF object, not the recorded pin. |
| `qwen3.8-27b.mq4` (15,662,615,552 B) | md5 `2fb2edc27865b697c5209c104de91f32` | the fixture the 2026-08 BT2 deep A/B was taken on (legacy HFQ4-era trunk) |

## 4B MQ4 — product path (interleaved, median of 3 windows × 5 runs)

| arm | pp64 prefill | pp2048 prefill | decode @ctx64 | decode @ctx2048 |
|---|---|---|---|---|
| master | 3113.8 | 5239.0 | 200.20 | 195.92 |
| branch (default) | **3806.8** | 5235.7 | **212.76** | **208.09** |
| branch + `HIPFIRE_FA_KVWRITE_FOLD=1` | 3794.0 | 5232.9 | 213.51 | 208.57 |
| branch + `HIPFIRE_BT2_DISABLE=1` | 3129.7 | 5233.4 | 212.80 | 207.95 |

- **Decode: branch default is +6.3% over master @ctx64** (200.20 → 212.76;
  every branch sample above every master sample) and **+6.2% @ctx2048**.
  Carried by the four admitted default fusions (conv `scalar_prep`,
  `gated_norm_mq_rotate`, `fa_prep` 16Q/4K + rope-contraction fix,
  `fa_epilogue` 16Q/4K).
- **Prefill pp64: +22.3%** (3113.8 → 3806.8), and the BT2 kill switch
  isolates it: `BT2_DISABLE=1` collapses pp64 to 3129.7 ≈ master
  (within +0.5%). **pp2048 is flat** (5239.0 → 5235.7) — the BT2 win is a
  small-grid/occupancy effect and does not survive a saturated grid.
- `HIPFIRE_BT2_DISABLE=1` is therefore a faithful master-equivalence
  switch: on the 4B it removes the entire pp64 delta and none of the
  decode delta (as designed).

## Levers re-measured against the campaign records

| Lever | Campaign record | This re-verification |
|---|---|---|
| `HIPFIRE_FA_KVWRITE_FOLD=1` | +0.5–1% decode, 3/3 wins | **+0.55%** (25 samples/arm, interleaved; t=2.88, df=40.7, **p=4.0e-3**, Cohen's d=0.81; 6 wins / 2 losses across 8 pairs) |
| `HIPFIRE_QKVZA_FUSEDNORM=1` | −26% decode (negative oracle) | **−25.6% / −26.4%** (212.4 → 158.0; 214.3 → 157.7) |
| `HIPFIRE_LM_HEAD_HFQ3=1` | +~10% decode (research-only, lossy) | **+9.8%** (213.3 → 234.3) |
| `HIPFIRE_BT2_DISABLE=1` | +29% pp32 prefill on 27B | see below |

## 27B — BT2 does not apply to MQ4V2 XT, and the +29% does not reproduce on today's bytes

Product path, `qwen3.8-27b.mq4-xt` (the current default artifact):

| arm | pp64 prefill | pp2048 prefill | decode @ctx64 |
|---|---|---|---|
| master | 588.4 | 2240.4 | 51.90 |
| branch | 590.8 | 2234.3 | 52.06 |
| branch + `HIPFIRE_BT2_DISABLE=1` | 590.0 | 2234.5 | 52.17 |

branch ≈ branch+`BT2_DISABLE` on every cell: the BT2 kernels do not engage
on this fixture (they are `gemm_*_hfq4g256_wmma_bt*`; MQ4V2 XT tensors take
a different GEMM family) and branch-vs-master is flat within noise.

Client: `bench_qwen35_mq4 --prefill 32` (**the harness the 2026-08 deep A/B
used**), `HIPFIRE_VERIFY_GRAPH=0`, DPM warmup, 3 interleaved windows,
current `qwen3.8-27b.mq4` bytes:

| arm | prefill tok/s (3 windows) | median | decode |
|---|---|---|---|
| master | 469.3 / 459.5 / 460.5 | 460.5 | 49.5 |
| branch | 460.5 / 460.5 / 461.7 | 460.5 | 49.5 |
| branch + `BT2_DISABLE=1` | 458.1 / 461.0 | 458.9 | 49.5 |

Two conclusions:

1. **BT2 is inert on this fixture too** (`BT2_DISABLE` changes nothing), so
   the 2026-08 `+29%` BT2 figure (baseline median 452.55 → bt2 583.95) does
   not reproduce: master and branch both read ~460, nowhere near 584.
   **Likely cause: fixture drift** — the recorded fixture md5
   `2fb2edc2…` no longer matches what HF serves. Today's
   `qwen3.8-27b.mq4` is md5 `d1292b4d5bd6046693604201a6ca8074`
   (sha256 `5bb556a6…e507`, size unchanged at 15,662,615,552 B), and that
   sha256 is byte-identical to the current HF LFS object — i.e. HF
   re-uploaded the file, exactly as with `qwen3.8-27b.mq4-xt`. The
   artifact the 2026-08 measurement was taken on is no longer retrievable
   by digest.
2. **No 27B regression**: branch ≡ master within noise (median 460.5 both).
   A single earlier window read master 480.6 / branch 459.0 (−4.5%); the
   interleaved re-run above shows that was window-to-window drift, not a
   branch effect. Recorded here so the wrong reading is not propagated.

Consequence for the claim line: the `+29% pp32 prefill on 27B` headline is
fixture-bound **and** not reproducible today. It is retained in
`docs/env-vars.md` only as a dated 2026-08 record with its fixture named.


## Correctness / bit-exactness (gfx1100)

Kernel-level probes (branch binary):

| Probe | Result |
|---|---|
| `test_kernels` | **17 passed / 0 failed / 1 skipped** — identical on master and branch |
| `probe_fa_kvwrite` | **PROBE PASS**: `fa_q`/`fa_gate`/`fa_k` and *every byte of both Q8_0 cache rows* bitwise identical to the legacy pair-writer |
| `probe_qkvza_fusednorm` | **BITEXACT** on y_qkv/y_z/y_beta/y_alpha (mism=0), gate_up, qkv |
| `probe_fa_prep` | `fa_q`/`fa_gate`/`fa_k` IDENTICAL |

End-to-end determinism (native serve, `temperature=0`,
`enable_thinking=false`, 64 tokens, content compared byte-for-byte):

| Comparison | Short prompt | Long prompt (1.6 kB → multi-chunk/BT2 prefill) |
|---|---|---|
| master vs branch (default) | **IDENTICAL** | **IDENTICAL** |
| branch vs branch (repeat) | IDENTICAL | IDENTICAL |
| branch vs `+BT2_DISABLE=1` | — | **IDENTICAL** (BT2 kernels are token-exact vs the plain WMMA path) |
| branch vs `+FA_KVWRITE_FOLD=1` | IDENTICAL | IDENTICAL |
| branch vs `+RESIDUAL_DUALROW=1` | IDENTICAL | — |
| branch vs `+QKVZA_FUSEDNORM=1` | IDENTICAL | — |
| branch vs `+XLDS_R=4` | **DIFFER** | **DIFFER** (degenerates to a repeating-token attractor) |

`cargo test --lib --workspace --locked`: **4582 passed, 0 failed**.

Serve-path battery (`serve_harness.py --mode battery --kv q8 --speculation off
--thinking off --seed 42 --max-tokens 256`, per-turn rows in
`battery_branch.json` / `battery_master.json` in the run artefacts):

| arm | turns | runaway | empty | attractor | avg prefill | avg decode |
|---|---|---|---|---|---|---|
| master | 5/5 | 1 (length cap) | 0 | 0 | 2199.6 | 190.0 |
| branch | 5/5 | 1 (length cap) | 0 | 0 | 2284.7 | **202.9** |

Both arms are coherent on all five genres; the single runaway on each is the
reasoning turn hitting the 256-token cap at the registry's temperature 1.0
(documented behaviour, not degeneracy). The serve-path decode delta
(+6.8%) matches the bench delta (+6.3%) — the branch's default gains are
visible on the user-facing path, not only in the bench harness.

### Invalid oracle removed: the LDS-staged activation probe

The gfx1100 LDS-staged activation probe carried two independent defects:

1. **Launch:** the kernel stages the full activation row in
   `extern __shared__ float xs[]` (writing `xs[0..k)`), but the wired dispatch
   requested **0 bytes** of dynamic shared memory — an out-of-bounds
   configuration for every measurement taken through it.
2. **Kernel math:** each lane reads one 4-byte word (nibbles `8t..8t+7` at
   `x` offsets `8t..8t+7`), but `acc1` pairs nibbles 4..7 with `xv.x..w`
   (x offsets `8t+0..8t+3`) instead of a second `float4` at offset `+4`. The
   kernel never computed the GEMV it claimed.

End-to-end parity shows the engaged arm degenerating into a repeating-token
attractor, while the 2026-08 amendment 2k record claims "BITEXACT" and a
`−16%..−37%` schedule table. Both the claim and the table are void. The
oracle was **removed** (dispatch, kernel constant, `.hip` source) rather than
kept as falsification evidence, because a broken oracle is not evidence; the
env-var table now records the removal. The `−26%` residual-LUT and
persist-dual-row oracles are unaffected (they are separately wired and their
schedules are correct as written).

## Gate status

- `scripts/leanup-ratchets.sh` — **OK** (21 assertions, 0 violations;
  `bypass_total` 310).
- `scripts/check-crate-maps.py --check`, `check-scratch-growth.py`,
  `check-changelog.py` — **OK**.
- `scripts/ratchet-diff.sh origin/master` — **FAIL** until the
  `bypass_total 307 → 310` raise is declared with a `RATCHET-RAISE:`
  trailer in a commit message **and** the PR carries the `ratchet-raise`
  label. (The branch's prose "Traded:" note does not match the trailer the
  gate greps for.)
- `scripts/speed-gate.sh --fast` on the branch tip — **withdrawn floor**: the
  campaign committed `4b_mq4_pp32_prefill_tok_s=1843.5`, which a clean ROCm
  7.15 container cannot reach (branch 1663.4, master 1601.0 — the branch is
  +3.9% over master, not regressed), so the gate and the pre-commit hook that
  runs it failed on the branch itself. This PR restores the merge-base 4B rows,
  after which the gate passes (observed 1796.6 pp32 / 198.0 gen). Re-capture on
  the reference machine if the tighter floor is wanted.

## Quant-class sweep: do the gains survive the Magnum V2 class?

The 2026-08 campaign was measured on the *legacy* quant family
(`HFQ4G256` / `MQ4G256`). The current model class is **Magnum V2**:
`MQ4G256V2` (qt44), `MQ4G256V2Lloyd` (qt52), `MQ4CG256` (qt45), `MQ6G256V2`
(qt47), `MQ5G256V2` (qt48), `MQ3G256V2` (qt49), `MQ4G128V2` (qt53) — with
`pro` / `xt` / `xts` recipe tiers layered on the same dtype families. Swept on
gfx1100, product path, `--kv-mode q8 --spec off`, 3 interleaved windows ×
5 runs per arm (single window for the 27B tiers):

### Which kernels engage (prefill profile, branch)

| artifact | dense prefill GEMMs | BT2 present? |
|---|---|---|
| `qwen3.5-4b.mq4` (legacy) | `gemm_{gate_up,qkvza,qkv}_hfq4g256_wmma_bt2`, `gemm_hfq4g256_residual_wmma_ksplit_det_bt2` | **yes** |
| `qwen3.8-27b.mq4-xt` (V2 XT) | `gemm_{gate_up,qkvza,qkv}_mq4g256v2_wmma`, `gemm_mq4g256v2_residual_wmma` | no |
| `qwen3.8-27b.mq4` (V2 base) | `gemm_*_mq4g256v2_wmma` | no |
| `qwen3.8-27b.mq4-pro` (V2 PRO) | `gemm_*_mq4g256v2_wmma`, `gemm_q8_0_residual_wmma` | no |

The BT2 kernels are `..._hfq4g256_...` — they are only reachable from the
legacy HFQ4/MQ4-G256 GEMM family, so **no V2 quant can engage them**.

### Measured deltas, branch vs master (4B geometry, decode @ctx64)

| artifact | master | branch | Δ | BT2 kill switch | conv-prep off | gated-norm off |
|---|---|---|---|---|---|---|
| `qwen3.5-4b.mq4` (legacy) | 200.20 | 212.76 | **+6.3%** | −17.8% pp64 | −0.9% | −1.2% |
| `qwen3.5-4b.mq3` (V2) | 208.32 | 214.98 | **+3.2%** | −0.7% (inert) | −2.0% | −0.7% |
| `qwen3.5-4b.mq6` (V2) | 158.04 | 162.04 | **+2.5%** | +0.2% (inert) | −1.5% | +0.1% |

Every window of the branch is above every window of master in all three rows
(per-window spreads ≤0.2% within an arm). The opt-in lm_head requant — which
keys off the model's Q8_0 head, not the body quant — also generalises:
**+9.8% / +11.0% / +7.8%** on legacy mq4 / mq3 / mq6.

### Reading

- **Decode gains hold across the new class, but attenuate by ~half.** The
  survivors are the dtype-agnostic fusions (conv scalar-prep, `fa_prep`,
  `fa_epilogue` — all arch+shape gated, no dtype term). The piece that drops
  out is `gated_norm_mq_rotate`, whose dtype list is
  `MQ4G256 | MQ4G256V2 | MQ4CG256`: it fires on the mq4 tiers but not on
  `MQ3G256V2` / `MQ5G256V2` / `MQ6G256V2` (nor on `MQ4G256V2Lloyd`). The
  kill-switch columns show exactly that: −1.2% on legacy mq4, ~0 on mq6.
- **The BT2 prefill win does not carry over at all.** `+22.3%` pp64 is a
  legacy-HFQ4/MQ4-G256-only result; on mq3/mq6 it is inside noise and the kill
  switch is inert, and it is flat on every 27B V2 tier (base/XT/PRO).
- **The 27B tiers gain ~nothing from this branch** — `pro`/`xt`/`xts` all
  resolve to the same `MQ4G256V2` GEMM family, where the branch's new levers
  either do not apply (BT2, fusednorm oracles: MQ4G256/HFQ4G256-only) or were
  already present at that shape on master (the four fusions). Measured:
  `mq4-xt` +0.3%, `mq4` (legacy bytes) flat, `mq4-pro` +0.3% decode.
- **Opt-in levers**: `FA_KVWRITE_FOLD` is arch+shape gated, so it applies to
  any dtype at 16Q/4K; the fusednorm oracles require
  `MQ4G256 | HFQ4G256` and therefore do not apply to V2.

