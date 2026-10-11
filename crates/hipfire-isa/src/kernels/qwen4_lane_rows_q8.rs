// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 Q8_0 row kernels of the Flash-Next lane step: exact multi-column
//! (1..=8 activation columns) twins of the HIP incumbents `gemv_q8_0_k8` /
//! `gemv_q8_0_k320_staged` that read each activation once for several weight
//! rows instead of once per weight row.
//!
//! Every column `c` of every output row keeps the incumbent's arithmetic DAG
//! (`fn-batch/s7/DAG.md` §2, §3), so `y[c*M + row]` is byte-identical to the
//! 1-row incumbent on `x[c*K..]`:
//!  * a Q8_0 block is 34 bytes (f16 `d`, then 32 int8 `q`); lane `l` of a
//!    block takes `q[l]` and `x[32*b + l]`; `dq = v_fma_mix_f32(d_f16,
//!    f32(q), -0.0)` (`op_sel_hi:[1,0,0]`), `sum = fma(x, dq, sum)` from `+0`
//!    in increasing block order;
//!  * lane reduction: `ds_swizzle_b32 "1pppp"` (offset 16), then
//!    `ds_bpermute_b32` 8/4/2/1 with own-lane addresses for the out-of-range
//!    lanes, each level `own + other`; lane 0 holds the wave's value;
//!  * k8 only: lane 0 of wave `v` parks `part[row][col][v]` in LDS, one thread
//!    per (row, column) folds `((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))` and, for
//!    `hc_act != 0`, runs the BF16 round trip, `scale * B`, a second round
//!    trip, the incumbent's exact exp/division expansion (SiLU) and a final
//!    round trip, all as VALU (the HIP object's `s_mul_f32` / `s_add_f32` /
//!    exec-masked sequences become the same operations lane-wise; the
//!    BF16 exceptional-exponent path is a `v_cndmask` select that yields the
//!    same bits). The F32 bits are stored.
//!
//! # `gemv_q8_0_k8_xbatch_pm`
//! Kernargs `A@0, x@8, y@16, M@24, K@28, B@32, scale@36 (f32), hc_act@40`
//! (44 B). Grid `ceil(M/4)`, block 256 (8 waves), static LDS 1024 B
//! (`part[4][8][8]` f32). Workgroup `w` owns weight rows `4w..4w+3` (loads
//! clamped to `M-1`, stores only for `row < M`); wave `v` runs K-slice `v`
//! (`K/8` elements, `K/256` blocks, `K % 2048 == 0`) of the four rows for
//! every column, and each activation load feeds the four rows. The slice is
//! processed as `K/2048` iterations of 8 unrolled blocks. Loads run in a
//! four-block software pipeline (a ring of four block slots; the first four
//! blocks of the next iteration load while the last four compute; the last
//! iteration re-reads its own first blocks instead of running off the slice).
//! A and the per-row offsets are 32-bit VGPR offsets from the kernarg bases:
//! `M * (K/32) * 34` must stay below 2^32 bytes and `B * K * 4` below 2^32.
//!
//! # `gemv_q8_0_k320_xbatch_pm`
//! Kernargs `A@0, x@8, y@16, M@24, B@28` (32 B), `K == 320` (10 blocks, 340
//! bytes per row). Grid `ceil(M/16)`, block 64 (2 waves), static LDS 1536 B.
//! Wave `v` of workgroup `w` owns rows `16w + 8v .. +7` (clamped to `M-1`),
//! keeps its `B x 10` activations in registers, and per row stages the 340
//! byte row in wave-local LDS (85 dwords: lane, lane+32, lane+64 for lane <
//! 21; lanes past 21 re-read dword 84 into a 44-byte pad of the 384-byte wave
//! region), then reads `d` / `q` back from LDS. Rows are double-buffered over
//! two LDS slots with one workgroup barrier per row; the global loads of a
//! row run four rows ahead through a register ring.
//!
//! B is specialized (one straight-line path per column count, selected once
//! from the kernarg); `B` outside `1..=8` exits without storing.
use crate::insn::{Instruction, MemoryClass};
use crate::ledger::Counter;
use crate::kernels::common::{lit, mem, op, s, smem, sop, sr, v, vr};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

pub const K8_SYMBOL: &str = "gemv_q8_0_k8_xbatch_pm";
pub const K320_SYMBOL: &str = "gemv_q8_0_k320_xbatch_pm";
pub const MAX_COLUMNS: u8 = 8;

/// Static LDS of k8: `part[4 rows][8 columns][8 waves]` f32.
pub const K8_LDS_BYTES: u32 = 4 * 8 * 8 * 4;
/// Static LDS of k320: two row slots of two 384-byte wave regions.
pub const K320_LDS_BYTES: u32 = 2 * 2 * K320_WAVE_REGION;
/// One wave's staging region of k320: 340 row bytes plus a 44-byte pad that
/// absorbs the third staging store of the lanes past 20.
const K320_WAVE_REGION: u32 = 384;
/// One k320 slot (both waves' regions).
const K320_SLOT: u32 = 2 * K320_WAVE_REGION;

/// Values per lane-tree batch (temporaries).
const TREE_CHUNK: usize = 8;

// ------------------------------------------------------------------ helpers

/// Declare `base..base+len` as the largest aligned pieces the allocator admits.
fn declare(p: &mut RegPlan, name: &str, kind: Kind, base: u8, len: u8) -> R {
    let mut at = 0u8;
    while at < len {
        let rest = len - at;
        let piece = [8u8, 4, 2, 1].into_iter()
            .find(|w| *w <= rest && (kind == Kind::V || (base + at) % *w == 0))
            .ok_or("unalignable register range")?;
        p.add_range(&format!("{name}{at}"), kind, base + at, piece, Live::Whole)?;
        at += piece;
    }
    Ok(())
}

/// One VALU instruction over VGPR defs/uses plus SGPR uses.
fn vop(b: &mut Builder, text: impl Into<String>, defs: &[u8], uses: &[u8], suses: &[u8]) -> R {
    let d: Vec<RegRef> = defs.iter().map(|&n| v(n)).collect();
    let mut u: Vec<RegRef> = uses.iter().map(|&n| v(n)).collect();
    u.extend(suses.iter().map(|&n| s(n)));
    op(b, text, &d, &u)
}

fn off(imm: u32) -> String { if imm == 0 { String::new() } else { format!(" offset:{imm}") } }

/// `global_load_*` of one dword-or-narrower value at `s[base:base+1] + v{voff} + imm`.
fn gload(b: &mut Builder, mnemonic: &str, dst: u8, voff: u8, base: u8, imm: u32) -> R {
    mem(b, format!("{mnemonic} v{dst}, v{voff}, s[{base}:{}]{}", base + 1, off(imm)),
        &[v(dst)], &[v(voff), sr(base, 2)], MemoryClass::VmemLoad)
}

fn ds_st(b: &mut Builder, slot: usize, addr: u8, data: u8, imm: u32) -> R {
    b.ds_store(slot, Instruction::new(format!("ds_store_b32 v{addr}, v{data}{}", off(imm)), vec![], vec![v(addr), v(data)])
        .memory(MemoryClass::DsStore))
}

/// `ds_load_u16_d16` (low half, high half kept) and `ds_load_i8` (sign extended).
fn ds_ld(b: &mut Builder, slot: usize, mnemonic: &str, dst: u8, addr: u8, imm: u32, keeps_dst: bool) -> R {
    let mut uses = vec![v(addr)];
    if keeps_dst { uses.push(v(dst)); }
    b.ds_load(slot, Instruction::new(format!("{mnemonic} v{dst}, v{addr}{}", off(imm)), vec![v(dst)], uses)
        .memory(MemoryClass::DsLoad))
}

/// The wave-wide lane reduction of every value in `values` (VGPRs), batched
/// by `TREE_CHUNK` through `tmp..tmp+TREE_CHUNK`: swizzle offset 16, then the
/// own-lane-clamped bpermutes 8/4/2/1 whose byte addresses sit in
/// `addr..addr+4`; each level is `own + other`, lane 0 ends with the sum.
fn lane_tree(b: &mut Builder, values: &[u8], tmp: u8, addr: u8) -> R {
    let add = |b: &mut Builder, chunk: &[u8]| -> R {
        for (i, &val) in chunk.iter().enumerate() {
            let t = tmp + i as u8;
            vop(b, format!("v_add_f32_e32 v{val}, v{val}, v{t}"), &[val], &[val, t], &[])?;
        }
        Ok(())
    };
    for chunk in values.chunks(TREE_CHUNK) {
        for (i, &val) in chunk.iter().enumerate() {
            let t = tmp + i as u8;
            b.ds_crosslane(Instruction::new(format!("ds_swizzle_b32 v{t}, v{val} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                vec![v(t)], vec![v(val)]).memory(MemoryClass::DsLoad))?;
        }
        add(b, chunk)?;
        for level in 0..4u8 {
            for (i, &val) in chunk.iter().enumerate() {
                let t = tmp + i as u8;
                b.ds_crosslane(Instruction::new(format!("ds_bpermute_b32 v{t}, v{}, v{val}", addr + level),
                    vec![v(t)], vec![v(addr + level), v(val)]).memory(MemoryClass::DsLoad))?;
            }
            add(b, chunk)?;
        }
    }
    Ok(())
}

/// Byte addresses of the shfl_down 8/4/2/1 bpermutes into `addr..addr+4`:
/// `(lane + (lane < 32 - o ? o : 0)) * 4`, i.e. the own lane past the end.
fn bpermute_addresses(b: &mut Builder, addr: u8, lane: u8) -> R {
    for (i, o) in [8u32, 4, 2, 1].into_iter().enumerate() {
        let a = addr + i as u8;
        vop(b, format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{lane}", 32 - o), &[], &[lane], &[])?;
        vop(b, format!("v_cndmask_b32_e64 v{a}, 0, {o}, vcc_lo"), &[a], &[], &[])?;
        vop(b, format!("v_add_lshl_u32 v{a}, v{a}, v{lane}, 2"), &[a], &[a, lane], &[])?;
    }
    Ok(())
}

/// The incumbent's BF16 round trip of `x` in place (RNE, exceptional exponent
/// preserved; `DAG.md` §2 / §6): the finite result
/// `(u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000`, and for an all-ones
/// exponent `u` (low half zero) or `u | 0x10000` (low half non-zero), selected
/// by `v_cndmask` instead of the HIP object's exec-masked branches. `t` are
/// four scratch VGPRs.
fn bf16_round(b: &mut Builder, x: u8, t: [u8; 4]) -> R {
    vop(b, format!("v_and_b32_e32 v{}, 0x7f800000, v{x}", t[0]), &[t[0]], &[x], &[])?;
    vop(b, format!("v_bfe_u32 v{}, v{x}, 16, 1", t[1]), &[t[1]], &[x], &[])?;
    vop(b, format!("v_add3_u32 v{0}, v{x}, v{0}, 0x7fff", t[1]), &[t[1]], &[x, t[1]], &[])?;
    vop(b, format!("v_and_b32_e32 v{}, 0xffff, v{x}", t[2]), &[t[2]], &[x], &[])?;
    vop(b, format!("v_or_b32_e32 v{}, 0x10000, v{x}", t[3]), &[t[3]], &[x], &[])?;
    vop(b, format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{}", t[2]), &[], &[t[2]], &[])?;
    vop(b, format!("v_cndmask_b32_e32 v{0}, v{1}, v{x}, vcc_lo", t[2], t[3]), &[t[2]], &[t[3], x], &[])?;
    vop(b, format!("v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v{}", t[0]), &[], &[t[0]], &[])?;
    vop(b, format!("v_cndmask_b32_e32 v{0}, v{1}, v{0}, vcc_lo", t[1], t[2]), &[t[1]], &[t[2], t[1]], &[])?;
    vop(b, format!("v_and_b32_e32 v{x}, 0xffff0000, v{}", t[1]), &[x], &[t[1]], &[])
}

/// SiLU of `x` in place: the r8 incumbent's expansion of `x / (1 + expf(-x))`
/// instruction for instruction (`DAG.md` §2, lines 10645-10679), registers
/// `t[0..6]` standing for the listing's v2..v7 and `x` for v1.
fn silu(b: &mut Builder, x: u8, t: [u8; 6]) -> R {
    let [t0, t1, t2, t3, t4, t5] = t;
    vop(b, format!("v_mul_f32_e32 v{t0}, 0xbfb8aa3b, v{x}"), &[t0], &[x], &[])?;
    vop(b, format!("v_fma_f32 v{t1}, 0xbfb8aa3b, v{x}, -v{t0}"), &[t1], &[x, t0], &[])?;
    vop(b, format!("v_rndne_f32_e32 v{t2}, v{t0}"), &[t2], &[t0], &[])?;
    vop(b, format!("v_sub_f32_e32 v{t0}, v{t0}, v{t2}"), &[t0], &[t0, t2], &[])?;
    vop(b, format!("v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v{x}"), &[], &[x], &[])?;
    vop(b, format!("v_fmamk_f32 v{t1}, v{x}, 0xb2a5705f, v{t1}"), &[t1], &[x, t1], &[])?;
    vop(b, format!("v_add_f32_e32 v{t0}, v{t0}, v{t1}"), &[t0], &[t0, t1], &[])?;
    vop(b, format!("v_cvt_i32_f32_e32 v{t1}, v{t2}"), &[t1], &[t2], &[])?;
    vop(b, format!("v_exp_f32_e32 v{t0}, v{t0}"), &[t0], &[t0], &[])?;
    vop(b, format!("v_ldexp_f32 v{t0}, v{t0}, v{t1}"), &[t0], &[t0, t1], &[])?;
    vop(b, format!("v_cndmask_b32_e32 v{t0}, 0, v{t0}, vcc_lo"), &[t0], &[t0], &[])?;
    vop(b, format!("v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v{x}"), &[], &[x], &[])?;
    vop(b, format!("v_cndmask_b32_e32 v{t0}, 0x7f800000, v{t0}, vcc_lo"), &[t0], &[t0], &[])?;
    vop(b, format!("v_add_f32_e32 v{t0}, 1.0, v{t0}"), &[t0], &[t0], &[])?;
    vop(b, format!("v_div_scale_f32 v{t1}, null, v{t0}, v{t0}, v{x}"), &[t1], &[t0, x], &[])?;
    vop(b, format!("v_rcp_f32_e32 v{t2}, v{t1}"), &[t2], &[t1], &[])?;
    vop(b, format!("v_fma_f32 v{t3}, -v{t1}, v{t2}, 1.0"), &[t3], &[t1, t2], &[])?;
    vop(b, format!("v_fmac_f32_e32 v{t2}, v{t3}, v{t2}"), &[t2], &[t2, t3], &[])?;
    vop(b, format!("v_div_scale_f32 v{t3}, vcc_lo, v{x}, v{t0}, v{x}"), &[t3], &[x, t0], &[])?;
    vop(b, format!("v_mul_f32_e32 v{t4}, v{t3}, v{t2}"), &[t4], &[t3, t2], &[])?;
    vop(b, format!("v_fma_f32 v{t5}, -v{t1}, v{t4}, v{t3}"), &[t5], &[t1, t4, t3], &[])?;
    vop(b, format!("v_fmac_f32_e32 v{t4}, v{t5}, v{t2}"), &[t4], &[t4, t5, t2], &[])?;
    vop(b, format!("v_fma_f32 v{t1}, -v{t1}, v{t4}, v{t3}"), &[t1], &[t1, t4, t3], &[])?;
    vop(b, format!("v_div_fmas_f32 v{t1}, v{t1}, v{t2}, v{t4}"), &[t1], &[t1, t2, t4], &[])?;
    vop(b, format!("v_div_fixup_f32 v{x}, v{t1}, v{t0}, v{x}"), &[x], &[t1, t0, x], &[])
}

/// `dq = fma_mix(d_f16, q, -0.0)` in place over the block's `d` register and
/// `q = f32(int8)` in place (`DAG.md` §2: `neg(0) op_sel_hi:[1,0,0]`).
fn dequant(b: &mut Builder, d: u8, q: u8) -> R {
    vop(b, format!("v_cvt_f32_i32_e32 v{q}, v{q}"), &[q], &[q], &[])?;
    vop(b, format!("v_fma_mix_f32 v{d}, v{d}, v{q}, neg(0) op_sel_hi:[1,0,0]"), &[d], &[d, q], &[])
}

// ===================================================================== k8
// VGPRs.
const V_TID: u8 = 0;
/// `tid & 31`.
const V_LANE: u8 = 1;
/// `(tid >> 5) * 4`: the wave's partial slot byte offset.
const V_WAVE4: u8 = 2;
/// Byte addresses of the shfl_down 8/4/2/1 bpermutes.
const V_ADDR: u8 = 3;
/// Activation byte offsets of the columns (`c*K*4 + wave*K/2 + lane*4`, +iteration).
const V_XOFF: u8 = 8;
/// Block-start byte offsets of the four rows (scale loads).
const V_WOFFD: u8 = 16;
/// The same plus the lane (int8 loads).
const V_WOFFQ: u8 = 20;
/// Four block slots of 16 VGPRs: q[4], d[4], x[8].
const V_SLOT: u8 = 32;
/// `sum[row][col]` at `+ 8*row + col`.
const V_SUM: u8 = 96;
const V_TMP: u8 = 128;
/// The eight wave partials of this thread's (row, column).
const V_PART: u8 = 136;
/// Epilogue scratch (six for SiLU, four for BF16).
const V_EP: u8 = 144;
/// Epilogue indices: row-in-group, column, LDS read address, global row, y byte offset.
const V_IR: u8 = 152;
const V_IC: u8 = 153;
const V_ILDS: u8 = 154;
const V_IROW: u8 = 155;
const V_IY: u8 = 156;
/// VGPR count of the k8 plan (`V_IY + 1`).
pub const K8_VGPRS: u16 = 157;

// SGPRs.
const S_KARG: u8 = 0;
const S_WG: u8 = 2;
/// A at s[4:5], x at s[6:7], y at s[8:9].
const S_A: u8 = 4;
const S_X: u8 = 6;
const S_Y: u8 = 8;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_B: u8 = 12;
const S_SCALE: u8 = 13;
const S_HC: u8 = 14;
const S_WAVE: u8 = 15;
const S_ROW0: u8 = 16;
const S_ROWBYTES: u8 = 17;
const S_SLICE: u8 = 18;
const S_XWAVE: u8 = 19;
/// `rowoff[r] = min(row0 + r, M-1) * rowbytes + slice` of the four rows.
const S_ROWOFF: u8 = 20;
const S_KX4: u8 = 24;
const S_ITERS: u8 = 25;
const S_WSTEP: u8 = 26;
const S_XSTEP: u8 = 27;
const S_T0: u8 = 28;
const S_MM1: u8 = 29;
/// Lane-0 exec mask.
const S_LANE0: u8 = 30;
/// Bytes of one 8-block iteration: weights 8*34, activations 8*128.
const K8_WSTEP: u32 = 8 * 34;
const K8_XSTEP: u32 = 8 * 128;

fn k8_q(slot: u8, r: u8) -> u8 { V_SLOT + 16 * slot + r }
fn k8_d(slot: u8, r: u8) -> u8 { V_SLOT + 16 * slot + 4 + r }
fn k8_x(slot: u8, c: u8) -> u8 { V_SLOT + 16 * slot + 8 + c }
fn k8_sum(r: u8, c: u8) -> u8 { V_SUM + 8 * r + c }

fn k8_plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(K8_VGPRS, 40)?;
    for (name, base, len) in [
        ("kernarg_ptr", S_KARG, 2), ("wg_x", S_WG, 1), ("a_x_ptrs", S_A, 4), ("y_ptr", S_Y, 2),
        ("m_k", S_M, 2), ("b_scale", S_B, 2), ("hc_act", S_HC, 1),
    ] { declare(&mut p, name, Kind::S, base, len)?; }
    for n in S_WAVE..=S_LANE0 { declare(&mut p, "scalar", Kind::S, n, 1)?; }
    for (name, base, len) in [
        ("tid", V_TID, 1), ("lane", V_LANE, 1), ("wave4", V_WAVE4, 1), ("bpermute_addr", V_ADDR, 4),
        ("x_offset", V_XOFF, 8), ("w_offset_d", V_WOFFD, 4), ("w_offset_q", V_WOFFQ, 4),
        ("block_slots", V_SLOT, 64), ("sums", V_SUM, 32), ("tree_tmp", V_TMP, 8),
        ("partials", V_PART, 8), ("epilogue_tmp", V_EP, 8), ("epilogue_index", V_IR, 5),
    ] { declare(&mut p, name, Kind::V, base, len)?; }
    Ok(p)
}

fn k8_spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "qwen4_lane_rows_q8".into(), variant: "k8_xbatch".into(), arch: Arch::Gfx1151, symbol: K8_SYMBOL.into(),
        kernargs: KernargLayout::new(44).pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly).pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("B", 32, 4, "by_value")
            .hidden("scale", 36, 4, "by_value").hidden("hc_act", 40, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 256,
        group_segment_fixed_size: K8_LDS_BYTES, wave32: true, cu_mode: false,
    }
}

fn k8() -> Result<Emitted, String> {
    let mut b = Builder::new(k8_spec(), k8_plan()?);
    b.enable_delay_alu();
    let slot = b.lds_slot("part", 0, K8_LDS_BYTES)?;
    k8_prologue(&mut b)?;
    for nb in 1..=MAX_COLUMNS {
        sop(&mut b, format!("s_cmp_eq_u32 s{S_B}, {nb}"), &[], &[S_B])?;
        op(&mut b, format!("s_cbranch_scc1 .Lq8k8_b{nb}"), &[], &[])?;
    }
    op(&mut b, "s_branch .Lq8k8_end", &[], &[])?;
    for nb in 1..=MAX_COLUMNS { k8_section(&mut b, nb, slot)?; }
    b.label(".Lq8k8_end")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

/// Kernarg loads, lane/wave constants, row and slice offsets; exits when the
/// workgroup has no row or `K < 2048`.
fn k8_prologue(b: &mut Builder) -> R {
    smem(b, S_A, 4, S_KARG, 0)?;
    smem(b, S_Y, 2, S_KARG, 0x10)?;
    smem(b, S_M, 2, S_KARG, 0x18)?;
    smem(b, S_B, 2, S_KARG, 0x20)?;
    smem(b, S_HC, 1, S_KARG, 0x28)?;
    vop(b, format!("v_and_b32_e32 v{V_LANE}, 31, v{V_TID}"), &[V_LANE], &[V_TID], &[])?;
    vop(b, format!("v_lshrrev_b32_e32 v{V_WAVE4}, 5, v{V_TID}"), &[V_WAVE4], &[V_TID], &[])?;
    op(b, format!("v_readfirstlane_b32 s{S_WAVE}, v{V_WAVE4}"), &[s(S_WAVE)], &[v(V_WAVE4)])?;
    vop(b, format!("v_lshlrev_b32_e32 v{V_WAVE4}, 2, v{V_WAVE4}"), &[V_WAVE4], &[V_WAVE4], &[])?;
    bpermute_addresses(b, V_ADDR, V_LANE)?;
    vop(b, format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{V_LANE}"), &[], &[V_LANE], &[])?;
    op(b, format!("s_mov_b32 s{S_LANE0}, vcc_lo"), &[s(S_LANE0)], &[])?;
    b.wait_all()?;
    sop(b, format!("s_lshl_b32 s{S_ROW0}, s{S_WG}, 2"), &[S_ROW0], &[S_WG])?;
    sop(b, format!("s_cmp_ge_i32 s{S_ROW0}, s{S_M}"), &[], &[S_ROW0, S_M])?;
    op(b, "s_cbranch_scc1 .Lq8k8_end", &[], &[])?;
    sop(b, format!("s_lshr_b32 s{S_ITERS}, s{S_K}, 11"), &[S_ITERS], &[S_K])?;
    sop(b, format!("s_cmp_eq_u32 s{S_ITERS}, 0"), &[], &[S_ITERS])?;
    op(b, "s_cbranch_scc1 .Lq8k8_end", &[], &[])?;
    // rowbytes = (K/32)*34; slice = wave*(K/256)*34; xwave = wave*K/2 (activation bytes); K*4.
    sop(b, format!("s_lshr_b32 s{S_ROWBYTES}, s{S_K}, 5"), &[S_ROWBYTES], &[S_K])?;
    sop(b, format!("s_mul_i32 s{S_ROWBYTES}, s{S_ROWBYTES}, 34"), &[S_ROWBYTES], &[S_ROWBYTES])?;
    sop(b, format!("s_lshr_b32 s{S_SLICE}, s{S_K}, 8"), &[S_SLICE], &[S_K])?;
    sop(b, format!("s_mul_i32 s{S_SLICE}, s{S_SLICE}, 34"), &[S_SLICE], &[S_SLICE])?;
    sop(b, format!("s_mul_i32 s{S_SLICE}, s{S_SLICE}, s{S_WAVE}"), &[S_SLICE], &[S_SLICE, S_WAVE])?;
    sop(b, format!("s_lshr_b32 s{S_XWAVE}, s{S_K}, 1"), &[S_XWAVE], &[S_K])?;
    sop(b, format!("s_mul_i32 s{S_XWAVE}, s{S_XWAVE}, s{S_WAVE}"), &[S_XWAVE], &[S_XWAVE, S_WAVE])?;
    sop(b, format!("s_lshl_b32 s{S_KX4}, s{S_K}, 2"), &[S_KX4], &[S_K])?;
    sop(b, format!("s_add_i32 s{S_MM1}, s{S_M}, -1"), &[S_MM1], &[S_M])?;
    for r in 0..4u8 {
        let ro = S_ROWOFF + r;
        sop(b, format!("s_add_i32 s{S_T0}, s{S_ROW0}, {r}"), &[S_T0], &[S_ROW0])?;
        sop(b, format!("s_min_i32 s{S_T0}, s{S_T0}, s{S_MM1}"), &[S_T0], &[S_T0, S_MM1])?;
        sop(b, format!("s_mul_i32 s{ro}, s{S_T0}, s{S_ROWBYTES}"), &[ro], &[S_T0, S_ROWBYTES])?;
        sop(b, format!("s_add_i32 s{ro}, s{ro}, s{S_SLICE}"), &[ro], &[ro, S_SLICE])?;
        vop(b, format!("v_mov_b32_e32 v{}, s{ro}", V_WOFFD + r), &[V_WOFFD + r], &[], &[ro])?;
        vop(b, format!("v_add_nc_u32_e32 v{}, s{ro}, v{V_LANE}", V_WOFFQ + r), &[V_WOFFQ + r], &[V_LANE], &[ro])?;
    }
    Ok(())
}

/// Loads of block `jb` (0..8 of the current base) into ring slot `slot`:
/// the four rows' int8 `q` and f16 `d`, then the columns' activations.
fn k8_issue(b: &mut Builder, nb: u8, slot: u8, jb: u32) -> R {
    for r in 0..4u8 { gload(b, "global_load_i8", k8_q(slot, r), V_WOFFQ + r, S_A, 34 * jb + 2)?; }
    for r in 0..4u8 { gload(b, "global_load_u16", k8_d(slot, r), V_WOFFD + r, S_A, 34 * jb)?; }
    for c in 0..nb { gload(b, "global_load_b32", k8_x(slot, c), V_XOFF + c, S_X, 128 * jb)?; }
    Ok(())
}

/// One block of all four rows against every column from ring slot `slot`:
/// `dq` once per row, then `sum[r][c] = fma(x[c], dq[r], sum[r][c])`.
fn k8_compute(b: &mut Builder, nb: u8, slot: u8) -> R {
    // At nb = 8 a slot's first int8 load has 63 younger loads in flight (three
    // 16-load blocks plus 15 of its own), which `vmcnt` (6 bits) cannot
    // express; drain the oldest two so the slot's waits fit the counter.
    if nb == 8 { b.wait(Counter::Vm, 62)?; }
    for r in 0..4u8 { vop(b, format!("v_cvt_f32_i32_e32 v{0}, v{0}", k8_q(slot, r)), &[k8_q(slot, r)], &[k8_q(slot, r)], &[])?; }
    for r in 0..4u8 {
        let (d, q) = (k8_d(slot, r), k8_q(slot, r));
        vop(b, format!("v_fma_mix_f32 v{d}, v{d}, v{q}, neg(0) op_sel_hi:[1,0,0]"), &[d], &[d, q], &[])?;
    }
    for c in 0..nb {
        for r in 0..4u8 {
            let (sum, x, d) = (k8_sum(r, c), k8_x(slot, c), k8_d(slot, r));
            vop(b, format!("v_fmac_f32_e32 v{sum}, v{x}, v{d}"), &[sum], &[sum, x, d], &[])?;
        }
    }
    Ok(())
}

fn k8_section(b: &mut Builder, nb: u8, slot: usize) -> R {
    let tag = format!(".Lq8k8_b{nb}");
    b.label(&tag)?;
    for c in 0..nb {
        sop(b, format!("s_mul_i32 s{S_T0}, s{S_KX4}, {c}"), &[S_T0], &[S_KX4])?;
        sop(b, format!("s_add_i32 s{S_T0}, s{S_T0}, s{S_XWAVE}"), &[S_T0], &[S_T0, S_XWAVE])?;
        op(b, format!("v_lshl_add_u32 v{}, v{V_LANE}, 2, s{S_T0}", V_XOFF + c), &[v(V_XOFF + c)], &[v(V_LANE), s(S_T0)])?;
    }
    for r in 0..4u8 { for c in 0..nb { vop(b, format!("v_mov_b32_e32 v{}, 0", k8_sum(r, c)), &[k8_sum(r, c)], &[], &[])?; } }
    for slot_i in 0..4u8 { k8_issue(b, nb, slot_i, u32::from(slot_i))?; }
    let head = format!("{tag}_loop");
    b.loop_(&head, |b| {
        // The loads of the next iteration's first four blocks advance by a full
        // iteration; in the last iteration they re-read the current blocks.
        sop(b, format!("s_cmp_gt_u32 s{S_ITERS}, 1"), &[], &[S_ITERS])?;
        sop(b, format!("s_cselect_b32 s{S_WSTEP}, {}, 0", lit(K8_WSTEP)), &[S_WSTEP], &[])?;
        sop(b, format!("s_cselect_b32 s{S_XSTEP}, {}, 0", lit(K8_XSTEP)), &[S_XSTEP], &[])?;
        for j in 0..8u8 {
            let ring = j % 4;
            k8_compute(b, nb, ring)?;
            if j < 4 {
                k8_issue(b, nb, ring, u32::from(j) + 4)?;
                if j == 3 {
                    for r in 0..4u8 {
                        vop(b, format!("v_add_nc_u32_e32 v{0}, s{S_WSTEP}, v{0}", V_WOFFD + r), &[V_WOFFD + r], &[V_WOFFD + r], &[S_WSTEP])?;
                        vop(b, format!("v_add_nc_u32_e32 v{0}, s{S_WSTEP}, v{0}", V_WOFFQ + r), &[V_WOFFQ + r], &[V_WOFFQ + r], &[S_WSTEP])?;
                    }
                    for c in 0..nb {
                        vop(b, format!("v_add_nc_u32_e32 v{0}, s{S_XSTEP}, v{0}", V_XOFF + c), &[V_XOFF + c], &[V_XOFF + c], &[S_XSTEP])?;
                    }
                }
            } else {
                k8_issue(b, nb, ring, u32::from(j) - 4)?;
            }
        }
        sop(b, format!("s_add_i32 s{S_ITERS}, s{S_ITERS}, -1"), &[S_ITERS], &[S_ITERS])?;
        sop(b, format!("s_cmp_lg_u32 s{S_ITERS}, 0"), &[], &[S_ITERS])?;
        op(b, format!("s_cbranch_scc1 {head}"), &[], &[])
    })?;
    b.wait_all()?;
    // Wave reduction of every (row, column) value, then lane 0 parks the wave's
    // partial: part[(row*8 + col)*8 + wave].
    let values: Vec<u8> = (0..4u8).flat_map(|r| (0..nb).map(move |c| k8_sum(r, c))).collect();
    lane_tree(b, &values, V_TMP, V_ADDR)?;
    op(b, format!("s_mov_b32 exec_lo, s{S_LANE0}"), &[], &[s(S_LANE0)])?;
    for r in 0..4u8 {
        for c in 0..nb { ds_st(b, slot, V_WAVE4, k8_sum(r, c), (u32::from(r) * 8 + u32::from(c)) * 32)?; }
    }
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
    // Thread t (< 4*nb) folds (row t / nb, column t % nb): the quotient is
    // (t * ceil(2^16 / nb)) >> 16, exact for t < 32.
    let magic = (65536 + u32::from(nb) - 1) / u32::from(nb);
    vop(b, format!("v_mul_u32_u24_e32 v{V_IR}, {}, v{V_TID}", lit(magic)), &[V_IR], &[V_TID], &[])?;
    vop(b, format!("v_lshrrev_b32_e32 v{V_IR}, 16, v{V_IR}"), &[V_IR], &[V_IR], &[])?;
    vop(b, format!("v_mul_u32_u24_e32 v{V_IC}, {nb}, v{V_IR}"), &[V_IC], &[V_IR], &[])?;
    vop(b, format!("v_sub_nc_u32_e32 v{V_IC}, v{V_TID}, v{V_IC}"), &[V_IC], &[V_TID, V_IC], &[])?;
    vop(b, format!("v_lshl_add_u32 v{V_ILDS}, v{V_IR}, 3, v{V_IC}"), &[V_ILDS], &[V_IR, V_IC], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{V_ILDS}, 5, v{V_ILDS}"), &[V_ILDS], &[V_ILDS], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{V_IROW}, s{S_ROW0}, v{V_IR}"), &[V_IROW], &[V_IR], &[S_ROW0])?;
    op(b, format!("v_mul_lo_u32 v{V_IY}, v{V_IC}, s{S_M}"), &[v(V_IY)], &[v(V_IC), s(S_M)])?;
    vop(b, format!("v_add_nc_u32_e32 v{V_IY}, v{V_IROW}, v{V_IY}"), &[V_IY], &[V_IROW, V_IY], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{V_IY}, 2, v{V_IY}"), &[V_IY], &[V_IY], &[])?;
    b.barrier(&[Transition::Ready(slot)])?;
    vop(b, format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{V_TID}", 4 * u32::from(nb)), &[], &[V_TID], &[])?;
    op(b, "s_mov_b32 exec_lo, vcc_lo", &[], &[])?;
    for half in 0..2u8 {
        b.ds_load(slot, Instruction::new(format!("ds_load_b128 {}, v{V_ILDS}{}", vrange(V_PART + 4 * half, 4), off(16 * u32::from(half))),
            vec![vr(V_PART + 4 * half, 4)], vec![v(V_ILDS)]).memory(MemoryClass::DsLoad))?;
    }
    b.wait_all()?;
    b.barrier(&[Transition::Retire(slot)])?;
    // Only wave 0 holds threads t < 4*nb.
    op(b, "s_cbranch_execz .Lq8k8_end", &[], &[])?;
    let p = |n: u8| V_PART + n;
    // ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
    for (dst, a, c) in [(0u8, 0u8, 1u8), (2, 2, 3), (4, 4, 5), (6, 6, 7), (0, 0, 2), (4, 4, 6), (0, 0, 4)] {
        vop(b, format!("v_add_f32_e32 v{}, v{}, v{}", p(dst), p(a), p(c)), &[p(dst)], &[p(a), p(c)], &[])?;
    }
    let x = p(0);
    let store = format!("{tag}_store");
    sop(b, format!("s_cmp_eq_u32 s{S_HC}, 0"), &[], &[S_HC])?;
    op(b, format!("s_cbranch_scc1 {store}"), &[], &[])?;
    let e = |n: u8| V_EP + n;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    vop(b, format!("v_mul_f32_e32 v{x}, s{S_SCALE}, v{x}"), &[x], &[x], &[S_SCALE])?;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    silu(b, x, [e(0), e(1), e(2), e(3), e(4), e(5)])?;
    bf16_round(b, x, [e(0), e(1), e(2), e(3)])?;
    b.label(&store)?;
    vop(b, format!("v_cmp_gt_i32_e32 vcc_lo, s{S_M}, v{V_IROW}"), &[], &[V_IROW], &[S_M])?;
    op(b, "s_and_b32 exec_lo, exec_lo, vcc_lo", &[], &[])?;
    mem(b, format!("global_store_b32 v{V_IY}, v{x}, s[{S_Y}:{}]", S_Y + 1), &[], &[v(V_IY), v(x), sr(S_Y, 2)], MemoryClass::VmemStore)?;
    b.wait_all()?;
    op(b, "s_branch .Lq8k8_end", &[], &[])
}

fn vrange(base: u8, len: u8) -> String { format!("v[{base}:{}]", u16::from(base) + u16::from(len) - 1) }

// ================================================================== k320
// VGPRs.
/// `(tid >> 5) * 384`: the wave's LDS region (scale reads).
const W_BASED: u8 = 2;
/// `+ lane`: the int8 reads.
const W_BASEQ: u8 = 3;
/// `+ lane*4`: the staging stores.
const W_STA: u8 = 4;
const W_LANE4: u8 = 5;
/// `min(lane, 20) * 4`: the third staging load.
const W_LANE4C: u8 = 6;
const W_ADDR: u8 = 7;
const W_XOFF: u8 = 12;
const W_YOFF: u8 = 20;
/// Global row ring: four rows of three dwords.
const W_G: u8 = 28;
const W_OFFA: u8 = 40;
const W_OFFB: u8 = 41;
const W_D: u8 = 44;
const W_Q: u8 = 54;
const W_TMP: u8 = 64;
/// Four `sum[col]` sets of eight, one per row in flight.
const W_SUM: u8 = 72;
/// `x[block][col]` at `+ 8*block + col`.
const W_X: u8 = 104;
/// VGPR count of the k320 plan.
pub const K320_VGPRS: u16 = 184;

// SGPRs (A at s[4:5], x at s[6:7], y at s[8:9]).
const T_M: u8 = 10;
const T_B: u8 = 11;
const T_WAVE: u8 = 12;
const T_ROW0: u8 = 13;
const T_MM1: u8 = 14;
const T_T0: u8 = 15;
const T_LANE0: u8 = 16;
/// `rowoff[q] = min(row0 + q, M-1) * 340` of the wave's eight rows.
const T_ROWOFF: u8 = 17;

fn k320_x(blk: u8, c: u8) -> u8 { W_X + 8 * blk + c }
fn k320_sum(set: u8, c: u8) -> u8 { W_SUM + 8 * set + c }

fn k320_plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(K320_VGPRS, 40)?;
    for (name, base, len) in [
        ("kernarg_ptr", S_KARG, 2), ("wg_x", S_WG, 1), ("a_x_ptrs", S_A, 4), ("y_ptr", S_Y, 2), ("m_b", T_M, 2),
    ] { declare(&mut p, name, Kind::S, base, len)?; }
    for n in T_WAVE..T_ROWOFF + 8 { declare(&mut p, "scalar", Kind::S, n, 1)?; }
    for (name, base, len) in [
        ("tid", V_TID, 1), ("lane", V_LANE, 1), ("lds_base_d", W_BASED, 1), ("lds_base_q", W_BASEQ, 1),
        ("lds_stage_addr", W_STA, 1), ("lane4", W_LANE4, 1), ("lane4_clamped", W_LANE4C, 1),
        ("bpermute_addr", W_ADDR, 4), ("x_offset", W_XOFF, 8), ("y_offset", W_YOFF, 8),
        ("row_ring", W_G, 12), ("row_offset_a", W_OFFA, 1), ("row_offset_b", W_OFFB, 1),
        ("scales", W_D, 10), ("weights", W_Q, 10), ("tree_tmp", W_TMP, 8), ("sums", W_SUM, 32),
        ("activations", W_X, 80),
    ] { declare(&mut p, name, Kind::V, base, len)?; }
    Ok(p)
}

fn k320_spec() -> KernelSpec {
    KernelSpec {
        kernel_id: "qwen4_lane_rows_q8".into(), variant: "k320_xbatch".into(), arch: Arch::Gfx1151, symbol: K320_SYMBOL.into(),
        kernargs: KernargLayout::new(32).pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly).pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value").hidden("B", 28, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 64,
        group_segment_fixed_size: K320_LDS_BYTES, wave32: true, cu_mode: false,
    }
}

fn k320() -> Result<Emitted, String> {
    let mut b = Builder::new(k320_spec(), k320_plan()?);
    b.enable_delay_alu();
    let slots = [b.lds_slot("rows0", 0, K320_SLOT)?, b.lds_slot("rows1", K320_SLOT, K320_SLOT)?];
    k320_prologue(&mut b)?;
    for nb in 1..=MAX_COLUMNS {
        sop(&mut b, format!("s_cmp_eq_u32 s{T_B}, {nb}"), &[], &[T_B])?;
        op(&mut b, format!("s_cbranch_scc1 .Lq8k320_b{nb}"), &[], &[])?;
    }
    op(&mut b, "s_branch .Lq8k320_end", &[], &[])?;
    for nb in 1..=MAX_COLUMNS { k320_section(&mut b, nb, slots)?; }
    b.label(".Lq8k320_end")?;
    op(&mut b, "s_endpgm", &[], &[])?;
    b.finish()
}

fn k320_prologue(b: &mut Builder) -> R {
    smem(b, S_A, 4, S_KARG, 0)?;
    smem(b, S_Y, 2, S_KARG, 0x10)?;
    smem(b, T_M, 2, S_KARG, 0x18)?;
    vop(b, format!("v_and_b32_e32 v{V_LANE}, 31, v{V_TID}"), &[V_LANE], &[V_TID], &[])?;
    vop(b, format!("v_lshrrev_b32_e32 v{W_BASED}, 5, v{V_TID}"), &[W_BASED], &[V_TID], &[])?;
    op(b, format!("v_readfirstlane_b32 s{T_WAVE}, v{W_BASED}"), &[s(T_WAVE)], &[v(W_BASED)])?;
    vop(b, format!("v_mul_u32_u24_e32 v{W_BASED}, {}, v{W_BASED}", lit(K320_WAVE_REGION)), &[W_BASED], &[W_BASED], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{W_BASEQ}, v{W_BASED}, v{V_LANE}"), &[W_BASEQ], &[W_BASED, V_LANE], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{W_LANE4}, 2, v{V_LANE}"), &[W_LANE4], &[V_LANE], &[])?;
    vop(b, format!("v_add_nc_u32_e32 v{W_STA}, v{W_BASED}, v{W_LANE4}"), &[W_STA], &[W_BASED, W_LANE4], &[])?;
    vop(b, format!("v_min_u32_e32 v{W_LANE4C}, 20, v{V_LANE}"), &[W_LANE4C], &[V_LANE], &[])?;
    vop(b, format!("v_lshlrev_b32_e32 v{W_LANE4C}, 2, v{W_LANE4C}"), &[W_LANE4C], &[W_LANE4C], &[])?;
    bpermute_addresses(b, W_ADDR, V_LANE)?;
    vop(b, format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{V_LANE}"), &[], &[V_LANE], &[])?;
    op(b, format!("s_mov_b32 s{T_LANE0}, vcc_lo"), &[s(T_LANE0)], &[])?;
    // `ds_load_u16_d16` keeps the high half of its destination.
    for blk in 0..10u8 { vop(b, format!("v_mov_b32_e32 v{}, 0", W_D + blk), &[W_D + blk], &[], &[])?; }
    b.wait_all()?;
    // The workgroup's first row must exist.
    sop(b, format!("s_lshl_b32 s{T_T0}, s{S_WG}, 4"), &[T_T0], &[S_WG])?;
    sop(b, format!("s_cmp_ge_i32 s{T_T0}, s{T_M}"), &[], &[T_T0, T_M])?;
    op(b, "s_cbranch_scc1 .Lq8k320_end", &[], &[])?;
    sop(b, format!("s_lshl_b32 s{T_ROW0}, s{T_WAVE}, 3"), &[T_ROW0], &[T_WAVE])?;
    sop(b, format!("s_add_i32 s{T_ROW0}, s{T_ROW0}, s{T_T0}"), &[T_ROW0], &[T_ROW0, T_T0])?;
    sop(b, format!("s_add_i32 s{T_MM1}, s{T_M}, -1"), &[T_MM1], &[T_M])?;
    for q in 0..8u8 {
        let ro = T_ROWOFF + q;
        sop(b, format!("s_add_i32 s{ro}, s{T_ROW0}, {q}"), &[ro], &[T_ROW0])?;
        sop(b, format!("s_min_i32 s{ro}, s{ro}, s{T_MM1}"), &[ro], &[ro, T_MM1])?;
        sop(b, format!("s_mul_i32 s{ro}, s{ro}, {}", lit(340)), &[ro], &[ro])?;
    }
    Ok(())
}

/// Global loads of row `q` (clamped) into ring slot `q % 4`: dwords lane,
/// lane+32 and min(lane, 20)+64.
fn k320_row_load(b: &mut Builder, q: u8) -> R {
    let g = W_G + 3 * (q % 4);
    let ro = T_ROWOFF + q;
    vop(b, format!("v_add_nc_u32_e32 v{W_OFFA}, s{ro}, v{W_LANE4}"), &[W_OFFA], &[W_LANE4], &[ro])?;
    vop(b, format!("v_add_nc_u32_e32 v{W_OFFB}, s{ro}, v{W_LANE4C}"), &[W_OFFB], &[W_LANE4C], &[ro])?;
    gload(b, "global_load_b32", g, W_OFFA, S_A, 0)?;
    gload(b, "global_load_b32", g + 1, W_OFFA, S_A, 128)?;
    gload(b, "global_load_b32", g + 2, W_OFFB, S_A, 256)
}

/// Stage row `q` from its ring slot into LDS slot `q % 2` (wave-local region).
fn k320_stage(b: &mut Builder, slots: [usize; 2], q: u8) -> R {
    let g = W_G + 3 * (q % 4);
    let base = u32::from(q % 2) * K320_SLOT;
    for dword in 0..3u8 { ds_st(b, slots[usize::from(q % 2)], W_STA, g + dword, base + 128 * u32::from(dword))?; }
    Ok(())
}

/// Row `r` from LDS slot `r % 2` against every column into sum set `r % 4`;
/// then the lane tree and lane 0's stores of valid rows.
fn k320_row(b: &mut Builder, nb: u8, slot: usize, r: u8) -> R {
    let base = u32::from(r % 2) * K320_SLOT;
    let set = r % 4;
    for blk in 0..10u8 {
        ds_ld(b, slot, "ds_load_u16_d16", W_D + blk, W_BASED, base + 34 * u32::from(blk), true)?;
        ds_ld(b, slot, "ds_load_i8", W_Q + blk, W_BASEQ, base + 34 * u32::from(blk) + 2, false)?;
    }
    for blk in 0..10u8 {
        dequant(b, W_D + blk, W_Q + blk)?;
        for c in 0..nb {
            let (sum, x, d) = (k320_sum(set, c), k320_x(blk, c), W_D + blk);
            if blk == 0 {
                vop(b, format!("v_fma_f32 v{sum}, v{x}, v{d}, 0"), &[sum], &[x, d], &[])?;
            } else {
                vop(b, format!("v_fmac_f32_e32 v{sum}, v{x}, v{d}"), &[sum], &[sum, x, d], &[])?;
            }
        }
    }
    Ok(())
}

fn k320_section(b: &mut Builder, nb: u8, slots: [usize; 2]) -> R {
    let tag = format!(".Lq8k320_b{nb}");
    b.label(&tag)?;
    for c in 0..nb {
        if c == 0 {
            vop(b, format!("v_mov_b32_e32 v{W_XOFF}, v{W_LANE4}"), &[W_XOFF], &[W_LANE4], &[])?;
        } else {
            vop(b, format!("v_add_nc_u32_e32 v{}, {}, v{W_LANE4}", W_XOFF + c, lit(1280 * u32::from(c))), &[W_XOFF + c], &[W_LANE4], &[])?;
        }
        // y byte offset of (column c, this wave's first row).
        sop(b, format!("s_mul_i32 s{T_T0}, s{T_M}, {c}"), &[T_T0], &[T_M])?;
        sop(b, format!("s_add_i32 s{T_T0}, s{T_T0}, s{T_ROW0}"), &[T_T0], &[T_T0, T_ROW0])?;
        sop(b, format!("s_lshl_b32 s{T_T0}, s{T_T0}, 2"), &[T_T0], &[T_T0])?;
        vop(b, format!("v_mov_b32_e32 v{}, s{T_T0}", W_YOFF + c), &[W_YOFF + c], &[], &[T_T0])?;
    }
    for q in 0..4u8 { k320_row_load(b, q)?; }
    for blk in 0..10u8 {
        for c in 0..nb { gload(b, "global_load_b32", k320_x(blk, c), W_XOFF + c, S_X, 128 * u32::from(blk))?; }
    }
    k320_stage(b, slots, 0)?;
    b.barrier(&[Transition::Ready(slots[0])])?;
    for r in 0..8u8 {
        let par = usize::from(r % 2);
        if r + 1 < 8 { k320_stage(b, slots, r + 1)?; }
        if r + 4 < 8 { k320_row_load(b, r + 4)?; }
        // Row r reuses the sum set that row r-4 stored. Those stores may be
        // skipped (row >= M) and retire out of order, so only vscnt(0)
        // releases their sources; four rows later they have long retired.
        if r >= 4 { b.wait(Counter::Vs, 0)?; }
        k320_row(b, nb, slots[par], r)?;
        let mut transitions = vec![Transition::Retire(slots[par])];
        if r + 1 < 8 { transitions.push(Transition::Ready(slots[1 - par])); }
        b.barrier(&transitions)?;
        let set = r % 4;
        let values: Vec<u8> = (0..nb).map(|c| k320_sum(set, c)).collect();
        lane_tree(b, &values, W_TMP, W_ADDR)?;
        let skip = format!("{tag}_r{r}_skip");
        sop(b, format!("s_add_i32 s{T_T0}, s{T_ROW0}, {r}"), &[T_T0], &[T_ROW0])?;
        sop(b, format!("s_cmp_ge_i32 s{T_T0}, s{T_M}"), &[], &[T_T0, T_M])?;
        op(b, format!("s_cbranch_scc1 {skip}"), &[], &[])?;
        op(b, format!("s_mov_b32 exec_lo, s{T_LANE0}"), &[], &[s(T_LANE0)])?;
        for c in 0..nb {
            mem(b, format!("global_store_b32 v{}, v{}, s[{S_Y}:{}]{}", W_YOFF + c, k320_sum(set, c), S_Y + 1, off(4 * u32::from(r))),
                &[], &[v(W_YOFF + c), v(k320_sum(set, c)), sr(S_Y, 2)], MemoryClass::VmemStore)?;
        }
        op(b, "s_mov_b32 exec_lo, -1", &[], &[])?;
        b.label(&skip)?;
    }
    b.wait_all()?;
    op(b, "s_branch .Lq8k320_end", &[], &[])
}

/// Both Q8 row exports of the lane step, in table order.
pub fn build_gfx1151() -> Result<Vec<Emitted>, String> {
    Ok(vec![k8()?, k320()?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_plans_fit_the_ring_layout() {
        for slot in 0..4u8 {
            assert_eq!(k8_q(slot, 3) + 1, k8_d(slot, 0));
            assert_eq!(k8_d(slot, 3) + 1, k8_x(slot, 0));
            assert!(u16::from(k8_x(slot, 7)) < u16::from(V_SUM));
        }
        assert_eq!(k8_sum(3, 7) + 1, V_TMP);
        assert_eq!(u16::from(V_IY) + 1, K8_VGPRS);
        assert!(u16::from(k320_x(9, 7)) + 1 == K320_VGPRS);
        assert_eq!(W_Q, W_D + 10);
        assert_eq!(W_SUM, W_TMP + 8);
        assert_eq!(W_X, k320_sum(3, 7) + 1);
        // Eight k8 columns keep 64 loads in flight, the vmcnt field's reach.
        assert_eq!(4 * (4 + 4 + 8), 64);
        // k320: the third staging store of lane 31 still ends inside the wave region.
        assert!(256 + 4 * 31 + 4 <= K320_WAVE_REGION);
        assert_eq!(K320_LDS_BYTES, 1536);
        assert_eq!(K8_LDS_BYTES, 1024);
    }

    #[test]
    fn exports_build_with_the_frozen_budgets() {
        let emitted = build_gfx1151().expect("q8 lane rows");
        assert_eq!(emitted.len(), 2);
        let (k8, k320) = (&emitted[0], &emitted[1]);
        assert_eq!(k8.shape.next_free_vgpr, K8_VGPRS);
        assert_eq!(k320.shape.next_free_vgpr, K320_VGPRS);
        assert!(k8.shape.next_free_sgpr <= 40 && k320.shape.next_free_sgpr <= 40);
        assert!(k8.s_text.contains(&format!(".amdhsa_group_segment_fixed_size {K8_LDS_BYTES}")));
        assert!(k320.s_text.contains(&format!(".amdhsa_group_segment_fixed_size {K320_LDS_BYTES}")));
        assert_eq!(k8.proof.lds_slots.len(), 1);
        assert_eq!(k320.proof.lds_slots.len(), 2);
        for (e, symbol) in [(k8, K8_SYMBOL), (k320, K320_SYMBOL)] {
            assert!(e.s_text.contains(&format!("{symbol}:")));
            // The DAG nodes: dq = fma_mix(d, q, -0.0); sum = fma(x, dq, sum); the lane tree.
            assert!(e.s_text.contains("v_fma_mix_f32") && e.s_text.contains("neg(0) op_sel_hi:[1,0,0]"));
            assert!(e.s_text.contains("v_fmac_f32_e32") && e.s_text.contains("v_cvt_f32_i32_e32"));
            assert!(e.s_text.contains("ds_swizzle_b32") && e.s_text.contains("ds_bpermute_b32"));
            assert!(e.s_text.contains("s_barrier"));
        }
        // Only k8 carries the epilogue.
        assert!(k8.s_text.contains("v_div_fixup_f32") && k8.s_text.contains("v_div_fmas_f32") && k8.s_text.contains("v_exp_f32_e32"));
        assert!(!k320.s_text.contains("v_div_fixup_f32"));
        assert_eq!(k8.proof.loop_fixpoints.len(), usize::from(MAX_COLUMNS));
        assert!(k320.proof.loop_fixpoints.is_empty());
    }

    #[cfg(feature = "toolchain")]
    #[test]
    fn q8_lane_rows_m7() {
        let dir = std::path::PathBuf::from("/home/kaden/qcal/release-0.4.2/fn-batch/s7/pm");
        std::fs::create_dir_all(&dir).unwrap();
        for (emitted, symbol) in build_gfx1151().expect("q8 lane rows").into_iter().zip([K8_SYMBOL, K320_SYMBOL]) {
            let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1151).expect("native assemble");
            std::fs::write(dir.join(format!("{symbol}.test.s")), &emitted.s_text).unwrap();
            let co = dir.join(format!("{symbol}.test.co"));
            std::fs::write(&co, &elf).unwrap();
            let report = crate::pm_check::m7(&co, "gfx1151", symbol).expect("M7");
            println!("{symbol}: {report}");
            assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
        }
    }
}
