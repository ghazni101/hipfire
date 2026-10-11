// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 multi-column row kernels of the Flash-Next lane step, MQ6 half:
//! `gemv_mq6g256v2_xbatch_pm` and `gemv_mq6g256v2_x4_xbatch_pm`.
//!
//! Every column `c` of `y[c*M + row]` is byte-identical to the 1-row
//! incumbent `gemv_mq6g256v2` on `x[c*K..]` (and to the incumbent rows kernel's
//! column `c`); `fn-batch/s7/DAG.md` §1 is the arithmetic spec. Each
//! workgroup (one wave32) owns four weight rows and keeps one register
//! ring of activation loads that all four rows share, which is what bounds
//! the incumbents (they re-read the activations per weight row).
//!
//! ABI (frozen by `fn-batch/S7-CONTRACT.md`), x is F32 `[B][K]`, y is F32
//! `[B][M]`, `B` in 1..=8, `K % 256 == 0`, block 32, no LDS, no scratch:
//! - `gemv_mq6g256v2_xbatch_pm`: `A@0 x@8 y@16 M@24 K@28 B@32` (36 B), grid
//!   `ceil(M/4)`; workgroup `w` owns rows `4w..4w+3` (loads clamp to `M-1`,
//!   only valid rows are stored).
//! - `gemv_mq6g256v2_x4_xbatch_pm`: `A0..A3@0..24 x@32 y0..y3@40..64
//!   M0..M3@72..84 K@88 B@92` (96 B), grid `sum ceil(Mi/4)`; workgroup `w`
//!   maps to matrix `i` and local row `4*(w - sum_{j<i} ceil(Mj/4))`, then runs
//!   the same body (`Mi <= 0` owns no workgroups).
//!
//! Per-matrix byte sizes must stay below 4 GiB: each row's byte offset is a
//! 32-bit buffer soffset against a raw, unbounded buffer descriptor.
//!
//! Arithmetic per (row, column) (`DAG.md` §1):
//! - Group `g` (200 B: half headers `h0` for lanes 0-15 / `h1` for lanes 16-31
//!   in bytes 0..7, then six payload bytes per lane at `8 + 6*lane`): lane
//!   `l` covers weights `8l..8l+7`. The payload is read as a dword at `8+6l`
//!   plus a zero-extended word at `12+6l` (`p` = bytes b0..b3, `u` = bytes
//!   b4,b5), and `q0 = p & 63`, `q1..q4 = bfe(p, 6/12/18/24, 6)`,
//!   `q5 = alignbit(u, p, 30) & 63`, `q6 = bfe(u, 4, 6)`, `q7 = bfe(u, 10, 6)`
//!   are the unsigned values of `DAG.md` §1 (`b0&63`, `(b0>>6)|((b1&15)<<2)`,
//!   ...); `w_i = fma_mix(sc_f16, f32(q_i), zp_f16)` with `op_sel:[0,0,1]
//!   op_sel_hi:[1,0,1]` (one rounding), `f32(q_i)` from `v_cvt_f32_ubyte0`.
//! - The group dot is `d = x1*w1` (`v_mul`) and then `fma(x0,w0,d)`,
//!   `fma(x2,w2,d)` .. `fma(x7,w7,d)` in that fixed order (`v_fmac`, one
//!   rounding each); multiplicand order is `w, x` for every column.
//! - Group `g` adds into stream `g % 4` as `a_s = a_s + d` in increasing
//!   `g` (quads in a loop, then up to three tail groups into streams
//!   0..2); the four streams start at +0 and fold as `(a0 + a1) + (a3 + a2)`.
//! - Lane tree: `ds_swizzle_b32` offset 16 (`"1pppp"`: lanes 16..31 read
//!   themselves) then `ds_bpermute_b32` at lane + 8/4/2/1 (own lane when
//!   out of range), `own + other` at every level; lane 0 stores.
//!
//! Instruction selection: columns are paired two at a time so the two halves
//! of a `v_dual_*` packet read different weight rows (src0 banks differ) and
//! different activation slots (src1 banks differ: ring slots sit 9 VGPRs
//! apart), which is what gfx11's VOPD rules require; a lone odd column runs
//! plain VALU. Every VOPD pair is validated by `vopd::validate_pair` while
//! the kernel is built.
use super::common::smem_off;
use crate::insn::{Instruction, MemoryClass};
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::vopd::{Operand, VopdF32, VopdOp};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub const SYMBOL: &str = "gemv_mq6g256v2_xbatch_pm";
pub const SYMBOL_X4: &str = "gemv_mq6g256v2_x4_xbatch_pm";
pub const MAX_COLUMNS: u8 = 8;

/// MQ6 group bytes and the four rows per workgroup.
const GROUP: u32 = 200;
const VGPRS: u16 = 256;
const SGPRS: u16 = 88;
/// Group-dot term order: x1 first (`v_mul`), then x0, x2 .. x7.
const TERMS: [u8; 8] = [1, 0, 2, 3, 4, 5, 6, 7];

// VGPRs.
const LANE: u8 = 0;
/// `(lane >> 4) * 4`: the lane's half-header byte offset.
const HDR_OFF: u8 = 1;
/// `lane * 6`: the lane's payload byte offset (plus 8 immediate).
const PK_OFF: u8 = 2;
/// `lane * 32`: the lane's eight activations.
const X_OFF: u8 = 3;
const SHUF: u8 = 4;
/// Raw loads of (buffer, row): header, payload dword, payload u16 (3 each;
/// two buffers alternate between the groups of a body).
fn raw(buf: u8, r: u8, k: u8) -> u8 { 8 + 12 * buf + 3 * r + k }
/// Dequantized weights of row r (bases differ by 2 mod 4 within a row pair).
const W: [u8; 4] = [32, 42, 52, 62];
/// Group-dot registers (column slot, row): even rows even, odd rows odd, and
/// `DOT[s][r] % 4 == r`, which makes every VOPD pairing of the add phase legal.
const DOT: [[u8; 4]; 2] = [[40, 41, 50, 51], [60, 61, 70, 71]];
const ACC0: u16 = 72;
/// Stream accumulator (column c, stream s, row r): parity `r % 2`, bank `r % 4`.
fn acc(c: u8, s: u8, r: u8) -> u8 { (ACC0 + 16 * u16::from(c) + 4 * u16::from(s) + u16::from(r)) as u8 }
fn ring_base(nb: u8) -> u16 { ACC0 + 16 * u16::from(nb) }
/// Activation ring slots (8 VGPRs at a 9-VGPR pitch) of the `nb`-column
/// path: every free register after the accumulators, at most eight. The
/// pitch makes consecutive items' slots differ in bank (mod 4) even across
/// the wrap, so any two consecutive items form a legal VOPD src1 pair.
fn slots(nb: u8) -> u16 { ((VGPRS - ring_base(nb) - 8) / 9 + 1).min(8) }
fn ring(nb: u8, item: u16) -> u8 { (ring_base(nb) + 9 * (item % slots(nb))) as u8 }
/// One past the highest VGPR any column count touches.
fn vgpr_high() -> u16 { (1..=MAX_COLUMNS).map(|nb| ring_base(nb) + 9 * (slots(nb) - 1) + 8).max().unwrap_or(0) }
// Epilogue registers (raw loads, weights and the ring are dead by then).
fn red_tmp(c: u8, r: u8) -> u8 { 8 + 4 * c + r }
fn y_addr(c: u8) -> u8 { 40 + c }

// SGPRs. s0:1 kernarg, s2 workgroup id x (`user_sgpr_count = 2`).
const S_WG: u8 = 2;
const S_B: u8 = 3;
const S_A: u8 = 4;
const S_X: u8 = 6;
const S_Y: u8 = 8;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_ROW0: u8 = 12;
const S_NV: u8 = 13;
const S_GROUPS: u8 = 14;
const S_STRIDE: u8 = 15;
/// Weight soffset of row r (row offset plus the current quad base).
fn s_row(r: u8) -> u8 { 16 + r }
const S_ASRD: u8 = 20;
const S_XSRD: u8 = 24;
/// Activation soffset of column c (column offset plus the current quad base).
fn s_xcol(c: u8) -> u8 { 28 + c }
const S_QUADS: u8 = 36;
const S_TAIL: u8 = 37;
const S_KX4: u8 = 38;
const S_MX4: u8 = 39;
const S_T0: u8 = 40;
const S_T1: u8 = 41;
const S_ROW0X4: u8 = 42;
// x4 routing (kernargs of the four matrices, loaded whole).
const R_A: u8 = 48;
const R_X: u8 = 56;
const R_Y: u8 = 64;
const R_M: u8 = 72;
const R_K: u8 = 76;
const R_B: u8 = 77;
/// Cumulative block ends of matrices 0..3 (s78..s81) and a temporary.
const R_END: u8 = 78;
const R_TMP: u8 = 82;

/// Both exports, in contract order.
pub fn build_gfx1151() -> Result<Vec<Emitted>, String> { Ok(vec![kernel(false)?, kernel(true)?]) }

/// Declare registers `0..end` in aligned 8/4/2/1-register pieces.
fn declare(regs: &mut RegPlan, kind: Kind, end: u16) -> Result<(), String> {
    let (prefix, mut at) = (if kind == Kind::V { "v" } else { "s" }, 0u16);
    while at < end {
        let piece = [8u16, 4, 2, 1].into_iter().find(|p| at % p == 0 && at + p <= end).unwrap_or(1);
        regs.add_range(&format!("{prefix}{at}"), kind, at as u8, piece as u8, Live::Whole)?;
        at += piece;
    }
    Ok(())
}

fn kernel(x4: bool) -> Result<Emitted, String> {
    let mut regs = RegPlan::new(VGPRS, SGPRS)?;
    declare(&mut regs, Kind::V, vgpr_high())?;
    declare(&mut regs, Kind::S, if x4 { u16::from(R_TMP) + 1 } else { u16::from(S_ROW0X4) + 1 })?;
    let (symbol, variant, prefix, kernargs) = if x4 {
        let mut k = KernargLayout::new(96);
        for (i, name) in ["A0", "A1", "A2", "A3"].into_iter().enumerate() { k = k.pointer_access(name, 8 * i as u32, Access::ReadOnly); }
        k = k.pointer_access("x", 32, Access::ReadOnly);
        for (i, name) in ["y0", "y1", "y2", "y3"].into_iter().enumerate() { k = k.pointer_access(name, 40 + 8 * i as u32, Access::WriteOnly); }
        for (i, name) in ["M0", "M1", "M2", "M3"].into_iter().enumerate() { k = k.hidden(name, 72 + 4 * i as u32, 4, "by_value"); }
        (SYMBOL_X4, "x4_xbatch_r4_b8", "mq6x4", k.hidden("K", 88, 4, "by_value").hidden("B", 92, 4, "by_value"))
    } else {
        (SYMBOL, "xbatch_r4_b8", "mq6", KernargLayout::new(36).pointer_access("A", 0, Access::ReadOnly)
            .pointer_access("x", 8, Access::ReadOnly).pointer_access("y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("B", 32, 4, "by_value"))
    };
    let mut b = Builder::new(KernelSpec {
        kernel_id: "qwen4_lane_rows_mq6".into(), variant: variant.into(), arch: Arch::Gfx1151, symbol: symbol.into(), kernargs,
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 32,
        group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, regs);
    b.enable_delay_alu();
    if x4 { route(&mut b, prefix)?; } else { select(&mut b, prefix)?; }
    setup(&mut b)?;
    // One specialized path per column count; each section ends in `s_endpgm`,
    // so every branch stays inside one section.
    for nb in 1..=MAX_COLUMNS {
        salu(&mut b, &format!("s_cmp_eq_u32 s{S_B}, {nb}"), &[], &[S_B])?;
        branch(&mut b, &format!("s_cbranch_scc0 .L{prefix}_skip{nb}"))?;
        section(&mut b, prefix, nb)?;
        b.label(&format!(".L{prefix}_skip{nb}"))?;
    }
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// Single-matrix workgroup row block and the early exit.
fn select(b: &mut Builder, prefix: &str) -> Result<(), String> {
    smem_load(b, S_M, 2, 0x18)?;
    salu(b, &format!("s_lshl_b32 s{S_ROW0}, s{S_WG}, 2"), &[S_ROW0], &[S_WG])?;
    b.wait_all()?;
    salu(b, &format!("s_cmp_ge_i32 s{S_ROW0}, s{S_M}"), &[], &[S_ROW0, S_M])?;
    exit_unless(b, prefix, "go")?;
    smem_load(b, S_A, 4, 0)?;
    smem_load(b, S_Y, 2, 0x10)?;
    smem_load(b, S_B, 1, 0x20)
}

/// `s_cbranch_scc0` over a lone `s_endpgm`: leaves the wave when SCC is set.
fn exit_unless(b: &mut Builder, prefix: &str, name: &str) -> Result<(), String> {
    branch(b, &format!("s_cbranch_scc0 .L{prefix}_{name}"))?;
    branch(b, "s_endpgm")?;
    b.label(&format!(".L{prefix}_{name}"))
}

/// x4: workgroup `w` -> matrix `i` (the smallest `i` with `w < end_i`, `end_i`
/// the prefix sum of `ceil(max(Mi,0)/4)`), its local row block and its A/y/M.
fn route(b: &mut Builder, prefix: &str) -> Result<(), String> {
    smem_load(b, R_M, 4, 0x48)?;
    smem_load(b, R_K, 2, 0x58)?;
    smem_load(b, R_A, 8, 0)?;
    smem_load(b, R_X, 2, 0x20)?;
    smem_load(b, R_Y, 8, 0x28)?;
    b.wait_all()?;
    for i in 0..4u8 {
        let (m, end) = (R_M + i, R_END + i);
        salu(b, &format!("s_max_i32 s{end}, s{m}, 0"), &[end], &[m])?;
        salu(b, &format!("s_add_i32 s{end}, s{end}, 3"), &[end], &[end])?;
        salu(b, &format!("s_lshr_b32 s{end}, s{end}, 2"), &[end], &[end])?;
        if i > 0 { salu(b, &format!("s_add_i32 s{end}, s{end}, s{}", end - 1), &[end], &[end, end - 1])?; }
    }
    salu(b, &format!("s_cmp_ge_u32 s{S_WG}, s{}", R_END + 3), &[], &[S_WG, R_END + 3])?;
    exit_unless(b, prefix, "route")?;
    // Matrix 3, then matrices 2, 1, 0 override it while `w < end_i`.
    pair(b, "s_mov_b64", S_A, R_A + 6)?;
    pair(b, "s_mov_b64", S_Y, R_Y + 6)?;
    salu(b, &format!("s_mov_b32 s{S_M}, s{}", R_M + 3), &[S_M], &[R_M + 3])?;
    salu(b, &format!("s_sub_i32 s{S_ROW0}, s{S_WG}, s{}", R_END + 2), &[S_ROW0], &[S_WG, R_END + 2])?;
    for i in (0..3u8).rev() {
        let local = if i == 0 { S_WG } else {
            salu(b, &format!("s_sub_i32 s{R_TMP}, s{S_WG}, s{}", R_END + i - 1), &[R_TMP], &[S_WG, R_END + i - 1])?;
            R_TMP
        };
        salu(b, &format!("s_cmp_lt_u32 s{S_WG}, s{}", R_END + i), &[], &[S_WG, R_END + i])?;
        pair(b, "s_cselect_b64", S_A, R_A + 2 * i)?;
        pair(b, "s_cselect_b64", S_Y, R_Y + 2 * i)?;
        salu(b, &format!("s_cselect_b32 s{S_M}, s{}, s{S_M}", R_M + i), &[S_M], &[R_M + i, S_M])?;
        salu(b, &format!("s_cselect_b32 s{S_ROW0}, s{local}, s{S_ROW0}"), &[S_ROW0], &[local, S_ROW0])?;
    }
    salu(b, &format!("s_lshl_b32 s{S_ROW0}, s{S_ROW0}, 2"), &[S_ROW0], &[S_ROW0])?;
    pair(b, "s_mov_b64", S_X, R_X)?;
    salu(b, &format!("s_mov_b32 s{S_K}, s{R_K}"), &[S_K], &[R_K])?;
    salu(b, &format!("s_mov_b32 s{S_B}, s{R_B}"), &[S_B], &[R_B])
}

/// `op dst_pair, src_pair` over 64-bit SGPR pairs; `s_cselect_b64` also keeps
/// the pair it does not select.
fn pair(b: &mut Builder, op: &str, dst: u8, src: u8) -> Result<(), String> {
    let text = if op == "s_cselect_b64" { format!("{op} s[{dst}:{}], s[{src}:{}], s[{dst}:{}]", dst + 1, src + 1, dst + 1) }
        else { format!("{op} s[{dst}:{}], s[{src}:{}]", dst + 1, src + 1) };
    let uses: &[u8] = if op == "s_cselect_b64" { &[src, src + 1, dst, dst + 1] } else { &[src, src + 1] };
    salu(b, &text, &[dst, dst + 1], uses)
}

/// Shared scalar/vector setup: valid-row count, row soffsets, lane offsets
/// and the two raw buffer resources.
fn setup(b: &mut Builder) -> Result<(), String> {
    salu(b, &format!("s_sub_i32 s{S_NV}, s{S_M}, s{S_ROW0}"), &[S_NV], &[S_M, S_ROW0])?;
    salu(b, &format!("s_min_i32 s{S_NV}, s{S_NV}, 4"), &[S_NV], &[S_NV])?;
    salu(b, &format!("s_add_i32 s{S_T0}, s{S_M}, -1"), &[S_T0], &[S_M])?;
    salu(b, &format!("s_lshr_b32 s{S_GROUPS}, s{S_K}, 8"), &[S_GROUPS], &[S_K])?;
    salu(b, &format!("s_mul_i32 s{S_STRIDE}, s{S_GROUPS}, {GROUP:#x}"), &[S_STRIDE], &[S_GROUPS])?;
    for r in 0..4u8 {
        salu(b, &format!("s_add_i32 s{S_T1}, s{S_ROW0}, {r}"), &[S_T1], &[S_ROW0])?;
        salu(b, &format!("s_min_i32 s{S_T1}, s{S_T1}, s{S_T0}"), &[S_T1], &[S_T1, S_T0])?;
        salu(b, &format!("s_mul_i32 s{}, s{S_T1}, s{S_STRIDE}", s_row(r)), &[s_row(r)], &[S_T1, S_STRIDE])?;
    }
    salu(b, &format!("s_lshl_b32 s{S_KX4}, s{S_K}, 2"), &[S_KX4], &[S_K])?;
    salu(b, &format!("s_lshl_b32 s{S_MX4}, s{S_M}, 2"), &[S_MX4], &[S_M])?;
    salu(b, &format!("s_lshl_b32 s{S_ROW0X4}, s{S_ROW0}, 2"), &[S_ROW0X4], &[S_ROW0])?;
    for c in 0..MAX_COLUMNS {
        salu(b, &format!("s_mul_i32 s{}, s{S_KX4}, {c}", s_xcol(c)), &[s_xcol(c)], &[S_KX4])?;
    }
    valu(b, &format!("v_lshrrev_b32_e32 v{HDR_OFF}, 4, v{LANE}"), &[HDR_OFF], &[LANE], &[])?;
    valu(b, &format!("v_lshlrev_b32_e32 v{HDR_OFF}, 2, v{HDR_OFF}"), &[HDR_OFF], &[HDR_OFF], &[])?;
    valu(b, &format!("v_mul_u32_u24_e32 v{PK_OFF}, 6, v{LANE}"), &[PK_OFF], &[LANE], &[])?;
    valu(b, &format!("v_lshlrev_b32_e32 v{X_OFF}, 5, v{LANE}"), &[X_OFF], &[LANE], &[])?;
    b.wait_all()?;
    // Raw buffer resources (stride 0, unbounded records) over A and x.
    for (res, ptr) in [(S_ASRD, S_A), (S_XSRD, S_X)] {
        salu(b, &format!("s_mov_b32 s{res}, s{ptr}"), &[res], &[ptr])?;
        salu(b, &format!("s_and_b32 s{}, s{}, 0xffff", res + 1, ptr + 1), &[res + 1], &[ptr + 1])?;
        salu(b, &format!("s_mov_b32 s{}, -1", res + 2), &[res + 2], &[])?;
        salu(b, &format!("s_mov_b32 s{}, {:#x}", res + 3, super::common::SRD_WORD3), &[res + 3], &[])?;
    }
    Ok(())
}

/// The complete kernel body for `nb` columns (ends in `s_endpgm`).
fn section(b: &mut Builder, prefix: &str, nb: u8) -> Result<(), String> {
    for c in 0..nb {
        for s in 0..4 {
            for r in 0..4 {
                let a = acc(c, s, r);
                valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
    }
    salu(b, &format!("s_lshr_b32 s{S_QUADS}, s{S_GROUPS}, 2"), &[S_QUADS], &[S_GROUPS])?;
    salu(b, &format!("s_and_b32 s{S_TAIL}, s{S_GROUPS}, 3"), &[S_TAIL], &[S_GROUPS])?;
    salu(b, &format!("s_cmp_eq_u32 s{S_QUADS}, 0"), &[], &[S_QUADS])?;
    branch(b, &format!("s_cbranch_scc1 .L{prefix}_b{nb}_tail"))?;
    b.loop_(&format!(".L{prefix}_b{nb}_quad"), |b| {
        body(b, nb, &[0, 1, 2, 3])?;
        for r in 0..4 {
            salu(b, &format!("s_add_i32 s{0}, s{0}, {1:#x}", s_row(r), 4 * GROUP), &[s_row(r)], &[s_row(r)])?;
        }
        for c in 0..nb {
            salu(b, &format!("s_add_i32 s{0}, s{0}, 0x1000", s_xcol(c)), &[s_xcol(c)], &[s_xcol(c)])?;
        }
        salu(b, &format!("s_add_i32 s{S_QUADS}, s{S_QUADS}, -1"), &[S_QUADS], &[S_QUADS])?;
        salu(b, &format!("s_cmp_lg_u32 s{S_QUADS}, 0"), &[], &[S_QUADS])?;
        branch(b, &format!("s_cbranch_scc1 .L{prefix}_b{nb}_quad"))
    })?;
    b.label(&format!(".L{prefix}_b{nb}_tail"))?;
    // Tail group t (at the post-quad bases) accumulates into stream t.
    for t in 0..3u8 {
        salu(b, &format!("s_cmp_gt_u32 s{S_TAIL}, {t}"), &[], &[S_TAIL])?;
        branch(b, &format!("s_cbranch_scc0 .L{prefix}_b{nb}_fold"))?;
        body(b, nb, &[t])?;
    }
    b.label(&format!(".L{prefix}_b{nb}_fold"))?;
    fold(b, nb)?;
    reduce(b, nb)?;
    store(b, prefix, nb)
}

/// Groups `js` (slot = stream = immediate index) of the four rows against
/// every column. The first two groups' weights and the first activations
/// issue up front; the weights of group `i + 2` reuse the raw registers
/// group `i` just dequantized, and activations of later items stream
/// through the ring as earlier items retire.
fn body(b: &mut Builder, nb: u8, js: &[u8]) -> Result<(), String> {
    let items = js.len() as u16 * u16::from(nb);
    let ring_slots = slots(nb);
    weight_loads(b, js[0], 0)?;
    let mut issued = 0u16;
    while issued < ring_slots.min(items) {
        x_load(b, nb, js, issued)?;
        issued += 1;
    }
    if js.len() > 1 { weight_loads(b, js[1], 1)?; }
    for (index, &j) in js.iter().enumerate() {
        let buf = (index % 2) as u8;
        dequant(b, buf)?;
        if let Some(&next) = js.get(index + 2) { weight_loads(b, next, buf)?; }
        let mut c = 0u8;
        while c < nb {
            let item = index as u16 * u16::from(nb) + u16::from(c);
            while issued < (item + ring_slots).min(items) {
                x_load(b, nb, js, issued)?;
                issued += 1;
            }
            let cols: Vec<(u8, u8)> = (c..nb.min(c + 2)).map(|col| (col, ring(nb, item + u16::from(col - c)))).collect();
            columns(b, j, &cols)?;
            c += 2;
        }
    }
    b.wait_all()
}

/// Payload dwords, payload u16s and lane-selected half headers of group
/// slot `j`, all four rows, into raw buffer `buf`.
fn weight_loads(b: &mut Builder, j: u8, buf: u8) -> Result<(), String> {
    let off = u32::from(j) * GROUP;
    let srd = [S_ASRD, S_ASRD + 1, S_ASRD + 2, S_ASRD + 3];
    for r in 0..4u8 {
        let p = raw(buf, r, 1);
        let mut scalar = srd.to_vec();
        scalar.push(s_row(r));
        vmem(b, &format!("buffer_load_b32 v{p}, v{PK_OFF}, s[{S_ASRD}:{}], s{} offen offset:{}", S_ASRD + 3, s_row(r), off + 8), &[p], &[PK_OFF], &scalar, false)?;
    }
    for r in 0..4u8 {
        let u = raw(buf, r, 2);
        let mut scalar = srd.to_vec();
        scalar.push(s_row(r));
        vmem(b, &format!("buffer_load_u16 v{u}, v{PK_OFF}, s[{S_ASRD}:{}], s{} offen offset:{}", S_ASRD + 3, s_row(r), off + 12), &[u], &[PK_OFF], &scalar, false)?;
    }
    for r in 0..4u8 {
        let h = raw(buf, r, 0);
        let mut scalar = srd.to_vec();
        scalar.push(s_row(r));
        let suffix = if off == 0 { String::new() } else { format!(" offset:{off}") };
        vmem(b, &format!("buffer_load_b32 v{h}, v{HDR_OFF}, s[{S_ASRD}:{}], s{} offen{suffix}", S_ASRD + 3, s_row(r)), &[h], &[HDR_OFF], &scalar, false)?;
    }
    Ok(())
}

/// The eight activations of item `item` (group `js[item / nb]`, column
/// `item % nb`) into its ring slot.
fn x_load(b: &mut Builder, nb: u8, js: &[u8], item: u16) -> Result<(), String> {
    let j = js[usize::from(item / u16::from(nb))];
    let c = (item % u16::from(nb)) as u8;
    let x = ring(nb, item);
    for half in 0..2u8 {
        let off = u32::from(j) * 1024 + u32::from(half) * 16;
        let suffix = if off == 0 { String::new() } else { format!(" offset:{off}") };
        let d = x + 4 * half;
        vmem(b, &format!("buffer_load_b128 v[{d}:{}], v{X_OFF}, s[{S_XSRD}:{}], s{} offen{suffix}", d + 3, S_XSRD + 3, s_xcol(c)),
            &[d, d + 1, d + 2, d + 3], &[X_OFF], &[S_XSRD, S_XSRD + 1, S_XSRD + 2, S_XSRD + 3, s_xcol(c)], false)?;
    }
    Ok(())
}

/// `w[r][n] = fma_mix(sc, f32(q_n), zp)` for the four rows of the group in
/// raw buffer `buf`, extracting `q0..q7` from the payload dword `p` and word
/// `u` exactly as `DAG.md` §1 (unsigned 6-bit fields; `q5` straddles `p` and `u`).
fn dequant(b: &mut Builder, buf: u8) -> Result<(), String> {
    let rows = |f: &mut dyn FnMut(u8, u8, u8, u8) -> Result<(), String>| -> Result<(), String> {
        for r in 0..4u8 { f(W[usize::from(r)], raw(buf, r, 0), raw(buf, r, 1), raw(buf, r, 2))?; }
        Ok(())
    };
    rows(&mut |w, _, p, _| valu(b, &format!("v_and_b32_e32 v{w}, 63, v{p}"), &[w], &[p], &[]))?;
    for (n, lsb) in [(1u8, 6u32), (2, 12), (3, 18), (4, 24)] {
        rows(&mut |w, _, p, _| valu(b, &format!("v_bfe_u32 v{}, v{p}, {lsb}, 6", w + n), &[w + n], &[p], &[]))?;
    }
    rows(&mut |w, _, p, u| valu(b, &format!("v_alignbit_b32 v{}, v{u}, v{p}, 30", w + 5), &[w + 5], &[u, p], &[]))?;
    rows(&mut |w, _, _, _| valu(b, &format!("v_and_b32_e32 v{0}, 63, v{0}", w + 5), &[w + 5], &[w + 5], &[]))?;
    rows(&mut |w, _, _, u| valu(b, &format!("v_bfe_u32 v{}, v{u}, 4, 6", w + 6), &[w + 6], &[u], &[]))?;
    rows(&mut |w, _, _, u| valu(b, &format!("v_bfe_u32 v{}, v{u}, 10, 6", w + 7), &[w + 7], &[u], &[]))?;
    for n in 0..8u8 {
        rows(&mut |w, _, _, _| valu(b, &format!("v_cvt_f32_ubyte0_e32 v{0}, v{0}", w + n), &[w + n], &[w + n], &[]))?;
    }
    for n in 0..8u8 {
        rows(&mut |w, h, _, _| {
            let d = w + n;
            valu(b, &format!("v_fma_mix_f32 v{d}, v{h}, v{d}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"), &[d], &[h, d], &[])
        })?;
    }
    Ok(())
}

/// Dot the four dequantized rows against one or two columns (each `(column,
/// ring slot)`) and add the dots into stream `s`. Terms run x1, x0, x2..x7.
/// Two columns: each packet pairs (row rx, column 0) with (row ry, column 1)
/// for `rx != ry` of one row pair, so the src0 banks (weights) differ by
/// construction, the src1 banks (ring slots) differ by the slot pitch and the
/// destinations alternate parity by row.
fn columns(b: &mut Builder, s: u8, cols: &[(u8, u8)]) -> Result<(), String> {
    let row_pairs = [(0u8, 1u8), (2, 3)];
    for (term, n) in TERMS.into_iter().enumerate() {
        let op = if term == 0 { VopdF32::Mul } else { VopdF32::Fmac };
        match cols {
            [(_, x0), (_, x1)] => {
                for (ra, rb) in row_pairs {
                    for (rx, ry) in [(ra, rb), (rb, ra)] {
                        let (wx, wy) = (W[usize::from(rx)] + n, W[usize::from(ry)] + n);
                        b.vopd(
                            VopdOp { op, dst: DOT[0][usize::from(rx)], src0: Operand::V(wx), src1: x0 + n },
                            VopdOp { op, dst: DOT[1][usize::from(ry)], src0: Operand::V(wy), src1: x1 + n },
                        )?;
                    }
                }
            }
            [(_, x)] => {
                for r in 0..4u8 {
                    let (d, w) = (DOT[0][usize::from(r)], W[usize::from(r)] + n);
                    if term == 0 {
                        valu(b, &format!("v_mul_f32_e32 v{d}, v{w}, v{}", x + n), &[d], &[w, x + n], &[])?;
                    } else {
                        valu(b, &format!("v_fmac_f32_e32 v{d}, v{w}, v{}", x + n), &[d], &[d, w, x + n], &[])?;
                    }
                }
            }
            _ => return Err("columns takes one or two columns".into()),
        }
    }
    match cols {
        [(c0, _), (c1, _)] => {
            for (ra, rb) in row_pairs {
                for (rx, ry) in [(ra, rb), (rb, ra)] {
                    let (ax, ay) = (acc(*c0, s, rx), acc(*c1, s, ry));
                    b.vopd(
                        VopdOp { op: VopdF32::Add, dst: ax, src0: Operand::V(ax), src1: DOT[0][usize::from(rx)] },
                        VopdOp { op: VopdF32::Add, dst: ay, src0: Operand::V(ay), src1: DOT[1][usize::from(ry)] },
                    )?;
                }
            }
        }
        [(c, _)] => {
            for r in 0..4u8 {
                let (a, d) = (acc(*c, s, r), DOT[0][usize::from(r)]);
                valu(b, &format!("v_add_f32_e32 v{a}, v{a}, v{d}"), &[a], &[a, d], &[])?;
            }
        }
        _ => unreachable!("checked above"),
    }
    Ok(())
}

/// `(a0 + a1) + (a3 + a2)` per column and row, into stream 0.
fn fold(b: &mut Builder, nb: u8) -> Result<(), String> {
    for (dst, other) in [(0u8, 1u8), (3, 2)] {
        for c in 0..nb {
            for r in 0..4u8 {
                let (x, y) = (acc(c, dst, r), acc(c, other, r));
                valu(b, &format!("v_add_f32_e32 v{x}, v{x}, v{y}"), &[x], &[x, y], &[])?;
            }
        }
    }
    for c in 0..nb {
        for r in 0..4u8 {
            let (x, y) = (acc(c, 0, r), acc(c, 3, r));
            valu(b, &format!("v_add_f32_e32 v{x}, v{x}, v{y}"), &[x], &[x, y], &[])?;
        }
    }
    Ok(())
}

/// Frozen shfl_down 16/8/4/2/1 tree, batched over every (column, row) value.
fn reduce(b: &mut Builder, nb: u8) -> Result<(), String> {
    let values: Vec<(u8, u8)> = (0..nb).flat_map(|c| (0..4u8).map(move |r| (c, r))).collect();
    for &(c, r) in &values {
        let (v, t) = (acc(c, 0, r), red_tmp(c, r));
        b.ds_crosslane(Instruction::new(format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
            refs(Kind::V, &[t]), refs(Kind::V, &[v])).memory(MemoryClass::DsLoad))?;
    }
    for &(c, r) in &values {
        let (v, t) = (acc(c, 0, r), red_tmp(c, r));
        valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
    }
    for offset in [8u8, 4, 2, 1] {
        // Lanes whose partner is out of range read themselves.
        valu(b, &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - offset), &[], &[LANE], &[])?;
        valu(b, &format!("v_cndmask_b32_e64 v{SHUF}, 0, {offset}, vcc_lo"), &[SHUF], &[], &[])?;
        valu(b, &format!("v_add_lshl_u32 v{SHUF}, v{SHUF}, v{LANE}, 2"), &[SHUF], &[SHUF, LANE], &[])?;
        for &(c, r) in &values {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            b.ds_crosslane(Instruction::new(format!("ds_bpermute_b32 v{t}, v{SHUF}, v{v}"),
                refs(Kind::V, &[t]), refs(Kind::V, &[SHUF, v])).memory(MemoryClass::DsLoad))?;
        }
        for &(c, r) in &values {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
        }
    }
    Ok(())
}

/// Lane 0 stores `y[c*M + row0 + r]` for every column and each valid row.
fn store(b: &mut Builder, prefix: &str, nb: u8) -> Result<(), String> {
    for c in 0..nb {
        let a = y_addr(c);
        if c == 0 {
            valu(b, &format!("v_mov_b32_e32 v{a}, s{S_ROW0X4}"), &[a], &[], &[S_ROW0X4])?;
        } else {
            salu(b, &format!("s_mul_i32 s{S_T0}, s{S_MX4}, {c}"), &[S_T0], &[S_MX4])?;
            salu(b, &format!("s_add_i32 s{S_T0}, s{S_T0}, s{S_ROW0X4}"), &[S_T0], &[S_T0, S_ROW0X4])?;
            valu(b, &format!("v_mov_b32_e32 v{a}, s{S_T0}"), &[a], &[], &[S_T0])?;
        }
    }
    valu(b, &format!("v_cmpx_eq_u32_e32 0, v{LANE}"), &[], &[LANE], &[])?;
    for r in 0..4u8 {
        if r > 0 {
            salu(b, &format!("s_cmp_le_i32 s{S_NV}, {r}"), &[], &[S_NV])?;
            branch(b, &format!("s_cbranch_scc1 .L{prefix}_b{nb}_stored"))?;
        }
        for c in 0..nb {
            let (a, v) = (y_addr(c), acc(c, 0, r));
            let suffix = if r == 0 { String::new() } else { format!(" offset:{}", 4 * r) };
            vmem(b, &format!("global_store_b32 v{a}, v{v}, s[{S_Y}:{}]{suffix}", S_Y + 1), &[], &[a, v], &[S_Y, S_Y + 1], true)?;
        }
    }
    b.label(&format!(".L{prefix}_b{nb}_stored"))?;
    b.wait_all()?;
    branch(b, "s_endpgm")
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<RegRef> {
    indices.iter().map(|&base| RegRef { kind, base, len: 1 }).collect()
}
fn salu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(text, refs(Kind::S, defs), refs(Kind::S, uses)))
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses))
}
/// `s_load_b{32,64,128,256}` of `len` dwords at kernarg byte `off`.
fn smem_load(b: &mut Builder, dst: u8, len: u8, off: u32) -> Result<(), String> {
    let name = match len { 1 => "s_load_b32", 2 => "s_load_b64", 4 => "s_load_b128", 8 => "s_load_b256", _ => return Err("SMEM width".into()) };
    let target = if len == 1 { format!("s{dst}") } else { format!("s[{dst}:{}]", dst + len - 1) };
    let defs: Vec<u8> = (dst..dst + len).collect();
    let offset = smem_off(b.spec.arch, off);
    b.push(Instruction::new(format!("{name} {target}, s[0:1], {offset}"), refs(Kind::S, &defs), refs(Kind::S, &[0, 1]))
        .memory(MemoryClass::SmemLoad))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses)
        .memory(if store { MemoryClass::VmemStore } else { MemoryClass::VmemLoad }))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> { b.control(Instruction::new(text, vec![], vec![])) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_plan_fits() {
        for nb in 1..=MAX_COLUMNS {
            let k = slots(nb);
            assert!(k >= 2, "nb={nb}");
            // Consecutive items (including the wrap) sit in different banks.
            for item in 0..2 * k {
                assert_ne!(ring(nb, item) % 4, ring(nb, item + 1) % 4, "nb={nb} item={item}");
            }
            // Slots stay disjoint, above the accumulators and inside the file.
            assert_eq!(ring_base(nb), u16::from(acc(nb - 1, 3, 3)) + 1, "nb={nb}");
            assert!(ring_base(nb) + 9 * (k - 1) + 8 <= VGPRS, "nb={nb}");
            // Outstanding VMEM loads (two weight groups + the full ring) fit the 6-bit counter.
            assert!(2 * 12 + 2 * k <= 63, "nb={nb}");
            // Epilogue scratch lives in dead raw/weight registers below the accumulators.
            assert!(u16::from(red_tmp(nb - 1, 3)) < u16::from(y_addr(0)), "nb={nb}");
            assert!(u16::from(y_addr(nb - 1)) < ACC0, "nb={nb}");
        }
        assert!(vgpr_high() <= VGPRS);
        // Raw buffers, weights and dot registers tile v8..v71 without overlap.
        assert_eq!(raw(1, 3, 2), 31);
        assert_eq!(W[0] + 8, DOT[0][0]);
        assert_eq!(u16::from(DOT[1][3]) + 1, ACC0);
    }

    #[test]
    fn dot_registers_follow_the_vopd_rules() {
        for s in 0..2 {
            for r in 0..4usize {
                assert_eq!(usize::from(DOT[s][r]) % 4, r);
                assert_eq!(usize::from(DOT[s][r]) % 2, r % 2);
            }
        }
        for (a, b) in [(0usize, 1usize), (2, 3)] {
            assert_ne!(W[a] % 4, W[b] % 4);
        }
    }

    #[test]
    fn builds_both_exports_for_every_column_count() {
        let emitted = build_gfx1151().expect("mq6 lane rows");
        assert_eq!(emitted.len(), 2);
        for (e, symbol) in emitted.iter().zip([SYMBOL, SYMBOL_X4]) {
            assert!(e.s_text.contains(&format!("{symbol}:")), "{symbol}");
            assert!(e.s_text.contains("v_dual_fmac_f32"), "{symbol}");
            assert!(e.s_text.contains("ds_swizzle_b32") && e.s_text.contains("ds_bpermute_b32"), "{symbol}");
            assert!(!e.s_text.contains("ttmp") && !e.s_text.contains("s_wait_alu"), "{symbol}");
        }
    }

    #[cfg(feature = "toolchain")]
    #[test]
    fn lane_rows_mq6_m7() {
        let dir = std::path::PathBuf::from("/home/kaden/qcal/release-0.4.2/fn-batch/s7/pm");
        std::fs::create_dir_all(&dir).unwrap();
        for (emitted, symbol) in build_gfx1151().expect("mq6 lane rows").into_iter().zip([SYMBOL, SYMBOL_X4]) {
            // The two columns of the first packet: x1 mul against rows (0, 1) of slots 0 and 1.
            let packet = format!("v_dual_mul_f32 v{}, v{}, v{} :: v_dual_mul_f32 v{}, v{}, v{}",
                DOT[0][0], W[0] + 1, ring(2, 0) + 1, DOT[1][1], W[1] + 1, ring(2, 1) + 1);
            assert!(emitted.s_text.contains(&packet), "{symbol}: {packet}");
            let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1151).expect("native assemble");
            std::fs::write(dir.join(format!("{symbol}.test.s")), &emitted.s_text).unwrap();
            let co = dir.join(format!("{symbol}.test.co"));
            std::fs::write(&co, &elf).unwrap();
            let report = crate::pm_check::m7(&co, "gfx1151", symbol).expect("M7");
            println!("{symbol}: {report}");
            assert_eq!(report["obligations"], serde_json::json!({}), "{symbol}: {report}");
        }
    }
}
