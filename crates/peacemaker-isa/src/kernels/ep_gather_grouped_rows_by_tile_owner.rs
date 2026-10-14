// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 tile pull of the Flash-Next
//! owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`). For every live
//! tile whose expert another rank owns, copy the tile's sixteen grouped rows of
//! the OWNER's `y_down_grouped` (peer-mapped device memory) onto the same rows
//! of this rank's buffer. All ranks run the same stable grouping, so a tile has
//! the same row offset on every rank. Rows of tiles this rank owns, and of the
//! padding tail, are never written. Pure `buffer_load_b128` /
//! `buffer_store_b128`: the copy is bit-exact for every pattern (+-0, NaN
//! payloads, denormals). The kernel is model-free (no Qwen4 type or constant).
//!
//! Two certified specializations of one schedule ([`Variant`]); the launcher
//! (`Gpu::ep_gather_grouped_rows_by_tile_owner`) picks one from `ranks`:
//!
//! * [`Variant::Ep2`], `ep_gather_grouped_rows_by_tile_owner_pm_<arch>`: the
//!   original two-rank kernel (48-byte kernarg). tile_ids `*i32` @0, owner_table
//!   `*i32` @8 (one owner rank in `{0, 1}` per expert), peer_src `*u8` @16 (the
//!   one peer's `y_down_grouped` base), dst `*u8` @24 (this rank's), n_experts
//!   i32 @32, rank i32 @36, tile_u4 i32 @40, tiles i32 @44. Owner semantics: the
//!   only other valid owner is the peer, so `owner != rank` pulls from the one
//!   peer base with no owner range check (an owner outside `{0, 1}` is a caller
//!   contract violation, never produced).
//! * [`Variant::N4`], `ep_gather_grouped_rows_by_tile_owner_n4_pm_<arch>`: the
//!   rank-generic kernel (72-byte kernarg, serves 2 and 4 ranks). tile_ids @0,
//!   owner_table @8, src0..src3 `*u8` @16/@24/@32/@40 (the bases of rank 0..3's
//!   `y_down_grouped`; the entry of this rank may be the destination itself and
//!   the entries of ranks the launch does not use are never read), dst @48,
//!   n_experts @56, rank @60, tile_u4 @64, tiles @68. The owner must be a rank
//!   in `[0, 4)`, otherwise the tile is not pulled.
//!
//! Both: tile_u4 = uint4 per tile = 16 rows x row_bytes / 16. Grid `[tiles, 1,
//! 1]` (the padded tile capacity), block 256, no LDS, no barrier, no scratch.
//! Contract (the caller's checks): `tile_u4 % 1024 == 0`, `tiles * tile_u4 * 16
//! <= 2^32 - 1`, `tile_ids` holds `tiles` words, `owner_table` holds `n_experts`
//! words, every source slot an owner can name is mapped and the tile range is
//! inside it and inside dst.
//!
//! Law (the same predicate as the mask kernel's complement):
//! ```text
//!   pull(t) = 0 <= tile_ids[t] < n_experts && owner_table[tile_ids[t]] != rank
//!             [&& owner_table[tile_ids[t]] < 4]                          (N4 only)
//!   pull(t) => dst[t * 16 rows ..+ 16 rows] = src[owner_table[tile_ids[t]]][same bytes]
//! ```
//!
//! Schedule (shared copy loop and workgroup structure): workgroup `t` reads its
//! id with an SMEM load and returns (workgroup-uniform) when it is the padding
//! sentinel / out of range, the owner is this rank (N4: or not a rank in
//! `[0, 4)`), before any VMEM access. A pulling workgroup walks the tile `1024 *
//! 16` bytes per trip through two raw buffer descriptors (the owner's virtual
//! address is just a descriptor base; N4 loads it with one SMEM load from the
//! kernarg segment at `16 + 8 * owner`, EP2 already holds the one peer base):
//! each lane keeps four `buffer_load_b128` in flight at byte `16 tid + {0, 4096,
//! 8192, 12288}` of the trip, then stores them to the same offsets; the two bases
//! advance by 16384 bytes per trip (`tile_u4 / 1024` trips).
use super::common::{add64, add64_imm, bload, bstore_b128, op, s, smem, sop, srd_tail, v};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, insn::Instruction, plan::Access, reg::{Kind, Live}};
use peacemaker_author::{End, Gfx1201, Scc, WgUniform, Workgroup};

type Wg<'b> = Workgroup<'b, Gfx1201, Builder>;
type R = Result<(), String>;

/// Threads per workgroup.
pub const BLOCK: u32 = 256;
/// Loads in flight per lane and trip.
pub const UNROLL: u32 = 4;
/// uint4 moved per workgroup trip; `tile_u4` is a multiple of it.
pub const TRIP_U4: u32 = BLOCK * UNROLL;
/// Source slots of the rank-generic variant (ranks 0..4); an EP2 launch passes two live bases.
pub const MAX_RANKS: u32 = 4;

/// The two certified specializations of the tile pull.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Two ranks, one peer base (48-byte kernarg): the original kernel.
    Ep2,
    /// Two or four ranks, four source slots (72-byte kernarg).
    N4,
}

impl Variant {
    /// Every variant, in emission order.
    pub const ALL: [Variant; 2] = [Variant::Ep2, Variant::N4];

    /// The `peacemaker-isa emit --kernel` name.
    pub fn kernel_id(self) -> &'static str {
        match self {
            Variant::Ep2 => "ep_gather_grouped_rows_by_tile_owner_pm",
            Variant::N4 => "ep_gather_grouped_rows_by_tile_owner_n4_pm",
        }
    }

    /// Kernarg bytes.
    pub fn kernarg_bytes(self) -> u32 {
        match self {
            Variant::Ep2 => 48,
            Variant::N4 => 72,
        }
    }
}

pub fn symbol(variant: Variant, arch: Arch) -> String { format!("{}_{}", variant.kernel_id(), arch.name()) }
pub fn module(variant: Variant, arch: Arch) -> String { symbol(variant, arch) }

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// EP2: kernarg words 0..8 -> s8..s15 (tile_ids 8:9, table 10:11, peer 12:13, dst 14:15), words 8..12 -> s16..s19.
// N4: kernarg words 0..4 -> s8..s11 (tile_ids 8:9, table 10:11), dst (@48) -> s12:13, words @56..72 -> s16..s19.
// N4 loads the per-tile source base from kernarg[16 + 8 * owner] into SRC; KSLOT is that kernarg address.
const TIDS: u8 = 8; const TABLE: u8 = 10; const PEER: u8 = 12; const DST_EP2: u8 = 14;
const DST_N4: u8 = 12; const KSLOT: u8 = 14;
const NEXP: u8 = 16; const RANK: u8 = 17; const TU4: u8 = 18; const TILES: u8 = 19;
const WGX: u8 = 20; const ID: u8 = 21; const OWN: u8 = 22; const T0: u8 = 23; const T64: u8 = 24;
const TRIPS: u8 = 26; const BYTES: u8 = 27; const SRC: u8 = 28;
// Source / destination raw buffer descriptors of the trip loop (4-aligned SGPR quads).
const SRD_SRC: u8 = 32; const SRD_OUT: u8 = 36;
const S_END: u8 = 40;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0; const VOFF: u8 = 1;
/// Four 16-byte loads per lane and trip (each quad inside one 8-VGPR plan block).
const DATA: u8 = 8;
const V_END: u8 = 24;

fn kernargs(variant: Variant) -> KernargLayout {
    let layout = KernargLayout::new(variant.kernarg_bytes())
        .pointer_access("tile_ids", 0, Access::ReadOnly)
        .pointer_access("owner_table", 8, Access::ReadOnly);
    match variant {
        Variant::Ep2 => layout
            .pointer_access("peer_src", 16, Access::ReadOnly)
            .pointer_access("dst", 24, Access::WriteOnly)
            .hidden("n_experts", 32, 4, "by_value")
            .hidden("rank", 36, 4, "by_value")
            .hidden("tile_u4", 40, 4, "by_value")
            .hidden("tiles", 44, 4, "by_value"),
        Variant::N4 => layout
            .pointer_access("src0", 16, Access::ReadOnly)
            .pointer_access("src1", 24, Access::ReadOnly)
            .pointer_access("src2", 32, Access::ReadOnly)
            .pointer_access("src3", 40, Access::ReadOnly)
            .pointer_access("dst", 48, Access::WriteOnly)
            .hidden("n_experts", 56, 4, "by_value")
            .hidden("rank", 60, 4, "by_value")
            .hidden("tile_u4", 64, 4, "by_value")
            .hidden("tiles", 68, 4, "by_value"),
    }
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..S_END).step_by(8) { p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?; }
    for base in (0u8..V_END).step_by(8) { p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?; }
    Ok(p)
}

/// The builder-emitted kernel (gfx1201 only; every other target is refused).
pub fn emit(variant: Variant, arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 { return Err(format!("{}: built for gfx1201 only, not {}", variant.kernel_id(), arch.name())) }
    let kspec = KernelSpec {
        kernel_id: variant.kernel_id().into(), variant: "pull".into(), arch, symbol: symbol(variant, arch), kernargs: kernargs(variant),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: BLOCK as u16, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?, variant)?;
    b.finish()
}

/// The single-symbol module `<symbol>` of one variant.
pub fn emit_module(variant: Variant, arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(variant, arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(variant, arch))?;
    Ok((emitted, text, proof))
}

// ---- helpers ----------------------------------------------------------------
fn cmp_wg(wg: &mut Wg, text: String, uses: &[crate::reg::RegRef]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}

// ---- kernel -----------------------------------------------------------------
fn kernel(wg: &mut Wg, variant: Variant) -> R {
    let end = wg.exit(".Lpull_end")?;
    prologue(wg, &end, variant)?;
    copy_tile(wg)?;
    wg.end(end)
}

/// Kernarg loads and the workgroup-uniform exits: tile out of range, tile
/// without a live expert, tile owned by this rank (N4: or by no rank of `[0, 4)`).
/// Leaves the running source and destination pairs at the tile's first byte and
/// `TRIPS` = `tile_u4 / 1024`.
fn prologue(wg: &mut Wg, end: &End, variant: Variant) -> R {
    let dst = match variant { Variant::Ep2 => DST_EP2, Variant::N4 => DST_N4 };
    let b = wg.isa();
    match variant {
        Variant::Ep2 => {
            smem(b, TIDS, 8, KARG, 0x00)?;
            smem(b, NEXP, 4, KARG, 0x20)?;
        }
        Variant::N4 => {
            smem(b, TIDS, 4, KARG, 0x00)?;
            smem(b, DST_N4, 2, KARG, 0x30)?;
            smem(b, NEXP, 4, KARG, 0x38)?;
        }
    }
    // Workgroup id x (gfx12: ttmp9) = the tile.
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    let outside = cmp_wg(wg, format!("s_cmp_ge_u32 s{WGX}, s{TILES}"), &[s(WGX), s(TILES)])?;
    wg.exit_if(outside, end)?;
    // id = tile_ids[tile]; the unsigned compare also rejects the -1 sentinel.
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGX}, 2"), &[T0], &[WGX])?;
    add64(b, T64, TIDS, T0)?;
    smem(b, ID, 1, T64, 0)?;
    let dead = cmp_wg(wg, format!("s_cmp_ge_u32 s{ID}, s{NEXP}"), &[s(ID), s(NEXP)])?;
    wg.exit_if(dead, end)?;
    // owner = owner_table[id].
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{ID}, 2"), &[T0], &[ID])?;
    add64(b, T64, TABLE, T0)?;
    smem(b, OWN, 1, T64, 0)?;
    let mine = cmp_wg(wg, format!("s_cmp_eq_u32 s{OWN}, s{RANK}"), &[s(OWN), s(RANK)])?;
    wg.exit_if(mine, end)?;
    if variant == Variant::N4 {
        // The owner is a rank in [0, 4): anything else (never produced by the caller) is not pulled,
        // so the source-base lookup below stays inside the four source slots.
        let range = cmp_wg(wg, format!("s_cmp_ge_u32 s{OWN}, {MAX_RANKS}"), &[s(OWN)])?;
        wg.exit_if(range, end)?;
    }
    // Tile byte offset = tile * tile_u4 * 16 (fits 32 bits by contract); trips; the
    // source / destination raw descriptors (base = pointer + tile offset, num_records =
    // the tile's bytes). N4: the source base is kernarg[16 + 8 * owner] (one SMEM load at a
    // scalar-computed kernarg address); EP2: the one peer base is already in PEER.
    let b = wg.isa();
    let src = match variant {
        Variant::Ep2 => PEER,
        Variant::N4 => {
            sop(b, format!("s_lshl_b32 s{T0}, s{OWN}, 3"), &[T0], &[OWN])?;
            add64(b, KSLOT, KARG, T0)?;
            smem(b, SRC, 2, KSLOT, 16)?;
            SRC
        }
    };
    sop(b, format!("s_lshl_b32 s{BYTES}, s{TU4}, 4"), &[BYTES], &[TU4])?;
    sop(b, format!("s_mul_i32 s{T0}, s{WGX}, s{BYTES}"), &[T0], &[WGX, BYTES])?;
    add64(b, SRD_SRC, src, T0)?;
    srd_tail(b, SRD_SRC, Some(BYTES))?;
    add64(b, SRD_OUT, dst, T0)?;
    srd_tail(b, SRD_OUT, Some(BYTES))?;
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{TU4}, 10"), &[TRIPS], &[TU4])?;
    op(b, format!("v_lshlrev_b32_e32 v{VOFF}, 4, v{TID}"), &[v(VOFF)], &[v(TID)])
}

/// One trip: four peer loads in flight, four local stores, advance the two descriptor bases.
fn copy_tile(wg: &mut Wg) -> R {
    wg.loop_carried(".Lpull_trip", (), |wg, ()| {
        let b = wg.isa();
        for k in 0..UNROLL as u8 {
            bload(b, 4, DATA + 4 * k, VOFF, SRD_SRC, 4096 * u32::from(k))?;
        }
        for k in 0..UNROLL as u8 {
            bstore_b128(b, DATA + 4 * k, VOFF, SRD_OUT, 4096 * u32::from(k))?;
        }
        add64_imm(b, SRD_SRC, 4096 * UNROLL)?;
        add64_imm(b, SRD_OUT, 4096 * UNROLL)?;
        sop(b, format!("{} s{TRIPS}, s{TRIPS}, -1", super::common::s_add_i32(b.spec.arch)), &[TRIPS], &[TRIPS])?;
        let more = cmp_wg(wg, format!("s_cmp_gt_i32 s{TRIPS}, 0"), &[s(TRIPS)])?;
        Ok(((), more))
    })?;
    wg.isa().wait_all()
}

/// Shape contract of one emitted variant (the `--shape` input of
/// `peacemaker-isa emit --co`): exact counts of the memory and control mnemonics
/// the kernel is allowed, the forbidden families and the resource limits.
pub fn contract(variant: Variant, arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in ["s_load_b32", "s_load_b128", "s_load_b256", "buffer_load_b128", "buffer_store_b128", "s_barrier_signal", "s_barrier_wait", "s_endpgm"] {
        let n = e.s_text.lines().filter(|l| l.trim().split_whitespace().next() == Some(name)).count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(variant, arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
