// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next owner-routed chunk MoE tile pull builder modules (gfx1201), two certified
//! specializations of one schedule: the original two-rank kernel
//! `ep_gather_grouped_rows_by_tile_owner_pm_gfx1201` (`Variant::Ep2`, 48-byte kernarg, one peer
//! base, owners in `{0, 1}`) and the rank-generic kernel
//! `ep_gather_grouped_rows_by_tile_owner_n4_pm_gfx1201` (`Variant::N4`, 72-byte kernarg, four
//! source slots). Each emitted program runs on the PeaceMaker CPU interpreter (256-thread
//! workgroups, one per tile) against a host reference: tiles of experts another rank owns are
//! byte copies of that owner's rows, every other byte of the destination (owned tiles, the -1
//! padding tail, bytes past the grid) is untouched, and no source buffer is ever written. Each
//! rank's source is a distinct mapped region at a high virtual address (N4: or the destination
//! itself for the local slot; EP2 maps no local slot at all, so a read of it faults). No GPU.
use peacemaker_isa::Arch;
use peacemaker_isa::kernels::ep_gather_grouped_rows_by_tile_owner as pull;
use pull::Variant;
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const BLOCK: u32 = pull::BLOCK;
/// Production tile: 16 grouped rows x 2560 BF16 = 5120 uint4.
const PRODUCTION_TILE_U4: usize = 16 * 2560 * 2 / 16;

/// The launches the module pair serves: the EP2 kernel at two ranks, the rank-generic kernel at two and four.
const LAUNCHES: [(Variant, usize); 3] = [(Variant::Ep2, 2), (Variant::N4, 2), (Variant::N4, 4)];

#[test]
fn emission_is_deterministic_and_targets_gfx1201_only() {
    for variant in Variant::ALL {
        let (a, b) = (pull::emit(variant, ARCH).unwrap(), pull::emit(variant, ARCH).unwrap());
        assert_eq!(a.s_text, b.s_text, "{variant:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{variant:?}");
        assert!(a.s_text.contains(&format!("{}:", pull::symbol(variant, ARCH))));
        assert_eq!(a.shape.barriers, 0);
        assert!(a.shape.next_free_vgpr <= 24 && a.shape.next_free_sgpr <= 104, "{variant:?} {:?}", a.shape);
        let (emitted, text, _) = pull::emit_module(variant, ARCH).unwrap();
        assert_eq!(emitted.len(), 1);
        assert!(text.contains(&format!("{}:", pull::symbol(variant, ARCH))));
        assert_eq!(pull::module(variant, ARCH), pull::symbol(variant, ARCH));
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            assert!(pull::emit(variant, arch).is_err(), "{variant:?} {arch:?}");
            assert!(pull::emit_module(variant, arch).is_err(), "{variant:?} {arch:?}");
        }
    }
    // Stable model-free module keys; the EP2 key is the original one.
    assert_eq!(pull::module(Variant::Ep2, ARCH), "ep_gather_grouped_rows_by_tile_owner_pm_gfx1201");
    assert_eq!(pull::symbol(Variant::Ep2, ARCH), "ep_gather_grouped_rows_by_tile_owner_pm_gfx1201");
    assert_eq!(pull::module(Variant::N4, ARCH), "ep_gather_grouped_rows_by_tile_owner_n4_pm_gfx1201");
    assert_eq!(pull::symbol(Variant::N4, ARCH), "ep_gather_grouped_rows_by_tile_owner_n4_pm_gfx1201");
    assert_eq!(pull::emit_module(Variant::Ep2, ARCH).unwrap().1, pull::emit_module(Variant::Ep2, ARCH).unwrap().1);
    assert_ne!(pull::emit(Variant::Ep2, ARCH).unwrap().s_text, pull::emit(Variant::N4, ARCH).unwrap().s_text);
}

/// Kernarg ABI of `Gpu::ep_gather_grouped_rows_by_tile_owner`: EP2 (48 bytes) tile_ids, owner_table,
/// peer_src, dst (four pointers), then n_experts, rank, tile_u4, tiles (four i32); N4 (72 bytes)
/// tile_ids, owner_table, four source bases, dst (seven pointers), then the same four i32.
/// 256-thread workgroup, no LDS, no atomics, no scratch, no flat / buffer / ds access; the peer
/// loads are `buffer_load_b128` through a raw descriptor whose base is the owner's address.
#[test]
fn kernarg_layout_matches_the_launcher() {
    for (variant, offsets, size) in [
        (Variant::Ep2, vec![0u32, 8, 16, 24, 32, 36, 40, 44], 48u32),
        (Variant::N4, vec![0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 68], 72),
    ] {
        assert_eq!(variant.kernarg_bytes(), size);
        let text = pull::emit_module(variant, ARCH).unwrap().1;
        let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
        let got: Vec<u32> = metadata.lines()
            .filter_map(|l| l.trim().strip_prefix(".offset:").map(|v| v.trim().parse::<u32>().unwrap())).collect();
        assert_eq!(got, offsets, "{variant:?}");
        assert!(text.contains(&format!(".amdhsa_kernarg_size {size}\n")), "{variant:?} kernarg size");
        assert!(text.contains(".amdhsa_group_segment_fixed_size 0\n"), "LDS");
        assert!(text.contains(".max_flat_workgroup_size: 256"), "workgroup size");
        assert!(!text.contains("atomic"));
        for l in text.lines().map(str::trim_start) {
            assert!(!(l.starts_with("scratch_") || l.starts_with("flat_") || l.starts_with("global_") || l.starts_with("ds_")), "{l}");
        }
        let loads = text.lines().filter(|l| l.trim_start().starts_with("buffer_load_b128")).count();
        let stores = text.lines().filter(|l| l.trim_start().starts_with("buffer_store_b128")).count();
        assert_eq!((loads, stores), (pull::UNROLL as usize, pull::UNROLL as usize), "{variant:?}");
    }
}

/// The two specializations differ exactly where the ABI differs: EP2 has no source-slot lookup and
/// no owner range check (one peer base straight into the descriptor), N4 has both.
#[test]
fn the_specializations_differ_only_in_the_source_selection() {
    let count = |variant: Variant, mnemonic: &str| pull::emit(variant, ARCH).unwrap().s_text.lines()
        .filter(|l| l.trim().split_whitespace().next() == Some(mnemonic)).count();
    assert_eq!(count(Variant::Ep2, "s_load_b256"), 1, "EP2 loads ids+table+peer+dst in one b256");
    assert_eq!(count(Variant::N4, "s_load_b256"), 0);
    assert_eq!(count(Variant::Ep2, "s_load_b32"), 2, "EP2: tile id and owner");
    assert_eq!(count(Variant::N4, "s_load_b32"), 2);
    assert_eq!(count(Variant::Ep2, "s_load_b128"), 1);
    assert_eq!(count(Variant::N4, "s_load_b128"), 2, "N4: ids+table and the n_experts words");
    assert_eq!(count(Variant::N4, "s_load_b64"), 2, "N4: dst and the owner's source base");
    assert_eq!(count(Variant::Ep2, "s_load_b64"), 0);
}

#[test]
fn committed_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    for variant in Variant::ALL {
        let module = pull::module(variant, ARCH);
        let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
        let text = pull::emit_module(variant, ARCH).unwrap().1;
        assert!(text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed),
            "{module}: fresh emission differs from the committed bundle");
    }
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-ep-tile-pull-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (dir.join(format!("{stem}.s")), dir.join(format!("{stem}.o")), dir.join(format!("{stem}.co")));
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc).args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={arch}"), "-filetype=obj"]).arg(&s).arg("-o").arg(&o).status().unwrap().success());
    assert!(Command::new(linker).arg("-shared").arg(&o).arg("-o").arg(&co).status().unwrap().success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum).find_map(|i| {
        let name = &e[names + u32_(sh(i))..];
        name.starts_with(b".text\0").then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
    }).unwrap()
}

// ---- CPU interpreter vs the host reference ------------------------------------

fn lift(variant: Variant) -> peacemaker_ir::Program {
    let text = pull::emit_module(variant, ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_lift::lift_object(&elf, peacemaker_lift::Options { frontend: peacemaker_ir::inst::Frontend::Builder }).unwrap().program
}
static PROGRAM_EP2: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(Variant::Ep2));
static PROGRAM_N4: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| lift(Variant::N4));
fn program(variant: Variant) -> &'static peacemaker_ir::Program {
    match variant { Variant::Ep2 => &PROGRAM_EP2, Variant::N4 => &PROGRAM_N4 }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// Bytes past the grid in the destination and the sentinel an unpulled tile keeps.
const TAIL: usize = 4096;
const SENTINEL: u8 = 0xC3;
/// Filler of a local source buffer that is mapped apart from the destination (nothing may read it).
const LOCAL_FILL: u8 = 0xEE;
const MAX_RANKS: usize = pull::MAX_RANKS as usize;
const DST_VA: u64 = 0x1000_0000;
/// Peer `r`'s `y_down_grouped` base: one distinct high mapped region per rank.
fn source_va(r: usize) -> u64 { 0x7f00_0000_0000 + r as u64 * 0x10_0000_0000 }

/// One launch of a `ranks`-rank split (2 or 4) of one specialization. N4: slot `r < ranks` holds rank
/// `r`'s base, the slots above are null (a 2-rank launch never names them); the local slot is the
/// destination itself (`alias_local`) or a separate mapped buffer that nothing may read or write.
/// EP2: the kernarg holds the one peer base `source_va(1 - rank)`; no local buffer is mapped, so any
/// read of it faults.
struct Case<'a> {
    variant: Variant,
    tile_ids: &'a [i32],
    owner_table: &'a [i32],
    rank: i32,
    ranks: usize,
    tile_u4: usize,
    /// Workgroups past `tile_ids.len()` (the grid is the padded tile capacity).
    grid_extra: usize,
    alias_local: bool,
}

/// Whether source buffer `r` is mapped as its own region in this launch.
fn separately_mapped(c: &Case, r: usize) -> bool {
    match c.variant {
        Variant::Ep2 => r != c.rank as usize,
        Variant::N4 => !(r == c.rank as usize && c.alias_local),
    }
}

/// One full grid. `sources[r]` is rank `r`'s buffer (the local entry is the separate local buffer when
/// `!alias_local`, else ignored). Returns the destination; every source buffer is checked unmodified.
fn run(c: &Case, sources: &[Vec<u8>]) -> Vec<u8> {
    let tiles = c.tile_ids.len();
    let tile_bytes = c.tile_u4 * 16;
    let rank = c.rank as usize;
    assert!(c.ranks == 2 || c.ranks == 4);
    assert!(c.variant == Variant::N4 || c.ranks == 2, "the EP2 kernel serves two ranks only");
    assert_eq!(sources.len(), c.ranks);
    assert!(sources.iter().all(|s| s.len() == tiles * tile_bytes));
    let emu = Emulator::new(program(c.variant), &pull::symbol(c.variant, ARCH)).unwrap();
    let (kp, ip, tp) = (0x1_0000u64, 0x10_0000u64, 0x20_0000u64);
    let words = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let mut kern = Vec::new();
    match c.variant {
        Variant::Ep2 => {
            for p in [ip, tp, source_va(1 - rank), DST_VA] { kern.extend(p.to_le_bytes()); }
        }
        Variant::N4 => {
            let mut bases = [0u64; MAX_RANKS];
            for (r, base) in bases.iter_mut().enumerate().take(c.ranks) {
                *base = if r == rank && c.alias_local { DST_VA } else { source_va(r) };
            }
            for p in [ip, tp].into_iter().chain(bases).chain([DST_VA]) { kern.extend(p.to_le_bytes()); }
        }
    }
    for x in [c.owner_table.len() as i32, c.rank, c.tile_u4 as i32, tiles as i32] { kern.extend(x.to_le_bytes()); }
    assert_eq!(kern.len(), c.variant.kernarg_bytes() as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    mem.map("tile_ids".into(), ip, words(c.tile_ids)).unwrap();
    mem.map("table".into(), tp, words(c.owner_table)).unwrap();
    for r in 0..c.ranks {
        if separately_mapped(c, r) { mem.map(format!("src{r}"), source_va(r), sources[r].clone()).unwrap(); }
    }
    mem.map("dst".into(), DST_VA, vec![SENTINEL; tiles * tile_bytes + TAIL]).unwrap();
    for wg in 0..tiles + c.grid_extra {
        emu.run_workgroup(&Launch { kernarg_ptr: kp, workgroup: [wg as u32, 0, 0], block: [BLOCK, 1, 1], dynamic_lds: 0 }, &mut mem)
            .unwrap_or_else(|e| panic!("{:?} tiles {tiles} rank {rank}/{} tile_u4 {} wg {wg}: {e}", c.variant, c.ranks, c.tile_u4));
    }
    let region = |name: &str| mem.regions.iter().find(|r| r.name == name).unwrap().bytes.clone();
    for r in 0..c.ranks {
        if separately_mapped(c, r) { assert_eq!(region(&format!("src{r}")), sources[r], "source {r} modified"); }
    }
    assert_eq!(region("tile_ids"), words(c.tile_ids), "tile ids modified");
    assert_eq!(region("table"), words(c.owner_table), "owner table modified");
    region("dst")
}

/// Peer bytes: every byte pattern class (NaN payloads, +-0, denormals, all-ones) among random bytes.
fn peer_bytes(rng: &mut Rng, len: usize) -> Vec<u8> {
    let specials: [u32; 8] = [0x7fc0_0001, 0xffc0_0000, 0x8000_0000, 0x0000_0001, 0x0080_0000, 0x7f80_0000, 0xffff_ffff, 0];
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let w = if rng.below(4) == 0 { specials[rng.below(8)] } else { rng.next() as u32 };
        out.extend(w.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// The owner rank a tile is pulled from. N4: a live expert whose owner is another rank in `0..4`.
/// EP2: a live expert whose owner is not this rank (its owners are in `{0, 1}` by contract, so the
/// owner is the one peer); an owner outside `{0, 1}` is outside the EP2 contract and never generated.
fn pulled_owner(variant: Variant, id: i32, owner_table: &[i32], rank: i32) -> Option<usize> {
    if !(id >= 0 && (id as usize) < owner_table.len()) { return None }
    let owner = owner_table[id as usize];
    match variant {
        Variant::Ep2 => {
            assert!((0..2).contains(&owner), "EP2 owner {owner} outside {{0, 1}}");
            (owner != rank).then_some(owner as usize)
        }
        Variant::N4 => (owner != rank && (0..MAX_RANKS as i32).contains(&owner)).then_some(owner as usize),
    }
}

/// Distinct random contents per rank (the local entry is the never-read filler), launch, compare to the host reference.
fn check(name: &str, rng: &mut Rng, variant: Variant, tile_ids: &[i32], owner_table: &[i32], rank: i32, ranks: usize, tile_u4: usize, grid_extra: usize) {
    let tiles = tile_ids.len();
    let tile_bytes = tile_u4 * 16;
    let alias_local = rng.below(2) == 0;
    let sources: Vec<Vec<u8>> = (0..ranks)
        .map(|r| if r == rank as usize { vec![LOCAL_FILL; tiles * tile_bytes] } else { peer_bytes(rng, tiles * tile_bytes) })
        .collect();
    let case = Case { variant, tile_ids, owner_table, rank, ranks, tile_u4, grid_extra, alias_local };
    let dst = run(&case, &sources);
    let mut want = vec![SENTINEL; tiles * tile_bytes + TAIL];
    for (t, &id) in tile_ids.iter().enumerate() {
        let range = t * tile_bytes..(t + 1) * tile_bytes;
        if let Some(owner) = pulled_owner(variant, id, owner_table, rank) {
            assert!(owner < ranks, "{name}: owner {owner} outside the {ranks}-rank launch");
            want[range.clone()].copy_from_slice(&sources[owner][range.clone()]);
            // The sources differ in every pulled tile, so a wrong slot cannot pass.
            for r in (0..ranks).filter(|&r| r != owner && r != rank as usize) {
                assert_ne!(sources[r][range.clone()], sources[owner][range.clone()], "{name}: sources {r} and {owner} equal");
            }
        }
    }
    if dst != want {
        let first = dst.iter().zip(&want).position(|(a, b)| a != b).unwrap();
        panic!("{name}: {variant:?} tiles {tiles} rank {rank}/{ranks} tile_u4 {tile_u4} alias_local {alias_local}: first differing byte {first} (tile {}) ids {tile_ids:?}", first / tile_bytes);
    }
}

fn table(experts: usize, ranks: usize) -> Vec<i32> { (0..experts).map(|e| (e % ranks) as i32).collect() }

/// Arbitrary ownership (not `e % ranks`): every expert is owned by a random rank of the split.
fn scattered(rng: &mut Rng, experts: usize, ranks: usize) -> Vec<i32> { (0..experts).map(|_| rng.below(ranks) as i32).collect() }

/// Mixed ownership and the -1 tail on small tiles (1024, 2048 and 3072 uint4: one to three trips)
/// for every rank of every launch (EP2 at 2 ranks, N4 at 2 and 4 ranks), `e % ranks` and arbitrary owner tables.
#[test]
fn emulated_launches_equal_the_host_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut cases = 0usize;
    for (variant, ranks) in LAUNCHES {
        for &tile_u4 in &[1024usize, 2048, 3072] {
            for &tiles in &[1usize, 2, 7, 24] {
                for trial in 0..3 {
                    let owner = if trial == 2 { scattered(&mut rng, 512, ranks) } else { table(512, ranks) };
                    let live = tiles - rng.below(tiles / 3 + 1);
                    let mut ids: Vec<i32> = (0..live).map(|_| rng.below(512) as i32).collect();
                    if trial == 0 { for (i, id) in ids.iter_mut().enumerate() { *id = (i % ranks) as i32 + ranks as i32 * (i as i32 / 3); } }
                    ids.resize(tiles, -1);
                    for rank in 0..ranks as i32 {
                        check(&format!("ranks {ranks} tile_u4 {tile_u4} trial {trial}"), &mut rng, variant, &ids, &owner, rank, ranks, tile_u4, 0);
                        cases += 1;
                    }
                }
            }
        }
    }
    println!("emulated_launches_equal_the_host_reference: {cases} launches");
}

/// The production tile (16 rows x 2560 BF16 = 5120 uint4, five 1024-uint4 trips of the unrolled
/// walk) with tiles of every owner and sentinels, every rank of every launch.
#[test]
fn emulated_production_tile_equals_the_host_reference() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    assert_eq!(PRODUCTION_TILE_U4 % pull::TRIP_U4 as usize, 0);
    let ids = [0, 1, 1, 2, 3, 5, 6, -1, -1];
    for (variant, ranks) in LAUNCHES {
        for owner in [table(512, ranks), scattered(&mut rng, 512, ranks)] {
            for rank in 0..ranks as i32 {
                check("production tile", &mut rng, variant, &ids, &owner, rank, ranks, PRODUCTION_TILE_U4, 0);
            }
        }
    }
}

/// Grid larger than the tile count (the workgroups past `tiles` return), ids outside the table and the
/// padding tail (never pulled), an all-local grid, an all-peer grid, one owner for every expert, and
/// hand-written arbitrary owners, for every launch; then, N4 only, owners outside `0..4` (never pulled,
/// never dereferenced; the EP2 contract has owners in `{0, 1}` only).
#[test]
fn emulated_edge_launches_equal_the_host_reference() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for (variant, ranks) in LAUNCHES {
        let owner = table(512, ranks);
        for rank in 0..ranks as i32 {
            check("grid past tiles", &mut rng, variant, &[3, 4, -1], &owner, rank, ranks, 1024, 5);
            check("odd ids", &mut rng, variant, &[-1, i32::MIN, i32::MAX, 512, 513, 0x4000_0000, 511, 0, 1], &owner, rank, ranks, 1024, 0);
            check("one tile", &mut rng, variant, &[9], &owner, rank, ranks, 1024, 0);
            check("all sentinel", &mut rng, variant, &[-1; 6], &owner, rank, ranks, 1024, 0);
            check("padding tail", &mut rng, variant, &[5, 6, -1, -1, 7, -1, -1, -1], &owner, rank, ranks, 2048, 3);
        }
        let local: Vec<i32> = (0..6).map(|i| (i * ranks) as i32).collect();
        check("all local", &mut rng, variant, &local, &owner, 0, ranks, 1024, 0);
        let peers: Vec<i32> = (0..6).map(|i| (i * ranks + 1) as i32).collect();
        check("all peer", &mut rng, variant, &peers, &owner, 0, ranks, 1024, 0);
        for sole in 0..ranks as i32 {
            let one_owner = vec![sole; 512];
            for rank in 0..ranks as i32 {
                check(&format!("one owner {sole}"), &mut rng, variant, &[0, 7, 8, -1, 100, 511], &one_owner, rank, ranks, 1024, 0);
            }
        }
    }
    let ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1];
    for variant in Variant::ALL {
        for rank in 0..2 {
            check("arbitrary owners ranks 2", &mut rng, variant, &ids, &[1, 0, 0, 1, 1, 1, 0, 0], rank, 2, 1024, 0);
        }
    }
    for rank in 0..4 {
        check("arbitrary owners ranks 4", &mut rng, Variant::N4, &ids, &[3, 0, 2, 1, 1, 3, 0, 2], rank, 4, 1024, 0);
        check("owner outside 0..4", &mut rng, Variant::N4, &ids, &[0, 1, 4, 7, -1, i32::MIN, i32::MAX, 2, 3], rank, 4, 1024, 0);
    }
    for rank in 0..2 {
        // The null slots 2 and 3 of a 2-rank launch are never dereferenced.
        check("owner outside 0..4 ranks 2", &mut rng, Variant::N4, &ids, &[0, 1, 4, 7, -1, i32::MIN, i32::MAX, 1, 0], rank, 2, 1024, 0);
    }
}

/// Random shapes: tiles 1..=40, tile widths 1024..=4096, any launch, any rank, `e % ranks` or arbitrary
/// owners, ids uniform / runs / odd.
#[test]
fn emulated_random_shapes_equal_the_host_reference() {
    let mut rng = Rng(0x2F0B_4A32_7C15_91E3);
    let cases = 90usize;
    for case in 0..cases {
        let tiles = 1 + rng.below(40);
        let tile_u4 = 1024 * (1 + rng.below(4));
        let (variant, ranks) = LAUNCHES[rng.below(LAUNCHES.len())];
        let rank = rng.below(ranks) as i32;
        let owner = if rng.below(2) == 0 { table(512, ranks) } else { scattered(&mut rng, 512, ranks) };
        let ids: Vec<i32> = (0..tiles).map(|_| match rng.below(8) {
            0 => -1,
            1 => [i32::MIN, 512, 700, 0x4000_0000][rng.below(4)],
            _ => rng.below(512) as i32,
        }).collect();
        let extra = rng.below(3);
        check(&format!("shape #{case}"), &mut rng, variant, &ids, &owner, rank, ranks, tile_u4, extra);
    }
    println!("emulated_random_shapes_equal_the_host_reference: {cases} launches");
}

/// EP2 copied-byte oracle, independent of `check`: the original two-rank law (a live expert whose owner is
/// not this rank is copied from the one peer, rows at the same offset) over the EP2 launch shape. Both
/// specializations run it (EP2 with its 48-byte one-peer kernarg; N4 with the peer in slot `1 - rank`, the
/// local slot the destination or a separate buffer, the other two slots null): the copied byte count and
/// every byte of the destination equal the oracle's, so the two kernels agree byte for byte.
#[test]
fn ep2_launch_equals_the_two_rank_oracle() {
    let mut rng = Rng(0xE9_2C0A_1E5C_ED01);
    for rank in 0..2usize {
        for tile_u4 in [1024usize, 2048] {
            for owner in [table(512, 2), scattered(&mut rng, 512, 2)] {
                let ids: Vec<i32> = (0..14).map(|_| if rng.below(5) == 0 { -1 } else { rng.below(512) as i32 }).collect();
                let tile_bytes = tile_u4 * 16;
                let peer = peer_bytes(&mut rng, ids.len() * tile_bytes);
                let mut want = vec![SENTINEL; peer.len() + TAIL];
                let mut copied = 0usize;
                for (t, &id) in ids.iter().enumerate() {
                    if id >= 0 && (id as usize) < owner.len() && owner[id as usize] != rank as i32 {
                        want[t * tile_bytes..(t + 1) * tile_bytes].copy_from_slice(&peer[t * tile_bytes..(t + 1) * tile_bytes]);
                        copied += tile_bytes;
                    }
                }
                assert!(copied > 0, "oracle case copies nothing");
                for variant in Variant::ALL {
                    for alias_local in [true, false] {
                        let mut sources = vec![vec![LOCAL_FILL; peer.len()]; 2];
                        sources[1 - rank] = peer.clone();
                        let case = Case { variant, tile_ids: &ids, owner_table: &owner, rank: rank as i32, ranks: 2, tile_u4, grid_extra: 2, alias_local };
                        let dst = run(&case, &sources);
                        assert_eq!(dst, want, "{variant:?} rank {rank} tile_u4 {tile_u4} alias_local {alias_local}: {copied} bytes expected");
                    }
                }
            }
        }
    }
}

/// Every instruction of every module has interpreter semantics.
#[test]
fn interpreter_covers_the_module() {
    for variant in Variant::ALL {
        let mut missing = std::collections::BTreeSet::new();
        let program = program(variant);
        for k in &program.kernels {
            for id in &k.body.layout {
                let i = k.body.insts.get(*id).unwrap();
                let row = peacemaker_ir::isa::lookup(program.target.arch, i.op, i.form).unwrap();
                if peacemaker_ir::emu::semantic_class(row.name).is_none() { missing.insert(row.name); }
            }
        }
        assert!(missing.is_empty(), "{variant:?}: no interpreter semantics: {missing:?}");
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check, toolchain::{build, certify, Toolchain}};

    /// Each symbol certifies: parse-back, independent wait replay, the committed shape contract,
    /// the static LDS bound, and M7's lift identity plus wait/hazard/barrier/window analyses of the
    /// linked ELF, obligation-free; the emitted program is identical on a second emission (determinism).
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for variant in Variant::ALL {
            let (emitted, text, _) = pull::emit_module(variant, ARCH).unwrap();
            assert_eq!(text, pull::emit_module(variant, ARCH).unwrap().1, "{variant:?}: emission is not deterministic");
            for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
            let dir = std::env::temp_dir().join(format!("peacemaker-isa-ep-tile-pull-cert-{:?}-{}", variant, std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
            let symbol = pull::symbol(variant, ARCH);
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}: {m7}");
            let path = format!("{}/kernels/{}.{}.contract.json", env!("CARGO_MANIFEST_DIR"), variant.kernel_id(), ARCH.name());
            let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
            assert_eq!(json, pull::contract(variant, ARCH, &emitted[0]), "{path} is stale");
            let contract = serde_json::from_slice(&raw).unwrap();
            certify(&toolchain, &build, &s, ARCH.name(), &dir.join("manifest.json"), Some(&contract), "test", "test")
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        }
    }
}
