// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1201 pointer-addressed F4 read-once C8 lane-union decode GEMV module
//! `mq_i4_moe_gemv_pm_gfx1201` (`mq_i4_moe_gemv_{mq3i4,mq4sym}_{r256_gate_up,r128_down}`):
//! symbols, the frozen 88-byte kernarg ABI and its typed ownership, exact-arch gating,
//! the instruction census of the frozen numerical tree, the read-once structure
//! (no weight load inside any slot branch), the derived shape contract, native
//! assembly, and (feature `toolchain`) wait replay, M7, certification and the native
//! code object / bundle identity against the ROCm oracle for the four entries and the module.
use peacemaker_isa::Arch;
use peacemaker_isa::kernels::mq_i4_moe_gemv::{self, Codec, Kind, Spec};
#[cfg(feature = "toolchain")]
#[path = "support/native_compare.rs"]
mod native_compare;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const MODULE: &str = "mq_i4_moe_gemv_pm_gfx1201";
const SYMBOLS: [&str; 4] = [
    "mq_i4_moe_gemv_mq3i4_r256_gate_up", "mq_i4_moe_gemv_mq3i4_r128_down",
    "mq_i4_moe_gemv_mq4sym_r256_gate_up", "mq_i4_moe_gemv_mq4sym_r128_down",
];
/// Frozen ABI (DESIGN §4.2): `(offset, size, name)` in kernarg order.
const ARGS: [(u32, u32, &str); 15] = [
    (0, 8, "expert_weight_ptrs"), (8, 8, "expert_ids"), (16, 8, "expert_offsets"), (24, 8, "slot_ids"),
    (32, 8, "Xrot"), (40, 8, "Y0"), (48, 8, "Y1"),
    (56, 4, "M"), (60, 4, "K"), (64, 4, "U"), (68, 4, "C"), (72, 4, "x_row_div"), (76, 4, "x_src_rows"), (80, 4, "x_row_stride"), (84, 4, "y_slot_stride"),
];
const SLOTS: usize = 8;

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines().filter_map(|l| l.trim().strip_prefix(key)).map(|v| v.trim().parse().unwrap()).collect()
}
fn count(text: &str, mnemonic: &str) -> usize {
    text.lines().filter(|l| l.split_whitespace().next() == Some(mnemonic)).count()
}
fn spec(codec: Codec, kind: Kind) -> Spec { Spec { arch: ARCH, codec, kind } }
fn mnemonic(line: &str) -> &str { line.split_whitespace().next().unwrap_or("") }
/// Operand `n` (0-based) of an instruction line, split on `", "`.
fn operand(line: &str, n: usize) -> String {
    line.trim().split_once(' ').map(|(_, rest)| rest.split(", ").nth(n).unwrap_or("").trim().to_owned()).unwrap_or_default()
}
fn vreg(token: &str) -> u32 { token.trim_start_matches('v').parse().unwrap_or_else(|_| panic!("not a VGPR: {token}")) }

#[test]
fn module_exports_are_the_four_frozen_symbols_in_order() {
    let specs = mq_i4_moe_gemv::module_specs(ARCH);
    assert_eq!(specs, [
        spec(Codec::Mq3I4, Kind::GateUp), spec(Codec::Mq3I4, Kind::Down),
        spec(Codec::Mq4Sym, Kind::GateUp), spec(Codec::Mq4Sym, Kind::Down),
    ]);
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), SYMBOLS);
    assert_eq!(Spec::module(ARCH), MODULE);
    let (emitted, text, proof) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    assert_eq!(text.matches(".amdhsa_kernel ").count(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(text.contains(&format!(".amdhsa_kernel {symbol}\n")), "{symbol}: missing from module source");
        assert!(text.contains(&format!(".globl {symbol}\n")), "{symbol}: not global");
        assert!(text.contains(&format!(".name: {symbol}\n")), "{symbol}: missing from module metadata");
    }
}

/// Emission is bit-deterministic, and every non-gfx1201 architecture is refused (the builder
/// is exact-gfx1201: no implicit alias for gfx1100 or gfx1151).
#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let (a, b) = (mq_i4_moe_gemv::emit(spec).unwrap(), mq_i4_moe_gemv::emit(spec).unwrap());
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
    }
    let (_, ta, pa) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    let (_, tb, pb) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        for spec in mq_i4_moe_gemv::module_specs(arch) { assert!(mq_i4_moe_gemv::emit(spec).is_err(), "{arch:?} {spec:?}"); }
        assert!(mq_i4_moe_gemv::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// Every symbol is wave32, one wave per workgroup, without scratch, spills, LDS or barriers.
#[test]
fn every_export_is_wave32_without_scratch_or_lds() {
    let (emitted, text, _) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    for (e, symbol) in emitted.iter().zip(SYMBOLS) {
        assert_eq!(e.proof.arch, ARCH, "{symbol}");
        assert!(e.s_text.contains(".amdhsa_wavefront_size32 1"), "{symbol}: wave32");
        assert!(e.s_text.contains(".private_segment_fixed_size: 0\n"), "{symbol}: private segment");
        assert!(e.s_text.contains(".sgpr_spill_count: 0\n") && e.s_text.contains(".vgpr_spill_count: 0\n"), "{symbol}: spills");
        assert!(e.s_text.contains(".group_segment_fixed_size: 0\n"), "{symbol}: no LDS");
        assert!(e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"), "{symbol}: grid y is the expert");
        assert_eq!(e.shape.barriers, 0, "{symbol}: barriers");
        let ds = e.s_text.lines().find(|l| mnemonic(l).starts_with("ds_load") || mnemonic(l).starts_with("ds_store") || mnemonic(l).starts_with("ds_swizzle"));
        assert!(ds.is_none(), "{symbol}: LDS access {ds:?}");
        assert!(!e.s_text.lines().any(|l| mnemonic(l).starts_with("scratch_") || mnemonic(l).starts_with("flat_")), "{symbol}: scratch/flat");
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1201"));
}

/// The 88-byte kernarg block: per-argument offsets/sizes/names, access, the launch bound, the
/// first loads, and the typed ownership of the registers that carry it.
#[test]
fn kernarg_abi_matches_the_frozen_88_byte_layout() {
    assert_eq!(mq_i4_moe_gemv::KERNARG_BYTES, 88);
    assert_eq!(mq_i4_moe_gemv::WORKGROUP_SIZE, 32);
    assert_eq!(mq_i4_moe_gemv::MAX_SLOTS as usize, SLOTS);
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert!(e.s_text.contains(".kernarg_segment_size: 88\n"), "{symbol}: kernarg size");
        assert!(e.s_text.contains(".amdhsa_kernarg_size 88\n"), "{symbol}: descriptor kernarg size");
        assert_eq!(field(&e.s_text, ".offset:"), ARGS.iter().map(|a| a.0).collect::<Vec<_>>(), "{symbol}: kernarg offsets");
        assert_eq!(field(&e.s_text, ".size:"), ARGS.iter().map(|a| a.1).collect::<Vec<_>>(), "{symbol}: kernarg sizes");
        let names: Vec<String> = e.s_text.lines().filter_map(|l| l.trim().strip_prefix(".name: ")).take(ARGS.len()).map(str::to_owned).collect();
        assert_eq!(names, ARGS.iter().map(|a| a.2.to_owned()).collect::<Vec<_>>(), "{symbol}: kernarg names");
        assert_eq!(field(&e.s_text, ".max_flat_workgroup_size:"), [32], "{symbol}: block 32");
        let (last_off, last_size, _) = *ARGS.last().unwrap();
        assert_eq!(last_off + last_size, mq_i4_moe_gemv::KERNARG_BYTES, "{symbol}: arguments cover the kernarg block");
        // Tables and X are read-only, Y0/Y1 write-only (LLVM key order: the first key carries the list dash).
        for (name, access) in [("expert_weight_ptrs", "read_only"), ("expert_ids", "read_only"), ("expert_offsets", "read_only"),
                               ("slot_ids", "read_only"), ("Xrot", "read_only"), ("Y0", "write_only"), ("Y1", "write_only")] {
            assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{symbol}: {name} access {access}");
        }
        // The kernarg is loaded as four scalar loads covering offsets 0..88 with no over-read.
        for load in ["s_load_b256 s[8:15], s[0:1], 0x0", "s_load_b256 s[16:23], s[0:1], 0x20", "s_load_b128 s[24:27], s[0:1], 0x40", "s_load_b64 s[28:29], s[0:1], 0x50"] {
            assert!(e.s_text.contains(load), "{symbol}: kernarg load {load}");
        }
        // Typed ownership: the register plan declares the C8 slot state and every access is inside a declared range.
        let plan: Vec<&str> = e.proof.reg_plan.iter().map(|r| r.name.as_ref()).collect();
        for owner in ["kernel_arguments_ptrs", "kernel_arguments_x_y_m_k", "kernel_arguments_u_c_div_src", "kernel_arguments_strides",
                      "x_descriptor", "y_select", "accumulators", "x_values", "weights", "slot_x_offsets", "slot_y_addresses", "shuffle_masks", "shuffle_addresses"] {
            assert!(plan.contains(&owner), "{symbol}: reg plan misses {owner}: {plan:?}");
        }
    }
}

/// Honest register ceilings of the C8 state: eight slots of accumulators (four chains per slot for gate)
/// and X registers, whole-granule VGPR reservation, SGPR bookkeeping, the contract equals the plan.
#[test]
fn register_ceilings_reflect_c8_slot_state() {
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let symbol = spec.symbol();
        let (nw, chains) = (spec.lane_elems() as u16, spec.kind.chains() as u16);
        let floor = SLOTS as u16 * (chains + nw) + 2 * SLOTS as u16;
        assert!(e.shape.next_free_vgpr >= floor && e.shape.next_free_vgpr % 8 == 0 && e.shape.next_free_vgpr <= 256, "{symbol}: vgpr {}", e.shape.next_free_vgpr);
        assert_eq!(e.shape.next_free_sgpr, 80, "{symbol}: sgpr");
        assert!(e.s_text.contains(&format!(".vgpr_count: {}\n", e.shape.next_free_vgpr)), "{symbol}: metadata vgpr count");
        let c = mq_i4_moe_gemv::contract(spec, &e);
        assert_eq!((c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()), (Some(u64::from(e.shape.next_free_vgpr)), Some(u64::from(e.shape.next_free_sgpr))), "{symbol}");
    }
}

/// The frozen numerical tree per symbol: weight/dot/accumulate chain per group instance, the gate
/// four-chain fold vs the down single chain, five snapshot shuffle steps per slot with out-of-range lanes
/// unchanged (sum into scratch, then a mask select), eight lane-0 stores, `Mq4Sym` affine fma vs `Mq3I4`
/// signed `2c-7` grid (never an XOR rebias).
#[test]
fn instruction_census_matches_the_frozen_dag() {
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        let nw = spec.lane_elems() as usize;
        let planar = spec.codec == Codec::Mq3I4;
        let gate = spec.kind == Kind::GateUp;
        // One slot-id load in the prologue; weight-class loads per group instance by codec/kind.
        let (u8s, u16s, b32s) = match (spec.codec, spec.kind) {
            (Codec::Mq3I4, _) => (2 * inst, inst, 1),
            (Codec::Mq4Sym, Kind::GateUp) => (0, 0, 2 * inst + 1),
            (Codec::Mq4Sym, Kind::Down) => (0, inst, inst + 1),
        };
        assert_eq!(count(&t, "global_load_u8"), u8s, "{symbol}: packet byte loads");
        assert_eq!(count(&t, "global_load_u16"), u16s, "{symbol}: u16 loads");
        assert_eq!(count(&t, "global_load_b32"), b32s, "{symbol}: dword loads");
        assert_eq!(count(&t, "buffer_load_b128"), inst * SLOTS * nw / 4, "{symbol}: X loads");
        assert_eq!(count(&t, "global_store_b32"), SLOTS, "{symbol}: stores");
        assert_eq!(count(&t, "v_bfe_u32"), nw * inst, "{symbol}: code extracts");
        assert_eq!(count(&t, "v_cvt_f32_i32_e32"), if planar { 4 * inst } else { 0 }, "{symbol}: signed grid converts");
        assert_eq!(count(&t, "v_cvt_f32_u32_e32"), 2 + if planar { 0 } else { nw * inst }, "{symbol}: unsigned converts (2 are the slot divide)");
        assert_eq!(count(&t, "v_fma_f32"), if planar { 0 } else { nw * inst }, "{symbol}: affine fma");
        assert_eq!(count(&t, "v_cvt_f32_f16_e64"), inst * if planar { 1 } else { 2 }, "{symbol}: header converts");
        assert_eq!(count(&t, "v_lshl_or_b32"), if planar { inst } else { 0 }, "{symbol}: W3 packet combine");
        // First dot term per slot per group (plus 4 weight multiplies per group for Mq3I4, plus the divide's one multiply).
        assert_eq!(count(&t, "v_mul_f32_e32"), SLOTS * inst + if planar { 4 * inst } else { 0 } + 1, "{symbol}: multiplies");
        assert_eq!(count(&t, "v_fmac_f32_e32"), (nw - 1) * SLOTS * inst, "{symbol}: fused steps");
        let fold = if gate { 3 } else { 0 };
        assert_eq!(count(&t, "v_add_f32_e32"), SLOTS * inst + SLOTS * (fold + 5), "{symbol}: group-partial, fold and shuffle adds");
        assert_eq!(count(&t, "ds_bpermute_b32"), 5 * SLOTS, "{symbol}: shuffle steps");
        assert_eq!(count(&t, "v_readlane_b32"), 2 * SLOTS, "{symbol}: slot broadcasts");
        assert_eq!(count(&t, "v_cmpx_eq_u32_e32"), 1, "{symbol}: lane-0 store mask");
        for forbidden in ["ds_swizzle", "ds_load", "ds_store", "v_fma_mix_f32", "v_xor_b32", "flat_", "scratch_", "v_wmma", "global_atomic", "buffer_atomic"] {
            let hit = t.lines().find(|l| mnemonic(l).starts_with(forbidden));
            assert!(hit.is_none(), "{symbol}: emits {hit:?}");
        }
        if planar {
            assert!(t.lines().any(|l| l.trim().starts_with("v_add_nc_u32_e32 v") && l.trim().ends_with(", -7, v24") ), "{symbol}: 2c-7 grid");
        }
        // Left-associated chain: p = RN(w0*x0), then fma(w_j, x_j, p) j = 1..nw-1 with adjacent registers.
        // (the builder's `s_delay_alu` issue hints and waits sit between dependent instructions and are not part of the tree)
        let lines: Vec<&str> = t.lines().map(str::trim).filter(|l| !mnemonic(l).starts_with("s_delay_alu") && !mnemonic(l).starts_with("s_wait_") && !mnemonic(l).starts_with("s_nop")).collect();
        let first_fmac = lines.iter().position(|l| mnemonic(l) == "v_fmac_f32_e32").unwrap();
        let mul = lines[first_fmac - 1];
        assert_eq!(mnemonic(mul), "v_mul_f32_e32", "{symbol}: first dot term");
        // Mq3I4: p = RN(w0*x0), fma(w1,x1,p), fma(w2,x2,p)...; Mq4Sym (production contraction of the old MQ4 kernels):
        // p = RN(w1*x1), fma(w0,x0,p), then fma(w_j,x_j,p) for j = 2.. ascending. Adjacent registers per element.
        let order: Vec<u32> = if planar { (0..nw as u32).collect() } else { [1, 0].into_iter().chain(2..nw as u32).collect() };
        let (dot, wf, xf) = (vreg(&operand(mul, 0)), vreg(&operand(mul, 1)), vreg(&operand(mul, 2)));
        let (w0, x0) = (wf - order[0], xf - order[0]);
        for (k, &j) in order.iter().enumerate().skip(1) {
            let l = lines[first_fmac + k - 1];
            assert_eq!(mnemonic(l), "v_fmac_f32_e32", "{symbol}: fused step {k}");
            assert_eq!((vreg(&operand(l, 0)), vreg(&operand(l, 1)), vreg(&operand(l, 2))), (dot, w0 + j, x0 + j), "{symbol}: fused step {k} (element {j}) operands");
        }
        // The partial is added separately: `acc = acc + p`.
        let add = lines[first_fmac + nw - 1];
        assert_eq!((mnemonic(add), vreg(&operand(add, 0)), vreg(&operand(add, 1)), vreg(&operand(add, 2))), ("v_add_f32_e32", vreg(&operand(add, 0)), vreg(&operand(add, 0)), dot), "{symbol}: group partial add");
        // Shuffle offsets 16, 8, 4, 2, 1 in that order.
        let offsets: Vec<u32> = lines.iter().filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 1) == "0").map(|l| operand(l, 2).parse().unwrap()).collect();
        assert_eq!(offsets, [16, 8, 4, 2, 1], "{symbol}: shuffle offsets");
        // Out-of-range lanes stay unchanged: per slot and step the sum goes to the exchange register, and v_cndmask
        // keeps the old accumulator where the step's compare mask is clear.
        let exch = vreg(&operand(lines.iter().find(|l| mnemonic(l) == "ds_bpermute_b32").unwrap(), 0));
        let selects = lines.iter().filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 0) == operand(l, 1) && vreg(&operand(l, 2)) == exch).count();
        assert_eq!(selects, 5 * SLOTS, "{symbol}: accumulator selects");
        let sums = lines.iter().filter(|l| mnemonic(l) == "v_add_f32_e32" && vreg(&operand(l, 0)) == exch && vreg(&operand(l, 2)) == exch).count();
        assert_eq!(sums, 5 * SLOTS, "{symbol}: shuffle sums into scratch");
        assert!(!lines.iter().any(|l| mnemonic(l) == "v_add_f32_e32" && operand(l, 0) == operand(l, 1) && vreg(&operand(l, 2)) == exch), "{symbol}: unconditional shuffle add");
        if spec.codec == Codec::Mq4Sym {
            // w = fma(d, float(q), z): one (d, z) pair across every fma, weight in place.
            let fmas: Vec<&&str> = lines.iter().filter(|l| mnemonic(l) == "v_fma_f32").collect();
            let (d, z) = (operand(fmas[0], 1), operand(fmas[0], 3));
            assert!(fmas.iter().all(|l| operand(l, 0) == operand(l, 2) && operand(l, 1) == d && operand(l, 3) == z), "{symbol}: affine weight fma(d, q, z)");
        }
    }
}

/// Read-once C8: every weight-class load of a group sits before that group's first slot branch and
/// never inside one; the number of weight-class loads does not depend on the slot count (it is the same
/// per group instance for one and for eight slots), and every slot branch targets a join label of the
/// same group (or the epilogue ladders), so a skipped slot can never skip a weight load.
#[test]
fn weights_are_loaded_once_per_group_outside_every_slot_branch() {
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let t = mq_i4_moe_gemv::emit(spec).unwrap().s_text;
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        let mut inside = false;
        let (mut gated, mut weight_loads_outside) = (0usize, 0usize);
        for line in t.lines().map(str::trim) {
            if line.ends_with(':') && !line.contains(' ') { inside = false; continue }
            let m = mnemonic(line);
            if m == "s_cbranch_scc0" && (line.ends_with("_xdone") || line.ends_with("_done")) { inside = true; gated += 1; continue }
            if inside {
                assert!(!m.starts_with("global_load") && !m.starts_with("s_load"), "{symbol}: weight/scalar load inside a slot branch: {line}");
            } else if m.starts_with("global_load") { weight_loads_outside += 1 }
        }
        // 7 X-ladder + 7 compute-ladder gates per group instance, 7 + 7 in the reduce and store ladders.
        assert_eq!(gated, 14 * inst + 14, "{symbol}: slot gates");
        // Weight-class loads (plus the single slot-id load) are exactly the per-instance counts, never multiplied by the slot count.
        let per_group = match spec.codec { Codec::Mq3I4 => 3, Codec::Mq4Sym => 2 }; // Mq3I4: two packet bytes + scale; Mq4Sym: header dword + payload
        assert_eq!(weight_loads_outside, per_group * inst + 1, "{symbol}: weight loads per group (read once)");
        // No loop over slots around a group: the group instance count is the unrolled one (4 loop + 3 tail groups, or 1), not slots * groups.
        assert_eq!(count(&t, "s_cbranch_scc0"), 14 * inst + 14 + if spec.kind == Kind::GateUp { 3 } else { 0 }, "{symbol}: scc0 branches");
        assert_eq!(count(&t, "s_cbranch_scc1"), 7, "{symbol}: scc1 branches (5 guards, loop entry skip, back edge)");
    }
}

/// The shape contract derives deterministically from emission: counts of the characteristic
/// instructions, the forbidden families (LDS and scratch, no `buffer_*` ban: X goes through a raw
/// descriptor), register ceilings equal to the plan, no private or LDS memory.
#[test]
fn shape_contract_is_derived_from_the_emission() {
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let e = mq_i4_moe_gemv::emit(spec).unwrap();
        let c = mq_i4_moe_gemv::contract(spec, &e);
        assert_eq!(c, mq_i4_moe_gemv::contract(spec, &mq_i4_moe_gemv::emit(spec).unwrap()));
        assert_eq!(c["symbol"], spec.symbol().as_str());
        assert_eq!(c["counts"]["s_endpgm"], 1);
        assert_eq!(c["counts"]["global_store_b32"], SLOTS);
        assert_eq!(c["counts"]["ds_bpermute_b32"], 5 * SLOTS);
        assert_eq!(c["counts"]["v_readlane_b32"], 2 * SLOTS);
        assert_eq!((c["launch_dynamic_lds_bytes"].as_u64(), c["group_segment_fixed_bytes"].as_u64()), (Some(0), Some(0)));
        assert_eq!((c["require_wave32"].as_bool(), c["require_zero_spills"].as_bool(), c["require_zero_private"].as_bool()), (Some(true), Some(true), Some(true)));
        let forbidden: Vec<&str> = c["forbidden"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        for family in ["scratch_*", "flat_*", "ds_load*", "ds_store*", "ds_swizzle*", "s_barrier*", "v_wmma*", "v_fma_mix*", "s_waitcnt*"] { assert!(forbidden.contains(&family), "{family}"); }
        assert!(!forbidden.contains(&"buffer_*") && !forbidden.contains(&"v_readlane*"));
    }
}

/// Production `emit` is the reference `Mq3I4` network (two `global_load_u8` at offsets 0 and 1 plus `v_lshl_or_b32`);
/// `emit_optimized` retains the single `global_load_u16` packet network (byte-exact on GPU, slower by timing, not production).
/// Both read the same 16 bits at the same address into the same register, the packet never crosses the 48-byte
/// group, every other VALU instruction (all arithmetic) is identical, and `Mq4Sym` is byte-identical.
#[test]
fn optimized_w3_packet_is_the_same_sixteen_bits_as_the_reference_two_byte_network() {
    // Lane l reads group bytes (3l)>>1 and +1: at most byte 47 of 48.
    assert!((0..32u32).all(|l| (3 * l) / 2 + 2 <= 48));
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let (opt, reference) = (mq_i4_moe_gemv::emit_optimized(spec).unwrap().s_text, mq_i4_moe_gemv::emit(spec).unwrap().s_text);
        let symbol = spec.symbol();
        let inst = spec.group_instances() as usize;
        if spec.codec == Codec::Mq4Sym { assert_eq!(opt, reference, "{symbol}: Mq4Sym is network-independent"); continue }
        assert_eq!((count(&reference, "global_load_u8"), count(&reference, "global_load_u16"), count(&reference, "v_lshl_or_b32")), (2 * inst, inst, inst), "{symbol}: reference network");
        assert_eq!((count(&opt, "global_load_u8"), count(&opt, "global_load_u16"), count(&opt, "v_lshl_or_b32")), (0, 2 * inst, 0), "{symbol}: optimized network");
        let loads = |t: &str, m: &str| -> Vec<String> { t.lines().map(str::trim).filter(|l| mnemonic(l) == m).map(str::to_owned).collect() };
        let bytes = loads(&reference, "global_load_u8");
        let (first, second): (Vec<&String>, Vec<&String>) = bytes.iter().partition(|l| !l.contains("offset:1"));
        assert_eq!((first.len(), second.len()), (inst, inst), "{symbol}: byte loads at offsets 0 and 1");
        let packet_dst = operand(first[0], 0);
        let optimized_packet: Vec<String> = loads(&opt, "global_load_u16").into_iter().filter(|l| operand(l, 0) == packet_dst).collect();
        assert_eq!(optimized_packet.len(), inst, "{symbol}: one u16 packet load per group");
        for ((lo, hi), wide) in first.iter().zip(&second).zip(&optimized_packet) {
            // Same destination register and address as the offset-0 byte load, no offset; the offset-1 byte shares its address operands.
            assert_eq!(operand(lo, 0), operand(wide, 0), "{symbol}: packet register");
            assert_eq!(operand(lo, 1), operand(wide, 1), "{symbol}: packet address register");
            assert_eq!(wide.split_once(' ').unwrap().1, lo.split_once(' ').unwrap().1, "{symbol}: packet address and base, no offset");
            assert!(operand(hi, 2).ends_with(" offset:1") && operand(hi, 1) == operand(lo, 1), "{symbol}: second byte is +1");
        }
        // All arithmetic is the same: the VALU stream differs only by the removed packet combine.
        let valu = |t: &str, skip_combine: bool| -> Vec<String> { t.lines().map(str::trim).filter(|l| mnemonic(l).starts_with("v_") && !(skip_combine && mnemonic(l) == "v_lshl_or_b32")).map(str::to_owned).collect() };
        assert_eq!(valu(&reference, true), valu(&opt, false), "{symbol}: arithmetic is identical");
    }
}

#[test]
fn codec_and_kind_constants_match_the_frozen_format() {
    assert_eq!((Codec::Mq3I4.tag(), Codec::Mq4Sym.tag()), ("mq3i4", "mq4sym"));
    assert_eq!((Kind::GateUp.tag(), Kind::Down.tag()), ("r256_gate_up", "r128_down"));
    assert_eq!((Kind::GateUp.chains(), Kind::Down.chains()), (4, 1));
    assert_eq!(mq_i4_moe_gemv::GROUP, 128);
    // The Mq4Sym gate lane owns eight elements of a G256 group; every other lane owns four.
    for spec in mq_i4_moe_gemv::module_specs(ARCH) {
        let wide = spec.codec == Codec::Mq4Sym && spec.kind == Kind::GateUp;
        assert_eq!(spec.lane_elems(), if wide { 8 } else { 4 }, "{spec:?}");
        assert_eq!(spec.group_instances(), if spec.kind == Kind::GateUp { 7 } else { 1 }, "{spec:?}");
    }
}

/// The module and every entry assemble with the native writer (no ROCm), and the bundle wraps the object.
#[test]
fn native_writer_assembles_every_entry_and_the_module() {
    let (emitted, text, _) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
    for e in &emitted {
        let elf = peacemaker_isa::native::assemble(&e.s_text, ARCH).unwrap_or_else(|err| panic!("{}: {err}", e.proof.kernel_id));
        assert!(elf.starts_with(b"\x7fELF"));
    }
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let bundle = peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(bundle.len() > elf.len());
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, native, pm_check, toolchain::{build, certify, oracle_assemble_link_bundle, IsaShapeContract, Toolchain}};

    const HOST: &str = "host-x86_64-unknown-linux-gnu-";

    /// Every symbol's waits and hazards replay independently of the builder's ledger (source and
    /// module text), the retained-replay dependency audit holds, the ABI is unchanged under
    /// replay, and M7 lifts the natively linked object byte-exactly with no outstanding
    /// obligations (no ROCm tools).
    #[test]
    fn every_export_passes_wait_hazard_replay_and_m7() {
        let (emitted, text, _) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
            assert!(e.s_text.contains(".kernarg_segment_size: 88\n"));
        }
        ledger_replay::replay_waits(&text, ARCH).unwrap();
        let dir = std::env::temp_dir().join(format!("peacemaker-isa-mqi4moe-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, ARCH).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 = pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Native code object and offload bundle against the ROCm oracle (`llvm-mc` + `ld.lld -shared`,
    /// `clang-offload-bundler`) for each of the four entries alone and for the module: field-by-field and
    /// byte-for-byte equal in everything except the intentionally different `.comment` stamp (the same
    /// comparison as `native_identity`, restricted to this family).
    #[test]
    fn native_code_objects_and_bundles_equal_the_rocm_oracle_for_all_entries_and_the_module() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let toolchain = Toolchain { host_target: HOST.into(), ..Toolchain::default() };
        let dir = std::env::temp_dir().join(format!("peacemaker-isa-mqi4moe-identity-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (emitted, module_text, _) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
        let mut units: Vec<(String, String)> = emitted.iter().zip(SYMBOLS).map(|(e, symbol)| (symbol.to_owned(), e.s_text.clone())).collect();
        units.push((format!("{MODULE} (module)"), module_text));
        let mut failures = Vec::new();
        for (i, (name, text)) in units.iter().enumerate() {
            let s = dir.join(format!("u{i}.s"));
            std::fs::write(&s, text).unwrap();
            let oracle = oracle_assemble_link_bundle(&toolchain, &s, &dir.join(format!("u{i}.hxaco")), ARCH.name()).unwrap();
            let (oracle_elf, oracle_bundle) = (std::fs::read(&oracle.elf).unwrap(), std::fs::read(&oracle.hsaco).unwrap());
            match native::assemble(text, ARCH) {
                Ok(elf) => {
                    let bundle = native::bundle(&elf, ARCH, HOST);
                    if let Some(why) = native_compare::compare_native_to_oracle(&elf, &bundle, &oracle_elf, &oracle_bundle) { failures.push(format!("{name}: {why}")) }
                }
                Err(e) => failures.push(format!("{name}: native: {e}")),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(failures.is_empty(), "{} of {} units differ from the oracle:\n{}", failures.len(), units.len(), failures.join("\n"));
    }

    /// Every symbol certifies against the ROCm oracle build: parse-back, independent wait replay,
    /// its shape contract (derived from the same fresh emission), and M7 of the linked ELF, obligation-free.
    #[test]
    fn module_passes_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = mq_i4_moe_gemv::emit_module(ARCH).unwrap();
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("peacemaker-isa-mqi4moe-cert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        for (spec, e) in mq_i4_moe_gemv::module_specs(ARCH).into_iter().zip(&emitted) {
            let symbol = spec.symbol();
            let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol).unwrap_or_else(|err| panic!("{symbol}: {err}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
            let tag = spec.tag();
            let contract: IsaShapeContract = serde_json::from_value(mq_i4_moe_gemv::contract(spec, e)).unwrap();
            certify(&toolchain, &build, &s, ARCH.name(), &dir.join(format!("{tag}.manifest.json")), Some(&contract), "test", "test")
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
