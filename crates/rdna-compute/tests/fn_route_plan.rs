// SPDX-License-Identifier: Apache-2.0
// Flash-Next walked route plan: CPU-only planner tests. No compiler process or HIP.
//
// `route_plans/qwen3.8-flash-next.gfx1201.tsv` is generated from module-load
// audits of the real dispatch (`scripts/route_plan_walk.py`). These tests check
// that every walked kernel is planned (walk ⊆ plan), that every planned HIP
// module is the release pack's object with the walked source (plan ⊆ pack),
// and that every load the walks did not cover refuses (lazy loading) instead
// of planning partially. Older Flash-Next traces (jit-g0, kernel-trace) ran
// other policies (Q8 QSA, host-mapped experts, an earlier dispatch) and are
// not a closure oracle for this route.
use rdna_compute::kernel_registry::{
    entries, route_entries, route_layout_sha256, walked_route_plans, KernelSpecRoute, KernelTensorMeta,
    PlannedKernel, RegistryError, RouteKernelInput, RouteKernelPlan, WalkedRoutePlan,
};
use rdna_compute::{DType, FeatureFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");

fn fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(Path::new(FIXTURES).join(name)).unwrap()).unwrap()
}

/// `hipfire-runtime` `loader_api::kernel_dtype_for_quant_type` for the tags a
/// Flash-Next header carries.
fn dtype(tag: u64) -> DType {
    match tag {
        3 => DType::Q8_0,
        16 => DType::BF16,
        44 => DType::MQ4G256V2,
        47 => DType::MQ6G256V2,
        53 => DType::MQ4G128V2,
        54 => DType::Raw,
        other => panic!("header fixture quant tag {other} has no test mapping"),
    }
}

/// The canonical Flash-Next GPTQ3 header (`jit-g0/header-flash-next.json`,
/// header SHA-256 8703bb16…, the artifact the gfx1201 walks loaded).
fn flash_next_header() -> Vec<(String, DType, Vec<usize>)> {
    fixture("jit-g0/header-flash-next.json")["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["name"].as_str().unwrap().to_owned(),
                dtype(t["quant_tag"].as_u64().unwrap()),
                t["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect(),
            )
        })
        .collect()
}

fn metas(header: &[(String, DType, Vec<usize>)]) -> Vec<KernelTensorMeta<'_>> {
    header.iter().map(|(name, dtype, shape)| KernelTensorMeta { name, dtype: *dtype, shape }).collect()
}

fn builtin_flags(arch: &str) -> FeatureFlags {
    let resolved = hipfire_config::resolve(std::iter::empty()).unwrap();
    let config = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
    FeatureFlags::from_process_config(arch, &config)
}

fn walked(arch: &str) -> &'static WalkedRoutePlan {
    walked_route_plans()
        .unwrap()
        .iter()
        .find(|plan| plan.model_arch == 16 && plan.arch == arch)
        .unwrap_or_else(|| panic!("no walked Flash-Next plan for {arch}"))
}

/// The walked axes, each at its first walked value (what the loader resolves
/// for the default route).
fn walked_axes(plan: &WalkedRoutePlan) -> Vec<(&'static str, String)> {
    plan.axes.iter().map(|(name, values)| (*name, values[0].to_owned())).collect()
}

struct Load<'a> {
    arch: &'a str,
    tensors: &'a [KernelTensorMeta<'a>],
    max_seq: usize,
    spec: KernelSpecRoute,
    flags: &'a FeatureFlags,
    axes: &'a [(&'a str, String)],
}

fn plan(load: &Load<'_>) -> Result<RouteKernelPlan, RegistryError> {
    let axis = |name: &str| load.axes.iter().find(|(n, _)| *n == name).map_or("", |(_, v)| v.as_str());
    route_entries(&RouteKernelInput {
        arch: load.arch,
        model_arch: 16,
        tensors: load.tensors,
        n_heads: 0,
        n_kv_heads: 0,
        head_dim: 0,
        max_seq: load.max_seq,
        prefill_chunk_rows: 0,
        kv_k: axis("qsa"),
        kv_v: axis("qsa"),
        state_quant: axis("gdn"),
        spec: load.spec,
        host_mapped_experts: false,
        retained_decode: true,
        hipcc_extra_flags: &load.flags.hipcc_extra_flags,
        flags: load.flags,
        route_axes: load.axes,
    })
}

/// `symbol -> (module, kind)` of a plan; one owner per symbol.
fn bindings(plan: &RouteKernelPlan) -> BTreeMap<&'static str, (&'static str, bool)> {
    let mut out = BTreeMap::new();
    for entry in &plan.entries {
        for &symbol in entry.symbols() {
            let embedded = matches!(entry, PlannedKernel::Embedded { .. });
            assert!(out.insert(symbol, (entry.module(), embedded)).is_none(), "symbol {symbol} planned twice");
        }
    }
    out
}

fn gfx1201_plans() -> (RouteKernelPlan, RouteKernelPlan) {
    let header = flash_next_header();
    let tensors = metas(&header);
    let walked = walked("gfx1201");
    let flags = builtin_flags("gfx1201");
    let axes = walked_axes(walked);
    let load = |spec| Load { arch: "gfx1201", tensors: &tensors, max_seq: 16_384, spec, flags: &flags, axes: &axes };
    let ar = plan(&load(KernelSpecRoute::Ar)).expect("Flash-Next AR route plan");
    let mtp = plan(&load(KernelSpecRoute::NativeMtp)).expect("Flash-Next native-MTP route plan");
    (ar, mtp)
}

#[test]
fn walked_layout_is_the_canonical_flash_next_header() {
    let header = flash_next_header();
    assert_eq!(route_layout_sha256(&metas(&header)), walked("gfx1201").layout);
}

/// walk ⊆ plan: every (module, symbol) the walked dispatch bound is planned
/// under its route with the walked identity, and nothing else is.
#[test]
fn every_walked_kernel_is_planned() {
    let (ar, mtp) = gfx1201_plans();
    let walked = walked("gfx1201");
    for (plan, native_mtp) in [(&ar, false), (&mtp, true)] {
        let rows: Vec<_> = walked.rows.iter().filter(|row| row.native_mtp == native_mtp).collect();
        assert_eq!(plan.entries.len(), rows.len(), "native_mtp={native_mtp}");
        for row in rows {
            let entry = plan
                .entries
                .iter()
                .find(|entry| entry.module() == row.module)
                .unwrap_or_else(|| panic!("walked {} is not planned", row.module));
            assert_eq!(entry.symbols(), &row.symbols[..], "{}", row.module);
            let sha = match entry {
                PlannedKernel::Hip(e) => format!("{:x}", Sha256::digest(e.source().as_bytes())),
                PlannedKernel::Embedded { image, .. } => format!("{:x}", Sha256::digest(image)),
            };
            assert_eq!(sha, row.sha256, "{}", row.module);
            assert_eq!(row.embedded, matches!(entry, PlannedKernel::Embedded { .. }), "{}", row.module);
        }
    }
    // The MTP route falls back to AR, so its plan carries every AR binding.
    let (ar, mtp) = (bindings(&ar), bindings(&mtp));
    for (symbol, owner) in &ar {
        assert_eq!(mtp.get(symbol), Some(owner), "AR binding {symbol} missing from the MTP plan");
    }
}

/// plan ⊆ pack: every planned HIP module is an installer-inventory entry (what
/// `scripts/build-kernel-pack.sh` ships) with the planned source, recipe and
/// every planned symbol in its index, so an installed pack satisfies the whole
/// plan with zero compiles.
#[test]
fn every_planned_hip_module_is_in_the_release_pack() {
    let (_, mtp) = gfx1201_plans();
    let flags = builtin_flags("gfx1201");
    let pack = entries("gfx1201", &flags.hipcc_extra_flags).unwrap();
    let mut hip = 0;
    for entry in &mtp.entries {
        let PlannedKernel::Hip(planned) = entry else { continue };
        let packed = pack
            .iter()
            .find(|e| e.module == planned.module)
            .unwrap_or_else(|| panic!("planned {} is not in the gfx1201 pack", planned.module));
        assert_eq!(packed.source(), planned.source(), "{}", planned.module);
        assert_eq!(packed.flags, planned.flags, "{}", planned.module);
        assert_eq!(packed.scheduler_profile, planned.scheduler_profile, "{}", planned.module);
        for symbol in planned.symbols {
            assert!(packed.symbols.contains(symbol), "pack index of {} lacks {symbol}", planned.module);
        }
        hip += 1;
    }
    assert!(hip > 0);
}

/// Loads with policies the walks did not cover get no plan (the caller keeps
/// lazy loading); a kernel the walks missed on a planned load warns and loads
/// (`scratch::route_load_tests`).
#[test]
fn unwalked_flash_next_policies_keep_lazy_loading() {
    let header = flash_next_header();
    let tensors = metas(&header);
    let walked = walked("gfx1201");
    let flags = builtin_flags("gfx1201");
    let axes = walked_axes(walked);
    let base = Load {
        arch: "gfx1201",
        tensors: &tensors,
        max_seq: walked.max_seq,
        spec: KernelSpecRoute::NativeMtp,
        flags: &flags,
        axes: &axes,
    };
    let refused = |load: &Load<'_>| matches!(plan(load), Err(RegistryError::UnsupportedRoute(_)));
    assert!(plan(&base).is_ok(), "the walked ceiling itself is admitted");
    assert!(refused(&Load { max_seq: walked.max_seq + 1, ..base }));
    for spec in [KernelSpecRoute::DFlash, KernelSpecRoute::DDTree, KernelSpecRoute::DSpark] {
        assert!(refused(&Load { spec, ..base }), "{spec:?}");
    }
    // Another header layout (a re-quantized or split artifact).
    let mut other = tensors.clone();
    other.pop();
    assert!(refused(&Load { tensors: &other, ..base }));
    // A policy axis outside the walks, a missing axis, an unknown axis.
    let mut q8 = axes.clone();
    q8.iter_mut().find(|(name, _)| *name == "qsa").unwrap().1 = "q8".into();
    assert!(refused(&Load { axes: &q8, ..base }));
    assert!(refused(&Load { axes: &axes[1..], ..base }));
    let mut extra = axes.clone();
    extra.push(("unwalked_axis", "on".into()));
    assert!(refused(&Load { axes: &extra, ..base }));
    // Any non-default feature flag, including kernel.pm_decode.
    let mut changed = flags.clone();
    changed.hipcc_extra_flags.push_str(" -DFN_ROUTE_TEST=1");
    assert!(refused(&Load { flags: &changed, ..base }));
    let mut pm = flags.clone();
    pm.pm_decode = !pm.pm_decode;
    assert!(refused(&Load { flags: &pm, ..base }));
    // Archs without a walked plan have no closed route.
    for arch in ["gfx1100", "gfx906"] {
        let arch_flags = builtin_flags(arch);
        assert!(refused(&Load { arch, flags: &arch_flags, ..base }), "{arch}");
    }
}
