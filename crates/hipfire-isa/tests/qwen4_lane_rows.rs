// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1151 lane-step row module `qwen4_lane_rows_pm`: frozen exports, the
//! kernarg ABI and launch bounds of every export, determinism, exact-arch
//! gating, the committed bundle's identity with fresh native emission, and
//! (feature `toolchain`) M7 on every export of the linked object.
use hipfire_isa::Arch;
use hipfire_isa::kernels::qwen4_lane_rows::{self, KERNARGS, KERNARG_BYTES, MODULE, SYMBOLS, WORKGROUP_SIZE};

const ARCH: Arch = Arch::Gfx1151;

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines().filter_map(|l| l.trim().strip_prefix(key)).map(|v| v.trim().parse().unwrap()).collect()
}

#[test]
fn module_exports_are_the_four_frozen_symbols_in_order() {
    assert_eq!(MODULE, "qwen4_lane_rows_pm");
    assert_eq!(
        SYMBOLS,
        ["gemv_mq6g256v2_xbatch_pm", "gemv_mq6g256v2_x4_xbatch_pm", "gemv_q8_0_k8_xbatch_pm", "gemv_q8_0_k320_xbatch_pm"]
    );
    assert_eq!(qwen4_lane_rows::bundle_name(), "qwen4_lane_rows_pm_gfx1151.hxaco");
    let (emitted, text, proof) = qwen4_lane_rows::emit_module().unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(text.contains(&format!(".globl {symbol}\n")), "{symbol}: missing from module source");
        assert!(text.contains(&format!(".name: {symbol}\n")), "{symbol}: missing from module metadata");
    }
}

#[test]
fn emission_is_deterministic() {
    let (_, ta, pa) = qwen4_lane_rows::emit_module().unwrap();
    let (_, tb, pb) = qwen4_lane_rows::emit_module().unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
}

/// Every export is built for exact gfx1151 and wave32 without scratch.
#[test]
fn every_export_is_exact_gfx1151_wave32_without_scratch() {
    let (emitted, text, _) = qwen4_lane_rows::emit_module().unwrap();
    for (e, symbol) in emitted.iter().zip(SYMBOLS) {
        assert_eq!(e.proof.arch, ARCH, "{symbol}");
        assert!(e.s_text.contains(".amdhsa_wavefront_size32 1"), "{symbol}: wave32");
        assert!(e.s_text.contains(".private_segment_fixed_size: 0\n"), "{symbol}: private segment");
        assert!(e.s_text.contains(".sgpr_spill_count: 0\n") && e.s_text.contains(".vgpr_spill_count: 0\n"), "{symbol}: spills");
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1151"));
}

/// The kernarg block, per-argument offsets/sizes and the launch bound of every
/// export are the frozen contract's.
#[test]
fn kernarg_abi_and_launch_bounds_match_the_contract() {
    let (emitted, _, _) = qwen4_lane_rows::emit_module().unwrap();
    for (i, (e, symbol)) in emitted.iter().zip(SYMBOLS).enumerate() {
        let size = KERNARG_BYTES[i];
        assert!(e.s_text.contains(&format!(".kernarg_segment_size: {size}\n")), "{symbol}: kernarg size {size}");
        let want: Vec<u32> = KERNARGS[i].iter().map(|a| a.0).collect();
        assert_eq!(field(&e.s_text, ".offset:"), want, "{symbol}: kernarg offsets");
        let want: Vec<u32> = KERNARGS[i].iter().map(|a| a.1).collect();
        assert_eq!(field(&e.s_text, ".size:"), want, "{symbol}: kernarg sizes");
        assert_eq!(field(&e.s_text, ".max_flat_workgroup_size:"), [WORKGROUP_SIZE[i]], "{symbol}: workgroup size");
        let (last_off, last_size) = *KERNARGS[i].last().unwrap();
        assert_eq!(last_off + last_size, size, "{symbol}: arguments cover the kernarg block");
    }
    // The frozen table itself: 36 / 96 / 44 / 32 bytes; 32 / 32 / 256 / 64 threads.
    assert_eq!(KERNARG_BYTES, [36, 96, 44, 32]);
    assert_eq!(WORKGROUP_SIZE, [32, 32, 256, 64]);
}

/// The native writer assembles the module and bundles it for the gfx1151 host entry.
#[test]
fn module_assembles_natively() {
    let (_, text, _) = qwen4_lane_rows::emit_module().unwrap();
    let elf = hipfire_isa::native::assemble(&text, ARCH).unwrap();
    let bundle = hipfire_isa::native::bundle(&elf, ARCH, hipfire_isa::native::DEFAULT_HOST_TARGET);
    assert!(bundle.len() > elf.len());
}

/// The runtime embeds `kernels/qwen4_lane_rows_pm_gfx1151.hxaco`: it must be
/// exactly what the builders emit today (native writer, no ROCm needed).
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = format!("{}/../../kernels/{}", env!("CARGO_MANIFEST_DIR"), qwen4_lane_rows::bundle_name());
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: hipfire-isa emit --kernel qwen4_lane_rows --arch gfx1151 ...)"));
    let (_, text, _) = qwen4_lane_rows::emit_module().unwrap();
    let elf = hipfire_isa::native::assemble(&text, ARCH).unwrap();
    let fresh = hipfire_isa::native::bundle(&elf, ARCH, hipfire_isa::native::DEFAULT_HOST_TARGET);
    assert!(committed == fresh, "{MODULE}: fresh emission differs from the committed bundle {path}");
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check};

    /// Every export's waits replay independently, and M7 lifts the linked
    /// object byte-exactly with no outstanding obligations.
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = qwen4_lane_rows::emit_module().unwrap();
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-lane-rows-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, hipfire_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 = pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}");
        }
    }
}
