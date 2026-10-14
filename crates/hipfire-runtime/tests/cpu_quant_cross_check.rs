// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Live cross-check of the host-side weight decoder over the **real** tensors of
//! the on-disk fixtures.
//!
//! `weight_backend::dequantize_to_f32` now delegates every block-quantized
//! format to `hipfire_cpu::quant::dequant_group`, so for those formats this test
//! compares the canonical decoder to itself. Its remaining two-sided value is
//! the formats `dequantize_to_f32` still decodes locally (qt 1/2/3/14/16). The
//! independent bit-exact oracle for the delegated formats is `hipfire-cpu`'s own
//! `EXPECTED_*` table test (`quant.rs`), whose tables were generated from the
//! canonical decoder before the cutover — keep both green.
//!
//! Not part of `scripts/no-gpu-ci.sh`'s curated list (it needs model files), and
//! it decodes real tensors, so run it optimized:
//!
//! ```text
//! cargo test --release -p hipfire-runtime --test cpu_quant_cross_check -- --nocapture
//! ```
//!
//! Fixtures are looked up in `$HIPFIRE_MODELS_DIR` (default `~/.hipfire/models`)
//! and a missing file is reported and skipped, like the `skip: no GPU` shape the
//! rest of the suite uses. A file that *is* present but yields no checkable
//! tensor is a failure: that would mean the check itself stopped exercising the
//! decode path.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Rows of each tensor checked. The group stride is fully exercised by one row
/// (all `k / group_elems` groups); the row stride by taking more than one, which
/// is why this is 4 and not 1. Bounded so the check stays fast on the 9B's
/// `[4096, 12288]` and `[12288, 4096]` tensors.
const CHECK_ROWS: usize = 4;

use hipfire_cpu::quant::{dequant_group, CpuQuant};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_backend::dequantize_weight_to_f32;

/// Registry fixtures, smallest first. The 2B variants are the per-format
/// fixtures (`mq3` = qt 17, `mq6` = qt 15, `hf6` = qt 8, `mq4` = qt 13).
const FIXTURES: [&str; 7] = [
    "qwen3.5-2b.mq4",
    "qwen3.5-2b.mq3",
    "qwen3.5-2b.mq6",
    "qwen3.5-2b.hf6",
    "qwen3.5-9b.mq4",
    "qwen3.8-27b.mq4",
    "ornith-1.5-35b-a3b.mq4",
];

fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("HIPFIRE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").unwrap_or_else(|| PathBuf::from("/").into());
    PathBuf::from(home).join(".hipfire").join("models")
}

#[test]
fn dequant_group_matches_the_canonical_decoder_on_real_tensors() {
    let dir = models_dir();
    let mut checked = 0usize;
    let mut files_seen = 0usize;
    let mut formats: BTreeSet<u8> = BTreeSet::new();

    for name in FIXTURES {
        let path = dir.join(name);
        if !path.exists() {
            eprintln!("skip: {} not present", path.display());
            continue;
        }
        files_seen += 1;
        let hfq = HfqFile::open(&path).expect("open fixture");
        for info in hfq.tensors() {
            let Some(q) = CpuQuant::from_quant_type(info.quant_type) else {
                continue;
            };
            let elems: usize = info.shape.iter().map(|d| *d as usize).product();
            if elems == 0 || elems % q.group_elems() != 0 {
                continue;
            }
            let (_, data) = hfq
                .tensor_data_vec(&info.name)
                .unwrap_or_else(|| panic!("{}: no bytes for {}", name, info.name));
            assert_eq!(
                data.len(),
                info.data_size,
                "{name}: {} data_size",
                info.name
            );

            // A 2-D tensor's byte count must be exactly `m` rows of the group
            // layout — the property `hipfire_cpu::gemv` derives its row pointer
            // from, checked here against the file rather than against itself.
            let m = if info.shape.len() == 2 {
                let rows = info.shape[0] as usize;
                assert_eq!(
                    data.len(),
                    rows * hipfire_cpu::gemv::row_bytes(q, elems / rows),
                    "{name}: {} byte count vs m * row_bytes",
                    info.name
                );
                rows
            } else {
                1
            };
            let k = elems / m;
            let (ge, gb) = (q.group_elems(), q.group_bytes());
            let rb = hipfire_cpu::gemv::row_bytes(q, k);

            let rows = m.min(CHECK_ROWS);
            let mut got = vec![0.0f32; k];
            for row in 0..rows {
                let row_bytes = &data[row * rb..(row + 1) * rb];
                // Canonical decoder over the row …
                let want = dequantize_weight_to_f32(info.quant_type, row_bytes, k);
                assert_eq!(want.len(), k, "{name}: {} decode length", info.name);

                // … versus the transcription, group by group.
                for g in 0..k / ge {
                    if q.row_header_bytes() == 0 {
                        dequant_group(q, &row_bytes[g * gb..], &mut got[g * ge..(g + 1) * ge]);
                    } else {
                        // E8 keeps its row scale once, before all FWHT groups.
                        let mut group = [0u8; 152];
                        group[..16].copy_from_slice(&row_bytes[..16]);
                        group[16..].copy_from_slice(&row_bytes[16 + g * gb..16 + (g + 1) * gb]);
                        dequant_group(q, &group, &mut got[g * ge..(g + 1) * ge]);
                    }
                }

                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{name}: {} (qt {}) row {row} element {i}: {a} (0x{:08x}) != {b} (0x{:08x})",
                        info.name,
                        info.quant_type,
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
            checked += 1;
            formats.insert(info.quant_type);
        }
    }

    eprintln!(
        "cpu_quant_cross_check: {checked} tensors over {files_seen} fixtures, quant types {formats:?}"
    );
    if files_seen == 0 {
        eprintln!("skip: no fixture found in {}", dir.display());
        return;
    }
    assert!(
        checked > 0,
        "no tensor in {files_seen} fixture(s) mapped to a CpuQuant — the cross-check \
         stopped exercising the decode path"
    );
    // Every fixture on disk is MQ4/MQ3/MQ6/HF6/F16/Q8, so a run against real
    // files must cover at least the rotated 4-bit path.
    assert!(
        formats.contains(&13) || formats.contains(&17),
        "expected at least one FWHT-rotated format among the checked tensors, got {formats:?}"
    );
}
/// Bit-exact equivalence of the canonical `dequantize_to_f32` arms for the
/// V2 MQ family (qt 44/45/47-50) against `hipfire_cpu::quant::dequant_group`,
/// on synthetic groups — the fixture files carry none of these dtypes, so the
/// live test above never exercises them. An fp16 header is built from
/// non-trivial bit patterns and the code payload cycles every valid code, so
/// both the scale/zero plumbing and the bit unpacking are covered.
fn f16le(v: f32) -> [u8; 2] {
    // Only need a stable, non-trivial bit pattern — the decoders widen it.
    let bits = half::f16_from_f32_bits(v);
    bits.to_le_bytes()
}

/// Minimal f32→f16 encoder (round-to-nearest-even) sufficient for the test's
/// chosen values — avoids pulling a half crate for one test.
mod half {
    pub fn f16_from_f32_bits(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let mant = b & 0x7fffff;
        if exp == 0 {
            return sign;
        }
        let e = exp - 127 + 15;
        if e >= 31 {
            return sign | 0x7c00;
        }
        if e <= 0 {
            // subnormal / underflow — flush through round-to-nearest.
            let m = mant | 0x800000;
            let shift = (14 - e) as u32;
            let mut m16 = (m >> shift) as u16;
            if (m >> (shift - 1)) & 1 == 1 {
                m16 = m16.wrapping_add(1);
            }
            return sign | m16;
        }
        let mut m16 = (mant >> 13) as u16;
        if mant & 0x1000 != 0 && (mant & 0x0fff != 0 || m16 & 1 == 1) {
            m16 = m16.wrapping_add(1);
        }
        sign | ((e as u16) << 10) | (m16 & 0x3ff)
    }
}

/// Build one synthetic group for `qt`, `v2_halves` layout `[s0 z0 s1 z1]` as
/// fp16 (qt45 reuses half-0's pair for both halves, per its single-grid spec).
fn synthetic_group(qt: u8) -> Vec<u8> {
    let (payload_len, group_bytes) = match qt {
        44 | 45 => (128, 136),
        47 => (192, 200),
        48 => (160, 168),
        49 => (96, 104),
        50 => (64, 72),
        _ => unreachable!(),
    };
    let max_code = match qt {
        47 => 63u32,
        48 => 31,
        49 => 7,
        50 => 3,
        _ => 15, // 44, 45
    };
    let mut g = vec![0u8; group_bytes];
    // Header: distinct half scale/zeros so per-half plumbing is exercised.
    // Chosen values are exactly representable in f16, so encode/decode is lossless.
    let (s0, z0, s1, z1) = (0.75f32, -0.5f32, 1.25f32, 0.25f32);
    g[0..2].copy_from_slice(&f16le(s0));
    g[2..4].copy_from_slice(&f16le(z0));
    g[4..6].copy_from_slice(&f16le(if qt == 45 { s0 } else { s1 }));
    g[6..8].copy_from_slice(&f16le(if qt == 45 { z0 } else { z1 }));

    // Codes 0..max_code cycling.
    let codes: Vec<u32> = (0..256).map(|i| (i as u32) % (max_code + 1)).collect();
    let p = &mut g[8..8 + payload_len];
    match qt {
        44 | 45 => {
            for (i, c) in codes.iter().enumerate() {
                let b = &mut p[i / 2];
                if i % 2 == 0 {
                    *b |= *c as u8;
                } else {
                    *b |= (*c as u8) << 4;
                }
            }
        }
        47 => {
            for ch in codes.chunks(4).enumerate() {
                let (bo, c) = (ch.0 * 3, ch.1);
                p[bo] = (c[0] | (c[1] << 6)) as u8;
                p[bo + 1] = ((c[1] >> 2) | (c[2] << 4)) as u8;
                p[bo + 2] = ((c[2] >> 4) | (c[3] << 2)) as u8;
            }
        }
        48 => {
            for ch in codes.chunks(8).enumerate() {
                let (bo, c) = (ch.0 * 5, ch.1);
                let w = (0..8).fold(0u64, |w, i| w | (c[i] as u64) << (5 * i));
                p[bo..bo + 5].copy_from_slice(&w.to_le_bytes()[..5]);
            }
        }
        49 => {
            for ch in codes.chunks(8).enumerate() {
                let (bo, c) = (ch.0 * 3, ch.1);
                let w = (0..8).fold(0u32, |w, i| w | c[i] << (3 * i));
                p[bo..bo + 3].copy_from_slice(&w.to_le_bytes()[..3]);
            }
        }
        50 => {
            for (i, c) in codes.iter().enumerate() {
                p[i / 4] |= (*c as u8) << (2 * (i % 4));
            }
        }
        _ => unreachable!(),
    }
    g
}

#[test]
fn dequantize_v2_family_matches_cpu_transcription_on_synthetic_groups() {
    for qt in [44u8, 45, 47, 48, 49, 50] {
        let q = CpuQuant::from_quant_type(qt)
            .unwrap_or_else(|| panic!("qt {qt} unmapped in CpuQuant"));
        let group = synthetic_group(qt);
        assert_eq!(group.len(), q.group_bytes(), "qt {qt} group size");

        // Two groups back-to-back so the canonical decoder's group loop and
        // the per-group stride are both exercised.
        let mut data = group.clone();
        data.extend_from_slice(&group);

        let want = dequantize_weight_to_f32(qt, &data, 512);
        assert_eq!(want.len(), 512, "qt {qt} decode length");

        let gb = q.group_bytes();
        let ge = q.group_elems();
        let mut got = vec![0.0f32; 512];
        for gidx in 0..2 {
            dequant_group(q, &data[gidx * gb..(gidx + 1) * gb], &mut got[gidx * ge..(gidx + 1) * ge]);
        }
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "qt {qt} element {i}: {a} (0x{:08x}) != {b} (0x{:08x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }
}
