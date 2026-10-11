// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Column-identity oracle of `Gpu::gemv_mq4g256v2_rows_x4` (MTP verify, 2..=8
//! activation rows of MQ4G256V2 trunk projections).
//!
//! Every output element of the few-row launch must be bitwise the AR decode
//! GEMV `Gpu::gemv_mq4g256v2` run on that activation row alone. Weights are
//! random valid MQ4G256V2 (finite normal f16 scale/zero per half group, random
//! nibbles) at the Qwen3.8-Flash-Next trunk shapes (M, K): GDN qkv (10240,
//! 2560), z (6144, 2560), a (48, 2560), b (48, 2560), out (2560, 6144); QSA q
//! (12288, 2560), k (512, 2560), v (512, 2560), indexer (640, 2560), o (2560,
//! 6144). For seeds 1..=3 and rows 1..=8:
//!
//!  (a) every matrix alone in slot 0 (the other slots m = 0);
//!  (b) the real groupings in one launch: GDN qkv|z|a|b and QSA q|k|v|indexer
//!      (K = 2560, four matrices), out|o (K = 6144, two matrices).
//!
//! Outputs are pre-filled with NaN with a NaN guard on both sides of every y
//! that must stay untouched; each case runs eagerly and replayed from an
//! instantiated HIP graph into freshly NaN-filled buffers. Comparison is
//! `to_bits()`. Negative controls (perturbed activation, swapped matrices)
//! must disagree, so a vacuous pass is impossible.
//!
//! `cargo run --release -p rdna-compute --features lab --example mq4_rows_oracle`
//! Exits nonzero on any mismatch.

use rdna_compute::{Gpu, GpuTensor};

const GUARD: usize = 64;
const MAX_ROWS: usize = 8;
const NAN_BITS: u32 = 0x7fc0_0000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// Host packer: `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0),
/// hB (s1, z1), then 128 bytes of nibbles (element e at byte e / 2, low
/// nibble = even e).
fn pack_random_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 136);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            out.extend_from_slice(&random_scale(rng).to_le_bytes());
            out.extend_from_slice(&random_zero(rng).to_le_bytes());
        }
        for _ in 0..128 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}

struct Mat {
    name: &'static str,
    m: usize,
    k: usize,
    w: GpuTensor,
    /// Singleton reference, `[MAX_ROWS][m]` result bits; row r is
    /// `gemv_mq4g256v2` on activation row r. Rows `..R` are the reference of R rows.
    reference: Vec<u32>,
}

const SHAPES: [(&str, usize, usize); 10] = [
    ("gdn.qkv", 10240, 2560),
    ("gdn.z", 6144, 2560),
    ("gdn.a", 48, 2560),
    ("gdn.b", 48, 2560),
    ("gdn.out", 2560, 6144),
    ("qsa.q", 12288, 2560),
    ("qsa.k", 512, 2560),
    ("qsa.v", 512, 2560),
    ("qsa.indexer", 640, 2560),
    ("qsa.o", 2560, 6144),
];

/// The real groupings, by index into `SHAPES`.
const GROUPS: [(&str, &[usize]); 3] = [
    ("group gdn qkv|z|a|b K=2560", &[0, 1, 2, 3]),
    ("group qsa q|k|v|indexer K=2560", &[5, 6, 7, 8]),
    ("group out|o K=6144", &[4, 9]),
];

fn nan_bytes(words: usize) -> Vec<u8> {
    NAN_BITS.to_le_bytes().repeat(words)
}

fn fill_nan(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip.device_synchronize().expect("sync before NaN fill");
    gpu.hip
        .memcpy_htod(&t.buf, &nan_bytes(t.numel()))
        .expect("NaN fill");
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `[GUARD NaN][expected][GUARD NaN]`.
fn guarded(expected: &[u32]) -> Vec<u32> {
    let mut v = vec![NAN_BITS; GUARD];
    v.extend_from_slice(expected);
    v.extend(std::iter::repeat_n(NAN_BITS, GUARD));
    v
}

fn mismatches(got: &[u32], want: &[u32]) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

fn upload_x(gpu: &mut Gpu, k: usize, rng: &mut Rng) -> GpuTensor {
    let x: Vec<f32> = (0..MAX_ROWS * k)
        .map(|_| (rng.unit() + rng.unit() - 1.0) * 1.5)
        .collect();
    gpu.upload_f32(&x, &[x.len()]).expect("upload x")
}

/// One few-row launch of `mats` (<= 4 parts) over `rows` activation rows into
/// NaN-prefilled guarded outputs; returns each output tensor's full bits.
/// `graph`: capture the launch into a HIP graph, refill NaN, and replay it.
fn launch(gpu: &mut Gpu, mats: &[&Mat], x: &GpuTensor, rows: usize, graph: bool) -> Vec<Vec<u32>> {
    let k = mats[0].k;
    let outs: Vec<GpuTensor> = mats
        .iter()
        .map(|mat| {
            let n = GUARD + rows * mat.m + GUARD;
            gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n]).expect("alloc y")
        })
        .collect();
    let views: Vec<GpuTensor> = mats
        .iter()
        .zip(&outs)
        .map(|(mat, y)| y.sub_offset(GUARD, rows * mat.m))
        .collect();
    let part = |i: usize| {
        if i < mats.len() {
            (&mats[i].w, &views[i], mats[i].m)
        } else {
            (&mats[0].w, &views[0], 0)
        }
    };
    let parts = [part(0), part(1), part(2), part(3)];
    if graph {
        gpu.ensure_capture_stream().expect("capture stream");
        gpu.hip.device_synchronize().expect("sync");
        gpu.begin_stream_capture().expect("begin capture");
        let captured = gpu.gemv_mq4g256v2_rows_x4(parts, x, k, rows);
        let g = gpu.end_stream_capture().expect("end capture");
        captured.expect("captured launch");
        let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
        for y in &outs {
            fill_nan(gpu, y);
        }
        gpu.launch_graph(&exec).expect("graph replay");
        gpu.hip.device_synchronize().expect("sync");
        gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
        gpu.hip.graph_destroy(g).expect("destroy graph");
    } else {
        gpu.gemv_mq4g256v2_rows_x4(parts, x, k, rows).expect("rows x4 launch");
    }
    let bits = outs.iter().map(|y| fetch(gpu, y)).collect();
    for y in outs {
        gpu.free_tensor(y).expect("free y");
    }
    bits
}

/// Compare a launch against per-matrix expected row bits (`rows * m` each).
fn compare(got: &[Vec<u32>], want: &[Vec<u32>]) -> usize {
    got.iter().zip(want).map(|(g, w)| mismatches(g, &guarded(w))).sum()
}

fn expected(mats: &[&Mat], rows: usize) -> Vec<Vec<u32>> {
    mats.iter().map(|m| m.reference[..rows * m.m].to_vec()).collect()
}

/// The AR singleton on every activation row of `x` into its own buffer.
fn singleton_reference(gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, m: usize, k: usize) -> Vec<u32> {
    let y = gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); MAX_ROWS * m], &[MAX_ROWS * m]).expect("alloc ref");
    for r in 0..MAX_ROWS {
        gpu.gemv_mq4g256v2(w, &x.sub_offset(r * k, k), &y.sub_offset(r * m, m), m, k)
            .expect("singleton gemv_mq4g256v2");
    }
    let bits = fetch(gpu, &y);
    gpu.free_tensor(y).expect("free ref");
    assert!(
        bits.iter().all(|&b| f32::from_bits(b).is_finite()),
        "singleton reference is not finite: the random weights/activations are invalid"
    );
    bits
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    let flag = hipfire_config::developer_bool("HIPFIRE_QWEN4_MQ4_ROWS_PM", true);
    let pm = flag && matches!(arch.as_str(), "gfx1151" | "gfx1201");
    println!("mq4_rows_oracle: arch={arch}");
    println!(
        "mq4_rows_oracle: route={} (HIPFIRE_QWEN4_MQ4_ROWS_PM {})",
        if pm { "embedded PM image mq4_rows_x4_pm" } else { "per-row gemv_mq4g256v2 fallback" },
        if flag { "on" } else { "0" }
    );
    let mut cases = 0usize;
    let mut bad_words = 0usize;
    let mut summary: Vec<(String, usize, usize)> = Vec::new();

    for seed in 1..=3u64 {
        let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407));
        let x2560 = upload_x(&mut gpu, 2560, &mut rng);
        let x6144 = upload_x(&mut gpu, 6144, &mut rng);
        let x_for = |k: usize| if k == 2560 { &x2560 } else { &x6144 };
        let mut mats: Vec<Mat> = Vec::new();
        for &(name, m, k) in &SHAPES {
            let bytes = pack_random_mq4g256v2(m, k, &mut rng);
            let w = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload weights");
            let reference = singleton_reference(&mut gpu, &w, x_for(k), m, k);
            mats.push(Mat { name, m, k, w, reference });
        }

        // (a) one matrix per launch, (b) the real groupings.
        let mut work: Vec<(String, Vec<usize>)> = SHAPES
            .iter()
            .enumerate()
            .map(|(i, &(name, m, k))| (format!("single {name} M={m} K={k}"), vec![i]))
            .collect();
        work.extend(GROUPS.iter().map(|&(name, idx)| (name.to_string(), idx.to_vec())));
        for (label, idx) in &work {
            let group: Vec<&Mat> = idx.iter().map(|&i| &mats[i]).collect();
            let x = x_for(group[0].k);
            let (mut case_runs, mut case_bad) = (0usize, 0usize);
            for rows in 1..=MAX_ROWS {
                for graph in [false, true] {
                    let got = launch(&mut gpu, &group, x, rows, graph);
                    let bad = compare(&got, &expected(&group, rows));
                    if bad != 0 {
                        eprintln!(
                            "MISMATCH seed={seed} {label} rows={rows} {}: {bad} words differ",
                            if graph { "graph" } else { "eager" }
                        );
                    }
                    case_runs += 1;
                    case_bad += bad;
                }
            }
            cases += case_runs;
            bad_words += case_bad;
            match summary.iter_mut().find(|(l, _, _)| l == label) {
                Some(entry) => {
                    entry.1 += case_runs;
                    entry.2 += case_bad;
                }
                None => summary.push((label.clone(), case_runs, case_bad)),
            }
        }

        // Negative controls on the GDN grouping at 3 rows: the launch must
        // disagree with a reference computed from a perturbed activation, and
        // gdn.a's output must disagree with gdn.b's reference (swapped matrices).
        if seed == 1 {
            let group: Vec<&Mat> = GROUPS[0].1.iter().map(|&i| &mats[i]).collect();
            let rows = 3;
            let got = launch(&mut gpu, &group, &x2560, rows, false);
            let mut host_x = gpu.download_f32(&x2560).expect("download x");
            host_x[0] += 1.0;
            let x_perturbed = gpu.upload_f32(&host_x, &[host_x.len()]).expect("upload x'");
            let perturbed: Vec<Vec<u32>> = group
                .iter()
                .map(|mat| {
                    singleton_reference(&mut gpu, &mat.w, &x_perturbed, mat.m, mat.k)[..rows * mat.m].to_vec()
                })
                .collect();
            let perturbed_bad: Vec<usize> = got
                .iter()
                .zip(&perturbed)
                .map(|(g, w)| mismatches(g, &guarded(w)))
                .collect();
            let (a, b) = (&group[2], &group[3]);
            assert_eq!((a.name, b.name, a.m, b.m), ("gdn.a", "gdn.b", 48, 48));
            let swapped_bad = mismatches(&got[2], &guarded(&b.reference[..rows * b.m]));
            println!(
                "mq4_rows_oracle: negative control perturbed-x mismatch words per output {perturbed_bad:?}, swapped a/b {swapped_bad}"
            );
            if perturbed_bad.iter().any(|&n| n == 0) || swapped_bad == 0 {
                eprintln!("negative control FAILED: a wrong reference agreed with the launch (vacuous oracle)");
                std::process::exit(1);
            }
        }

        for mat in mats {
            gpu.free_tensor(mat.w).expect("free weights");
        }
        gpu.free_tensor(x2560).expect("free x");
        gpu.free_tensor(x6144).expect("free x");
    }

    println!("mq4_rows_oracle: per-shape summary (runs = seeds 1..=3 x rows 1..=8 x eager+graph)");
    for (label, runs, bad) in &summary {
        println!("  {label:<34} runs={runs:<4} mismatch_words={bad}");
    }
    if bad_words != 0 {
        println!("mq4_rows_oracle: FAIL cases={cases} mismatch_bytes={}", bad_words * 4);
        std::process::exit(1);
    }
    println!("mq4_rows_oracle: PASS cases={cases} mismatch_bytes=0");
}
