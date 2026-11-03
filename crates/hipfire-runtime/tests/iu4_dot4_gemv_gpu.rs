// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerics gate for the W4A8 IU4G256 decode kernel (`gemv_iu4g256_dot4`,
//! `HIPFIRE_IU4_DECODE=dot4`, `v_dot4_i32_iu8`).
//!
//! 1. Contract: matches the CPU `iu_quant::gemv_iu4_dot4` (integer dot is
//!    exact; only the f32 combine order may differ).
//! 2. Reference: error vs an f64 dot of the dequantized weights with the
//!    unquantized x, side by side with the f32 kernel and the W4A4 dot8 one.
//! 3. Residual and two-run bit determinism.
//!
//! M in {1, 7, 129, 4097}, K in {256, 512, 768, 2560, 9216}; uniform and
//! heavy-tailed x. Optional real tensors: `IU4_REAL=<model.iu4p>`.
//!
//!   cargo test -p hipfire-runtime --release --test iu4_dot4_gemv_gpu -- --ignored --nocapture

use half::f16;
use hipfire_runtime::iu_quant::{dequant, gemv_iu4_dot4, repack_mq4v2, Layout};
use rdna_compute::{DType, Gpu};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn mq4v2_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let g = k / 256;
    let mut out = vec![0u8; m * g * 136];
    for blk in out.chunks_mut(136) {
        for h in 0..2 {
            let sc = 0.002 + 0.03 * rng.f().abs();
            let zp = -8.0 * sc + 0.01 * rng.f();
            blk[h * 4..h * 4 + 2].copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
            blk[h * 4 + 2..h * 4 + 4].copy_from_slice(&f16::from_f32(zp).to_bits().to_le_bytes());
        }
        for b in &mut blk[8..] {
            *b = rng.next() as u8;
        }
    }
    out
}

/// rel = max abs error / max |reference|.
fn rel(got: &[f32], want: &[f64]) -> f64 {
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0, f64::max);
    max_abs / want.iter().fold(0.0f64, |a, b| a.max(b.abs())).max(1e-12)
}

fn run(
    gpu: &mut Gpu,
    which: &str,
    w: &rdna_compute::GpuTensor,
    xd: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    seed: Option<&[f32]>,
) -> Vec<f32> {
    let y = match seed {
        Some(s) => gpu.upload_f32(s, &[m]).unwrap(),
        None => gpu.alloc_tensor(&[m], DType::F32).unwrap(),
    };
    let res = seed.is_some();
    match (which, res) {
        ("dot4", r) => gpu.gemv_iu4g256_dot4(w, xd, &y, m, k, r).unwrap(),
        ("dot4q", r) => gpu.gemv_iu4g256_dot4q(w, xd, &y, m, k, r).unwrap(),
        ("dot8", r) => gpu.gemv_iu4g256_dot8(w, xd, &y, m, k, r).unwrap(),
        ("f32", false) => gpu.gemv_iu4g256(w, xd, &y, m, k).unwrap(),
        ("f32", true) => gpu.gemv_iu4g256_residual(w, xd, &y, m, k).unwrap(),
        _ => unreachable!(),
    }
    gpu.hip.device_synchronize().unwrap();
    let out = gpu.download_f32(&y).unwrap();
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{which} m={m} k={k}: non-finite output"
    );
    gpu.free_tensor(y).ok();
    out
}

struct Worst {
    contract: f64,
    d4: f64,
    d8: f64,
    f32_: f64,
    n: usize,
}

fn check(gpu: &mut Gpu, label: &str, iu: &[u8], x: &[f32], m: usize, k: usize, w: &mut Worst) {
    let wd = gpu.upload_raw(iu, &[iu.len()]).unwrap();
    let xd = gpu.upload_f32(x, &[k]).unwrap();
    let d4 = run(gpu, "dot4", &wd, &xd, m, k, None);
    let d4b = run(gpu, "dot4", &wd, &xd, m, k, None);
    let d8 = run(gpu, "dot8", &wd, &xd, m, k, None);
    let f = run(gpu, "f32", &wd, &xd, m, k, None);
    let seed: Vec<f32> = (0..m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let d4r = run(gpu, "dot4", &wd, &xd, m, k, Some(&seed));
    // v2 (x quantized once): bit-identical to v1, plain and residual, and
    // the fused 3-matrix launch (third matrix = first ceil(m/2) rows) is
    // bit-identical to the per-matrix kernel.
    let q = run(gpu, "dot4q", &wd, &xd, m, k, None);
    let qb = run(gpu, "dot4q", &wd, &xd, m, k, None);
    let qr = run(gpu, "dot4q", &wd, &xd, m, k, Some(&seed));
    let m2 = m.div_ceil(2);
    let ys: Vec<_> = [m, m, m2]
        .iter()
        .map(|&mm| gpu.alloc_tensor(&[mm], DType::F32).unwrap())
        .collect();
    gpu.fused_iu4g256_rows_impl(
        &[(&wd, &ys[0], m), (&wd, &ys[1], m), (&wd, &ys[2], m2)],
        &xd,
        k,
        true,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let fz: Vec<Vec<f32>> = ys.iter().map(|y| gpu.download_f32(y).unwrap()).collect();
    for y in ys {
        gpu.free_tensor(y).ok();
    }
    let bits = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    let q_v1 = bits(&q, &d4);
    let qr_v1 = bits(&qr, &d4r);
    let q_det = bits(&q, &qb);
    let fused = bits(&fz[0], &q) + bits(&fz[1], &q) + bits(&fz[2], &q[..m2]);
    eprintln!("{label:<28} m={m:<6} k={k:<5} dot4q!=dot4v1 {q_v1} resid {qr_v1} fused!=plain {fused} nondet {q_det}");
    assert_eq!(q_v1 + qr_v1 + q_det + fused, 0, "{label}: dot4q parity");

    let cpu = gemv_iu4_dot4(iu, x, m, k).unwrap();
    let cpu64: Vec<f64> = cpu.iter().map(|&v| v as f64).collect();
    let wq = dequant(iu, m, k, Layout::Iu4).unwrap();
    let refv: Vec<f64> = (0..m)
        .map(|r| (0..k).map(|c| wq[r * k + c] as f64 * x[c] as f64).sum())
        .collect();
    let refr: Vec<f64> = refv.iter().zip(&seed).map(|(a, s)| a + *s as f64).collect();

    let r_con = rel(&d4, &cpu64);
    let (r4, r8, rf) = (rel(&d4, &refv), rel(&d8, &refv), rel(&f, &refv));
    // Residual = plain + seed (same body, ADD only in the store): compare to
    // d4 + seed directly, scale = |d4| + |seed| per row.
    let r4r = d4r
        .iter()
        .zip(&d4)
        .zip(&seed)
        .fold(0f64, |a, ((&g, &p), &s)| {
            a.max(
                (g as f64 - (p as f64 + s as f64)).abs()
                    / (p.abs() as f64 + s.abs() as f64).max(1e-12),
            )
        });
    let _ = &refr;
    let det = d4
        .iter()
        .zip(&d4b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("{label:<28} m={m:<6} k={k:<5} dot4_vs_cpu={r_con:.2e} | vs f64 ref: f32={rf:.2e} dot4={r4:.2e} dot8={r8:.2e} dot4_resid={r4r:.2e} nondet={det}");
    assert!(r_con < 1e-5, "{label}: dot4 vs CPU contract {r_con:.3e}");
    assert_eq!(det, 0, "{label}: dot4 not deterministic");
    assert!(r4r < 1e-6, "{label}: residual != plain + seed ({r4r:.3e})");
    // dot8 vs f64 is reported, not gated: it measures int4 activation-quant
    // error (0.4 on heavy-tailed x at m=1), not kernel correctness. dot8's
    // contract gate is `gemv_iu4_dot8` in iu_quant_gemv_gpu.rs.
    w.contract = w.contract.max(r_con);
    w.d4 = w.d4.max(r4);
    w.d8 = w.d8.max(r8);
    w.f32_ = w.f32_.max(rf);
    w.n += 1;
    gpu.free_tensor(wd).ok();
    gpu.free_tensor(xd).ok();
}

#[test]
#[ignore]
fn iu4_dot4_gemv_matches_contract() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut w = Worst {
        contract: 0.0,
        d4: 0.0,
        d8: 0.0,
        f32_: 0.0,
        n: 0,
    };
    for &m in &[1usize, 7, 129, 4097] {
        for &k in &[256usize, 512, 768, 2560, 9216] {
            if m * k > 4097 * 2560 {
                continue;
            }
            let mq = mq4v2_random(&mut rng, m, k);
            let iu = repack_mq4v2(&mq, m, k, Layout::Iu4).unwrap();
            let x: Vec<f32> = (0..k).map(|_| rng.f() * 3.0).collect();
            check(&mut gpu, "random-uniform", &iu, &x, m, k, &mut w);
            // Heavy tail: 1 in 64 entries x30 (outlier channels).
            let xh: Vec<f32> = (0..k)
                .map(|i| rng.f() * if i % 64 == 5 { 30.0 } else { 1.0 })
                .collect();
            check(&mut gpu, "random-heavy", &iu, &xh, m, k, &mut w);
        }
    }
    if let Ok(iu_path) = std::env::var("IU4_REAL") {
        use hipfire_runtime::hfq::HfqFile;
        let f = HfqFile::open(std::path::Path::new(&iu_path)).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        let mut done = 0;
        for t in f.tensors().to_vec() {
            let (ti, ib) = f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 57 || ti.shape.len() < 2 {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if !seen.insert((m, k)) {
                continue;
            }
            let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
            let label = format!(
                "real:{}",
                t.name
                    .chars()
                    .rev()
                    .take(23)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
            );
            check(&mut gpu, &label, ib, &x, m, k, &mut w);
            done += 1;
            if done >= 8 {
                break;
            }
        }
        eprintln!("real tensors checked: {done}");
    }
    eprintln!(
        "iu4 dot4: {} cases; worst dot4 vs CPU contract {:.3e}; worst vs f64 ref: f32 {:.3e}, dot4 {:.3e}, dot8 {:.3e}",
        w.n, w.contract, w.f32_, w.d4, w.d8
    );
}
