// SPDX-License-Identifier: Apache-2.0
//! Numerics gate for the IU8G32 (qt59) decode GEMV v2 (`gemv_iu8g32_v2_*`).
//!
//! Every v2 variant (R = 1, 2, 4; plain and residual) vs an f64 reference of
//! the dequantized row dot f32 x, next to v1 on the same case:
//! rel = max abs / max_row sum|w x|. Gate: v2 rel <= max(4 * v1 rel, 2e-6)
//! and < 1e-4; v2 run twice is bit-identical. Real tensors with
//! `IU8_REAL=<.iu4p>` (embedding/lm_head and every qt59 shape).
//!
//!   cargo test -p hipfire-runtime --release --test iu8g32_v2_gemv_gpu -- --ignored --nocapture

use half::f16;
use rdna_compute::Gpu;

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

fn iu8_random(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 128, 0, "qt59 rows are one scale per 128");
    let g = k / 128;
    let mut out = vec![0u8; m * (k + 2 * g)];
    for r in 0..m {
        let row = &mut out[r * (k + 2 * g)..(r + 1) * (k + 2 * g)];
        for b in &mut row[..k] {
            *b = (rng.next() & 0xff) as u8;
        }
        for b in 0..g {
            let sc = 0.001 + 0.02 * rng.f().abs();
            row[k + 2 * b..k + 2 * b + 2]
                .copy_from_slice(&f16::from_f32(sc).to_bits().to_le_bytes());
        }
    }
    out
}

fn reference(iu: &[u8], x: &[f32], m: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let g = k / 128;
    let rb = k + 2 * g;
    let mut y = vec![0f64; m];
    let mut l1 = vec![0f64; m];
    for r in 0..m {
        let row = &iu[r * rb..(r + 1) * rb];
        let (mut s, mut a) = (0f64, 0f64);
        for b in 0..g {
            let sc =
                f16::from_bits(u16::from_le_bytes([row[k + 2 * b], row[k + 2 * b + 1]])).to_f64();
            for i in 0..128 {
                let v = sc * (row[b * 128 + i] as i8) as f64 * x[b * 128 + i] as f64;
                s += v;
                a += v.abs();
            }
        }
        y[r] = s;
        l1[r] = a;
    }
    (y, l1)
}

const V2: [(&str, &str); 3] = [
    ("gemv_iu8g32_v2_r1", "gemv_iu8g32_v2_residual_r1"),
    ("gemv_iu8g32_v2_r2", "gemv_iu8g32_v2_residual_r2"),
    ("gemv_iu8g32_v2_r4", "gemv_iu8g32_v2_residual_r4"),
];

fn run(
    gpu: &mut Gpu,
    f: &'static str,
    w: &rdna_compute::GpuTensor,
    x: &rdna_compute::GpuTensor,
    seed: &[f32],
    m: usize,
    k: usize,
) -> Vec<f32> {
    let y = gpu.upload_f32(seed, &[m]).unwrap();
    gpu.gemv_iu8g32_named(f, w, x, &y, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let out = gpu.download_f32(&y).unwrap();
    gpu.free_tensor(y).ok();
    out
}

fn rel_of(got: &[f32], seed: &[f32], add: bool, want: &[f64], l1: &[f64]) -> f64 {
    let mut abs = 0f64;
    let mut sc = 1e-12f64;
    for r in 0..want.len() {
        let w = want[r] + if add { seed[r] as f64 } else { 0.0 };
        abs = abs.max((got[r] as f64 - w).abs());
        sc = sc.max(l1[r]);
    }
    abs / sc
}

fn check(gpu: &mut Gpu, label: &str, iu: &[u8], m: usize, k: usize, rng: &mut Rng) -> (f64, f64) {
    let x: Vec<f32> = (0..k).map(|_| rng.f() * 2.0).collect();
    let seed: Vec<f32> = (0..m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
    let zero = vec![0f32; m];
    let w = gpu.upload_raw(iu, &[iu.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[k]).unwrap();
    let (want, l1) = reference(iu, &x, m, k);
    let v1 = rel_of(
        &run(gpu, "gemv_iu8g32", &w, &xd, &zero, m, k),
        &zero,
        false,
        &want,
        &l1,
    );
    let v1r = rel_of(
        &run(gpu, "gemv_iu8g32_residual", &w, &xd, &seed, m, k),
        &seed,
        true,
        &want,
        &l1,
    );
    let mut worst = 0f64;
    let aligned = (k + k / 64) % 16 == 0;
    if aligned {
        for (p, r) in V2 {
            let a = run(gpu, p, &w, &xd, &zero, m, k);
            let a2 = run(gpu, p, &w, &xd, &zero, m, k);
            assert!(
                a.iter().zip(&a2).all(|(u, v)| u.to_bits() == v.to_bits()),
                "{label} {p}: nondeterministic"
            );
            let rp = rel_of(&a, &zero, false, &want, &l1);
            let rr = rel_of(&run(gpu, r, &w, &xd, &seed, m, k), &seed, true, &want, &l1);
            for (n, v, base) in [(p, rp, v1), (r, rr, v1r)] {
                assert!(
                    v < 1e-4 && v <= f64::max(4.0 * base, 2e-6),
                    "{label} m={m} k={k} {n}: rel {v:.3e} (v1 {base:.3e})"
                );
                worst = worst.max(v);
            }
        }
    }
    eprintln!(
        "{label:<28} m={m:<7} k={k:<5} v1={v1:.3e} v1r={v1r:.3e} v2_worst={worst:.3e}{}",
        if aligned {
            ""
        } else {
            " (v2 n/a: unaligned rows -> v1)"
        }
    );
    gpu.free_tensor(w).ok();
    gpu.free_tensor(xd).ok();
    (v1.max(v1r), worst)
}

#[test]
#[ignore]
fn iu8g32_v2_gemv_matches_reference() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (mut w1, mut w2, mut cases) = (0f64, 0f64, 0);
    for &m in &[1usize, 3, 7, 129, 4097, 40000] {
        for &k in &[128usize, 256, 384, 2560, 4096, 9216] {
            if m * k > 40000 * 2560 {
                continue;
            }
            let iu = iu8_random(&mut rng, m, k);
            let (a, b) = check(&mut gpu, "random", &iu, m, k, &mut rng);
            w1 = w1.max(a);
            w2 = w2.max(b);
            cases += 1;
        }
    }
    if let Ok(p) = std::env::var("IU8_REAL") {
        use hipfire_runtime::hfq::HfqFile;
        let f = HfqFile::open(std::path::Path::new(&p)).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for t in f.tensors().to_vec() {
            let (ti, ib) = f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 59 || ti.shape.len() < 2 {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if !seen.insert((k, m)) {
                continue;
            }
            let ib = ib.to_vec();
            let (a, b) = check(
                &mut gpu,
                &format!("real:{}", &t.name[t.name.len().saturating_sub(24)..]),
                &ib,
                m,
                k,
                &mut rng,
            );
            w1 = w1.max(a);
            w2 = w2.max(b);
            cases += 1;
        }
    }
    eprintln!("iu8g32 v2 gemv: {cases} cases, worst rel v1 = {w1:.3e}, v2 = {w2:.3e}");
}

/// CPU contract of `gemv_iu8g32_a8_*`: x quantized per 32 to int8
/// (d_x = amax/127, round-half-even, clamp ±127), integer dot per 32,
/// y = sum_g (d_w * d_x) * dot.
fn a8_contract(iu: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f64> {
    let gw = k / 128;
    let g = k / 32;
    let rb = k + 2 * gw;
    let mut qx = vec![0i32; k];
    let mut dx = vec![0f32; g];
    for b in 0..g {
        let amax = x[b * 32..b * 32 + 32]
            .iter()
            .fold(0f32, |a, v| a.max(v.abs()));
        dx[b] = amax * (1.0 / 127.0);
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for i in 0..32 {
            qx[b * 32 + i] = ((x[b * 32 + i] * id).round_ties_even() as i32).clamp(-127, 127);
        }
    }
    (0..m)
        .map(|r| {
            let row = &iu[r * rb..(r + 1) * rb];
            (0..g)
                .map(|b| {
                    let sc = f16::from_bits(u16::from_le_bytes([
                        row[k + 2 * (b >> 2)],
                        row[k + 2 * (b >> 2) + 1],
                    ]))
                    .to_f32();
                    let dot: i32 = (0..32)
                        .map(|i| (row[b * 32 + i] as i8) as i32 * qx[b * 32 + i])
                        .sum();
                    ((sc * dx[b]) * dot as f32) as f64
                })
                .sum()
        })
        .collect()
}

const A8: [(&str, &str); 3] = [
    ("gemv_iu8g32_a8_r1", "gemv_iu8g32_a8_residual_r1"),
    ("gemv_iu8g32_a8_r2", "gemv_iu8g32_a8_residual_r2"),
    ("gemv_iu8g32_a8_r4", "gemv_iu8g32_a8_residual_r4"),
];

/// W8A8 (`HIPFIRE_IU8_GEMV=dot4`): every R variant matches the CPU contract
/// (rel < 1e-5 of the row L1), residual adds the seed, two runs are
/// bit-identical. Reports the int8-activation error vs the f64 reference
/// next to v2's (f32 x). Uniform and heavy-tailed x.
#[test]
#[ignore]
fn iu8g32_a8_gemv_matches_contract() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut wc, mut wa, mut wv, mut cases) = (0f64, 0f64, 0f64, 0);
    let mut shapes: Vec<(String, Vec<u8>, usize, usize)> = Vec::new();
    for &m in &[1usize, 7, 129, 4097, 40000] {
        for &k in &[128usize, 256, 2560, 4096] {
            if m * k > 40000 * 2560 {
                continue;
            }
            shapes.push(("random".into(), iu8_random(&mut rng, m, k), m, k));
        }
    }
    if let Ok(p) = std::env::var("IU8_REAL") {
        use hipfire_runtime::hfq::HfqFile;
        let f = HfqFile::open(std::path::Path::new(&p)).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for t in f.tensors().to_vec() {
            let (ti, ib) = f.tensor_data(&t.name).unwrap();
            if ti.quant_type != 59 || ti.shape.len() < 2 {
                continue;
            }
            let m = ti.shape[0] as usize;
            let k: usize = ti.shape[1..].iter().map(|&s| s as usize).product();
            if seen.insert((k, m)) {
                shapes.push((
                    format!("real:{}", &t.name[t.name.len().saturating_sub(24)..]),
                    ib.to_vec(),
                    m,
                    k,
                ));
            }
        }
    }
    for (label, iu, m, k) in &shapes {
        let (m, k) = (*m, *k);
        for heavy in [false, true] {
            let x: Vec<f32> = (0..k)
                .map(|i| rng.f() * if heavy && i % 64 == 5 { 30.0 } else { 1.0 })
                .collect();
            let seed: Vec<f32> = (0..m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
            let zero = vec![0f32; m];
            let w = gpu.upload_raw(iu, &[iu.len()]).unwrap();
            let xd = gpu.upload_f32(&x, &[k]).unwrap();
            let (want, l1) = reference(iu, &x, m, k);
            let con = a8_contract(iu, &x, m, k);
            let v2 = rel_of(
                &run(&mut gpu, "gemv_iu8g32_v2_r1", &w, &xd, &zero, m, k),
                &zero,
                false,
                &want,
                &l1,
            );
            let (mut rc, mut ra) = (0f64, 0f64);
            for (p, r) in A8 {
                let a = run(&mut gpu, p, &w, &xd, &zero, m, k);
                let a2 = run(&mut gpu, p, &w, &xd, &zero, m, k);
                assert!(
                    a.iter().zip(&a2).all(|(u, v)| u.to_bits() == v.to_bits()),
                    "{label} {p}: nondeterministic"
                );
                let c = rel_of(&a, &zero, false, &con, &l1);
                let cr = rel_of(
                    &run(&mut gpu, r, &w, &xd, &seed, m, k),
                    &seed,
                    true,
                    &con,
                    &l1,
                );
                assert!(
                    c < 1e-5 && cr < 1e-5,
                    "{label} m={m} k={k} {p}: vs contract {c:.3e} resid {cr:.3e}"
                );
                rc = rc.max(c.max(cr));
                ra = ra.max(rel_of(&a, &zero, false, &want, &l1));
            }
            eprintln!("{label:<28} {} m={m:<7} k={k:<5} a8_vs_contract={rc:.2e} | vs f64 ref: v2(f32 x)={v2:.2e} a8(int8 x)={ra:.2e}", if heavy { "heavy" } else { "unif " });
            wc = wc.max(rc);
            wa = wa.max(ra);
            wv = wv.max(v2);
            cases += 1;
            gpu.free_tensor(w).ok();
            gpu.free_tensor(xd).ok();
        }
    }
    eprintln!("iu8g32 a8 gemv: {cases} cases, worst vs contract {wc:.3e}; vs f64 ref v2 {wv:.3e}, a8 {wa:.3e}");
}
