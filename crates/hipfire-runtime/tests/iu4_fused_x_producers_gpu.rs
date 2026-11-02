//! C11b: the IU4 fused producers (RMSNorm / SwiGLU / plain FWHT, each with
//! the int4 `block_i4_128` and int8 `block_i8_128` sidecar) are bit-exact
//! to the f32 producer followed by the standalone quantizer, f32 `x_rot`
//! included, and deterministic. The int4 producers' emit is the reduced
//! candidate-set quantizer, so their sidecar bytes are reported, not asserted. GPU-only:
//! `cargo test --release -p hipfire-runtime --test iu4_fused_x_producers_gpu -- --ignored --nocapture`
use rdna_compute::{Gpu, GpuTensor};
use std::ffi::c_void;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn bytes_at(gpu: &mut Gpu, p: *mut c_void, len: usize) -> Vec<u8> {
    let t = gpu.upload_raw(&vec![0u8; len], &[len]).unwrap();
    gpu.hip.device_synchronize().unwrap();
    unsafe {
        gpu.hip.memcpy_dtod_raw(t.buf.as_ptr(), p as *const c_void, len).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let v = gpu.download_raw_bytes(&t).unwrap();
    gpu.free_tensor(t).ok();
    v
}

fn f32_bits(gpu: &mut Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t).unwrap().iter().map(|v| v.to_bits()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Prod {
    Norm,
    Silu,
    Rot,
}

/// Returns (f32 x_rot bits, sidecar bytes) for the reference (`fused=false`)
/// or the fused producer.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &mut Gpu,
    prod: Prod,
    a8: bool,
    fused: bool,
    awq: Option<&GpuTensor>,
    a: &GpuTensor,
    b: &GpuTensor,
    w: &GpuTensor,
    k: usize,
    n: usize,
) -> (Vec<u32>, Vec<u8>) {
    let eps = 1e-6f32;
    let xr = gpu.upload_f32(&vec![f32::NAN; n * k], &[n * k]).unwrap();
    let len = (k / 128) * n * if a8 { 136 } else { 72 };
    let side = if !fused {
        match (prod, awq) {
            (Prod::Norm, Some(s)) => gpu.fused_rmsnorm_rotate_mq_awq_batched(a, w, s, &xr, k, eps, n).unwrap(),
            (Prod::Norm, None) => gpu.fused_rmsnorm_rotate_mq_batched(a, w, &xr, k, eps, n).unwrap(),
            (Prod::Silu, Some(s)) => gpu.fused_silu_mul_rotate_mq_awq_batched(a, b, s, &xr, k, n).unwrap(),
            (Prod::Silu, None) => gpu.fused_silu_mul_rotate_mq_batched(a, b, &xr, k, n).unwrap(),
            (Prod::Rot, Some(s)) => gpu.rotate_x_mq_awq_batched(a, s, &xr, k, n).unwrap(),
            (Prod::Rot, None) => gpu.rotate_x_mq_batched(a, &xr, k, n).unwrap(),
        }
        let p = if a8 { gpu.ensure_int8_mmq_x(&xr, n, k).unwrap() } else { gpu.ensure_int4_mmq_x(&xr, n, k).unwrap() };
        bytes_at(gpu, p, len)
    } else if a8 {
        let res = gpu.reserve_int8_mmq(k, n).unwrap();
        let h = match prod {
            Prod::Norm => gpu.fused_rmsnorm_rotate_mq_i8_batched(a, w, awq, Some(&xr), res, k, eps, n).unwrap(),
            Prod::Silu => gpu.fused_silu_mul_rotate_mq_i8_batched(a, b, Some(&xr), res, k, n).unwrap(),
            Prod::Rot => gpu.rotate_x_mq_i8_batched(a, Some(&xr), res, k, n).unwrap(),
        };
        let p = gpu.int8_mmq_prepared_ptr(&h, k, n).unwrap();
        bytes_at(gpu, p, len)
    } else {
        let res = gpu.reserve_int4_mmq(k, n).unwrap();
        let h = match prod {
            Prod::Norm => gpu.fused_rmsnorm_rotate_mq_i4_batched(a, w, awq, Some(&xr), res, k, eps, n).unwrap(),
            Prod::Silu => gpu.fused_silu_mul_rotate_mq_i4_batched(a, b, awq, Some(&xr), res, k, n).unwrap(),
            Prod::Rot => gpu.rotate_x_mq_i4_batched(a, awq, Some(&xr), res, k, n).unwrap(),
        };
        let p = gpu.int4_mmq_prepared_ptr(&h, k, n).unwrap();
        bytes_at(gpu, p, len)
    };
    let bits = f32_bits(gpu, &xr);
    gpu.free_tensor(xr).ok();
    (bits, side)
}

#[test]
#[ignore]
fn iu4_fused_x_producers_bit_exact() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.iu4g256_mmq_eligible(256, 64) {
        eprintln!("SKIP — IU4 MMQ not eligible on this arch.");
        return;
    }
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (mut cases, mut bad) = (0usize, 0usize);
    for &k in &[256usize, 768, 2560, 9728, 17408] {
        for &n in &[1usize, 7, 64, 333, 2048] {
            if k * n > 17408 * 512 && n == 2048 && k > 9728 {
                continue;
            }
            let scale = 1.0 + (rng.next() % 40) as f32;
            let va: Vec<f32> = (0..n * k).map(|_| rng.f() * scale).collect();
            let vb: Vec<f32> = (0..n * k).map(|_| rng.f() * 3.0).collect();
            let vw: Vec<f32> = (0..k).map(|_| 1.0 + rng.f()).collect();
            let vs: Vec<f32> = (0..k).map(|_| 0.25 + (rng.f() + 0.5) * 2.0).collect();
            let a = gpu.upload_f32(&va, &[n * k]).unwrap();
            let b = gpu.upload_f32(&vb, &[n * k]).unwrap();
            let w = gpu.upload_f32(&vw, &[k]).unwrap();
            let s = gpu.upload_f32(&vs, &[k]).unwrap();
            for prod in [Prod::Norm, Prod::Silu, Prod::Rot] {
                for a8 in [false, true] {
                    for use_awq in [false, true] {
                        // The i8 SwiGLU / plain-FWHT producers are no-AWQ only
                        // (the IU4 route falls back to the f32 producer with AWQ).
                        // RMSNorm rows live in LDS: the f32 reference itself
                        // rejects k = 17408 (the route only fuses k + 256 <= 16384).
                        if prod == Prod::Norm && (k + 256) * 4 > 65536 {
                            continue;
                        }
                        if a8 && use_awq && prod != Prod::Norm {
                            continue;
                        }
                        let awq = if use_awq { Some(&s) } else { None };
                        let r = run(&mut gpu, prod, a8, false, awq, &a, &b, &w, k, n);
                        let f1 = run(&mut gpu, prod, a8, true, awq, &a, &b, &w, k, n);
                        let f2 = run(&mut gpu, prod, a8, true, awq, &a, &b, &w, k, n);
                        let dx = r.0.iter().zip(&f1.0).filter(|(p, q)| p != q).count();
                        let ds = r.1.iter().zip(&f1.1).filter(|(p, q)| p != q).count();
                        let det = f1 == f2;
                        cases += 1;
                        // i4: the producer emit uses the reduced MSE candidate set
                        // (by design not the standalone 8-candidate quantizer), so
                        // only x_rot + determinism are asserted; the byte diff is
                        // reported.
                        let ok = dx == 0 && det && (ds == 0 || !a8);
                        if !ok {
                            bad += 1;
                        }
                        eprintln!(
                            "{:<4} {} awq={} k={k:<6} n={n:<5} xrot_bitdiff={dx} sidecar_bytediff={ds}/{} det={det} {}",
                            format!("{prod:?}"),
                            if a8 { "i8" } else { "i4" },
                            use_awq as u8,
                            r.1.len(),
                            if !ok { "FAIL" } else if ds != 0 { "OK(i4-candset)" } else { "OK" }
                        );
                    }
                }
            }
            for t in [a, b, w, s] {
                gpu.free_tensor(t).ok();
            }
        }
    }
    eprintln!("SUMMARY cases={cases} bad={bad}");
    assert_eq!(bad, 0);
}
