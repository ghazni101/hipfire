//! C15: the `I8_XPLANAR` int8 X layout holds exactly the legacy
//! `block_i8_128` bytes, relocated (standalone quantizer and all three gfx11
//! fused producers), and the IU4 W4A8 GEMM reading it is bitwise equal to the
//! legacy-layout GEMM (t128 and t256, set and add). GPU-only:
//! `cargo test --release -p hipfire-runtime --test iu4_i8_xplanar_gpu -- --ignored --nocapture`
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

/// Legacy [K/128][N] x 136 B -> XPLANAR per-kb [N x 128 qs][N x 8 hdr].
fn relayout(l: &[u8], k: usize, n: usize) -> Vec<u8> {
    let mut o = vec![0u8; l.len()];
    for kb in 0..k / 128 {
        let c = kb * n * 136;
        for t in 0..n {
            let b = &l[(kb * n + t) * 136..(kb * n + t + 1) * 136];
            o[c + t * 128..c + t * 128 + 128].copy_from_slice(&b[8..]);
            o[c + n * 128 + t * 8..c + n * 128 + t * 8 + 8].copy_from_slice(&b[..8]);
        }
    }
    o
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(p, q)| p != q).count()
}

#[allow(clippy::too_many_arguments)]
fn prod(gpu: &mut Gpu, which: u8, xp: bool, a: &GpuTensor, b: &GpuTensor, w: &GpuTensor, awq: Option<&GpuTensor>, k: usize, n: usize) -> Vec<u8> {
    let xr = gpu.upload_f32(&vec![0.0; n * k], &[n * k]).unwrap();
    let res = gpu.reserve_int8_mmq(k, n).unwrap();
    gpu.i8_xplanar_emit = xp;
    let h = match which {
        0 => gpu.fused_rmsnorm_rotate_mq_i8_batched(a, w, awq, Some(&xr), res, k, 1e-6, n).unwrap(),
        1 => gpu.fused_silu_mul_rotate_mq_i8_batched(a, b, Some(&xr), res, k, n).unwrap(),
        _ => gpu.rotate_x_mq_i8_batched(a, Some(&xr), res, k, n).unwrap(),
    };
    gpu.i8_xplanar_emit = false;
    let p = gpu.int8_mmq_prepared_ptr(&h, k, n).unwrap();
    let v = bytes_at(gpu, p, (k / 128) * n * 136);
    gpu.free_tensor(xr).ok();
    v
}

#[test]
#[ignore]
fn iu4_i8_xplanar_bit_exact() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.iu4g256_mmq_eligible(256, 64) || gpu.arch != "gfx1100" {
        eprintln!("SKIP — gfx11 IU4 MMQ only.");
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut bad = 0usize;
    // Layout: standalone + producers.
    for &k in &[256usize, 768, 2560, 9728, 17408] {
        for &n in &[1usize, 7, 64, 333, 2048] {
            let scale = 1.0 + (rng.next() % 40) as f32;
            let va: Vec<f32> = (0..n * k).map(|_| rng.f() * scale).collect();
            let vb: Vec<f32> = (0..n * k).map(|_| rng.f() * 3.0).collect();
            let vw: Vec<f32> = (0..k).map(|_| 1.0 + rng.f()).collect();
            let vs: Vec<f32> = (0..k).map(|_| 0.25 + (rng.f() + 0.5) * 2.0).collect();
            let a = gpu.upload_f32(&va, &[n * k]).unwrap();
            let b = gpu.upload_f32(&vb, &[n * k]).unwrap();
            let w = gpu.upload_f32(&vw, &[k]).unwrap();
            let s = gpu.upload_f32(&vs, &[k]).unwrap();
            let len = (k / 128) * n * 136;
            let pl = gpu.ensure_int8_mmq_x_layout(&a, n, k, false).unwrap();
            let l = bytes_at(&mut gpu, pl, len);
            let px = gpu.ensure_int8_mmq_x_layout(&a, n, k, true).unwrap();
            let x = bytes_at(&mut gpu, px, len);
            let want = relayout(&l, k, n);
            let d0 = diff(&x, &want);
            if d0 != 0 {
                bad += 1;
            }
            eprintln!("quant k={k:<6} n={n:<5} xp_vs_relayout(legacy) bytediff={d0}/{len}");
            for which in 0u8..3 {
                for use_awq in [false, true] {
                    if which == 0 && (k + 256) * 4 > 65536 {
                        continue;
                    }
                    if use_awq && which != 0 {
                        continue;
                    }
                    let awq = if use_awq { Some(&s) } else { None };
                    let pl = prod(&mut gpu, which, false, &a, &b, &w, awq, k, n);
                    let px = prod(&mut gpu, which, true, &a, &b, &w, awq, k, n);
                    let px2 = prod(&mut gpu, which, true, &a, &b, &w, awq, k, n);
                    let dd = diff(&px, &relayout(&pl, k, n));
                    let det = px == px2;
                    if dd != 0 || !det {
                        bad += 1;
                    }
                    eprintln!("prod{which} awq={} k={k:<6} n={n:<5} bytediff={dd} det={det}", use_awq as u8);
                }
            }
            for t in [a, b, w, s] {
                gpu.free_tensor(t).ok();
            }
        }
    }
    // GEMM: legacy X + legacy kernel vs XPLANAR X + XPLANAR kernel.
    // (m, k, n): small grids pick t128, big ones t256 (HIPFIRE_IU4_I8_TILE auto).
    for &(m, k, n) in &[(256usize, 256usize, 1usize), (512, 2560, 7), (1024, 2560, 64), (2048, 9728, 333), (4096, 2560, 2048), (17408, 5120, 2048), (512, 768, 130)] {
        let wb: Vec<u8> = (0..m * (k / 256) * 136).map(|i| { let r = rng.next() as u8; if i % 2 == 1 { 0x30 + r % 12 } else { r } }).collect();
        let x: Vec<f32> = (0..n * k).map(|_| rng.f() * 2.0).collect();
        let seed: Vec<f32> = (0..n * m).map(|i| 0.37 * (i as f32 % 11.0) - 1.1).collect();
        let wt = gpu.upload_raw(&wb, &[wb.len()]).unwrap();
        let xd = gpu.upload_f32(&x, &[n * k]).unwrap();
        for add in [false, true] {
            let yl = gpu.upload_f32(&seed, &[n * m]).unwrap();
            let yx = gpu.upload_f32(&seed, &[n * m]).unwrap();
            let yx2 = gpu.upload_f32(&seed, &[n * m]).unwrap();
            let ql = gpu.ensure_int8_mmq_x_layout(&xd, n, k, false).unwrap();
            gpu.gemm_iu4g256_w4a8_prequant_layout(&wt, ql, &yl, m, k, n, add, false).unwrap();
            let qx = gpu.ensure_int8_mmq_x_layout(&xd, n, k, true).unwrap();
            gpu.gemm_iu4g256_w4a8_prequant_layout(&wt, qx, &yx, m, k, n, add, true).unwrap();
            gpu.gemm_iu4g256_w4a8_prequant_layout(&wt, qx, &yx2, m, k, n, add, true).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let bits = |gpu: &mut Gpu, t: &GpuTensor| gpu.download_f32(t).unwrap().iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
            let (a, b, c) = (bits(&mut gpu, &yl), bits(&mut gpu, &yx), bits(&mut gpu, &yx2));
            let nd = a.iter().zip(&b).filter(|(p, q)| p != q).count();
            let nan = a.iter().filter(|v| f32::from_bits(**v).is_nan()).count();
            let det = b == c;
            if nd != 0 || !det {
                bad += 1;
            }
            eprintln!("gemm m={m:<6} k={k:<6} n={n:<5} add={} bitdiff={nd}/{} det={det} nan_in_ref={nan}", add as u8, a.len());
            for t in [yl, yx, yx2] {
                gpu.free_tensor(t).ok();
            }
        }
        gpu.free_tensor(wt).ok();
        gpu.free_tensor(xd).ok();
    }
    eprintln!("SUMMARY bad={bad}");
    assert_eq!(bad, 0);
}
