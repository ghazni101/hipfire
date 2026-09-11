// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU smoke test for the grouped RMSNorm kernel.
//!
//! Verifies that `grouped_rmsnorm_f32` produces results matching a CPU
//! reference implementation. The test creates a small tensor, runs the
//! kernel, reads back the result, and compares.
//!
//! Run with: `cargo test -p hipfire-arch-k2-horizon -- --ignored`

#[cfg(test)]
mod gpu_tests {
    use rdna_compute::Gpu;

    /// CPU reference for grouped RMSNorm.
    /// Splits each row into `n_groups` contiguous chunks, computes RMS per
    /// chunk, applies weight elementwise.
    fn cpu_grouped_rmsnorm(
        x: &[f32],
        weight: &[f32],
        batch: usize,
        n: usize,
        n_groups: usize,
        eps: f32,
    ) -> Vec<f32> {
        let chunk_len = n / n_groups;
        let mut out = vec![0.0f32; batch * n];
        for r in 0..batch {
            for g in 0..n_groups {
                let offset = r * n + g * chunk_len;
                let mut sum_sq = 0.0f32;
                for i in 0..chunk_len {
                    let v = x[offset + i];
                    sum_sq += v * v;
                }
                let rms = 1.0 / (sum_sq / chunk_len as f32 + eps).sqrt();
                for i in 0..chunk_len {
                    let idx = offset + i;
                    out[idx] = x[idx] * weight[idx] * rms;
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "requires GPU — run with --ignored"]
    fn grouped_rmsnorm_matches_cpu_reference() {
        let mut gpu = Gpu::init().expect("GPU init");

        // K2-Horizon dimensions: dim=2560, n_groups=2, batch=1
        let batch = 1usize;
        let n = 2560usize;
        let n_groups = 2usize;
        let eps = 1e-6f32;

        // Fill with a deterministic pattern (not all-ones, which would mask
        // per-group variance differences).
        let x_host: Vec<f32> = (0..batch * n)
            .map(|i| {
                let group = (i % n) / (n / n_groups);
                ((i as f32) * 0.001 + group as f32).sin()
            })
            .collect();
        let w_host: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0001).collect();

        // CPU reference
        let expected = cpu_grouped_rmsnorm(&x_host, &w_host, batch, n, n_groups, eps);

        // GPU
        let x_gpu = gpu.upload_f32(&x_host, &[batch, n]).expect("upload x");
        let w_gpu = gpu.upload_f32(&w_host, &[n]).expect("upload w");
        let out_gpu = gpu
            .alloc_tensor(&[batch, n], rdna_compute::DType::F32)
            .expect("alloc out");

        gpu.grouped_rmsnorm_f32(&x_gpu, &w_gpu, &out_gpu, batch, n, n_groups, eps)
            .expect("grouped_rmsnorm_f32 launch");

        let result = gpu.download_f32(&out_gpu).expect("readback");

        // Compare with tolerance — fp32 arithmetic, reduction order may differ.
        let mut max_err = 0.0f32;
        for (got, want) in result.iter().zip(expected.iter()) {
            let err = (got - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-4,
            "grouped_rmsnorm: max error {max_err:.2e} exceeds 1e-4"
        );
    }

    #[test]
    #[ignore = "requires GPU — run with --ignored"]
    fn grouped_rmsnorm_differs_from_standard_rmsnorm() {
        let mut gpu = Gpu::init().expect("GPU init");

        let batch = 1usize;
        let n = 2560usize;
        let n_groups = 2usize;
        let eps = 1e-6f32;

        // Use input where the two halves have very different magnitudes so
        // grouped vs standard RMSNorm produce visibly different outputs.
        let mut x_host = vec![0.0f32; batch * n];
        for i in 0..n / 2 {
            x_host[i] = 0.01; // small first half
        }
        for i in n / 2..n {
            x_host[i] = 10.0; // large second half
        }
        let w_host: Vec<f32> = vec![1.0; n];

        let x_gpu = gpu.upload_f32(&x_host, &[batch, n]).expect("upload x");
        let w_gpu = gpu.upload_f32(&w_host, &[n]).expect("upload w");

        // Standard RMSNorm
        let out_std = gpu
            .alloc_tensor(&[batch, n], rdna_compute::DType::F32)
            .expect("alloc std");
        gpu.rmsnorm_f32(&x_gpu, &w_gpu, &out_std, eps)
            .expect("rmsnorm_f32 launch");
        let result_std = gpu.download_f32(&out_std).expect("readback std");

        // Grouped RMSNorm
        let out_grp = gpu
            .alloc_tensor(&[batch, n], rdna_compute::DType::F32)
            .expect("alloc grp");
        gpu.grouped_rmsnorm_f32(&x_gpu, &w_gpu, &out_grp, batch, n, n_groups, eps)
            .expect("grouped_rmsnorm_f32 launch");
        let result_grp = gpu.download_f32(&out_grp).expect("readback grp");

        // The first half should be much larger under grouped norm (RMS
        // computed only over the small-valued half) than under standard
        // norm (RMS over both halves).
        let std_first = result_std[0].abs();
        let grp_first = result_grp[0].abs();
        assert!(
            grp_first > std_first * 10.0,
            "grouped norm first half ({grp_first:.4}) should be >> standard ({std_first:.4})"
        );

        // The second half should be much smaller under grouped norm.
        // (index n/2 = first element of the second group; n is out of bounds)
        let std_second = result_std[n / 2].abs();
        let grp_second = result_grp[n / 2].abs();
        assert!(
            grp_second < std_second,
            "grouped norm second half ({grp_second:.4}) should be < standard ({std_second:.4})"
        );
    }

    /// Batched grouped RMSNorm must produce per-row results identical to
    /// batch=1. Catches row-offset / weight-indexing bugs that only appear
    /// when `batch > 1` (e.g. the K2-Horizon prefill path).
    #[test]
    #[ignore = "requires GPU — run with --ignored"]
    fn grouped_rmsnorm_batched_matches_single() {
        let mut gpu = Gpu::init().expect("GPU init");

        let batch = 4usize;
        let n = 2560usize;
        let n_groups = 2usize;
        let eps = 1e-6f32;

        let x_host: Vec<f32> = (0..batch * n)
            .map(|i| ((i * 37 % 1000) as f32 - 500.0) * 0.001)
            .collect();
        let w_host: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.0003).collect();

        let x_gpu = gpu.upload_f32(&x_host, &[batch, n]).expect("upload x");
        let w_gpu = gpu.upload_f32(&w_host, &[n]).expect("upload w");

        // Batched call.
        let out_b = gpu
            .alloc_tensor(&[batch, n], rdna_compute::DType::F32)
            .expect("alloc batched out");
        gpu.grouped_rmsnorm_f32(&x_gpu, &w_gpu, &out_b, batch, n, n_groups, eps)
            .expect("batched launch");
        let got_b = gpu.download_f32(&out_b).expect("readback batched");

        // Per-row single calls.
        let mut got_s = vec![0.0f32; batch * n];
        for r in 0..batch {
            let x_row = gpu
                .upload_f32(&x_host[r * n..(r + 1) * n], &[1, n])
                .expect("upload row");
            let out_r = gpu
                .alloc_tensor(&[1, n], rdna_compute::DType::F32)
                .expect("alloc row out");
            gpu.grouped_rmsnorm_f32(&x_row, &w_gpu, &out_r, 1, n, n_groups, eps)
                .expect("single launch");
            let row = gpu.download_f32(&out_r).expect("readback row");
            got_s[r * n..(r + 1) * n].copy_from_slice(&row);
        }

        let mut max_err = 0.0f32;
        for (b, s) in got_b.iter().zip(got_s.iter()) {
            max_err = max_err.max((b - s).abs());
        }
        assert!(
            max_err < 1e-4,
            "batched vs single max_err={max_err:.6} (expected < 1e-4)"
        );
    }
}
