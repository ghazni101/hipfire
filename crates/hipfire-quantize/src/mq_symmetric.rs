// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared per-128 symmetric grid search for the MQ4V2 / MQ3V2 encoders.
//!
//! The wire format stays the affine fp16 `(scale, zero)` header; the stored
//! zero point is exactly `-mid * d` (`mid` = 8 for MQ4, 4 for MQ3), so code
//! `mid` reconstructs 0. The main `hipfire-quantize` encoders
//! (`quantize_mq4g256v2_symmetric`, `quantize_mq3g256v2_symmetric`) and the
//! `dflash_convert` draft converter both call this one routine.

use crate::float16::{f16_to_f32, f32_to_f16};

/// Quantize one 128-weight FWHT-rotated half onto a symmetric grid.
///
/// `max_code` is 15 (MQ4V2, mid code 8) or 7 (MQ3V2, mid code 4). Four scale
/// candidates (the runtime A4 producer ladder shifted onto the
/// `max_code`-interval midpoint grid) are scored by MSE including the fp16
/// round-trip of both scale and zero. Returns `(scale_bits, zero_bits)` and
/// fills `codes` (len 128).
pub fn quantize_symmetric_half(
    values: &[f32],
    max_code: u8,
    codes: &mut [u8],
) -> (u16, u16) {
    debug_assert!(max_code == 15 || max_code == 7);
    debug_assert_eq!(values.len(), 128);
    debug_assert_eq!(codes.len(), 128);
    let mid = (max_code as f32 + 1.0) * 0.5;
    let max_code_f = max_code as f32;

    let amax = values.iter().fold(0.0f32, |acc, &value| acc.max(value.abs()));
    if amax == 0.0 {
        codes.fill(mid as u8);
        return (0, f32_to_f16(-0.0));
    }

    let candidate_base = (amax / (max_code_f * 0.5)) * 0.5;
    let multipliers = [
        1.0f32,
        f32::from_bits(0x3fa4_9249),
        f32::from_bits(0x3fdb_6db7),
        2.0f32,
    ];
    let mut scale_bits_best = 0u16;
    let mut zero_bits_best = 0u16;
    let mut best_mse = f64::INFINITY;
    for multiplier in multipliers {
        let scale_bits = f32_to_f16(candidate_base * multiplier);
        let scale = f16_to_f32(scale_bits);
        if scale == 0.0 {
            continue;
        }
        let zero_bits = f32_to_f16(-mid * scale);
        let zero = f16_to_f32(zero_bits);
        let inverse = 1.0 / scale;
        let mut mse = 0.0f64;
        for &value in values {
            let code = ((value - zero) * inverse + 0.5).floor().clamp(0.0, max_code_f);
            let error = value - code.mul_add(scale, zero);
            mse += (error as f64) * (error as f64);
        }
        if mse < best_mse {
            best_mse = mse;
            scale_bits_best = scale_bits;
            zero_bits_best = zero_bits;
        }
    }
    let scale = f16_to_f32(scale_bits_best);
    let zero = f16_to_f32(zero_bits_best);
    let inverse = 1.0 / scale;
    for (code, &value) in codes.iter_mut().zip(values) {
        *code = ((value - zero) * inverse + 0.5).floor().clamp(0.0, max_code_f) as u8;
    }
    (scale_bits_best, zero_bits_best)
}
