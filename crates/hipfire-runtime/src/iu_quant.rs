// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! IU4 / IU8 storage for the integer matrix units (HFQ qt=57 / qt=58).
//!
//! The codes and the per-128 affine grid are MQ4G256V2's (qt=44): FWHT-256,
//! unsigned codes `0..=15`, `w = f32(scale) * code + f32(zero)` on each
//! 128-wide half. What changes is the byte layout. MQ4V2 interleaves an 8-byte
//! header in front of every 128 nibble bytes, so an IU4 or IU8 WMMA load has
//! to skip those headers (and, for IU8, expand each nibble into a byte).
//!
//! An IU row is one contiguous operand plane, then the scale pairs:
//!
//! ```text
//! [codes for all K][f16 scale, f16 zero] * (K/128)
//! ```
//!
//! * `K % 256 == 0`. The FWHT group is still 256 weights, which is two
//!   128-wide affine halves. `K == 0` is rejected.
//! * **IU4 (qt 57).** Unsigned nibbles, low nibble = even K, high nibble =
//!   odd K. That is the packing `v_wmma_i32_16x16x16_iu4` consumes with the
//!   weight side unsigned: nibble `m` of a dword is K element `m`. 128 code
//!   bytes per 256 weights. Stored bpw 4.25, same as MQ4V2.
//! * **IU8 (qt 58).** One byte per weight, value `0..=15`, K-linear. That is
//!   the byte operand `v_wmma_i32_16x16x16_iu8` consumes after today's
//!   `pk & 0x0F0F0F0F` / `(pk >> 4) & 0x0F0F0F0F` expand, already expanded.
//!   256 code bytes per 256 weights. Stored bpw 8.25. Not an 8-bit grid:
//!   the high bits of each byte are zero.
//!
//! Scales sit at the end of the row, half 0 then half 1 of each 256-group,
//! little-endian fp16, the same `[scale0, zero0, scale1, zero1]` header
//! MQ4V2 stores at the front of the group. Symmetric MQ4V2 still stores
//! `zero = f16(-8 * scale)`, so a kernel that wants the integer seed
//! `sum((q - 8) * x)` can use it when that identity holds after the fp16
//! round trip.
//!
//! No GPU kernel reads these tensors yet. Generic GEMV and GEMM refuse them.
//! A kernel should load `row_bytes(k)` with the code plane at offset 0
//! (`code_bytes(k)` long) and the scales immediately after.
//!
//! ## IU8G32 (qt 59), the Q8 tier
//!
//! Embeddings, `lm_head`, and the fixed tier are not MQ4 codes. They stay on
//! the Q8F16 grid: signed int8, one fp16 scale per 32 weights, zero point 0,
//! `w = f32(scale) * f32(i8(code))`, no FWHT. Q8F16 interleaves that scale in
//! front of every 32 codes, so an IU8 WMMA load cannot take the row as one
//! operand. [`repack_q8f16`] moves the scales to the tail:
//!
//! ```text
//! [i8 codes for all K][f16 scale] * (K/32)
//! ```
//!
//! `K % 32 == 0`. The bytes are the full int8 range, which
//! `v_wmma_i32_16x16x16_iu8` multiplies with the weight operand signed. This
//! is not [`Layout::Iu8`]: that layout is qt 58, MQ4 nibbles stored in a byte.

use half::f16;

/// FWHT / affine group, in weights. Two 128-wide halves.
pub const GROUP: usize = 256;
/// Affine half width. One `(scale, zero)` pair covers this many codes.
pub const HALF: usize = 128;
/// MQ4G256V2 group stride this layout is repacked from.
pub const MQ4V2_GROUP_BYTES: usize = 136;
/// Bytes of one `(f16 scale, f16 zero)` pair.
pub const SCALE_PAIR_BYTES: usize = 4;
/// Q8F16 group. IU8G32 (qt 59) keeps this grid and only moves the scales.
pub const Q8_GROUP: usize = 32;
/// Interleaved Q8F16 block: one f16 scale, then 32 signed codes.
pub const Q8_BLOCK_BYTES: usize = 34;
/// HFQ quant-type byte of the planar Q8 operand ([`repack_q8f16`]).
pub const Q8_QUANT_TYPE: u8 = 59;

/// Which integer operand the codes are stored as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// qt=57. Packed unsigned nibbles.
    Iu4,
    /// qt=58. One byte per code, `0..=15`.
    Iu8,
}

impl Layout {
    /// HFQ quant-type byte.
    pub const fn quant_type(self) -> u8 {
        match self {
            Self::Iu4 => 57,
            Self::Iu8 => 58,
        }
    }

    /// Wire name. Matches `QuantType` and the qt register.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Iu4 => "IU4G256",
            Self::Iu8 => "IU8G256",
        }
    }

    /// Parse a `--format` token. `iu4` / `iu4g256`, `iu8` / `iu8g256`.
    pub fn from_format(format: &str) -> Option<Self> {
        match format {
            "iu4" | "iu4g256" => Some(Self::Iu4),
            "iu8" | "iu8g256" => Some(Self::Iu8),
            _ => None,
        }
    }
}

/// `K/256` when `K` is a positive multiple of [`GROUP`].
pub fn groups_per_row(k: usize) -> Option<usize> {
    if k == 0 || k % GROUP != 0 {
        None
    } else {
        Some(k / GROUP)
    }
}

/// Code-plane bytes of one row.
pub fn code_bytes(k: usize, layout: Layout) -> Option<usize> {
    groups_per_row(k)?;
    match layout {
        Layout::Iu4 => Some(k / 2),
        Layout::Iu8 => Some(k),
    }
}

/// Scale-plane bytes of one row: one pair per 128 weights.
pub fn scale_bytes(k: usize) -> Option<usize> {
    groups_per_row(k)?;
    Some((k / HALF) * SCALE_PAIR_BYTES)
}

/// Bytes of one row: codes, then scales.
pub fn row_bytes(k: usize, layout: Layout) -> Option<usize> {
    code_bytes(k, layout)?.checked_add(scale_bytes(k)?)
}

/// Bytes of an `[m, k]` tensor.
pub fn tensor_bytes(m: usize, k: usize, layout: Layout) -> Option<usize> {
    if m == 0 {
        return None;
    }
    m.checked_mul(row_bytes(k, layout)?)
}

/// Stored bits per weight, header included. IU4 is 4.25, IU8 is 8.25.
pub fn stored_bpw(layout: Layout) -> f32 {
    let row = row_bytes(GROUP, layout).expect("256 is a legal K");
    (row * 8) as f32 / GROUP as f32
}

fn shape_error(m: usize, k: usize) -> String {
    format!("IU quant requires M > 0 and K % {GROUP} == 0, got M={m} K={k}")
}

/// Reject a tensor whose length, shape, or (IU8) code range is not the layout.
pub fn validate(bytes: &[u8], m: usize, k: usize, layout: Layout) -> Result<(), String> {
    let expected = tensor_bytes(m, k, layout).ok_or_else(|| shape_error(m, k))?;
    if bytes.len() != expected {
        return Err(format!(
            "{} blob length mismatch: expected {expected}, got {} (M={m} K={k})",
            layout.name(),
            bytes.len()
        ));
    }
    if layout == Layout::Iu8 {
        let codes = code_bytes(k, layout).expect("shape already checked");
        let row = row_bytes(k, layout).expect("shape already checked");
        for r in 0..m {
            let plane = &bytes[r * row..r * row + codes];
            if let Some(bad) = plane.iter().position(|&b| b > 15) {
                return Err(format!(
                    "IU8 code at row {r} K {bad} is {plane_b}, not an MQ4 nibble 0..=15",
                    plane_b = plane[bad]
                ));
            }
        }
    }
    Ok(())
}

/// Rebuild one MQ4G256V2 tensor (`136 * M * K/256` bytes) into an IU row layout.
///
/// Codes and scale bits are copied. Nothing is re-rounded and the FWHT is not
/// run again: the source blob is already in the rotated basis.
pub fn repack_mq4v2(src: &[u8], m: usize, k: usize, layout: Layout) -> Result<Vec<u8>, String> {
    let gpr = groups_per_row(k).ok_or_else(|| shape_error(m, k))?;
    if m == 0 {
        return Err(shape_error(m, k));
    }
    let groups = m
        .checked_mul(gpr)
        .ok_or_else(|| format!("IU shape overflows: M={m} K={k}"))?;
    let expect = groups
        .checked_mul(MQ4V2_GROUP_BYTES)
        .ok_or_else(|| format!("MQ4V2 blob length overflows: M={m} K={k}"))?;
    if src.len() != expect {
        return Err(format!(
            "MQ4V2 blob length mismatch: expected {expect}, got {} (M={m} K={k})",
            src.len()
        ));
    }
    let row_b = row_bytes(k, layout).expect("shape already checked");
    let code_b = code_bytes(k, layout).expect("shape already checked");
    let mut out = vec![0u8; m * row_b];
    for row in 0..m {
        let dst = &mut out[row * row_b..(row + 1) * row_b];
        let mut code_at = 0usize;
        let mut scale_at = code_b;
        for g in 0..gpr {
            let base = (row * gpr + g) * MQ4V2_GROUP_BYTES;
            let header = &src[base..base + 8];
            let nibbles = &src[base + 8..base + MQ4V2_GROUP_BYTES];
            match layout {
                Layout::Iu4 => {
                    dst[code_at..code_at + 128].copy_from_slice(nibbles);
                    code_at += 128;
                }
                Layout::Iu8 => {
                    for (i, byte) in nibbles.iter().copied().enumerate() {
                        dst[code_at + 2 * i] = byte & 0x0f;
                        dst[code_at + 2 * i + 1] = byte >> 4;
                    }
                    code_at += 256;
                }
            }
            dst[scale_at..scale_at + 8].copy_from_slice(header);
            scale_at += 8;
        }
        debug_assert_eq!(code_at, code_b);
        debug_assert_eq!(scale_at, row_b);
    }
    Ok(out)
}

/// Bytes of one IU8G32 row: `K` signed codes, then one f16 scale per 32.
///
/// `None` unless `K` is a positive multiple of [`Q8_GROUP`]. A partial group
/// is not rounded up: the interleaved Q8F16 block would cross the row, and
/// there is no legal plane for it.
pub fn q8_row_bytes(k: usize) -> Option<usize> {
    if k == 0 || k % Q8_GROUP != 0 {
        None
    } else {
        Some(k + (k / Q8_GROUP) * 2)
    }
}

/// Bytes of an `[m, k]` IU8G32 tensor.
pub fn q8_tensor_bytes(m: usize, k: usize) -> Option<usize> {
    if m == 0 {
        return None;
    }
    m.checked_mul(q8_row_bytes(k)?)
}

fn q8_shape_error(m: usize, k: usize) -> String {
    format!("IU8G32 requires M > 0 and K % {Q8_GROUP} == 0, got M={m} K={k}")
}

/// Reject an IU8G32 tensor whose length or shape is not the layout.
///
/// Every byte of the code plane is a legal signed code, including values
/// above 15. That is the difference from [`validate`] on [`Layout::Iu8`].
pub fn validate_q8(bytes: &[u8], m: usize, k: usize) -> Result<(), String> {
    let expected = q8_tensor_bytes(m, k).ok_or_else(|| q8_shape_error(m, k))?;
    if bytes.len() != expected {
        return Err(format!(
            "IU8G32 blob length mismatch: expected {expected}, got {} (M={m} K={k})",
            bytes.len()
        ));
    }
    Ok(())
}

/// Rebuild a Q8F16 tensor (`34 * M * K/32` bytes) into an IU8G32 row.
///
/// Codes and scale bits are copied. Nothing is re-rounded. The source is
/// the natural-basis Q8 grid, not an FWHT MQ4 blob.
pub fn repack_q8f16(src: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    if q8_row_bytes(k).is_none() || m == 0 {
        return Err(q8_shape_error(m, k));
    }
    let gpr = k / Q8_GROUP;
    let groups = m
        .checked_mul(gpr)
        .ok_or_else(|| format!("IU8G32 shape overflows: M={m} K={k}"))?;
    let expect = groups
        .checked_mul(Q8_BLOCK_BYTES)
        .ok_or_else(|| format!("Q8F16 blob length overflows: M={m} K={k}"))?;
    if src.len() != expect {
        return Err(format!(
            "Q8F16 blob length mismatch: expected {expect}, got {} (M={m} K={k})",
            src.len()
        ));
    }
    let row_b = q8_row_bytes(k).expect("shape already checked");
    let mut out = vec![0u8; m * row_b];
    for row in 0..m {
        let dst = &mut out[row * row_b..(row + 1) * row_b];
        for g in 0..gpr {
            let base = (row * gpr + g) * Q8_BLOCK_BYTES;
            dst[k + g * 2..k + g * 2 + 2].copy_from_slice(&src[base..base + 2]);
            let code_at = g * Q8_GROUP;
            dst[code_at..code_at + Q8_GROUP]
                .copy_from_slice(&src[base + 2..base + Q8_BLOCK_BYTES]);
        }
    }
    Ok(out)
}

/// Dequantise an IU8G32 tensor. `w = f32(scale) * f32(i8(code))` for the
/// scale of `k / 32`. Natural basis: these weights were not FWHT-rotated.
pub fn dequant_q8(bytes: &[u8], m: usize, k: usize) -> Result<Vec<f32>, String> {
    validate_q8(bytes, m, k)?;
    let row_b = q8_row_bytes(k).expect("validate checked the shape");
    let mut out = vec![0.0f32; m * k];
    for row in 0..m {
        let raw = &bytes[row * row_b..(row + 1) * row_b];
        let codes = &raw[..k];
        let scales = &raw[k..];
        for col in 0..k {
            let scale = load_f16(&scales[(col / Q8_GROUP) * 2..]);
            let q = f32::from(codes[col] as i8);
            out[row * k + col] = scale * q;
        }
    }
    Ok(out)
}

fn load_f16(bytes: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

fn code_at(codes: &[u8], index: usize, layout: Layout) -> u8 {
    match layout {
        Layout::Iu4 => {
            let byte = codes[index / 2];
            if index % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            }
        }
        Layout::Iu8 => codes[index],
    }
}

/// Dequantise an IU tensor back to f32 in the rotated basis.
///
/// `w[row, k] = f32(scale) * code + f32(zero)` with the pair for
/// `k / 128`. This is the CPU oracle a kernel test should match.
pub fn dequant(bytes: &[u8], m: usize, k: usize, layout: Layout) -> Result<Vec<f32>, String> {
    validate(bytes, m, k, layout)?;
    let row_b = row_bytes(k, layout).expect("validate checked the shape");
    let code_b = code_bytes(k, layout).expect("validate checked the shape");
    let mut out = vec![0.0f32; m * k];
    for row in 0..m {
        let raw = &bytes[row * row_b..(row + 1) * row_b];
        let codes = &raw[..code_b];
        let scales = &raw[code_b..];
        for col in 0..k {
            let pair = (col / HALF) * SCALE_PAIR_BYTES;
            let scale = load_f16(&scales[pair..pair + 2]);
            let zero = load_f16(&scales[pair + 2..pair + 4]);
            let q = code_at(codes, col, layout);
            out[row * k + col] = scale * f32::from(q) + zero;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_mq4v2_group(scales: [u16; 2], zeros: [u16; 2], codes: &[u8; 256]) -> [u8; 136] {
        let mut out = [0u8; 136];
        out[0..2].copy_from_slice(&scales[0].to_le_bytes());
        out[2..4].copy_from_slice(&zeros[0].to_le_bytes());
        out[4..6].copy_from_slice(&scales[1].to_le_bytes());
        out[6..8].copy_from_slice(&zeros[1].to_le_bytes());
        for i in 0..128 {
            out[8 + i] = codes[2 * i] | (codes[2 * i + 1] << 4);
        }
        out
    }

    fn sample(m: usize, k: usize) -> (Vec<u8>, [u16; 2], [u16; 2]) {
        let gpr = k / GROUP;
        let scales = [f16::from_f32(0.25).to_bits(), f16::from_f32(0.5).to_bits()];
        let zeros = [
            f16::from_f32(-1.0).to_bits(),
            f16::from_f32(0.125).to_bits(),
        ];
        let mut src = Vec::with_capacity(m * gpr * MQ4V2_GROUP_BYTES);
        for row in 0..m {
            for g in 0..gpr {
                let mut codes = [0u8; 256];
                for (i, code) in codes.iter_mut().enumerate() {
                    *code = ((row * 3 + g * 5 + i) % 16) as u8;
                }
                src.extend_from_slice(&pack_mq4v2_group(scales, zeros, &codes));
            }
        }
        (src, scales, zeros)
    }

    #[test]
    fn stored_width_matches_mq4v2_for_iu4_and_doubles_codes_for_iu8() {
        assert_eq!(row_bytes(256, Layout::Iu4), Some(136));
        assert_eq!(row_bytes(256, Layout::Iu8), Some(264));
        assert_eq!(row_bytes(512, Layout::Iu4), Some(272));
        assert_eq!(code_bytes(512, Layout::Iu8), Some(512));
        assert_eq!(scale_bytes(512), Some(16));
        assert!((stored_bpw(Layout::Iu4) - 4.25).abs() < 1e-6);
        assert!((stored_bpw(Layout::Iu8) - 8.25).abs() < 1e-6);
        assert_eq!(row_bytes(128, Layout::Iu4), None);
        assert_eq!(tensor_bytes(0, 256, Layout::Iu4), None);
        assert_eq!(Layout::Iu4.quant_type(), 57);
        assert_eq!(Layout::Iu8.quant_type(), 58);
        assert_eq!(Layout::from_format("iu4g256"), Some(Layout::Iu4));
        assert_eq!(Layout::from_format("mq4"), None);
    }

    #[test]
    fn repack_keeps_mq4v2_nibbles_and_both_layouts_dequant_equal() {
        let m = 2;
        let k = 512;
        let (src, scales, zeros) = sample(m, k);
        let iu4 = repack_mq4v2(&src, m, k, Layout::Iu4).unwrap();
        let iu8 = repack_mq4v2(&src, m, k, Layout::Iu8).unwrap();
        assert_eq!(iu4.len(), m * 272);
        assert_eq!(iu8.len(), m * (512 + 16));

        // First 256-group of row 0: nibbles copied verbatim, header moved to the scale tail.
        assert_eq!(&iu4[..128], &src[8..136]);
        assert_eq!(&iu4[256..264], &src[0..8]);
        let scale0 = f16::from_bits(scales[0]).to_f32();
        let zero0 = f16::from_bits(zeros[0]).to_f32();
        assert_eq!(iu8[0], src[8] & 0x0f);
        assert_eq!(iu8[1], src[8] >> 4);
        assert!(iu8[..512].iter().all(|b| *b <= 15));

        let d4 = dequant(&iu4, m, k, Layout::Iu4).unwrap();
        let d8 = dequant(&iu8, m, k, Layout::Iu8).unwrap();
        assert_eq!(d4, d8);
        let q0 = (src[8] & 0x0f) as f32;
        assert!((d4[0] - (scale0 * q0 + zero0)).abs() < 1e-6);

        // A code past 15 is not an MQ4 nibble.
        let mut bad = iu8.clone();
        bad[3] = 16;
        assert!(validate(&bad, m, k, Layout::Iu8).is_err());
        assert!(repack_mq4v2(&src[..src.len() - 1], m, k, Layout::Iu4).is_err());
    }

    #[test]
    fn q8_repack_keeps_signed_codes_and_moves_scales_to_the_tail() {
        let m = 2usize;
        let k = 64usize;
        let scales = [f16::from_f32(0.125).to_bits(), f16::from_f32(-0.5).to_bits()];
        let mut src = vec![0u8; m * (k / Q8_GROUP) * Q8_BLOCK_BYTES];
        for row in 0..m {
            for g in 0..(k / Q8_GROUP) {
                let base = (row * (k / Q8_GROUP) + g) * Q8_BLOCK_BYTES;
                src[base..base + 2].copy_from_slice(&scales[g].to_le_bytes());
                for i in 0..Q8_GROUP {
                    // 200 is i8 -56, which qt 58 would reject as a nibble.
                    let code: i8 = if i == 0 { -56 } else { ((row + g + i) as i8) % 20 };
                    src[base + 2 + i] = code as u8;
                }
            }
        }
        let packed = repack_q8f16(&src, m, k).unwrap();
        assert_eq!(packed.len(), m * (k + 4));
        assert_eq!(q8_row_bytes(32), Some(34));
        assert_eq!(q8_row_bytes(256), Some(272));
        assert_eq!(q8_row_bytes(31), None);
        assert_eq!(Q8_QUANT_TYPE, 59);
        // Row 0: codes first, then the two scale words that headed the blocks.
        assert_eq!(packed[0], (-56i8) as u8);
        assert_eq!(&packed[k..k + 2], &scales[0].to_le_bytes());
        assert_eq!(&packed[k + 2..k + 4], &scales[1].to_le_bytes());
        let got = dequant_q8(&packed, m, k).unwrap();
        let scale0 = f16::from_bits(scales[0]).to_f32();
        assert!((got[0] - scale0 * f32::from(-56i8)).abs() < 1e-6);
        // qt 58 stores MQ4 nibbles. A signed Q8 code does not fit that range.
        assert!(packed[0] > 15);
        assert!(repack_q8f16(&src, m, 48).is_err());
    }
}
