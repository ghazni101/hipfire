// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Double-precision, 64-bit integer and carry-out vector opcodes (wave32).
//!
//! Pseudocode is the RDNA3/RDNA4 ISA text. f64 arithmetic is host IEEE binary64 (RNE; the launch mode checked by
//! `State::new` keeps f64/f16 denormals), bit-exact for every non-NaN operand and result. No NaN behaviour of the
//! double-precision opcodes is measured on any architecture, so a NaN operand or a NaN result (`inf - inf`) is a hard
//! error. `v_cvt_i32_f64` documents its NaN result (0) and saturation, which Rust's cast implements exactly.
//! Carry/borrow masks are wave32 dwords whose inactive lanes are zero (ISA: "Inactive lanes write zero into this
//! carry-out"). DPP, SDWA, OMOD and OP_SEL have no meaning here and are errors.
use super::{Result, State, Wide};
use crate::{operand::Omod, Arch, Inst};

fn stem(name: &str) -> &str {
    name.strip_suffix("_e32").or_else(|| name.strip_suffix("_e64")).unwrap_or(name)
}

pub(super) fn handles(name: &str) -> bool {
    matches!(
        stem(name),
        "v_cvt_f64_i32" | "v_cvt_f64_u32" | "v_cvt_f64_f32" | "v_cvt_i32_f64" | "v_cvt_f32_f64" | "v_add_f64" | "v_min_f64"
            | "v_lshrrev_b64" | "v_subrev_co_u32" | "v_sub_co_ci_u32"
    )
}

fn clamp01(x: f64) -> f64 {
    if x <= 0.0 { 0.0 } else if x >= 1.0 { 1.0 } else { x }
}

/// Source modifiers of a double: absolute value first, negate after.
fn modified(bits: u64, abs: bool, neg: bool) -> u64 {
    let bits = if abs { bits & !(1 << 63) } else { bits };
    if neg { bits ^ (1 << 63) } else { bits }
}

pub(super) fn execute(_arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let m = &i.mods;
    let o = &i.operands;
    let op = stem(name);
    if m.dpp.is_some() || m.sdwa.is_some() {
        return Err(format!("{name}: DPP/SDWA is not modeled"));
    }
    if m.op_sel != 0 || m.omod != Omod::None {
        return Err(format!("{name}: op_sel/omod have no modeled meaning"));
    }
    let float_src = matches!(op, "v_cvt_f64_f32" | "v_cvt_i32_f64" | "v_cvt_f32_f64" | "v_add_f64" | "v_min_f64");
    if !float_src && m.neg | m.abs != 0 {
        return Err(format!("{name}: neg/abs on an integer source"));
    }
    if m.clamp && !matches!(op, "v_cvt_i32_f64" | "v_cvt_f32_f64" | "v_add_f64" | "v_min_f64") {
        return Err(format!("{name}: clamp is not modeled"));
    }
    let src = |idx: usize| o.get(idx).ok_or_else(|| format!("{name}: missing operand {idx}"));
    let bit = |n: u8, k: u8| n >> k & 1 != 0;
    let mut vals = [0u64; 32];
    let mut carry = 0u32;
    let explicit_carry = matches!(op, "v_sub_co_ci_u32") && o.len() >= 5 || op == "v_subrev_co_u32";
    for lane in 0..32 {
        if s.exec >> lane & 1 == 0 {
            continue;
        }
        let f64_src = |idx: usize, k: u8| -> Result<f64> {
            let x = f64::from_bits(modified(s.read_wide(src(idx)?, lane, Wide::Float)?, bit(m.abs, k), bit(m.neg, k)));
            if x.is_nan() {
                return Err(format!("{name}: NaN double operand: propagation is not measured"));
            }
            Ok(x)
        };
        vals[lane] = match op {
            "v_cvt_f64_i32" => f64::from(s.read(src(1)?, lane, 0)? as i32).to_bits(),
            "v_cvt_f64_u32" => f64::from(s.read(src(1)?, lane, 0)?).to_bits(),
            "v_cvt_f64_f32" => {
                let bits = (modified(u64::from(s.read(src(1)?, lane, 0)?) << 32, bit(m.abs, 0), bit(m.neg, 0)) >> 32) as u32;
                let x = f32::from_bits(bits);
                if x.is_nan() {
                    return Err(format!("{name}: NaN f32 operand: quieting/payload is not measured"));
                }
                f64::from(x).to_bits()
            }
            // Truncate toward zero, saturate (infinity included), NaN -> 0: Rust's `as` is exactly that.
            "v_cvt_i32_f64" => {
                let x = f64::from_bits(modified(s.read_wide(src(1)?, lane, Wide::Float)?, bit(m.abs, 0), bit(m.neg, 0)));
                u64::from(x as i32 as u32)
            }
            "v_cvt_f32_f64" => {
                let y = f64_src(1, 0)? as f32;
                let y = if m.clamp { clamp01(f64::from(y)) as f32 } else { y };
                u64::from(y.to_bits())
            }
            "v_add_f64" => {
                let r = f64_src(1, 0)? + f64_src(2, 1)?;
                if r.is_nan() {
                    return Err(format!("{name}: NaN double result (inf - inf): default NaN is not measured"));
                }
                (if m.clamp { clamp01(r) } else { r }).to_bits()
            }
            "v_min_f64" => {
                let (a, b) = (f64_src(1, 0)?, f64_src(2, 1)?);
                let r = if a < b || (a == 0.0 && b == 0.0 && a.is_sign_negative() && !b.is_sign_negative()) { a } else { b };
                (if m.clamp { clamp01(r) } else { r }).to_bits()
            }
            // D0.u64 = S1.u64 >> S0[5:0]
            "v_lshrrev_b64" => s.read_wide(src(2)?, lane, Wide::Unsigned)? >> (s.read(src(1)?, lane, 0)? & 63),
            // tmp = S1 - S0; carry-out = S0 > S1 (unsigned borrow).
            "v_subrev_co_u32" => {
                let (a, b) = (s.read(src(2)?, lane, 0)?, s.read(src(3)?, lane, 0)?);
                carry |= u32::from(a > b) << lane;
                u64::from(b.wrapping_sub(a))
            }
            // tmp = S0 - S1 - carry_in; carry-out = S1 + carry_in > S0 (unsigned).
            "v_sub_co_ci_u32" => {
                let (a_idx, b_idx) = if o.len() >= 5 { (2, 3) } else { (1, 2) };
                let (a, b) = (s.read(src(a_idx)?, lane, 0)?, s.read(src(b_idx)?, lane, 0)?);
                let cin = if o.len() >= 5 { s.read(src(4)?, 0, 0)? } else { s.vcc } >> lane & 1;
                carry |= u32::from(u64::from(b) + u64::from(cin) > u64::from(a)) << lane;
                u64::from(a.wrapping_sub(b).wrapping_sub(cin))
            }
            _ => return Err(format!("unsupported wide opcode {name}")),
        };
    }
    for lane in 0..32 {
        if s.exec >> lane & 1 == 0 {
            continue;
        }
        if matches!(op, "v_cvt_f64_i32" | "v_cvt_f64_u32" | "v_cvt_f64_f32" | "v_add_f64" | "v_min_f64" | "v_lshrrev_b64") {
            s.put_wide(&o[0], lane, vals[lane])?;
        } else {
            s.put(&o[0], lane, 0, vals[lane] as u32)?;
        }
    }
    match op {
        "v_subrev_co_u32" | "v_sub_co_ci_u32" if explicit_carry => s.put(src(1)?, 0, 0, carry)?,
        "v_sub_co_ci_u32" => s.vcc = carry,
        _ => {}
    }
    Ok(())
}
