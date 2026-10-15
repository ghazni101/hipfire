// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `I4ExpertFinalCodeRecord`: one expert's final MQ3-I4 / MQ2-I4 codes and f16
//! scale bits (DESIGN F2), plus the importer from the R1 calibration NPZ schema.
//!
//! The record holds *final* codes — the packer never re-quantises. Import
//! validates the NPZ identity (`bits`, `grid`, `scale_rule`, unit `sd`), converts
//! the f16-valued F32 scale arrays exactly (no rounding), range-checks codes and
//! then runs the canonical F1 encoder ([`mq_i4::encode_tensor`], which applies
//! the full F1 validation). Every record is decoded again and compared with the
//! imported codes/scales before it is returned; that check cannot be disabled.
//!
//! R1 NPZ schema (one file per expert, `L{L}/E{e}.npz`):
//! `gds f32[Mgu, Kgu/128]`, `gc u8[Mgu, Kgu]`, `dds f32[Md, Kd/128]`,
//! `dc u8[Md, Kd]`, `sd f32[Kd]` (must be exactly `1.0` everywhere — there is no
//! runtime alpha vector), `bits i64`, `grid <U`, `scale_rule <U`.

use std::fmt;
use std::path::Path;

use hipfire_arch_qwen4::expert_slabs::{expert_entry_name, ExpertFormat, ExpertProj};
use hipfire_runtime::mq_i4::{self, MqI4Error, MqI4Profile, GROUP};
use sha2::{Digest, Sha256};

use crate::npz::{NpzArchive, NpzError};

/// The only grid the approved MQ3-I4 / MQ2-I4 wire types may carry. The
/// `signed_27` challenger is calibration-only and is rejected at import.
pub const GRID_UNIFORM: &str = "uniform";

/// Explicit allow-list of recipe identifiers accepted at import.
pub const ALLOWED_SCALE_RULES: [&str; 3] = [
    "mqi4-canonical-v1-rne-tree128",
    "mqi4-recover-s1-v1",
    "mqi4-recover-s2-v1",
];

/// Lower-case hex of `bytes`.
pub fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 15)] as char);
    }
    out
}

/// SHA-256 of `bytes`, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// `[M, K]` of the two projections of an expert. Tests use the tiny fixture
/// geometry; the FN CLI pins [`I4RecordShapes::FN`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I4RecordShapes {
    pub gate_up: [usize; 2],
    pub down: [usize; 2],
}

impl I4RecordShapes {
    /// Flash-Next: gate_up `[1280, 2560]` (R256), down `[2560, 640]` (R128).
    pub const FN: Self = Self {
        gate_up: [1280, 2560],
        down: [2560, 640],
    };
}

/// Failure importing, validating or encoding a record.
#[derive(Debug)]
pub enum I4RecordError {
    Npz(NpzError),
    Io(String),
    /// `bits` is neither 3 nor 2.
    UnsupportedBits(i64),
    /// The NPZ's `bits` does not match the format the caller expects.
    FormatMismatch { expected: ExpertFormat, bits: i64 },
    /// A record can only hold MQ3-I4 or MQ2-I4, never `Mq4Xts`.
    NotAnI4Format(ExpertFormat),
    GridNotUniform(String),
    ScaleRuleNotAllowed(String),
    /// `sd` has a value that is not exactly `1.0` (no runtime alpha exists).
    NonUnitSd { index: usize, bits: u32 },
    /// A scale is not an exact non-negative finite f16 (no rounding at import).
    Scale {
        projection: &'static str,
        index: usize,
        error: MqI4Error,
    },
    /// A code exceeds `L = 2^b - 1`.
    CodeOutOfRange {
        projection: &'static str,
        row: usize,
        col: usize,
        code: u8,
        max: u8,
    },
    /// F1 shape/length/zero-scale validation failed.
    Codec {
        projection: &'static str,
        error: MqI4Error,
    },
    /// decode(encode(x)) != x.
    RoundTrip { projection: &'static str },
}

impl fmt::Display for I4RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Npz(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "record i/o error: {e}"),
            Self::UnsupportedBits(b) => {
                write!(f, "record bits={b}: only 3 (MQ3-I4) and 2 (MQ2-I4) are supported")
            }
            Self::FormatMismatch { expected, bits } => write!(
                f,
                "record bits={bits} does not match the expected format {expected:?}"
            ),
            Self::NotAnI4Format(fmt_) => {
                write!(f, "{fmt_:?} is not an MQ3-I4/MQ2-I4 record format")
            }
            Self::GridNotUniform(g) => write!(
                f,
                "record grid `{g}` rejected: only `{GRID_UNIFORM}` may be emitted under qt131-134"
            ),
            Self::ScaleRuleNotAllowed(r) => {
                write!(f, "record scale_rule `{r}` is not in the allow-list {ALLOWED_SCALE_RULES:?}")
            }
            Self::NonUnitSd { index, bits } => write!(
                f,
                "record sd[{index}] bits {bits:#010x} != 1.0: runtime alpha is not supported"
            ),
            Self::Scale {
                projection,
                index,
                error,
            } => write!(f, "{projection} scale[{index}]: {error}"),
            Self::CodeOutOfRange {
                projection,
                row,
                col,
                code,
                max,
            } => write!(f, "{projection} code {code} at [{row},{col}] exceeds {max}"),
            Self::Codec { projection, error } => write!(f, "{projection}: {error}"),
            Self::RoundTrip { projection } => {
                write!(f, "{projection}: decode(encode(record)) differs from the record")
            }
        }
    }
}

impl std::error::Error for I4RecordError {}

impl From<NpzError> for I4RecordError {
    fn from(e: NpzError) -> Self {
        Self::Npz(e)
    }
}

/// One projection of an expert: final codes, f16 scale bits and the F1 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I4ProjectionRecord {
    /// Slab entry name, e.g. `model.language_model.layers.3.mlp.experts.17.gate_up_proj`.
    pub tensor_name: String,
    /// `[M, K]`.
    pub shape: [usize; 2],
    pub profile: MqI4Profile,
    /// One code per weight, row-major.
    pub codes: Vec<u8>,
    /// f16 bit patterns, `[M, K/128]` row-major.
    pub scale_bits: Vec<u16>,
    /// The encoded tensor (`codes plane ++ f16 scale plane`).
    pub bytes: Vec<u8>,
}

impl I4ProjectionRecord {
    fn encode(
        tensor_name: String,
        projection: &'static str,
        profile: MqI4Profile,
        shape: [usize; 2],
        codes: Vec<u8>,
        scale_bits: Vec<u16>,
    ) -> Result<Self, I4RecordError> {
        let [m, k] = shape;
        let codec = |error| I4RecordError::Codec { projection, error };
        // Shape legality (K % 128, K % R, size overflow) before any indexing.
        mq_i4::tensor_bytes(profile, m, k).map_err(codec)?;
        let max = profile.codec.levels();
        if let Some(pos) = codes.iter().position(|&c| c > max) {
            return Err(I4RecordError::CodeOutOfRange {
                projection,
                row: pos / k,
                col: pos % k,
                code: codes[pos],
                max,
            });
        }
        let bytes = mq_i4::encode_tensor(profile, m, k, &codes, &scale_bits).map_err(codec)?;
        let back = mq_i4::decode_tensor(profile, m, k, &bytes).map_err(codec)?;
        if back.codes != codes || back.scale_bits != scale_bits {
            return Err(I4RecordError::RoundTrip { projection });
        }
        Ok(Self {
            tensor_name,
            shape,
            profile,
            codes,
            scale_bits,
            bytes,
        })
    }
}

/// Final codes of one expert (gate_up + down) for MQ3-I4 or MQ2-I4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I4ExpertFinalCodeRecord {
    pub layer: usize,
    pub expert: usize,
    /// `Mq3I4` or `Mq2I4`; gives the qtype pair (131,132) / (133,134).
    pub format: ExpertFormat,
    pub gate_up: I4ProjectionRecord,
    pub down: I4ProjectionRecord,
    /// Recipe identifier (member of [`ALLOWED_SCALE_RULES`]).
    pub scale_rule: String,
    /// Always [`GRID_UNIFORM`].
    pub grid: String,
    /// SHA-256 (hex) of the source NPZ bytes (empty when built from parts).
    pub source_npz_sha256: String,
}

impl I4ExpertFinalCodeRecord {
    /// Build and fully validate a record from already-decoded codes and f16
    /// scale bits.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        layer: usize,
        expert: usize,
        format: ExpertFormat,
        shapes: I4RecordShapes,
        gate_up: (Vec<u8>, Vec<u16>),
        down: (Vec<u8>, Vec<u16>),
        scale_rule: &str,
        grid: &str,
        source_npz_sha256: String,
    ) -> Result<Self, I4RecordError> {
        if format == ExpertFormat::Mq4Xts {
            return Err(I4RecordError::NotAnI4Format(format));
        }
        if grid != GRID_UNIFORM {
            return Err(I4RecordError::GridNotUniform(grid.to_string()));
        }
        if !ALLOWED_SCALE_RULES.contains(&scale_rule) {
            return Err(I4RecordError::ScaleRuleNotAllowed(scale_rule.to_string()));
        }
        let (gu_qt, down_qt) = format.pair();
        let gu_profile = MqI4Profile::from_quant_type(gu_qt).expect("I4 format has a gate_up profile");
        let down_profile =
            MqI4Profile::from_quant_type(down_qt).expect("I4 format has a down profile");
        let gate_up = I4ProjectionRecord::encode(
            expert_entry_name(layer, expert, ExpertProj::GateUp),
            "gate_up",
            gu_profile,
            shapes.gate_up,
            gate_up.0,
            gate_up.1,
        )?;
        let down = I4ProjectionRecord::encode(
            expert_entry_name(layer, expert, ExpertProj::Down),
            "down",
            down_profile,
            shapes.down,
            down.0,
            down.1,
        )?;
        Ok(Self {
            layer,
            expert,
            format,
            gate_up,
            down,
            scale_rule: scale_rule.to_string(),
            grid: grid.to_string(),
            source_npz_sha256,
        })
    }

    /// Payload bytes of the whole slab (gate_up ++ down).
    pub fn slab_len(&self) -> usize {
        self.gate_up.bytes.len() + self.down.bytes.len()
    }

    /// Import one expert from the bytes of an R1 NPZ. `expected` is the format
    /// the caller's record directory is supposed to hold (`bits` must agree).
    pub fn import_npz_bytes(
        npz_bytes: &[u8],
        layer: usize,
        expert: usize,
        expected: ExpertFormat,
        shapes: I4RecordShapes,
    ) -> Result<Self, I4RecordError> {
        let npz_sha = sha256_hex(npz_bytes);
        let z = NpzArchive::from_bytes(npz_bytes)?;
        let bits = z.i64_scalar("bits")?;
        let format = match bits {
            3 => ExpertFormat::Mq3I4,
            2 => ExpertFormat::Mq2I4,
            other => return Err(I4RecordError::UnsupportedBits(other)),
        };
        if format != expected {
            return Err(I4RecordError::FormatMismatch { expected, bits });
        }
        let grid = z.str_scalar("grid")?;
        let scale_rule = z.str_scalar("scale_rule")?;
        // Check identity before decoding any large array.
        if grid != GRID_UNIFORM {
            return Err(I4RecordError::GridNotUniform(grid.to_string()));
        }
        if !ALLOWED_SCALE_RULES.contains(&scale_rule) {
            return Err(I4RecordError::ScaleRuleNotAllowed(scale_rule.to_string()));
        }
        let [mg, kg] = shapes.gate_up;
        let [md, kd] = shapes.down;
        if kg % GROUP != 0 || kd % GROUP != 0 {
            return Err(I4RecordError::Codec {
                projection: "shape",
                error: MqI4Error::KNotGroupAligned {
                    k: if kg % GROUP != 0 { kg } else { kd },
                },
            });
        }
        let gds = z.f32_array("gds", &[mg, kg / GROUP])?;
        let gc = z.u8_array("gc", &[mg, kg])?;
        let dds = z.f32_array("dds", &[md, kd / GROUP])?;
        let dc = z.u8_array("dc", &[md, kd])?;
        let sd = z.f32_array("sd", &[kd])?;
        if let Some(index) = sd.iter().position(|v| v.to_bits() != 1.0f32.to_bits()) {
            return Err(I4RecordError::NonUnitSd {
                index,
                bits: sd[index].to_bits(),
            });
        }
        let gate_up_scales = scales_to_bits("gate_up", gds)?;
        let down_scales = scales_to_bits("down", dds)?;
        Self::from_parts(
            layer,
            expert,
            format,
            shapes,
            (gc.to_vec(), gate_up_scales),
            (dc.to_vec(), down_scales),
            scale_rule,
            grid,
            npz_sha,
        )
    }

    /// [`Self::import_npz_bytes`] from a file path.
    pub fn import_npz_file(
        path: &Path,
        layer: usize,
        expert: usize,
        expected: ExpertFormat,
        shapes: I4RecordShapes,
    ) -> Result<Self, I4RecordError> {
        let bytes = std::fs::read(path)
            .map_err(|e| I4RecordError::Io(format!("{}: {e}", path.display())))?;
        Self::import_npz_bytes(&bytes, layer, expert, expected, shapes)
    }
}

fn scales_to_bits(projection: &'static str, scales: &[f32]) -> Result<Vec<u16>, I4RecordError> {
    scales
        .iter()
        .enumerate()
        .map(|(index, &v)| {
            mq_i4::scale_bits_from_f16_valued_f32(v).map_err(|error| I4RecordError::Scale {
                projection,
                index,
                error,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::testutil::build_npz;
    use crate::npz::{NpyArray, NpyData};
    use std::collections::BTreeMap;

    const W3: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w3.npz");
    const W2: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w2.npz");
    const W3_EXPECTED: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w3.expected.bin");
    const W2_EXPECTED: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w2.expected.bin");

    const TINY: I4RecordShapes = I4RecordShapes {
        gate_up: [4, 256],
        down: [8, 128],
    };

    fn import(bytes: &[u8], expected: ExpertFormat) -> Result<I4ExpertFinalCodeRecord, I4RecordError> {
        I4ExpertFinalCodeRecord::import_npz_bytes(bytes, 5, 9, expected, TINY)
    }

    fn mutated(
        bytes: &[u8],
        f: impl FnOnce(&mut BTreeMap<String, NpyArray>),
    ) -> Vec<u8> {
        let mut arrays = NpzArchive::from_bytes(bytes).unwrap().into_arrays();
        f(&mut arrays);
        build_npz(&arrays)
    }

    fn f32s(a: &mut NpyArray) -> &mut Vec<f32> {
        match &mut a.data {
            NpyData::F32(v) => v,
            _ => panic!("not f32"),
        }
    }

    fn u8s(a: &mut NpyArray) -> &mut Vec<u8> {
        match &mut a.data {
            NpyData::U8(v) => v,
            _ => panic!("not u8"),
        }
    }

    #[test]
    fn fixtures_encode_to_the_independent_python_packer_bytes() {
        for (npz, expected, format, npz_sha) in [
            (
                W3,
                W3_EXPECTED,
                ExpertFormat::Mq3I4,
                "9917fec8f16447a0989649992e5eb94579e7d96fbbd3f5eedcc8270dd77ce1be",
            ),
            (
                W2,
                W2_EXPECTED,
                ExpertFormat::Mq2I4,
                "4fc8200ff3fafb4d73287b927a60b98f7a6ebde7beaf069f153bb622a778ee60",
            ),
        ] {
            let rec = import(npz, format).unwrap();
            assert_eq!(rec.source_npz_sha256, npz_sha);
            assert_eq!((rec.layer, rec.expert, rec.format), (5, 9, format));
            assert_eq!(rec.grid, "uniform");
            assert_eq!(rec.scale_rule, "mqi4-canonical-v1-rne-tree128");
            let mut got = rec.gate_up.bytes.clone();
            got.extend_from_slice(&rec.down.bytes);
            assert_eq!(got.len(), expected.len());
            assert_eq!(got, expected, "{format:?} bytes differ from the Python packer");
            assert_eq!(rec.slab_len(), expected.len());
            assert_eq!(
                rec.gate_up.tensor_name,
                "model.language_model.layers.5.mlp.experts.9.gate_up_proj"
            );
            assert_eq!(
                rec.down.tensor_name,
                "model.language_model.layers.5.mlp.experts.9.down_proj"
            );
        }
        // README.json pins the expected-bin hashes.
        assert_eq!(
            sha256_hex(W3_EXPECTED),
            "83c69ed3929991c6ca08492dd7ffaa28cc0c74149d8d661a64702b3c7a2b8221"
        );
        assert_eq!(
            sha256_hex(W2_EXPECTED),
            "cc347c8feb98e99b94ce864896a4ff39e1b32ce15beb9009ef7f6b823c535502"
        );
    }

    #[test]
    fn decode_round_trips_to_the_npz_codes_and_scale_bits() {
        for (npz, format) in [(W3, ExpertFormat::Mq3I4), (W2, ExpertFormat::Mq2I4)] {
            let z = NpzArchive::from_bytes(npz).unwrap();
            let rec = import(npz, format).unwrap();
            for (proj, gk, dk, shape) in [
                (&rec.gate_up, "gc", "gds", [4usize, 256]),
                (&rec.down, "dc", "dds", [8usize, 128]),
            ] {
                let dec =
                    mq_i4::decode_tensor(proj.profile, shape[0], shape[1], &proj.bytes).unwrap();
                assert_eq!(dec.codes, z.u8_array(gk, &shape).unwrap());
                let want: Vec<u16> = z
                    .f32_array(dk, &[shape[0], shape[1] / 128])
                    .unwrap()
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect();
                assert_eq!(dec.scale_bits, want);
                assert_eq!(proj.codes, dec.codes);
                assert_eq!(proj.scale_bits, dec.scale_bits);
            }
        }
    }

    #[test]
    fn profiles_follow_the_format_pair() {
        let r = import(W3, ExpertFormat::Mq3I4).unwrap();
        assert_eq!(r.gate_up.profile, MqI4Profile::MQ3_R256);
        assert_eq!(r.down.profile, MqI4Profile::MQ3_R128);
        let r = import(W2, ExpertFormat::Mq2I4).unwrap();
        assert_eq!(r.gate_up.profile, MqI4Profile::MQ2_R256);
        assert_eq!(r.down.profile, MqI4Profile::MQ2_R128);
    }

    #[test]
    fn format_mismatch_and_foreign_bits_are_rejected() {
        assert!(matches!(
            import(W3, ExpertFormat::Mq2I4),
            Err(I4RecordError::FormatMismatch { .. })
        ));
        assert!(matches!(
            import(W2, ExpertFormat::Mq4Xts),
            Err(I4RecordError::FormatMismatch { .. })
        ));
        let bad = mutated(W3, |a| a.get_mut("bits").unwrap().data = NpyData::I64(vec![4]));
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::UnsupportedBits(4))
        ));
    }

    #[test]
    fn non_unit_sd_is_rejected() {
        for v in [0.5f32, 1.000_000_1, -1.0, 0.0, f32::NAN] {
            let bad = mutated(W3, |a| f32s(a.get_mut("sd").unwrap())[5] = v);
            assert!(
                matches!(
                    import(&bad, ExpertFormat::Mq3I4),
                    Err(I4RecordError::NonUnitSd { index: 5, .. })
                ),
                "sd={v}"
            );
        }
    }

    #[test]
    fn grid_signed_27_and_unknown_scale_rules_are_rejected() {
        let bad = mutated(W2, |a| {
            a.get_mut("grid").unwrap().data = NpyData::Str(vec!["signed_27".into()])
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq2I4),
            Err(I4RecordError::GridNotUniform(g)) if g == "signed_27"
        ));
        for rule in ["mqi4-legacy-rtz", "", "MQI4-CANONICAL-V1-RNE-TREE128"] {
            let bad = mutated(W3, |a| {
                a.get_mut("scale_rule").unwrap().data = NpyData::Str(vec![rule.into()])
            });
            assert!(
                matches!(
                    import(&bad, ExpertFormat::Mq3I4),
                    Err(I4RecordError::ScaleRuleNotAllowed(_))
                ),
                "rule {rule:?}"
            );
        }
    }

    #[test]
    fn every_allow_listed_scale_rule_imports() {
        for rule in ALLOWED_SCALE_RULES {
            let ok = mutated(W3, |a| {
                a.get_mut("scale_rule").unwrap().data = NpyData::Str(vec![rule.into()])
            });
            let rec = import(&ok, ExpertFormat::Mq3I4).unwrap();
            assert_eq!(rec.scale_rule, rule);
            assert_eq!(
                [rec.gate_up.bytes, rec.down.bytes].concat(),
                W3_EXPECTED,
                "the recipe label does not change the bytes"
            );
        }
    }

    #[test]
    fn code_above_l_is_rejected() {
        let bad = mutated(W3, |a| u8s(a.get_mut("gc").unwrap())[0] = 8);
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::CodeOutOfRange {
                projection: "gate_up",
                row: 0,
                col: 0,
                code: 8,
                max: 7
            })
        ));
        let bad = mutated(W2, |a| u8s(a.get_mut("dc").unwrap())[130] = 4);
        assert!(matches!(
            import(&bad, ExpertFormat::Mq2I4),
            Err(I4RecordError::CodeOutOfRange {
                projection: "down",
                row: 1,
                col: 2,
                code: 4,
                max: 3
            })
        ));
        // W3's 4 is legal for W3 but the same byte is illegal for W2 (> L = 3).
        let bad = mutated(W2, |a| u8s(a.get_mut("gc").unwrap())[7] = 4);
        assert!(import(&bad, ExpertFormat::Mq2I4).is_err());
    }

    #[test]
    fn non_f16_exact_negative_and_nonfinite_scales_are_rejected() {
        for (v, _label) in [
            (0.1f32, "not f16-exact"),
            (-0.5f32, "negative"),
            (-0.0f32, "negative zero"),
            (f32::INFINITY, "inf"),
            (f32::NAN, "nan"),
            (65520.0f32, "above f16 max"),
        ] {
            let bad = mutated(W3, |a| f32s(a.get_mut("gds").unwrap())[1] = v);
            assert!(
                matches!(
                    import(&bad, ExpertFormat::Mq3I4),
                    Err(I4RecordError::Scale {
                        projection: "gate_up",
                        index: 1,
                        ..
                    })
                ),
                "scale {v}"
            );
        }
        let bad = mutated(W2, |a| f32s(a.get_mut("dds").unwrap())[3] = 0.1);
        assert!(matches!(
            import(&bad, ExpertFormat::Mq2I4),
            Err(I4RecordError::Scale {
                projection: "down",
                index: 3,
                ..
            })
        ));
    }

    #[test]
    fn zero_scale_group_with_non_midpoint_code_is_rejected() {
        // gate_up row 1 group 1 (scale +0): columns 128..256 of row 1.
        for (npz, format, mid) in [(W3, ExpertFormat::Mq3I4, 4u8), (W2, ExpertFormat::Mq2I4, 2u8)] {
            let z = NpzArchive::from_bytes(npz).unwrap();
            let gc = z.u8_array("gc", &[4, 256]).unwrap();
            assert!(gc[256 + 128..256 + 256].iter().all(|&c| c == mid));
            let bad = mutated(npz, |a| u8s(a.get_mut("gc").unwrap())[256 + 130] = mid - 1);
            assert!(matches!(
                import(&bad, format),
                Err(I4RecordError::Codec {
                    projection: "gate_up",
                    error: MqI4Error::ZeroScaleNonMidpoint { row: 1, group: 1, .. }
                })
            ));
        }
    }

    #[test]
    fn missing_and_misshapen_members_are_rejected() {
        let bad = mutated(W3, |a| {
            a.remove("sd");
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::Npz(NpzError::MissingEntry(_)))
        ));
        let bad = mutated(W3, |a| {
            let sd = a.get_mut("sd").unwrap();
            f32s(sd).pop();
            sd.shape = vec![127];
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::Npz(NpzError::ShapeMismatch { .. }))
        ));
        // Wrong geometry parameters are caught by the shape check, too.
        let wrong = I4RecordShapes {
            gate_up: [4, 256],
            down: [8, 256],
        };
        assert!(
            I4ExpertFinalCodeRecord::import_npz_bytes(W3, 0, 0, ExpertFormat::Mq3I4, wrong)
                .is_err()
        );
    }

    #[test]
    fn from_parts_rejects_mq4_format_and_validates_directly() {
        let shapes = TINY;
        let ok = I4ExpertFinalCodeRecord::from_parts(
            0,
            0,
            ExpertFormat::Mq2I4,
            shapes,
            (vec![2; 4 * 256], vec![0x3C00; 8]),
            (vec![1; 8 * 128], vec![0x3C00; 8]),
            "mqi4-recover-s2-v1",
            "uniform",
            String::new(),
        );
        assert!(ok.is_ok());
        let err = I4ExpertFinalCodeRecord::from_parts(
            0,
            0,
            ExpertFormat::Mq4Xts,
            shapes,
            (vec![], vec![]),
            (vec![], vec![]),
            "mqi4-recover-s2-v1",
            "uniform",
            String::new(),
        );
        assert!(matches!(err, Err(I4RecordError::NotAnI4Format(_))));
    }
}
