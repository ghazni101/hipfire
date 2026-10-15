// SPDX-License-Identifier: Apache-2.0
//! Raw PM K1 MoE decode-GEMV G1 byte oracle; never participates in runtime dispatch.
//!
//! Loads the code object `mq_i4_moe_gemv_pm_gfx1201` and its four symbols
//! `mq_i4_moe_gemv_{mq3i4,mq4sym}_{r256_gate_up,r128_down}`, launches them with the
//! frozen 88-byte F4 kernarg ABI (DESIGN §4.2, block 32, grid `(M, U, 1)`) and compares every
//! output word bit-for-bit (signed zeros and NaN payloads included) with a CPU oracle:
//!
//! * W3 (`mq3i4`): `mq_i4_reference::decode_moe_slots` (cross-checked against
//!   `decode_gemv_f32` on a bounded slot sample).
//! * MQ4 (`mq4sym`): [`mq4_row_gate`]/[`mq4_row_down`], a transcription of the pinned legacy
//!   MQ4 `.hip` arithmetic (`mq_i4_reference` supports only W3/W2): per 8 (gate) / 4 (down)
//!   elements `d_i = fma(sc, float(q_i), zp)`, `s = RN(d1*x1)` then `fma(d0,x0,s)` then
//!   `fma(d_i,x_i,s)` ascending (the hipcc `-ffp-contract=fast` orientation audited on
//!   gfx1201 ISA for both `qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`
//!   {k8 loop, k2560 rows8 helper} and `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip`),
//!   gate: `acc[g&3] += s` over G256 groups folded `(a0+a1)+(a2+a3)`, down: one chain over
//!   G128 groups, then the 16,8,4,2,1 wave shuffle tree. `--old-kernel-parity` additionally runs the
//!   production HIP kernels (JIT through `rdna_compute::Gpu`) on all-MQ4 fixtures and compares them
//!   bit-for-bit with this CPU oracle, so the transcription never rests on the ISA audit alone.
//!
//! MQ4 headers are symmetric (`zp = f16(-8*sc)`), so the legitimate finite maximum MQ4 scale is
//! f16 0x6fff (8188; `-8*sc` = -65504); W3 fixtures use the full f16 max 0x7bff (65504).
//!
//! Build:
//!   cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_byte_oracle
//! G1 (expected card, fixtures first, then optional real layer):
//!   target/release/examples/mq_i4_moe_gemv_byte_oracle --codeobject /abs/mq_i4_moe_gemv_pm_gfx1201.hxaco \
//!     [--artifact /abs/ctl3-hot.hfq --layer 0 [--real-experts 24]] [--old-kernel-parity] \
//!     [--device N] [--card NAME] [--expect-uuid GPU-… --expect-bdf 0000:… --expect-arch gfx1201] [--only SUBSTR]…
//! Certified R1 records (real weights, no packed HFQM; cases labelled origin `real_records_layer`):
//!   … --records-dir …/r1/q/w3-gptq3/L0 --records-manifest …/r1/pack/w3-gptq3/L0/manifest.json \
//!     --alloc …/alloc.ctl3hot.json [--point ctl3-hot] --xts /models/….mq4 --layer0 \
//!     [--map …/ctl3-hot.map.json] [--records-manifest-sha256 HEX] [--xts-sha256 HEX] [--real-experts 24]
//!   Cold = W3 experts imported with `I4ExpertFinalCodeRecord::import_npz_file` (NPZ sha256 and encoded
//!   gate++down sha256 checked against the manifest); hot = MQ4-XTS per-expert slices of the XTS aggregate
//!   tensors read through `HfqFile`. Activations stay synthetic. Runs after the 13 synthetic cases.
//! Timing (separate fabric run after G1 passed; raw device-event samples, R6 takes the median):
//!   … --codeobject … --timing --reps 30 --warmup 5 [--launches 32] --card NAME \
//!     --fabric-owner OWNER --fabric-wrapper /path/fabric-run.sh [--fabric-card NAME]
//!   (`--card`/`--fabric-owner`/`--fabric-wrapper` fall back to env `FABRIC_CARD`/`FABRIC_OWNER`/`FABRIC_WRAPPER`.)
//! Prints one machine JSON line on stdout; exit code 1 on any mismatch or error.
use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use hipfire_arch_qwen4::expert_slabs::{ExpertFormat, ExpertProj};
use hipfire_quantize::i4_records::{I4ExpertFinalCodeRecord, I4RecordShapes};
use hipfire_quantize::i4_slabs::{load_allocation, SlabGeometry};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::mq_i4::{self, MqI4Profile};
use hipfire_runtime::mq_i4_reference::{self, DecodeKind, ExpertTensor};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const SCHEMA: &str = "mq-i4-moe-gemv-byte-oracle-v1";
const MODULE: &str = "mq_i4_moe_gemv_pm_gfx1201";
const BUILD_COMMAND: &str =
    "cargo build -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_byte_oracle";
const DEFAULT_UUID: &str = "GPU-6109a4cb5f833235";
const DEFAULT_BDF: &str = "0000:9a:00.0";
const DEFAULT_ARCH: &str = "gfx1201";
const KERNARG_BYTES: usize = 88;
const N_EXPERTS: usize = 512;
const GATE_M: usize = 1280;
const GATE_K: usize = 2560;
const DOWN_M: usize = 2560;
const DOWN_K: usize = 640;
/// Unwritten output words: the byte pattern `memset(0xa5)` leaves (a NaN, compared by bits).
const SENTINEL_BITS: u32 = 0xa5a5_a5a5;
const PAD_BITS: u32 = 0x7fc0_0000;

// ---------------------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Codec {
    W3,
    Mq4,
}
impl Codec {
    const ALL: [Codec; 2] = [Codec::W3, Codec::Mq4];
    fn tag(self) -> &'static str {
        match self {
            Codec::W3 => "mq3i4",
            Codec::Mq4 => "mq4sym",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Codec::W3 => "w3",
            Codec::Mq4 => "mq4",
        }
    }
    /// Codec name of the R6 timing records.
    fn timing_name(self) -> &'static str {
        match self {
            Codec::W3 => "mq3i4",
            Codec::Mq4 => "mq4",
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    GateUp,
    Down,
}
impl Kind {
    const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    fn suffix(self) -> &'static str {
        match self {
            Kind::GateUp => "r256_gate_up",
            Kind::Down => "r128_down",
        }
    }
    /// Profile name of the R6 timing records.
    fn timing_profile(self) -> &'static str {
        match self {
            Kind::GateUp => "gate_up",
            Kind::Down => "down",
        }
    }
    fn m(self) -> usize {
        match self {
            Kind::GateUp => GATE_M,
            Kind::Down => DOWN_M,
        }
    }
    fn k(self) -> usize {
        match self {
            Kind::GateUp => GATE_K,
            Kind::Down => DOWN_K,
        }
    }
    /// Rows written to each of Y0/Y1 per slot.
    fn out_rows(self) -> usize {
        match self {
            Kind::GateUp => GATE_M / 2,
            Kind::Down => DOWN_M,
        }
    }
    fn x_row_div(self) -> usize {
        match self {
            Kind::GateUp => 10,
            Kind::Down => 1,
        }
    }
    fn decode_kind(self) -> DecodeKind {
        match self {
            Kind::GateUp => DecodeKind::GateUp,
            Kind::Down => DecodeKind::Down,
        }
    }
    fn w3_profile(self) -> MqI4Profile {
        match self {
            Kind::GateUp => MqI4Profile::MQ3_R256,
            Kind::Down => MqI4Profile::MQ3_R128,
        }
    }
    /// Bytes of one expert's projection tensor.
    fn tensor_bytes(self, codec: Codec) -> usize {
        match codec {
            Codec::W3 => mq_i4::tensor_bytes(self.w3_profile(), self.m(), self.k())
                .expect("FN shapes are legal W3 shapes"),
            Codec::Mq4 => match self {
                Kind::GateUp => GATE_M * (GATE_K / 256) * 136,
                Kind::Down => DOWN_M * (DOWN_K / 128) * 68,
            },
        }
    }
}
fn symbol(codec: Codec, kind: Kind) -> String {
    format!("mq_i4_moe_gemv_{}_{}", codec.tag(), kind.suffix())
}

// ---------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------

struct Rng(u64);
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(mix64(seed))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}
fn name_seed(name: &str) -> u64 {
    let d = Sha256::digest(name.as_bytes());
    u64::from_le_bytes(d[..8].try_into().unwrap())
}
fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn f16f(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
/// Symmetric MQ4 zero: `f16(-8 * sc)` (exact for every finite `sc <= 0x6fff`).
fn sym_zero_bits(sc: u16) -> u16 {
    half::f16::from_f32(-8.0 * f16f(sc)).to_bits()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64_bytes(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bits_hex(v: u32) -> String {
    format!("0x{v:08x}")
}

// ---------------------------------------------------------------------------------------
// MQ4 exact CPU oracle (transcription of the pinned legacy MQ4 HIP kernels)
// ---------------------------------------------------------------------------------------

/// Wave32 shuffle-down tree 16,8,4,2,1; lane 0 is the result.
fn lane_tree(mut lanes: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let snap = lanes;
        for l in 0..32 - off {
            lanes[l] = snap[l] + snap[l + off];
        }
    }
    lanes[0]
}

/// One lane's eight-element group dot: `d_i = fma(sc, q_i, zp)`; `RN(d1*x1)`, `fma(d0,x0)`,
/// then `fma(d_i, x_i)` for i = 2..8 (the "element 1 first" contraction hipcc picks on gfx1201).
fn group_dot8(sc: f32, zp: f32, pk: u32, x: &[f32]) -> f32 {
    let d = |i: usize| sc.mul_add(((pk >> (4 * i)) & 15) as f32, zp);
    let mut s = d(1) * x[1];
    s = d(0).mul_add(x[0], s);
    for i in 2..8 {
        s = d(i).mul_add(x[i], s);
    }
    s
}

/// MQ4G256V2 gate/up row: `row.len() / 136` groups of `[sc0 zp0 sc1 zp1 : f16][128 nibble bytes]`.
/// Lane `tid` reads header half `tid >= 16`, u32 `tid` of the payload (elements `tid*8..+8`).
fn mq4_row_gate(row: &[u8], x: &[f32]) -> f32 {
    let groups = row.len() / 136;
    debug_assert_eq!(x.len(), groups * 256);
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let hoff = if tid < 16 { 0 } else { 4 };
        let mut acc = [0.0f32; 4];
        for g in 0..groups {
            let gp = &row[g * 136..(g + 1) * 136];
            let sc = f16f(u16::from_le_bytes([gp[hoff], gp[hoff + 1]]));
            let zp = f16f(u16::from_le_bytes([gp[hoff + 2], gp[hoff + 3]]));
            let pk = u32::from_le_bytes(gp[8 + tid * 4..12 + tid * 4].try_into().unwrap());
            let xb = g * 256 + tid * 8;
            acc[g & 3] += group_dot8(sc, zp, pk, &x[xb..xb + 8]);
        }
        *lane = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    lane_tree(lanes)
}

/// MQ4G128V2 down row: `row.len() / 68` groups of `[sc zp : f16][64 nibble bytes]`; lane `tid`
/// owns bytes `2*tid, 2*tid+1` (elements `tid*4..+4`), one accumulation chain.
fn mq4_row_down(row: &[u8], x: &[f32]) -> f32 {
    let groups = row.len() / 68;
    debug_assert_eq!(x.len(), groups * 128);
    let mut lanes = [0.0f32; 32];
    for (tid, lane) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..groups {
            let gp = &row[g * 68..(g + 1) * 68];
            let sc = f16f(u16::from_le_bytes([gp[0], gp[1]]));
            let zp = f16f(u16::from_le_bytes([gp[2], gp[3]]));
            let (b0, b1) = (gp[4 + tid * 2], gp[5 + tid * 2]);
            let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
            let d = |i: usize| sc.mul_add(q[i] as f32, zp);
            let xb = g * 128 + tid * 4;
            let mut s = d(1) * x[xb + 1];
            s = d(0).mul_add(x[xb], s);
            s = d(2).mul_add(x[xb + 2], s);
            s = d(3).mul_add(x[xb + 3], s);
            acc += s;
        }
        *lane = acc;
    }
    lane_tree(lanes)
}

/// Bucketed MQ4 decode with the `decode_moe_slots` contract (slot `s` reads X row `s/div`, writes
/// `Y[s*ys + r]`; gate rows `>= m/2` go to Y1).
#[allow(clippy::too_many_arguments)]
fn mq4_decode_moe_slots(
    kind: Kind,
    experts: &[Option<&[u8]>],
    b: &Bucket,
    x: &[f32],
    xrs: usize,
    ys: usize,
    y0: &mut [f32],
    mut y1: Option<&mut [f32]>,
) {
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let row_bytes = match kind {
        Kind::GateUp => (k / 256) * 136,
        Kind::Down => (k / 128) * 68,
    };
    for (u, &id) in b.ids.iter().enumerate() {
        let bytes = experts[id as usize].expect("MQ4 expert present");
        for &slot in &b.slots[b.offs[u] as usize..b.offs[u + 1] as usize] {
            let slot = slot as usize;
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..k];
            let base = slot * ys;
            for r in 0..m {
                let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
                let v = match kind {
                    Kind::GateUp => mq4_row_gate(row, xs),
                    Kind::Down => mq4_row_down(row, xs),
                };
                if r < out {
                    y0[base + r] = v;
                } else {
                    y1.as_deref_mut().expect("gate has Y1")[base + r - out] = v;
                }
            }
        }
    }
}

/// Encode an MQ4 projection tensor (legacy 136 B / 68 B group layout, symmetric headers).
/// `codes` is one nibble per weight (`m*k`), `scales` one f16 per 128 columns (`m*k/128`).
fn encode_mq4(kind: Kind, m: usize, k: usize, codes: &[u8], scales: &[u16]) -> Vec<u8> {
    assert_eq!(codes.len(), m * k);
    assert_eq!(scales.len(), m * (k / 128));
    let pack = |dst: &mut [u8], base: usize| {
        for (j, byte) in dst.iter_mut().enumerate() {
            *byte = (codes[base + 2 * j] & 15) | ((codes[base + 2 * j + 1] & 15) << 4);
        }
    };
    match kind {
        Kind::GateUp => {
            let groups = k / 256;
            let mut out = vec![0u8; m * groups * 136];
            for r in 0..m {
                for g in 0..groups {
                    let off = (r * groups + g) * 136;
                    for half in 0..2 {
                        let sc = scales[r * (k / 128) + 2 * g + half];
                        out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
                        out[off + 4 * half + 2..off + 4 * half + 4]
                            .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
                    }
                    pack(&mut out[off + 8..off + 136], r * k + g * 256);
                }
            }
            out
        }
        Kind::Down => {
            let groups = k / 128;
            let mut out = vec![0u8; m * groups * 68];
            for r in 0..m {
                for g in 0..groups {
                    let off = (r * groups + g) * 68;
                    let sc = scales[r * groups + g];
                    out[off..off + 2].copy_from_slice(&sc.to_le_bytes());
                    out[off + 2..off + 4].copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
                    pack(&mut out[off + 4..off + 68], r * k + g * 128);
                }
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------------------
// Synthetic expert + activation fixtures
// ---------------------------------------------------------------------------------------

struct Expert {
    codec: Codec,
    gate: Vec<u8>,
    down: Vec<u8>,
    gate_sha: [u8; 32],
    down_sha: [u8; 32],
}

/// Scale bits for a row/group class. W3: f16 max 0x7bff is legal; MQ4 (symmetric zero `-8*sc`)
/// tops out at 0x6fff. Class 3 is the zero scale (codes forced to the zero-scale contract).
fn class_scale(class: usize, codec: Codec, rng: &mut Rng) -> u16 {
    let max = match codec {
        Codec::W3 => 0x7bff,
        Codec::Mq4 => 0x6fff,
    };
    match class {
        1 => max,
        2 => 1 + rng.below(0x3ff) as u16, // f16 subnormal
        3 => 0,
        4 => 0x0400, // smallest normal
        5 => 0x0001, // smallest subnormal
        6 => 0x3c00,
        _ => (((8 + rng.below(13)) as u16) << 10) | (rng.u32() as u16 & 0x3ff),
    }
}

/// Deterministic expert tensor for `(id, codec, kind)`. Every non-plain expert rotates the code
/// so each code value lands on every in-group position across rows, and spreads zero, f16-max,
/// subnormal and smallest-normal scales over (row, group).
fn gen_tensor(codec: Codec, kind: Kind, id: u32, special: bool) -> Vec<u8> {
    let (m, k) = (kind.m(), kind.k());
    let groups = k / 128;
    let levels: usize = match codec {
        Codec::W3 => 7,
        Codec::Mq4 => 15,
    };
    let zero_code: u8 = match codec {
        Codec::W3 => 4,
        Codec::Mq4 => 0,
    };
    let mut rng = Rng::new(mix64(id as u64 * 8 + kind as u64 * 2 + codec as u64 + 1));
    let salt = id as usize;
    let mut codes = vec![0u8; m * k];
    let mut scales = vec![0u16; m * groups];
    for r in 0..m {
        for g in 0..groups {
            let class = if special { (r * 5 + g * 3 + salt) % 10 } else { 0 };
            let sc = class_scale(class, codec, &mut rng);
            scales[r * groups + g] = sc;
            let base = r * k + g * 128;
            let rotate = special && (r + g) % 2 == 0;
            for j in 0..128 {
                codes[base + j] = if sc == 0 {
                    zero_code
                } else if rotate {
                    ((j + r + g + salt) % (levels + 1)) as u8
                } else {
                    ((rng.u32() as usize) & levels) as u8
                };
            }
        }
    }
    match codec {
        Codec::W3 => mq_i4::encode_tensor(kind.w3_profile(), m, k, &codes, &scales)
            .expect("synthetic W3 tensor is valid"),
        Codec::Mq4 => encode_mq4(kind, m, k, &codes, &scales),
    }
}

fn gen_expert(id: u32, codec: Codec) -> Expert {
    let special = id % 3 != 2;
    let gate = gen_tensor(codec, Kind::GateUp, id, special);
    let down = gen_tensor(codec, Kind::Down, id, special);
    Expert { codec, gate_sha: sha(&gate), down_sha: sha(&down), gate, down }
}

fn gen_x_row(style: usize, n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = match style % 4 {
            0 => {
                let (s, e, m) = (rng.u32() >> 31, 124 + rng.below(5) as u32, rng.u32() & 0x7f_ffff);
                f32::from_bits((s << 31) | (e << 23) | m)
            }
            1 => {
                let (s, e, m) = (rng.u32() >> 31, 107 + rng.below(24) as u32, rng.u32() & 0x7f_ffff);
                f32::from_bits((s << 31) | (e << 23) | m)
            }
            2 => [16777216.0f32, 1.0, -16777216.0, 1.0][i % 4],
            _ => [0.0f32, -0.0, 1.0e-30, -1.0e-30, 1.0, -1.0, 0.5, -0.5][i % 8],
        };
        out.push(v);
    }
    out
}

// ---------------------------------------------------------------------------------------
// Cases and buckets
// ---------------------------------------------------------------------------------------

struct Case {
    name: String,
    origin: &'static str,
    c: usize,
    /// `route[lane*10 + rank]` = expert id.
    route: Vec<u32>,
    experts: BTreeMap<u32, Rc<Expert>>,
    x_gate: Vec<f32>,
    x_down: Vec<f32>,
    xrs_gate: usize,
    xrs_down: usize,
    ys_gate: usize,
    ys_down: usize,
    old_parity: bool,
}

impl Case {
    fn x(&self, kind: Kind) -> (&[f32], usize, usize) {
        match kind {
            Kind::GateUp => (&self.x_gate, self.xrs_gate, self.ys_gate),
            Kind::Down => (&self.x_down, self.xrs_down, self.ys_down),
        }
    }
    fn tight(&self) -> bool {
        self.xrs_gate == GATE_K
            && self.xrs_down == DOWN_K
            && self.ys_gate == GATE_M / 2
            && self.ys_down == DOWN_M
    }
    fn codec_of_slot(&self, slot: usize) -> Codec {
        self.experts[&self.route[slot]].codec
    }
    fn input_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.name.as_bytes());
        h.update((self.c as u64).to_le_bytes());
        h.update(u32_bytes(&self.route));
        for (id, e) in &self.experts {
            h.update(id.to_le_bytes());
            h.update(e.codec.tag().as_bytes());
            h.update(e.gate_sha);
            h.update(e.down_sha);
        }
        for v in [self.xrs_gate, self.xrs_down, self.ys_gate, self.ys_down] {
            h.update((v as u64).to_le_bytes());
        }
        h.update(f32_bytes(&self.x_gate));
        h.update(f32_bytes(&self.x_down));
        hex(&h.finalize())
    }
}

/// Activations: per-lane (gate) / per-slot (down) style rotation; padded strides are poisoned
/// with NaN so any read outside a row's `K` words is visible.
#[allow(clippy::too_many_arguments)]
fn build_case(
    name: &str,
    origin: &'static str,
    c: usize,
    route: Vec<u32>,
    experts: BTreeMap<u32, Rc<Expert>>,
    padded: bool,
    old_parity: bool,
    xshift: usize,
) -> Case {
    assert_eq!(route.len(), 10 * c);
    let (xrs_gate, ys_gate, xrs_down, ys_down) = if padded {
        (GATE_K + 64, GATE_M / 2 + 32, DOWN_K + 64, DOWN_M + 32)
    } else {
        (GATE_K, GATE_M / 2, DOWN_K, DOWN_M)
    };
    let mut rng = Rng::new(name_seed(name));
    let fill = |rows: usize, k: usize, xrs: usize, rng: &mut Rng, shift: usize| {
        let mut x = vec![f32::from_bits(PAD_BITS); rows * xrs];
        for r in 0..rows {
            let row = gen_x_row(r + shift, k, rng);
            x[r * xrs..r * xrs + k].copy_from_slice(&row);
        }
        x
    };
    let x_gate = fill(c, GATE_K, xrs_gate, &mut rng, xshift);
    let x_down = fill(10 * c, DOWN_K, xrs_down, &mut rng, xshift + 1);
    Case {
        name: name.to_string(),
        origin,
        c,
        route,
        experts,
        x_gate,
        x_down,
        xrs_gate,
        xrs_down,
        ys_gate,
        ys_down,
        old_parity,
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}

/// Per-codec descriptor: ascending expert ids, slots in stable original order per expert,
/// offsets spanning only this bucket's slots.
fn build_bucket(route: &[u32], codec_of: &dyn Fn(u32) -> Codec, codec: Codec) -> Bucket {
    let ids: BTreeSet<u32> = route.iter().copied().filter(|&e| codec_of(e) == codec).collect();
    let mut b = Bucket::default();
    for id in ids {
        b.ids.push(id);
        b.offs.push(b.slots.len() as u32);
        for (s, &e) in route.iter().enumerate() {
            if e == id {
                b.slots.push(s as u32);
            }
        }
    }
    b.offs.push(b.slots.len() as u32);
    b
}

fn buckets_of(case: &Case) -> [Bucket; 2] {
    let f = |e: u32| case.experts[&e].codec;
    [build_bucket(&case.route, &f, Codec::W3), build_bucket(&case.route, &f, Codec::Mq4)]
}

fn pick_ids(rng: &mut Rng, n: usize, edges: bool) -> Vec<u32> {
    let mut set = BTreeSet::new();
    if edges {
        set.insert(0u32);
        set.insert(511u32);
    }
    while set.len() < n {
        set.insert(rng.below(N_EXPERTS) as u32);
    }
    set.into_iter().collect()
}

fn experts_for(ids: &[u32], codec_of: impl Fn(usize, u32) -> Codec) -> BTreeMap<u32, Rc<Expert>> {
    ids.iter().enumerate().map(|(i, &id)| (id, Rc::new(gen_expert(id, codec_of(i, id))))).collect()
}

const MIXED10: [Codec; 10] = [
    Codec::W3,
    Codec::Mq4,
    Codec::Mq4,
    Codec::W3,
    Codec::W3,
    Codec::Mq4,
    Codec::W3,
    Codec::Mq4,
    Codec::Mq4,
    Codec::W3,
];

fn lane_routes(rng: &mut Rng, pool: &[u32], c: usize) -> Vec<u32> {
    let mut route = Vec::with_capacity(10 * c);
    for _ in 0..c {
        let mut p = pool.to_vec();
        rng.shuffle(&mut p);
        route.extend_from_slice(&p[..10]);
    }
    route
}

/// Random codec per id with `w3_tenths`/10 W3 probability; both codecs forced present when mixed.
fn mixed_codecs(rng: &mut Rng, n: usize, w3_tenths: usize) -> Vec<Codec> {
    let mut codecs: Vec<Codec> =
        (0..n).map(|_| if rng.below(10) < w3_tenths { Codec::W3 } else { Codec::Mq4 }).collect();
    if w3_tenths > 0 && w3_tenths < 10 {
        codecs[0] = Codec::W3;
        codecs[n - 1] = Codec::Mq4;
    }
    codecs
}

const SYNTH_CASES: &[&str] = &[
    "c1_mixed_interleaved",
    "c1_mixed_stress_x",
    "c1_all_w3",
    "c1_all_mq4",
    "c1_mixed_1w3_9mq4",
    "c1_mixed_9w3_1mq4",
    "c3_mixed_partial_lanes",
    "c8_union_repeated_mixed",
    "c8_unique_80_mixed",
    "c8_identical_lanes_mixed",
    "c8_union_all_w3",
    "c8_union_all_mq4",
    "c8_padded_strides_mixed",
];

fn build_synthetic(name: &str) -> Case {
    let mut rng = Rng::new(name_seed(name) ^ 0x5eed);
    match name {
        "c1_mixed_interleaved" | "c1_mixed_stress_x" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i, _| MIXED10[i]);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            let shift = if name == "c1_mixed_stress_x" { 1 } else { 0 };
            build_case(name, "synthetic", 1, route, experts, false, false, shift)
        }
        "c1_all_w3" | "c1_all_mq4" => {
            let codec = if name == "c1_all_w3" { Codec::W3 } else { Codec::Mq4 };
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |_, _| codec);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, "synthetic", 1, route, experts, false, codec == Codec::Mq4, 0)
        }
        "c1_mixed_1w3_9mq4" | "c1_mixed_9w3_1mq4" => {
            let ids = pick_ids(&mut rng, 10, true);
            let single_w3 = name == "c1_mixed_1w3_9mq4";
            // Never an edge id, so the edge experts keep the majority codec.
            let lone = 1 + rng.below(8);
            let experts = experts_for(&ids, |i, _| {
                if (i == lone) == single_w3 {
                    Codec::W3
                } else {
                    Codec::Mq4
                }
            });
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, "synthetic", 1, route, experts, false, false, 2)
        }
        "c3_mixed_partial_lanes" => {
            let ids = pick_ids(&mut rng, 12, false);
            let codecs = mixed_codecs(&mut rng, ids.len(), 5);
            let experts = experts_for(&ids, |i, _| codecs[i]);
            let route = lane_routes(&mut rng, &ids, 3);
            build_case(name, "synthetic", 3, route, experts, false, false, 0)
        }
        "c8_union_repeated_mixed" | "c8_union_all_w3" | "c8_union_all_mq4" | "c8_padded_strides_mixed" => {
            let ids = pick_ids(&mut rng, 14, true);
            let codecs: Vec<Codec> = match name {
                "c8_union_all_w3" => vec![Codec::W3; ids.len()],
                "c8_union_all_mq4" => vec![Codec::Mq4; ids.len()],
                _ => mixed_codecs(&mut rng, ids.len(), 5),
            };
            let experts = experts_for(&ids, |i, _| codecs[i]);
            let route = lane_routes(&mut rng, &ids, 8);
            let padded = name == "c8_padded_strides_mixed";
            build_case(name, "synthetic", 8, route, experts, padded, name == "c8_union_all_mq4", 0)
        }
        "c8_unique_80_mixed" => {
            let ids = pick_ids(&mut rng, 80, true);
            let codecs = mixed_codecs(&mut rng, ids.len(), 5);
            let experts = experts_for(&ids, |i, _| codecs[i]);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            build_case(name, "synthetic", 8, route, experts, false, false, 3)
        }
        "c8_identical_lanes_mixed" => {
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |i, _| MIXED10[(i + 3) % 10]);
            let mut lane = ids.clone();
            rng.shuffle(&mut lane);
            let route: Vec<u32> = (0..8).flat_map(|_| lane.clone()).collect();
            build_case(name, "synthetic", 8, route, experts, false, false, 0)
        }
        other => panic!("unknown synthetic case {other}"),
    }
}

// ---------------------------------------------------------------------------------------
// Real artifact (one HFQM layer, selected expert records only)
// ---------------------------------------------------------------------------------------

fn expert_entry(layer: usize, expert: usize, proj: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}")
}

struct Real {
    experts: BTreeMap<u32, Rc<Expert>>,
    meta: Value,
    mq4_sym_violations: u64,
}

fn mq4_sym_violations(kind: Kind, bytes: &[u8]) -> u64 {
    let mut bad = 0u64;
    let (stride, hdrs): (usize, &[usize]) = match kind {
        Kind::GateUp => (136, &[0usize, 4][..]),
        Kind::Down => (68, &[0usize][..]),
    };
    for g in bytes.chunks_exact(stride) {
        for &h in hdrs {
            let sc = u16::from_le_bytes([g[h], g[h + 1]]);
            let zp = u16::from_le_bytes([g[h + 2], g[h + 3]]);
            let finite = sc & 0x7c00 != 0x7c00 && sc & 0x8000 == 0;
            // -8*sc must itself be a finite f16 and equal the stored zero (+-0 compare equal).
            if !finite || f16f(zp) != -8.0 * f16f(sc) || (-8.0 * f16f(sc)).abs() > 65504.0 {
                bad += 1;
            }
        }
    }
    bad
}

fn load_real(path: &str, layer: usize, want: usize) -> Result<Real> {
    let hfq = HfqFile::open_at_offset(std::path::Path::new(path), 0).map_err(err)?;
    let mut avail: BTreeMap<Codec, Vec<u32>> = BTreeMap::new();
    let (mut w2, mut other) = (0usize, 0usize);
    for e in 0..N_EXPERTS {
        let (gn, dn) = (expert_entry(layer, e, "gate_up_proj"), expert_entry(layer, e, "down_proj"));
        let (Some(g), Some(d)) = (hfq.find_tensor_info(&gn), hfq.find_tensor_info(&dn)) else {
            continue;
        };
        let codec = match (g.quant_type, d.quant_type) {
            (131, 132) => Codec::W3,
            (44, 53) => Codec::Mq4,
            (133, 134) => {
                w2 += 1;
                continue;
            }
            _ => {
                other += 1;
                continue;
            }
        };
        let shape_ok = g.shape == [GATE_M as u32, GATE_K as u32] && d.shape == [DOWN_M as u32, DOWN_K as u32];
        let size_ok =
            g.data_size == Kind::GateUp.tensor_bytes(codec) && d.data_size == Kind::Down.tensor_bytes(codec);
        if !shape_ok || !size_ok {
            return Err(format!(
                "layer {layer} expert {e}: unexpected shape/size for {} (gate {:?}/{} down {:?}/{})",
                codec.label(),
                g.shape,
                g.data_size,
                d.shape,
                d.data_size
            ));
        }
        avail.entry(codec).or_default().push(e as u32);
    }
    let counts = |c: Codec| avail.get(&c).map_or(0, |v| v.len());
    if counts(Codec::W3) == 0 || counts(Codec::Mq4) == 0 {
        return Err(format!(
            "layer {layer} of {path} has no mixed W3/MQ4 experts (w3={}, mq4={}, w2={w2}, other={other})",
            counts(Codec::W3),
            counts(Codec::Mq4)
        ));
    }
    let want = want.max(10);
    let half = want / 2;
    let (a, b) = (avail[&Codec::W3].clone(), avail[&Codec::Mq4].clone());
    let mut take_w3 = half.min(a.len());
    let mut take_mq4 = (want - half).min(b.len());
    if take_w3 + take_mq4 < want {
        // One side is short: top up from the other.
        take_w3 = (want - take_mq4).min(a.len());
        take_mq4 = (want - take_w3).min(b.len());
    }
    let spread = |v: &Vec<u32>, n: usize| -> Vec<u32> { (0..n).map(|i| v[i * v.len() / n]).collect() };
    let (sel_w3, sel_mq4) = (spread(&a, take_w3), spread(&b, take_mq4));
    if sel_w3.len() + sel_mq4.len() < 10 {
        return Err(format!(
            "layer {layer}: only {} usable experts; a top-10 lane needs 10",
            sel_w3.len() + sel_mq4.len()
        ));
    }
    let mut experts = BTreeMap::new();
    let mut lines = String::new();
    let mut recs = Vec::new();
    let mut violations = 0u64;
    for (codec, ids) in [(Codec::W3, &sel_w3), (Codec::Mq4, &sel_mq4)] {
        for &id in ids {
            let gn = expert_entry(layer, id as usize, "gate_up_proj");
            let dn = expert_entry(layer, id as usize, "down_proj");
            let (gi, gate) = hfq.tensor_data_vec(&gn).ok_or_else(|| format!("short read of {gn}"))?;
            let (gq, goff, gsz) = (gi.quant_type, gi.data_offset, gi.data_size);
            let (di, down) = hfq.tensor_data_vec(&dn).ok_or_else(|| format!("short read of {dn}"))?;
            let (dq, doff, dsz) = (di.quant_type, di.data_offset, di.data_size);
            match codec {
                Codec::W3 => {
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R256, GATE_M, GATE_K, &gate)
                        .map_err(|e| format!("{gn}: {e}"))?;
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R128, DOWN_M, DOWN_K, &down)
                        .map_err(|e| format!("{dn}: {e}"))?;
                }
                Codec::Mq4 => {
                    violations += mq4_sym_violations(Kind::GateUp, &gate) + mq4_sym_violations(Kind::Down, &down)
                }
            }
            let (gs, ds) = (sha(&gate), sha(&down));
            lines.push_str(&format!(
                "{gn} {goff} {gsz} {gq} {}\n{dn} {doff} {dsz} {dq} {}\n",
                hex(&gs),
                hex(&ds)
            ));
            recs.push(json!({
                "expert": id, "codec": codec.label(), "gate_quant_type": gq, "down_quant_type": dq,
                "gate_offset": goff, "gate_bytes": gsz, "gate_sha256": hex(&gs),
                "down_offset": doff, "down_bytes": dsz, "down_sha256": hex(&ds),
                "contiguous_gate_down": goff + gsz == doff,
            }));
            experts.insert(id, Rc::new(Expert { codec, gate_sha: gs, down_sha: ds, gate, down }));
        }
    }
    let meta = json!({
        "path": path, "layer": layer, "file_len": hfq.file_len(), "arch_id": hfq.arch_id,
        "index_sha256": hex(&hfq.index_sha256().map_err(err)?),
        "mq4v2_symmetric_metadata": hfq.mq4v2_symmetric(),
        "layer_expert_formats": {"w3": counts(Codec::W3), "mq4": counts(Codec::Mq4), "w2": w2, "other": other},
        "selected": {"w3": sel_w3.len(), "mq4": sel_mq4.len()},
        "mq4_sym_header_violations": violations,
        "selected_input_sha256": hex(&sha(lines.as_bytes())),
        "experts": recs,
    });
    Ok(Real { experts, meta, mq4_sym_violations: violations })
}

fn real_cases(real: &Real, origin: &'static str, prefix: &str) -> Vec<Case> {
    let all: Vec<u32> = real.experts.keys().copied().collect();
    let by = |c: Codec| -> Vec<u32> { all.iter().copied().filter(|id| real.experts[id].codec == c).collect() };
    let (w3, mq4) = (by(Codec::W3), by(Codec::Mq4));
    let mut rng = Rng::new(0x7ea1_1a7e);
    // C1: 10 experts, as balanced as the pool allows, route order shuffled.
    let mut ids: Vec<u32> = Vec::new();
    let (mut i3, mut i4) = (0usize, 0usize);
    while ids.len() < 10 {
        if (ids.len() % 2 == 0 && i3 < w3.len()) || i4 >= mq4.len() {
            ids.push(w3[i3]);
            i3 += 1;
        } else {
            ids.push(mq4[i4]);
            i4 += 1;
        }
    }
    let mut route1 = ids.clone();
    rng.shuffle(&mut route1);
    let sub: BTreeMap<u32, Rc<Expert>> = ids.iter().map(|id| (*id, real.experts[id].clone())).collect();
    let c1 = build_case(&format!("{prefix}_c1_mixed"), origin, 1, route1, sub, false, false, 0);
    let route8 = lane_routes(&mut rng, &all, 8);
    let c8 = build_case(&format!("{prefix}_c8_union_mixed"), origin, 8, route8, real.experts.clone(), false, false, 1);
    vec![c1, c8]
}

// ---------------------------------------------------------------------------------------
// Certified R1 records + XTS aggregate (real weights, no packed HFQM)
// ---------------------------------------------------------------------------------------

/// Origin label of the certified-records cases; deliberately not "packed HFQM".
const RECORDS_ORIGIN: &str = "real_records_layer";

/// sha256 and length of a whole file, streamed.
fn file_sha256(path: &str) -> Result<(String, u64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let (mut h, mut n) = (Sha256::new(), 0u64);
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let k = f.read(&mut buf).map_err(|e| format!("{path}: {e}"))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex(&h.finalize()), n))
}

/// `layers["<layer>"]` of a `ctl3-hot.map.json`: `"w3"` / `"xts_mq4"` (/`"w2"`) per expert.
fn hot_map_formats(root: &Value, layer: usize) -> Result<Vec<ExpertFormat>> {
    let row = root
        .get("layers")
        .and_then(|l| l.get(layer.to_string()))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("map has no layers[\"{layer}\"] array"))?;
    if row.len() != N_EXPERTS {
        return Err(format!("map layers[\"{layer}\"] has {} experts, expected {N_EXPERTS}", row.len()));
    }
    row.iter()
        .enumerate()
        .map(|(e, v)| match v.as_str() {
            Some("w3") => Ok(ExpertFormat::Mq3I4),
            Some("w2") => Ok(ExpertFormat::Mq2I4),
            Some("xts_mq4") => Ok(ExpertFormat::Mq4Xts),
            other => Err(format!("map layers[\"{layer}\"][{e}] = {other:?} is not w3/w2/xts_mq4")),
        })
        .collect()
}

/// How many W3 (cold) and MQ4-XTS (hot) experts to take for `want` total, half each, topping a short
/// side up from the other.
fn select_counts(w3_len: usize, mq4_len: usize, want: usize) -> (usize, usize) {
    let want = want.max(10);
    let half = want / 2;
    let mut take_w3 = half.min(w3_len);
    let mut take_mq4 = (want - half).min(mq4_len);
    if take_w3 + take_mq4 < want {
        take_w3 = (want - take_mq4).min(w3_len);
        take_mq4 = (want - take_w3).min(mq4_len);
    }
    (take_w3, take_mq4)
}

fn spread(v: &[u32], n: usize) -> Vec<u32> {
    (0..n).map(|i| v[i * v.len() / n]).collect()
}

/// Strict real weights: cold experts are the certified R1 NPZ records imported through
/// `I4ExpertFinalCodeRecord::import_npz_file` (S2 encode + round trip), hot experts are the
/// per-expert gate/down slices of the XTS aggregate tensors, read through `HfqFile` + the source entry span.
fn load_records(cfg: &Cfg) -> Result<Real> {
    let layer = cfg.layer;
    let (dir, manifest_path, alloc_path, xts_path) = match (&cfg.records_dir, &cfg.records_manifest, &cfg.alloc, &cfg.xts) {
        (Some(d), Some(m), Some(a), Some(x)) => (d.as_str(), m.as_str(), a.as_str(), x.as_str()),
        _ => return Err("certified-records mode needs --records-dir, --records-manifest, --alloc and --xts".into()),
    };
    // Manifest identity.
    let manifest_bytes = std::fs::read(manifest_path).map_err(|e| format!("{manifest_path}: {e}"))?;
    let manifest_sha = hex(&sha(&manifest_bytes));
    if let Some(want) = &cfg.records_manifest_sha256 {
        if *want != manifest_sha {
            return Err(format!("manifest sha256 {manifest_sha} != expected {want}"));
        }
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(|e| format!("{manifest_path}: {e}"))?;
    let want_expert_bytes = Kind::GateUp.tensor_bytes(Codec::W3) + Kind::Down.tensor_bytes(Codec::W3);
    let header_ok = manifest.get("layer").and_then(Value::as_u64) == Some(layer as u64)
        && manifest.get("bits").and_then(Value::as_u64) == Some(3)
        && manifest.get("grid").and_then(Value::as_str) == Some("uniform")
        && manifest.get("expert_bytes").and_then(Value::as_u64) == Some(want_expert_bytes as u64);
    if !header_ok {
        return Err(format!("{manifest_path}: header is not layer {layer} / bits 3 / grid uniform / expert_bytes {want_expert_bytes}"));
    }
    // Allocation (+ optional hot map cross-check).
    let alloc = load_allocation(Path::new(alloc_path), &cfg.alloc_point, SlabGeometry::FN).map_err(err)?;
    let formats: Vec<ExpertFormat> = alloc.formats[layer * N_EXPERTS..(layer + 1) * N_EXPERTS].to_vec();
    let mut map_meta = Value::Null;
    if let Some(map_path) = &cfg.map {
        let bytes = std::fs::read(map_path).map_err(|e| format!("{map_path}: {e}"))?;
        let root: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{map_path}: {e}"))?;
        let from_map = hot_map_formats(&root, layer)?;
        if let Some(e) = (0..N_EXPERTS).find(|&e| from_map[e] != formats[e]) {
            return Err(format!("{map_path} disagrees with allocation point {} at layer {layer} expert {e}: {:?} vs {:?}", alloc.point, from_map[e], formats[e]));
        }
        map_meta = json!({"path": map_path, "sha256": hex(&sha(&bytes)), "arm": root.get("arm"), "cross_checked_against_alloc": true});
    }
    let (mut cold, mut hot, mut w2) = (Vec::new(), Vec::new(), 0usize);
    for (e, f) in formats.iter().enumerate() {
        match f {
            ExpertFormat::Mq3I4 => cold.push(e as u32),
            ExpertFormat::Mq4Xts => hot.push(e as u32),
            ExpertFormat::Mq2I4 => w2 += 1,
        }
    }
    if cold.is_empty() || hot.is_empty() {
        return Err(format!("layer {layer} allocation has no mixed W3/MQ4 experts (w3={}, mq4={}, w2={w2})", cold.len(), hot.len()));
    }
    let (take_w3, take_mq4) = select_counts(cold.len(), hot.len(), cfg.real_experts);
    let (sel_w3, sel_mq4) = (spread(&cold, take_w3), spread(&hot, take_mq4));
    if sel_w3.len() + sel_mq4.len() < 10 {
        return Err(format!("layer {layer}: only {} usable experts; a top-10 lane needs 10", sel_w3.len() + sel_mq4.len()));
    }
    // XTS source: identity, aggregate layout, then only the selected spans.
    let (xts_sha, xts_len) = file_sha256(xts_path)?;
    if let Some(want) = &cfg.xts_sha256 {
        if *want != xts_sha {
            return Err(format!("{xts_path}: sha256 {xts_sha} != expected {want}"));
        }
    }
    let hfq = HfqFile::open_at_offset(Path::new(xts_path), 0).map_err(err)?;
    let mut aggregates = Vec::new();
    let mut agg_span: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for (kind, proj, qt, group) in [(Kind::GateUp, ExpertProj::GateUp, 44u8, 256u32), (Kind::Down, ExpertProj::Down, 53u8, 128u32)] {
        let name = SlabGeometry::aggregate_name(layer, proj);
        let info = hfq.find_tensor_info(&name).ok_or_else(|| format!("{xts_path}: missing aggregate `{name}`"))?;
        let per = kind.tensor_bytes(Codec::Mq4);
        let want_shape = vec![N_EXPERTS as u32, kind.m() as u32, kind.k() as u32];
        if info.quant_type != qt || info.shape != want_shape || info.group_size != group || info.data_size != N_EXPERTS * per {
            return Err(format!(
                "aggregate `{name}` is qt{} shape {:?} group {} {} B; expected qt{qt} shape {want_shape:?} group {group} {} B",
                info.quant_type, info.shape, info.group_size, info.data_size, N_EXPERTS * per
            ));
        }
        aggregates.push(json!({"name": name, "quant_type": qt, "shape": want_shape, "group_size": group,
            "data_offset": info.data_offset, "data_size": info.data_size, "per_expert_bytes": per}));
        agg_span.insert(kind.suffix(), (info.data_offset, per));
    }
    let xts_slice = |kind: Kind, e: u32| -> Result<(Vec<u8>, usize)> {
        let (base, per) = agg_span[kind.suffix()];
        let off = base + e as usize * per;
        let s = hfq.data_range(off, per).ok_or_else(|| format!("{xts_path}: cannot map {} bytes at {off} (mmap dropped or range past EOF)", per))?;
        Ok((s.to_vec(), off))
    };
    let shapes = I4RecordShapes { gate_up: [GATE_M, GATE_K], down: [DOWN_M, DOWN_K] };
    let manifest_entry = |e: u32| manifest.get("experts").and_then(|x| x.get(format!("E{e}")));
    let mut experts = BTreeMap::new();
    let mut lines = String::new();
    let mut recs = Vec::new();
    let mut violations = 0u64;
    for (codec, ids) in [(Codec::W3, &sel_w3), (Codec::Mq4, &sel_mq4)] {
        for &id in ids {
            let (gate, down, rec_meta) = match codec {
                Codec::W3 => {
                    let npz = Path::new(dir).join(format!("E{id}.npz"));
                    let rec = I4ExpertFinalCodeRecord::import_npz_file(&npz, layer, id as usize, ExpertFormat::Mq3I4, shapes)
                        .map_err(|e| format!("{}: {e}", npz.display()))?;
                    let entry = manifest_entry(id).ok_or_else(|| format!("{manifest_path}: no experts.E{id}"))?;
                    if entry.get("npz_sha256").and_then(Value::as_str) != Some(rec.source_npz_sha256.as_str()) {
                        return Err(format!("{}: npz sha256 {} != manifest {:?}", npz.display(), rec.source_npz_sha256, entry.get("npz_sha256")));
                    }
                    let mut pack = rec.gate_up.bytes.clone();
                    pack.extend_from_slice(&rec.down.bytes);
                    let pack_sha = hex(&sha(&pack));
                    if entry.get("sha256").and_then(Value::as_str) != Some(pack_sha.as_str()) {
                        return Err(format!("{}: encoded gate++down sha256 {pack_sha} != manifest {:?}", npz.display(), entry.get("sha256")));
                    }
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R256, GATE_M, GATE_K, &rec.gate_up.bytes).map_err(err)?;
                    mq_i4::decode_tensor(MqI4Profile::MQ3_R128, DOWN_M, DOWN_K, &rec.down.bytes).map_err(err)?;
                    let meta = json!({"source": "r1_npz_record", "npz_path": npz.display().to_string(),
                        "npz_sha256": rec.source_npz_sha256, "manifest_npz_sha256_match": true,
                        "manifest_pack_sha256": pack_sha, "manifest_pack_sha256_match": true,
                        "scale_rule": rec.scale_rule, "grid": rec.grid});
                    (rec.gate_up.bytes, rec.down.bytes, meta)
                }
                Codec::Mq4 => {
                    let (gate, goff) = xts_slice(Kind::GateUp, id)?;
                    let (down, doff) = xts_slice(Kind::Down, id)?;
                    violations += mq4_sym_violations(Kind::GateUp, &gate) + mq4_sym_violations(Kind::Down, &down);
                    let meta = json!({"source": "xts_aggregate_slice", "gate_source_offset": goff, "down_source_offset": doff});
                    (gate, down, meta)
                }
            };
            let (gs, ds) = (sha(&gate), sha(&down));
            lines.push_str(&format!("{} {id} {} {}\n", codec.label(), hex(&gs), hex(&ds)));
            let mut rec_json = json!({"expert": id, "codec": codec.label(), "gate_bytes": gate.len(), "gate_sha256": hex(&gs),
                "down_bytes": down.len(), "down_sha256": hex(&ds)});
            rec_json["provenance"] = rec_meta;
            recs.push(rec_json);
            experts.insert(id, Rc::new(Expert { codec, gate_sha: gs, down_sha: ds, gate, down }));
        }
    }
    let meta = json!({
        "origin": RECORDS_ORIGIN, "packed_hfqm": false, "layer": layer,
        "records_dir": dir,
        "manifest": {"path": manifest_path, "sha256": manifest_sha, "expected_sha256": cfg.records_manifest_sha256,
            "script_sha256": manifest.get("script_sha256"), "input_receipts": manifest.get("input_receipts"),
            "expert_bytes": want_expert_bytes},
        "alloc": {"path": alloc_path, "sha256": alloc.file_sha256, "point": alloc.point,
            "layer_formats": {"w3": cold.len(), "mq4_xts": hot.len(), "w2": w2}},
        "map": map_meta,
        "xts": {"path": xts_path, "sha256": xts_sha, "expected_sha256": cfg.xts_sha256, "bytes": xts_len,
            "file_len": hfq.file_len(), "index_sha256": hex(&hfq.index_sha256().map_err(err)?),
            "aggregates": aggregates},
        "selected": {"w3_cold": sel_w3.len(), "mq4_hot": sel_mq4.len()},
        "mq4_sym_header_violations": violations,
        "selected_input_sha256": hex(&sha(lines.as_bytes())),
        "experts": recs,
    });
    Ok(Real { experts, meta, mq4_sym_violations: violations })
}

// ---------------------------------------------------------------------------------------
// Device plumbing
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Buf {
    idx: usize,
    ptr: u64,
}
struct Dev<'a> {
    hip: &'a HipRuntime,
    bufs: Vec<DeviceBuffer>,
}
impl<'a> Dev<'a> {
    fn new(hip: &'a HipRuntime) -> Self {
        Dev { hip, bufs: Vec::new() }
    }
    fn alloc(&mut self, n: usize) -> Result<Buf> {
        let b = self.hip.malloc(n.max(4)).map_err(err)?;
        let ptr = b.as_ptr() as u64;
        self.bufs.push(b);
        Ok(Buf { idx: self.bufs.len() - 1, ptr })
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<Buf> {
        let b = self.alloc(bytes.len())?;
        self.hip.memcpy_htod(&self.bufs[b.idx], bytes).map_err(err)?;
        Ok(b)
    }
    fn fill(&mut self, n: usize, value: i32) -> Result<Buf> {
        let b = self.alloc(n)?;
        self.hip.memset(&self.bufs[b.idx], value, n.max(4)).map_err(err)?;
        Ok(b)
    }
    fn download(&self, b: Buf, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        self.hip.memcpy_dtoh(&mut out, &self.bufs[b.idx]).map_err(err)?;
        Ok(out)
    }
    fn release(self) {
        let hip = self.hip;
        for b in self.bufs {
            let _ = hip.free(b);
        }
    }
}

/// One slab per expert (gate_up tensor then down tensor, as in the artifact), allocated in
/// shuffled order between random-size fillers so pointers are neither monotone nor uniformly
/// strided. Returns `id -> (gate_ptr, down_ptr)`.
fn upload_experts(dev: &mut Dev, case: &Case) -> Result<BTreeMap<u32, (u64, u64)>> {
    let mut rng = Rng::new(name_seed(&case.name) ^ 0xa110c);
    let mut order: Vec<u32> = case.experts.keys().copied().collect();
    rng.shuffle(&mut order);
    let mut out = BTreeMap::new();
    for id in order {
        if rng.below(3) != 0 {
            dev.alloc(256 * (1 + rng.below(300)))?;
        }
        let e = &case.experts[&id];
        let mut slab = Vec::with_capacity(e.gate.len() + e.down.len());
        slab.extend_from_slice(&e.gate);
        slab.extend_from_slice(&e.down);
        let b = dev.upload(&slab)?;
        out.insert(id, (b.ptr, b.ptr + e.gate.len() as u64));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn kernarg(
    ptrs: [u64; 7],
    m: usize,
    k: usize,
    u: usize,
    c: usize,
    xdiv: usize,
    xsrc: usize,
    xrs: usize,
    ys: usize,
) -> [u8; KERNARG_BYTES] {
    let mut a = [0u8; KERNARG_BYTES];
    for (i, p) in ptrs.iter().enumerate() {
        a[i * 8..i * 8 + 8].copy_from_slice(&p.to_le_bytes());
    }
    for (off, v) in [(56, m), (60, k), (64, u), (68, c), (72, xdiv), (76, xsrc), (80, xrs), (84, ys)] {
        a[off..off + 4].copy_from_slice(&(v as u32).to_le_bytes());
    }
    a
}

struct Ctx {
    hip: HipRuntime,
    funcs: Vec<(Codec, Kind, Function)>,
    codeobject_sha: String,
    device: i32,
    old_parity: bool,
    gpu: Option<Gpu>,
}
impl Ctx {
    fn func(&self, codec: Codec, kind: Kind) -> &Function {
        &self.funcs.iter().find(|(c, k, _)| *c == codec && *k == kind).expect("all four symbols loaded").2
    }
}

struct TimingCfg {
    reps: usize,
    warmup: usize,
    launches: usize,
}

// ---------------------------------------------------------------------------------------
// CPU expected buffers
// ---------------------------------------------------------------------------------------

struct Expected {
    y0: Vec<f32>,
    y1: Option<Vec<f32>>,
    twin_mismatches: u64,
    twin_slots_checked: usize,
}

fn proj_bytes(e: &Expert, kind: Kind) -> &[u8] {
    match kind {
        Kind::GateUp => &e.gate,
        Kind::Down => &e.down,
    }
}

fn cpu_expected(case: &Case, kind: Kind, buckets: &[Bucket; 2]) -> Expected {
    let (x, xrs, ys) = case.x(kind);
    let n = 10 * case.c * ys;
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let mut y0 = vec![sentinel; n];
    let mut y1 = if kind == Kind::GateUp { Some(vec![sentinel; n]) } else { None };
    let _ = n;
    let mut twin_mismatches = 0u64;
    let mut checked = 0usize;
    // W3: the mq_i4_reference twin.
    let b = &buckets[Codec::W3.idx()];
    if !b.ids.is_empty() {
        let mut exps: Vec<Option<ExpertTensor<'_>>> = (0..N_EXPERTS).map(|_| None).collect();
        for (&id, e) in &case.experts {
            if e.codec == Codec::W3 {
                exps[id as usize] = Some(ExpertTensor { profile: kind.w3_profile(), bytes: proj_bytes(e, kind) });
            }
        }
        mq_i4_reference::decode_moe_slots(
            kind.decode_kind(),
            kind.m(),
            kind.k(),
            &exps,
            &b.ids,
            &b.offs,
            &b.slots,
            case.c,
            x,
            xrs,
            ys,
            &mut y0,
            y1.as_deref_mut(),
        );
        // decode_moe_slots vs decode_gemv_f32 on a bounded slot sample (plumbing self-check).
        let mut sample: Vec<u32> = b.slots.clone();
        sample.sort_unstable();
        sample.truncate(12);
        for &slot in &sample {
            let slot = slot as usize;
            let e = &case.experts[&case.route[slot]];
            let xs = &x[(slot / kind.x_row_div()) * xrs..][..kind.k()];
            let row = mq_i4_reference::decode_gemv_f32(kind.w3_profile(), kind.m(), kind.k(), proj_bytes(e, kind), xs);
            for (r, v) in row.iter().enumerate() {
                let got = if r < kind.out_rows() {
                    y0[slot * ys + r]
                } else {
                    y1.as_ref().unwrap()[slot * ys + r - kind.out_rows()]
                };
                if got.to_bits() != v.to_bits() {
                    twin_mismatches += 1;
                }
            }
        }
        checked = sample.len();
    }
    // MQ4: the transcribed legacy arithmetic.
    let b = &buckets[Codec::Mq4.idx()];
    if !b.ids.is_empty() {
        let mut exps: Vec<Option<&[u8]>> = vec![None; N_EXPERTS];
        for (&id, e) in &case.experts {
            if e.codec == Codec::Mq4 {
                exps[id as usize] = Some(proj_bytes(e, kind));
            }
        }
        mq4_decode_moe_slots(kind, &exps, b, x, xrs, ys, &mut y0, y1.as_deref_mut());
    }
    Expected { y0, y1, twin_mismatches, twin_slots_checked: checked }
}

// ---------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Mm {
    count: u64,
    first: Vec<Value>,
}
impl Mm {
    fn push(&mut self, v: Value) {
        self.count += 1;
        if self.first.len() < 8 {
            self.first.push(v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_buf(
    part: &str,
    gpu: &[u8],
    exp: &[f32],
    ys: usize,
    out_rows: usize,
    slot_codec: &[Codec],
    per: &mut [Mm; 2],
    stray: &mut Mm,
) {
    for (i, (b, e)) in gpu.chunks_exact(4).zip(exp).enumerate() {
        let g = u32::from_le_bytes(b.try_into().unwrap());
        if g == e.to_bits() {
            continue;
        }
        let (slot, row) = (i / ys, i % ys);
        let rec = json!({"part": part, "word_index": i, "slot": slot, "row": row,
            "gpu_bits": bits_hex(g), "oracle_bits": bits_hex(e.to_bits()),
            "oracle_unwritten": e.to_bits() == SENTINEL_BITS});
        if row < out_rows {
            per[slot_codec[slot].idx()].push(rec);
        } else {
            stray.push(rec);
        }
    }
}

// ---------------------------------------------------------------------------------------
// One (case, kind) GPU run
// ---------------------------------------------------------------------------------------

struct KindOut {
    symbols: Vec<Value>,
    mismatches: u64,
    twin_mismatches: u64,
    launches: u64,
    expected: Expected,
    timing: Vec<(Codec, Kind, usize, usize, Vec<f64>)>,
}

fn run_kind(
    ctx: &Ctx,
    dev: &mut Dev,
    case: &Case,
    kind: Kind,
    slabs: &BTreeMap<u32, (u64, u64)>,
    timing: Option<&TimingCfg>,
) -> Result<KindOut> {
    let c = case.c;
    let slots = 10 * c;
    let (m, k, out) = (kind.m(), kind.k(), kind.out_rows());
    let (x, xrs, ys) = case.x(kind);
    let x_src_rows = x.len() / xrs;
    let buckets = buckets_of(case);
    let expected = cpu_expected(case, kind, &buckets);

    let xb = dev.upload(&f32_bytes(x))?;
    let mut table = vec![0u64; N_EXPERTS];
    for (&id, &(g, d)) in slabs {
        table[id as usize] = if kind == Kind::GateUp { g } else { d };
    }
    let tb = dev.upload(&u64_bytes(&table))?;
    let n_words = slots * ys;
    let y0 = dev.fill(n_words * 4, 0xa5)?;
    let y1 = if kind == Kind::GateUp { Some(dev.fill(n_words * 4, 0xa5)?) } else { None };

    let mut symbols = Vec::new();
    let mut prepared: Vec<(Codec, [u8; KERNARG_BYTES], [u32; 3])> = Vec::new();
    for codec in Codec::ALL {
        let b = &buckets[codec.idx()];
        if b.ids.is_empty() {
            symbols.push(json!({"symbol": symbol(codec, kind), "codec": codec.label(), "profile": kind.suffix(),
                "launched": false, "slots": 0, "unique_experts": 0}));
            continue;
        }
        let ib = dev.upload(&u32_bytes(&b.ids))?;
        let ob = dev.upload(&u32_bytes(&b.offs))?;
        let sb = dev.upload(&u32_bytes(&b.slots))?;
        let args = kernarg(
            [tb.ptr, ib.ptr, ob.ptr, sb.ptr, xb.ptr, y0.ptr, y1.map_or(0, |b| b.ptr)],
            m,
            k,
            b.ids.len(),
            c,
            kind.x_row_div(),
            x_src_rows,
            xrs,
            ys,
        );
        prepared.push((codec, args, [m as u32, b.ids.len() as u32, 1]));
    }
    for (codec, args, grid) in &prepared {
        let mut a = *args;
        // SAFETY: exact 88-byte ABI; every pointer is a live allocation of `dev` until release.
        unsafe { ctx.hip.launch_kernel_blob(ctx.func(*codec, kind), *grid, [32, 1, 1], 0, None, &mut a) }
            .map_err(err)?;
    }
    ctx.hip.device_synchronize().map_err(err)?;

    let slot_codec: Vec<Codec> = (0..slots).map(|s| case.codec_of_slot(s)).collect();
    let mut per = [Mm::default(), Mm::default()];
    let mut stray = Mm::default();
    let g0 = dev.download(y0, n_words * 4)?;
    compare_buf("y0", &g0, &expected.y0, ys, out, &slot_codec, &mut per, &mut stray);
    if let (Some(yb), Some(e1)) = (y1, expected.y1.as_ref()) {
        let g1 = dev.download(yb, n_words * 4)?;
        compare_buf("y1", &g1, e1, ys, out, &slot_codec, &mut per, &mut stray);
    }
    let mut mismatches = stray.count;
    for (i, codec) in Codec::ALL.iter().enumerate() {
        let b = &buckets[i];
        if b.ids.is_empty() {
            continue;
        }
        mismatches += per[i].count;
        symbols.push(json!({
            "symbol": symbol(*codec, kind), "codec": codec.label(), "profile": kind.suffix(), "launched": true,
            "grid": [m, b.ids.len(), 1], "slots": b.slots.len(), "unique_experts": b.ids.len(),
            "mismatch_count": per[i].count, "first_mismatches": per[i].first,
        }));
    }
    symbols.push(json!({"symbol": "stray_or_pad_words", "profile": kind.suffix(),
        "mismatch_count": stray.count, "first_mismatches": stray.first}));

    let mut timing_rows = Vec::new();
    if let Some(t) = timing {
        if mismatches != 0 {
            return Err(format!("{} not byte-exact; refusing to time a wrong kernel", kind.suffix()));
        }
        for (codec, args, grid) in &prepared {
            let f = ctx.func(*codec, kind);
            let start = ctx.hip.event_create().map_err(err)?;
            let stop = ctx.hip.event_create().map_err(err)?;
            let mut samples = Vec::with_capacity(t.reps);
            for i in 0..t.warmup + t.reps {
                ctx.hip.event_record(&start, None).map_err(err)?;
                for _ in 0..t.launches {
                    let mut a = *args;
                    // SAFETY: as above; the launch reads the blob during the call.
                    unsafe { ctx.hip.launch_kernel_blob(f, *grid, [32, 1, 1], 0, None, &mut a) }.map_err(err)?;
                }
                ctx.hip.event_record(&stop, None).map_err(err)?;
                ctx.hip.event_synchronize(&stop).map_err(err)?;
                let ms = ctx.hip.event_elapsed_ms(&start, &stop).map_err(err)?;
                if i >= t.warmup {
                    samples.push(ms as f64 / 1000.0);
                }
            }
            let _ = ctx.hip.event_destroy(start);
            let _ = ctx.hip.event_destroy(stop);
            timing_rows.push((*codec, kind, slots, grid[1] as usize, samples));
        }
    }
    Ok(KindOut {
        symbols,
        mismatches,
        twin_mismatches: expected.twin_mismatches,
        launches: prepared.len() as u64,
        expected,
        timing: timing_rows,
    })
}

// ---------------------------------------------------------------------------------------
// Independent old-kernel parity (production HIP kernels, JIT via rdna_compute::Gpu)
// ---------------------------------------------------------------------------------------

fn old_kernel_parity(gpu: &mut Gpu, case: &Case, gate: &Expected, down: &Expected) -> Result<Value> {
    let c = case.c;
    fn up(gpu: &mut Gpu, bytes: &[u8]) -> Result<GpuTensor> {
        let t = gpu.alloc_tensor(&[bytes.len().max(4)], DType::Raw).map_err(err)?;
        gpu.hip.memcpy_htod(&t.buf, bytes).map_err(err)?;
        Ok(t)
    }
    fn upf(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor> {
        let t = gpu.alloc_tensor(&[v.len()], DType::F32).map_err(err)?;
        gpu.hip.memcpy_htod(&t.buf, &f32_bytes(v)).map_err(err)?;
        Ok(t)
    }
    fn blank(gpu: &mut Gpu, n: usize) -> Result<GpuTensor> {
        let t = gpu.alloc_tensor(&[n], DType::F32).map_err(err)?;
        gpu.hip.memset(&t.buf, 0xa5, n * 4).map_err(err)?;
        Ok(t)
    }
    fn read(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; n * 4];
        gpu.hip.memcpy_dtoh(&mut out, &t.buf).map_err(err)?;
        Ok(out)
    }
    fn cmp(got: &[u8], exp: &[f32]) -> (u64, Vec<Value>) {
        let mut mm = Mm::default();
        for (i, (b, e)) in got.chunks_exact(4).zip(exp).enumerate() {
            let g = u32::from_le_bytes(b.try_into().unwrap());
            if g != e.to_bits() {
                mm.push(json!({"word_index": i, "old_kernel_bits": bits_hex(g), "oracle_bits": bits_hex(e.to_bits())}));
            }
        }
        (mm.count, mm.first)
    }
    let mut keep: Vec<GpuTensor> = Vec::new();
    let (mut gtab, mut dtab) = (vec![0u64; N_EXPERTS], vec![0u64; N_EXPERTS]);
    for (&id, e) in &case.experts {
        let g = up(gpu, &e.gate)?;
        let d = up(gpu, &e.down)?;
        gtab[id as usize] = g.buf.as_ptr() as u64;
        dtab[id as usize] = d.buf.as_ptr() as u64;
        keep.push(g);
        keep.push(d);
    }
    let gt = up(gpu, &u64_bytes(&gtab))?;
    let dt = up(gpu, &u64_bytes(&dtab))?;
    let topk: Vec<u8> = case.route.iter().flat_map(|&e| (e as i32).to_le_bytes()).collect();
    let tk = up(gpu, &topk)?;
    let xg = upf(gpu, &case.x_gate)?;
    let xd = upf(gpu, &case.x_down)?;
    let yg = blank(gpu, c * 10 * GATE_M / 2)?;
    let yu = blank(gpu, c * 10 * GATE_M / 2)?;
    let yd = blank(gpu, c * 10 * DOWN_M)?;
    gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(&gt, &tk, &xg, &yg, &yu, GATE_M, GATE_K, c)
        .map_err(err)?;
    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(&dt, &tk, &xd, &yd, DOWN_M, DOWN_K, c, N_EXPERTS)
        .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let (n_gate, n_down) = (c * 10 * GATE_M / 2, c * 10 * DOWN_M);
    let (g0, g1, d0) = (read(gpu, &yg, n_gate)?, read(gpu, &yu, n_gate)?, read(gpu, &yd, n_down)?);
    let rows = [
        ("gate_up_y0", cmp(&g0, &gate.y0)),
        ("gate_up_y1", cmp(&g1, gate.y1.as_ref().expect("gate has Y1"))),
        ("down_y0", cmp(&d0, &down.y0)),
    ];
    let total: u64 = rows.iter().map(|(_, (n, _))| *n).sum();
    let detail: Vec<Value> = rows
        .iter()
        .map(|(n, (count, first))| json!({"part": n, "mismatch_count": count, "first_mismatches": first}))
        .collect();
    for t in keep.into_iter().chain([gt, dt, tk, xg, xd, yg, yu, yd]) {
        gpu.free_tensor(t).map_err(err)?;
    }
    Ok(json!({"kernels": ["gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded"],
        "mismatch_count": total, "parts": detail}))
}

// ---------------------------------------------------------------------------------------
// Case driver
// ---------------------------------------------------------------------------------------

#[derive(Default)]
struct Totals {
    cases: u64,
    synthetic: u64,
    real: u64,
    real_records: u64,
    launches: u64,
    mismatches: u64,
    twin_mismatches: u64,
    old_kernel_mismatches: u64,
    old_kernel_cases: u64,
}

type TimingSamples = Vec<(Codec, Kind, usize, usize, Vec<f64>)>;

fn run_case(ctx: &mut Ctx, case: &Case, t: &mut Totals, timing: Option<&TimingCfg>) -> Result<(Value, TimingSamples)> {
    let mut dev = Dev::new(&ctx.hip);
    let slabs = upload_experts(&mut dev, case);
    let gate = match &slabs {
        Ok(s) => run_kind(ctx, &mut dev, case, Kind::GateUp, s, timing),
        Err(e) => Err(e.clone()),
    };
    let down = match (&slabs, &gate) {
        (Ok(s), Ok(_)) => run_kind(ctx, &mut dev, case, Kind::Down, s, timing),
        _ => Err("skipped after earlier failure".to_string()),
    };
    dev.release();
    slabs?;
    let (gate, down) = (gate?, down?);
    let mut counts = BTreeMap::new();
    for codec in Codec::ALL {
        counts.insert(codec.label(), case.route.iter().filter(|e| case.experts[*e].codec == codec).count());
    }
    let unique_by_codec: BTreeMap<&str, usize> = Codec::ALL
        .iter()
        .map(|c| (c.label(), case.experts.values().filter(|e| e.codec == *c).count()))
        .collect();
    t.cases += 1;
    match case.origin {
        "synthetic" => t.synthetic += 1,
        RECORDS_ORIGIN => {
            t.real += 1;
            t.real_records += 1;
        }
        _ => t.real += 1,
    }
    t.launches += gate.launches + down.launches;
    t.mismatches += gate.mismatches + down.mismatches;
    t.twin_mismatches += gate.twin_mismatches + down.twin_mismatches;
    let mut row = json!({
        "case": case.name, "origin": case.origin, "c": case.c, "slots": 10 * case.c,
        "unique_experts": case.experts.len(), "slots_by_codec": counts, "unique_experts_by_codec": unique_by_codec,
        "strides": {"x_row_stride_gate": case.xrs_gate, "x_row_stride_down": case.xrs_down,
            "y_slot_stride_gate": case.ys_gate, "y_slot_stride_down": case.ys_down},
        "input_sha256": case.input_sha256(),
        "gate_up": gate.symbols, "down": down.symbols,
        "mismatch_count": gate.mismatches + down.mismatches,
        "twin_selfcheck": {"slots_checked_gate": gate.expected.twin_slots_checked,
            "slots_checked_down": down.expected.twin_slots_checked,
            "mismatch_count": gate.twin_mismatches + down.twin_mismatches},
    });
    if ctx.old_parity && case.old_parity && case.tight() {
        if ctx.gpu.is_none() {
            ctx.gpu = Some(Gpu::init_with_device(ctx.device).map_err(err)?);
        }
        let gpu = ctx.gpu.as_mut().expect("initialised above");
        let r = old_kernel_parity(gpu, case, &gate.expected, &down.expected)?;
        t.old_kernel_cases += 1;
        t.old_kernel_mismatches += r["mismatch_count"].as_u64().unwrap_or(1);
        row["old_kernel_parity"] = r;
    }
    let mut samples = gate.timing;
    samples.extend(down.timing);
    Ok((row, samples))
}

// ---------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------

#[derive(Debug)]
struct Cfg {
    codeobject: String,
    artifact: Option<String>,
    layer: usize,
    real_experts: usize,
    old_parity: bool,
    timing: bool,
    reps: usize,
    warmup: usize,
    launches: usize,
    device: Option<i32>,
    expect_uuid: String,
    expect_bdf: String,
    expect_arch: String,
    card: Option<String>,
    fabric_card: Option<String>,
    fabric_owner: Option<String>,
    fabric_wrapper: Option<String>,
    only: Vec<String>,
    /// Directory holding the layer's `E{e}.npz` certified R1 records (e.g. `.../q/w3-gptq3/L0`).
    records_dir: Option<String>,
    /// The layer's pack `manifest.json` (npz_sha256 / sha256 per expert).
    records_manifest: Option<String>,
    records_manifest_sha256: Option<String>,
    /// R1 allocation JSON and its point (`ctl3-hot`): per-(layer, expert) W3 / MQ4-XTS map.
    alloc: Option<String>,
    alloc_point: String,
    /// Optional `ctl3-hot.map.json`, cross-checked against the allocation.
    map: Option<String>,
    /// MQ4-XTS source HFQM holding the aggregate expert tensors of the hot experts.
    xts: Option<String>,
    xts_sha256: Option<String>,
    layer0: bool,
}

const USAGE: &str = "usage: mq_i4_moe_gemv_byte_oracle --codeobject PATH [--artifact HFQM --layer N [--real-experts N]] \
[--records-dir DIR --records-manifest FILE --alloc FILE [--point NAME] --xts HFQ [--layer0|--layer N] [--map FILE] \
[--records-manifest-sha256 HEX] [--xts-sha256 HEX] [--real-experts N]] \
[--old-kernel-parity] [--timing --reps N(>=20) --warmup N(>=3) [--launches N] --card NAME --fabric-owner OWNER \
--fabric-wrapper fabric-run.sh [--fabric-card NAME]] [--device N] [--expect-uuid U] [--expect-bdf B] [--expect-arch A] \
[--only SUBSTR]...  (card/fabric flags fall back to env FABRIC_CARD / FABRIC_OWNER / FABRIC_WRAPPER)";

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn parse_args(argv: &[String]) -> Result<Cfg> {
    let mut cfg = Cfg {
        codeobject: String::new(),
        artifact: None,
        layer: 0,
        real_experts: 24,
        old_parity: false,
        timing: false,
        reps: 30,
        warmup: 5,
        launches: 32,
        device: None,
        expect_uuid: DEFAULT_UUID.into(),
        expect_bdf: DEFAULT_BDF.into(),
        expect_arch: DEFAULT_ARCH.into(),
        card: env_nonempty("FABRIC_CARD"),
        fabric_card: None,
        fabric_owner: env_nonempty("FABRIC_OWNER"),
        fabric_wrapper: env_nonempty("FABRIC_WRAPPER"),
        only: Vec::new(),
        records_dir: None,
        records_manifest: None,
        records_manifest_sha256: None,
        alloc: None,
        alloc_point: "ctl3-hot".into(),
        map: None,
        xts: None,
        xts_sha256: None,
        layer0: false,
    };
    let mut i = 0;
    let mut layer_explicit: Option<usize> = None;
    let value = |i: &mut usize| -> Result<String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("{} needs a value\n{USAGE}", argv[*i - 1]))
    };
    let number = |i: &mut usize| -> Result<usize> {
        let flag = argv[*i].clone();
        value(i)?.parse::<usize>().map_err(|e| format!("{flag}: {e}"))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--codeobject" => cfg.codeobject = value(&mut i)?,
            "--artifact" => cfg.artifact = Some(value(&mut i)?),
            "--layer" => {
                let l = number(&mut i)?;
                layer_explicit = Some(l);
                cfg.layer = l;
            }
            "--real-experts" => cfg.real_experts = number(&mut i)?,
            "--old-kernel-parity" => cfg.old_parity = true,
            "--timing" => cfg.timing = true,
            "--reps" => cfg.reps = number(&mut i)?,
            "--warmup" => cfg.warmup = number(&mut i)?,
            "--launches" => cfg.launches = number(&mut i)?,
            "--device" => cfg.device = Some(number(&mut i)? as i32),
            "--expect-uuid" => cfg.expect_uuid = value(&mut i)?,
            "--expect-bdf" => cfg.expect_bdf = value(&mut i)?.to_ascii_lowercase(),
            "--expect-arch" => cfg.expect_arch = value(&mut i)?,
            "--card" => cfg.card = Some(value(&mut i)?),
            "--fabric-card" => cfg.fabric_card = Some(value(&mut i)?),
            "--fabric-owner" => cfg.fabric_owner = Some(value(&mut i)?),
            "--fabric-wrapper" => cfg.fabric_wrapper = Some(value(&mut i)?),
            "--only" => cfg.only.push(value(&mut i)?),
            "--records-dir" => cfg.records_dir = Some(value(&mut i)?),
            "--records-manifest" => cfg.records_manifest = Some(value(&mut i)?),
            "--records-manifest-sha256" => cfg.records_manifest_sha256 = Some(value(&mut i)?.to_ascii_lowercase()),
            "--alloc" => cfg.alloc = Some(value(&mut i)?),
            "--point" => cfg.alloc_point = value(&mut i)?,
            "--map" => cfg.map = Some(value(&mut i)?),
            "--xts" => cfg.xts = Some(value(&mut i)?),
            "--xts-sha256" => cfg.xts_sha256 = Some(value(&mut i)?.to_ascii_lowercase()),
            "--layer0" => cfg.layer0 = true,
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 1;
    }
    if cfg.codeobject.is_empty() {
        return Err(USAGE.into());
    }
    if cfg.timing {
        if cfg.reps < 20 {
            return Err("--timing requires --reps >= 20".into());
        }
        if cfg.warmup < 3 {
            return Err("--timing requires --warmup >= 3".into());
        }
        if cfg.launches < 1 {
            return Err("--timing requires --launches >= 1".into());
        }
        if cfg.artifact.is_some() || cfg.old_parity {
            return Err("--timing is a separate run: no --artifact / --old-kernel-parity".into());
        }
        for (what, v) in [("--card", &cfg.card), ("--fabric-owner", &cfg.fabric_owner), ("--fabric-wrapper", &cfg.fabric_wrapper)] {
            if v.is_none() {
                return Err(format!("--timing requires {what} (or its env fallback): fabric metadata is part of the record"));
            }
        }
    }
    if cfg.layer0 {
        if layer_explicit.is_some_and(|l| l != 0) {
            return Err("--layer0 conflicts with --layer N (N != 0)".into());
        }
        cfg.layer = 0;
    }
    let records_flags = [
        ("--records-dir", cfg.records_dir.is_some()),
        ("--records-manifest", cfg.records_manifest.is_some()),
        ("--alloc", cfg.alloc.is_some()),
        ("--xts", cfg.xts.is_some()),
    ];
    let any_records = records_flags.iter().any(|(_, set)| *set);
    if any_records {
        if let Some((missing, _)) = records_flags.iter().find(|(_, set)| !*set) {
            return Err(format!("certified-records mode needs --records-dir, --records-manifest, --alloc and --xts together; {missing} is missing"));
        }
        if cfg.timing {
            return Err("--timing is a separate run: no certified-records flags".into());
        }
    } else if cfg.layer0 || cfg.map.is_some() || cfg.xts_sha256.is_some() || cfg.records_manifest_sha256.is_some() {
        return Err("--layer0/--map/--xts-sha256/--records-manifest-sha256 only apply to certified-records mode".into());
    }
    Ok(cfg)
}

fn selected(cfg: &Cfg, name: &str) -> bool {
    cfg.only.is_empty() || cfg.only.iter().any(|s| name.contains(s.as_str()))
}

fn run() -> Result<(Value, bool)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cfg = parse_args(&argv)?;
    let hip = HipRuntime::load().map_err(err)?;
    if let Some(d) = cfg.device {
        hip.set_device(d).map_err(err)?;
    }
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let identity_match = uuid == cfg.expect_uuid && bdf == cfg.expect_bdf && arch == cfg.expect_arch;
    let device = json!({"index": id, "uuid": uuid, "bdf": bdf, "arch": arch, "card": cfg.card,
        "expected_uuid": cfg.expect_uuid, "expected_bdf": cfg.expect_bdf, "expected_arch": cfg.expect_arch,
        "identity_match": identity_match});
    let command = json!({"argv": argv, "build_command": BUILD_COMMAND});
    if !identity_match {
        return Ok((
            json!({"schema": SCHEMA, "command": command, "device": device,
                "error": "MQ-I4 MoE GEMV device identity mismatch", "byte_exact": false, "pass": false, "rc": 1}),
            false,
        ));
    }
    let image = std::fs::read(&cfg.codeobject).map_err(|e| format!("{}: {e}", cfg.codeobject))?;
    let codeobject_sha = hex(&sha(&image));
    let module = hip.module_load_data(&image).map_err(err)?;
    let mut funcs = Vec::new();
    let mut loads = Vec::new();
    for kind in Kind::ALL {
        for codec in Codec::ALL {
            let name = symbol(codec, kind);
            let f = hip.module_get_function(&module, &name).map_err(|e| format!("symbol {name}: {e}"))?;
            loads.push(name);
            funcs.push((codec, kind, f));
        }
    }
    let mut ctx = Ctx { hip, funcs, codeobject_sha: codeobject_sha.clone(), device: id, old_parity: cfg.old_parity, gpu: None };
    let mut totals = Totals::default();
    let mut rows: Vec<Value> = Vec::new();
    let mut runs: Vec<Value> = Vec::new();
    let mut artifact_meta = Value::Null;
    let mut mq4_violations = 0u64;
    let mut records_meta = Value::Null;
    let mut records_violations = 0u64;

    if cfg.timing {
        let tcfg = TimingCfg { reps: cfg.reps, warmup: cfg.warmup, launches: cfg.launches };
        let card = cfg.card.clone().unwrap_or_default();
        let fabric = json!({"class": "timing", "card": cfg.fabric_card.clone().unwrap_or_else(|| card.clone()),
            "owner": cfg.fabric_owner, "wrapper": cfg.fabric_wrapper});
        for codec in Codec::ALL {
            // 10 distinct experts of one codec, C1: one launch covers `M * 10` rows per profile.
            let name = format!("timing_c1_all_{}", codec.label());
            let mut rng = Rng::new(name_seed(&name));
            let ids = pick_ids(&mut rng, 10, true);
            let experts = experts_for(&ids, |_, _| codec);
            let mut route = ids.clone();
            rng.shuffle(&mut route);
            let case = build_case(&name, "synthetic", 1, route, experts, false, false, 0);
            eprintln!("[timing] {name}");
            let (row, samples) = run_case(&mut ctx, &case, &mut totals, Some(&tcfg))?;
            rows.push(row);
            for (codec, kind, slots, unique, elapsed) in samples {
                runs.push(json!({
                    "codec": codec.timing_name(), "profile": kind.timing_profile(),
                    "symbol": symbol(codec, kind), "m": kind.m(), "k": kind.k(), "M": kind.m(), "K": kind.k(),
                    "row_count": kind.m() * 10, "c": 1, "slots": slots, "unique_experts": unique,
                    "elapsed_s": elapsed, "reps": tcfg.reps, "warmup": tcfg.warmup,
                    "launches_per_sample": tcfg.launches,
                    "symbol_codeobject": MODULE, "code_object_sha": ctx.codeobject_sha,
                    "timing_source": "device", "event_timing": "hipEventElapsedTime over launches_per_sample launches",
                    "device": {"arch": arch, "card": card, "uuid": uuid, "bdf": bdf},
                    "fabric": fabric,
                    "verified_byte_exact_before_timing": true,
                }));
            }
        }
    } else {
        for name in SYNTH_CASES.iter().filter(|n| selected(&cfg, n)) {
            eprintln!("[g1] {name}");
            let case = build_synthetic(name);
            let (row, _) = run_case(&mut ctx, &case, &mut totals, None)?;
            rows.push(row);
        }
        if let Some(path) = &cfg.artifact {
            eprintln!("[g1] loading layer {} of {path}", cfg.layer);
            let real = load_real(path, cfg.layer, cfg.real_experts)?;
            mq4_violations = real.mq4_sym_violations;
            artifact_meta = real.meta.clone();
            for case in real_cases(&real, "real", "real").iter().filter(|c| selected(&cfg, &c.name)) {
                eprintln!("[g1] {}", case.name);
                let (row, _) = run_case(&mut ctx, case, &mut totals, None)?;
                rows.push(row);
            }
        }
        if cfg.records_dir.is_some() {
            eprintln!("[g1] loading certified L{} records", cfg.layer);
            let real = load_records(&cfg)?;
            records_violations = real.mq4_sym_violations;
            records_meta = real.meta.clone();
            for case in real_cases(&real, RECORDS_ORIGIN, "real_records").iter().filter(|c| selected(&cfg, &c.name)) {
                eprintln!("[g1] {}", case.name);
                let (row, _) = run_case(&mut ctx, case, &mut totals, None)?;
                rows.push(row);
            }
        }
    }

    let byte_exact = totals.mismatches == 0
        && totals.twin_mismatches == 0
        && totals.old_kernel_mismatches == 0
        && !rows.is_empty()
        && (!cfg.old_parity || totals.old_kernel_cases > 0);
    let mut out = json!({
        "schema": SCHEMA,
        "mode": if cfg.timing { "timing" } else { "g1" },
        "module": MODULE,
        "command": command,
        "config": {"layer": cfg.layer, "real_experts": cfg.real_experts, "old_kernel_parity": cfg.old_parity,
            "only": cfg.only, "timing": cfg.timing, "reps": cfg.reps, "warmup": cfg.warmup,
            "launches_per_sample": cfg.launches},
        "codeobject": {"path": cfg.codeobject, "sha256": codeobject_sha, "bytes": image.len()},
        "code_object_sha": codeobject_sha,
        "kernarg_bytes": KERNARG_BYTES,
        "symbols_loaded": loads,
        "device": device,
        "case_counts": {"total": totals.cases, "synthetic": totals.synthetic, "real": totals.real,
            "real_records_layer": totals.real_records, "old_kernel_parity": totals.old_kernel_cases},
        "symbol_launches": totals.launches,
        "cases": rows,
        "mismatch_count": totals.mismatches,
        "twin_selfcheck_mismatch_count": totals.twin_mismatches,
        "old_kernel_mismatch_count": totals.old_kernel_mismatches,
        "mq4_sym_header_violations_real": mq4_violations,
        "mq4_sym_header_violations_records": records_violations,
        "byte_exact": byte_exact,
        "pass": byte_exact,
        "rc": if byte_exact { 0 } else { 1 },
    });
    if !artifact_meta.is_null() {
        out["artifact"] = artifact_meta;
    }
    if !records_meta.is_null() {
        out["real_records_layer"] = records_meta;
    }
    if cfg.timing {
        out["runs"] = Value::Array(runs);
    }
    Ok((out, byte_exact))
}

fn main() {
    match run() {
        Ok((v, pass)) => {
            println!("{v}");
            if !pass {
                eprintln!("MQ-I4 MoE GEMV byte oracle FAILED");
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("{}", json!({"schema": SCHEMA, "error": e, "byte_exact": false, "pass": false, "rc": 1}));
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Local known-answer tests (no GPU):
//   cargo test -p hipfire-runtime --release --features lab --example mq_i4_moe_gemv_byte_oracle
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cyc(n: usize) -> Vec<u8> {
        (0..n).map(|j| (j % 16) as u8).collect()
    }

    #[test]
    fn sym_zero_is_exact_at_extremes() {
        assert_eq!(sym_zero_bits(0x6fff), 0xfbff); // -65504
        assert_eq!(sym_zero_bits(0x0001), 0x8008); // -2^-21
        assert_eq!(f16f(sym_zero_bits(0x3c00)), -8.0);
        assert_eq!(f16f(0x6fff), 8188.0);
    }

    /// Exactly representable data: every partial sum is an integer, so rounding cannot hide a
    /// header-half, nibble-order or lane-ownership error. Row 2's second half uses scale 2.
    #[test]
    fn mq4_gate_known_answer() {
        let (m, k) = (3usize, 256usize);
        let mut codes = cyc(m * k);
        codes[k..2 * k].fill(8); // row 1: d = 0 everywhere
        let mut scales = vec![0x3c00u16; m * 2];
        scales[2 * 2 + 1] = 0x4000; // row 2: half0 scale 1, half1 scale 2
        let bytes = encode_mq4(Kind::GateUp, m, k, &codes, &scales);
        let x = vec![1.0f32; k];
        let got: Vec<f32> = (0..m).map(|r| mq4_row_gate(&bytes[r * 136..(r + 1) * 136], &x)).collect();
        // Each 16-code cycle sums to -8; 128 columns = 8 cycles = -64.
        assert_eq!(got, vec![-128.0, 0.0, -192.0]);
    }

    #[test]
    fn mq4_down_known_answer() {
        let (m, k) = (2usize, 640usize);
        let mut codes = cyc(m * k);
        codes[k..].fill(8);
        let scales = vec![0x3800u16; m * (k / 128)]; // 0.5, zp = -4
        let bytes = encode_mq4(Kind::Down, m, k, &codes, &scales);
        let x = vec![2.0f32; k];
        let rb = (k / 128) * 68;
        let got: Vec<f32> = (0..m).map(|r| mq4_row_down(&bytes[r * rb..(r + 1) * rb], &x)).collect();
        assert_eq!(got, vec![-320.0, 0.0]); // 5 groups * 8 cycles * (-8)
    }

    /// The oracle takes the audited gfx1201 contraction ("element 1 first"); the opposite
    /// orientation must differ on full-mantissa data or a pass would prove nothing.
    #[test]
    fn mq4_orientation_is_element1_first_and_sensitive() {
        let k = 640usize;
        let mut rng = Rng::new(7);
        let mut codes = vec![0u8; 64 * k];
        for c in codes.iter_mut() {
            *c = (rng.u32() & 15) as u8;
        }
        let scales: Vec<u16> = (0..64 * 5).map(|_| 0x2800 + (rng.u32() as u16 & 0x3ff)).collect();
        let bytes = encode_mq4(Kind::Down, 64, k, &codes, &scales);
        let x = gen_x_row(0, k, &mut rng);
        let rb = 5 * 68;
        let alt = |row: &[u8]| -> f32 {
            let mut lanes = [0.0f32; 32];
            for (tid, lane) in lanes.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for g in 0..5 {
                    let gp = &row[g * 68..];
                    let sc = f16f(u16::from_le_bytes([gp[0], gp[1]]));
                    let zp = f16f(u16::from_le_bytes([gp[2], gp[3]]));
                    let (b0, b1) = (gp[4 + tid * 2], gp[5 + tid * 2]);
                    let q = [b0 & 15, b0 >> 4, b1 & 15, b1 >> 4];
                    let d = |i: usize| sc.mul_add(q[i] as f32, zp);
                    let xb = g * 128 + tid * 4;
                    let mut s = d(0) * x[xb];
                    for i in 1..4 {
                        s = d(i).mul_add(x[xb + i], s);
                    }
                    acc += s;
                }
                *lane = acc;
            }
            lane_tree(lanes)
        };
        let differing = (0..64)
            .filter(|&r| {
                let row = &bytes[r * rb..(r + 1) * rb];
                mq4_row_down(row, &x).to_bits() != alt(row).to_bits()
            })
            .count();
        assert!(differing > 0, "fixture is not contraction-sensitive");
    }

    /// 2^24 + 1 + 1 + 1: the 16,8,4,2,1 tree folds lane1+lane3 first (1+1 = 2, exact) so the
    /// result is 2^24 + 2; a left-to-right sum rounds every +1 away and stays 2^24.
    #[test]
    fn lane_tree_order_is_the_shuffle_tree() {
        let mut lanes = [0.0f32; 32];
        lanes[0] = 16777216.0;
        lanes[1] = 1.0;
        lanes[2] = 1.0;
        lanes[3] = 1.0;
        // step 2: l0 = 2^24 + l2 (=1) rounds to 2^24 (ties-to-even); l1 = 1 + l3 = 2; step 1: 2^24 + 2.
        assert_eq!(lane_tree(lanes), 16777218.0);
    }

    fn check_partition(route: &[u32], codecs: &BTreeMap<u32, Codec>) {
        let f = |e: u32| codecs[&e];
        let bs = [build_bucket(route, &f, Codec::W3), build_bucket(route, &f, Codec::Mq4)];
        let mut seen = vec![false; route.len()];
        for (ci, b) in bs.iter().enumerate() {
            assert_eq!(b.offs.len(), b.ids.len() + 1);
            assert_eq!(b.offs[0], 0);
            assert_eq!(*b.offs.last().unwrap() as usize, b.slots.len());
            assert!(b.ids.windows(2).all(|w| w[0] < w[1]), "ids ascending and unique");
            for (u, &id) in b.ids.iter().enumerate() {
                let seg = &b.slots[b.offs[u] as usize..b.offs[u + 1] as usize];
                assert!(!seg.is_empty());
                assert!(seg.windows(2).all(|w| w[0] < w[1]), "stable original slot order");
                for &s in seg {
                    assert_eq!(route[s as usize], id);
                    assert_eq!(codecs[&id], Codec::ALL[ci]);
                    assert!(!seen[s as usize], "slot appears once");
                    seen[s as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "buckets partition all 10C slots");
    }

    #[test]
    fn buckets_partition_slots_for_union_and_unique_routes() {
        let mut rng = Rng::new(11);
        let pool = pick_ids(&mut rng, 14, true);
        let codecs: BTreeMap<u32, Codec> = pool.iter().enumerate().map(|(i, &id)| (id, MIXED10[i % 10])).collect();
        check_partition(&lane_routes(&mut rng, &pool, 8), &codecs); // repeated across lanes
        let uniq = pick_ids(&mut rng, 80, true);
        let codecs: BTreeMap<u32, Codec> =
            uniq.iter().enumerate().map(|(i, &id)| (id, MIXED10[(i * 7) % 10])).collect();
        let mut route = uniq.clone();
        rng.shuffle(&mut route);
        check_partition(&route, &codecs); // 80 unique ids
    }

    #[test]
    fn kernarg_layout_is_the_frozen_88_bytes() {
        let a = kernarg([1, 2, 3, 4, 5, 6, 7], 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq!(a.len(), 88);
        for i in 0..7usize {
            assert_eq!(u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap()), i as u64 + 1);
        }
        for (j, off) in [56usize, 60, 64, 68, 72, 76, 80, 84].iter().enumerate() {
            assert_eq!(u32::from_le_bytes(a[*off..*off + 4].try_into().unwrap()), 8 + j as u32);
        }
    }

    #[test]
    fn synthetic_tensors_have_exact_sizes_and_valid_w3() {
        for kind in Kind::ALL {
            for codec in Codec::ALL {
                assert_eq!(gen_tensor(codec, kind, 5, true).len(), kind.tensor_bytes(codec));
            }
            mq_i4::decode_tensor(kind.w3_profile(), kind.m(), kind.k(), &gen_tensor(Codec::W3, kind, 5, true))
                .unwrap();
        }
    }

    #[test]
    fn mq4_expert_headers_are_symmetric_and_finite() {
        for kind in Kind::ALL {
            let t = gen_tensor(Codec::Mq4, kind, 9, true);
            assert_eq!(mq4_sym_violations(kind, &t), 0);
        }
    }

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    const RECORDS: [&str; 8] = ["--records-dir", "/r/q/w3-gptq3/L0", "--records-manifest", "/r/m.json", "--alloc", "/r/a.json", "--xts", "/x.mq4"];

    fn with(extra: &[&str]) -> Vec<String> {
        let mut v = vec!["--codeobject", "/co.hxaco"];
        v.extend_from_slice(&RECORDS);
        v.extend_from_slice(extra);
        args(&v)
    }

    #[test]
    fn records_flags_parse_and_default_to_layer0_point_ctl3_hot() {
        let c = parse_args(&with(&["--layer0", "--map", "/r/map.json", "--xts-sha256", "AB12", "--records-manifest-sha256", "CD34"])).unwrap();
        assert_eq!(c.layer, 0);
        assert!(c.layer0);
        assert_eq!(c.alloc_point, "ctl3-hot");
        assert_eq!(c.records_dir.as_deref(), Some("/r/q/w3-gptq3/L0"));
        assert_eq!(c.xts_sha256.as_deref(), Some("ab12"));
        assert_eq!(c.records_manifest_sha256.as_deref(), Some("cd34"));
        assert_eq!(c.map.as_deref(), Some("/r/map.json"));
        assert_eq!(c.real_experts, 24);
        let c = parse_args(&with(&["--point", "other", "--layer", "0"])).unwrap();
        assert_eq!(c.alloc_point, "other");
    }

    #[test]
    fn records_flags_are_all_or_nothing_and_exclusive_of_timing() {
        let partial = args(&["--codeobject", "/co", "--records-dir", "/d", "--alloc", "/a", "--xts", "/x"]);
        assert!(parse_args(&partial).unwrap_err().contains("--records-manifest is missing"));
        assert!(parse_args(&args(&["--codeobject", "/co", "--layer0"])).is_err());
        assert!(parse_args(&with(&["--layer0", "--layer", "3"])).is_err());
        let timing = with(&["--timing", "--card", "c", "--fabric-owner", "o", "--fabric-wrapper", "w"]);
        assert!(parse_args(&timing).is_err());
        // The old flags are unchanged: --artifact alone still parses.
        assert!(parse_args(&args(&["--codeobject", "/co", "--artifact", "/a.hfq", "--layer", "5"])).is_ok());
    }

    #[test]
    fn hot_map_labels_parse_strictly() {
        let mut row: Vec<Value> = (0..N_EXPERTS).map(|e| json!(if e % 3 == 0 { "xts_mq4" } else { "w3" })).collect();
        let root = json!({"arm": "ctl3-hot", "layers": {"0": row.clone()}});
        let f = hot_map_formats(&root, 0).unwrap();
        assert_eq!(f[0], ExpertFormat::Mq4Xts);
        assert_eq!(f[1], ExpertFormat::Mq3I4);
        assert!(hot_map_formats(&root, 1).is_err());
        row[7] = json!("mq4");
        assert!(hot_map_formats(&json!({"layers": {"0": row.clone()}}), 0).is_err());
        row.pop();
        assert!(hot_map_formats(&json!({"layers": {"0": row}}), 0).is_err());
    }

    /// The R1 allocation shape (`profile_indices[48][512]`, 0=MQ2 1=MQ3 2=MQ4-XTS) through S2's loader,
    /// cross-checked with a map row in the other vocabulary.
    #[test]
    fn allocation_layer_slice_matches_map_vocabulary() {
        let rows: Vec<Vec<u64>> = (0..48).map(|l| (0..N_EXPERTS).map(|e| if (e + l) % 3 == 0 { 2 } else { 1 }).collect()).collect();
        let root = json!({"schema": "ctl3-hot-allocation-v1", "profiles": ["mq2-i4", "mq3-i4", "mq4-xts"],
            "allocations": {"ctl3-hot": {"feasible": true, "profile_indices": rows}}});
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("alloc.json");
        std::fs::write(&p, root.to_string()).unwrap();
        let a = load_allocation(&p, "ctl3-hot", SlabGeometry::FN).unwrap();
        let l2 = &a.formats[2 * N_EXPERTS..3 * N_EXPERTS];
        let map_row: Vec<Value> = (0..N_EXPERTS).map(|e| json!(if (e + 2) % 3 == 0 { "xts_mq4" } else { "w3" })).collect();
        assert_eq!(l2, &hot_map_formats(&json!({"layers": {"2": map_row}}), 2).unwrap()[..]);
        assert!(load_allocation(&p, "nope", SlabGeometry::FN).is_err());
    }

    #[test]
    fn selection_is_balanced_and_tops_up_the_short_side() {
        assert_eq!(select_counts(100, 100, 24), (12, 12));
        assert_eq!(select_counts(100, 4, 24), (20, 4));
        assert_eq!(select_counts(3, 100, 24), (3, 21));
        assert_eq!(select_counts(100, 100, 2), (5, 5)); // a top-10 lane needs 10 experts
        let v: Vec<u32> = (0..100).collect();
        let s = spread(&v, 12);
        assert!(s.windows(2).all(|w| w[0] < w[1]) && s[0] == 0 && *s.last().unwrap() < 100);
    }
}
