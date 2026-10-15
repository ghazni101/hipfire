// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sealed expert-slab route (`MoeRouteCapability::SlabCodecGrouped`) stage
//! helpers shared by the decode and prefill lowerings.
//!
//! The layer's experts live in one slab arena and carry a codec each (MQ4
//! symmetric or MQ3-I4). Two kernel families serve them:
//!
//! * `route_rows <= 8` (decode, MTP verify, a one-row forward): the K1 lane-union
//!   GEMV over codec buckets ([`rdna_compute::slab_moe`]). The layer sits in the
//!   path-1 stage shape (gate/up F32 rows, SwiGLU activation + 128-wide
//!   rotation, unweighted F32 down rows, top-10 combine); only the two GEMVs
//!   are K1 launches.
//! * longer prefill: the K2 grouped IU4 kernels, MQ4 and W3, over the same
//!   grouped buffers with codec-masked tile ids (disjoint tiles).
//!
//! A layer whose experts all share one codec carries no codec table; a mixed
//! layer's table is the layer's dtype-tag tensor: `[512]` u8 codec bytes
//! (`0` = MQ4, `1` = W3) followed by the same codes as `[512]` i32 (the
//! table-driven tile mask's owner table). The tensor's logical shape is the u8
//! table; its allocation is [`CODEC_TAG_ALLOC_BYTES`].

use super::layer_ops::hip;
use crate::families::moe::MoeDtypes;
use crate::types::DispatchError;
use rdna_compute::slab_moe::{SlabBuckets, SLAB_EXPERTS, SLAB_MAX_LANES};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Largest `route_rows` served by the K1 decode GEMV.
pub const K1_MAX_ROWS: usize = SLAB_MAX_LANES;
/// Bytes of the u8 codec table.
pub const CODEC_U8_BYTES: usize = SLAB_EXPERTS;
/// Bytes of the whole codec-tag allocation: u8 table, then i32 table.
pub const CODEC_TAG_ALLOC_BYTES: usize = CODEC_U8_BYTES + 4 * SLAB_EXPERTS;

fn refuse(what: &str) -> DispatchError {
    DispatchError::Hip(format!("sealed expert-slab route: {what}"))
}

/// The K1 bucket layout of a layer: its codec table when mixed, else the one
/// codec its (uniform) routed dtype names.
pub(crate) fn buckets<'a>(
    dtypes: &MoeDtypes<'_>,
    tags: Option<&'a GpuTensor>,
) -> Result<SlabBuckets<'a>, DispatchError> {
    match tags {
        Some(table) => {
            if table.buf.size() < CODEC_TAG_ALLOC_BYTES {
                return Err(refuse("codec tag allocation is smaller than the u8 + i32 tables"));
            }
            Ok(SlabBuckets::Table(table))
        }
        None => match dtypes.routed_gate_up {
            DType::MQ3I4G128R256 => Ok(SlabBuckets::Uniform(0)),
            DType::MQ4G256V2 => Ok(SlabBuckets::Uniform(1)),
            other => Err(refuse(&format!("uniform layer dtype {other:?} is not a slab codec"))),
        },
    }
}

/// The `[512]` i32 codec table of a mixed layer's tag tensor (the table-driven
/// tile mask's owner table).
pub(crate) fn codec_i32(tags: &GpuTensor) -> Result<GpuTensor, DispatchError> {
    if tags.buf.size() < CODEC_TAG_ALLOC_BYTES {
        return Err(refuse("codec tag allocation is smaller than the u8 + i32 tables"));
    }
    Ok(GpuTensor {
        buf: tags.buf.byte_view(CODEC_U8_BYTES, 4 * SLAB_EXPERTS),
        shape: vec![SLAB_EXPERTS],
        dtype: DType::F32,
    })
}

/// K1 gate/up of `lanes` lanes: plans the lane union of `topk` and launches
/// every non-empty bucket. `xr` `[lanes, 2560]` F32 rotated activations;
/// `gate_out`/`up_out` `[lanes * 10, 640]` F32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k1_gate_up(
    gpu: &mut Gpu,
    buckets: SlabBuckets<'_>,
    gate_up_ptrs: &GpuTensor,
    topk: &GpuTensor,
    xr: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    lanes: usize,
) -> Result<(), DispatchError> {
    hip(gpu.qwen4_slab_union_plan(topk, buckets, lanes))?;
    hip(gpu.mq_i4_moe_gemv_slab(false, buckets, gate_up_ptrs, xr, gate_out, Some(up_out), lanes))
}

/// K1 down of `lanes` lanes over the union the same call's gate/up planned:
/// `xrot` `[lanes * 10, 640]` F32, `y0` `[lanes * 10, 2560]` F32 unweighted.
pub(crate) fn k1_down(
    gpu: &mut Gpu,
    buckets: SlabBuckets<'_>,
    down_ptrs: &GpuTensor,
    xrot: &GpuTensor,
    y0: &GpuTensor,
    lanes: usize,
) -> Result<(), DispatchError> {
    hip(gpu.mq_i4_moe_gemv_slab(true, buckets, down_ptrs, xrot, y0, None, lanes))
}

#[cfg(test)]
mod tests {
    use super::super::slab_moe_decode::{self as k1, BucketKey, Codec, Kind};
    use rdna_compute::slab_moe as gpu;

    /// The launcher's constants are the certified K1 contract's: symbols in
    /// bucket order (W3 first), geometry, lanes, and the codec byte mapping.
    #[test]
    fn launcher_contract_is_the_k1_contract() {
        for (bucket, codec) in [(0usize, Codec::Mq3I4), (1usize, Codec::Mq4Sym)] {
            assert_eq!(Codec::ORDER[bucket], codec);
            for (down, kind) in [(0usize, Kind::GateUp), (1usize, Kind::Down)] {
                assert_eq!(
                    gpu::MQ_I4_MOE_GEMV_SYMBOLS[bucket][down],
                    BucketKey::new(codec, kind).symbol()
                );
            }
        }
        assert_eq!(gpu::SLAB_MAX_LANES, k1::MAX_LANES);
        assert_eq!(gpu::SLAB_TOP_K, k1::TOP_K);
        assert_eq!(gpu::SLAB_EXPERTS, k1::MAX_EXPERTS);
        assert_eq!((gpu::SLAB_GATE_UP_M, gpu::SLAB_GATE_UP_K), (k1::GATE_UP_M, k1::GATE_UP_K));
        assert_eq!((gpu::SLAB_DOWN_M, gpu::SLAB_DOWN_K), (k1::DOWN_M, k1::DOWN_K));
        // Codec bytes of the per-layer table: 1 = W3 -> bucket 0, 0 = MQ4 -> bucket 1.
        assert_eq!(gpu::slab_bucket_of_codec(1), Codec::Mq3I4.order_index());
        assert_eq!(gpu::slab_bucket_of_codec(0), Codec::Mq4Sym.order_index());
    }

    /// The planner's fixed-capacity layout is `6 S + 2` words per call.
    #[test]
    fn union_scratch_words_match_the_descriptor_layout() {
        for lanes in 1..=gpu::SLAB_MAX_LANES {
            let s = 10 * lanes;
            assert_eq!(rdna_compute::scratch::qwen4_slab_union_words(lanes), 2 * s + 2 * (s + 1) + 2 * s);
        }
    }
}
