// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched DFlash lanes of the VMM continuous-batching store.
//!
//! [`DflashVmmLaneState`] is the persistent private draft/verify state of one
//! continuous-batching lane: the singleton [`DflashState`] owners plus the
//! request-local cursors the singleton [`DflashSpeculator`] keeps beside it.
//! It is built by consuming a [`DflashLaneSnapshot`] (`take_vmm_lane`, which
//! transfers every owner and cursor losslessly) or cold, from a lane
//! `DflashState` (`new_dflash_lane_state` with `DflashOwnerKind::VmmLane`)
//! after the lane prompt prefill.
//!
//! A VMM lane owns NO FWHT rotation scratch (`mq_x_rot`/`mq_x_rot_f16`) and
//! no `VerifyScratch::prefill_batch` (the batched verify runs on the shared
//! trunk scratch; the lane never runs a target forward through its own
//! scratch). The rotation buffers are a per-GEMM-call temporary of the draft
//! forward, identical in size for every draft scratch, so the engine's ONE
//! shared draft scratch ([`DflashCbScratch::reserve_draft`]) owns them: the
//! batched draft borrows them directly (`draft_forward_lanes`), and every
//! other lane-draft path (the singleton-call fallback, the lane prompt
//! prefill backfill) lends them to the lane for the duration of the call
//! ([`DflashCbScratch::lend_mq`] / [`DflashCbScratch::reclaim_mq`]). All
//! calls run on the single stream, so the borrowed temporary is never live
//! across two calls. A lane promoted from the singleton arrives with the
//! singleton's own buffers, which `DflashState::shed_vmm_lane_scratch` frees.
//!
//! The in-crate halves of the shared DFlash verify:
//!
//! * [`dflash_lane_draft`] — the singleton greedy draft call of
//!   `spec_step_dflash` (`draft_dflash_block_rank`) on a lane's own draft
//!   state; returns `[seed, candidates..]`.
//! * [`dflash_cb_head_argmax`] — the singleton DFlash verify head
//!   (`dflash_enqueue_verify_lm_head_argmax` + one packed i32 D2H) over the
//!   first `rows` rows of a shared [`VerifyScratch`], for greedy picks only.
//!   Sampled lanes instead run a private Product head at their singleton b:
//!   Product dispatch is not row-count-independent above 63 aggregate rows.
//! * [`dflash_cb_verify`] — whole lanes packed into trunk chunks of at most
//!   the effective cap (`<= 63` rows; up to 128 with
//!   `HIPFIRE_CB_VERIFY_CHUNK128` on an admitted dense MQ4G256V2 gfx1201
//!   target) (`forward_prefill_batch_multi`, ChainVerify fusion, each lane's
//!   own hidden ring and tape at offset 0). Greedy lanes share a packed head;
//!   sampled lanes retain actual selector q and private FastTargetRows until
//!   their shared singleton accept/commit consumer runs.

use crate::dflash_spec::{
    DflashCheckpointPolicy, DflashLaneSnapshot, DflashState, DflashWindowMark,
};
use crate::qwen35::prefill::multi::{
    cb_dn_save_free_enabled, dflash_iu4_verify_admitted, forward_prefill_batch_multi, lane_gdn_batch_ready,
    multi_chunk_pack_cap, multi_chunk_scratch_rows, pack_whole_lanes, MultiChunkRequest,
    MultiChunkScratch, MULTI_CHUNK_MAX_LANE_ROWS, MULTI_CHUNK_PRODUCT_MAX_ROWS,
};
use crate::qwen35::prefill::{fp8_verify_open, verify_dense_math, DenseBatchMath};
use crate::qwen35::{self, DeltaNetState, DflashFusionCtx};
use crate::qwen35::{Qwen35Config, Qwen35Weights};
use crate::speculative::{
    dflash_download_verify_argmax, dflash_draft_batch_eligible, dflash_enqueue_verify_lm_head_argmax,
    draft_dflash_block_rank, draft_dflash_blocks_batched, DeltaNetSnapshot, DflashCbDraft, DflashDraftLane,
    dflash_sampled_selector_draft_into, dflash_target_rows_into, DflashSampleCfg, DflashSampledDraft,
    dflash_draft_batch_max_rows, FastTargetRows, VerifyScratch,
};
use hipfire_runtime::dflash::DflashScratch;
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::spec::{request_rng_state, SpecRequestConfig};
use rdna_compute::Gpu;
use std::ops::Range;

/// The singleton's verify block for remaining output budget `max_emit`
/// (`E > 0`): `B = min(block_size, max(E, 2))`. Even `E = 1` verifies two rows
/// (the accept clamp `max_accept = E - 1` then keeps zero drafts).
pub fn dflash_block_for_emit(block_size: usize, max_emit: usize) -> usize {
    block_size.min(max_emit.max(2))
}

/// Private DFlash state of one lane (see the module docs).
pub struct DflashVmmLaneState {
    /// The singleton owners: draft scratch/weights, hidden ring, verify
    /// scratch, target DeltaNet snapshot, GDN tape.
    pub df: DflashState,
    /// Committed target position at capture; the store keeps it equal to the
    /// request's position after every commit.
    pub position: usize,
    pub rng_state: u64,
    pub sample_temp: f32,
    pub sample_top_p: f32,
    pub sample_top_k: usize,
    pub sample_cactus: f32,
    /// Divergent-render checkpoints `(position, snapshot)`.
    pub checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub policy: DflashCheckpointPolicy,
    /// Pre-window mark of the last committed window (terminal-prefix repair).
    pub last_window: Option<DflashWindowMark>,
    /// Target picks of this lane's verified window (`B` ids), kept from the
    /// shared head until the greedy accept consumer runs.
    pub picks: Vec<u32>,
    /// Request-owned normalized target rows, consumed before chunk scratch reuse.
    pub sampled_target: FastTargetRows,
    /// Shared singleton first-token / acceptance workspace.
    pub sample_scratch: Vec<f32>,
    /// Actual selector proposal and q, retained beside the planned verify.
    pub sampled_draft: Option<DflashSampledDraft>,
    /// Remaining output budget `E` installed by the driver for the next
    /// window (`0` = not installed; a window needs `E >= 1`).
    pub max_emit: usize,
    /// `picks` hold the verify outcome of the lane's planned window.
    pub verified: bool,
    /// Tokens the last committed window emitted (`committed[1..]`).
    pub last_emit: Vec<u32>,
    /// The request's history length before the last window's emit was
    /// appended (terminal-prefix repair truncates back to it).
    pub last_history_len: usize,
}

// SAFETY: a lane lives inside the store (`Qwen35Bundle::vmm_store`, which must
// be `Send` to ride the model across threads) and, like every GPU owner there
// (`DeviceBuffer` is `Send`), is used only by the thread holding the model.
// The one non-`Send` part of a `DflashState` is its retained-PM4 route, which a
// lane keeps explicitly disabled; the shared draft weights `Arc` is a counted
// reference to the one resident set, read only through GPU calls under that
// same single owner.
unsafe impl Send for DflashVmmLaneState {}

impl DflashVmmLaneState {
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        df: DflashState,
        position: usize,
        rng_state: u64,
        sample_temp: f32,
        sample_top_p: f32,
        sample_top_k: usize,
        sample_cactus: f32,
        checkpoints: Vec<(usize, DeltaNetSnapshot)>,
        policy: DflashCheckpointPolicy,
        last_window: Option<DflashWindowMark>,
    ) -> Self {
        let block = df.block_size;
        Self {
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
            picks: Vec::with_capacity(block),
            sampled_target: FastTargetRows { probs: Vec::new(), tau: Vec::new(), z: Vec::new() },
            sample_scratch: Vec::new(),
            sampled_draft: None,
            max_emit: 0,
            verified: false,
            last_emit: Vec::with_capacity(block),
            last_history_len: 0,
        }
    }

    /// Consume `snapshot`, moving every owner and cursor into the lane.
    /// Read `snapshot.rows()` before calling.
    pub fn from_snapshot(snapshot: DflashLaneSnapshot) -> Self {
        let DflashLaneSnapshot {
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
        } = snapshot;
        Self::assemble(
            df,
            position,
            rng_state,
            sample_temp,
            sample_top_p,
            sample_top_k,
            sample_cactus,
            checkpoints,
            policy,
            last_window,
        )
    }

    /// A lane that just finished its cold prompt prefill (`position` prompt
    /// rows committed): request sampling configured exactly as
    /// `DflashSpeculator::configure_request`, no completed window yet.
    pub fn cold(
        df: DflashState,
        checkpoints: Vec<(usize, DeltaNetSnapshot)>,
        policy: DflashCheckpointPolicy,
        position: usize,
        request: &SpecRequestConfig,
    ) -> Self {
        Self::assemble(
            df,
            position,
            request_rng_state(request.rng_seed),
            request.temp,
            request.top_p,
            request.ar_candidate_cap(),
            request.cactus_delta,
            checkpoints,
            policy,
            None,
        )
    }

    pub fn sample_cfg(&self) -> DflashSampleCfg {
        DflashSampleCfg {
            temp: self.sample_temp,
            top_p: self.sample_top_p,
            top_k: self.sample_top_k,
            cactus_delta: self.sample_cactus,
        }
    }

    /// Free every owned GPU object exactly once (shared draft weights go
    /// through `DflashState::free_gpu`'s final-owner release).
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let DflashVmmLaneState {
            df,
            position: _,
            rng_state: _,
            sample_temp: _,
            sample_top_p: _,
            sample_top_k: _,
            sample_cactus: _,
            checkpoints,
            policy: _,
            last_window: _,
            picks: _,
            sampled_target: _,
            sample_scratch: _,
            sampled_draft: _,
            max_emit: _,
            verified: _,
            last_emit: _,
            last_history_len: _,
        } = self;
        df.free_gpu(gpu);
        for (_, snap) in checkpoints {
            snap.free_gpu(gpu);
        }
    }
}

/// Greedy chain draft of one lane window of `b` rows at `position` with
/// pending `seed`: exactly the singleton `spec_step_dflash` greedy fast-path
/// call (`ctx_slice=None`, dense target so no draft FFN graph). Mutates the
/// lane's draft scratch (projection watermarks, draft K/V) as the singleton
/// draft does. Returns `[seed, candidates..]` (`b` tokens).
pub fn dflash_lane_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    compact_offset: i32,
    df: &mut DflashState,
    position: usize,
    seed: u32,
    b: usize,
) -> HipResult<Vec<u32>> {
    if config.num_experts > 0 {
        return Err(hip_bridge::HipError::new(
            0,
            "dflash_lane_draft: MoE targets use the draft FFN graph route (not reproduced)",
        ));
    }
    if b < 2 {
        return Err(hip_bridge::HipError::new(0, "dflash_lane_draft: block size must be >= 2"));
    }
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let mut block: Vec<u32> = vec![df.draft_config.mask_token_id; b];
    block[0] = seed;
    draft_dflash_block_rank(
        gpu,
        weights,
        &df.draft_weights,
        &df.draft_config,
        &mut df.draft_scratch,
        &df.verify_scratch,
        &df.target_hidden_host,
        &block,
        position,
        compact_offset,
        None,
        false,
        seed,
        config.vocab_size,
    )
}

/// Shared verify head: the singleton DFlash dispatcher (rotate + batched
/// lm-head + GPU argmax) over `vs.final_hidden[0..rows]`, then one packed i32
/// D2H. Logits land in `vs.logits[0..rows]`, picks in `vs.argmax[0..rows]`.
/// A head over 64+ rows (only a wide chunk reaches it) uses the exact
/// singleton-WMMA lm-head, whose rows equal the singleton head's; below 64
/// rows the product head is the singleton's own, except inside the PM FP8
/// verify scope ([`verify_dense_math`]) where every row count takes the
/// `*VerifyExact` head (FP8-routed). Call it inside the same scope as the
/// trunk forward it follows. Sampled callers do not consume these aggregate
/// logits: their private head runs at exactly the singleton's own window size.
pub fn dflash_cb_head_argmax(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    vs: &VerifyScratch,
    rows: usize,
) -> HipResult<Vec<u32>> {
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let final_hidden = vs.final_hidden.sub_offset(0, rows * config.dim);
    let math = if rows > MULTI_CHUNK_PRODUCT_MAX_ROWS { DenseBatchMath::SingletonWmma } else { verify_dense_math(gpu) };
    dflash_enqueue_verify_lm_head_argmax(
        gpu,
        &weights.output,
        &final_hidden,
        vs,
        rows,
        config.vocab_size,
        math,
    )?;
    dflash_download_verify_argmax(gpu, vs, rows)
}

/// Shared verify buffers of one DFlash CB engine: the trunk row scratch, ONE
/// head scratch (`final_hidden`/`logits`/`rot`/`argmax`, `draft_rows` rows,
/// `hidden_k = dim.next_power_of_two()` like a singleton lane's), and the
/// reusable chunk tables. Allocated once; a cycle allocates no GPU memory.
/// `max_rows` is the EFFECTIVE cap (63, or up to 128 on the wide verify route).
pub struct DflashCbScratch {
    max_rows: usize,
    /// Head-scratch rows (`>= max_rows`): see [`Self::draft_rows`].
    draft_rows: usize,
    trunk: MultiChunkScratch,
    head: VerifyScratch,
    rows: Vec<usize>,
    ranges: Vec<Range<usize>>,
    /// Diagnostic: also copy each lane's head rows (post-norm hidden, logits,
    /// argmax) into its own `df.verify_scratch` (the oracle's proof arrays).
    /// Greedy production leaves this off.
    pub keep_lane_rows: bool,
    /// Row scratch of the batched draft forward, allocated by
    /// [`Self::reserve_draft`] (or on the first draft call that needs it): the
    /// shared draft activations (lane draft state stays in each lane) and the
    /// ONE set of FWHT rotation buffers every VMM lane borrows
    /// ([`Self::lend_mq`]).
    draft: Option<DflashScratch>,
    /// Fewest lanes a draft chunk needs to take the batched forward (a lone
    /// lane runs the singleton call; the oracle sets 1 to prove the batched
    /// forward at one lane).
    pub draft_min_lanes: usize,
    /// `(lanes, chunks)` the last [`dflash_cb_draft`] call drafted through the
    /// batched forward (the rest ran the singleton call); evidence for the
    /// oracle that the batched path, not its fallback, produced the drafts.
    pub draft_stats: (usize, usize),
}

impl DflashCbScratch {
    /// `max_rows` is the planned rows; the scratch allocates this target's
    /// effective cap ([`multi_chunk_scratch_rows`]: `2..=63`, or up to 128
    /// only when the wide verify route is admitted for `weights`/`config`).
    /// The scratch stores that cap ([`Self::max_rows`]). The head scratch holds
    /// `max(cap, min(planned, dflash_draft_batch_max_rows(gpu)))` rows so a
    /// 64-row batched draft chunk fits even when the verify cap is 63
    /// ([`Self::draft_rows`]); with the 48-row draft cap this equals the cap.
    pub fn new(gpu: &mut Gpu, weights: &Qwen35Weights, config: &Qwen35Config, planned_rows: usize) -> HipResult<Self> {
        let max_rows = multi_chunk_scratch_rows(gpu, weights, config, planned_rows);
        let draft_rows = max_rows.max(planned_rows.min(dflash_draft_batch_max_rows(gpu)));
        let trunk = MultiChunkScratch::new(gpu, config, max_rows)?;
        let head = match VerifyScratch::new(gpu, draft_rows, config.dim, config.vocab_size, config.dim.next_power_of_two()) {
            Ok(h) => h,
            Err(e) => {
                let _ = trunk.free_gpu(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            max_rows,
            draft_rows,
            trunk,
            head,
            rows: Vec::with_capacity(max_rows),
            ranges: Vec::with_capacity(max_rows),
            keep_lane_rows: false,
            draft: None,
            draft_min_lanes: 2,
            draft_stats: (0, 0),
        })
    }

    /// Allocate the batched-draft row scratch now (idempotent), so a serving
    /// engine never allocates inside a step; `dflash_cb_draft` otherwise
    /// allocates it on the first batched chunk. The scratch also OWNS the
    /// FWHT rotation buffers every lane borrows ([`Self::lend_mq`]).
    pub fn reserve_draft(&mut self, gpu: &mut Gpu, draft_config: &hipfire_runtime::dflash::DflashConfig) -> HipResult<()> {
        if self.draft.is_none() {
            self.draft = Some(DflashScratch::new_with_mq(gpu, draft_config, dflash_draft_batch_max_rows(gpu), 1, true)?);
        }
        Ok(())
    }

    /// Lend the shared FWHT rotation buffers to `lane` (a lane draft scratch
    /// allocated without them) so a lane-local draft call can run its
    /// `gemm_dispatch`es. `Ok(false)`: the lane owns its own buffers (a
    /// singleton-origin scratch), nothing moved. `Ok(true)`: both buffers
    /// moved from the shared scratch into the lane; the caller MUST call
    /// [`Self::reclaim_mq`] with the returned flag on every path (success and
    /// error) before the lane or the engine is used again. `Err` when no
    /// shared draft scratch (or no rotation buffers in it) exists.
    ///
    /// Exact: the buffers have the same element count as a lane's own would
    /// (`MQ_X_ROT_CHUNK_ROWS × max(inter, q_dim, num_extract × hidden)`), so
    /// `gemm_dispatch` chunks identically, and each call rotates into them
    /// before its GEMM reads them (a per-call temporary, single stream).
    pub fn lend_mq(&mut self, lane: &mut DflashScratch) -> HipResult<bool> {
        if lane.mq_x_rot.is_some() {
            return Ok(false);
        }
        let shared = self
            .draft
            .as_mut()
            .ok_or_else(|| HipError::new(0, "lend_mq: no shared draft scratch to lend rotation buffers from"))?;
        if shared.mq_x_rot.is_none() || shared.mq_x_rot_f16.is_none() {
            return Err(HipError::new(0, "lend_mq: the shared draft scratch has no FWHT rotation buffers"));
        }
        lane.mq_x_rot = shared.mq_x_rot.take();
        lane.mq_x_rot_f16 = shared.mq_x_rot_f16.take();
        Ok(true)
    }

    /// Return buffers lent by [`Self::lend_mq`] (`lent` is its `Ok` value) to
    /// the shared scratch. No-op when nothing was lent.
    pub fn reclaim_mq(&mut self, lane: &mut DflashScratch, lent: bool) {
        if !lent {
            return;
        }
        if let Some(shared) = self.draft.as_mut() {
            shared.mq_x_rot = lane.mq_x_rot.take();
            shared.mq_x_rot_f16 = lane.mq_x_rot_f16.take();
        }
    }

    /// Rows one trunk chunk can hold (the effective cap: `<= 63`, or up to
    /// 128 on the wide verify route).
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Rows one batched draft chunk may hold: the head scratch's row count.
    /// `max_rows` unless the planned rows exceed it and the 64-row draft
    /// chunk ([`dflash_draft_batch_max_rows`]) is admitted, in which case
    /// the head (and so this cap) is widened to hold it.
    pub fn draft_rows(&self) -> usize {
        self.draft_rows
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> HipResult<()> {
        let DflashCbScratch {
            max_rows: _,
            draft_rows: _,
            trunk,
            head,
            rows: _,
            ranges: _,
            keep_lane_rows: _,
            draft,
            draft_min_lanes: _,
            draft_stats: _,
        } = self;
        if let Some(d) = draft {
            d.free_gpu(gpu);
        }
        let r = trunk.free_gpu(gpu);
        head.free_gpu(gpu);
        r
    }
}

/// One lane of [`dflash_cb_draft`]: its private state and the window to draft
/// (`b` rows at `position` with pending `seed`).
pub struct DflashCbDraftLane<'a> {
    pub state: &'a mut DflashVmmLaneState,
    pub position: usize,
    pub seed: u32,
    pub b: usize,
    pub compact_offset: i32,
}

/// Draft phase over lanes. With `batched` (and the exact-gfx1201 MQ4 v2
/// eligibility of [`dflash_draft_batch_eligible`]), greedy full-block lanes are
/// packed in order into chunks of at most
/// `min(dflash_draft_batch_max_rows(gpu), cb.draft_rows())` rows (48, or 64
/// on exact gfx1201 where the scoped 64-row draft route is exact); each chunk of at
/// least `cb.draft_min_lanes` lanes is ONE draft-model forward and ONE shared
/// lm-head ([`draft_dflash_blocks_batched`]) with every lane keeping its own
/// context rings, positions and hidden context. Every other lane (budget
/// tails, lone lanes, ineligible machines/drafts, `batched == false`) runs
/// the singleton call [`dflash_lane_draft`]. Sampled lanes always use the shared
/// singleton selector draft, retaining its genuine q receipt on the lane.
/// Returns each lane's `[seed, candidates..]`.
pub fn dflash_cb_draft(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    cb: &mut DflashCbScratch,
    lanes: &mut [DflashCbDraftLane<'_>],
    batched: bool,
) -> HipResult<Vec<Vec<u32>>> {
    let n = lanes.len();
    let mut out: Vec<Option<Vec<u32>>> = vec![None; n];
    // chunk id per lane (None = singleton draft).
    let mut chunk_of: Vec<Option<usize>> = vec![None; n];
    let mut n_chunks = 0usize;
    if batched && n > 0 && dflash_draft_batch_eligible(gpu, weights, config, &lanes[0].state.df.draft_weights) {
        let cap = dflash_draft_batch_max_rows(gpu).min(cb.draft_rows);
        let mut rows = 0usize;
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (i, l) in lanes.iter().enumerate() {
            if l.state.sample_temp > 1e-6 || l.b < 3 || l.b != l.state.df.block_size || l.b > cap {
                continue;
            }
            if members.last().is_none() || rows + l.b > cap {
                members.push(Vec::new());
                rows = 0;
            }
            members.last_mut().unwrap().push(i);
            rows += l.b;
        }
        for m in members {
            if m.len() >= cb.draft_min_lanes.max(1) {
                for &i in &m {
                    chunk_of[i] = Some(n_chunks);
                }
                n_chunks += 1;
            }
        }
    }
    cb.draft_stats = (chunk_of.iter().flatten().count(), n_chunks);
    if n_chunks > 0 && cb.draft.is_none() {
        let cfg = lanes.iter().zip(&chunk_of).find(|(_, c)| c.is_some()).map(|(l, _)| l.state.df.draft_config.clone());
        if let Some(cfg) = cfg {
            cb.draft = Some(DflashScratch::new_with_mq(gpu, &cfg, dflash_draft_batch_max_rows(gpu), 1, true)?);
        }
    }
    for c in 0..n_chunks {
        let mut dl: Vec<DflashDraftLane<'_>> = Vec::new();
        let mut idx: Vec<usize> = Vec::new();
        for (i, l) in lanes.iter_mut().enumerate() {
            if chunk_of[i] == Some(c) {
                idx.push(i);
                dl.push(DflashDraftLane {
                    df: &mut l.state.df,
                    position: l.position,
                    seed: l.seed,
                    b: l.b,
                    compact_offset: l.compact_offset,
                });
            }
        }
        let DflashCbScratch { draft, head, .. } = &mut *cb;
        let shared = draft.as_mut().ok_or_else(|| HipError::new(0, "dflash_cb_draft: no draft batch scratch"))?;
        let toks = draft_dflash_blocks_batched(gpu, weights, config.vocab_size, shared, head, &mut dl)?;
        for (i, t) in idx.into_iter().zip(toks) {
            out[i] = Some(t);
        }
    }
    for (i, l) in lanes.iter_mut().enumerate() {
        if out[i].is_none() {
            // A VMM lane has no rotation buffers of its own: borrow the
            // shared scratch's for this call and hand them back on EVERY
            // path (no `?` between lend and reclaim).
            if cb.draft.is_none() && l.state.df.draft_scratch.mq_x_rot.is_none() {
                cb.reserve_draft(gpu, &l.state.df.draft_config)?;
            }
            let lent = cb.lend_mq(&mut l.state.df.draft_scratch)?;
            let r = if l.state.sample_temp > 1e-6 {
                let cfg = l.state.sample_cfg();
                match l.state.sampled_draft.as_mut() {
                    None => Err(HipError::new(0, "dflash_cb_draft: sampled receipt not reserved")),
                    Some(receipt) => dflash_sampled_selector_draft_into(
                        gpu, weights, &mut l.state.df, l.position, l.compact_offset,
                        l.seed, l.b, config.vocab_size, cfg, &mut l.state.rng_state, receipt,
                    )
                    .map(|()| receipt.block.clone()),
                }
            } else {
                dflash_lane_draft(
                    gpu,
                    weights,
                    config,
                    l.compact_offset,
                    &mut l.state.df,
                    l.position,
                    l.seed,
                    l.b,
                )
            };
            cb.reclaim_mq(&mut l.state.df.draft_scratch, lent);
            out[i] = Some(r?);
        }
    }
    Ok(out.into_iter().map(|t| t.expect("every lane drafted")).collect())
}

/// One drafted lane between verify and accept: its trunk owners, its private
/// DFlash state and the window it drafted.
pub struct DflashCbVerifyLane<'a> {
    pub kv_cache: &'a mut KvCache,
    pub dn_state: &'a mut DeltaNetState,
    pub state: &'a mut DflashVmmLaneState,
    pub draft: &'a DflashCbDraft,
}

/// Verify phase over drafted lanes: the pre-verify DeltaNet state of each
/// lane is preserved (right after its draft, as the singleton) either by a
/// SAVE copy into its `target_snap` or, on an admitted chunk, by the
/// save-free owner exchange ([`cb_dn_save_free_admitted`]); then whole lanes
/// in order
/// through `forward_prefill_batch_multi` chunks of at most the packing cap
/// ([`multi_chunk_pack_cap`] of `cb.max_rows()`: C1 `[16]`, C3 `[48]`, C4
/// `[48,16]`, C8 `[48,48,32]` at 63 rows; C8 `[128]` with the wide route on a
/// dense MQ4G256V2 gfx1201 target) with each lane's own ChainVerify fusion,
/// hidden ring and tape (offset 0). Greedy lanes share one head/argmax/D2H;
/// sampled lanes use their own singleton-b Product head and materialize all b
/// normalized target rows before scratch reuse. Outcomes remain lane-owned.
///
/// Everything refusable is checked before the first snapshot is saved (or
/// owner exchanged) or row launched (capture/recording, block sizes,
/// ring/tape capacity, batched-body eligibility). A later failure leaves the
/// lanes of executed chunks advanced (after an exchange, also the failing
/// chunk's lanes with their live bank unwritten): the caller poisons the
/// participants, and nothing here undoes an exchange.
pub fn dflash_cb_verify(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    cb: &mut DflashCbScratch,
    lanes: &mut [DflashCbVerifyLane<'_>],
) -> HipResult<()> {
    let refuse = |why: String| Err(HipError::new(0, &format!("dflash_cb_verify: {why}")));
    if lanes.is_empty() {
        return Ok(());
    }
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return refuse("graph capture / replay recording is not supported".into());
    }
    if config.num_experts > 0 {
        return refuse("MoE targets are not admitted".into());
    }
    // Effective packing cap: the scratch capacity, held to the product cap
    // unless the wide exact route is admitted for this target.
    let pack_cap = multi_chunk_pack_cap(gpu, weights, config, cb.max_rows);
    cb.rows.clear();
    for (i, lane) in lanes.iter().enumerate() {
        let b = lane.draft.verify_tokens.len();
        if lane.draft.verify_tokens.first() != Some(&lane.draft.seed) {
            return refuse(format!("lane {i}: window does not start with its seed"));
        }
        if b > pack_cap.min(MULTI_CHUNK_MAX_LANE_ROWS) {
            return refuse(format!("lane {i}: {b} rows > chunk capacity {pack_cap}"));
        }
        let df = &lane.state.df;
        if b > df.gdn_tape.max_n || b > df.hidden_rb.max_batch || b > df.hidden_rb.max_positions {
            return refuse(format!("lane {i}: {b} rows exceed its tape/ring staging"));
        }
        if !qwen35::prefill_batch_pbs_eligible(weights, config, lane.dn_state, b, gpu.arch.as_str(), true) {
            return refuse(format!("lane {i}: {b} rows not eligible for the batched prefill body"));
        }
        cb.rows.push(b);
    }
    pack_whole_lanes(&cb.rows, pack_cap, &mut cb.ranges).map_err(|e| HipError::new(0, &format!("dflash_cb_verify: {e}")))?;
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    let (dim, vocab) = (config.dim, config.vocab_size);
    for ci in 0..cb.ranges.len() {
        let range = cb.ranges[ci].clone();
        let rows: usize = cb.rows[range.clone()].iter().sum();
        // One decision per packed chunk, never per lane (no half admission).
        let save_free = cb_dn_save_free_admitted(gpu, config, &cb.trunk, &lanes[range.clone()]);
        if save_free {
            // Save-free DeltaNet verify (`HIPFIRE_CB_DN_SAVE_FREE`). Instead of
            // copying every lane's live state A into its snapshot B, exchange
            // the owners of the four state families, so the canonical live
            // `dn_state` owns the old snapshot bank and the canonical
            // `target_snap` owns A, the pre-window state. Verify then reads A
            // (the request's `dn_state_src`) and writes only the live bank:
            // the conv ring is copied A -> B (pure bytes) and the in-place conv
            // kernel runs on B, and `gated_delta_net_q8_fast_ml_from` is the
            // in-place recurrence with only the first-use LOAD addresses
            // changed, so every byte the live bank ends with is the byte the
            // in-place verify of a SAVEd lane leaves. The existing commit then
            // runs unchanged on the canonical fields: full accept keeps the
            // live bank, partial / zero accept replay snapshot A -> live bank
            // (D8), and A stays the pre-window state terminal repair restores.
            // One stream drain before the exchange (everything queued on the
            // active stream, incl. the previous step's commit, has finished
            // with the old owners). The exchange itself runs below, after the
            // last fallible step that precedes the forward, and is host-only
            // and infallible after the whole-chunk preflight.
            if let Some(stream) = gpu.active_stream.as_ref() {
                gpu.hip.stream_synchronize(stream)?;
            }
        } else {
            for lane in lanes[range.clone()].iter_mut() {
                lane.state.df.target_snap.save_from(lane.dn_state, gpu)?;
            }
        }
        // PM FP8 verify scope (`HIPFIRE_CB_VERIFY_FP8`): trunk + head run the one
        // E4M3 verify arithmetic at every row count, so this batched chunk equals
        // each lane's singleton verify (same admission: the lanes share target
        // and KV tier); closed on every path below.
        let prev = fp8_verify_open(gpu, weights, config, &*lanes[range.start].kv_cache)?;
        if save_free {
            for lane in lanes[range.clone()].iter_mut() {
                lane.state.df.target_snap.exchange_buffers(&mut *lane.dn_state);
            }
        }
        let fwd = {
            let mut reqs: Vec<MultiChunkRequest<'_>> = Vec::with_capacity(range.len());
            for lane in lanes[range.clone()].iter_mut() {
                let DflashCbVerifyLane { kv_cache, dn_state, state, draft } = lane;
                let df = &mut state.df;
                reqs.push(MultiChunkRequest {
                    tokens: &draft.verify_tokens,
                    start_pos: draft.position,
                    kv_cache: &mut **kv_cache,
                    dn_state: &mut **dn_state,
                    gdn_tape: Some(&df.gdn_tape),
                    fusion: DflashFusionCtx::ChainVerify,
                    hidden_rb: Some(&mut df.hidden_rb),
                    dn_state_src: if save_free { Some(&df.target_snap) } else { None },
                });
            }
            // Trunk row-byte identity at aggregates 63/64/65/96/128, chunk128
            // off/HIP/PM: cb/chunk128/STAGE4.md:39-58 (the aggregate probe
            // freezes verify.final_hidden as well as logits against singleton).
            // DFlash IU4 verify (this load's decision): the trunk runs inside the IU4 verify
            // scope, the same A4 producers + IU4 GEMMs as the singleton verify;
            // the heads below stay F16.
            let iu4 = gpu.iu4_verify_enter(dflash_iu4_verify_admitted(gpu, weights, config));
            let r = forward_prefill_batch_multi(gpu, weights, config, scratch, &cb.trunk, &mut reqs, Some(&cb.head.final_hidden));
            gpu.iu4_verify_exit(iu4);
            r
        };
        // Greedy retains the packed head. Sampled acceptance reads every float:
        // each sampled head must use Product with its singleton's own b.
        let any_greedy = lanes[range.clone()].iter().any(|l| l.state.sample_temp <= 1e-6);
        let picks = match fwd {
            Ok(()) if any_greedy => dflash_cb_head_argmax(gpu, weights, config, &cb.head, rows).map(Some),
            Ok(()) => Ok(None),
            Err(e) => Err(e),
        };
        // Per-lane heads run inside the FP8 scope too (the singleton's verify
        // head does); the scope is closed on every path.
        let heads = picks.and_then(|picks| {
            let mut row = 0usize;
            for lane in lanes[range].iter_mut() {
                let b = lane.draft.verify_tokens.len();
                let st = &mut *lane.state;
                st.picks.clear();
                if st.sample_temp > 1e-6 {
                    let cfg = st.sample_cfg();
                    let v = &st.df.verify_scratch;
                    gpu.memcpy_dtod_at_auto(&v.final_hidden.buf, 0, &cb.head.final_hidden.buf, row * dim * 4, b * dim * 4)?;
                    let hidden = v.final_hidden.sub_offset(0, b * dim);
                    dflash_enqueue_verify_lm_head_argmax(
                        gpu, &weights.output, &hidden, v, b, vocab, DenseBatchMath::Product,
                    )?;
                    st.picks = dflash_download_verify_argmax(gpu, v, b)?;
                    let logits = v.logits.sub_offset(0, b * vocab);
                    // Materialize all b rows now, before another chunk overwrites
                    // shared hidden scratch (and before next draft reuses v.logits).
                    dflash_target_rows_into(gpu, &logits, b, vocab, cfg, &mut st.sampled_target)?;
                } else {
                    st.picks.extend_from_slice(&picks.as_ref().expect("greedy head")[row..row + b]);
                    if cb.keep_lane_rows {
                        let v = &st.df.verify_scratch;
                        gpu.memcpy_dtod_at_auto(&v.final_hidden.buf, 0, &cb.head.final_hidden.buf, row * dim * 4, b * dim * 4)?;
                        gpu.memcpy_dtod_at_auto(&v.logits.buf, 0, &cb.head.logits.buf, row * vocab * 4, b * vocab * 4)?;
                        gpu.memcpy_dtod_at_auto(&v.argmax.buf, 0, &cb.head.argmax.buf, row * 4, b * 4)?;
                    }
                }
                st.verified = true;
                row += b;
            }
            Ok(())
        });
        gpu.fp8_verify_exit(prev);
        heads?;
    }
    Ok(())
}

/// Whole-chunk admission of the save-free DeltaNet verify for one packed chunk
/// (`lanes` = the chunk's lanes), checked BEFORE any owner is exchanged. All
/// of the following must hold, otherwise the chunk takes today's SAVE route
/// unchanged:
///
/// * `HIPFIRE_CB_DN_SAVE_FREE` on (`cb_dn_save_free_enabled`), exact gfx1201;
/// * the shared scratch armed its snapshot-source lane table
///   ([`MultiChunkScratch::gdn_from_ready`]);
/// * the lane-batched GDN admission's process half
///   ([`lane_gdn_batch_ready`]: `HIPFIRE_CB_LANE_GDN_BATCH` /
///   `HIPFIRE_CB_TAPE_COPY_BATCH`, `1..=CONV_LANES_MAX` lanes, exact gfx1201
///   head geometry, no capture / recording / G0 observation) and its chunk
///   half (Sequential, ChainVerify, tape per lane — all structural here —
///   and EQUAL row count `b` on every lane, each within its tape);
/// * `2 <= b <= 16` (the D8 replay's step bound, so every accept class can
///   replay from the snapshot);
/// * no enabled retained Redline controller and no existing replay graph
///   (`replay_graph_count() == 0`: a graph's fast launch bypasses the
///   table re-fingerprinting an owner exchange relies on);
/// * per lane: D8 on + the replay-ML route admitted for `b` steps + the
///   snapshot mirrors the state ([`GdnTape::replay_from_snapshot_admits`]),
///   the tape's snapshot-source tables armed
///   ([`GdnTape::snapshot_source_armed`]), all four families exact: Q8 + EF,
///   conv kernel dimension 4, the config's DeltaNet layer count, per-layer
///   equal-size banks each at least the exact logical extent (pool rounding
///   allowed) and HIP-owned buffers on both banks
///   ([`DeltaNetSnapshot::cb_verify_owned_exact`]), and an uncompacted Q8/fp8
///   KV cache (the batched forward's own admission, so it cannot refuse after
///   the exchange);
/// * the chunk's `2 * 4 * n_layers` owner pointers per lane are pairwise
///   distinct across all lanes and both banks.
///
/// Host-only: one small pointer vector, no device access.
pub(crate) fn cb_dn_save_free_admitted(
    gpu: &Gpu,
    config: &Qwen35Config,
    trunk: &MultiChunkScratch,
    lanes: &[DflashCbVerifyLane<'_>],
) -> bool {
    use rdna_compute::dflash_gdn_replay::DFLASH_GDN_REPLAY_MAX_STEPS;
    let Some(first) = lanes.first() else {
        return false;
    };
    let b = first.draft.verify_tokens.len();
    if !cb_dn_save_free_enabled()
        || !gpu.arch_caps.is_gfx1201()
        || !trunk.gdn_from_ready()
        || !lane_gdn_batch_ready(gpu, config, lanes.len())
        || gpu.replay.is_enabled()
        || gpu.graphs.replay_graph_count() != 0
        || !(2..=DFLASH_GDN_REPLAY_MAX_STEPS).contains(&b)
    {
        return false;
    }
    let mut ptrs: Vec<u64> = Vec::with_capacity(lanes.len() * 8 * config.n_layers);
    for lane in lanes {
        let df = &lane.state.df;
        let state: &DeltaNetState = &*lane.dn_state;
        let kv: &KvCache = &*lane.kv_cache;
        if lane.draft.verify_tokens.len() != b
            || b > df.gdn_tape.max_n
            || !df.gdn_tape.snapshot_source_armed()
            || !df.gdn_tape.replay_from_snapshot_admits(gpu, state, &df.target_snap, b)
            || !df.target_snap.cb_verify_owned_exact(state, config)
            || !(kv.quant_q8 || kv.quant_fp8)
            || kv.compact_offset != 0
        {
            return false;
        }
        df.target_snap.push_owner_ptrs(state, &mut ptrs);
    }
    ptrs.sort_unstable();
    ptrs.windows(2).all(|w| w[0] != w[1])
}

#[cfg(test)]
mod tests {
    use super::dflash_block_for_emit;

    #[test]
    fn block_follows_the_singleton_budget_rule() {
        // B = min(16, max(E, 2)); E = 1 still verifies two rows.
        let b: Vec<usize> = [1, 2, 3, 15, 16, 17, 1000].iter().map(|&e| dflash_block_for_emit(16, e)).collect();
        assert_eq!(b, vec![2, 2, 3, 15, 16, 16, 16]);
    }
}
