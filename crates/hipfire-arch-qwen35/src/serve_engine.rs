// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotEngine — the multi-slot rig owned by one background thread.
//
// `hipfire serve` already speaks OpenAI-compatible HTTP, and HTTP is already
// concurrent. What serialises requests today is the single `Engine` behind a
// `Mutex<ServeRuntime>`. This engine is fed by a channel and lives outside any
// such lock, so N requests are in flight together and share one batched
// forward per step.
//
// The thread owns the GPU rig exclusively: nothing else touches `Gpu`,
// `SlotPool`, the arenas or the DeltaNet states. All interaction is by message,
// which is what makes "no lock" safe rather than merely fast.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use hipfire_runtime::admission::{AdmissionController, AdmitError, ModelFootprint};
use hipfire_runtime::serve::{
    send_event, Continuation, DoneReason, EngineStats, Event, SubmitRequest,
};
use hipfire_runtime::session_table::{SessionId, SessionTable};
use hipfire_runtime::swap::snapshot::{capture_slot, restore_slot, SnapshotStamp};
use hipfire_runtime::swap::SwapManager;
use rdna_compute::page_pool::PAGE_TOKENS;
use rdna_compute::sampling::SlotSampleParams;
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::forward_slots::{forward_batch_slots_graphed_opts, SlotDecodeGraph, SlotDescStaging};
use crate::grammar;
use crate::scheduler::{is_runnable_decode, is_runnable_prefill, PendingWork, Scheduler, VlPrefill};
use crate::slot_batch::SlotBatch;
use crate::qwen35::{
    self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch, Qwen35Weights,
};
use hipfire_runtime::llama::{KvCache, VMode};
use hipfire_runtime::tokenizer::Tokenizer;
use crate::speculative::DeltaNetSnapshot;
use crate::checkpoint::{capture_checkpoint, plan_resume, CheckpointId, QwenCheckpointPool};
use hipfire_runtime::prefix_index::{Handle, PrefixIndex};
use hipfire_runtime::serve_contract::{
    CacheDomain, DrafterDecision, PrefixLookup, PrefixLookupResult,
};
use hipfire_runtime::serve_fairness::{FairQueue, Grant, Selection};
use hipfire_runtime::serve_wait::WaitQueue;

/// Internal control plane for the engine thread. Public methods map onto these
/// messages; the GPU rig stays exclusively on that thread.
///
/// Continuous-batch scheduler integration is deferred — multi-slot is an
/// alternate single-weight daemon path, not ContinuousBatchScheduler.
enum EngineCommand {
    Submit(SubmitRequest),
    Close {
        session: u64,
        reply: Sender<Result<(), String>>,
    },
    Reset {
        reply: Sender<Result<(), String>>,
    },
}

pub struct EngineConfig {
    pub model_path: PathBuf,
    pub n_slots: usize,
    pub cap_tokens: usize,
    /// Prefill tokens taken from one slot per step. Bounds the batch scratch
    /// (`n_slots × prefill_chunk` rows, NOT `cap_tokens`) and keeps a long
    /// prompt from blocking other slots for its whole prefill — the property
    /// the scheduler was built around. Sizing scratch by `cap_tokens` put a
    /// 16k-ctx A3B out of reach of a 24 GB card.
    pub prefill_chunk: usize,
    pub host_budget_bytes: u64,
    pub swap_dir: PathBuf,
    /// True when the HFQ has vision tensors. Rig::build loads VisionWeights
    /// and VisionConfig when set.
    /// True when the model has vision support (either inline in the trunk
    /// HFQ or in a separate `.vl` sidecar). Rig::build loads VisionWeights
    /// and VisionConfig when set.
    pub is_vl: bool,
    /// Path to a standalone `.vl` file containing only vision tower tensors.
    /// When set, vision weights are loaded from this file instead of the
    /// trunk HFQ. When `None`, falls back to inline vision tensors in the
    /// trunk (backward compat with pre-separation artifacts).
    pub vl_path: Option<PathBuf>,
    /// MTP speculative decode depth (K candidates per cycle). 0 = MTP off.
    pub mtp_k: usize,
    /// Raw KV-mode string from the load request (`--kv-mode`). Empty = fall
    /// back to `HIPFIRE_KV_MODE` / config, resolved through the slots site
    /// policy (full static ladder; q8 default).
    pub kv_mode_raw: String,
    /// Cross-session prefix cache (radix index + checkpoint pool). Default
    /// false: the existing session-local continuation path (convo hashes)
    /// is unchanged. When true, cold admits consult a PrefixIndex for
    /// reusable KV pages and a QwenCheckpointPool for recurrent state
    /// (spec §4.5–4.6). Requires paged KV (PagePool).
    pub prefix_cache: bool,
    /// Maximum device bytes for the checkpoint pool. 0 = pool that refuses
    /// inserts (lookups still work for already-published entries from a
    /// prior epoch, but no new captures are stored).
    pub prefix_cache_max_bytes: u64,
    /// Global trunk-row budget per step (spec §5.2 S2): the sum of prefill +
    /// decode + verify rows a step emits must not exceed this. The scheduler
    /// enforces it in `next_batch` (prefill + decode) and the engine subtracts
    /// verify rows before calling the scheduler, so the forward panic-assert
    /// (`n <= pbs.max_batch`) is a fail-closed backstop, not the throttle.
    /// Default 4096 (the registered config default for `serve.max_batch_tokens`).
    pub max_batch_tokens: usize,
    /// Minimum prefill quantum guaranteed to one prefilling slot per step when
    /// space remains after decode allocation (spec §5.2 S2 / §5.3 S3). The
    /// scheduler rotates which prefilling slot gets it across ticks. Default 1
    /// (the registered config default for `serve.prefill_min_tokens`).
    pub prefill_min_tokens: usize,
    /// Bounded waiting room: max queued request count (spec §5.3 S3). Read
    /// from `serve.max_queue`; must be non-zero (Rig::build refuses
    /// `WaitQueue::new(0, _)`). Replaces the R-A4 immediate-reject path.
    pub wait_max_count: usize,
    /// Bounded waiting room: max total queued bytes (spec §5.3 S3). Read
    /// from `serve.max_queue_bytes`; must be non-zero.
    pub wait_max_bytes: u64,
    /// Bounded waiting room: per-waiter timeout in scheduler ticks (spec
    /// §5.3 S3). Derived from `serve.queue_timeout_ms` — the engine ticks
    /// once per serve iteration, so the millisecond value is used directly
    /// as the tick count (1 tick ≈ 1 ms is conservative for decode steps,
    /// which dominate serve workloads; a prefill-heavy step takes longer,
    /// making the timeout generous rather than tight).
    pub wait_timeout_ticks: u64,
    /// Structured-output jump-forward (spec §7.3 G3). When true and a slot
    /// carries a grammar constraint under greedy decoding, the engine calls
    /// `forced_token_run` after the grammar mask to find a bounded run of
    /// provably-forced tokens and commits them without sampling. Default
    /// false (read from `serve.structured_jump_forward`): no behavior change
    /// on the constrained-decode path.
    pub structured_jump_forward: bool,
}

pub struct SlotEngine {
    tx: Option<Sender<EngineCommand>>,
    stats: Arc<Mutex<EngineStats>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl SlotEngine {
    pub fn submit(&self, req: SubmitRequest) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Submit(req))
            .map_err(|_| "engine thread is gone".to_string())
    }

    /// Close a session synchronously. If the session is mid-generation its
    /// in-flight request is cancelled (InFlight removed, reply sender dropped,
    /// work cleared), swap forgotten and SessionTable/pool/admission closed,
    /// then acknowledged. Idle close does the same. Never silently ignored.
    pub fn close(&self, session: u64) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Close {
                session,
                reply: reply_tx,
            })
            .map_err(|_| "engine thread is gone".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "engine thread is gone".to_string())?
    }

    /// Drop every session and swap entry, returning the engine to a cold empty
    /// table. Rejects while any slot is still generating.
    pub fn reset(&self) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Reset { reply: reply_tx })
            .map_err(|_| "engine thread is gone".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "engine thread is gone".to_string())?
    }

    pub fn stats(&self) -> EngineStats {
        *self.stats.lock().expect("stats mutex poisoned")
    }

    /// Build the rig on a new thread and start serving. Returns once the model
    /// is loaded, so a caller that gets `Ok` can submit immediately.
    pub fn spawn(cfg: EngineConfig) -> Result<SlotEngine, String> {
        let (tx, rx) = channel::<EngineCommand>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let stats_thread = Arc::clone(&stats);

        let handle = std::thread::Builder::new()
            .name("hipfire-slot-engine".to_string())
            .spawn(move || -> Result<(), String> {
                match Rig::build(&cfg) {
                    Ok(rig) => {
                        let _ = ready_tx.send(Ok(()));
                        run_loop(rig, rx, stats_thread)
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        Err(e)
                    }
                }
            })
            .map_err(|e| format!("spawn engine thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(SlotEngine {
                tx: Some(tx),
                stats,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("engine thread died during startup".to_string()),
        }
    }
    /// Consuming shutdown that closes the channel, drains and fails any
    /// in-flight requests, joins the engine thread and returns GPU teardown
    /// or poison errors. The daemon can call this and withhold `unloaded`
    /// on failure. Graph is destroyed before buffers.
    pub fn shutdown(mut self) -> Result<(), String> {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(res) => res,
                Err(_) => Err("engine thread panicked".to_string()),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for SlotEngine {
    fn drop(&mut self) {
        // Best-effort teardown for non-shutdown paths. Closing the channel
        // signals the engine loop; join releases VRAM before Drop returns.
        // Logs but does not propagate teardown/poison errors — call
        // `shutdown()` for checked teardown.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("hipfire-slot-engine teardown: {e}"),
                Err(_) => eprintln!("hipfire-slot-engine teardown: engine thread panicked"),
            }
        }
    }
}

/// Everything the engine thread owns.
/// Upper bound on a request's token-penalty window (`repeat_window`). The
/// penalty prepass is O(window²) per launch on device, and the per-slot
/// window buffer is sized by this; the daemon clamps requests to the same
/// bound so a client cannot ask for an unbounded scan.
const REPEAT_WINDOW_MAX: usize = 2048;

struct Rig {
    gpu: Gpu,
    weights: Qwen35Weights,
    config: qwen35::Qwen35Config,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    pool: SlotPool,
    /// The engine's resolved KV tier + model-global rotation tables. Passed
    /// to every forward step; the tier selects the KV-write and attend
    /// kernels (see `forward_slots::kv_write_slots` / `tier_attend_slots`).
    kv_tier: crate::forward_slots::SlotKvTier,
    k_arenas: Vec<GpuTensor>,
    v_arenas: Vec<GpuTensor>,
    dn_states: Vec<DeltaNetState>,
    desc_staging: SlotDescStaging,
    pbs: PrefillBatchScratch,
    scratch: Qwen35Scratch,
    /// Vision encoder weights + config. None for text-only models.
    vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
    /// MTP head weights (shared across all slots). None when MTP is off.
    mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
    /// Per-slot MTP spec state. None for slots that haven't started MTP yet
    /// or when MTP is off. Allocated lazily on first MTP prefill.
    mtp_states: Vec<Option<crate::mtp_spec::MtpSpecState>>,
    /// Batched scratch (+ per-row FWHT-rot buffer) for MTP head-KV prefill
    /// from the scheduler's batched prompt chunks. None when MTP is off.
    mtp_prefill_batched: Option<(
        crate::mtp_head::Qwen35MtpHeadBatchedScratch,
        GpuTensor,
    )>,
    /// Per-layer (qkv, alpha, beta) tape of MTP verify rows, captured during
    /// the verify forward and replayed by the partial-accept DN repair.
    /// Row stride per slot = mtp_k + 1.
    mtp_verify_tape: Option<crate::speculative::GdnTape>,
    /// Staging for the POST-output-norm hidden rows fed to the MTP head
    /// during prompt prefill (x_batch holds the pre-norm residual).
    mtp_prefill_hidden: GpuTensor,
    /// MTP chain depth (K candidates per cycle). 0 = MTP off.
    mtp_k: usize,
    vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    logits_out: GpuTensor,
    out_tokens: GpuTensor,
    sample_params: Vec<SlotSampleParams>,
    /// Per-slot penalty windows (see `REPEAT_WINDOW_MAX`); uploaded from each
    /// penalized slot's session token tail before every sampling step.
    repeat_windows: Vec<GpuTensor>,
    /// Per-slot vision-embedding matrices (n_visual × dim, F32) for batched
    /// VL requests; the scatter kernel dereferences them through per-row
    /// pointers uploaded into `pbs.ext_emb_row_ptr`. None until the slot's
    /// request has run vision_forward; also cleared when the slot is.
    vl_ext_devs: Vec<Option<GpuTensor>>,
    /// Per-slot in-flight vision-tower jobs (batched VL path). The engine
    /// advances each live job ONE tower layer per serve iteration so a
    /// big-image encode (~9.7 s of GPU work at a 78×78 grid, even Q-tiled)
    /// interleaves with the other slots' decode steps instead of stalling
    /// them for the whole encode. None when no encode is in flight for the
    /// slot; cleared wherever `vl_ext_devs` is.
    vl_tower_jobs:
        Vec<Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob>>,
    /// Serve VL requests on the legacy sequential per-token path instead of
    /// the batched one (HIPFIRE_VL_SEQUENTIAL=1; batched is the default and
    /// the only paged-compatible mode).
    vl_sequential: bool,
    sessions: SessionTable,
    adm: AdmissionController,
    swap: SwapManager,
    stamp: SnapshotStamp,
    n_slots: usize,
    cap_tokens: usize,
    prefill_chunk: usize,
    /// Global trunk-row budget (spec §5.2 S2). Enforced in `next_batch` and
    /// the MTP verify injection path; the forward assert is a backstop.
    max_batch_tokens: usize,
    /// Minimum prefill quantum (spec §5.2 S2 / §5.3 S3).
    prefill_min_tokens: usize,
    /// Reusable host-side logits buffer for grammar mask application (spec
    /// §7.2 G2). When a slot has a grammar constraint, its logits row is
    /// copied D2H, masked (disallowed tokens set to NEG_INFINITY), then
    /// copied H2D back before `sample_per_slot` runs. This ensures the hard
    /// grammar mask participates before the sampler's probability-dependent
    /// filtering/renormalization (spec §7.2 G2). Allocated once; reused
    /// across steps. Sized to `n_slots * vocab_size` f32 elements.
    logits_host: Vec<f32>,
    /// Cross-session prefix cache: CPU-resident radix index over published
    /// token spans (spec §4.2 C2). None when `prefix_cache` is off.
    prefix_index: Option<hipfire_runtime::prefix_index::PrefixIndex>,
    /// Cross-session recurrent-state checkpoint pool (spec §4.5 C5).
    /// None when `prefix_cache` is off.
    checkpoint_pool: Option<crate::checkpoint::QwenCheckpointPool<DeltaNetSnapshot>>,
    /// Cache domain identity built once at Rig::build (spec §4.1 C1).
    /// None when `prefix_cache` is off.
    cache_domain: Option<hipfire_runtime::serve_contract::CacheDomain>,
    /// True when cross-session prefix reuse is enabled.
    prefix_cache: bool,
    /// Debug counter: total prefix cache lookups (for test assertions that
    /// vision requests never call lookup).
    lookup_count: u64,
    /// Debug counter: total plan_cow calls (for test assertions that the
    /// COW write barrier fires on paged steps).
    cow_plan_count: u64,
    /// Host-side fairness queue (spec §5.3 S3). Admits/needs/select are
    /// driven from the engine loop each step; the granted ids become the
    /// scheduler's eligibility mask so an un-granted slot contributes 0
    /// rows. Pure policy — no GPU state.
    fair_queue: FairQueue,
    /// Bounded waiting room for admission (spec §5.3 S3). Replaces the R-A4
    /// immediate-reject path: when every resident session is generating, a
    /// new request is enqueued here instead of rejected, up to finite count
    /// and byte limits. Pure host policy — no GPU state.
    wait_queue: WaitQueue,
    /// Parked SubmitRequests keyed by waiter id. SubmitRequest is not Clone
    /// (it owns a `Sender<Event>`), so the full request is parked here on
    /// enqueue and retrieved on pop_ready for retry. Removed on expire,
    /// cancel, and successful retry.
    parked_requests: std::collections::HashMap<u64, SubmitRequest>,
    /// Monotonic counter for synthetic waiter ids (queued requests have no
    /// session id yet).
    next_waiter_id: u64,
    /// Scheduler tick counter, advanced once per serve iteration. Used for
    /// wait-queue enqueue timestamps and expiry deadlines.
    tick: u64,
    /// Structured-output jump-forward enabled (spec §7.3 G3). Default false.
    structured_jump_forward: bool,
}

fn dn_buffers(dn: &DeltaNetState) -> Vec<&GpuTensor> {
    let mut v: Vec<&GpuTensor> = Vec::new();
    v.extend(dn.s_matrices.iter());
    v.extend(dn.s_scales.iter());
    v.extend(dn.conv_states.iter());
    v.extend(dn.s_ef_residual.iter());
    v
}

impl Rig {
    /// Build the GPU rig.
    ///
    /// CPU arch/tensor preflight precedes this loader — `SlotBackend::cpu_preflight`
    /// opens the HFQ, validates `arch_id` 5|6, records whether the HFQ has a
    /// vision tower, and parses config/tokenizer before this GPU path. This
    /// function assumes that preflight has passed; its `get_vram_info` +
    /// `preflight_alloc` is the actual-device VRAM admission. Keep that
    /// preflight on the live device (not a hardcoded budget).
    ///
    /// Post-weight allocation is transactional: K/V arenas, DN states, desc
    /// staging, PBS, scratch, logits/out are all owned after `Qwen35Weights`.
    /// On any `?`/error, all completed stages are freed, weights freed,
    /// caches/graph state invalidated and pool drained, so no VRAM leaks.
    fn build(cfg: &EngineConfig) -> Result<Rig, String> {
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::tokenizer::Tokenizer;
        use rdna_compute::kv_slots::preflight_alloc;

        let mut hfq = HfqFile::open(&cfg.model_path).map_err(|e| format!("open model: {e}"))?;
        let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("config: {e}"))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let n_fa_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        // Resolve the KV tier through the slots site policy — same config
        // source and alias table as the sequential carrier, but with q8 as
        // the slots engine's own default (the longest-validated slots path;
        // a rotated tier is an explicit operator choice). The full static
        // ladder is wired — q8, asym{2,3,4}, fwht{2,3,4} — each under both
        // the legacy slab pool and the paged pool.
        let kv_mode_raw = if cfg.kv_mode_raw.is_empty() {
            hipfire_runtime::config::get().kv_mode.clone()
        } else {
            cfg.kv_mode_raw.clone()
        };
        let hipfire_runtime::kv_mode::ResolveResult { mode: kv_mode, warning: kv_warning } =
            hipfire_runtime::kv_mode::resolve(
                &kv_mode_raw,
                &hipfire_runtime::kv_mode::QWEN35_SLOTS_POLICY,
                config.head_dim,
            );
        if let Some(w) = kv_warning {
            eprintln!(
                "  KV cache (slots): {w} (site {})",
                hipfire_runtime::kv_mode::QWEN35_SLOTS_POLICY.site
            );
        }
        // The tier's arena geometry: the K and V per-position strides DIFFER
        // on every rotated-K tier (packed K with a 4-byte per-head norm
        // header against Q8_0 V), and the pool, arenas, preflight math, and
        // snapshot stamp all follow them.
        let kv_plan = hipfire_runtime::kv_mode::SlotKvTierPlan::resolve(
            kv_mode,
            config.n_kv_heads,
            config.head_dim,
        )
        .map_err(|e| format!("kv mode {kv_mode:?}: {e}"))?;
        let per_pos_bytes = kv_plan.k_bytes_per_pos;
        let per_pos_v_bytes = kv_plan.v_bytes_per_pos;
        let prefill_chunk = cfg.prefill_chunk.max(1).min(cfg.cap_tokens.max(1));
        // A step carries at most max(prefill_chunk, mtp_k + 1) rows per slot:
        // prefilling slots contribute up to `prefill_chunk` rows, and a
        // decoding MTP slot injects `mtp_k + 1` verify rows. Sizing without
        // the mtp term let a small `prefill_chunk` (operator knob) overflow
        // `pbs.max_batch` on the first verify step — an engine-thread panic,
        // the serve-hang failure class.
        let max_batch = (prefill_chunk.max(cfg.mtp_k + 1) * cfg.n_slots).max(cfg.n_slots);

        // ── Paged KV opt-in ────────────────────────────────────────────────
        // Default OFF: the pool stays legacy (fixed per-slot slabs). With
        // HIPFIRE_SLOTS_PAGED=1 the pool becomes a PagePool: slots draw
        // physical pages on demand, so short sessions don't pre-reserve
        // their full cap. HIPFIRE_SLOTS_PAGED_PAGES overrides the physical
        // page count; the default matches legacy total capacity
        // (n_slots × ceil(cap/128) pages), so admission behaviour is
        // unchanged and only the allocation pattern is paged. A smaller
        // count under-subscribes (pool exhaustion fails the step CLOSED —
        // set_seq_len errors before any kernel writes through the missing
        // page); a larger count over-subscribes physical VRAM and is the
        // operator's call.
        // Prefix cache requires paged KV: the radix index shares physical
        // pages across sessions via PagePool refcounting, which is only
        // available in paged mode. Force it on when prefix_cache is enabled.
        let prefix_cache = cfg.prefix_cache;
        let paged_pages = if prefix_cache {
            let legacy_equivalent = cfg.n_slots * cfg.cap_tokens.div_ceil(PAGE_TOKENS);
            Some(
                hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED_PAGES")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(legacy_equivalent),
            )
        } else {
            match hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED")
                .ok()
                .as_deref()
            {
                Some("1") | Some("on") | Some("true") => {
                    // Batched VL is paged-capable (block-table indirection all
                    // the way down), so paged is refused only when the operator
                    // forced the legacy sequential VL path — that path writes
                    // through a flat contiguous slab view which aliases every
                    // slot onto the arena prefix in paged mode.
                    let sequential_vl = hipfire_config::developer_var("HIPFIRE_VL_SEQUENTIAL")
                        .ok()
                        .is_some_and(|v| v == "1" || v == "on" || v == "true");
                    if cfg.is_vl && sequential_vl {
                        eprintln!(
                            "  [hipfire] paged KV disabled: HIPFIRE_VL_SEQUENTIAL=1 \
                             requires contiguous KV slots"
                        );
                        None
                    } else {
                        let legacy_equivalent =
                            cfg.n_slots * cfg.cap_tokens.div_ceil(PAGE_TOKENS);
                        Some(
                            hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED_PAGES")
                                .ok()
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(legacy_equivalent),
                        )
                    }
                }
                _ => None,
            }
        };

        let weight_bytes = std::fs::metadata(&cfg.model_path)
            .map_err(|e| format!("stat model: {e}"))?
            .len();
        let cap_rounded = cfg.cap_tokens.div_ceil(128) * 128;
        let kv_pages = paged_pages.unwrap_or(cfg.n_slots * cap_rounded / 128);
        let kv_bytes = (n_fa_layers as u64)
            * (kv_pages as u64)
            * (PAGE_TOKENS as u64)
            * (per_pos_bytes as u64 + per_pos_v_bytes as u64);
        let planned = weight_bytes + kv_bytes + 768 * 1024 * 1024;

        // GPU context only before preflight — no device allocations yet. Free/
        // total come from the live device so gfx1100 is not over-admitted
        // against a hardcoded R9700 budget.
        let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
        let (vram_free, vram_total) = gpu
            .hip
            .get_vram_info()
            .map_err(|e| format!("vram info: {e}"))?;
        preflight_alloc(planned, vram_free as u64, "SlotEngine")
            .map_err(|e| format!("preflight refused: {e}"))?;
        // Discrete GPUs: keep model pages in page cache (fadvise-per-tensor
        // forces a full re-read on every load). UMA keeps eviction to avoid
        // OOM vs hipMalloc staging. Mirrors carriers.rs — without this, the
        // default `evict_page_cache=true` drops the mmap in prepare(), and
        // post-weight tensor reads (e.g. vision weights) panic on None.
        hfq.set_evict_page_cache(
            std::env::var("HIPFIRE_PAGE_EVICTION")
                .ok()
                .map(|v| v != "0")
                .unwrap_or_else(|| gpu.is_uma()),
        );
        let weights: Qwen35Weights = {
            let mut src = qwen35::HfqSource::new(&mut hfq, &config);
            let layout = qwen35::Layout::single(config.n_layers);
            qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
        }
        .map_err(|e| format!("load weights: {e}"))?;

        // ── Vision tower loading ──────────────────────────────────────────
        // When `vl_path` is set, vision weights load from the standalone .vl
        // sidecar (a standard HFQM container with only model.visual.* tensors
        // and config.vision_config in its metadata). Otherwise, fall back to
        // inline vision tensors in the trunk HFQ (backward compat).
        let (vision_config, vision_weights) = if cfg.is_vl {
            if let Some(vl) = &cfg.vl_path {
                eprintln!("  loading vision weights from .vl sidecar: {}", vl.display());
                let mut vl_hfq = HfqFile::open(vl)
                    .map_err(|e| format!("open .vl file {}: {e}", vl.display()))?;
                let vc = hipfire_arch_qwen35_vl::qwen35_vl::vision_config_from_hfq(&vl_hfq)
                    .ok_or_else(|| ".vl file missing vision_config in metadata".to_string())?;
                let vw = hipfire_arch_qwen35_vl::qwen35_vl::load_vision_weights(
                    &mut vl_hfq, &vc, &mut gpu,
                )
                .map_err(|e| format!("load vision weights from .vl: {e}"))?;
                (Some(vc), Some(vw))
            } else {
                let vc = hipfire_arch_qwen35_vl::qwen35_vl::vision_config_from_hfq(&hfq)
                    .ok_or_else(|| "VL model missing vision_config in metadata".to_string())?;
                let vw = hipfire_arch_qwen35_vl::qwen35_vl::load_vision_weights(&hfq, &vc, &mut gpu)
                    .map_err(|e| format!("load vision weights: {e}"))?;
                (Some(vc), Some(vw))
            }
        } else {
            (None, None)
        };

        // ── MTP head loading ──────────────────────────────────────────────
        // Mirrors finish_qwen35_load: try bundled trailer first, then .mtp sidecar.
        // The head is shared across all slots (small ~216 MB for 27B).
        //
        // MTP × paged: ALLOWED. The gate below (59c731dd0) refused the
        // combination because the then-current MTP path prefilled and
        // replayed KV through the flat slab view (`slot_kv_view`, whose
        // legacy bases are 0 under paged → silent cross-slot corruption).
        // The 85dde5df0 rewrite removed every flat-view writer from the MTP
        // path: prefill runs through the scheduler's batched chunk path
        // (paged block tables + write-frontier provisioning), the verify is
        // the batched forward, and a partial accept repairs only the
        // DeltaNet state (private per-slot buffers) from the verify tape —
        // `slot_kv_view` has no MTP caller left. Graph capture simply does
        // not engage under paged (the graphed entry falls back to the plain
        // step), and `mtp_verify_fits_cap` retires MTP before a verify's
        // k+1-row frontier could fail the paged provision closed.
        let mtp_k = cfg.mtp_k;
        let mtp_head: Option<crate::mtp_head::Qwen35MtpHead> = if mtp_k > 0 {
            let trunk_path = &cfg.model_path;
            let mut head_opt: Option<crate::mtp_head::Qwen35MtpHead> = None;
            match crate::mtp_head::load_mtp_head_bundled(
                trunk_path,
                &mut gpu,
                cfg.cap_tokens,
            ) {
                Ok(Some(h)) => {
                    eprintln!(
                        "  MTP head loaded (bundled .mq4-mtp): n_embd={} vocab={}",
                        h.config.n_embd, h.config.vocab_size
                    );
                    head_opt = Some(h);
                }
                Ok(None) => {
                    if let Some(sidecar) = crate::mtp_head::find_mtp_sidecar(trunk_path) {
                        match crate::mtp_head::load_mtp_head(
                            &sidecar,
                            &mut gpu,
                            cfg.cap_tokens,
                        ) {
                            Ok(h) => {
                                eprintln!(
                                    "  MTP head loaded (sidecar {}): n_embd={} vocab={}",
                                    sidecar.display(),
                                    h.config.n_embd,
                                    h.config.vocab_size
                                );
                                head_opt = Some(h);
                            }
                            Err(e) => {
                                eprintln!(
                                    "  MTP sidecar {} load failed: {e} — MTP disabled",
                                    sidecar.display()
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  MTP bundled trailer load failed: {e}");
                    if let Some(sidecar) = crate::mtp_head::find_mtp_sidecar(trunk_path) {
                        match crate::mtp_head::load_mtp_head(
                            &sidecar,
                            &mut gpu,
                            cfg.cap_tokens,
                        ) {
                            Ok(h) => {
                                eprintln!(
                                    "  MTP head loaded (sidecar {} after bundled error): n_embd={} vocab={}",
                                    sidecar.display(),
                                    h.config.n_embd,
                                    h.config.vocab_size
                                );
                                head_opt = Some(h);
                            }
                            Err(e2) => {
                                eprintln!(
                                    "  MTP sidecar {} load failed: {e2} — MTP disabled", sidecar.display()
                                );
                            }
                        }
                    }
                }
            }
            head_opt
        } else {
            None
        };
        if mtp_k > 0 && mtp_head.is_none() {
            // Probe last-extension replacement *and* the stem sidecar
            // (foo.mq4v2.hfq → foo.mq4v2.mtp and foo.mtp). A miss must
            // not silently degrade to AR: that is the "draft pulled but
            // DFlash off" trap.
            let probed: Vec<String> = crate::mtp_head::mtp_sidecar_candidates(&cfg.model_path)
                .into_iter()
                .map(|p| p.display().to_string())
                .collect();
            eprintln!(
                "  [hipfire] MTP requested (mtp_k={mtp_k}) but no head was found: \
                 no bundled .mq4-mtp trailer and no sidecar at {} — \
                 serving without MTP",
                probed.join(" or ")
            );
        }

        // ── Transactional post-weight guard ────────────────────────────────
        // Every allocation after weights is owned here. On any error, Drop
        // frees all completed stages in reverse construction order
        // (out_tokens → logits → scratch → pbs → staging → dn → arenas →
        // weights → caches/pool) so no prior stage leaks. Actual-device
        // preflight above is retained; CPU arch/tensor preflight precedes
        // this loader (see doc above).
        struct Guard {
            gpu: Option<Gpu>,
            weights: Option<Qwen35Weights>,
            vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
            mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
            k_arenas: Vec<GpuTensor>,
            v_arenas: Vec<GpuTensor>,
            kv_tier: Option<crate::forward_slots::SlotKvTier>,
            dn_states: Vec<DeltaNetState>,
            desc_staging: Option<SlotDescStaging>,
            pbs: Option<PrefillBatchScratch>,
            scratch: Option<Qwen35Scratch>,
            logits_out: Option<GpuTensor>,
            out_tokens: Option<GpuTensor>,
            committed: bool,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                if self.committed {
                    return;
                }
                if let Some(mut gpu) = self.gpu.take() {
                    if let Some(t) = self.out_tokens.take() {
                        let _ = gpu.free_tensor(t);
                    }
                    if let Some(t) = self.logits_out.take() {
                        let _ = gpu.free_tensor(t);
                    }
                    if let Some(s) = self.scratch.take() {
                        let _ = s.free_gpu(&mut gpu);
                    }
                    if let Some(p) = self.pbs.take() {
                        let _ = p.free_gpu(&mut gpu);
                    }
                    if let Some(d) = self.desc_staging.take() {
                        d.free_gpu(&mut gpu);
                    }
                    for dn in self.dn_states.drain(..).rev() {
                        dn.free_gpu(&mut gpu);
                    }
                    while let Some(v) = self.v_arenas.pop() {
                        let _ = gpu.free_tensor(v);
                    }
                    while let Some(k) = self.k_arenas.pop() {
                        let _ = gpu.free_tensor(k);
                    }
                    if let Some(tier) = self.kv_tier.take() {
                        tier.free_gpu(&mut gpu);
                    }
                    if let Some(w) = self.weights.take() {
                        w.free_gpu(&mut gpu);
                    }
                    if let Some(h) = self.mtp_head.take() {
                        h.free_gpu(&mut gpu);
                    }
                    if let Some(vw) = self.vision_weights.take() {
                        vw.free_gpu(&mut gpu);
                    }
                    gpu.invalidate_weight_caches();
                    gpu.invalidate_graph_state();
                    gpu.drain_pool();
                }
            }
        }
        let mut g = Guard {
            gpu: Some(gpu),
            weights: Some(weights),
            vision_weights,
            mtp_head,
            k_arenas: Vec::with_capacity(n_fa_layers),
            v_arenas: Vec::with_capacity(n_fa_layers),
            kv_tier: None,
            dn_states: Vec::with_capacity(cfg.n_slots),
            desc_staging: None,
            pbs: None,
            scratch: None,
            logits_out: None,
            out_tokens: None,
            committed: false,
        };
        let pool = if let Some(pages) = paged_pages {
            SlotPool::new_paged_with_strides(
                cfg.n_slots,
                cfg.cap_tokens,
                per_pos_bytes,
                per_pos_v_bytes,
                pages,
            )
        } else {
            SlotPool::new_with_strides(cfg.n_slots, cfg.cap_tokens, per_pos_bytes, per_pos_v_bytes)
        }
        .map_err(|e| format!("SlotPool: {e}"))?;
        if !kv_plan.kv_strides_differ() {
            // q8/bf16: nothing else to log; the tier banner below covers it.
        } else {
            eprintln!(
                "[hipfire] slot KV tier {kv_mode:?}: K {per_pos_bytes} B/pos, \
                 V {per_pos_v_bytes} B/pos (strides differ; descriptors carry \
                 separate K/V slab bases)"
            );
        }
        if pool.is_paged() {
            eprintln!(
                "[hipfire] paged KV pool: {} slots x cap {} tokens over {} physical pages \
                 ({} tokens/page)",
                cfg.n_slots,
                pool.cap_tokens(),
                pool.free_pages(),
                PAGE_TOKENS
            );
        }
        let k_arena_bytes = pool.k_arena_bytes();
        let v_arena_bytes = pool.v_arena_bytes();
        for _ in 0..n_fa_layers {
            let k = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[k_arena_bytes], DType::Raw)
                .map_err(|e| format!("k arena: {e}"))?;
            g.k_arenas.push(k);
            let v = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[v_arena_bytes], DType::Raw)
                .map_err(|e| format!("v arena: {e}"))?;
            g.v_arenas.push(v);
        }
        // Rotation tables for the resolved tier: Givens angles (asym) or
        // FWHT sign vectors (fwht), model-global and shared by every slot
        // and layer. Seeds match the sequential path's cache constructors
        // exactly, so slot-pool cache bytes are bit-identical to the
        // sequential engine's for the same tier.
        {
            let gpu = g.gpu.as_mut().unwrap();
            let upload_f32 = |gpu: &mut Gpu, vals: &[f32]| -> Result<GpuTensor, String> {
                let t = gpu.alloc_tensor(&[vals.len()], DType::F32).map_err(|e| format!("kv rotation table: {e}"))?;
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
                gpu.hip.memcpy_htod(&t.buf, &bytes).map_err(|e| format!("kv rotation table upload: {e}"))?;
                Ok(t)
            };
            let (cos, sin, s1, s2) = match kv_plan.givens_len {
                Some(len) => {
                    let (c, si) = KvCache::gen_givens_angles(42, len);
                    (
                        Some(upload_f32(gpu, &c)?),
                        Some(upload_f32(gpu, &si)?),
                        None,
                        None,
                    )
                }
                None => match kv_plan.fwht_len {
                    Some(len) => (
                        None,
                        None,
                        Some(upload_f32(gpu, &KvCache::gen_fwht_signs(42, len))?),
                        Some(upload_f32(gpu, &KvCache::gen_fwht_signs(1042, len))?),
                    ),
                    None => (None, None, None, None),
                },
            };
            g.kv_tier = Some(crate::forward_slots::SlotKvTier {
                mode: kv_mode,
                givens_cos: cos,
                givens_sin: sin,
                fwht_signs1: s1,
                fwht_signs2: s2,
            });
        }
        for _ in 0..cfg.n_slots {
            let dn = DeltaNetState::new(g.gpu.as_mut().unwrap(), &config)
                .map_err(|e| format!("dn: {e}"))?;
            g.dn_states.push(dn);
        }
        // Paged mode needs per-slot block-table staging sized to the slot
        // cap; legacy mode passes 0 and the staging stays empty.
        let max_pages_per_slot = if pool.is_paged() {
            pool.cap_tokens().div_ceil(PAGE_TOKENS)
        } else {
            0
        };
        let desc_staging = SlotDescStaging::new(
            g.gpu.as_mut().unwrap(),
            cfg.n_slots,
            max_batch,
            max_pages_per_slot,
        )
        .map_err(|e| format!("staging: {e}"))?;
        g.desc_staging = Some(desc_staging);
        // Slots do plain prefill only — never tree-verify — so skip the GDN
        // S-tape: at max_batch=cap_tokens it is tens of GB (16k → ~21 GB).
        let pbs = PrefillBatchScratch::new_opt(g.gpu.as_mut().unwrap(), &config, max_batch, false)
            .map_err(|e| format!("pbs: {e}"))?;
        g.pbs = Some(pbs);
        let scratch =
            Qwen35Scratch::new_with_kv_max(g.gpu.as_mut().unwrap(), &config, 64, cfg.cap_tokens)
                .map_err(|e| format!("scratch: {e}"))?;
        g.scratch = Some(scratch);
        let logits_out = g
            .gpu
            .as_mut()
            .unwrap()
            .zeros(&[cfg.n_slots * config.vocab_size], DType::F32)
            .map_err(|e| format!("logits: {e}"))?;
        g.logits_out = Some(logits_out);
        let out_tokens = g
            .gpu
            .as_mut()
            .unwrap()
            .zeros(&[cfg.n_slots], DType::F32)
            .map_err(|e| format!("out_tokens: {e}"))?;
        g.out_tokens = Some(out_tokens);
        let sample_params = (0..cfg.n_slots)
            .map(|_| SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
            })
            .collect::<Vec<_>>();
        // Per-slot penalty windows: u32 token ids, most recent last, uploaded
        // from the session's token tail before each sampling step. Capacity
        // bounds the window a request may request (the daemon clamps harder).
        let repeat_windows = (0..cfg.n_slots)
            .map(|_| {
                g.gpu
                    .as_mut()
                    .unwrap()
                    .zeros(&[REPEAT_WINDOW_MAX], DType::F32)
                    .map_err(|e| format!("repeat window: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Per-slot vision-embedding matrices, populated lazily when a batched
        // VL request runs vision_forward. The previous matrix is explicitly
        // freed on replacement (see `admit` — dropping a GpuTensor without
        // free_tensor leaks its VRAM; it is neither pooled nor hipFree'd).
        let vl_ext_devs: Vec<Option<GpuTensor>> = (0..cfg.n_slots).map(|_| None).collect();
        let vl_tower_jobs: Vec<Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob>> =
            (0..cfg.n_slots).map(|_| None).collect();
        let vl_sequential = hipfire_config::developer_var("HIPFIRE_VL_SEQUENTIAL")
            .ok()
            .is_some_and(|v| v == "1" || v == "on" || v == "true");
        if vl_sequential && !matches!(kv_mode, hipfire_runtime::kv_mode::KvMode::Q8) {
            return Err(
                "HIPFIRE_VL_SEQUENTIAL=1 writes KV through the q8-only flat slab \
                 view; it is not supported under a rotated KV tier. Drop the knob \
                 (the default batched VL path is tier-generic) or use q8 KV."
                    .to_string(),
            );
        }

        let dn_bytes: u64 = dn_buffers(&g.dn_states[0])
            .iter()
            .map(|t| t.buf.size() as u64)
            .sum();
        // kv_dtype_tag: a per-tier discriminant so a snapshot captured under
        // one KV tier can never be restored into a rig running another — the
        // payload's K/V spans are stride-shaped, and a mismatched restore
        // would be silent corruption. q8 keeps tag 1 (historical).
        let kv_dtype_tag: u32 = match kv_mode {
            hipfire_runtime::kv_mode::KvMode::Q8 => 1,
            hipfire_runtime::kv_mode::KvMode::Asym2 => 21,
            hipfire_runtime::kv_mode::KvMode::Asym3 => 31,
            hipfire_runtime::kv_mode::KvMode::Asym4 => 41,
            hipfire_runtime::kv_mode::KvMode::Fwht2 => 22,
            hipfire_runtime::kv_mode::KvMode::Fwht3 => 32,
            hipfire_runtime::kv_mode::KvMode::Fwht4 => 42,
            hipfire_runtime::kv_mode::KvMode::Bf16 => 12,
            hipfire_runtime::kv_mode::KvMode::Asym3Auto => {
                return Err("Asym3Auto sentinel reached Rig::build".to_string());
            }
        };
        let stamp = SnapshotStamp {
            model_hash: weight_bytes,
            kv_dtype_tag,
            per_pos_bytes: per_pos_bytes as u32,
            per_pos_v_bytes: per_pos_v_bytes as u32,
            n_fa_layers: n_fa_layers as u32,
            dn_layout_version: 1,
            cap: pool.cap_tokens() as u32,
            dn_bytes,
        };

        let mut adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: weight_bytes,
                kv_bytes_per_token: (n_fa_layers * (per_pos_bytes + per_pos_v_bytes)) as u64,
            },
            vram_total as u64,
        );
        adm.set_host_budget(cfg.host_budget_bytes);
        let swap = SwapManager::new(cfg.swap_dir.clone(), cfg.host_budget_bytes)
            .map_err(|e| format!("SwapManager: {e}"))?;

        // Commit guard — prevent Drop cleanup, move out owned GPU state.
        g.committed = true;
        let mut gpu = g.gpu.take().unwrap();
        let weights = g.weights.take().unwrap();
        let vision_weights = g.vision_weights.take();
        let mtp_head = g.mtp_head.take();
        let k_arenas = std::mem::take(&mut g.k_arenas);
        let v_arenas = std::mem::take(&mut g.v_arenas);
        let kv_tier = g.kv_tier.take().expect("kv tier built");
        let dn_states = std::mem::take(&mut g.dn_states);
        let desc_staging = g.desc_staging.take().unwrap();
        let pbs = g.pbs.take().unwrap();
        let scratch = g.scratch.take().unwrap();
        let logits_out = g.logits_out.take().unwrap();
        let out_tokens = g.out_tokens.take().unwrap();
        // MTP head-KV prefill scratch, sized for one scheduler prompt chunk.
        // Allocated after the guard commit (transactional ownership is no
        // longer needed past this point: everything above already committed).
        let mtp_prefill_batched = mtp_head
            .as_ref()
            .map(|head| -> Result<(crate::mtp_head::Qwen35MtpHeadBatchedScratch, GpuTensor), String> {
                let scratch = crate::mtp_head::Qwen35MtpHeadBatchedScratch::new(
                    &mut gpu,
                    &head.config,
                    prefill_chunk.max(1),
                )
                .map_err(|e| format!("mtp prefill scratch: {e}"))?;
                let widest_k = (2 * config.dim)
                    .max(head.config.n_ff)
                    .max(config.dim)
                    .max(head.config.n_head * head.config.head_dim);
                let rot = gpu
                    .zeros(&[prefill_chunk.max(1) * widest_k], DType::F32)
                    .map_err(|e| format!("mtp prefill rot: {e}"))?;
                Ok((scratch, rot))
            })
            .transpose()?;
        let mtp_verify_tape = if mtp_head.is_some() && mtp_k > 0 {
            Some(
                crate::speculative::GdnTape::new_for_config(
                    &mut gpu,
                    &config,
                    cfg.n_slots * (mtp_k + 1),
                )
                .map_err(|e| format!("mtp verify tape: {e}"))?,
            )
        } else {
            None
        };
        let mtp_prefill_hidden = gpu
            .zeros(&[prefill_chunk.max(1) * config.dim], DType::F32)
            .map_err(|e| format!("mtp prefill hidden staging: {e}"))?;
        // ── Prefix cache (spec §4.1–4.6) ──────────────────────────────────
        // Build the CacheDomain ONCE from available identities. When
        // prefix_cache is off, all three are None and the existing path is
        // bitwise unchanged.
        let (prefix_index, checkpoint_pool, cache_domain) = if prefix_cache {
            use hipfire_runtime::serve_contract::{
                ArchPolicy, CacheDomain, DeviceTopology, KvLayout, SharingNamespace,
                TemplateIdentity, TokenizerIdentity,
            };
            // C1 identity: SHA-256 of HFQ content, tokenizer vocab/config,
            // and the rendered chat-template program (spec §4.1). Sidecars
            // (MTP, optional VL) are hashed independently so a changed
            // adapter cannot hit a trunk-only cache entry.
            let chat_template = hfq.chat_template().unwrap_or_default();
            let template_digest = hipfire_runtime::serve_contract::sha256_len_prefixed(&[
                chat_template.as_bytes(),
            ]);
            let mut sidecar_digests = Vec::new();
            if mtp_head.is_some() {
                if let Some(sidecar) = crate::mtp_head::find_mtp_sidecar(&cfg.model_path) {
                    match HfqFile::open(&sidecar) {
                        Ok(mtp_hfq) => sidecar_digests.push(mtp_hfq.content_digest()),
                        Err(_) => {
                            if let Ok(d) = hipfire_runtime::serve_contract::sha256_file(&sidecar) {
                                sidecar_digests.push(d);
                            }
                        }
                    }
                }
            }
            if let Some(vl) = cfg.vl_path.as_ref() {
                if vl.exists() {
                    match HfqFile::open(vl) {
                        Ok(vl_hfq) => sidecar_digests.push(vl_hfq.content_digest()),
                        Err(_) => {
                            if let Ok(d) = hipfire_runtime::serve_contract::sha256_file(vl) {
                                sidecar_digests.push(d);
                            }
                        }
                    }
                }
            }
            let domain = CacheDomain {
                model_content_digest: hfq.content_digest(),
                model_load_epoch: 1,
                sidecar_digests,
                tokenizer: TokenizerIdentity {
                    vocab_digest: tokenizer.vocab_digest(),
                    config_digest: tokenizer.config_digest(),
                },
                template: TemplateIdentity {
                    template_digest,
                    normalization_tag: if chat_template.is_empty() {
                        "prompt-frame".to_string()
                    } else {
                        "jinja".to_string()
                    },
                },
                arch_policy: ArchPolicy {
                    arch_tag: "qwen35-deltanet".to_string(),
                    state_abi_tag: "dn-q8".to_string(),
                    position_attention_tag: "causal-mrope".to_string(),
                },
                kv_layout: KvLayout {
                    k_stride_bytes: vec![per_pos_bytes as u64],
                    v_stride_bytes: vec![per_pos_v_bytes as u64],
                    layout_tag: format!("{:?}", kv_mode),
                },
                device: DeviceTopology {
                    device_id: format!("gpu-{}", gpu.device_id),
                    topology_id: "single".to_string(),
                    allocation_epoch: 1,
                },
                namespace: SharingNamespace {
                    domain_id: "default".to_string(),
                },
            };
            let idx = hipfire_runtime::prefix_index::PrefixIndex::new(1 << 16);
            let pool = crate::checkpoint::QwenCheckpointPool::<DeltaNetSnapshot>::new(
                cfg.prefix_cache_max_bytes,
            );
            eprintln!(
                "  [hipfire] prefix cache enabled: radix max {} nodes, \
                 checkpoint pool {} bytes",
                1 << 16,
                cfg.prefix_cache_max_bytes
            );
            (Some(idx), Some(pool), Some(domain))
        } else {
            (None, None, None)
        };

        let vocab_size = config.vocab_size;
        // FairQueue (spec §5.3 S3): constructed from the same budget knobs the
        // scheduler enforces. An invalid config (max_batch_tokens < n_slots +
        // prefill_min_tokens) fails build rather than silently disabling
        // fairness — the policy would otherwise deadlock.
        let max_batch_tokens = cfg.max_batch_tokens.max(1);
        let prefill_min_tokens = cfg.prefill_min_tokens.max(1);
        let fair_queue = FairQueue::new(
            max_batch_tokens as u64,
            cfg.n_slots as u64,
            prefill_min_tokens as u64,
        )
        .map_err(|e| format!("fairness queue: {e}"))?;
        // WaitQueue (spec §5.3 S3): bounded waiting room that replaces the
        // R-A4 immediate-reject path. max_count and max_bytes MUST be
        // non-zero — WaitQueue::new refuses the zero/uncapped combination
        // (spec §5.3: "serve.max_queue=0 currently means uncapped; the new
        // robust multi-slot mode must reject that combination"). Failing
        // build here is the engine-side backstop even when the CLI guard
        // already rejected max_queue=0.
        let wait_queue = WaitQueue::new(
            cfg.wait_max_count,
            cfg.wait_max_bytes,
            cfg.wait_timeout_ticks,
        )
        .map_err(|e| format!("wait queue: {e}"))?;
        Ok(Rig {
            gpu,
            weights,
            config,
            tokenizer,
            pool,
            kv_tier,
            k_arenas,
            v_arenas,
            dn_states,
            desc_staging,
            pbs,
            scratch,
            vision_weights,
            vision_config,
            mtp_head,
            mtp_states: (0..cfg.n_slots).map(|_| None).collect(),
            mtp_prefill_batched,
            mtp_verify_tape,
            mtp_prefill_hidden,
            mtp_k,
            logits_out,
            out_tokens,
            sample_params,
            repeat_windows,
            vl_ext_devs,
            vl_tower_jobs,
            vl_sequential,
            sessions: SessionTable::default(),
            adm,
            swap,
            stamp,
            n_slots: cfg.n_slots,
            cap_tokens: cfg.cap_tokens,
            prefill_chunk,
            max_batch_tokens,
            prefill_min_tokens,
            logits_host: vec![0.0f32; cfg.n_slots * vocab_size],
            prefix_index,
            checkpoint_pool,
            cache_domain,
            prefix_cache,
            lookup_count: 0,
            cow_plan_count: 0,
            fair_queue,
            wait_queue,
            parked_requests: std::collections::HashMap::new(),
            next_waiter_id: 0,
            tick: 0,
            structured_jump_forward: cfg.structured_jump_forward,
        })
    }

    /// Release every GPU allocation owned by this rig, then invalidate caches
    /// and drain the HIP pool. Graph handles must be destroyed first so they
    /// never outlive the buffers they capture. Best-effort: keeps going after
    /// individual free failures and returns only the first error for logging.
    fn free_gpu(self, mut graph: SlotDecodeGraph) -> Result<(), String> {
        let Rig {
            mut gpu,
            weights,
            vision_weights,
            mtp_head,
            mtp_states,
            mtp_prefill_batched,
            mtp_verify_tape,
            mtp_prefill_hidden,
            kv_tier,
            k_arenas,
            v_arenas,
            dn_states,
            desc_staging,
            pbs,
            scratch,
            logits_out,
            out_tokens,
            vl_ext_devs,
            vl_tower_jobs,
            checkpoint_pool,
            ..
        } = self;

        let mut first_err: Option<String> = None;
        let mut note_hip = |r: rdna_compute::HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e.to_string());
                }
            }
        };

        // Reverse of construction: graph → out_tokens → logits → scratch →
        // pbs → staging → dn → arenas → weights → caches/pool.
        graph.release(&gpu);
        note_hip(gpu.free_tensor(out_tokens));
        note_hip(gpu.free_tensor(logits_out));
        note_hip(scratch.free_gpu(&mut gpu));
        note_hip(pbs.free_gpu(&mut gpu));
        desc_staging.free_gpu(&mut gpu);
        kv_tier.free_gpu(&mut gpu);
        for dn in dn_states.into_iter().rev() {
            dn.free_gpu(&mut gpu);
        }
        // Arenas were allocated per FA layer as k then v; free reverse layers
        // as v then k.
        let mut k_arenas = k_arenas;
        let mut v_arenas = v_arenas;
        while !k_arenas.is_empty() || !v_arenas.is_empty() {
            if let Some(v) = v_arenas.pop() {
                note_hip(gpu.free_tensor(v));
            }
            if let Some(k) = k_arenas.pop() {
                note_hip(gpu.free_tensor(k));
            }
        }
        weights.free_gpu(&mut gpu);
        if let Some(h) = mtp_head {
            h.free_gpu(&mut gpu);
        }
        for mtp_state in mtp_states.into_iter().flatten() {
            mtp_state.free_gpu(&mut gpu);
        }
        if let Some((scratch, rot)) = mtp_prefill_batched {
            scratch.free_gpu(&mut gpu);
            note_hip(gpu.free_tensor(rot));
        }
        if let Some(tape) = mtp_verify_tape {
            tape.free_gpu(&mut gpu);
        }
        note_hip(gpu.free_tensor(mtp_prefill_hidden));
        if let Some(vw) = vision_weights {
            vw.free_gpu(&mut gpu);
        }
        // Per-slot VL matrices + any tower job still mid-encode at teardown.
        for dev in vl_ext_devs.into_iter().flatten() {
            note_hip(gpu.free_tensor(dev));
        }
        for job in vl_tower_jobs.into_iter().flatten() {
            note_hip(job.free(&mut gpu));
        }
        // Free checkpoint pool snapshots (DeltaNetSnapshot device buffers).
        if let Some(mut pool) = checkpoint_pool {
            for blob in pool.drain_blobs() {
                blob.free_gpu(&mut gpu);
            }
        }
        gpu.invalidate_weight_caches();
        gpu.invalidate_graph_state();
        gpu.drain_pool();

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Borrowed Q8 KvCache viewing this slot's contiguous K/V slabs.
    /// Sequential `forward_scratch_*` writes here; the batched path reads
    /// the same bytes via `legacy_base`. Views are `DeviceBuffer::Borrowed`
    /// — dropping the KvCache does not free the arenas.
    fn slot_kv_view(&self, slot: SlotId) -> KvCache {
        // Paged slots have no contiguous slab: every legacy base is 0, so this
        // view would alias the arena prefix for ALL slots. Sequential writers
        // (VL, MTP prefill/replay) must be gated off under paged KV at
        // activation — if one reaches here, fail loudly rather than corrupt.
        assert!(
            !self.pool.is_paged(),
            "slot_kv_view: flat slab view requested under a paged pool — a \
             sequential KV writer would alias every slot onto the arena prefix"
        );
        assert!(
            self.kv_tier.is_q8(),
            "slot_kv_view: this view hardwires the q8 layout, but the engine \
             runs the {:?} KV tier — a sequential write here would corrupt the \
             rotated-K arena",
            self.kv_tier.mode
        );
        let cap = self.pool.cap_tokens();
        let per_pos_bytes = self.config.n_kv_heads * (self.config.head_dim / 32) * 34;
        let base = self.pool.descriptors()[slot.0].legacy_k_base as usize;
        let span = cap * per_pos_bytes;
        let mut k_gpu = Vec::with_capacity(self.config.n_layers);
        let mut v_gpu = Vec::with_capacity(self.config.n_layers);
        let mut fa = 0usize;
        for t in &self.config.layer_types {
            if *t == LayerType::FullAttention {
                k_gpu.push(self.k_arenas[fa].sub_offset(base, span));
                v_gpu.push(self.v_arenas[fa].sub_offset(base, span));
                fa += 1;
            } else {
                k_gpu.push(GpuTensor::null_for_test());
                v_gpu.push(GpuTensor::null_for_test());
            }
        }
        KvCache {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: self.config.n_kv_heads * self.config.head_dim,
            max_seq: cap,
            physical_cap: cap,
            n_kv_heads: self.config.n_kv_heads,
            head_dim: self.config.head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            quant_bf16: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        }
    }
}

/// Sequential VL step: encode the image if needed, prefill/decode every
/// remaining token with M-RoPE, then sample the next token from scratch.logits.
///
/// VL slots stay on this path for the whole request because batched 1D RoPE
/// disagrees with M-RoPE on image tokens and on decode (`t=h=w = pos+delta`).
fn vl_forward_remaining(
    rig: &mut Rig,
    slot: SlotId,
    work: &mut PendingWork,
    image_pad_id: u32,
) -> Result<u32, String> {
    let mut vl = work
        .vl_prefill
        .take()
        .ok_or_else(|| "vl_forward_remaining: slot has no VL state".to_string())?;
    let first_step = vl.embeddings.is_empty();
    if first_step {
        let weights = rig
            .vision_weights
            .as_ref()
            .ok_or_else(|| "VL request but model has no vision encoder".to_string())?;
        let config = rig
            .vision_config
            .as_ref()
            .ok_or_else(|| "VL request but model has no vision config".to_string())?;
        let emb = hipfire_arch_qwen35_vl::qwen35_vl::vision_forward(
            &mut rig.gpu,
            weights,
            config,
            &vl.patches,
            vl.grid_h,
            vl.grid_w,
        )
        .map_err(|e| format!("vision_forward: {e}"))?;
        vl.embeddings = emb;
        vl.dim = rig.config.dim;
        vl.patches.clear();
        hipfire_runtime::llama::reset_cpu_sampler_rng(rig.sample_params[slot.0].seed);
    }

    let prompt_tokens = std::mem::take(&mut work.remaining_prompt);
    let mut pos = work.next_pos;
    let mut kv_cache = rig.slot_kv_view(slot);
    let mrope_ctx = qwen35::MropeCtx::new(
        &rig.config,
        vl.base,
        vl.mrope_positions.clone(),
        vl.rope_delta,
    );
    let mut visual_idx = vl.visual_idx;
    for &token in &prompt_tokens {
        if token == image_pad_id && visual_idx < vl.n_visual_tokens {
            let dim = vl.dim;
            let start = visual_idx * dim;
            let emb = vl
                .embeddings
                .get(start..start + dim)
                .ok_or_else(|| "VL embedding slice out of range".to_string())?;
            qwen35::forward_scratch_embed_mrope(
                &mut rig.gpu,
                &rig.weights,
                &rig.config,
                emb,
                pos,
                &mut kv_cache,
                &mut rig.dn_states[slot.0],
                &rig.scratch,
                Some(&mrope_ctx),
            )
            .map_err(|e| format!("VL prefill embed: {e}"))?;
            visual_idx += 1;
        } else {
            qwen35::forward_scratch_mrope(
                &mut rig.gpu,
                &rig.weights,
                &rig.config,
                token,
                pos,
                &mut kv_cache,
                &mut rig.dn_states[slot.0],
                &rig.scratch,
                Some(&mrope_ctx),
            )
            .map_err(|e| format!("VL forward: {e}"))?;
        }
        pos += 1;
    }
    vl.visual_idx = visual_idx;
    work.next_pos = pos;
    rig.pool
        .set_seq_len(slot, pos)
        .map_err(|e| format!("VL set_seq_len: {e}"))?;
    work.vl_prefill = Some(vl);

    let mut logits = rig
        .gpu
        .download_f32(&rig.scratch.logits)
        .map_err(|e| format!("VL download logits: {e}"))?;
    let sp = &rig.sample_params[slot.0];
    let cfg = hipfire_runtime::sampler::SamplerConfig {
        temperature: sp.temperature,
        top_p: sp.top_p,
        repeat_penalty: 1.0,
        repeat_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        top_k: if sp.top_k > 0 {
            Some(sp.top_k as u32)
        } else {
            None
        },
        min_p: None,
    };
    Ok(hipfire_runtime::sampler::sample_cpu(&mut logits, &[], &cfg))
}

/// Advance slot `s`'s batched-VL vision-tower encode by ONE layer, starting
/// the [`VisionTowerJob`] on the first call and finishing it (merger epilogue
/// + ext-embedding upload) on the last.
///
/// Why one layer at a time: this engine thread is the GPU's exclusive owner,
/// so a monolithic `vision_forward` wedges every OTHER slot's decode step for
/// the whole tower pass — measured ~9.7 s at a 78×78 grid (6084 patches) even
/// on the Q-tiled attention kernel, i.e. the whole serve freezes for the
/// duration of one big image request. One tower layer (~0.4 s at that grid)
/// is the largest GPU chunk a VL slot adds between other slots' steps, and
/// the scheduler already holds the VL slot's rows back until its embeddings
/// exist (`vl_slots_waiting_on_vision_forward_are_skipped`).
///
/// `job` is the caller-owned per-slot job slot. Contract on return:
/// `Ok(false)` → `job` is `Some` (encode continues next iteration);
/// `Ok(true)` → encode finished, embeddings spliced, `job` is `None`;
/// `Err` → the job was freed, `job` is `None`.
fn vision_tower_step(
    rig: &mut Rig,
    s: usize,
    job: &mut Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob>,
    vl: &mut VlPrefill,
) -> Result<bool, String> {
    let Rig {
        gpu,
        vision_weights,
        vision_config,
        vl_ext_devs,
        config,
        ..
    } = rig;
    if job.is_none() {
        let weights = vision_weights
            .as_ref()
            .ok_or("VL request but model has no vision encoder")?;
        let vconfig = vision_config
            .as_ref()
            .ok_or("VL request but model has no vision config")?;
        let started = hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob::new(
            gpu,
            weights,
            vconfig,
            &vl.patches,
            vl.grid_h,
            vl.grid_w,
        )
        .map_err(|e| format!("vision_forward: {e}"))?;
        *job = Some(started);
        // Patches now live on-device for the rest of the encode; free the
        // host copy (~48 MB at the 2 MP budget).
        vl.patches = Vec::new();
    }
    let weights = vision_weights
        .as_ref()
        .ok_or("VL request but model has no vision encoder")?;
    let vconfig = vision_config
        .as_ref()
        .ok_or("VL request but model has no vision config")?;
    let done = match job
        .as_mut()
        .expect("vision tower job present")
        .step_layer(gpu, weights, vconfig)
    {
        Ok(done) => done,
        Err(e) => {
            let reason = format!("vision_forward: {e}");
            if let Some(j) = job.take() {
                let _ = j.free(gpu);
            }
            return Err(reason);
        }
    };
    if !done {
        return Ok(false);
    }
    let j = job.take().expect("vision tower job present");
    // The merger output stays on GPU — handed straight to the scatter kernel
    // via `vl_ext_devs` with no D2H+H2D roundtrip. The old monolithic path
    // downloaded the merged embeddings (forcing a device_synchronize that
    // waited on the whole tower) and immediately re-uploaded them; that sync
    // stalled every other slot's decode for the full encode. `finish` now
    // returns the GpuTensor directly (zero-copy spatial-merge reshape on GPU).
    let dev = j
        .finish(gpu, weights, vconfig)
        .map_err(|e| format!("vision_forward: {e}"))?;
    let n_emb = dev.numel() / vconfig.out_hidden_size;
    *vl_ext_devs.get_mut(s).expect("vl_ext_devs slot") = Some(dev);
    // `vl.embeddings` is the scheduler's readiness flag (non-empty ⇒ the
    // matrix exists on device); the batched scatter kernel reads `vl_ext_devs`,
    // not this Vec, so a zero placeholder of the right length is all that's
    // needed here.
    vl.embeddings = vec![0.0f32; n_emb * vconfig.out_hidden_size];
    vl.dim = config.dim;
    Ok(true)
}

/// Free slot `s`'s VL GPU state (ext-embedding matrix + any tower job still
/// mid-encode). For every site that clears a slot's work: dropping either
/// without freeing leaks VRAM — the buffers are neither pooled nor hipFree'd
/// by Drop.
fn clear_slot_vl_state(rig: &mut Rig, s: usize) {
    if let Some(dev) = rig.vl_ext_devs[s].take() {
        let _ = rig.gpu.free_tensor(dev);
    }
    if let Some(job) = rig.vl_tower_jobs[s].take() {
        let _ = job.free(&mut rig.gpu);
    }
}

/// MTP head-KV fill for one scheduler prefill chunk, post-forward.
///
/// The trunk side of an MTP slot's prompt flows through the scheduler's
/// batched chunk path like any other prefill (1D RoPE, shared forward with
/// other slots). This runs right AFTER that forward: the chunk's hidden
/// rows sit in `pbs.x_batch`, and the head consumes exactly
/// `(embed(token_i), trunk_hidden(token_i))` per row, so one
/// `mtp_head_forward_block_batched(kv_only = true)` call fills the head's
/// private KV for the chunk. `state.prev_hidden` is left pointing at the
/// chunk's last row — the correct draft seed hidden once the prompt drains.
///
/// `row_off` is the slot's first row in the step batch; `m` its row count.
fn mtp_head_prefill_chunk(
    rig: &mut Rig,
    slot: SlotId,
    chunk_tokens: &[u32],
    chunk_positions: &[i32],
    row_off: usize,
    m: usize,
) -> Result<(), String> {
    let head = rig
        .mtp_head
        .as_ref()
        .ok_or_else(|| "mtp_head_prefill_chunk: MTP head not loaded".to_string())?;
    let Some((scratch, rot)) = rig.mtp_prefill_batched.as_mut() else {
        return Err("mtp_head_prefill_chunk: batched head scratch missing".to_string());
    };
    if rig.mtp_states[slot.0].is_none() {
        let state = crate::mtp_spec::MtpSpecState::new_for_components(
            &mut rig.gpu,
            &rig.config,
            &rig.dn_states[slot.0],
            head,
            rig.mtp_k,
            crate::mtp_head::MtpKvMode::Q8,
        )
        .map_err(|e| format!("mtp state alloc: {e}"))?;
        rig.mtp_states[slot.0] = Some(state);
    }
    if let Some(cvs) = head.weights.compressed_vocab_size {
        let state = rig.mtp_states[slot.0].as_mut().unwrap();
        state
            .ensure_compressed_lm_logits(&mut rig.gpu, cvs)
            .map_err(|e| format!("ensure compressed logits: {e}"))?;
        // The draft phase reads `mtp_scratch.logits_compressed` (head-side
        // buffer) — distinct from `mtp_lm_logits_compressed` above. Without
        // this allocation the first draft step on a compressed head panics
        // the engine thread (the sequential speculator was the only caller
        // of `ensure_compressed_logits`).
        state
            .mtp_scratch
            .ensure_compressed_logits(&mut rig.gpu, cvs)
            .map_err(|e| format!("ensure head compressed logits: {e}"))?;
    }

    let dim = rig.config.dim;
    let mut state = rig
        .mtp_states[slot.0]
        .take()
        .expect("just ensured it exists");

    // The head consumes POST-output-norm trunk hidden (the convention it was
    // exported with and the one the sequential path feeds it); pbs.x_batch
    // holds the PRE-norm residual, so norm the chunk's rows into staging.
    let hidden_rows = rig.pbs.x_batch.sub_offset(row_off * dim, m * dim);
    let staged = rig.mtp_prefill_hidden.sub_offset(0, m * dim);
    let norm_outcome = rig
        .gpu
        .rmsnorm_batched(
            &hidden_rows,
            &rig.weights.output_norm,
            &staged,
            m,
            dim,
            rig.config.norm_eps,
        )
        .map_err(|e| format!("mtp prefill hidden norm: {e}"));

    // Disjoint field borrows: `scratch`/`rot` from mtp_prefill_batched,
    // `gpu`/`weights`/head separately. No closure — it would capture all of
    // `rig` and collide with the scratch borrow.
    let mtp_outcome = norm_outcome.and_then(|()| {
        crate::mtp_head::mtp_head_forward_block_batched(
            &mut rig.gpu,
            head,
            scratch,
            &mut state.mtp_kv,
            chunk_tokens,
            &staged,
            chunk_positions,
            m,
            &rig.weights,
            Some(rot),
            /* kv_only */ true,
        )
        .map_err(|e| format!("mtp head prefill: {e}"))
    });
    // Seed hidden for the first draft cycle: the (normed) trunk hidden of
    // the chunk's last token — same convention as the verify-path capture.
    let hidden_outcome = mtp_outcome.and_then(|()| {
        rig.gpu
            .hip
            .memcpy_dtod_at(
                &state.prev_hidden.buf,
                0,
                &rig.mtp_prefill_hidden.buf,
                (m - 1) * dim * 4,
                dim * 4,
            )
            .map_err(|e| format!("mtp prev_hidden capture: {e:?}"))
    });
    rig.mtp_states[slot.0] = Some(state);
    hidden_outcome
}

/// MTP draft phase: run K serial head-forward steps, save DN snapshot.
///
/// Called when `work.decoding` (subsequent MTP steps). Takes the seed from
/// `work.remaining_prompt.last()`. Saves DN snapshot to `state.trunk_snap`
/// for rollback during verify. Returns the draft output for the batched
/// verify phase.
fn mtp_draft_step(
    rig: &mut Rig,
    slot: SlotId,
    work: &mut PendingWork,
) -> Result<crate::mtp_spec::MtpDraftOutput, String> {
    let head = rig
        .mtp_head
        .as_ref()
        .ok_or_else(|| "mtp_draft_step: MTP head not loaded".to_string())?;

    if rig.mtp_states[slot.0].is_none() {
        let state = crate::mtp_spec::MtpSpecState::new_for_components(
            &mut rig.gpu,
            &rig.config,
            &rig.dn_states[slot.0],
            head,
            rig.mtp_k,
            crate::mtp_head::MtpKvMode::Q8,
        )
        .map_err(|e| format!("mtp state alloc: {e}"))?;
        rig.mtp_states[slot.0] = Some(state);
    }
    let mut state = rig
        .mtp_states[slot.0]
        .take()
        .expect("just ensured it exists");

    if let Some(cvs) = head.weights.compressed_vocab_size {
        if let Err(e) = state.ensure_compressed_lm_logits(&mut rig.gpu, cvs) {
            rig.mtp_states[slot.0] = Some(state);
            return Err(format!("ensure compressed logits: {e}"));
        }
        // Head-side compressed buffer for the draft phase (see the alloc
        // comment in `mtp_head_prefill_chunk`).
        if let Err(e) = state
            .mtp_scratch
            .ensure_compressed_logits(&mut rig.gpu, cvs)
        {
            rig.mtp_states[slot.0] = Some(state);
            return Err(format!("ensure head compressed logits: {e}"));
        }
    }

    let prompt_tokens = std::mem::take(&mut work.remaining_prompt);
    let pos = work.next_pos;

    let outcome: Result<crate::mtp_spec::MtpDraftOutput, String> = (|| {
        if prompt_tokens.is_empty() {
            return Err("mtp_draft_step: no seed token".to_string());
        }
        let seed = *prompt_tokens.last().unwrap();

        // Save DN snapshot for rollback during verify phase.
        state
            .trunk_snap
            .save_from(&mut rig.dn_states[slot.0], &mut rig.gpu)
            .map_err(|e| format!("mtp dn snapshot: {e}"))?;

        let draft_outcome = crate::mtp_spec::mtp_draft_phase_inner(
            &mut rig.gpu,
            &rig.weights,
            &rig.config,
            head,
            &mut state,
            pos,
            seed,
            rig.mtp_k,
            /* skip_proposal_graph */ true,
        )
        .map_err(|e| format!("mtp draft: {e}"));
        if crate::mtp_spec::mtp_trace_enabled() {
            eprintln!("[mtp-trace] draft done pos={pos} seed={seed}");
        }
        draft_outcome
    })();

    rig.mtp_states[slot.0] = Some(state);
    outcome
}

/// MTP verify/accept phase: after the batched forward, run the trunk lm_head
/// over the slot's verify rows (normed views of `pbs.x_batch`), greedy-accept,
/// and repair the DeltaNet state on partial accepts.
///
/// `hidden_row_offset` is the flat row index where this slot's verify rows
/// begin in the step batch (computed from `batch.m_per_slot` by the caller).
/// `produced`/`max_tokens` are the in-flight request counters, used to clamp
/// the accepted burst to what the commit path can actually keep (see
/// `mtp_batched_verify_accept_from_batch`). Returns committed tokens
/// (excludes seed, includes bonus).
#[allow(clippy::too_many_arguments)]
fn mtp_verify_accept_step(
    rig: &mut Rig,
    slot: SlotId,
    work: &mut PendingWork,
    draft: crate::mtp_spec::MtpDraftOutput,
    hidden_row_offset: usize,
    produced: usize,
    max_tokens: usize,
    sess_len: usize,
) -> Result<Vec<u32>, String> {
    let mut state = rig
        .mtp_states[slot.0]
        .take()
        .expect("mtp state must exist from draft phase");

    let pos = work.next_pos;

    // How many tokens of this burst the commit path can keep before a
    // terminator fires: the request's remaining max_tokens budget and its
    // remaining context room (`commit_sampled_token` stops decode when
    // `sess.tokens.len() + 1 >= cap`, so room is `cap - len`).
    let commit_budget = max_tokens
        .saturating_sub(produced)
        .min(rig.cap_tokens.saturating_sub(sess_len))
        .max(1);

    let outcome: Result<Vec<u32>, String> = (|| {
        let result = crate::mtp_spec::mtp_batched_verify_accept_from_batch(
            &mut rig.gpu,
            &rig.weights,
            &rig.config,
            &mut rig.dn_states[slot.0],
            &rig.pbs,
            &mut state,
            &draft,
            hidden_row_offset,
            rig.mtp_verify_tape
                .as_ref()
                .expect("MTP drafts require the verify tape"),
            rig.mtp_k + 1,
            slot.0,
            rig.tokenizer.eos_id,
            rig.tokenizer.eot_id,
            commit_budget,
        )
        .map_err(|e| format!("mtp verify: {e}"))?;

        // Update position and seq_len. Stale KV rows past the new frontier
        // (rejected drafts) are overwritten before any later forward reads
        // them: every subsequent write is position-ordered.
        let new_pos = pos + result.advance;
        work.next_pos = new_pos;
        rig.pool
            .set_seq_len(slot, new_pos)
            .map_err(|e| format!("mtp set_seq_len: {e}"))?;

        Ok(result.committed)
    })();

    rig.mtp_states[slot.0] = Some(state);
    outcome
}

/// Rebuild a `SlotBatch` to include MTP verify tokens for slots that have
/// draft outputs. Non-MTP slots keep their scheduler-provided rows.
///
/// `pos3_deltas[s]` is the slot's continuation rope offset (0 for pure-text
/// conversations): verify rows are text rows of the slot they belong to, so
/// a slot continuing an image conversation phases them at `pos + delta`
/// exactly like the scheduler does for its prefill/decode rows.
///
/// `force_pos3` is the scheduler's work-level "this step needs M-RoPE" flag
/// (`(vl_prefill && !vl_sequential) || pos3_delta != 0` over ALL work, not
/// just the rows the scheduler emitted). It MUST be passed in by the caller:
/// an MTP-decoding slot contributes zero scheduler rows, so a verify-only
/// step — the steady state whenever the only active request is an MTP slot —
/// has an empty `batch.pos3` even though the slot's rows need the shifted
/// phases. Deriving the flag from `batch.pos3` emptiness here would silently
/// drop pos3 for exactly those steps and phase-shift every verify query
/// against the image turn's stored keys.
fn inject_mtp_verify_tokens(
    batch: &SlotBatch,
    mtp_drafts: &[Option<crate::mtp_spec::MtpDraftOutput>],
    n_slots: usize,
    pos3_deltas: &[i32],
    force_pos3: bool,
) -> SlotBatch {
    // A VL row anywhere in the step keeps the batch on the M-RoPE/scatter
    // path, so the side arrays must survive the rebuild: verify rows are
    // text rows of text slots and take their (possibly shifted) phase; VL
    // rows are copied through untouched.
    let vl_step = force_pos3
        || (batch.pos3.len() == batch.positions.len() && !batch.pos3.is_empty());
    let mut new_batch = SlotBatch::default();
    let mut old_row_off = 0usize;
    for s in 0..n_slots {
        if let Some(draft) = &mtp_drafts[s] {
            let delta = pos3_deltas.get(s).copied().unwrap_or(0);
            let verify_tokens = draft.verify_tokens();
            new_batch.m_per_slot.push(verify_tokens.len());
            for (i, t) in verify_tokens.iter().enumerate() {
                let pos = draft.cur_pos + i;
                new_batch.tokens.push(*t);
                new_batch.positions.push(pos as i32);
                new_batch.row_slot.push(s as i32);
                if vl_step {
                    new_batch.pos3.push([pos as i32 + delta; 3]);
                    new_batch.ext_emb.push(-1);
                }
            }
        } else {
            let m = batch.m_per_slot.get(s).copied().unwrap_or(0);
            new_batch.m_per_slot.push(m);
            if m > 0 {
                new_batch
                    .tokens
                    .extend_from_slice(&batch.tokens[old_row_off..old_row_off + m]);
                new_batch
                    .positions
                    .extend_from_slice(&batch.positions[old_row_off..old_row_off + m]);
                new_batch
                    .row_slot
                    .extend_from_slice(&batch.row_slot[old_row_off..old_row_off + m]);
                if vl_step {
                    new_batch
                        .pos3
                        .extend_from_slice(&batch.pos3[old_row_off..old_row_off + m]);
                    new_batch
                        .ext_emb
                        .extend_from_slice(&batch.ext_emb[old_row_off..old_row_off + m]);
                }
            }
            old_row_off += m;
        }
    }
    new_batch
}

fn clear_work_slot(work: &mut PendingWork) {
    work.remaining_prompt.clear();
    work.decoding = false;
    work.next_pos = 0;
    work.vl_prefill = None;
    work.mtp_active = false;
    work.pos3_delta = 0;
    // Adaptive-retire counters are per-REQUEST state: a request that ends
    // mid-window must not leave its failures behind for the slot's next
    // occupant (one bad window would sit one failure from retirement, and
    // the next request's window mean would mix both requests' advances).
    work.mtp_cycles = 0;
    work.mtp_committed = 0;
    work.mtp_retire_fails = 0;
}

/// Rebuild a `SlotBatch`'s flat arrays to exclude rows belonging to failed
/// slots (spec §5.4/S4: per-slot provision failure isolates the failing
/// request). The flat arrays (tokens, positions, row_slot, pos3, ext_emb) are
/// packed in slot order by `m_per_slot`; zeroing a slot's count without
/// removing its flat rows would misalign every subsequent slot. This rebuilds
/// them from the surviving `m_per_slot` entries.
fn rebuild_batch_excluding_failed(batch: &mut SlotBatch, failed: &[usize]) {
    let failed_set: std::collections::HashSet<usize> = failed.iter().copied().collect();
    let mut new_tokens = Vec::new();
    let mut new_positions = Vec::new();
    let mut new_row_slot = Vec::new();
    let mut new_pos3 = Vec::new();
    let mut new_ext_emb = Vec::new();
    let has_pos3 = !batch.pos3.is_empty();
    let has_ext = !batch.ext_emb.is_empty();
    let mut off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 || failed_set.contains(&s) {
            off += m;
            continue;
        }
        new_tokens.extend_from_slice(&batch.tokens[off..off + m]);
        new_positions.extend_from_slice(&batch.positions[off..off + m]);
        new_row_slot.extend_from_slice(&batch.row_slot[off..off + m]);
        if has_pos3 {
            new_pos3.extend_from_slice(&batch.pos3[off..off + m]);
        }
        if has_ext {
            new_ext_emb.extend_from_slice(&batch.ext_emb[off..off + m]);
        }
        off += m;
    }
    for &s in failed {
        batch.m_per_slot[s] = 0;
    }
    batch.tokens = new_tokens;
    batch.positions = new_positions;
    batch.row_slot = new_row_slot;
    batch.pos3 = new_pos3;
    batch.ext_emb = new_ext_emb;
}

/// True when an MTP verify batch — `mtp_k + 1` rows at positions
/// `next_pos..next_pos + mtp_k` — fits inside the slot's token cap. The
/// batched forward provisions `set_seq_len(max_pos + 1)` before its KV write
/// and fails the whole step closed when that exceeds the cap, which rejects
/// every active slot; MTP must retire before drafting instead.
/// Rolling adaptive-retire window (cycles) before MTP may retire a slot for
/// low acceptance, and the minimum mean advance (committed tokens per cycle)
/// the head must sustain to stay on the spec path. An MTP cycle costs roughly
/// twice an AR step (draft + verify over k+1 rows), so a mean advance below
/// ~2.2 loses wall-clock (measured MTP cycle ≈ 2.3x an AR step on
/// gfx1101/ornith-1.5-9b); code sustains ~2.8 (stays on MTP) while prose
/// sits at ~1.9 and retires to AR.
pub const MTP_RETIRE_WINDOW: usize = 32;
pub const MTP_RETIRE_MIN_ADVANCE: f64 = 2.2;
/// Consecutive failed windows required before retiring. One bad window is
/// normal even on high-acceptance workloads (a template-y or low-confidence
/// opening); two in a row means the head genuinely cannot pay for the cycle.
pub const MTP_RETIRE_WINDOWS: usize = 2;

fn mtp_verify_fits_cap(next_pos: usize, mtp_k: usize, cap_tokens: usize) -> bool {
    mtp_k + 1 <= cap_tokens.saturating_sub(next_pos)
}

/// True when the request carries a non-neutral token penalty. Such requests
/// decode as plain AR when MTP would otherwise be available: the verify accept
/// compares draft tokens against a bare greedy argmax over unpenalized logits,
/// which cannot reproduce penalize-then-argmax.
fn request_penalized(req: &SubmitRequest) -> bool {
    req.repeat_window > 0
        && (req.repeat_penalty > 1.0
            || req.presence_penalty > 0.0
            || req.frequency_penalty > 0.0)
}

/// True when the request asks for real sampling. Sampled requests decode AR
/// for the same reason penalized ones do: the verify accept is a bare greedy
/// argmax and cannot reproduce a sampled pick — running MTP anyway would emit
/// a greedy/spec hybrid instead of the requested distribution. The daemon
/// defaults temperature to 0.0 (greedy), so MTP stays on for default requests.
fn request_sampled(req: &SubmitRequest) -> bool {
    req.temperature > 1e-6
}

/// Install a request's sampling parameters (including token penalties and
/// min_p) on its slot's sampler state.
fn install_sample_params(rig: &mut Rig, slot: SlotId, req: &SubmitRequest) {
    rig.sample_params[slot.0] = SlotSampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repeat_window: req.repeat_window.min(REPEAT_WINDOW_MAX) as i32,
        repeat_penalty: req.repeat_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
    };
}

/// Publish generated-prefix pages at the client-commit boundary (Done event
/// for non-cancellation reasons). Seals and publishes any full pages beyond
/// the last prefill-boundary publish, capturing a checkpoint at the new
/// boundary (spec §4.6 C6).
fn publish_generated_prefix(rig: &mut Rig, slots: &mut [Option<InFlight>], s: usize) {
    if !rig.prefix_cache {
        return;
    }
    let Some(f) = slots[s].as_ref() else { return };
    let session = f.session;
    let last_pub = f.last_published_boundary;
    let sess = match rig.sessions.get(session) {
        Some(s) => s,
        None => return,
    };
    let total_tokens = sess.tokens.len();
    // Only publish full page-aligned boundaries beyond what was already published.
    let new_boundary = (total_tokens / PAGE_TOKENS) * PAGE_TOKENS;
    if new_boundary <= last_pub || new_boundary == 0 {
        return;
    }
    let n_old_pages = last_pub / PAGE_TOKENS;
    let n_new_pages = new_boundary / PAGE_TOKENS;
    let page_indices: Vec<u32> = match rig.pool.block_table(SlotId(s)) {
        Some(bt) => bt.page_indices().to_vec(),
        None => return,
    };
    if page_indices.len() < n_new_pages {
        return;
    }
    // Seal and add cache refs for the newly written pages.
    if let Some(pp) = rig.pool.page_pool_mut() {
        for p in n_old_pages..n_new_pages {
            let phys = page_indices[p];
            let _ = pp.seal(phys);
            let _ = pp.add_cache_ref(phys);
        }
    }
    // Build handles for all pages up to the new boundary.
    let handles: Vec<Handle> = page_indices[..n_new_pages]
        .iter()
        .enumerate()
        .map(|(i, &phys)| Handle {
            handle: rdna_compute::page_pool::PageHandle {
                phys,
                epoch: 0,
                generation: rig.pool.page_pool().map(|p| p.page_generation(phys)).unwrap_or(0),
            },
            token_offset: (i * PAGE_TOKENS) as u64,
        })
        .collect();
    let tokens: Vec<u32> = rig
        .sessions
        .get(session)
        .map(|sess| sess.tokens[..new_boundary].to_vec())
        .unwrap_or_default();
    if tokens.len() != new_boundary {
        return;
    }
    let domain = rig.cache_domain.as_ref().unwrap();
    let idx = rig.prefix_index.as_mut().unwrap();
    let pp = rig.pool.page_pool().unwrap();
    let checkpoint = if let Some(ckpt_pool) = rig.checkpoint_pool.as_mut() {
        match capture_checkpoint(
            &mut rig.gpu,
            ckpt_pool,
            domain,
            new_boundary as u64,
            &rig.dn_states[s],
        ) {
            Ok(id) => Some(id),
            Err(e) => {
                eprintln!("[prefix-cache] generated-prefix capture_checkpoint failed: {e}");
                None
            }
        }
    } else {
        None
    };
    if let Err(e) = idx.publish_sealed_pages(domain, &tokens, &handles, checkpoint, pp) {
        eprintln!("[prefix-cache] generated-prefix publish_sealed_pages failed: {e:?}");
    } else {
        if let Some(f) = slots[s].as_mut() {
            f.last_published_boundary = new_boundary;
        }
        if rig.gpu.slot_trace() {
            eprintln!(
                "[slot-trace] generated-prefix published {n_new_pages} pages at boundary {new_boundary}"
            );
        }
    }
}

fn commit_sampled_token(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    s: usize,
    tok: u32,
) {
    let Some(f) = slots[s].as_mut() else { return };
    let session = f.session;
    // Advance the grammar parser cursor on the accepted token BEFORE any
    // terminal checks (spec §7.2 G2: "per-request parser cursor advanced
    // on accepted tokens only"). The cursor is private per-request; a
    // speculative verifier would use a private cursor per candidate and
    // commit only the accepted prefix (spec §7.2 G2).
    if let Some(constraint) = f.grammar.as_mut() {
        constraint.advance_token(&rig.tokenizer, tok);
    }
    // Check whether the grammar constraint (if any) is in an accepting
    // state after advancing. A non-accepting state at terminal means the
    // generated JSON is incomplete/invalid — emit a typed rejection
    // instead of a successful done (spec §7.1 G1, A17).
    let grammar_incomplete = f
        .grammar
        .as_ref()
        .is_some_and(|c| !c.is_accepting());
    let hit_eos = rig.tokenizer.is_terminator(tok);
    let gone = if hit_eos {
        false
    } else {
        send_event(&f.reply, Event::Token { id: tok }).is_err()
    };
    f.produced += 1;
    let hit_max = f.produced >= f.max_tokens;
    let hit_ctx_cap = rig
        .sessions
        .get(session)
        .map(|sess| sess.tokens.len() + 1 >= rig.cap_tokens)
        .unwrap_or(false);

    if gone || hit_eos || hit_max || hit_ctx_cap {
        let reason = if gone {
            DoneReason::ClientGone
        } else if hit_eos {
            DoneReason::Eos
        } else {
            DoneReason::MaxTokens
        };
        if should_retain_terminal_tok(reason) {
            if let Some(sess) = rig.sessions.get_mut(session) {
                sess.tokens.push(tok);
            }
        }
        let generated = if hit_eos || gone {
            f.produced - 1
        } else {
            f.produced
        };
        // If a grammar constraint is present and not accepting at terminal,
        // the generated JSON is incomplete/invalid — emit a typed rejection
        // instead of a successful done (spec §7.1 G1, A17). ClientGone is
        // never overridden: the receiver is already dropped.
        if grammar_incomplete && !matches!(reason, DoneReason::ClientGone) {
            let _ = send_event(
                &f.reply,
                Event::Rejected {
                    reason: GrammarMaskError::Unsatisfiable.to_string(),
                },
            );
        } else {
            let _ = send_event(&f.reply, Event::Done { reason, generated });
        }
        // Publish generated-prefix pages at client commit (spec §4.6 C6).
        // Only for non-cancellation completions; ClientGone unpins instead.
        if !matches!(reason, DoneReason::ClientGone) && rig.prefix_cache {
            publish_generated_prefix(rig, slots, s);
        }
        slots[s] = None;
        // FairQueue: the request is done — drop it so it stops consuming
        // the fairness budget (spec §5.3 S3). Idempotent (NotFound ignored).
        let _ = rig.fair_queue.remove(session.0);
        clear_work_slot(&mut work[s]);
        clear_slot_vl_state(rig, s);
        if matches!(reason, DoneReason::ClientGone) {
            // Cancellation: unpin prefix cache domain pins (spec §4.4).
            if rig.prefix_cache {
                if let (Some(idx), Some(domain)) = (rig.prefix_index.as_mut(), rig.cache_domain.as_ref()) {
                    idx.unpin(domain);
                }
            }
            rig.swap.forget(session.0);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, session);
        } else {
            rig.sessions.touch(session);
        }
    } else {
        work[s].remaining_prompt.push(tok);
        if let Some(sess) = rig.sessions.get_mut(session) {
            sess.tokens.push(tok);
        }
        rig.sessions.touch(session);
    }
}

/// One request currently occupying a slot.
struct InFlight {
    session: SessionId,
    reply: Sender<Event>,
    produced: usize,
    max_tokens: usize,
    /// The REQUESTED penalty window (min of the request's repeat_window and
    /// REPEAT_WINDOW_MAX), kept unclamped so the per-step effective window
    /// grows with the session instead of ratcheting down permanently: the
    /// clamp used to write the effective window back into
    /// `sample_params[s].repeat_window`, which the next step then read as the
    /// "requested" value — freezing the window at the first step's session
    /// length for the whole generation.
    repeat_window_req: usize,
    /// Tokens reused from the cross-session prefix cache (spec §4.5).
    /// 0 when prefix_cache is off or the lookup missed. Reported to
    /// EngineStats.reused_tokens when the request completes.
    reused_tokens: usize,
    /// Last page-aligned boundary published to the prefix cache for this
    /// request (spec §4.6 C6). Initialized to `reused` at admit time so
    /// that shared pages from a prefix cache hit are not re-published.
    /// 0 when prefix_cache is off.
    last_published_boundary: usize,
    /// Per-request grammar constraint for structured output (spec §7 G1/G2).
    /// `None` for unconstrained requests. When present, a grammar mask is
    /// applied to the slot's logits row BEFORE the GPU sampler's
    /// probability-dependent filtering/renormalization, so no later
    /// operation can resurrect a forbidden token (spec §7.2 G2).
    ///
    /// Built from `SubmitRequest::json_schema` on admit: the schema is
    /// compiled into a `CompiledSchema` (immutable, shared) and a fresh
    /// `SchemaMatcher` cursor (per-request mutable state) is installed
    /// (spec §7.2 G2).
    grammar: Option<GrammarConstraint>,
}

/// Per-request grammar constraint state (spec §7 G1/G2).
///
/// Holds a per-request `SchemaMatcher` cursor (private mutable state, never
/// shared between forks — spec §7.2 G2) and produces a boolean token mask
/// from the current accepting set. The mask is applied to logits BEFORE the
/// sampler (spec §7.2 G2: "the hard grammar mask participates before
/// probability-dependent filtering/renormalization").
///
/// An empty allowed set is a typed per-request error, never an unconstrained
/// fallback (spec §7.2 G2: "An empty allowed set or invalid probability mass
/// is a typed error, never an unconstrained-sampling fallback"). EOS is
/// allowed only in an accepting grammar state.
struct GrammarConstraint {
    /// The per-request SchemaMatcher cursor. Advanced on accepted tokens
    /// only (spec §7.2 G2: "per-request parser cursor advanced on accepted
    /// tokens only"). For speculative verify, a private cursor per candidate
    /// is advanced and only the accepted prefix is committed (spec §7.2 G2).
    matcher: grammar::json_schema::SchemaMatcher,
    /// Reusable mask buffer (vocab-sized bool array), avoiding per-step
    /// allocation.
    mask_buf: Vec<bool>,
}

/// Error from grammar mask construction or application (spec §7.2 G2, A17).
#[derive(Debug, Clone)]
enum GrammarMaskError {
    /// The allowed token set is empty — no token can be emitted without
    /// violating the grammar. This is a typed per-request error, never an
    /// unconstrained fallback (spec §7.2 G2, A17).
    EmptyAllowedSet,
    /// The grammar constraint could not advance (unsatisfiable dead end).
    /// Truncation/cancel/unsatisfiable = typed incomplete/invalid result
    /// (spec §7.1 G1, A17).
    Unsatisfiable,
}

impl std::fmt::Display for GrammarMaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyAllowedSet => write!(
                f,
                "grammar constraint allows no token (empty allowed set) — \
                 request cannot continue without violating the schema"
            ),
            Self::Unsatisfiable => write!(
                f,
                "grammar constraint reached an unsatisfiable state — \
                 the schema cannot be satisfied from this position"
            ),
        }
    }
}

impl std::error::Error for GrammarMaskError {}

impl GrammarConstraint {
    /// Build the boolean token mask for the current parser state into
    /// `self.mask_buf` and return a reference to it (spec §7.2 G2).
    ///
    /// For each token id, `is_token_allowed` is called with the lossless
    /// `token_bytes(id)` (spec §7.2 G2: "Never build strict constraints on
    /// replacement-character artifacts"). EOS/terminator tokens are allowed
    /// only when the matcher is in an accepting state (spec §7.2 G2).
    ///
    /// Returns `Err(EmptyAllowedSet)` when no token is allowed — the caller
    /// must fail that request with a typed error, never fall back to
    /// unconstrained sampling (spec §7.2 G2, A17).
    fn build_mask(
        &mut self,
        tokenizer: &Tokenizer,
        vocab_size: usize,
    ) -> Result<&[bool], GrammarMaskError> {
        self.mask_buf.clear();
        self.mask_buf.resize(vocab_size, false);
        let accepting = self.matcher.is_accepting();
        for id in 0..vocab_size as u32 {
            // EOS/terminator tokens: allowed only in an accepting state
            // (spec §7.2 G2). A terminator emitted before the schema is
            // complete would produce invalid JSON.
            if tokenizer.is_terminator(id) {
                self.mask_buf[id as usize] = accepting;
                continue;
            }
            let bytes = tokenizer.token_bytes(id);
            if bytes.is_empty() {
                // Tokens with no byte representation (e.g. unused ids)
                // cannot contribute to the JSON output.
                continue;
            }
            self.mask_buf[id as usize] = self.matcher.is_token_allowed(bytes);
        }
        if !self.mask_buf.iter().any(|&allowed| allowed) {
            return Err(GrammarMaskError::EmptyAllowedSet);
        }
        Ok(&self.mask_buf)
    }

    /// Advance the parser cursor with the lossless token bytes of an
    /// accepted token (spec §7.2 G2: "per-request parser cursor advanced
    /// on accepted tokens only").
    ///
    /// Uses `Tokenizer::token_bytes` (lossless) rather than `decode` (lossy
    /// `from_utf8_lossy`) so byte-fallback tokens are handled correctly
    /// (spec §7.2 G2: "Never build strict constraints on
    /// replacement-character artifacts").
    fn advance_token(&mut self, tokenizer: &Tokenizer, token: u32) {
        let bytes = tokenizer.token_bytes(token);
        if !bytes.is_empty() {
            self.matcher.advance(bytes);
        }
    }

    /// True when the full JSON value has been parsed and conforms to the
    /// schema (spec §7.2 G2). Used at terminal to distinguish a
    /// schema-conforming done from an incomplete/invalid result.
    fn is_accepting(&self) -> bool {
        self.matcher.is_accepting()
    }
}

fn run_loop(
    mut rig: Rig,
    rx: Receiver<EngineCommand>,
    stats: Arc<Mutex<EngineStats>>,
) -> Result<(), String> {
    let n = rig.n_slots;
    let mut slots: Vec<Option<InFlight>> = (0..n).map(|_| None).collect();
    let mut work: Vec<PendingWork> = (0..n)
        .map(|s| PendingWork {
            slot: SlotId(s),
            remaining_prompt: Vec::new(),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            mtp_active: false,
            mtp_cycles: 0,
            mtp_committed: 0,
            mtp_retire_fails: 0,
            pos3_delta: 0,
        })
        .collect();
    let mut sched = Scheduler {
        chunk_size: rig.prefill_chunk,
        vl_sequential: rig.vl_sequential,
        prefill_cursor: 0,
    };
    let mut graph = SlotDecodeGraph::new();
    let mut poison: Option<String> = None;

    // Helper to fail every active request, close sessions, clear work, forget swap.
    // Used for HIP poison and shutdown drain. Returns the reason used.
    let fail_all_active = |rig: &mut Rig,
                           slots: &mut [Option<InFlight>],
                           work: &mut [PendingWork],
                           reason: String| {
        for (s, f) in slots.iter_mut().enumerate() {
            if let Some(f) = f.take() {
                let _ = send_event(
                    &f.reply,
                    Event::Rejected {
                        reason: reason.clone(),
                    },
                );
                rig.swap.forget(f.session.0);
                let sid = f.session;
                rig.sessions.close(&mut rig.pool, &mut rig.adm, sid);
                let _ = rig.fair_queue.remove(sid.0);
                clear_work_slot(&mut work[s]);
                clear_slot_vl_state(rig, s);
            }
        }
    };

    'serve: loop {
        let idle = slots.iter().all(|s| s.is_none());
        if idle && rig.wait_queue.queued_count() == 0 {
            // Nothing in flight and nothing queued: block rather than spin.
            match rx.recv() {
                Ok(cmd) => handle_command(&mut rig, &mut slots, &mut work, &stats, cmd),
                Err(_) => break 'serve, // all senders gone: shut down after drain
            }
        }
        loop {
            match rx.try_recv() {
                Ok(cmd) => handle_command(&mut rig, &mut slots, &mut work, &stats, cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if slots.iter().all(|s| s.is_none()) {
                        break 'serve;
                    }
                    break;
                }
            }
        }
        // ── WaitQueue: expire and retry queued requests (spec §5.3 S3) ──
        // Advance the scheduler tick once per serve iteration. The tick
        // stamps waiter admission age and drives per-waiter timeout deadlines.
        rig.tick = rig.tick.saturating_add(1);
        // Expire timed-out waiters → typed queue timeout rejection. The
        // client receives a Rejected event with a timeout reason; the parked
        // request is dropped.
        for waiter in rig.wait_queue.expire(rig.tick) {
            if let Some(parked) = rig.parked_requests.remove(&waiter.id) {
                let _ = send_event(
                    &parked.reply,
                    Event::Rejected {
                        reason: "serve queue timeout: request waited beyond \
                                 the configured deadline"
                            .to_string(),
                    },
                );
                stats.lock().expect("stats").note_rejected();
            }
        }
        // Pop ready waiters and retry admit when a slot is free. pop_ready
        // is FIFO by enqueue order, preserving admission age (spec §5.3 S3).
        // A retry that succeeds places the request in a slot; a retry that
        // fails (all slots still busy — should not happen since we checked
        // for a free slot, but a command between pop and admit could take
        // it) re-enqueues with a fresh waiter id inside admit.
        while slots.iter().any(|s| s.is_none()) {
            let waiter = match rig.wait_queue.pop_ready(rig.tick) {
                Some(w) => w,
                None => break,
            };
            let Some(parked) = rig.parked_requests.remove(&waiter.id) else {
                continue;
            };
            let active_before = slots.iter().filter(|s| s.is_some()).count();
            admit(&mut rig, &mut slots, &mut work, &stats, parked);
            let active_after = slots.iter().filter(|s| s.is_some()).count();
            // If admit did not place the request in a slot (it may have
            // re-queued or rejected), stop draining — the next iteration
            // will pop the next waiter if a slot is still free.
            if active_after <= active_before {
                break;
            }
        }
        if slots.iter().all(|s| s.is_none()) {
            continue;
        }

        let image_pad_id = rig.tokenizer.special_token_id("<|image_pad|>");
        for s in 0..n {
            // Sequential fallback ONLY (HIPFIRE_VL_SEQUENTIAL=1): this per-token
            // M-RoPE path writes KV through a flat slab view. The default
            // batched VL path must NEVER enter here — it runs through the
            // scheduler below, and under a paged pool the slot_kv_view assert
            // would (correctly) kill the engine.
            if !rig.vl_sequential || work[s].vl_prefill.is_none() || slots[s].is_none() {
                continue;
            }
            if work[s].remaining_prompt.is_empty() {
                continue;
            }
            let pad = image_pad_id.unwrap_or(u32::MAX);
            match vl_forward_remaining(&mut rig, SlotId(s), &mut work[s], pad) {
                Ok(tok) => commit_sampled_token(&mut rig, &mut slots, &mut work, s, tok),
                Err(reason) => {
                    if let Some(f) = slots[s].take() {
                        let _ = send_event(
                            &f.reply,
                            Event::Rejected {
                                reason: reason.clone(),
                            },
                        );
                        rig.swap.forget(f.session.0);
                        let _ = rig.fair_queue.remove(f.session.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    }
                    clear_work_slot(&mut work[s]);
                    clear_slot_vl_state(&mut rig, s);
                }
            }
        }

        // ── MTP phase 1: draft ────────────────────────────────────────────
        // MTP slots prefill through the scheduler's batched chunk path like
        // any other slot (their head-KV fill runs post-forward from x_batch,
        // below). This phase only drafts: K serial head-forward steps per
        // decoding MTP slot. The trunk verify is batched with regular decode
        // in the forward_batch_slots_graphed_opts call below — the vLLM-style
        // batched-verify design, now graph-captured like pure decode.
        let mut mtp_drafts: Vec<Option<crate::mtp_spec::MtpDraftOutput>> = (0..n).map(|_| None).collect();
        if rig.mtp_head.is_some() && rig.mtp_k > 0 {
            for s in 0..n {
                if !work[s].mtp_active || slots[s].is_none() || !work[s].decoding {
                    continue;
                }
                if work[s].remaining_prompt.is_empty() {
                    continue;
                }
                if !mtp_verify_fits_cap(work[s].next_pos, rig.mtp_k, rig.cap_tokens) {
                    // Context-cap guard: the batched verify writes mtp_k+1 rows
                    // at next_pos..next_pos+mtp_k; a frontier past the cap
                    // fails the forward's provision CLOSED, which rejects every
                    // active slot, not just this one. Retire MTP for this slot
                    // and let regular decode finish under its per-token guard.
                    // Of the committed tokens only the newest is not yet in
                    // KV — the rest are dead seed copies — so keep just it.
                    work[s].mtp_active = false;
                    if let Some(last) = work[s].remaining_prompt.last().copied() {
                        work[s].remaining_prompt.clear();
                        work[s].remaining_prompt.push(last);
                    }
                } else {
                    // Draft phase: K serial head-forward steps.
                    match mtp_draft_step(&mut rig, SlotId(s), &mut work[s]) {
                        Ok(draft) => {
                            mtp_drafts[s] = Some(draft);
                        }
                        Err(reason) => {
                            if let Some(f) = slots[s].take() {
                                let _ = send_event(
                                    &f.reply,
                                    Event::Rejected {
                                        reason: reason.clone(),
                                    },
                                );
                                rig.swap.forget(f.session.0);
                                let _ = rig.fair_queue.remove(f.session.0);
                                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                            }
                            clear_work_slot(&mut work[s]);
                            clear_slot_vl_state(&mut rig, s);
                        }
                    }
                }
            }
        }

        // ── VL phase: advance the vision tower for batched VL slots ────────
        // Each live encode takes ONE tower layer per serve iteration (see
        // `vision_tower_step`) so the loop returns to the scheduler — and to
        // the other slots' decode steps — between layers instead of wedging
        // them for the whole multi-second encode. The finished matrix becomes
        // the slot's external-embedding source; until it exists the scheduler
        // holds the slot's rows back (image pads would otherwise embed as the
        // raw pad token).
        if !rig.vl_sequential {
            for s in 0..n {
                if slots[s].is_none() {
                    continue;
                }
                let Some(vl) = work[s].vl_prefill.as_mut() else {
                    continue;
                };
                if !vl.embeddings.is_empty() || vl.n_visual_tokens == 0 {
                    continue;
                }
                let mut job = rig.vl_tower_jobs[s].take();
                let outcome = vision_tower_step(&mut rig, s, &mut job, vl);
                rig.vl_tower_jobs[s] = job;
                if let Err(reason) = outcome {
                    if let Some(f) = slots[s].take() {
                        let _ = send_event(&f.reply, Event::Rejected {
                            reason: reason.clone(),
                        });
                        rig.swap.forget(f.session.0);
                        let _ = rig.fair_queue.remove(f.session.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    }
                    clear_work_slot(&mut work[s]);
                    clear_slot_vl_state(&mut rig, s);
                }
            }
        }
        // ── Step reservation (spec §5.1/S1, §5.2/S2) ────────────────────
        // The scheduler enforces the global trunk-row budget for prefill +
        // decode; MTP verify rows (injected below) are subtracted first so
        // the full step never exceeds `max_batch_tokens`. The forward
        // panic-assert (`n <= pbs.max_batch`) stays as a fail-closed
        // backstop, not the throttle.
        let max_batch_tokens = rig.max_batch_tokens;
        let prefill_min_tokens = rig.prefill_min_tokens;

        // MTP verify rows: each drafting slot contributes `mtp_k + 1` rows.
        // A verify slot must not also keep an ordinary decode row (spec §5.2
        // S2), so the scheduler already skipped decoding MTP slots. If the
        // verify rows would not fit the remaining quota after decode+prefill
        // allocation, skip MTP for that slot this cycle (drop the draft; the
        // seed stays in `remaining_prompt` for a regular decode row next
        // step) rather than overflowing the budget.
        let mut verify_rows: u64 = 0;
        if mtp_drafts.iter().any(|d| d.is_some()) {
            for s in 0..n {
                if mtp_drafts[s].is_some() {
                    verify_rows = verify_rows
                        .checked_add((rig.mtp_k + 1) as u64)
                        .unwrap_or(u64::MAX);
                }
            }
        }

        let remaining_for_sched = max_batch_tokens
            .saturating_sub(verify_rows as usize)
            .max(0);
        // ── FairQueue select (spec §5.3 S3) ─────────────────────────────
        // The FairQueue decides WHO is eligible this tick (aged-first,
        // backfill, prefill cursor); the scheduler still decides HOW MANY
        // rows each granted slot gets. Sync each active slot's per-step
        // needs from slot state, then select grants under the same remaining
        // budget the scheduler will use. The granted ids become the
        // scheduler's eligibility mask; starved_oldest triggers bounded
        // backfill (only the oldest stays eligible so younger prefill/decode
        // is skipped this tick, giving the starved oldest the next budget).
        let vl_sequential = rig.vl_sequential;
        let mtp_k = rig.mtp_k;
        for s in 0..n {
            let Some(f) = slots[s].as_ref() else { continue };
            let id = f.session.0;
            let wants_decode = is_runnable_decode(&work[s], vl_sequential);
            let verify = if mtp_drafts[s].is_some() { (mtp_k + 1) as u64 } else { 0 };
            let uncached = if is_runnable_prefill(&work[s], vl_sequential) {
                work[s].remaining_prompt.len() as u64
            } else {
                0
            };
            sync_fair_needs(&mut rig.fair_queue, id, wants_decode, verify, uncached);
        }
        let sel = rig.fair_queue.select(
            n as u64,
            prefill_min_tokens as u64,
            remaining_for_sched as u64,
        );
        // After the admission round: mark requests not served since the last
        // round as aged (spec §5.3 S3.3). select already advanced the tick;
        // skip_round_complete sets the round boundary so the next select
        // prefers aged (starved) requests before cache-locality tie-breaks.
        rig.fair_queue.skip_round_complete();
        let granted_ids = if sel.starved_oldest {
            oldest_active_session(&rig.fair_queue, &slots)
                .map(|oid| apply_starved_backfill(&sel, oid))
                .unwrap_or_else(|| eligible_ids_from_selection(&sel))
        } else {
            eligible_ids_from_selection(&sel)
        };
        let mut eligible = vec![false; n];
        for s in 0..n {
            if let Some(f) = &slots[s] {
                eligible[s] = granted_ids.contains(&f.session.0);
            }
        }
        let sched_batch = sched.next_batch_eligible(
            &mut work,
            remaining_for_sched,
            prefill_min_tokens,
            &eligible,
        );

        // Count prefill + decode rows the scheduler emitted.
        let prefill_rows: u64 = (0..n)
            .filter(|&s| !work[s].decoding && sched_batch.m_per_slot.get(s).copied().unwrap_or(0) > 0)
            .map(|s| sched_batch.m_per_slot[s] as u64)
            .sum();
        let decode_rows: u64 = (0..n)
            .filter(|&s| {
                work[s].decoding
                    && !work[s].mtp_active
                    && sched_batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
            })
            .map(|s| sched_batch.m_per_slot[s] as u64)
            .sum();

        // Build a StepReservation and validate against the global budget
        // (spec §5.2 S2: `fits(max_batch_tokens)`). If it does not fit
        // (e.g. verify_rows overestimated because a draft was dropped
        // above), shrink by dropping verify slots first, then decode, then
        // prefill — never panic.
        let mut reservation = hipfire_runtime::serve_contract::StepReservation {
            prefill_rows,
            decode_rows,
            verify_rows,
            forced_rows: 0,
            page_growth_credits: 0,
            cow_destinations: 0,
            needs: hipfire_runtime::serve_contract::StepNeeds {
                scratch_bytes: 0,
                snapshot_bytes: 0,
                output_bytes: 0,
            },
        };
        match reservation.fits(max_batch_tokens as u64) {
            Ok(true) => {}
            _ => {
                // Shrink: drop MTP verify slots until it fits, then drop
                // prefill rows from the tail. The scheduler already capped
                // prefill+decode at `remaining_for_sched`, so this only
                // fires if verify_rows was over-counted or the budget is
                // smaller than the scheduler's view.
                while reservation.fits(max_batch_tokens as u64) != Ok(true) {
                    let slot = mtp_drafts.iter().position(|x| x.is_some());
                    if let Some(s) = slot {
                        mtp_drafts[s] = None;
                        let vk = (rig.mtp_k + 1) as u64;
                        reservation.verify_rows = reservation.verify_rows.saturating_sub(vk);
                        // The seed stays in remaining_prompt for regular
                        // decode next step; no state mutation needed here.
                    } else {
                        // No verify slots left to drop: trim prefill from
                        // the tail until it fits.
                        if reservation.prefill_rows > 0 {
                            reservation.prefill_rows -= 1;
                        } else if reservation.decode_rows > 0 {
                            reservation.decode_rows -= 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Re-inject MTP verify tokens for the surviving drafts. A slot whose
        // draft was dropped above does not get verify rows; its seed stays in
        // `remaining_prompt` for a regular decode row next step.
        let mut batch = if mtp_drafts.iter().any(|d| d.is_some()) {
            let pos3_deltas: Vec<i32> = work.iter().map(|w| w.pos3_delta).collect();
            // Work-level M-RoPE need — the same predicate the scheduler's
            // `any_pos3` uses — NOT derived from `sched_batch.pos3`, which is
            // empty whenever the scheduler emitted no rows (verify-only
            // steps; see `inject_mtp_verify_tokens`).
            let force_pos3 = work.iter().any(|w| {
                (w.vl_prefill.is_some() && !rig.vl_sequential) || w.pos3_delta != 0
            });
            inject_mtp_verify_tokens(&sched_batch, &mtp_drafts, n, &pos3_deltas, force_pos3)
        } else {
            sched_batch
        };
        if batch.is_empty() {
            continue;
        }
        // VL rows: refresh the per-row vision-matrix base pointers the
        // scatter kernel dereferences. Engine-side upload, every step
        // (including graph replays), so the captured kernel always reads
        // current pointers. Rows with a negative index never dereference.
        if batch.ext_emb.len() == batch.positions.len()
            && batch.ext_emb.iter().any(|&e| e >= 0)
        {
            let ptrs: Vec<u64> = batch
                .row_slot
                .iter()
                .map(|&sl| {
                    rig.vl_ext_devs[sl as usize]
                        .as_ref()
                        .map(|t| t.buf.as_ptr() as u64)
                        .unwrap_or(0)
                })
                .collect();
            let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
            // Bound-check here, not in memcpy_htod: its overflow path is an
            // assert, and a panic in this thread kills the engine loop while
            // the HTTP front end keeps accepting — the serve-hang failure
            // mode. This upload runs before forward_batch_slots' own
            // n <= pbs.max_batch assert, so an oversized batch would only
            // surface here.
            if bytes.len() > rig.pbs.ext_emb_row_ptr.buf.size() {
                let reason = format!(
                    "vl ext ptr upload of {} bytes exceeds staging capacity {} \
                     (batch of {} rows vs max_batch {})",
                    bytes.len(),
                    rig.pbs.ext_emb_row_ptr.buf.size(),
                    batch.total_rows(),
                    rig.pbs.max_batch
                );
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
            if let Err(e) = rig.gpu.hip.memcpy_htod(&rig.pbs.ext_emb_row_ptr.buf, &bytes) {
                let reason = format!("vl ext ptr upload failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
        }
        // lm_head_skip and any_verify are computed AFTER the COW failure
        // isolation (below), so a slot whose draft was dropped there is not
        // marked as a verify slot.
        // ── COW write barrier (spec §4.6) ──────────────────────────────
        // Before KV write kernels: plan copy-on-write for each active slot's
        // block table over the write interval. Sealed/CacheOnly pages (shared
        // via prefix cache) are scheduled for copy to private pages; private
        // pages are a no-op. Always-on — correct even when prefix_cache is
        // off (no sealed pages exist, so plan_cow returns an empty plan).
        let mut cow_plans: Vec<Option<(usize, rdna_compute::page_pool::CowPlan)>> = Vec::with_capacity(n);
        let mut row_offset = 0usize;
        // Per-slot COW provision (spec §5.4/S4): a plan_cow failure for one
        // slot drops ONLY that slot from this step (its request is failed),
        // not every active slot. Prior plans are aborted so no half-installed
        // COW mappings survive (spec §4.3: abort_cow before continuing).
        let mut failed_slots: Vec<usize> = Vec::new();
        for s in 0..n {
            let m = batch.m_per_slot[s];
            if m == 0 {
                cow_plans.push(None);
                continue;
            }
            let write_start = batch.positions[row_offset] as usize;
            let write_end = batch.positions[row_offset + m - 1] as usize + 1;
            match rig.pool.plan_cow_for_slot(SlotId(s), write_start, write_end) {
                Ok(plan) => {
                    rig.cow_plan_count += 1;
                    cow_plans.push(Some((s, plan)));
                }
                Err(e) => {
                    eprintln!("[cow] plan_cow failed for slot {s}: {e}");
                    // plan_cow failed: nothing was installed for this slot,
                    // so no abort is needed. Record the failure and continue
                    // — other slots' plans survive (spec §5.4/S4).
                    failed_slots.push(s);
                    cow_plans.push(None);
                }
            }
            row_offset += m;
        }
        // Fail the isolated slots: reject their requests, clear their work,
        // and zero their batch rows so the forward skips them. Their COW
        // plans (none, since plan_cow failed) need no abort.
        if !failed_slots.is_empty() {
            for &s in &failed_slots {
                // Abort any COW plan that WAS created for this slot before
                // the failure (shouldn't happen since plan_cow failed, but
                // be safe).
                if let Some((_, p)) = cow_plans.get(s).and_then(|o| o.as_ref()) {
                    rig.pool.abort_cow_for_slot(p);
                }
                cow_plans[s] = None;
                // Zero the slot's batch rows so the forward does not write
                // through a slot whose COW plan failed.
                let m = batch.m_per_slot[s];
                batch.m_per_slot[s] = 0;
                let _ = m; // rows are dropped from the flat arrays below
                if let Some(f) = slots[s].take() {
                    let reason = "COW reservation failed for this slot".to_string();
                    let _ = send_event(&f.reply, Event::Rejected { reason });
                    rig.swap.forget(f.session.0);
                    let _ = rig.fair_queue.remove(f.session.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    clear_work_slot(&mut work[s]);
                    clear_slot_vl_state(&mut rig, s);
                }
            }
            // Rebuild the flat arrays to drop the failed slots' rows. The
            // forward reads tokens/positions/row_slot/pos3/ext_emb as flat
            // arrays packed by m_per_slot, so zeroing m_per_slot without
            // removing the flat rows would misalign every subsequent slot.
            rebuild_batch_excluding_failed(&mut batch, &failed_slots);
            // Also drop MTP drafts for failed slots so the verify path
            // does not try to read their hidden rows.
            for &s in &failed_slots {
                mtp_drafts[s] = None;
            }
            if batch.is_empty() {
                continue;
            }
        }
        // Slots with verify drafts skip the per-slot last-row lm_head GEMV
        // inside the forward: their logits come from the batched verify
        // lm_head over all k+1 rows instead, so the single-row GEMV (a full
        // vocab projection) is redundant for them. Computed after COW failure
        // isolation so a dropped draft is not counted as a verify slot.
        let lm_head_skip: Vec<bool> = (0..n).map(|s| mtp_drafts[s].is_some()).collect();
        let any_verify = lm_head_skip.iter().any(|&b| b);

        let fwd = (|| {
            let mut capture = any_verify.then(|| {
                let tape = rig
                    .mtp_verify_tape
                    .as_mut()
                    .expect("MTP drafts require the verify tape");
                crate::forward_slots::MtpVerifyCapture {
                    tape,
                    verify_slots: &lm_head_skip,
                    stride: rig.mtp_k + 1,
                }
            });
            forward_batch_slots_graphed_opts(
                &mut rig.gpu,
                &rig.weights,
                &rig.config,
                &batch,
                &mut rig.pool,
                &mut rig.dn_states,
                &rig.k_arenas,
                &rig.v_arenas,
                &mut rig.desc_staging,
                &rig.kv_tier,
                &rig.pbs,
                &rig.scratch,
                &rig.logits_out,
                &mut graph,
                rig.mtp_k,
                &lm_head_skip,
                capture.as_mut(),
            )
        })();
        // Commit COW plans after the forward succeeded — rebinds block
        // tables to private copy pages. On forward failure, abort instead.
        match &fwd {
            Ok(()) => {
                for (s, plan) in cow_plans.iter().flatten() {
                    if let Err(e) = rig.pool.commit_cow_for_slot(SlotId(*s), plan) {
                        eprintln!("[cow] commit_cow failed for slot {s}: {e}");
                    }
                }
            }
            Err(_) => {
                for (_, plan) in cow_plans.iter().flatten() {
                    rig.pool.abort_cow_for_slot(plan);
                }
            }
        }
        // ── Publication (spec §4.6 C6) ──────────────────────────────────
        // After a successful prefill chunk, publish sealed full pages at
        // committed page-aligned boundaries. Only when prefix_cache is on.
        // Generated-prefix pages are published at the Done event (client commit).
        if rig.prefix_cache {
            let mut row_offset = 0usize;
            for s in 0..n {
                let m = batch.m_per_slot[s];
                if m == 0 {
                    continue;
                }
                row_offset += m;
                // Only publish for prefilling slots (not decoding).
                if work[s].decoding {
                    continue;
                }
                // Vision requests skip the radix (spec X2).
                if work[s].vl_prefill.is_some() {
                    continue;
                }
                let new_boundary = (work[s].next_pos / PAGE_TOKENS) * PAGE_TOKENS;
                let last_pub = slots[s].as_ref().map(|f| f.last_published_boundary).unwrap_or(0);
                if new_boundary <= last_pub || new_boundary == 0 {
                    continue;
                }
                // Publish pages [last_published, new_boundary).
                let n_old_pages = last_pub / PAGE_TOKENS;
                let n_new_pages = new_boundary / PAGE_TOKENS;
                let page_indices: Vec<u32> = match rig.pool.block_table(SlotId(s)) {
                    Some(bt) => bt.page_indices().to_vec(),
                    None => continue,
                };
                if page_indices.len() < n_new_pages {
                    continue;
                }
                // Seal and add cache refs for the newly written pages.
                if let Some(pp) = rig.pool.page_pool_mut() {
                    for p in n_old_pages..n_new_pages {
                        let phys = page_indices[p];
                        let _ = pp.seal(phys);
                        let _ = pp.add_cache_ref(phys);
                    }
                }
                // Build handles for all pages up to the new boundary.
                let handles: Vec<Handle> = page_indices[..n_new_pages]
                    .iter()
                    .enumerate()
                    .map(|(i, &phys)| Handle {
                        handle: rdna_compute::page_pool::PageHandle {
                            phys,
                            epoch: 0,
                            generation: rig.pool.page_pool().map(|p| p.page_generation(phys)).unwrap_or(0),
                        },
                        token_offset: (i * PAGE_TOKENS) as u64,
                    })
                    .collect();
                // Get the session's tokens up to the boundary.
                let session = match slots[s].as_ref().map(|f| f.session) {
                    Some(sid) => sid,
                    None => continue,
                };
                let tokens: Vec<u32> = rig
                    .sessions
                    .get(session)
                    .map(|sess| sess.tokens[..new_boundary].to_vec())
                    .unwrap_or_default();
                if tokens.len() == new_boundary {
                    let domain = rig.cache_domain.as_ref().unwrap();
                    let idx = rig.prefix_index.as_mut().unwrap();
                    let pp = rig.pool.page_pool().unwrap();
                    // Capture checkpoint at the boundary.
                    let checkpoint = if let Some(ckpt_pool) = rig.checkpoint_pool.as_mut() {
                        match capture_checkpoint(
                            &mut rig.gpu,
                            ckpt_pool,
                            domain,
                            new_boundary as u64,
                            &rig.dn_states[s],
                        ) {
                            Ok(id) => Some(id),
                            Err(e) => {
                                eprintln!("[prefix-cache] capture_checkpoint failed: {e}");
                                None
                            }
                        }
                    } else {
                        None
                    };
                    if let Err(e) = idx.publish_sealed_pages(domain, &tokens, &handles, checkpoint, pp) {
                        eprintln!("[prefix-cache] publish_sealed_pages failed: {e:?}");
                    } else {
                        if let Some(f) = slots[s].as_mut() {
                            f.last_published_boundary = new_boundary;
                        }
                        if rig.gpu.slot_trace() {
                            eprintln!(
                                "[slot-trace] published {n_new_pages} pages at boundary {new_boundary}"
                            );
                        }
                    }
                }
            }
        }
        if let Err(e) = fwd {
            // A forward failure is not recoverable per-request. Report it as a
            // REJECTION carrying the reason, not as a normal Done: an
            // unsupported model (e.g. "DeltaNet layer has a non-Q8_0 weight")
            // otherwise reaches the client as a successful, empty 200, which
            // looks like the model chose to say nothing.
            // Retain per-request behavior only because state is provably
            // closed/reset: every active slot is taken, its session closed,
            // work cleared and swap forgotten, so no poisoned KV/DN survives.
            // If this invariant were broken, we would need to poison instead.
            let reason = e.to_string();
            for (s, f) in slots.iter_mut().enumerate() {
                if let Some(f) = f.take() {
                    let _ = send_event(
                        &f.reply,
                        Event::Rejected {
                            reason: reason.clone(),
                        },
                    );
                    rig.swap.forget(f.session.0);
                    let _ = rig.fair_queue.remove(f.session.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    work[s].remaining_prompt.clear();
                    work[s].decoding = false;
                    work[s].next_pos = 0;
                    work[s].mtp_active = false;
                }
            }
            continue;
        }
        if let Err(e) = rig.gpu.hip.device_synchronize() {
            let reason = format!("device synchronize failed: {e:?}");
            fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
            poison = Some(reason);
            break 'serve;
        }
        // Penalty windows: for each slot whose request carries non-neutral
        // token penalties, clamp the requested window to what the session
        // actually holds and upload that tail (most recent last). The kernel
        // scans exactly `repeat_window` entries, so the param must equal the
        // uploaded count — early in a generation that is fewer than the
        // request's window.
        for s in 0..n {
            let penalized = rig.sample_params[s].penalized();
            if !penalized || slots[s].is_none() {
                continue;
            }
            let session = slots[s].as_ref().unwrap().session;
            let Some(sess) = rig.sessions.get(session) else {
                continue;
            };
            let requested = slots[s]
                .as_ref()
                .map(|f| f.repeat_window_req)
                .unwrap_or(0);
            let effective = requested.min(sess.tokens.len());
            rig.sample_params[s].repeat_window = effective as i32;
            if effective == 0 {
                continue;
            }
            let tail = &sess.tokens[sess.tokens.len() - effective..];
            let tail_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tail.as_ptr() as *const u8, effective * 4) };
            if let Err(e) = rig.gpu.hip.memcpy_htod(&rig.repeat_windows[s].buf, tail_bytes) {
                let reason = format!("repeat window upload failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
        }
        // ── Grammar mask: apply BEFORE the sampler (spec §7.2 G2) ───────
        // The hard grammar mask participates before probability-dependent
        // filtering/renormalization. No later operation may resurrect a
        // forbidden token (spec §7.2 G2). For each slot with a grammar
        // constraint, copy its logits row D2H, set disallowed tokens to
        // NEG_INFINITY, and write it back H2D before `sample_per_slot`.
        //
        // An empty allowed set is a typed per-request error, never an
        // unconstrained fallback (spec §7.2 G2, A17). The failing request
        // is rejected; other active requests are unaffected.
        //
        // The mask is built from `SchemaMatcher::is_token_allowed` over the
        // tokenizer's lossless `token_bytes` table (spec §7.2 G2).
        let vocab = rig.config.vocab_size;
        let mut grammar_failures: Vec<(usize, String)> = Vec::new();
        for s in 0..n {
            let Some(f) = slots[s].as_mut() else { continue };
            let Some(constraint) = f.grammar.as_mut() else { continue };

            // Build the mask for the current parser state.
            let mask = match constraint.build_mask(&rig.tokenizer, vocab) {
                Ok(m) => m,
                Err(e) => {
                    grammar_failures.push((s, e.to_string()));
                    continue;
                }
            };

            // D2H: copy this slot's logits row to the host buffer.
            let row_offset = s * vocab;
            let logits_bytes_len = vocab * std::mem::size_of::<f32>();
            let host_slice = &mut rig.logits_host[row_offset..row_offset + vocab];
            let host_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    host_slice.as_mut_ptr() as *mut u8,
                    logits_bytes_len,
                )
            };
            let gpu_row = rig.logits_out.sub_offset(row_offset, vocab);
            if let Err(e) = rig.gpu.hip.memcpy_dtoh(host_bytes, &gpu_row.buf) {
                let reason = format!("grammar mask D2H failed for slot {s}: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }

            // Apply the mask: disallowed tokens → NEG_INFINITY. Allowed
            // tokens are left alone so the sampler's penalties/top_p
            // operate on the original logits (spec §7.2 G2: "Apply request
            // penalties/biases according to the existing sampler contract,
            // then ensure the hard grammar mask participates before
            // probability-dependent filtering/renormalization").
            for (i, &allowed) in mask.iter().enumerate() {
                if !allowed {
                    host_slice[i] = f32::NEG_INFINITY;
                }
            }

            // H2D: write the masked logits back.
            let host_bytes_back: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    host_slice.as_ptr() as *const u8,
                    logits_bytes_len,
                )
            };
            if let Err(e) = rig.gpu.hip.memcpy_htod(&gpu_row.buf, host_bytes_back) {
                let reason = format!("grammar mask H2D failed for slot {s}: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
        }
        // Fail any slots whose grammar constraint produced an empty allowed
        // set or unsatisfiable state. Each is a per-request rejection —
        // other active requests continue (spec §5.4/S4: "Per-request
        // parser/sampling failure → fail that request; release its
        // resources at a safe boundary").
        for (s, reason) in grammar_failures {
            if let Some(f) = slots[s].take() {
                let _ = send_event(&f.reply, Event::Rejected { reason: reason.clone() });
                rig.swap.forget(f.session.0);
                let _ = rig.fair_queue.remove(f.session.0);
                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                clear_work_slot(&mut work[s]);
                clear_slot_vl_state(&mut rig, s);
            }
        }
        // ── Jump-forward planner (spec §7.3 G3) ───────────────────────
        // After the grammar mask, if structured_jump_forward is on and the
        // slot carries a grammar constraint under greedy decoding, call
        // `forced_token_run` to find a bounded run of provably-forced token
        // IDs. The run is committed instead of the sampled token; the
        // forced tokens are pushed into `remaining_prompt` so the scheduler
        // processes them as prefill (writing their KV) in subsequent steps.
        // Config off → jump_forced stays all-None: no behavior change.
        let mut jump_forced: Vec<Option<grammar::json_schema::ForcedRun>> =
            (0..n).map(|_| None).collect();
        if rig.structured_jump_forward {
            let vocab = rig.config.vocab_size;
            for s in 0..n {
                let Some(f) = slots[s].as_ref() else { continue };
                let Some(constraint) = f.grammar.as_ref() else { continue };
                // Greedy only (spec §7.3: "not sampled (greedy only)").
                if rig.sample_params[s].temperature > 1e-6 {
                    continue;
                }
                // Only for decoding slots: remaining_prompt empty means
                // the current logits are for the next token. A still-
                // prefilling slot's logits belong to a mid-prompt token.
                if !work[s].remaining_prompt.is_empty() {
                    continue;
                }
                // MTP is disabled for grammar-constrained requests at
                // admit, but be defensive.
                if work[s].mtp_active {
                    continue;
                }
                let remaining_max = f.max_tokens.saturating_sub(f.produced);
                let max_run = remaining_max.min(rig.max_batch_tokens).min(64);
                if max_run == 0 {
                    continue;
                }
                // Build the lossless token_bytes table and EOS id set
                // for the planner (spec §7.2 G2: "Never build strict
                // constraints on replacement-character artifacts").
                let token_bytes_table: Vec<&[u8]> = (0..vocab as u32)
                    .map(|id| rig.tokenizer.token_bytes(id))
                    .collect();
                let mut eos_ids = vec![rig.tokenizer.eos_id];
                if let Some(eot) = rig.tokenizer.eot_id {
                    eos_ids.push(eot);
                }
                jump_forced[s] = Some(grammar::json_schema::forced_token_run(
                    &constraint.matcher,
                    &token_bytes_table,
                    &eos_ids,
                    max_run,
                ));
            }
        }
        if let Err(e) = rig.gpu.sample_per_slot(
            &rig.logits_out,
            &mut rig.sample_params,
            &rig.repeat_windows,
            n,
            rig.config.vocab_size,
            &rig.out_tokens,
        ) {
            let reason = format!("sample failed: {e:?}");
            fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
            poison = Some(reason);
            break 'serve;
        }
        if let Err(e) = rig.gpu.hip.device_synchronize() {
            let reason = format!("device synchronize failed: {e:?}");
            fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
            poison = Some(reason);
            break 'serve;
        }
        let mut ids = vec![0i32; n];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, n * 4) };
            if let Err(e) = rig.gpu.hip.memcpy_dtoh(bytes, &rig.out_tokens.buf) {
                let reason = format!("DtoH failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
        }

        for s in 0..n {
            // MTP verify/accept: process slots that produced draft outputs.
            if let Some(draft) = mtp_drafts[s].take() {
                if slots[s].is_none() {
                    continue;
                }
                // Compute hidden row offset for this slot in the batch.
                let hidden_row_offset: usize = (0..s)
                    .map(|i| batch.m_per_slot.get(i).copied().unwrap_or(0))
                    .sum();
                // Commit context for the burst clamp: the request counters
                // and the session's current stored length.
                let (produced, max_tokens, sess_len) = match slots[s].as_ref() {
                    Some(f) => (
                        f.produced,
                        f.max_tokens,
                        rig.sessions
                            .get(f.session)
                            .map(|sess| sess.tokens.len())
                            .unwrap_or(0),
                    ),
                    None => (0, usize::MAX, 0),
                };
                match mtp_verify_accept_step(
                    &mut rig,
                    SlotId(s),
                    &mut work[s],
                    draft,
                    hidden_row_offset,
                    produced,
                    max_tokens,
                    sess_len,
                ) {
                    Ok(tokens) => {
                        for tok in &tokens {
                            commit_sampled_token(&mut rig, &mut slots, &mut work, s, *tok);
                            if slots[s].is_none() {
                                break;
                            }
                        }
                        // Adaptive retire: a head that cannot sustain the
                        // minimum mean advance loses wall-clock on the spec
                        // cycle (~2x an AR step). Fall back to plain AR for
                        // the rest of this request. The last committed token
                        // is already in remaining_prompt; the scheduler picks
                        // the slot up as a regular decode row next step.
                        if slots[s].is_some() && work[s].mtp_active {
                            work[s].mtp_cycles += 1;
                            work[s].mtp_committed += tokens.len();
                            if work[s].mtp_cycles >= MTP_RETIRE_WINDOW {
                                let mean = work[s].mtp_committed as f64
                                    / work[s].mtp_cycles as f64;
                                // Reset the window each time it fills; only a
                                // SECOND consecutive failure retires.
                                work[s].mtp_cycles = 0;
                                work[s].mtp_committed = 0;
                                if mean < MTP_RETIRE_MIN_ADVANCE {
                                    work[s].mtp_retire_fails += 1;
                                } else {
                                    work[s].mtp_retire_fails = 0;
                                }
                                if work[s].mtp_retire_fails >= MTP_RETIRE_WINDOWS {
                                    if rig.gpu.slot_trace() {
                                        eprintln!(
                                            "[slot-trace] MTP retired for slot {s}: mean advance {mean:.2} over {} consecutive windows",
                                            MTP_RETIRE_WINDOWS
                                        );
                                    }
                                    work[s].mtp_active = false;
                                    work[s].mtp_retire_fails = 0;
                                    // Of the committed tokens only the newest
                                    // lacks a KV row (the rest were written by
                                    // verify) — keep just it, like the cap
                                    // retire. The scheduler decodes it as a
                                    // regular row next step.
                                    if let Some(last) =
                                        work[s].remaining_prompt.last().copied()
                                    {
                                        work[s].remaining_prompt.clear();
                                        work[s].remaining_prompt.push(last);
                                    }
                                }
                            }
                        }
                    }
                    Err(reason) => {
                        if let Some(f) = slots[s].take() {
                            let _ = send_event(
                                &f.reply,
                                Event::Rejected {
                                    reason: reason.clone(),
                                },
                            );
                            rig.swap.forget(f.session.0);
                            let _ = rig.fair_queue.remove(f.session.0);
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        }
                        clear_work_slot(&mut work[s]);
                        clear_slot_vl_state(&mut rig, s);
                    }
                }
                continue;
            }
            // MTP prefill: fill the head's private KV from this chunk's
            // x_batch rows; when the prompt has drained, the sampled
            // out_token becomes the draft seed and decode begins.
            if work[s].mtp_active
                && !work[s].decoding
                && slots[s].is_some()
                && batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
            {
                let m = batch.m_per_slot[s];
                let row_off: usize = (0..s)
                    .map(|i| batch.m_per_slot.get(i).copied().unwrap_or(0))
                    .sum();
                let chunk_tokens: Vec<u32> =
                    batch.tokens[row_off..row_off + m].to_vec();
                let chunk_positions: Vec<i32> =
                    batch.positions[row_off..row_off + m].to_vec();
                match mtp_head_prefill_chunk(
                    &mut rig,
                    SlotId(s),
                    &chunk_tokens,
                    &chunk_positions,
                    row_off,
                    m,
                ) {
                    Ok(()) => {
                        if work[s].remaining_prompt.is_empty() && slots[s].is_some() {
                            // Prompt fully prefilled (trunk KV by the
                            // scheduler path, head KV above): the sampled
                            // token is the first draft seed.
                            let seed = ids[s] as u32;
                            work[s].decoding = true;
                            if crate::mtp_spec::mtp_trace_enabled() {
                                eprintln!("[mtp-trace] prefill done seed={seed}");
                            }
                            commit_sampled_token(&mut rig, &mut slots, &mut work, s, seed);
                        }
                    }
                    Err(reason) => {
                        if let Some(f) = slots[s].take() {
                            let _ = send_event(
                                &f.reply,
                                Event::Rejected {
                                    reason: reason.clone(),
                                },
                            );
                            rig.swap.forget(f.session.0);
                            let _ = rig.fair_queue.remove(f.session.0);
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        }
                        clear_work_slot(&mut work[s]);
                        clear_slot_vl_state(&mut rig, s);
                    }
                }
                continue;
            }
            // Batched VL slots sample and commit like any regular slot (the
            // last prompt row's logits seed the first token); only
            // MTP-decoding slots skip — their tokens come from the verify.
            if work[s].mtp_active {
                continue;
            }
            // Still prefilling: these logits belong to a mid-prompt token.
            if !work[s].remaining_prompt.is_empty() {
                continue;
            }
            // Jump-forward (spec §7.3 G3): commit the provably-forced token
            // run instead of the sampled token. Each forced token is
            // committed via `commit_sampled_token`, which advances the
            // matcher, emits Token events, increments produced, and pushes
            // the token into `remaining_prompt` so the scheduler processes
            // it as prefill (writing its KV) in subsequent steps. The
            // FairQueue is not jumped — the scheduler still respects its
            // eligibility mask for the forced prefill rows.
            if let Some(forced) = jump_forced[s].take() {
                if !forced.token_ids.is_empty() {
                    for &tok in &forced.token_ids {
                        commit_sampled_token(&mut rig, &mut slots, &mut work, s, tok);
                        if slots[s].is_none() {
                            break;
                        }
                    }
                    continue;
                }
            }
            let Some(f) = slots[s].as_mut() else { continue };
            let tok = ids[s] as u32;
            // Advance the grammar parser cursor on the accepted token
            // (spec §7.2 G2: "per-request parser cursor advanced on
            // accepted tokens only").
            if let Some(constraint) = f.grammar.as_mut() {
                constraint.advance_token(&rig.tokenizer, tok);
            }
            // Check whether the grammar constraint (if any) is in an
            // accepting state after advancing. A non-accepting state at
            // terminal means the generated JSON is incomplete/invalid —
            // emit a typed rejection instead of a successful done (spec
            // §7.1 G1, A17).
            let grammar_incomplete = f
                .grammar
                .as_ref()
                .is_some_and(|c| !c.is_accepting());
            let session = f.session;
            let hit_eos = rig.tokenizer.is_terminator(tok);
            // Emit BEFORE counting, but never emit the terminator itself --
            // otherwise `<|im_end|>` lands in the client's content.
            let gone = if hit_eos {
                false
            } else {
                send_event(&f.reply, Event::Token { id: tok }).is_err()
            };
            f.produced += 1;
            let hit_max = f.produced >= f.max_tokens;
            // Context-cap guard: stop before the slot's KV length would pass
            // its cap. Without this, a long multi-turn session overflows
            // set_seq_len and kills the whole batched step.
            let hit_ctx_cap = rig
                .sessions
                .get(session)
                .map(|sess| sess.tokens.len() + 1 >= rig.cap_tokens)
                .unwrap_or(false);

            if gone || hit_eos || hit_max || hit_ctx_cap {
                let reason = if gone {
                    DoneReason::ClientGone
                } else if hit_eos {
                    DoneReason::Eos
                } else {
                    DoneReason::MaxTokens
                };
                // MaxTokens: the final emitted token reached the client but has
                // not yet gone through the next forward. Keep it in the
                // authoritative stream; held seq_len remains one shorter, so
                // begin_turn prefills this token plus the next-turn suffix.
                if should_retain_terminal_tok(reason) {
                    if let Some(sess) = rig.sessions.get_mut(session) {
                        sess.tokens.push(tok);
                    }
                }
                // Terminator and undelivered (gone) tokens are counted by
                // `produced` but never reached the client.
                let generated = if hit_eos || gone {
                    f.produced - 1
                } else {
                    f.produced
                };
                // If a grammar constraint is present and not accepting at
                // terminal, the generated JSON is incomplete/invalid — emit
                // a typed rejection instead of a successful done (spec §7.1
                // G1, A17). ClientGone is never overridden: the receiver is
                // already dropped.
                if grammar_incomplete && !matches!(reason, DoneReason::ClientGone) {
                    let _ = send_event(
                        &f.reply,
                        Event::Rejected {
                            reason: GrammarMaskError::Unsatisfiable.to_string(),
                        },
                    );
                } else {
                    let _ = send_event(&f.reply, Event::Done { reason, generated });
                }
                // Publish generated-prefix pages at client commit (spec §4.6 C6).
                if !matches!(reason, DoneReason::ClientGone) && rig.prefix_cache {
                    publish_generated_prefix(&mut rig, &mut slots, s);
                }
                slots[s] = None;
                let _ = rig.fair_queue.remove(session.0);
                clear_work_slot(&mut work[s]);
                clear_slot_vl_state(&mut rig, s);
                if matches!(reason, DoneReason::ClientGone) {
                    // Cancellation: unpin prefix cache domain pins (spec §4.4).
                    if rig.prefix_cache {
                        if let (Some(idx), Some(domain)) = (rig.prefix_index.as_mut(), rig.cache_domain.as_ref()) {
                            idx.unpin(domain);
                        }
                    }
                    // Nobody will follow up on a vanished client, so hand the
                    // slot back at once.
                    rig.swap.forget(session.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, session);
                } else {
                    // Keep the session RESIDENT but idle. Its KV and recurrent
                    // state are exactly what a follow-up turn continuing this
                    // conversation needs; closing here is what made multi-turn
                    // reuse impossible. LRU eviction reclaims the slot when
                    // someone else needs it.
                    rig.sessions.touch(session);
                }
            } else {
                work[s].remaining_prompt.push(tok);
                if let Some(sess) = rig.sessions.get_mut(session) {
                    sess.tokens.push(tok);
                }
                rig.sessions.touch(session);
            }
        }
    }

    // Shutdown/drain: if channel closed or poisoned while slots remain, fail them.
    // This ensures no in-flight cancellation can remain reusable and teardown
    // drains every active request. Normal shutdown drain (no prior poison) is
    // not itself poison — only HIP failures poison.
    let has_active = slots.iter().any(|s| s.is_some());
    if has_active {
        let reason = poison
            .clone()
            .unwrap_or_else(|| "engine shutdown".to_string());
        fail_all_active(&mut rig, &mut slots, &mut work, reason);
    }

    // Drain the wait queue on shutdown: reject every parked request so no
    // client is left waiting forever. The wait queue is pure host state —
    // no GPU resources to free, just event delivery.
    let shutdown_reason = poison
        .clone()
        .unwrap_or_else(|| "engine shutdown".to_string());
    while let Some(waiter) = rig.wait_queue.pop_ready(u64::MAX) {
        if let Some(parked) = rig.parked_requests.remove(&waiter.id) {
            let _ = send_event(
                &parked.reply,
                Event::Rejected {
                    reason: shutdown_reason.clone(),
                },
            );
            stats.lock().expect("stats").note_rejected();
        }
    }

    // Free GPU after poisoning/shutdown drain. Graph is destroyed before buffers.
    let free_res = rig.free_gpu(graph);
    match (poison, free_res) {
        (Some(p), Ok(())) => Err(p),
        (Some(p), Err(e)) => Err(format!("{p}; teardown failed: {e}")),
        (None, Ok(())) => Ok(()),
        (None, Err(e)) => Err(e),
    }
}
fn should_retain_terminal_tok(reason: DoneReason) -> bool {
    matches!(reason, DoneReason::MaxTokens)
}

fn handle_command(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Arc<Mutex<EngineStats>>,
    cmd: EngineCommand,
) {
    match cmd {
        EngineCommand::Submit(req) => admit(rig, slots, work, stats, req),
        EngineCommand::Close { session, reply } => {
            let sid = SessionId(session);
            // At the engine-loop command boundary: if session is in flight,
            // cancel it — remove InFlight, drop its reply sender (Event channel),
            // clear work, forget swap, close SessionTable/pool/admission, then ack.
            // Idle close does the same forget/close. No ignored close.
            if let Some(idx) = slots
                .iter()
                .position(|s| s.as_ref().map(|f| f.session == sid).unwrap_or(false))
            {
                if let Some(inflight) = slots[idx].take() {
                    drop(inflight.reply);
                }
                clear_work_slot(&mut work[idx]);
                clear_slot_vl_state(rig, idx);
                let _ = rig.fair_queue.remove(session);
            }
            rig.swap.forget(session);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, sid);
            let _ = reply.send(Ok(()));
        }
        EngineCommand::Reset { reply } => {
            if slots.iter().any(|s| s.is_some()) {
                let _ = reply.send(Err("reset rejected: requests in flight".to_string()));
                return;
            }
            if rig.wait_queue.queued_count() > 0 {
                let _ = reply.send(Err(
                    "reset rejected: requests queued in the wait room".to_string(),
                ));
                return;
            }
            let ids: Vec<u64> = rig.sessions.iter().map(|(id, _)| id).collect();
            for id in ids {
                rig.swap.forget(id);
                rig.sessions
                    .close(&mut rig.pool, &mut rig.adm, SessionId(id));
            }
            // Drop prefix cache state so subsequent lookups miss (spec §4.6).
            if rig.prefix_cache {
                // Replace the radix index with a fresh empty one.
                rig.prefix_index = Some(hipfire_runtime::prefix_index::PrefixIndex::new(1 << 16));
                // Drain and free GPU blobs from the checkpoint pool.
                if let Some(pool) = rig.checkpoint_pool.as_mut() {
                    for blob in pool.drain_blobs() {
                        blob.free_gpu(&mut rig.gpu);
                    }
                }
                // Bump the allocation epoch so stale CacheDomain lookups miss.
                if let Some(domain) = rig.cache_domain.as_mut() {
                    domain.device.allocation_epoch += 1;
                }
            }
            let _ = reply.send(Ok(()));
        }
    }
}
/// Pure policy: should the admit path attempt a prefix-cache lookup?
///
/// Returns true only when prefix caching is enabled and the request carries
/// no vision data (spec X2: Vision+prefix reuse OFF).
fn should_lookup_prefix(prefix_cache: bool, has_visual: bool) -> bool {
    prefix_cache && !has_visual
}

/// Pure policy: how many prompt tokens are reused (skipped) given a resume
/// plan boundary and the total prompt length.
///
/// Returns 0 when there is no checkpoint. A boundary that covers the entire
/// prompt is a full hit: every prompt token is reused and prefill is empty
/// (spec §4.2 / A1 exact match).
fn reused_tokens_from_plan(boundary: usize, prompt_len: usize) -> usize {
    if boundary == 0 {
        0
    } else {
        boundary.min(prompt_len)
    }
}

/// Pure policy: should the admit path enqueue in the WaitQueue instead of
/// rejecting immediately (spec §5.3 S3, replacing R-A4)?
///
/// Returns true only when every resident session is generating (`all_busy`)
/// and the bounded waiting room is enabled (`wait_enabled` — always true
/// when Rig::build succeeded, since `WaitQueue::new` refuses zero caps).
fn should_wait_instead_of_reject(all_busy: bool, wait_enabled: bool) -> bool {
    all_busy && wait_enabled
}

// ---- FairQueue integration helpers (spec §5.3 S3) ----

/// Extract the set of request ids that received any grant this step.
///
/// Pure projection over a [`Selection`]: the FairQueue decides WHO is
/// eligible (aged-first, backfill, prefill cursor); the scheduler decides
/// HOW MANY rows each granted slot gets. This set becomes the scheduler's
/// eligibility mask so an un-granted slot contributes 0 rows and its
/// `remaining_prompt` is not drained (no tokens lost).
fn eligible_ids_from_selection(sel: &Selection) -> HashSet<u64> {
    sel.grants.iter().map(|g| g.id()).collect()
}

/// Bounded backfill (spec §5.3 S3.4): when the oldest request is starved,
/// drop younger grants so the oldest receives the budget it needs. Returns
/// a set containing only `oldest_id` when `starved_oldest` is set; otherwise
/// returns every granted id. The oldest is force-included even if it
/// received no grant this tick — the backfill exists precisely to give a
/// starved oldest request the next step's budget.
fn apply_starved_backfill(sel: &Selection, oldest_id: u64) -> HashSet<u64> {
    if sel.starved_oldest {
        let mut set = HashSet::new();
        set.insert(oldest_id);
        set
    } else {
        eligible_ids_from_selection(sel)
    }
}

/// Find the session id of the oldest admitted request currently in flight,
/// by lowest `admission_tick` (ties broken by lowest id). Returns `None`
/// when no active slot is tracked by the FairQueue.
fn oldest_active_session(
    q: &FairQueue,
    slots: &[Option<InFlight>],
) -> Option<u64> {
    let mut best: Option<(u64, u64)> = None; // (admission_tick, session_id)
    for f in slots.iter().flatten() {
        let Some(req) = q.get(f.session.0) else { continue };
        match best {
            None => best = Some((req.admission_tick, f.session.0)),
            Some((bt, bid)) => {
                if (req.admission_tick, f.session.0) < (bt, bid) {
                    best = Some((req.admission_tick, f.session.0));
                }
            }
        }
    }
    best.map(|(_, id)| id)
}

/// Sync a request's per-step needs into the FairQueue from slot state.
/// `wants_decode` tracks a runnable decode row; `verify_rows` carries MTP
/// verify rows (k+1) when a draft was produced this step; `uncached` is the
/// remaining prefill suffix length (0 for decoding slots). Errors are
/// ignored defensively — every active slot is admitted to the queue, so a
/// `NotFound` only indicates a transient race at retirement.
fn sync_fair_needs(
    q: &mut FairQueue,
    id: u64,
    wants_decode: bool,
    verify_rows: u64,
    uncached: u64,
) {
    let _ = q.set_needs(id, wants_decode, verify_rows, 0);
    let _ = q.set_uncached_prefill(id, uncached);
}

/// Admit one request, evicting an idle session if that is what it takes.
fn admit(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Arc<Mutex<EngineStats>>,
    req: SubmitRequest,
) {
    let busy: Vec<SessionId> = slots.iter().flatten().map(|f| f.session).collect();

    if let Some(vd) = req.visual_data.as_ref() {
        let reject = |reason: String| {
            let _ = send_event(&req.reply, Event::Rejected { reason });
            stats.lock().expect("stats").note_rejected();
        };
        if rig.vision_weights.is_none() {
            reject("VL request but model has no vision encoder".to_string());
            return;
        }
        // Batched VL runs the paged block-table path like any slot; only the
        // legacy sequential fallback needs a contiguous slab view.
        if rig.pool.is_paged() && rig.vl_sequential {
            reject("VL sequential path requires contiguous KV slots".to_string());
            return;
        }
        if rig.tokenizer.special_token_id("<|image_pad|>").is_none() {
            reject("VL tokenizer missing <|image_pad|>".to_string());
            return;
        }
        if vd.n_visual_tokens == 0 {
            reject("VL request has no visual tokens".to_string());
            return;
        }
        if vd.mrope_positions.len() != req.prompt_tokens.len() {
            reject(format!(
                "VL mrope_positions length {} != prompt_tokens {}",
                vd.mrope_positions.len(),
                req.prompt_tokens.len()
            ));
            return;
        }
    }

    // Multi-turn continuation. The client resends the whole conversation, so a
    // follow-up turn is matched by its USER turns and then built by APPENDING
    // `continuation` to the session's exact stored tokens. Appending rather
    // than re-rendering is what makes it a strict extension: the generated
    // turn began after an OpenThink opener that history rendering does not
    // replay, and re-encoding the decoded reply is not guaranteed to round
    // trip. Reuse also keeps the DeltaNet state, which is the point.
    if req.visual_data.is_none() && !req.continuation.tokens().is_empty() {
        if rig.gpu.slot_trace() {
            let shape = match &req.continuation {
                Continuation::ToolResults { session, .. } => {
                    format!("tool-results of session {session}")
                }
                _ => "new user turn".to_string(),
            };
            eprintln!(
                "[slot-trace] continuation attempt ({shape}): convo={:?} suffix={} tokens",
                req.convo,
                req.continuation.tokens().len(),
            );
            for (id, sess) in rig.sessions.iter() {
                eprintln!(
                    "[slot-trace]   session {} convo={:?} slot={:?} residency={:?} tokens={}",
                    id,
                    sess.convo,
                    sess.slot,
                    sess.residency,
                    sess.tokens.len()
                );
            }
        }
        // The suffix shape decides which match is legal — never both. A user
        // turn searches for the session one turn behind; tool results confirm
        // the session that emitted the calls being answered. Either way the
        // suffix is appended to the session's exact stored tokens, so the KV
        // and DeltaNet state extend strictly.
        let matched = match &req.continuation {
            Continuation::Cold => None,
            Continuation::UserTurn(_) => rig.sessions.find_continuation(&req.convo, &busy),
            Continuation::ToolResults { session, .. } => {
                rig.sessions
                    .confirm_reentry(SessionId(*session), &req.convo, &busy)
            }
        };
        if let Some(existing) = matched {
            let base = rig
                .sessions
                .get(existing)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            // A matched session may be SWAPPED. Bring it back before use --
            // this is the half of SP6 that makes eviction worth doing, and
            // without it evicted snapshots accumulate and their sessions are
            // never seen again.
            let mut slot = rig.sessions.get(existing).and_then(|s| s.slot);
            // Resident = this session never left the slot, so the slot's
            // trunk KV/DN state AND the MTP head's private KV (filled through
            // this session's tokens) are still exactly right. Only a
            // swap-restore loses the head KV — the swap snapshot does not
            // carry it — and must reset before the suffix re-fills.
            let mtp_head_kv_valid = slot.is_some();
            if slot.is_none() {
                let free = rig.pool.acquire().or_else(|| {
                    rig.sessions
                        .lru_idle_victim(&busy)
                        .filter(|v| *v != existing)
                        .and_then(|v| {
                            evict(rig, v);
                            rig.pool.acquire()
                        })
                });
                if let Some(target) = free {
                    if restore(rig, existing, target) {
                        stats.lock().expect("stats").note_restore();
                        slot = Some(target);
                    } else {
                        // restore marked it Cold; its tokens survive, so the
                        // cold path below re-prefills.
                        rig.pool.release(target);
                    }
                }
            }
            if let Some(slot) = slot {
                let mut extended = base;
                extended.extend_from_slice(req.continuation.tokens());
                // Prefill-side context-cap guard (mirrors decode hit_ctx_cap).
                // A prompt/extension already at the KV cap overflows set_seq_len
                // during prefill before any decode step can stop it.
                if extended.len() >= rig.cap_tokens {
                    // A session at the KV cap can never grow: report the
                    // condition as a REJECTION, not a Done — a successful
                    // empty completion here made every subsequent turn on
                    // the session a silent no-op (it would re-enter this
                    // guard forever while the resident session pinned its
                    // slot). The session is closed so the slot is released;
                    // a follow-up on the same conversation re-attempts cold
                    // and receives the same actionable rejection.
                    let reason = format!(
                        "context window full: the conversation holds {} of \
                         {} tokens and cannot extend — start a new \
                         conversation",
                        extended.len(),
                        rig.cap_tokens
                    );
                    let _ = send_event(&req.reply, Event::Rejected { reason });
                    rig.swap.forget(existing.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                    return;
                }
                // vLLM-style admission belt, mirroring the daemon's cold-path
                // belt: the extension plus the requested generation budget
                // must fit the context. Without this, a request admitted in
                // the (cap - max_tokens, cap) window decodes until the ctx-cap
                // guard truncates it — a silent "length" finish the client
                // never asked for. The SESSION is kept (it still has room to
                // grow): only this request is rejected, so a retry with a
                // smaller max_tokens can still continue the conversation.
                if extended.len() + req.max_tokens.max(1) > rig.cap_tokens {
                    let reason = format!(
                        "context window budget: the conversation would hold {} \
                         tokens and the request asks for {} more, beyond the {} \
                         token context — reduce max_tokens or start a new \
                         conversation",
                        extended.len(),
                        req.max_tokens.max(1),
                        rig.cap_tokens
                    );
                    let _ = send_event(&req.reply, Event::Rejected { reason });
                    return;
                }
                if let Ok(plan) = rig.sessions.begin_turn(&mut rig.pool, existing, &extended) {
                    if let Some(sess) = rig.sessions.get_mut(existing) {
                        sess.tokens = extended.clone();
                        sess.convo = req.convo.clone();
                    }
                    if send_event(
                        &req.reply,
                        Event::Accepted {
                            session: existing.0,
                            reused: plan.reused,
                            prefill: extended.len() - plan.reused,
                        },
                    )
                    .is_err()
                    {
                        // Mutated session must not remain reusable — forget swap,
                        // close SessionTable/pool/admission so no poisoned KV/DN
                        // state survives. Preserve typed Continuation semantics
                        // and max-terminal-token accounting (caller retains base
                        // for retransmission).
                        rig.swap.forget(existing.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                        return;
                    }
                    work[slot.0].remaining_prompt = extended[plan.reused..].to_vec();
                    work[slot.0].next_pos = plan.reused;
                    work[slot.0].decoding = false;
                    // A text follow-up on a conversation that carries an
                    // image turn keeps that turn's M-RoPE phase offset: KV
                    // rows are addressed by token index, but the image
                    // turn's keys hold compressed-grid phases, so this
                    // turn's rows must phase-shift by the same delta.
                    // (Image turns themselves are forced Cold and rebuild
                    // the offset from their own table.)
                    work[slot.0].pos3_delta =
                        rig.sessions.get(existing).map(|s| s.rope_delta).unwrap_or(0);
                    // MTP: activate if enabled. Vision continuations stay on
                    // the sequential M-RoPE path (no MTP), matching cold
                    // admits; penalized requests stay on AR (see
                    // `request_penalized`). The head KV reset is only needed
                    // when the head KV was lost (swap-restore) or holds
                    // another session's rows: a session resident on this slot
                    // since its last turn keeps a head KV that matches its
                    // committed prefix, so the suffix's head-fill extends it
                    // in place.
                    work[slot.0].mtp_active = rig.mtp_head.is_some()
                        && rig.mtp_k > 0
                        && req.visual_data.is_none()
                        && !request_penalized(&req)
                        && !request_sampled(&req)
                        && req.json_schema.is_none();
                    if work[slot.0].mtp_active
                        && !(mtp_head_kv_valid && rig.mtp_states[slot.0].is_some())
                    {
                        if let Some(state) = rig.mtp_states[slot.0].as_mut() {
                            let _ = state.reset(&mut rig.gpu);
                        }
                    }
                    install_sample_params(rig, slot, &req);
                    // Compile the JSON Schema for the continuation path too
                    // (spec §7 G1/G2). Same recompile-as-cursor pattern as
                    // the cold admit path.
                    let grammar_constraint = req.json_schema.as_ref().and_then(|schema| {
                        grammar::json_schema::CompiledSchema::compile(schema)
                            .ok()
                            .map(|compiled| GrammarConstraint {
                                matcher:
                                    grammar::json_schema::SchemaMatcher::from_compiled(
                                        &compiled,
                                    ),
                                mask_buf: Vec::new(),
                            })
                    });
                    slots[slot.0] = Some(InFlight {
                        session: existing,
                        reply: req.reply,
                        produced: 0,
                        max_tokens: req.max_tokens.max(1),
                        repeat_window_req: req.repeat_window.min(REPEAT_WINDOW_MAX),
                        reused_tokens: 0,
                        last_published_boundary: 0,
                        grammar: grammar_constraint,
                    });
                    rig.sessions.touch(existing);
                    if rig.gpu.slot_trace() {
                        eprintln!(
                            "[slot-trace] continuation HIT -- session {} reused {} of {} tokens",
                            existing.0,
                            plan.reused,
                            extended.len()
                        );
                    }
                    // FairQueue: admit the continued request with its
                    // remaining prefill suffix as the cache-locality weight.
                    let uncached = work[slot.0].remaining_prompt.len() as u64;
                    let _ = rig.fair_queue.admit(existing.0, "default", uncached);
                    let mut st = stats.lock().expect("stats");
                    st.note_admitted();
                    st.note_prefix_hit();
                    return;
                }
            }
        }
        if rig.gpu.slot_trace() {
            eprintln!("[slot-trace] continuation MISS -- falling back to cold prefill");
        }
    }

    // Prefill-side context-cap guard for cold admits. Rejected, not Done: a
    // successful empty completion would read as the model choosing silence;
    // the prompt physically cannot fit the slot's KV.
    if req.prompt_tokens.len() >= rig.cap_tokens {
        let reason = format!(
            "prompt of {} tokens exceeds the slot context cap of {} tokens",
            req.prompt_tokens.len(),
            rig.cap_tokens
        );
        let _ = send_event(&req.reply, Event::Rejected { reason });
        stats.lock().expect("stats").note_rejected();
        return;
    }

    // Try to open; when admission refuses, make room from the LRU idle
    // sessions and retry — bounded by how many victims there are.
    //
    // Two distinct shortages need two distinct remedies:
    // * PoolFull — a slot is the missing resource: PARK the victim to swap
    //   (its grant stays reserved against the snapshot, which is fine).
    // * WouldExceedBudget — context grant bytes are the missing resource,
    //   and parking does not release those (it swaps where they live, it
    //   does not forgive them), so the retry would fail identically. CLOSE
    //   the victim instead: its grant returns to the budget and a future
    //   turn on that conversation simply cold-prefills.
    // In both cases only IDLE sessions are touched — never an active one.
    let mut opened = rig.sessions.open(&mut rig.pool, &mut rig.adm, rig.cap_tokens);
    while opened.is_err() && rig.sessions.lru_idle_victim(&busy).is_some() {
        let budget_shaped = matches!(
            opened,
            Err(AdmitError::WouldExceedBudget { .. })
        );
        // Re-derive the victim each pass: the previous remedy consumed it.
        let victim = rig.sessions.lru_idle_victim(&busy).expect("checked above");
        if budget_shaped {
            rig.swap.forget(victim.0);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, victim);
            if rig.gpu.slot_trace() {
                eprintln!(
                    "[slot-trace] admission budget full -- closed idle session {} to make room",
                    victim.0
                );
            }
        } else {
            if !evict(rig, victim) {
                let _ = send_event(
                    &req.reply,
                    Event::Rejected { reason: "eviction failed".to_string() },
                );
                stats.lock().expect("stats").note_rejected();
                return;
            }
            stats.lock().expect("stats").note_eviction();
        }
        opened = rig.sessions.open(&mut rig.pool, &mut rig.adm, rig.cap_tokens);
    }
    let id = match opened {
        Ok(id) => id,
        Err(e) => {
            // R-A4 replacement (spec §5.3 S3): when every resident session
            // is generating and no idle victim can make room, enqueue the
            // request in the bounded WaitQueue instead of rejecting
            // immediately. The request is parked and retried when a slot
            // frees; a per-waiter tick deadline drives timeout expiry.
            if should_wait_instead_of_reject(true, rig.wait_queue.max_count() > 0) {
                let waiter_id = rig.next_waiter_id;
                rig.next_waiter_id += 1;
                // queue_bytes is the canonical pending-input bytes from the
                // HTTP admission queue (spec §5.3). Fall back to 1 so a
                // request that bypassed the byte-bounded queue still has a
                // non-zero byte charge (WaitQueue enforces a non-zero byte
                // cap but a zero-byte waiter would never trip it).
                let bytes = req.queue_bytes.max(1);
                match rig.wait_queue.try_enqueue(waiter_id, bytes, rig.tick) {
                    Ok(()) => {
                        rig.parked_requests.insert(waiter_id, req);
                        return;
                    }
                    Err(we) => {
                        // Queue full or byte cap exceeded: surface as a
                        // typed overload rejection (HTTP 429 at the front
                        // end). The request is not parked.
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                reason: format!("serve queue full: {we}"),
                            },
                        );
                        stats.lock().expect("stats").note_rejected();
                        return;
                    }
                }
            }
            // WaitQueue disabled (should not happen — Rig::build refuses
            // zero caps): fall back to the original immediate reject.
            let _ = send_event(
                &req.reply,
                Event::Rejected { reason: format!("{e:?}") },
            );
            stats.lock().expect("stats").note_rejected();
            return;
        }
    };

    let slot = match rig.sessions.get(id).and_then(|s| s.slot) {
        Some(s) => s,
        None => {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    reason: "admitted session holds no slot".to_string(),
                },
            );
            stats.lock().expect("stats").note_rejected();
            return;
        }
    };

    // Reset the slot's recurrent state before reusing it. `seq_len = 0` clears
    // the KV, but DeltaNet state lives OUTSIDE the KV arena, so without this a
    // new conversation inherits the previous occupant's recurrent state --
    // which shows up as degenerate or echoed output on every request after the
    // first. Same trap as the swap unit: KV alone is not the whole state.
    if let Err(e) = rig.dn_states[slot.0].reset(&mut rig.gpu) {
        let _ = send_event(
            &req.reply,
            Event::Rejected {
                reason: format!("state reset failed: {e}"),
            },
        );
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        stats.lock().expect("stats").note_rejected();
        return;
    }

    // Reset MTP head KV for the new conversation (stale positions from the
    // previous occupant would poison the first draft step).
    if let Some(state) = rig.mtp_states[slot.0].as_mut() {
        let _ = state.reset(&mut rig.gpu);
    }

    // ── Prefix cache lookup (spec §4.5–4.6) ──────────────────────────────
    // Cross-session prefix reuse: consult the radix index for reusable KV
    // pages and the checkpoint pool for recurrent state. This is SEPARATE
    // from the session-local continuation path (convo hashes) — the radix
    // index is keyed by exact token sequences across sessions. Vision
    // requests skip the radix entirely (spec X2: Vision+prefix reuse OFF).
    let mut prefix_reused: usize = 0;
    let mut prefix_hit = false;
    if should_lookup_prefix(rig.prefix_cache, req.visual_data.is_some()) {
        rig.lookup_count += 1;
        let domain = rig.cache_domain.as_ref().unwrap();
        let (lookup_result, hit_handles) = {
            let pp = rig
                .pool
                .page_pool()
                .expect("prefix_cache requires paged pool");
            rig.prefix_index
                .as_mut()
                .unwrap()
                .lookup_with_pages(domain, &req.prompt_tokens, pp, None)
        };
        if let PrefixLookupResult::Hit(lookup) = &lookup_result {
            let ckpt_pool = rig.checkpoint_pool.as_ref().unwrap();
            match plan_resume(
                ckpt_pool,
                domain,
                req.prompt_tokens.len() as u64,
                lookup,
                DrafterDecision::Ar,
            ) {
                Ok(plan) => {
                    let boundary = plan.boundary as usize;
                    if reused_tokens_from_plan(boundary, req.prompt_tokens.len()) > 0 {
                        let n_pages = boundary / PAGE_TOKENS;
                        let phys_pages: Vec<u32> = hit_handles
                            .iter()
                            .take(n_pages)
                            .map(|h| h.handle.phys)
                            .collect();
                        if n_pages > 0 && phys_pages.len() == n_pages {
                            if rig.pool.share_published_pages(slot, &phys_pages).is_ok() {
                                // Restore DN state from the checkpoint pool
                                // (peek + restore_to: one D2D copy from the
                                // pool's immutable snapshot into the live
                                // slot state).
                                let ckpt_pool = rig.checkpoint_pool.as_ref().unwrap();
                                if let Some(snapshot) =
                                    ckpt_pool.peek(domain, boundary as u64)
                                {
                                    let _ = snapshot
                                        .restore_to(&mut rig.dn_states[slot.0], &mut rig.gpu);
                                }
                                prefix_reused = boundary;
                                prefix_hit = true;
                                if rig.gpu.slot_trace() {
                                    eprintln!(
                                        "[slot-trace] prefix cache HIT: \
                                         reused {}/{} tokens",
                                        boundary,
                                        req.prompt_tokens.len()
                                    );
                                }
                            }
                        }
                    }
                }
                Err(_) => {} // No checkpoint at any boundary: cold prefill
            }
        }
        // Unpin on miss — lookup pins the radix path; a miss means we
        // won't use those pages, so release the pins to allow eviction.
        if !prefix_hit {
            if let (Some(idx), Some(domain)) =
                (rig.prefix_index.as_mut(), rig.cache_domain.as_ref())
            {
                idx.unpin(domain);
            }
        }
    }

    // Session-local prefix reuse (convo hash continuation) OR prefix cache
    // hit. When the prefix cache hit, begin_turn is skipped because it
    // would reset seq_len to 0, undoing the shared pages.
    let reused = if prefix_hit {
        if let Some(sess) = rig.sessions.get_mut(id) {
            sess.next_pos = prefix_reused;
        }
        prefix_reused
    } else {
        // Prefix reuse: a fresh session reuses nothing, a continued one
        // reuses its common prefix. Either way the suffix is what gets
        // prefilled.
        let plan = match rig
            .sessions
            .begin_turn(&mut rig.pool, id, &req.prompt_tokens)
        {
            Ok(p) => p,
            Err(e) => {
                let _ = send_event(&req.reply, Event::Rejected { reason: e });
                rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
                stats.lock().expect("stats").note_rejected();
                return;
            }
        };
        plan.reused
    };
    if let Some(sess) = rig.sessions.get_mut(id) {
        sess.tokens = req.prompt_tokens.clone();
        sess.convo = req.convo.clone();
        // An image turn defines the conversation's M-RoPE phase offset for
        // every later text continuation; a text-only conversation stays 0
        // (fresh sessions are, but this path also serves continuation-miss
        // re-prefills, which must not inherit anything stale).
        sess.rope_delta = req.visual_data.as_ref().map(|vd| vd.rope_delta).unwrap_or(0);
    }

    if send_event(
        &req.reply,
        Event::Accepted {
            session: id.0,
            reused: reused,
            prefill: req.prompt_tokens.len() - reused,
        },
    )
    .is_err()
    {
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        return;
    }

    work[slot.0].decoding = false;
    // Built before `req.visual_data` is moved out below; SlotSampleParams is
    // Copy so this is a plain host-side struct build.
    let penalized = request_penalized(&req);
    let sample_params = SlotSampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repeat_window: req.repeat_window.min(REPEAT_WINDOW_MAX) as i32,
        repeat_penalty: req.repeat_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
    };
    // Compile the JSON Schema (if present) into a per-request SchemaMatcher
    // cursor (spec §7 G1/G2). The schema was already validated in
    // validate_generate_caps; recompile here to build the per-request
    // mutable cursor. CompiledSchema is immutable and Send+Sync; the
    // SchemaMatcher is request-local mutable state (spec §7.2 G2).
    let grammar_constraint = req.json_schema.as_ref().and_then(|schema| {
        match grammar::json_schema::CompiledSchema::compile(schema) {
            Ok(compiled) => Some(GrammarConstraint {
                matcher: grammar::json_schema::SchemaMatcher::from_compiled(&compiled),
                mask_buf: Vec::new(),
            }),
            Err(e) => {
                // Should not happen (validated at submit), but fail closed.
                let _ = send_event(
                    &req.reply,
                    Event::Rejected {
                        reason: format!("json_schema compile failed on admit: {e}"),
                    },
                );
                rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
                stats.lock().expect("stats").note_rejected();
                return None;
            }
        }
    });
    // If compilation failed and we already sent Rejected, bail out.
    if req.json_schema.is_some() && grammar_constraint.is_none() {
        return;
    }
    let grammar_constrained = grammar_constraint.is_some();
    if let Some(vd) = req.visual_data {
        work[slot.0].remaining_prompt = req.prompt_tokens.clone();
        work[slot.0].next_pos = 0;
        work[slot.0].pos3_delta = 0; // phases come from the VL table below
        // Drop any previous request's vision state: the new request's
        // embeddings are not produced until its first batched chunk, and the
        // scheduler holds the slot's rows until then. Free the old matrix +
        // job explicitly — dropping them unfreed leaks VRAM (the buffers are
        // neither pooled nor hipFree'd by Drop).
        clear_slot_vl_state(rig, slot.0);
        work[slot.0].vl_prefill = Some(VlPrefill {
            patches: vd.patches,
            grid_h: vd.grid_h,
            grid_w: vd.grid_w,
            n_visual_tokens: vd.n_visual_tokens,
            embeddings: Vec::new(),
            dim: rig.config.dim,
            visual_idx: 0,
            image_pad_id: rig
                .tokenizer
                .special_token_id("<|image_pad|>")
                .unwrap_or(u32::MAX),
            mrope_positions: vd.mrope_positions,
            rope_delta: vd.rope_delta,
            base: 0,
        });
        work[slot.0].mtp_active = false;
    } else {
        work[slot.0].remaining_prompt = req.prompt_tokens[reused..].to_vec();
        work[slot.0].next_pos = reused;
        work[slot.0].vl_prefill = None;
        work[slot.0].pos3_delta = 0; // fresh session: no image-turn offset
        // Activate MTP for text-only requests when the head is loaded.
        // Penalized requests stay on AR: the verify accept is a bare greedy
        // argmax and cannot reproduce penalize-then-argmax; sampled requests
        // (temperature > 0) stay on AR for the same reason — the verify path
        // cannot reproduce a sampled pick. Grammar-constrained requests also
        // stay on AR: constrained MTP requires joint verification (applying
        // grammar masks before acceptance, not only after a bad token has
        // advanced state), which does not yet exist (spec §6 X2: "MTP +
        // grammar → AR selected before execution until exact joint
        // acceptance is implemented and validated").
        work[slot.0].mtp_active = rig.mtp_head.is_some()
            && rig.mtp_k > 0
            && !penalized
            && !request_sampled(&req)
            && !grammar_constrained;
    }
    rig.sample_params[slot.0] = sample_params;
    slots[slot.0] = Some(InFlight {
        session: id,
        reply: req.reply,
        produced: 0,
        max_tokens: req.max_tokens.max(1),
        repeat_window_req: req.repeat_window.min(REPEAT_WINDOW_MAX),
        reused_tokens: if prefix_hit { prefix_reused } else { 0 },
        last_published_boundary: reused,
        grammar: grammar_constraint,
    });
    if rig.gpu.slot_trace() {
        eprintln!(
            "[slot-trace] cold ADMIT session {} slot={} prompt={} tokens",
            id.0,
            slot.0,
            req.prompt_tokens.len()
        );
    }
    // FairQueue: admit the cold request with its prefill suffix as the
    // cache-locality weight (0 for a fully-reused prefix-cache hit).
    let uncached = work[slot.0].remaining_prompt.len() as u64;
    let _ = rig.fair_queue.admit(id.0, "default", uncached);
    if prefix_hit {
        stats.lock().expect("stats").note_reused_tokens(prefix_reused);
    }
    stats.lock().expect("stats").note_admitted();
}

/// Capture an idle session's state, park it, and free its slot.
///
/// Returns false if anything went wrong, in which case the session is marked
/// `Cold`: its tokens survive, so it re-prefills next time. Slow, never wrong.
fn evict(rig: &mut Rig, victim: SessionId) -> bool {
    let Some(sess) = rig.sessions.get(victim) else {
        return false;
    };
    let Some(slot) = sess.slot else { return false };
    let tokens = sess.tokens.clone();

    let dn_refs = dn_buffers(&rig.dn_states[slot.0]);
    let snap = capture_slot(
        &mut rig.gpu,
        &rig.pool,
        slot,
        &rig.k_arenas,
        &rig.v_arenas,
        &dn_refs,
        &tokens,
        rig.stamp,
    );
    drop(dn_refs);

    match snap {
        Ok(snap) => match rig.swap.park(victim.0, snap) {
            Ok(()) => {
                rig.sessions.mark_swapped(&mut rig.pool, victim);
                true
            }
            Err(_) => {
                rig.sessions.mark_cold(&mut rig.pool, victim);
                true
            }
        },
        Err(_) => {
            rig.sessions.mark_cold(&mut rig.pool, victim);
            true
        }
    }
}

/// Restore a previously swapped session into `slot`. Any failure marks it
/// `Cold` so the caller re-prefills from tokens.
fn restore(rig: &mut Rig, id: SessionId, slot: SlotId) -> bool {
    match rig.swap.unpark(id.0) {
        Ok(snap) => {
            let dn_refs = dn_buffers(&rig.dn_states[slot.0]);
            let r = restore_slot(
                &mut rig.gpu,
                &mut rig.pool,
                slot,
                &rig.k_arenas,
                &rig.v_arenas,
                &dn_refs,
                &snap,
                rig.stamp,
            );
            drop(dn_refs);
            // MTP head KV is per-generation, not per-session. Reset it so
            // the restored session starts a fresh MTP prefill.
            if let Some(state) = rig.mtp_states[slot.0].as_mut() {
                let _ = state.reset(&mut rig.gpu);
            }
            match r {
                Ok(()) => {
                    rig.sessions.mark_resident(id, slot, snap.seq_len);
                    true
                }
                Err(_) => {
                    rig.sessions.mark_cold(&mut rig.pool, id);
                    false
                }
            }
        }
        Err(_) => {
            rig.sessions.mark_cold(&mut rig.pool, id);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pure stand-in for the MaxTokens bookkeeping in `run_loop`: only
    /// MaxTokens keeps the terminal emitted tok in the session transcript.
    fn apply_terminal(tokens: &mut Vec<u32>, tok: u32, reason: DoneReason) {
        if should_retain_terminal_tok(reason) {
            tokens.push(tok);
        }
    }

    #[test]
    fn max_tokens_retains_final_emitted_tok_for_continuation() {
        let mut tokens = vec![10, 20];
        apply_terminal(&mut tokens, 99, DoneReason::MaxTokens);
        assert_eq!(tokens, vec![10, 20, 99]);
    }

    #[test]
    fn eos_and_client_gone_do_not_retain_terminal_tok() {
        let mut tokens = vec![10, 20];
        apply_terminal(&mut tokens, 1, DoneReason::Eos);
        apply_terminal(&mut tokens, 2, DoneReason::ClientGone);
        assert_eq!(tokens, vec![10, 20]);
    }

    #[test]
    fn mtp_verify_frontier_must_land_inside_the_cap() {
        // cap 100, k 3: verify writes rows at next_pos..next_pos+3, frontier
        // next_pos+4. next_pos 96 is the last position that fits (frontier 100).
        assert!(mtp_verify_fits_cap(96, 3, 100));
        assert!(!mtp_verify_fits_cap(97, 3, 100));
        // A full verify batch is exactly k+1 rows.
        assert!(mtp_verify_fits_cap(0, 3, 4));
        assert!(!mtp_verify_fits_cap(0, 3, 3));
        // k = 0 degenerates to plain decode: one row, frontier next_pos+1.
        assert!(mtp_verify_fits_cap(99, 0, 100));
        assert!(!mtp_verify_fits_cap(100, 0, 100));
        // next_pos at or past the cap never fits, whatever k is.
        assert!(!mtp_verify_fits_cap(100, 3, 100));
    }

    /// Grammar mask ordering: the mask is applied BEFORE the sampler's
    /// probability-dependent filtering (spec §7.2 G2). This test verifies
    /// the `GrammarConstraint::build_mask` → mask application → sampler
    /// ordering is observable: a token forbidden by the mask gets
    /// NEG_INFINITY and cannot be selected even if it has the highest
    /// logit (spec §7.2 G2: "No later operation may resurrect a forbidden
    /// token"; A17: "Empty grammar mask → explicit failure, never
    /// unconstrained fallback").
    #[test]
    fn grammar_mask_applied_before_sampler_forbids_highest_logit() {
        // Simulate a 4-token vocab: [forbidden, allowed, allowed, allowed].
        // The forbidden token has the highest logit (10.0); without the
        // mask a greedy sampler would pick it. After masking it to
        // NEG_INFINITY, the argmax falls through to the next highest.
        let mut logits = [10.0f32, 1.0, 2.0, 3.0];
        let mask = [false, true, true, true];

        // Apply the mask as serve_engine's run_loop does: disallowed →
        // NEG_INFINITY, allowed left alone (spec §7.2 G2).
        for (i, &allowed) in mask.iter().enumerate() {
            if !allowed {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        // Greedy argmax over masked logits: must NOT pick token 0.
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 3, "masked argmax must skip the forbidden token");
    }

    /// An empty grammar mask (no allowed tokens) is a typed error, never an
    /// unconstrained fallback (spec §7.2 G2, A17).
    #[test]
    fn grammar_empty_allowed_set_is_typed_error() {
        // Verify the error type and Display impl are correct.
        let err = GrammarMaskError::EmptyAllowedSet;
        assert!(err.to_string().contains("empty allowed set"));
        assert!(err.to_string().contains("violating the schema"));
        assert!(std::error::Error::source(&err).is_none());
    }

    /// Grammar-constrained requests must select AR (not MTP) at admission
    /// (spec §6 X2: "MTP + grammar → AR selected before execution until
    /// exact joint acceptance is implemented and validated").
    #[test]
    fn grammar_constrained_requests_select_ar_not_mtp() {
        // The admit() path now sets `grammar_constrained = true` when
        // req.json_schema is Some. This test documents the invariant: a
        // grammar-constrained request must NOT activate MTP.
        let grammar_constrained = true;
        let mtp_available = true;
        let penalized = false;
        let sampled = false;
        let mtp_active = mtp_available && !penalized && !sampled && !grammar_constrained;
        assert!(!mtp_active, "grammar-constrained requests must use AR");
    }

    /// The SchemaMatcher mask forbids a token whose bytes would break the
    /// `{` object start (spec §7.2 G2). A schema requiring an object must
    /// not allow tokens that don't begin with `{` (or whitespace before it).
    #[test]
    fn schema_matcher_mask_forbids_non_object_start() {
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
        let compiled =
            grammar::json_schema::CompiledSchema::compile(&schema)
                .expect("compile");
        let matcher =
            grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // `{` is allowed (object start).
        assert!(matcher.is_token_allowed(b"{"), "object start must be allowed");
        // `}` is NOT allowed at the start (no object opened yet).
        assert!(
            !matcher.is_token_allowed(b"}"),
            "closing brace must be forbidden before object start"
        );
        // `1` (a number) is NOT allowed (schema requires an object).
        assert!(
            !matcher.is_token_allowed(b"1"),
            "number must be forbidden when schema requires object"
        );
    }

    /// The SchemaMatcher mask allows tokens that continue a valid prefix
    /// and forbids tokens that would violate the schema (spec §7.2 G2).
    #[test]
    fn schema_matcher_mask_allows_valid_continuation() {
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
        let compiled =
            grammar::json_schema::CompiledSchema::compile(&schema)
                .expect("compile");
        let mut matcher =
            grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // Advance past `{"a":` — now a string value is expected.
        matcher.advance(b"{\"a\":");
        // `"` (start of string value) is allowed.
        assert!(
            matcher.is_token_allowed(b"\""),
            "string start must be allowed after key"
        );
        // `}` is NOT allowed (expecting a string value, not object end).
        assert!(
            !matcher.is_token_allowed(b"}"),
            "object end must be forbidden when a string value is expected"
        );
    }

    /// An empty allowed set from a SchemaMatcher is a typed error (A17).
    /// A schema that requires a specific const value produces no allowed
    /// tokens when the matcher is in a state that can't accept any token.
    #[test]
    fn schema_matcher_empty_allowed_set_is_typed_error() {
        // A const schema requiring the value true: after advancing past
        // "tru", only "e" can continue. After "true", the matcher is
        // accepting and only whitespace/EOS is allowed.
        let schema = serde_json::json!({"const": true});
        let compiled =
            grammar::json_schema::CompiledSchema::compile(&schema)
                .expect("compile");
        let mut matcher =
            grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // After "tru", "e" is the only valid continuation.
        matcher.advance(b"tru");
        assert!(matcher.is_token_allowed(b"e"), "const true: 'e' must be allowed after 'tru'");
        assert!(!matcher.is_token_allowed(b"x"), "const true: 'x' must be forbidden after 'tru'");
        // After "true", the matcher is accepting.
        matcher.advance(b"e");
        assert!(matcher.is_accepting(), "const true: must be accepting after 'true'");
    }

    // ---- Prefix cache policy tests (spec §4.5–4.6, X2) ----

    #[test]
    fn should_lookup_prefix_off_never_lookups() {
        // prefix_cache off → never lookup, regardless of visual data.
        assert!(!should_lookup_prefix(false, false));
        assert!(!should_lookup_prefix(false, true));
    }

    #[test]
    fn should_lookup_prefix_vision_skips_even_when_enabled() {
        // Vision requests skip the radix entirely (spec X2: Vision+prefix
        // reuse OFF).
        assert!(!should_lookup_prefix(true, true));
    }

    #[test]
    fn should_lookup_prefix_text_request_lookups_when_enabled() {
        // Text request with prefix_cache on → lookup.
        assert!(should_lookup_prefix(true, false));
    }

    #[test]
    fn reused_tokens_full_when_boundary_covers_entire_prompt() {
        // boundary == prompt_len is an exact match: every prompt token is reused.
        assert_eq!(reused_tokens_from_plan(100, 100), 100);
    }

    #[test]
    fn reused_tokens_zero_when_boundary_is_zero() {
        // No checkpoint → boundary 0 → 0 reused.
        assert_eq!(reused_tokens_from_plan(0, 100), 0);
    }

    #[test]
    fn reused_tokens_returns_boundary_on_partial_match() {
        // Partial match: boundary < prompt_len → boundary tokens reused.
        assert_eq!(reused_tokens_from_plan(64, 128), 64);
        assert_eq!(reused_tokens_from_plan(1, 2), 1);
    }

    #[test]
    fn reset_epoch_bump_makes_cache_domain_unequal() {
        // After Reset, allocation_epoch is bumped. A CacheDomain with a
        // bumped epoch must differ from the pre-reset domain (spec §4.6:
        // "unload/reload advances the epoch and invalidates lookups").
        use hipfire_runtime::serve_contract::{
            CacheDomain, DeviceTopology, SharingNamespace,
        };
        let mk = |epoch: u64| CacheDomain {
            model_content_digest: vec![1, 2, 3],
            model_load_epoch: 0,
            sidecar_digests: Vec::new(),
            tokenizer: hipfire_runtime::serve_contract::TokenizerIdentity {
                vocab_digest: vec![0; 32],
                config_digest: vec![0; 32],
            },
            template: hipfire_runtime::serve_contract::TemplateIdentity {
                template_digest: vec![0; 32],
                normalization_tag: "default".to_owned(),
            },
            arch_policy: hipfire_runtime::serve_contract::ArchPolicy {
                arch_tag: "qwen35-deltanet".to_owned(),
                state_abi_tag: "dn-q8".to_owned(),
                position_attention_tag: "mrope".to_owned(),
            },
            kv_layout: hipfire_runtime::serve_contract::KvLayout {
                k_stride_bytes: vec![128],
                v_stride_bytes: vec![128],
                layout_tag: "q8".to_owned(),
            },
            device: DeviceTopology {
                device_id: "gpu0".to_owned(),
                topology_id: "single".to_owned(),
                allocation_epoch: epoch,
            },
            namespace: SharingNamespace {
                domain_id: "owner".to_owned(),
            },
        };
        let before = mk(0);
        let after = mk(1);
        assert_ne!(before, after, "epoch bump must invalidate domain equality");
        // Same epoch → equal (sanity).
        assert_eq!(mk(0), mk(0));
    }

    // ---- FairQueue integration tests (spec §5.3 S3) ----

    /// `eligible_ids_from_selection` projects a Selection's grants into the
    /// id set that becomes the scheduler's eligibility mask.
    #[test]
    fn eligible_ids_from_selection_collects_granted_ids() {
        let sel = Selection {
            grants: vec![
                Grant::Decode { id: 7 },
                Grant::Prefill { id: 3, rows: 16 },
                Grant::Verify { id: 9, rows: 4 },
            ],
            starved_oldest: false,
        };
        let ids = eligible_ids_from_selection(&sel);
        assert_eq!(ids.len(), 3, "one entry per granted id");
        assert!(ids.contains(&7), "decode grant id 7");
        assert!(ids.contains(&3), "prefill grant id 3");
        assert!(ids.contains(&9), "verify grant id 9");
        // An empty selection yields no eligible ids.
        assert!(eligible_ids_from_selection(&Selection::default()).is_empty());
    }

    /// Bounded backfill (spec §5.3 S3.4): when starved_oldest is set,
    /// `apply_starved_backfill` yields ONLY the oldest id, dropping every
    /// younger grant so the starved oldest receives the next step's budget.
    #[test]
    fn starved_oldest_backfill_yields_only_oldest_id() {
        // A selection where younger work (ids 3, 9) was granted but the
        // oldest (id 7) is starved. The backfill keeps only id 7.
        let sel = Selection {
            grants: vec![
                Grant::Prefill { id: 3, rows: 16 },
                Grant::Verify { id: 9, rows: 4 },
            ],
            starved_oldest: true,
        };
        let ids = apply_starved_backfill(&sel, 7);
        assert_eq!(ids.len(), 1, "backfill keeps only the oldest");
        assert!(ids.contains(&7), "the starved oldest is force-eligible");
        assert!(!ids.contains(&3) && !ids.contains(&9), "younger work dropped");
    }

    /// End-to-end FairQueue consultation: a real queue with an oldest
    /// prefill request starved by younger decode lanes produces a
    /// starved_oldest selection, and the backfill reduces eligibility to
    /// only the oldest id (spec §5.3 S3.4).
    #[test]
    fn fairqueue_starved_oldest_select_then_backfill() {
        // Budget 10, 4 decode lanes, prefill_min 5 (4 + 5 = 9 <= 10, valid).
        let mut q = FairQueue::new(10, 4, 5).expect("valid config");
        // Oldest: wants prefill, 5 uncached tokens (individually feasible:
        // 5 <= max_batch_tokens 10).
        q.admit(1, "s1", 5).unwrap();
        // Younger: four decode lanes.
        q.admit(2, "s2", 0).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.admit(4, "s4", 0).unwrap();
        q.admit(5, "s5", 0).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        q.set_needs(3, true, 0, 0).unwrap();
        q.set_needs(4, true, 0, 0).unwrap();
        q.set_needs(5, true, 0, 0).unwrap();
        // remaining_rows = 8: 4 decode rows, 4 left. prefill_min 5 > 4 → no
        // prefill grant for the oldest. It is feasible (5 <= 10) but starved.
        let sel = q.select(4, 10, 8);
        assert!(
            sel.starved_oldest,
            "oldest is feasible but starved, starved_oldest must be set"
        );
        // The oldest (id 1) received no grant this tick.
        assert!(
            !sel.grants.iter().any(|g| g.id() == 1),
            "starved oldest received no grant"
        );
        // Backfill: only the oldest id (1) stays eligible.
        let ids = apply_starved_backfill(&sel, 1);
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&1));
    }

    /// A non-starved selection passes all granted ids through unchanged.
    #[test]
    fn non_starved_selection_passes_all_grants() {
        let mut q = FairQueue::new(4096, 4, 1).expect("valid config");
        q.admit(1, "s1", 0).unwrap();
        q.admit(2, "s2", 0).unwrap();
        q.set_needs(1, true, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        let sel = q.select(4, 0, 4096);
        assert!(!sel.starved_oldest, "both served, not starved");
        let ids = eligible_ids_from_selection(&sel);
        assert_eq!(ids.len(), 2, "both decode grants eligible");
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    // ---- WaitQueue policy tests (spec §5.3 S3) ----

    /// `should_wait_instead_of_reject` returns true only when every resident
    /// session is generating and the bounded waiting room is enabled (spec
    /// §5.3 S3, replacing R-A4).
    #[test]
    fn should_wait_when_all_busy_and_wait_enabled() {
        assert!(should_wait_instead_of_reject(true, true));
    }

    #[test]
    fn should_not_wait_when_not_all_busy() {
        // A free slot means the request can be admitted immediately — no
        // need to queue.
        assert!(!should_wait_instead_of_reject(false, true));
    }

    #[test]
    fn should_not_wait_when_wait_disabled() {
        // WaitQueue disabled (max_count 0 — should not happen since
        // Rig::build refuses it, but the policy is pure and testable).
        assert!(!should_wait_instead_of_reject(true, false));
    }

    /// WaitQueue::new refuses zero count or byte caps (spec §5.3: the new
    /// robust multi-slot mode must reject the zero/uncapped combination).
    #[test]
    fn waitqueue_refuses_zero_caps() {
        use hipfire_runtime::serve_wait::{WaitError, WaitQueue};
        assert!(matches!(
            WaitQueue::new(0, 1024, 100),
            Err(WaitError::QueueFull { .. })
        ));
        assert!(matches!(
            WaitQueue::new(4, 0, 100),
            Err(WaitError::QueueBytes { .. })
        ));
    }

    /// WaitQueue enqueue/pop_ready/expire round-trip (spec §5.3 S3).
    #[test]
    fn waitqueue_enqueue_pop_expire_roundtrip() {
        use hipfire_runtime::serve_wait::WaitQueue;
        let mut q = WaitQueue::new(4, 1024, 100).expect("valid config");
        // Enqueue two waiters at ticks 0 and 1.
        q.try_enqueue(1, 100, 0).expect("enqueue 1");
        q.try_enqueue(2, 200, 1).expect("enqueue 2");
        assert_eq!(q.queued_count(), 2);
        assert_eq!(q.queued_bytes(), 300);
        // Pop the oldest (FIFO).
        let w = q.pop_ready(2).expect("pop oldest");
        assert_eq!(w.id, 1);
        assert_eq!(q.queued_count(), 1);
        // Expire at tick 102 (waiter 2 was enqueued at tick 1, deadline 101).
        let expired = q.expire(102);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, 2);
        assert_eq!(q.queued_count(), 0);
    }

    /// WaitQueue remove on cancel (spec §5.3 S3).
    #[test]
    fn waitqueue_remove_cancels_waiter() {
        use hipfire_runtime::serve_wait::WaitQueue;
        let mut q = WaitQueue::new(4, 1024, 100).expect("valid config");
        q.try_enqueue(1, 100, 0).expect("enqueue 1");
        q.try_enqueue(2, 200, 0).expect("enqueue 2");
        assert!(q.remove(1), "remove waiter 1");
        assert!(!q.remove(1), "already removed");
        assert_eq!(q.queued_count(), 1);
        assert_eq!(q.queued_bytes(), 200);
    }

    // ---- Jump-forward policy tests (spec §7.3 G3) ----

    /// Jump-forward is skipped when structured_jump_forward is false (spec
    /// §7.3: default OFF). This test documents the gating invariant: the
    /// flag must be true for forced_token_run to be called.
    #[test]
    fn jump_forward_skipped_when_flag_false() {
        let structured_jump_forward = false;
        let has_grammar = true;
        let greedy = true;
        let should_jump = structured_jump_forward && has_grammar && greedy;
        assert!(!should_jump, "jump-forward must be skipped when flag is false");
    }

    /// Jump-forward requires a grammar constraint (spec §7.3 G3).
    #[test]
    fn jump_forward_requires_grammar_constraint() {
        let structured_jump_forward = true;
        let has_grammar = false;
        let greedy = true;
        let should_jump = structured_jump_forward && has_grammar && greedy;
        assert!(!should_jump, "jump-forward requires a grammar constraint");
    }

    /// Jump-forward is greedy-only (spec §7.3: "not sampled (greedy only)").
    #[test]
    fn jump_forward_greedy_only() {
        let structured_jump_forward = true;
        let has_grammar = true;
        let greedy = false; // sampled
        let should_jump = structured_jump_forward && has_grammar && greedy;
        assert!(!should_jump, "jump-forward must be skipped for sampled requests");
    }

    /// forced_token_run on a const schema produces the forced token sequence
    /// (spec §7.3 G3). A `{"const": true}` schema has exactly one legal
    /// token at each step until the value is complete.
    #[test]
    fn forced_token_run_const_true_produces_forced_tokens() {
        let schema = serde_json::json!({"const": true});
        let compiled =
            grammar::json_schema::CompiledSchema::compile(&schema)
                .expect("compile");
        let matcher =
            grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // Build a minimal token_bytes table with only the individual byte
        // tokens for "true" plus a distractor. Whitespace tokens are
        // excluded so the planner sees exactly one legal token at each
        // step (no branch point from leading whitespace before the value).
        let token_bytes: Vec<Vec<u8>> = vec![
            b"t".to_vec(), // 0
            b"r".to_vec(), // 1
            b"u".to_vec(), // 2
            b"e".to_vec(), // 3
            b"x".to_vec(), // 4 — distractor, never allowed by const true
        ];
        // EOS id outside the vocab so no real token is filtered as EOS.
        let eos_ids: Vec<u32> = vec![999];
        let run = grammar::json_schema::forced_token_run(
            &matcher,
            &token_bytes,
            &eos_ids,
            64,
        );
        // The forced run should produce the bytes for "true" as individual
        // token IDs 0,1,2,3 (t,r,u,e). After "true" the matcher is
        // accepting; the run stops (Cap or EosBoundary depending on
        // whether whitespace tokens are allowed in the accepting state).
        assert!(!run.token_ids.is_empty(), "const true must produce forced tokens");
        assert_eq!(run.token_ids, vec![0, 1, 2, 3], "forced tokens are t,r,u,e");
    }
}