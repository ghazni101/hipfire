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

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use hipfire_runtime::admission::{AdmissionController, ModelFootprint};
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
use crate::slot_batch::SlotBatch;
use crate::qwen35::{
    self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch, Qwen35Weights,
};
use crate::scheduler::{PendingWork, Scheduler, VlPrefill};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::llama::VMode;

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
struct Rig {
    gpu: Gpu,
    weights: Qwen35Weights,
    config: qwen35::Qwen35Config,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    pool: SlotPool,
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
    sessions: SessionTable,
    adm: AdmissionController,
    swap: SwapManager,
    stamp: SnapshotStamp,
    n_slots: usize,
    cap_tokens: usize,
    prefill_chunk: usize,
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
        let per_pos_bytes = config.n_kv_heads * (config.head_dim / 32) * 34;
        let prefill_chunk = cfg.prefill_chunk.max(1).min(cfg.cap_tokens.max(1));
        let max_batch = (prefill_chunk * cfg.n_slots).max(cfg.n_slots);

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
        let paged_pages = match hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED")
            .ok()
            .as_deref()
        {
            Some("1") | Some("on") | Some("true") if !cfg.is_vl => {
                let legacy_equivalent = cfg.n_slots * cfg.cap_tokens.div_ceil(PAGE_TOKENS);
                Some(
                    hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED_PAGES")
                        .ok()
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(legacy_equivalent),
                )
            }
            Some("1") | Some("on") | Some("true") if cfg.is_vl => {
                eprintln!(
                    "  [hipfire] paged KV disabled for VL model \
                     (vision sequential path requires contiguous KV slots)"
                );
                None
            }
            _ => None,
        };

        let weight_bytes = std::fs::metadata(&cfg.model_path)
            .map_err(|e| format!("stat model: {e}"))?
            .len();
        let cap_rounded = cfg.cap_tokens.div_ceil(128) * 128;
        let kv_pages = paged_pages.unwrap_or(cfg.n_slots * cap_rounded / 128);
        let kv_bytes = (n_fa_layers as u64)
            * 2
            * (kv_pages as u64)
            * (PAGE_TOKENS as u64)
            * (per_pos_bytes as u64);
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
        // MTP is incompatible with a paged pool: the sequential MTP path
        // (prompt prefill and the post-accept KV/DN replay) writes through the
        // flat per-slot slab view (`slot_kv_view`), whose `legacy_k_base` is 0
        // for every slot in paged mode — prefill would land in the arena
        // prefix instead of the slot's own pages, while the batched verify
        // (paged-aware) reads the slot's real pages. Silent garbage + cross-
        // slot corruption. The batching itself has no paged fix yet, so the
        // combination is refused at load: paged wins, MTP stays off (mirror
        // of the VL gate above). Zeroing mtp_k also skips the head load.
        let mtp_k = if paged_pages.is_some() && cfg.mtp_k > 0 {
            eprintln!(
                "  [hipfire] MTP disabled: paged KV pool is incompatible with \
                 the sequential MTP path (flat slab view)"
            );
            0
        } else {
            cfg.mtp_k
        };
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
                    let sidecar = trunk_path.with_extension("mtp");
                    if sidecar.exists() {
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
                    let sidecar = trunk_path.with_extension("mtp");
                    if sidecar.exists() {
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
            // The sidecar probe replaces the trunk's final extension
            // (foo.mq4v2.hfq → foo.mq4v2.mtp). A sidecar named after a
            // different trunk spelling — or any miss — must not silently
            // degrade to AR: that is the "draft pulled but DFlash off" trap.
            eprintln!(
                "  [hipfire] MTP requested (mtp_k={mtp_k}) but no head was found: \
                 no bundled .mq4-mtp trailer and no sidecar at {} — \
                 serving without MTP",
                cfg.model_path.with_extension("mtp").display()
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
            dn_states: Vec::with_capacity(cfg.n_slots),
            desc_staging: None,
            pbs: None,
            scratch: None,
            logits_out: None,
            out_tokens: None,
            committed: false,
        };
        let pool = if let Some(pages) = paged_pages {
            SlotPool::new_paged(cfg.n_slots, cfg.cap_tokens, per_pos_bytes, pages)
        } else {
            SlotPool::new(cfg.n_slots, cfg.cap_tokens, per_pos_bytes)
        }
        .map_err(|e| format!("SlotPool: {e}"))?;
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
        let arena_bytes = pool.arena_bytes();
        for _ in 0..n_fa_layers {
            let k = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[arena_bytes], DType::Raw)
                .map_err(|e| format!("k arena: {e}"))?;
            g.k_arenas.push(k);
            let v = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[arena_bytes], DType::Raw)
                .map_err(|e| format!("v arena: {e}"))?;
            g.v_arenas.push(v);
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
            })
            .collect::<Vec<_>>();

        let dn_bytes: u64 = dn_buffers(&g.dn_states[0])
            .iter()
            .map(|t| t.buf.size() as u64)
            .sum();
        let stamp = SnapshotStamp {
            model_hash: weight_bytes,
            kv_dtype_tag: 1,
            per_pos_bytes: per_pos_bytes as u32,
            n_fa_layers: n_fa_layers as u32,
            dn_layout_version: 1,
            cap: pool.cap_tokens() as u32,
            dn_bytes,
        };

        let mut adm = AdmissionController::new(
            ModelFootprint {
                weights_bytes: weight_bytes,
                kv_bytes_per_token: (n_fa_layers * 2 * per_pos_bytes) as u64,
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
        // g now committed and empty; Drop is no-op.

        Ok(Rig {
            gpu,
            weights,
            config,
            tokenizer,
            pool,
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
            sessions: SessionTable::default(),
            adm,
            swap,
            stamp,
            n_slots: cfg.n_slots,
            cap_tokens: cfg.cap_tokens,
            prefill_chunk,
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
            k_arenas,
            v_arenas,
            dn_states,
            desc_staging,
            pbs,
            scratch,
            logits_out,
            out_tokens,
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
/// over the slot's verify rows (a view of `pbs.x_batch`), greedy-accept, and
/// repair the DeltaNet state on partial accepts.
///
/// `hidden_row_offset` is the flat row index where this slot's verify rows
/// begin in the step batch (computed from `batch.m_per_slot` by the caller).
/// Returns committed tokens (excludes seed, includes bonus).
fn mtp_verify_accept_step(
    rig: &mut Rig,
    slot: SlotId,
    work: &mut PendingWork,
    draft: crate::mtp_spec::MtpDraftOutput,
    hidden_row_offset: usize,
) -> Result<Vec<u32>, String> {
    let mut state = rig
        .mtp_states[slot.0]
        .take()
        .expect("mtp state must exist from draft phase");

    let pos = work.next_pos;

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
fn inject_mtp_verify_tokens(
    batch: &SlotBatch,
    mtp_drafts: &[Option<crate::mtp_spec::MtpDraftOutput>],
    n_slots: usize,
) -> SlotBatch {
    let mut new_batch = SlotBatch::default();
    let mut old_row_off = 0usize;
    for s in 0..n_slots {
        if let Some(draft) = &mtp_drafts[s] {
            let verify_tokens = draft.verify_tokens();
            new_batch.m_per_slot.push(verify_tokens.len());
            for (i, t) in verify_tokens.iter().enumerate() {
                new_batch.tokens.push(*t);
                new_batch.positions.push((draft.cur_pos + i) as i32);
                new_batch.row_slot.push(s as i32);
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

fn commit_sampled_token(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    s: usize,
    tok: u32,
) {
    let Some(f) = slots[s].as_mut() else { return };
    let session = f.session;
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
        let _ = send_event(&f.reply, Event::Done { reason, generated });
        slots[s] = None;
        clear_work_slot(&mut work[s]);
        if matches!(reason, DoneReason::ClientGone) {
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
        })
        .collect();
    let mut sched = Scheduler {
        chunk_size: rig.prefill_chunk,
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
                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                clear_work_slot(&mut work[s]);
            }
        }
    };

    'serve: loop {
        let idle = slots.iter().all(|s| s.is_none());
        if idle {
            // Nothing in flight: block rather than spin.
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
        if slots.iter().all(|s| s.is_none()) {
            continue;
        }

        let image_pad_id = rig.tokenizer.special_token_id("<|image_pad|>");
        for s in 0..n {
            if work[s].vl_prefill.is_none() || slots[s].is_none() {
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
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    }
                    clear_work_slot(&mut work[s]);
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
                                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                            }
                            clear_work_slot(&mut work[s]);
                        }
                    }
                }
            }
        }

        // Build the batch: scheduler handles regular slots (including MTP
        // prefill chunks), then inject MTP verify tokens for slots that
        // produced draft outputs.
        let sched_batch = sched.next_batch(&mut work);
        let batch = if mtp_drafts.iter().any(|d| d.is_some()) {
            inject_mtp_verify_tokens(&sched_batch, &mtp_drafts, n)
        } else {
            sched_batch
        };
        if batch.is_empty() {
            continue;
        }
        // Slots with verify drafts skip the per-slot last-row lm_head GEMV
        // inside the forward: their logits come from the batched verify
        // lm_head over all k+1 rows instead, so the single-row GEMV (a full
        // lm_head weight re-read) would be discarded work.
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
                &rig.pbs,
                &rig.scratch,
                &rig.logits_out,
                &mut graph,
                rig.mtp_k,
                &lm_head_skip,
                capture.as_mut(),
            )
        })();
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
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    work[s].remaining_prompt.clear();
                    work[s].decoding = false;
                    work[s].next_pos = 0;
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
        if let Err(e) = rig.gpu.sample_per_slot(
            &rig.logits_out,
            &mut rig.sample_params,
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
                match mtp_verify_accept_step(
                    &mut rig,
                    SlotId(s),
                    &mut work[s],
                    draft,
                    hidden_row_offset,
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
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        }
                        clear_work_slot(&mut work[s]);
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
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        }
                        clear_work_slot(&mut work[s]);
                    }
                }
                continue;
            }
            if work[s].vl_prefill.is_some() || work[s].mtp_active {
                continue;
            }
            let Some(f) = slots[s].as_mut() else { continue };
            // Still prefilling: these logits belong to a mid-prompt token.
            if !work[s].remaining_prompt.is_empty() {
                continue;
            }
            let tok = ids[s] as u32;
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
                let _ = send_event(&f.reply, Event::Done { reason, generated });
                slots[s] = None;
                clear_work_slot(&mut work[s]);
                if matches!(reason, DoneReason::ClientGone) {
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
            let ids: Vec<u64> = rig.sessions.iter().map(|(id, _)| id).collect();
            for id in ids {
                rig.swap.forget(id);
                rig.sessions
                    .close(&mut rig.pool, &mut rig.adm, SessionId(id));
            }
            let _ = reply.send(Ok(()));
        }
    }
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
        if rig.pool.is_paged() {
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
                    let _ = send_event(
                        &req.reply,
                        Event::Done {
                            reason: DoneReason::MaxTokens,
                            generated: 0,
                        },
                    );
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
                    // MTP: activate if enabled. Vision continuations stay on
                    // the sequential M-RoPE path (no MTP), matching cold
                    // admits. The head KV reset is only needed when the head
                    // KV was lost (swap-restore) or holds another session's
                    // rows: a session resident on this slot since its last
                    // turn keeps a head KV that matches its committed prefix,
                    // so the suffix's head-fill extends it in place.
                    work[slot.0].mtp_active = rig.mtp_head.is_some()
                        && rig.mtp_k > 0
                        && req.visual_data.is_none();
                    if work[slot.0].mtp_active
                        && !(mtp_head_kv_valid && rig.mtp_states[slot.0].is_some())
                    {
                        if let Some(state) = rig.mtp_states[slot.0].as_mut() {
                            let _ = state.reset(&mut rig.gpu);
                        }
                    }
                    rig.sample_params[slot.0] = SlotSampleParams {
                        temperature: req.temperature,
                        top_p: req.top_p,
                        top_k: req.top_k,
                        seed: req.seed,
                    };
                    slots[slot.0] = Some(InFlight {
                        session: existing,
                        reply: req.reply,
                        produced: 0,
                        max_tokens: req.max_tokens.max(1),
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

    // Prefill-side context-cap guard for cold admits. Same wire reporting as
    // the decode path (DoneReason::MaxTokens → finish_reason "length").
    if req.prompt_tokens.len() >= rig.cap_tokens {
        let _ = send_event(
            &req.reply,
            Event::Done {
                reason: DoneReason::MaxTokens,
                generated: 0,
            },
        );
        return;
    }

    // Try to open; if the pool is full, evict the LRU idle session first.
    let id = match rig
        .sessions
        .open(&mut rig.pool, &mut rig.adm, rig.cap_tokens)
    {
        Ok(id) => id,
        Err(_) => {
            let victim = rig.sessions.lru_idle_victim(&busy);
            match victim {
                Some(v) => {
                    if !evict(rig, v) {
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                reason: "eviction failed".to_string(),
                            },
                        );
                        stats.lock().expect("stats").note_rejected();
                        return;
                    }
                    stats.lock().expect("stats").note_eviction();
                    match rig
                        .sessions
                        .open(&mut rig.pool, &mut rig.adm, rig.cap_tokens)
                    {
                        Ok(id) => id,
                        Err(e) => {
                            let _ = send_event(
                                &req.reply,
                                Event::Rejected {
                                    reason: format!("{e:?}"),
                                },
                            );
                            stats.lock().expect("stats").note_rejected();
                            return;
                        }
                    }
                }
                None => {
                    // Every resident session is generating. Reject with a
                    // reason rather than preempting one or queueing forever.
                    let _ = send_event(
                        &req.reply,
                        Event::Rejected {
                            reason: "all slots busy".to_string(),
                        },
                    );
                    stats.lock().expect("stats").note_rejected();
                    return;
                }
            }
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

    // Prefix reuse: a fresh session reuses nothing, a continued one reuses its
    // common prefix. Either way the suffix is what gets prefilled.
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
    if let Some(sess) = rig.sessions.get_mut(id) {
        sess.tokens = req.prompt_tokens.clone();
        sess.convo = req.convo.clone();
    }

    if send_event(
        &req.reply,
        Event::Accepted {
            session: id.0,
            reused: plan.reused,
            prefill: req.prompt_tokens.len() - plan.reused,
        },
    )
    .is_err()
    {
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        return;
    }

    work[slot.0].decoding = false;
    if let Some(vd) = req.visual_data {
        work[slot.0].remaining_prompt = req.prompt_tokens.clone();
        work[slot.0].next_pos = 0;
        work[slot.0].vl_prefill = Some(VlPrefill {
            patches: vd.patches,
            grid_h: vd.grid_h,
            grid_w: vd.grid_w,
            n_visual_tokens: vd.n_visual_tokens,
            embeddings: Vec::new(),
            dim: rig.config.dim,
            visual_idx: 0,
            mrope_positions: vd.mrope_positions,
            rope_delta: vd.rope_delta,
            base: 0,
        });
        work[slot.0].mtp_active = false;
    } else {
        work[slot.0].remaining_prompt = req.prompt_tokens[plan.reused..].to_vec();
        work[slot.0].next_pos = plan.reused;
        work[slot.0].vl_prefill = None;
        // Activate MTP for text-only requests when the head is loaded.
        work[slot.0].mtp_active = rig.mtp_head.is_some() && rig.mtp_k > 0;
    }
    rig.sample_params[slot.0] = SlotSampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
    };
    slots[slot.0] = Some(InFlight {
        session: id,
        reply: req.reply,
        produced: 0,
        max_tokens: req.max_tokens.max(1),
    });
    if rig.gpu.slot_trace() {
        eprintln!(
            "[slot-trace] cold ADMIT session {} slot={} prompt={} tokens",
            id.0,
            slot.0,
            req.prompt_tokens.len()
        );
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
}
