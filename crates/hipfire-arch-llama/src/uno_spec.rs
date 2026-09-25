// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Conditional-LoRA Uno linear windows, using the base target as verifier.
//! Reference: https://github.com/ifm-ai/uno (Apache-2.0).
use crate::{uno::{uno_forward_row, uno_forward_batch, uno_forward_batch_device, uno_upload_window, UnoAdapter, UnoScratch, UnoBatchScratch}, LlamaBundle};
use hip_bridge::{Graph, GraphExec};
use hipfire_runtime::{llama, spec::{request_rng_state, terminal_prefix_replay, PrefillOutcome, SpecAdvance, SpecGrammar, SpecRequestConfig, SpecStep, SpecTarget, Speculator}};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Draft noise law, matching upstream nano_vllm_uno/engine/noise.py.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NoiseMode {
    /// `torch.randint` in `[noise_low, noise_high)` — the law the adapter was
    /// trained on (upstream default).
    RandomUniform,
    /// Deterministic splitmix64 stream mixed with the window seed token
    /// (upstream `_make_deterministic_noise` semantics: reproducible for a
    /// given request/window).
    DeterministicUniform,
    /// Constant `noise_high` (= mask token id) rows (upstream `_make_noise`
    /// mask mode). Out-of-distribution for the adapter; kept for parity.
    Mask,
}

impl NoiseMode {
    fn parse(s: &str) -> NoiseMode {
        match s {
            "deterministic_uniform" => NoiseMode::DeterministicUniform,
            "mask" => NoiseMode::Mask,
            _ => NoiseMode::RandomUniform,
        }
    }
}

pub struct UnoSpeculator {
    adapter: UnoAdapter,
    scratch: UnoScratch,
    batch: Option<UnoBatchScratch>,
    capacity: usize,
    noise_mode: NoiseMode,
    request: SpecRequestConfig,
    /// Model EOS plus tokenizer terminators (`<|ifm|endoftext|>`, `<|ifm|im_end|>`).
    stop_ids: Vec<u32>,
    rng: u32,
    /// Ψ-Spec tree verify config: (max tree nodes incl. root, candidate top-k
    /// per depth). `None` = linear chain windows.
    tree: Option<(usize, usize)>,
    /// Upstream `_sequence_noise_seed`: splitmix64 hash of the prompt tokens.
    prompt_noise_seed: u64,
    /// Upstream `_noise_salt`: per-request salt (request RNG seed).
    noise_salt: u64,
    /// Diffusion-training mask token (`--mask-token-id`, K2 = vocab_size).
    mask_token: u32,
    /// Captured draft (LoRA) / verify (base) window graphs. Replay is one
    /// launch; capture uses a 256-token LDS cap so short-ctx windows don't
    /// bake physical_cap=200k into shared memory.
    draft_graph: Option<(Graph, GraphExec, Vec<Vec<u8>>)>,
    verify_graph: Option<(Graph, GraphExec, Vec<Vec<u8>>)>,
    graph_draft_warm: bool,
    graph_verify_warm: bool,
    /// Last up-to-64 prompt tokens of the current request. AR scopes its repeat
    /// history to the turn's prompt (`ar.rs` repeat upload uses
    /// `conversation_tokens[max(scope_start, len - min(repeat_window, 64))..]`),
    /// so the verifier needs the prompt tail to reproduce the penalty exactly.
    prompt_tail: Vec<u32>,
    /// Device ring for `Gpu::apply_repeat_penalty_row` — 64 slots holding token
    /// ids as `u32`, matching `ForwardScratch::repeat_buf`. `None` only if the
    /// allocation failed at load.
    repeat_buf: Option<GpuTensor>,
}

impl UnoSpeculator {
    pub fn load(gpu: &mut Gpu, target: &LlamaBundle, dir: &std::path::Path, capacity: usize, stop_ids: &[u32]) -> Result<Box<dyn Speculator>, String> {
        let c = &target.config;
        if c.dim != 4096 || c.n_layers != 36 || c.n_heads != 32 || c.n_kv_heads != 8 || c.head_dim != 128 || c.norm_groups != 4 || c.vocab_size != 250624 {
            return Err("Uno requires the K2-Horizon-7B target".into());
        }
        // Diffusion block length L (window = L rows: seed + L-1 noise).
        // Default 4: the 8-row xbatch GEMV drops to 4 waves/SIMD and loses to
        // 1-token AR; 4-row keeps 8-wave occupancy. Override with HIPFIRE_UNO_BLOCK.
        let block_len = parse_block_len(hipfire_config::developer_var("HIPFIRE_UNO_BLOCK").ok().as_deref());
        // Ψ-Spec tree verify (upstream "tree sampler for high per-request
        // throughput"): HIPFIRE_UNO_TREE = max tree nodes incl. root (0 =
        // linear chain), HIPFIRE_UNO_TREE_K = candidates exposed per depth.
        // Config default is 16 (`nano_vllm_uno.config.tree_candidate_top_k`);
        // the inference CLI default is 32. Stay with the engine default.
        let tree_nodes = hipfire_config::developer_var("HIPFIRE_UNO_TREE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.clamp(0, 64))
            .unwrap_or(0);
        let tree_top_k = hipfire_config::developer_var("HIPFIRE_UNO_TREE_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|k| k.clamp(1, 32))
            .unwrap_or(16);
        let tree = (tree_nodes >= 2).then_some((tree_nodes, tree_top_k));
        let noise_mode = hipfire_config::developer_var("HIPFIRE_UNO_NOISE")
            .ok()
            .map(|s| NoiseMode::parse(&s))
            .unwrap_or(NoiseMode::RandomUniform);
        let batch_rows = tree.map_or(block_len, |(nodes, _)| nodes.max(block_len));
        // Upstream `_noise_bounds`: uniform draws from `[1, min(mask, vocab))`.
        // K2 recipe `--mask-token-id 250624` equals vocab_size, so the half-open
        // range is `[1, 250624)`. Mask mode fills with the mask token itself.
        let mask_token = c.vocab_size as u32;
        let adapter = UnoAdapter::open(dir, c, gpu, block_len, 1, mask_token)?;
        let scratch = match UnoScratch::new(gpu, c) {
            Ok(s) => s,
            Err(e) => { adapter.free_gpu(gpu); return Err(format!("Uno scratch: {e:?}")); }
        };
        let kv_ok = target.kv.quant_q8
            || target.kv.quant_asym2
            || target.kv.quant_asym3
            || target.kv.quant_asym4;
        let batch = if llama::mq4g256v2_window_batch_ok(&gpu.arch) && kv_ok {
            match UnoBatchScratch::new(gpu, c, adapter.rank, batch_rows, target.kv.physical_cap) {
                Ok(batch) => Some(batch),
                Err(e) => { scratch.free_gpu(gpu); adapter.free_gpu(gpu); return Err(format!("Uno batch scratch: {e:?}")); }
            }
        } else { None };
        if batch.is_none() {
            eprintln!(
                "  Uno: batch scratch not allocated (need gfx11/gfx12 + quantized KV + K2-Horizon-7B); verify downloads full-vocab logits (arch={} q8={})",
                gpu.arch, target.kv.quant_q8
            );
        } else {
            eprintln!(
                "  Uno: device-resident batch path (arch={} rows={} q8={}) — logits stay on GPU",
                gpu.arch, batch_rows, target.kv.quant_q8
            );
        }
        let mut stop_ids: Vec<u32> = stop_ids.iter().copied().filter(|&id| (id as usize) < c.vocab_size).collect();
        stop_ids.sort_unstable();
        stop_ids.dedup();
        if stop_ids.is_empty() {
            stop_ids.push(c.eos_token);
        }
        if tree.is_some() && !target.kv.quant_q8 {
            scratch.free_gpu(gpu);
            adapter.free_gpu(gpu);
            if let Some(batch) = batch { batch.free_gpu(gpu); }
            return Err("Uno tree sampler requires Q8 KV (compact_tree_kv is Q8_0)".into());
        }
        Ok(Box::new(Self {
            adapter, scratch, batch, capacity,
            noise_mode,
            request: SpecRequestConfig::default(),
            stop_ids, rng: 0x1357_9BDF, tree,
            prompt_noise_seed: 0xD6E8_FEB8_6659_FD93,
            noise_salt: 0,
            mask_token,
            draft_graph: None, verify_graph: None,
            graph_draft_warm: false, graph_verify_warm: false,
            prompt_tail: Vec::new(),
            repeat_buf: Some(gpu.alloc_tensor(&[64], DType::F32).map_err(|e| format!("Uno repeat buffer: {e:?}"))?),
        }))
    }

    fn stops(&self, token: u32) -> bool {
        self.stop_ids.contains(&token)
    }

    /// AR's Phase-0 repeat/presence/frequency penalty applied in place to one
    /// logits row, before that row's argmax/sampling.
    ///
    /// This is what makes greedy spec decode lossless versus AR: AR scales the
    /// logits of tokens inside the repeat window *before* its argmax
    /// (`sample_apply_repeat_penalty`), and the shipped `repeat_penalty` default
    /// is 1.05 — non-neutral — so a verifier that skips this emits different
    /// tokens than AR. `history` is oldest-first and already truncated to the
    /// window, matching what AR uploads into `ForwardScratch::repeat_buf`.
    ///
    /// No-op when every penalty is neutral, so the common case costs nothing.
    fn penalize_row(
        &self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        row: usize,
        vocab: usize,
        history: &[u32],
    ) -> Result<(), String> {
        let r = &self.request;
        if history.is_empty()
            || (r.repeat_penalty <= 1.0 && r.presence_penalty <= 0.0 && r.frequency_penalty <= 0.0)
        {
            return Ok(());
        }
        let buf = self.repeat_buf.as_ref().ok_or("uno: repeat buffer missing")?;
        // The kernel reads `const unsigned int*`, so token ids go up as u32
        // bytes (same as AR's upload into the nominally-F32 repeat_buf).
        let bytes: Vec<u8> = history.iter().flat_map(|t| t.to_ne_bytes()).collect();
        gpu.hip
            .memcpy_htod(&buf.buf, &bytes)
            .map_err(|e| format!("uno repeat upload: {e:?}"))?;
        gpu.apply_repeat_penalty_row(
            &logits.sub_offset(row * vocab, vocab),
            buf,
            vocab,
            history.len(),
            r.repeat_penalty,
            r.presence_penalty,
            r.frequency_penalty,
        )
        .map_err(|e| format!("uno repeat penalty: {e:?}"))
    }

    fn forward(&mut self, gpu: &mut Gpu, target: &mut LlamaBundle, tokens: &[u32], pos: usize, gated: bool) -> Result<Vec<f32>, String> {
        if let Some(batch) = &self.batch {
            return uno_forward_batch(gpu, &target.weights, &target.config,
                gated.then_some(&self.adapter), tokens, pos, &mut target.kv,
                &target.scratch, batch).map_err(|e| format!("Uno batch: {e:?}"));
        }
        let mut logits = Vec::with_capacity(tokens.len() * target.config.vocab_size);
        for (row, &token) in tokens.iter().enumerate() {
            if gated && row > 0 {
                logits.extend(uno_forward_row(gpu, &target.weights, &target.config,
                    &self.adapter, token, pos + row, true, &mut target.kv,
                    &target.scratch, &mut self.scratch).map_err(|e| format!("Uno draft: {e:?}"))?);
            } else {
                llama::forward_scratch_embed(gpu, &target.weights, &target.config, token, pos + row, &target.scratch).map_err(|e| format!("Uno embed: {e:?}"))?;
                llama::forward_scratch_compute(gpu, &target.weights, &target.config, pos + row, &mut target.kv, &target.scratch).map_err(|e| format!("Uno forward: {e:?}"))?;
                logits.extend(gpu.download_f32(&target.scratch.logits).map_err(|e| format!("Uno logits: {e:?}"))?);
            }
        }
        Ok(logits)
    }

    fn sample_target(&mut self, gpu: &mut Gpu, target: &LlamaBundle) -> Result<u32, String> {
        let logits = gpu.download_f32(&target.scratch.logits).map_err(|e| format!("Uno prefill logits: {e:?}"))?;
        Ok(Distribution::from_logits(&logits, self.request)?.sample(&mut self.rng))
    }

    /// Draft noise rows. `HIPFIRE_UNO_NOISE` selects the upstream noise mode
    /// (noise.py): `random_uniform` (default; the law the conditional LoRA was
    /// trained on — tokens from `[noise_low, noise_high)`), `deterministic_uniform`
    /// (reproducible splitmix64 stream mixed with the window seed token), or
    /// `mask` (constant `noise_high` = mask-token rows — a non-default choice
    /// that is out-of-distribution for the adapter and collapses deep-row
    /// acceptance).
    fn draw_noise(&mut self, n: usize, seed_token: u32, seq_len: usize, n_completion: usize) -> Vec<u32> {
        let low = self.adapter.noise_low;
        let high = self.adapter.noise_high;
        match self.noise_mode {
            // Upstream mask mode fills with `_noise_bounds().1` which equals
            // the mask token. Clamp into the embedding so a vocab-sized mask
            // id cannot OOB the lookup; uniform still draws from `[low, high)`.
            NoiseMode::Mask => vec![self.mask_token.min(high.saturating_sub(1)).max(low); n],
            NoiseMode::RandomUniform => random_uniform_noise(low, high, n, &mut self.rng),
            NoiseMode::DeterministicUniform => deterministic_uniform_noise(
                low, high, seed_token, n,
                self.prompt_noise_seed, self.noise_salt, n_completion, seq_len,
            ),
        }
    }

    /// Ψ-Spec tree window (upstream tree sampler): draft once, grow a
    /// best-first candidate tree from the per-depth draft top-k, verify the
    /// whole tree in ONE tree-attention forward, walk the target picks
    /// through it, and compact the accepted path's KV into the committed
    /// slots. Lossless by the same per-position argument as upstream: every
    /// committed token is the target's own draw at its prefix.
    fn step_tree(&mut self, gpu: &mut Gpu, target: &mut LlamaBundle, position: usize, seed: u32, emitted: &[u32], budget: usize, temp: f32, greedy: bool) -> Result<SpecStep, String> {
        let (max_nodes, top_k) = self.tree.ok_or("Uno tree config missing")?;
        let vocab = target.config.vocab_size;
        let n = budget.saturating_sub(2).min(max_nodes.saturating_sub(1));
        let mut noise = self.draw_noise(n, seed, position + 1, emitted.len());
        noise.insert(0, seed);
        let batch = self.batch.as_ref().ok_or("Uno batch scratch missing")?;

        // DRAFT: conditional-LoRA forward over [seed, noise...] — logits stay
        // on device in batch.draft_logits.
        uno_forward_batch_device(gpu, &target.weights, &target.config, Some(&self.adapter),
            &noise, position, &mut target.kv, &target.scratch, batch, &batch.draft_logits)
            .map_err(|e| format!("Uno draft forward: {e:?}"))?;
        let rows = n + 1;
        let inv_temp = if greedy { 1.0 } else { 1.0 / temp };

        // Target-pick law for the root and every node pick: the reference
        // tree sampler draws ALL target tokens through the full request
        // filters (two_pass_decoding.py runs root and per-node picks through
        // the same `Sampler`), so mirror `step_device`'s request law instead
        // of an unfiltered draw. `sample_row` drives the fused AR sampler, so
        // the picks stay byte-identical to AR decode at this request config.
        // Greedy ignores the filter (argmax arm below).
        let (pick_top_p, pick_top_k) = if greedy {
            (1.0, None)
        } else if self.request.top_p > 0.0 && self.request.top_p < 1.0 {
            (self.request.top_p.min(1.0),
                (self.request.top_k > 0).then_some(self.request.top_k.min(64) as u32))
        } else if self.request.top_k > 0 && self.request.top_k < vocab {
            (1.0, Some(self.request.top_k.min(64) as u32))
        } else {
            (1.0, None)
        };

        // AR's repeat history for this window, exactly as the linear path builds
        // it: last `min(repeat_window, 64)` of (prompt tail ++ emitted), where
        // `emitted` already ends with the pending seed. Only built when a penalty
        // is non-neutral.
        let penalty_active = self.request.repeat_penalty > 1.0
            || self.request.presence_penalty > 0.0
            || self.request.frequency_penalty > 0.0;
        let rw = self.request.repeat_window.min(64);
        let mut hist_base: Vec<u32> = Vec::new();
        if greedy && penalty_active && rw > 0 {
            let e = emitted.len().min(rw);
            let mut all: Vec<u32> = Vec::with_capacity(self.prompt_tail.len() + e);
            all.extend_from_slice(&self.prompt_tail);
            all.extend_from_slice(&emitted[emitted.len() - e..]);
            let last = all.len().min(rw);
            hist_base = all[all.len() - last..].to_vec();
        }

        // Root token: the target's own draw after the seed. It is committed
        // unconditionally, so it carries AR's penalty.
        if greedy {
            if !hist_base.is_empty() {
                self.penalize_row(gpu, &batch.draft_logits, 0, vocab, &hist_base)?;
            }
            gpu.argmax_f32_batched(&batch.draft_logits, &batch.picks, vocab, rows)
                .map_err(|e| format!("Uno tree argmax: {e:?}"))?;
        }
        let clean = if greedy {
            let mut raw = [0u8; 4];
            gpu.hip.memcpy_dtoh(&mut raw, &batch.picks.buf).map_err(|e| format!("Uno tree picks: {e:?}"))?;
            u32::from_ne_bytes(raw)
        } else {
            sample_row(gpu, &batch.draft_logits.sub_offset(0, vocab), vocab, temp, pick_top_p, pick_top_k,
                &batch.sample_result, &batch.sample_repeat, &mut self.rng)?
        };
        if budget == 1 || self.stops(clean) {
            return Ok(SpecStep::new([clean], clean, 0, 0));
        }

        // Per-depth candidate lists: logsumexp then top_k masked-argmax
        // rounds over the draft rows (row d carries depth d; row 0 is the
        // seed row and is skipped). Mutates draft_logits — the tree path does
        // not reuse it.
        gpu.uno_row_lse(&batch.draft_logits, &batch.lse, rows, vocab, inv_temp)
            .map_err(|e| format!("Uno tree lse: {e:?}"))?;
        let mut lse_bytes = vec![0u8; rows * 4];
        gpu.hip.memcpy_dtoh(&mut lse_bytes, &batch.lse.buf).map_err(|e| format!("Uno tree lse: {e:?}"))?;
        let lse: Vec<f32> = lse_bytes.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
        // n depth lists: draft row d (1..=n) carries the depth-d candidates.
        let mut depth_candidates: Vec<Vec<crate::uno::TreeCandidate>> =
            (0..n).map(|_| Vec::with_capacity(top_k)).collect();
        let mut round = vec![0u8; rows * 8];
        for _ in 0..top_k {
            gpu.uno_topk_round(&batch.draft_logits, &batch.topk, rows, vocab)
                .map_err(|e| format!("Uno tree topk: {e:?}"))?;
            gpu.hip.memcpy_dtoh(&mut round, &batch.topk.buf).map_err(|e| format!("Uno tree topk: {e:?}"))?;
            // Buffer slot = draft row (block id); the embedded f32 is the
            // row's argmax token id (exact below 2^24) with its logit value.
            for (row, chunk) in round.chunks_exact(8).enumerate() {
                let token = f32::from_ne_bytes(chunk[0..4].try_into().unwrap()) as usize;
                let val = f32::from_ne_bytes(chunk[4..8].try_into().unwrap());
                if row == 0 || row >= rows || (row - 1) >= depth_candidates.len() {
                    continue;
                }
                depth_candidates[row - 1].push(crate::uno::TreeCandidate {
                    token: token as u32,
                    log_prob: f64::from(val) * f64::from(inv_temp) - f64::from(lse[row]),
                });
            }
        }

        let tree = crate::uno::build_best_first_tree(clean, &depth_candidates, max_nodes);
        if tree.tokens.len() < 2 {
            return Ok(SpecStep::new([clean], clean, 0, 0));
        }
        let picks = crate::uno::uno_tree_verify_picks(
            gpu, &target.weights, &target.config, &tree, position,
            &mut target.kv, &target.scratch, batch, greedy, temp, pick_top_p, pick_top_k,
            &mut self.rng,
            &hist_base, self.repeat_buf.as_ref(),
            self.request.repeat_penalty, self.request.presence_penalty,
            self.request.frequency_penalty, rw,
        ).map_err(|e| format!("Uno tree verify: {e:?}"))?;
        let (committed, path) = crate::uno::walk_draft_tree(&tree, &picks, (n) as i32, |t| self.stops(t));
        crate::uno::compact_tree_kv(gpu, &target.kv, &path, position)
            .map_err(|e| format!("Uno tree compact: {e:?}"))?;
        // The walk's picks are the tokens GENERATED after the root; the root
        // (clean) itself is committed too and its KV sits at position+1, so
        // it must lead the emit or every subsequent position drifts by one.
        let mut emit = Vec::with_capacity(committed.len() + 1);
        emit.push(clean);
        emit.extend_from_slice(&committed);
        let next = *emit.last().unwrap();
        Ok(SpecStep::new(emit, next, tree.tokens.len() - 1, path.len() - 1))
    }

    /// Batch-scratch fast path: both forwards leave their logits on device;
    /// only token ids (and the fused verifier's 8 B/row decisions) cross to
    /// the host. Mirrors the reference implementation's device-resident
    /// sampling/verify (ifm-ai/uno two_pass_decoding.py).
    fn step_device(&mut self, gpu: &mut Gpu, target: &mut LlamaBundle, position: usize, seed: u32, emitted: &[u32], budget: usize, temp: f32, greedy: bool) -> Result<SpecStep, String> {
        let n = budget.saturating_sub(2);
        let vocab = target.config.vocab_size;
        let rows = n + 1;
        // Device-resident filters. Greedy ignores the filter (argmax).
        // top-k-only requests mask draft + verify rows to their top-K support
        // so the fused dual-softmax kernel computes the filtered p/q (upstream
        // `build_sparse_top_k_probs` support). top-p requests (optionally with
        // top_k, min_p=0) use the candidate-gather verifier (`uno_filter_verify`),
        // which replicates the fused AR sampler's exact law: top-K logit gather
        // (cap = request top_k else the sampler's 20), softmax at temp, nucleus
        // truncation at top_p over the descending-order list, renormalization.
        // min-p-only filtering keeps the host Distribution law (the gather
        // kernel has no min-p selector).
        let dev_topk = if greedy { None } else {
            (self.request.top_k > 0 && self.request.top_k < vocab
                && !(self.request.top_p > 0.0 && self.request.top_p < 1.0)
                && self.request.min_p <= 0.0)
                .then_some(self.request.top_k.min(1024))
        };
        let dev_nucleus = !greedy
            && self.request.top_p > 0.0
            && self.request.top_p < 1.0
            && self.request.min_p <= 0.0;
        // Sampling law for non-greedy rows: dev_nucleus passes the request
        // top_p/top_k through to the fused AR sampler (identical law to what
        // AR decode emits at this request config); every other non-greedy path
        // keeps the unfiltered draw (`top_k` None → sampler cap 20, top_p 1.0 —
        // the dev_topk mask already restricted the row's support).
        let (sample_top_p, sample_top_k) = if dev_nucleus {
            (self.request.top_p.min(1.0),
             (self.request.top_k > 0).then_some(self.request.top_k.min(64) as u32))
        } else {
            (1.0, None)
        };
        let mut noise = self.draw_noise(n, seed, position + 1, emitted.len());
        noise.insert(0, seed);
        let batch = self.batch.as_ref().ok_or("Uno batch scratch missing")?;
        // DRAFT: conditional-LoRA forward, logits stay in batch.draft_logits.
        uno_window_forward(gpu, target, &self.adapter, batch, &noise, position, true,
            &mut self.draft_graph, &mut self.graph_draft_warm)?;
        if let Some(k) = dev_topk {
            gpu.uno_apply_topk_mask(&batch.draft_logits, rows, vocab, k)
                .map_err(|e| format!("Uno draft topk mask: {e:?}"))?;
        }
        // AR's repeat history for this window: the last `min(repeat_window, 64)`
        // conversation tokens, never before the turn's prompt (mirrors `ar.rs`'s
        // repeat upload, which passes `hist_slice.len()` as the kernel window).
        // Only built when a penalty is non-neutral, so the neutral path pays
        // nothing.
        let penalty_active = self.request.repeat_penalty > 1.0
            || self.request.presence_penalty > 0.0
            || self.request.frequency_penalty > 0.0;
        let rw = self.request.repeat_window.min(64);
        let mut hist_base: Vec<u32> = Vec::new();
        if greedy && penalty_active && rw > 0 {
            // AR pushes the freshly sampled token into `conversation_tokens`
            // (ar.rs:5220) BEFORE uploading the history (ar.rs:5247), so the
            // history INCLUDES the token being forwarded — here, `seed`, the
            // token this window forwards at `position`. Build the full suffix
            // (prompt tail ++ committed ++ seed) then keep its last `rw`.
            // Build (prompt tail ++ newest generated) and keep the last `rw`.
            // Order matters: filling the window with the prompt tail FIRST and
            // only then appending what fits leaves no room for generated tokens
            // (prompt_tail.len() >= rw), so the window would omit every repeat
            // among the generated text — a large, penalty-dependent logit error.
            // `emitted` already ends with the pending seed: the trace confirms
            // `position - prompt_len + 1 == emitted.len()` at every window, i.e.
            // it spans positions `prompt_len .. position` inclusive, so the
            // "history includes the token being forwarded" convention is already
            // satisfied — pushing `seed` again would fabricate a duplicate.
            let e = emitted.len().min(rw);
            let mut all: Vec<u32> = Vec::with_capacity(self.prompt_tail.len() + e);
            all.extend_from_slice(&self.prompt_tail);
            all.extend_from_slice(&emitted[emitted.len() - e..]);
            let last = all.len().min(rw);
            hist_base = all[all.len() - last..].to_vec();
        }
        let mut proposal: Vec<u32> = Vec::with_capacity(rows);
        if greedy {
            // Draft row 0's argmax becomes `clean`, which is committed
            // unconditionally, so it must carry AR's penalty. The remaining
            // draft rows only produce proposals (verified against the target),
            // so they stay unpenalised — that can only affect acceptance, never
            // losslessness.
            if !hist_base.is_empty() {
                self.penalize_row(gpu, &batch.draft_logits, 0, vocab, &hist_base)?;
            }
            let picks = argmax_rows(gpu, &batch.draft_logits, &batch.picks, vocab, rows)?;
            proposal.extend_from_slice(&picks);
        } else {
            for row in 0..rows {
                proposal.push(sample_row(gpu, &batch.draft_logits.sub_offset(row * vocab, vocab),
                    vocab, temp, sample_top_p, sample_top_k, &batch.sample_result,
                    &batch.sample_repeat, &mut self.rng)?);
            }
        }
        let clean = proposal[0];
        if budget == 1 || self.stops(clean) { return Ok(SpecStep::new([clean], clean, 0, 0)); }
        // VERIFY: base-only forward, logits stay in batch.verify_logits.
        uno_window_forward(gpu, target, &self.adapter, batch, &proposal, position + 1, false,
            &mut self.verify_graph, &mut self.graph_verify_warm)?;
        if let Some(k) = dev_topk {
            gpu.uno_apply_topk_mask(&batch.verify_logits, rows, vocab, k)
                .map_err(|e| format!("Uno verify topk mask: {e:?}"))?;
        }
        let mut emit = Vec::with_capacity(rows + 1);
        emit.push(clean);
        let mut accepted = 0;
        if greedy {
            // Verify row `row` sits at position+1+row and predicts the token
            // after it, so AR's history there is everything before that input:
            // the base history plus this window's earlier proposals. Applying it
            // makes every `picks[row]` — which is the committed token whether it
            // matches the proposal or becomes the correction — equal to AR's
            // penalised argmax.
            if !hist_base.is_empty() {
                let mut hist = hist_base.clone();
                for row in 0..rows {
                    // Same convention: the row's own input token is part of the
                    // history for the token it predicts.
                    hist.push(proposal[row]);
                    let last = hist.len().min(rw);
                    let window = hist[hist.len() - last..].to_vec();
                    if hipfire_config::developer_var("HIPFIRE_UNO_TRACE").is_ok() {
                        // Predicted position: verify row `row` forwards proposal[row]
                        // at position+1+row and predicts the token at position+2+row.
                        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                        for &t in window.iter() {
                            h ^= t as u64;
                            h = h.wrapping_mul(0x100_0000_01b3);
                        }
                        eprintln!(
                            "[uno-win] pred_pos={} row={row} win_len={} hist_len={} fnv={h:016x}",
                            position + 2 + row,
                            window.len(),
                            hist.len(),
                        );
                    }
                    self.penalize_row(gpu, &batch.verify_logits, row, vocab, &window)?;
                }
            }
            let picks = argmax_rows(gpu, &batch.verify_logits, &batch.picks, vocab, rows)?;
            for row in 0..n {
                emit.push(picks[row]);
                if picks[row] != proposal[row + 1] { break; }
                accepted += 1;
                if self.stops(picks[row]) { break; }
            }
            if accepted == n && !self.stops(*emit.last().unwrap()) {
                emit.push(picks[n]);
            }
            if hipfire_config::developer_var("HIPFIRE_UNO_TRACE").is_ok() {
                eprintln!(
                    "[uno-trace] pos={position} seed={seed} n={n} rows={rows} \
                     proposal={proposal:?} verify_picks={picks:?} emit={emit:?} accepted={accepted} \
                     rw={rw} hist_base={} emitted={} prompt_tail={} penalty_active={penalty_active} \
                     hist_tail={:?} seed_tail={:?}",
                    hist_base.len(),
                    emitted.len(),
                    self.prompt_tail.len(),
                    &hist_base[hist_base.len().saturating_sub(8)..],
                    &emitted[emitted.len().saturating_sub(8)..],
                );
            }
        } else {
            if n > 0 {
                let bytes: Vec<u8> = proposal[1..].iter().flat_map(|t| t.to_ne_bytes()).collect();
                gpu.hip.memcpy_htod(&batch.proposals.buf, &bytes).map_err(|e| format!("Uno proposals: {e:?}"))?;
                // Reserve a fresh request-seeded stream for each verification window.
                let _ = uniform(&mut self.rng);
                if dev_nucleus {
                    // Candidate-gather verifier: per-row top-K +
                    // softmax + nucleus truncation, matching the fused AR
                    // sampler law the proposals were drawn with.
                    let top_k_req = if self.request.top_k > 0 {
                        self.request.top_k.min(64) as i32
                    } else {
                        20
                    };
                    gpu.uno_filter_verify(&batch.verify_logits, &batch.draft_logits.sub_offset(vocab, n * vocab),
                        &batch.proposals, &batch.decisions, &batch.filter_vals,
                        &batch.filter_idxs, n, vocab, temp,
                        sample_top_p, top_k_req, self.rng)
                        .map_err(|e| format!("Uno filter verify: {e:?}"))?;
                } else {
                    gpu.uno_verify_logits(&batch.verify_logits, &batch.draft_logits.sub_offset(vocab, n * vocab),
                        &batch.proposals, &batch.decisions, n, vocab, temp, self.rng)
                        .map_err(|e| format!("Uno fused verify: {e:?}"))?;
                }
                let mut decisions = vec![0u8; n * 8];
                gpu.hip.memcpy_dtoh(&mut decisions, &batch.decisions.buf).map_err(|e| format!("Uno decisions: {e:?}"))?;
                for (row, pair) in decisions.chunks_exact(8).enumerate() {
                    let status = u32::from_ne_bytes(pair[..4].try_into().unwrap());
                    let correction = u32::from_ne_bytes(pair[4..].try_into().unwrap());
                    match status {
                        1 => { emit.push(proposal[row + 1]); accepted += 1; }
                        0 if (correction as usize) < vocab => { emit.push(correction); break; }
                        _ => return Err("Uno fused verifier rejected invalid logits or residual mass".into()),
                    }
                    if self.stops(*emit.last().unwrap()) { break; }
                }
            }
            if accepted == n && !self.stops(*emit.last().unwrap()) {
                emit.push(sample_row(gpu, &batch.verify_logits.sub_offset(n * vocab, vocab),
                    vocab, temp, sample_top_p, sample_top_k, &batch.sample_result,
                    &batch.sample_repeat, &mut self.rng)?);
            }
        }
        let next = *emit.last().unwrap();
        Ok(SpecStep::new(emit, next, n, accepted))
    }
}

fn parse_block_len(raw: Option<&str>) -> usize {
    // Upstream `max_diffusion_block_size` is 16; inference default is 8.
    // xbatch GEMV is 1..=8 so LoRA deltas chunk above 8 (see UnoBatchScratch::delta).
    raw.and_then(|s| s.parse::<usize>().ok())
        .map(|b| b.clamp(2, 16))
        .unwrap_or(4)
}

/// hipGraph capture cap: LDS Q8 attention sizes scores[] from this, while the
/// causal bound still comes from positions[b]. So a capture taken at cap `N`
/// replays correctly for any window whose `pos + rows <= N`, and a window that
/// does not fit falls back to eager (never a mis-sized replay).
///
/// 256 covered the original HTTP A/B (prompt 17 + 64 gen). Every longer prompt
/// fell back to eager and paid a full launch sequence per window (hundreds of
/// launches per window forward). Raise with `HIPFIRE_UNO_GRAPH_CTX` to amortise
/// those launches on long prompts.
/// DISABLED by default (0). Window graph capture is retained behind
/// `HIPFIRE_UNO_GRAPH_CTX` for experiments only: a captured window replayed at
/// a later position returned the CAPTURE window's draft/verify logits instead
/// of recomputing for the current tokens/positions, so two consecutive windows
/// emitted byte-identical proposals from different seeds and the second
/// window's commit could be a token the target never chose (observed
/// divergence at `top2 gap 0.9994`, i.e. a wide-gap flip, not a ULP tie).
/// `uno_perf_probe` passes token identity with capture off and fails with it
/// on at the same positions.
const UNO_GRAPH_CTX: usize = 0;

/// Resolved capture cap (`HIPFIRE_UNO_GRAPH_CTX`, clamped to 0..=8192; 0
/// disables capture). The clamp is the LDS budget: the legacy Q8 batched
/// attention kernel sizes its dynamic shared memory as
/// `(max_ctx_len + block + head_dim) * 4`, so a cap near 15000 already needs
/// ~61 KB per workgroup (one workgroup per CU) and past that the launch fails
/// outright. 8192 keeps the capture at 34 KB.
fn graph_ctx_cap() -> usize {
    hipfire_config::developer_var("HIPFIRE_UNO_GRAPH_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.clamp(0, 8192))
        .unwrap_or(UNO_GRAPH_CTX)
}

fn uno_window_eager(
    gpu: &mut Gpu, target: &mut LlamaBundle, adapter: &UnoAdapter,
    batch: &UnoBatchScratch, tokens: &[u32], pos: usize, lora: bool,
) -> Result<(), String> {
    let output = if lora { &batch.draft_logits } else { &batch.verify_logits };
    uno_forward_batch_device(
        gpu, &target.weights, &target.config,
        lora.then_some(adapter), tokens, pos, &mut target.kv, &target.scratch,
        batch, output,
    ).map_err(|e| format!("Uno {} forward: {e:?}", if lora { "draft" } else { "verify" }))
}

fn uno_window_forward(
    gpu: &mut Gpu, target: &mut LlamaBundle, adapter: &UnoAdapter,
    batch: &UnoBatchScratch, tokens: &[u32], pos: usize, lora: bool,
    graph: &mut Option<(Graph, GraphExec, Vec<Vec<u8>>)>,
    warm: &mut bool,
) -> Result<(), String> {
    let rows = tokens.len();
    uno_upload_window(gpu, batch, tokens, pos)
        .map_err(|e| format!("Uno window upload: {e:?}"))?;
    let cap = graph_ctx_cap();
    let fits = pos.saturating_add(rows) <= cap && rows == adapter.block_len;
    if fits {
        if let Some((_, exec, _)) = graph.as_ref() {
            gpu.ensure_capture_stream().map_err(|e| format!("Uno graph stream: {e:?}"))?;
            gpu.launch_graph(exec).map_err(|e| format!("Uno graph launch: {e:?}"))?;
            return Ok(());
        }
    }
    if fits && *warm {
        if gpu.ensure_capture_stream().is_ok() {
            gpu.graphs.capture_blobs.clear();
            gpu.graphs.capture_max_ctx = Some(cap);
            gpu.graphs.capture_mode = true;
            if gpu.begin_stream_capture().is_ok() {
                let r = uno_window_eager(gpu, target, adapter, batch, tokens, pos, lora);
                if r.is_err() {
                    gpu.graphs.capture_mode = false;
                    gpu.graphs.capture_max_ctx = None;
                    gpu.graphs.capture_blobs.clear();
                    if let Some(stream) = gpu.active_stream.as_ref() {
                        if let Ok(g) = gpu.hip.stream_end_capture(stream) {
                            let _ = gpu.hip.graph_destroy(g);
                        }
                    }
                    return r;
                }
                gpu.graphs.capture_mode = false;
                gpu.graphs.capture_max_ctx = None;
                match gpu.end_stream_capture() {
                    Ok(captured) => match gpu.hip.graph_instantiate(&captured) {
                        Ok(exec) => {
                            let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
                            *graph = Some((captured, exec, blobs));
                        }
                        Err(_) => {
                            let _ = gpu.hip.graph_destroy(captured);
                            gpu.graphs.capture_blobs.clear();
                        }
                    },
                    Err(_) => gpu.graphs.capture_blobs.clear(),
                }
                return Ok(());
            }
            gpu.graphs.capture_mode = false;
            gpu.graphs.capture_max_ctx = None;
        }
    }
    let r = uno_window_eager(gpu, target, adapter, batch, tokens, pos, lora);
    if r.is_ok() && fits {
        *warm = true;
    }
    r
}

impl Speculator for UnoSpeculator {
    fn name(&self) -> &'static str { "uno" }
    fn requires_greedy(&self) -> bool { false }
    fn supports_temp_verify(&self) -> bool { true }
    fn supports_chain_nucleus_verify(&self) -> bool { true }
    fn supports_grammar(&self) -> bool { false }
    fn admission_error(&self) -> Option<&'static str> {
        if self.request.min_p > 0.0 { Some("Uno does not support min_p") } else { None }
    }
    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
        self.rng = request_rng_state(cfg.rng_seed) as u32;
        self.noise_salt = u64::from(self.rng);
    }
    fn prefill(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, prompt_tokens: &[u32], tokens: &[u32], start: usize, cache_hit: bool, _resume: Option<usize>, abort: &dyn Fn() -> bool) -> Result<PrefillOutcome, String> {
        if self.request.min_p > 0.0 {
            return Err("Uno does not support min_p".into());
        }
        self.prompt_noise_seed = hash_prompt_tokens(prompt_tokens);
        // AR scopes its repeat history to the turn's prompt (`ar.rs` repeat
        // upload), so the verifier needs that tail to reproduce the penalty.
        let tail = prompt_tokens.len().min(64);
        self.prompt_tail.clear();
        self.prompt_tail
            .extend_from_slice(&prompt_tokens[prompt_tokens.len() - tail..]);
        match target.spec_advance(gpu, tokens, start, !cache_hit, abort, None)? {
            SpecAdvance::Ready { .. } => {
                let target = target.as_any_mut().downcast_mut::<LlamaBundle>().ok_or("Uno target is not LlamaBundle")?;
                Ok(PrefillOutcome::Ready { first_token: self.sample_target(gpu, target)? })
            }
            SpecAdvance::Aborted => Ok(PrefillOutcome::Aborted),
        }
    }
    fn step(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, position: usize, seed: u32, emitted: &[u32], grammar: Option<&mut dyn SpecGrammar>, temp: f32, max_emit: usize) -> Result<SpecStep, String> {
        if max_emit == 0 || grammar.is_some() { return Err("Uno requires a positive output budget and no grammar".into()); }
        let budget = max_emit.min(self.capacity.saturating_sub(position)).min(self.adapter.block_len + 1);
        if budget == 0 { return Err("Uno context exhausted".into()); }
        self.request.temp = temp;
        let target = target.as_any_mut().downcast_mut::<LlamaBundle>().ok_or("Uno target is not LlamaBundle")?;
        if target.kv.compact_offset != 0 { return Err("Uno does not support compacted KV".into()); }
        let n = budget.saturating_sub(2);
        let vocab = target.config.vocab_size;
        // GPU-resident fast path: greedy verify via batched GPU argmax (16 B
        // D2H per pass) and unfiltered temp>0 via the fused dual-softmax
        // kernel + `sample_top_p_pf` proposals — full logits never leave the
        // device. Filtered stochastic requests keep the host Distribution
        // law (the fused kernel computes an unfiltered softmax q).
        let greedy = !temp.is_finite() || temp <= 1e-6;
        let filtered = (self.request.top_k > 0 && self.request.top_k < vocab)
            || (self.request.top_p > 0.0 && self.request.top_p < 1.0)
            || self.request.min_p > 0.0;
        // Filtered requests are device-resident when the filter is
        // expressible on-device: top-k-only masks rows to their top-K support
        // (fused dual-softmax over the masked row); top-p (optionally combined
        // with top_k) uses the candidate-gather verifier that replicates the
        // fused AR sampler's law. min-p-only filtering keeps the host
        // Distribution law (the gather kernel has no min-p selector). Greedy
        // ignores all filters (argmax).
        let dev_filterable = filtered && self.request.min_p <= 0.0;
        // The tree takes every request the device paths can express (greedy,
        // unfiltered, and all min_p-free filters): like the reference tree
        // sampler, its target picks run through the full request law
        // (`step_tree` passes the filters into `sample_row`), and top-p needs
        // no verifier-kernel support there because acceptance is by
        // construction, not by p/q ratio. Requires batch scratch (the tree
        // forwards are batch-device-resident); without it, fall through to the
        // host law below.
        if self.tree.is_some() && self.batch.is_some() && (greedy || !filtered || dev_filterable) {
            return self.step_tree(gpu, target, position, seed, emitted, budget, temp, greedy);
        }
        if self.batch.is_some() && (greedy || !filtered || dev_filterable) {
            return self.step_device(gpu, target, position, seed, emitted, budget, temp, greedy);
        }
        let mut noise = self.draw_noise(n, seed, position + 1, emitted.len());
        noise.insert(0, seed);
        let draft_logits = self.forward(gpu, target, &noise, position, true)?;
        let clean = Distribution::from_logits(&draft_logits[..vocab], self.request)?.sample(&mut self.rng);
        if budget == 1 || self.stops(clean) { return Ok(SpecStep::new([clean], clean, 0, 0)); }
        let mut proposal = Vec::with_capacity(n + 1);
        proposal.push(clean);
        let mut distributions = Vec::with_capacity(n);
        for row in draft_logits.chunks_exact(vocab).skip(1) {
            let q = Distribution::from_logits(row, self.request)?;
            proposal.push(q.sample(&mut self.rng));
            distributions.push(q);
        }
        // Base seed KV persists; verification overwrites the noise suffix.
        let verify_logits = self.forward(gpu, target, &proposal, position + 1, false)?;
        let mut emit = Vec::with_capacity(budget);
        emit.push(clean);
        let mut accepted = 0;
        for (row, q) in distributions.iter().enumerate() {
            let p = Distribution::from_logits(&verify_logits[row * vocab..(row + 1) * vocab], self.request)?;
            let token = proposal[row + 1];
            let (next, keep) = verify_proposal(&p, q, token, greedy, &mut self.rng)?;
            emit.push(next);
            if !keep { break; }
            accepted += 1;
            if self.stops(next) { break; }
        }
        if accepted == n && !self.stops(*emit.last().unwrap()) {
            emit.push(Distribution::from_logits(&verify_logits[n * vocab..(n + 1) * vocab], self.request)?.sample(&mut self.rng));
        }
        let next = *emit.last().unwrap();
        Ok(SpecStep::new(emit, next, n, accepted))
    }

    fn repair_terminal_prefix(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, window_start: usize, window_seed: u32, consumed: &[u32]) -> Result<bool, String> {
        // Linear Uno overwrites draft-noise KV during verify and has no
        // recurrent snapshot. Replay the pending seed plus the consumed
        // prefix (minus the new pending token) through the same per-token
        // advance AR uses, so a ThinkCap/EOS mid-window does not 200k-reprefill.
        // Leftover proposal slots past the replayed length are unattended:
        // attention is keyed by `pos`, not physical_cap.
        let replay = terminal_prefix_replay(window_seed, consumed);
        if replay.is_empty() {
            return Ok(true);
        }
        match target.spec_advance(gpu, &replay, window_start, false, &|| false, None)? {
            SpecAdvance::Ready { .. } => Ok(true),
            SpecAdvance::Aborted => Ok(false),
        }
    }
    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> {
        self.rng = request_rng_state(self.request.rng_seed) as u32;
        self.noise_salt = u64::from(self.rng);
        Ok(())
    }
    fn block_size(&self) -> usize { self.adapter.block_len + 1 }
    fn ctx_capacity(&self) -> usize { self.capacity }
    fn free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(batch) = self.batch { batch.free_gpu(gpu); }
        if let Some(buf) = self.repeat_buf { let _ = gpu.free_tensor(buf); }
        self.scratch.free_gpu(gpu);
        self.adapter.free_gpu(gpu);
    }
}


/// All `rows` argmaxes in one launch; only `rows × 4` bytes cross to host.
fn argmax_rows(gpu: &mut Gpu, logits: &GpuTensor, picks: &GpuTensor, vocab: usize, rows: usize) -> Result<Vec<u32>, String> {
    gpu.argmax_f32_batched(logits, picks, vocab, rows)
        .map_err(|e| format!("Uno argmax: {e:?}"))?;
    let mut bytes = vec![0u8; rows * 4];
    gpu.hip.memcpy_dtoh(&mut bytes, &picks.buf)
        .map_err(|e| format!("Uno picks: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

/// One fused GPU draw (softmax + categorical) via the AR decode sampler —
/// the same kernel AR decode uses, so the drawn token matches the request
/// sampling law exactly (temp + top_p + top_k). `top_p_eff` = 1.0 with
/// `top_k` = None reproduces the unfiltered legacy draw byte-for-byte.
pub(crate) fn sample_row(gpu: &mut Gpu, logits: &GpuTensor, vocab: usize, temp: f32,
    top_p_eff: f32, top_k: Option<u32>, result: &GpuTensor, repeat: &GpuTensor,
    rng: &mut u32) -> Result<u32, String> {
    let (tok, next) = gpu
        .sample_top_p_pf(logits, result, repeat, vocab, temp, top_p_eff, *rng, 0, 1.0, 0.0, 0.0, top_k, None)
        .map_err(|e| format!("Uno sample: {e:?}"))?;
    *rng = next;
    Ok(tok)
}

fn verify_proposal(p: &Distribution, q: &Distribution, token: u32, greedy: bool, rng: &mut u32) -> Result<(u32, bool), String> {
    let keep = if greedy { p.probability(token) > 0.0 }
        else { uniform(rng) * q.probability(token) < p.probability(token) };
    if keep { Ok((token, true)) }
    else { Ok((p.residual(q)?.sample(rng), false)) }
}

/// Normalized categorical distribution in token-ID order. Unfiltered softmax
/// is linear; explicit top-k/nucleus requests rank before filtering.
struct Distribution(Vec<(u32, f64)>);

impl Distribution {
    fn from_logits(logits: &[f32], cfg: SpecRequestConfig) -> Result<Self, String> {
        if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
            return Err("Uno nonfinite or empty logits".into());
        }
        if !cfg.temp.is_finite() { return Err("Uno nonfinite temperature".into()); }
        if cfg.temp <= 1e-6 { return Ok(Self(vec![(llama::argmax(logits), 1.0)])); }
        let nucleus = if cfg.top_p > 0.0 { f64::from(cfg.top_p.min(1.0)) } else { 1.0 };
        let ranked_filter = (cfg.top_k > 0 && cfg.top_k < logits.len()) || nucleus < 1.0;
        let mut ranked: Vec<(u32, f64)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, f64::from(v))).collect();
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        if ranked_filter {
            ranked.sort_unstable_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            if cfg.top_k > 0 { ranked.truncate(cfg.top_k.min(ranked.len())); }
        }
        for (_, p) in &mut ranked { *p = ((*p - max) / f64::from(cfg.temp)).exp(); }
        if cfg.min_p > 0.0 { ranked.retain(|&(_, p)| p >= f64::from(cfg.min_p)); }
        if nucleus < 1.0 {
            let total: f64 = ranked.iter().map(|x| x.1).sum();
            let mut cumulative = 0.0;
            ranked.retain(|(_, p)| {
                let keep = cumulative <= nucleus * total;
                cumulative += *p;
                keep
            });
        }
        let total: f64 = ranked.iter().map(|x| x.1).sum();
        if total <= 0.0 || !total.is_finite() { return Err("Uno empty filtered distribution".into()); }
        for (_, p) in &mut ranked { *p /= total; }
        if ranked_filter { ranked.sort_unstable_by_key(|x| x.0); }
        Ok(Self(ranked))
    }

    fn probability(&self, token: u32) -> f64 {
        self.0.binary_search_by_key(&token, |x| x.0).map(|i| self.0[i].1).unwrap_or(0.0)
    }

    fn sample(&self, rng: &mut u32) -> u32 {
        if self.0.len() == 1 { return self.0[0].0; }
        self.sample_at(uniform(rng))
    }

    fn sample_at(&self, u: f64) -> u32 {
        let total: f64 = self.0.iter().map(|x| x.1).sum();
        let mut remainder = u * total;
        for &(token, p) in &self.0 {
            if remainder < p { return token; }
            remainder -= p;
        }
        self.0.iter().rev().find(|x| x.1 > 0.0).expect("positive categorical mass").0
    }

    fn residual(&self, draft: &Self) -> Result<Self, String> {
        let mut residual = Vec::with_capacity(self.0.len());
        let mut j = 0;
        for &(token, p) in &self.0 {
            while j < draft.0.len() && draft.0[j].0 < token { j += 1; }
            let q = draft.0.get(j).filter(|x| x.0 == token).map_or(0.0, |x| x.1);
            if p > q { residual.push((token, p-q)); }
        }
        if residual.is_empty() { return Err("Uno rejected identical distributions".into()); }
        Ok(Self(residual))
    }
}

fn uniform(state: &mut u32) -> f64 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (f64::from(*state) + 0.5) / 4294967296.0
}

/// Upstream noise.py `_mix_u64`: the splitmix64 finalizer.
#[inline]
fn mix_u64(value: u64) -> u64 {
    let mut v = value & 0xFFFF_FFFF_FFFF_FFFF;
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9) & 0xFFFF_FFFF_FFFF_FFFF;
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB) & 0xFFFF_FFFF_FFFF_FFFF;
    (v ^ (v >> 31)) & 0xFFFF_FFFF_FFFF_FFFF
}

/// Upstream `random_uniform` noise rows: uniform in `[low, high)`.
fn random_uniform_noise(low: u32, high: u32, n: usize, rng: &mut u32) -> Vec<u32> {
    let span = high.saturating_sub(low).max(1);
    let last = span - 1;
    (0..n)
        .map(|_| low + ((uniform(rng) * f64::from(span)) as u32).min(last))
        .collect()
}

/// Upstream `_sequence_noise_seed`: hash the prompt once with splitmix64.
fn hash_prompt_tokens(tokens: &[u32]) -> u64 {
    let mut seed = 0xD6E8_FEB8_6659_FD93;
    for &token in tokens {
        seed = mix_u64(seed ^ u64::from(token));
    }
    seed
}

/// Upstream `_make_deterministic_noise`: mix prompt seed, per-request salt,
/// completion count, window seed token, and sequence length, then
/// `low + mix_u64(base + slot * C) % span` per noise row.
fn deterministic_uniform_noise(
    low: u32,
    high: u32,
    seed_token: u32,
    n: usize,
    prompt_seed: u64,
    salt: u64,
    n_completion: usize,
    seq_len: usize,
) -> Vec<u32> {
    let span = (high.saturating_sub(low)).max(1) as u64;
    let base = prompt_seed
        .wrapping_mul(0x9E37_79B1_85EB_CA87)
        .wrapping_add(salt.wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add((n_completion as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(u64::from(seed_token).wrapping_mul(0x1656_67B1_9E37_79F9))
        .wrapping_add((seq_len as u64).wrapping_mul(0x85EB_CA77_C2B2_AE63));
    (0..n)
        .map(|slot| {
            let mixed = mix_u64(base.wrapping_add((slot as u64).wrapping_mul(0x27D4_EB2F_1656_67C5)));
            low + (mixed % span) as u32
        })
        .collect()
}

#[cfg(test)]
mod uno_sampling_tests {
    use super::*;

    #[test]
    fn production_decision_accepts_or_corrects_from_residual_support() {
        let cfg = SpecRequestConfig { temp: 1.5, ..Default::default() };
        let p = Distribution::from_logits(&[0.1f32.ln() * 1.5, 0.6f32.ln() * 1.5, 0.3f32.ln() * 1.5], cfg).unwrap();
        let q = Distribution::from_logits(&[0.7f32.ln() * 1.5, 0.2f32.ln() * 1.5, 0.1f32.ln() * 1.5], cfg).unwrap();
        // Seed 42: acceptance draw 0.2523 > 1/7; correction draw
        // 0.0881 selects token 1 from residual probabilities [0, 2/3, 1/3].
        assert_eq!(verify_proposal(&p, &q, 0, false, &mut 42).unwrap(), (1, false));
        assert_eq!(verify_proposal(&p, &q, 1, false, &mut 42).unwrap(), (1, true));
        // Seed 1972 accepts even when p<q: acceptance is not argmax matching.
        assert_eq!(verify_proposal(&p, &q, 0, false, &mut 1972).unwrap(), (0, true));
        let mut rng = 42;
        let mut counts = [0usize; 3];
        let mut rejected = 0;
        for _ in 0..100_000 {
            let proposal = q.sample(&mut rng);
            let (emitted, accepted) = verify_proposal(&p, &q, proposal, false, &mut rng).unwrap();
            counts[emitted as usize] += 1;
            rejected += usize::from(!accepted);
        }
        for (count, expected) in counts.into_iter().zip([0.1, 0.6, 0.3]) {
            assert!((count as f64 / 100_000.0 - expected).abs() < 0.005);
        }
        assert!((rejected as f64 / 100_000.0 - 0.6).abs() < 0.005);
    }

    #[test]
    fn unfiltered_temperature_and_rejection_draws_match_target() {
        let cfg = SpecRequestConfig { temp: 0.8, ..Default::default() };
        let p = Distribution::from_logits(&[0.1f32.ln() * 0.8 + 5.0, 0.6f32.ln() * 0.8 + 5.0, 0.3f32.ln() * 0.8 + 5.0], cfg).unwrap();
        let q = Distribution::from_logits(&[0.7f32.ln() * 0.8 - 3.0, 0.2f32.ln() * 0.8 - 3.0, 0.1f32.ln() * 0.8 - 3.0], cfg).unwrap();
        for (token, expected) in [0.1, 0.6, 0.3].into_iter().enumerate() {
            assert!((p.probability(token as u32) - expected).abs() < 1e-6);
        }
        let residual = p.residual(&q).unwrap();
        let rejected: f64 = q.0.iter().map(|&(t, mass)| (mass - p.probability(t)).max(0.0)).sum();
        let mut counts = [0usize; 3];
        for i in 0..10000 {
            counts[residual.sample_at((i as f64 + 0.5) / 10000.0) as usize] += 1;
        }
        assert_eq!(counts[0], 0);
        for token in 0..3 {
            let mass = p.probability(token as u32).min(q.probability(token as u32))
                + rejected * counts[token] as f64 / 10000.0;
            assert!((mass - p.probability(token as u32)).abs() < 1e-4);
        }
    }

    #[test]
    fn greedy_rejection_corrects_to_target_without_rng_draw() {
        let cfg = SpecRequestConfig { temp: 0.0, ..Default::default() };
        let p = Distribution::from_logits(&[0.0, 3.0, 1.0], cfg).unwrap();
        let q = Distribution::from_logits(&[4.0, 0.0, 1.0], cfg).unwrap();
        let mut rng = 42;
        assert_eq!(q.sample(&mut rng), 0);
        assert_eq!(p.probability(0), 0.0);
        assert_eq!(p.residual(&q).unwrap().sample(&mut rng), 1);
        assert_eq!(rng, 42);
    }

    #[test]
    fn rejection_mass_recovers_target_including_disjoint_support() {
        for (p, q) in [
            (vec![(0, 0.1), (1, 0.6), (2, 0.3)], vec![(0, 0.7), (1, 0.2), (3, 0.1)]),
            (vec![(2, 1.0)], vec![(0, 1.0)]),
        ] {
            let p = Distribution(p);
            let q = Distribution(q);
            let residual = p.residual(&q).unwrap();
            let rejected: f64 = q.0.iter().map(|&(t, mass)| mass - mass.min(p.probability(t))).sum();
            let residual_sum: f64 = residual.0.iter().map(|x| x.1).sum();
            for token in 0..4 {
                let emitted = q.probability(token).min(p.probability(token))
                    + rejected * residual.probability(token) / residual_sum;
                assert!((emitted - p.probability(token)).abs() < 1e-12);
            }
            assert!(residual.probability(residual.sample_at(0.0)) > 0.0);
            assert!(residual.probability(residual.sample_at(1.0 - f64::EPSILON)) > 0.0);
        }
    }

    #[test]
    fn nucleus_includes_crossing_token_after_top_k() {
        let cfg = SpecRequestConfig { temp: 1.0, top_k: 2, top_p: 0.7, ..Default::default() };
        let p = Distribution::from_logits(&[0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()], cfg).unwrap();
        assert!((p.probability(0) - 0.625).abs() < 1e-6);
        assert!((p.probability(1) - 0.375).abs() < 1e-6);
        assert_eq!(p.probability(2), 0.0);
    }

    #[test]
    fn noise_mode_parse_maps_laws() {
        assert_eq!(NoiseMode::parse("random_uniform"), NoiseMode::RandomUniform);
        assert_eq!(NoiseMode::parse("deterministic_uniform"), NoiseMode::DeterministicUniform);
        assert_eq!(NoiseMode::parse("mask"), NoiseMode::Mask);
        assert_eq!(NoiseMode::parse("garbage"), NoiseMode::RandomUniform);
    }

    #[test]
    fn random_uniform_noise_stays_in_law_bounds() {
        let mut rng = 0x1357_9BDF;
        let noise = random_uniform_noise(1, 250624, 8192, &mut rng);
        assert!(noise.iter().all(|&t| (1..250624).contains(&t)), "random_uniform must draw from [low, high)");
        // Two fresh streams with the same seed replay identically.
        let mut rng_a = 0x1357_9BDF;
        let mut rng_b = 0x1357_9BDF;
        assert_eq!(
            random_uniform_noise(1, 250624, 16, &mut rng_a),
            random_uniform_noise(1, 250624, 16, &mut rng_b),
            "must replay from a fresh seed"
        );
    }

    #[test]
    fn deterministic_noise_matches_upstream_noise_py() {
        let prompt_seed = hash_prompt_tokens(&[1, 2, 3]);
        assert_eq!(prompt_seed, 8767098978563511914);
        let tokens = deterministic_uniform_noise(1, 250624, 42, 8, prompt_seed, 7, 4, 10);
        assert_eq!(tokens, vec![108876, 197631, 173207, 6829, 189567, 22995, 194265, 135812]);
        assert!(tokens.iter().all(|&t| (1..250624).contains(&t)));
        let replay = deterministic_uniform_noise(1, 250624, 42, 8, prompt_seed, 7, 4, 10);
        assert_eq!(tokens, replay, "deterministic mode must reproduce");
        assert_ne!(
            tokens,
            deterministic_uniform_noise(1, 250624, 42, 8, prompt_seed, 7, 5, 10),
            "a new completion count must not resample identical noise"
        );
        assert_ne!(
            tokens,
            deterministic_uniform_noise(1, 250624, 43, 8, prompt_seed, 7, 4, 10),
            "a different seed token must decorrelate"
        );
    }

    #[test]
    fn block_len_defaults_to_occupancy_four() {
        assert_eq!(parse_block_len(None), 4);
        assert_eq!(parse_block_len(Some("8")), 8);
        assert_eq!(parse_block_len(Some("1")), 2);
        assert_eq!(parse_block_len(Some("16")), 16);
        assert_eq!(parse_block_len(Some("64")), 16);
        assert_eq!(parse_block_len(Some("nope")), 4);
    }

    #[test]
    fn window_batch_admits_gfx1100() {
        assert!(llama::mq4g256v2_window_batch_ok("gfx1100"));
        assert!(llama::mq4g256v2_window_batch_ok("gfx1101"));
        assert!(!llama::mq4g256v2_window_batch_ok("gfx1030"));
    }

    #[test]
    fn nonfinite_logits_refuse_without_inventing_a_token() {
        let cfg = SpecRequestConfig { temp: 1.0, ..Default::default() };
        assert!(Distribution::from_logits(&[1.0, f32::NAN], cfg).is_err());
        assert!(Distribution::from_logits(&[f32::INFINITY, 0.0], cfg).is_err());
        assert!(Distribution::from_logits(&[], cfg).is_err());
    }

    #[test]
    fn k2_uniform_noise_uses_upstream_half_open_vocab_range() {
        // Recipe mask_token_id=250624, vocab=250624 → [1, 250624).
        let mut rng = 0xC0FF_EE11;
        let noise = random_uniform_noise(1, 250624, 4096, &mut rng);
        assert!(noise.iter().all(|&t| (1..250624).contains(&t)));
        // high-1 is a legal draw (the id the previous exclusive high dropped).
        let mut rng = 0x1357_9BDF;
        let small = random_uniform_noise(1, 4, 4096, &mut rng);
        assert!(small.contains(&3), "exclusive high must be reachable as high-1");
        assert!(small.iter().all(|&t| (1..4).contains(&t)));
    }
}
