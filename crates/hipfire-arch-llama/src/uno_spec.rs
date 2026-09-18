// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Conditional-LoRA Uno linear windows, using the base target as verifier.
//! Reference: https://github.com/ifm-ai/uno (Apache-2.0).
use crate::{uno::{uno_forward_row, uno_forward_batch, uno_forward_batch_device, UnoAdapter, UnoScratch, UnoBatchScratch}, LlamaBundle};
use hipfire_runtime::{llama, spec::{request_rng_state, PrefillOutcome, SpecAdvance, SpecGrammar, SpecRequestConfig, SpecStep, SpecTarget, Speculator}};
use rdna_compute::{Gpu, GpuTensor};

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
    noise_state: u64,
    noise_mode: NoiseMode,
    request: SpecRequestConfig,
    rng: u32,
    /// Ψ-Spec tree verify config: (max tree nodes incl. root, candidate top-k
    /// per depth). `None` = linear chain windows.
    tree: Option<(usize, usize)>,
}

impl UnoSpeculator {
    pub fn load(gpu: &mut Gpu, target: &LlamaBundle, dir: &std::path::Path, capacity: usize) -> Result<Box<dyn Speculator>, String> {
        let c = &target.config;
        if c.dim != 4096 || c.n_layers != 36 || c.n_heads != 32 || c.n_kv_heads != 8 || c.head_dim != 128 || c.norm_groups != 4 || c.vocab_size != 250624 {
            return Err("Uno requires the K2-Horizon-7B target".into());
        }
        // Diffusion block length L (window = L rows: seed + L-1 noise). The
        // reference implementation defaults to 8; 4 was the bring-up value.
        // Tunable because the window arithmetic dominates: two full forwards
        // amortize over up to L+1 emitted tokens.
        let block_len = hipfire_config::developer_var("HIPFIRE_UNO_BLOCK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|b| b.clamp(2, 8))
            .unwrap_or(4);
        // Ψ-Spec tree verify (upstream "tree sampler for high per-request
        // throughput"): HIPFIRE_UNO_TREE = max tree nodes incl. root (0 =
        // linear chain), HIPFIRE_UNO_TREE_K = candidates exposed per depth.
        let tree_nodes = hipfire_config::developer_var("HIPFIRE_UNO_TREE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.clamp(0, 64))
            .unwrap_or(0);
        let tree_top_k = hipfire_config::developer_var("HIPFIRE_UNO_TREE_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|k| k.clamp(1, 32))
            .unwrap_or(8);
        let tree = (tree_nodes >= 2).then_some((tree_nodes, tree_top_k));
        let noise_mode = hipfire_config::developer_var("HIPFIRE_UNO_NOISE")
            .ok()
            .map(|s| NoiseMode::parse(&s))
            .unwrap_or(NoiseMode::RandomUniform);
        let batch_rows = tree.map_or(block_len, |(nodes, _)| nodes.max(block_len));
        let adapter = UnoAdapter::open(dir, c, gpu, block_len, 1, 250623)?;
        let scratch = match UnoScratch::new(gpu, c) {
            Ok(s) => s,
            Err(e) => { adapter.free_gpu(gpu); return Err(format!("Uno scratch: {e:?}")); }
        };
        let batch = if gpu.arch == "gfx1101" && target.kv.quant_q8 {
            match UnoBatchScratch::new(gpu, c, adapter.rank, batch_rows, target.kv.physical_cap) {
                Ok(batch) => Some(batch),
                Err(e) => { scratch.free_gpu(gpu); adapter.free_gpu(gpu); return Err(format!("Uno batch scratch: {e:?}")); }
            }
        } else { None };
        Ok(Box::new(Self { adapter, scratch, batch, capacity, noise_state: 42, noise_mode, request: SpecRequestConfig::default(), rng: 0x1357_9BDF, tree }))
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
    fn draw_noise(&mut self, n: usize, seed_token: u32) -> Vec<u32> {
        let low = self.adapter.noise_low;
        let high = self.adapter.noise_high;
        match self.noise_mode {
            NoiseMode::Mask => vec![high; n],
            NoiseMode::RandomUniform => random_uniform_noise(low, high, n, &mut self.rng),
            NoiseMode::DeterministicUniform => {
                // Advance the per-request stream once per window so consecutive
                // windows decorrelate while staying reproducible from the
                // request RNG seed (configure_request re-seeds noise_state).
                self.noise_state = self.noise_state.wrapping_add(0x9E37_79B1_85EB_CA87);
                deterministic_uniform_noise(low, high, seed_token, self.noise_state, n)
            }
        }
    }

    /// Ψ-Spec tree window (upstream tree sampler): draft once, grow a
    /// best-first candidate tree from the per-depth draft top-k, verify the
    /// whole tree in ONE tree-attention forward, walk the target picks
    /// through it, and compact the accepted path's KV into the committed
    /// slots. Lossless by the same per-position argument as upstream: every
    /// committed token is the target's own draw at its prefix.
    fn step_tree(&mut self, gpu: &mut Gpu, target: &mut LlamaBundle, position: usize, seed: u32, budget: usize, temp: f32, greedy: bool) -> Result<SpecStep, String> {
        let (max_nodes, top_k) = self.tree.ok_or("Uno tree config missing")?;
        let vocab = target.config.vocab_size;
        let n = budget.saturating_sub(2).min(max_nodes.saturating_sub(1));
        let mut noise = self.draw_noise(n, seed);
        noise.insert(0, seed);
        let batch = self.batch.as_ref().ok_or("Uno batch scratch missing")?;

        // DRAFT: conditional-LoRA forward over [seed, noise...] — logits stay
        // on device in batch.draft_logits.
        uno_forward_batch_device(gpu, &target.weights, &target.config, Some(&self.adapter),
            &noise, position, &mut target.kv, &target.scratch, batch, &batch.draft_logits)
            .map_err(|e| format!("Uno draft forward: {e:?}"))?;
        let rows = n + 1;
        let inv_temp = if greedy { 1.0 } else { 1.0 / temp };

        // Root token: the target's own draw after the seed.
        if greedy {
            gpu.argmax_f32_batched(&batch.draft_logits, &batch.picks, vocab, rows)
                .map_err(|e| format!("Uno tree argmax: {e:?}"))?;
        }
        let clean = if greedy {
            let mut raw = [0u8; 4];
            gpu.hip.memcpy_dtoh(&mut raw, &batch.picks.buf).map_err(|e| format!("Uno tree picks: {e:?}"))?;
            u32::from_ne_bytes(raw)
        } else {
            sample_row(gpu, &batch.draft_logits.sub_offset(0, vocab), vocab, temp,
                &batch.sample_result, &batch.sample_repeat, &mut self.rng)?
        };
        if budget == 1 || is_stop(clean) {
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
            &mut target.kv, &target.scratch, batch, greedy, temp, &mut self.rng,
        ).map_err(|e| format!("Uno tree verify: {e:?}"))?;
        let (committed, path) = crate::uno::walk_draft_tree(&tree, &picks, (n) as i32, is_stop);
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
    fn step_device(&mut self, gpu: &mut Gpu, target: &mut LlamaBundle, position: usize, seed: u32, budget: usize, temp: f32, greedy: bool) -> Result<SpecStep, String> {
        let n = budget.saturating_sub(2);
        let vocab = target.config.vocab_size;
        let rows = n + 1;
        // Filtered (top-k-only) device path: mask draft + verify logits to
        // their top-K support so the fused dual-softmax kernel computes the
        // filtered p/q — the upstream `build_sparse_top_k_probs` support.
        // Greedy ignores the filter (argmax); top-p/min-p-only stay host-side.
        let dev_topk = if greedy { None } else {
            (self.request.top_k > 0 && self.request.top_k < vocab
                && !(self.request.top_p > 0.0 && self.request.top_p < 1.0)
                && self.request.min_p <= 0.0)
                .then_some(self.request.top_k.min(1024))
        };
        let mut noise = self.draw_noise(n, seed);
        noise.insert(0, seed);
        let batch = self.batch.as_ref().ok_or("Uno batch scratch missing")?;
        // DRAFT: conditional-LoRA forward, logits stay in batch.draft_logits.
        uno_forward_batch_device(gpu, &target.weights, &target.config, Some(&self.adapter),
            &noise, position, &mut target.kv, &target.scratch, batch, &batch.draft_logits)
            .map_err(|e| format!("Uno draft forward: {e:?}"))?;
        if let Some(k) = dev_topk {
            gpu.uno_apply_topk_mask(&batch.draft_logits, rows, vocab, k)
                .map_err(|e| format!("Uno draft topk mask: {e:?}"))?;
        }
        let mut proposal: Vec<u32> = Vec::with_capacity(rows);
        if greedy {
            let picks = argmax_rows(gpu, &batch.draft_logits, &batch.picks, vocab, rows)?;
            proposal.extend_from_slice(&picks);
        } else {
            for row in 0..rows {
                proposal.push(sample_row(gpu, &batch.draft_logits.sub_offset(row * vocab, vocab),
                    vocab, temp, &batch.sample_result, &batch.sample_repeat, &mut self.rng)?);
            }
        }
        let clean = proposal[0];
        if budget == 1 || is_stop(clean) { return Ok(SpecStep::new([clean], clean, 0, 0)); }
        // VERIFY: base-only forward, logits stay in batch.verify_logits.
        uno_forward_batch_device(gpu, &target.weights, &target.config, None,
            &proposal, position + 1, &mut target.kv, &target.scratch, batch, &batch.verify_logits)
            .map_err(|e| format!("Uno verify forward: {e:?}"))?;
        if let Some(k) = dev_topk {
            gpu.uno_apply_topk_mask(&batch.verify_logits, rows, vocab, k)
                .map_err(|e| format!("Uno verify topk mask: {e:?}"))?;
        }
        let mut emit = Vec::with_capacity(rows + 1);
        emit.push(clean);
        let mut accepted = 0;
        if greedy {
            let picks = argmax_rows(gpu, &batch.verify_logits, &batch.picks, vocab, rows)?;
            for row in 0..n {
                emit.push(picks[row]);
                if picks[row] != proposal[row + 1] { break; }
                accepted += 1;
                if is_stop(picks[row]) { break; }
            }
            if accepted == n && !is_stop(*emit.last().unwrap()) {
                emit.push(picks[n]);
            }
        } else {
            if n > 0 {
                let bytes: Vec<u8> = proposal[1..].iter().flat_map(|t| t.to_ne_bytes()).collect();
                gpu.hip.memcpy_htod(&batch.proposals.buf, &bytes).map_err(|e| format!("Uno proposals: {e:?}"))?;
                // Reserve a fresh request-seeded stream for each verification window.
                let _ = uniform(&mut self.rng);
                gpu.uno_verify_logits(&batch.verify_logits, &batch.draft_logits.sub_offset(vocab, n * vocab),
                    &batch.proposals, &batch.decisions, n, vocab, temp, self.rng)
                    .map_err(|e| format!("Uno fused verify: {e:?}"))?;
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
                    if is_stop(*emit.last().unwrap()) { break; }
                }
            }
            if accepted == n && !is_stop(*emit.last().unwrap()) {
                emit.push(sample_row(gpu, &batch.verify_logits.sub_offset(n * vocab, vocab),
                    vocab, temp, &batch.sample_result, &batch.sample_repeat, &mut self.rng)?);
            }
        }
        let next = *emit.last().unwrap();
        Ok(SpecStep::new(emit, next, n, accepted))
    }
}

impl Speculator for UnoSpeculator {
    fn name(&self) -> &'static str { "uno" }
    fn requires_greedy(&self) -> bool { false }
    fn supports_temp_verify(&self) -> bool { true }
    fn supports_chain_nucleus_verify(&self) -> bool { true }
    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
        self.rng = request_rng_state(cfg.rng_seed) as u32;
        self.noise_state = u64::from(self.rng).max(1);
    }
    fn prefill(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, _prompt_tokens: &[u32], tokens: &[u32], start: usize, cache_hit: bool, _resume: Option<usize>, abort: &dyn Fn() -> bool) -> Result<PrefillOutcome, String> {
        match target.spec_advance(gpu, tokens, start, !cache_hit, abort, None)? {
            SpecAdvance::Ready { .. } => {
                let target = target.as_any_mut().downcast_mut::<LlamaBundle>().ok_or("Uno target is not LlamaBundle")?;
                Ok(PrefillOutcome::Ready { first_token: self.sample_target(gpu, target)? })
            }
            SpecAdvance::Aborted => Ok(PrefillOutcome::Aborted),
        }
    }
    fn step(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, position: usize, seed: u32, _emitted: &[u32], grammar: Option<&mut dyn SpecGrammar>, temp: f32, max_emit: usize) -> Result<SpecStep, String> {
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
        // top-k-only filters are device-resident: step_device masks the draft
        // and verify logits to their top-K support so the fused kernel computes
        // the filtered p/q. top-p/min-p-only filtering keeps the host
        // Distribution law (the fused kernel's full-vocab softmax can't
        // express a nucleus support without the top-K selection).
        let dev_filterable = filtered
            && !(self.request.top_p > 0.0 && self.request.top_p < 1.0)
            && self.request.min_p <= 0.0;
        if self.batch.is_some() && (greedy || !filtered || dev_filterable) {
            if self.tree.is_some() {
                return self.step_tree(gpu, target, position, seed, budget, temp, greedy);
            }
            return self.step_device(gpu, target, position, seed, budget, temp, greedy);
        }
        let mut noise = self.draw_noise(n, seed);
        noise.insert(0, seed);
        let draft_logits = self.forward(gpu, target, &noise, position, true)?;
        let clean = Distribution::from_logits(&draft_logits[..vocab], self.request)?.sample(&mut self.rng);
        if budget == 1 || is_stop(clean) { return Ok(SpecStep::new([clean], clean, 0, 0)); }
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
            if is_stop(next) { break; }
        }
        if accepted == n && !is_stop(*emit.last().unwrap()) {
            emit.push(Distribution::from_logits(&verify_logits[n * vocab..(n + 1) * vocab], self.request)?.sample(&mut self.rng));
        }
        let next = *emit.last().unwrap();
        Ok(SpecStep::new(emit, next, n, accepted))
    }

    fn repair_terminal_prefix(&mut self, _gpu: &mut Gpu, _target: &mut dyn SpecTarget, _start: usize, _seed: u32, _consumed: &[u32]) -> Result<bool, String> { Ok(true) }
    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> { self.noise_state = 42; Ok(()) }
    fn block_size(&self) -> usize { self.adapter.block_len + 1 }
    fn ctx_capacity(&self) -> usize { self.capacity }
    fn free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(batch) = self.batch { batch.free_gpu(gpu); }
        self.scratch.free_gpu(gpu);
        self.adapter.free_gpu(gpu);
    }
}

fn is_stop(token: u32) -> bool { matches!(token, 1 | 250019) }

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
/// unfiltered only (the caller routes filtered requests to the host law).
pub(crate) fn sample_row(gpu: &mut Gpu, logits: &GpuTensor, vocab: usize, temp: f32,
    result: &GpuTensor, repeat: &GpuTensor, rng: &mut u32) -> Result<u32, String> {
    let (tok, next) = gpu
        .sample_top_p_pf(logits, result, repeat, vocab, temp, 1.0, *rng, 0, 1.0, 0.0, 0.0, None, None)
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
    let span = (high - low) as f64;
    (0..n)
        .map(|_| low + (uniform(rng) * span) as u32)
        .collect()
}

/// Upstream `deterministic_uniform` noise rows: a splitmix64 stream keyed by
/// `stream` and the window `seed_token`, drawing `low + (mix % span)`.
fn deterministic_uniform_noise(low: u32, high: u32, seed_token: u32, stream: u64, n: usize) -> Vec<u32> {
    let span = (high - low) as u64;
    let mut s = stream ^ (u64::from(seed_token).wrapping_mul(0x1656_67B1_9E37_79F9));
    (0..n)
        .map(|_| {
            s = mix_u64(s.wrapping_add(0x9E37_79B1_85EB_CA87));
            low + (s % span) as u32
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
        let noise = random_uniform_noise(1, 250623, 8192, &mut rng);
        assert!(noise.iter().all(|&t| (1..250623).contains(&t)), "random_uniform must draw from [low, high)");
        // Two fresh streams with the same seed replay identically.
        let mut rng_a = 0x1357_9BDF;
        let mut rng_b = 0x1357_9BDF;
        assert_eq!(
            random_uniform_noise(1, 250623, 16, &mut rng_a),
            random_uniform_noise(1, 250623, 16, &mut rng_b),
            "must replay from a fresh seed"
        );
    }

    #[test]
    fn deterministic_noise_is_reproducible_and_in_bounds() {
        let a = deterministic_uniform_noise(1, 250623, 42, 0xA_B_C, 256);
        let b = deterministic_uniform_noise(1, 250623, 42, 0xA_B_C, 256);
        assert_eq!(a, b, "deterministic mode must reproduce for a given seed/stream");
        assert!(a.iter().all(|&t| (1..250623).contains(&t)), "deterministic rows in [low, high)");
        // A different window stream (or seed token) decorrelates.
        assert_ne!(a, deterministic_uniform_noise(1, 250623, 42, 0xA_B_C + 1, 256),
            "a new window must not resample identical noise");
        assert_ne!(a, deterministic_uniform_noise(1, 250623, 43, 0xA_B_C, 256),
            "a different seed token must decorrelate");
    }
}
