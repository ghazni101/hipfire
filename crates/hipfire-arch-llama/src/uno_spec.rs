// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Conditional-LoRA Uno linear windows, using the base target as verifier.
//! Reference: https://github.com/ifm-ai/uno (Apache-2.0).
use crate::{uno::{uno_forward_row, UnoAdapter, UnoScratch}, LlamaBundle};
use hipfire_runtime::{llama, spec::{accept_greedy_prefix, PrefillOutcome, SpecAdvance, SpecGrammar, SpecStep, SpecTarget, Speculator}};
use rdna_compute::Gpu;

pub struct UnoSpeculator {
    adapter: UnoAdapter,
    scratch: UnoScratch,
    capacity: usize,
    noise_state: u64,
}

impl UnoSpeculator {
    pub fn load(gpu: &mut Gpu, target: &LlamaBundle, dir: &std::path::Path, capacity: usize) -> Result<Box<dyn Speculator>, String> {
        let c = &target.config;
        if c.dim != 4096 || c.n_layers != 36 || c.n_heads != 32 || c.n_kv_heads != 8 || c.head_dim != 128 || c.norm_groups != 4 || c.vocab_size != 250624 {
            return Err("Uno requires the K2-Horizon-7B target".into());
        }
        let adapter = UnoAdapter::open(dir, c, gpu, 4, 1, 250623)?;
        let scratch = match UnoScratch::new(gpu, c) {
            Ok(s) => s,
            Err(e) => { adapter.free_gpu(gpu); return Err(format!("Uno scratch: {e:?}")); }
        };
        Ok(Box::new(Self { adapter, scratch, capacity, noise_state: 42 }))
    }

    fn row(&mut self, gpu: &mut Gpu, target: &mut LlamaBundle, token: u32, pos: usize, gated: bool) -> Result<u32, String> {
        let logits = if gated {
            uno_forward_row(gpu, &target.weights, &target.config, &self.adapter, token, pos, true, &mut target.kv, &target.scratch, &mut self.scratch).map_err(|e| format!("Uno draft: {e:?}"))?
        } else {
            llama::forward_scratch_embed(gpu, &target.weights, &target.config, token, pos, &target.scratch).map_err(|e| format!("Uno base embed: {e:?}"))?;
            llama::forward_scratch_compute(gpu, &target.weights, &target.config, pos, &mut target.kv, &target.scratch).map_err(|e| format!("Uno base forward: {e:?}"))?;
            gpu.download_f32(&target.scratch.logits).map_err(|e| format!("Uno logits: {e:?}"))?
        };
        if logits.iter().any(|v| !v.is_finite()) { return Err("Uno nonfinite logits".into()); }
        Ok(llama::argmax(&logits))
    }
}

impl Speculator for UnoSpeculator {
    fn name(&self) -> &'static str { "uno" }
    fn prefill(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, _prompt_tokens: &[u32], tokens: &[u32], start: usize, cache_hit: bool, _resume: Option<usize>, abort: &dyn Fn() -> bool) -> Result<PrefillOutcome, String> {
        match target.spec_advance(gpu, tokens, start, !cache_hit, abort, None)? {
            SpecAdvance::Ready { last_argmax, .. } => Ok(PrefillOutcome::Ready { first_token: last_argmax }),
            SpecAdvance::Aborted => Ok(PrefillOutcome::Aborted),
        }
    }
    fn step(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget, position: usize, seed: u32, _emitted: &[u32], grammar: Option<&mut dyn SpecGrammar>, temp: f32, max_emit: usize) -> Result<SpecStep, String> {
        if max_emit == 0 || temp > 1e-6 || grammar.is_some() { return Err("Uno requires a positive output budget, greedy sampling, and no grammar".into()); }
        let target = target.as_any_mut().downcast_mut::<LlamaBundle>().ok_or("Uno target is not LlamaBundle")?;
        if target.kv.compact_offset != 0 { return Err("Uno does not support compacted KV".into()); }
        let budget = max_emit.min(self.capacity.saturating_sub(position)).min(self.adapter.block_len + 1);
        if budget == 0 { return Err("Uno context exhausted".into()); }
        let clean = self.row(gpu, target, seed, position, false)?;
        if budget == 1 || is_stop(clean) { return Ok(SpecStep::new([clean], clean, 0, 0)); }
        let n = budget - 2;
        let mut drafts = Vec::with_capacity(n);
        for row in 1..=n {
            // Deterministic uniform noise in the reference checkpoint's [1, mask) range.
            self.noise_state ^= self.noise_state << 13;
            self.noise_state ^= self.noise_state >> 7;
            self.noise_state ^= self.noise_state << 17;
            let token = self.adapter.noise_low + (self.noise_state % u64::from(self.adapter.noise_high - self.adapter.noise_low)) as u32;
            drafts.push(self.row(gpu, target, token, position + row, true)?);
        }
        let mut picks = Vec::with_capacity(n + 1);
        for (row, token) in std::iter::once(clean).chain(drafts.iter().copied()).enumerate() {
            picks.push(self.row(gpu, target, token, position + 1 + row, false)?);
        }
        let verdict = accept_greedy_prefix(&drafts, &picks, None);
        let mut emit: Vec<u32> = std::iter::once(clean).chain(verdict.committed).collect();
        if let Some(stop) = emit.iter().position(|&t| is_stop(t)) { emit.truncate(stop + 1); }
        let accepted = verdict.accepted.min(emit.len().saturating_sub(1));
        let next = *emit.last().unwrap();
        // Pure attention: valid seed/accepted-prefix KV is already base-only.
        // Rejected slots remain outside the next position's attention range.
        Ok(SpecStep::new(emit, next, n, accepted))
    }
    fn repair_terminal_prefix(&mut self, _gpu: &mut Gpu, _target: &mut dyn SpecTarget, _start: usize, _seed: u32, _consumed: &[u32]) -> Result<bool, String> { Ok(true) }
    fn reset(&mut self, _gpu: &mut Gpu) -> Result<(), String> { self.noise_state = 42; Ok(()) }
    fn block_size(&self) -> usize { self.adapter.block_len + 1 }
    fn ctx_capacity(&self) -> usize { self.capacity }
    fn free(self: Box<Self>, gpu: &mut Gpu) { self.scratch.free_gpu(gpu); self.adapter.free_gpu(gpu); }
}

fn is_stop(token: u32) -> bool { matches!(token, 1 | 250019) }
