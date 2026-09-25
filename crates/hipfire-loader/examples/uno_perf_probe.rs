// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! uno_perf_probe: phase-attributed perf A/B for the K2-Horizon Uno speculator.
//!
//! Loads the target + conditional-LoRA adapter through the production loader
//! (`load_model` with `draft_path` = the adapter dir), then measures:
//!
//!   A. AR decode        — per-token `forward_scratch_{embed,compute}` + logits
//!                         download + CPU argmax (the uncaptured dense path).
//!   B. Uno greedy step  — full `Speculator::step` windows at temp=0.
//!   C. Uno fused step   — full `Speculator::step` windows at temp=1.0
//!                         (unfiltered → GPU fused verify).
//!   D. Phase split      — the two `uno_forward_batch` calls (draft w/ LoRA,
//!                         verify base) timed alone, brackets GPU+download time;
//!                         (step total − forwards) is host-side acceptance cost.
//!
//! Usage:
//!   uno_perf_probe MODEL.mq4 ADAPTER_DIR [AR_TOKENS [WINDOWS]]

use hipfire_runtime::llama;
use hipfire_runtime::spec::{PrefillOutcome, Speculator};
use std::time::Instant;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap()
        .0 as u32
}

/// Repeat/presence/frequency penalty config for the mirror, applied through the
/// SAME GPU kernel the speculator uses (`apply_repeat_penalty_row`) but on
/// *scalar* logits. That is the point of this harness: it removes the
/// batched-vs-scalar forward difference from the comparison, so a mismatch
/// indicts the penalty LAW/HISTORY rather than the window kernels.
struct Pen {
    rp: f32,
    pp: f32,
    fp: f32,
    rw: usize,
    buf: Option<rdna_compute::GpuTensor>,
}

impl Pen {
    fn neutral() -> Pen {
        Pen { rp: 1.0, pp: 0.0, fp: 0.0, rw: 0, buf: None }
    }

    fn from_env(gpu: &mut rdna_compute::Gpu) -> Pen {
        let rp: f32 = std::env::var("HIPFIRE_BENCH_REPEAT_PENALTY").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let pp: f32 = std::env::var("HIPFIRE_BENCH_PRESENCE_PENALTY").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let fp: f32 = std::env::var("HIPFIRE_BENCH_FREQUENCY_PENALTY").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0.0);
        // AR's effective window is `min(repeat_window, 64)`.
        let rw: usize = std::env::var("HIPFIRE_BENCH_REPEAT_WINDOW").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(128usize).min(64);
        let active = rw > 0 && (rp > 1.0 || pp > 0.0 || fp > 0.0);
        let buf = if active {
            gpu.alloc_tensor(&[64], rdna_compute::DType::F32).ok()
        } else {
            None
        };
        eprintln!("[pen] mirror penalty rp={rp} pp={pp} fp={fp} rw={rw} active={active}");
        Pen { rp, pp, fp, rw: if active { rw } else { 0 }, buf }
    }

    /// Apply in place to a logits tensor, using the last `rw` of `history`
    /// (oldest first, and history must already END with the token being
    /// forwarded — AR pushes the sampled token before uploading).
    fn apply(
        &self,
        gpu: &mut rdna_compute::Gpu,
        logits: &rdna_compute::GpuTensor,
        history: &[u32],
        vocab: usize,
    ) -> Result<(), String> {
        let Some(buf) = self.buf.as_ref() else { return Ok(()) };
        if self.rw == 0 || history.is_empty() {
            return Ok(());
        }
        let last = history.len().min(self.rw);
        let window = &history[history.len() - last..];
        let bytes: Vec<u8> = window.iter().flat_map(|t| t.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&buf.buf, &bytes).map_err(|e| format!("{e:?}"))?;
        gpu.apply_repeat_penalty_row(
            logits, buf, vocab, window.len(), self.rp, self.pp, self.fp,
        )
        .map_err(|e| format!("{e:?}"))
    }
}

/// One greedy AR decode on the mirror KV/scratch; returns the picked token
/// and the top-2 logit gap (batched-vs-scalar kernels differ by ULPs, so a
/// near-tie argmax flip is not a correctness bug — a wide-gap flip is).
///
/// `history` is the committed conversation with the token being forwarded
/// already appended; `pen` applies AR's penalty law to the scalar logits.
fn ar_pick(
    gpu: &mut rdna_compute::Gpu,
    bundle: &mut hipfire_arch_llama::LlamaBundle,
    ar_scratch: &llama::ForwardScratch,
    ar_kv: &mut llama::KvCache,
    token: u32,
    pos: usize,
    history: &[u32],
    pen: &Pen,
) -> (u32, f32) {
    llama::forward_scratch_embed(gpu, &bundle.weights, &bundle.config, token, pos, ar_scratch)
        .map_err(|e| format!("{e:?}"))
        .unwrap();
    llama::forward_scratch_compute(gpu, &bundle.weights, &bundle.config, pos, ar_kv, ar_scratch)
        .map_err(|e| format!("{e:?}"))
        .unwrap();
    pen.apply(gpu, &ar_scratch.logits, history, bundle.config.vocab_size)
        .map_err(|e| format!("mirror penalty: {e}"))
        .unwrap();
    let logits = gpu.download_f32(&ar_scratch.logits).map_err(|e| format!("{e:?}")).unwrap();
    let mut best = (f32::NEG_INFINITY, 0usize);
    let mut second = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best.0 {
            second = best.0;
            best = (v, i);
        } else if v > second {
            second = v;
        }
    }
    (best.1 as u32, best.0 - second)
}


/// Token-identity check: decode the mirror AR chain over `emit` and require
/// exact agreement. The top-2 logit gap is printed on mismatch so a near-tie
/// is visible in the failure, but a mismatch still fails.
fn check_identity(
    gpu: &mut rdna_compute::Gpu,
    target: &mut dyn hipfire_runtime::spec::SpecTarget,
    ar_scratch: &llama::ForwardScratch,
    ar_kv: &mut llama::KvCache,
    seed: u32,
    position: usize,
    emit: &[u32],
    tag: &str,
    history: &mut Vec<u32>,
    pen: &Pen,
) -> Result<Vec<u32>, String> {
    // Any token mismatch fails. The top-2 gap is printed beside it.
    let bundle = target
        .as_any_mut()
        .downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        .ok_or("target is not LlamaBundle")?;
    let mut picks = Vec::with_capacity(emit.len());
    let mut previous = seed;
    for (row, &token) in emit.iter().enumerate() {
        let (expected, gap) = ar_pick(
            gpu, bundle, ar_scratch, ar_kv, previous, position + row, history, pen,
        );
        if std::env::var("HIPFIRE_UNO_IDENTITY_VERBOSE").is_ok() {
            eprintln!(
                "[ident] pos={} mirror={expected} committed={token} top2_gap={gap:.6}{}",
                position + row + 1,
                if token == expected { "" } else { "  <-- MISMATCH" }
            );
        }
        if token != expected {
            let wlin = history.len().min(pen.rw.max(1));
            let w = &history[history.len() - wlin.min(history.len())..];
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &t in w.iter() {
                h ^= t as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            eprintln!(
                "[mirror-win] pred_pos={} win_len={} hist_len={} fnv={h:016x}",
                position + row + 1,
                w.len(),
                history.len()
            );
            let tail = &history[history.len().saturating_sub(8)..];
            let mut with_token = history.clone();
            with_token.push(token);
            return Err(format!(
                "{tag}: UNO diverged from AR at pos {} (expected {expected}, got {token}, top2 gap {gap:.4})\n  \
                 mirror hist len={} tail={tail:?}\n  \
                 uno   hist len={} tail={:?}",
                position + row + 1,
                history.len(),
                with_token.len(),
                &with_token[with_token.len().saturating_sub(8)..],
            ));
        }
        picks.push(expected);
        // Teacher-force UNO's committed token: the main path's KV holds it,
        // so the mirror must follow to stay aligned at later positions. It also
        // becomes the mirror's next forwarded token, so it joins the history.
        previous = token;
        history.push(token);
        if matches!(previous, 1 | 250019) { break; }
    }
    Ok(picks)
}

fn stats(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let mid = s.len() / 2;
    let median = if s.len() % 2 == 1 {
        s[mid]
    } else {
        (s[mid - 1] + s[mid]) / 2.0
    };
    (median, s.iter().sum::<f64>() / s.len() as f64)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: uno_perf_probe MODEL.mq4 ADAPTER_DIR [AR_TOKENS [WINDOWS]]".into());
    }
    let ar_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48);
    let windows: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(24);

    let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("Gpu::init: {e:?}"))?;
    eprintln!("[perf] GPU: {}", gpu.arch);

    let max_seq = 2048usize;
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg::default();
    let mut m = hipfire_loader::load_model(
        &args[1],
        max_seq,
        Some(&args[2]),
        Some("q8"),
        None,
        None,
        &cask,
        1,
        spec_cfg,
        &mut gpu,
    )?;
    eprintln!("[perf] model loaded (arch_id={})", m.arch_id);

    let spec = m
        .speculator
        .as_mut()
        .ok_or("no speculator loaded — draft_path must be the Uno adapter dir")?;
    let arch_id = m.arch_id;
    let carrier = hipfire_loader::carrier_for(arch_id)
        .ok_or_else(|| format!("no carrier for arch_id {arch_id}"))?;
    let mut guard = carrier
        .spec_target_guard(&mut m.state, &m.model_path)
        .map_err(|e| format!("spec_target_guard: {e}"))?;
    let target = guard.slot().map_err(|e| format!("guard.slot: {e}"))?;

    // Prompt (ChatML-ish, matches the parity probe).
    let prompt_owned;
    let prompt: &str = match std::env::var("HIPFIRE_UNO_PROBE_PROMPT") {
        Ok(path) => {
            prompt_owned = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
            &prompt_owned
        }
        Err(_) => {
            "<|ifm|im_start|>user\nWrite a short paragraph about why the sky is blue.<|ifm|im_end|><|ifm|im_start|>assistant\n"
        }
    };
    let tokenizer = m.tokenizer.as_ref().ok_or("no tokenizer")?;
    let bos = target
        .as_any_mut()
        .downcast_ref::<hipfire_arch_llama::LlamaBundle>()
        .ok_or("target is not LlamaBundle")?
        .config
        .bos_token;
    let mut prompt_tokens: Vec<u32> = Vec::new();
    prompt_tokens.push(bos);
    prompt_tokens.extend(tokenizer.encode(prompt));
    eprintln!("[perf] prompt tokens: {}", prompt_tokens.len());

    // ── Prefill through the speculator (same as the daemon path) ──
    let out = spec.prefill(
        &mut gpu,
        target,
        &prompt_tokens,
        &prompt_tokens,
        0,
        false,
        None,
        &|| false,
    )?;
    let mut seed = match out {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => return Err("prefill aborted".into()),
    };
    let mut position = prompt_tokens.len();

    // ── A. AR decode baseline (uncaptured scratch path) ──
    // A mirror KV+scratch decodes the AR reference alongside every UNO
    // configuration; identity is on by default (UNO_IDENTITY=0 to skip).
    let identity = std::env::var("UNO_IDENTITY").ok().as_deref() != Some("0");
    let bundle_for_mirror = target
        .as_any_mut()
        .downcast_mut::<hipfire_arch_llama::LlamaBundle>()
        .ok_or("target is not LlamaBundle")?;
    let mut ar_kv = llama::KvCache::new_gpu_q8(
        &mut gpu,
        bundle_for_mirror.config.n_layers,
        bundle_for_mirror.config.n_kv_heads,
        bundle_for_mirror.config.head_dim,
        2048,
    )
    .map_err(|e| format!("mirror KV: {e:?}"))?;
    let ar_scratch = llama::ForwardScratch::new_with_max_seq(&mut gpu, &bundle_for_mirror.config, 2048)
        .map_err(|e| format!("mirror scratch: {e:?}"))?;
    for (i, &token) in prompt_tokens.iter().enumerate() {
        llama::forward_scratch_embed(&mut gpu, &bundle_for_mirror.weights, &bundle_for_mirror.config, token, i, &ar_scratch)
            .map_err(|e| format!("{e:?}"))?;
        llama::forward_scratch_compute(&mut gpu, &bundle_for_mirror.weights, &bundle_for_mirror.config, i, &mut ar_kv, &ar_scratch)
            .map_err(|e| format!("{e:?}"))?;
    }
    let mirror_first = argmax(&gpu.download_f32(&ar_scratch.logits).map_err(|e| format!("{e:?}"))?);
    assert_eq!(mirror_first, seed, "mirror prefill disagrees with speculator first token");
    // Scalar-reference penalty (see `Pen`): the mirror applies the same law
    // through the same kernel, so identity failures here indict the law/history
    // rather than the batched window kernels.
    let pen = Pen::from_env(&mut gpu);
    let pen_neutral = Pen::neutral();
    // Committed conversation, oldest first, WITH the token being forwarded.
    // The prefill's first token is already committed (it is this window's seed).
    let mut history: Vec<u32> = prompt_tokens.clone();
    history.push(seed);
    // The daemon passes its committed-token list as `step`'s `emitted` argument
    // (the prefill token first, then every committed token). The probe used to
    // pass `&[]`, which left the speculator's penalty history without any
    // generated tokens — so the scalar-reference discriminator was comparing
    // against a *different window* and its failures said nothing about the law.
    let mut committed: Vec<u32> = vec![seed];
    {
        let bundle = target
            .as_any_mut()
            .downcast_mut::<hipfire_arch_llama::LlamaBundle>()
            .ok_or("target is not LlamaBundle")?;
        // The AR baseline decodes ON THE MIRROR KV so it stays in lockstep
        // with the identity gate (same frontier, same slots).
        let mut tok = seed;
        for i in 0..3 {
            let (next, _) = ar_pick(&mut gpu, bundle, &ar_scratch, &mut ar_kv, tok, position + i, &[], &pen_neutral);
            tok = next;
        }
        let mut ar_ms = Vec::new();
        let mut pos = position + 3;
        for _ in 0..ar_tokens {
            let t0 = Instant::now();
            let (next, _) = ar_pick(&mut gpu, bundle, &ar_scratch, &mut ar_kv, tok, pos, &[], &pen_neutral);
            ar_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            tok = next;
            pos += 1;
            if matches!(tok, 1 | 250019) {
                break;
            }
        }
        let (med, avg) = stats(&ar_ms);
        eprintln!("[AR]    tokens={} median={:.2} ms avg={:.2} ms  ({:.1} tok/s uncaptured)", ar_ms.len(), med, avg, 1000.0 / avg);
        // bundle.kv stays at the post-prefill frontier: the AR baseline ran
        // entirely on the mirror, and the spec windows must start where the
        // mirror's identity tracking starts (position = prompt len).
        let _ = tok;
    }

    // ── B/C. Uno spec windows (greedy, then temp=1.0 when fused is available) ──
    let mut emitted_texts: Vec<u32> = Vec::new();
    let mut mirror_texts: Vec<u32> = Vec::new();
    for (label, temp) in [("greedy", 0.0f32), ("temp1.0", 1.0)] {
        let probe_top_k: usize = std::env::var("HIPFIRE_UNO_TOP_K").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(0);
        let probe_top_p: f32 = std::env::var("HIPFIRE_UNO_TOP_P").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1.0);
        spec.configure_request(hipfire_runtime::spec::SpecRequestConfig {
            temp,
            top_k: probe_top_k,
            top_p: probe_top_p,
            repeat_penalty: pen.rp,
            repeat_window: pen.rw,
            presence_penalty: pen.pp,
            frequency_penalty: pen.fp,
            ..Default::default()
        });
        // Warmup 2 windows — identity-checked as well, so the mirror AR
        // decode stays in lockstep with the main path across every window.
        for _ in 0..2 {
            let step = spec.step(&mut gpu, target, position, seed, &committed, None, temp, usize::MAX)?;
            if step.emit.is_empty() {
                return Err("empty emit in warmup".into());
            }
            if identity && temp <= 1e-6 {
                let picks = check_identity(&mut gpu, target, &ar_scratch, &mut ar_kv, seed, position, &step.emit, "warmup", &mut history, &pen)?;
                mirror_texts.extend_from_slice(&picks);
            }
            committed.extend_from_slice(&step.emit);
            position += step.emit.len();
            seed = step.next_seed;
            if matches!(seed, 1 | 250019) {
                return Err("EOS during warmup".into());
            }
        }
        let mut win_ms = Vec::new();
        let mut emitted = 0usize;
        let mut accepted = 0usize;
        let mut proposed = 0usize;
        for _ in 0..windows {
            let t0 = Instant::now();
            let step = spec.step(&mut gpu, target, position, seed, &committed, None, temp, usize::MAX)?;
            win_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            // Token-identity gate: every committed UNO token must equal the
            // mirror AR decode.
            if identity && temp <= 1e-6 {
                let picks = check_identity(&mut gpu, target, &ar_scratch, &mut ar_kv, seed, position, &step.emit, "window", &mut history, &pen)?;
                mirror_texts.extend_from_slice(&picks);
            }
            emitted_texts.extend_from_slice(&step.emit);
            mirror_texts.extend_from_slice(&step.emit);
            committed.extend_from_slice(&step.emit);
            emitted += step.emit.len();
            accepted += step.accepted;
            proposed += step.proposed;
            position += step.emit.len();
            seed = step.next_seed;
            if matches!(seed, 1 | 250019) {
                break;
            }
        }
        if win_ms.is_empty() {
            continue;
        }
        let (med, avg) = stats(&win_ms);
        let tok_s = emitted as f64 / (win_ms.iter().sum::<f64>() / 1000.0);
        let nwin = win_ms.len().max(1) as f64;
        // IFM τ is tokens per (draft+verify) step. TPF is tokens per weight
        // pass = τ/2. `accepted/proposed` is the extra-token accept rate, not τ.
        let tau = emitted as f64 / nwin;
        let tpf = emitted as f64 / (nwin * 2.0);
        eprintln!(
            "[UNO {label}] windows={} emitted={} tau={:.3} tpf={:.2} extra_accept={:.3} median={:.2} ms/window avg={:.2} ms/window  ({:.1} tok/s) ms/token={:.2}{}",
            win_ms.len(),
            emitted,
            tau,
            tpf,
            accepted as f64 / proposed.max(1) as f64,
            med,
            avg,
            tok_s,
            win_ms.iter().sum::<f64>() / emitted as f64,
            if identity && temp <= 1e-6 { " identity=PASS" } else { "" },
        );
    }

    eprintln!("[text] UNO  : {:?}", tokenizer.decode(&emitted_texts));
    eprintln!("[text] MIRROR: {:?}", tokenizer.decode(&mirror_texts));

    // Phase split downloads the full vocab per row. Off unless UNO_PHASE=1;
    // the scale/TPF ablation only needs the window loop above.
    if std::env::var("UNO_PHASE").ok().as_deref() != Some("1") {
        return Ok(());
    }

    // ── D. Phase split: forwards alone (draft w/ LoRA + verify base) ──
    {
        use hipfire_arch_llama::uno::{uno_forward_batch, UnoBatchScratch};
        let bundle = target
            .as_any_mut()
            .downcast_mut::<hipfire_arch_llama::LlamaBundle>()
            .ok_or("target is not LlamaBundle")?;
        let adapter_dir = std::path::Path::new(&args[2]);
        // Re-open a private adapter + batch scratch against the same weights.
        let adapter = hipfire_arch_llama::uno::UnoAdapter::open(adapter_dir, &bundle.config, &mut gpu, 4, 1, 250623)
            .map_err(|e| format!("adapter: {e}"))?;
        let mut batch = UnoBatchScratch::new(&mut gpu, &bundle.config, adapter.rank, adapter.block_len, 512)
            .map_err(|e| format!("batch scratch: {e:?}"))?;
        let noise: Vec<u32> = std::iter::once(seed)
            .chain(std::iter::repeat(250623).take(3))
            .collect();
        let mut t_draft = Vec::new();
        let mut t_verify = Vec::new();
        for i in 0..8 {
            let t0 = Instant::now();
            let logits = uno_forward_batch(&mut gpu, &bundle.weights, &bundle.config, Some(&adapter), &noise, position + i * 8, &mut bundle.kv, &bundle.scratch, &batch)
                .map_err(|e| format!("{e:?}"))?;
            t_draft.push(t0.elapsed().as_secs_f64() * 1e3);
            let picks: Vec<u32> = logits.chunks_exact(bundle.config.vocab_size).map(argmax).collect();
            let t1 = Instant::now();
            uno_forward_batch(&mut gpu, &bundle.weights, &bundle.config, None, &picks, position + 1 + i * 8, &mut bundle.kv, &bundle.scratch, &batch)
                .map_err(|e| format!("{e:?}"))?;
            t_verify.push(t1.elapsed().as_secs_f64() * 1e3);
        }
        let (dm, da) = stats(&t_draft);
        let (vm, va) = stats(&t_verify);
        eprintln!("[phase] draft fwd (incl 4MB D2H): median={:.2} avg={:.2} ms | verify fwd (incl 4MB D2H): median={:.2} avg={:.2} ms", dm, da, vm, va);

        // ── E. Per-kernel GPU-time attribution (rdna-compute profile timers) ──
        for (label, gated) in [("verify", false), ("draft", true)] {
            rdna_compute::profile::start();
            for i in 0..4 {
                uno_forward_batch(&mut gpu, &bundle.weights, &bundle.config, gated.then_some(&adapter), &noise, position + 128 + i * 8, &mut bundle.kv, &bundle.scratch, &batch)
                    .map_err(|e| format!("{e:?}"))?;
            }
            let entries = rdna_compute::profile::stop().unwrap_or_default();
            let mut by_kernel: std::collections::HashMap<(&str, &str), (f64, usize)> = std::collections::HashMap::new();
            let mut total_us = 0.0;
            for e in &entries {
                let slot = by_kernel.entry((e.category, e.kernel)).or_insert((0.0, 0));
                slot.0 += e.time_us;
                slot.1 += 1;
                total_us += e.time_us;
            }
            let mut rows: Vec<_> = by_kernel.into_iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            eprintln!("[profile] {} forward x4: gpu kernel time {:.2} ms/forward (top 10):", label, total_us / 4.0 / 1000.0);
            for ((cat, k), (us, n)) in rows.into_iter().take(10) {
                eprintln!("[profile]   {:<10} {:<34} {:8.3} ms/fwd  ({} launches, {:.1} us avg)", cat, k, us / 4.0 / 1000.0, n / 4, us / n as f64);
            }
        }
        batch.free_gpu(&mut gpu);
        adapter.free_gpu(&mut gpu);
    }
    Ok(())
}
