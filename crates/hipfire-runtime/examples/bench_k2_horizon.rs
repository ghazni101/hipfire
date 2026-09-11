// SPDX-License-Identifier: Apache-2.0
//! Focused perf benchmark for the K2-Horizon (MoVA-36B-A4B) decode path.
//!
//! Drives `decode_step_sampled` in-process so rocprofv3 can capture a clean
//! per-kernel trace (the daemon never exits, so its .dat never finalizes).
//!
//! Usage:
//!   bench_k2_horizon <model.mq4r> [--prefill N] [--warmup N] [--gen N]
//!
//! Env:
//!   HIPFIRE_KV_MODE=q8          KV cache mode (default q8)
//!   HIPFIRE_DPM_WARMUP_SECS     DPM stabilization before timed decode
//!   HIPFIRE_REPLAY_BACKEND      hip | pm4 | auto (default: auto → PM4 on mq4r)

use hipfire_arch_k2_horizon as k2;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_k2_horizon <model.mq4r> [--prefill N] [--warmup N] [--gen N]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let mut prefill_len = 32usize;
    let mut warmup_len = 3usize;
    let mut gen_len = 50usize;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => {
                prefill_len = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--warmup" => {
                warmup_len = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--gen" => {
                gen_len = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }

    eprintln!("=== bench_k2_horizon ===");
    eprintln!("Model: {model_path}");
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = <k2::K2Horizon as Architecture>::config_from_hfq(&hfq).expect("config");
    eprintln!(
        "Config: dim={} layers={} heads={} kv_heads={} vocab={} experts={}x{}",
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.num_experts,
        config.num_experts_per_tok
    );
    let t_load = Instant::now();
    let weights = <k2::K2Horizon as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
        .expect("load_weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let max_seq = (prefill_len + warmup_len + gen_len + 16).max(512);
    let mut state =
        k2::K2HorizonState::new_with_max_seq(&mut gpu, &config, max_seq).expect("state");

    // Prefill with a fixed token id (BOS-ish); content doesn't matter for perf.
    let prompt: Vec<u32> = vec![1u32; prefill_len];
    let ps = k2::prefill::PrefillScratch::new(&mut gpu, &config).expect("prefill scratch");
    let t_prefill = Instant::now();
    let logits =
        k2::prefill::forward_prefill_batch(&config, &weights, &ps, &mut state, &mut gpu, &prompt)
            .expect("prefill");
    eprintln!(
        "prefill: {} tokens in {:.1}ms ({:.1} tok/s)",
        prefill_len,
        t_prefill.elapsed().as_secs_f64() * 1000.0,
        prefill_len as f64 / t_prefill.elapsed().as_secs_f64()
    );
    let mut next_tok = llama::argmax(&logits);
    state.n_tokens = prefill_len;

    // Warmup (untimed; lets PM4 arm + JIT settle).
    for _ in 0..warmup_len {
        let pos = state.n_tokens as u32;
        let (tok, _rng) = k2::forward::decode_step_sampled(
            &config, &weights, &mut state, &mut gpu, next_tok, pos, 0.0, 1.0, 0x1234,
        )
        .expect("warmup decode");
        next_tok = tok;
        state.n_tokens += 1;
    }

    if let Ok(s) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
        let secs: f32 = s.parse().unwrap_or(0.0);
        if secs > 0.0 {
            gpu.dpm_warmup(secs).expect("dpm warmup");
        }
    }

    // Timed decode.
    let mut per_token_ms = Vec::with_capacity(gen_len);
    let t_gen = Instant::now();
    for _ in 0..gen_len {
        let pos = state.n_tokens as u32;
        let t = Instant::now();
        let (tok, _rng) = k2::forward::decode_step_sampled(
            &config, &weights, &mut state, &mut gpu, next_tok, pos, 0.0, 1.0, 0x1234,
        )
        .expect("gen decode");
        per_token_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        next_tok = tok;
        state.n_tokens += 1;
    }
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    per_token_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = gen_ms / per_token_ms.len() as f64;
    let p50 = per_token_ms[per_token_ms.len() / 2];
    let min = per_token_ms[0];
    let max = *per_token_ms.last().unwrap();
    eprintln!(
        "gen: {} tokens, avg={:.2}ms p50={:.2} min={:.2} max={:.2} → {:.1} tok/s",
        per_token_ms.len(),
        avg,
        p50,
        min,
        max,
        1000.0 / avg
    );
    println!(
        "SUMMARY gen_tok_s={:.1} avg_ms={:.2} p50_ms={:.2}",
        1000.0 / avg,
        avg,
        p50
    );
}
