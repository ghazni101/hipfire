// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P5 GPU oracle: cross-session radix prefix reuse on the slot engine.
//!
//! Sequential greedy requests sharing a >=128-token prefix must:
//!   * report `Accepted.reused >= 128` on the second request,
//!   * produce the same continuation tokens as a cold first request when
//!     the prompt is identical,
//!   * not replay another branch for a divergent suffix.
//!
//! Build: `cargo run -p hipfire-runtime --example test_serve_prefix_cache --features lab -- <model.hfq>`

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features lab (deltanet is default)");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::serve::{Continuation, Event, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;

    const MAX_TOKENS: usize = 8;
    const PAGE: usize = 128;

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.5-4b.mq4v2.hfq")
    });
    println!("=== test_serve_prefix_cache ===\nmodel: {model_path}");

    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    drop(hfq);

    let mut prefix = tokenizer.encode("The capital of France is Paris. ");
    let filler = tokenizer.encode("It is known for museums, food, and art. ");
    while prefix.len() < PAGE * 2 {
        prefix.extend_from_slice(&filler);
    }
    prefix.truncate(PAGE * 2);

    let italy: Vec<u32> = {
        let mut v = prefix.clone();
        v.extend(tokenizer.encode(" The capital of Italy is"));
        v
    };
    let germany: Vec<u32> = {
        let mut v = prefix.clone();
        v.extend(tokenizer.encode(" The capital of Germany is"));
        v
    };
    assert!(italy.len() > PAGE * 2);
    assert_ne!(italy, germany);

    let engine = SlotEngine::spawn(EngineConfig {
        model_path: PathBuf::from(&model_path),
        n_slots: 2,
        cap_tokens: 1024,
        prefill_chunk: 256,
        host_budget_bytes: 4 * 1024 * 1024 * 1024,
        swap_dir: std::env::temp_dir().join("hipfire-prefix-cache-swap"),
        is_vl: false,
        vl_path: None,
        mtp_k: 0,
        kv_mode_raw: String::new(),
        prefix_cache: true,
        prefix_cache_max_bytes: 256 * 1024 * 1024,
        max_batch_tokens: 4096,
        prefill_min_tokens: 1,
        wait_max_count: 64,
        wait_max_bytes: 256 * 1024 * 1024,
        wait_timeout_ticks: 30_000,
        structured_jump_forward: false,
    })
    .expect("SlotEngine::spawn");
    println!("engine up: prefix_cache on, {}-token shared prefix", prefix.len());

    fn run(engine: &SlotEngine, prompt: Vec<u32>, max_tokens: usize) -> (usize, Vec<u32>) {
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                prompt_tokens: prompt,
                convo: Vec::new(),
                continuation: Continuation::Cold,
                max_tokens,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                visual_data: None,
                json_schema: None,
                queue_bytes: 0,
                reply: tx,
            })
            .expect("submit");
        let mut reused = 0usize;
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.recv() {
            match ev {
                Event::Accepted { reused: r, .. } => reused = r,
                Event::Token { id } => tokens.push(id),
                Event::Rejected { reason } => panic!("rejected: {reason}"),
                Event::Done { .. } => break,
            }
        }
        (reused, tokens)
    }

    let (reused_cold, toks_italy_1) = run(&engine, italy.clone(), MAX_TOKENS);
    println!(
        "  cold Italy: reused={reused_cold} generated={}",
        toks_italy_1.len()
    );
    assert_eq!(reused_cold, 0, "first request must be a cold miss");
    assert!(!toks_italy_1.is_empty(), "cold request produced no tokens");

    let (reused_warm, toks_italy_2) = run(&engine, italy.clone(), MAX_TOKENS);
    println!(
        "  warm Italy: reused={reused_warm} generated={}",
        toks_italy_2.len()
    );
    assert!(
        reused_warm >= PAGE,
        "warm identical prompt must reuse at least one full page, got {reused_warm}"
    );
    assert_eq!(
        toks_italy_1, toks_italy_2,
        "warm identical prompt must match the cold greedy continuation"
    );

    let (reused_branch, toks_germany) = run(&engine, germany, MAX_TOKENS);
    println!(
        "  branch Germany: reused={reused_branch} generated={}",
        toks_germany.len()
    );
    assert!(
        reused_branch >= PAGE,
        "divergent suffix must still reuse the shared prefix pages, got {reused_branch}"
    );
    assert_ne!(
        toks_germany, toks_italy_1,
        "divergent suffix must not replay the other branch"
    );

    let stats = engine.stats();
    println!(
        "  stats: admitted={} reused_tokens={} prefix_hits={}",
        stats.admitted, stats.reused_tokens, stats.prefix_hits
    );
    assert!(
        stats.reused_tokens >= PAGE * 2,
        "engine must account reused tokens from both warm requests"
    );
    println!("PASS");
}
