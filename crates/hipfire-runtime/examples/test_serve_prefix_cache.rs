// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P5 GPU compose oracle: cross-session radix prefix reuse on the slot engine.
//!
//! Cells (X2):
//!   * greedy AR/MTP: warm identical prompt reuses ≥128 tokens and matches
//!     the cold continuation; a divergent suffix still reuses the prefix
//!   * sampled AR: same seed is deterministic and still reuses
//!   * grammar AR: JSON Schema mask + prefix reuse; output parses as JSON
//!   * A20 reset: `SlotEngine::reset` drops the radix so the next request is cold
//!
//! Build: `cargo run -p hipfire-runtime --example test_serve_prefix_cache --features lab -- <model.hfq>`
//! Optional: `--mtp-k N` (default 4 when a sidecar exists, else 0).

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features lab (deltanet is default)");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::mtp_head::find_mtp_sidecar;
    use hipfire_arch_qwen35::serve_engine::{EngineConfig, SlotEngine};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::serve::{Continuation, Event, SubmitRequest};
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::channel;

    const PAGE: usize = 128;

    let mut model_path = None;
    let mut mtp_k_arg: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--mtp-k" {
            mtp_k_arg = Some(
                args.next()
                    .expect("--mtp-k needs a value")
                    .parse()
                    .expect("mtp_k"),
            );
        } else if !a.starts_with('-') {
            model_path = Some(a);
        }
    }
    let model_path = model_path.unwrap_or_else(|| {
        let home = std::env::var("HOME").expect("HOME not set");
        format!("{home}/.hipfire/models/qwen3.5-4b.mq4v2.hfq")
    });
    let sidecar = find_mtp_sidecar(Path::new(&model_path));
    let mtp_k = mtp_k_arg.unwrap_or(if sidecar.is_some() { 4 } else { 0 });
    println!("=== test_serve_prefix_cache ===\nmodel: {model_path}");
    println!(
        "mtp_k={mtp_k} sidecar={}",
        sidecar
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    if mtp_k > 0 && sidecar.is_none() {
        panic!("mtp_k={mtp_k} requested but no .mtp sidecar next to the trunk");
    }

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
    let json_prompt: Vec<u32> = {
        let mut v = prefix.clone();
        v.extend(tokenizer.encode(
            " Reply with a JSON object {\"city\": \"<name>\"} naming Italy's capital.",
        ));
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
        mtp_k,
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
    println!(
        "engine up: prefix_cache on, {}-token shared prefix",
        prefix.len()
    );

    #[derive(Clone)]
    struct RunSpec {
        temperature: f32,
        seed: u32,
        json_schema: Option<serde_json::Value>,
        max_tokens: usize,
    }
    impl RunSpec {
        fn greedy(max_tokens: usize) -> Self {
            Self {
                temperature: 0.0,
                seed: 0,
                json_schema: None,
                max_tokens,
            }
        }
    }

    fn run(
        engine: &SlotEngine,
        prompt: Vec<u32>,
        spec: &RunSpec,
    ) -> (usize, Vec<u32>) {
        let (tx, rx) = channel::<Event>();
        engine
            .submit(SubmitRequest {
                prompt_tokens: prompt,
                convo: Vec::new(),
                continuation: Continuation::Cold,
                max_tokens: spec.max_tokens,
                temperature: spec.temperature,
                top_p: 1.0,
                top_k: 0,
                seed: spec.seed,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                visual_data: None,
                json_schema: spec.json_schema.clone(),
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

    let greedy = RunSpec::greedy(8);
    let (reused_cold, toks_italy_1) = run(&engine, italy.clone(), &greedy);
    println!(
        "  greedy cold Italy: reused={reused_cold} generated={}",
        toks_italy_1.len()
    );
    assert_eq!(reused_cold, 0, "first request must be a cold miss");
    assert!(!toks_italy_1.is_empty(), "cold request produced no tokens");

    let (reused_warm, toks_italy_2) = run(&engine, italy.clone(), &greedy);
    println!(
        "  greedy warm Italy: reused={reused_warm} generated={}",
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

    let (reused_branch, toks_germany) = run(&engine, germany, &greedy);
    println!(
        "  greedy branch Germany: reused={reused_branch} generated={}",
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

    // Sampled AR (X2): MTP is off for temperature>0; prefix reuse still applies.
    let sampled = RunSpec {
        temperature: 0.8,
        seed: 7,
        json_schema: None,
        max_tokens: 8,
    };
    let (reused_s1, toks_s1) = run(&engine, italy.clone(), &sampled);
    let (reused_s2, toks_s2) = run(&engine, italy.clone(), &sampled);
    println!(
        "  sampled Italy: reused={reused_s1}/{reused_s2} generated={}",
        toks_s1.len()
    );
    assert!(
        reused_s1 >= PAGE && reused_s2 >= PAGE,
        "sampled AR must reuse the already-published prefix, got {reused_s1}/{reused_s2}"
    );
    assert_eq!(
        toks_s1, toks_s2,
        "same seed + same prompt must be deterministic under request-private RNG"
    );

    // Grammar AR (X2): constrained decode must reuse prefix and stay valid JSON.
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
        "additionalProperties": false
    });
    let grammar = RunSpec {
        temperature: 0.0,
        seed: 0,
        json_schema: Some(schema),
        max_tokens: 48,
    };
    let (reused_g, toks_g) = run(&engine, json_prompt, &grammar);
    let json_text = String::from_utf8_lossy(&tokenizer.decode_bytes(&toks_g)).into_owned();
    println!(
        "  grammar JSON: reused={reused_g} generated={} text={json_text:?}",
        toks_g.len()
    );
    assert!(
        reused_g >= PAGE,
        "grammar AR must reuse the published prefix, got {reused_g}"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(json_text.trim()).unwrap_or_else(|e| {
            panic!("grammar output is not JSON ({e}): {json_text:?}")
        });
    assert!(
        parsed.get("city").and_then(|v| v.as_str()).is_some(),
        "grammar JSON missing city string: {parsed}"
    );

    // A20: repeated warm hits stay bounded, then reset forces a cold miss.
    for i in 0..4 {
        let (r, toks) = run(&engine, italy.clone(), &greedy);
        println!("  soak[{i}]: reused={r} generated={}", toks.len());
        assert!(r >= PAGE, "soak request {i} lost prefix reuse ({r})");
        assert_eq!(toks, toks_italy_1, "soak request {i} drifted from greedy");
    }
    engine.reset().expect("reset after idle");
    let (reused_after_reset, _) = run(&engine, italy.clone(), &greedy);
    println!("  after reset: reused={reused_after_reset}");
    assert_eq!(
        reused_after_reset, 0,
        "reset must drop the radix (allocation_epoch bump); got reused={reused_after_reset}"
    );

    let stats = engine.stats();
    println!(
        "  stats: admitted={} reused_tokens={} prefix_hits={}",
        stats.admitted, stats.reused_tokens, stats.prefix_hits
    );
    assert!(
        stats.reused_tokens >= PAGE * 2,
        "engine must account reused tokens from warm requests"
    );
    println!("PASS");
}
