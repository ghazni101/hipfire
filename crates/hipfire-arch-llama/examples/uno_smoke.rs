// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! GPU parity probe: actual conditional-LoRA proposal windows against base AR.
use hipfire_arch_llama::uno::{uno_forward_row, UnoAdapter, UnoScratch};
use hipfire_runtime::hfq::{config_from_hfq, load_weights_hfq, HfqFile};
use hipfire_runtime::llama::{embedding_lookup_dispatch, forward_scratch_compute, ForwardScratch, KvCache};
use hipfire_runtime::spec::{accept_greedy_prefix, SpecStep};
use rdna_compute::Gpu;

fn pick(logits: &[f32]) -> u32 {
    assert!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0 as u32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: uno_smoke MODEL.mq4 ADAPTER_DIR");
    let mut gpu = Gpu::init().expect("gpu");
    let hfq = HfqFile::open(std::path::Path::new(&args[1])).expect("model");
    let config = config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    assert_eq!(config.norm_groups, 4);
    let weights = load_weights_hfq(&hfq, &config, &mut gpu).expect("weights");
    let mut kv = KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 128).expect("spec KV");
    let mut ar_kv = KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 128).expect("AR KV");
    let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, 128).expect("scratch");
    let ar_scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, 128).expect("AR scratch");
    let uno = UnoAdapter::open(std::path::Path::new(&args[2]), &config, &mut gpu, 4, 1, 250623).expect("adapter");
    let mut lora = UnoScratch::new(&mut gpu, &config).expect("LoRA scratch");
    let mut ar = |gpu: &mut Gpu, token, pos| {
        embedding_lookup_dispatch(gpu, weights.embd_format, &weights.token_embd, &ar_scratch.x, token, config.dim).unwrap();
        gpu.hip.memcpy_htod(&ar_scratch.pos_buf, &(pos as i32).to_ne_bytes()).unwrap();
        forward_scratch_compute(gpu, &weights, &config, pos, &mut ar_kv, &ar_scratch).unwrap();
        pick(&gpu.download_f32(&ar_scratch.logits).unwrap())
    };
    let prompt = "<|ifm|im_start|>user\nSolve 2+2.<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think_faster>\n";
    let mut prompt_tokens = vec![config.bos_token];
    prompt_tokens.extend(tokenizer.encode(prompt));
    let mut seed = 0;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        seed = pick(&uno_forward_row(&mut gpu, &weights, &config, &uno, token, pos, false, &mut kv, &scratch, &mut lora).unwrap());
        assert_eq!(seed, ar(&mut gpu, token, pos), "prefill mismatch at {pos}");
    }
    let mut position = prompt_tokens.len();
    let mut uno_tokens = vec![seed];
    let mut base_tokens = vec![seed];
    let mut total = 0;
    let mut accepted = 0;
    'windows: for window in 0..6 {
        // Seed is a target draw. Draft noise rows predict future proposal slots.
        let clean = pick(&uno_forward_row(&mut gpu, &weights, &config, &uno, seed, position, false, &mut kv, &scratch, &mut lora).unwrap());
        let mut drafts = Vec::new();
        for row in 1..uno.block_len {
            let noise = 1 + ((window * 7919 + row * 104729) % 250622) as u32;
            drafts.push(pick(&uno_forward_row(&mut gpu, &weights, &config, &uno, noise, position + row, true, &mut kv, &scratch, &mut lora).unwrap()));
        }
        // Verify starts AFTER the cached seed; overwrites every noise KV slot.
        let mut target_picks = Vec::new();
        for (row, token) in std::iter::once(clean).chain(drafts.iter().copied()).enumerate() {
            target_picks.push(pick(&uno_forward_row(&mut gpu, &weights, &config, &uno, token, position + 1 + row, false, &mut kv, &scratch, &mut lora).unwrap()));
        }
        let verdict = accept_greedy_prefix(&drafts, &target_picks, None);
        let emit: Vec<u32> = std::iter::once(clean).chain(verdict.committed).collect();
        let next_seed = *emit.last().unwrap();
        let step = SpecStep::new(emit, next_seed, drafts.len(), verdict.accepted);
        let mut previous = seed;
        for (row, &token) in step.emit.iter().enumerate() {
            let expected = ar(&mut gpu, previous, position + row);
            uno_tokens.push(token);
            base_tokens.push(expected);
            println!("token pos={} base={expected} uno={token} base_text={:?} uno_text={:?}", position + row + 1, tokenizer.decode(&[expected]), tokenizer.decode(&[token]));
            assert_eq!(token, expected, "window {window}, emitted row {row}, position {}", position + row + 1);
            previous = token;
            if matches!(token, 1 | 250019) {
                println!("terminal token={token}, stopping inside window {window}");
                total += row + 1;
                break 'windows;
            }
        }
        println!("window={window} pos={position} draft={drafts:?} emitted={:?} accepted={}/{} AR parity=PASS", step.emit, step.accepted, step.proposed);
        total += step.emit.len();
        accepted += step.accepted;
        position += step.emit.len();
        seed = step.next_seed;
    }
    println!("Uno speculative parity PASS: {total} emitted token IDs, {accepted} accepted drafts");
    println!("BASE TEXT: {}", tokenizer.decode(&base_tokens));
    println!("UNO TEXT: {}", tokenizer.decode(&uno_tokens));
    assert_eq!(uno_tokens, base_tokens);
    lora.free_gpu(&mut gpu);
    uno.free_gpu(&mut gpu);
}
