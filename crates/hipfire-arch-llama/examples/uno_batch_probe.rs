// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! GPU parity probe: actual conditional-LoRA proposal windows against base AR.
use hipfire_arch_llama::uno::{uno_forward_row, uno_forward_batch, UnoAdapter, UnoScratch, UnoBatchScratch};
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
    let mut kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 128).expect("spec KV");
    let mut ar_kv = KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, 128).expect("AR KV");
    let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, 128).expect("scratch");
    let ar_scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, 128).expect("AR scratch");
    let uno = UnoAdapter::open(std::path::Path::new(&args[2]), &config, &mut gpu, 4, 1, 250623).expect("adapter");
    let mut lora = UnoScratch::new(&mut gpu, &config).expect("LoRA scratch");
    let batch = UnoBatchScratch::new(&mut gpu, &config, uno.rank, 4, 128).expect("batch scratch");
    let batched = std::env::var_os("UNO_BATCH_PROBE").is_some();
    eprintln!("batch fixture: cap={} compact={} q8={} max_batch={}", kv.physical_cap, kv.compact_offset, kv.quant_q8, batch.pbs.max_batch);
    for (i, layer) in weights.layers.iter().enumerate() {
        for (name, w) in [("q", &layer.wq), ("k", &layer.wk), ("v", &layer.wv), ("o", &layer.wo), ("gate", &layer.w_gate), ("up", &layer.w_up), ("down", &layer.w_down)] {
            if !hipfire_runtime::llama::is_batchable_la(w.gpu_dtype, &gpu.arch) {
                eprintln!("ineligible layer={i} projection={name} dtype={:?}", w.gpu_dtype);
            }
        }
    }
    let mut ar = |gpu: &mut Gpu, token, pos| {
        embedding_lookup_dispatch(gpu, weights.embd_format, &weights.token_embd, &ar_scratch.x, token, config.dim).unwrap();
        gpu.hip.memcpy_htod(&ar_scratch.pos_buf, &(pos as i32).to_ne_bytes()).unwrap();
        forward_scratch_compute(gpu, &weights, &config, pos, &mut ar_kv, &ar_scratch).unwrap();
        gpu.download_f32(&ar_scratch.logits).unwrap()
    };
    let prompt_owned;
    let prompt: &str = match std::env::var("HIPFIRE_UNO_PROBE_PROMPT") {
        Ok(path) => {
            prompt_owned = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            &prompt_owned
        }
        Err(_) => "<|ifm|im_start|>user\nSolve 2+2.<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think_faster>\n",
    };
    let mut prompt_tokens = vec![config.bos_token];
    prompt_tokens.extend(tokenizer.encode(prompt));
    let mut seed = 0;
    let mut worst_gap = f32::INFINITY;
    let mut worst_pos = 0usize;
    let mut worst_abs = 0.0f32;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let row_l = uno_forward_row(&mut gpu, &weights, &config, &uno, token, pos, false, &mut kv, &scratch, &mut lora).unwrap();
        // Same token through the AR path; the raw logits expose whether a
        // mismatch is a near-tie (numerics) or a wide-gap flip (a real bug).
        let row_r = ar(&mut gpu, token, pos);
        let a = pick(&row_l);
        let b = pick(&row_r);
        // Top-2 gap of each side, and the largest absolute logit difference.
        let gap = |v: &[f32]| {
            let mut top = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &x in v.iter() {
                if x > top.0 { top.1 = top.0; top.0 = x; } else if x > top.1 { top.1 = x; }
            }
            top.0 - top.1
        };
        let ma = row_l.iter().zip(row_r.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        if a != b && gap(&row_r).min(gap(&row_l)) < worst_gap {
            worst_gap = gap(&row_r).min(gap(&row_l));
            worst_pos = pos;
            worst_abs = ma;
        }
        seed = a;
        assert_eq!(a, b, "prefill mismatch at {pos}: uno={a} ar={b} gap_uno={:.4} gap_ar={:.4} max_abs={ma}", gap(&row_l), gap(&row_r));
    }
    eprintln!("[prefill] worst mismatch: pos={worst_pos} min_gap={worst_gap:.4} max_abs={worst_abs}");
    let mut position = prompt_tokens.len();
    let mut uno_tokens = vec![seed];
    let mut base_tokens = vec![seed];
    let mut total = 0;
    let mut accepted = 0;
    'windows: for window in 0..6 {
        // Seed is a target draw. Draft noise rows predict future proposal slots.
        let mut input = vec![seed];
        for row in 1..uno.block_len {
            input.push(1 + ((window * 7919 + row * 104729) % 250622) as u32);
        }
        let mut reference = Vec::new();
        for (row, &token) in input.iter().enumerate() {
            reference.extend(uno_forward_row(&mut gpu, &weights, &config, &uno, token, position + row, row > 0, &mut kv, &scratch, &mut lora).unwrap());
        }
        let logits = if batched {
            let candidate = uno_forward_batch(&mut gpu, &weights, &config, Some(&uno), &input, position, &mut kv, &scratch, &batch).unwrap();
            for row in 0..input.len() {
                let a = &reference[row * config.vocab_size..(row + 1) * config.vocab_size];
                let b = &candidate[row * config.vocab_size..(row + 1) * config.vocab_size];
                let max_error = a.iter().zip(b).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
                let rms = (a.iter().zip(b).map(|(a,b)| f64::from(a-b).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
                println!("draft window={window} row={row} max_abs={max_error} rms={rms} scalar={} batch={}", pick(a), pick(b));
                assert!(rms < 0.1, "draft logits diverged");
                assert_eq!(pick(a), pick(b), "draft argmax divergence");
            }
            candidate
        } else { reference };
        let clean = pick(&logits[..config.vocab_size]);
        let drafts: Vec<u32> = logits.chunks_exact(config.vocab_size).skip(1).map(pick).collect();
        let verify: Vec<u32> = std::iter::once(clean).chain(drafts.iter().copied()).collect();
        let target_picks: Vec<u32> = if batched {
            let candidate = uno_forward_batch(&mut gpu, &weights, &config, None, &verify, position + 1, &mut kv, &scratch, &batch).unwrap();
            // Verify rows are where spec-decode divergences actually surface, and
            // unlike the draft rows above they were never compared batched-vs-
            // scalar. Measure them: this is the quantity that decides whether the
            // window's committed token matches the per-token AR decode.
            for (row, &token) in verify.iter().enumerate() {
                let a = uno_forward_row(&mut gpu, &weights, &config, &uno, token, position + 1 + row, false, &mut kv, &scratch, &mut lora).unwrap();
                let b = &candidate[row * config.vocab_size..(row + 1) * config.vocab_size];
                let max_error = a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let rms = (a.iter().zip(b).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
                // Top-2 gap of the batched row, to compare against the delta.
                let mut top = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for &v in b.iter() {
                    if v > top.0 { top.1 = top.0; top.0 = v; } else if v > top.1 { top.1 = v; }
                }
                println!("verify window={window} row={row} max_abs={max_error} rms={rms} gap={:.4} scalar={} batch={}", top.0 - top.1, pick(&a), pick(b));
            }
            candidate.chunks_exact(config.vocab_size).map(pick).collect()
        } else {
            verify.iter().enumerate().map(|(row, &token)| pick(&uno_forward_row(&mut gpu, &weights, &config, &uno, token, position + 1 + row, false, &mut kv, &scratch, &mut lora).unwrap())).collect()
        };
        let verdict = accept_greedy_prefix(&drafts, &target_picks, None);
        let emit: Vec<u32> = std::iter::once(clean).chain(verdict.committed).collect();
        let next_seed = *emit.last().unwrap();
        let step = SpecStep::new(emit, next_seed, drafts.len(), verdict.accepted);
        let mut previous = seed;
        for (row, &token) in step.emit.iter().enumerate() {
            let expected = pick(&ar(&mut gpu, previous, position + row));
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
    batch.free_gpu(&mut gpu);
    uno.free_gpu(&mut gpu);
}
