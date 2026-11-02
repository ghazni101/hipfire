// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! iu4_kld — decode-path logit oracle for quant-kernel changes.
//!
//! `iu4_kld ref <model> <out.json> gen <max_gen> PROMPT...` greedy-generates
//! with the per-token decode path (`forward_scratch`) and records, per
//! scored step, the top-64 log-probs, the argmax and an FNV-1a hash of the
//! full f32 logit row.
//! `iu4_kld ref <model> <out.json> text <n_tokens> <file>` teacher-forces the
//! first `n_tokens` raw tokens of `file` instead (PPL text).
//! `iu4_kld score <model> <ref.json> <out.json>` teacher-forces the reference
//! token sequence on another model/build and reports: logit rows bit-identical
//! to the reference, top-1 agreement, first divergence, KL(ref||cand) over the
//! reference top-64 plus the lumped remainder bucket, and PPL of the sequence.
#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use std::path::Path;
    const TOPK: usize = 64;

    fn fnv(v: &[f32]) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for x in v {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        format!("{h:016x}")
    }
    fn log_softmax(v: &[f32]) -> Vec<f64> {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
        let l = m + s.ln();
        v.iter().map(|&x| x as f64 - l).collect()
    }
    fn topk(lp: &[f64], k: usize) -> Vec<(u32, f64)> {
        let mut idx: Vec<u32> = (0..lp.len() as u32).collect();
        idx.select_nth_unstable_by(k, |&a, &b| {
            lp[b as usize].partial_cmp(&lp[a as usize]).unwrap()
        });
        let mut t: Vec<(u32, f64)> = idx[..k].iter().map(|&i| (i, lp[i as usize])).collect();
        t.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        t
    }

    let a: Vec<String> = std::env::args().collect();
    let usage = "iu4_kld ref <model> <out.json> gen <max_gen> PROMPT... | ref <model> <out.json> text <n> <file> | score <model> <ref.json> <out.json>";
    if a.len() < 5 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let mode = a[1].as_str();
    let model_path = &a[2];
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tok =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let weights = {
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
    }
    .expect("load");
    let max_seq = 8192usize;
    let mut kv = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        max_seq,
    )
    .unwrap();
    let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();
    let t0 = std::time::Instant::now();

    // (tokens, first scored index): logits after token[i] predict token[i+1]
    // for i in start-1 .. len-1.
    let (tokens, start, gen): (Vec<u32>, usize, Option<usize>) = match mode {
        "ref" => match a[4].as_str() {
            "gen" => {
                let n: usize = a[5].parse().unwrap();
                let p = a[6..].join(" ");
                let enc = |s: &str| tok.encode(s);
                let mut t = Vec::new();
                for s in ["<|im_start|>", "user", "\n"] {
                    t.extend(enc(s));
                }
                t.extend(enc(&p));
                for s in [
                    "<|im_end|>",
                    "\n",
                    "<|im_start|>",
                    "assistant",
                    "\n",
                    "<think>\n</think>\n\n",
                ] {
                    t.extend(enc(s));
                }
                let s = t.len();
                (t, s, Some(n))
            }
            "text" => {
                let n: usize = a[5].parse().unwrap();
                let txt = std::fs::read_to_string(&a[6]).expect("text");
                let mut t = tok.encode(&txt);
                t.truncate(n);
                (t, 1, None)
            }
            _ => panic!("{usage}"),
        },
        "score" | "pref" | "pscore" => {
            let r: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&a[3]).unwrap()).unwrap();
            let t: Vec<u32> = r["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            (t, r["start"].as_u64().unwrap() as usize, None)
        }
        _ => panic!("{usage}"),
    };
    // Prefill-path oracle: `pref <model> <tokens-ref.json> <out.json>` and
    // `pscore <model> <ref.json> <out.json>` run ONE batched prompt prefill
    // (`forward_prefill_batch_capture_hidden`, production prompt route) over
    // the reference tokens and take every row's logits from the lm_head GEMV
    // on the captured hidden rows, so the prefill GEMMs are what is measured.
    if mode == "pref" || mode == "pscore" {
        let n = tokens.len();
        let hidden = gpu
            .alloc_tensor(&[n, config.dim], rdna_compute::DType::F32)
            .unwrap();
        qwen35::forward_prefill_batch_capture_hidden(
            &mut gpu, &weights, &config, &tokens, 0, &mut kv, &mut dn, &scratch, &hidden,
        )
        .expect("prefill");
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&a[3]).unwrap()).unwrap();
        let (mut same_hash, mut top1, mut nll_sum, mut kl_sum, mut kl_max, mut n_sc) =
            (0usize, 0usize, 0f64, 0f64, 0f64, 0usize);
        let mut first_div: Option<usize> = None;
        let mut rows: Vec<serde_json::Value> = Vec::new();
        let nsteps_ref = r["steps"].as_array().unwrap().len();
        for i in (start - 1)..(n - 1) {
            let step = i + 1 - start;
            if step >= nsteps_ref {
                break;
            }
            let row = hidden.sub_offset(i * config.dim, config.dim);
            let ctx = llama::DispatchCtx::new(&gpu);
            let _ = llama::gemv_family()
                .run_auto(
                    &ctx,
                    &mut gpu,
                    &weights.output.dispatch_ref(),
                    &row,
                    &scratch.logits,
                )
                .expect("lm_head");
            let logits = gpu.download_f32(&scratch.logits).unwrap();
            let logits = &logits[..config.vocab_size];
            let am = llama::argmax(logits);
            let lp = log_softmax(logits);
            let h = fnv(logits);
            if mode == "pscore" {
                let rr = &r["steps"][step];
                if rr["hash"].as_str() == Some(h.as_str()) {
                    same_hash += 1;
                }
                if rr["argmax"].as_u64() == Some(am as u64) {
                    top1 += 1;
                } else if first_div.is_none() {
                    first_div = Some(step);
                }
                let mut kl = 0f64;
                let (mut pin, mut qin) = (0f64, 0f64);
                for e in rr["top"].as_array().unwrap() {
                    let id = e[0].as_u64().unwrap() as usize;
                    let lpr = e[1].as_f64().unwrap();
                    let p = lpr.exp();
                    kl += p * (lpr - lp[id]);
                    pin += p;
                    qin += lp[id].exp();
                }
                let (pr, qr) = ((1.0 - pin).max(1e-300), (1.0 - qin).max(1e-300));
                kl += pr * (pr.ln() - qr.ln());
                kl_sum += kl;
                kl_max = kl_max.max(kl);
            } else {
                let t = topk(&lp, TOPK);
                rows.push(serde_json::json!({"argmax": am, "hash": h, "top": t.iter().map(|(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>()}));
            }
            n_sc += 1;
            nll_sum += -lp[tokens[i + 1] as usize];
        }
        let ppl = (nll_sum / n_sc.max(1) as f64).exp();
        let secs = t0.elapsed().as_secs_f64();
        let out = if mode == "pscore" {
            serde_json::json!({"model": model_path, "ref": a[3], "mode": "prefill", "steps": n_sc,
                "logit_rows_bit_identical": same_hash, "top1_agree": top1, "first_divergence": first_div,
                "mean_kl_top64": kl_sum / n_sc.max(1) as f64, "max_kl_top64": kl_max, "ppl": ppl, "secs": secs})
        } else {
            serde_json::json!({"model": model_path, "mode": "prefill", "start": start, "tokens": tokens, "steps": rows, "ppl": ppl, "secs": secs})
        };
        std::fs::write(&a[4], out.to_string()).unwrap();
        if mode == "pscore" {
            println!("{out}");
        } else {
            println!("{{\"steps\":{n_sc},\"ppl\":{ppl},\"secs\":{secs}}}");
        }
        return;
    }
    let mut tokens = tokens;
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let refj: Option<serde_json::Value> = if mode == "score" {
        Some(serde_json::from_str(&std::fs::read_to_string(&a[3]).unwrap()).unwrap())
    } else {
        None
    };
    let (mut same_hash, mut top1, mut nll_sum, mut kl_sum, mut kl_max, mut n_sc) =
        (0usize, 0usize, 0f64, 0f64, 0f64, 0usize);
    let mut first_div: Option<usize> = None;
    let mut pos = 0usize;
    loop {
        if pos >= tokens.len() || pos >= max_seq - 1 {
            break;
        }
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            tokens[pos],
            pos,
            &mut kv,
            &mut dn,
            &scratch,
        )
        .expect("fwd");
        pos += 1;
        if pos < start {
            continue;
        }
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let am = llama::argmax(&logits);
        if let Some(n) = gen {
            if pos == tokens.len() {
                if pos - start >= n {
                    break;
                }
                tokens.push(am);
            }
        }
        if pos >= tokens.len() {
            break;
        }
        let next = tokens[pos];
        let lp = log_softmax(&logits);
        let h = fnv(&logits);
        let step = rows.len().max(n_sc);
        if let Some(r) = &refj {
            let rr = &r["steps"][step];
            if rr.is_null() {
                break;
            }
            if rr["hash"].as_str() == Some(h.as_str()) {
                same_hash += 1;
            }
            if rr["argmax"].as_u64() == Some(am as u64) {
                top1 += 1;
            } else if first_div.is_none() {
                first_div = Some(step);
            }
            let mut kl = 0f64;
            let (mut pin, mut qin) = (0f64, 0f64);
            for e in rr["top"].as_array().unwrap() {
                let id = e[0].as_u64().unwrap() as usize;
                let lpr = e[1].as_f64().unwrap();
                let p = lpr.exp();
                kl += p * (lpr - lp[id]);
                pin += p;
                qin += lp[id].exp();
            }
            let (pr, qr) = ((1.0 - pin).max(1e-300), (1.0 - qin).max(1e-300));
            kl += pr * (pr.ln() - qr.ln());
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            n_sc += 1;
        } else {
            let t = topk(&lp, TOPK);
            rows.push(serde_json::json!({"argmax": am, "hash": h, "top": t.iter().map(|(i, v)| serde_json::json!([i, v])).collect::<Vec<_>>()}));
        }
        nll_sum += -lp[next as usize];
        if gen.is_some() && (am == config.eos_token) && refj.is_none() {
            break;
        }
    }
    let nsteps = if refj.is_some() { n_sc } else { rows.len() };
    let ppl = (nll_sum / nsteps.max(1) as f64).exp();
    let secs = t0.elapsed().as_secs_f64();
    if let Some(_) = refj {
        let out = serde_json::json!({"model": model_path, "ref": a[3], "steps": nsteps,
            "logit_rows_bit_identical": same_hash, "top1_agree": top1, "first_divergence": first_div,
            "mean_kl_top64": kl_sum / nsteps.max(1) as f64, "max_kl_top64": kl_max, "ppl": ppl, "secs": secs});
        println!("{out}");
        std::fs::write(&a[4], out.to_string()).unwrap();
    } else {
        tokens.truncate(start + rows.len());
        let text = tok.decode(&tokens[start..]);
        let out = serde_json::json!({"model": model_path, "start": start, "tokens": tokens, "steps": rows, "text": text, "ppl": ppl, "secs": secs});
        std::fs::write(&a[3], out.to_string()).unwrap();
        println!(
            "{{\"steps\":{},\"ppl\":{ppl},\"secs\":{secs},\"text_sha\":\"{}\"}}",
            rows.len(),
            fnv(&text.bytes().map(|b| b as f32).collect::<Vec<_>>())
        );
        eprintln!("TEXT<<<{text}>>>");
    }
}
