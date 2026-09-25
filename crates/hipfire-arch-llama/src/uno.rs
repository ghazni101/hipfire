// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Uno two-pass diffusion decoding for the LLaMA-family dense target
//! (K2-Horizon + K2-Horizon-Uno conditional LoRA).
//!
//! Algorithm per window of block_len L (adapted from ifm-ai/uno
//! `nano_vllm_uno/engine/two_pass_decoding.py`; attribution required by
//! the upstream Apache-2.0 NOTICE):
//!
//! 1. DRAFT: forward `[seed, noise_1..noise_{L-1}]`. The gated LoRA delta
//!    applies ONLY to noise rows (row 0 mask 0). Proposals: sample/argmax of
//!    the clean row logit (the already-committed seed) and each noise row.
//!    The draft-noise KV is discarded (the verify forward overwrites those
//!    slots).
//! 2. VERIFY: forward `[clean, proposal_1..proposal_{L-1}]` base-only.
//!    Greedy: accept the prefix whose target argmax equals the proposal, then
//!    a lookahead argmax from the last verify row. Stochastic (temp > 0):
//!    dual-softmax rejection acceptance with residual correction, fused on
//!    device. Unfiltered requests use the full-vocab fused verifier
//!    (`uno_verify_logits.hip`, a HIP port of the upstream
//!    `fused_verify_kernel.py`); filtered requests are device-resident too —
//!    top-k rows are masked to their top-K support, and top-p (optionally
//!    with top_k) goes through the candidate-gather verifier
//!    (`uno_verify_logits.hip`), which replicates the fused AR sampler's
//!    law (top-K gather → softmax → nucleus truncation → renormalize) so the
//!    emitted distribution matches the daemon's non-speculative AR decode at
//!    the same request config. min-p-only requests keep the host
//!    `Distribution` law. Truncate at EOS.
//! 3. KV: committed tokens keep base computations only — the linear path
//!    overwrites the draft-noise slots during verify; the tree path walks the
//!    accepted path and compacts its KV (`compact_tree_kv`).
//!
//! Lossless: every committed token is a draw from the target's own
//! distribution at its prefix (greedy argmax or stochastic rejection).

use hip_bridge::HipResult;
use hipfire_runtime::llama::{
    embedding_lookup_dispatch, weight_gemv, weight_gemm, ForwardScratch, KvCache, LlamaConfig,
    LlamaWeights,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use half::f16;

/// One module's low-rank factors, already scaled: delta = B·(A·x).
/// `a` is [rank × k], `b` is [m × rank], both F32 on GPU.
struct UnoProj {
    a: GpuTensor,
    b: GpuTensor,
    #[allow(dead_code)]
    rank: usize,
    #[allow(dead_code)]
    k: usize,
    #[allow(dead_code)]
    m: usize,
}

/// Per-layer conditional LoRA over the 7 PEFT target modules.
struct UnoLayer {
    q_proj: UnoProj,
    k_proj: UnoProj,
    v_proj: UnoProj,
    o_proj: UnoProj,
    gate_proj: UnoProj,
    up_proj: UnoProj,
    down_proj: UnoProj,
}

pub struct UnoAdapter {
    pub rank: usize,
    pub scale: f32,
    pub block_len: usize,
    pub noise_low: u32,
    pub noise_high: u32,
    layers: Vec<UnoLayer>,
}


fn free_proj(gpu: &mut Gpu, p: UnoProj) {
    let _ = gpu.free_tensor(p.a);
    let _ = gpu.free_tensor(p.b);
}

fn free_layer(gpu: &mut Gpu, l: UnoLayer) {
    for p in [l.q_proj, l.k_proj, l.v_proj, l.o_proj, l.gate_proj, l.up_proj, l.down_proj] {
        free_proj(gpu, p);
    }
}

impl UnoAdapter {
    /// Read adapter_config.json + adapter_model.safetensors from `dir` and
    /// upload every factor to the GPU.
    pub fn open(
        dir: &std::path::Path,
        config: &LlamaConfig,
        gpu: &mut Gpu,
        block_len: usize,
        noise_low: u32,
        noise_high: u32,
    ) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("adapter_config.json"))
                .map_err(|e| format!("uno: adapter_config.json: {e}"))?,
        )
        .map_err(|e| format!("uno: adapter_config parse: {e}"))?;
        let peft = cfg
            .get("peft_type")
            .and_then(|v| v.as_str())
            .ok_or("uno: peft_type missing")?;
        if peft != "LORA" {
            return Err(format!("uno: refusing adapter peft_type {peft} (only LORA)"));
        }
        let rank = cfg
            .get("r")
            .and_then(|v| v.as_u64())
            .ok_or("uno: rank missing")? as usize;
        let alpha = cfg
            .get("lora_alpha")
            .and_then(|v| v.as_f64())
            .ok_or("uno: lora_alpha missing")? as f32;
        let targets: Vec<String> = cfg
            .get("target_modules")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .ok_or("uno: target_modules missing")?;
        let expected = [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ];
        for t in expected {
            if !targets.iter().any(|x| x == t) {
                return Err(format!("uno: target module {t} missing from adapter"));
            }
        }
        // PEFT scale is alpha/rank. The released K2 adapter_config has
        // lora_alpha=8192 (scale 64); the paper trains alpha=256 (scale 2).
        // HIPFIRE_UNO_LORA_SCALE replaces the folded scale so those two laws
        // can be compared without rewriting the checkpoint.
        let file_scale = alpha / rank as f32;
        let scale = hipfire_config::developer_var("HIPFIRE_UNO_LORA_SCALE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(file_scale);
        eprintln!(
            "[uno] lora scale {scale:.4} (file alpha/rank={file_scale:.4}, alpha={alpha}, rank={rank})"
        );

        let path = dir.join("adapter_model.safetensors");
        let file = std::fs::File::open(&path).map_err(|e| format!("uno: {}: {e}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("uno: safetensors parse: {e}"))?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            let q_out = config.n_heads * config.head_dim;
            let kv_out = config.n_kv_heads * config.head_dim;
            let dim = config.dim;
            let hidden = config.hidden_dim;
            let mk = |module: &str, m: usize, k: usize, gpu: &mut Gpu| -> Result<UnoProj, String> {
                let a_name = format!("model.layers.{layer}.{module}.lora_A.weight");
                let b_name = format!("model.layers.{layer}.{module}.lora_B.weight");
                let load = |name: &str| -> Result<(Vec<u8>, Vec<usize>), String> {
                    let view = st
                        .tensor(name)
                        .map_err(|_| format!("uno: tensor {name} missing"))?;
                    if view.dtype() != safetensors::Dtype::F32 {
                        return Err(format!("uno: {name} dtype {}", view.dtype()));
                    }
                    Ok((view.data().to_vec(), view.shape().to_vec()))
                };
                let (a_data, a_shape) = load(&a_name)?;
                let (b_data, b_shape) = load(&b_name)?;
                if a_shape != [rank, k] || b_shape != [m, rank] {
                    return Err(format!(
                        "uno: {module} layer {layer} shape mismatch A={a_shape:?} B={b_shape:?} (rank {rank}, k {k}, m {m})"
                    ));
                }
                // Fold the PEFT scale (alpha/rank) into A at upload time so
                // the GPU path is a plain B·(A·x) with no extra kernel.
                let mut a_f32: Vec<f32> = a_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                for v in a_f32.iter_mut() {
                    *v *= scale;
                }
                // P2b: store the LoRA A/B factors as F16. The ~1.36 GB/fwd F32
                // weight read is the dominant draft cost (profile: splitk 11.8ms
                // + xbatch 7.2ms per forward); halving it to F16 cuts that to
                // ~9.6ms. x/y accumulate stays F32, and the small rank-128 delta
                // (scaled by alpha/rank) is a small additive delta on the
                // base projection; greedy token-identity is the pass bar.
                let a_bits: Vec<u16> = a_f32.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
                let a = gpu
                    .upload_f16_bits(&a_bits, &[rank, k])
                    .map_err(|e| format!("uno: upload A {module}/{layer}: {e:?}"))?;
                let b_f32: Vec<f32> = b_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let b_bits: Vec<u16> = b_f32.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
                let b = match gpu.upload_f16_bits(&b_bits, &[m, rank]) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = gpu.free_tensor(a);
                        return Err(format!("uno: upload B {module}/{layer}: {e:?}"));
                    }
                };
                Ok(UnoProj { a, b, rank, k, m })
            };
            let specs = [
                ("self_attn.q_proj", q_out, dim),
                ("self_attn.k_proj", kv_out, dim),
                ("self_attn.v_proj", kv_out, dim),
                ("self_attn.o_proj", dim, q_out),
                ("mlp.gate_proj", hidden, dim),
                ("mlp.up_proj", hidden, dim),
                ("mlp.down_proj", dim, hidden),
            ];
            let mut projs = Vec::with_capacity(7);
            let mut err = None;
            for &(module, m, k) in &specs {
                match mk(module, m, k, gpu) {
                    Ok(p) => projs.push(p),
                    Err(e) => {
                        for p in projs.drain(..) {
                            free_proj(gpu, p);
                        }
                        err = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = err {
                for l in layers.drain(..) {
                    free_layer(gpu, l);
                }
                return Err(e);
            }
            let mut it = projs.into_iter();
            layers.push(UnoLayer {
                q_proj: it.next().expect("q_proj"),
                k_proj: it.next().expect("k_proj"),
                v_proj: it.next().expect("v_proj"),
                o_proj: it.next().expect("o_proj"),
                gate_proj: it.next().expect("gate_proj"),
                up_proj: it.next().expect("up_proj"),
                down_proj: it.next().expect("down_proj"),
            });
        }
        Ok(Self {
            rank,
            scale,
            block_len,
            noise_low,
            noise_high,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for l in self.layers {
            free_layer(gpu, l);
        }
    }

    /// y += scale·B·(A·x) for one projection (scale folded into A).
    fn apply_delta(
        gpu: &mut Gpu,
        p: &UnoProj,
        x: &GpuTensor,
        y: &GpuTensor,
        ax: &GpuTensor,
        delta: &GpuTensor,
    ) -> HipResult<()> {
        let ax = ax.sub_offset(0, p.rank);
        let delta = delta.sub_offset(0, p.m);
        gpu.gemv_f16_xbatch(&p.a, x, &ax, p.rank, p.k, 1, false)?;
        gpu.gemv_f16_xbatch(&p.b, &ax, &delta, p.m, p.rank, 1, false)?;
        gpu.add_inplace_f32(y, &delta)
    }
}

/// A single-token forward with optional gated LoRA delta. `lora` selects
/// whether the delta applies (seed row: false; noise rows: true).
/// Returns logits over vocab. Advances KV by one position at `pos`.
#[allow(clippy::too_many_arguments)]
pub fn uno_forward_row(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    uno: &UnoAdapter,
    token: u32,
    pos: usize,
    lora: bool,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    lora_scratch: &mut UnoScratch,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = n_kv_heads * head_dim;

    embedding_lookup_dispatch(
        gpu,
        weights.embd_format,
        &weights.token_embd,
        &scratch.x,
        token,
        dim,
    )?;

    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    for layer_idx in 0..config.n_layers {
        let layer = &weights.layers[layer_idx];
        let ul = &uno.layers[layer_idx];
        config.rmsnorm(gpu, &scratch.x, &layer.attn_norm, &scratch.tmp)?;

        // Q/K/V + gated deltas.
        weight_gemv(gpu, &layer.wq, &scratch.tmp, &scratch.q)?;
        weight_gemv(gpu, &layer.wk, &scratch.tmp, &scratch.k)?;
        weight_gemv(gpu, &layer.wv, &scratch.tmp, &scratch.v)?;
        if lora {
            UnoAdapter::apply_delta(
                gpu,
                &ul.q_proj,
                &scratch.tmp,
                &scratch.q,
                &lora_scratch.ax_q,
                &lora_scratch.delta_q,
            )?;
            UnoAdapter::apply_delta(
                gpu,
                &ul.k_proj,
                &scratch.tmp,
                &scratch.k,
                &lora_scratch.ax_kv,
                &lora_scratch.delta_kv,
            )?;
            UnoAdapter::apply_delta(
                gpu,
                &ul.v_proj,
                &scratch.tmp,
                &scratch.v,
                &lora_scratch.ax_kv,
                &lora_scratch.delta_kv,
            )?;
        }

        gpu.rope_f32(
            &scratch.q,
            &scratch.k,
            &scratch.pos_buf,
            n_heads,
            n_kv_heads,
            head_dim,
            config.rope_freq_base,
        )?;

    hipfire_runtime::llama::llama_kv_write_attend(
        gpu, kv_cache, scratch, layer_idx, pos, n_heads, n_kv_heads, head_dim, kv_dim,
    )?;

        weight_gemv(gpu, &layer.wo, &scratch.attn_out, &scratch.o)?;
        if lora {
            UnoAdapter::apply_delta(
                gpu,
                &ul.o_proj,
                &scratch.attn_out,
                &scratch.o,
                &lora_scratch.ax_q,
                &lora_scratch.delta_q,
            )?;
        }
        gpu.add_inplace_f32(&scratch.x, &scratch.o)?;

        config.rmsnorm(gpu, &scratch.x, &layer.ffn_norm, &scratch.tmp)?;
        weight_gemv(gpu, &layer.w_gate, &scratch.tmp, &scratch.gate)?;
        weight_gemv(gpu, &layer.w_up, &scratch.tmp, &scratch.up)?;
        if lora {
            UnoAdapter::apply_delta(
                gpu,
                &ul.gate_proj,
                &scratch.tmp,
                &scratch.gate,
                &lora_scratch.ax_h,
                &lora_scratch.delta_h,
            )?;
            UnoAdapter::apply_delta(
                gpu,
                &ul.up_proj,
                &scratch.tmp,
                &scratch.up,
                &lora_scratch.ax_h,
                &lora_scratch.delta_h,
            )?;
        }
        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.ffn_hidden)?;
        weight_gemv(gpu, &layer.w_down, &scratch.ffn_hidden, &scratch.ffn_out)?;
        if lora {
            UnoAdapter::apply_delta(
                gpu,
                &ul.down_proj,
                &scratch.ffn_hidden,
                &scratch.ffn_out,
                &lora_scratch.ax_h,
                &lora_scratch.delta_h,
            )?;
        }
        gpu.add_inplace_f32(&scratch.x, &scratch.ffn_out)?;
    }

    config.rmsnorm(gpu, &scratch.x, &weights.output_norm, &scratch.tmp)?;
    weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
    gpu.download_f32(&scratch.logits)
}

/// Reusable F32 scratch for the LoRA deltas (largest shapes dominate).
pub struct UnoScratch {
    ax_q: GpuTensor,
    delta_q: GpuTensor,
    ax_kv: GpuTensor,
    delta_kv: GpuTensor,
    ax_h: GpuTensor,
    delta_h: GpuTensor,
}

impl UnoScratch {
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig) -> HipResult<Self> {
        let q = config.n_heads * config.head_dim;
        let kv = config.n_kv_heads * config.head_dim;
        Ok(Self {
            ax_q: gpu.alloc_tensor(&[q], DType::F32)?,
            delta_q: gpu.alloc_tensor(&[q], DType::F32)?,
            ax_kv: gpu.alloc_tensor(&[kv], DType::F32)?,
            delta_kv: gpu.alloc_tensor(&[kv], DType::F32)?,
            ax_h: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
            delta_h: gpu.alloc_tensor(&[config.hidden_dim], DType::F32)?,
        })
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.ax_q,
            self.delta_q,
            self.ax_kv,
            self.delta_kv,
            self.ax_h,
            self.delta_h,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Persistent batch workspace; separate from the scalar reference workspace.
pub struct UnoBatchScratch {
    pub pbs: hipfire_runtime::llama::PrefillBatchScratch,
    norm: GpuTensor,
    ax: GpuTensor,
    hidden: GpuTensor,
    pub(crate) draft_logits: GpuTensor,
    pub(crate) verify_logits: GpuTensor,
    pub(crate) proposals: GpuTensor,
    pub(crate) decisions: GpuTensor,
    /// Per-row argmax sink (`argmax_f32_batched` writes one i32 per row).
    pub(crate) picks: GpuTensor,
    /// `sample_top_p_pf` scratch: result `[2]` + repeat `[1]` (no penalty).
    pub(crate) sample_result: GpuTensor,
    pub(crate) sample_repeat: GpuTensor,
    /// Tree verify: per-row logsumexp, top-k round output, ancestor bias.
    pub(crate) lse: GpuTensor,
    pub(crate) topk: GpuTensor,
    pub(crate) bias: GpuTensor,
    /// Filter-verify candidate gather scratch: 64 (idx, val) per row for the
    /// target and draft rows (`uno_filter_gather` writes p then q into the
    /// first/second half; `uno_filter_finalize` consumes both).
    pub(crate) filter_vals: GpuTensor,
    pub(crate) filter_idxs: GpuTensor,
}

impl UnoBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig, rank: usize, rows: usize, capacity: usize) -> HipResult<Self> {
        let pbs = hipfire_runtime::llama::PrefillBatchScratch::new(gpu, config, rows, capacity)?;
        let tree_cols = rows * rows;
        let mut tensors = Vec::with_capacity(15);
        for size in [rows * config.dim, rows * rank,
            rows * config.hidden_dim, rows * config.vocab_size, rows * config.vocab_size, rows, rows * 2,
            rows, 2, 1,
            rows, rows * 2 * 8, tree_cols.max(1),
            rows * 64 * 2, rows * 64 * 2] {
            match gpu.alloc_tensor(&[size], DType::F32) {
                Ok(t) => tensors.push(t),
                Err(e) => {
                    for t in tensors { let _ = gpu.free_tensor(t); }
                    pbs.free_gpu(gpu);
                    return Err(e);
                }
            }
        }
        let mut tensors = tensors.into_iter();
        Ok(Self {
            pbs, norm: tensors.next().unwrap(), ax: tensors.next().unwrap(),
            hidden: tensors.next().unwrap(),
            draft_logits: tensors.next().unwrap(), verify_logits: tensors.next().unwrap(),
            proposals: tensors.next().unwrap(), decisions: tensors.next().unwrap(),
            picks: tensors.next().unwrap(), sample_result: tensors.next().unwrap(),
            sample_repeat: tensors.next().unwrap(),
            lse: tensors.next().unwrap(), topk: tensors.next().unwrap(),
            bias: tensors.next().unwrap(),
            filter_vals: tensors.next().unwrap(), filter_idxs: tensors.next().unwrap(),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.pbs.free_gpu(gpu);
        for tensor in [self.norm, self.ax, self.hidden, self.draft_logits, self.verify_logits, self.proposals, self.decisions, self.picks, self.sample_result, self.sample_repeat, self.lse, self.topk, self.bias, self.filter_vals, self.filter_idxs] { let _ = gpu.free_tensor(tensor); }
    }

    fn delta(&self, gpu: &mut Gpu, p: &UnoProj, x: &GpuTensor, y: &GpuTensor, rows: usize) -> HipResult<()> {
        // Gate row zero out by never launching or adding its delta. The
        // B side (M=proj, K=rank) lands directly on the base projection
        // (accumulate epilogue). A used to be split-K + memset: M=rank
        // (128) × 8 splits is 1024 workgroups plus a device memset per
        // module × 7 × 36 layers — hundreds of extra launches on the
        // draft forward. Plain xbatch overwrites `ax` and is enough
        // waves for rank-128. `gemv_f16_xbatch` admits b in 1..=8.
        // Upstream `max_diffusion_block_size` is 16, so a block of 16
        // is 15 noise rows — chunk into 8-wide groups rather than
        // refusing the width.
        const XBATCH: usize = 8;
        let n = rows - 1;
        if n == 0 { return Ok(()); }
        let x = x.sub_offset(p.k, n * p.k);
        let y = y.sub_offset(p.m, n * p.m);
        let mut off = 0;
        while off < n {
            let chunk = (n - off).min(XBATCH);
            let x_c = x.sub_offset(off * p.k, chunk * p.k);
            let y_c = y.sub_offset(off * p.m, chunk * p.m);
            let ax = self.ax.sub_offset(0, chunk * p.rank);
            gpu.gemv_f16_xbatch(&p.a, &x_c, &ax, p.rank, p.k, chunk, false)?;
            gpu.gemv_f16_xbatch(&p.b, &ax, &y_c, p.m, p.rank, chunk, true)?;
            off += chunk;
        }
        Ok(())
    }

    /// Normed, original-basis input for a conditional projection delta. On
    /// layouts whose `weight_gemm` arm rotates internally (MQ-V2 family) or
    /// not at all (plain HFQ4/Q8), the chunk's `x_rot_batch` already holds
    /// the plain normed rows — recompute only for the V1-MQ layouts where
    /// the chunk pre-rotated it.
    fn normed_input<'a>(
        gpu: &mut Gpu,
        config: &LlamaConfig,
        weights: &LlamaWeights,
        pbs: &'a hipfire_runtime::llama::PrefillBatchScratch,
        norm_buf: &'a GpuTensor,
        layer: usize,
        rows: usize,
        attn: bool,
    ) -> HipResult<&'a GpuTensor> {
        let projection = if attn { &weights.layers[layer].wq } else { &weights.layers[layer].w_gate };
        if matches!(projection.gpu_dtype, DType::MQ4G256 | DType::MQ6G256 | DType::MQ3G256 | DType::MFP4G32) {
            let weight = if attn { &weights.layers[layer].attn_norm } else { &weights.layers[layer].ffn_norm };
            config.rmsnorm_batch(gpu, &pbs.x_batch, weight, norm_buf, rows)?;
            Ok(norm_buf)
        } else {
            Ok(&pbs.x_rot_batch)
        }
    }
}

/// Upload window token ids + positions into the batch scratch. Kept outside
/// hipGraph capture so replay can swap the 8 i32s without recapturing.
pub(crate) fn uno_upload_window(
    gpu: &Gpu, batch: &UnoBatchScratch, tokens: &[u32], pos: usize,
) -> HipResult<()> {
    let n = tokens.len();
    let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4)
    };
    gpu.hip.memcpy_htod(&batch.pbs.tokens.buf, token_bytes)?;
    let positions_host: Vec<i32> = (0..n).map(|i| (pos + i) as i32).collect();
    let pos_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4)
    };
    gpu.hip.memcpy_htod(&batch.pbs.positions.buf, pos_bytes)
}

/// Causal conditional-LoRA batch. Row zero is base-only; subsequent rows use
/// the adapter. `None` performs base verification. Returns token-major logits.
#[allow(clippy::too_many_arguments)]
pub fn uno_forward_batch(
    gpu: &mut Gpu, weights: &LlamaWeights, config: &LlamaConfig,
    uno: Option<&UnoAdapter>, tokens: &[u32], pos: usize,
    kv: &mut KvCache, scratch: &ForwardScratch, batch: &UnoBatchScratch,
) -> HipResult<Vec<f32>> {
    let output = if uno.is_some() { &batch.draft_logits } else { &batch.verify_logits };
    uno_forward_batch_device(gpu, weights, config, uno, tokens, pos, kv, scratch, batch, output)?;
    gpu.download_f32(&output.sub_offset(0, tokens.len() * config.vocab_size))
}

/// Device-output variant for fused verification; does not download logits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uno_forward_batch_device(
    gpu: &mut Gpu, weights: &LlamaWeights, config: &LlamaConfig,
    uno: Option<&UnoAdapter>, tokens: &[u32], pos: usize,
    kv: &mut KvCache, scratch: &ForwardScratch, batch: &UnoBatchScratch,
    output: &GpuTensor,
) -> HipResult<()> {
    use hipfire_runtime::llama::{forward_prefill_adapter_batch, PrefillProjectionStage};
    if !gpu.graphs.capture_mode {
        uno_upload_window(gpu, batch, tokens, pos)?;
    }
    let mut hook = |gpu: &mut Gpu, c: &LlamaConfig, w: &LlamaWeights,
        pbs: &hipfire_runtime::llama::PrefillBatchScratch, layer: usize, rows: usize,
        stage: PrefillProjectionStage| -> HipResult<()> {
        let adapter = uno.expect("hook only installed with adapter");
        let ul = &adapter.layers[layer];
        match stage {
            PrefillProjectionStage::Qkv => {
                let normed = UnoBatchScratch::normed_input(gpu, c, w, pbs, &batch.norm, layer, rows, true)?;
                batch.delta(gpu, &ul.q_proj, normed, &pbs.fa_q_batch, rows)?;
                batch.delta(gpu, &ul.k_proj, normed, &pbs.fa_k_batch, rows)?;
                batch.delta(gpu, &ul.v_proj, normed, &pbs.fa_v_batch, rows)?;
            }
            PrefillProjectionStage::AttentionOutput => {
                batch.delta(gpu, &ul.o_proj, &pbs.fa_attn_out_batch, &pbs.x_batch, rows)?;
            }
            PrefillProjectionStage::GateUp => {
                let normed = UnoBatchScratch::normed_input(gpu, c, w, pbs, &batch.norm, layer, rows, false)?;
                batch.delta(gpu, &ul.gate_proj, normed, &pbs.gate_ffn_batch, rows)?;
                batch.delta(gpu, &ul.up_proj, normed, &pbs.up_batch, rows)?;
            }
            PrefillProjectionStage::Down => {
                let size = rows * c.hidden_dim;
                gpu.silu_mul_f32(&pbs.gate_ffn_batch.sub_offset(0, size),
                    &pbs.up_batch.sub_offset(0, size), &batch.hidden.sub_offset(0, size))?;
                batch.delta(gpu, &ul.down_proj, &batch.hidden, &pbs.x_batch, rows)?;
            }
        }
        Ok(())
    };
    forward_prefill_adapter_batch(gpu, weights, config, tokens, pos, kv,
        scratch, &batch.pbs, if uno.is_some() { Some(&mut hook) } else { None })?;
    // Batched output head: one norm + one batched GEMM reads the lm_head
    // weights ONCE for all rows (the per-row GEMV loop re-read them per row —
    // 8 full lm_head passes per window at block_len 4).
    let rows = tokens.len();
    let x_rows = batch.pbs.x_batch.sub_offset(0, rows * config.dim);
    config.rmsnorm_batch(gpu, &x_rows, &weights.output_norm, &batch.norm, rows)?;
    // Window-sized lm_head: WMMA residual gets one 16-wide N-tile and
    // underfills; xbatch reads the 250k-row head once against all rows.
    let logits = output.sub_offset(0, rows * config.vocab_size);
    if weights.output.gpu_dtype == DType::MQ4G256V2 {
        let x_rot = batch.pbs.v2_rot.sub_offset(0, rows * config.dim);
        hipfire_runtime::llama::mq4g256v2_window_rotate_project(
            gpu, &weights.output, &batch.norm, &logits, &x_rot, rows,
        )?;
    } else {
        weight_gemm(gpu, &weights.output, &batch.norm, &logits, rows)?;
    }
    Ok(())
}

// ── Ψ-Spec tree verification ────────────────────────────────────────────────
// Port of ifm-ai/uno nano_vllm_uno/engine/draft_tree.py (best-first
// prefix-closed tree over per-depth draft top-k candidates) plus the
// tree-attention verify forward on hipfire's existing TreeMaskRef machinery
// and the accepted-path KV compaction (upstream compact_tree_kv).

/// One draft candidate: token id and temperature-scaled log-prob.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeCandidate {
    pub token: u32,
    pub log_prob: f64,
}

struct TreeNode {
    token: u32,
    depth: i32,
    parent: i32,
    log_mass: f64,
    children: std::collections::HashMap<u32, usize>,
}

/// A built draft tree in parent-before-child (root-first) order.
pub(crate) struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
    pub depths: Vec<i32>,
}

/// Fixed-budget best-first tree: expose each built node's depth candidates,
/// always grow the highest accumulated log-mass candidate (ties: lower depth,
/// lower rank, lower token id, lower parent — the upstream deterministic
/// ordering). The root (the already-committed clean token) is node 0.
pub(crate) fn build_best_first_tree(
    root: u32,
    depth_candidates: &[Vec<TreeCandidate>],
    max_nodes: usize,
) -> DraftTree {
    use std::cmp::{Ordering, Reverse};
    let mut nodes = vec![TreeNode {
        token: root,
        depth: 0,
        parent: -1,
        log_mass: 0.0,
        children: Default::default(),
    }];
    #[derive(PartialEq)]
    struct Cand {
        neg_mass: f64,
        depth: i32,
        rank: usize,
        token: u32,
        parent: usize,
    }
    impl Eq for Cand {}
    impl Ord for Cand {
        fn cmp(&self, other: &Self) -> Ordering {
            self.neg_mass
                .total_cmp(&other.neg_mass)
                .then(self.depth.cmp(&other.depth))
                .then(self.rank.cmp(&other.rank))
                .then(self.token.cmp(&other.token))
                .then(self.parent.cmp(&other.parent))
        }
    }
    impl PartialOrd for Cand {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut heap: std::collections::BinaryHeap<Reverse<Cand>> = Default::default();

    fn expose(
        nodes: &[TreeNode],
        depth_candidates: &[Vec<TreeCandidate>],
        heap: &mut std::collections::BinaryHeap<Reverse<Cand>>,
        parent_index: usize,
    ) {
        let parent_node = &nodes[parent_index];
        let position = parent_node.depth as usize;
        if position >= depth_candidates.len() {
            return;
        }
        for (rank, cand) in depth_candidates[position].iter().enumerate() {
            if parent_node.children.contains_key(&cand.token) {
                continue;
            }
            heap.push(Reverse(Cand {
                neg_mass: -(parent_node.log_mass + cand.log_prob),
                depth: parent_node.depth + 1,
                rank,
                token: cand.token,
                parent: parent_index,
            }));
        }
    }

    expose(&nodes, depth_candidates, &mut heap, 0);
    while nodes.len() < max_nodes {
        let Some(Reverse(cand)) = heap.pop() else { break };
        if nodes[cand.parent].children.contains_key(&cand.token) {
            continue;
        }
        let index = nodes.len();
        nodes[cand.parent].children.insert(cand.token, index);
        nodes.push(TreeNode {
            token: cand.token,
            depth: cand.depth,
            parent: cand.parent as i32,
            log_mass: -cand.neg_mass,
            children: Default::default(),
        });
        expose(&nodes, depth_candidates, &mut heap, index);
    }

    DraftTree {
        tokens: nodes.iter().map(|n| n.token).collect(),
        parents: nodes.iter().map(|n| n.parent).collect(),
        depths: nodes.iter().map(|n| n.depth).collect(),
    }
}

/// Greedy-or-sampled traversal of the target picks through the draft tree
/// (port of upstream `walk_tree`): at the current node take its target pick;
/// commit it; descend into the child carrying that token while one exists and
/// depth remains. Returns the committed tokens and the matched node path
/// (root first) whose KV is base-correct for the committed prefix.
pub(crate) fn walk_draft_tree(
    tree: &DraftTree,
    picks: &[u32],
    max_depth: i32,
    is_stop: impl Fn(u32) -> bool,
) -> (Vec<u32>, Vec<usize>) {
    let mut committed = Vec::new();
    let mut path = vec![0usize];
    let mut current = 0usize;
    let mut depth = 0i32;
    loop {
        let pick = picks[current];
        committed.push(pick);
        if is_stop(pick) || depth >= max_depth {
            break;
        }
        let child = (1..tree.tokens.len()).find(|&i| {
            tree.parents[i] == current as i32 && tree.tokens[i] == pick
        });
        match child {
            Some(c) => {
                path.push(c);
                current = c;
                depth += 1;
            }
            None => break,
        }
    }
    (committed, path)
}

/// Gather the accepted path's per-slot KV rows into the contiguous committed
/// positions (upstream `compact_tree_kv`). Path nodes are parent-before-child,
/// so every source slot index is ≥ its destination — the in-place left shift
/// below never overwrites a row that is still needed.
pub(crate) fn compact_tree_kv(
    gpu: &mut Gpu,
    kv: &hipfire_runtime::llama::KvCache,
    path: &[usize],
    position: usize,
) -> HipResult<()> {
    debug_assert!(kv.quant_q8, "tree KV compaction requires the Q8_0 layout");
    let blocks_per_token = kv.n_kv_heads * (kv.head_dim / 32);
    let row_bytes = blocks_per_token * 34; // Q8_0: 32 int8 + f16 scale per block
    for (j, &node) in path.iter().enumerate().skip(1) {
        let dst = (position + 1 + j) * row_bytes;
        let src = (position + 1 + node) * row_bytes;
        if dst == src {
            continue;
        }
        for layer in 0..kv.k_gpu.len() {
            gpu.hip.memcpy_dtod_at(&kv.k_gpu[layer].buf, dst, &kv.k_gpu[layer].buf, src, row_bytes)?;
            gpu.hip.memcpy_dtod_at(&kv.v_gpu[layer].buf, dst, &kv.v_gpu[layer].buf, src, row_bytes)?;
        }
    }
    Ok(())
}

/// Tree verify forward + per-node picks: one tree-attention batched forward
/// (KV writes land on contiguous slots `position+1 ..`, RoPE uses depth
/// positions), then one batched lm_head over all node rows and a pick per
/// node (GPU argmax, or a fused GPU sample at temp>0 under the request's
/// top_p/top_k law — the reference runs every node pick through the full
/// request sampler). Returns the picks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uno_tree_verify_picks(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tree: &DraftTree,
    position: usize,
    kv: &mut KvCache,
    scratch: &ForwardScratch,
    batch: &UnoBatchScratch,
    greedy: bool,
    temp: f32,
    // Request filter law for non-greedy picks (same expression as
    // `step_device`): the tree's target draws must match AR decode at the
    // same request config, exactly as the reference tree sampler filters
    // every pick.
    pick_top_p: f32,
    pick_top_k: Option<u32>,
    rng: &mut u32,
    // AR's repeat/presence/frequency penalty, shared by every node row. The
    // per-node history is `base_hist` plus that node's root path, since a node's
    // token sits at the end of its own ancestor chain — the same convention the
    // linear window uses (history includes the token being forwarded).
    base_hist: &[u32],
    repeat_buf: Option<&GpuTensor>,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_window: usize,
) -> HipResult<Vec<u32>> {
    let nodes = tree.tokens.len();
    // Ancestor bias: row i sees itself and its tree ancestors (0.0); every
    // other in-block column is masked off (-INF). Prompt keys before the
    // block stay fully visible in the attention kernel.
    let mut bias = vec![f32::NEG_INFINITY; nodes * nodes];
    for i in 0..nodes {
        bias[i * nodes + i] = 0.0;
        let mut anc = tree.parents[i];
        while anc >= 0 {
            bias[i * nodes + anc as usize] = 0.0;
            anc = tree.parents[anc as usize];
        }
    }
    let bytes: Vec<u8> = bias.iter().flat_map(|v| v.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&batch.bias.buf, &bytes)?;

    let depth_positions: Vec<i32> =
        tree.depths.iter().map(|&d| (position as i32 + 1 + d)).collect();

    hipfire_runtime::llama::forward_prefill_batch_tree(
        gpu,
        weights,
        config,
        &tree.tokens,
        position + 1,
        &batch.bias,
        &depth_positions,
        kv,
        scratch,
        &batch.pbs,
        None,
    )?;

    // Batched output head over all node rows → verify_logits [nodes × vocab].
    let vocab = config.vocab_size;
    let x_rows = batch.pbs.x_batch.sub_offset(0, nodes * config.dim);
    config.rmsnorm_batch(gpu, &x_rows, &weights.output_norm, &batch.norm, nodes)?;
    weight_gemm(
        gpu,
        &weights.output,
        &batch.norm,
        &batch.verify_logits.sub_offset(0, nodes * vocab),
        nodes,
    )?;

    // Phase 0 (AR's law) per node row, before the picks are taken. No-op when
    // every penalty is neutral, so the common path costs nothing.
    if let Some(buf) = repeat_buf {
        if repeat_window > 0
            && (repeat_penalty > 1.0 || presence_penalty > 0.0 || frequency_penalty > 0.0)
        {
            let mut hist: Vec<u32> = base_hist.to_vec();
            let base_len = hist.len();
            for node in 0..nodes {
                hist.truncate(base_len);
                // Root-first ancestor chain ending at this node.
                let mut chain: Vec<u32> = vec![tree.tokens[node]];
                let mut anc = tree.parents[node];
                while anc >= 0 {
                    chain.push(tree.tokens[anc as usize]);
                    anc = tree.parents[anc as usize];
                }
                chain.reverse();
                hist.extend_from_slice(&chain);
                let last = hist.len().min(repeat_window);
                let window = &hist[hist.len() - last..];
                let bytes: Vec<u8> = window.iter().flat_map(|t| t.to_ne_bytes()).collect();
                gpu.hip.memcpy_htod(&buf.buf, &bytes)?;
                gpu.apply_repeat_penalty_row(
                    &batch.verify_logits.sub_offset(node * vocab, vocab),
                    buf,
                    vocab,
                    window.len(),
                    repeat_penalty,
                    presence_penalty,
                    frequency_penalty,
                )?;
            }
        }
    }

    if greedy {
        gpu.argmax_f32_batched(&batch.verify_logits, &batch.picks, vocab, nodes)?;
        let mut raw = vec![0u8; nodes * 4];
        gpu.hip.memcpy_dtoh(&mut raw, &batch.picks.buf)?;
        Ok(raw
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    } else {
        let mut picks = Vec::with_capacity(nodes);
        for row in 0..nodes {
            let tok = crate::uno_spec::sample_row(
                gpu,
                &batch.verify_logits.sub_offset(row * vocab, vocab),
                vocab,
                temp,
                pick_top_p,
                pick_top_k,
                &batch.sample_result,
                &batch.sample_repeat,
                rng,
            ).map_err(|e| hip_bridge::HipError::new(0, &e))?;
            picks.push(tok);
        }
        Ok(picks)
    }
}

#[cfg(test)]
mod uno_tree_tests {
    use super::*;

    fn cands(spec: &[(u32, f64)]) -> Vec<TreeCandidate> {
        spec.iter().map(|&(t, p)| TreeCandidate { token: t, log_prob: p }).collect()
    }

    #[test]
    fn best_first_tree_is_prefix_closed_budgeted_and_deterministic() {
        // Depth 1: A (logp -0.1) beats B (-0.5); depth 2: C (-0.2), D (-0.3).
        // Budget 4 → root + A + (A,C) + (A,D): the best-first order grows the
        // highest-mass chain first and never emits an orphan.
        let depths = vec![
            cands(&[(100, -0.1), (200, -0.5)]),
            cands(&[(300, -0.2), (400, -0.3)]),
        ];
        let tree = build_best_first_tree(7, &depths, 4);
        assert_eq!(tree.tokens, vec![7, 100, 300, 400]);
        assert_eq!(tree.parents, vec![-1, 0, 1, 1]);
        assert_eq!(tree.depths, vec![0, 1, 2, 2]);
        // Determinism: same inputs, same tree.
        let again = build_best_first_tree(7, &depths, 4);
        assert_eq!(again.tokens, tree.tokens);
        // Budget 1 → root only.
        let root_only = build_best_first_tree(7, &depths, 1);
        assert_eq!(root_only.tokens, vec![7]);
        // A second root child is preferred over a depth-2 grandchild when its
        // mass wins: B(-0.5) vs (A,C) mass -0.3 → after root+A, next pop is
        // (A,C) at -0.3, then B at -0.5, then (A,D) at -0.4... order check:
        let tree3 = build_best_first_tree(7, &depths, 5);
        assert_eq!(tree3.tokens, vec![7, 100, 300, 400, 200]);
        assert_eq!(tree3.parents, vec![-1, 0, 1, 1, 0]);
    }

    #[test]
    fn walk_commits_the_picked_path_and_stops_at_mismatch() {
        // Tree: root 7 → {100 (node 1), 200 (node 4)}; 100 → {300 (2), 400 (3)}.
        let tree = DraftTree {
            tokens: vec![7, 100, 300, 400, 200],
            parents: vec![-1, 0, 1, 1, 0],
            depths: vec![0, 1, 2, 2, 1],
        };
        // Production supplies one pick per node. Root pick 100 descends; the
        // depth-1 pick 300 descends again; the depth-2 pick 999 has no child
        // so it is committed as the correction and the walk stops.
        let picks = [100, 300, 999, 400, 200];
        let (committed, path) = walk_draft_tree(&tree, &picks, 3, |t| t == 1);
        assert_eq!(committed, vec![100, 300, 999]);
        assert_eq!(path, vec![0, 1, 2]);
        // A stop pick halts immediately after committing.
        let picks = [100, 1, 999, 400, 200];
        let (committed, path) = walk_draft_tree(&tree, &picks, 3, |t| t == 1);
        assert_eq!(committed, vec![100, 1]);
        assert_eq!(path, vec![0, 1]);
        // Depth cap 1: the root pick descends once; the depth-1 node's pick
        // commits, then the budget is spent.
        let picks = [100, 300, 999, 400, 200];
        let (committed, path) = walk_draft_tree(&tree, &picks, 1, |t| t == 1);
        assert_eq!(committed, vec![100, 300]);
        assert_eq!(path, vec![0, 1]);
    }
}
