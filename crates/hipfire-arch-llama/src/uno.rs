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
//!    applies ONLY to noise rows (row 0 mask 0). Proposals: argmax of the
//!    clean row logit (seed token, already committed) and each noise row.
//!    Noise KV is discarded (verify overwrites those slots).
//! 2. VERIFY: forward `[clean, proposal_1..proposal_{L-1}]` base-only.
//!    Accept the greedy prefix where target argmax == proposal; on
//!    rejection the first rejected slot is replaced by the target's
//!    correction token; append the lookahead argmax from the last verify
//!    row; truncate at EOS.
//! 3. KV repair: committed tokens are REPLAYED as a base prefill from the
//!    window start so the KV frontier holds base computations only.
//!
//! Greedy only in this bring-up (temp 0). Stochastic rejection sampling
//! needs the sparse draft distribution retained at draft time — deferred.

use hip_bridge::HipResult;
use hipfire_runtime::llama::{
    embedding_lookup_dispatch, weight_gemv, ForwardScratch, KvCache, LlamaConfig, LlamaWeights,
};
use rdna_compute::{DType, Gpu, GpuTensor};

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
        let scale = alpha / rank as f32;

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
                let a = gpu
                    .upload_f32(&a_f32, &[rank, k])
                    .map_err(|e| format!("uno: upload A {module}/{layer}: {e:?}"))?;
                let mut b = match gpu.upload_raw(&b_data, &[m, rank]) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = gpu.free_tensor(a);
                        return Err(format!("uno: upload B {module}/{layer}: {e:?}"));
                    }
                };
                b.dtype = DType::F32;
                Ok(UnoProj { a, b, rank, k, m })
            };
            layers.push(UnoLayer {
                q_proj: mk("self_attn.q_proj", q_out, dim, gpu)?,
                k_proj: mk("self_attn.k_proj", kv_out, dim, gpu)?,
                v_proj: mk("self_attn.v_proj", kv_out, dim, gpu)?,
                o_proj: mk("self_attn.o_proj", dim, q_out, gpu)?,
                gate_proj: mk("mlp.gate_proj", hidden, dim, gpu)?,
                up_proj: mk("mlp.up_proj", hidden, dim, gpu)?,
                down_proj: mk("mlp.down_proj", dim, hidden, gpu)?,
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
            let UnoLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
            } = l;
            for p in [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj] {
                let _ = gpu.free_tensor(p.a);
                let _ = gpu.free_tensor(p.b);
            }
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
        gpu.gemv_f32(&p.a, x, &ax)?;
        gpu.gemv_f32(&p.b, &ax, &delta)?;
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
    delta: GpuTensor,
    hidden: GpuTensor,
    pub(crate) draft_logits: GpuTensor,
    pub(crate) verify_logits: GpuTensor,
    pub(crate) proposals: GpuTensor,
    pub(crate) decisions: GpuTensor,
}

impl UnoBatchScratch {
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig, rank: usize, rows: usize, capacity: usize) -> HipResult<Self> {
        let pbs = hipfire_runtime::llama::PrefillBatchScratch::new(gpu, config, rows, capacity)?;
        let mut tensors = Vec::with_capacity(8);
        for size in [rows * config.dim, rows * rank, rows * config.hidden_dim.max(config.dim),
            rows * config.hidden_dim, rows * config.vocab_size, rows * config.vocab_size, rows, rows * 2] {
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
            delta: tensors.next().unwrap(), hidden: tensors.next().unwrap(),
            draft_logits: tensors.next().unwrap(), verify_logits: tensors.next().unwrap(),
            proposals: tensors.next().unwrap(), decisions: tensors.next().unwrap(),
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.pbs.free_gpu(gpu);
        for tensor in [self.norm, self.ax, self.delta, self.hidden, self.draft_logits, self.verify_logits, self.proposals, self.decisions] { let _ = gpu.free_tensor(tensor); }
    }

    fn delta(&self, gpu: &mut Gpu, p: &UnoProj, x: &GpuTensor, y: &GpuTensor, rows: usize) -> HipResult<()> {
        // Gate row zero out by never launching or adding its delta.
        let n = rows - 1;
        if n == 0 { return Ok(()); }
        let x = x.sub_offset(p.k, n * p.k);
        let y = y.sub_offset(p.m, n * p.m);
        let ax = self.ax.sub_offset(0, n * p.rank);
        let delta = self.delta.sub_offset(0, n * p.m);
        gpu.gemm_f32_batched(&p.a, &x, &ax, p.rank, p.k, n)?;
        gpu.gemm_f32_batched(&p.b, &ax, &delta, p.m, p.rank, n)?;
        gpu.add_inplace_f32(&y, &delta)
    }
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
    let mut hook = |gpu: &mut Gpu, c: &LlamaConfig, w: &LlamaWeights,
        pbs: &hipfire_runtime::llama::PrefillBatchScratch, layer: usize, rows: usize,
        stage: PrefillProjectionStage| -> HipResult<()> {
        let adapter = uno.expect("hook only installed with adapter");
        let ul = &adapter.layers[layer];
        match stage {
            PrefillProjectionStage::Qkv => {
                c.rmsnorm_batch(gpu, &pbs.x_batch, &w.layers[layer].attn_norm, &batch.norm, rows)?;
                batch.delta(gpu, &ul.q_proj, &batch.norm, &pbs.fa_q_batch, rows)?;
                batch.delta(gpu, &ul.k_proj, &batch.norm, &pbs.fa_k_batch, rows)?;
                batch.delta(gpu, &ul.v_proj, &batch.norm, &pbs.fa_v_batch, rows)?;
            }
            PrefillProjectionStage::AttentionOutput => {
                batch.delta(gpu, &ul.o_proj, &pbs.fa_attn_out_batch, &pbs.x_batch, rows)?;
            }
            PrefillProjectionStage::GateUp => {
                c.rmsnorm_batch(gpu, &pbs.x_batch, &w.layers[layer].ffn_norm, &batch.norm, rows)?;
                batch.delta(gpu, &ul.gate_proj, &batch.norm, &pbs.gate_ffn_batch, rows)?;
                batch.delta(gpu, &ul.up_proj, &batch.norm, &pbs.up_batch, rows)?;
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
    for row in 0..tokens.len() {
        let x = batch.pbs.x_batch.sub_offset(row * config.dim, config.dim);
        config.rmsnorm(gpu, &x, &weights.output_norm, &scratch.tmp)?;
        let logits = output.sub_offset(row * config.vocab_size, config.vocab_size);
        weight_gemv(gpu, &weights.output, &scratch.tmp, &logits)?;
    }
    Ok(())
}
