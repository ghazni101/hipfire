// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for K2-Horizon.
//!
//! The trait surface provides config parsing, weight loading, and state
//! allocation entry points. Forward-pass dispatch is NOT routed through
//! the trait — the daemon calls arch-specific forward functions directly,
//! matching the qwen35 pattern (see `hipfire-arch-qwen35/src/arch.rs`).

use crate::config::{config_from_hfq, K2HorizonConfig};
use crate::forward::K2HorizonState;
use crate::weights::{
    DenseLayerWeights, K2HorizonWeights, MoeExpertWeights, MoeFfnWeights, MovaAttnWeights,
    MovaLayerWeights, SharedExpertWeights,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Type marker for the K2-Horizon architecture (MoVA attention +
/// sigmoid-routed MoE FFN). arch_id = 15.
pub struct K2Horizon;

// ─── HFQ load helpers (mirrors cohere2moe) ──────────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("k2_horizon: tensor not found in HFQ: {name}"))?;
    Ok((info.quant_type, data))
}

/// Load a 1D norm/embedding weight (F16/F32/Q8) → F32 GpuTensor.
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => dequant_q8_0(&data),
        _ => {
            return Err(format!(
                "k2_horizon: expected F16/F32/Q8 for {name}, got qt={qt}"
            ))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("k2_horizon: upload {name}: {e:?}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

/// Load a 2D weight tensor (quantized or F16/F32) → WeightTensor.
fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k)
        .map_err(|e| format!("k2_horizon: load_wt {name}: {e}"))
}

/// quant_type → DType mapping (mirrors cohere2moe::wt_from_raw).
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        1 => DType::F16,
        2 => DType::F32,
        16 => DType::BF16,
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

// ─── Architecture impl ──────────────────────────────────────────────────

impl Architecture for K2Horizon {
    type Weights = K2HorizonWeights;
    type State = K2HorizonState;
    type Config = K2HorizonConfig;

    fn arch_id() -> u32 {
        15
    }

    fn name() -> &'static str {
        "k2_horizon"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        config_from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        let hidden = cfg.dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let dense_inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let n_exp = cfg.num_experts;
        let mova_n_exp = cfg.mova_num_experts;

        // Determine which layers are dense vs MoE.
        let dense_set: std::collections::HashSet<usize> =
            cfg.mlp_only_layers.iter().copied().collect();

        // Globals.
        let token_embd = load_f32(hfq, gpu, "model.embed_tokens.weight", &[cfg.vocab_size, hidden])?;
        let final_norm = load_f32(hfq, gpu, "model.norm.weight", &[hidden])?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_wt(
                hfq,
                gpu,
                "lm_head.weight",
                cfg.vocab_size,
                hidden,
            )?)
        };

        let mut dense_layers = Vec::new();
        let mut moe_layers = Vec::new();

        for l in 0..cfg.n_layers {
            let p = format!("model.layers.{l}");

            let attn_norm =
                load_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let ffn_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;

            if dense_set.contains(&l) {
                // ── Dense layer (0–2): standard MHA + dense SwiGLU MLP ──
                let wq = load_wt(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, hidden)?;
                let wk = load_wt(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden)?;
                let wv = load_wt(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden)?;
                let wo = load_wt(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), hidden, q_dim)?;
                let w_gate = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    dense_inter,
                    hidden,
                )?;
                let w_up = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    dense_inter,
                    hidden,
                )?;
                let w_down = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    dense_inter,
                )?;

                dense_layers.push(DenseLayerWeights {
                    attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                });
            } else {
                // ── MoE layer (3–47): MoVA attention + sigmoid MoE FFN ──
                let wq = load_wt(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, hidden)?;
                let wk = load_wt(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden)?;
                let wo = load_wt(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), hidden, q_dim)?;

                // MoVA: v_router [64, 2560], v_experts[64] [4096, 2560], attn_gate [2560, 2560]
                let v_router = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_router.weight"),
                    mova_n_exp,
                    hidden,
                )?;
                let mut v_experts = Vec::with_capacity(mova_n_exp);
                for e in 0..mova_n_exp {
                    let expert = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.v_experts.{e}.weight"),
                        q_dim, // [n_heads * head_dim, dim] = [4096, 2560]
                        hidden,
                    )?;
                    v_experts.push(expert);
                }
                let attn_gate = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    hidden, // [dim, dim] = [2560, 2560]
                    hidden,
                )?;

                // MoE FFN: router [100, 2560], router_bias [100], experts[100], shared
                let router = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate.weight"),
                    n_exp,
                    hidden,
                )?;
                let router_bias = if cfg.moe_gate_bias {
                    Some(load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.mlp.gate.bias"),
                        &[n_exp],
                    )?)
                } else {
                    None
                };

                let mut experts = Vec::with_capacity(n_exp);
                for e in 0..n_exp {
                    let ep = format!("{p}.mlp.experts.{e}");
                    let gate = load_wt(
                        hfq,
                        gpu,
                        &format!("{ep}.gate_proj.weight"),
                        moe_inter,
                        hidden,
                    )?;
                    let up = load_wt(
                        hfq,
                        gpu,
                        &format!("{ep}.up_proj.weight"),
                        moe_inter,
                        hidden,
                    )?;
                    let down = load_wt(
                        hfq,
                        gpu,
                        &format!("{ep}.down_proj.weight"),
                        hidden,
                        moe_inter,
                    )?;
                    experts.push(MoeExpertWeights { gate, up, down });
                }

                // Shared expert (always-on).
                let shared = SharedExpertWeights {
                    gate: load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.mlp.shared_experts.gate_proj.weight"),
                        moe_inter,
                        hidden,
                    )?,
                    up: load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.mlp.shared_experts.up_proj.weight"),
                        moe_inter,
                        hidden,
                    )?,
                    down: load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.mlp.shared_experts.down_proj.weight"),
                        hidden,
                        moe_inter,
                    )?,
                };

                moe_layers.push(MovaLayerWeights {
                    attn_norm,
                    attn: MovaAttnWeights {
                        wq,
                        wk,
                        v_router,
                        v_experts,
                        wo,
                        attn_gate,
                    },
                    ffn_norm,
                    ffn: MoeFfnWeights {
                        router,
                        router_bias,
                        experts,
                        shared,
                    },
                });
            }
        }

        Ok(K2HorizonWeights {
            token_embd,
            dense_layers,
            moe_layers,
            final_norm,
            lm_head,
        })
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        K2HorizonState::new(gpu, cfg)
    }
}
