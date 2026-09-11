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
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("k2_horizon: load_wt {name}: {e}"))
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
        37 => DType::MFP2G32E8,
        41 => DType::BQ1G128,
        44 => DType::MQ4G256V2,
        45 => DType::MQ4CG256,
        47 => DType::MQ6G256V2,
        48 => DType::MQ5G256V2,
        49 => DType::MQ3G256V2,
        50 => DType::MQ2G256V2,
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

/// Quant types that share the 136 B/group stride and can be packed into
/// layer-level blobs. All three have identical byte layout for
/// concatenation — only the 8-byte group header interpretation differs,
/// handled by kernel dispatch on `gpu_dtype`. qt=30 (MQ4G256Lloyd, 160
/// B/group) and qt=47 (MQ6G256V2, 200 B/group) are excluded (different
/// strides). Mirrors the fix/pack-mq4g256v2-experts approach.
fn packable_mq4_dtype(qt: u8) -> Option<DType> {
    match qt {
        13 => Some(DType::MQ4G256),
        44 => Some(DType::MQ4G256V2),
        45 => Some(DType::MQ4CG256),
        _ => None,
    }
}

/// Restrict expert packing to the gfx11 family where the packed-blob layout
/// has been validated. On gfx12, placing every expert view inside two large
/// layer allocations makes the HIP/graph route treat those views as one
/// coarse allocation domain — gfx1201 tg128 fell from 171 to 77 tok/s in the
/// qwen35 port of this fix (fix/pack-mq4g256v2-experts). Preserve per-expert
/// allocations there until a gfx12-safe packing granularity is established.
fn packed_experts_supported(gpu: &Gpu) -> bool {
    gpu.arch_caps.is_rdna3()
}

/// Pack N uniform-quant-type expert tensors into a single GPU allocation,
/// returning WeightTensor views (one per expert) + the owning GpuTensor.
/// Falls back to per-expert loading if the quant type is not packable or
/// if any expert has a different quant type.
///
/// `tensor_specs` is (name, m, k) per expert. The host bytes are read
/// once and concatenated into one buffer before a single H2D.
fn load_packed_experts(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    tensor_specs: &[(String, usize, usize)],
) -> Result<(Vec<WeightTensor>, Option<GpuTensor>), String> {
    if tensor_specs.is_empty() || !packed_experts_supported(gpu) {
        return Ok((vec![], None));
    }

    // Single pass: read each tensor once, requiring a uniform packable
    // quant type and stride across all experts.
    let mut expert_dtype: Option<DType> = None;
    let mut stride: Option<usize> = None;
    let mut host_blob: Vec<u8> = Vec::new();
    for (name, _m, _k) in tensor_specs {
        let (qt, data) = read_tensor(hfq, name)?;
        let Some(dt) = packable_mq4_dtype(qt) else {
            // Not packable — fall back to per-expert loading.
            return Ok((vec![], None));
        };
        match expert_dtype {
            None => expert_dtype = Some(dt),
            Some(existing) if existing == dt => {}
            Some(_) => return Ok((vec![], None)), // mixed dtypes — fallback
        }
        match stride {
            None => stride = Some(data.len()),
            Some(s) if s == data.len() => {}
            Some(_) => return Ok((vec![], None)), // stride mismatch — fallback
        }
        host_blob.extend_from_slice(&data);
    }

    let dtype = expert_dtype.unwrap();
    let stride = stride.unwrap();

    // Single GPU allocation for all experts.
    let owner = gpu
        .upload_raw(&host_blob, &[host_blob.len()])
        .map_err(|e| format!("k2_horizon: packed expert upload: {e:?}"))?;

    // Create views — sub_offset on Raw dtype uses byte offsets.
    let mut views = Vec::with_capacity(tensor_specs.len());
    for (slot, (_name, m, k)) in tensor_specs.iter().enumerate() {
        let view = owner.sub_offset(slot * stride, stride);
        views.push(WeightTensor {
            buf: view,
            gpu_dtype: dtype,
            m: *m,
            k: *k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        });
    }

    Ok((views, Some(owner)))
}

/// The indexed MoE GEMV kernels used by the K2-Horizon forward path
/// (`gemv_mq4g256v2_moe_*_k8_indexed_batched*`) decode fp16 per-128 group
/// headers — the MQ4G256V2 (qt=44) layout. They are the only indexed kernels
/// that support K2-Horizon's K dims (768 / 2560); the V1 kernels are
/// hardcoded to K=512. Any other expert dtype would be silently misdecoded,
/// so refuse it at load time rather than serve garbage.
fn require_v2_expert_dtype(experts: &[WeightTensor], what: &str) -> Result<(), String> {
    for (i, e) in experts.iter().enumerate() {
        if e.gpu_dtype != DType::MQ4G256V2 {
            return Err(format!(
                "k2_horizon: {what}[{i}] has dtype {:?} — indexed MoE GEMV requires MQ4G256V2 (qt=44); re-quantize with --format mq4",
                e.gpu_dtype
            ));
        }
    }
    Ok(())
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
        // Upload embedding as raw Q8_0 bytes (681 MB) instead of dequantizing
        // to F32 (2.57 GB). embedding_lookup_q8 dequantizes one row on-GPU
        // at lookup time. Matches minimax/deepseek4/cohere2moe pattern.
        let (_qt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let token_embd = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("k2_horizon: upload embed: {e:?}"))?;
        let final_norm = load_f32(hfq, gpu, "model.norm.weight", &[hidden])?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(load_wt(hfq, gpu, "lm_head.weight", cfg.vocab_size, hidden)?)
        };

        let mut dense_layers = Vec::new();
        let mut moe_layers = Vec::new();

        for l in 0..cfg.n_layers {
            let p = format!("model.layers.{l}");

            let attn_norm = load_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), &[hidden])?;
            let ffn_norm = load_f32(
                hfq,
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
            )?;

            if dense_set.contains(&l) {
                // ── Dense layer (0–2): standard MHA + dense SwiGLU MLP ──
                let wq = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    hidden,
                )?;
                let wk = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wv = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wo = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                    q_dim,
                )?;
                let attn_gate = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    q_dim, // [num_attention_heads * head_dim, dim] = [4096, 2560]
                    hidden,
                )?;
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
                    attn_gate,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                });
            } else {
                // ── MoE layer (3–47): MoVA attention + sigmoid MoE FFN ──
                let wq = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    hidden,
                )?;
                let wk = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wo = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                    q_dim,
                )?;

                // MoVA: v_router [64, 2560] + v_router.bias [64],
                // v_experts[64] [kv_dim=1024, 2560], attn_gate [2560, 2560].
                let v_router = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_router.weight"),
                    mova_n_exp,
                    hidden,
                )?;
                let v_router_bias = if cfg.moe_gate_bias {
                    Some(load_f32(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.v_router.bias"),
                        &[mova_n_exp],
                    )?)
                } else {
                    None
                };
                // Pack v_experts into one GPU allocation (64 experts → 1
                // hipMalloc instead of 64). Falls back to per-expert if not
                // packable.
                let v_expert_specs: Vec<(String, usize, usize)> = (0..mova_n_exp)
                    .map(|e| {
                        (
                            format!("{p}.self_attn.v_experts.{e}.weight"),
                            kv_dim,
                            hidden,
                        )
                    })
                    .collect();
                let (v_experts, v_experts_owner) = load_packed_experts(hfq, gpu, &v_expert_specs)?;
                let (v_experts, v_experts_owner) = if v_experts.is_empty() {
                    // Fallback: per-expert loading (each expert owns its buffer).
                    let mut v = Vec::with_capacity(mova_n_exp);
                    for e in 0..mova_n_exp {
                        v.push(load_wt(
                            hfq,
                            gpu,
                            &format!("{p}.self_attn.v_experts.{e}.weight"),
                            kv_dim,
                            hidden,
                        )?);
                    }
                    (v, None)
                } else {
                    (v_experts, v_experts_owner)
                };
                // The MoVA indexed GEMV is V2-only (fp16 per-128 headers);
                // refuse any other expert dtype instead of misdecoding it.
                require_v2_expert_dtype(&v_experts, &format!("{p}.self_attn.v_experts"))?;
                // Device pointer table for indexed MoE GEMV: mova_n_exp u64
                // device addresses stored as [2*mova_n_exp] F32 (8 B/ptr).
                let ve_ptrs: Vec<u8> = v_experts
                    .iter()
                    .flat_map(|e| (e.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let v_expert_ptrs = gpu
                    .alloc_tensor(&[2 * mova_n_exp], DType::F32)
                    .map_err(|e| format!("k2_horizon: alloc v_expert_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&v_expert_ptrs.buf, &ve_ptrs)
                    .map_err(|e| format!("k2_horizon: htod v_expert_ptrs: {e:?}"))?;

                let attn_gate = load_wt(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    q_dim, // [num_attention_heads * head_dim, dim] = [4096, 2560]
                    hidden,
                )?;

                // MoE FFN: router [100, 2560], router_bias [100],
                // experts[100] (fused gate_up + down), shared expert.
                let router = load_wt(hfq, gpu, &format!("{p}.mlp.gate.weight"), n_exp, hidden)?;
                let router_bias = if cfg.moe_gate_bias {
                    Some(load_f32(hfq, gpu, &format!("{p}.mlp.gate.bias"), &[n_exp])?)
                } else {
                    None
                };

                // Pack MoE experts: fuse gate‖up per expert, then pack all
                // 100 experts into 2 GPU allocations (gate_up blob + down blob)
                // instead of 200 individual hipMalloc calls. Falls back to
                // per-expert loading if quant types are not packable.
                let (experts, gu_owner, dn_owner) = if !packed_experts_supported(gpu) {
                    // gfx12+ or unsupported arch: per-expert loading (see
                    // packed_experts_supported for the gfx1201 regression).
                    let mut experts = Vec::with_capacity(n_exp);
                    for e in 0..n_exp {
                        let ep = format!("{p}.mlp.experts.{e}");
                        let (qt_g, g) = read_tensor(hfq, &format!("{ep}.gate_proj.weight"))?;
                        let (qt_u, u) = read_tensor(hfq, &format!("{ep}.up_proj.weight"))?;
                        if qt_g != qt_u {
                            return Err(format!(
                                "k2_horizon L{l}E{e}: gate/up dtype mismatch ({qt_g:?} vs {qt_u:?}) — cannot byte-fuse gate_up"
                            ));
                        }
                        let mut gate_up_bytes = g;
                        gate_up_bytes.extend_from_slice(&u);
                        let gate_up = wt_from_raw(gpu, qt_g, &gate_up_bytes, 2 * moe_inter, hidden)
                            .map_err(|e2| format!("k2_horizon: fuse gate_up L{l}E{e}: {e2}"))?;
                        let down = load_wt(
                            hfq,
                            gpu,
                            &format!("{ep}.down_proj.weight"),
                            hidden,
                            moe_inter,
                        )?;
                        experts.push(MoeExpertWeights { gate_up, down });
                    }
                    (experts, None, None)
                } else {
                    // Build specs and check packability.
                    let mut specs = Vec::with_capacity(n_exp);
                    let mut gu_dtype: Option<DType> = None;
                    let mut gu_stride: Option<usize> = None;
                    let mut dn_stride: Option<usize> = None;
                    let mut packable = true;
                    for e in 0..n_exp {
                        let ep = format!("{p}.mlp.experts.{e}");
                        let (qt_g, g) = read_tensor(hfq, &format!("{ep}.gate_proj.weight"))?;
                        let (qt_u, u) = read_tensor(hfq, &format!("{ep}.up_proj.weight"))?;
                        if qt_g != qt_u {
                            return Err(format!(
                                "k2_horizon L{l}E{e}: gate/up dtype mismatch ({qt_g:?} vs {qt_u:?}) — cannot byte-fuse gate_up"
                            ));
                        }
                        let Some(dt) = packable_mq4_dtype(qt_g) else {
                            packable = false;
                            break;
                        };
                        let gu_bytes_len = g.len() + u.len();
                        match gu_dtype {
                            None => gu_dtype = Some(dt),
                            Some(existing) if existing == dt => {}
                            Some(_) => {
                                packable = false;
                                break;
                            }
                        }
                        match gu_stride {
                            None => gu_stride = Some(gu_bytes_len),
                            Some(s) if s == gu_bytes_len => {}
                            Some(_) => {
                                packable = false;
                                break;
                            }
                        }
                        let (qt_d, d) = read_tensor(hfq, &format!("{ep}.down_proj.weight"))?;
                        if packable_mq4_dtype(qt_d) != Some(dt) {
                            packable = false;
                            break;
                        }
                        match dn_stride {
                            None => dn_stride = Some(d.len()),
                            Some(s) if s == d.len() => {}
                            Some(_) => {
                                packable = false;
                                break;
                            }
                        }
                        specs.push((e, g, u, d));
                    }

                    if packable && !specs.is_empty() {
                        let dt = gu_dtype.unwrap();
                        let gu_s = gu_stride.unwrap();
                        let dn_s = dn_stride.unwrap();

                        // Concatenate all experts into two host blobs.
                        let mut gu_blob = Vec::with_capacity(gu_s * n_exp);
                        let mut dn_blob = Vec::with_capacity(dn_s * n_exp);
                        for (_e, g, u, d) in &specs {
                            gu_blob.extend_from_slice(g);
                            gu_blob.extend_from_slice(u);
                            dn_blob.extend_from_slice(d);
                        }

                        // Two GPU allocations for all 100 experts.
                        let gu_owner = gpu
                            .upload_raw(&gu_blob, &[gu_blob.len()])
                            .map_err(|e| format!("k2_horizon: packed gu upload: {e:?}"))?;
                        let dn_owner = gpu
                            .upload_raw(&dn_blob, &[dn_blob.len()])
                            .map_err(|e| format!("k2_horizon: packed dn upload: {e:?}"))?;

                        // Create views.
                        let mut experts = Vec::with_capacity(n_exp);
                        for slot in 0..n_exp {
                            let gu_view = gu_owner.sub_offset(slot * gu_s, gu_s);
                            let dn_view = dn_owner.sub_offset(slot * dn_s, dn_s);
                            experts.push(MoeExpertWeights {
                                gate_up: WeightTensor {
                                    buf: gu_view,
                                    gpu_dtype: dt,
                                    m: 2 * moe_inter,
                                    k: hidden,
                                    row_stride: 0,
                                    paro: None,
                                    awq_scale: None,
                                },
                                down: WeightTensor {
                                    buf: dn_view,
                                    gpu_dtype: dt,
                                    m: hidden,
                                    k: moe_inter,
                                    row_stride: 0,
                                    paro: None,
                                    awq_scale: None,
                                },
                            });
                        }
                        (experts, Some(gu_owner), Some(dn_owner))
                    } else {
                        // Fallback: per-expert loading.
                        let mut experts = Vec::with_capacity(n_exp);
                        for e in 0..n_exp {
                            let ep = format!("{p}.mlp.experts.{e}");
                            let (qt_g, g) = read_tensor(hfq, &format!("{ep}.gate_proj.weight"))?;
                            let (qt_u, u) = read_tensor(hfq, &format!("{ep}.up_proj.weight"))?;
                            if qt_g != qt_u {
                                return Err(format!(
                                    "k2_horizon L{l}E{e}: gate/up dtype mismatch ({qt_g:?} vs {qt_u:?}) — cannot byte-fuse gate_up"
                                ));
                            }
                            let mut gate_up_bytes = g;
                            gate_up_bytes.extend_from_slice(&u);
                            let gate_up =
                                wt_from_raw(gpu, qt_g, &gate_up_bytes, 2 * moe_inter, hidden)
                                    .map_err(|e2| {
                                        format!("k2_horizon: fuse gate_up L{l}E{e}: {e2}")
                                    })?;
                            let down = load_wt(
                                hfq,
                                gpu,
                                &format!("{ep}.down_proj.weight"),
                                hidden,
                                moe_inter,
                            )?;
                            experts.push(MoeExpertWeights { gate_up, down });
                        }
                        (experts, None, None)
                    }
                };

                // Same V2-only constraint as the MoVA v_experts — the indexed
                // gate_up/down GEMV kernels decode fp16 per-128 headers only.
                for (i, e) in experts.iter().enumerate() {
                    if e.gate_up.gpu_dtype != DType::MQ4G256V2
                        || e.down.gpu_dtype != DType::MQ4G256V2
                    {
                        return Err(format!(
                            "k2_horizon: {p}.mlp.experts.{i} has dtype {:?}/{:?} — indexed MoE GEMV requires MQ4G256V2 (qt=44); re-quantize with --format mq4",
                            e.gate_up.gpu_dtype, e.down.gpu_dtype
                        ));
                    }
                }

                // Device pointer tables for indexed MoE GEMV kernels.
                let gu_ptrs: Vec<u8> = experts
                    .iter()
                    .flat_map(|e| (e.gate_up.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let dn_ptrs: Vec<u8> = experts
                    .iter()
                    .flat_map(|e| (e.down.buf.buf.as_ptr() as u64).to_ne_bytes())
                    .collect();
                let expert_gate_up_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("k2_horizon: alloc gu_ptrs: {e:?}"))?;
                let expert_down_ptrs = gpu
                    .alloc_tensor(&[2 * n_exp], DType::F32)
                    .map_err(|e| format!("k2_horizon: alloc dn_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_gate_up_ptrs.buf, &gu_ptrs)
                    .map_err(|e| format!("k2_horizon: htod gu_ptrs: {e:?}"))?;
                gpu.hip
                    .memcpy_htod(&expert_down_ptrs.buf, &dn_ptrs)
                    .map_err(|e| format!("k2_horizon: htod dn_ptrs: {e:?}"))?;

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
                        v_router_bias,
                        v_experts,
                        v_experts_owner,
                        v_expert_ptrs,
                        wo,
                        attn_gate,
                    },
                    ffn_norm,
                    ffn: MoeFfnWeights {
                        router,
                        router_bias,
                        experts,
                        experts_gate_up_owner: gu_owner,
                        experts_down_owner: dn_owner,
                        expert_gate_up_ptrs,
                        expert_down_ptrs,
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
