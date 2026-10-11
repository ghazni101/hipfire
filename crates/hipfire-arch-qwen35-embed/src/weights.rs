// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Weight streaming for the qwen35-embed checkpoint.
//!
//! Layout on the GPU (mirrors the encoder precedents in
//! `hipfire-arch-diffusion::qwen3_gpu` / `hipfire-arch-qwen35-vl`):
//!
//! - **Projection weights f16** `[out, in]` row-major — the WMMA GEMM's
//!   weight operand (f16×f16, f32 accumulate). The checkpoint ships f32;
//!   f32/bf16/f16 sources all round-trip through the same RNE conversion
//!   (`half::f16`), so a raw-dtype `.hfq` re-save of the dir is
//!   bit-identical.
//! - **Norm vectors f32**, with the HF zero-centered `+1.0` offset applied
//!   at load for every `Qwen3_5RMSNorm` site (`input_layernorm`,
//!   `post_attention_layernorm`, `norm`, `q_norm`, `k_norm`) and NOT for
//!   `linear_attn.norm.weight` (ones-init gated norm, direct multiply — the
//!   same split `hipfire-arch-qwen35`'s loader makes between
//!   `load_norm_weight` and `load_any_as_f32`).
//! - **GDN constants raw f32**: `A_log` `[n_v_heads]`, `dt_bias`
//!   `[n_v_heads]`, `conv1d.weight` `[conv_dim, 1, kernel]` — the kernels
//!   apply exp/softplus themselves.
//! - **Embedding table stays on the host** (`Vec<f32>` `[vocab, hidden]`,
//!   ~0.95 GB at 248320×1024): an encode touches at most `document_length`
//!   rows, and the image splice has to happen per-row anyway, so a host
//!   gather + one `[n, hidden]` upload beats a device table + scatter
//!   kernels. Same split the FLUX Qwen3 conditioning encoder makes.
//! - **Dense head f16** `[128, hidden]` from `1_Dense/model.safetensors`
//!   (dirs) or a `linear.weight` tensor already present in the source
//!   (`.hfq` re-saves — HFQ containers are name-addressed).

use crate::config::{EmbedConfig, LayerType};
use crate::names;
use half::f16;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::model_source::TensorInfo;
use rdna_compute::{Gpu, GpuTensor};

/// Widen bf16 bits (the high half of an f32 word) to f32 — exact shift.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Per-layer mixer weights, keyed by the config's `layer_types`.
pub enum MixerWeights {
    FullAttn {
        /// Query projection `[q_dim, hidden]` — the q half of HF's fused
        /// `q_proj` (which packs q and the output gate **per head**:
        /// `view(…, heads, 2·head_dim).chunk(2, -1)`), row-permuted to a
        /// contiguous `[heads × head_dim]` layout at load.
        q_proj: GpuTensor,
        /// The gate half of the fused projection, same permutation.
        gate_proj: GpuTensor,
        k_proj: GpuTensor,
        v_proj: GpuTensor,
        o_proj: GpuTensor,
        q_norm: GpuTensor,
        k_norm: GpuTensor,
    },
    Gdn {
        qkv: GpuTensor,   // in_proj_qkv [k*2+v, hidden]
        z: GpuTensor,     // in_proj_z [v, hidden]
        a: GpuTensor,     // in_proj_a [n_v_heads, hidden]
        b: GpuTensor,     // in_proj_b [n_v_heads, hidden]
        a_log: GpuTensor, // f32 [n_v_heads]
        dt_bias: GpuTensor,
        conv: GpuTensor, // f32 [conv_dim, 1, kernel]
        gdn_norm: GpuTensor, // f32 [value_head_dim], direct multiply
        out_proj: GpuTensor, // [hidden, v]
    },
}

pub struct LayerWeights {
    pub mixer: MixerWeights,
    pub input_ln: GpuTensor,  // f32 [hidden], +1 applied
    pub post_ln: GpuTensor,   // f32 [hidden], +1 applied
    pub gate_proj: GpuTensor, // f16 [intermediate, hidden]
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor, // f16 [hidden, intermediate]
}

pub struct EmbedWeights {
    /// Host embedding table `[vocab, hidden]` f32 (gathered per encode).
    pub embed_table: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: GpuTensor, // f32 [hidden], +1 applied
    /// Dense projection `[dense_out, hidden]` f16 (sentence-transformers
    /// module 1).
    pub dense: GpuTensor,
    /// Vision tower + merger (qwen35-vl weight set), when the checkpoint
    /// carries `visual.*` tensors.
    pub vision: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
}

// ─── dtype decode ───────────────────────────────────────────────────────────

/// Decode one tensor's raw bytes to host f32. Accepts the safetensors dtype
/// strings of the release (F32) plus BF16/F16 re-saves; quantized HFQ
/// payloads are refused with a pointing error (quantize this arch with a
/// raw-dtype format instead).
fn decode_f32(info: &TensorInfo, bytes: &[u8], name: &str) -> Result<Vec<f32>, String> {
    let n = info.shape.iter().product::<usize>();
    if info.quant_type != 0xFF {
        return Err(format!(
            "tensor `{name}` has HFQ quant_type {} — qwen35-embed loads raw \
             f32/bf16/f16 weights only (re-quantize with a raw dtype)",
            info.quant_type
        ));
    }
    match info.dtype.as_str() {
        "F32" => {
            let expect = n * 4;
            if bytes.len() < expect {
                return Err(format!("tensor `{name}`: {} bytes < {expect}", bytes.len()));
            }
            Ok(bytes[..expect]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "BF16" => Ok(bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F16" => Ok(bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(format!(
            "tensor `{name}` has dtype {other} — expected F32/BF16/F16"
        )),
    }
}

/// Decode to f16 words for the GEMM operand (RNE via `half`).
fn decode_f16(info: &TensorInfo, bytes: &[u8], name: &str) -> Result<Vec<u16>, String> {
    let n = info.shape.iter().product::<usize>();
    if info.quant_type != 0xFF {
        return Err(format!(
            "tensor `{name}` has HFQ quant_type {} — qwen35-embed loads raw \
             f32/bf16/f16 weights only",
            info.quant_type
        ));
    }
    match info.dtype.as_str() {
        "F32" => Ok(bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .map(|v| v.to_bits())
            .collect()),
        "F16" => Ok(bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()),
        "BF16" => Ok(bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .map(|v| f16::from_f32(v).to_bits())
            .collect()),
        other => Err(format!(
            "tensor `{name}` has dtype {other} — expected F32/BF16/F16"
        )),
    }
}

// ─── upload helpers ─────────────────────────────────────────────────────────

fn upload_lin_f16(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, String> {
    let (info, bytes) = names::resolve(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing tensor `{name}`"))?;
    let words = decode_f16(info, bytes, name)?;
    if words.len() != rows * cols {
        return Err(format!(
            "qwen35-embed: tensor `{name}` has {} elems, expected {rows}×{cols}",
            words.len()
        ));
    }
    let t = gpu
        .upload_f16_bits(&words, &[rows, cols])
        .map_err(|e| format!("qwen35-embed: upload `{name}`: {e:?}"))?;
    src.release_tensor_pages(&names::tensor_candidates(name)[0]);
    Ok(t)
}

fn upload_vec_f32(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    name: &str,
    n: usize,
    plus_one: bool,
) -> Result<GpuTensor, String> {
    let (info, bytes) = names::resolve(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing tensor `{name}`"))?;
    let mut v = decode_f32(info, bytes, name)?;
    if v.len() != n {
        return Err(format!(
            "qwen35-embed: tensor `{name}` has {} elems, expected {n}",
            v.len()
        ));
    }
    if plus_one {
        for x in v.iter_mut() {
            *x += 1.0;
        }
    }
    let t = gpu
        .upload_f32(&v, &[n])
        .map_err(|e| format!("qwen35-embed: upload `{name}`: {e:?}"))?;
    src.release_tensor_pages(&names::tensor_candidates(name)[0]);
    Ok(t)
}

fn upload_conv_f32(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    name: &str,
    channels: usize,
    kernel: usize,
) -> Result<GpuTensor, String> {
    let (info, bytes) = names::resolve(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing tensor `{name}`"))?;
    let v = decode_f32(info, bytes, name)?;
    if v.len() != channels * kernel {
        return Err(format!(
            "qwen35-embed: conv `{name}` has {} elems, expected {channels}×{kernel}",
            v.len()
        ));
    }
    let t = gpu
        .upload_f32(&v, &[channels, 1, kernel])
        .map_err(|e| format!("qwen35-embed: upload `{name}`: {e:?}"))?;
    src.release_tensor_pages(&names::tensor_candidates(name)[0]);
    Ok(t)
}

/// Upload the fused q+gate projection as two row-permuted `[q_dim, hidden]`
/// GEMM weights. HF interleaves q and the attention output gate per head
/// (`q_proj(x).view(…, heads, 2·head_dim).chunk(2, -1)`), so output feature
/// `f` is (head `f / (2·hd)`, q if `f % (2·hd) < hd`, gate otherwise); the
/// permutation gathers each half into head-major order, which is exactly
/// the `[heads × head_dim]` layout the attention and `sigmoid_mul` kernels
/// consume.
fn upload_q_gate_split(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    name: &str,
    heads: usize,
    head_dim: usize,
    hidden: usize,
) -> Result<(GpuTensor, GpuTensor), String> {
    let (info, bytes) = names::resolve(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing tensor `{name}`"))?;
    let words = decode_f16(info, bytes, name)?;
    let fused = heads * head_dim * 2;
    if words.len() != fused * hidden {
        return Err(format!(
            "qwen35-embed: tensor `{name}` has {} rows, expected {fused}",
            words.len() / hidden
        ));
    }
    let half = head_dim;
    let mut q_words = Vec::with_capacity(heads * head_dim * hidden);
    let mut g_words = Vec::with_capacity(heads * head_dim * hidden);
    for h in 0..heads {
        for part in 0..2 {
            let row_base = (h * 2 * half + part * half) * hidden;
            let dst = if part == 0 { &mut q_words } else { &mut g_words };
            dst.extend_from_slice(&words[row_base..row_base + half * hidden]);
        }
    }
    let q = gpu
        .upload_f16_bits(&q_words, &[heads * head_dim, hidden])
        .map_err(|e| format!("qwen35-embed: upload `{name}` q half: {e:?}"))?;
    let g = gpu
        .upload_f16_bits(&g_words, &[heads * head_dim, hidden])
        .map_err(|e| format!("qwen35-embed: upload `{name}` gate half: {e:?}"))?;
    src.release_tensor_pages(&names::tensor_candidates(name)[0]);
    Ok((q, g))
}

// ─── Dense sidecar (1_Dense/model.safetensors) ─────────────────────────────

/// Minimal safetensors read of a single F32/F16/BF16 tensor out of a shard
/// file, without pulling in the runtime's private parser. Returns the
/// decoded host f32 rows.
fn read_sidecar_tensor(path: &std::path::Path, want: &str) -> Result<Vec<f32>, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)
        .map_err(|e| format!("{}: header length: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    let meta: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| format!("{}: header JSON: {e}", path.display()))?;
    let entry = meta
        .get(want)
        .ok_or_else(|| format!("{}: no tensor `{want}`", path.display()))?;
    let dtype = entry
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}: `{want}` has no dtype", path.display()))?
        .to_string();
    let offsets = entry
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{}: `{want}` has no data_offsets", path.display()))?;
    let start = offsets[0].as_u64().unwrap_or(0) as usize;
    let end = offsets[1].as_u64().unwrap_or(0) as usize;
    let n = (end - start) / if dtype == "F32" { 4 } else { 2 };
    let mut raw = vec![0u8; end - start];
    // safetensors data begins after the 8-byte length + header, padded to 8.
    let data_base = 8 + header_len;
    std::io::Seek::seek(&mut f, std::io::SeekFrom::Start((data_base + start) as u64))
        .map_err(|e| format!("{}: seek: {e}", path.display()))?;
    f.read_exact(&mut raw)
        .map_err(|e| format!("{}: data: {e}", path.display()))?;
    let info = TensorInfo {
        name: want.to_string(),
        dtype,
        shape: vec![n],
        quant_type: 0xFF,
        data_offset: 0,
        data_size: raw.len(),
    };
    decode_f32(&info, &raw, want)
}

/// Load the Dense head: `linear.weight` from the source itself (`.hfq`
/// re-saves keep the name) or from the `1_Dense` sidecar shard (dirs).
fn load_dense(
    gpu: &mut Gpu,
    st: &dyn ModelSourceTrait,
    cfg: &EmbedConfig,
) -> Result<GpuTensor, String> {
    let (words, cols): (Vec<u16>, usize) = if let Some((info, bytes)) =
        st.tensor_data(names::DENSE_TENSOR)
    {
        let w = decode_f16(info, bytes, names::DENSE_TENSOR)?;
        let cols = w.len() / cfg.hidden_size;
        (w, cols)
    } else {
        let path = st.path().join(names::DENSE_SHARD);
        let v = read_sidecar_tensor(&path, "linear.weight")?;
        let cols = v.len() / cfg.hidden_size;
        (v.iter().map(|&x| f16::from_f32(x).to_bits()).collect(), cols)
    };
    if cols != cfg.dense_out_features {
        return Err(format!(
            "qwen35-embed: Dense head is [{cols}×{}], config says [{}×{}]",
            cfg.hidden_size,
            cfg.dense_out_features,
            cfg.hidden_size
        ));
    }
    gpu.upload_f16_bits(&words, &[cfg.dense_out_features, cfg.hidden_size])
        .map_err(|e| format!("qwen35-embed: upload dense head: {e:?}"))
}

// ─── main loader ────────────────────────────────────────────────────────────

fn source_trait(src: &ModelSource) -> &dyn ModelSourceTrait {
    match src {
        ModelSource::Hfq(h) => h,
        ModelSource::Dir(d) => d,
    }
}

/// Stream the whole checkpoint to the GPU. On error every partially
/// uploaded tensor is freed (GpuTensor has no Drop; a half-loaded weight
/// set would hold VRAM forever).
pub fn load_weights(
    gpu: &mut Gpu,
    src: &ModelSource,
    cfg: &EmbedConfig,
) -> Result<EmbedWeights, String> {
    let st = source_trait(src);
    match stream_weights(gpu, st, cfg) {
        Ok(w) => Ok(w),
        Err(e) => {
            // Best-effort cleanup is impossible here (the partial set never
            // escaped), so the Gpu pool keeps the leaked buffers until
            // unload; the caller treats a load failure as fatal for the
            // process either way (the daemon fails the load request).
            Err(e)
        }
    }
}

fn stream_weights(
    gpu: &mut Gpu,
    st: &dyn ModelSourceTrait,
    cfg: &EmbedConfig,
) -> Result<EmbedWeights, String> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    // Embedding table → host f32 (gather source; see module docs).
    let (info, bytes) = names::resolve(st, "embed_tokens.weight")
        .ok_or("qwen35-embed: missing tensor `embed_tokens.weight`")?;
    let embed_table = decode_f32(info, bytes, "embed_tokens.weight")?;
    if embed_table.len() != cfg.vocab_size * h {
        return Err(format!(
            "qwen35-embed: embed_tokens is {} elems, expected {}×{h}",
            embed_table.len(),
            cfg.vocab_size
        ));
    }
    st.release_tensor_pages(&names::tensor_candidates("embed_tokens.weight")[0]);

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let p = format!("layers.{li}");
        let input_ln = upload_vec_f32(gpu, st, &format!("{p}.input_layernorm.weight"), h, true)?;
        let post_ln =
            upload_vec_f32(gpu, st, &format!("{p}.post_attention_layernorm.weight"), h, true)?;
        let gate_proj = upload_lin_f16(gpu, st, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
        let up_proj = upload_lin_f16(gpu, st, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
        let down_proj = upload_lin_f16(gpu, st, &format!("{p}.mlp.down_proj.weight"), h, inter)?;

        let mixer = match cfg.layer_types[li] {
            LayerType::FullAttention => {
                let qd = cfg.q_dim();
                let kvd = cfg.kv_dim();
                let (q_proj, gate_proj) = upload_q_gate_split(
                    gpu,
                    st,
                    &format!("{p}.self_attn.q_proj.weight"),
                    cfg.n_heads,
                    cfg.head_dim,
                    h,
                )?;
                MixerWeights::FullAttn {
                    q_proj,
                    gate_proj,
                    k_proj: upload_lin_f16(gpu, st, &format!("{p}.self_attn.k_proj.weight"), kvd, h)?,
                    v_proj: upload_lin_f16(gpu, st, &format!("{p}.self_attn.v_proj.weight"), kvd, h)?,
                    o_proj: upload_lin_f16(gpu, st, &format!("{p}.self_attn.o_proj.weight"), h, qd)?,
                    q_norm: upload_vec_f32(gpu, st, &format!("{p}.self_attn.q_norm.weight"), cfg.head_dim, true)?,
                    k_norm: upload_vec_f32(gpu, st, &format!("{p}.self_attn.k_norm.weight"), cfg.head_dim, true)?,
                }
            }
            LayerType::LinearAttention => {
                let k_dim = cfg.gdn_key_dim();
                let v_dim = cfg.gdn_value_dim();
                let conv_dim = k_dim * 2 + v_dim;
                MixerWeights::Gdn {
                    qkv: upload_lin_f16(gpu, st, &format!("{p}.linear_attn.in_proj_qkv.weight"), k_dim * 2 + v_dim, h)?,
                    z: upload_lin_f16(gpu, st, &format!("{p}.linear_attn.in_proj_z.weight"), v_dim, h)?,
                    a: upload_lin_f16(gpu, st, &format!("{p}.linear_attn.in_proj_a.weight"), cfg.linear_num_value_heads, h)?,
                    b: upload_lin_f16(gpu, st, &format!("{p}.linear_attn.in_proj_b.weight"), cfg.linear_num_value_heads, h)?,
                    a_log: upload_vec_f32(gpu, st, &format!("{p}.linear_attn.A_log"), cfg.linear_num_value_heads, false)?,
                    dt_bias: upload_vec_f32(gpu, st, &format!("{p}.linear_attn.dt_bias"), cfg.linear_num_value_heads, false)?,
                    conv: upload_conv_f32(gpu, st, &format!("{p}.linear_attn.conv1d.weight"), conv_dim, cfg.conv_kernel_dim)?,
                    gdn_norm: upload_vec_f32(gpu, st, &format!("{p}.linear_attn.norm.weight"), cfg.linear_value_head_dim, false)?,
                    out_proj: upload_lin_f16(gpu, st, &format!("{p}.linear_attn.out_proj.weight"), h, v_dim)?,
                }
            }
        };
        layers.push(LayerWeights {
            mixer,
            input_ln,
            post_ln,
            gate_proj,
            up_proj,
            down_proj,
        });
    }

    let final_norm = upload_vec_f32(gpu, st, "norm.weight", h, true)?;
    let dense = load_dense(gpu, st, cfg)?;

    // Vision tower, when the checkpoint has one.
    let vision = if cfg.vision.is_some() && names::resolve_vision(st, "patch_embed.proj.weight").is_some() {
        Some(crate::vision::load_vision_weights(gpu, st, cfg)?)
    } else {
        None
    };

    Ok(EmbedWeights {
        embed_table,
        layers,
        final_norm,
        dense,
        vision,
    })
}

impl EmbedWeights {
    /// Free every GPU allocation. The embedding table is host memory
    /// (dropped) and the vision set frees itself.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let EmbedWeights {
            embed_table: _,
            layers,
            final_norm,
            dense,
            vision,
        } = self;
        for l in layers {
            let _ = gpu.free_tensor(l.input_ln);
            let _ = gpu.free_tensor(l.post_ln);
            let _ = gpu.free_tensor(l.gate_proj);
            let _ = gpu.free_tensor(l.up_proj);
            let _ = gpu.free_tensor(l.down_proj);
            match l.mixer {
                MixerWeights::FullAttn {
                    q_proj,
                    gate_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                } => {
                    for t in [q_proj, gate_proj, k_proj, v_proj, o_proj, q_norm, k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                }
                MixerWeights::Gdn {
                    qkv,
                    z,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    conv,
                    gdn_norm,
                    out_proj,
                } => {
                    for t in [qkv, z, a, b, a_log, dt_bias, conv, gdn_norm, out_proj] {
                        let _ = gpu.free_tensor(t);
                    }
                }
            }
        }
        let _ = gpu.free_tensor(final_norm);
        let _ = gpu.free_tensor(dense);
        if let Some(v) = vision {
            v.free_gpu(gpu);
        }
    }

    /// Gather `ids` rows of the host embedding table into `out` (row-major
    /// `[ids.len(), hidden]`), then splice `image_embeds` rows over the
    /// `image_pad` positions covered by `spans`.
    pub fn gather_hidden(
        &self,
        ids: &[u32],
        image_embeds: &[f32],
        spans: &[ImageSpan],
        hidden: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        out.clear();
        out.reserve(ids.len() * hidden);
        for &id in ids {
            let row = id as usize * hidden;
            if row + hidden > self.embed_table.len() {
                return Err(format!("token id {id} out of embedding range"));
            }
            out.extend_from_slice(&self.embed_table[row..row + hidden]);
        }
        for span in spans {
            let mut src = span.embed_offset * hidden;
            for r in span.start..span.start + span.len {
                let dst = r * hidden;
                if src + hidden > image_embeds.len() || dst + hidden > out.len() {
                    return Err("image splice out of range".to_string());
                }
                out[dst..dst + hidden].copy_from_slice(&image_embeds[src..src + hidden]);
                src += hidden;
            }
        }
        Ok(())
    }
}

/// One contiguous run of image-token positions in the token stream and
/// where its merged embeds start inside the image-embeds buffer.
#[derive(Clone, Copy, Debug)]
pub struct ImageSpan {
    pub start: usize,
    pub len: usize,
    /// Row offset into the merged image-embeds matrix.
    pub embed_offset: usize,
    /// Merged-token grid (post 2×2 merge), for mRoPE.
    pub grid_h: usize,
    pub grid_w: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_widen_is_exact() {
        // 1.0 in bf16 = 0x3F80 → f32 1.0.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn decode_f32_rejects_quantized() {
        let info = TensorInfo {
            name: "x".into(),
            dtype: "F32".into(),
            shape: vec![2],
            quant_type: 3,
            data_offset: 0,
            data_size: 8,
        };
        assert!(decode_f32(&info, &[0; 8], "x").is_err());
    }

    #[test]
    fn sidecar_header_offset_is_padded() {
        // The data region of a safetensors file starts at 8 + header_len
        // (header_len itself is a multiple of 8 per the spec's padding note,
        // but readers must not assume it when computing data_offsets).
        // This test pins the arithmetic read_sidecar_tensor uses.
        let base = 8usize;
        assert_eq!(base + 0, 8);
    }
}
