// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision path: image bytes → SigLIP-style tower → merged 1024-d tokens.
//!
//! The tower itself is `hipfire-arch-qwen35-vl`'s (patch-embed GEMM,
//! interpolated learned pos-embed, 2D rope, ViT blocks, LayerNorm + 2×2
//! merge + GELU MLP merger) — the pplx tower is the same architecture at
//! hidden 768 / depth 12, and nothing in that code is size-pinned. This
//! module only owns the **loading** of `visual.*` tensors out of a
//! safetensors dir / raw `.hfq` (the vl crate's loader is HFQ- and
//! `model.visual.`-prefixed) and the request-side preprocessing +
//! `vision_forward` call.
//!
//! Known numerical deviation, inherited from the shared tower: the merger
//! GELU is the tanh approximation (`gelu_tanh_f32`) where HF
//! `Qwen3_5VisionPatchMerger` uses exact-erf `nn.GELU()`. Same tradeoff
//! production Qwen3.5-VL serving makes; ~1e-3 relative on merged tokens.

use crate::config::EmbedConfig;
use crate::names;
use half::f16;
use hipfire_arch_qwen35_vl::qwen35_vl::{
    vision_forward, VisionConfig, VisionLayerWeights, VisionWeights,
};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use rdna_compute::Gpu;

/// Vision tensors are kept raw by the quantizer (F16 under
/// `--include-vision`), so both the safetensors-dir (`dtype` string) and the
/// `.hfq` (`quant_type` 1/2) conventions resolve here.
fn vision_storage(
    info: &hipfire_runtime::model_source::TensorInfo,
    name: &str,
) -> Result<u8, String> {
    // 0 = f32, 1 = f16
    if info.quant_type != 0xFF {
        return match info.quant_type {
            1 => Ok(1),
            2 => Ok(0),
            other => Err(format!(
                "qwen35-embed: vision tensor `{name}` has HFQ quant_type {other} \
                 (raw f16/f32 only)"
            )),
        };
    }
    match info.dtype.as_str() {
        "F16" => Ok(1),
        "F32" => Ok(0),
        other => Err(format!(
            "qwen35-embed: vision tensor `{name}` dtype {other} (F16/F32 only)"
        )),
    }
}

fn to_f32_vec(width: u8, bytes: &[u8], n: usize) -> Vec<f32> {
    if width == 1 {
        bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    } else {
        bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn up_lin(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    name: &str,
) -> Result<rdna_compute::GpuTensor, String> {
    let (info, bytes) = names::resolve_vision(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing vision tensor `{name}`"))?;
    let n = info.shape.iter().product::<usize>();
    let words: Vec<u16> = to_f32_vec(vision_storage(info, name)?, bytes, n)
        .into_iter()
        .map(|v| f16::from_f32(v).to_bits())
        .collect();
    let rows = info.shape.first().copied().unwrap_or(0);
    // N-D → 2-D row-major flatten: the 5-D conv-style patch-embed weight
    // `[out, in, T, h, w]` becomes `[out, in·T·h·w]`, which is exactly the
    // GEMM form for the (C, T, h, w) channel-outer patch layout.
    let cols: usize = if info.shape.len() >= 2 {
        info.shape[1..].iter().product()
    } else {
        1
    };
    let t = gpu
        .upload_f16_bits(&words, &[rows, cols])
        .map_err(|e| format!("qwen35-embed: upload vision `{name}`: {e:?}"))?;
    src.release_tensor_pages(&names::vision_tensor_candidates(name)[0]);
    Ok(t)
}

fn up_vec(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    name: &str,
    n: usize,
) -> Result<rdna_compute::GpuTensor, String> {
    let (info, bytes) = names::resolve_vision(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing vision tensor `{name}`"))?;
    let total = info.shape.iter().product::<usize>();
    let v = to_f32_vec(vision_storage(info, name)?, bytes, total);
    if v.len() != n {
        return Err(format!(
            "qwen35-embed: vision `{name}` is {} elems, expected {n}",
            v.len()
        ));
    }
    let t = gpu
        .upload_f32(&v, &[n])
        .map_err(|e| format!("qwen35-embed: upload vision `{name}`: {e:?}"))?;
    src.release_tensor_pages(&names::vision_tensor_candidates(name)[0]);
    Ok(t)
}

fn host_vec(src: &dyn ModelSourceTrait, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let (info, bytes) = names::resolve_vision(src, name)
        .ok_or_else(|| format!("qwen35-embed: missing vision tensor `{name}`"))?;
    let total = info.shape.iter().product::<usize>();
    let v = to_f32_vec(vision_storage(info, name)?, bytes, total);
    if v.len() != n {
        return Err(format!(
            "qwen35-embed: vision `{name}` is {} elems, expected {n}",
            v.len()
        ));
    }
    src.release_tensor_pages(&names::vision_tensor_candidates(name)[0]);
    Ok(v)
}

/// Load the tower into the qwen35-vl weight set (see module docs for why
/// this isn't `qwen35_vl::load_vision_weights`).
pub fn load_vision_weights(
    gpu: &mut Gpu,
    src: &dyn ModelSourceTrait,
    cfg: &EmbedConfig,
) -> Result<VisionWeights, String> {
    let vis: &VisionConfig = cfg
        .vision
        .as_ref()
        .ok_or("qwen35-embed: vision weights present but no vision_config")?;
    let side = (vis.num_position_embeddings as f64).sqrt().round() as usize;
    if side == 0 || side * side != vis.num_position_embeddings {
        return Err(format!(
            "qwen35-embed: num_position_embeddings ({}) must be a perfect square",
            vis.num_position_embeddings
        ));
    }
    let h = vis.hidden_size;
    let mut layers = Vec::with_capacity(vis.num_layers);
    for i in 0..vis.num_layers {
        let p = format!("blocks.{i}");
        layers.push(VisionLayerWeights {
            norm1_w: up_vec(gpu, src, &format!("{p}.norm1.weight"), h)?,
            norm1_b: up_vec(gpu, src, &format!("{p}.norm1.bias"), h)?,
            qkv_w: up_lin(gpu, src, &format!("{p}.attn.qkv.weight"))?,
            qkv_b: up_vec(gpu, src, &format!("{p}.attn.qkv.bias"), 3 * h)?,
            proj_w: up_lin(gpu, src, &format!("{p}.attn.proj.weight"))?,
            proj_b: up_vec(gpu, src, &format!("{p}.attn.proj.bias"), h)?,
            norm2_w: up_vec(gpu, src, &format!("{p}.norm2.weight"), h)?,
            norm2_b: up_vec(gpu, src, &format!("{p}.norm2.bias"), h)?,
            fc1_w: up_lin(gpu, src, &format!("{p}.mlp.linear_fc1.weight"))?,
            fc1_b: up_vec(gpu, src, &format!("{p}.mlp.linear_fc1.bias"), vis.mlp_dim)?,
            fc2_w: up_lin(gpu, src, &format!("{p}.mlp.linear_fc2.weight"))?,
            fc2_b: up_vec(gpu, src, &format!("{p}.mlp.linear_fc2.bias"), h)?,
        });
    }
    let merge_dim = h * vis.spatial_merge_size * vis.spatial_merge_size;
    Ok(VisionWeights {
        patch_embed_w: up_lin(gpu, src, "patch_embed.proj.weight")?,
        patch_embed_b: up_vec(gpu, src, "patch_embed.proj.bias", h)?,
        pos_embed: host_vec(src, "pos_embed.weight", vis.num_position_embeddings * h)?,
        layers,
        merger_norm_w: up_vec(gpu, src, "merger.norm.weight", h)?,
        merger_norm_b: up_vec(gpu, src, "merger.norm.bias", h)?,
        merger_fc1_w: up_lin(gpu, src, "merger.linear_fc1.weight")?,
        merger_fc1_b: up_vec(gpu, src, "merger.linear_fc1.bias", merge_dim)?,
        merger_fc2_w: up_lin(gpu, src, "merger.linear_fc2.weight")?,
        merger_fc2_b: up_vec(gpu, src, "merger.linear_fc2.bias", vis.out_hidden_size)?,
    })
}

/// One merged image token per `spatial_merge_size²` patch group.
pub fn merged_grid(cfg: &EmbedConfig, grid_h: usize, grid_w: usize) -> (usize, usize) {
    let m = cfg
        .vision
        .as_ref()
        .map(|v| v.spatial_merge_size)
        .unwrap_or(2);
    (grid_h / m, grid_w / m)
}

/// Decode + preprocess + encode an image to merged `[n_merged,
/// out_hidden]` host embeds. `patch_size`/`merge` come from the vision
/// config; preprocessing (smart-resize, `/127.5 − 1` normalize, 2×2-grouped
/// patch extraction) is `hipfire-arch-qwen35-vl::image`.
pub fn encode_image(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &EmbedConfig,
    image_bytes: &[u8],
) -> Result<(Vec<f32>, usize, usize), String> {
    let vis = cfg
        .vision
        .as_ref()
        .ok_or("qwen35-embed: image input on a text-only checkpoint")?;
    // `load_and_preprocess_from_bytes` returns (CHW, **pixel** h, pixel w) —
    // the pixel grid is a multiple of `patch_size * spatial_merge_size`. The
    // tower consumes patch vectors + a PATCH grid, so run `extract_patches`
    // and divide by `patch_size` (the qwen35-vl serve path does the same:
    // print/`hipfire-generate/src/vision.rs` computes grid = img/patch_size).
    // Passing the pixel dims as the grid was the first-cut bug here: a 64×64
    // input became a 256×256 grid (65,536 patches) and OOM'd.
    let (pixels, img_h, img_w) = hipfire_arch_qwen35_vl::image::load_and_preprocess_from_bytes(
        image_bytes,
        vis.patch_size,
        vis.spatial_merge_size,
    )?;
    let patches = hipfire_arch_qwen35_vl::image::extract_patches(
        &pixels,
        3,
        img_h,
        img_w,
        vis.patch_size,
        vis.temporal_patch_size,
        vis.spatial_merge_size,
    );
    let grid_h = img_h / vis.patch_size;
    let grid_w = img_w / vis.patch_size;
    if std::env::var("HIPFIRE_EMBED_DEBUG").ok().as_deref() == Some("1") {
        let (f, t) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        eprintln!(
            "  [embed-debug] vision: {}x{} px -> {}x{} grid ({} patches), weights loaded; VRAM {} MB free / {} MB",
            img_w, img_h, grid_w, grid_h, grid_h * grid_w, f / (1024*1024), t / (1024*1024)
        );
    }
    let merged = vision_forward(gpu, weights, vis, &patches, grid_h, grid_w)
        .map_err(|e| format!("qwen35-embed: vision forward: {e:?}"))?;
    let (mh, mw) = merged_grid(cfg, grid_h, grid_w);
    let n_merged = mh * mw;
    if merged.len() != n_merged * vis.out_hidden_size {
        return Err(format!(
            "qwen35-embed: vision tower returned {} floats, expected {n_merged}×{}",
            merged.len(),
            vis.out_hidden_size
        ));
    }
    Ok((merged, mh, mw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{config_from_metadata, parse_config};

    fn pplx_cfg() -> crate::config::EmbedConfig {
        // The composite shape both source kinds produce (config under a
        // `config` key) — the same string the config module tests.
        let v: serde_json::Value = serde_json::from_str(
            r#"{"config":{"architectures":["Qwen3_5Model"],"model_type":"qwen3_5",
                 "text_config":{"hidden_size":1024,"num_hidden_layers":2,
                   "num_attention_heads":8,"num_key_value_heads":2,"head_dim":256,
                   "intermediate_size":3584,"vocab_size":10,"rms_norm_eps":1e-6,
                   "hidden_act":"silu","attn_output_gate":true,
                   "linear_num_key_heads":16,"linear_num_value_heads":16,
                   "linear_key_head_dim":128,"linear_value_head_dim":128,
                   "linear_conv_kernel_dim":4,
                   "rope_parameters":{"rope_theta":1e7,"partial_rotary_factor":0.25,
                                      "mrope_section":[11,11,10]},
                   "layer_types":["linear_attention","full_attention"]},
                 "vision_config":{"hidden_size":768,"num_heads":12,"depth":2,
                   "intermediate_size":3072,"out_hidden_size":1024,"patch_size":16,
                   "temporal_patch_size":2,"spatial_merge_size":2,
                   "num_position_embeddings":2304}}}"#,
        )
        .unwrap();
        parse_config(config_from_metadata(&v)).unwrap()
    }

    #[test]
    fn merged_grid_divides_by_spatial_merge_size() {
        let cfg = pplx_cfg();
        // A 16x16 patch grid merges 2x2 -> 8x8 = 64 tokens (the value the
        // gfx1101 tower smoke produced for a 256x256-pixel input).
        assert_eq!(merged_grid(&cfg, 16, 16), (8, 8));
        assert_eq!(merged_grid(&cfg, 1, 1), (0, 0)); // partial group drops
        assert_eq!(merged_grid(&cfg, 32, 48), (16, 24));
    }
}
