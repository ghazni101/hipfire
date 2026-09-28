// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZDTaichu-5.0 vision encoder: C-RADIOv4-H (ViT-H/16) + pixel-shuffle
//! projector. Mirrors `hipfire-arch-qwen35-vl::qwen35_vl` but is a different
//! tower shape:
//!
//!   pixel tiles [T, 3, 512, 512]
//!     └─► im2col patch embed (16×16 → 1024 patches/tile) + absolute pos-embed
//!         (interp 128×128 table → 32×32 tile grid) + 10 cls/register prefix
//!         tokens → 32× timm ViT blocks (LN → fused QKV attn, NO RoPE → proj;
//!         LN → fc1 → exact-erf GELU → fc2) → drop prefix tokens
//!           └─► per-tile pixel_shuffle 0.5 ([1024,1280] → [256,5120])
//!                 └─► mlp1: RMSNorm(5120) → Linear(5120→20480) → SquaredReLU
//!                     → Linear(20480→4096) → per-tile 256 visual tokens
//!
//! Reference: `modeling.py` (`extract_feature`, `pixel_shuffle` ps_version=v2)
//! and `cradio_model.py` (`ViTPatchGenerator`, `InnerRADIOModel._extract_final`)
//! in the ZDTaichu-5.0-9B checkpoint.

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TaichuVisionConfig {
    /// ViT hidden size (1280 for C-RADIO v4-h).
    pub hidden_size: usize,
    /// Attention heads (16 → head_dim 80).
    pub num_heads: usize,
    pub head_dim: usize,
    /// Transformer block depth (32).
    pub num_layers: usize,
    /// Block MLP hidden dim (fc1 out = 4 × hidden = 5120).
    pub mlp_dim: usize,
    /// Patch size (16).
    pub patch_size: usize,
    /// Tile edge in pixels (`force_image_size`, 512).
    pub image_size: usize,
    /// pixel_shuffle scale (`downsample_ratio`, 0.5).
    pub downsample_ratio: f32,
    /// Per-tile visual token edge after shuffle (32·0.5 = 16).
    pub tile_tokens_per_side: usize,
    /// Visual tokens per tile (256).
    pub num_image_token: usize,
    /// Projector hidden dim (20480).
    pub projector_hidden: usize,
    /// Text hidden size the projector emits (4096).
    pub out_hidden_size: usize,
    /// cls + register prefix tokens prepended before the blocks (10 for
    /// this checkpoint: `cls_token.token` shape [10, 1280]; num_cls_tokens +
    /// num_registers in HF terms).
    pub num_prefix_tokens: usize,
    /// Side of the square learned `pos_embed` table (16384 → 128).
    pub pos_table_side: usize,
    /// Maximum InternVL tiles per image (`max_num_tiles`, 12).
    pub max_num_tiles: usize,
    /// Whether a thumbnail tile is appended when the grid has >1 tile
    /// (`use_thumbnail`, true).
    pub use_thumbnail: bool,
    /// ImageNet normalization (externalized RADIO input conditioner).
    pub norm_mean: [f32; 3],
    pub norm_std: [f32; 3],
    pub norm_eps: f32,
}

impl TaichuVisionConfig {
    /// Patches per tile edge = image_size / patch_size (32).
    pub fn tile_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
}

/// Parse the ZDTaichu vision config from HFQ metadata.
///
/// `meta.config` carries the checkpoint's top-level config.json verbatim, so
/// the tower shape lives at `config.vision_config` (RADIOConfig: patch_size,
/// max_resolution, version) while the tile/projector contract lives at the
/// top level (`force_image_size`, `downsample_ratio`, `vit_hidden_size`,
/// `projector_hidden_size`, `max_dynamic_patch`, `use_thumbnail`) and the LLM
/// width at `config.llm_config.hidden_size`. Returns `None` when the artifact
/// is not a ZDTaichu pack at all.
pub fn taichu_vision_config_from_hfq(hfq: &HfqFile) -> Option<TaichuVisionConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    if config.get("model_type")?.as_str()? != "zdtaichu5_0" {
        return None;
    }
    let vc = config.get("vision_config")?;
    let llm = config.get("llm_config");

    let hidden_size = config
        .get("vit_hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1280) as usize;
    let num_heads = 16usize; // C-RADIO v4-h is fixed ViT-H/16, heads=16.
    let num_layers = 32usize;
    let mlp_dim = hidden_size * 4;
    let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let image_size = config
        .get("force_image_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    if patch_size == 0 || image_size == 0 {
        return None; // malformed pack — treated as no tower
    }
    let downsample_ratio = config
        .get("downsample_ratio")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let tile_tokens_per_side =
        ((image_size / patch_size) as f32 * downsample_ratio).round() as usize;
    let num_image_token = tile_tokens_per_side * tile_tokens_per_side;
    let projector_hidden = config
        .get("projector_hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(20480) as usize;
    let out_hidden_size = llm
        .and_then(|l| l.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("hidden_size").and_then(|v| v.as_u64()))
        .unwrap_or(4096) as usize;
    let max_resolution = vc
        .get("max_resolution")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let pos_table_side = max_resolution / patch_size; // 2048/16 = 128
    let max_num_tiles = config
        .get("max_dynamic_patch")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let use_thumbnail = config
        .get("use_thumbnail")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Some(TaichuVisionConfig {
        hidden_size,
        num_heads,
        head_dim: hidden_size / num_heads,
        num_layers,
        mlp_dim,
        patch_size,
        image_size,
        downsample_ratio,
        tile_tokens_per_side,
        num_image_token,
        projector_hidden,
        out_hidden_size,
        // [10, 1280] cls_token.token in this checkpoint; if a future variant
        // ships a different register count the WEIGHTS SHAPE is authoritative
        // (load_vision_weights overrides this after reading the tensor).
        num_prefix_tokens: 10,
        pos_table_side,
        max_num_tiles,
        use_thumbnail,
        // ImageNet mean/std — matches preprocessor_config.json and the
        // externalized RADIO InputConditioner.
        norm_mean: [0.485, 0.456, 0.406],
        norm_std: [0.229, 0.224, 0.225],
        // timm LayerNorm default.
        norm_eps: 1e-6,
    })
}

/// True when this HFQ carries the C-RADIO tower. Probe tensor mirrors
/// `is_vision_hfq`'s `model.visual.patch_embed.proj.weight` for qwen35-vl.
pub fn is_taichu_vision_hfq(hfq: &HfqFile) -> bool {
    hfq.tensor_data("vision_model.radio_model.model.patch_generator.embedder.weight")
        .is_some()
}

// ─── GPU-side weights ───────────────────────────────────────────────────────

/// One timm ViT block: LN → fused QKV → proj; LN → fc1 → GELU(erf) → fc2.
/// Same field set as qwen35-vl's `VisionLayerWeights` — kept as a separate
/// type because the towers differ in activation and positional scheme.
pub struct TaichuLayerWeights {
    pub norm1_w: GpuTensor,
    pub norm1_b: GpuTensor,
    pub qkv_w: GpuTensor,
    pub qkv_b: GpuTensor,
    pub proj_w: GpuTensor,
    pub proj_b: GpuTensor,
    pub norm2_w: GpuTensor,
    pub norm2_b: GpuTensor,
    pub fc1_w: GpuTensor,
    pub fc1_b: GpuTensor,
    pub fc2_w: GpuTensor,
    pub fc2_b: GpuTensor,
}

pub struct TaichuVisionWeights {
    /// im2col patch embedder `[hidden, 3*patch²]` (768→1280), F16.
    pub patch_embed_w: GpuTensor,
    /// `[num_prefix_tokens, hidden]` cls/register tokens, F32 host — prepended
    /// per tile in the job prologue.
    pub cls_prefix: Vec<f32>,
    /// Learned absolute pos-embed table `[pos_table_side², hidden]`, F32 host
    /// — every tile bilinearly interpolates it to the patch grid.
    pub pos_embed: Vec<f32>,
    pub layers: Vec<TaichuLayerWeights>,
    /// mlp1.0: RMSNorm weight `[pixel_shuffle_dim]` (no bias — raw w, NOT the
    /// qwen-style +1 convention).
    pub proj_norm_w: GpuTensor,
    /// mlp1.1: Linear `[projector_hidden, pixel_shuffle_dim]`, F16.
    pub proj_fc1_w: GpuTensor,
    /// mlp1.3: Linear `[out_hidden, projector_hidden]`, F16.
    pub proj_fc2_w: GpuTensor,
}

impl TaichuVisionWeights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patch_embed_w);
        for l in self.layers {
            for t in [
                l.norm1_w, l.norm1_b, l.qkv_w, l.qkv_b, l.proj_w, l.proj_b, l.norm2_w, l.norm2_b,
                l.fc1_w, l.fc1_b, l.fc2_w, l.fc2_b,
            ] {
                let _ = gpu.free_tensor(t);
            }
        }
        let _ = gpu.free_tensor(self.proj_norm_w);
        let _ = gpu.free_tensor(self.proj_fc1_w);
        let _ = gpu.free_tensor(self.proj_fc2_w);
    }
}

// ─── Weight loading ─────────────────────────────────────────────────────────

fn load_f32_cpu(hfq: &HfqFile, name: &str, n: usize) -> HipResult<Vec<f32>> {
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        HipError::new(1, &format!("taichu vision tensor not found: {name}"))
    })?;
    let mut vals: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => {
            return Err(HipError::new(
                1,
                &format!("expected F16/F32 for {name}, got qt={other}"),
            ))
        }
    };
    if vals.len() < n {
        return Err(HipError::new(
            1,
            &format!(
                "taichu vision tensor {name} truncated: {} elems < expected {n}",
                vals.len()
            ),
        ));
    }
    vals.truncate(n);
    Ok(vals)
}

const F16_FINITE_MAX: f32 = 65504.0;

fn checked_f32_to_f16(value: f32, tensor: &str, index: usize) -> HipResult<u16> {
    if !value.is_finite() {
        return Err(HipError::new(
            1,
            &format!(
                "{tensor}: element {index} is non-finite ({value}) — cannot narrow F32 oracle to F16"
            ),
        ));
    }
    if value.abs() > F16_FINITE_MAX {
        return Err(HipError::new(
            1,
            &format!(
                "{tensor}: element {index} value {value} exceeds finite F16 range (±{F16_FINITE_MAX})"
            ),
        ));
    }
    Ok(f32_to_f16(value))
}

fn load_f16_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        HipError::new(1, &format!("taichu vision tensor not found: {name}"))
    })?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match info.quant_type {
        1 => gpu.upload_raw(data, &[n]),
        2 => {
            // F32 oracle → narrow to F16 for the vision GEMMs (validated).
            let mut f16_bytes = Vec::with_capacity(n * 2);
            for (i, c) in data.chunks_exact(4).take(n).enumerate() {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                f16_bytes.extend_from_slice(&checked_f32_to_f16(v, name, i)?.to_le_bytes());
            }
            gpu.upload_raw(&f16_bytes, &[n])
        }
        other => Err(HipError::new(
            1,
            &format!("{name}: unsupported vision quant_type={other} (expected F16=1, F32=2)"),
        )),
    }
}

fn load_f32_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let vals = load_f32_cpu(hfq, name, n)?;
    gpu.upload_f32(&vals, &[n])
}

/// Load the C-RADIO tower + mlp1 projector from a ZDTaichu HFQ pack.
///
/// Tensor names are verbatim safetensors names (quantize writes them
/// unchanged): `vision_model.radio_model.model.*` for the ViT and `mlp1.*`
/// for the projector. `input_conditioner.*` and `summary_idxs` are dropped
/// at quantize time (normalization is external; summary tokens are unused).
pub fn load_taichu_vision_weights(
    hfq: &HfqFile,
    config: &TaichuVisionConfig,
    gpu: &mut Gpu,
) -> HipResult<TaichuVisionWeights> {
    let h = config.hidden_size;
    // Divisor/shape preconditions the tower math relies on. A pack that
    // violates them would panic mid-load (div-by-zero) or mid-forward
    // (slice OOB) — refuse it here, at load time.
    if config.patch_size == 0
        || config.image_size == 0
        || config.image_size % config.patch_size != 0
        || config.downsample_ratio <= 0.0
        || h == 0
        || config.pos_table_side == 0
    {
        return Err(HipError::new(
            1,
            &format!(
                "taichu vision config invalid: patch_size={} image_size={} \
                 downsample_ratio={} hidden={h} pos_table_side={}",
                config.patch_size,
                config.image_size,
                config.downsample_ratio,
                config.pos_table_side
            ),
        ));
    }
    // pixel_shuffle_v2 and the mlp1.1 fan-in are hard-coded to the 0.5
    // downsample (2×2 groups, 4h input) — a different ratio must fail loud.
    if (config.downsample_ratio - 0.5).abs() > 1e-6 {
        return Err(HipError::new(
            1,
            &format!(
                "taichu downsample_ratio={} unsupported: only 0.5 is wired \
                 (pixel_shuffle_v2 + mlp1 shapes are hard-coded to it)",
                config.downsample_ratio
            ),
        ));
    }
    let tower = "vision_model.radio_model.model";

    let patch_embed_w = load_f16_gpu(
        hfq,
        gpu,
        &format!("{tower}.patch_generator.embedder.weight"),
    )?;

    // cls/register prefix — host-side, prepended per tile in the job prologue.
    let (cls_info, _) = hfq
        .tensor_data(&format!("{tower}.patch_generator.cls_token.token"))
        .ok_or_else(|| {
            HipError::new(0, "taichu: patch_generator.cls_token.token not found")
        })?;
    let n_prefix = cls_info.shape.first().copied().unwrap_or(0) as usize;
    let cls_prefix = load_f32_cpu(
        hfq,
        &format!("{tower}.patch_generator.cls_token.token"),
        n_prefix * h,
    )?;

    let pos_table = config.pos_table_side * config.pos_table_side;
    let pos_embed = load_f32_cpu(
        hfq,
        &format!("{tower}.patch_generator.pos_embed"),
        pos_table * h,
    )?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for l in 0..config.num_layers {
        let p = format!("{tower}.blocks.{l}");
        layers.push(TaichuLayerWeights {
            norm1_w: load_f32_gpu(hfq, gpu, &format!("{p}.norm1.weight"), h)?,
            norm1_b: load_f32_gpu(hfq, gpu, &format!("{p}.norm1.bias"), h)?,
            qkv_w: load_f16_gpu(hfq, gpu, &format!("{p}.attn.qkv.weight"))?,
            qkv_b: load_f32_gpu(hfq, gpu, &format!("{p}.attn.qkv.bias"), 3 * h)?,
            proj_w: load_f16_gpu(hfq, gpu, &format!("{p}.attn.proj.weight"))?,
            proj_b: load_f32_gpu(hfq, gpu, &format!("{p}.attn.proj.bias"), h)?,
            norm2_w: load_f32_gpu(hfq, gpu, &format!("{p}.norm2.weight"), h)?,
            norm2_b: load_f32_gpu(hfq, gpu, &format!("{p}.norm2.bias"), h)?,
            fc1_w: load_f16_gpu(hfq, gpu, &format!("{p}.mlp.fc1.weight"))?,
            fc1_b: load_f32_gpu(hfq, gpu, &format!("{p}.mlp.fc1.bias"), config.mlp_dim)?,
            fc2_w: load_f16_gpu(hfq, gpu, &format!("{p}.mlp.fc2.weight"))?,
            fc2_b: load_f32_gpu(hfq, gpu, &format!("{p}.mlp.fc2.bias"), h)?,
        });
    }

    let ps_dim = h * (1.0f32 / config.downsample_ratio).powi(2) as usize;
    let proj_norm_w = load_f32_gpu(hfq, gpu, "mlp1.0.weight", ps_dim)?;
    let proj_fc1_w = load_f16_gpu(hfq, gpu, "mlp1.1.weight")?;
    let proj_fc2_w = load_f16_gpu(hfq, gpu, "mlp1.3.weight")?;

    // The loaded prefix-token count is authoritative over the config guess.
    let mut weights = TaichuVisionWeights {
        patch_embed_w,
        cls_prefix,
        pos_embed,
        layers,
        proj_norm_w,
        proj_fc1_w,
        proj_fc2_w,
    };
    let _ = &mut weights; // shape note only
    if n_prefix != config.num_prefix_tokens {
        eprintln!(
            "  taichu-vl: cls_token.token has {n_prefix} rows (config guessed {}) — using tensor shape",
            config.num_prefix_tokens
        );
    }
    Ok(weights)
}

// ─── Position-embedding interpolation ───────────────────────────────────────

/// Bilinear-interpolate the `[K*K, h]` pos table to `[rows*cols, h]`,
/// matching torch `F.interpolate(mode="bilinear", align_corners=False)` on
/// the `(1, h, K, K)` reshape: `src = (dst + 0.5) * scale − 0.5`, clamped.
/// CPU-side; K²·h ≈ 20 MB for the 128² table — a 32² tile grid samples 1/16
/// of it per image.
pub fn pos_embed_bilinear(
    table: &[f32],
    h: usize,
    src_side: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols * h];
    let scale_r = src_side as f32 / rows as f32;
    let scale_c = src_side as f32 / cols as f32;
    for r in 0..rows {
        let fr = ((r as f32 + 0.5) * scale_r - 0.5).clamp(0.0, (src_side - 1) as f32);
        let r0 = fr.floor() as usize;
        let r1 = (r0 + 1).min(src_side - 1);
        let wr = fr - r0 as f32;
        for c in 0..cols {
            let fc = ((c as f32 + 0.5) * scale_c - 0.5).clamp(0.0, (src_side - 1) as f32);
            let c0 = fc.floor() as usize;
            let c1 = (c0 + 1).min(src_side - 1);
            let wc = fc - c0 as f32;
            let dst = &mut out[(r * cols + c) * h..(r * cols + c) * h + h];
            let p00 = &table[(r0 * src_side + c0) * h..];
            let p01 = &table[(r0 * src_side + c1) * h..];
            let p10 = &table[(r1 * src_side + c0) * h..];
            let p11 = &table[(r1 * src_side + c1) * h..];
            for i in 0..h {
                let top = p00[i] * (1.0 - wc) + p01[i] * wc;
                let bot = p10[i] * (1.0 - wc) + p11[i] * wc;
                dst[i] = top * (1.0 - wr) + bot * wr;
            }
        }
    }
    out
}

/// Per-tile pixel_shuffle (HF `pixel_shuffle` ps_version=v2, scale 0.5):
/// `[P*P, h]` patches of one tile → `[(P/2)², 4h]` visual tokens, concat
/// order [top-left, top-right, bottom-left, bottom-right] inside each 2×2
/// group. Hard-coded to factor 2 (downsample_ratio 0.5 is the only shipped
/// variant; assert so a different checkpoint fails loud).
pub fn pixel_shuffle_v2(tile: &[f32], p: usize, h: usize) -> Vec<f32> {
    let half = p / 2;
    let out_dim = 4 * h;
    let mut out = vec![0.0f32; half * half * out_dim];
    for r in 0..half {
        for c in 0..half {
            let dst = &mut out[(r * half + c) * out_dim..(r * half + c) * out_dim + out_dim];
            for (k, &(dr, dc)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)].iter().enumerate() {
                let src = ((2 * r + dr) * p + (2 * c + dc)) * h;
                dst[k * h..k * h + h].copy_from_slice(&tile[src..src + h]);
            }
        }
    }
    out
}

// ─── Tower forward ──────────────────────────────────────────────────────────

/// `linear_f16` shared semantics with qwen35-vl: WMMA on capable arches,
/// gemm+transpose otherwise, then broadcast bias-add.
fn linear_f16(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
        gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)?;
    } else {
        let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
        gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
        gpu.transpose_f32(&yt, &y, out_dim, n)?;
        gpu.free_tensor(yt)?;
    }
    gpu.bias_add_f32(&y, bias, n, out_dim)?;
    Ok(y)
}

/// Bias-free variant for the patch embedder and the projector linears.
fn linear_f16_nobias(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    if gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12() {
        gpu.gemm_f16_wmma_mb8(w, x, &y, out_dim, in_dim, n)?;
    } else {
        let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
        gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
        gpu.transpose_f32(&yt, &y, out_dim, n)?;
        gpu.free_tensor(yt)?;
    }
    Ok(y)
}

/// Resumable C-RADIO tower encode, mirroring `VisionTowerJob`'s contract so
/// the slot engine can interleave other slots' decode between layers:
/// prologue once, ONE block per [`Self::step_layer`], projector epilogue in
/// [`Self::finish`].
///
/// Row layout of the resident `x`: `[t * (num_prefix + P²) + row]` — each
/// tile carries its own 10-token cls/register prefix. Prefix rows are
/// dropped in `finish` before pixel_shuffle, matching
/// `InnerRADIOModel._extract_final` (features start at `num_skip`).
pub struct TaichuTowerJob {
    x: GpuTensor,
    /// Tiles × (prefix + patches_per_tile).
    n: usize,
    n_tiles: usize,
    /// Rows per tile including the prefix block.
    rows_per_tile: usize,
    next_layer: usize,
    attn_naive: bool,
    t0: std::time::Instant,
}

impl TaichuTowerJob {
    /// `patches`: `[n_tiles * patches_per_tile, 3*patch²]` normalized im2col
    /// rows from `crate::image::decode_and_tile`.
    pub fn new(
        gpu: &mut Gpu,
        weights: &TaichuVisionWeights,
        config: &TaichuVisionConfig,
        patches: &[f32],
    ) -> HipResult<Self> {
        let h = config.hidden_size;
        let p_side = config.tile_patches_per_side();
        let p_per_tile = p_side * p_side;
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let n_patches = patches.len() / patch_dim;
        if n_patches % p_per_tile != 0 || n_patches == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "taichu-vl: {} patch rows is not a positive multiple of {p_per_tile} (tile grid)",
                    n_patches
                ),
            ));
        }
        let n_tiles = n_patches / p_per_tile;
        let n_prefix = weights.cls_prefix.len() / h;
        let rows_per_tile = n_prefix + p_per_tile;
        let n = n_tiles * rows_per_tile;
        let t0 = std::time::Instant::now();
        eprintln!(
            "  taichu vision forward (GPU): {n_tiles} tiles × {p_per_tile} patches + {n_prefix} prefix"
        );

        // Patch embed on GPU: [n_patches, patch_dim] → [n_patches, h].
        let x_patches = gpu.upload_f32(patches, &[n_patches * patch_dim])?;
        let emb = match linear_f16_nobias(
            gpu,
            &weights.patch_embed_w,
            &x_patches,
            h,
            patch_dim,
            n_patches,
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(x_patches);
                return Err(e);
            }
        };
        gpu.free_tensor(x_patches)?;

        // Pos-embed add + cls/register prepend: assembled on host (one
        // ~26 MB round-trip at 12 tiles — cheap vs the 32-layer tower), then
        // uploaded once as the residual stream. This keeps the GPU layout
        // exactly `[tile][prefix|patches]` with no gather kernel.
        let pos = pos_embed_bilinear(
            &weights.pos_embed,
            h,
            config.pos_table_side,
            p_side,
            p_side,
        );
        let mut emb_host = gpu.download_f32(&emb)?;
        gpu.free_tensor(emb)?;
        let mut x_host = vec![0.0f32; n * h];
        for t in 0..n_tiles {
            // Prefix block: broadcast cls_prefix.
            for r in 0..n_prefix {
                let dst = &mut x_host[(t * rows_per_tile + r) * h..(t * rows_per_tile + r) * h + h];
                dst.copy_from_slice(&weights.cls_prefix[r * h..r * h + h]);
            }
            // Patch rows: emb + interpolated pos.
            for r in 0..p_per_tile {
                let src = &emb_host[(t * p_per_tile + r) * h..(t * p_per_tile + r) * h + h];
                let dst = &mut x_host
                    [(t * rows_per_tile + n_prefix + r) * h
                        ..(t * rows_per_tile + n_prefix + r) * h + h];
                let pr = &pos[r * h..r * h + h];
                for i in 0..h {
                    dst[i] = src[i] + pr[i];
                }
            }
        }
        emb_host.clear();
        let x = gpu.upload_f32(&x_host, &[n * h])?;

        let attn_naive = matches!(
            hipfire_config::developer_var("HIPFIRE_VIT_ATTN").as_deref(),
            Ok("naive")
        ) || config.head_dim % 16 != 0
            || config.head_dim > 128;

        Ok(Self {
            x,
            n,
            n_tiles,
            rows_per_tile,
            next_layer: 0,
            attn_naive,
            t0,
        })
    }

    /// Run the NEXT tower block. Returns `true` when the last block has run.
    pub fn step_layer(
        &mut self,
        gpu: &mut Gpu,
        weights: &TaichuVisionWeights,
        config: &TaichuVisionConfig,
    ) -> HipResult<bool> {
        assert!(
            self.next_layer < config.num_layers,
            "step_layer past the last tower block"
        );
        let h = config.hidden_size;
        let n = self.n;
        let lw = &weights.layers[self.next_layer];
        // Every tensor local to this step is freed before an error escapes —
        // an `Err` without the frees would strand VRAM on the rig for the
        // life of the model.
        macro_rules! step {
            ($e:expr, $($t:expr),+ $(,)?) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => { $(let _ = gpu.free_tensor($t);)+ return Err(e); }
                }
            };
        }

        // LN1 → fused QKV (no positional encoding inside the blocks).
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        step!(gpu.layernorm_batched(&self.x, &lw.norm1_w, &lw.norm1_b, &tmp, n, h, config.norm_eps), tmp);
        let qkv = step!(linear_f16(gpu, &lw.qkv_w, &tmp, &lw.qkv_b, 3 * h, h, n), tmp);
        gpu.free_tensor(tmp)?;

        let attn_out = gpu.alloc_tensor(&[n * h], DType::F32)?;
        // Each tile is an independent batch element (the per-tile duplicated
        // cls/register prefix exists for exactly that reason): the softmax
        // must NOT cross tile boundaries, so segment at rows_per_tile.
        if self.attn_naive {
            step!(gpu.vit_attention_f32(&qkv, &attn_out, n, h, config.num_heads, config.head_dim, self.rows_per_tile), qkv, attn_out);
        } else {
            step!(gpu.vit_attention_qtiled_f32(&qkv, &attn_out, n, h, config.num_heads, config.head_dim, self.rows_per_tile), qkv, attn_out);
        }
        gpu.free_tensor(qkv)?;

        let proj = step!(linear_f16(gpu, &lw.proj_w, &attn_out, &lw.proj_b, h, h, n), attn_out);
        gpu.free_tensor(attn_out)?;
        step!(gpu.add_inplace_f32(&self.x, &proj), proj);
        gpu.free_tensor(proj)?;

        // LN2 → fc1 → exact-erf GELU → fc2 → residual.
        let tmp2 = gpu.alloc_tensor(&[n * h], DType::F32)?;
        step!(gpu.layernorm_batched(&self.x, &lw.norm2_w, &lw.norm2_b, &tmp2, n, h, config.norm_eps), tmp2);
        let fc1 = step!(linear_f16(gpu, &lw.fc1_w, &tmp2, &lw.fc1_b, config.mlp_dim, h, n), tmp2);
        gpu.free_tensor(tmp2)?;
        step!(gpu.gelu_erf_f32(&fc1, &fc1, n * config.mlp_dim), fc1);
        let fc2 = step!(linear_f16(gpu, &lw.fc2_w, &fc1, &lw.fc2_b, h, config.mlp_dim, n), fc1);
        gpu.free_tensor(fc1)?;
        step!(gpu.add_inplace_f32(&self.x, &fc2), fc2);
        gpu.free_tensor(fc2)?;

        self.next_layer += 1;
        Ok(self.next_layer == config.num_layers)
    }

    /// Epilogue: drop prefix rows, per-tile pixel_shuffle 0.5, then
    /// RMSNorm → Linear → SquaredReLU → Linear projector. Returns
    /// `[n_tiles * num_image_token, out_hidden]` host rows in
    /// tile-then-raster order — the order the prompt splice expects.
    pub fn finish(
        self,
        gpu: &mut Gpu,
        weights: &TaichuVisionWeights,
        config: &TaichuVisionConfig,
    ) -> HipResult<Vec<f32>> {
        let Self {
            x,
            n: _,
            n_tiles,
            rows_per_tile,
            t0,
            ..
        } = self;
        let h = config.hidden_size;
        macro_rules! step {
            ($e:expr, $($t:expr),+ $(,)?) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => { $(let _ = gpu.free_tensor($t);)+ return Err(e); }
                }
            };
        }
        step!(gpu.hip.device_synchronize(), x);

        let x_host = step!(gpu.download_f32(&x), x);
        gpu.free_tensor(x)?;
        let n_prefix = rows_per_tile - config.tile_patches_per_side().pow(2);
        let p_side = config.tile_patches_per_side();
        let p_per_tile = p_side * p_side;
        let tps = config.tile_tokens_per_side;
        let ps_dim = 4 * h;

        // Per tile: drop prefix rows → [P², h] → pixel_shuffle → [T², 4h].
        let mut shuffled = vec![0.0f32; n_tiles * tps * tps * ps_dim];
        for t in 0..n_tiles {
            let base = (t * rows_per_tile + n_prefix) * h;
            let tile = &x_host[base..base + p_per_tile * h];
            let s = pixel_shuffle_v2(tile, p_side, h);
            shuffled[t * tps * tps * ps_dim..(t + 1) * tps * tps * ps_dim]
                .copy_from_slice(&s);
        }
        drop(x_host);

        let n_out = n_tiles * tps * tps;
        let merged = gpu.upload_f32(&shuffled, &[n_out * ps_dim])?;
        // RMSNorm (raw weight — no +1 bias convention here; it was written
        // as an F32 oracle so elements pass through unmodified).
        let normed = gpu.alloc_tensor(&[n_out * ps_dim], DType::F32)?;
        step!(gpu.rmsnorm_batched(&merged, &weights.proj_norm_w, &normed, n_out, ps_dim, 1e-5), merged, normed);
        gpu.free_tensor(merged)?;

        let h1 = step!(
            linear_f16_nobias(
                gpu,
                &weights.proj_fc1_w,
                &normed,
                config.projector_hidden,
                ps_dim,
                n_out,
            ),
            normed
        );
        gpu.free_tensor(normed)?;
        // SquaredReLU: relu(x)² elementwise (NOT plain x²).
        step!(gpu.squared_relu_f32(&h1, &h1, n_out * config.projector_hidden), h1);
        let out = step!(
            linear_f16_nobias(
                gpu,
                &weights.proj_fc2_w,
                &h1,
                config.out_hidden_size,
                config.projector_hidden,
                n_out,
            ),
            h1
        );
        gpu.free_tensor(h1)?;

        let result = step!(gpu.download_f32(&out), out);
        gpu.free_tensor(out)?;
        eprintln!(
            "  taichu vision done: {n_out} tokens × {} dims ({:.2}s)",
            config.out_hidden_size,
            t0.elapsed().as_secs_f32()
        );
        Ok(result)
    }

    /// Free a job that will never reach `finish` (slot aborted/evicted
    /// mid-encode). Best-effort, mirrors `VisionTowerJob::free`.
    pub fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.free_tensor(self.x)
    }
}

/// Monolithic encode: prologue + all blocks + projector epilogue.
/// `patches` is `[n_tiles * 1024, 3*16*16]` im2col rows; `n_tiles` is
/// inferred. Output order: tile-major, raster within tile — matches the
/// `<|image_pad|>` expansion order (grid tiles raster, then thumbnail).
pub fn taichu_vision_forward(
    gpu: &mut Gpu,
    weights: &TaichuVisionWeights,
    config: &TaichuVisionConfig,
    patches: &[f32],
) -> HipResult<Vec<f32>> {
    let mut job = TaichuTowerJob::new(gpu, weights, config, patches)?;
    let mut done = false;
    while !done {
        done = job.step_layer(gpu, weights, config)?;
    }
    job.finish(gpu, weights, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_shuffle_v2_groups_2x2_tl_tr_bl_br() {
        // 4×4 tile of h=2: patch value = row*4+col marker.
        let p = 4;
        let h = 2;
        let mut tile = vec![0.0f32; p * p * h];
        for r in 0..p {
            for c in 0..p {
                tile[(r * p + c) * h] = (r * p + c) as f32;
                tile[(r * p + c) * h + 1] = 1000.0 + (r * p + c) as f32;
            }
        }
        let out = pixel_shuffle_v2(&tile, p, h);
        // Output token (0,0): patches (0,0),(0,1),(1,0),(1,1) = 0,1,4,5.
        let tok0 = &out[0..8];
        assert_eq!(tok0[0], 0.0);
        assert_eq!(tok0[2], 1.0);
        assert_eq!(tok0[4], 4.0);
        assert_eq!(tok0[6], 5.0);
        // Token (0,1): patches (0,2),(0,3),(1,2),(1,3) = 2,3,6,7.
        let tok1 = &out[8..16];
        assert_eq!(tok1[0], 2.0);
        assert_eq!(tok1[2], 3.0);
        assert_eq!(tok1[4], 6.0);
        assert_eq!(tok1[6], 7.0);
    }

    #[test]
    fn pos_embed_bilinear_identity_when_sizes_match() {
        let h = 4;
        let k = 4;
        let mut table = vec![0.0f32; k * k * h];
        for i in 0..k * k {
            for j in 0..h {
                table[i * h + j] = (i * h + j) as f32;
            }
        }
        let out = pos_embed_bilinear(&table, h, k, k, k);
        assert_eq!(out, table);
    }
}
