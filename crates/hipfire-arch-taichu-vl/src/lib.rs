// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-taichu-vl: ZDTaichu-5.0 vision-language front-end.
//!
//! ZDTaichu-5.0 is "C-RADIOv4-H vision tower + Qwen3.5-hybrid text decoder"
//! (HF `model_type = zdtaichu5_0`, `architectures =
//! ["ZDTaichu5_0_ForConditionalGeneration"]`). The text half reuses
//! `hipfire-arch-qwen35` wholesale — same 3×linear+1×full hybrid block, same
//! kernels, same bundle — so this crate owns ONLY the vision surface that
//! differs from Qwen3.5-VL (`hipfire-arch-qwen35-vl`):
//!
//!   - `vision` — C-RADIO tower weights + GPU forward. ViT-H/16 (1280-dim,
//!     32 layers, fused QKV, exact-erf GELU MLP, NO RoPE inside the tower),
//!     10 prepend cls/register tokens, an absolute pos-embed table
//!     bilinearly interpolated to the tile grid, then per-tile pixel_shuffle
//!     (downsample_ratio 0.5) + RMSNorm→Linear→SquaredReLU→Linear projector.
//!   - `image` — InternVL-style dynamic tiling (512×512 tiles, ≤12,
//!     aspect-ratio snapped, +1 thumbnail tile) and ImageNet mean/std
//!     normalization — the RADIO input conditioner is externalized.
//!   - `mrope` — per-token 3D position builder matching
//!     `ZDTaichu5_0_ForConditionalGeneration.get_vision_position_ids`:
//!     tile-level h/w coordinates over the merged 16×16-per-tile grid, with
//!     the thumbnail tile's coordinates stretched by (tile_rows, tile_cols).
//!
//! Forward dispatch is NOT trait-routed (mirroring qwen35-vl): the daemon /
//! generate layer calls `taichu_vision_forward` directly when the loaded
//! bundle carries `taichu_vision_weights`.

#[cfg(feature = "deltanet")]
pub mod image;
pub mod mrope;
#[cfg(feature = "deltanet")]
pub mod vision;
