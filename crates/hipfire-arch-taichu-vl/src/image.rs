// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZDTaichu-5.0 image preprocessing: InternVL-style dynamic tiling.
//!
//! Ports `image_processing.py` (`select_tile_grid`, `dynamic_preprocess`,
//! `_preprocess`) from the ZDTaichu checkpoint: the source image is
//! aspect-snapped to a tile grid of ≤ `max_num_tiles` square `image_size`
//! tiles, bicubic-resized to the grid extent, cropped into tiles (raster
//! order), then — when the grid is >1 tile — a full-image thumbnail tile is
//! appended last. Each tile is normalized with ImageNet mean/std (the RADIO
//! input conditioner is externalized) and emitted as `[3·P²]` im2col rows
//! per tile for the linear patch embedder.
//!
//! Output contract with `vision.rs`: patch row `t*P² + (r*P + c)` is patch
//! `(r, c)` of tile `t`, channel-major `[c][y][x]` — `Im2Patches`/`F.unfold`
//! ordering.

use image::imageops::FilterType;
use std::path::Path;

/// Output of [`decode_and_tile`].
pub struct TiledImage {
    /// `[n_tiles * P², 3 * patch²]` f32 im2col rows (ImageNet-normalized).
    pub patches: Vec<f32>,
    /// Grid tiles incl. thumbnail.
    pub n_tiles: usize,
    /// InternVL grid shape — needed by the mrope builder. `tile_cols` counts
    /// GRID columns only (thumbnail excluded).
    pub tile_rows: usize,
    pub tile_cols: usize,
    /// `n_tiles * num_image_token` — `<|image_pad|>` count for prompt splice.
    pub n_visual_tokens: usize,
}

/// InternVL target-ratio enumeration: all (i, j) with min_num ≤ i·j ≤
/// max_num, sorted by area — `get_internvl_target_ratios`.
fn target_ratios(min_num: usize, max_num: usize) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                if i * j >= min_num && i * j <= max_num {
                    v.push((i, j));
                }
            }
        }
    }
    v.sort_by_key(|&(i, j)| i * j);
    v.dedup();
    v
}

/// `find_closest_aspect_ratio` (OpenGVLab InternVL): nearest (w-tiles,
/// h-tiles) by aspect difference; ties broken toward grids the source has
/// enough pixels to fill.
fn find_closest_aspect_ratio(
    aspect: f32,
    ratios: &[(usize, usize)],
    width: usize,
    height: usize,
    image_size: usize,
) -> (usize, usize) {
    let mut best_diff = f32::INFINITY;
    let mut best = (1usize, 1usize);
    let area = (width * height) as f32;
    for &(rw, rh) in ratios {
        let target = rw as f32 / rh as f32;
        let diff = (aspect - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (rw, rh);
        } else if diff == best_diff && area > 0.5 * (image_size * image_size * rw * rh) as f32 {
            best = (rw, rh);
        }
    }
    best
}

/// `select_tile_grid`: area cap + 3× aspect-sanity guard, then closest ratio.
/// Returns `(grid_rows, grid_cols)` — rows/cols of TILES (thumbnail excluded).
fn select_tile_grid(
    orig_w: usize,
    orig_h: usize,
    image_size: usize,
    min_tiles: usize,
    max_tiles: usize,
) -> (usize, usize) {
    // Guard 1: never emit more tiles than the source has pixels for.
    let tile_px = (image_size * image_size) as f32;
    let area_max = ((orig_w * orig_h) as f32 / tile_px).ceil().max(1.0) as usize;
    let effective_max = max_tiles.min(area_max).max(min_tiles);

    let mut ratios = target_ratios(min_tiles, effective_max);
    // Guard 2: drop candidates whose aspect differs from the source by >3×.
    let src_ar = orig_w as f32 / orig_h as f32;
    let filtered: Vec<(usize, usize)> = ratios
        .iter()
        .copied()
        .filter(|&(rw, rh)| {
            let r = (rw as f32 / rh as f32) / src_ar;
            (1.0 / 3.0) <= r && r <= 3.0
        })
        .collect();
    if !filtered.is_empty() {
        ratios = filtered;
    }
    // find_closest_aspect_ratio returns (target_width_tiles, target_height_tiles).
    let (rw, rh) = find_closest_aspect_ratio(src_ar, &ratios, orig_w, orig_h, image_size);
    // HF: target_width = image_size*rw, target_height = image_size*rh;
    // tile_rows = target_height/image_size, tile_cols = target_width/image_size.
    (rh, rw)
}

/// Decode + tile + normalize an image for the C-RADIO tower.
///
/// Returns im2col patch rows ready for `taichu_vision_forward`. `config` is
/// the parsed tower config (image_size, patch_size, max_num_tiles,
/// use_thumbnail, norm stats).
pub fn tile_dynamic_image(
    img: image::DynamicImage,
    cfg: &crate::vision::TaichuVisionConfig,
) -> TiledImage {
    let img = match img {
        image::DynamicImage::ImageRgb8(_) => img,
        other => image::DynamicImage::ImageRgb8(other.to_rgb8()),
    };
    let orig_w = img.width() as usize;
    let orig_h = img.height() as usize;
    let image_size = cfg.image_size;
    let p = cfg.tile_patches_per_side();
    let patch_dim = 3 * cfg.patch_size * cfg.patch_size;

    let (tile_rows, tile_cols) =
        select_tile_grid(orig_w, orig_h, image_size, 1, cfg.max_num_tiles);
    let target_w = tile_cols * image_size;
    let target_h = tile_rows * image_size;

    // Bicubic to the grid extent — CatmullRom is the `image` crate's bicubic
    // (same filter family as PIL's resample=3 the HF processor uses).
    let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom);
    let resized = resized.to_rgb8();

    let n_grid = tile_rows * tile_cols;
    let n_tiles = if cfg.use_thumbnail && n_grid != 1 {
        n_grid + 1
    } else {
        n_grid
    };

    let mut patches = vec![0.0f32; n_tiles * p * p * patch_dim];
    let mut write_tile = |tile_idx: usize, tile: &image::RgbImage| {
        let w = tile.width() as usize;
        debug_assert_eq!(w, image_size);
        for py in 0..p {
            for px in 0..p {
                let row = &mut patches
                    [(tile_idx * p * p + py * p + px) * patch_dim
                        ..(tile_idx * p * p + py * p + px) * patch_dim + patch_dim];
                let mut k = 0;
                for c in 0..3 {
                    for dy in 0..cfg.patch_size {
                        for dx in 0..cfg.patch_size {
                            let px8 = tile.get_pixel(
                                (px * cfg.patch_size + dx) as u32,
                                (py * cfg.patch_size + dy) as u32,
                            )[c] as f32
                                / 255.0;
                            row[k] = (px8 - cfg.norm_mean[c]) / cfg.norm_std[c];
                            k += 1;
                        }
                    }
                }
            }
        }
    };

    // Grid tiles in raster order.
    for i in 0..n_grid {
        let col = i % tile_cols;
        let row = i / tile_cols;
        let tile = image::imageops::crop_imm(
            &resized,
            (col * image_size) as u32,
            (row * image_size) as u32,
            image_size as u32,
            image_size as u32,
        )
        .to_image();
        write_tile(i, &tile);
    }
    // Thumbnail: whole source resized to one tile.
    if n_tiles > n_grid {
        let thumb = img.resize_exact(image_size as u32, image_size as u32, FilterType::CatmullRom);
        write_tile(n_grid, &thumb.to_rgb8());
    }

    TiledImage {
        patches,
        n_tiles,
        tile_rows,
        tile_cols,
        n_visual_tokens: n_tiles * cfg.num_image_token,
    }
}

/// Decode an image file and tile it. Errors are strings for the daemon's
/// request-error path (parity with qwen35-vl's `load_and_preprocess`).
pub fn decode_and_tile(
    path: &Path,
    cfg: &crate::vision::TaichuVisionConfig,
) -> Result<TiledImage, String> {
    let img = hipfire_runtime::imagedec::decode_dynamic_path(path)?;
    Ok(tile_dynamic_image(img, cfg))
}

/// Byte-buffer variant for `image_base64` requests.
pub fn decode_and_tile_bytes(
    data: &[u8],
    cfg: &crate::vision::TaichuVisionConfig,
) -> Result<TiledImage, String> {
    let (w, h) = hipfire_runtime::imagedec::probe_dimensions(data)?;
    if (w as usize) * (h as usize) > 50_000_000 {
        return Err(format!(
            "image dimensions ({w}x{h}) exceed maximum (50 MP)"
        ));
    }
    let img = hipfire_runtime::imagedec::decode_dynamic(data)?;
    Ok(tile_dynamic_image(img, cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_image_gets_single_tile() {
        // 512×512 source: area cap = 1 → 1×1 grid, no thumbnail.
        let (r, c) = select_tile_grid(512, 512, 512, 1, 12);
        assert_eq!((r, c), (1, 1));
    }

    #[test]
    fn wide_image_prefers_wide_grid() {
        // 2048×512: aspect 4 → (4,1) grid (4 tiles), never (1,4).
        let (r, c) = select_tile_grid(2048, 512, 512, 1, 12);
        assert_eq!((r, c), (1, 4));
    }

    #[test]
    fn thumbnail_rule_matches_hf() {
        // count_tiles: grid tiles +1 when n_grid != 1 and use_thumbnail.
        for (w, h, want) in [(512usize, 512usize, 1usize), (1024, 512, 3)] {
            let (r, c) = select_tile_grid(w, h, 512, 1, 12);
            let n_grid = r * c;
            let n_tiles = if n_grid != 1 { n_grid + 1 } else { n_grid };
            assert_eq!(n_tiles, want, "{w}x{h}");
        }
    }
}
