// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 3D mrope positions for ZDTaichu-5.0's InternVL-style tiled images.
//!
//! Ports `ZDTaichu5_0_ForConditionalGeneration.get_vision_position_ids` /
//! `get_rope_index` (modeling.py): text tokens take `[cursor; 3]`; an image
//! run takes `(t=cursor, h=cursor+global_row, w=cursor+global_col)` where the
//! grid tiles contribute global raster coordinates over a
//! `(tile_rows·S)×(tile_cols·S)` visual field (S = tile_tokens_per_side) and
//! the thumbnail tile's local coords are stretched by (tile_rows, tile_cols)
//! so its positions overlay the grid coarsely. After an image run the cursor
//! advances by `max(tile_rows·S, tile_cols·S)` — the spatial extent, not the
//! token count — same as qwen35-vl's `build_mrope_positions`.
//!
//! Pure CPU, no GPU, no I/O — unit-tested below.

/// One contiguous `<|image_pad|>` run produced by a single image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaichuImageSpan {
    /// Index of the first visual token.
    pub start: usize,
    /// Total visual tokens in the run (== n_tiles × num_image_token).
    pub len: usize,
    /// InternVL tile grid (thumbnail excluded).
    pub tile_rows: usize,
    pub tile_cols: usize,
    /// True when the run's last tile is the appended thumbnail
    /// (`len == (tile_rows·tile_cols + 1) × num_image_token`).
    pub has_thumbnail: bool,
}

/// Positions + decode delta; mirrors `qwen35_vl::mrope::MropePositions`
/// without depending on the sibling crate (no crate cycle pressure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaichuMropePositions {
    /// Per-token (t, h, w).
    pub positions: Vec<[i32; 3]>,
    /// `positions.max() + 1 - n_tokens` — added to the running sequence
    /// length to get decode-step positions.
    pub rope_delta: i32,
}

/// Build per-token (t, h, w) for a prompt of `n_tokens` containing `spans`
/// image runs (sorted by `start`, non-overlapping). `tile_side` is
/// `tile_tokens_per_side` (16 for the shipped checkpoint).
pub fn build_taichu_mrope_positions(
    n_tokens: usize,
    spans: &[TaichuImageSpan],
    tile_side: usize,
) -> TaichuMropePositions {
    assert!(tile_side > 0);
    let _npt = tile_side * tile_side;
    let mut positions = Vec::with_capacity(n_tokens);
    let mut cursor: i32 = 0;
    let mut tok = 0usize;

    for span in spans {
        debug_assert!(span.start >= tok, "image spans must be sorted/disjoint");
        while tok < span.start && tok < n_tokens {
            positions.push([cursor; 3]);
            cursor += 1;
            tok += 1;
        }
        // Grid tiles: token (r, c) of tile (tr, tc) →
        //   h = tr·S + r, w = tc·S + c (raster within tile, raster across tiles)
        let mut emitted = 0usize;
        for t in 0..(span.tile_rows * span.tile_cols) {
            let tr = t / span.tile_cols;
            let tc = t % span.tile_cols;
            for r in 0..tile_side {
                for c in 0..tile_side {
                    positions.push([
                        cursor,
                        cursor + (tr * tile_side + r) as i32,
                        cursor + (tc * tile_side + c) as i32,
                    ]);
                    emitted += 1;
                }
            }
        }
        // Thumbnail tile: local (r, c) → global (r·tile_rows, c·tile_cols).
        if span.has_thumbnail {
            for r in 0..tile_side {
                for c in 0..tile_side {
                    positions.push([
                        cursor,
                        cursor + (r * span.tile_rows) as i32,
                        cursor + (c * span.tile_cols) as i32,
                    ]);
                    emitted += 1;
                }
            }
        }
        debug_assert_eq!(
            emitted, span.len,
            "span len {} != emitted {emitted} (grid {}x{} thumb {})",
            span.len, span.tile_rows, span.tile_cols, span.has_thumbnail
        );
        tok += emitted.min(span.len);
        // Advance by the spatial extent (matches HF
        // `current_pos += max(tile_rows*S, tile_cols*S)`).
        cursor += (span.tile_rows.max(span.tile_cols) * tile_side) as i32;
    }

    while tok < n_tokens {
        positions.push([cursor; 3]);
        cursor += 1;
        tok += 1;
    }

    let max_pos = positions
        .iter()
        .flat_map(|p| p.iter().copied())
        .max()
        .unwrap_or(0);
    TaichuMropePositions {
        rope_delta: max_pos + 1 - n_tokens as i32,
        positions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single 1×1-tile image mid-prompt: visual tokens get the 4×4 (S=4)
    /// local grid; text before/after stays on the 1D cursor which then jumps
    /// by the 4-unit spatial extent.
    #[test]
    fn single_tile_positions_and_cursor() {
        // prompt: 3 text + 16 image + 2 text, S=4, grid 1×1, no thumbnail.
        let span = TaichuImageSpan {
            start: 3,
            len: 16,
            tile_rows: 1,
            tile_cols: 1,
            has_thumbnail: false,
        };
        let m = build_taichu_mrope_positions(21, &[span], 4);
        assert_eq!(m.positions.len(), 21);
        // Text prefix.
        assert_eq!(m.positions[0], [0, 0, 0]);
        assert_eq!(m.positions[2], [2, 2, 2]);
        // First visual token: t=cursor=3, h=3+0, w=3+0.
        assert_eq!(m.positions[3], [3, 3, 3]);
        // Visual token (r=0,c=1): w advances.
        assert_eq!(m.positions[4], [3, 3, 4]);
        // Visual token (r=1,c=0): h advances.
        assert_eq!(m.positions[7], [3, 4, 3]);
        // Trailing text resumes at cursor 3 + max(1,1)*4 = 7.
        assert_eq!(m.positions[19], [7, 7, 7]);
        assert_eq!(m.positions[20], [8, 8, 8]);
        // max pos = 8 → rope_delta = 8+1-21 = -12.
        assert_eq!(m.rope_delta, -12);
    }

    /// 2×1 grid + thumbnail: thumb coords stretch by (rows, cols).
    #[test]
    fn thumbnail_stretches_over_grid() {
        // S=2, grid 1×2 → grid tokens 2·4=8, thumb 4, len 12, span at 0.
        let span = TaichuImageSpan {
            start: 0,
            len: 12,
            tile_rows: 1,
            tile_cols: 2,
            has_thumbnail: true,
        };
        let m = build_taichu_mrope_positions(12, &[span], 2);
        // Last grid token: tile (0,1), local (1,1) → (0, 0+1, 0+3).
        assert_eq!(m.positions[7], [0, 1, 3]);
        // First thumb token (r=0,c=0) → (0, 0·1, 0·2) = (0,0,0).
        assert_eq!(m.positions[8], [0, 0, 0]);
        // Thumb (r=1,c=1) → (0, 1·1, 1·2) = (0,1,2).
        assert_eq!(m.positions[11], [0, 1, 2]);
        assert_eq!(m.rope_delta, 3 + 1 - 12);
    }
}
