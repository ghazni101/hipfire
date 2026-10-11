// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batch-invariant Flash-Next prefill profile.
//!
//! Today several kernels on the qwen4 prefill path are routed by the row count
//! `n` of the forward (thresholds 8 / 512 / 2048 / `x_src_rows >= 5120`, ...),
//! so a row's bytes depend on how many other rows share its forward.  Under
//! the invariant profile every row-count ROUTING decision is taken as if `n`
//! were [`QWEN4_PREFILL_CANON_ROWS`] (the gfx1151 default prefill chunk, above
//! every threshold) while launches keep using the real `n`.  At `n == 8192`
//! the profile therefore equals the product route byte for byte.
//!
//! The profile is opt-in (`HIPFIRE_PREFILL_PM_INVARIANT`) and only engaged by
//! the qwen4 forward on gfx1151 via `Gpu::prefill_invariant_scope`.

use crate::dispatch::Gpu;
use std::sync::LazyLock;

/// Row count every routing predicate sees under the invariant profile.
pub const QWEN4_PREFILL_CANON_ROWS: usize = 8192;

static PREFILL_PM_INVARIANT: LazyLock<bool> =
    LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_PREFILL_PM_INVARIANT", false));

/// `HIPFIRE_PREFILL_PM_INVARIANT` (developer bool, default off), read once.
pub fn prefill_pm_invariant_enabled() -> bool {
    *PREFILL_PM_INVARIANT
}

/// Lab selector for the qwen4 prefill GDN recurrence
/// (`HIPFIRE_QWEN4_GDN_PREFILL`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4GdnPrefill {
    /// Product route: WY chunked recurrence at >= 512 rows, persistent256
    /// below.
    Wy,
    /// Always the persistent256 recurrence (launch-end Q8 requant).
    Persistent,
    /// Persistent256 with the token-canonical Q8 requant (every row requants
    /// at its own position), independent of how rows split into launches.
    Canon,
}

static QWEN4_GDN_PREFILL: LazyLock<Qwen4GdnPrefill> = LazyLock::new(|| {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_GDN_PREFILL")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "persistent" => Qwen4GdnPrefill::Persistent,
        "canon" => Qwen4GdnPrefill::Canon,
        _ => Qwen4GdnPrefill::Wy,
    }
});

/// `HIPFIRE_QWEN4_GDN_PREFILL` = `wy` (default) | `persistent` | `canon`,
/// read once.
pub fn qwen4_gdn_prefill() -> Qwen4GdnPrefill {
    *QWEN4_GDN_PREFILL
}

impl Gpu {
    /// Row count to feed ROUTING predicates (never launch extents): the
    /// canonical prefill chunk under the invariant profile, else `rows`.
    #[inline]
    pub fn route_rows(&self, rows: usize) -> usize {
        if self.prefill_invariant_scope {
            QWEN4_PREFILL_CANON_ROWS
        } else {
            rows
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{
        QWEN4_F16_WMMA_MIN_TOKENS, QWEN4_MOE_SYM_IU4_MIN_ROWS, QWEN4_MQ6_X4_PM_MIN_N_GFX1151,
    };

    /// The canonical row count must sit above every row threshold the profile
    /// routes on, or "taken as if n were canonical" would pick a short route.
    #[test]
    fn canonical_rows_clear_every_prefill_threshold() {
        assert_eq!(QWEN4_PREFILL_CANON_ROWS, 8192);
        assert!(QWEN4_PREFILL_CANON_ROWS >= QWEN4_F16_WMMA_MIN_TOKENS);
        assert!(QWEN4_PREFILL_CANON_ROWS >= 10 * QWEN4_F16_WMMA_MIN_TOKENS);
        assert!(QWEN4_PREFILL_CANON_ROWS >= QWEN4_MOE_SYM_IU4_MIN_ROWS);
        assert!(QWEN4_PREFILL_CANON_ROWS >= QWEN4_MQ6_X4_PM_MIN_N_GFX1151);
        // HC-down 160 x 64 tile (rows >= 2048) and the few-row arms (<= 8).
        assert!(QWEN4_PREFILL_CANON_ROWS >= 2048);
        assert!(QWEN4_PREFILL_CANON_ROWS > 8);
        assert_eq!(QWEN4_PREFILL_CANON_ROWS % 128, 0);
    }
}
