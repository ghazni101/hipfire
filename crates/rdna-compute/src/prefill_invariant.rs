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
//!
//! The same canonical-routing switch is also engaged per tile by the agent-turn
//! append route (`HIPFIRE_QWEN4_APPEND_WMMA`, on by default, exact gfx1151,
//! prefill tiles of `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS..512` rows), which
//! additionally suppresses the symmetric IU4 MoE arm for that tile
//! (`Gpu::qwen4_append_f16_moe_scope`).

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

/// Default of `HIPFIRE_QWEN4_APPEND_WMMA`: on (KLD, coherence and timing gates
/// on Strix Halo in the 0.4.3 CHANGELOG).
pub const QWEN4_APPEND_WMMA_DEFAULT: bool = true;
/// Default of `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS`: above the measured
/// F32/append-route crossover (the route loses at 33 rows, wins at 55 and 81).
pub const QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT: usize = 48;

static QWEN4_APPEND_WMMA: LazyLock<bool> = LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_APPEND_WMMA", QWEN4_APPEND_WMMA_DEFAULT)
});

/// `HIPFIRE_QWEN4_APPEND_WMMA` (developer bool, default [`QWEN4_APPEND_WMMA_DEFAULT`]), read once.
pub fn qwen4_append_wmma_enabled() -> bool {
    *QWEN4_APPEND_WMMA
}

static QWEN4_APPEND_WMMA_MIN_ROWS: LazyLock<usize> = LazyLock::new(|| {
    hipfire_config::developer_var("HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 2)
        .unwrap_or(QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT)
});

/// `HIPFIRE_QWEN4_APPEND_WMMA_MIN_ROWS` (developer var, default
/// [`QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT`]; values below 2 are ignored), read once.
pub fn qwen4_append_wmma_min_rows() -> usize {
    *QWEN4_APPEND_WMMA_MIN_ROWS
}

/// Whether a qwen4 forward tile takes the agent-turn append route: an
/// enabled prefill tile of `min_rows..QWEN4_F16_WMMA_MIN_TOKENS` rows on
/// exact gfx1151 that is not already under the invariant profile. Decode,
/// MTP verify, other arches, tiles of 512+ rows and the
/// `HIPFIRE_PREFILL_PM_INVARIANT` profile (whose MoE must route the same at
/// every row count) keep their route.
pub fn qwen4_append_wmma_route(
    enabled: bool,
    prefill: bool,
    arch: &str,
    rows: usize,
    min_rows: usize,
    pm_invariant: bool,
) -> bool {
    enabled
        && prefill
        && !pm_invariant
        && arch == "gfx1151"
        && rows >= min_rows
        && rows < crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
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

    #[test]
    fn append_route_disabled_is_never_taken() {
        for prefill in [false, true] {
            for pm in [false, true] {
                for arch in ["gfx1151", "gfx1201", "gfx942"] {
                    for rows in [0, 1, 64, 511, 512, 8192] {
                        assert!(!qwen4_append_wmma_route(false, prefill, arch, rows, 64, pm));
                    }
                }
            }
        }
    }

    #[test]
    fn append_route_row_window_and_gates() {
        let route = |prefill, arch, rows, pm| {
            qwen4_append_wmma_route(true, prefill, arch, rows, 64, pm)
        };
        assert!(!route(true, "gfx1151", 63, false));
        assert!(route(true, "gfx1151", 64, false));
        assert!(route(true, "gfx1151", 511, false));
        assert!(!route(true, "gfx1151", 512, false));
        assert!(!route(true, "gfx1151", 8192, false));
        assert!(!route(false, "gfx1151", 256, false));
        assert!(!route(true, "gfx1151", 256, true));
        for arch in ["gfx1201", "gfx942", "gfx1100", "gfx1150"] {
            assert!(!route(true, arch, 256, false));
        }
    }

    #[test]
    fn append_route_defaults() {
        assert!(QWEN4_APPEND_WMMA_DEFAULT);
        assert!(QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT >= 2);
        assert!(QWEN4_APPEND_WMMA_MIN_ROWS_DEFAULT < QWEN4_F16_WMMA_MIN_TOKENS);
    }
}
