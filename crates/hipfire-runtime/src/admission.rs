// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// AdmissionController — decides whether a session can be admitted, and with how
// much context.
//
// In the test harnesses `kv_slots::preflight_alloc` is what stops an oversized
// configuration. In the daemon that job is HERE. The difference matters: on this
// hardware the GPU allocates from system RAM and the cgroup does NOT contain
// amdgpu GTT, so a wrong decision here does not fail a request — it takes down
// the user's desktop with a global OOM.

/// What one loaded model costs, split into the part charged once and the part
/// charged per session.
#[derive(Debug, Clone, Copy)]
pub struct ModelFootprint {
    /// Charged ONCE, however many sessions are admitted.
    pub weights_bytes: u64,
    /// Charged per session, per token of granted context.
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitError {
    PoolFull,
    WouldExceedBudget { need: u64, available: u64 },
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gib = |b: u64| b as f64 / 1073741824.0;
        match self {
            AdmitError::PoolFull => write!(f, "no free slot"),
            AdmitError::WouldExceedBudget { need, available } => write!(
                f,
                "needs {:.2} GiB but only {:.2} GiB of the budget remains",
                gib(*need),
                gib(*available)
            ),
        }
    }
}

pub struct AdmissionController {
    footprint: ModelFootprint,
    budget_bytes: u64,
    /// Granted context per admitted session, in tokens.
    admitted: Vec<usize>,
    /// Host-tier budget for swapped-out snapshots. Separate from the VRAM
    /// budget: admission is the production memory gate for BOTH, because the
    /// control group does not contain amdgpu GTT.
    host_budget: u64,
    host_used: u64,
}

impl AdmissionController {
    pub fn new(footprint: ModelFootprint, budget_bytes: u64) -> Self {
        Self {
            footprint,
            budget_bytes,
            admitted: Vec::new(),
            host_budget: crate::swap::DEFAULT_HOST_BUDGET_BYTES,
            host_used: 0,
        }
    }

    /// Bytes currently committed: weights once (if anything is admitted) plus
    /// each session's KV.
    pub fn used_bytes(&self) -> u64 {
        if self.admitted.is_empty() {
            return 0;
        }
        let kv: u64 = self
            .admitted
            .iter()
            .map(|&ctx| ctx as u64 * self.footprint.kv_bytes_per_token)
            .sum();
        self.footprint.weights_bytes + kv
    }

    /// Admit a session at `requested_ctx` tokens, or explain why not.
    ///
    /// Rejects rather than silently capping: a caller that asked for 128K and
    /// silently got 8K would produce baffling truncation far from here.
    pub fn admit(&mut self, requested_ctx: usize) -> Result<usize, AdmitError> {
        let kv_need = requested_ctx as u64 * self.footprint.kv_bytes_per_token;
        // Weights are charged once, on the first admission.
        let weights_need = if self.admitted.is_empty() {
            self.footprint.weights_bytes
        } else {
            0
        };
        let need = kv_need + weights_need;
        let available = self.budget_bytes.saturating_sub(self.used_bytes());
        // >= rather than >: an admission that would consume the LAST byte of
        // budget is refused too, not just one that overflows it. On this
        // hardware (no swap, cgroup does not contain amdgpu GTT) landing
        // exactly on the edge leaves zero headroom for anything else running
        // on the box, so it is treated the same as exceeding the budget.
        if need >= available {
            return Err(AdmitError::WouldExceedBudget { need, available });
        }
        self.admitted.push(requested_ctx);
        Ok(requested_ctx)
    }

    /// Return a session's context allowance to the budget.
    /// Reserve host-tier bytes for a swapped-out session. Returns false when
    /// the budget cannot cover it, in which case the caller spills to disk
    /// rather than exceeding the budget.
    pub fn admit_host(&mut self, bytes: u64) -> bool {
        if self.host_used.saturating_add(bytes) > self.host_budget {
            return false;
        }
        self.host_used += bytes;
        true
    }

    pub fn release_host(&mut self, bytes: u64) {
        self.host_used = self.host_used.saturating_sub(bytes);
    }

    pub fn host_used_bytes(&self) -> u64 {
        self.host_used
    }

    pub fn host_budget_bytes(&self) -> u64 {
        self.host_budget
    }

    /// Set the host-tier budget. Defaults to `DEFAULT_HOST_BUDGET_BYTES`.
    pub fn set_host_budget(&mut self, bytes: u64) {
        self.host_budget = bytes;
    }

    pub fn release(&mut self, granted_ctx: usize) {
        if let Some(i) = self.admitted.iter().position(|&c| c == granted_ctx) {
            self.admitted.remove(i);
        }
    }
}


// =========================================================================
// S1 physical capacity accounting (spec §5.1)
// =========================================================================

/// Page size in tokens. Matches `rdna-compute::page_pool::PAGE_TOKENS`.
pub const PAGE_TOKENS: u64 = 128;

/// KV bytes for one 128-token page bundle (spec §5.1).
///
/// `page_bytes = sum_attention_layers B * (k_stride_bytes[layer] + v_stride_bytes[layer])`
/// where `B = 128` (`PAGE_TOKENS`). Strides include quant scales/headers.
/// Returns `None` on stride-length mismatch or arithmetic overflow (checked
/// arithmetic, spec §5.1: "Use checked arithmetic").
pub fn page_bytes(k_strides: &[u64], v_strides: &[u64]) -> Option<u64> {
    if k_strides.len() != v_strides.len() {
        return None;
    }
    let mut total: u64 = 0;
    for (&k, &v) in k_strides.iter().zip(v_strides.iter()) {
        let per_layer = PAGE_TOKENS.checked_mul(k.checked_add(v)?)?;
        total = total.checked_add(per_layer)?;
    }
    Some(total)
}

/// Typed capacity error for the serving admission path (spec §5.1/S1, §5.4/S4).
///
/// Distinguishes pool exhaustion from arithmetic overflow so a caller can
/// fail closed on an accounting fault rather than busy-loop retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityError {
    /// The requested bytes would exceed the remaining pool capacity.
    WouldExceedPool { need: u64, available: u64 },
    /// Checked arithmetic overflowed during accounting.
    ArithmeticOverflow,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WouldExceedPool { need, available } => write!(
                f,
                "capacity exceeded: needs {need} bytes but {available} remain"
            ),
            Self::ArithmeticOverflow => write!(f, "capacity accounting overflow"),
        }
    }
}

impl std::error::Error for CapacityError {}

/// Physical capacity accounting for the serving cache/scheduler (spec §5.1/S1).
///
/// Tracks four separate quantities that must not be conflated:
/// - **allocated pool capacity** — total bytes the page pool can hold
/// - **uniquely resident page bytes** — shared pages counted once, not per ref
/// - **unmaterialized growth/COW credits** — reserved for future private suffix
///   and copy-on-write tail, not yet allocated as physical pages
/// - **logical context limits** — per-request max token grants (not physical
///   bytes; `max_seq` is not proof of physical allocation)
///
/// Invariant (spec §5.1): `resident_page_bytes + growth_credits_bytes ≤
/// pool_capacity_bytes`. All arithmetic is checked; overflow is a
/// [`CapacityError::ArithmeticOverflow`], not silent wraparound.
#[derive(Debug, Clone)]
pub struct ServeCapacityAccount {
    pool_capacity_bytes: u64,
    resident_page_bytes: u64,
    growth_credits_bytes: u64,
    logical_ctx_limit_tokens: usize,
}

impl ServeCapacityAccount {
    pub fn new(pool_capacity_bytes: u64, logical_ctx_limit_tokens: usize) -> Self {
        Self {
            pool_capacity_bytes,
            resident_page_bytes: 0,
            growth_credits_bytes: 0,
            logical_ctx_limit_tokens,
        }
    }

    /// Bytes remaining under the pool capacity invariant.
    pub fn available_bytes(&self) -> u64 {
        self.pool_capacity_bytes
            .saturating_sub(self.resident_page_bytes)
            .saturating_sub(self.growth_credits_bytes)
    }

    pub fn pool_capacity_bytes(&self) -> u64 {
        self.pool_capacity_bytes
    }

    pub fn resident_page_bytes(&self) -> u64 {
        self.resident_page_bytes
    }

    pub fn growth_credits_bytes(&self) -> u64 {
        self.growth_credits_bytes
    }

    pub fn logical_ctx_limit_tokens(&self) -> usize {
        self.logical_ctx_limit_tokens
    }

    /// Charge `bytes` of uniquely resident page bytes (spec §5.1).
///
/// Shared physical prefix pages count once; the request's future private
/// suffix and recurrent state count separately via [`Self::reserve_growth`].
    pub fn charge_resident(&mut self, bytes: u64) -> Result<(), CapacityError> {
        let new_resident = self
            .resident_page_bytes
            .checked_add(bytes)
            .ok_or(CapacityError::ArithmeticOverflow)?;
        let committed = new_resident
            .checked_add(self.growth_credits_bytes)
            .ok_or(CapacityError::ArithmeticOverflow)?;
        if committed > self.pool_capacity_bytes {
            return Err(CapacityError::WouldExceedPool {
                need: bytes,
                available: self.available_bytes(),
            });
        }
        self.resident_page_bytes = new_resident;
        Ok(())
    }

    /// Release `bytes` of resident page bytes back to the pool.
    pub fn release_resident(&mut self, bytes: u64) {
        self.resident_page_bytes = self.resident_page_bytes.saturating_sub(bytes);
    }

    /// Reserve `bytes` of unmaterialized growth/COW credits (spec §5.1).
///
/// Credits turn into allocated private pages as work advances. Allocation
/// must not exceed the already granted credits.
    pub fn reserve_growth(&mut self, bytes: u64) -> Result<(), CapacityError> {
        let new_credits = self
            .growth_credits_bytes
            .checked_add(bytes)
            .ok_or(CapacityError::ArithmeticOverflow)?;
        let committed = self
            .resident_page_bytes
            .checked_add(new_credits)
            .ok_or(CapacityError::ArithmeticOverflow)?;
        if committed > self.pool_capacity_bytes {
            return Err(CapacityError::WouldExceedPool {
                need: bytes,
                available: self.available_bytes(),
            });
        }
        self.growth_credits_bytes = new_credits;
        Ok(())
    }

    /// Release `bytes` of growth credits (e.g. request cancelled before
    /// materializing its growth).
    pub fn release_growth(&mut self, bytes: u64) {
        self.growth_credits_bytes = self.growth_credits_bytes.saturating_sub(bytes);
    }

    /// Materialize `bytes` of growth credits into resident page bytes
/// (spec §5.1: "Credits turn into allocated private pages as work
/// advances"). The credits are released and the bytes are charged as
/// resident in one checked operation.
    pub fn materialize_growth(&mut self, bytes: u64) -> Result<(), CapacityError> {
        self.release_growth(bytes);
        self.charge_resident(bytes)
    }

    /// Check whether `bytes` would fit under the pool capacity invariant
/// without mutating state.
    pub fn fits(&self, bytes: u64) -> bool {
        self.available_bytes().checked_sub(bytes).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// qwen3.6:27b — 15.0 GB of weights, 34 KB of KV per token.
    fn f27b() -> ModelFootprint {
        ModelFootprint {
            weights_bytes: 15 * GIB,
            kv_bytes_per_token: 34 * 1024,
        }
    }

    /// qwen3.6:35b-a3b — ~20 GB of weights, 10.6 KB of KV per token.
    fn f35b() -> ModelFootprint {
        ModelFootprint {
            weights_bytes: 20 * GIB,
            kv_bytes_per_token: 10_854,
        }
    }

    #[test]
    fn weights_are_charged_once_not_per_session() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        a.admit(1024).unwrap();
        let after_one = a.used_bytes();
        a.admit(1024).unwrap();
        let after_two = a.used_bytes();
        // The second session adds only its KV, never another copy of the weights.
        assert!(after_two - after_one < GIB, "weights charged twice");
        assert!(after_one >= 15 * GIB, "weights not charged at all");
    }

    #[test]
    fn the_27b_cannot_take_four_agents_at_128k() {
        // 15 GB + 4 x 4.25 GB = 32.25 GB against a 32 GB card.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).expect("first three must fit");
        }
        let e = a.admit(128 * 1024).unwrap_err();
        assert!(
            matches!(e, AdmitError::WouldExceedBudget { .. }),
            "got {e:?}"
        );
    }

    #[test]
    fn the_27b_does_take_four_agents_at_96k() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for i in 0..4 {
            a.admit(96 * 1024)
                .unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn the_35b_does_take_four_agents_at_128k() {
        let mut a = AdmissionController::new(f35b(), 32 * GIB);
        for i in 0..4 {
            a.admit(128 * 1024)
                .unwrap_or_else(|e| panic!("agent {i} rejected: {e:?}"));
        }
    }

    #[test]
    fn release_returns_budget_so_a_later_session_fits() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).unwrap();
        }
        assert!(a.admit(128 * 1024).is_err());
        a.release(128 * 1024);
        a.admit(128 * 1024)
            .expect("budget must be reusable after release");
    }

    #[test]
    fn rejection_reports_the_numbers_not_just_a_failure() {
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        for _ in 0..3 {
            a.admit(128 * 1024).unwrap();
        }
        match a.admit(128 * 1024).unwrap_err() {
            AdmitError::WouldExceedBudget { need, available } => {
                // `>=`, not `>`. Zero headroom is a rejection: 15 GiB of weights
                // plus 4 x 4.25 GiB of KV is an EXACT tie with a 32 GiB budget,
                // and a card with nothing left for activations, scratch and
                // driver overhead does not fit the workload. The plan's comment
                // claiming 32.25 GB was wrong -- 34 * 1024 IS the real per-token
                // cost and the sum lands exactly on the budget.
                assert!(
                    need >= available,
                    "need {need} should be at least available {available}"
                );
                assert!(available < 32 * GIB);
            }
            other => panic!("expected a budget rejection, got {other:?}"),
        }
    }

    #[test]
    fn a_single_session_over_budget_is_rejected_not_silently_capped() {
        // One agent asking for more than the whole card can hold.
        let mut a = AdmissionController::new(f27b(), 32 * GIB);
        assert!(
            a.admit(2 * 1024 * 1024).is_err(),
            "must reject, not silently truncate"
        );
    }

    #[test]
    fn the_host_tier_has_its_own_budget() {
        let mut a = AdmissionController::new(
            ModelFootprint {
                weights_bytes: 0,
                kv_bytes_per_token: 0,
            },
            1 << 30,
        );
        a.set_host_budget(1000);
        assert!(a.admit_host(600));
        assert_eq!(a.host_used_bytes(), 600);
        assert!(
            !a.admit_host(600),
            "the second must not fit; the caller spills to disk instead"
        );
        assert_eq!(a.host_used_bytes(), 600, "a refused admit reserves nothing");
        a.release_host(600);
        assert_eq!(a.host_used_bytes(), 0);
        assert!(a.admit_host(600), "released budget must be reusable");
    }

    // ---- page_bytes helper (spec §5.1) ----

    #[test]
    fn page_bytes_computes_sum_over_layers() {
        // 2 layers, k_stride=128, v_stride=64 → per layer: 128*(128+64) = 24576
        // total: 2 * 24576 = 49152
        let k = [128u64, 128];
        let v = [64u64, 64];
        assert_eq!(page_bytes(&k, &v), Some(49152));
    }

    #[test]
    fn page_bytes_single_layer() {
        // 1 layer, k=256, v=128 → 128*(256+128) = 49152
        assert_eq!(page_bytes(&[256], &[128]), Some(49152));
    }

    #[test]
    fn page_bytes_mismatched_strides_returns_none() {
        assert_eq!(page_bytes(&[128, 128], &[64]), None);
        assert_eq!(page_bytes(&[128], &[64, 64]), None);
    }

    #[test]
    fn page_bytes_overflow_returns_none() {
        // u64::MAX stride would overflow when multiplied by PAGE_TOKENS.
        assert_eq!(page_bytes(&[u64::MAX], &[1]), None);
    }

    #[test]
    fn page_bytes_empty_layers_is_zero() {
        assert_eq!(page_bytes(&[], &[]), Some(0));
    }

    // ---- ServeCapacityAccount (spec §5.1/S1) ----

    #[test]
    fn capacity_account_charges_and_releases_resident() {
        let mut acct = ServeCapacityAccount::new(1024, 8192);
        assert_eq!(acct.available_bytes(), 1024);
        assert!(acct.charge_resident(400).is_ok());
        assert_eq!(acct.resident_page_bytes(), 400);
        assert_eq!(acct.available_bytes(), 624);
        acct.release_resident(200);
        assert_eq!(acct.resident_page_bytes(), 200);
        assert_eq!(acct.available_bytes(), 824);
    }

    #[test]
    fn capacity_account_rejects_resident_over_pool() {
        let mut acct = ServeCapacityAccount::new(1000, 8192);
        assert!(acct.charge_resident(600).is_ok());
        let err = acct.charge_resident(500).unwrap_err();
        assert_eq!(err, CapacityError::WouldExceedPool { need: 500, available: 400 });
    }

    #[test]
    fn capacity_account_reserves_and_releases_growth_credits() {
        let mut acct = ServeCapacityAccount::new(1000, 8192);
        assert!(acct.reserve_growth(300).is_ok());
        assert_eq!(acct.growth_credits_bytes(), 300);
        assert_eq!(acct.available_bytes(), 700);
        // Resident + growth must not exceed pool.
        assert!(acct.charge_resident(800).is_err());
        assert!(acct.charge_resident(600).is_ok());
        acct.release_growth(200);
        assert_eq!(acct.growth_credits_bytes(), 100);
    }

    #[test]
    fn capacity_account_materialize_growth_converts_credit_to_resident() {
        let mut acct = ServeCapacityAccount::new(1000, 8192);
        assert!(acct.reserve_growth(500).is_ok());
        assert_eq!(acct.growth_credits_bytes(), 500);
        assert_eq!(acct.resident_page_bytes(), 0);
        assert!(acct.materialize_growth(300).is_ok());
        assert_eq!(acct.growth_credits_bytes(), 200);
        assert_eq!(acct.resident_page_bytes(), 300);
    }

    #[test]
    fn capacity_account_overflow_is_typed_error() {
        let mut acct = ServeCapacityAccount::new(u64::MAX, 8192);
        // Charge near-max to set up overflow on the next add.
        acct.charge_resident(u64::MAX - 10).ok();
        let err = acct.charge_resident(20).unwrap_err();
        assert_eq!(err, CapacityError::ArithmeticOverflow);
    }

    #[test]
    fn capacity_account_fits_is_non_mutating() {
        let mut acct = ServeCapacityAccount::new(1000, 8192);
        assert!(acct.fits(500));
        assert!(!acct.fits(1500));
        // fits must not mutate state.
        assert_eq!(acct.available_bytes(), 1000);
        acct.charge_resident(500).ok();
        assert!(acct.fits(500));
        assert!(!acct.fits(501));
    }
}
