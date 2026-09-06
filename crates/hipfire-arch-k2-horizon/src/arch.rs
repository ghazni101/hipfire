// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Architecture` trait implementation for K2-Horizon.
//!
//! The trait surface provides config parsing, weight loading, and state
//! allocation entry points. Forward-pass dispatch is NOT routed through
//! the trait — the daemon calls arch-specific forward functions directly,
//! matching the qwen35 pattern (see `hipfire-arch-qwen35/src/arch.rs`).
//!
//! `load_weights` returns "not yet implemented" until Phase 6 (loader).
//! `new_state` is implemented via `K2HorizonState::new`.

use crate::config::{config_from_hfq, K2HorizonConfig};
use crate::forward::K2HorizonState;
use crate::weights::K2HorizonWeights;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for the K2-Horizon architecture (MoVA attention +
/// sigmoid-routed MoE FFN). arch_id = 15.
pub struct K2Horizon;

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
        _hfq: &mut HfqFile,
        _cfg: &Self::Config,
        _gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        Err("k2_horizon: load_weights not yet implemented — Phase 6".into())
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        K2HorizonState::new(gpu, cfg)
    }
}
