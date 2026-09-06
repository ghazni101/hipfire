// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-k2-horizon: K2-Horizon architecture (MoVA attention +
//! sigmoid-routed MoE FFN).
//!
//! Implements the [`hipfire_runtime::arch::Architecture`] trait for the
//! K2-Horizon model family (IFM/K2-Horizon-MoVA-36B-A4B and future
//! variants). Key architectural features vs the qwen3.5-MoE baseline:
//!
//! - **MoVA attention** — value states come from a mixture of 64 value
//!   experts (4 per token), routed by a sigmoid router, not a single
//!   `v_proj`. Followed by a softplus post-attention gate.
//! - **Sigmoid-routed MoE FFN** — 100 routed experts + 1 shared expert,
//!   sigmoid router (not softmax) with 2.5× scaling and bias-for-selection.
//! - **Grouped RMSNorm** — variance computed per `hidden_size /
//!   layernorm_num_groups` chunk (n_groups=2), not over the full vector.
//! - **Dense prefix** — layers 0–2 are standard attention + dense MLP
//!   (`mlp_only_layers: [0, 1, 2]`).
//!
//! See `docs/plans/k2-horizon-arch-spec.md` for the full design.

pub mod arch;
pub mod config;
pub mod forward;
pub mod load;
pub mod weights;

pub use arch::K2Horizon;
pub use config::K2HorizonConfig;
pub use forward::K2HorizonState;
pub use load::K2HorizonBundle;
