// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K2-Horizon weight loading from HFQ + bundle assembly.
//!
//! Mirrors the cohere2moe carrier pattern: `load_k2_horizon_bundle`
//! is called by `K2HorizonCarrier::load` in the loader crate.

use crate::arch::K2Horizon;
use crate::config::K2HorizonConfig;
use crate::forward::K2HorizonState;
use crate::weights::K2HorizonWeights;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use rdna_compute::Gpu;

/// The loaded K2-Horizon model — config + weights + state + eos.
///
/// Stored as `Box<dyn ArchModel>` inside `LoadedModel.state`; the daemon
/// downcasts via `k2_horizon_mut()` to call `forward::decode_step`.
pub struct K2HorizonBundle {
    pub config: K2HorizonConfig,
    pub weights: K2HorizonWeights,
    pub state: K2HorizonState,
    pub eos_tok: u32,
}

impl ArchModel for K2HorizonBundle {
    fn dim(&self) -> usize {
        self.config.dim
    }

    fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "k2_horizon"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        Some(&mut self.state.kv)
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        // Drop weights (frees WeightTensor GPU buffers) then state (frees
        // scratch + KV cache). Order doesn't matter — both are GPU-owned.
        let bundle = *self;
        bundle.state.free_gpu(gpu);
        // Weights' WeightTensor drop handles GPU free internally.
        drop(bundle.weights);
    }
}

/// Build the K2-Horizon GPU bundle from an HFQ source.
///
/// Dir (safetensors) source is not supported — K2-Horizon is HFQ-only
/// (quantize first, then serve).
pub fn load_k2_horizon_bundle(
    src: ModelSource,
    ctx: &mut LoadCtx,
) -> Result<K2HorizonBundle, String> {
    match src {
        ModelSource::Hfq(mut hfq) => {
            let config = <K2Horizon as Architecture>::config_from_hfq(&hfq)?;
            let weights =
                <K2Horizon as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
            let state = K2HorizonState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("k2_horizon: new_with_max_seq failed: {e}"))?;

            // EOS: K2-Horizon uses <|ifm|endoftext|> (id 1) as primary EOS.
            // The config carries eos_token_id; fall back to 1.
            let eos_tok = config.eos_token;

            Ok(K2HorizonBundle {
                config,
                weights,
                state,
                eos_tok,
            })
        }
        ModelSource::Dir(_) => {
            Err("k2_horizon: safetensors-dir source not supported — quantize first, then serve the HFQ".into())
        }
    }
}
