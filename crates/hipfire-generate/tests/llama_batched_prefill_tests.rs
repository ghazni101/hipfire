// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Llama batched prefill tests.
//!
//! Moved out of `hipfire-daemon`'s `main.rs`. Compiled into a bin crate these
//! never appeared as their own test target; as integration tests they are
//! reported individually.

#![allow(unused_imports, dead_code, clippy::all)]

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::ar::*;
use hipfire_generate::batch::*;
use hipfire_generate::common::*;

    
    use hipfire_runtime::llama::ModelArch;

    #[test]
    fn route_stays_inside_validated_envelopes() {
        // (arch, model, enabled, q8, eviction, tokens, v2_weights, expected)
        // Qwen3: the original validated envelope. Llama: admitted only with
        // all-V2 weights (K2-Horizon) — a plain-llama model keeps the
        // per-token route, as does any non-V2 or gfx10 layout.
        let cases = [
            ("gfx1100", ModelArch::Qwen3, true, true, false, 256, false, true),
            ("gfx1201", ModelArch::Qwen3, true, true, false, 4, false, true),
            ("gfx1200", ModelArch::Qwen3, true, true, false, 256, false, false),
            ("gfx1100", ModelArch::Llama, true, true, false, 256, false, false),
            ("gfx1100", ModelArch::Qwen3, true, false, false, 256, false, false),
            ("gfx1100", ModelArch::Qwen3, true, true, true, 256, false, false),
            ("gfx1100", ModelArch::Qwen3, true, true, false, 3, false, false),
            ("gfx1100", ModelArch::Qwen3, false, true, false, 256, false, false),
            // K2-Horizon: Llama arch + all-MQ4G256V2 weights.
            ("gfx1100", ModelArch::Llama, true, true, false, 256, true, true),
            ("gfx1201", ModelArch::Llama, true, true, false, 8, true, true),
            ("gfx1100", ModelArch::Llama, true, true, false, 3, true, false),
            ("gfx1100", ModelArch::Llama, true, false, false, 256, true, false),
            ("gfx1100", ModelArch::Llama, true, true, true, 256, true, false),
            ("gfx1030", ModelArch::Llama, true, true, false, 256, true, false),
        ];
        for (arch, model, enabled, q8, eviction, tokens, v2, expected) in cases {
            assert_eq!(
                llama_qwen3_batched_prefill_eligible(
                    arch, model, enabled, q8, eviction, tokens, v2,
                ),
                expected,
                "arch={arch} model={model:?} v2={v2}",
            );
        }
    }

    #[test]
    fn sampled_prefill_preserves_discarded_xorshift_draws() {
        assert_eq!(llama_prefill_sample_seed(42, 4, 0.0), 42);
        assert_eq!(llama_prefill_sample_seed(42, 1, 1.0), 42);
        assert_eq!(llama_prefill_sample_seed(42, 4, 1.0), 476_557_059);
    }
