// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Loader-facing bundle construction for Maple-Preview.
//!
//! Mirrors `hipfire_arch_cohere2moe::carrier`. HFQ only: there is no
//! safetensors-directory path, because the published checkpoint is 40 GB of
//! dequantized BF16 masters whose whole point is to be packed losslessly into
//! qt=51 first — serving it unpacked would need a BF16 MoE decode path that
//! does not exist and would be ~7× the memory.

use crate::bundle::{load_maple_from_hfq, MapleBundle};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

/// Build the Maple GPU bundle from a loader `ModelSource`.
pub fn load_maple_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<MapleBundle, String> {
    if ctx.pp > 1 {
        return Err("maple: pp>1 unsupported via registry".into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => load_maple_from_hfq(&mut hfq, ctx.gpu, ctx.max_seq),
        ModelSource::Dir(_) => Err(
            "maple: safetensors-directory loading is unsupported — convert first with \
             `hipfire-quantize --format maple --input <dir> --output <model.hfq>`, which packs \
             the native ternary weights losslessly into qt=51"
                .into(),
        ),
    }
}
