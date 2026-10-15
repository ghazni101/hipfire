// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Expert-slab support of the product Qwen4 carrier.
//!
//! A version-1 expert-slab artifact declares one synthetic `slab_arena` entry
//! per trunk layer instead of the aggregate expert tensors. That entry is not
//! an index tensor of the container: its physical size is the recipe's arena
//! extent and its payload is one span of the container. These two helpers
//! route the carrier's per-entry byte sizing (placement inputs) and weight
//! fulfillment through the recipe's arena plan first, and leave every other
//! entry on the unchanged by-name path.

use hipfire_arch_qwen4::expert_arena::{recipe_arena_payload, ARENA_SUFFIX};
use hipfire_arch_qwen4::weights::Qwen4TierRecipe;
use hipfire_runtime::hfq::HfqModelSource;
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::weight_manifest::WeightEntry;

/// Physical source bytes of `entry` for placement sizing.
///
/// Under [`Qwen4TierRecipe::FlashNextSlab`] a layer arena entry is sized from
/// the recipe's plan (its physical arena bytes; the allocation tail pad is the
/// weight store's own) and never looked up in the tensor index, where it does
/// not exist: an arena naming no planned layer is `None`. Every other entry,
/// and every entry of any other recipe, is sized by `ordinary` exactly as
/// before.
pub(crate) fn qwen4_entry_bytes(
    recipe: &Qwen4TierRecipe,
    entry: &WeightEntry,
    ordinary: impl FnOnce(&WeightEntry) -> Option<u64>,
) -> Option<u64> {
    if let Qwen4TierRecipe::FlashNextSlab { arenas, .. } = recipe {
        if entry.name.ends_with(ARENA_SUFFIX) {
            return entry
                .layer
                .and_then(|layer| arenas.layer(layer))
                .map(|layer| layer.bytes as u64);
        }
    }
    ordinary(entry)
}

/// The fulfillment payload of one manifest entry: a slab layer arena (one
/// container span) when the recipe declares it, otherwise the tensor read the
/// carrier always made. `use_range` is the caller's
/// `qwen4_use_range_payload` decision (unified memory or an external entry
/// reads a checked source range; a resident discrete entry borrows the mapped
/// bytes).
pub(crate) fn qwen4_fulfillment_payload<'a>(
    source: &'a HfqModelSource,
    recipe: &Qwen4TierRecipe,
    entry: &WeightEntry,
    use_range: bool,
) -> Result<SourcePayload<'a>, String> {
    if let Some(payload) = recipe_arena_payload(source, recipe, entry) {
        return payload;
    }
    if use_range {
        return source
            .tensor_range(&entry.name)
            .map_err(|error| error.to_string())?
            .map(SourcePayload::Range)
            .ok_or_else(|| {
                if entry.residency.is_external() {
                    format!("missing external tensor '{}'", entry.name)
                } else {
                    format!("missing resident tensor '{}'", entry.name)
                }
            });
    }
    let (info, bytes) = source
        .tensor_data(&entry.name)
        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
    Ok(SourcePayload::Borrowed { info, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_qwen4::expert_arena::ExpertArenaPlan;
    use hipfire_arch_qwen4::expert_slab_table::build_expert_slab_table;
    use hipfire_arch_qwen4::expert_slabs::{
        expert_entry_name, ExpertFormat, ExpertProj, ExpertSlabsV1, EXPERTS_PER_LAYER,
        TRUNK_LAYERS,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor, HfqTensorInfo};
    use hipfire_runtime::weight_manifest::{ShardPolicy, WeightResidency};
    use rdna_compute::DType;
    use std::sync::Arc;

    /// The all-MQ3-I4 version-1 index: every trunk expert, back to back from
    /// 4096, in canonical order (the table builder's own layout).
    fn all_w3_plan() -> Arc<ExpertArenaPlan> {
        let format = ExpertFormat::Mq3I4;
        let mut tensors = Vec::new();
        let mut cursor = 4096usize;
        for layer in 0..TRUNK_LAYERS as usize {
            for expert in 0..EXPERTS_PER_LAYER as usize {
                let (gate_qt, down_qt) = format.pair();
                for (proj, qt, bytes) in [
                    (ExpertProj::GateUp, gate_qt, format.gate_up_bytes()),
                    (ExpertProj::Down, down_qt, format.down_bytes()),
                ] {
                    tensors.push(HfqTensorInfo {
                        name: expert_entry_name(layer, expert, proj),
                        quant_type: qt,
                        shape: proj.shape().to_vec(),
                        group_size: 128,
                        data_offset: cursor,
                        data_size: bytes,
                    });
                    cursor += bytes;
                }
            }
        }
        let table = build_expert_slab_table(&ExpertSlabsV1::fn_v1(), &tensors)
            .expect("all-W3 index is a valid version-1 layout");
        Arc::new(ExpertArenaPlan::from_table(&table).expect("contiguous arenas"))
    }

    fn slab_recipe(plan: &Arc<ExpertArenaPlan>) -> Qwen4TierRecipe {
        Qwen4TierRecipe::FlashNextSlab {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
            arenas: plan.clone(),
        }
    }

    fn packed_recipe() -> Qwen4TierRecipe {
        Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        }
    }

    fn arena_entry(layer: usize, bytes: usize, residency: WeightResidency) -> WeightEntry {
        let mut entry = WeightEntry::model(
            ExpertArenaPlan::arena_name(&format!("model.language_model.layers.{layer}")),
            vec![bytes],
            DType::Raw,
            ShardPolicy::Replicate,
        );
        entry.layer = Some(layer);
        entry.residency = residency;
        entry
    }

    fn ordinary_entry(name: &str, layer: Option<usize>) -> WeightEntry {
        let mut entry = WeightEntry::model(name, vec![4, 4], DType::F32, ShardPolicy::Replicate);
        entry.layer = layer;
        entry
    }

    #[test]
    fn arena_entries_are_sized_from_the_recipe_plan_without_a_tensor_lookup() {
        let plan = all_w3_plan();
        let recipe = slab_recipe(&plan);
        let expected = (ExpertFormat::Mq3I4.slab_bytes() * EXPERTS_PER_LAYER as usize) as u64;
        for layer in [0usize, 1, 47] {
            let entry = arena_entry(layer, expected as usize, WeightResidency::Resident);
            let bytes = qwen4_entry_bytes(&recipe, &entry, |_| {
                panic!("an arena entry must not be looked up by name")
            });
            assert_eq!(bytes, Some(expected));
            assert_eq!(bytes, Some(plan.layer(layer).unwrap().bytes as u64));
        }
        // Host-mapped arenas size identically (residency is placement, not size).
        let host = arena_entry(3, expected as usize, WeightResidency::HostMapped);
        assert_eq!(
            qwen4_entry_bytes(&recipe, &host, |_| panic!("no lookup")),
            Some(expected)
        );
    }

    #[test]
    fn ordinary_entries_keep_the_by_name_size_under_every_recipe() {
        let plan = all_w3_plan();
        let entry = ordinary_entry("model.language_model.layers.0.self_attn.q_proj.weight", Some(0));
        for recipe in [slab_recipe(&plan), packed_recipe()] {
            assert_eq!(
                qwen4_entry_bytes(&recipe, &entry, |e| {
                    assert_eq!(e.name, entry.name);
                    Some(1234)
                }),
                Some(1234)
            );
            assert_eq!(qwen4_entry_bytes(&recipe, &entry, |_| None), None);
        }
    }

    #[test]
    fn arena_named_entry_outside_a_slab_recipe_stays_a_by_name_lookup() {
        // A packed (non-slab) manifest declares no arenas: the name is just a
        // name, sized (or missing) exactly as before.
        let entry = arena_entry(0, 100, WeightResidency::Resident);
        assert_eq!(qwen4_entry_bytes(&packed_recipe(), &entry, |_| Some(77)), Some(77));
        assert_eq!(qwen4_entry_bytes(&packed_recipe(), &entry, |_| None), None);
    }

    #[test]
    fn an_arena_naming_no_planned_layer_is_missing() {
        let plan = all_w3_plan();
        let recipe = slab_recipe(&plan);
        let past_end = arena_entry(TRUNK_LAYERS as usize, 100, WeightResidency::Resident);
        assert_eq!(
            qwen4_entry_bytes(&recipe, &past_end, |_| panic!("no lookup")),
            None
        );
        let mut unlayered = arena_entry(0, 100, WeightResidency::Resident);
        unlayered.layer = None;
        assert_eq!(
            qwen4_entry_bytes(&recipe, &unlayered, |_| panic!("no lookup")),
            None
        );
    }

    /// A one-tensor HFQM container: enough source for the by-name path and a
    /// proof that arena entries never reach it.
    fn tiny_source(tag: &str) -> (HfqModelSource, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-qwen4-slab-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.hfq");
        let tensors = [HfqMemTensor {
            name: "model.embed_tokens.weight".into(),
            quant_type: 1,
            shape: vec![4, 4],
            group_size: 0,
            data: (0u8..32).collect(),
        }];
        write_hfqm_package_mem(&path, 16, "{}", &tensors).unwrap();
        let hfq = HfqFile::open(&path).expect("tiny container opens");
        (HfqModelSource::from_hfq(hfq), path)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    #[test]
    fn payload_routes_arenas_to_the_plan_and_everything_else_by_name() {
        let (source, path) = tiny_source("route");
        let plan = all_w3_plan();
        let recipe = slab_recipe(&plan);
        let arena_bytes = plan.layer(0).unwrap().bytes;

        // An ordinary tensor under the slab recipe is the unchanged borrow.
        let embed = ordinary_entry("model.embed_tokens.weight", None);
        match qwen4_fulfillment_payload(&source, &recipe, &embed, false) {
            Ok(SourcePayload::Borrowed { bytes, .. }) => {
                assert_eq!(bytes, (0u8..32).collect::<Vec<_>>().as_slice())
            }
            other => panic!(
                "ordinary tensor must borrow its bytes, got {:?}",
                other.map(|_| ())
            ),
        }

        // A name the index lacks keeps the legacy diagnostic.
        let missing = ordinary_entry("model.nope.weight", None);
        assert_eq!(
            qwen4_fulfillment_payload(&source, &recipe, &missing, false).map(|_| ()),
            Err("missing resident tensor 'model.nope.weight'".to_string())
        );

        // A host-mapped arena reads the plan's container span (this tiny file
        // does not hold it), so the error is the arena's, not a missing tensor.
        let host = arena_entry(0, arena_bytes, WeightResidency::HostMapped);
        let error = qwen4_fulfillment_payload(&source, &recipe, &host, false)
            .map(|_| ())
            .expect_err("the tiny file holds no arena span");
        assert!(
            error.contains("slab arena") && !error.contains("missing resident tensor"),
            "arena entry must be routed through the plan first: {error}"
        );

        // A mis-declared arena shape is the plan's refusal, again not a lookup.
        let wrong = arena_entry(0, arena_bytes + 1, WeightResidency::Resident);
        let error = qwen4_fulfillment_payload(&source, &recipe, &wrong, true)
            .map(|_| ())
            .expect_err("shape disagrees with the plan");
        assert!(error.contains("declares"), "{error}");

        // Outside a slab recipe the same name is an ordinary missing tensor.
        let error = qwen4_fulfillment_payload(&source, &packed_recipe(), &host, false)
            .map(|_| ())
            .expect_err("a packed recipe has no arenas");
        assert_eq!(error, format!("missing resident tensor '{}'", host.name));
        cleanup(&path);
    }
}
