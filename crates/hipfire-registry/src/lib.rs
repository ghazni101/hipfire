// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed model-registry loading and model identity resolution.
//!
//! The checked-in v1 registry is compiled into every binary as the offline
//! floor. Dynamic loading preserves the established fallback order: fresh
//! cache, network, stale cache, then bundled. A malformed registry is rejected
//! wholesale; model code never consumes a partially validated catalog.

use hipfire_config::{ConfigLayer, ConfigValue};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
pub mod xdna;
pub use xdna::{XdnaManifest, XdnaProfile, XdnaSidecarDescriptor};

pub const REGISTRY_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/warpfront/hipfire/master/registry/v1.json";
pub const REGISTRY_CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub const REGISTRY_FETCH_TIMEOUT: Duration = Duration::from_millis(3500);
const BUNDLED_REGISTRY: &str = include_str!("../../../registry/v1.json");

pub const FLASH_NEXT_XTS_LEGACY_FILE: &str = "qwen3.8-flash-next-gptq3.mq4";
pub const FLASH_NEXT_XTS_CANONICAL_FILE: &str = "qwen3.8-flash-next.mq4-xts";
const FLASH_NEXT_XTS_REPO: &str = "hipfire-models/qwen3.8-flash-next";
const FLASH_NEXT_XTS_SIZE: u64 = 125288544792;
const FLASH_NEXT_XTS_SHA256: &str =
    "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";

/// Suffix appended to a split base's file name for its exact Q8F16 PLE rows.
pub const PLE_Q8_SIDECAR_SUFFIX: &str = ".ple-q8";
/// Suffix appended to a split base's file name for its 4-bit Lloyd-Max PLE rows.
pub const PLE_Q4_SIDECAR_SUFFIX: &str = ".ple-q4";
/// Marker inserted before the extension of a split base's file name
/// (`qwen3.8-flash-next.mq4-xts` -> `qwen3.8-flash-next-base.mq4-xts`); the
/// mirror of `hipfire_runtime::ple_convert::default_split_base_path`.
const SPLIT_BASE_STEM_SUFFIX: &str = "-base";

/// True when `file` is a bare split-base file name (`<name>-base.<ext>`).
pub fn is_split_base_file(file: &str) -> bool {
    monolithic_file_for_split_base(file).is_some()
}

/// The monolithic sibling file name of a split-base file name: drops the
/// `-base` marker from the stem. `None` when `file` is not a bare split-base
/// name (a path, an empty stem, or no `-base` marker).
pub fn monolithic_file_for_split_base(file: &str) -> Option<String> {
    if file.contains('/') || file.contains('\\') {
        return None;
    }
    let path = Path::new(file);
    let stem = path.file_stem()?.to_str()?;
    let prefix = stem.strip_suffix(SPLIT_BASE_STEM_SUFFIX)?;
    if prefix.is_empty() {
        return None;
    }
    Some(
        match path.extension().and_then(|extension| extension.to_str()) {
            Some(extension) => format!("{prefix}.{extension}"),
            None => prefix.to_owned(),
        },
    )
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse registry from {source_name}: {message}")]
    Parse {
        source_name: String,
        message: String,
    },
    #[error("invalid registry from {source_name}: {message}")]
    Invalid {
        source_name: String,
        message: String,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T> = std::result::Result<T, RegistryError>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sidecar {
    pub file: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

/// Content identity of one trunk overlay sidecar and of the exact base
/// artifact it is bound to.
///
/// The overlay is an HFQM container of pre-encoded QT44 trunk tensors
/// (`<base file>.trunk-overlay`) that the Qwen4 loader applies by default on
/// gfx1151 (`HIPFIRE_QWEN4_TRUNK_OVERLAY=0` is the kill switch). The pin lives
/// in the client instead of a registry field so that older clients' strict
/// `deny_unknown_fields` v1 parser keeps accepting the published registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkOverlayPin {
    /// SHA-256 of the base artifact (the registry entry's `sha256`).
    pub base_sha256: &'static str,
    /// Size in bytes of the base artifact (the registry entry's `size_bytes`).
    pub base_size: u64,
    /// Hugging Face repo that hosts the overlay file.
    pub repo: &'static str,
    /// Overlay file name: the canonical base file name plus
    /// `hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX`.
    pub file: &'static str,
    /// SHA-256 of the overlay file.
    pub sha256: &'static str,
    /// Size in bytes of the overlay file.
    pub size_bytes: u64,
}

/// Trunk overlays known to this client, keyed by exact base artifact identity.
pub const TRUNK_OVERLAY_PINS: &[TrunkOverlayPin] = &[TrunkOverlayPin {
    base_sha256: FLASH_NEXT_XTS_SHA256,
    base_size: FLASH_NEXT_XTS_SIZE,
    repo: FLASH_NEXT_XTS_REPO,
    file: "qwen3.8-flash-next.mq4-xts.trunk-overlay",
    sha256: "df49bc0aa601c2e26556261effd680f0be5d216804a1c5976a8a07ccd66840fd",
    size_bytes: 305_543_168,
}];

/// A resolved trunk overlay: the repo that hosts it and its sidecar identity.
#[derive(Clone, Debug, PartialEq)]
pub struct TrunkOverlay {
    pub repo: &'static str,
    pub sidecar: Sidecar,
}

/// The pin in `pins` whose base identity equals the entry's `sha256` and
/// `size_bytes`, as a [`TrunkOverlay`]. An entry without a pinned digest and
/// size never matches.
pub fn trunk_overlay_in(entry: &ModelEntry, pins: &[TrunkOverlayPin]) -> Option<TrunkOverlay> {
    let sha256 = entry.sha256.as_deref()?;
    let size_bytes = entry.size_bytes?;
    pins.iter()
        .find(|pin| pin.base_size == size_bytes && pin.base_sha256.eq_ignore_ascii_case(sha256))
        .map(|pin| TrunkOverlay {
            repo: pin.repo,
            sidecar: Sidecar {
                file: pin.file.to_owned(),
                sha256: Some(pin.sha256.to_owned()),
                size_bytes: Some(pin.size_bytes),
            },
        })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecommendedSettings {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Parent-model reasoning effort. This is prompt semantics, independent of
    /// any explicit token cap selected by `thinking_budget`.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Optional named cap policy for reasoning tokens
    /// (`off|low|med|high|xhigh|max|uncapped`). Absence means uncapped.
    /// Effort-native families (Qwen3.8, DeepSeek4, Muse Glimmer, Ornith) omit this field.
    #[serde(default)]
    pub thinking_budget: Option<String>,
}

impl RecommendedSettings {
    pub fn config_layer(&self) -> std::result::Result<ConfigLayer, String> {
        let mut layer = ConfigLayer::default();
        if let Some(value) = self.temperature {
            layer
                .set("generation.temperature", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.top_p {
            layer
                .set("generation.top_p", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.top_k {
            let value = i64::try_from(value).map_err(|_| "top_k is too large".to_owned())?;
            layer
                .set("generation.top_k", ConfigValue::Integer(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.min_p {
            layer
                .set("generation.min_p", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.presence_penalty {
            layer
                .set("generation.presence_penalty", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = self.repeat_penalty {
            layer
                .set("generation.repeat_penalty", ConfigValue::Float(value))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.system_prompt {
            layer
                .set("prompt.system", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.reasoning_effort {
            layer
                .set("reasoning.effort", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        if let Some(value) = &self.thinking_budget {
            layer
                .set("reasoning.budget", ConfigValue::String(value.clone()))
                .map_err(|error| error.to_string())?;
        }
        Ok(layer)
    }
}

/// Per-mode sampling profiles for a model, mirroring the model card's
/// documented modes. Each is a full [`RecommendedSettings`] blob, including
/// optional reasoning effort and optional legacy budget policy. `general` is the thinking-mode
/// default (equals the entry's `recommended_settings`); `coding` is the precise
/// thinking-coding profile; `instruct` is the non-thinking profile. Profiles
/// are entry-level metadata and are selected client-side (e.g. serve_harness
/// `--sampling registry:<profile>`), lowering through the typed config/request
/// path rather than architecture-specific environment variables.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingProfiles {
    #[serde(default)]
    pub general: Option<RecommendedSettings>,
    #[serde(default)]
    pub coding: Option<RecommendedSettings>,
    #[serde(default)]
    pub instruct: Option<RecommendedSettings>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub repo: String,
    pub file: String,
    pub size_gb: f64,
    pub min_vram_gb: f64,
    pub desc: String,
    #[serde(default)]
    pub triattn: Option<Sidecar>,
    #[serde(default)]
    pub mtp: Option<Sidecar>,
    #[serde(default)]
    pub dspark: Option<Sidecar>,
    // Image-generation component sidecars (arch 40+ trunks, see
    // docs/architecture-ids.md — ids provisional until the HFQM pack ships).
    // `t5` is the T5-XXL text-encoder component (sidecar arch 41), `clip` the
    // CLIP-L pooled-text encoder (arch 42), `vae` the VAE decoder (arch 43).
    // Shared across FLUX.1 entries; a FLUX.2 Klein entry uses `qwen3` (its
    // Qwen3 text encoder, arch 45) plus the shared `vae` instead of
    // `t5`/`clip`.
    #[serde(default)]
    pub t5: Option<Sidecar>,
    #[serde(default)]
    pub clip: Option<Sidecar>,
    #[serde(default)]
    pub qwen3: Option<Sidecar>,
    #[serde(default)]
    pub vae: Option<Sidecar>,
    #[serde(default)]
    pub dflash: Option<Sidecar>,
    /// Opt-in gfx1151 XDNA NPU spillover sidecar (`.xdna.zip` with manifest
    /// v1). Capability data only: `kernel.npu_spillover` must also be on, the
    /// archive must verify, and the host arch must be exactly `gfx1151`.
    /// No certified entries declare one yet; the first certified model binds
    /// its payload hashes here via `sidecar.sha256`.
    #[serde(default)]
    pub xdna: Option<Sidecar>,
    /// Shared Qwen3.8-27B vision-tower sidecar (`qwen3.8-27b-vision.hfq`,
    /// llm.cpp mmproj-style). Every `qwen3.8:27b*` tier declares the same
    /// file so each text quant tier serves images without requantizing the
    /// trunk; sha256/size_bytes stay absent until the pack ships.
    #[serde(default)]
    pub vision: Option<Sidecar>,
    /// Exact Q8F16 PLE n-gram rows of a Flash-Next split base
    /// (`<file>.ple-q8`). Only valid on an entry whose `file` is a split base
    /// (`<name>-base.<ext>`); the file name must be exactly `<file>.ple-q8`.
    #[serde(default)]
    pub ple_q8: Option<Sidecar>,
    /// 4-bit Lloyd-Max PLE rows of a Flash-Next split base (`<file>.ple-q4`),
    /// bit-identical in decode to the q4 derived locally from the q8 rows.
    /// Same placement rules as `ple_q8`.
    #[serde(default)]
    pub ple_q4: Option<Sidecar>,
    /// Alternative `lm_head` carriers, keyed by short name (`q4k`, `bf16`).
    ///
    /// Each is a single-tensor `.hfq` from `hipfire-quantize --head-only` that
    /// shadows the base's `lm_head.weight` at load time. The BASE ships the
    /// recommended head and runs standalone; these only exist so a different
    /// carrier costs a 188-635 MB download instead of a near-identical 6.5 GB
    /// model. Three full variants would be 19.63 GB; base plus two overlays is
    /// 7.30 GB.
    #[serde(default)]
    pub heads: std::collections::BTreeMap<String, Sidecar>,
    #[serde(default)]
    pub default_tool_format: Option<String>,
    #[serde(default)]
    pub default_kv_mode: Option<String>,
    #[serde(default)]
    pub quant_recipe: Option<String>,
    #[serde(default)]
    pub sampling: Option<SamplingDefaults>,
    #[serde(default)]
    pub recommended_settings: Option<RecommendedSettings>,
    #[serde(default)]
    pub sampling_profiles: Option<SamplingProfiles>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub arch_id: Option<u32>,
    #[serde(default)]
    pub quant: Option<String>,
}

impl ModelEntry {
    /// Exact XTS artifact identity, including the two admitted filename spellings.
    /// This selection has no filesystem side effects; CLI migration verifies bytes.
    pub fn is_flash_next_xts(&self) -> bool {
        self.repo == FLASH_NEXT_XTS_REPO
            && self.size_bytes == Some(FLASH_NEXT_XTS_SIZE)
            && self.sha256.as_deref() == Some(FLASH_NEXT_XTS_SHA256)
            && matches!(
                self.file.as_str(),
                FLASH_NEXT_XTS_LEGACY_FILE | FLASH_NEXT_XTS_CANONICAL_FILE
            )
    }

    /// The Flash-Next trunk overlay sidecar of this entry's base artifact, from
    /// the compiled-in [`TRUNK_OVERLAY_PINS`]; `None` for every other entry.
    pub fn trunk_overlay(&self) -> Option<TrunkOverlay> {
        trunk_overlay_in(self, TRUNK_OVERLAY_PINS)
    }

    /// True when this entry is a split base that publishes PLE row sidecars.
    pub fn has_ple_sidecars(&self) -> bool {
        self.ple_q4.is_some() || self.ple_q8.is_some()
    }

    /// Resolve a named sampling profile (`general` | `coding` | `instruct`).
    /// `general` falls back to `recommended_settings` (the default profile)
    /// when no explicit profile map is present. Unknown names return `None`.
    pub fn sampling_profile(&self, name: &str) -> Option<&RecommendedSettings> {
        let profiles = self.sampling_profiles.as_ref();
        match name {
            "general" => profiles
                .and_then(|p| p.general.as_ref())
                .or(self.recommended_settings.as_ref()),
            "coding" => profiles.and_then(|p| p.coding.as_ref()),
            "instruct" => profiles.and_then(|p| p.instruct.as_ref()),
            _ => None,
        }
    }

    /// Lower this entry's load/sampling defaults into a sparse config layer.
    ///
    /// Starts from [`RecommendedSettings::config_layer`] when present (sampling
    /// and reasoning only), then overlays model-specific load defaults onto
    /// their canonical keys. `default_kv_mode = "q8"` is the universal fallback,
    /// not a model-specific opinion, so it is left for `auto` resolution rather
    /// than lowered here. Callers merge this under `RegistryModel` precedence so
    /// global/model/one-shot user config still wins.
    pub fn config_layer(&self) -> std::result::Result<ConfigLayer, String> {
        let mut layer = match &self.recommended_settings {
            Some(settings) => settings.config_layer()?,
            None => ConfigLayer::default(),
        };
        if let Some(mode) = &self.default_kv_mode {
            if mode != "q8" {
                layer
                    .set("memory.kv_cache", ConfigValue::String(mode.clone()))
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(layer)
    }
}

/// Tag-aware registry config layer. Starts from [`ModelEntry::config_layer`]
/// then applies automatic load policy by canonical tag. This is the single
/// canonical place for tag policy so validation and CLI stay in sync.
///
/// Policy (static; no registry/v1 wire fields): Qwen3.5/3.6/3.8 and
/// DeepSeek V4 Flash tags supply generation max_tokens only — never
/// `memory.max_seq` (resolved at load time) and never a KV backend.
/// Backend selection is source-independent at admission. Qwen3.8 entries
/// with `arch_id=16` (Qwen4) and Muse Glimmer tags write neither length.
///
/// Qwen3.8-27B trunk tiers (`qwen3.8:27b` and `qwen3.8:27b-*`, never a
/// longer lookalike such as `27boops`) that declare a paired `dflash` sidecar
/// also lower `speculation.dflash = "auto"`. The load resolver wires the draft
/// only when that registry-declared file is present; without it MTP `auto`
/// stays in effect. The schema/BuiltIn default stays `off` for every other
/// model; `arch_id=16`, draft/dflash sidecars, and tiers with no `dflash`
/// declaration get no speculation policy.
///
/// Explicit global/model/one-shot user config (including `dflash=off`) remains
/// higher precedence than this registry layer.
pub fn config_layer_for_tag(
    tag: &str,
    entry: &ModelEntry,
) -> std::result::Result<ConfigLayer, String> {
    let mut layer = entry.config_layer()?;
    let family = tag.split(':').next().unwrap_or(tag);
    let is_sidecar = tag.contains("draft") || tag.contains("dflash");
    let is_qwen_tag_policy = matches!(family, "qwen3.5" | "qwen3.6" | "qwen3.8")
        && !is_sidecar
        && entry.arch_id != Some(16);
    if is_qwen_tag_policy {
        layer
            .set("generation.max_tokens", ConfigValue::Integer(81920))
            .map_err(|error| error.to_string())?;
    }
    let is_qwen38_27b_tier = tag == "qwen3.8:27b" || tag.starts_with("qwen3.8:27b-");
    if is_qwen_tag_policy && is_qwen38_27b_tier && entry.dflash.is_some() {
        layer
            .set("speculation.dflash", ConfigValue::String("auto".into()))
            .map_err(|error| error.to_string())?;
    }
    let is_deepseek_tag_policy =
        matches!(family, "deepseek-v4-flash" | "deepseek-v4-flash-preview") && !is_sidecar;
    if is_deepseek_tag_policy {
        layer
            .set("generation.max_tokens", ConfigValue::Integer(393216))
            .map_err(|error| error.to_string())?;
    }
    Ok(layer)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryV1 {
    pub schema_version: u32,
    pub generated_at: String,
    #[serde(default)]
    pub _comment: Option<String>,
    pub models: BTreeMap<String, ModelEntry>,
    pub aliases: BTreeMap<String, String>,
}

impl RegistryV1 {
    pub fn parse(raw: &str, source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        let mut registry: Self =
            serde_json::from_str(raw).map_err(|error| RegistryError::Parse {
                source_name: source_name.clone(),
                message: error.to_string(),
            })?;
        registry.validate(&source_name)?;
        registry.normalize_flash_next_xts();
        // An alias is non-authoritative convenience data. Match v1 behavior by
        // dropping dangling redirects instead of rejecting an otherwise valid
        // registry.
        registry
            .aliases
            .retain(|_, target| registry.models.contains_key(target));
        Ok(registry)
    }

    fn normalize_flash_next_xts(&mut self) {
        for entry in self.models.values_mut() {
            if entry.is_flash_next_xts() && entry.file != FLASH_NEXT_XTS_CANONICAL_FILE {
                entry.file.clear();
                entry.file.push_str(FLASH_NEXT_XTS_CANONICAL_FILE);
            }
        }
    }

    pub fn validate(&self, source_name: &str) -> Result<()> {
        let fail = |message: String| RegistryError::Invalid {
            source_name: source_name.to_owned(),
            message,
        };
        if self.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err(fail(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        validate_generated_at(&self.generated_at).map_err(fail)?;
        if self.models.is_empty() {
            return Err(fail("model catalog is empty".into()));
        }
        for (tag, entry) in &self.models {
            if tag.trim().is_empty() || entry.file.trim().is_empty() {
                return Err(fail(format!("model '{tag}' has an empty tag or file")));
            }
            if !entry.size_gb.is_finite()
                || entry.size_gb < 0.0
                || !entry.min_vram_gb.is_finite()
                || entry.min_vram_gb < 0.0
            {
                return Err(fail(format!("model '{tag}' has invalid size metadata")));
            }
            validate_digest(entry.sha256.as_deref(), tag).map_err(fail)?;
            for sidecar in [
                &entry.triattn,
                &entry.mtp,
                &entry.dspark,
                &entry.dflash,
                &entry.xdna,
                &entry.vision,
                &entry.t5,
                &entry.clip,
                &entry.qwen3,
                &entry.vae,
                &entry.ple_q4,
                &entry.ple_q8,
            ]
            .into_iter()
            .flatten()
            .chain(entry.heads.values())
            {
                if sidecar.file.trim().is_empty() {
                    return Err(fail(format!("model '{tag}' has an empty sidecar file")));
                }
                validate_digest(sidecar.sha256.as_deref(), tag).map_err(fail)?;
            }
            for (slot, suffix, sidecar) in [
                ("ple_q4", PLE_Q4_SIDECAR_SUFFIX, &entry.ple_q4),
                ("ple_q8", PLE_Q8_SIDECAR_SUFFIX, &entry.ple_q8),
            ] {
                let Some(sidecar) = sidecar else {
                    continue;
                };
                if !is_split_base_file(&entry.file) {
                    return Err(fail(format!(
                        "model '{tag}' declares {slot} but its file '{}' is not a split base \
                         (expected <name>-base.<ext>)",
                        entry.file
                    )));
                }
                let expected = format!("{}{suffix}", entry.file);
                if sidecar.file != expected {
                    return Err(fail(format!(
                        "model '{tag}' {slot} file '{}' must be '{expected}'",
                        sidecar.file
                    )));
                }
            }
            if let Some(format) = entry.default_tool_format.as_deref() {
                if !matches!(format, "hermes" | "qwen_xml") {
                    return Err(fail(format!(
                        "model '{tag}' has invalid default_tool_format '{format}'"
                    )));
                }
            }
            if let Some(settings) = &entry.recommended_settings {
                validate_recommendations(tag, settings).map_err(fail)?;
            }
            config_layer_for_tag(tag, entry)
                .map_err(|error| fail(format!("model '{tag}': {error}")))?;
            if let Some(profiles) = &entry.sampling_profiles {
                for (name, settings) in [
                    ("general", &profiles.general),
                    ("coding", &profiles.coding),
                    ("instruct", &profiles.instruct),
                ] {
                    if let Some(settings) = settings {
                        validate_recommendations(tag, settings).map_err(|error| {
                            fail(format!("model '{tag}' profile '{name}': {error}"))
                        })?;
                        settings.config_layer().map_err(|error| {
                            fail(format!("model '{tag}' profile '{name}': {error}"))
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn resolve_tag(&self, input: &str) -> String {
        let normalized = input
            .replace("-hfq4", "-hf4")
            .replace("-hfq6", "-hf6")
            .strip_suffix(".hfq")
            .map(|prefix| format!("{prefix}.hf4"))
            .unwrap_or_else(|| input.replace("-hfq4", "-hf4").replace("-hfq6", "-hf6"));
        if self.models.contains_key(&normalized) {
            return normalized;
        }
        if let Some(tag) = self.aliases.get(&normalized) {
            return tag.clone();
        }
        let qwen = format!("qwen3.5:{normalized}");
        if self.models.contains_key(&qwen) {
            return qwen;
        }
        // A bare file name matches its entry. A path never does: matching on
        // `file_name()` let a lookalike file outside the models directory
        // inherit the installed artifact's identity (sidecars, kv/max_seq
        // policy, rm targets). Callers that know the models directory resolve
        // paths with canonical comparison instead (`registry_entry_for_path`
        // in hipfire-cli); a bare file name carries no directory to confuse.
        self.models
            .iter()
            .find_map(|(tag, entry)| {
                (entry.file == normalized
                    || entry.file == input
                    || (input == FLASH_NEXT_XTS_LEGACY_FILE && entry.is_flash_next_xts()))
                .then(|| tag.clone())
            })
            .unwrap_or(normalized)
    }

    /// Exact bare filename match for callers that already established the input
    /// is the installed artifact (e.g. via canonical path comparison against
    /// the models directory). The pinned XTS legacy basename also matches its
    /// normalized entry. Unlike [`RegistryV1::model`], no tag/alias normalization
    /// is applied.
    pub fn entry_for_file(&self, file_name: &str) -> Option<(&str, &ModelEntry)> {
        self.models
            .iter()
            .find(|(_, entry)| {
                entry.file == file_name
                    || (file_name == FLASH_NEXT_XTS_LEGACY_FILE && entry.is_flash_next_xts())
            })
            .map(|(tag, entry)| (tag.as_str(), entry))
    }

    pub fn model(&self, input: &str) -> Option<(&str, &ModelEntry)> {
        let tag = self.resolve_tag(input);
        self.models
            .get_key_value(&tag)
            .map(|(tag, entry)| (tag.as_str(), entry))
    }

    /// The monolithic (PLE embedded) entry a split `entry` was cut from: the
    /// entry of the same repo whose file is `entry.file` without the `-base`
    /// marker and that has no PLE sidecars itself. `None` for an entry that is
    /// not a split base or whose monolithic sibling is not catalogued. Lets a
    /// user who still has the grandfathered monolithic keep serving it under
    /// the split tag instead of being forced to re-download.
    pub fn monolithic_sibling(&self, entry: &ModelEntry) -> Option<(&str, &ModelEntry)> {
        if !entry.has_ple_sidecars() {
            return None;
        }
        let file = monolithic_file_for_split_base(&entry.file)?;
        self.models
            .iter()
            .find(|(_, other)| {
                other.repo == entry.repo && other.file == file && !other.has_ple_sidecars()
            })
            .map(|(tag, other)| (tag.as_str(), other))
    }
}

fn validate_digest(digest: Option<&str>, label: &str) -> std::result::Result<(), String> {
    if let Some(digest) = digest {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("'{label}' has an invalid SHA-256"));
        }
    }
    Ok(())
}

fn validate_generated_at(ts: &str) -> std::result::Result<(), String> {
    // Strict normalized RFC3339 UTC: YYYY-MM-DDTHH:MM:SSZ (20 bytes).
    // Lexical order equals chronological order only in this normalized form,
    // which is what `prefer_bundled_if_newer` relies on. Reject any
    // non-normalized representation (offsets, fractional seconds, whitespace,
    // lowercase, etc.) and validate calendar ranges so malformed strings
    // cannot invert precedence via lexical comparison.
    if ts.len() != 20 {
        return Err(format!(
            "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
            ts
        ));
    }
    let b = ts.as_bytes();
    if b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return Err(format!(
            "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
            ts
        ));
    }
    for i in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !b[i].is_ascii_digit() {
            return Err(format!(
                "generated_at '{}' is not strict RFC3339 UTC (expected YYYY-MM-DDTHH:MM:SSZ)",
                ts
            ));
        }
    }
    let year = (b[0] - b'0') as u16 * 1000
        + (b[1] - b'0') as u16 * 100
        + (b[2] - b'0') as u16 * 10
        + (b[3] - b'0') as u16;
    let month = (b[5] - b'0') * 10 + (b[6] - b'0');
    let day = (b[8] - b'0') * 10 + (b[9] - b'0');
    let hour = (b[11] - b'0') * 10 + (b[12] - b'0');
    let minute = (b[14] - b'0') * 10 + (b[15] - b'0');
    let second = (b[17] - b'0') * 10 + (b[18] - b'0');
    if month == 0 || month > 12 {
        return Err(format!("generated_at '{}' has invalid month", ts));
    }
    if day == 0 || day > days_in_month(year, month) {
        return Err(format!("generated_at '{}' has invalid day", ts));
    }
    if hour > 23 {
        return Err(format!("generated_at '{}' has invalid hour", ts));
    }
    if minute > 59 {
        return Err(format!("generated_at '{}' has invalid minute", ts));
    }
    if second > 59 {
        return Err(format!("generated_at '{}' has invalid second", ts));
    }
    Ok(())
}

fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn is_leap_year(year: u16) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn is_effort_native_tag(tag: &str) -> bool {
    // Mirrors scripts/registry_gen.py:_effort_native_tag and the family/tag
    // conventions already used by config_layer_for_tag. Effort-native families
    // (Qwen3.8, DeepSeek V4 Flash/preview, Muse Glimmer product SKUs, Ornith
    // 1.5/legacy) have no registry thinking_budget; absence means uncapped.
    // Draft/dflash sidecars are excluded.
    let base = tag.split(" sampling_profiles.").next().unwrap_or(tag);
    if base.contains("draft") || base.contains("dflash") {
        return false;
    }
    let family = base.split(':').next().unwrap_or(base);
    if family == "qwen3.8" {
        return true;
    }
    if matches!(family, "deepseek-v4-flash" | "deepseek-v4-flash-preview") {
        return true;
    }
    if base == "muse-glimmer" || base == "muse-glimmer:fast" {
        return true;
    }
    if matches!(family, "ornith-1.5" | "ornith1.5" | "ornith") {
        return true;
    }
    false
}

fn validate_recommendations(
    tag: &str,
    value: &RecommendedSettings,
) -> std::result::Result<(), String> {
    let ranged = |name: &str, value: Option<f64>, min: f64, max: f64| {
        if value.is_some_and(|value| !value.is_finite() || value < min || value > max) {
            Err(format!("model '{tag}' has invalid {name}"))
        } else {
            Ok(())
        }
    };
    ranged("temperature", value.temperature, 0.0, 2.0)?;
    ranged("top_p", value.top_p, 0.0, 1.0)?;
    ranged("min_p", value.min_p, 0.0, 1.0)?;
    ranged("presence_penalty", value.presence_penalty, 0.0, 2.0)?;
    ranged("repeat_penalty", value.repeat_penalty, 0.5, 2.0)?;
    if value
        .top_k
        .is_some_and(|value| value == 0 || value > 100_000)
    {
        return Err(format!("model '{tag}' has invalid top_k"));
    }
    const REASONING_EFFORTS: &[&str] = &["auto", "none", "low", "medium", "high", "xhigh", "max"];
    const THINKING_BUDGETS: &[&str] = &["off", "low", "med", "high", "xhigh", "max", "uncapped"];
    if let Some(effort) = &value.reasoning_effort {
        if !REASONING_EFFORTS.contains(&effort.as_str()) {
            return Err(format!(
                "model '{tag}' has invalid reasoning_effort '{effort}'"
            ));
        }
    }
    if let Some(budget) = &value.thinking_budget {
        if is_effort_native_tag(tag) {
            return Err(format!(
                "model '{tag}' has unsupported thinking_budget '{budget}' on effort-native model (omit the field; absence means uncapped)"
            ));
        }
        if !THINKING_BUDGETS.contains(&budget.as_str()) {
            return Err(format!(
                "model '{tag}' has invalid thinking_budget '{budget}'"
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegistrySource {
    Cache,
    Network,
    StaleCache,
    Bundled,
}

#[derive(Clone, Debug)]
pub struct LoadedRegistry {
    pub registry: RegistryV1,
    pub source: RegistrySource,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RegistryPaths {
    pub cache: PathBuf,
}

impl RegistryPaths {
    pub fn discover() -> Self {
        let root = env::var_os("HIPFIRE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".hipfire")))
            .unwrap_or_else(|| PathBuf::from(".hipfire"));
        Self {
            cache: root.join("registry.cache.json"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryCache {
    fetched_at: u64,
    url: String,
    registry: RegistryV1,
}

pub fn bundled() -> Result<RegistryV1> {
    RegistryV1::parse(BUNDLED_REGISTRY, "bundled registry/v1.json")
}

/// Prefer the BUNDLED registry when it is newer than whatever was fetched.
///
/// `registry/v1.json` is compiled into the binary, so on a branch the bundled
/// copy IS that branch's registry — while the fetch targets master. Without
/// this, editing the registry on a branch changes nothing for a locally built
/// binary: the 24h cache or a master fetch silently wins, and nothing says so.
/// That cost a real debugging detour (a branch's `heads` map read as empty).
///
/// `generated_at` is the existing signal and needs no new configuration:
/// `scripts/registry_gen.py` stamps it on every regeneration, and its
/// `%Y-%m-%dT%H:%M:%SZ` form compares correctly as a plain string.
///
/// This does NOT break distribution. A released binary's bundled registry is
/// older than master's by construction, so the fetch keeps winning there and
/// users still get new models without upgrading. Only a freshly regenerated
/// local registry — i.e. someone editing it — takes precedence.
fn prefer_bundled_if_newer(
    loaded: LoadedRegistry,
    bundled: RegistryV1,
    warnings: &mut Vec<String>,
) -> LoadedRegistry {
    if loaded.source == RegistrySource::Bundled
        || bundled.generated_at <= loaded.registry.generated_at
    {
        return loaded;
    }
    warnings.push(format!(
        "using the bundled registry ({}), which is newer than the fetched one ({})",
        bundled.generated_at, loaded.registry.generated_at
    ));
    LoadedRegistry {
        registry: bundled,
        source: RegistrySource::Bundled,
        warnings: std::mem::take(warnings),
    }
}

pub fn load(paths: &RegistryPaths) -> LoadedRegistry {
    let url = env::var("HIPFIRE_REGISTRY_URL").unwrap_or_else(|_| DEFAULT_REGISTRY_URL.into());
    let fetch = env::var_os("HIPFIRE_NO_REGISTRY_FETCH").as_deref() != Some("1".as_ref());
    load_with_options(paths, fetch.then_some(url.as_str()))
}

fn load_with_options(paths: &RegistryPaths, url: Option<&str>) -> LoadedRegistry {
    let mut warnings = Vec::new();
    let bundled = bundled().expect("checked-in registry/v1.json must validate");
    let Some(url) = url else {
        return LoadedRegistry {
            registry: bundled,
            source: RegistrySource::Bundled,
            warnings,
        };
    };
    let now = epoch_millis();
    let cache = read_cache(&paths.cache, url, &mut warnings);
    if cache
        .as_ref()
        .is_some_and(|cache| cache_is_fresh(cache, now, REGISTRY_CACHE_TTL))
    {
        let loaded = LoadedRegistry {
            registry: cache.expect("checked above").registry,
            source: RegistrySource::Cache,
            warnings: std::mem::take(&mut warnings),
        };
        return prefer_bundled_if_newer(loaded, bundled, &mut warnings);
    }

    let loaded = match fetch_registry(&url) {
        Ok(registry) => {
            let cache_file = RegistryCache {
                fetched_at: now,
                url: url.to_owned(),
                registry: registry.clone(),
            };
            if let Err(error) = write_cache(&paths.cache, &cache_file) {
                warnings.push(error.to_string());
            }
            LoadedRegistry {
                registry,
                source: RegistrySource::Network,
                warnings: std::mem::take(&mut warnings),
            }
        }
        Err(error) => {
            warnings.push(error);
            if let Some(cache) = cache {
                LoadedRegistry {
                    registry: cache.registry,
                    source: RegistrySource::StaleCache,
                    warnings: std::mem::take(&mut warnings),
                }
            } else {
                LoadedRegistry {
                    registry: bundled.clone(),
                    source: RegistrySource::Bundled,
                    warnings: std::mem::take(&mut warnings),
                }
            }
        }
    };
    prefer_bundled_if_newer(loaded, bundled, &mut warnings)
}

fn read_cache(path: &Path, url: &str, warnings: &mut Vec<String>) -> Option<RegistryCache> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warnings.push(format!("registry cache read failed: {error}"));
            return None;
        }
    };
    let mut cache: RegistryCache = match serde_json::from_str(&raw) {
        Ok(cache) => cache,
        Err(error) => {
            warnings.push(format!("registry cache parse failed: {error}"));
            return None;
        }
    };
    if cache.url != url {
        return None;
    }
    if let Err(error) = cache.registry.validate("registry cache") {
        warnings.push(error.to_string());
        return None;
    }
    cache.registry.normalize_flash_next_xts();
    cache
        .registry
        .aliases
        .retain(|_, target| cache.registry.models.contains_key(target));
    Some(cache)
}

fn cache_is_fresh(cache: &RegistryCache, now_ms: u64, ttl: Duration) -> bool {
    cache.fetched_at <= now_ms && now_ms.saturating_sub(cache.fetched_at) < ttl.as_millis() as u64
}

fn fetch_registry(url: &str) -> std::result::Result<RegistryV1, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(REGISTRY_FETCH_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|error| format!("registry fetch failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "registry fetch returned HTTP {}",
            response.status()
        ));
    }
    let raw = response
        .body_mut()
        .read_to_string()
        .map_err(|error| format!("registry response read failed: {error}"))?;
    RegistryV1::parse(&raw, url).map_err(|error| error.to_string())
}

fn write_cache(path: &Path, cache: &RegistryCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| RegistryError::Write {
            path: parent.to_owned(),
            source,
        })?;
    }
    let bytes = serde_json::to_vec(cache).map_err(|error| RegistryError::Parse {
        source_name: "registry cache serialization".into(),
        message: error.to_string(),
    })?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&tmp, bytes).map_err(|source| RegistryError::Write {
        path: tmp.clone(),
        source,
    })?;
    fs::rename(&tmp, path).map_err(|source| RegistryError::Write {
        path: path.to_owned(),
        source,
    })
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLASH_NEXT_FIXTURE_TAGS: [&str; 4] = [
        "qwen3.8:flash-next",
        "qwen3.8:flash-next-mq4",
        "qwen3.8:flash-next-gptq3",
        "qwen3.8:flash-next-mq4-xts",
    ];

    fn pinned_flash_next_entry(file: &str) -> ModelEntry {
        ModelEntry {
            repo: FLASH_NEXT_XTS_REPO.into(),
            file: file.into(),
            size_gb: 125.29,
            min_vram_gb: 8.0,
            desc: "Qwen3.8-Flash-Next MQ4 XTS fixture".into(),
            sha256: Some(FLASH_NEXT_XTS_SHA256.into()),
            size_bytes: Some(FLASH_NEXT_XTS_SIZE),
            arch_id: Some(16),
            quant: Some("mq4".into()),
            ..Default::default()
        }
    }

    /// A registry as the pre-split wire format spelled it: three tags on the
    /// legacy filename and the canonical pinned monolithic under its own tag.
    fn flash_next_wire_registry(stamp: &str) -> RegistryV1 {
        let mut models = BTreeMap::new();
        for tag in &FLASH_NEXT_FIXTURE_TAGS[..3] {
            models.insert(
                (*tag).to_owned(),
                pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE),
            );
        }
        models.insert(
            "qwen3.8:flash-next-mq4-xts".into(),
            pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE),
        );
        RegistryV1 {
            schema_version: REGISTRY_SCHEMA_VERSION,
            generated_at: stamp.into(),
            _comment: None,
            models,
            aliases: BTreeMap::from([("fixture".into(), "qwen3.8:flash-next".into())]),
        }
    }

    fn assert_flash_next_xts(registry: &RegistryV1, tags: &[&str]) {
        for tag in tags {
            let (_, entry) = registry.model(tag).unwrap();
            assert!(entry.is_flash_next_xts(), "{tag}");
            assert_eq!(entry.file, FLASH_NEXT_XTS_CANONICAL_FILE, "{tag}");
            assert_eq!(entry.arch_id, Some(16));
            assert_eq!(entry.quant.as_deref(), Some("mq4"));
        }
    }

    /// The pinned monolithic stays catalogued under its own tag whatever the
    /// canonical `qwen3.8:flash-next` tag maps to (the pinned monolithic, or the
    /// split base once the registry switches).
    const BUNDLED_PINNED_TAGS: [&str; 1] = ["qwen3.8:flash-next-mq4-xts"];

    #[test]
    fn flash_next_xts_bundle_preserves_wire_compatibility_and_normalizes_locally() {
        let wire: RegistryV1 = serde_json::from_str(BUNDLED_REGISTRY).unwrap();
        assert_eq!(
            wire.models["qwen3.8:flash-next-mq4-xts"].file,
            FLASH_NEXT_XTS_CANONICAL_FILE
        );
        let parsed = bundled().unwrap();
        assert_flash_next_xts(&parsed, &BUNDLED_PINNED_TAGS);
        assert_eq!(wire.aliases["qwen3.8:flash"], "qwen3.8:flash-next");
        let (short_tag, _) = parsed.model("qwen3.8:flash").unwrap();
        assert_eq!(short_tag, "qwen3.8:flash-next");
        assert!(wire
            .models
            .values()
            .all(|entry| !entry.desc.to_ascii_lowercase().contains("gptq3")));
        for (tag, entry) in &wire.models {
            let mut expected = entry.clone();
            if expected.is_flash_next_xts() {
                assert!(!entry.desc.contains("GPTQ3"), "{tag}");
                expected.file = FLASH_NEXT_XTS_CANONICAL_FILE.into();
            }
            assert_eq!(
                &parsed.models[tag], &expected,
                "{tag}: only filename may change"
            );
        }
        assert!(!parsed.models["qwen3.8:flash-next-rtn-asym"].is_flash_next_xts());
        assert_eq!(
            parsed.models["qwen3.8:flash-next-rtn-asym"].file,
            "qwen3.8-flash-next.mq4"
        );
    }

    #[test]
    fn flash_next_zab_named_tier_preserves_stock_aliases() {
        let wire: RegistryV1 = serde_json::from_str(BUNDLED_REGISTRY).unwrap();
        let curated: serde_json::Value =
            serde_json::from_str(include_str!("../../../registry/models.json")).unwrap();
        let tag = "qwen3.8:flash-next-mq4-xts-zab";
        let zab = &wire.models[tag];
        assert!(zab.repo.is_empty(), "staged artifact must not be pullable");
        assert_eq!(zab.file, "qwen3.8-flash-next.mq4-xts-zab");
        assert_eq!(zab.arch_id, Some(16));
        assert_eq!(zab.quant.as_deref(), Some("mq4"));
        assert_eq!(
            zab.sha256.as_deref(),
            Some("a85a00d192ee7a352d5d1f55f0421691a2553fe90f05d9ae1d7e59fec5b686fa")
        );
        assert_eq!(zab.size_bytes, Some(125144775192));
        let curated_zab: ModelEntry =
            serde_json::from_value(curated["models"][tag].clone()).unwrap();
        assert_eq!(zab.sha256, curated_zab.sha256);
        assert_eq!(zab.size_bytes, curated_zab.size_bytes);
        assert!(!zab.is_flash_next_xts());
        assert!(zab.desc.contains("LOCAL-ONLY"));
        assert!(zab.desc.contains("108 GDN"));
        assert!(zab.desc.contains("symmetric QT44"));
        assert!(zab.desc.contains("non-zab bytes unchanged"));
        assert!(zab.desc.contains("qwen4_trunk_iu4_default=gfx1151-zab-v1"));
        assert!(zab.desc.contains("HIPFIRE_QWEN4_TRUNK_IU4=0"));
        assert!(zab.desc.contains("NOT uploaded/pullable"));
        let settings = zab.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(20));
        assert_eq!(serde_json::to_value(&wire.aliases).unwrap(), curated["aliases"]);
        assert_eq!(wire.aliases["qwen3.8:flash"], "qwen3.8:flash-next");
        assert!(!wire.aliases.values().any(|target| target == tag));
        for stock in FLASH_NEXT_FIXTURE_TAGS {
            let entry = &wire.models[stock];
            assert_eq!(entry.repo, FLASH_NEXT_XTS_REPO);
            assert_eq!(entry.sha256.as_deref(), Some(FLASH_NEXT_XTS_SHA256));
            assert_eq!(entry.size_bytes, Some(FLASH_NEXT_XTS_SIZE));
            assert_eq!(
                entry.file,
                if stock == "qwen3.8:flash-next-mq4-xts" {
                    FLASH_NEXT_XTS_CANONICAL_FILE
                } else {
                    FLASH_NEXT_XTS_LEGACY_FILE
                }
            );
            assert_eq!(entry.recommended_settings, zab.recommended_settings);
        }
    }

    const TRUNK_OVERLAY_FILE: &str = "qwen3.8-flash-next.mq4-xts.trunk-overlay";

    #[test]
    fn trunk_overlay_pin_names_the_canonical_base_file_plus_the_suffix() {
        assert_eq!(TRUNK_OVERLAY_PINS.len(), 1);
        let pin = &TRUNK_OVERLAY_PINS[0];
        assert_eq!(
            pin.file,
            format!(
                "{FLASH_NEXT_XTS_CANONICAL_FILE}{}",
                hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX
            )
        );
        assert_eq!(pin.file, TRUNK_OVERLAY_FILE);
        assert_eq!(pin.base_sha256, FLASH_NEXT_XTS_SHA256);
        assert_eq!(pin.base_size, FLASH_NEXT_XTS_SIZE);
        assert_eq!(pin.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(pin.size_bytes, 305_543_168);
        for digest in [pin.base_sha256, pin.sha256] {
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert!(!pin.file.contains('/') && !pin.file.contains('\\'));
    }

    #[test]
    fn trunk_overlay_lookup_matches_the_base_identity_only() {
        let entry = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        let overlay = entry.trunk_overlay().unwrap();
        assert_eq!(overlay.repo, FLASH_NEXT_XTS_REPO);
        assert_eq!(
            overlay.sidecar,
            Sidecar {
                file: TRUNK_OVERLAY_FILE.into(),
                sha256: Some(TRUNK_OVERLAY_PINS[0].sha256.into()),
                size_bytes: Some(305_543_168),
            }
        );
        // The legacy spelling of the same pinned bytes resolves to the canonical overlay name.
        let legacy = pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE);
        assert_eq!(legacy.trunk_overlay(), Some(overlay));

        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.sha256 = Some(digest('a'));
        assert!(other.trunk_overlay().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.size_bytes = Some(FLASH_NEXT_XTS_SIZE - 1);
        assert!(other.trunk_overlay().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.sha256 = None;
        assert!(other.trunk_overlay().is_none());
        let mut other = pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE);
        other.size_bytes = None;
        assert!(other.trunk_overlay().is_none());
    }

    #[test]
    fn trunk_overlay_in_uses_the_supplied_pin_table() {
        let mut entry = pinned_flash_next_entry("custom.mq4");
        entry.sha256 = Some(digest('b'));
        entry.size_bytes = Some(123);
        assert!(trunk_overlay_in(&entry, TRUNK_OVERLAY_PINS).is_none());
        let pins = [TrunkOverlayPin {
            base_sha256: "b".repeat(64).leak(),
            base_size: 123,
            repo: "test/repo",
            file: "custom.mq4.trunk-overlay",
            sha256: "c".repeat(64).leak(),
            size_bytes: 7,
        }];
        let overlay = trunk_overlay_in(&entry, &pins).unwrap();
        assert_eq!(overlay.repo, "test/repo");
        assert_eq!(overlay.sidecar.file, "custom.mq4.trunk-overlay");
        assert_eq!(overlay.sidecar.sha256, Some(digest('c')));
        assert_eq!(overlay.sidecar.size_bytes, Some(7));
        assert!(trunk_overlay_in(&entry, &[]).is_none());
    }

    #[test]
    fn bundled_registry_resolves_the_trunk_overlay_on_exactly_the_stock_xts_tags() {
        let parsed = bundled().unwrap();
        let mut resolved: Vec<&str> = parsed
            .models
            .iter()
            .filter(|(_, entry)| entry.trunk_overlay().is_some())
            .map(|(tag, _)| tag.as_str())
            .collect();
        resolved.sort_unstable();
        let mut expected = FLASH_NEXT_FIXTURE_TAGS.to_vec();
        expected.sort_unstable();
        assert_eq!(resolved, expected);
        assert!(parsed.models["qwen3.8:flash-next-rtn-asym"]
            .trunk_overlay()
            .is_none());
        assert!(parsed.models["qwen3.8:flash-next-mq4-xts-zab"]
            .trunk_overlay()
            .is_none());
        let first = parsed.models["qwen3.8:flash-next"].trunk_overlay().unwrap();
        for tag in FLASH_NEXT_FIXTURE_TAGS {
            assert_eq!(
                parsed.models[tag].trunk_overlay().as_ref(),
                Some(&first),
                "{tag}"
            );
        }
        let (alias_tag, aliased) = parsed.model("qwen3.8:flash").unwrap();
        assert_eq!(alias_tag, "qwen3.8:flash-next");
        assert_eq!(aliased.trunk_overlay().as_ref(), Some(&first));
    }

    #[test]
    fn the_wire_registry_has_no_trunk_overlay_field() {
        // The overlay pin is compiled into the client; the registry schema is
        // unchanged so older clients' strict v1 parser keeps accepting it.
        assert!(!BUNDLED_REGISTRY.contains("trunk_overlay"));
        assert!(!include_str!("../../../registry/models.json").contains("trunk_overlay"));
        let mut value: serde_json::Value = serde_json::from_str(BUNDLED_REGISTRY).unwrap();
        value["models"]["qwen3.8:flash-next"]["trunk_overlay"] = serde_json::json!({
            "file": TRUNK_OVERLAY_FILE
        });
        assert!(
            RegistryV1::parse(&value.to_string(), "fixture").is_err(),
            "ModelEntry stays deny_unknown_fields"
        );
    }

    #[test]
    fn flash_next_legacy_spelled_wire_entries_normalize_locally() {
        let wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        for tag in &FLASH_NEXT_FIXTURE_TAGS[..3] {
            assert_eq!(wire.models[*tag].file, FLASH_NEXT_XTS_LEGACY_FILE);
        }
        let parsed = RegistryV1::parse(&serde_json::to_string(&wire).unwrap(), "fixture").unwrap();
        assert_flash_next_xts(&parsed, &FLASH_NEXT_FIXTURE_TAGS);
        assert_eq!(parsed.model("fixture").unwrap().0, "qwen3.8:flash-next");
    }

    const SPLIT_BASE_FILE: &str = "qwen3.8-flash-next-base.mq4-xts";

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    /// The registry after the split switch: `qwen3.8:flash-next` is the split
    /// base with both PLE slots; the pinned monolithic keeps its own tags.
    fn split_registry_json() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": {
                    "repo": FLASH_NEXT_XTS_REPO,
                    "file": SPLIT_BASE_FILE,
                    "size_gb": 70.9,
                    "min_vram_gb": 8.0,
                    "desc": "Qwen3.8-Flash-Next MQ4 XTS split base",
                    "arch_id": 16,
                    "quant": "mq4",
                    "sha256": digest('a'),
                    "size_bytes": 70_900_000_000_u64,
                    "ple_q8": {
                        "file": format!("{SPLIT_BASE_FILE}.ple-q8"),
                        "sha256": digest('b'),
                        "size_bytes": 54_400_000_000_u64
                    },
                    "ple_q4": {
                        "file": format!("{SPLIT_BASE_FILE}.ple-q4"),
                        "sha256": digest('c'),
                        "size_bytes": 28_800_000_000_u64
                    }
                },
                "qwen3.8:flash-next-mq4": serde_json::to_value(
                    pinned_flash_next_entry(FLASH_NEXT_XTS_LEGACY_FILE)).unwrap(),
                "qwen3.8:flash-next-mq4-xts": serde_json::to_value(
                    pinned_flash_next_entry(FLASH_NEXT_XTS_CANONICAL_FILE)).unwrap()
            },
            "aliases": { "qwen3.8:flash": "qwen3.8:flash-next" }
        })
    }

    fn parse_split(value: &serde_json::Value) -> Result<RegistryV1> {
        RegistryV1::parse(&value.to_string(), "split fixture")
    }

    #[test]
    fn split_entry_parses_and_keeps_the_pinned_monolithic_distinct() {
        let registry = parse_split(&split_registry_json()).unwrap();
        let (tag, split) = registry.model("qwen3.8:flash").unwrap();
        assert_eq!(tag, "qwen3.8:flash-next");
        assert_eq!(split.file, SPLIT_BASE_FILE);
        assert!(split.has_ple_sidecars());
        assert!(
            !split.is_flash_next_xts(),
            "the split base is not the pinned monolithic"
        );
        assert_eq!(
            split.ple_q8.as_ref().unwrap().file,
            format!("{SPLIT_BASE_FILE}{PLE_Q8_SIDECAR_SUFFIX}")
        );
        assert_eq!(
            split.ple_q4.as_ref().unwrap().file,
            format!("{SPLIT_BASE_FILE}{PLE_Q4_SIDECAR_SUFFIX}")
        );
        // The pinned monolithic still normalizes to the canonical file under its tags.
        assert_flash_next_xts(
            &registry,
            &["qwen3.8:flash-next-mq4", "qwen3.8:flash-next-mq4-xts"],
        );
        // Either legacy spelling of the pin resolves to a pinned entry, never the split.
        let (legacy_tag, legacy) = registry.model(FLASH_NEXT_XTS_LEGACY_FILE).unwrap();
        assert!(legacy.is_flash_next_xts(), "{legacy_tag}");
        assert_eq!(
            registry.entry_for_file(SPLIT_BASE_FILE).unwrap().0,
            "qwen3.8:flash-next"
        );
        let reparsed = parse_split(&serde_json::to_value(&registry).unwrap()).unwrap();
        assert_eq!(registry, reparsed);
    }

    #[test]
    fn split_base_without_slots_or_with_one_slot_is_valid() {
        for keep in [None, Some("ple_q8"), Some("ple_q4")] {
            let mut value = split_registry_json();
            let entry = value["models"]["qwen3.8:flash-next"]
                .as_object_mut()
                .unwrap();
            for slot in ["ple_q8", "ple_q4"] {
                if Some(slot) != keep {
                    entry.remove(slot);
                }
            }
            let registry = parse_split(&value).unwrap();
            assert_eq!(
                registry.models["qwen3.8:flash-next"].has_ple_sidecars(),
                keep.is_some()
            );
        }
    }

    #[test]
    fn ple_slot_validation_rejects_misplaced_or_malformed_sidecars() {
        let cases: [(&str, Box<dyn Fn(&mut serde_json::Value)>, &str); 7] = [
            (
                "slot on a monolithic file",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["file"] = "qwen3.8-flash-next.mq4-xts".into();
                    entry["ple_q8"]["file"] = "qwen3.8-flash-next.mq4-xts.ple-q8".into();
                    entry["ple_q4"]["file"] = "qwen3.8-flash-next.mq4-xts.ple-q4".into();
                }),
                "not a split base",
            ),
            (
                "q8 file does not follow the base",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple_q8"]["file"] = "other.ple-q8".into()
                }),
                "must be",
            ),
            (
                "q4 slot names the q8 file",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple_q4"]["file"] = format!("{SPLIT_BASE_FILE}.ple-q8").into();
                }),
                "must be",
            ),
            (
                "sidecar in a subdirectory",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple_q4"]["file"] = format!("sub/{SPLIT_BASE_FILE}.ple-q4").into();
                }),
                "must be",
            ),
            (
                "bad q8 digest",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["ple_q8"]["sha256"] = "8b15b6".into()
                }),
                "invalid SHA-256",
            ),
            (
                "empty q4 file",
                Box::new(|entry: &mut serde_json::Value| entry["ple_q4"]["file"] = "  ".into()),
                "empty sidecar file",
            ),
            (
                "empty stem before -base",
                Box::new(|entry: &mut serde_json::Value| {
                    entry["file"] = "-base.mq4-xts".into();
                    entry["ple_q8"]["file"] = "-base.mq4-xts.ple-q8".into();
                    entry["ple_q4"]["file"] = "-base.mq4-xts.ple-q4".into();
                }),
                "not a split base",
            ),
        ];
        for (name, mutate, needle) in cases {
            let mut value = split_registry_json();
            mutate(&mut value["models"]["qwen3.8:flash-next"]);
            let error = parse_split(&value).expect_err(name).to_string();
            assert!(error.contains(needle), "{name}: {error}");
        }
        let mut value = split_registry_json();
        value["models"]["qwen3.8:flash-next"]["ple_q8"]["unknown"] = 1.into();
        assert!(
            parse_split(&value).is_err(),
            "sidecar slots stay deny_unknown_fields"
        );
    }

    #[test]
    fn split_base_file_helpers_mirror_the_converter_naming() {
        for (base, monolithic) in [
            (
                "qwen3.8-flash-next-base.mq4-xts",
                Some("qwen3.8-flash-next.mq4-xts"),
            ),
            ("model-base.mq4", Some("model.mq4")),
            ("model-base", Some("model")),
            ("model.mq4", None),
            ("-base.mq4", None),
            ("model-base.mq4.ple-q8", None),
            ("dir/model-base.mq4", None),
            ("", None),
        ] {
            assert_eq!(
                monolithic_file_for_split_base(base).as_deref(),
                monolithic,
                "{base}"
            );
            assert_eq!(is_split_base_file(base), monolithic.is_some(), "{base}");
        }
    }

    #[test]
    fn monolithic_sibling_maps_the_split_base_to_the_catalogued_pin() {
        let registry = parse_split(&split_registry_json()).unwrap();
        let split = &registry.models["qwen3.8:flash-next"];
        let (_, sibling) = registry.monolithic_sibling(split).unwrap();
        assert!(sibling.is_flash_next_xts());
        assert_eq!(sibling.file, FLASH_NEXT_XTS_CANONICAL_FILE);
        // A monolithic entry has no sibling of its own.
        assert!(registry
            .monolithic_sibling(&registry.models["qwen3.8:flash-next-mq4-xts"])
            .is_none());

        let mut orphan = split_registry_json();
        orphan["models"]
            .as_object_mut()
            .unwrap()
            .remove("qwen3.8:flash-next-mq4");
        orphan["models"]
            .as_object_mut()
            .unwrap()
            .remove("qwen3.8:flash-next-mq4-xts");
        let orphan = parse_split(&orphan).unwrap();
        assert!(orphan
            .monolithic_sibling(&orphan.models["qwen3.8:flash-next"])
            .is_none());

        let mut foreign = split_registry_json();
        foreign["models"]["qwen3.8:flash-next-mq4"]["repo"] = "other/repo".into();
        foreign["models"]["qwen3.8:flash-next-mq4-xts"]["repo"] = "other/repo".into();
        let foreign = parse_split(&foreign).unwrap();
        assert!(foreign
            .monolithic_sibling(&foreign.models["qwen3.8:flash-next"])
            .is_none());
    }

    #[test]
    fn bundled_split_entries_name_their_monolithic_sibling() {
        let registry = bundled().unwrap();
        for (tag, entry) in &registry.models {
            if entry.has_ple_sidecars() {
                assert!(
                    registry.monolithic_sibling(entry).is_some(),
                    "{tag}: a split entry must keep its grandfathered monolithic catalogued"
                );
            }
        }
    }

    #[test]
    fn flash_next_xts_normalization_requires_the_complete_pin_and_exact_filename() {
        let wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        let original = wire.models["qwen3.8:flash-next"].clone();
        assert!(original.is_flash_next_xts());
        for mismatch in 0..7 {
            let mut entry = original.clone();
            match mismatch {
                0 => entry.repo = "other/qwen3.8-flash-next".into(),
                1 => entry.size_bytes = Some(FLASH_NEXT_XTS_SIZE - 1),
                2 => entry.sha256 = Some("a".repeat(64)),
                3 => entry.file = "qwen3.8-flash-next.mq4".into(),
                4 => entry.file = format!("subdir/{FLASH_NEXT_XTS_LEGACY_FILE}"),
                5 => entry.sha256 = None,
                6 => entry.size_bytes = None,
                _ => unreachable!(),
            }
            assert!(!entry.is_flash_next_xts());
            let mut changed = wire.clone();
            changed
                .models
                .insert("qwen3.8:flash-next".into(), entry.clone());
            let parsed =
                RegistryV1::parse(&serde_json::to_string(&changed).unwrap(), "fixture").unwrap();
            assert_eq!(parsed.models["qwen3.8:flash-next"], entry);
        }
        let parsed = RegistryV1::parse(&serde_json::to_string(&wire).unwrap(), "fixture").unwrap();
        assert_eq!(
            parsed.model("fixture").unwrap().1.file,
            FLASH_NEXT_XTS_CANONICAL_FILE
        );
        let reparsed =
            RegistryV1::parse(&serde_json::to_string(&parsed).unwrap(), "fixture").unwrap();
        assert_eq!(parsed, reparsed);
        assert!(parsed
            .model(FLASH_NEXT_XTS_LEGACY_FILE)
            .unwrap()
            .1
            .is_flash_next_xts());
        assert!(parsed
            .entry_for_file(FLASH_NEXT_XTS_LEGACY_FILE)
            .unwrap()
            .1
            .is_flash_next_xts());
        for path in [
            format!("./{FLASH_NEXT_XTS_LEGACY_FILE}"),
            format!("/tmp/{FLASH_NEXT_XTS_LEGACY_FILE}"),
        ] {
            assert!(parsed.model(&path).is_none());
            assert!(parsed.entry_for_file(&path).is_none());
        }
        let mut vintage = wire.clone();
        vintage.models.remove("qwen3.8:flash-next-mq4-xts");
        let parsed =
            RegistryV1::parse(&serde_json::to_string(&vintage).unwrap(), "v0.4.1 remote").unwrap();
        for entry in parsed.models.values() {
            assert!(entry.is_flash_next_xts());
            assert_eq!(entry.file, FLASH_NEXT_XTS_CANONICAL_FILE);
        }
        let mut invalid = wire;
        invalid.models.get_mut("qwen3.8:flash-next").unwrap().sha256 = Some("8b15b6".into());
        assert!(RegistryV1::parse(&serde_json::to_string(&invalid).unwrap(), "fixture").is_err());
    }

    struct RegistryTestDir(PathBuf);

    impl RegistryTestDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = env::temp_dir().join(format!(
                "hipfire-registry-{}-{}-{}",
                std::process::id(),
                epoch_millis(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for RegistryTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn serve_registry_once(registry: &RegistryV1) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!(
            "http://{}/alternate-registry.json",
            listener.local_addr().unwrap()
        );
        let body = serde_json::to_string(registry).unwrap();
        let handle = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = [0; 4096];
            let read = socket.read(&mut request).unwrap();
            assert!(read > 0, "client sent no request");
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            ).unwrap();
        });
        (url, handle)
    }

    #[test]
    fn flash_next_xts_normalizes_bundle_remote_fresh_and_stale_cache_and_alternate_url() {
        let dir = RegistryTestDir::new();
        let paths = RegistryPaths {
            cache: dir.0.join("registry.cache.json"),
        };
        let offline = load_with_options(&paths, None);
        assert_eq!(offline.source, RegistrySource::Bundled);
        assert_flash_next_xts(&offline.registry, &BUNDLED_PINNED_TAGS);
        assert!(
            !paths.cache.exists(),
            "bundle-only must not fetch or write cache"
        );

        let wire = flash_next_wire_registry("2099-01-01T00:00:00Z");
        let (url, server) = serve_registry_once(&wire);
        let remote = load_with_options(&paths, Some(&url));
        server.join().unwrap();
        assert_eq!(remote.source, RegistrySource::Network);
        assert_flash_next_xts(&remote.registry, &FLASH_NEXT_FIXTURE_TAGS);
        let cached = load_with_options(&paths, Some(&url));
        assert_eq!(cached.source, RegistrySource::Cache);
        assert_flash_next_xts(&cached.registry, &FLASH_NEXT_FIXTURE_TAGS);

        // Simulate a pre-upgrade cache: deserialization alone has not normalized it.
        let legacy_cache = RegistryCache {
            fetched_at: epoch_millis(),
            url: url.clone(),
            registry: wire.clone(),
        };
        write_cache(&paths.cache, &legacy_cache).unwrap();
        let cached = load_with_options(&paths, Some(&url));
        assert_eq!(cached.source, RegistrySource::Cache);
        assert_flash_next_xts(&cached.registry, &FLASH_NEXT_FIXTURE_TAGS);
        assert!(cached
            .registry
            .model("fixture")
            .unwrap()
            .1
            .is_flash_next_xts());

        let mut stale = legacy_cache;
        stale.fetched_at = 0;
        write_cache(&paths.cache, &stale).unwrap();
        let loaded = load_with_options(&paths, Some(&url));
        assert_eq!(loaded.source, RegistrySource::StaleCache);
        assert_flash_next_xts(&loaded.registry, &FLASH_NEXT_FIXTURE_TAGS);
        assert!(!loaded.warnings.is_empty());

        let (alternate, server) = serve_registry_once(&wire);
        assert_ne!(url, alternate);
        let loaded = load_with_options(&paths, Some(&alternate));
        server.join().unwrap();
        assert_eq!(loaded.source, RegistrySource::Network, "cache is URL-bound");
        assert_flash_next_xts(&loaded.registry, &FLASH_NEXT_FIXTURE_TAGS);
        let offline = load_with_options(&paths, None);
        assert_eq!(
            offline.source,
            RegistrySource::Bundled,
            "disabled fetch ignores cache"
        );

        let old_wire = flash_next_wire_registry("2000-01-01T00:00:00Z");
        let (old_url, server) = serve_registry_once(&old_wire);
        let loaded = load_with_options(&paths, Some(&old_url));
        server.join().unwrap();
        assert_eq!(loaded.source, RegistrySource::Bundled, "newer bundle wins");
        assert_flash_next_xts(&loaded.registry, &BUNDLED_PINNED_TAGS);
        assert!(loaded
            .warnings
            .iter()
            .any(|warning| warning.contains("newer")));
    }

    #[test]
    fn config_layer_omits_universal_q8_fallback_but_lowers_model_opinions() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "fallback":{"repo":"r","file":"fallback.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","default_kv_mode":"q8"},
                "opinion":{"repo":"r","file":"opinion.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","default_kv_mode":"fwht3"}
            },
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        let fallback = registry
            .model("fallback")
            .unwrap()
            .1
            .config_layer()
            .unwrap();
        assert!(fallback.get("memory.kv_cache").is_none());

        let opinion = registry.model("opinion").unwrap().1.config_layer().unwrap();
        assert_eq!(
            opinion.get("memory.kv_cache"),
            Some(&ConfigValue::String("fwht3".into()))
        );
    }

    /// The `heads` map must be VALIDATED, not merely parsed.
    ///
    /// Adding a field to the struct makes it round-trip; it does not make the
    /// validator look at it. This asserts the negative directly: a head with a
    /// malformed digest is REJECTED. Without the `.chain(entry.heads.values())`
    /// in `validate`, this test fails and the bundled-registry check would
    /// happily ship an unverifiable head overlay.
    #[test]
    fn heads_sidecars_are_digest_validated() {
        let with_bad_head = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{"repo":"r","file":"f.hfq","size_gb":1,"min_vram_gb":1,"desc":"d",
              "heads":{"q4k":{"file":"h.hfq","sha256":"not-a-sha"}}}},
            "aliases":{}
        }"#;
        let err = RegistryV1::parse(with_bad_head, "test")
            .expect_err("a malformed head digest must be rejected");
        assert!(
            format!("{err}").contains("invalid SHA-256"),
            "expected a digest complaint, got: {err}"
        );

        // Control: the SAME registry with a well-formed digest parses, so the
        // rejection above is about the digest and not about `heads` being
        // unparseable.
        let good = with_bad_head.replace("not-a-sha", &"a".repeat(64));
        let reg = RegistryV1::parse(&good, "test").expect("valid head must parse");
        let (_, entry) = reg.model("m").unwrap();
        assert_eq!(entry.heads.len(), 1);
        assert_eq!(entry.heads["q4k"].file, "h.hfq");
    }
    /// The `xdna` sidecar must be VALIDATED like every other sidecar slot:
    /// a malformed digest is REJECTED and a well-formed one parses onto the
    /// entry. Without `&entry.xdna` in the `validate` walk, an unverifiable
    /// `.xdna.zip` declaration would ship silently.
    #[test]
    fn xdna_sidecar_is_digest_validated() {
        let with_bad_xdna = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{"repo":"r","file":"f.hfq","size_gb":1,"min_vram_gb":1,"desc":"d",
              "xdna":{"file":"m.xdna.zip","sha256":"not-a-sha"}}},
            "aliases":{}
        }"#;
        let err = RegistryV1::parse(with_bad_xdna, "test")
            .expect_err("a malformed xdna digest must be rejected");
        assert!(
            format!("{err}").contains("invalid SHA-256"),
            "expected a digest complaint, got: {err}"
        );
        let good = with_bad_xdna.replace("not-a-sha", &"b".repeat(64));
        let reg = RegistryV1::parse(&good, "test").expect("valid xdna must parse");
        let (_, entry) = reg.model("m").unwrap();
        assert_eq!(entry.xdna.as_ref().unwrap().file, "m.xdna.zip");
    }

    fn reg_at(stamp: &str) -> RegistryV1 {
        RegistryV1::parse(
            &format!(
                r#"{{"schema_version":1,"generated_at":"{stamp}",
                   "models":{{"m":{{"repo":"r","file":"f","size_gb":1,"min_vram_gb":1,"desc":"d"}}}},
                   "aliases":{{}}}}"#
            ),
            "test",
        )
        .unwrap()
    }

    /// A NEWER bundled registry wins — this is what makes a branch's registry
    /// edits visible to a locally built binary instead of being silently
    /// overridden by the 24h cache or a master fetch.
    #[test]
    fn newer_bundled_registry_beats_a_stale_fetch() {
        let mut w = Vec::new();
        let fetched = LoadedRegistry {
            registry: reg_at("2026-08-31T05:32:38Z"),
            source: RegistrySource::Cache,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at("2026-09-01T13:14:39Z"), &mut w);
        assert_eq!(out.source, RegistrySource::Bundled);
        assert_eq!(out.registry.generated_at, "2026-09-01T13:14:39Z");
        assert!(
            out.warnings.iter().any(|x| x.contains("newer")),
            "the override must be reported, not silent: {:?}",
            out.warnings
        );
    }

    /// The other direction is what keeps DISTRIBUTION working: a released
    /// binary's bundled registry is older than master's, so the fetch must
    /// still win and users get new models without upgrading. Without this the
    /// change above would freeze every client at its build-time registry.
    #[test]
    fn older_bundled_registry_defers_to_the_fetch() {
        let mut w = Vec::new();
        let fetched = LoadedRegistry {
            registry: reg_at("2026-09-01T13:14:39Z"),
            source: RegistrySource::Network,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at("2026-08-28T08:31:54Z"), &mut w);
        assert_eq!(out.source, RegistrySource::Network);
        assert_eq!(out.registry.generated_at, "2026-09-01T13:14:39Z");
        assert!(out.warnings.is_empty(), "no override, so nothing to report");
    }

    /// Equal stamps must not flap between sources.
    #[test]
    fn equal_timestamps_keep_the_fetched_registry() {
        let mut w = Vec::new();
        let same = "2026-09-01T13:14:39Z";
        let fetched = LoadedRegistry {
            registry: reg_at(same),
            source: RegistrySource::Cache,
            warnings: Vec::new(),
        };
        let out = prefer_bundled_if_newer(fetched, reg_at(same), &mut w);
        assert_eq!(out.source, RegistrySource::Cache);
    }

    #[test]
    fn bundled_registry_is_strictly_valid() {
        let registry = bundled().unwrap();
        assert!(registry.models.len() > 20);
        let (tag, model) = registry.model("qwen3.6:35b-a3b-mq4r").unwrap();
        assert_eq!(tag, "qwen3.6:35b-a3b-mq4r");
        assert_eq!(model.quant.as_deref(), Some("mq4r"));
        assert_eq!(
            model.sha256.as_deref(),
            Some("4685c140c46b1a6f31a0fd9053bf09d5faf1d2529d715b84794249b66cde0428")
        );

        let (tag, model) = registry.model("ornith-1.5:35b-a3b-mq4r").unwrap();
        assert_eq!(tag, "ornith-1.5:35b-a3b-mq4r");
        for alias in ["ornith-1.5:fast", "ornith-1.5:35b-a3b-fast"] {
            let (resolved, fast) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(
                resolved, "ornith-1.5:35b-a3b-mq4r",
                "{alias} is the MQ4R speed SKU"
            );
            assert_eq!(fast.file, "ornith-1.5-35b-a3b.mq4r");
        }
        assert_eq!(model.file, "ornith-1.5-35b-a3b.mq4r");
        assert_eq!(model.quant.as_deref(), Some("mq4r"));
        assert_eq!(model.size_bytes, Some(18_700_570_368));
        assert_eq!(
            model.quant_recipe.as_deref(),
            Some("mq4v2-uniform-no-q8-router@10fbf86f")
        );
        assert_eq!(
            model.sha256.as_deref(),
            Some("84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6")
        );
    }

    /// Pins Muse Glimmer's sampling contract to its model card's "Best Practices"
    /// section (`meta-models/Muse-Glimmer-30B`): temperature 1.0, top_p 0.95, top_k 64.
    ///
    /// `top_k` is the one that matters. Before this entry existed the tag resolved to
    /// nothing and callers silently fell back to `recipe(general)`, which uses the Qwen
    /// family's `top_k = 20` — a different model's sampling contract applied to Glimmer
    /// with only a warning on stderr.
    #[test]
    fn bundled_muse_glimmer_matches_the_model_card_sampling_contract() {
        let registry = bundled().unwrap();
        // Only one Glimmer size exists, so the canonical tag carries no size
        // suffix. `muse-glimmer:30b` stays a back-compat alias because
        // scripts/serve_harness.py infers it from the artifact filename.
        let (tag, model) = registry.model("muse-glimmer").unwrap();
        assert_eq!(tag, "muse-glimmer");
        assert_eq!(model.arch_id, Some(14));
        assert_eq!(model.file, "muse-glimmer-30b.mq4");
        for alias in [
            "muse-glimmer:30b",
            "muse-glimmer:latest",
            "muse-glimmer:quality",
        ] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(resolved, "muse-glimmer", "{alias} resolves to the trunk");
        }

        // The speed SKU is a distinct entry on the MQ4-attention artifact, not an
        // alias of the trunk.
        let (fast_tag, fast) = registry.model("muse-glimmer:fast").unwrap();
        assert_eq!(fast_tag, "muse-glimmer:fast");
        assert_eq!(fast.file, "muse-glimmer-30b.mq4r");
        assert_eq!(fast.arch_id, Some(14));
        assert_ne!(
            fast.file, model.file,
            "the two SKUs are different artifacts"
        );

        // The drafter is arch 23 and pairs with either SKU.
        let (draft_tag, draft) = registry.model("muse-glimmer:draft").unwrap();
        assert_eq!(draft_tag, "muse-glimmer:draft");
        assert_eq!(draft.arch_id, Some(23));

        let settings = model
            .recommended_settings
            .as_ref()
            .expect("muse-glimmer must carry the card's recommended_settings");
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(64));
        assert_eq!(settings.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.top_k, Some(64));
        assert_eq!(general.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(
            general.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // The card specifies no separate coding/instruct sampling, so those are
        // deliberately absent rather than invented.
        assert!(model.sampling_profile("coding").is_none());
        assert!(model.sampling_profile("instruct").is_none());
    }

    /// Pins Qwen3.8-27B's release contract: aliases, artifact identity, arch/quant,
    /// default KV, and the upstream sampling card (effort-native; no thinking_budget).
    #[test]
    fn bundled_qwen38_matches_the_release_contract() {
        let registry = bundled().unwrap();
        let (tag, model) = registry.model("qwen3.8:27b").unwrap();
        assert_eq!(tag, "qwen3.8:27b");
        assert_eq!(model.file, "qwen3.8-27b.mq4");
        assert_eq!(model.arch_id, Some(5));
        assert_eq!(model.quant.as_deref(), Some("mq4"));
        assert_eq!(model.default_kv_mode.as_deref(), Some("q8"));
        // Pin the canonical MQ4V2 body by content as well as filename: MQ4V2
        // supersedes the former MQ4V1/MQ4R Qwen3.8 products.
        assert_eq!(
            model.sha256.as_deref(),
            Some("5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507")
        );
        assert_eq!(model.size_bytes, Some(15662615552));

        for alias in ["qwen3.8", "qwen3.8:latest", "qwen3.8:27b-mq4"] {
            let (resolved, _) = registry
                .model(alias)
                .unwrap_or_else(|| panic!("{alias} must resolve"));
            assert_eq!(
                resolved, "qwen3.8:27b",
                "{alias} resolves to the canonical tag"
            );
        }

        // The speed tier is MQ4V2 XT. Both former `fast` tags migrate to it;
        // the superseded `.mq4r` artifact is no longer in the registry.
        let (fast_tag, fast) = registry.model("qwen3.8:27b-fast").unwrap();
        assert_eq!(fast_tag, "qwen3.8:27b-mq4-xt");
        assert_eq!(fast.file, "qwen3.8-27b.mq4-xt");
        assert_eq!(fast.arch_id, Some(5));
        assert_eq!(fast.quant.as_deref(), Some("mq4"));
        assert_eq!(fast.default_kv_mode.as_deref(), Some("q8"));
        assert_eq!(
            fast.sha256.as_deref(),
            Some("80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3")
        );
        assert_eq!(fast.size_bytes, Some(14987185152));
        assert_ne!(
            fast.sha256, model.sha256,
            "the two tiers must not share a content digest"
        );

        for alias in ["qwen3.8:fast", "qwen3.8:27b-fast"] {
            let (resolved, _) = registry.model(alias).expect("fast alias must resolve");
            assert_eq!(resolved, "qwen3.8:27b-mq4-xt");
        }
        assert!(
            registry.model("qwen3.8-27b.mq4r").is_none(),
            "superseded MQ4R filename must not remain addressable"
        );

        // The symmetric XT checkpoint (H2) is the canonical dense fixture. It has
        // exactly the asymmetric XT's size, so only the digest tells them apart;
        // it shares the MQ4 DFlash draft.
        let (xts_tag, xts) = registry.model("qwen3.8:27b-mq4-xts").unwrap();
        assert_eq!(xts_tag, "qwen3.8:27b-mq4-xts");
        assert_eq!(xts.file, "qwen3.8-27b.mq4-xts");
        assert_eq!(xts.arch_id, Some(5));
        assert_eq!(xts.quant.as_deref(), Some("mq4"));
        assert_eq!(xts.default_kv_mode.as_deref(), Some("q8"));
        assert_eq!(
            xts.sha256.as_deref(),
            Some("3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae")
        );
        assert_eq!(xts.size_bytes, Some(14987185152));
        assert_ne!(xts.sha256, fast.sha256, "XTS and XT are distinct artifacts");
        assert_eq!(
            xts.dflash.as_ref().map(|draft| &draft.file),
            fast.dflash.as_ref().map(|draft| &draft.file)
        );

        // The native Qwen3.8-27B MTP head (public checkpoint's `mtp.*` layer,
        // `mtp_extract --quant mq4`) ships as one trunk-independent sidecar:
        // it reuses the trunk's embed/lm_head, so every trunk tier pulls the
        // same bytes, pinned by content.
        let expected_mtp = Sidecar {
            file: "qwen3.8-27b.mtp".into(),
            sha256: Some("f0d46d07ded75abc095ebfdbccc167c68418a6515b187bf810395faa460bd32e".into()),
            size_bytes: Some(225716224),
        };
        for (tag, entry) in [(tag, model), (fast_tag, fast), (xts_tag, xts)] {
            assert_eq!(entry.mtp.as_ref(), Some(&expected_mtp), "{tag} MTP head");
        }
        let mut trunk_tiers = 0;
        for (tag, entry) in &registry.models {
            if !tag.starts_with("qwen3.8:27b") {
                continue;
            }
            if tag.contains("draft") {
                assert!(
                    entry.mtp.is_none(),
                    "{tag}: a DFlash draft carries no MTP head"
                );
            } else {
                assert_eq!(entry.mtp.as_ref(), Some(&expected_mtp), "{tag} MTP head");
                trunk_tiers += 1;
            }
        }
        assert!(
            trunk_tiers >= 13,
            "every Qwen3.8-27B trunk tier declares the head"
        );

        let settings = model
            .recommended_settings
            .as_ref()
            .expect("qwen3.8:27b must carry recommended_settings");
        assert_eq!(settings.temperature, Some(1.0));
        assert_eq!(settings.top_p, Some(0.95));
        assert_eq!(settings.top_k, Some(20));
        assert_eq!(settings.min_p, Some(0.0));
        assert_eq!(settings.presence_penalty, Some(0.0));
        assert_eq!(settings.repeat_penalty, Some(1.0));
        assert_eq!(settings.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // `general` must resolve to the same contract, so `--sampling registry:general`
        // and the bare default cannot diverge.
        let general = model
            .sampling_profile("general")
            .expect("general profile resolves");
        assert_eq!(general.temperature, Some(1.0));
        assert_eq!(general.top_p, Some(0.95));
        assert_eq!(general.top_k, Some(20));
        assert_eq!(general.min_p, Some(0.0));
        assert_eq!(general.presence_penalty, Some(0.0));
        assert_eq!(general.repeat_penalty, Some(1.0));
        assert_eq!(general.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            general.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );

        // Tag policy pins generation.max_tokens only for Qwen3.8; never max_seq or a KV backend.
        let layer =
            config_layer_for_tag(tag, model).expect("qwen3.8:27b tag policy lowers cleanly");
        assert!(layer.get("memory.kv_cache").is_none());
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(
            layer.get("memory.max_seq").is_none(),
            "tag layer must not write memory.max_seq"
        );
        assert_eq!(
            layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );
        // Tag policy keys off the family before ':' and excludes only draft/dflash
        // tags, so the fast SKU receives the same max_tokens pin (no max_seq/backend).
        let fast_layer = config_layer_for_tag(fast_tag, fast)
            .expect("qwen3.8:27b-fast tag policy lowers cleanly");
        assert!(fast_layer.get("memory.kv_backend").is_none());
        assert!(
            fast_layer.get("memory.max_seq").is_none(),
            "tag layer must not write memory.max_seq"
        );
        assert_eq!(
            fast_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );
        // Flash-Next carries arch_id=16 inside the qwen3.8 naming family;
        // do not inherit the ordinary Qwen3.8 VMM/262K policy.
        let (flash_tag, flash) = registry.model("qwen3.8:flash-next").unwrap();
        assert_eq!(flash.arch_id, Some(16));
        let flash_layer = config_layer_for_tag(flash_tag, flash)
            .expect("qwen3.8:flash-next must lower without a family policy");
        assert!(flash_layer.get("memory.kv_backend").is_none());
        assert!(flash_layer.get("memory.max_seq").is_none());
        assert!(flash_layer.get("generation.max_tokens").is_none());
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("medium".into()))
        );
        // Original Qwen3 family (without .5/.6/.8) receives no automatic policy.
        let qwen3_raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"}},
            "aliases":{}
        }"#;
        let qwen3_registry = RegistryV1::parse(qwen3_raw, "test").unwrap();
        let (q3_tag, q3_entry) = qwen3_registry.model("qwen3:8b").unwrap();
        let q3_layer = config_layer_for_tag(q3_tag, q3_entry).unwrap();
        assert!(q3_layer.get("memory.kv_cache").is_none());
        assert!(q3_layer.get("memory.kv_backend").is_none());
        assert!(q3_layer.get("memory.max_seq").is_none());
        assert!(q3_layer.get("generation.max_tokens").is_none());
    }

    /// Qwen3.8 dense tiers and both Ornith 1.5 artifacts share the effort-native
    /// Qwen3.8 contract: the registry default and general profile are medium.
    /// Flash-Next (Qwen4, arch_id=16), sidecar drafts, and the intentional
    /// Ornith coding/instruct profiles are unchanged.
    #[test]
    fn bundled_qwen38_family_effort_defaults_medium() {
        let registry = bundled().unwrap();
        let mut tiers = 0;
        for (tag, entry) in &registry.models {
            let is_dense_trunk = tag.starts_with("qwen3.8:27b") && !tag.contains("-draft");
            let is_ornith = tag.starts_with("ornith-1.5:");
            if !is_dense_trunk && !is_ornith {
                continue;
            }
            let settings = entry
                .recommended_settings
                .as_ref()
                .unwrap_or_else(|| panic!("{tag} must carry recommended_settings"));
            assert_eq!(
                settings.reasoning_effort.as_deref(),
                Some("medium"),
                "{tag} recommended_settings effort"
            );
            let general = entry
                .sampling_profile("general")
                .unwrap_or_else(|| panic!("{tag} must carry a general profile"));
            assert_eq!(
                general.reasoning_effort.as_deref(),
                Some("medium"),
                "{tag} general profile effort"
            );
            tiers += 1;
        }
        assert!(tiers >= 15, "13 Qwen3.8-27B tiers plus 2 Ornith artifacts");

        // Flash-Next (arch_id=16) never carried an effort default.
        let (_, flash) = registry.model("qwen3.8:flash-next").unwrap();
        assert_eq!(flash.arch_id, Some(16));
        assert!(!flash
            .recommended_settings
            .as_ref()
            .is_some_and(|settings| settings.reasoning_effort.is_some()));
    }

    #[test]
    fn bundled_0731_mq2r_is_default_and_mq2lloyd_stays_addressable() {
        let registry = bundled().unwrap();
        // MQ2R became the default on 2026-08-14 on the DeepSeek V4 contributor's
        // recommendation. The bare tag and the explicit `:mq2r` name must serve
        // the SAME artifact, while `:mq2lloyd` stays a distinct, still-pullable
        // identity — a silent collapse of the two would ship 86 GB under an 82 GB
        // name or vice versa.
        let (_, default_sku) = registry.model("deepseek-v4-flash").unwrap();
        let (_, mq2r) = registry.model("deepseek-v4-flash:mq2r").unwrap();
        let (_, mq2lloyd) = registry.model("deepseek-v4-flash:mq2lloyd").unwrap();

        assert_eq!(default_sku.file, "deepseek-v4-flash-0731.mq2r");
        assert_eq!(mq2r.file, "deepseek-v4-flash-0731.mq2r");
        assert_eq!(mq2lloyd.file, "deepseek-v4-flash-0731.mq2lloyd");
        assert_ne!(mq2r.sha256, mq2lloyd.sha256);

        assert_eq!(default_sku.default_kv_mode.as_deref(), Some("f32"));
        assert_eq!(mq2lloyd.default_kv_mode.as_deref(), Some("f32"));
        assert_eq!(
            mq2r.sha256.as_deref(),
            Some("cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce")
        );
        assert_eq!(
            mq2lloyd.sha256.as_deref(),
            Some("521c9687a3f5c12fb3d89bde4a3ed202698b95ae5a102d0b5ba7f3abb87982d0")
        );
        assert_eq!(
            mq2r.quant_recipe.as_deref(),
            Some("deepseek4-mq2r-e8-p3-v1")
        );

        assert_eq!(
            mq2r.dspark
                .as_ref()
                .and_then(|sidecar| sidecar.sha256.as_deref()),
            Some("bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e")
        );
        assert!(registry.model("deepseek4:preview-mq2r").is_none());

        // Effort-native DeepSeek defaults: main low / coding max / instruct none;
        // no registry thinking_budget (absence = uncapped). Preview defaults high.
        let settings = default_sku.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.reasoning_effort.as_deref(), Some("low"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
        let general = default_sku.sampling_profile("general").unwrap();
        assert_eq!(general.reasoning_effort.as_deref(), Some("low"));
        assert!(general.thinking_budget.is_none());
        let coding = default_sku.sampling_profile("coding").unwrap();
        assert_eq!(coding.reasoning_effort.as_deref(), Some("max"));
        assert!(coding.thinking_budget.is_none());
        let instruct = default_sku.sampling_profile("instruct").unwrap();
        assert_eq!(instruct.reasoning_effort.as_deref(), Some("none"));
        assert!(instruct.thinking_budget.is_none());
        let (_, preview) = registry.model("deepseek-v4-flash-preview").unwrap();
        let preview_rs = preview.recommended_settings.as_ref().unwrap();
        assert_eq!(preview_rs.reasoning_effort.as_deref(), Some("high"));
        assert!(
            preview_rs.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
    }
    #[test]
    fn bundled_dflash_sidecars_name_pullable_files() {
        // Every `dflash.file` must name a file that some registry entry's
        // `file` also names, so `hipfire pull <target>` fetching the sidecar
        // always lands a file that `hipfire pull <tag>-draft` could fetch too.
        let registry = bundled().unwrap();
        let files: std::collections::BTreeSet<&str> = registry
            .models
            .values()
            .map(|entry| entry.file.as_str())
            .collect();
        let mut paired = 0;
        for (tag, entry) in &registry.models {
            if let Some(sidecar) = entry.dflash.as_ref() {
                assert!(
                    files.contains(sidecar.file.as_str()),
                    "model '{tag}' declares dflash sidecar '{}' with no matching entry file",
                    sidecar.file
                );
                paired += 1;
            }
        }
        assert!(
            paired > 0,
            "bundled registry should pair at least one dflash sidecar"
        );
    }

    #[test]
    fn bundled_vision_sidecar_is_shared_across_qwen38_tiers() {
        // Every non-draft `qwen3.8:27b*` target tier declares the same
        // vision-tower sidecar (`qwen3.8-27b-vision.hfq`) so each text quant
        // tier serves images without requantizing the trunk. `-draft*` tags
        // are text-only DFlash controller sidecars and carry no vision. Unlike
        // dflash, the vision file is standalone: no registry entry's `file`
        // names it.
        let registry = bundled().unwrap();
        let mut declared = 0;
        for (tag, entry) in &registry.models {
            if !tag.starts_with("qwen3.8:27b") || tag.contains("-draft") {
                continue;
            }
            let sidecar = entry
                .vision
                .as_ref()
                .unwrap_or_else(|| panic!("model '{tag}' must declare the shared vision sidecar"));
            assert_eq!(
                sidecar.file, "qwen3.8-27b-vision.hfq",
                "model '{tag}' must share qwen3.8-27b-vision.hfq"
            );
            declared += 1;
        }
        assert!(
            declared > 0,
            "bundled registry should declare a vision sidecar"
        );
    }

    #[test]
    fn aliases_and_filenames_resolve_to_canonical_tags() {
        let registry = bundled().unwrap();
        assert_eq!(registry.resolve_tag("qwen3.6"), "qwen3.6:35b-a3b");
        assert_eq!(
            registry.resolve_tag("qwen3.6-35b-a3b.mq4r"),
            "qwen3.6:35b-a3b-mq4r"
        );
        assert_eq!(registry.resolve_tag("qwen3.8-27b.mq4"), "qwen3.8:27b");
        assert_eq!(
            registry.resolve_tag("qwen3.8-27b.mq4-xt"),
            "qwen3.8:27b-mq4-xt"
        );
        assert_eq!(registry.resolve_tag("qwen3.8:fast"), "qwen3.8:27b-mq4-xt");
        // A path is not a tag: even a path into the models directory resolves
        // to itself here. Callers that know the models directory establish
        // identity with canonical path comparison (`registry_entry_for_path`
        // in hipfire-cli), so `serve --model ~/.hipfire/models/<file>` still
        // sees the entry's sidecars without letting a same-basename lookalike
        // elsewhere inherit them (PR #686 hw-gate regression).
        assert_eq!(
            registry.resolve_tag("/home/u/.hipfire/models/qwen3.8-27b.mq5"),
            "/home/u/.hipfire/models/qwen3.8-27b.mq5"
        );
        assert_eq!(
            registry.resolve_tag("/home/u/.hipfire/models/qwen3.8-27b.mq4"),
            "/home/u/.hipfire/models/qwen3.8-27b.mq4"
        );
        assert!(registry
            .model("/home/u/.hipfire/models/qwen3.8-27b.mq4")
            .is_none());

        assert_eq!(registry.resolve_tag("deepseek4"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("deepseek4:0731"), "deepseek-v4-flash");
        // `:mq2r` names now land on the default tag, since MQ2R *is* the default.
        assert_eq!(
            registry.resolve_tag("deepseek4:0731-mq2r"),
            "deepseek-v4-flash"
        );
        assert_eq!(
            registry.resolve_tag("deepseek-v4-flash-0731.mq2r"),
            "deepseek-v4-flash"
        );
        // The superseded artifact must still be reachable by its own filename.
        assert_eq!(
            registry.resolve_tag("deepseek-v4-flash-0731.mq2lloyd"),
            "deepseek-v4-flash:mq2lloyd"
        );
        assert_eq!(
            registry.resolve_tag("deepseek4:preview"),
            "deepseek-v4-flash-preview"
        );
        assert!(registry.model("deepseek4:preview-mq2r").is_none());

        // `ds4` is the short name used throughout the tree (ds4-adapter-r128.bin,
        // ds4_length_sweep.sh, hipfire-arch-deepseek4), so it resolves too, and
        // mirrors the full `deepseek4:*` surface.
        assert_eq!(registry.resolve_tag("ds4"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("ds4:0731"), "deepseek-v4-flash");
        assert_eq!(registry.resolve_tag("ds4:mq2r"), "deepseek-v4-flash");
        assert_eq!(
            registry.resolve_tag("ds4:preview"),
            "deepseek-v4-flash-preview"
        );
        // Every SKU reachable by a short `:mq2r` name must also be reachable by
        // the matching `:mq2lloyd` one — otherwise demoting MQ2-Lloyd from the
        // default silently strands it behind long-form names only.
        for name in [
            "deepseek4:mq2lloyd",
            "deepseek-v4:mq2lloyd",
            "deepseek4:0731-mq2lloyd",
            "ds4:mq2lloyd",
            "ds4:0731-mq2lloyd",
        ] {
            assert_eq!(
                registry.resolve_tag(name),
                "deepseek-v4-flash:mq2lloyd",
                "{name} must resolve to the MQ2-Lloyd tag"
            );
        }
        let (_, default_sku) = registry.model("deepseek4:0731").unwrap();
        let settings = default_sku.recommended_settings.as_ref().unwrap();
        assert_eq!(settings.reasoning_effort.as_deref(), Some("low"));
        assert!(
            settings.thinking_budget.is_none(),
            "effort-native: absence means uncapped"
        );
    }

    #[test]
    fn path_inputs_never_resolve_to_registry_tags() {
        // PR #686 hw-gate regression: `hipfire rm /elsewhere/qwen3.6-27b.mq4`
        // basename-matched the `qwen3.6:27b` entry and deleted the installed
        // sidecars. A path — anywhere, including the models directory itself —
        // is not a tag and resolves to itself.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-10T00:00:00Z",
            "models":{"qwen3.6:27b":{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        assert_eq!(
            registry.resolve_tag("/elsewhere/qwen3.6-27b.mq4"),
            "/elsewhere/qwen3.6-27b.mq4"
        );
        assert!(registry.model("/elsewhere/qwen3.6-27b.mq4").is_none());
        // The bare file name still resolves: it carries no directory to confuse
        // with the installed artifact.
        assert_eq!(registry.resolve_tag("qwen3.6-27b.mq4"), "qwen3.6:27b");
        assert!(registry.model("qwen3.6-27b.mq4").is_some());
    }

    #[test]
    fn entry_for_file_matches_exact_bare_names_only() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-10T00:00:00Z",
            "models":{"qwen3.6:27b":{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{"qwen36":"qwen3.6:27b"}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, _) = registry.entry_for_file("qwen3.6-27b.mq4").unwrap();
        assert_eq!(tag, "qwen3.6:27b");
        assert!(registry
            .entry_for_file("/elsewhere/qwen3.6-27b.mq4")
            .is_none());
        assert!(registry.entry_for_file("qwen3.6:27b").is_none());
        assert!(registry.entry_for_file("qwen36").is_none());
        assert!(registry.entry_for_file("other.mq4").is_none());
    }

    #[test]
    fn recommended_settings_lower_the_full_sampling_contract_to_config() {
        let settings = RecommendedSettings {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(40),
            min_p: Some(0.05),
            presence_penalty: Some(1.5),
            repeat_penalty: Some(1.05),
            system_prompt: Some("You are MiniMax.".into()),
            reasoning_effort: Some("high".into()),
            thinking_budget: Some("xhigh".into()),
        };
        let layer = settings.config_layer().unwrap();
        assert_eq!(
            layer.get("generation.temperature"),
            Some(&ConfigValue::Float(1.0))
        );
        assert_eq!(
            layer.get("generation.top_p"),
            Some(&ConfigValue::Float(0.95))
        );
        assert_eq!(
            layer.get("generation.top_k"),
            Some(&ConfigValue::Integer(40))
        );
        assert_eq!(
            layer.get("generation.min_p"),
            Some(&ConfigValue::Float(0.05))
        );
        assert_eq!(
            layer.get("generation.presence_penalty"),
            Some(&ConfigValue::Float(1.5))
        );
        assert_eq!(
            layer.get("generation.repeat_penalty"),
            Some(&ConfigValue::Float(1.05))
        );
        assert_eq!(
            layer.get("prompt.system"),
            Some(&ConfigValue::String("You are MiniMax.".into()))
        );
        assert_eq!(
            layer.get("reasoning.effort"),
            Some(&ConfigValue::String("high".into()))
        );
        assert_eq!(
            layer.get("reasoning.budget"),
            Some(&ConfigValue::String("xhigh".into()))
        );
    }

    #[test]
    fn malformed_entry_rejects_the_whole_registry() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"magic4"}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn tag_policy_pins_qwen_deepseek_and_glimmer_targets() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:4b":{"repo":"x","file":"qwen3.5-4b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.6:35b-a3b":{"repo":"x","file":"qwen3.6-35b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.5:27b-draft":{"repo":"x","file":"qwen35-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.5:27b-dflash":{"repo":"x","file":"qwen35-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash":{"repo":"x","file":"deepseek-v4-flash-0731.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"deepseek-v4-flash-0731.mq2lloyd","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash-preview":{"repo":"x","file":"deepseek-v4-flash-preview.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:draft":{"repo":"x","file":"deepseek-v4-flash-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:fast":{"repo":"x","file":"muse-glimmer-30b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:draft":{"repo":"x","file":"muse-glimmer-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "other:model":{"repo":"x","file":"other.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{
                "deepseek4":"deepseek-v4-flash",
                "ds4":"deepseek-v4-flash",
                "deepseek4:preview":"deepseek-v4-flash-preview",
                "muse-glimmer:quality":"muse-glimmer"
            }
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Exact Qwen families get max_tokens 81920 only; never max_seq or a KV backend.
        for tag in ["qwen3.5:4b", "qwen3.6:35b-a3b", "qwen3.8:27b"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{tag} must not write memory.max_seq"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(81920)),
                "{tag} should get 81920"
            );
        }

        // Original Qwen3 (no dot) receives none
        let (_, entry) = registry.model("qwen3:8b").unwrap();
        let layer = config_layer_for_tag("qwen3:8b", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Draft/dflash sidecars do not get the Qwen policy
        for tag in ["qwen3.5:27b-draft", "qwen3.5:27b-dflash"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} sidecar must not write a KV backend"
            );
            assert!(layer.get("memory.max_seq").is_none());
            assert!(layer.get("generation.max_tokens").is_none());
        }

        // DeepSeek official / MQ2Lloyd / preview targets get max_tokens 393216 only.
        for tag in [
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(resolved, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{tag} must not write memory.max_seq"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(393216)),
                "{tag} should get 393216"
            );
        }
        // Aliases resolve to canonical families that carry the same policy.
        for alias in ["deepseek4", "ds4", "deepseek4:preview"] {
            let (resolved, entry) = registry.model(alias).unwrap();
            let layer = config_layer_for_tag(resolved, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{alias}->{resolved} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{alias}->{resolved} must not write memory.max_seq"
            );
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(393216)),
                "{alias}->{resolved} should get 393216"
            );
        }
        // DeepSeek draft sidecar receives none
        let (_, entry) = registry.model("deepseek-v4-flash:draft").unwrap();
        let layer = config_layer_for_tag("deepseek-v4-flash:draft", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Muse Glimmer quality and fast targets write nothing (no max_seq/max_tokens/backend).
        for tag in ["muse-glimmer", "muse-glimmer:fast"] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("memory.kv_backend").is_none(),
                "{tag} must not write a KV backend"
            );
            assert!(
                layer.get("memory.max_seq").is_none(),
                "{tag} must not write memory.max_seq"
            );
            assert!(
                layer.get("generation.max_tokens").is_none(),
                "{tag} must not get max_tokens"
            );
        }
        // quality alias lands on trunk and inherits the same (empty) policy.
        let (resolved, entry) = registry.model("muse-glimmer:quality").unwrap();
        assert_eq!(resolved, "muse-glimmer");
        let layer = config_layer_for_tag(resolved, entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Muse Glimmer draft receives none
        let (_, entry) = registry.model("muse-glimmer:draft").unwrap();
        let layer = config_layer_for_tag("muse-glimmer:draft", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());

        // Absent policy: unrelated models get no automatic policy
        let (_, entry) = registry.model("other:model").unwrap();
        let layer = config_layer_for_tag("other:model", entry).unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());
    }

    #[test]
    fn tag_policy_does_not_expand_wire_schema() {
        // Old v1 JSON without the invented fields must still parse (deny_unknown_fields).
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"ok":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("ok").unwrap();
        let layer = entry.config_layer().unwrap();
        assert!(layer.get("memory.kv_backend").is_none());
        assert!(layer.get("memory.max_seq").is_none());
        assert!(layer.get("generation.max_tokens").is_none());
        // Tag policy also absent for non-target.
        let tagged = config_layer_for_tag("ok", entry).unwrap();
        assert!(tagged.get("memory.kv_backend").is_none());

        // Invented wire fields must be rejected (no schema expansion).
        let bad = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_backend":"vmm"}},
            "aliases":{}
        }"#;
        assert!(
            RegistryV1::parse(bad, "test").is_err(),
            "default_kv_backend must be rejected as unknown field"
        );
        let bad2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_seq":262144}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad2, "test").is_err());
        let bad3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"bad":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","default_max_tokens":81920}},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(bad3, "test").is_err());
    }

    #[test]
    fn qwen38_27b_dflash_auto_is_scoped_to_paired_trunk_tiers() {
        // `speculation.dflash = "auto"` is lowered only for canonical Qwen3.8-27B
        // trunk tags (`qwen3.8:27b`, `qwen3.8:27b-*`) that declare a `dflash`
        // sidecar, are not draft/dflash sidecars, and are not arch_id=16.
        let dflash = r#""dflash":{"file":"qwen38-27b-dflash-mq4.hfq"}"#;
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{{
                "qwen3.8:27b":{{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq3":{{"repo":"x","file":"qwen3.8-27b.mq3","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq4-pro":{{"repo":"x","file":"qwen3.8-27b.mq4-pro","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq4-xt":{{"repo":"x","file":"qwen3.8-27b.mq4-xt","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:27b-mq5":{{"repo":"x","file":"qwen3.8-27b.mq5","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5}},
                "qwen3.8:27boops":{{"repo":"x","file":"qwen3.8-27boops.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,{dflash}}},
                "qwen3.8:flash-next":{{"repo":"x","file":"qwen3.8-flash-next.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":16,{dflash}}},
                "qwen3.8:27b-draft-mq4":{{"repo":"x","file":"qwen38-27b-draft.hfq","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":20,{dflash}}},
                "qwen3.8:27b-dflash-mq4":{{"repo":"x","file":"qwen38-27b-dflash.hfq","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":20,{dflash}}},
                "qwen3.6:27b":{{"repo":"x","file":"qwen3.6-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3.5:27b":{{"repo":"x","file":"qwen3.5-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3:8b":{{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}},
                "qwen3.8:9b":{{"repo":"x","file":"qwen3.8-9b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",{dflash}}}
            }},
            "aliases":{{
                "qwen3.8":"qwen3.8:27b",
                "qwen3.8:latest":"qwen3.8:27b",
                "qwen3.8:27b-mq4":"qwen3.8:27b",
                "qwen3.8:fast":"qwen3.8:27b-mq4-xt",
                "qwen3.8:27b-fast":"qwen3.8:27b-mq4-xt"
            }}
        }}"#
        );
        let registry = RegistryV1::parse(&raw, "test").unwrap();
        let auto_value = ConfigValue::String("auto".into());
        let auto = Some(&auto_value);

        // Canonical tags and every alias that resolves to an eligible tag.
        for input in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq3",
            "qwen3.8:27b-mq4-pro",
            "qwen3.8:27b-mq4-xt",
            "qwen3.8",
            "qwen3.8:latest",
            "qwen3.8:27b-mq4",
            "qwen3.8:fast",
            "qwen3.8:27b-fast",
        ] {
            let (tag, entry) = registry.model(input).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert_eq!(layer.get("speculation.dflash"), auto, "{input}->{tag}");
            // Only the dflash mode is added; MTP and the ngram/mode knobs are untouched.
            assert!(layer.get("speculation.mtp").is_none(), "{input}->{tag}");
            assert!(layer.get("speculation.mode").is_none(), "{input}->{tag}");
            assert_eq!(
                layer.get("generation.max_tokens"),
                Some(&ConfigValue::Integer(81920)),
                "{input}->{tag} keeps the existing max_tokens policy"
            );
        }

        // Same family/tier without a registry dflash declaration, a longer
        // lookalike tier, arch_id=16, draft/dflash sidecar tags, and other
        // families/sizes never get the policy.
        for tag in [
            "qwen3.8:27b-mq5",
            "qwen3.8:27boops",
            "qwen3.8:flash-next",
            "qwen3.8:27b-draft-mq4",
            "qwen3.8:27b-dflash-mq4",
            "qwen3.6:27b",
            "qwen3.5:27b",
            "qwen3:8b",
            "qwen3.8:9b",
        ] {
            let (_, entry) = registry.model(tag).unwrap();
            let layer = config_layer_for_tag(tag, entry).unwrap();
            assert!(
                layer.get("speculation.dflash").is_none(),
                "{tag} must not default DFlash"
            );
        }

        // arch_id and the dflash declaration are both required, not just the tag.
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let mut no_dflash = entry.clone();
        no_dflash.dflash = None;
        assert!(config_layer_for_tag(tag, &no_dflash)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
        let mut qwen4 = entry.clone();
        qwen4.arch_id = Some(16);
        assert!(config_layer_for_tag(tag, &qwen4)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
        // A different tag with the same dflash-less entry stays off too.
        assert!(config_layer_for_tag("qwen3.8:27b-mq4-pro", &no_dflash)
            .unwrap()
            .get("speculation.dflash")
            .is_none());
    }

    #[test]
    fn bundled_dflash_auto_matches_only_declared_qwen38_27b_tiers() {
        // Source-grounded against checked-in registry/v1.json: DFlash-paired
        // Qwen3.8-27B trunk tiers lower `auto`; mq5/mq6 tiers (no dflash
        // declaration), Flash-Next, drafts, and other families do not.
        let registry = bundled().unwrap();
        let mut auto_tiers = 0;
        for (tag, entry) in &registry.models {
            let layer = config_layer_for_tag(tag, entry).unwrap();
            let expected = (tag == "qwen3.8:27b" || tag.starts_with("qwen3.8:27b-"))
                && !tag.contains("draft")
                && !tag.contains("dflash")
                && entry.arch_id != Some(16)
                && entry.dflash.is_some();
            assert_eq!(
                layer.get("speculation.dflash") == Some(&ConfigValue::String("auto".into())),
                expected,
                "{tag}"
            );
            if expected {
                auto_tiers += 1;
            } else {
                assert!(layer.get("speculation.dflash").is_none(), "{tag}");
            }
        }
        assert!(
            auto_tiers >= 1,
            "at least one bundled Qwen3.8-27B tier declares a DFlash draft"
        );
        for tag in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq3",
            "qwen3.8:27b-mq3-xt",
            "qwen3.8:27b-mq3-pro",
            "qwen3.8:27b-mq4-xt",
            "qwen3.8:27b-mq4-xts",
            "qwen3.8:27b-mq4-pro",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            assert!(entry.dflash.is_some(), "{tag} declares a DFlash draft");
            assert_eq!(
                config_layer_for_tag(resolved, entry)
                    .unwrap()
                    .get("speculation.dflash"),
                Some(&ConfigValue::String("auto".into())),
                "{tag}"
            );
        }
        for tag in [
            "qwen3.8:27b-mq5",
            "qwen3.8:27b-mq6-pro",
            "qwen3.8:flash-next",
            "qwen3.8:27b-draft-mq4",
        ] {
            let (resolved, entry) = registry.model(tag).unwrap();
            assert!(
                config_layer_for_tag(resolved, entry)
                    .unwrap()
                    .get("speculation.dflash")
                    .is_none(),
                "{tag}"
            );
        }
        for alias in [
            "qwen3.8",
            "qwen3.8:latest",
            "qwen3.8:fast",
            "qwen3.8:27b-fast",
        ] {
            let (resolved, entry) = registry.model(alias).unwrap();
            assert_eq!(
                config_layer_for_tag(resolved, entry)
                    .unwrap()
                    .get("speculation.dflash"),
                Some(&ConfigValue::String("auto".into())),
                "{alias}->{resolved}"
            );
        }
    }

    #[test]
    fn qwen38_27b_dflash_auto_yields_to_explicit_user_off() {
        // Registry layer is lowest; global/model/one-shot dflash=off wins.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","arch_id":5,"dflash":{"file":"qwen38-27b-dflash-mq4.hfq"}}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let registry_named = || hipfire_config::NamedLayer {
            source: hipfire_config::ConfigSource::RegistryModel {
                tag: tag.to_owned(),
                revision: "v1".into(),
            },
            layer: config_layer_for_tag(tag, entry).unwrap(),
        };
        let dflash_of = |layers: Vec<hipfire_config::NamedLayer>| {
            hipfire_config::resolve(layers)
                .unwrap()
                .get("speculation.dflash")
                .map(|item| item.value.clone())
        };
        // Registry alone: auto beats the BuiltIn off default.
        assert_eq!(
            dflash_of(vec![registry_named()]),
            Some(ConfigValue::String("auto".into()))
        );
        // Explicit global-user off wins.
        let mut global_off = ConfigLayer::default();
        global_off
            .set("speculation.dflash", ConfigValue::String("off".into()))
            .unwrap();
        assert_eq!(
            dflash_of(vec![
                registry_named(),
                hipfire_config::NamedLayer {
                    source: hipfire_config::ConfigSource::GlobalUser {
                        path: std::path::PathBuf::from("/tmp/global.toml"),
                    },
                    layer: global_off,
                },
            ]),
            Some(ConfigValue::String("off".into()))
        );
        // Explicit per-model-user off wins too.
        let mut model_off = ConfigLayer::default();
        model_off
            .set("speculation.dflash", ConfigValue::String("off".into()))
            .unwrap();
        assert_eq!(
            dflash_of(vec![
                registry_named(),
                hipfire_config::NamedLayer {
                    source: hipfire_config::ConfigSource::ModelUser {
                        model: tag.to_owned(),
                        path: std::path::PathBuf::from("/tmp/model.toml"),
                    },
                    layer: model_off,
                },
            ]),
            Some(ConfigValue::String("off".into()))
        );
    }

    #[test]
    fn tag_policy_explicit_override_wins_over_registry() {
        // Registry tag policy is a low-precedence layer; global/model/one-shot user config wins.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let registry_layer = config_layer_for_tag(tag, entry).unwrap();
        assert!(registry_layer.get("memory.kv_backend").is_none());
        assert!(
            registry_layer.get("memory.max_seq").is_none(),
            "tag layer must not write memory.max_seq"
        );
        assert_eq!(
            registry_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(81920))
        );

        // Simulate user global override to legacy + different max_seq/max_tokens.
        let mut user_layer = ConfigLayer::default();
        user_layer
            .set("memory.kv_backend", ConfigValue::String("legacy".into()))
            .unwrap();
        user_layer
            .set("memory.max_seq", ConfigValue::Integer(32768))
            .unwrap();
        user_layer
            .set("generation.max_tokens", ConfigValue::Integer(1024))
            .unwrap();

        let resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test.toml"),
                },
                layer: user_layer,
            },
        ])
        .unwrap();
        // User wins.
        assert_eq!(
            resolved.get("memory.kv_backend").map(|value| &value.value),
            Some(&ConfigValue::String("legacy".into()))
        );
        assert_eq!(
            resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(32768))
        );
        assert_eq!(
            resolved
                .get("generation.max_tokens")
                .map(|value| &value.value),
            Some(&ConfigValue::Integer(1024))
        );

        // Glimmer target override likewise wins (backend + max_seq); tag writes neither.
        let raw2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry2 = RegistryV1::parse(raw2, "test").unwrap();
        let (g_tag, g_entry) = registry2.model("muse-glimmer").unwrap();
        let g_registry_layer = config_layer_for_tag(g_tag, g_entry).unwrap();
        assert!(g_registry_layer.get("memory.kv_backend").is_none());
        assert!(g_registry_layer.get("memory.max_seq").is_none());
        assert!(g_registry_layer.get("generation.max_tokens").is_none());
        let mut g_user = ConfigLayer::default();
        g_user
            .set("memory.kv_backend", ConfigValue::String("legacy".into()))
            .unwrap();
        g_user
            .set("memory.max_seq", ConfigValue::Integer(8192))
            .unwrap();
        let g_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: g_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: g_registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test2.toml"),
                },
                layer: g_user,
            },
        ])
        .unwrap();
        assert_eq!(
            g_resolved
                .get("memory.kv_backend")
                .map(|value| &value.value),
            Some(&ConfigValue::String("legacy".into()))
        );
        assert_eq!(
            g_resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(8192))
        );

        // DeepSeek target override wins over max_tokens policy; tag writes no max_seq.
        let raw3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"deepseek-v4-flash":{"repo":"x","file":"ds4.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry3 = RegistryV1::parse(raw3, "test").unwrap();
        let (d_tag, d_entry) = registry3.model("deepseek-v4-flash").unwrap();
        let d_registry_layer = config_layer_for_tag(d_tag, d_entry).unwrap();
        assert!(d_registry_layer.get("memory.max_seq").is_none());
        assert_eq!(
            d_registry_layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(393216))
        );
        let mut d_user = ConfigLayer::default();
        d_user
            .set("memory.max_seq", ConfigValue::Integer(65536))
            .unwrap();
        d_user
            .set("generation.max_tokens", ConfigValue::Integer(2048))
            .unwrap();
        let d_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: d_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: d_registry_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test3.toml"),
                },
                layer: d_user,
            },
        ])
        .unwrap();
        assert_eq!(
            d_resolved.get("memory.max_seq").map(|value| &value.value),
            Some(&ConfigValue::Integer(65536))
        );
        assert_eq!(
            d_resolved
                .get("generation.max_tokens")
                .map(|value| &value.value),
            Some(&ConfigValue::Integer(2048))
        );
    }

    #[test]
    fn dangling_aliases_are_dropped() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"ok":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{"good":"ok","bad":"missing"}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        assert_eq!(registry.aliases.len(), 1);
        assert_eq!(registry.resolve_tag("good"), "ok");
    }

    #[test]
    fn future_cache_timestamps_are_stale() {
        let cache = RegistryCache {
            fetched_at: 101,
            url: "x".into(),
            registry: bundled().unwrap(),
        };
        assert!(!cache_is_fresh(&cache, 100, REGISTRY_CACHE_TTL));
    }

    #[test]
    fn sampling_profiles_resolve_per_mode_with_general_fallback() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "recommended_settings":{"temperature":1.0,"presence_penalty":1.5},
                "sampling_profiles":{
                    "coding":{"temperature":0.6,"presence_penalty":0.0},
                    "instruct":{"temperature":0.7,"top_p":0.8}
                }
            }},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("m").unwrap();
        assert_eq!(
            entry.sampling_profile("coding").unwrap().temperature,
            Some(0.6)
        );
        assert_eq!(entry.sampling_profile("instruct").unwrap().top_p, Some(0.8));
        // general has no explicit profile → falls back to recommended_settings.
        assert_eq!(
            entry.sampling_profile("general").unwrap().presence_penalty,
            Some(1.5)
        );
        assert!(entry.sampling_profile("nope").is_none());
    }

    #[test]
    fn out_of_range_sampling_profile_rejects_the_whole_registry() {
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"m":{
                "repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x",
                "sampling_profiles":{"coding":{"temperature":9.0}}
            }},
            "aliases":{}
        }"#;
        assert!(RegistryV1::parse(raw, "test").is_err());
    }

    #[test]
    fn bundled_registry_passes_reasoning_validation() {
        // Bundled v1.json must validate under the mirrored generator invariants:
        // effort-native families have no thinking_budget (absence = uncapped)
        // and all present enums are recognized.
        let registry = bundled().unwrap();
        // Spot-check that effort-native bundled entries indeed omit budget
        for tag in [
            "qwen3.8:27b",
            "qwen3.8:27b-mq4-xt",
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
            "muse-glimmer",
            "muse-glimmer:fast",
            "ornith-1.5:35b-a3b",
            "ornith-1.5:35b-a3b-mq4r",
        ] {
            let (_, entry) = registry
                .model(tag)
                .unwrap_or_else(|| panic!("{tag} must exist"));
            if let Some(rs) = &entry.recommended_settings {
                assert!(
                    rs.thinking_budget.is_none(),
                    "{tag} bundled thinking_budget must be absent"
                );
            }
            if let Some(profiles) = &entry.sampling_profiles {
                for (_, settings) in [
                    ("general", &profiles.general),
                    ("coding", &profiles.coding),
                    ("instruct", &profiles.instruct),
                ] {
                    if let Some(settings) = settings {
                        assert!(
                            settings.thinking_budget.is_none(),
                            "{tag} profile thinking_budget must be absent"
                        );
                    }
                }
            }
        }

        // Both Ornith 1.5 artifacts share the Qwen3.8 effort-native contract:
        // default/general medium (hipfire default; xhigh stays selectable),
        // explicit coding profile xhigh, instruct none, and no named
        // thinking_budget (absence = uncapped).
        for tag in ["ornith-1.5:35b-a3b", "ornith-1.5:35b-a3b-mq4r"] {
            let (_, ornith) = registry
                .model(tag)
                .unwrap_or_else(|| panic!("{tag} must exist"));
            let ornith_rs = ornith.recommended_settings.as_ref().unwrap();
            assert_eq!(ornith_rs.reasoning_effort.as_deref(), Some("medium"));
            assert!(
                ornith_rs.thinking_budget.is_none(),
                "{tag} effort-native: absence means uncapped"
            );
            let ornith_general = ornith.sampling_profile("general").unwrap();
            assert_eq!(ornith_general.reasoning_effort.as_deref(), Some("medium"));
            assert!(ornith_general.thinking_budget.is_none());
            let ornith_coding = ornith.sampling_profile("coding").unwrap();
            assert_eq!(ornith_coding.reasoning_effort.as_deref(), Some("xhigh"));
            assert!(ornith_coding.thinking_budget.is_none());
            let ornith_instruct = ornith.sampling_profile("instruct").unwrap();
            assert_eq!(ornith_instruct.reasoning_effort.as_deref(), Some("none"));
            assert!(ornith_instruct.thinking_budget.is_none());
        }
    }

    #[test]
    fn effort_native_thinking_budget_rejected_for_fetched_or_cached_registry() {
        // Mirrors scripts/registry_gen.py:_effort_native_tag. A fetched/cache
        // registry that reintroduces a named thinking_budget on effort-native
        // families must fail validation wholesale so load() falls back to
        // bundled (network) or discards the cache entry (fresh/stale).
        let cases = [
            // Qwen3.8 product SKUs
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"high"}}},"aliases":{}}"#,
            // DeepSeek V4 Flash (also covers :mq2lloyd via family)
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"low","thinking_budget":"uncapped"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash-preview":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"high","thinking_budget":"med"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"low"}}},"aliases":{}}"#,
            // Muse Glimmer product SKUs
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"xhigh"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer:fast":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"max"}}},"aliases":{}}"#,
            // Ornith 1.5 product + legacy family spellings
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith-1.5:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"xhigh","thinking_budget":"high"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith1.5:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"uncapped"}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"med"}}},"aliases":{}}"#,
            // Effort-native sampling_profiles also rejected
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"coding":{"reasoning_effort":"xhigh","thinking_budget":"high"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"deepseek-v4-flash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"low"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"muse-glimmer":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"uncapped"}}}},"aliases":{}}"#,
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"ornith-1.5:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"high"}}}},"aliases":{}}"#,
        ];
        for raw in cases {
            let err = RegistryV1::parse(raw, "network/cache")
                .expect_err("must reject effort-native thinking_budget");
            let msg = err.to_string();
            assert!(
                msg.contains("thinking_budget") && msg.contains("effort-native"),
                "unexpected error for effort-native budget rejection: {msg}"
            );
        }
        // A cache entry that violates the invariant is also rejected via
        // validate(), causing read_cache to return None and load() to fall back.
        let stale_raw = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.8:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}}},"aliases":{}}"#;
        let stale: RegistryV1 = serde_json::from_str(stale_raw).unwrap();
        assert!(
            stale.validate("registry cache").is_err(),
            "stale cache with effort-native thinking_budget must fail validate()"
        );
    }

    #[test]
    fn invalid_reasoning_enums_rejected() {
        // Mirrors hipfire-config/registry_gen enum allowlists: recognizable
        // invalid values are rejected with a clear error; malformed types
        // already fail via surrounding validation and are not re-tested here.
        let invalid_effort = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"m":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"turbo"}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_effort, "test")
            .expect_err("invalid reasoning_effort must be rejected");
        assert!(err.to_string().contains("reasoning_effort"));

        let invalid_effort_profile = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"m":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"coding":{"reasoning_effort":"ultra"}}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_effort_profile, "test")
            .expect_err("invalid profile effort must be rejected");
        assert!(err.to_string().contains("reasoning_effort"));

        // thinking_budget invalid on legacy (non-effort-native) model
        let invalid_budget = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.5:9b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"yolo"}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_budget, "test")
            .expect_err("invalid thinking_budget must be rejected");
        assert!(err.to_string().contains("thinking_budget"));

        let invalid_budget_profile = r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{"qwen3.6:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"superhigh"}}}},"aliases":{}}"#;
        let err = RegistryV1::parse(invalid_budget_profile, "test")
            .expect_err("invalid profile budget must be rejected");
        assert!(err.to_string().contains("thinking_budget"));
    }

    #[test]
    fn legacy_qwen35_qwen36_thinking_budget_accepted() {
        // Legacy Qwen3.5/3.6 families are not effort-native; named budgets
        // remain valid and pass validation for both top-level and profiles.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:9b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"reasoning_effort":"high","thinking_budget":"high"}},
                "qwen3.5:27b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"general":{"thinking_budget":"med"},"coding":{"reasoning_effort":"max","thinking_budget":"max"}}},
                "qwen3.6:35b-a3b":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"uncapped"}},
                "qwen3.6:35b-a3b-mq4r":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","sampling_profiles":{"instruct":{"thinking_budget":"off","reasoning_effort":"none"}}}
            },
            "aliases":{}
        }"#;
        RegistryV1::parse(raw, "test").expect("legacy thinking_budget must remain valid");

        // Draft/dflash sidecars are excluded from effort-native gating, so they
        // also accept thinking_budget even when family would otherwise be native.
        let sidecars = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.8:27b-draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}},
                "qwen3.8:27b-dflash":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"low"}},
                "deepseek-v4-flash:draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"xhigh"}},
                "muse-glimmer:draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"med"}},
                "ornith-1.5:35b-a3b-draft":{"repo":"x","file":"x","size_gb":1,"min_vram_gb":1,"desc":"x","recommended_settings":{"thinking_budget":"high"}}
            },
            "aliases":{}
        }"#;
        RegistryV1::parse(sidecars, "test")
            .expect("draft/dflash sidecars must allow thinking_budget");
    }
}
