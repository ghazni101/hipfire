// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use hipfire_client::ServiceEndpoint;
use hipfire_config::{
    field, load_catalog, load_global, resolve, secret::BearerToken, ConfigFormat, ConfigPaths,
    ConfigValue, NamedLayer,
};

use super::HipfirePaths;

#[derive(Clone, Debug)]
pub struct ConfigState {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub values: BTreeMap<String, String>,
    /// Keys explicitly present in `~/.hipfire/config.toml` — i.e. user overrides,
    /// as opposed to inherited/hardcoded defaults. `values` always carries every
    /// default key merged with the disk overlay, so this set is the only way to
    /// tell an override apart from a default (drives reset gating + the 5c
    /// override marker).
    pub overrides: BTreeSet<String>,
    pub per_model_count: usize,
    pub loaded_from_disk: bool,
    pub warning: Option<String>,
}

impl ConfigState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut values = defaults();
        let mut overrides = BTreeSet::new();
        let mut loaded_from_disk = false;
        let mut warning = None;

        match load_global(&ConfigPaths::under(&paths.root)) {
            Ok(loaded) => {
                loaded_from_disk = loaded.format != ConfigFormat::Empty;
                for (canonical, value) in loaded.layer.values {
                    if let Some(schema) = field(&canonical) {
                        overrides.insert(schema.legacy_key.to_owned());
                        values.insert(schema.legacy_key.to_owned(), value_to_string(&value));
                    }
                }
                if !loaded.warnings.is_empty() {
                    warning = Some(loaded.warnings.join("; "));
                }
            }
            Err(error) => warning = Some(format!("config parse error: {error}")),
        }

        // `HIPFIRE_SERVE_API_KEY_FILE` outranks the config file, matching the
        // CLI's resolution order for this key.
        if let Some(schema) = field("serve.api_key_file") {
            if let Some(name) = schema.env_compat {
                if let Ok(raw) = std::env::var(name) {
                    match schema.parse_cli(&raw) {
                        Ok(value) => {
                            values.insert(schema.legacy_key.to_owned(), value_to_string(&value));
                        }
                        Err(error) => {
                            // Retain the invalid override so HTTP callers fail
                            // closed instead of using the old or absent key.
                            values.insert(schema.legacy_key.to_owned(), raw);
                            let message = format!("{name}: {error}");
                            warning = Some(match warning.take() {
                                Some(previous) => format!("{previous}; {message}"),
                                None => message,
                            });
                        }
                    }
                }
            }
        }

        let per_model_count = load_catalog(&ConfigPaths::under(&paths.root))
            .map(|loaded| loaded.catalog.models.len())
            .unwrap_or(0);

        let host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "0.0.0.0".into());
        let port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);
        let default_model = values
            .get("default_model")
            .cloned()
            .unwrap_or_else(|| "qwen3.5:9b".into());

        Self {
            host,
            port,
            default_model,
            values,
            overrides,
            per_model_count,
            loaded_from_disk,
            warning,
        }
    }

    /// True when `key` is an explicit user override in config.toml (vs an
    /// inherited/hardcoded default). Drives reset gating ("already default") and
    /// the override marker in the Settings view.
    pub fn is_override(&self, key: &str) -> bool {
        self.overrides.contains(key)
    }

    /// The hardcoded default map, for cross-checking the knob explainer defaults
    /// (5e) against the real defaults so the two can't drift.
    #[cfg(test)]
    pub(crate) fn defaults_for_test() -> BTreeMap<String, String> {
        defaults()
    }

    pub fn probe_host(&self) -> String {
        match self.host.as_str() {
            "0.0.0.0" | "" => "127.0.0.1".into(),
            "::" => "::1".into(),
            other => other.to_string(),
        }
    }

    /// Authenticated endpoint for the local serve process. Loads the bearer
    /// from the resolved `serve.api_key_file` (empty = keyless). Errors name
    /// the key file path and the violated rule, never the token.
    pub fn service_endpoint(&self) -> Result<ServiceEndpoint, String> {
        let key_file = self.values.get("api_key_file").map_or("", String::as_str);
        if !key_file.is_empty()
            && (key_file.trim() != key_file || !Path::new(key_file).is_absolute())
        {
            return Err("serve.api_key_file must be an absolute path".to_owned());
        }
        let bearer = if key_file.is_empty() {
            None
        } else {
            Some(
                BearerToken::load(Path::new(key_file))
                    .map_err(|error| format!("serve.api_key_file: {error}"))?,
            )
        };
        ServiceEndpoint::local(&self.host, self.port, bearer).map_err(|error| error.to_string())
    }

    /// Config keys behind each easy-mode row, in display order. `None` marks a
    /// row that is composite/non-editable inline (e.g. the `host:port` serve
    /// row, or the Model row which is set from the Models tab). The TUI editor
    /// uses this to resolve a selected easy row to a writable config key.
    pub fn easy_keys(&self) -> Vec<Option<&'static str>> {
        vec![
            None,            // Model (set via Models tab)
            Some("max_seq"), // Context
            Some("dflash_mode"),
            Some("prefill_compression"), // Prefill (pflash)
            Some("kv_cache"),
            Some("gpu_layer_budget"),  // Offload
            Some("moe_expert_budget"), // Offload (MoE experts)
            Some("offload_exec"),      // Offload exec
            Some("offload_passback_share"), // Pass-back GPU share
            Some("thinking"),
            Some("reasoning_effort"),
            Some("thinking_budget"),
            None, // Serve host:port (composite)
        ]
    }

    /// True when prefill compression is requested (auto/always) but no
    /// `prefill_drafter` is set — pflash then silently no-ops. Mirrors the bun
    /// CLI warning (`prefill_compression=… but prefill_drafter is unset`). Surfaced
    /// honestly in the easy row + as a toast when the user enables compression.
    ///
    /// Scope: this reflects the GLOBAL config (`~/.hipfire/config.toml`), which is
    /// what the Settings tab edits — like every other row here. A per-model
    /// override (`hipfire config <tag> set prefill_drafter …`, counted in
    /// `per_model_count`) is resolved by the daemon at serve time and is NOT
    /// layered in here; the daemon's own gate is the final word. Empty-string =
    /// disabled matches the shared schema exactly (`"" disables`).
    pub fn pflash_needs_drafter(&self) -> bool {
        let compression = self
            .values
            .get("prefill_compression")
            .map(String::as_str)
            .unwrap_or("off");
        let drafter_empty = self
            .values
            .get("prefill_drafter")
            .map(|s| s.is_empty())
            .unwrap_or(true);
        compression != "off" && drafter_empty
    }

    /// Per-easy-row override status, parallel to [`easy_rows`]/[`easy_keys`].
    /// Unlike `easy_keys` (which is `None` for the non-inline-editable Model /
    /// Serve composite rows), this resolves override status for EVERY row from
    /// its underlying config key(s), so the 5c override marker is consistent with
    /// the advanced view — a user-set `default_model` / `host` / `port` shows as
    /// changed in easy mode too, not just advanced.
    pub fn easy_override_state(&self) -> Vec<bool> {
        vec![
            self.is_override("default_model"),                    // Model
            self.is_override("max_seq"),                          // Context
            self.is_override("dflash_mode"),                      // Spec decode
            self.is_override("prefill_compression"),              // Prefill
            self.is_override("kv_cache"),                         // KV cache
            self.is_override("gpu_layer_budget"),                 // Offload
            self.is_override("moe_expert_budget"),                // Offload (MoE experts)
            self.is_override("offload_exec"),                     // Offload exec
            self.is_override("offload_passback_share"),           // Pass-back GPU share
            self.is_override("thinking"),                         // Thinking
            self.is_override("reasoning_effort"),                 // Reasoning effort
            self.is_override("thinking_budget"),                  // Reasoning budget
            self.is_override("host") || self.is_override("port"), // Serve
        ]
    }

    /// Help-lookup key for each easy row, parallel to [`easy_rows`]/[`easy_keys`].
    /// Unlike `easy_keys`, this maps the composite Model/Serve rows to a
    /// representative key (default_model / host) so the 5e explainer pane has
    /// something to show for every row.
    pub fn easy_help_keys(&self) -> Vec<&'static str> {
        vec![
            "default_model",       // Model
            "max_seq",             // Context
            "dflash_mode",         // Spec decode
            "prefill_compression", // Prefill
            "kv_cache",            // KV cache
            "gpu_layer_budget",    // Offload
            "moe_expert_budget",   // Offload (MoE experts)
            "offload_exec",        // Offload exec
            "offload_passback_share", // Pass-back GPU share
            "thinking",            // Thinking
            "reasoning_effort",    // Reasoning effort
            "thinking_budget",     // Reasoning budget
            "host",                // Serve
        ]
    }

    pub fn easy_rows(&self) -> Vec<(&'static str, String, &'static str)> {
        vec![
            (
                "Model",
                self.default_model.clone(),
                "Default model pre-warmed by serve and used by chat.",
            ),
            (
                "Context",
                self.values
                    .get("max_seq")
                    .cloned()
                    .unwrap_or_else(|| "32768".into()),
                "KV cache capacity allocated at load.",
            ),
            (
                "Spec decode",
                self.values
                    .get("dflash_mode")
                    .cloned()
                    .unwrap_or_else(|| "off".into()),
                "Spec decode: the model's attached drafter — its DFlash draft, else its MTP head.",
            ),
            (
                "Prefill",
                {
                    let c = self
                        .values
                        .get("prefill_compression")
                        .cloned()
                        .unwrap_or_else(|| "off".into());
                    // Honest: compression on without a drafter is a no-op.
                    if self.pflash_needs_drafter() {
                        format!("{c} (needs drafter)")
                    } else {
                        c
                    }
                },
                "Prefill KV compression (pflash). Needs a prefill_drafter (.hfq) to engage.",
            ),
            (
                "KV cache",
                self.values
                    .get("kv_cache")
                    .cloned()
                    .unwrap_or_else(|| "auto".into()),
                "Precision/memory tradeoff for attention cache.",
            ),
            (
                // Says GPU, not "offload": this value counts layers LEFT ON the
                // GPU, and an "Offload" label invites reading it as the inverse.
                "GPU layers",
                {
                    // Unset keeps every layer resident on every model. Name
                    // the state rather than leaving a blank value column; the
                    // editor seeds its buffer from `values`, not from this label.
                    // Short: the easy table's value column is 24 wide (ui.rs).
                    match self
                        .values
                        .get("gpu_layer_budget")
                        .map(String::as_str)
                        .unwrap_or("")
                    {
                        "" => "all on GPU".into(),
                        // `-1` is the schema's reserved auto spelling, and the
                        // rule also accepts a literal `auto`; both mean fit-to-card
                        // and neither is a layer count.
                        "-1" | "auto" => "auto (fits to card)".into(),
                        v => format!("{v} on GPU"),
                    }
                },
                "How many layers stay on the GPU; the rest spill to system RAM. Frees VRAM, but much slower.",
            ),
            (
                // The routed-expert tier: a MoE model's experts usually dwarf its
                // attention weights, so this is the row that makes an over-VRAM
                // MoE model loadable.
                "Expert layers",
                {
                    match self
                        .values
                        .get("moe_expert_budget")
                        .map(String::as_str)
                        .unwrap_or("")
                    {
                        // Unset never spills routed experts.
                        "" => "all on GPU".into(),
                        "-1" | "auto" => "auto (fits to card)".into(),
                        v => format!("{v} on GPU"),
                    }
                },
                "MoE only: how many layers keep their routed experts on the GPU. Only experts move; attention, router, shared expert and KV stay.",
            ),
            (
                // The other half of the offload decision: placement is the row
                // above, who multiplies is this one.
                "Offload exec",
                match self
                    .values
                    .get("offload_exec")
                    .map(String::as_str)
                    .unwrap_or("")
                {
                    "" | "pcie" => "over PCIe".into(),
                    "cpu" => "on CPU".into(),
                    "passback" => "passback (CPU + GPU)".into(),
                    // Only reachable if the schema's list grows: show it rather
                    // than mislabel it.
                    other => other.to_string(),
                },
                "Who multiplies a spilled layer's weights: the GPU over the link, or the CPU.",
            ),
            (
                // The pass-back's modifier: read only by the mode above, which is
                // why the row says so rather than pretending it is standalone.
                "Pass-back GPU share",
                match self
                    .values
                    .get("offload_passback_share")
                    .map(String::as_str)
                    .unwrap_or("")
                {
                    "" | "auto" => "auto (scheduled)".into(),
                    other => other.to_string(),
                },
                "Share of each spilled step's rows the GPU takes back, for the passback mode.",
            ),
            (
                "Thinking",
                self.values
                    .get("thinking")
                    .cloned()
                    .unwrap_or_else(|| "on".into()),
                "Whether reasoning models emit a hidden think block.",
            ),
            (
                "Reasoning effort",
                self.values
                    .get("reasoning_effort")
                    .cloned()
                    .unwrap_or_else(|| "auto".into()),
                "How hard the model is told to think, in its own vocabulary.",
            ),
            (
                "Reasoning budget",
                self.values
                    .get("thinking_budget")
                    .cloned()
                    .unwrap_or_else(|| "med".into()),
                "A CAP, not a dial — truncates. Prefer effort where the model has one.",
            ),
            (
                "Serve",
                format!("{}:{}", self.host, self.port),
                "OpenAI-compatible endpoint used by chat and API clients.",
            ),
        ]
    }
}

fn defaults() -> BTreeMap<String, String> {
    resolve(Vec::<NamedLayer>::new())
        .expect("shared built-in config schema validates")
        .legacy_values()
        .into_iter()
        .map(|(key, value)| (key, value_to_string(&value)))
        .collect()
}

fn value_to_string(value: &ConfigValue) -> String {
    match value {
        ConfigValue::String(value) => value.clone(),
        ConfigValue::Integer(value) => value.to_string(),
        ConfigValue::Float(value) => value.to_string(),
        ConfigValue::Bool(value) => value.to_string(),
        ConfigValue::Null => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four easy-mode lists are positional: row i in `easy_rows` is edited
    /// through `easy_keys[i]`, marked by `easy_override_state[i]`, and
    /// explained by `easy_help_keys[i]`. Nothing but this test stops them
    /// drifting — adding a row to three of the four silently misattributes
    /// every row after it.
    #[test]
    fn easy_mode_lists_stay_positionally_parallel() {
        let c = state_with(&[]);
        let n = c.easy_rows().len();
        assert_eq!(c.easy_keys().len(), n, "easy_keys drifted from easy_rows");
        assert_eq!(
            c.easy_override_state().len(),
            n,
            "easy_override_state drifted from easy_rows"
        );
        assert_eq!(
            c.easy_help_keys().len(),
            n,
            "easy_help_keys drifted from easy_rows"
        );
    }

    /// A ConfigState seeded from an explicit value map (defaults overlaid).
    fn state_with(pairs: &[(&str, &str)]) -> ConfigState {
        let mut values = defaults();
        let mut overrides = BTreeSet::new();
        for (k, v) in pairs {
            overrides.insert((*k).to_string());
            values.insert((*k).to_string(), (*v).to_string());
        }
        ConfigState {
            host: "0.0.0.0".into(),
            port: 11435,
            default_model: "qwen3.5:9b".into(),
            values,
            overrides,
            per_model_count: 0,
            loaded_from_disk: true,
            warning: None,
        }
    }

    #[test]
    fn service_endpoint_keeps_keyless_local_behavior_and_rejects_bad_key_paths() {
        let endpoint = state_with(&[]).service_endpoint().unwrap();
        assert!(endpoint.bearer().is_none());
        assert_eq!(
            hipfire_client::service_url(&endpoint, "/health"),
            "http://127.0.0.1:11435/health"
        );
        assert!(state_with(&[("api_key_file", "relative-key")])
            .service_endpoint()
            .unwrap_err()
            .contains("absolute path"));
        let missing = std::env::temp_dir().join(format!(
            "hipfire-tui-key-missing-{}", std::process::id()
        ));
        assert!(state_with(&[("api_key_file", missing.to_str().unwrap())])
            .service_endpoint()
            .is_err());
    }

    #[test]
    fn pflash_needs_drafter_logic() {
        // 5d honest state: compression on + no drafter -> warns; with a drafter,
        // or compression off, -> fine.
        assert!(
            !state_with(&[]).pflash_needs_drafter(),
            "off by default = fine"
        );
        assert!(
            state_with(&[("prefill_compression", "auto")]).pflash_needs_drafter(),
            "auto without drafter needs one"
        );
        assert!(
            state_with(&[("prefill_compression", "always")]).pflash_needs_drafter(),
            "always without drafter needs one"
        );
        assert!(
            !state_with(&[
                ("prefill_compression", "auto"),
                ("prefill_drafter", "/d.hfq")
            ])
            .pflash_needs_drafter(),
            "auto WITH a drafter is fine"
        );
    }

    #[test]
    fn easy_rows_surface_pflash_with_honest_hint() {
        // The Prefill easy row exists and flags the missing drafter inline.
        let st = state_with(&[("prefill_compression", "auto")]);
        let prefill = st
            .easy_rows()
            .into_iter()
            .find(|(label, _, _)| *label == "Prefill")
            .expect("Prefill easy row present");
        assert!(
            prefill.1.contains("needs drafter"),
            "no-op state shown: {}",
            prefill.1
        );

        // With a drafter set, the hint goes away.
        let st2 = state_with(&[
            ("prefill_compression", "auto"),
            ("prefill_drafter", "/d.hfq"),
        ]);
        let prefill2 = st2
            .easy_rows()
            .into_iter()
            .find(|(label, _, _)| *label == "Prefill")
            .unwrap();
        assert!(
            !prefill2.1.contains("needs drafter"),
            "hint cleared: {}",
            prefill2.1
        );
    }

    #[test]
    fn easy_view_vectors_stay_parallel() {
        // All four easy vectors must stay equal length (5d adds a row, 5e adds
        // help keys) or the marker/help attaches to the wrong row.
        let st = state_with(&[]);
        let n = st.easy_rows().len();
        assert_eq!(st.easy_keys().len(), n);
        assert_eq!(st.easy_override_state().len(), n);
        assert_eq!(st.easy_help_keys().len(), n);
    }

    /// A row marked inline-editable (`Some(key)`) that has no `EDITABLE_FIELDS`
    /// spec cannot actually be edited: `App::handle_settings_key` looks the spec up
    /// and, on `None`, refuses — Enter never opens a buffer and no value can be
    /// typed. `gpu_layer_budget` and `reasoning_effort` both shipped that way, in
    /// `easy_keys()` with no spec, and nothing failed.
    #[test]
    fn every_inline_editable_easy_row_has_a_field_spec() {
        for key in state_with(&[]).easy_keys().into_iter().flatten() {
            assert!(
                crate::hipfire::writer::field_spec(key).is_some(),
                "easy row {key} is marked inline-editable but has no EDITABLE_FIELDS \
                 spec, so the editor refuses to type a value into it"
            );
        }
    }

    /// The GPU-layers row must render every state it can hold, and render it
    /// honestly: `-1` is the schema's *auto* spelling, not a count of layers, and
    /// a cleared key must read as unset rather than as the string the JSON layer
    /// happens to produce for null.
    #[test]
    fn offload_layer_row_renders_unset_count_and_auto() {
        let row = |pairs: &[(&str, &str)]| {
            state_with(pairs)
                .easy_rows()
                .into_iter()
                .find(|(l, _, _)| *l == "GPU layers")
                .map(|(_, value, _)| value)
                .expect("GPU layers row present")
        };

        // Unset never spills, including on a MoE model.
        assert_eq!(row(&[]), "all on GPU");
        assert_eq!(row(&[("gpu_layer_budget", "32")]), "32 on GPU");
        assert_eq!(
            row(&[("gpu_layer_budget", "-1")]),
            "auto (fits to card)",
            "-1 means auto, not -1 layers on the GPU"
        );
        assert_eq!(
            row(&[("gpu_layer_budget", "auto")]),
            "auto (fits to card)",
            "the literal auto spelling (accepted by the value rule) is not a count"
        );
        // A cleared key is unset, not the literal string the JSON layer produces
        // for null — the bug this replaced rendered "null on GPU".
        assert_eq!(row(&[("gpu_layer_budget", "")]), "all on GPU");
    }

    /// The routed-expert row: same cell grammar as its `GPU layers` sibling, and
    /// distinct labels so the two offload tiers cannot be confused in the UI.
    #[test]
    fn offload_expert_row_renders_unset_count_and_auto() {
        let row = |pairs: &[(&str, &str)]| {
            state_with(pairs)
                .easy_rows()
                .into_iter()
                .find(|(l, _, _)| *l == "Expert layers")
                .map(|(_, value, _)| value)
                .expect("Expert layers row present")
        };
        assert_eq!(row(&[]), "all on GPU");
        assert_eq!(row(&[("moe_expert_budget", "26")]), "26 on GPU");
        assert_eq!(row(&[("moe_expert_budget", "-1")]), "auto (fits to card)");
        assert_eq!(row(&[("moe_expert_budget", "auto")]), "auto (fits to card)");
        assert_eq!(row(&[("moe_expert_budget", "")]), "all on GPU");
    }

    /// The exec row's two arms, in the same cell grammar as its sibling.
    #[test]
    fn offload_exec_row_renders_both_arms() {
        let row = |pairs: &[(&str, &str)]| {
            state_with(pairs)
                .easy_rows()
                .into_iter()
                .find(|(l, _, _)| *l == "Offload exec")
                .map(|(_, value, _)| value)
                .expect("Offload exec row present")
        };

        assert_eq!(row(&[]), "over PCIe");
        assert_eq!(row(&[("offload_exec", "pcie")]), "over PCIe");
        assert_eq!(row(&[("offload_exec", "cpu")]), "on CPU");
        assert_eq!(
            row(&[("offload_exec", "passback")]),
            "passback (CPU + GPU)"
        );
    }

    /// The pass-back share row: unset and `auto` both mean "the engine schedules
    /// it", and a pinned number renders as itself.
    #[test]
    fn offload_passback_share_row_renders_scheduled_and_pinned() {
        let row = |pairs: &[(&str, &str)]| {
            state_with(pairs)
                .easy_rows()
                .into_iter()
                .find(|(l, _, _)| *l == "Pass-back GPU share")
                .map(|(_, value, _)| value)
                .expect("Pass-back GPU share row present")
        };

        assert_eq!(row(&[]), "auto (scheduled)");
        assert_eq!(row(&[("offload_passback_share", "auto")]), "auto (scheduled)");
        assert_eq!(row(&[("offload_passback_share", "0")]), "0");
        assert_eq!(row(&[("offload_passback_share", "0.375")]), "0.375");
    }

    #[test]
    fn easy_help_keys_have_explainers() {
        // 5e: every easy row's help key must resolve to a curated explainer.
        for key in state_with(&[]).easy_help_keys() {
            assert!(
                crate::hipfire::knobs::knob_info(key).is_some(),
                "easy help key {key} has no explainer"
            );
        }
    }
}
