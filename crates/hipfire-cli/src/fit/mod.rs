// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Read-only load admission: the loader decides, this command explains its
//! decision using machine counters and published receipts. Never initialize HIP
//! or start a daemon here; GPU-dependent placement remains explicitly bounded.

mod host;
mod qwen4;
mod remote;
pub(crate) mod types;

use anyhow::Result;
use clap::Args;
use hipfire_loader::admission::{
    admit_source_with_options, qwen4_kv_adaptive_requested, qwen_default_q8_enabled,
    EffectiveTopology, KvBackendHints, KvBackendRequest, Qwen35CardProjection, SourceAdmission,
    SourceAdmissionOptions, QWEN35_MIN_AUTO_CONTEXT,
};
use hipfire_registry::{receipts, ModelEntry, RegistryV1};
use hipfire_runtime::loader_api::{CaskConfig, ModelSource, SpecLoadCfg};
use serde::Serialize;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use types::{gib, Finding, FitDevice, FitHost, PlacementPlan, Verdict};

#[derive(Args, Debug)]
pub(crate) struct FitArgs {
    /// Installed model, alias, file, path, or registry tag.
    pub model: String,
    /// Context to admit (otherwise use the same configured/automatic bound as run).
    #[arg(long, value_parser = parse_context)]
    pub ctx: Option<usize>,
    /// Do not fetch headers for models that are not installed.
    #[arg(long)]
    pub no_fetch: bool,
    /// Print one hipfire-fit-1 JSON object.
    #[arg(short = 'j', long)]
    pub json: bool,
    /// Also print the source-cited derivation lines (always in --json).
    #[arg(short = 'v', long)]
    pub verbose: bool,
}

fn parse_context(raw: &str) -> std::result::Result<usize, String> {
    raw.parse::<usize>()
        .ok()
        .filter(|n| (512..=1_048_576).contains(n))
        .ok_or_else(|| "--ctx must be an integer between 512 and 1048576".into())
}

#[derive(Serialize)]
struct ModelInfo {
    tag: String,
    file: Option<String>,
    installed: bool,
    replica: bool,
    size_bytes: Option<u64>,
    arch_id: Option<u32>,
    quant: Option<String>,
}

#[derive(Serialize)]
struct Route {
    name: String,
    value: String,
    why: String,
}

#[derive(Serialize)]
struct Plan {
    device_index: usize,
    topology: String,
    kv_backend: String,
    kv_backend_reason: Option<String>,
    kv_mode: String,
    max_seq: usize,
    bound: String,
    model_ctx: Option<usize>,
    projection: Option<Value>,
    placement: PlacementPlan,
    decode_route: String,
    routes: Vec<Route>,
}

#[derive(Serialize)]
struct TierRow {
    tag: String,
    size_bytes: Option<u64>,
    verdict: Verdict,
    summary: String,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    hipfire: Value,
    input: String,
    model: ModelInfo,
    machine: FitHost,
    plan: Option<Plan>,
    findings: Vec<Finding>,
    verdict: Verdict,
    tiers: Vec<TierRow>,
    receipts: Vec<Value>,
    notes: Vec<String>,
    /// Source-cited derivation lines (`--verbose` in text).
    details: Vec<String>,
}

struct Analysis {
    model: ModelInfo,
    plan: Option<Plan>,
    findings: Vec<Finding>,
    notes: Vec<String>,
    details: Vec<String>,
}

/// Give admission the visible-device count without opening the HIP runtime.
///
/// The loader's `visible_device_count` (crates/hipfire-loader/src/admission.rs
/// `visible_device_count`) reads `hardware.devices`, then
/// `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`, and only then dlopens HIP for
/// `hipGetDeviceCount` — which initializes the ROCm runtime and opens
/// `/dev/kfd`. When none of the first three is set, every KFD GPU is visible to
/// the daemon, so this process (which never loads HIP itself) states exactly
/// that set in ROCr order: the count admission reads is the count the daemon's
/// HIP would report, and fit stays GPU-free.
fn supply_visible_device_count(process: &hipfire_config::ProcessConfig, machine: &FitHost) {
    let configured = process
        .legacy_value("HIPFIRE_DEVICES")
        .is_some_and(|v| !v.trim().is_empty());
    let inherited = [
        hipfire_config::devices::HIP_VISIBLE_DEVICES,
        hipfire_config::devices::ROCR_VISIBLE_DEVICES,
    ]
    .iter()
    .any(|name| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()));
    if configured || inherited || machine.devices.is_empty() {
        return;
    }
    let all = (0..machine.devices.len())
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(",");
    std::env::set_var(hipfire_config::devices::ROCR_VISIBLE_DEVICES, all);
}

fn fold_findings(findings: &[Finding]) -> Verdict {
    findings.iter().fold(Verdict::Fits, |v, f| v.and(f.verdict))
}

fn exit_code(verdict: Verdict) -> i32 {
    match verdict {
        Verdict::Fits => 0,
        Verdict::DoesNotFit => 1,
        Verdict::Undetermined => 2,
    }
}

pub(crate) fn fit_command(paths: &crate::Paths, args: FitArgs) -> Result<()> {
    let loaded = crate::load_registry(&paths.registry);
    let registry = &loaded.registry;
    let card = crate::model_card(paths, registry, &args.model);
    let resolved = crate::resolved_for_model(paths, &args.model, card.tag(), card.entry())?;
    let process = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    // Same immutable startup snapshot as run's spawn_configured, installed
    // before host/preflight can freeze process_value's compatibility table.
    // Family tiers use this one process policy; differing overlays are labelled.
    hipfire_config::install_process_config(process).map_err(|_| {
        anyhow::anyhow!("fit process configuration was initialized before model resolution")
    })?;
    let process = hipfire_config::active_process_config().expect("just installed");
    // Match daemon/main.rs:776's physical -> logical normalization, including
    // ACTIVE_DEVICES used by admission's mesh. Fit never reserves a card: the
    // claim callback only asks the pure selector to retain its ordered result.
    let visibility_error = hipfire_config::devices::apply_device_visibility(process, &mut |_| {
        Ok(hipfire_config::devices::Claim::Claimed)
    })
    .err();
    let (machine, host_notes) = host::detect_with_notes(paths);
    let mut findings = preflight_findings(paths);
    if let Some(error) = visibility_error {
        findings.push(Finding::fails("admission", error.to_string(),
            "set `hardware.devices` to a listed GPU, or unset incompatible ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES overrides"));
    }
    for note in host_notes
        .iter()
        .filter(|note| note.starts_with("load refused:"))
    {
        findings.push(Finding::fails("admission", note,
            note.split_once("fix:").map(|(_, fix)| fix.trim()).unwrap_or("correct hardware.devices or the GPU visibility override, then run `hipfire doctor`")));
    }
    if machine.devices.is_empty() {
        findings.push(Finding::undetermined(
            "vram",
            "no AMD GPU detected in the KFD topology",
            Some("run `hipfire doctor`".into()),
        ));
    } else if machine.devices[0].free.is_none() {
        findings.push(Finding::undetermined(
            "vram",
            "free device memory counter is unavailable; capacity cannot be determined",
            Some("run `hipfire doctor` and check amdgpu sysfs memory counters".into()),
        ));
    }
    supply_visible_device_count(process, &machine);
    let analysis = analyze_model(
        paths,
        registry,
        &args.model,
        args.ctx,
        args.no_fetch,
        &machine,
    )?;
    let mut tiers = Vec::new();
    let family = crate::pull_select::family_tiers(registry, &analysis.model.tag);
    // Every other tier runs the same admission concurrently: each one walks its
    // own artifact's source identity (seconds per tier on a large MoE), and the
    // analyses share only read-only state (the installed process config, the
    // registry, the machine facts) and write disjoint replica cache entries.
    let others: Vec<Option<Result<Analysis>>> = if family.len() > 1 {
        std::thread::scope(|scope| {
            let handles: Vec<_> = family
                .iter()
                .map(|tier| {
                    let tag = tier.tag;
                    let machine = &machine;
                    (tag != analysis.model.tag).then(|| {
                        scope.spawn(move || {
                            analyze_model(paths, registry, tag, args.ctx, args.no_fetch, machine)
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle.map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("tier analysis panicked")))
                    })
                })
                .collect()
        })
    } else {
        Vec::new()
    };
    for (tier, other) in family.iter().zip(others) {
        let owned;
        let evaluated = match other {
            None => &analysis,
            Some(Ok(analysis)) => {
                owned = analysis;
                &owned
            }
            // One tier that cannot be analysed (a failed header fetch, an
            // unreadable local file) is a row of its own, not a failed fit.
            Some(Err(error)) => {
                tiers.push(TierRow {
                    tag: tier.tag.into(),
                    size_bytes: tier.entry.size_bytes,
                    verdict: Verdict::Undetermined,
                    summary: format!("not evaluated: {error:#}"),
                });
                continue;
            }
        };
        // The tier's own fit: machine-level blockers (the runtime preflight,
        // device visibility) are reported once above and apply to every tier.
        let verdict = fold_findings(&evaluated.findings);
        // A tier that does not fit is summarized by the first term that fails;
        // otherwise by its capacity and placement.
        let failing = evaluated
            .findings
            .iter()
            .find(|f| f.verdict == Verdict::DoesNotFit);
        let mut summary = match failing {
            Some(finding) => format!("{}: {}", finding.id, finding.detail),
            None => evaluated
                .plan
                .as_ref()
                .map(|p| {
                    p.projection
                        .as_ref()
                        .and_then(|v| v["card_cap"].as_u64())
                        .map(|cap| format!("up to {cap} tokens; {}", p.placement.summary))
                        .unwrap_or_else(|| p.placement.summary.clone())
                })
                .unwrap_or_else(|| {
                    evaluated
                        .findings
                        .iter()
                        .map(|f| f.detail.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                }),
        };
        let differs = crate::resolved_for_model(paths, tier.tag, Some(tier.tag), Some(tier.entry))
            .ok()
            .and_then(|r| hipfire_config::ProcessConfig::from_resolved(&r).ok())
            .is_some_and(|tier_process| &tier_process != process);
        if differs {
            summary.push_str(&format!(
                "; per-model settings differ: run `hipfire fit {}` for its exact answer",
                tier.tag
            ));
        }
        tiers.push(TierRow {
            tag: tier.tag.into(),
            size_bytes: evaluated.model.size_bytes,
            verdict,
            summary,
        });
    }
    let published = matching_receipts(&analysis.model.tag, &machine, analysis.plan.as_ref());
    findings.extend(analysis.findings);
    let verdict = fold_findings(&findings);
    let mut notes = analysis.notes;
    notes.extend(host_notes);
    notes.push("Free memory is sampled without opening the GPU; HIP reports slightly less once a process opens it. Fit reserves no cards: device availability, locks and memory counters can change before the real load.".into());
    notes.push("Admission assumes HIP VMM is available on native Linux ROCm; the loader still applies its platform refusal rules.".into());
    let report = Report {
        schema: "hipfire-fit-1",
        hipfire: json!({"version": env!("HIPFIRE_BUILD_VERSION"), "commit": crate::BUILD_COMMIT}),
        input: safe_name(&args.model),
        model: analysis.model,
        machine,
        plan: analysis.plan,
        findings,
        verdict,
        tiers,
        receipts: published,
        notes,
        details: analysis.details,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&json_report(&report)?)?);
    } else {
        print!("{}", render_text(&report, args.verbose));
    }
    std::io::stdout().flush()?;
    std::process::exit(exit_code(verdict));
}

#[cfg(target_os = "linux")]
fn preflight_findings(paths: &crate::Paths) -> Vec<Finding> {
    let preflight = crate::doctor::runtime::preflight(paths);
    if preflight.blocked() {
        // CheckStatus lives in doctor's private report module. Use its stable
        // serialized status rather than duplicating that module's status type.
        let failed = preflight
            .checks
            .iter()
            .filter(|check| {
                serde_json::to_value(check.status).ok().as_ref() == Some(&json!("fail"))
            })
            .map(|check| check.id)
            .collect::<Vec<_>>()
            .join(", ");
        vec![Finding::fails(
            "preflight",
            format!("runtime preflight blocked: {failed}"),
            "run `hipfire doctor` and apply its fix lines",
        )]
    } else {
        vec![Finding::fits("preflight", "runtime preflight ok")]
    }
}

#[cfg(not(target_os = "linux"))]
fn preflight_findings(_: &crate::Paths) -> Vec<Finding> {
    Vec::new()
}

fn analyze_model(
    paths: &crate::Paths,
    registry: &RegistryV1,
    input: &str,
    ctx: Option<usize>,
    no_fetch: bool,
    machine: &FitHost,
) -> Result<Analysis> {
    let card = crate::model_card(paths, registry, input);
    let entry = card.entry();
    let tag = card.tag().unwrap_or(input);
    let local = crate::find_model_path(paths, registry, input);
    if local.is_none() && entry.is_none() {
        return Err(crate::unknown_model_error(registry, input));
    }
    let mut result = Analysis {
        model: ModelInfo {
            tag: safe_name(tag),
            file: local
                .as_deref()
                .and_then(Path::file_name)
                .map(|s| s.to_string_lossy().into_owned())
                .or_else(|| entry.map(|e| safe_name(&e.file))),
            installed: local.is_some(),
            replica: false,
            size_bytes: entry.and_then(|e| e.size_bytes),
            arch_id: entry.and_then(|e| e.arch_id),
            quant: entry.and_then(|e| e.quant.clone()),
        },
        plan: None,
        findings: Vec::new(),
        notes: Vec::new(),
        details: Vec::new(),
    };
    let path = match local {
        Some(path) => path,
        None => {
            let entry = entry.expect("registry entry checked above");
            if no_fetch {
                result.findings.push(missing_header(
                    tag,
                    "not installed; fit needs the model header",
                ));
                return Ok(result);
            }
            match remote::header_replica(paths, tag, entry) {
                Ok(replica) => {
                    result.model.replica = true;
                    result.notes.push(format!(
                        "{} header replica: {} bytes fetched; weights have not been downloaded",
                        if replica.cached { "cached" } else { "sparse" },
                        replica.fetched_bytes
                    ));
                    replica.path
                }
                Err(error) => {
                    result.findings.push(missing_header(
                        tag,
                        &format!("not installed; fit needs the model header: {error}"),
                    ));
                    return Ok(result);
                }
            }
        }
    };
    result.model.size_bytes = std::fs::metadata(&path)
        .ok()
        .map(|m| m.len())
        .or(result.model.size_bytes);
    let params = match model_params(paths, input, Some(tag), entry, &path, ctx) {
        Ok(params) => params,
        Err(error) => {
            let detail = error.to_string();
            result
                .findings
                .push(Finding::fails("admission", &detail, admission_fix(&detail)));
            return Ok(result);
        }
    };
    let Some(device) = machine.devices.first() else {
        return Ok(result);
    };
    let inputs = admission_inputs(&params);
    let admitted = match inputs.admit(&path, device) {
        Ok(admitted) => admitted,
        Err(detail) => {
            result
                .findings
                .push(Finding::fails("admission", &detail, admission_fix(&detail)));
            return Ok(result);
        }
    };
    result.model.arch_id = Some(admitted.arch_id);
    if entry.is_none() {
        // Same filename-label convention as pull_select::file_label. The HFQ
        // container extension is not a claim about each tensor's mixed dtype.
        result.model.quant = path.extension().map(|v| v.to_string_lossy().into_owned());
    }
    let model_ctx = admitted
        .sequence
        .as_ref()
        .map(|s| s.model_ctx)
        .or_else(|| source_context(&admitted.source));
    let mut plan = Plan {
        device_index: device.index,
        topology: topology_key(admitted.topology),
        kv_backend: admitted.kv_backend.as_str().into(),
        kv_backend_reason: admitted.kv_backend_reason.clone(),
        kv_mode: match admitted.sequence.as_ref() {
            Some(sequence) => sequence.kv_mode.to_owned(),
            // Qwen4 resolves its KV mode the way admission does
            // (admission.rs `resolve_qwen4`), so the plan names the
            // effective mode instead of the requested `auto`.
            None if admitted.arch_id == 16 => hipfire_runtime::kv_mode::resolve_qwen4(
                params["kv_mode"].as_str().unwrap_or(""),
                &device.arch,
            )
            .map(|kv| hipfire_runtime::kv_mode::qwen_k_display_name(kv.mode).to_owned())
            .unwrap_or_else(|_| params["kv_mode"].as_str().unwrap_or("auto").into()),
            None => params["kv_mode"].as_str().unwrap_or("auto").into(),
        },
        max_seq: admitted.max_seq,
        bound: admitted
            .sequence
            .as_ref()
            .map(|s| s.bound)
            .unwrap_or(if inputs.max_seq == 0 {
                "automatic"
            } else {
                "user"
            })
            .into(),
        model_ctx,
        projection: None,
        placement: PlacementPlan {
            summary: "fully resident".into(),
            expert_layers_resident: None,
            host_mapped_bytes: Some(0),
            decided_at_load: false,
        },
        decode_route: decode_route(&params),
        routes: link_routes(device),
    };
    match admitted.topology {
        EffectiveTopology::Pipeline(_) | EffectiveTopology::Expert(_) => {
            plan.placement.summary = "per-rank placement decided at load".into();
            plan.placement.decided_at_load = true;
            result.findings.push(Finding::undetermined(
                "context",
                "multi-GPU capacity is sized per rank at load",
                None,
            ));
            for rank in &machine.devices {
                result.notes.push(format!(
                    "GPU{} {}: {} free",
                    rank.index,
                    rank.arch,
                    rank.free.map(gib).unwrap_or_else(|| "unknown".into())
                ));
            }
        }
        EffectiveTopology::Single if matches!(admitted.arch_id, 5 | 6) => {
            match admitted.sequence.as_ref().and_then(|s| s.projection) {
                // The projection admission itself refused on: the load's own
                // context check, so its verdict is the fit verdict.
                Some(projection) => {
                    result.findings.push(context_finding(
                        projection,
                        admitted.max_seq,
                        inputs.max_seq == 0,
                    ));
                    plan.model_ctx = Some(projection.model_ctx);
                    plan.projection = Some(projection_json(projection));
                }
                // No stored projection (MoE on a discrete GPU, or legacy KV):
                // the daemon does not check this context before load, so fit
                // cannot decide it. The VMM projection is advisory only and
                // never replaces the plan's resolved KV mode.
                None => {
                    let projection = hipfire_loader::admission::qwen35_card_projection(
                        &admitted.source,
                        &device.arch,
                        inputs.hints(device),
                    );
                    let reason = admitted.sequence_reason.unwrap_or("legacy KV backend");
                    result.findings.push(unchecked_context_finding(
                        projection,
                        admitted.max_seq,
                        reason,
                    ));
                }
            }
        }
        EffectiveTopology::Single if admitted.arch_id == 16 => {
            if let ModelSource::Hfq(hfq) = &admitted.source {
                let analysis = qwen4::analyze(&qwen4::Qwen4Request {
                    hfq,
                    device,
                    host: machine,
                    max_seq: admitted.max_seq,
                    params: &params,
                });
                plan.placement = analysis.placement;
                result.findings.extend(analysis.findings);
                result.notes.extend(analysis.notes);
                result.details.extend(analysis.details);
            } else {
                result.findings.push(Finding::undetermined(
                    "context",
                    "Qwen4 placement requires an HFQ weight manifest",
                    Some("choose a published HFQ tier with `hipfire pull`".into()),
                ));
            }
        }
        EffectiveTopology::Single => {
            match (result.model.size_bytes, device.free) {
                (Some(bytes), Some(free)) if bytes <= free => result.findings.push(Finding::fits(
                    "weights",
                    format!("weights {} of {} free", gib(bytes), gib(free)),
                )),
                (Some(bytes), Some(free)) => result.findings.push(Finding::fails(
                    "weights",
                    format!("weights {} exceed {} free", gib(bytes), gib(free)),
                    "pick a smaller tier or free GPU memory",
                )),
                _ => result.findings.push(Finding::undetermined(
                    "weights",
                    "weight size or free GPU memory is unavailable",
                    Some("run `hipfire doctor`".into()),
                )),
            }
            result.findings.push(Finding::undetermined(
                "context",
                format!(
                    "this architecture sizes its KV at load (no pre-load projection); {} tokens",
                    admitted.max_seq
                ),
                None,
            ));
        }
    }
    result.plan = Some(plan);
    Ok(result)
}

fn missing_header(tag: &str, detail: &str) -> Finding {
    Finding::undetermined(
        "model",
        detail,
        Some(format!("`hipfire pull {tag}` or rerun without --no-fetch")),
    )
}

fn model_params(
    paths: &crate::Paths,
    input: &str,
    tag: Option<&str>,
    entry: Option<&ModelEntry>,
    path: &Path,
    ctx: Option<usize>,
) -> Result<Value> {
    let resolved = crate::resolved_for_model(paths, input, tag, entry)?;
    let mut params = crate::load_params(
        &resolved,
        entry,
        &paths.models,
        path,
        0,
        None,
        None,
        tag,
        false,
        None,
    )?;
    // main.rs run_command forwards --max-seq after load_params, without any
    // max_tokens inflation; --ctx has precisely that same precedence.
    if let Some(ctx) = ctx {
        params["max_seq"] = json!(ctx);
    }
    // daemon/main.rs:1675-1693 and vision_ladder.rs:49-59: sample the
    // developer override ladders outside the pure params mapper, including
    // explicit empty values (opt out). The mapper applies each mode gate.
    for (variable, key) in [
        ("HIPFIRE_DFLASH_DRAFT", "draft"),
        ("HIPFIRE_VISION_SIDECAR", "vision"),
    ] {
        if let Ok(value) = hipfire_config::developer_var(variable) {
            params[key] = json!(value);
        }
    }
    // daemon/main.rs:1813-1820 and :1845-1862 use these process fallbacks
    // for absent/non-numeric params (including an "auto" mtp_k).
    let runtime = hipfire_runtime::config::get();
    if params["mtp_mode"].as_str().is_none() {
        params["mtp_mode"] = json!(runtime.mtp_mode);
    }
    if params["mtp_k"].as_u64().is_none() {
        if let Some(k) = runtime.mtp_k {
            params["mtp_k"] = json!(k);
        }
    }
    if params["ngram_draft"].as_bool().is_none() {
        params["ngram_draft"] = json!(runtime.ngram_draft);
    }
    if params["ngram_k"].as_u64().is_none() {
        params["ngram_k"] = json!(runtime.ngram_k);
    }
    if params["ngram_min_count"].as_u64().is_none() {
        params["ngram_min_count"] = json!(runtime.ngram_min_count);
    }
    // daemon/main.rs:2132-2146: retain the explicit experimental PP gates.
    params["_pp_dflash"] = json!(
        hipfire_config::developer_var("HIPFIRE_PP_DFLASH")
            .ok()
            .as_deref()
            == Some("1")
    );
    params["_pp_pflash"] = json!(
        hipfire_config::developer_var("HIPFIRE_PP_PFLASH")
            .ok()
            .as_deref()
            == Some("1")
    );
    // daemon/main.rs:1270-1289 resolves the process topology before parsing
    // admission inputs. Keep environmental resolution outside the pure mapper.
    let mut message = json!({"params": params});
    hipfire_config::topology::resolve_load_pp(&mut message).map_err(anyhow::Error::msg)?;
    Ok(message["params"].take())
}

fn source_context(source: &ModelSource) -> Option<usize> {
    let ModelSource::Hfq(hfq) = source else {
        return None;
    };
    let metadata: Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    metadata
        .pointer("/config/text_config/max_position_embeddings")
        .or_else(|| metadata.pointer("/config/max_position_embeddings"))
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

fn topology_key(topology: EffectiveTopology) -> String {
    match topology {
        EffectiveTopology::Single => "single".into(),
        EffectiveTopology::Expert(n) => format!("ep{n}"),
        EffectiveTopology::Pipeline(n) => format!("pp{n}"),
    }
}

fn decode_route(params: &Value) -> String {
    if params["draft"].as_str().is_some_and(|v| !v.is_empty())
        && params["dflash_mode"].as_str() != Some("off")
    {
        "dflash".into()
    } else {
        match params["mtp_mode"].as_str().unwrap_or("auto") {
            "on" => "mtp".into(),
            "auto" => "mtp / ar (auto; weights and route decided at load)".into(),
            _ => "ar".into(),
        }
    }
}

fn link_routes(device: &FitDevice) -> Vec<Route> {
    let Some(link) = &device.link else {
        return Vec::new();
    };
    let Some(gbps) = link
        .gbps
        .filter(|_| link.calibrated && !device.unified_memory)
    else {
        return Vec::new();
    };
    hipfire_arch_qwen4::host_link::link_route_choices(
        &device.arch,
        hipfire_arch_qwen4::host_link::HostLink::Discrete { gbps },
    )
    .into_iter()
    .map(|r| Route {
        name: r.name.into(),
        value: r.value,
        why: r.why,
    })
    .collect()
}

fn projection_json(p: Qwen35CardProjection) -> Value {
    json!({"free": p.free, "weights": p.weights, "scratch": p.scratch,
        "page_rounding": p.page_rounding, "bytes_per_token": p.bytes_per_token,
        "card_cap": p.card_cap, "model_ctx": p.model_ctx, "kv_mode": p.kv_mode})
}

/// The context finding for a Qwen3.5-family load whose context the daemon
/// does not check before load (no stored admission projection: MoE on a
/// discrete GPU, or the legacy KV backend). Always `Undetermined`: the VMM
/// projection is reported as advisory, never as the load's verdict.
fn unchecked_context_finding(
    projection: Result<Qwen35CardProjection, String>,
    max_seq: usize,
    reason: &str,
) -> Finding {
    match projection {
        Ok(p) => Finding::undetermined(
            "context",
            format!(
                "the daemon does not check this context before load ({reason}); advisory only: \
                 the VMM admission projection would leave room for {} tokens against {max_seq} \
                 (free {} - weights {} - prefill scratch {} - page rounding {})",
                p.card_cap,
                gib(p.free as u64),
                gib(p.weights as u64),
                gib(p.scratch as u64),
                gib(p.page_rounding as u64),
            ),
            (max_seq > p.card_cap).then(|| {
                format!(
                    "lower --ctx to at most {} (`hipfire config set memory.max_seq {}`) or free GPU memory",
                    p.card_cap, p.card_cap
                )
            }),
        ),
        Err(error) => Finding::undetermined(
            "context",
            format!("the daemon does not check this context before load ({reason}); no projection: {error}"),
            None,
        ),
    }
}

fn context_finding(p: Qwen35CardProjection, requested: usize, automatic: bool) -> Finding {
    let charges = format!(
        "free {} - weights {} - prefill scratch {} - page rounding {}",
        gib(p.free as u64),
        gib(p.weights as u64),
        gib(p.scratch as u64),
        gib(p.page_rounding as u64)
    );
    let fix = format!("lower --ctx to at most {} (`hipfire config set memory.max_seq {}`), free GPU memory, or pick a smaller tier", p.card_cap, p.card_cap);
    if automatic && p.card_cap < QWEN35_MIN_AUTO_CONTEXT {
        return Finding::fails(
            "context",
            format!(
                "projected card capacity {} is below the minimum automatic context {} ({charges})",
                p.card_cap, QWEN35_MIN_AUTO_CONTEXT
            ),
            "free GPU memory or pick a smaller tier to provide the minimum automatic context",
        );
    }
    if automatic {
        Finding::fits("context", format!("automatic context: min(trained window {}, card capacity measured after weights load); projected {} tokens ({charges})", p.model_ctx, p.card_cap))
    } else {
        let detail = format!("a {requested}-token context needs {requested} x {} bytes of KV; the projection leaves room for {} tokens ({charges})", p.bytes_per_token, p.card_cap);
        if requested <= p.card_cap {
            Finding::fits("context", detail)
        } else {
            Finding::fails("context", detail, fix)
        }
    }
}

fn admission_fix(message: &str) -> String {
    if message.contains("max_seq") || message.contains("context") {
        "lower --ctx (or `hipfire config set memory.max_seq 512`), free GPU memory, or pick a smaller tier; apply any bound named above".into()
    } else if message.contains("reload without") || message.contains("set ") {
        format!("apply the setting or reload change named above: {message}")
    } else if message.contains("kv") || message.contains("KV") {
        "select a KV mode/backend supported by this model and GPU using `hipfire config set memory.kv_cache q8` and `hipfire config set memory.kv_backend vmm`; apply the admission message if it names a different required mode".into()
    } else if message.contains("topology") || message.contains("pp=") || message.contains("tp=") {
        "use a supported topology (`hipfire config set topology.pp 1` and `hipfire config set topology.ep 1`) or select the model/topology named above".into()
    } else {
        "see the message; change the named load setting or choose a supported smaller tier with `hipfire ls`, then rerun `hipfire fit`".into()
    }
}

fn receipt_matches(
    r: &receipts::Receipt,
    tag: &str,
    arch: &str,
    topology: &str,
    placement: &str,
) -> bool {
    r.tag == tag
        && r.arch == arch
        && r.topology == topology
        // Never across placement: a receipt whose source does not state its
        // placement ("unstated") matches no plan. Historical rows (an older
        // engine, or an artifact the registry no longer serves) are kept in
        // the table but never applied to this machine's expectation.
        && r.placement == placement
        && !r.historical
}

fn matching_receipts(tag: &str, machine: &FitHost, plan: Option<&Plan>) -> Vec<Value> {
    let (Some(device), Some(plan)) = (machine.devices.first(), plan) else {
        return Vec::new();
    };
    let placement = if plan
        .placement
        .host_mapped_bytes
        .is_some_and(|bytes| bytes > 0)
    {
        "host-experts"
    } else {
        "resident"
    };
    receipts::for_model(tag, &device.arch)
        .into_iter()
        .filter(|r| receipt_matches(r, tag, &device.arch, &plan.topology, placement))
        .map(receipt_json)
        .collect()
}

fn receipt_json(r: &receipts::Receipt) -> Value {
    let class = receipts::class_of(r.metric, r.tok_s);
    let band = receipts::class_band(r.metric, class);
    json!({"tag": r.tag, "arch": r.arch, "topology": r.topology,
        "placement": r.placement, "metric": match r.metric { receipts::Metric::Decode => "decode", receipts::Metric::Prefill => "prefill" },
        "route": r.route, "tokens": r.tokens, "context": r.context, "tok_s": r.tok_s,
        "date": r.date, "source": r.source, "quote": r.quote, "historical": r.historical, "note": r.note,
        "class": class.to_string(), "band": band})
}

fn safe_name(input: &str) -> String {
    if input.contains('/') || input.contains('\\') {
        input
            .rsplit(['/', '\\'])
            .find(|s| !s.is_empty())
            .unwrap_or("model")
            .into()
    } else {
        input.into()
    }
}

/// Error strings from the loader/network may carry a local path. The public
/// JSON contract intentionally contains only basenames, never home paths or
/// PCI addresses; scrub nested findings/notes as well as the explicit file.
fn json_report(report: &Report) -> Result<Value> {
    let mut value = serde_json::to_value(report)?;
    fn scrub_with_home(value: &mut Value, home: Option<&str>) {
        match value {
            Value::String(text) => {
                let mut out = String::new();
                for part in text.split_inclusive(char::is_whitespace) {
                    let word = part.trim_end_matches(char::is_whitespace);
                    let suffix = &part[word.len()..];
                    if word.contains("/home/")
                        || word.contains("/root/")
                        || home.is_some_and(|home| home != "/" && word.contains(home))
                    {
                        out.push_str("<local-file>");
                    } else if word.as_bytes().windows(12).any(|b| {
                        b[4] == b':'
                            && b[7] == b':'
                            && b[10] == b'.'
                            && b.iter()
                                .enumerate()
                                .all(|(i, c)| matches!(i, 4 | 7 | 10) || c.is_ascii_hexdigit())
                    }) {
                        out.push_str("<device>");
                    } else {
                        out.push_str(word);
                    }
                    out.push_str(suffix);
                }
                *text = out;
            }
            Value::Array(items) => items
                .iter_mut()
                .for_each(|item| scrub_with_home(item, home)),
            Value::Object(fields) => fields
                .values_mut()
                .for_each(|item| scrub_with_home(item, home)),
            _ => {}
        }
    }
    let home = std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty() && home != "/");
    scrub_with_home(&mut value, home.as_deref());
    Ok(value)
}

fn render_text(report: &Report, verbose: bool) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "hipfire fit {}", report.input);
    let _ = writeln!(out, "this machine");
    for device in &report.machine.devices {
        let free = device.free.map(gib).unwrap_or_else(|| "unknown".into());
        let total = device
            .vram_total
            .map(gib)
            .unwrap_or_else(|| "unknown".into());
        let link = device.link.as_ref().map(|l| l.detail.as_str()).unwrap_or(
            "not calibrated: run `hipfire calibrate` (the engine probes the link at load)",
        );
        let _ = writeln!(
            out,
            "  GPU{} {}  {free} free of {total}  link: {link}",
            device.index, device.arch
        );
    }
    let bytes = |v: Option<u64>| v.map(gib).unwrap_or_else(|| "unknown".into());
    let _ = writeln!(
        out,
        "  host  RAM {}, {} available; GTT cap {}, {} used",
        bytes(report.machine.ram_total),
        bytes(report.machine.mem_available),
        bytes(report.machine.gtt_limit),
        bytes(report.machine.gtt_used)
    );
    if let Some(preflight) = report.findings.iter().find(|f| f.id == "preflight") {
        let _ = writeln!(
            out,
            "  runtime preflight: {}",
            if preflight.verdict == Verdict::Fits {
                "ok"
            } else {
                "blocked (run `hipfire doctor`)"
            }
        );
    }
    let _ = writeln!(
        out,
        "model  {}  {}, {}, arch {}, {}",
        report.model.tag,
        if report.model.installed {
            "installed"
        } else if report.model.replica {
            "header replica (not installed)"
        } else {
            "not installed"
        },
        bytes(report.model.size_bytes),
        report
            .model
            .arch_id
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".into()),
        report.model.quant.as_deref().unwrap_or("quant unknown")
    );
    if let Some(plan) = &report.plan {
        let _ = writeln!(
            out,
            "plan (what `hipfire run` would load on GPU{})",
            plan.device_index
        );
        let _ = writeln!(
            out,
            "  topology   {}",
            if plan.topology == "single" {
                "single GPU"
            } else {
                &plan.topology
            }
        );
        let _ = writeln!(out, "  kv         {}, {}", plan.kv_backend, plan.kv_mode);
        if let Some(reason) = &plan.kv_backend_reason {
            let _ = writeln!(out, "               {reason}");
        }
        let _ = writeln!(
            out,
            "  context    {} ({}); trained window {}",
            plan.max_seq,
            plan.bound,
            plan.model_ctx
                .map(|v| v.to_string())
                .unwrap_or_else(|| "decided at load".into())
        );
        if let Some(p) = &plan.projection {
            let n = |key: &str| p[key].as_u64().unwrap_or(0);
            let _ = writeln!(out, "  capacity   up to {} tokens ({} free - {} weights - {} prefill scratch - {} rounding; {} B/token)", n("card_cap"), gib(n("free")), gib(n("weights")), gib(n("scratch")), gib(n("page_rounding")), n("bytes_per_token"));
        }
        let _ = writeln!(out, "  placement  {}", plan.placement.summary);
        let _ = writeln!(out, "  decode     {}", plan.decode_route);
        for route in &plan.routes {
            let _ = writeln!(
                out,
                "  routes     {}={} ({})",
                route.name, route.value, route.why
            );
        }
    }
    let _ = writeln!(out, "findings");
    for finding in &report.findings {
        let _ = writeln!(
            out,
            "  {} {}: {}",
            finding.verdict.label(),
            finding.id,
            finding.detail
        );
        if let Some(fix) = &finding.fix {
            let _ = writeln!(out, "    fix: {fix}");
        }
    }
    let context = report
        .plan
        .as_ref()
        .map(|p| {
            if p.max_seq == 0 {
                "automatic context".into()
            } else {
                format!("{} tokens", p.max_seq)
            }
        })
        .unwrap_or_else(|| "requested context".into());
    let _ = writeln!(out, "verdict: {} at {context}", report.verdict.label());
    if !report.tiers.is_empty() {
        let _ = writeln!(
            out,
            "quant tiers (model fit only; pull default: {})",
            report.model.tag
        );
        for tier in &report.tiers {
            let _ = writeln!(
                out,
                "  {}  {}  {}  {}",
                tier.tag,
                bytes(tier.size_bytes),
                tier.verdict.label(),
                tier.summary
            );
        }
        let largest = report
            .tiers
            .iter()
            .filter(|t| t.verdict == Verdict::Fits)
            .fold(None::<&TierRow>, |best, tier| match best {
                Some(best) if best.size_bytes.unwrap_or(0) >= tier.size_bytes.unwrap_or(0) => {
                    Some(best)
                }
                _ => Some(tier),
            });
        let _ = writeln!(
            out,
            "largest tier that fits at {context}: {}",
            largest
                .map(|t| t.tag.as_str())
                .unwrap_or("none established")
        );
    }
    let _ = writeln!(
        out,
        "expected speed (published receipts; same model and GPU arch only)"
    );
    if report.receipts.is_empty() {
        let arch = report
            .machine
            .devices
            .first()
            .map(|d| d.arch.as_str())
            .unwrap_or("unknown GPU arch");
        let _ = writeln!(out, "  no published receipt for {} on {arch} with this placement; measure with `hipfire bench {}`", report.model.tag, report.model.tag);
    }
    for row in &report.receipts {
        let s = |key: &str| row[key].as_str().unwrap_or("");
        let at = row["context"]
            .as_u64()
            .map(|v| format!(", at {v}"))
            .unwrap_or_default();
        let historical = if row["historical"].as_bool() == Some(true) {
            " [historical]"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "  {} {} tok/s ({}, {} tokens{at}) on {}, {} — {} ({}){historical} — {}",
            s("metric"),
            row["tok_s"],
            s("route"),
            row["tokens"],
            s("arch"),
            s("date"),
            s("class"),
            s("band"),
            s("source")
        );
        if row["placement"].as_str() == Some("unstated") {
            let _ = writeln!(out, "    (placement not stated by the source)");
        }
        if let Some(note) = row["note"].as_str() {
            let _ = writeln!(out, "    {note}");
        }
    }
    for note in &report.notes {
        let _ = writeln!(out, "note: {note}");
    }
    if verbose {
        for detail in &report.details {
            let _ = writeln!(out, "detail: {detail}");
        }
    } else if !report.details.is_empty() {
        let _ = writeln!(
            out,
            "note: --verbose (or --json) lists how each bound was derived, with engine sources"
        );
    }
    out
}

/// Borrowed controls plus the owned eviction config the load API expects.
/// Environment/process normalization happens before this pure mapper.
struct AdmissionInputs<'a> {
    tp: usize,
    pp: usize,
    max_seq: usize,
    request: KvBackendRequest,
    draft: Option<&'a str>,
    vision: Option<&'a str>,
    vision_mode: &'a str,
    head: Option<&'a str>,
    kv_mode: Option<&'a str>,
    kv_k: Option<&'a str>,
    kv_v: Option<&'a str>,
    kv_adaptive: Option<&'a str>,
    cask: CaskConfig,
    options: SourceAdmissionOptions,
    error: Option<String>,
}

impl AdmissionInputs<'_> {
    fn hints(&self, device: &FitDevice) -> KvBackendHints<'_> {
        KvBackendHints {
            kv_mode: self.kv_mode,
            kv_k: self.kv_k,
            kv_v: self.kv_v,
            kv_adaptive: self.kv_adaptive,
            cask: Some(&self.cask),
            deepseek4_heterogeneous: self.options.non_single_compute,
            qwen_default_q8: qwen_default_q8_enabled(),
            vmm_runtime_available: true,
            free_vram_bytes: device.free.and_then(|n| usize::try_from(n).ok()),
        }
    }

    fn admit(
        &self,
        path: &Path,
        device: &FitDevice,
    ) -> std::result::Result<SourceAdmission, String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        let path = path.to_str().ok_or("model path is not valid UTF-8")?;
        admit_source_with_options(
            path,
            self.tp,
            self.pp,
            self.request,
            self.draft,
            &device.arch,
            self.vision,
            self.vision_mode,
            self.head,
            self.max_seq,
            self.hints(device),
            self.options,
        )
    }
}

fn str_param<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn usize_param(params: &Value, key: &str) -> Option<usize> {
    params.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

/// The sole pure params -> admission mapping. Cited lines are in
/// crates/hipfire-daemon/src/main.rs; external ladders are sampled by
/// model_params so tests never mutate the environment or a process snapshot.
fn admission_inputs(params: &Value) -> AdmissionInputs<'_> {
    let mut error = None;
    // :1327-1339, :1640-1652: omitted is automatic; explicit zero refuses,
    // large explicit requests clamp to MAX_REQUESTED_SEQ (1 Mi tokens, :469).
    let requested = usize_param(params, "max_seq");
    if requested == Some(0) {
        error = Some("load refused: max_seq 0 below floor 1".into());
    }
    let max_seq = requested.unwrap_or(0).min(1_048_576);
    // :1270-1289 resolves pp outside this pure fn; :2094-2102 reads it and tp.
    let tp = usize_param(params, "tp").unwrap_or(1);
    let pp = usize_param(params, "pp").unwrap_or(1);
    if tp > 1 && pp > 1 {
        error.get_or_insert_with(|| hipfire_loader::bench_topology::PP_EP_EXCLUSIVE_MESSAGE.into());
    }
    // :1665-1693: the developer draft ladder is already sampled in params,
    // but dflash_mode=off still suppresses even an explicit draft.
    let dflash_mode = params["dflash_mode"].as_str().unwrap_or("auto");
    // An `auto` pairing is dropped on tp>1 (DFlash is not served on EP/TP);
    // an explicit `on` keeps the draft and meets the tp>1 refusal below.
    let draft = (dflash_mode != "off" && !(dflash_mode == "auto" && tp > 1))
        .then(|| str_param(params, "draft"))
        .flatten();
    // vision_ladder.rs:43-71, same ladder and hard mode gate.
    let vision_mode = params["vision_mode"]
        .as_str()
        .unwrap_or(hipfire_config::vision_mode_default());
    let vision = (vision_mode != "off")
        .then(|| str_param(params, "vision"))
        .flatten();
    // :1748-1777 and :1799-1804: empty overrides are omitted.
    let head = str_param(params, "head");
    let kv_mode = str_param(params, "kv_mode");
    let kv_k = str_param(params, "kv_k");
    let kv_v = str_param(params, "kv_v");
    let kv_adaptive = str_param(params, "kv_adaptive");
    // :2219-2236: use the engine parser, including automatic omission.
    let request =
        KvBackendRequest::from_override(str_param(params, "kv_backend")).unwrap_or_else(|e| {
            error.get_or_insert(e);
            KvBackendRequest::Automatic
        });
    // :1809-1895: process defaults for mtp/ngram are normalized by caller;
    // all fields otherwise retain exactly the daemon's typed per-load values.
    let (dflash_block, dflash_block_auto) =
        hipfire_runtime::dflash::DflashBlockPolicy::from_param(params.get("dflash_block"))
            .map(|p| {
                (
                    p.resolve(
                        params.get("serve_vmm_batch").is_some(),
                        usize_param(params, "continuous_batch_size").unwrap_or(1),
                    ),
                    p == hipfire_runtime::dflash::DflashBlockPolicy::Auto,
                )
            })
            .unwrap_or_else(|e| {
                error.get_or_insert(e);
                (0, false)
            });
    let spec = SpecLoadCfg {
        ngram_draft: params["ngram_draft"].as_bool(),
        ngram_k: usize_param(params, "ngram_k"),
        ngram_min_count: params["ngram_min_count"].as_u64().map(|v| v as u32),
        ddtree_budget: usize_param(params, "ddtree_budget"),
        ddtree_topk: usize_param(params, "ddtree_topk"),
        dspark: params["dspark_mode"]
            .as_str()
            .and_then(hipfire_config::parse_tri_state),
        dspark_conf_threshold: params["dspark_conf_threshold"].as_f64().map(|v| v as f32),
        dflash: hipfire_config::parse_tri_state(dflash_mode),
        mtp: hipfire_config::parse_tri_state(params["mtp_mode"].as_str().unwrap_or("auto")),
        mtp_k: usize_param(params, "mtp_k"),
        dflash_adaptive_b: Some(params["dflash_adaptive_b"].as_bool().unwrap_or(false)),
        dflash_block: Some(dflash_block),
        dflash_block_auto,
    };
    // :1906-1972: keep a CaskConfig even when eviction is off, exactly as
    // admission's hints do; cask option below is SIDECAR presence, not enabled.
    let enabled = params["cask"].as_bool().unwrap_or(false);
    let cask = CaskConfig {
        sidecar: str_param(params, "cask_sidecar").map(str::to_owned),
        cask_m_folding: enabled && draft.is_none(),
        budget: usize_param(params, "cask_budget").unwrap_or(512),
        beta: usize_param(params, "cask_beta").unwrap_or(128),
        handoff_tokens: usize_param(params, "cask_handoff_tokens").unwrap_or(0),
        core_frac: params["cask_core_frac"].as_f64().unwrap_or(0.5) as f32,
        fold_m: usize_param(params, "cask_fold_m").unwrap_or(2),
    };
    // :2152-2183: parse even an empty placement string (daemon refuses it).
    let placement = params["deepseek4_compute_placement"]
        .as_str()
        .unwrap_or("single")
        .parse::<hipfire_config::Deepseek4ComputePlacement>()
        .unwrap_or_else(|e| {
            error.get_or_insert_with(|| format!("invalid DeepSeek V4 compute placement: {e}"));
            hipfire_config::Deepseek4ComputePlacement::Single
        });
    // :2204-2218: every field of SourceAdmissionOptions is the real load option.
    let options = SourceAdmissionOptions {
        spec,
        kv_adaptive: qwen4_kv_adaptive_requested(kv_adaptive),
        eagle_drafter: str_param(params, "drafter").is_some(),
        cask: cask.sidecar.is_some(),
        state_quant: str_param(params, "state_quant").is_some(),
        non_single_compute: !matches!(placement, hipfire_config::Deepseek4ComputePlacement::Single),
        expert_count_override: usize_param(params, "deepseek4_experts_per_token").is_some(),
        pflash: str_param(params, "prefill_drafter").is_some()
            || params["prefill_compression"].as_str().unwrap_or("off") != "off",
    };
    // :2115-2197: source-independent pre-admission feature refusals. They must
    // not become a successful fit merely because a non-Qwen4 carrier is used.
    let refusal = if tp > 1 && draft.is_some() {
        Some("EP serving (tp>1) does not support DFlash drafters in v1; reload without a draft.")
    } else if (tp > 1 || pp > 1) && options.eagle_drafter {
        Some("EAGLE speculative decode requires a single GPU; reload without params.drafter.")
    } else if pp > 1 && draft.is_some() && params["_pp_dflash"].as_bool() != Some(true) {
        Some(
            "DFlash speculative decode requires pp=1; reload without a draft or set topology.pp=1.",
        )
    } else if pp > 1 && options.cask {
        Some(
            "CASK / TriAttention eviction requires pp=1; reload without CASK or set topology.pp=1.",
        )
    } else if pp > 1 && options.pflash && params["_pp_pflash"].as_bool() != Some(true) {
        Some("PFlash prefill compression requires pp=1; turn off prefill compression or set topology.pp=1.")
    } else if tp > 1 && options.expert_count_override {
        Some("DeepSeek V4 experts-per-token override requires tp=1; reload without the experts-per-token override.")
    } else {
        None
    };
    if let Some(refusal) = refusal {
        error.get_or_insert_with(|| refusal.into());
    }
    AdmissionInputs {
        tp,
        pp,
        max_seq,
        request,
        draft,
        vision,
        vision_mode,
        head,
        kv_mode,
        kv_k,
        kv_v,
        kv_adaptive,
        cask,
        options,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn unchecked_context_is_undetermined_and_advisory() {
        let p = projection();
        let fits = unchecked_context_finding(Ok(p), 4096, "legacy KV backend");
        assert_eq!(fits.verdict, Verdict::Undetermined);
        assert!(fits.detail.contains("advisory only"));
        assert!(fits.detail.contains("legacy KV backend"));
        assert_eq!(fits.fix, None);
        let over = unchecked_context_finding(Ok(p), p.card_cap + 1, "MoE on a discrete GPU");
        assert_eq!(over.verdict, Verdict::Undetermined);
        assert!(over
            .fix
            .as_deref()
            .is_some_and(|fix| fix.contains(&p.card_cap.to_string())));
        let failed = unchecked_context_finding(Err("no HFQ".into()), 4096, "legacy KV backend");
        assert_eq!(failed.verdict, Verdict::Undetermined);
        assert!(failed.detail.contains("no projection: no HFQ"));
    }

    fn projection() -> Qwen35CardProjection {
        Qwen35CardProjection {
            free: 32 << 30,
            weights: 20 << 30,
            scratch: 1 << 30,
            page_rounding: 128 << 20,
            bytes_per_token: 49_152,
            card_cap: 200_000,
            model_ctx: 262_144,
            kv_mode: "q8",
        }
    }

    fn device() -> FitDevice {
        FitDevice {
            index: 0,
            arch: "gfx1201".into(),
            unified_memory: false,
            vram_total: Some(32 << 30),
            free: Some(31 << 30),
            link: None,
        }
    }

    fn receipt() -> receipts::Receipt {
        receipts::Receipt {
            tag: "example:mq4".into(),
            arch: "gfx1201".into(),
            topology: "single".into(),
            placement: "resident".into(),
            metric: receipts::Metric::Decode,
            route: "ar".into(),
            tokens: 128,
            context: Some(4096),
            tok_s: 203.9,
            date: "2026-10-17".into(),
            source: "registry/v1.json".into(),
            quote: "203.9 tok/s".into(),
            historical: false,
            note: None,
        }
    }

    fn report() -> Report {
        let p = projection();
        Report {
            schema: "hipfire-fit-1",
            hipfire: json!({"version": "0.4.3", "commit": "test"}),
            input: "example:mq4".into(),
            model: ModelInfo {
                tag: "example:mq4".into(),
                file: Some("example.mq4".into()),
                installed: true,
                replica: false,
                size_bytes: Some(20 << 30),
                arch_id: Some(6),
                quant: Some("MQ4".into()),
            },
            machine: FitHost {
                ram_total: Some(128 << 30),
                mem_available: Some(64 << 30),
                ttm_pool: 0,
                gtt_limit: Some(64 << 30),
                gtt_used: Some(1 << 30),
                devices: vec![device()],
            },
            plan: Some(Plan {
                device_index: 0,
                topology: "single".into(),
                kv_backend: "vmm".into(),
                kv_backend_reason: None,
                kv_mode: "q8".into(),
                max_seq: 65_536,
                bound: "user".into(),
                model_ctx: Some(p.model_ctx),
                projection: Some(projection_json(p)),
                placement: PlacementPlan {
                    summary: "fully resident".into(),
                    expert_layers_resident: None,
                    host_mapped_bytes: Some(0),
                    decided_at_load: false,
                },
                decode_route: "ar".into(),
                routes: vec![Route {
                    name: "prefill_chunk".into(),
                    value: "512".into(),
                    why: "narrow link".into(),
                }],
            }),
            findings: vec![
                Finding::fits("preflight", "runtime preflight ok"),
                context_finding(p, 65_536, false),
                Finding::fails(
                    "host_ram",
                    "host RAM gate failed",
                    "free host RAM or pick a smaller tier",
                ),
            ],
            verdict: Verdict::DoesNotFit,
            tiers: Vec::new(),
            receipts: vec![receipt_json(&receipt())],
            notes: vec!["source /home/private-user/models/example.mq4 on 0000:03:00.0".into()],
            details: vec!["reserve term (expert_residency.rs:1)".into()],
        }
    }

    #[test]
    fn clap_fit_forms_and_command_wiring() {
        for flags in [
            vec![],
            vec!["--ctx", "65536", "--no-fetch"],
            vec!["-j"],
            vec!["--json"],
        ] {
            let mut argv = vec!["hipfire", "fit", "example:mq4"];
            argv.extend(flags);
            let cli = crate::Cli::try_parse_from(argv).unwrap();
            let Some(crate::Commands::Fit(args)) = cli.command else {
                panic!("fit dispatch variant missing")
            };
            assert_eq!(args.model, "example:mq4");
        }
        let cli = crate::Cli::try_parse_from([
            "hipfire",
            "fit",
            "example",
            "--ctx",
            "65536",
            "--no-fetch",
            "-j",
        ])
        .unwrap();
        let Some(crate::Commands::Fit(args)) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.ctx, Some(65_536));
        assert!(args.no_fetch && args.json);
    }

    #[test]
    fn clap_context_range_and_required_model() {
        for n in ["0", "511", "1048577", "-1", "not-a-number"] {
            assert!(crate::Cli::try_parse_from(["hipfire", "fit", "example", "--ctx", n]).is_err());
        }
        for n in ["512", "1048576"] {
            assert!(crate::Cli::try_parse_from(["hipfire", "fit", "example", "--ctx", n]).is_ok());
        }
        assert!(crate::Cli::try_parse_from(["hipfire", "fit"]).is_err());
    }

    #[test]
    fn admission_inputs_map_kv_context_pp_and_mtp() {
        let params = json!({"max_seq": 65536, "pp": 2, "kv_mode": "q8", "kv_backend": "vmm",
            "kv_k": "fwht3", "kv_v": "q8", "kv_adaptive": "off",
            "mtp_mode": "on", "mtp_k": 3, "ngram_draft": true, "ngram_k": 8,
            "ngram_min_count": 2, "ddtree_budget": 16, "ddtree_topk": 4,
            "dspark_mode": "off", "dflash_mode": "off", "dflash_block": 8,
            "dflash_adaptive_b": true, "head": "head.hfq"});
        let a = admission_inputs(&params);
        assert!(a.error.is_none());
        assert_eq!((a.tp, a.pp, a.max_seq), (1, 2, 65_536));
        assert_eq!(
            a.request,
            KvBackendRequest::Explicit(hipfire_runtime::kv_backend::KvBackend::Vmm)
        );
        assert_eq!(
            (a.kv_mode, a.kv_k, a.kv_v, a.kv_adaptive),
            (Some("q8"), Some("fwht3"), Some("q8"), Some("off"))
        );
        assert!(!a.options.kv_adaptive);
        assert_eq!(a.options.spec.mtp, Some(true));
        assert_eq!(a.options.spec.mtp_k, Some(3));
        assert_eq!(a.options.spec.ngram_draft, Some(true));
        assert_eq!(a.options.spec.ddtree_budget, Some(16));
        assert_eq!(a.options.spec.ddtree_topk, Some(4));
        assert_eq!(a.options.spec.dspark, Some(false));
        assert_eq!(a.options.spec.dflash_block, Some(8));
        assert_eq!(a.options.spec.dflash_adaptive_b, Some(true));
        assert_eq!(a.head, Some("head.hfq"));
    }

    #[test]
    fn admission_inputs_defaults_clamp_and_validate() {
        let empty = json!({});
        let a = admission_inputs(&empty);
        assert_eq!((a.tp, a.pp, a.max_seq), (1, 1, 0));
        assert_eq!(a.request, KvBackendRequest::Automatic);
        assert_eq!(a.vision_mode, hipfire_config::vision_mode_default());
        assert!(a.error.is_none());
        assert_eq!(
            admission_inputs(&json!({"max_seq": 2_000_000})).max_seq,
            1_048_576
        );
        assert!(admission_inputs(&json!({"max_seq": 0})).error.is_some());
        assert!(admission_inputs(&json!({"kv_backend": "invalid"}))
            .error
            .is_some());
        assert!(admission_inputs(&json!({"pp": 2, "tp": 2})).error.is_some());
        assert!(
            admission_inputs(&json!({"deepseek4_compute_placement": ""}))
                .error
                .is_some()
        );
    }

    #[test]
    fn admission_inputs_gates_sidecars_and_maps_cask_flags() {
        let params = json!({"draft": "draft.hfq", "dflash_mode": "off", "vision": "vision.hfq",
            "vision_mode": "off", "cask": true, "cask_sidecar": "eviction.hfq",
            "cask_budget": 768, "cask_beta": 256, "cask_handoff_tokens": 128,
            "cask_core_frac": 0.25, "cask_fold_m": 3, "state_quant": "off",
            "prefill_drafter": "prefill.hfq", "drafter": "eagle.hfq",
            "deepseek4_experts_per_token": 4});
        let a = admission_inputs(&params);
        assert!(a.error.is_none());
        assert!(a.draft.is_none() && a.vision.is_none());
        assert!(a.cask.cask_m_folding);
        assert_eq!(
            (
                a.cask.budget,
                a.cask.beta,
                a.cask.handoff_tokens,
                a.cask.fold_m
            ),
            (768, 256, 128, 3)
        );
        assert_eq!(a.cask.core_frac, 0.25);
        assert!(a.options.cask && a.options.state_quant && a.options.pflash);
        assert!(a.options.eagle_drafter && a.options.expert_count_override);
        let params = json!({"draft": "draft.hfq", "dflash_mode": "auto",
            "vision": "vision.hfq", "vision_mode": "auto", "cask": true});
        let a = admission_inputs(&params);
        assert_eq!(a.draft, Some("draft.hfq"));
        assert_eq!(a.vision, Some("vision.hfq"));
        assert!(!a.cask.cask_m_folding);
        assert!(!a.options.cask); // only a sidecar activates this admission gate
                                  // tp>1: an `auto` pairing is dropped (DFlash is not served on EP/TP),
                                  // an explicit `on` keeps the draft for the tp>1 refusal.
        let auto_tp = json!({"draft": "draft.hfq", "dflash_mode": "auto", "tp": 2});
        assert_eq!(admission_inputs(&auto_tp).draft, None);
        let on_tp = json!({"draft": "draft.hfq", "dflash_mode": "on", "tp": 2});
        assert_eq!(admission_inputs(&on_tp).draft, Some("draft.hfq"));
    }

    #[test]
    fn context_user_fits_and_exceeds() {
        assert_eq!(
            context_finding(projection(), 65_536, false).verdict,
            Verdict::Fits
        );
        let failure = context_finding(projection(), 262_144, false);
        assert_eq!(failure.verdict, Verdict::DoesNotFit);
        assert!(failure.detail.contains("49152 bytes"));
        assert!(failure.fix.unwrap().contains("200000"));
    }

    #[test]
    fn context_automatic_and_minimum_floor() {
        let automatic = context_finding(projection(), 0, true);
        assert_eq!(automatic.verdict, Verdict::Fits);
        assert!(automatic
            .detail
            .contains("automatic context: min(trained window"));
        let mut p = projection();
        p.card_cap = QWEN35_MIN_AUTO_CONTEXT - 1;
        let failure = context_finding(p, 0, true);
        assert_eq!(failure.verdict, Verdict::DoesNotFit);
        assert!(!failure.fix.unwrap().is_empty());
    }

    #[test]
    fn verdict_fold_and_exit_codes() {
        assert_eq!(fold_findings(&[]), Verdict::Fits);
        assert_eq!(
            fold_findings(&[
                Finding::fits("context", "ok"),
                Finding::undetermined("weights", "unknown", None)
            ]),
            Verdict::Undetermined
        );
        assert_eq!(
            fold_findings(&[
                Finding::undetermined("context", "unknown", None),
                Finding::fails("host_ram", "too much", "free host RAM")
            ]),
            Verdict::DoesNotFit
        );
        assert_eq!(
            (
                exit_code(Verdict::Fits),
                exit_code(Verdict::DoesNotFit),
                exit_code(Verdict::Undetermined)
            ),
            (0, 1, 2)
        );
    }

    #[test]
    fn receipts_require_exact_tag_arch_topology_and_placement() {
        let mut r = receipt();
        assert!(receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "single",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq3",
            "gfx1201",
            "single",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1100",
            "single",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "ep2",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "single",
            "host-experts"
        ));
        // A source that does not state its placement never matches a plan.
        r.placement = "unstated".into();
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "single",
            "host-experts"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "pp2",
            "host-experts"
        ));
    }

    #[test]
    fn text_report_has_plan_lines_and_every_fix() {
        let report = report();
        let text = render_text(&report, false);
        for line in [
            "plan (what `hipfire run` would load on GPU0)",
            "topology",
            "kv",
            "context",
            "capacity",
            "placement",
            "decode",
            "routes",
            "verdict: DOES NOT FIT",
            "expected speed",
        ] {
            assert!(text.contains(line), "missing {line}");
        }
        for finding in &report.findings {
            if let Some(fix) = &finding.fix {
                assert!(text.contains(&format!("fix: {fix}")));
            }
            if finding.verdict == Verdict::DoesNotFit {
                assert!(finding.fix.as_ref().is_some_and(|fix| !fix.is_empty()));
            }
        }
    }

    #[test]
    fn json_report_schema_and_privacy() {
        let value = json_report(&report()).unwrap();
        assert_eq!(value["schema"], "hipfire-fit-1");
        assert_eq!(value["model"]["file"], "example.mq4");
        assert_eq!(value["plan"]["topology"], "single");
        assert_eq!(value["verdict"], "does-not-fit");
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(!encoded.contains("/home/"));
        assert!(!encoded.contains("private-user"));
        assert!(!encoded.contains("0000:03:00.0"));
    }

    #[test]
    fn unstated_and_historical_receipts_never_match() {
        let mut r = receipt();
        assert!(receipt_matches(
            &r,
            &r.tag.clone(),
            &r.arch.clone(),
            &r.topology.clone(),
            &r.placement.clone()
        ));
        r.placement = "unstated".into();
        assert!(!receipt_matches(
            &r,
            &r.tag.clone(),
            &r.arch.clone(),
            &r.topology.clone(),
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            &r.tag.clone(),
            &r.arch.clone(),
            &r.topology.clone(),
            "host-experts"
        ));
        let mut h = receipt();
        h.historical = true;
        assert!(!receipt_matches(
            &h,
            &h.tag.clone(),
            &h.arch.clone(),
            &h.topology.clone(),
            &h.placement.clone()
        ));
    }

    #[test]
    fn unstated_receipt_placement_is_labelled() {
        let mut report = report();
        report.receipts[0]["placement"] = json!("unstated");
        assert!(render_text(&report, false).contains("(placement not stated by the source)"));
    }

    #[test]
    fn missing_header_has_pull_and_fetch_fixes() {
        let finding = missing_header("example:mq4", "not installed; fit needs the model header");
        assert_eq!(finding.verdict, Verdict::Undetermined);
        let fix = finding.fix.unwrap();
        assert!(fix.contains("hipfire pull example:mq4"));
        assert!(fix.contains("without --no-fetch"));
    }
}
