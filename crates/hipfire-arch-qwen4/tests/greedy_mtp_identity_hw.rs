// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Greedy Flash-Next MTP token identity against AR on a real prompt.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact named by
//! `HIPFIRE_MTP_IDENTITY_MODEL` (explicitly running it without one fails
//! rather than passing as model proof).
//!
//! The rendered `tests/fixtures/qwen4_mtp_identity_p1.txt` prompt (Q8 GDN
//! recurrent state, `max_seq` 16384, K=3, temperature 0) is prefilled once per
//! arm and 1200 greedy token IDs are generated in process through the public
//! bundle and `MtpDrafter` contracts, not the daemon wire. Every arm runs in
//! its own fresh child process of this test binary, because route knobs are
//! read from a process-start snapshot:
//!
//! - `ar`: `forward_chunk_final` then single-row `forward_token_or_argmax`;
//! - `batched`: `HIPFIRE_MTP_INCREMENTAL=0`;
//! - `incremental`: `HIPFIRE_MTP_INCREMENTAL=1`;
//! - `adaptive`: route knob unset;
//! - `full-head`: `HIPFIRE_MTP_INCREMENTAL=0`, `HIPFIRE_MTP_DRAFT_HEAD=full`.
//! - `ngram-batched`, `ngram-incremental`, `ngram-adaptive`: the drafter is built
//!   `.with_ngram(Some(qwen4_ngram_mod_config(ngram_mod_triple_for_arch(16, arch))))`
//!   and the request armed with `allow_ngram_modifier: true`, with route knob
//!   `HIPFIRE_MTP_INCREMENTAL` `0` / `1` / unset;
//! - `ngram-long`: `ngram-adaptive` with the inherited triple
//!   `HIPFIRE_NGRAM_MOD_N_MATCH=24 HIPFIRE_NGRAM_MOD_N_MIN=48 HIPFIRE_NGRAM_MOD_N_MAX=64`
//!   (`qwen4_ngram_mod_config` caps `n_max` to 63).
//!
//! Each MTP arm must emit AR's exact IDs. The route is confirmed from the
//! drafter's own `HIPFIRE_MTP_TRACE` window events (batched windows carry
//! `"baseline":true`), so a silent fallback cannot pass as coverage.
//! Per-arm IDs, decoded text and logs go to `HIPFIRE_MTP_IDENTITY_OUT`
//! (default: a fresh directory under the system temp dir).
//!
//! A second case, `seasons`, renders the serve battery's factual prompt
//! exactly as `hipfire serve` does for a single user message with thinking
//! disabled (`render_messages`, no system block; the token IDs are pinned to
//! the serve-captured ones) and generates 256 tokens. Its short context
//! exercises the 2..=8-row HC projections whose summation order once
//! differed from the single-row decode (Gate041z seasons divergence).
//!
//! A third case, `copy`, renders the committed copy-heavy
//! `benchmarks/prompts/code_edit_rewrite_copy.txt` (sha256 pinned) the way `p1`
//! is rendered and generates 256 tokens, so the n-gram pool actually hits.
//!
//! The n-gram tests (`greedy_mtp_ngram_matches_ar_on_flash_next_{p1,seasons,copy}`)
//! run `ar` plus the four n-gram arms over a case with the same identity check.
//! `HIPFIRE_MTP_TRACE` windows are counted by `"source"` (`ngram` / `mtp`) and
//! `"verify_route"` (`batched` / `interleaved`): `ngram-batched` must have no
//! interleaved windows, `ngram-incremental` no batched windows, and on `copy`
//! every n-gram arm must take at least one n-gram window.
//! `greedy_mtp_ngram_batched_is_deterministic_on_flash_next_copy` runs
//! `ngram-batched` on `copy` in two fresh children and requires identical IDs.
//! The tests spawn GPU children: run with `--test-threads=1`.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::mtp_spec::{
    native_mtp_row_capture, qwen4_ngram_mod_config, Qwen4MtpDrafter,
};
use hipfire_arch_qwen4::{admit_hfqm_artifact, GdnStateFormat, Qwen4KvBackend};
use hipfire_dispatch::pipeline::DraftHeadPolicy;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const OUT_ENV: &str = "HIPFIRE_MTP_IDENTITY_OUT";
const ARM_ENV: &str = "HIPFIRE_MTP_IDENTITY_ARM";
const CASE_ENV: &str = "HIPFIRE_MTP_IDENTITY_CASE";
const ARM_TEST: &str = "greedy_mtp_identity_arm";

const PROMPT: &[u8] = include_bytes!("fixtures/qwen4_mtp_identity_p1.txt");
/// `hipx:~/pm-wave/fns3/in/p1.txt` (md5 `c91f8380a035dc842416b1be5efa8197`).
const PROMPT_SHA256: &str = "4ba1d4d971b99ace1fd294e1cdb087c432d87a155c203bddef87552ce02353a7";
const PROMPT_TOKENS: usize = 1047;
const MAX_SEQ: usize = 16384;
const MTP_K: usize = 3;

/// The serve battery's factual prompt (`scripts/serve_harness.py`).
const SEASONS: &str = "What causes the seasons on Earth? Answer in exactly three sentences.";
/// Prompt IDs `hipfire serve` prefilled for SEASONS (Gate041z, daemon diag dump).
const SEASONS_IDS: [u32; 25] = [
    248045, 846, 198, 3710, 10814, 279, 15127, 383, 8964, 30, 21134, 303, 6681, 2250, 22157, 13,
    248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

/// The committed copy-heavy prompt (`benchmarks/prompts/code_edit_rewrite_copy.txt`).
const COPY_PROMPT: &[u8] =
    include_bytes!("../../../benchmarks/prompts/code_edit_rewrite_copy.txt");
const COPY_PROMPT_SHA256: &str = "5f223a71ca57719c256968a15031f78483459ac22fb649690e3f9077b211e79d";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Case {
    P1,
    Seasons,
    Copy,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::P1 => "p1",
            Case::Seasons => "seasons",
            Case::Copy => "copy",
        }
    }

    fn generated(self) -> usize {
        match self {
            Case::P1 => 1200,
            Case::Seasons => 256,
            Case::Copy => 256,
        }
    }
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    incremental: Option<&'static str>,
    draft_head: Option<&'static str>,
    route: Route,
    /// Build the drafter with the n-gram pool and arm `allow_ngram_modifier`.
    ngram: bool,
    /// `HIPFIRE_NGRAM_MOD_{N_MATCH,N_MIN,N_MAX}` for the child, if overridden.
    ngram_triple: Option<(&'static str, &'static str, &'static str)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Ar,
    Batched,
    Incremental,
    Adaptive,
}

const ARMS: [Arm; 5] = [
    Arm {
        name: "ar",
        incremental: None,
        draft_head: None,
        route: Route::Ar,
        ngram: false,
        ngram_triple: None,
    },
    Arm {
        name: "batched",
        incremental: Some("0"),
        draft_head: None,
        route: Route::Batched,
        ngram: false,
        ngram_triple: None,
    },
    Arm {
        name: "incremental",
        incremental: Some("1"),
        draft_head: None,
        route: Route::Incremental,
        ngram: false,
        ngram_triple: None,
    },
    Arm {
        name: "adaptive",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
        ngram: false,
        ngram_triple: None,
    },
    Arm {
        name: "full-head",
        incremental: Some("0"),
        draft_head: Some("full"),
        route: Route::Batched,
        ngram: false,
        ngram_triple: None,
    },
];

/// AR plus the n-gram combined arms.
const NGRAM_ARMS: [Arm; 5] = [
    Arm {
        name: "ar",
        incremental: None,
        draft_head: None,
        route: Route::Ar,
        ngram: false,
        ngram_triple: None,
    },
    Arm {
        name: "ngram-batched",
        incremental: Some("0"),
        draft_head: None,
        route: Route::Batched,
        ngram: true,
        ngram_triple: None,
    },
    Arm {
        name: "ngram-incremental",
        incremental: Some("1"),
        draft_head: None,
        route: Route::Incremental,
        ngram: true,
        ngram_triple: None,
    },
    Arm {
        name: "ngram-adaptive",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
        ngram: true,
        ngram_triple: None,
    },
    Arm {
        name: "ngram-long",
        incremental: None,
        draft_head: None,
        route: Route::Adaptive,
        ngram: true,
        ngram_triple: Some(("24", "48", "64")),
    },
];

fn model_path() -> PathBuf {
    let path = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    assert!(
        path.is_file(),
        "{MODEL_ENV}={} is not a file",
        path.display()
    );
    path
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen4-mtp-identity-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// Batched and incremental window counts from the drafter's trace events.
fn window_routes(stderr: &str) -> (usize, usize) {
    let windows = stderr.lines().filter(|line| {
        line.starts_with("QWEN4_MTP_TRACE ") && line.contains("\"event\":\"window\"")
    });
    windows.fold((0, 0), |(batched, incremental), line| {
        if line.contains("\"baseline\":true") {
            (batched + 1, incremental)
        } else {
            (batched, incremental + 1)
        }
    })
}

/// Trace windows by `"source"` and `"verify_route"`.
#[derive(Default, Clone, Copy)]
struct WindowCounts {
    ngram: usize,
    mtp: usize,
    batched: usize,
    interleaved: usize,
}

fn window_counts(stderr: &str) -> WindowCounts {
    let mut counts = WindowCounts::default();
    let windows = stderr.lines().filter(|line| {
        line.starts_with("QWEN4_MTP_TRACE ") && line.contains("\"event\":\"window\"")
    });
    for line in windows {
        if line.contains("\"source\":\"ngram\"") {
            counts.ngram += 1;
        } else if line.contains("\"source\":\"mtp\"") {
            counts.mtp += 1;
        }
        if line.contains("\"verify_route\":\"batched\"") {
            counts.batched += 1;
        } else if line.contains("\"verify_route\":\"interleaved\"") {
            counts.interleaved += 1;
        }
    }
    counts
}

/// Route and source assertions for an n-gram arm; returns the counts.
fn ngram_window_check(arm: Arm, case: Case, stderr: &str) -> WindowCounts {
    let w = window_counts(stderr);
    assert_eq!(
        w.ngram + w.mtp,
        w.batched + w.interleaved,
        "{} arm: every window must carry a source and a verify_route",
        arm.name
    );
    let route_ok = match arm.route {
        Route::Ar => w.batched == 0 && w.interleaved == 0,
        Route::Batched => w.batched > 0 && w.interleaved == 0,
        Route::Incremental => w.batched == 0 && w.interleaved > 0,
        Route::Adaptive => w.batched > 0,
    };
    assert!(
        route_ok,
        "{} arm took the wrong verify_route: {} batched and {} interleaved windows ({} ngram, {} mtp)",
        arm.name, w.batched, w.interleaved, w.ngram, w.mtp
    );
    if case == Case::Copy {
        assert!(
            w.ngram > 0,
            "{} arm took no n-gram window on the copy-heavy case ({} mtp windows)",
            arm.name, w.mtp
        );
    }
    w
}

fn first_difference(ar: &[u32], mtp: &[u32]) -> Option<usize> {
    ar.iter()
        .zip(mtp)
        .position(|(a, b)| a != b)
        .or_else(|| (ar.len() != mtp.len()).then(|| ar.len().min(mtp.len())))
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn greedy_mtp_matches_ar_on_flash_next_p1() {
    run_case(Case::P1, &ARMS, "p1");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn greedy_mtp_matches_ar_on_flash_next_seasons() {
    run_case(Case::Seasons, &ARMS, "seasons");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn greedy_mtp_ngram_matches_ar_on_flash_next_p1() {
    run_case(Case::P1, &NGRAM_ARMS, "ngram-p1");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn greedy_mtp_ngram_matches_ar_on_flash_next_seasons() {
    run_case(Case::Seasons, &NGRAM_ARMS, "ngram-seasons");
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn greedy_mtp_ngram_matches_ar_on_flash_next_copy() {
    run_case(Case::Copy, &NGRAM_ARMS, "ngram-copy");
}

/// Same-shape determinism: `ngram-batched` on `copy` in two fresh children.
#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn greedy_mtp_ngram_batched_is_deterministic_on_flash_next_copy() {
    let model = model_path();
    let base = out_dir().join("ngram-copy-determinism");
    let arm = NGRAM_ARMS
        .iter()
        .copied()
        .find(|arm| arm.name == "ngram-batched")
        .expect("ngram-batched arm");
    let mut runs = Vec::new();
    for run in 1..=2 {
        let dir = base.join(format!("run{run}"));
        fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
        let result = spawn_arm(arm, Case::Copy, &model, &dir);
        let w = ngram_window_check(arm, Case::Copy, &result.stderr);
        eprintln!(
            "ngram-batched run {run}: {} ngram and {} mtp windows",
            w.ngram, w.mtp
        );
        runs.push(result.ids);
    }
    if let Some(index) = first_difference(&runs[0], &runs[1]) {
        panic!(
            "ngram-batched copy runs differ at generated index {index} (logs in {})",
            base.display()
        );
    }
}

struct ArmRun {
    ids: Vec<u32>,
    stderr: String,
}

/// Run one arm in a fresh child of this test binary and collect its IDs and
/// trace.
fn spawn_arm(arm: Arm, case: Case, model: &Path, dir: &Path) -> ArmRun {
    let exe = std::env::current_exe().expect("test binary path");
    let mut child = Command::new(&exe);
    child
        .args([
            "--exact",
            ARM_TEST,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM_ENV, arm.name)
        .env(CASE_ENV, case.name())
        .env(MODEL_ENV, model)
        .env(OUT_ENV, dir)
        .env_remove("HIPFIRE_MTP_INCREMENTAL")
        .env_remove("HIPFIRE_MTP_DRAFT_HEAD")
        .env_remove("HIPFIRE_MTP_TRACE")
        .env_remove("HIPFIRE_MTP_NGRAM")
        .env_remove("HIPFIRE_NGRAM_MOD_N_MATCH")
        .env_remove("HIPFIRE_NGRAM_MOD_N_MIN")
        .env_remove("HIPFIRE_NGRAM_MOD_N_MAX")
        .stdin(Stdio::null());
    if arm.route != Route::Ar {
        child.env("HIPFIRE_MTP_TRACE", "1");
    }
    if let Some(value) = arm.incremental {
        child.env("HIPFIRE_MTP_INCREMENTAL", value);
    }
    if let Some(value) = arm.draft_head {
        child.env("HIPFIRE_MTP_DRAFT_HEAD", value);
    }
    if let Some((n_match, n_min, n_max)) = arm.ngram_triple {
        child
            .env("HIPFIRE_NGRAM_MOD_N_MATCH", n_match)
            .env("HIPFIRE_NGRAM_MOD_N_MIN", n_min)
            .env("HIPFIRE_NGRAM_MOD_N_MAX", n_max);
    }
    let output = child
        .output()
        .unwrap_or_else(|e| panic!("spawn {} arm: {e}", arm.name));
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::write(dir.join(format!("{}.out", arm.name)), &output.stdout).unwrap();
    fs::write(dir.join(format!("{}.err", arm.name)), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "{} arm failed ({}); logs in {}",
        arm.name,
        output.status,
        dir.display()
    );
    let ids: Vec<u32> = serde_json::from_slice(
        &fs::read(dir.join(format!("{}.ids.json", arm.name))).expect("arm ids"),
    )
    .expect("arm ids json");
    assert_eq!(ids.len(), case.generated(), "{} arm id count", arm.name);
    ArmRun { ids, stderr }
}

fn run_case(case: Case, arms: &[Arm], dir_name: &str) {
    let model = model_path();
    let dir = out_dir().join(dir_name);
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let mut ids: Vec<(Arm, Vec<u32>)> = Vec::new();
    let mut failures = Vec::new();
    let mut summary = String::new();
    for &arm in arms {
        let run = spawn_arm(arm, case, &model, &dir);
        let arm_ids = run.ids;
        let (batched, incremental) = window_routes(&run.stderr);
        let route_ok = match arm.route {
            Route::Ar => batched == 0 && incremental == 0,
            Route::Batched => batched > 0 && incremental == 0,
            Route::Incremental => batched == 0 && incremental > 0,
            Route::Adaptive => batched > 0,
        };
        assert!(
            route_ok,
            "{} arm took the wrong route: {batched} batched and {incremental} interleaved windows",
            arm.name
        );
        let extra = if arm.ngram {
            let w = ngram_window_check(arm, case, &run.stderr);
            format!(", {} ngram and {} mtp windows", w.ngram, w.mtp)
        } else {
            String::new()
        };
        let line = match ids.first() {
            None => format!("{}: {} ids\n", arm.name, arm_ids.len()),
            Some((_, ar)) => match first_difference(ar, &arm_ids) {
                None => format!(
                    "{}: identical to ar ({batched} batched, {incremental} interleaved windows{extra})\n",
                    arm.name
                ),
                Some(index) => {
                    let line = format!(
                        "{}: first difference at generated index {index}: ar {} vs mtp {} ({batched} batched, {incremental} interleaved windows{extra})\n",
                        arm.name, ar[index], arm_ids[index]
                    );
                    failures.push(line.clone());
                    line
                }
            },
        };
        eprint!("{line}");
        summary.push_str(&line);
        ids.push((arm, arm_ids));
    }
    fs::write(dir.join("summary.txt"), &summary).unwrap();
    assert!(
        failures.is_empty(),
        "greedy MTP differs from AR (logs in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

/// One arm in a fresh process; spawned by `spawn_arm`.
#[test]
#[ignore = "child process of the greedy_mtp_*_flash_next_* tests"]
fn greedy_mtp_identity_arm() {
    let Ok(name) = std::env::var(ARM_ENV) else {
        return;
    };
    let arm = ARMS
        .iter()
        .chain(NGRAM_ARMS.iter())
        .copied()
        .find(|arm| arm.name == name)
        .unwrap_or_else(|| panic!("unknown {ARM_ENV}={name}"));
    let case = match std::env::var(CASE_ENV).as_deref() {
        Ok("p1") => Case::P1,
        Ok("seasons") => Case::Seasons,
        Ok("copy") => Case::Copy,
        other => panic!("unknown {CASE_ENV}={other:?}"),
    };
    let dir = out_dir();
    if let Err(error) = run_arm(arm, case, &model_path(), &dir) {
        panic!("{} arm: {error}", arm.name);
    }
}

fn run_arm(arm: Arm, case: Case, model: &Path, dir: &Path) -> Result<(), String> {
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    let frame = |user: &'static str, system: Option<&'static str>| JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system,
        user,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    };
    let tokens = match case {
        Case::P1 => {
            let digest = Sha256::digest(PROMPT);
            let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            if digest != PROMPT_SHA256 {
                return Err(format!(
                    "prompt fixture sha256 {digest}, expected {PROMPT_SHA256}"
                ));
            }
            let prompt: &'static str = std::str::from_utf8(PROMPT).map_err(|e| e.to_string())?;
            let tokens = tokenizer.encode(&frame(prompt, Some("")).render()?);
            if tokens.len() != PROMPT_TOKENS {
                return Err(format!(
                    "rendered prompt is {} tokens, expected {PROMPT_TOKENS}",
                    tokens.len()
                ));
            }
            tokens
        }
        Case::Copy => {
            let digest = Sha256::digest(COPY_PROMPT);
            let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
            if digest != COPY_PROMPT_SHA256 {
                return Err(format!(
                    "copy prompt sha256 {digest}, expected {COPY_PROMPT_SHA256}"
                ));
            }
            let prompt: &'static str =
                std::str::from_utf8(COPY_PROMPT).map_err(|e| e.to_string())?;
            tokenizer.encode(&frame(prompt, Some("")).render()?)
        }
        Case::Seasons => {
            // `hipfire serve` renders an OpenAI messages request through
            // `render_messages` (no system block when none is sent).
            let user = Message {
                role: Role::User,
                content: SEASONS.to_string(),
                reasoning_content: None,
                name: None,
                rendered_name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            };
            let rendered = frame(SEASONS, None).render_messages(&[user], None, None)?;
            let tokens = tokenizer.encode(&rendered);
            if tokens != SEASONS_IDS {
                return Err(format!(
                    "rendered seasons prompt {tokens:?} is not the serve prompt {SEASONS_IDS:?}"
                ));
            }
            tokens
        }
    };
    let receipt = admit_hfqm_artifact(&hfq).map_err(|e| e.to_string())?;
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| e.to_string())?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|e| e.to_string())?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &receipt.config,
    )?;
    if state_format.gdn != GdnStateFormat::Q8 {
        return Err(format!(
            "GDN recurrent state resolved to {:?}, not Q8",
            state_format.gdn
        ));
    }
    let vocab = receipt.config.vocab_size;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        &mut gpu,
        MAX_SEQ,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    bundle
        .attach_forward(&mut gpu, MAX_SEQ)
        .map_err(|e| e.to_string())?;
    let capture = native_mtp_row_capture(&gpu, &bundle.config);
    let head = hipfire_config::developer_var("HIPFIRE_MTP_DRAFT_HEAD")
        .unwrap_or_else(|_| "mq2r".to_string());
    let head_copy = DraftHeadPolicy::parse(&head).copy;
    if (arm.draft_head == Some("full")) == head_copy.is_some() {
        return Err(format!(
            "draft head policy {head:?} resolved to copy {head_copy:?}"
        ));
    }
    println!(
        "ARM {} arch={} prompt_tokens={} state={state_format:?} backend={} row_capture={capture} draft_head={head}",
        arm.name,
        gpu.arch,
        tokens.len(),
        backend.name()
    );
    let mut ids = Vec::with_capacity(case.generated() + MTP_K);
    if arm.route == Route::Ar {
        let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        bundle
            .forward_chunk_final(&mut gpu, &tokens, &logits, None)
            .map_err(|e| e.to_string())?;
        while ids.len() < case.generated() {
            ids.push(
                bundle
                    .forward_token_or_argmax(&mut gpu, None, &logits)
                    .map_err(|e| e.to_string())?,
            );
        }
        gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        if arm.route == Route::Batched && !capture {
            return Err("this GPU's GDN route does not capture verify rows".into());
        }
        bundle
            .attach_mtp(&mut gpu, MAX_SEQ)
            .map_err(|e| e.to_string())?;
        let mut drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ, None);
        if arm.ngram {
            let triple = hipfire_config::ngram_mod_triple_for_arch(16, &gpu.arch);
            if arm.ngram_triple.is_some() && triple != (24, 48, 64) {
                return Err(format!(
                    "ngram-long resolved triple {triple:?}, expected (24, 48, 64)"
                ));
            }
            let config = qwen4_ngram_mod_config(triple)?;
            println!(
                "NGRAM triple={triple:?} n_match={} n_min={} n_max={}",
                config.n_match, config.n_min, config.n_max
            );
            drafter = drafter.with_ngram(Some(config));
            drafter.configure_request(SpecRequestConfig {
                allow_ngram_modifier: true,
                ..SpecRequestConfig::default()
            });
        }
        let mut seed =
            drafter.mtp_prefill(&mut gpu, &mut bundle, &tokens, &tokens, 0, false, &|| false)?;
        ids.push(seed);
        let eos = bundle.config.eos_token_id;
        while ids.len() < case.generated() {
            let position = bundle.state.position;
            let window = drafter.mtp_step(
                &mut gpu,
                &mut bundle,
                position,
                seed,
                &ids,
                MTP_K,
                eos,
                None,
            )?;
            seed = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
        }
        ids.truncate(case.generated());
        let stats = drafter.request_stats();
        println!("STATS {stats:?}");
        if arm.ngram && !stats.mtp_ngram {
            return Err("n-gram pool was not armed for the request".into());
        }
        Box::new(drafter).mtp_free(&mut gpu);
    }
    fs::write(
        dir.join(format!("{}.ids.json", arm.name)),
        serde_json::to_vec(&ids).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        dir.join(format!("{}.txt", arm.name)),
        tokenizer.decode(&ids),
    )
    .map_err(|e| e.to_string())?;
    bundle.free_gpu(&mut gpu).map_err(|e| e.to_string())
}
