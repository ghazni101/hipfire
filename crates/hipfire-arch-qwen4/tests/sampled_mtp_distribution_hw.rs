// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampled Flash-Next MTP against AR sampling, in distribution.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact named by
//! `HIPFIRE_MTP_IDENTITY_MODEL`.
//!
//! The serve battery's prose prompt (thinking off, rendered as `hipfire
//! serve` renders one user message) is decoded from a cold state
//! `HIPFIRE_SAMPLED_MTP_TRIALS` times (default 2000) per arm, each trial with
//! its own seed, keeping the first `TOKENS` generated IDs:
//!
//! - `ar`: `forward_chunk_final` / `forward_token_or_argmax` with the
//!   production host sampler (`sampler::sample_cpu`);
//! - `batched`: sampled native MTP, `HIPFIRE_MTP_INCREMENTAL=0`;
//! - `interleaved`: sampled native MTP, `HIPFIRE_MTP_INCREMENTAL=1`.
//!
//! Trial seeds are SplitMix64-scrambled (`trial_seed`), distinct per arm:
//! the host AR sampler's xorshift32 maps consecutive seeds to nearly equal
//! first draws, which production avoids by hashing its request seeds.
//!
//! Each MTP arm's per-position marginals and its (t1, t2) joint are compared
//! with AR's by a two-sample chi-square homogeneity test (cells with fewer
//! than 5 expected draws pooled); every p-value must exceed `ALPHA`. Each MTP
//! arm also replays its first `REPLAYS` seeds and must emit the same IDs.
//! Temperature / top_p / top_k / min_p come from `HIPFIRE_SAMPLED_MTP_TEMP` /
//! `HIPFIRE_SAMPLED_MTP_TOP_P` / `HIPFIRE_SAMPLED_MTP_TOP_K` (unset = absent)
//! / `HIPFIRE_SAMPLED_MTP_MIN_P` (default 1.0 / 0.95 / absent / 0); both arms
//! receive them as a request would. Per-arm IDs and the report go to
//! `HIPFIRE_SAMPLED_MTP_OUT`; an arm whose `<arm>.ids.json` is already there
//! is reused, so the arms can run first as separate time-boxed invocations of
//! `sampled_mtp_distribution_arm` (set `HIPFIRE_SAMPLED_MTP_ARM`,
//! `HIPFIRE_MTP_SAMPLED=1` and the arm's `HIPFIRE_MTP_INCREMENTAL` yourself).
//! The arms inherit `HIPFIRE_MTP_SAMPLED_MODE`, so the test checks whichever
//! sampled verifier that selects (unset: speculative rejection sampling).
//!
//! Repeat / presence / frequency penalties (default neutral) reach BOTH arms,
//! as a request's would: `HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY` (1.0),
//! `HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW` (0 = penalties off),
//! `HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY` (0.0) and
//! `HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY` (0.0). The AR arm applies them
//! over the full rendered prompt plus its generated IDs (host `sample_cpu`);
//! the MTP arm installs them on its `SpecRequestConfig`. A malformed value
//! fails the run rather than falling back to neutral.
//!
//! `HIPFIRE_SAMPLED_MTP_PROMPT` selects the prompt fixture: `prose` (default,
//! the prompt above, so old neutral runs reproduce), `repeat` (a
//! repetition-heavy pattern-continuation prompt) or `window` (a prompt of
//! roughly 250 tokens whose distinctive head lies beyond a 128-token window,
//! so W=128 crosses the prompt / generated boundary from the first token;
//! the arm fails unless the prompt is longer than the window). In
//! `naive_sampled_mtp_emits_seeded_ar_ids` a non-`prose` fixture is added as
//! a third prompt.
//!
//! Every arm records its full config (sampling, penalties, fixture, trials,
//! `HIPFIRE_MTP_SAMPLED_MODE`, `HIPFIRE_MTP_INCREMENTAL`) in a sidecar
//! `<ids file stem>.config.json` and prints it. An existing ids file is
//! reused only when its recorded config equals the current one; a missing or
//! different config is a hard error. Use one `HIPFIRE_SAMPLED_MTP_OUT`
//! directory per config (e.g. `.../neutral`, `.../presence1.5`).
//!
//! # N-gram composition arms
//!
//! `sampled_mtp_ngram_matches_ar_distribution_on_flash_next` repeats the
//! comparison with the request-local n-gram pool armed (drafter built
//! `.with_ngram(Some(qwen4_ngram_mod_config((5, 3, 3))?))`, request
//! `allow_ngram_modifier: true`, the same neutral T/top_p/top_k/min_p, no
//! penalties), `HIPFIRE_SAMPLED_MTP_TRIALS` (>= 2000) cold trials per arm:
//!
//! - `ngram_batched` / `ngram_interleaved`: the prose prompt, so the pool
//!   mostly misses and the native window runs; compared with the existing
//!   `ar` arm (a window-source mix must leave the target law unchanged);
//!   native windows (`mtp_windows`) must occur;
//! - `copy_ar`: AR on the copied-span fixture (the reference);
//! - `copy_batched` / `copy_interleaved`: the copied-span fixture, whose
//!   prompt is a "rewrite this passage" request with the passage's first
//!   `COPY_PRIMED` tokens already in the assistant turn, so the pool proposes
//!   the passage's continuation on the first window (the fixture is
//!   pre-checked on the CPU against `MtpNgramContext` for the prompt). Hits
//!   are REQUIRED: `request_stats().ngram_mod_windows`, summed over the
//!   trials, and the offered drafts must both be nonzero, in the arm and again
//!   in the parent from `<arm>.stats.json`.
//!
//! Every MTP arm is compared with its AR reference by the same per-position
//! marginals and (t1, t2) joint chi-square (every p-value > `ALPHA`), and
//! replays its first `REPLAYS` seeds: identical IDs and identical window
//! counters. Each arm runs in a fresh process of
//! `sampled_mtp_ngram_distribution_arm` (`ar` still comes from
//! `sampled_mtp_distribution_arm`; an arm whose `<arm>.ids.json` exists is
//! reused). The `HIPFIRE_SAMPLED_MTP_*_PENALTY` / `_REPEAT_WINDOW` env
//! reaches the n-gram arms and `copy_ar` exactly as above, so a presence-1.5
//! run checks penalized `p` on the n-gram rows (one output dir per config).
//!
//! `naive_sampled_mtp_emits_seeded_ar_ids` checks the stronger property of
//! `HIPFIRE_MTP_SAMPLED_MODE=naive`: with the same seed, every MTP route
//! (`batched`, `interleaved`, `adaptive` = route knob unset) emits the AR
//! arm's exact IDs. Cases: the serve bench's `lru_cache_pep8_strict` and
//! `prose_river_short` prompts × T0.7/top_p 0.8/top_k 20, T1.0/top_p 0.95
//! and T0.8/top_p 0.95/top_k 40/min_p 0.05 × `ID_SEEDS` seeds, `ID_TOKENS`
//! IDs each (cut after EOS). Arms run as fresh processes of
//! `sampled_mtp_identity_arm`; an arm whose `<arm>.identity.json` already
//! exists in `HIPFIRE_SAMPLED_MTP_OUT` is reused.
//!
//! Both identity arms read the same penalty and prompt-fixture env as above.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::mtp_spec::{qwen4_ngram_mod_config, Qwen4MtpDrafter};
use hipfire_arch_qwen4::{admit_hfqm_artifact, Qwen4KvBackend};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::ngram_mod::{MtpNgramContext, NgramModConfig};
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
use hipfire_runtime::spec::{MtpDrafter, MtpRequestStats, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const OUT_ENV: &str = "HIPFIRE_SAMPLED_MTP_OUT";
const ARM_ENV: &str = "HIPFIRE_SAMPLED_MTP_ARM";
const TRIALS_ENV: &str = "HIPFIRE_SAMPLED_MTP_TRIALS";
const TEMP_ENV: &str = "HIPFIRE_SAMPLED_MTP_TEMP";
const TOP_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_P";
const TOP_K_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_K";
const MIN_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_MIN_P";
const ARM_TEST: &str = "sampled_mtp_distribution_arm";
const REPEAT_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_REPEAT_PENALTY";
const REPEAT_WINDOW_ENV: &str = "HIPFIRE_SAMPLED_MTP_REPEAT_WINDOW";
const PRESENCE_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_PRESENCE_PENALTY";
const FREQUENCY_PENALTY_ENV: &str = "HIPFIRE_SAMPLED_MTP_FREQUENCY_PENALTY";
const PROMPT_ENV: &str = "HIPFIRE_SAMPLED_MTP_PROMPT";
const INCREMENTAL_ENV: &str = "HIPFIRE_MTP_INCREMENTAL";
const SAMPLED_MODE_ENV: &str = "HIPFIRE_MTP_SAMPLED_MODE";

/// The serve battery's prose prompt (`scripts/serve_harness.py`).
const PROSE: &str = "Write a four-sentence story about a lighthouse keeper who finds something unexpected washed up on the rocks.";
const TOKENS: usize = 4;
const MAX_SEQ: usize = 4096;
const MTP_K: usize = 3;
const REPLAYS: usize = 5;
const ALPHA: f64 = 1e-3;
const ARMS: [&str; 3] = ["ar", "batched", "interleaved"];

/// The n-gram triple under test (`n_match`, `n_min`, `n_max`): the arch-16
/// default of `hipfire_config::ngram_mod_triple_for_arch`.
const NGRAM_TRIPLE: (usize, usize, usize) = (5, 3, 3);
/// Fewest independent seeded trials per n-gram arm.
const NGRAM_MIN_TRIALS: usize = 2000;
const NGRAM_ARM_TEST: &str = "sampled_mtp_ngram_distribution_arm";
/// The AR reference arm of the copied-span fixture.
const COPY_AR_ARM: &str = "copy_ar";
const COPY_AR_SALT: u64 = 0xC0A11;

/// Copied-span fixture: the user asks for a rewrite that keeps the wording,
/// and the assistant turn already holds the passage's first `COPY_PRIMED`
/// tokens, so the last five context tokens before the first window's seed
/// (the passage's token `COPY_PRIMED`, when sampled) occur in the prompt.
const COPY_INSTRUCTION: &str = "Rewrite the following passage, keeping its wording wherever you can.\n\n";
const COPY_PASSAGE: &str = "The old lighthouse keeper climbed the spiral stairs every evening to light the great lamp, and every morning he polished the brass until it shone like gold.";
const COPY_PRIMED: usize = 8;

/// One n-gram combo arm.
struct NgramArm {
    name: &'static str,
    /// The AR arm whose distribution this arm must match.
    reference: &'static str,
    /// Runs the copied-span fixture (hits required) rather than the prose prompt.
    copy: bool,
    /// `trial_seed` salt, distinct per arm.
    salt: u64,
    /// `HIPFIRE_MTP_INCREMENTAL` the parent gives the arm's process.
    incremental: &'static str,
}

static NGRAM_ARMS: [NgramArm; 4] = [
    NgramArm {
        name: "ngram_batched",
        reference: "ar",
        copy: false,
        salt: 0x4E1,
        incremental: "0",
    },
    NgramArm {
        name: "ngram_interleaved",
        reference: "ar",
        copy: false,
        salt: 0x4E2,
        incremental: "1",
    },
    NgramArm {
        name: "copy_batched",
        reference: COPY_AR_ARM,
        copy: true,
        salt: 0xC0B1,
        incremental: "0",
    },
    NgramArm {
        name: "copy_interleaved",
        reference: COPY_AR_ARM,
        copy: true,
        salt: 0xC0B2,
        incremental: "1",
    },
];

/// SplitMix64 of `arm_salt ^ trial`, never zero.
fn trial_seed(arm_salt: u64, trial: usize) -> u64 {
    let mut z = (arm_salt ^ trial as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parses `name` strictly (unset = `default`): a malformed value is an error,
/// so a typo cannot silently run the neutral config.
fn strict_env<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(v) => v.trim().parse().map_err(|e| format!("{name}={v}: {e}")),
        Err(_) => Ok(default),
    }
}

/// The request's repeat / presence / frequency penalty controls.
#[derive(Clone, Copy)]
struct Penalties {
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl Penalties {
    fn from_env() -> Result<Self, String> {
        let p = Self {
            repeat_penalty: strict_env(REPEAT_PENALTY_ENV, 1.0)?,
            repeat_window: strict_env(REPEAT_WINDOW_ENV, 0)?,
            presence_penalty: strict_env(PRESENCE_PENALTY_ENV, 0.0)?,
            frequency_penalty: strict_env(FREQUENCY_PENALTY_ENV, 0.0)?,
        };
        for (name, v) in [
            (REPEAT_PENALTY_ENV, p.repeat_penalty),
            (PRESENCE_PENALTY_ENV, p.presence_penalty),
            (FREQUENCY_PENALTY_ENV, p.frequency_penalty),
        ] {
            if !v.is_finite() {
                return Err(format!("{name}={v} is not finite"));
            }
        }
        Ok(p)
    }
}

/// The prompt a run decodes from (`HIPFIRE_SAMPLED_MTP_PROMPT`).
#[derive(Clone, Copy, PartialEq)]
enum Fixture {
    Prose,
    Repeat,
    Window,
}

impl Fixture {
    fn from_env() -> Result<Self, String> {
        match std::env::var(PROMPT_ENV).as_deref() {
            Err(_) | Ok("prose") => Ok(Self::Prose),
            Ok("repeat") => Ok(Self::Repeat),
            Ok("window") => Ok(Self::Window),
            Ok(other) => Err(format!(
                "{PROMPT_ENV}={other}: expected prose, repeat or window"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::Repeat => "repeat",
            Self::Window => "window",
        }
    }

    fn text(self) -> String {
        match self {
            Self::Prose => PROSE.to_string(),
            Self::Repeat => format!(
                "Continue this pattern for as long as you can, writing only the continuation: {}",
                "red green blue red green blue red green blue ".repeat(24)
            ),
            // Head (its distinctive names occur only here), filler, tail: with
            // a 128-token window the head has left the window before the first
            // generated token, and the filler drains out as tokens are added.
            Self::Window => format!(
                "Harbor log, early shift: the copper ferry Marigold, the gray tug Osprey, \
                 and the salt barge Tern waited by the cold quay while rain fell on the cranes. {}\
                 Now forget the harbor log. Write a four-sentence story about a lighthouse \
                 keeper who finds something unexpected washed up on the rocks.",
                "Tide table, ordinary entry: water rose slowly, then fell again, as it always \
                 does along this quiet coast. "
                    .repeat(8)
            ),
        }
    }

    /// The window fixture must really cross the penalty window.
    fn check_prompt(self, prompt_tokens: usize, pen: &Penalties) -> Result<(), String> {
        if self == Self::Window && prompt_tokens <= pen.repeat_window.max(128) + TOKENS {
            return Err(format!(
                "window fixture has only {prompt_tokens} prompt tokens; it must exceed the {} \
                 token penalty window",
                pen.repeat_window.max(128)
            ));
        }
        Ok(())
    }
}

/// Penalties and fixture as recorded JSON fields.
fn config_base(kind: &str, arm: &str, pen: &Penalties, fixture: Fixture) -> serde_json::Value {
    serde_json::json!({
        "kind": kind,
        "arm": arm,
        "fixture": fixture.name(),
        "repeat_penalty": pen.repeat_penalty,
        "repeat_window": pen.repeat_window,
        "presence_penalty": pen.presence_penalty,
        "frequency_penalty": pen.frequency_penalty,
        "mtp_k": MTP_K,
    })
}

/// `HIPFIRE_MTP_INCREMENTAL` a distribution / identity arm runs under.
fn arm_incremental(arm: &str) -> Option<&'static str> {
    match arm {
        "batched" => Some("0"),
        "interleaved" => Some("1"),
        _ => None,
    }
}

/// What a child arm process actually runs under.
fn child_incremental(arm: &str) -> Option<String> {
    (arm != "ar")
        .then(|| std::env::var(INCREMENTAL_ENV).ok())
        .flatten()
}

fn dist_config(
    arm: &str,
    incremental: Option<String>,
    pen: &Penalties,
    fixture: Fixture,
    sampling: (f32, f32, Option<u32>, f32),
    trials: usize,
) -> serde_json::Value {
    let mut config = config_base("distribution", arm, pen, fixture);
    config["temp"] = serde_json::json!(sampling.0);
    config["top_p"] = serde_json::json!(sampling.1);
    config["top_k"] = serde_json::json!(sampling.2);
    config["min_p"] = serde_json::json!(sampling.3);
    config["trials"] = serde_json::json!(trials);
    config["tokens"] = serde_json::json!(TOKENS);
    config["incremental"] = serde_json::json!(incremental);
    config["sampled_mode"] = serde_json::json!(std::env::var(SAMPLED_MODE_ENV).ok());
    config
}

fn id_config(
    arm: &str,
    incremental: Option<String>,
    mode: Option<String>,
    pen: &Penalties,
    fixture: Fixture,
) -> serde_json::Value {
    let mut config = config_base("identity", arm, pen, fixture);
    config["tokens"] = serde_json::json!(ID_TOKENS);
    config["seeds"] = serde_json::json!(ID_SEEDS);
    config["incremental"] = serde_json::json!(incremental);
    config["sampled_mode"] = serde_json::json!(mode);
    config
}

/// An existing `ids_path` may only be reused (or overwritten) under the config
/// recorded next to it.
fn check_config(ids_path: &Path, expected: &serde_json::Value) -> Result<(), String> {
    let config_path = ids_path.with_extension("config.json");
    let recorded: serde_json::Value = match fs::read(&config_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: {e}", config_path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "{} exists but {} does not: its config is unknown; remove it or use another {OUT_ENV}",
                ids_path.display(),
                config_path.display()
            ));
        }
        Err(e) => return Err(format!("{}: {e}", config_path.display())),
    };
    if &recorded != expected {
        return Err(format!(
            "{} was produced under a different config; remove it or use another {OUT_ENV}\n  recorded: {recorded}\n  current:  {expected}",
            ids_path.display()
        ));
    }
    Ok(())
}

fn write_config(ids_path: &Path, config: &serde_json::Value) -> Result<(), String> {
    let config_path = ids_path.with_extension("config.json");
    fs::write(
        &config_path,
        serde_json::to_vec_pretty(config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", config_path.display()))
}

/// Sampling / trial / penalty / fixture settings of a distribution run.
struct DistSettings {
    trials: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    pen: Penalties,
    fixture: Fixture,
}

impl DistSettings {
    fn from_env() -> Result<Self, String> {
        Ok(Self {
            trials: env_or(TRIALS_ENV, 2000),
            temp: env_or(TEMP_ENV, 1.0),
            top_p: env_or(TOP_P_ENV, 0.95),
            top_k: std::env::var(TOP_K_ENV)
                .ok()
                .map(|v| v.parse().map_err(|e| format!("{TOP_K_ENV}={v}: {e}")))
                .transpose()?,
            min_p: env_or(MIN_P_ENV, 0.0),
            pen: Penalties::from_env()?,
            fixture: Fixture::from_env()?,
        })
    }

    fn config(&self, arm: &str, incremental: Option<String>) -> serde_json::Value {
        dist_config(
            arm,
            incremental,
            &self.pen,
            self.fixture,
            (self.temp, self.top_p, self.top_k, self.min_p),
            self.trials,
        )
    }
}

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
            std::env::temp_dir().join(format!("qwen4-sampled-mtp-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// ln Γ(x), Lanczos (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma Q(a, x) (series / continued fraction).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_pre = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * ln_pre.exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        ln_pre.exp() * h
    }
}

/// Two-sample chi-square homogeneity test of category counts `a` vs `b`.
/// Categories whose expected count is under 5 in either sample are pooled.
/// Returns `(statistic, degrees of freedom, p-value)`.
fn homogeneity<K: std::hash::Hash + Eq + Clone>(a: &[K], b: &[K]) -> (f64, usize, f64) {
    let mut cells: HashMap<K, (f64, f64)> = HashMap::new();
    for key in a {
        cells.entry(key.clone()).or_default().0 += 1.0;
    }
    for key in b {
        cells.entry(key.clone()).or_default().1 += 1.0;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let n = na + nb;
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    for &(x, y) in cells.values() {
        let total = x + y;
        if total * na.min(nb) / n < 5.0 {
            pooled.0 += x;
            pooled.1 += y;
        } else {
            kept.push((x, y));
        }
    }
    if pooled.0 + pooled.1 > 0.0 {
        kept.push(pooled);
    }
    let mut stat = 0.0;
    for &(x, y) in &kept {
        let total = x + y;
        let (ea, eb) = (total * na / n, total * nb / n);
        stat += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    let df = kept.len().saturating_sub(1).max(1);
    (stat, df, gamma_q(df as f64 / 2.0, stat / 2.0))
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let exe = std::env::current_exe().expect("test binary path");
    let mut samples: Vec<Vec<Vec<u32>>> = Vec::new();
    let settings = DistSettings::from_env().unwrap_or_else(|e| panic!("{e}"));
    for arm in ARMS {
        let ids_path = dir.join(format!("{arm}.ids.json"));
        if ids_path.exists() {
            let expected = settings.config(arm, arm_incremental(arm).map(str::to_string));
            check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
            eprintln!("reusing {}", ids_path.display());
            samples
                .push(serde_json::from_slice(&fs::read(&ids_path).unwrap()).expect("arm ids json"));
            continue;
        }
        let mut child = Command::new(&exe);
        child
            .args([
                "--exact",
                ARM_TEST,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ARM_ENV, arm)
            .env(MODEL_ENV, &model)
            .env(OUT_ENV, &dir)
            .env("HIPFIRE_MTP_SAMPLED", "1")
            .env_remove("HIPFIRE_MTP_INCREMENTAL")
            .stdin(Stdio::null());
        match arm {
            "batched" => {
                child.env("HIPFIRE_MTP_INCREMENTAL", "0");
            }
            "interleaved" => {
                child.env("HIPFIRE_MTP_INCREMENTAL", "1");
            }
            _ => {}
        }
        let output = child
            .output()
            .unwrap_or_else(|e| panic!("spawn {arm}: {e}"));
        fs::write(dir.join(format!("{arm}.out")), &output.stdout).unwrap();
        fs::write(dir.join(format!("{arm}.err")), &output.stderr).unwrap();
        assert!(
            output.status.success(),
            "{arm} arm failed ({}); logs in {}",
            output.status,
            dir.display()
        );
        let arm_samples: Vec<Vec<u32>> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm ids")).expect("arm ids json");
        samples.push(arm_samples);
    }
    let mut report = format!("config (ar arm): {}\n", settings.config("ar", None));
    let mut failures = Vec::new();
    let ar = &samples[0];
    for (arm, mtp) in ARMS.iter().zip(&samples).skip(1) {
        for position in 0..TOKENS {
            let a: Vec<u32> = ar.iter().map(|ids| ids[position]).collect();
            let b: Vec<u32> = mtp.iter().map(|ids| ids[position]).collect();
            let (stat, df, p) = homogeneity(&a, &b);
            let line = format!(
                "{arm} vs ar, token {}: chi2={stat:.2} df={df} p={p:.4} (n={}+{})\n",
                position + 1,
                a.len(),
                b.len()
            );
            if p < ALPHA {
                failures.push(line.clone());
            }
            report.push_str(&line);
        }
        let a: Vec<(u32, u32)> = ar.iter().map(|ids| (ids[0], ids[1])).collect();
        let b: Vec<(u32, u32)> = mtp.iter().map(|ids| (ids[0], ids[1])).collect();
        let (stat, df, p) = homogeneity(&a, &b);
        let line = format!("{arm} vs ar, (t1,t2) joint: chi2={stat:.2} df={df} p={p:.4}\n");
        if p < ALPHA {
            failures.push(line.clone());
        }
        report.push_str(&line);
    }
    eprint!("{report}");
    fs::write(dir.join("report.txt"), &report).unwrap();
    assert!(
        failures.is_empty(),
        "sampled MTP differs from AR in distribution (alpha {ALPHA}; logs in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

/// One arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of sampled_mtp_matches_ar_distribution_on_flash_next"]
fn sampled_mtp_distribution_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

/// A loaded Flash-Next target with its tokenizer and chat template.
struct Loaded {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    template: String,
    state: String,
}

fn load(model: &Path) -> Result<Loaded, String> {
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
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
    Ok(Loaded {
        gpu,
        bundle,
        tokenizer,
        template,
        state: format!("{state_format:?}"),
    })
}

impl Loaded {
    /// `text` as one user message, thinking off, rendered as `hipfire serve`
    /// renders it.
    fn render(&self, text: &str) -> Result<Vec<u32>, String> {
        let user = Message {
            role: Role::User,
            content: text.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let rendered = JinjaChatFrame {
            tokenizer: &self.tokenizer,
            template: &self.template,
            system: None,
            user: text,
            enable_thinking: false,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        }
        .render_messages(&[user], None, None)?;
        Ok(self.tokenizer.encode(&rendered))
    }

    /// Seeded AR from a cold state with the production host sampler: up to
    /// `tokens` IDs, cut after EOS.
    fn ar_ids(
        &mut self,
        logits: &rdna_compute::GpuTensor,
        prompt: &[u32],
        cfg: &SamplerConfig,
        seed: u32,
        tokens: usize,
    ) -> Result<Vec<u32>, String> {
        let eos = self.bundle.config.eos_token_id;
        hipfire_runtime::llama::reset_cpu_sampler_rng(seed);
        self.bundle
            .reset(&mut self.gpu)
            .map_err(|e| e.to_string())?;
        self.bundle
            .forward_chunk_final(&mut self.gpu, prompt, logits, None)
            .map_err(|e| e.to_string())?;
        let mut history = prompt.to_vec();
        let mut ids = Vec::with_capacity(tokens);
        loop {
            let mut row = self.gpu.download_f32(logits).map_err(|e| e.to_string())?;
            let token = sample_cpu(&mut row, &history, cfg);
            ids.push(token);
            history.push(token);
            if ids.len() == tokens || token == eos {
                return Ok(ids);
            }
            self.bundle
                .forward_token_or_argmax(&mut self.gpu, Some(token), logits)
                .map_err(|e| e.to_string())?;
        }
    }

    /// Sampled native MTP from a cold state: up to `tokens` IDs, cut after
    /// EOS. Returns the IDs and the windows' (drafts, accepted) totals.
    fn mtp_ids(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
    ) -> Result<(Vec<u32>, usize, usize), String> {
        self.mtp_request(drafter, prompt, cfg, tokens)
            .map(|(ids, drafted, accepted, _)| (ids, drafted, accepted))
    }

    /// [`mtp_ids`](Self::mtp_ids) plus the request's `request_stats()`.
    fn mtp_request(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
    ) -> Result<(Vec<u32>, usize, usize, MtpRequestStats), String> {
        let eos = self.bundle.config.eos_token_id;
        drafter.configure_request(cfg);
        let mut seed_token = drafter.mtp_prefill(
            &mut self.gpu,
            &mut self.bundle,
            prompt,
            prompt,
            0,
            false,
            &|| false,
        )?;
        let mut ids = vec![seed_token];
        let (mut drafted, mut accepted) = (0, 0);
        while ids.len() < tokens && seed_token != eos {
            let position = self.bundle.state.position;
            let window = drafter.mtp_step(
                &mut self.gpu,
                &mut self.bundle,
                position,
                seed_token,
                &ids,
                MTP_K.min(tokens - ids.len()),
                eos,
                None,
            )?;
            drafted += window.drafts_generated;
            accepted += window.accepted;
            seed_token = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
        }
        ids.truncate(tokens);
        Ok((ids, drafted, accepted, drafter.request_stats()))
    }
}

fn ar_sampler(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    pen: &Penalties,
) -> SamplerConfig {
    SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: pen.repeat_penalty,
        repeat_window: pen.repeat_window,
        presence_penalty: pen.presence_penalty,
        frequency_penalty: pen.frequency_penalty,
        blocked_tokens: Vec::new(),
        top_k,
        min_p: (min_p > 0.0).then_some(min_p),
    }
}

fn spec_request(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    seed: u64,
    pen: &Penalties,
) -> SpecRequestConfig {
    SpecRequestConfig {
        temp,
        top_p,
        top_k,
        min_p,
        cactus_delta: 0.0,
        rng_seed: seed,
        allow_ngram_modifier: false,
        repeat_penalty: pen.repeat_penalty,
        repeat_window: pen.repeat_window,
        presence_penalty: pen.presence_penalty,
        frequency_penalty: pen.frequency_penalty,
        ..SpecRequestConfig::default()
    }
}

/// [`spec_request`] with the n-gram pool armed (`allow_ngram_modifier`).
fn spec_request_ngram(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    seed: u64,
    pen: &Penalties,
) -> SpecRequestConfig {
    SpecRequestConfig {
        allow_ngram_modifier: true,
        ..spec_request(temp, top_p, top_k, min_p, seed, pen)
    }
}

fn new_drafter(loaded: &mut Loaded) -> Result<Qwen4MtpDrafter, String> {
    new_drafter_with(loaded, None)
}

/// A sampled-verify drafter; `ngram` is the pool configuration requests that
/// arm `allow_ngram_modifier` use (`None`: every window native).
fn new_drafter_with(
    loaded: &mut Loaded,
    ngram: Option<NgramModConfig>,
) -> Result<Qwen4MtpDrafter, String> {
    loaded
        .bundle
        .attach_mtp(&mut loaded.gpu, MAX_SEQ)
        .map_err(|e| e.to_string())?;
    let drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ, None).with_ngram(ngram);
    if !drafter.supports_temp_verify() {
        return Err("HIPFIRE_MTP_SAMPLED did not enable sampled verification".into());
    }
    Ok(drafter)
}

fn run_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let DistSettings {
        trials,
        temp,
        top_p,
        top_k,
        min_p,
        pen,
        fixture,
    } = DistSettings::from_env()?;
    let settings_config = dist_config(
        arm,
        child_incremental(arm),
        &pen,
        fixture,
        (temp, top_p, top_k, min_p),
        trials,
    );
    let ids_path = dir.join(format!("{arm}.ids.json"));
    if ids_path.exists() {
        check_config(&ids_path, &settings_config)?;
    }
    let mut loaded = load(model)?;
    let prompt = loaded.render(&fixture.text())?;
    fixture.check_prompt(prompt.len(), &pen)?;
    let eos = loaded.bundle.config.eos_token_id;
    println!(
        "ARM {arm} arch={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} repeat_penalty={} repeat_window={} presence_penalty={} frequency_penalty={} fixture={} state={}",
        loaded.gpu.arch,
        prompt.len(),
        pen.repeat_penalty,
        pen.repeat_window,
        pen.presence_penalty,
        pen.frequency_penalty,
        fixture.name(),
        loaded.state
    );
    println!("ARM {arm} config {settings_config}");
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(trials);
    if arm == "ar" {
        let logits = loaded
            .gpu
            .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
            .map_err(|e| e.to_string())?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
        for trial in 0..trials {
            let seed = trial_seed(0xA11, trial) as u32;
            samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, TOKENS)?);
        }
        loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        let mut drafter = new_drafter(&mut loaded)?;
        let mut run = |loaded: &mut Loaded, seed: u64| {
            let cfg = spec_request(temp, top_p, top_k, min_p, seed, &pen);
            loaded
                .mtp_ids(&mut drafter, &prompt, cfg, TOKENS)
                .map(|(ids, _, _)| ids)
        };
        for trial in 0..trials {
            samples.push(run(&mut loaded, trial_seed(0x5EC, trial))?);
        }
        for trial in 0..REPLAYS.min(trials) {
            let replay = run(&mut loaded, trial_seed(0x5EC, trial))?;
            if replay != samples[trial] {
                return Err(format!(
                    "seed {} replayed {replay:?}, first run {:?}",
                    trial + 1,
                    samples[trial]
                ));
            }
        }
        println!(
            "ARM {arm} replayed {} seeds identically",
            REPLAYS.min(trials)
        );
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} {trials} trials in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &settings_config)?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

const ID_ARM_TEST: &str = "sampled_mtp_identity_arm";
const ID_ARMS: [&str; 4] = ["ar", "batched", "interleaved", "adaptive"];
const ID_TOKENS: usize = 128;
const ID_SEEDS: usize = 6;
/// (name, temperature, top_p, top_k, min_p).
const ID_SAMPLINGS: [(&str, f32, f32, Option<u32>, f32); 3] = [
    ("T0.7/p0.8/k20", 0.7, 0.8, Some(20), 0.0),
    ("T1.0/p0.95", 1.0, 0.95, None, 0.0),
    ("T0.8/p0.95/k40/minp0.05", 0.8, 0.95, Some(40), 0.05),
];
const ID_PROMPTS: [(&str, &str); 2] = [
    (
        "lru_cache",
        include_str!("../../../benchmarks/prompts/lru_cache_pep8_strict.txt"),
    ),
    (
        "prose_river",
        include_str!("../../../benchmarks/prompts/prose_river_short.txt"),
    ),
];

/// One identity case's result in an arm.
struct IdCase {
    case: String,
    ids: Vec<u32>,
    drafted: usize,
    accepted: usize,
}

impl IdCase {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "case": self.case,
            "ids": self.ids,
            "drafted": self.drafted,
            "accepted": self.accepted,
        })
    }

    fn from_json(value: &serde_json::Value) -> Self {
        let count = |key: &str| value[key].as_u64().expect("identity count") as usize;
        Self {
            case: value["case"].as_str().expect("identity case").to_string(),
            ids: serde_json::from_value(value["ids"].clone()).expect("identity ids"),
            drafted: count("drafted"),
            accepted: count("accepted"),
        }
    }
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn naive_sampled_mtp_emits_seeded_ar_ids() {
    let model = model_path();
    let dir = out_dir();
    let exe = std::env::current_exe().expect("test binary path");
    let mut arms: Vec<Vec<IdCase>> = Vec::new();
    let pen = Penalties::from_env().unwrap_or_else(|e| panic!("{e}"));
    let fixture = Fixture::from_env().unwrap_or_else(|e| panic!("{e}"));
    for arm in ID_ARMS {
        let ids_path = dir.join(format!("{arm}.identity.json"));
        let expected = id_config(
            arm,
            arm_incremental(arm).map(str::to_string),
            Some("naive".to_string()),
            &pen,
            fixture,
        );
        if ids_path.exists() {
            check_config(&ids_path, &expected).unwrap_or_else(|e| panic!("{e}"));
            eprintln!("reusing {}", ids_path.display());
        } else {
            let mut child = Command::new(&exe);
            child
                .args([
                    "--exact",
                    ID_ARM_TEST,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(ARM_ENV, arm)
                .env(MODEL_ENV, &model)
                .env(OUT_ENV, &dir)
                .env("HIPFIRE_MTP_SAMPLED", "1")
                .env("HIPFIRE_MTP_SAMPLED_MODE", "naive")
                .env_remove("HIPFIRE_MTP_INCREMENTAL")
                .stdin(Stdio::null());
            match arm {
                "batched" => {
                    child.env("HIPFIRE_MTP_INCREMENTAL", "0");
                }
                "interleaved" => {
                    child.env("HIPFIRE_MTP_INCREMENTAL", "1");
                }
                _ => {}
            }
            let output = child
                .output()
                .unwrap_or_else(|e| panic!("spawn {arm}: {e}"));
            fs::write(dir.join(format!("{arm}.identity.out")), &output.stdout).unwrap();
            fs::write(dir.join(format!("{arm}.identity.err")), &output.stderr).unwrap();
            assert!(
                output.status.success(),
                "{arm} arm failed ({}); logs in {}",
                output.status,
                dir.display()
            );
        }
        let json: Vec<serde_json::Value> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm identity ids"))
                .expect("arm identity json");
        arms.push(json.iter().map(IdCase::from_json).collect());
    }
    let mut report = String::new();
    let mut failures = 0usize;
    let ar = &arms[0];
    for (arm, cases) in ID_ARMS.iter().zip(&arms).skip(1) {
        assert_eq!(cases.len(), ar.len(), "{arm}: case count");
        let mut same = 0usize;
        for (a, m) in ar.iter().zip(cases) {
            assert_eq!(a.case, m.case, "{arm}: case order");
            match a.ids.iter().zip(&m.ids).position(|(x, y)| x != y) {
                None if a.ids.len() == m.ids.len() => same += 1,
                first => {
                    failures += 1;
                    report.push_str(&format!(
                        "{arm} {}: first differs at {} (ar {} ids, mtp {} ids)\n",
                        m.case,
                        first.unwrap_or(a.ids.len().min(m.ids.len())),
                        a.ids.len(),
                        m.ids.len()
                    ));
                }
            }
        }
        let (drafted, accepted) = cases
            .iter()
            .fold((0, 0), |(d, a), c| (d + c.drafted, a + c.accepted));
        report.push_str(&format!(
            "{arm}: {same}/{} cases identical to AR; drafts {drafted}, accepted {accepted}\n",
            ar.len()
        ));
    }
    eprint!("{report}");
    fs::write(dir.join("identity-report.txt"), &report).unwrap();
    assert_eq!(
        failures,
        0,
        "naive sampled MTP left AR's seeded IDs (logs in {}):\n{report}",
        dir.display()
    );
}

/// One identity arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of naive_sampled_mtp_emits_seeded_ar_ids"]
fn sampled_mtp_identity_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_identity_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} identity arm: {error}");
    }
}

fn run_identity_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let pen = Penalties::from_env()?;
    let fixture = Fixture::from_env()?;
    let config = id_config(
        arm,
        child_incremental(arm),
        std::env::var(SAMPLED_MODE_ENV).ok(),
        &pen,
        fixture,
    );
    let ids_path = dir.join(format!("{arm}.identity.json"));
    if ids_path.exists() {
        check_config(&ids_path, &config)?;
    }
    println!("ARM {arm} config {config}");
    let mut prompts: Vec<(String, String)> = ID_PROMPTS
        .iter()
        .map(|(name, text)| (name.to_string(), text.to_string()))
        .collect();
    if fixture != Fixture::Prose {
        prompts.push((fixture.name().to_string(), fixture.text()));
    }
    let mut loaded = load(model)?;
    let started = std::time::Instant::now();
    let logits = loaded
        .gpu
        .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
        .map_err(|e| e.to_string())?;
    let mut drafter = if arm == "ar" {
        None
    } else {
        Some(new_drafter(&mut loaded)?)
    };
    let mut out = Vec::new();
    for (prompt_name, text) in &prompts {
        let prompt = loaded.render(text)?;
        if *prompt_name == fixture.name() {
            fixture.check_prompt(prompt.len(), &pen)?;
        }
        for (sampling, temp, top_p, top_k, min_p) in ID_SAMPLINGS {
            for n in 0..ID_SEEDS {
                let seed = trial_seed(0x1D, n) as u32;
                let case = format!("{prompt_name} {sampling} seed#{n}");
                let (ids, drafted, accepted) = match drafter.as_mut() {
                    None => {
                        let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
                        let ids = loaded.ar_ids(&logits, &prompt, &cfg, seed, ID_TOKENS)?;
                        (ids, 0, 0)
                    }
                    Some(drafter) => {
                        let cfg = spec_request(temp, top_p, top_k, min_p, seed as u64, &pen);
                        loaded.mtp_ids(drafter, &prompt, cfg, ID_TOKENS)?
                    }
                };
                println!(
                    "ARM {arm} {case}: {} ids, drafts {drafted}, accepted {accepted}",
                    ids.len()
                );
                out.push(IdCase {
                    case,
                    ids,
                    drafted,
                    accepted,
                });
            }
        }
    }
    println!(
        "ARM {arm} {} cases in {:.1}s (arch {}, state {})",
        out.len(),
        started.elapsed().as_secs_f64(),
        loaded.gpu.arch,
        loaded.state
    );
    fs::write(
        &ids_path,
        serde_json::to_vec(&out.iter().map(IdCase::to_json).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    write_config(&ids_path, &config)?;
    if let Some(drafter) = drafter {
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// N-gram composition arms
// ---------------------------------------------------------------------------

/// Per-position marginals and the (t1, t2) joint of `mtp` against its AR
/// `reference`; lines go to `report`, p-values under `ALPHA` to `failures`.
fn compare_to_reference(
    arm: &str,
    reference: &str,
    ar: &[Vec<u32>],
    mtp: &[Vec<u32>],
    report: &mut String,
    failures: &mut Vec<String>,
) {
    for position in 0..TOKENS {
        let a: Vec<u32> = ar.iter().map(|ids| ids[position]).collect();
        let b: Vec<u32> = mtp.iter().map(|ids| ids[position]).collect();
        let (stat, df, p) = homogeneity(&a, &b);
        let line = format!(
            "{arm} vs {reference}, token {}: chi2={stat:.2} df={df} p={p:.4} (n={}+{})\n",
            position + 1,
            a.len(),
            b.len()
        );
        if p < ALPHA {
            failures.push(line.clone());
        }
        report.push_str(&line);
    }
    let a: Vec<(u32, u32)> = ar.iter().map(|ids| (ids[0], ids[1])).collect();
    let b: Vec<(u32, u32)> = mtp.iter().map(|ids| (ids[0], ids[1])).collect();
    let (stat, df, p) = homogeneity(&a, &b);
    let line = format!("{arm} vs {reference}, (t1,t2) joint: chi2={stat:.2} df={df} p={p:.4}\n");
    if p < ALPHA {
        failures.push(line.clone());
    }
    report.push_str(&line);
}

/// Run one arm as a fresh process of `test`.
fn run_ngram_arm_process(
    test: &str,
    arm: &str,
    incremental: Option<&str>,
    model: &Path,
    dir: &Path,
) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("test binary path: {e}"))?;
    let mut child = Command::new(exe);
    child
        .args([
            "--exact",
            test,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM_ENV, arm)
        .env(MODEL_ENV, model)
        .env(OUT_ENV, dir)
        .env("HIPFIRE_MTP_SAMPLED", "1")
        .env_remove("HIPFIRE_MTP_INCREMENTAL")
        .stdin(Stdio::null());
    if let Some(value) = incremental {
        child.env("HIPFIRE_MTP_INCREMENTAL", value);
    }
    let output = child
        .output()
        .map_err(|e| format!("spawn {arm}: {e}"))?;
    fs::write(dir.join(format!("{arm}.out")), &output.stdout).map_err(|e| e.to_string())?;
    fs::write(dir.join(format!("{arm}.err")), &output.stderr).map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "{arm} arm failed ({}); logs in {}",
            output.status,
            dir.display()
        ));
    }
    Ok(())
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_ngram_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let trials: usize = env_or(TRIALS_ENV, NGRAM_MIN_TRIALS);
    assert!(
        trials >= NGRAM_MIN_TRIALS,
        "{TRIALS_ENV}={trials}: the n-gram arms need at least {NGRAM_MIN_TRIALS} trials"
    );
    let mut arms: Vec<(&str, Option<&NgramArm>)> = vec![("ar", None), (COPY_AR_ARM, None)];
    arms.extend(NGRAM_ARMS.iter().map(|arm| (arm.name, Some(arm))));
    let mut samples: HashMap<&str, Vec<Vec<u32>>> = HashMap::new();
    for (name, spec) in arms {
        let ids_path = dir.join(format!("{name}.ids.json"));
        if ids_path.exists() {
            eprintln!("reusing {}", ids_path.display());
        } else {
            let test = if name == "ar" {
                ARM_TEST
            } else {
                NGRAM_ARM_TEST
            };
            run_ngram_arm_process(test, name, spec.map(|arm| arm.incremental), &model, &dir)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let ids: Vec<Vec<u32>> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm ids")).expect("arm ids json");
        assert!(
            ids.len() >= NGRAM_MIN_TRIALS,
            "{name}: {} trials in {}, need at least {NGRAM_MIN_TRIALS}",
            ids.len(),
            ids_path.display()
        );
        samples.insert(name, ids);
    }
    let mut report = String::new();
    let mut failures = Vec::new();
    for arm in &NGRAM_ARMS {
        compare_to_reference(
            arm.name,
            arm.reference,
            &samples[arm.reference],
            &samples[arm.name],
            &mut report,
            &mut failures,
        );
        let stats_path = dir.join(format!("{}.stats.json", arm.name));
        let bytes = fs::read(&stats_path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (the arm writes its stats with its ids; delete {}.ids.json to rerun it)",
                stats_path.display(),
                arm.name
            )
        });
        let stats: serde_json::Value = serde_json::from_slice(&bytes).expect("arm stats json");
        let count = |key: &str| stats[key].as_u64().expect("arm stats count") as usize;
        let (windows, drafts, accepted, native) = (
            count("ngram_mod_windows"),
            count("ngram_mod_drafts"),
            count("ngram_mod_accepted"),
            count("mtp_windows"),
        );
        let line = format!(
            "{}: ngram_mod_windows={windows} drafts={drafts} accepted={accepted} mtp_windows={native} (summed over {} trials)\n",
            arm.name,
            count("trials")
        );
        if arm.copy && (windows == 0 || drafts == 0) {
            failures.push(format!("{line}  the copied-span fixture produced no n-gram hits\n"));
        } else if !arm.copy && native == 0 {
            failures.push(format!("{line}  no native MTP window ran on the prose prompt\n"));
        }
        report.push_str(&line);
    }
    eprint!("{report}");
    fs::write(dir.join("ngram-report.txt"), &report).unwrap();
    assert!(
        failures.is_empty(),
        "n-gram + sampled MTP differs from AR in distribution or never hit (alpha {ALPHA}; logs in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

/// One n-gram arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of sampled_mtp_ngram_matches_ar_distribution_on_flash_next"]
fn sampled_mtp_ngram_distribution_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_ngram_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

fn sampling_from_env() -> Result<(f32, f32, Option<u32>, f32), String> {
    let temp: f32 = env_or(TEMP_ENV, 1.0);
    let top_p: f32 = env_or(TOP_P_ENV, 0.95);
    let top_k: Option<u32> = std::env::var(TOP_K_ENV)
        .ok()
        .map(|v| v.parse().map_err(|e| format!("{TOP_K_ENV}={v}: {e}")))
        .transpose()?;
    let min_p: f32 = env_or(MIN_P_ENV, 0.0);
    Ok((temp, top_p, top_k, min_p))
}

/// The copied-span fixture's prompt: the rendered rewrite request plus the
/// passage's first `COPY_PRIMED` tokens as the start of the assistant turn.
/// Fails unless the passage tokens occur in the request and the pool, seeded
/// from this prompt, proposes the passage's next `n_max` tokens after the
/// passage token `COPY_PRIMED` (the first window's seed when sampled).
fn copy_fixture(loaded: &Loaded) -> Result<Vec<u32>, String> {
    let (_, n_min, n_max) = NGRAM_TRIPLE;
    let passage = loaded.tokenizer.encode(COPY_PASSAGE);
    if passage.len() < COPY_PRIMED + 1 + n_max.max(n_min) {
        return Err(format!(
            "copy passage has {} tokens, need {}",
            passage.len(),
            COPY_PRIMED + 1 + n_max.max(n_min)
        ));
    }
    let mut prompt = loaded.render(&format!("{COPY_INSTRUCTION}{COPY_PASSAGE}"))?;
    if !prompt
        .windows(passage.len())
        .any(|window| window == passage.as_slice())
    {
        return Err("the passage's own tokens do not occur in the rendered request".into());
    }
    prompt.extend_from_slice(&passage[..COPY_PRIMED]);
    let mut pool = MtpNgramContext::new(qwen4_ngram_mod_config(NGRAM_TRIPLE)?)
        .map_err(|e| e.to_string())?;
    pool.begin_request(&prompt);
    let seed = &passage[COPY_PRIMED..=COPY_PRIMED];
    let want = &passage[COPY_PRIMED + 1..COPY_PRIMED + 1 + n_max];
    match pool.propose(seed, n_max) {
        Some(candidates) if candidates == want => Ok(prompt),
        other => Err(format!(
            "copy fixture: the pool proposed {other:?} after the passage's token {COPY_PRIMED}, expected {want:?}"
        )),
    }
}

fn run_ngram_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let spec = NGRAM_ARMS.iter().find(|spec| spec.name == arm);
    if spec.is_none() && arm != COPY_AR_ARM {
        return Err(format!("unknown n-gram arm {arm}"));
    }
    let trials: usize = env_or(TRIALS_ENV, NGRAM_MIN_TRIALS);
    if trials < NGRAM_MIN_TRIALS {
        return Err(format!(
            "{TRIALS_ENV}={trials}: the n-gram arms need at least {NGRAM_MIN_TRIALS} trials"
        ));
    }
    let (temp, top_p, top_k, min_p) = sampling_from_env()?;
    // The env penalties reach both the AR reference and the n-gram arms.
    let pen = Penalties::from_env()?;
    let mut loaded = load(model)?;
    let copy = spec.is_none_or(|spec| spec.copy);
    let prompt = if copy {
        copy_fixture(&loaded)?
    } else {
        loaded.render(PROSE)?
    };
    let eos = loaded.bundle.config.eos_token_id;
    println!(
        "ARM {arm} arch={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} triple={NGRAM_TRIPLE:?} state={}",
        loaded.gpu.arch,
        prompt.len(),
        loaded.state
    );
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(trials);
    match spec {
        None => {
            let logits = loaded
                .gpu
                .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
                .map_err(|e| e.to_string())?;
            let cfg = ar_sampler(temp, top_p, top_k, min_p, &pen);
            for trial in 0..trials {
                let seed = trial_seed(COPY_AR_SALT, trial) as u32;
                samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, TOKENS)?);
            }
            loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
        }
        Some(spec) => {
            let config = qwen4_ngram_mod_config(NGRAM_TRIPLE)?;
            let mut drafter = new_drafter_with(&mut loaded, Some(config))?;
            let mut run = |loaded: &mut Loaded, trial: usize| {
                let cfg = spec_request_ngram(
                    temp,
                    top_p,
                    top_k,
                    min_p,
                    trial_seed(spec.salt, trial),
                    &pen,
                );
                loaded.mtp_request(&mut drafter, &prompt, cfg, TOKENS)
            };
            let (mut windows, mut drafts, mut accepted, mut native) = (0usize, 0usize, 0usize, 0usize);
            let mut first_stats: Vec<MtpRequestStats> = Vec::new();
            for trial in 0..trials {
                let (ids, _, _, stats) = run(&mut loaded, trial)?;
                if !stats.mtp_ngram {
                    return Err(format!("trial {trial}: the n-gram pool was not armed"));
                }
                windows += stats.ngram_mod_windows;
                drafts += stats.ngram_mod_drafts;
                accepted += stats.ngram_mod_accepted;
                native += stats.mtp_windows;
                if trial < REPLAYS {
                    first_stats.push(stats);
                }
                samples.push(ids);
            }
            for trial in 0..REPLAYS.min(trials) {
                let (replay, _, _, stats) = run(&mut loaded, trial)?;
                if replay != samples[trial] || stats != first_stats[trial] {
                    return Err(format!(
                        "trial {trial} replayed {replay:?} / {stats:?}, first run {:?} / {:?}",
                        samples[trial], first_stats[trial]
                    ));
                }
            }
            println!(
                "ARM {arm} replayed {} seeds identically; ngram_mod_windows={windows} drafts={drafts} accepted={accepted} mtp_windows={native}",
                REPLAYS.min(trials)
            );
            if spec.copy && (windows == 0 || drafts == 0) {
                return Err(format!(
                    "the copied-span fixture produced no n-gram hits in {trials} trials (ngram_mod_windows={windows}, drafts={drafts})"
                ));
            }
            if !spec.copy && native == 0 {
                return Err(format!(
                    "no native MTP window ran in {trials} prose trials (ngram_mod_windows={windows})"
                ));
            }
            fs::write(
                dir.join(format!("{arm}.stats.json")),
                serde_json::to_vec(&serde_json::json!({
                    "trials": trials,
                    "ngram_mod_windows": windows,
                    "ngram_mod_drafts": drafts,
                    "ngram_mod_accepted": accepted,
                    "mtp_windows": native,
                }))
                .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            Box::new(drafter).mtp_free(&mut loaded.gpu);
        }
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} {trials} trials in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        dir.join(format!("{arm}.ids.json")),
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}
