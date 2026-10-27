// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R6/R7 promotion pipeline: `can_promote` evaluates ordered stages and refuses
//! at the first failure. The final `kld_state` stage accepts either the
//! bit-identical exact-output proof or, for numeric candidates, promotion-grade
//! paired KLD evidence ([`kld_state_stage`]); a numeric candidate without that
//! evidence is never promoted.

use super::{AttemptV1, ClaimPolicyV1, KldStateEvidenceV1, PromotionV1, Refusal, TuneEvidenceV1};

pub const STAGES: [&str; 6] = [
    "m7",
    "correctness",
    "abba",
    "noise_floor",
    "confirm",
    "kld_state",
];

fn refuse(idx: usize, reason: String) -> Refusal {
    Refusal {
        stage: STAGES[idx].to_owned(),
        reason,
        passed: STAGES[..idx].iter().map(|s| (*s).to_owned()).collect(),
    }
}

fn attempts(ev: &TuneEvidenceV1) -> impl Iterator<Item = (String, &AttemptV1)> {
    std::iter::once(("discovery".to_owned(), &ev.discovery)).chain(
        ev.confirms
            .iter()
            .enumerate()
            .map(|(i, a)| (format!("confirm[{i}]"), a)),
    )
}

fn stage_m7(ev: &TuneEvidenceV1) -> Result<(), String> {
    for (n, a) in attempts(ev) {
        if a.m7_status.as_deref() != Some("pass") {
            return Err(format!(
                "{n}: m7_status is {:?}, need \"pass\"",
                a.m7_status
            ));
        }
    }
    Ok(())
}

fn stage_correctness(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    for (n, a) in attempts(ev) {
        if a.correctness_pass != Some(true) {
            return Err(format!(
                "{n}: correctness_pass is {:?}, need true",
                a.correctness_pass
            ));
        }
        match a.identical_to_baseline {
            Some(true) => {}
            Some(false) if policy.admit_numeric => {}
            Some(false) => {
                return Err(format!(
                    "{n}: identical_to_baseline is false; NUMERIC candidates are not admitted by this policy (admit_numeric=false)"
                ))
            }
            None => {
                return Err(format!(
                    "{n}: identical_to_baseline is None; need an explicit exact/numeric determination"
                ))
            }
        }
    }
    Ok(())
}

/// ABBA criteria for one attempt (stage 3).
fn abba_ok(n: &str, a: &AttemptV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if a.separable != Some(true) {
        return Err(format!("{n}: separable is {:?}, need true", a.separable));
    }
    if a.direction.as_deref() != Some("faster") {
        return Err(format!(
            "{n}: direction is {:?}, need \"faster\"",
            a.direction
        ));
    }
    match a.sign_p {
        Some(p) if p <= policy.sign_p_max => {}
        other => {
            return Err(format!(
                "{n}: sign_p {other:?} not <= {}",
                policy.sign_p_max
            ))
        }
    }
    if a.material != Some(true) {
        return Err(format!("{n}: material is {:?}, need true", a.material));
    }
    match a.block_ratio {
        Some(r) if (r - 1.0).abs() >= policy.min_effect => {}
        other => {
            return Err(format!(
                "{n}: block_ratio {other:?} effect below min_effect {}",
                policy.min_effect
            ))
        }
    }
    Ok(())
}

/// Noise-floor criteria for one attempt against the evidence floor (stage 4).
fn noise_ok(n: &str, a: &AttemptV1, floor: f64) -> Result<(), String> {
    match a.block_ratio {
        Some(r) if (r - 1.0).abs() > floor => {}
        other => {
            return Err(format!(
                "{n}: block_ratio {other:?} effect does not exceed noise floor {floor}"
            ))
        }
    }
    if a.exceeds_noise_floor != Some(true) {
        return Err(format!(
            "{n}: exceeds_noise_floor is {:?}, need true",
            a.exceeds_noise_floor
        ));
    }
    Ok(())
}

fn stage_abba(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    abba_ok("discovery", &ev.discovery, policy)
}

fn stage_noise(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if ev.aa_runs < policy.aa_processes {
        return Err(format!(
            "aa_runs {} < required {}",
            ev.aa_runs, policy.aa_processes
        ));
    }
    let Some(floor) = ev.noise_floor else {
        return Err("no noise floor measured; a missing floor is never a pass".to_owned());
    };
    noise_ok("discovery", &ev.discovery, floor)
}

fn stage_confirm(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    if ev.confirms.len() < policy.confirm_processes {
        return Err(format!(
            "{} confirmation(s) < required {}",
            ev.confirms.len(),
            policy.confirm_processes
        ));
    }
    let floor = ev
        .noise_floor
        .ok_or_else(|| "no noise floor for confirmation".to_owned())?;
    for (i, c) in ev.confirms.iter().enumerate() {
        let n = format!("confirm[{i}]");
        abba_ok(&n, c, policy)?;
        noise_ok(&n, c, floor)?;
        if c.candidate_code_sha256 != ev.discovery.candidate_code_sha256
            || c.candidate_code_sha256.is_none()
        {
            return Err(format!("{n}: candidate_code_sha256 differs from discovery"));
        }
        if c.baseline_code_sha256 != ev.discovery.baseline_code_sha256
            || c.baseline_code_sha256.is_none()
        {
            return Err(format!("{n}: baseline_code_sha256 differs from discovery"));
        }
    }
    Ok(())
}

/// Run/evidence consistency checks shared by every `kld_state` path.
fn kld_consistency(ev: &TuneEvidenceV1, k: &KldStateEvidenceV1) -> Result<(), String> {
    let (i, c) = (&k.incumbent, &k.candidate);
    if i.role != "incumbent" {
        return Err(format!("incumbent run role is {:?}", i.role));
    }
    if c.role != "candidate" {
        return Err(format!("candidate run role is {:?}", c.role));
    }
    for (n, r) in [("incumbent", i), ("candidate", c)] {
        if r.scoring_mode != "prefill" {
            return Err(format!(
                "{n}: scoring_mode is {:?}, need \"prefill\"",
                r.scoring_mode
            ));
        }
        for var in ["HIPFIRE_GRAPH", "HIPFIRE_NORMALIZE_PROMPT"] {
            if r.env.get(var).map(String::as_str) != Some("0") {
                return Err(format!(
                    "{n}: env {var} is {:?}, need \"0\"",
                    r.env.get(var)
                ));
            }
        }
        if !r.fresh_home {
            return Err(format!("{n}: fresh_home is false"));
        }
    }
    let same = [
        ("reference_sha256", &i.reference_sha256, &c.reference_sha256),
        ("model_sha256", &i.model_sha256, &c.model_sha256),
        ("evaluator_sha256", &i.evaluator_sha256, &c.evaluator_sha256),
        ("kv_mode", &i.kv_mode, &c.kv_mode),
        ("kv_v", &i.kv_v, &c.kv_v),
    ];
    for (n, a, b) in same {
        if a != b {
            return Err(format!("{n} differs between incumbent and candidate runs"));
        }
    }
    if i.max_chunks != c.max_chunks {
        return Err(format!(
            "max_chunks differs between runs ({} vs {})",
            i.max_chunks, c.max_chunks
        ));
    }
    if i.tune_table_path.is_some() {
        return Err("incumbent run used a tune table (tune_table_path is set)".to_owned());
    }
    match (
        &c.selected_code_object_sha256,
        &ev.discovery.candidate_co_sha256,
    ) {
        (Some(a), Some(b)) if a == b => {}
        (a, b) => {
            return Err(format!(
                "candidate selected_code_object_sha256 {a:?} != discovery candidate_co_sha256 {b:?}"
            ))
        }
    }
    Ok(())
}

/// Stage 6: model-level KLD / state-equality evidence. Exposed so callers can
/// judge it independently of stages 1-5.
pub fn kld_state_stage(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), String> {
    let all_identical = attempts(ev).all(|(_, a)| a.identical_to_baseline == Some(true));
    if all_identical {
        // Bit-identical path: the exact-output proof suffices; any attached
        // state evidence must agree.
        if let Some(k) = &ev.kld_state {
            if !k.exact {
                return Err(
                    "kld_state evidence attached to an identical-to-baseline candidate has exact=false (kldseq bytes differ)"
                        .to_owned(),
                );
            }
            kld_consistency(ev, k)?;
        }
        return Ok(());
    }
    let Some(k) = &ev.kld_state else {
        return Err(
            "numeric candidate has no kld_state evidence; no pass without evidence".to_owned(),
        );
    };
    if k.grade != "promotion" {
        return Err(format!(
            "kld_state grade is {:?}; screening-grade evidence never promotes (need \"promotion\")",
            k.grade
        ));
    }
    kld_consistency(ev, k)?;
    let min = policy.kld_promotion_min_chunks;
    if k.incumbent.max_chunks < min {
        return Err(format!(
            "max_chunks {} < kld_promotion_min_chunks {min}",
            k.incumbent.max_chunks
        ));
    }
    if k.paired.n < min {
        return Err(format!(
            "paired.n {} < kld_promotion_min_chunks {min}",
            k.paired.n
        ));
    }
    if k.paired.ci95_hi.is_nan() || k.paired.ci95_hi > policy.kld_margin {
        return Err(format!(
            "paired ci95_hi {} exceeds kld_margin {}",
            k.paired.ci95_hi, policy.kld_margin
        ));
    }
    Ok(())
}

fn run_stages_1_to_5(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<(), Refusal> {
    stage_m7(ev).map_err(|r| refuse(0, r))?;
    stage_correctness(ev, policy).map_err(|r| refuse(1, r))?;
    stage_abba(ev, policy).map_err(|r| refuse(2, r))?;
    stage_noise(ev, policy).map_err(|r| refuse(3, r))?;
    stage_confirm(ev, policy).map_err(|r| refuse(4, r))?;
    Ok(())
}

/// Stages 1-5 all pass (KLD/state not considered).
pub fn is_winner(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> bool {
    run_stages_1_to_5(ev, policy).is_ok()
}

pub fn can_promote(ev: &TuneEvidenceV1, policy: &ClaimPolicyV1) -> Result<PromotionV1, Refusal> {
    run_stages_1_to_5(ev, policy)?;
    kld_state_stage(ev, policy).map_err(|r| refuse(5, r))?;
    let ratios: Vec<f64> = attempts(ev).filter_map(|(_, a)| a.block_ratio).collect();
    let g = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
    Ok(PromotionV1 {
        passed: STAGES.iter().map(|s| (*s).to_owned()).collect(),
        speedup: 1.0 / g,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{CandidateSpecV1, KldRunV1, PairedKldV1};
    use std::collections::BTreeMap;

    fn attempt(phase: &str) -> AttemptV1 {
        AttemptV1 {
            candidate_id: "c".into(),
            phase: phase.into(),
            replicate: 0,
            record_path: "r.json".into(),
            exit_code: 0,
            error_stage: None,
            error_message: None,
            m7_status: Some("pass".into()),
            correctness_pass: Some(true),
            identical_to_baseline: Some(true),
            baseline_launch_symbol: None,
            candidate_launch_symbol: None,
            baseline_co_sha256: None,
            candidate_co_sha256: Some("co".into()),
            baseline_code_sha256: Some("b".into()),
            candidate_code_sha256: Some("c".into()),
            separable: Some(true),
            direction: Some("faster".into()),
            sign_p: Some(1e-5),
            block_ratio: Some(0.95),
            material: Some(true),
            rounds_run: Some(10),
            noise_floor: Some(0.01),
            exceeds_noise_floor: Some(true),
            device_uuid: None,
            fabric_job: None,
        }
    }

    fn ev() -> TuneEvidenceV1 {
        TuneEvidenceV1 {
            candidate: CandidateSpecV1 {
                kernel: "k".into(),
                arch: "gfx1100".into(),
                shape: "s".into(),
                knobs: BTreeMap::new(),
            },
            discovery: attempt("discovery"),
            confirms: vec![attempt("confirm")],
            noise_floor: Some(0.01),
            aa_runs: 3,
            kld_state: None,
        }
    }

    fn stage_of(ev: &TuneEvidenceV1) -> String {
        can_promote(ev, &ClaimPolicyV1::default())
            .unwrap_err()
            .stage
    }

    fn run(role: &str) -> KldRunV1 {
        KldRunV1 {
            role: role.into(),
            model_path: "m".into(),
            model_sha256: "m".into(),
            reference_path: "r".into(),
            reference_sha256: "r".into(),
            evaluator_path: "e".into(),
            evaluator_sha256: "e".into(),
            hipfire_commit: "c".into(),
            scoring_mode: "prefill".into(),
            max_chunks: 32,
            kv_mode: "q8".into(),
            kv_v: "q8".into(),
            env: BTreeMap::from([
                ("HIPFIRE_GRAPH".to_owned(), "0".to_owned()),
                ("HIPFIRE_NORMALIZE_PROMPT".to_owned(), "0".to_owned()),
            ]),
            fresh_home: true,
            tune_table_path: None,
            tune_table_sha256: None,
            selected_code_object_sha256: (role == "candidate").then(|| "co".to_owned()),
            kldseq_path: "k".into(),
            kldseq_sha256: "k".into(),
            device_uuid: "u".into(),
            fabric_job: Some("j".into()),
            slice_mean_kld: 0.01,
        }
    }

    fn kld(grade: &str, exact: bool, ci95_hi: f64) -> KldStateEvidenceV1 {
        KldStateEvidenceV1 {
            grade: grade.into(),
            incumbent: run("incumbent"),
            candidate: run("candidate"),
            paired: PairedKldV1 {
                n: 32,
                mean_diff: 0.0,
                sd_diff: 0.0001,
                ci95_lo: -0.0001,
                ci95_hi,
            },
            exact,
        }
    }

    fn numeric(k: Option<KldStateEvidenceV1>) -> TuneEvidenceV1 {
        let mut e = ev();
        e.discovery.identical_to_baseline = Some(false);
        e.confirms[0].identical_to_baseline = Some(false);
        e.kld_state = k;
        e
    }

    fn kld_reason(e: &TuneEvidenceV1) -> String {
        let r = can_promote(e, &ClaimPolicyV1::default()).unwrap_err();
        assert_eq!(r.stage, "kld_state");
        assert_eq!(r.passed, STAGES[..5].to_vec());
        r.reason
    }

    #[test]
    fn identical_path_passes_without_kld() {
        let e = ev();
        let p = ClaimPolicyV1::default();
        assert!(is_winner(&e, &p));
        let g = can_promote(&e, &p).unwrap();
        assert_eq!(g.passed, STAGES.to_vec());
        assert!(g.speedup > 1.0);
        // consistent exact evidence is fine
        let mut e2 = ev();
        e2.kld_state = Some(kld("promotion", true, 0.0));
        assert!(can_promote(&e2, &p).is_ok());
    }

    #[test]
    fn identical_with_non_exact_or_inconsistent_kld_refuses() {
        let mut e = ev();
        e.kld_state = Some(kld("promotion", false, 0.0));
        assert!(kld_reason(&e).contains("exact=false"));
        let mut e = ev();
        let mut k = kld("promotion", true, 0.0);
        k.candidate.scoring_mode = "decode".into();
        e.kld_state = Some(k);
        assert!(kld_reason(&e).contains("scoring_mode"));
    }

    #[test]
    fn numeric_requires_evidence_and_promotion_grade() {
        assert!(kld_reason(&numeric(None)).contains("no pass without evidence"));
        let r = kld_reason(&numeric(Some(kld("screen", false, 0.0))));
        assert!(r.contains("screening"), "{r}");
    }

    #[test]
    fn numeric_consistency_checks_each_refuse() {
        type Mutator = Box<dyn Fn(&mut KldStateEvidenceV1)>;
        let cases: Vec<(&str, Mutator)> = vec![
            (
                "scoring_mode",
                Box::new(|k| k.incumbent.scoring_mode = "decode".into()),
            ),
            (
                "reference_sha256",
                Box::new(|k| k.candidate.reference_sha256 = "x".into()),
            ),
            (
                "model_sha256",
                Box::new(|k| k.candidate.model_sha256 = "x".into()),
            ),
            (
                "evaluator_sha256",
                Box::new(|k| k.candidate.evaluator_sha256 = "x".into()),
            ),
            ("max_chunks", Box::new(|k| k.candidate.max_chunks = 40)),
            ("kv_mode", Box::new(|k| k.candidate.kv_mode = "x".into())),
            ("kv_v", Box::new(|k| k.candidate.kv_v = "x".into())),
            (
                "HIPFIRE_GRAPH",
                Box::new(|k| {
                    k.candidate.env.insert("HIPFIRE_GRAPH".into(), "1".into());
                }),
            ),
            (
                "HIPFIRE_NORMALIZE_PROMPT",
                Box::new(|k| {
                    k.incumbent.env.remove("HIPFIRE_NORMALIZE_PROMPT");
                }),
            ),
            ("fresh_home", Box::new(|k| k.incumbent.fresh_home = false)),
            (
                "tune table",
                Box::new(|k| k.incumbent.tune_table_path = Some("t.json".into())),
            ),
            (
                "selected_code_object_sha256",
                Box::new(|k| {
                    k.candidate.selected_code_object_sha256 = Some("other".into());
                }),
            ),
            (
                "selected_code_object_sha256",
                Box::new(|k| {
                    k.candidate.selected_code_object_sha256 = None;
                }),
            ),
            (
                "min chunks",
                Box::new(|k| {
                    k.incumbent.max_chunks = 8;
                    k.candidate.max_chunks = 8;
                }),
            ),
            ("paired.n", Box::new(|k| k.paired.n = 8)),
        ];
        for (needle, m) in cases {
            let mut k = kld("promotion", false, 0.0);
            m(&mut k);
            let r = kld_reason(&numeric(Some(k)));
            let needle = if needle == "min chunks" {
                "max_chunks"
            } else {
                needle
            };
            assert!(r.contains(needle), "{needle}: {r}");
        }
    }

    #[test]
    fn numeric_margin_gate() {
        let p = ClaimPolicyV1::default();
        let ok = numeric(Some(kld("promotion", false, 0.0005)));
        assert!(is_winner(&ok, &p));
        let g = can_promote(&ok, &p).unwrap();
        assert_eq!(g.passed, STAGES.to_vec());
        let r = kld_reason(&numeric(Some(kld("promotion", false, 0.0006))));
        assert!(r.contains("ci95_hi"), "{r}");
        assert!(kld_state_stage(&numeric(Some(kld("promotion", false, 0.0))), &p).is_ok());
    }

    #[test]
    fn each_stage_refuses_at_itself() {
        let p = ClaimPolicyV1::default();
        let mut e = ev();
        e.discovery.m7_status = Some("fail".into());
        assert_eq!(stage_of(&e), "m7");
        assert!(!is_winner(&e, &p));

        let mut e = ev();
        e.discovery.identical_to_baseline = Some(false);
        let strict = ClaimPolicyV1 {
            admit_numeric: false,
            ..ClaimPolicyV1::default()
        };
        let r = can_promote(&e, &strict).unwrap_err();
        assert_eq!(r.stage, "correctness");
        assert!(r.reason.contains("NUMERIC candidates are not admitted"));
        let mut e = ev();
        e.discovery.identical_to_baseline = None;
        assert_eq!(stage_of(&e), "correctness");
        let mut e = ev();
        e.discovery.correctness_pass = Some(false);
        assert_eq!(stage_of(&e), "correctness");

        let mut e = ev();
        e.discovery.separable = Some(false);
        assert_eq!(stage_of(&e), "abba");
        let mut e = ev();
        e.discovery.direction = Some("slower".into());
        e.discovery.block_ratio = Some(1.05);
        assert_eq!(stage_of(&e), "abba");
        let mut e = ev();
        e.discovery.block_ratio = Some(0.998);
        assert_eq!(stage_of(&e), "abba");

        let mut e = ev();
        e.noise_floor = None;
        assert_eq!(stage_of(&e), "noise_floor");
        let mut e = ev();
        e.aa_runs = 2;
        assert_eq!(stage_of(&e), "noise_floor");
        let mut e = ev();
        e.noise_floor = Some(0.2);
        assert_eq!(stage_of(&e), "noise_floor");

        let mut e = ev();
        e.confirms.clear();
        assert_eq!(stage_of(&e), "confirm");
        let mut e = ev();
        e.confirms[0].direction = Some("slower".into());
        e.confirms[0].block_ratio = Some(1.05);
        assert_eq!(stage_of(&e), "confirm");
        let mut e = ev();
        e.confirms[0].candidate_code_sha256 = Some("other".into());
        assert_eq!(stage_of(&e), "confirm");
    }
}
