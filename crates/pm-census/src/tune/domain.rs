// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R4 domain enumeration: the ordered candidate set of one corpus node.
//!
//! Two node families are supported, both on arch `gfx1201`:
//!
//! * kernel `qwen4_moe_sym` (the shipped entries have NT1/NT4/NT8 gate/up and down
//!   siblings plus the grouped-position `down_gp` twins). The node's launch symbol
//!   selects the *kind* (`gate_up` or `down`) and the incumbent `--epi`; the
//!   candidates are every `--epi` entry of that kind that `pm_loop` knows. The model
//!   family selects the expert-geometry preset, and the node's `M` / `m_total`
//!   params are checked against that preset so a corpus node can never be tuned at a
//!   shape it did not run.
//! * kernel `iu4_gemm`, the dense 27B prefill `gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`
//!   (the slab-activation silu gate/up GEMM; knobs `fold`/`tile`/`cacc`/`epi`/`alayout`
//!   as `pm_loop` names them). The rocprof-derived corpus node carries no params, only
//!   its launch grid, so the model-family preset (`dense-27b`: hidden K = 5120,
//!   intermediate M = 17408) is checked against the grid: `grid.x = 2*M/128` virtual
//!   128-row tiles and `grid.y = ceil(tokens/128)`. The candidates are the generator
//!   twin of the incumbent (the negative control), the one other tile (illegal for the
//!   slab silu symbol, so the offline screen prunes it with the generator's reason) and
//!   one candidate per `.rip` script, whose path is the `rip` knob (`pm_loop --rip`).
//!   The scripts are given to [`enumerate_with_rips`], or for the CLI (which calls
//!   [`enumerate`]) as the `*.rip` files of the directory named by the environment
//!   variable [`RIP_DIR_VAR`]. Their sha256 enter the derivation and the domain hash.
//!
//! This module decides **no legality**: a listed candidate may still be
//! rejected by the generator, by M7 or by `pm_loop`'s baseline resolution. That
//! screen is the offline phase of [`super::run`], which prunes with a recorded
//! reason. The incumbent's own knob set is listed on purpose: it emits code
//! byte-identical to the shipped object and serves as the negative control.

use super::{canonical_sha256, CandidateSpecV1, DomainV1, DOMAIN_SCHEMA};
use hipfire_atlas::workload::{ChainNodeV1, WorkloadCaseV1};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const KERNEL: &str = "qwen4_moe_sym";
const IU4_KERNEL: &str = "iu4_gemm";
const ARCH: &str = "gfx1201";

/// Environment variable naming a directory whose `*.rip` files [`enumerate`]
/// lists as the `rip` candidates of an `iu4_gemm` node.
pub const RIP_DIR_VAR: &str = "PM_CENSUS_RIP_DIR";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    GateUp,
    Down,
}

/// Expert-geometry preset of a model family (`pm_loop` `--shape` preset name).
struct Preset {
    name: &'static str,
    experts: u64,
    top: u64,
    hidden: u64,
    inter: u64,
}

fn preset(family: &str) -> Result<Preset, String> {
    match family {
        "a3b" => Ok(Preset {
            name: "a3b",
            experts: 256,
            top: 8,
            hidden: 2048,
            inter: 512,
        }),
        "flash-next" => Ok(Preset {
            name: "fn",
            experts: 512,
            top: 10,
            hidden: 2560,
            inter: 640,
        }),
        other => Err(format!(
            "model family {other:?} has no expert-geometry preset (a3b, flash-next)"
        )),
    }
}

/// `(kind, --epi)` of a corpus `qwen4_moe_sym` launch symbol.
fn classify_moe(symbol: &str) -> Option<(Kind, &'static str)> {
    const TABLE: [(&str, Kind, &str); 8] = [
        (
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
            Kind::GateUp,
            "gate_up",
        ),
        (
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
            Kind::GateUp,
            "gate_up_nt4",
        ),
        (
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
            Kind::GateUp,
            "gate_up_nt8",
        ),
        ("qwen4_moe_down_iu4_sym_pm_gfx1201", Kind::Down, "down"),
        (
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
            Kind::Down,
            "down_nt4",
        ),
        (
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
            Kind::Down,
            "down_nt8",
        ),
        (
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4",
            Kind::Down,
            "down_gp_nt4",
        ),
        (
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8",
            Kind::Down,
            "down_gp_nt8",
        ),
    ];
    TABLE
        .iter()
        .find(|(s, _, _)| *s == symbol)
        .map(|(_, k, e)| (*k, *e))
}

/// The shipped launch symbol of the dense 27B prefill gate/up silu GEMM on the
/// slab activation layout (module `gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s`).
pub const IU4_GATE_UP_SILU_B1S: &str = "gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s";

/// The `pm_loop --kernel iu4_gemm` knob values that select a shipped symbol.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Iu4Knobs {
    fold: &'static str,
    tile: &'static str,
    cacc: &'static str,
    epi: &'static str,
    alayout: &'static str,
}

/// What a corpus launch symbol is tunable as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Node {
    /// `qwen4_moe_sym`: its kind and incumbent `--epi`.
    Moe(Kind, &'static str),
    /// `iu4_gemm`: the knob set of the shipped symbol.
    Iu4(Iu4Knobs),
}

/// The tunable identity of a corpus launch symbol (`None`: not an enumerable node).
fn classify(symbol: &str) -> Option<Node> {
    if symbol == IU4_GATE_UP_SILU_B1S {
        return Some(Node::Iu4(Iu4Knobs {
            fold: "k128",
            tile: "128x128x8",
            cacc: "1",
            epi: "silu",
            alayout: "slab",
        }));
    }
    classify_moe(symbol).map(|(k, e)| Node::Moe(k, e))
}

fn candidate_epis(kind: Kind) -> &'static [&'static str] {
    match kind {
        Kind::GateUp => &["gate_up", "gate_up_nt4", "gate_up_nt8"],
        Kind::Down => &["down", "down_nt4", "down_nt8", "down_gp_nt4", "down_gp_nt8"],
    }
}

fn ceil16(x: u64) -> u64 {
    x.div_ceil(16) * 16
}

fn spec(shape: &str, epi: &str) -> CandidateSpecV1 {
    CandidateSpecV1 {
        kernel: KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape: shape.to_owned(),
        knobs: BTreeMap::from([("epi".to_owned(), epi.to_owned())]),
    }
}

/// The `*.rip` files of `dir` as absolute paths, sorted by file name.
pub fn rips_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut out = Vec::new();
    for ent in rd {
        let p = ent.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        if p.extension().is_some_and(|x| x == "rip") && p.is_file() {
            out.push(
                p.canonicalize()
                    .map_err(|e| format!("{}: {e}", p.display()))?,
            );
        }
    }
    out.sort();
    Ok(out)
}

/// The ordered domain of `case.chain[node]` at `tokens` rows. No GPU probing and
/// no legality decision (see the module docs). An `iu4_gemm` node takes its `rip`
/// candidates from the `*.rip` files of the directory named by [`RIP_DIR_VAR`]
/// (none when unset); every other node ignores that variable.
pub fn enumerate(case: &WorkloadCaseV1, node: usize, tokens: u64) -> Result<DomainV1, String> {
    let is_iu4 = case
        .chain
        .get(node)
        .is_some_and(|n| matches!(classify(&n.symbol), Some(Node::Iu4(_))));
    let rips = match std::env::var_os(RIP_DIR_VAR) {
        Some(dir) if is_iu4 => rips_in_dir(Path::new(&dir))?,
        _ => Vec::new(),
    };
    enumerate_with_rips(case, node, tokens, &rips)
}

/// [`enumerate`] with explicit `.rip` candidate scripts (`iu4_gemm` nodes only; a
/// node of another kernel refuses a non-empty list).
pub fn enumerate_with_rips(
    case: &WorkloadCaseV1,
    node: usize,
    tokens: u64,
    rips: &[PathBuf],
) -> Result<DomainV1, String> {
    if case.arch != ARCH {
        return Err(format!(
            "case arch is {:?}; only {ARCH} is enumerable",
            case.arch
        ));
    }
    let n = case.chain.get(node).ok_or_else(|| {
        format!(
            "node {node} out of range: case has {} chain nodes",
            case.chain.len()
        )
    })?;
    match classify(&n.symbol) {
        Some(Node::Moe(kind, incumbent_epi)) => {
            if !rips.is_empty() {
                return Err(format!(
                    "node {node} symbol {} is a {KERNEL} entry: .rip candidates are enumerated \
                     for {IU4_KERNEL} nodes only",
                    n.symbol
                ));
            }
            enumerate_moe(case, n, node, tokens, kind, incumbent_epi)
        }
        Some(Node::Iu4(k)) => enumerate_iu4(case, n, node, tokens, k, rips),
        None => Err(format!(
            "node {node} symbol {:?} is not an enumerable {ARCH} entry \
             ({KERNEL} entries, {IU4_GATE_UP_SILU_B1S})",
            n.symbol
        )),
    }
}

/// `pm_loop` `iu4_gemm` knob set of `k` plus `overrides`.
fn iu4_spec(shape: &str, k: Iu4Knobs, overrides: &[(&str, &str)]) -> CandidateSpecV1 {
    let mut knobs: BTreeMap<String, String> = [
        ("fold", k.fold),
        ("tile", k.tile),
        ("cacc", k.cacc),
        ("epi", k.epi),
        ("alayout", k.alayout),
    ]
    .iter()
    .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
    .collect();
    for (a, b) in overrides {
        knobs.insert((*a).to_owned(), (*b).to_owned());
    }
    CandidateSpecV1 {
        kernel: IU4_KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape: shape.to_owned(),
        knobs,
    }
}

/// The dense 27B prefill gate/up GEMM: `M` intermediate rows (gate and up each),
/// `K` hidden, `N = tokens`.
struct DensePreset {
    hidden: u64,
    inter: u64,
}

fn dense_preset(family: &str) -> Result<DensePreset, String> {
    match family {
        "dense-27b" => Ok(DensePreset {
            hidden: 5120,
            inter: 17408,
        }),
        other => Err(format!(
            "model family {other:?} has no dense gate/up geometry preset (dense-27b)"
        )),
    }
}

/// Rows per workgroup tile of the shipped `128x128x8` iu4 tile (`pm_loop` `--tile`).
const IU4_TILE_ROWS: u64 = 128;
/// Tokens per workgroup tile of that tile.
const IU4_TILE_TOKENS: u64 = 128;

fn enumerate_iu4(
    case: &WorkloadCaseV1,
    n: &ChainNodeV1,
    node: usize,
    tokens: u64,
    k: Iu4Knobs,
    rips: &[PathBuf],
) -> Result<DomainV1, String> {
    let p = dense_preset(&case.model.family)?;
    let mut derivation = vec![
        format!(
            "model family {:?} -> preset dense-27b (K={}, M={} intermediate rows)",
            case.model.family, p.hidden, p.inter
        ),
        format!(
            "node {node} symbol {} -> kernel {IU4_KERNEL}, incumbent knobs fold={} tile={} \
             cacc={} epi={} alayout={}",
            n.symbol, k.fold, k.tile, k.cacc, k.epi, k.alayout
        ),
    ];
    if let Some(&m) = n.params.get("M") {
        if m != p.inter {
            return Err(format!(
                "node {node} param M is {m}, preset dense-27b needs {}",
                p.inter
            ));
        }
    }
    if tokens == 0 {
        return Err("tokens must be at least 1".to_owned());
    }
    let grid = n.grid.ok_or_else(|| {
        format!(
            "node {node} has no launch grid: the {IU4_KERNEL} shape is bound by the grid \
             (x = 2*M/{IU4_TILE_ROWS}, y = ceil(tokens/{IU4_TILE_TOKENS}))"
        )
    })?;
    let want_x = (2 * p.inter).div_ceil(IU4_TILE_ROWS);
    if u64::from(grid[0]) != want_x {
        return Err(format!(
            "node {node} grid.x is {}, preset dense-27b needs 2*M/{IU4_TILE_ROWS} = {want_x}",
            grid[0]
        ));
    }
    let want_y = tokens.div_ceil(IU4_TILE_TOKENS);
    if u64::from(grid[1]) != want_y {
        return Err(format!(
            "node {node} grid.y is {}, tokens={tokens} needs ceil(tokens/{IU4_TILE_TOKENS}) \
             = {want_y}",
            grid[1]
        ));
    }
    derivation.push(format!(
        "grid.x = {} equals 2*{}/{IU4_TILE_ROWS}; grid.y = {} equals ceil({tokens}/{IU4_TILE_TOKENS})",
        grid[0], p.inter, grid[1]
    ));

    let shape = format!("M={},N={tokens},K={}", p.inter, p.hidden);
    let incumbent = iu4_spec(&shape, k, &[]);
    // Deterministic order: the generator twin (control), the other tile, then the scripts.
    let mut candidates = vec![
        incumbent.clone(),
        iu4_spec(&shape, k, &[("tile", "256x128x16")]),
    ];
    let mut rip_shas = Vec::new();
    for path in rips {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let p = path
            .to_str()
            .ok_or_else(|| format!("{}: the rip path is not valid UTF-8", path.display()))?;
        derivation.push(format!("rip candidate {p} sha256 {sha}"));
        rip_shas.push(json!({"path": p, "sha256": sha}));
        candidates.push(iu4_spec(&shape, k, &[("rip", p)]));
    }
    let domain_hash = canonical_sha256(&json!({
        "kernel": IU4_KERNEL,
        "arch": ARCH,
        "shape": shape,
        "incumbent": incumbent,
        "candidates": candidates,
        "rips": rip_shas,
    }));
    Ok(DomainV1 {
        schema: DOMAIN_SCHEMA.to_owned(),
        case_id: case.case_id.clone(),
        node_index: node,
        node_symbol: n.symbol.clone(),
        kernel: IU4_KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape,
        incumbent,
        candidates,
        derivation,
        domain_hash,
    })
}

fn enumerate_moe(
    case: &WorkloadCaseV1,
    n: &ChainNodeV1,
    node: usize,
    tokens: u64,
    kind: Kind,
    incumbent_epi: &'static str,
) -> Result<DomainV1, String> {
    let p = preset(&case.model.family)?;
    let mut derivation = vec![format!(
        "model family {:?} -> preset {} (E={}, TOP={}, H={}, I={})",
        case.model.family, p.name, p.experts, p.top, p.hidden, p.inter
    )];
    derivation.push(format!(
        "node {node} symbol {} -> kind {}, incumbent --epi {incumbent_epi}",
        n.symbol,
        if kind == Kind::GateUp {
            "gate_up"
        } else {
            "down"
        }
    ));

    let (want_m, m_rule) = match kind {
        Kind::GateUp => (2 * p.inter, "2*I"),
        Kind::Down => (p.hidden, "H"),
    };
    let m = *n
        .params
        .get("M")
        .ok_or_else(|| format!("node {node} has no param M"))?;
    if m != want_m {
        return Err(format!(
            "node {node} param M is {m}, preset {} needs {m_rule} = {want_m}",
            p.name
        ));
    }
    derivation.push(format!(
        "param M = {m} equals {m_rule} of preset {}",
        p.name
    ));

    let want_total = ceil16(
        tokens
            .saturating_mul(p.top)
            .saturating_add(p.experts.saturating_mul(15)),
    );
    let m_total = *n
        .params
        .get("m_total")
        .ok_or_else(|| format!("node {node} has no param m_total"))?;
    if m_total != want_total {
        return Err(format!(
            "node {node} param m_total is {m_total}, tokens={tokens} on preset {} needs \
             ceil16(T*TOP + E*15) = {want_total}",
            p.name
        ));
    }
    derivation.push(format!(
        "param m_total = {m_total} equals ceil16({tokens}*{} + {}*15) at T={tokens}",
        p.top, p.experts
    ));

    let shape = format!("{},T={tokens}", p.name);
    let incumbent = spec(&shape, incumbent_epi);
    let candidates: Vec<CandidateSpecV1> = candidate_epis(kind)
        .iter()
        .map(|e| spec(&shape, e))
        .collect();
    let domain_hash = canonical_sha256(&json!({
        "kernel": KERNEL,
        "arch": ARCH,
        "shape": shape,
        "incumbent": incumbent,
        "candidates": candidates,
    }));
    Ok(DomainV1 {
        schema: DOMAIN_SCHEMA.to_owned(),
        case_id: case.case_id.clone(),
        node_index: node,
        node_symbol: n.symbol.clone(),
        kernel: KERNEL.to_owned(),
        arch: ARCH.to_owned(),
        shape,
        incumbent,
        candidates,
        derivation,
        domain_hash,
    })
}

/// A minimal sealed A3B/FN case with one node, for the tune tests.
#[cfg(test)]
pub(crate) fn test_case(family: &str, symbol: &str, params: &[(&str, u64)]) -> WorkloadCaseV1 {
    test_case_grid(family, symbol, params, None)
}

/// [`test_case`] with the node's launch grid (a rocprof-derived dense node has a grid, no params).
#[cfg(test)]
pub(crate) fn test_case_grid(
    family: &str,
    symbol: &str,
    params: &[(&str, u64)],
    grid: Option<[u32; 3]>,
) -> WorkloadCaseV1 {
    let params: serde_json::Map<String, serde_json::Value> = params
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v)))
        .collect();
    let v = json!({
        "schema": "", "case_id": "",
        "model": {"name": "m", "family": family, "weight_format": null,
                  "file_sha256": null, "file_bytes": null, "sidecar_sha256": null},
        "prompt": null, "arch": "gfx1201", "device": null, "config": {},
        "phase": "prefill",
        "rows": {"tokens": 8192, "context": null, "chunk": null, "ragged": []},
        "mode": "eager", "placement": "resident",
        "chain": [{
            "symbol": symbol,
            "object": {"origin": "pm", "module": null, "artifact": null,
                       "artifact_sha256": null, "join": "test"},
            "abi": null, "params": params, "params_basis": "model_geometry",
            "grid": grid, "block": null, "dynamic_lds": null,
            "kernarg_sha256": null, "calls": 1, "time_ns": null
        }],
        "fixtures": [],
        "state": {"rollback": false, "prefix_commit": false, "graph_replay": false, "notes": []},
        "denominator": {"status": "unmeasured", "reason": "test"},
        "provenance": {"source_kind": "test", "source_path": "t", "source_sha256": "00",
                       "extracted_by": "test", "commit": null}
    });
    serde_json::from_value::<WorkloadCaseV1>(v)
        .expect("test case literal parses")
        .seal()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GU_NT8: &str = "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8";
    const DOWN_GP_NT4: &str = "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4";

    fn epis(d: &DomainV1) -> Vec<&str> {
        d.candidates
            .iter()
            .map(|c| c.knobs["epi"].as_str())
            .collect()
    }

    #[test]
    fn a3b_gate_up_nt8_domain() {
        let case = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        let d = enumerate(&case, 0, 8192).unwrap();
        assert_eq!(d.schema, DOMAIN_SCHEMA);
        assert_eq!(d.shape, "a3b,T=8192");
        assert_eq!(d.incumbent.knobs["epi"], "gate_up_nt8");
        assert_eq!(epis(&d), ["gate_up", "gate_up_nt4", "gate_up_nt8"]);
        assert!(d.candidates.contains(&d.incumbent));
        assert_eq!(d.node_symbol, GU_NT8);
        assert_eq!(d.case_id, case.case_id);
        assert_eq!(d.derivation.len(), 4);
    }

    // ---- iu4_gemm: dense 27B prefill gate/up silu (`gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`)

    fn iu4_case() -> WorkloadCaseV1 {
        test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], Some([272, 64, 1]))
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pm-census-domain-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_rips(dir: &Path, files: &[(&str, &str)]) -> Vec<PathBuf> {
        for (name, text) in files {
            std::fs::write(dir.join(name), text).unwrap();
        }
        rips_in_dir(dir).unwrap()
    }

    #[test]
    fn classify_knows_both_families() {
        assert_eq!(
            classify(IU4_GATE_UP_SILU_B1S),
            Some(Node::Iu4(Iu4Knobs {
                fold: "k128",
                tile: "128x128x8",
                cacc: "1",
                epi: "silu",
                alayout: "slab",
            }))
        );
        assert_eq!(
            classify(GU_NT8),
            Some(Node::Moe(Kind::GateUp, "gate_up_nt8"))
        );
        // The token-layout symbol and the other epilogues are not enumerable nodes.
        assert_eq!(classify("gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1"), None);
        assert_eq!(
            classify("gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s"),
            None
        );
    }

    #[test]
    fn iu4_domain_without_rips_is_control_and_tile() {
        let case = iu4_case();
        let d = enumerate_with_rips(&case, 0, 8192, &[]).unwrap();
        assert_eq!(d.schema, DOMAIN_SCHEMA);
        assert_eq!(d.kernel, "iu4_gemm");
        assert_eq!(d.arch, "gfx1201");
        assert_eq!(d.shape, "M=17408,N=8192,K=5120");
        assert_eq!(d.node_symbol, IU4_GATE_UP_SILU_B1S);
        assert_eq!(d.case_id, case.case_id);
        let k = &d.incumbent.knobs;
        assert_eq!(k.len(), 5);
        assert_eq!(
            [
                k["fold"].as_str(),
                k["tile"].as_str(),
                k["cacc"].as_str(),
                k["epi"].as_str(),
                k["alayout"].as_str()
            ],
            ["k128", "128x128x8", "1", "silu", "slab"]
        );
        assert!(!k.contains_key("rip"));
        assert_eq!(d.candidates.len(), 2);
        assert_eq!(d.candidates[0], d.incumbent);
        assert_eq!(d.candidates[1].knobs["tile"], "256x128x16");
        assert_eq!(d.candidates[1].knobs["alayout"], "slab");
        assert_eq!(d.derivation.len(), 3);
    }

    #[test]
    fn iu4_domain_lists_rip_candidates_in_name_order() {
        let dir = tmp_dir("rips");
        let rips = write_rips(
            &dir,
            &[
                ("b.rip", "fn emit() {}"),
                ("a.rip", "fn emit() { }"),
                ("notes.txt", "not a script"),
            ],
        );
        assert_eq!(rips.len(), 2);
        assert!(rips[0].ends_with("a.rip") && rips[1].ends_with("b.rip"));
        assert!(rips[0].is_absolute());
        let d = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        assert_eq!(d.candidates.len(), 4);
        let rip_of = |i: usize| d.candidates[i].knobs["rip"].clone();
        assert_eq!(rip_of(2), rips[0].to_str().unwrap());
        assert_eq!(rip_of(3), rips[1].to_str().unwrap());
        // The rip candidate keeps every incumbent knob and adds only `rip`.
        let mut k = d.candidates[2].knobs.clone();
        k.remove("rip");
        assert_eq!(k, d.incumbent.knobs);
        // `pm_loop` gets `--rip PATH` and the five knob flags.
        let args = d.candidates[2].pm_loop_args();
        let at = |flag: &str| {
            let i = args.iter().position(|a| a == flag).unwrap();
            args[i + 1].clone()
        };
        assert_eq!(at("--kernel"), "iu4_gemm");
        assert_eq!(at("--shape"), "M=17408,N=8192,K=5120");
        assert_eq!(at("--rip"), rips[0].to_str().unwrap());
        assert_eq!(at("--fold"), "k128");
        assert_eq!(at("--tile"), "128x128x8");
        assert_eq!(at("--cacc"), "1");
        assert_eq!(at("--epi"), "silu");
        assert_eq!(at("--alayout"), "slab");
        assert_eq!(
            d.derivation
                .iter()
                .filter(|l| l.starts_with("rip candidate "))
                .count(),
            2
        );
        // Distinct ids: the resume key tells the scripts apart.
        assert_ne!(d.candidates[2].id(), d.candidates[3].id());
        assert_ne!(d.candidates[0].id(), d.candidates[2].id());
    }

    #[test]
    fn iu4_domain_hash_tracks_script_content_and_shape() {
        let dir = tmp_dir("hash");
        let rips = write_rips(&dir, &[("a.rip", "one")]);
        let a = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        let again = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        assert_eq!(a, again);
        // Same path, edited content: the candidate spec is unchanged, the domain hash is not.
        write_rips(&dir, &[("a.rip", "two")]);
        let edited = enumerate_with_rips(&iu4_case(), 0, 8192, &rips).unwrap();
        assert_eq!(a.candidates, edited.candidates);
        assert_ne!(a.domain_hash, edited.domain_hash);
        // Another token count is another shape.
        let case_half = test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], Some([272, 32, 1]));
        let half = enumerate_with_rips(&case_half, 0, 4096, &rips).unwrap();
        assert_eq!(half.shape, "M=17408,N=4096,K=5120");
        assert_ne!(edited.domain_hash, half.domain_hash);
        // A partial last token tile still rounds up (N = 8191 -> 64 tiles).
        assert!(enumerate_with_rips(&iu4_case(), 0, 8191, &[]).is_ok());
    }

    #[test]
    fn iu4_refusals() {
        let g = |x: u32, y: u32| {
            test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], Some([x, y, 1]))
        };
        let e = enumerate_with_rips(&g(271, 64), 0, 8192, &[]).unwrap_err();
        assert!(
            e.contains("grid.x") && e.contains("271") && e.contains("272"),
            "{e}"
        );
        let e = enumerate_with_rips(&g(272, 32), 0, 8192, &[]).unwrap_err();
        assert!(
            e.contains("grid.y") && e.contains("32") && e.contains("64"),
            "{e}"
        );
        let none = test_case_grid("dense-27b", IU4_GATE_UP_SILU_B1S, &[], None);
        assert!(enumerate_with_rips(&none, 0, 8192, &[])
            .unwrap_err()
            .contains("no launch grid"));
        let fam = test_case_grid("a3b", IU4_GATE_UP_SILU_B1S, &[], Some([272, 64, 1]));
        assert!(enumerate_with_rips(&fam, 0, 8192, &[])
            .unwrap_err()
            .contains("preset"));
        let m = test_case_grid(
            "dense-27b",
            IU4_GATE_UP_SILU_B1S,
            &[("M", 4096)],
            Some([272, 64, 1]),
        );
        let e = enumerate_with_rips(&m, 0, 8192, &[]).unwrap_err();
        assert!(e.contains("4096") && e.contains("17408"), "{e}");
        assert!(enumerate_with_rips(&iu4_case(), 0, 0, &[])
            .unwrap_err()
            .contains("tokens"));
        let mut gfx = iu4_case();
        gfx.arch = "gfx1100".into();
        assert!(enumerate_with_rips(&gfx, 0, 8192, &[])
            .unwrap_err()
            .contains("gfx1100"));
        assert!(enumerate_with_rips(&iu4_case(), 1, 8192, &[])
            .unwrap_err()
            .contains("out of range"));
        let missing =
            enumerate_with_rips(&iu4_case(), 0, 8192, &[PathBuf::from("/nonexistent/x.rip")]);
        assert!(missing.unwrap_err().contains("/nonexistent/x.rip"));
    }

    #[test]
    fn rips_are_refused_for_moe_nodes() {
        let case = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        let e = enumerate_with_rips(&case, 0, 8192, &[PathBuf::from("/x/a.rip")]).unwrap_err();
        assert!(e.contains(".rip") && e.contains("iu4_gemm"), "{e}");
        // Without scripts the MoE path is unchanged.
        assert!(enumerate_with_rips(&case, 0, 8192, &[]).is_ok());
    }

    #[test]
    fn rips_in_dir_errors_on_missing_dir() {
        assert!(rips_in_dir(Path::new("/nonexistent/rip-dir")).is_err());
    }

    #[test]
    fn down_gp_nt4_has_five_candidates() {
        let case = test_case("a3b", DOWN_GP_NT4, &[("M", 2048), ("m_total", 69376)]);
        let d = enumerate(&case, 0, 8192).unwrap();
        assert_eq!(
            epis(&d),
            ["down", "down_nt4", "down_nt8", "down_gp_nt4", "down_gp_nt8"]
        );
        assert_eq!(d.incumbent.knobs["epi"], "down_gp_nt4");
    }

    #[test]
    fn flash_next_uses_fn_preset() {
        let case = test_case(
            "flash-next",
            "qwen4_moe_down_iu4_sym_pm_gfx1201",
            &[("M", 2560), ("m_total", 48640)],
        );
        let d = enumerate(&case, 0, 4096).unwrap();
        assert_eq!(d.shape, "fn,T=4096");
        assert_eq!(d.incumbent.knobs["epi"], "down");
    }

    #[test]
    fn refuses_mismatches() {
        let ok = |m: u64, mt: u64| test_case("a3b", GU_NT8, &[("M", m), ("m_total", mt)]);
        let e = enumerate(&ok(1024, 69377), 0, 8192).unwrap_err();
        assert!(e.contains("69377") && e.contains("69376"), "{e}");
        let e = enumerate(&ok(2048, 69376), 0, 8192).unwrap_err();
        assert!(e.contains("2048") && e.contains("1024"), "{e}");
        let e = enumerate(&ok(1024, 69376), 0, 4096).unwrap_err();
        assert!(e.contains("69376"), "{e}");
        let e = enumerate(&ok(1024, 69376), 3, 8192).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let unknown = test_case("a3b", "mystery_kernel", &[("M", 1024), ("m_total", 69376)]);
        assert!(enumerate(&unknown, 0, 8192).is_err());
        let fam = test_case("other", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        assert!(enumerate(&fam, 0, 8192).unwrap_err().contains("preset"));
        let mut gfx = ok(1024, 69376);
        gfx.arch = "gfx1100".into();
        assert!(enumerate(&gfx, 0, 8192).unwrap_err().contains("gfx1100"));
    }

    #[test]
    fn domain_hash_is_stable_and_shape_sensitive() {
        let case = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 69376)]);
        let a = enumerate(&case, 0, 8192).unwrap();
        let b = enumerate(&case, 0, 8192).unwrap();
        assert_eq!(a.domain_hash, b.domain_hash);
        assert_eq!(a, b);
        let case4 = test_case("a3b", GU_NT8, &[("M", 1024), ("m_total", 36608)]);
        let c = enumerate(&case4, 0, 4096).unwrap();
        assert_ne!(a.domain_hash, c.domain_hash);
    }
}
