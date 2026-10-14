// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-quantize trunk-overlay`: offline, no-GPU build of a
//! `<model>.trunk-overlay` sidecar from a base model and a directory of
//! pre-encoded QT44 payloads (`<tensor name>.bin`, the layout the developer
//! runtime map `HIPFIRE_QWEN4_REQUANT="...=qt44:<dir>"` reads).
//!
//! The container, binding and self-check are documented in
//! `hipfire_runtime::trunk_overlay`; this is only the command-line wrapper.

use clap::Parser;
use hipfire_runtime::trunk_overlay::write_trunk_overlay;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "hipfire-quantize trunk-overlay",
    about = "Build a `<model>.trunk-overlay` sidecar of pre-encoded QT44 tensors for a Qwen4 \
             base model",
    long_about = "Write the trunk overlay sidecar that substitutes pre-encoded QT44 tensors for \
                  the base model's tensors at load, and arms the named trunk A4 default.\n\n\
                  Base tensors whose names end with any --suffix (base index order) are \
                  replaced; each needs <qt44-payload-dir>/<tensor name>.bin of exactly the QT44 \
                  size. The sidecar binds to the base's file length and index digest, records \
                  the base's full sha256, and is re-bound and re-read as a self-check. The \
                  report is printed as JSON on stdout."
)]
pub(crate) struct TrunkOverlayArgs {
    /// Base HFQM model file the overlay binds to.
    #[arg(long, value_name = "BASE.hfq")]
    pub base: PathBuf,

    /// Directory of `<tensor name>.bin` QT44 payloads.
    #[arg(long, value_name = "DIR")]
    pub qt44_payload_dir: PathBuf,

    /// Tensor name suffix to replace (repeatable, at least one).
    #[arg(long, value_name = "SUFFIX", required = true)]
    pub suffix: Vec<String>,

    /// Trunk A4 default marker the overlay arms (e.g. `gfx1151-zab-v1`).
    #[arg(long, value_name = "MARKER")]
    pub trunk_iu4_default: String,

    /// Destination sidecar (an existing file is refused).
    #[arg(long, value_name = "FILE.trunk-overlay")]
    pub output: PathBuf,
}

/// Entry point for `hipfire-quantize trunk-overlay …` (`argv` excludes the
/// program name and the subcommand word). Returns the process exit code.
pub(crate) fn run(argv: Vec<String>) -> i32 {
    let args = TrunkOverlayArgs::parse_from(
        std::iter::once("hipfire-quantize trunk-overlay".to_string()).chain(argv),
    );
    match run_inner(&args) {
        Ok(report) => {
            println!("{report}");
            0
        }
        Err(e) => {
            eprintln!("error: trunk-overlay: {e}");
            1
        }
    }
}

fn run_inner(args: &TrunkOverlayArgs) -> Result<String, String> {
    let report = write_trunk_overlay(
        &args.base,
        &args.qt44_payload_dir,
        &args.suffix,
        &args.trunk_iu4_default,
        &args.output,
    )?;
    serde_json::to_string_pretty(&report).map_err(|e| format!("serialize report: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repeatable_suffixes() {
        let args = TrunkOverlayArgs::try_parse_from([
            "trunk-overlay",
            "--base",
            "m.mq4-xts",
            "--qt44-payload-dir",
            "d",
            "--suffix",
            ".in_proj_z.weight",
            "--suffix",
            ".in_proj_a.weight",
            "--trunk-iu4-default",
            "gfx1151-zab-v1",
            "--output",
            "m.mq4-xts.trunk-overlay",
        ])
        .unwrap();
        assert_eq!(args.suffix, [".in_proj_z.weight", ".in_proj_a.weight"]);
        assert_eq!(args.trunk_iu4_default, "gfx1151-zab-v1");
    }

    #[test]
    fn requires_a_suffix() {
        assert!(TrunkOverlayArgs::try_parse_from([
            "trunk-overlay",
            "--base",
            "m",
            "--qt44-payload-dir",
            "d",
            "--trunk-iu4-default",
            "x",
            "--output",
            "o",
        ])
        .is_err());
    }
}
