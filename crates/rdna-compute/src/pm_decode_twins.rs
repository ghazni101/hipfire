// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Oracle-accepted builder (peacemaker) twins of gfx1201 decode HIP modules.
//!
//! On exact `gfx1201` with `kernel.pm_decode` (default on; `HIPFIRE_PM_DECODE=0` opts out), the
//! common first-load funnels (`scratch::load_kernel_module`, reached by
//! `Gpu::ensure_kernel`, the scratch helpers and the planned-route preload)
//! load the embedded image of a listed module instead of compiling its HIP
//! source. Launchers are unchanged, so ordinary, graph-captured, lowered and
//! retained callers all bind the same function.
//!
//! Each entry pins the whole embedded bundle's SHA-256 (the module's
//! [`crate::code_object::CodeObjectId`]) and the SHA-256 of the incumbent HIP
//! source it replaces. Admission verifies both once and fails closed on any
//! mismatch: a re-emitted image or an edited source never loads under this
//! flag until it is re-accepted here. Only twins that passed the
//! full-symbol whole-buffer oracle (zero mismatch bytes) are listed; the
//! accepted gfx1201 ELF digest is recorded alongside for provenance.
//!
//! Re-accepting a new `source_sha256` after editing one of these HIP sources
//! takes a fresh whole-buffer oracle, even for a change that does not touch
//! the arithmetic. Under `-ffp-contract=fast` the backend's FMA fusion of the
//! group dot depends on the surrounding code, so an edit elsewhere in the
//! kernel can change its output bytes while the twin keeps the old ones (see
//! `hipfire_isa::kernels::pm_decode`).

use sha2::{Digest, Sha256};

/// One accepted twin: the module name the launcher passes, its embedded
/// bundle, and the frozen digests admission checks.
#[derive(Debug)]
pub struct NativeKernelBundle {
    pub module: &'static str,
    /// SHA-256 of the incumbent HIP source this image replaces.
    pub source_sha256: [u8; 32],
    /// SHA-256 of [`Self::image`] (the whole clang offload bundle).
    pub image_sha256: [u8; 32],
    /// SHA-256 of the gfx1201 ELF inside the bundle that the oracle accepted.
    pub accepted_elf_sha256: [u8; 32],
    pub image: &'static [u8],
    pub radiowave_json: &'static str,
    pub symbols: &'static [&'static str],
}

const fn hex32(hex: &str) -> [u8; 32] {
    let bytes = hex.as_bytes();
    assert!(bytes.len() == 64, "sha256 hex must be 64 digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (nibble(bytes[2 * i]) << 4) | nibble(bytes[2 * i + 1]);
        i += 1;
    }
    out
}

const fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("sha256 hex must be lowercase"),
    }
}

macro_rules! twin {
    ($module:literal, $file:literal, $symbols:expr, source = $src:literal, image = $img:literal, elf = $elf:literal) => {
        NativeKernelBundle {
            module: $module,
            source_sha256: hex32($src),
            image_sha256: hex32($img),
            accepted_elf_sha256: hex32($elf),
            image: include_bytes!(concat!("../../../kernels/pm-decode/gfx1201/", $file, ".hxaco")),
            radiowave_json: include_str!(concat!(
                "../../../kernels/pm-decode/gfx1201/",
                $file,
                ".radiowave.json"
            )),
            symbols: $symbols,
        }
    };
}

/// Accepted exact-gfx1201 twins. Oracle reports:
/// `release-0.4.2/pm-decode-twins/oracle-results/<module>/full-*/report.json`.
pub static GFX1201_TWINS: &[NativeKernelBundle] = &[
    twin!(
        "fused_qkvza_hfq4g256_mq4v2",
        "fused_qkvza_hfq4g256_mq4v2",
        &["fused_qkvza_mq4g256v2"],
        source = "24946fc118db455e19f96b0e02fd67bb6bfaf46ed4a917e23352bef6153e1ba5",
        image = "ded83395c4dc4fe78750d1669d029f725b9b69b53216ac8d25c9de5b9e79ac7d",
        elf = "31ac049859d799558b6f9e44176f94a68c6bfbcd3145ac23bf127b7ab4f36f38"
    ),
    twin!(
        "fused_qkv_hfq4g256_mq4v2",
        "fused_qkv_hfq4g256_mq4v2",
        &["fused_qkv_mq4g256v2"],
        source = "a0e7f85ec0a35eb777c07975cd888eb53aa21c855ae39d1c06ee02322836cd6b",
        image = "9e4f53092542077b6b97d4b462cfb5be0f58f109b4deba65d3a1892c8402420b",
        elf = "22c665a2548ad0a13d3b2a77212914d8ca42c3af4872dc9c394a3b0a70dbc4cb"
    ),
    twin!(
        "fused_gate_up_hfq4g256_mq4v2",
        "fused_gate_up_hfq4g256_mq4v2",
        &["fused_gate_up_mq4g256v2"],
        source = "f61a08770f8cfdfe2be45f85ef5002d4759c1f18d815e243dc6735d2e6215930",
        image = "23673ee66a0b3dc36b050b21282e4847cca468348868a7619e2ee09f7f193635",
        elf = "1e8ca901ec0f68269c92890b5eb69b3031db346530ea480aa7ffc35c70ede5c8"
    ),
    twin!(
        "gemv_hfq4g256_multirow_default_mq4v2",
        "gemv_hfq4g256_multirow_default_mq4v2",
        &["gemv_mq4g256v2_multirow_r2"],
        source = "64b1b7eededa4bce636b8799247d7a66e7ef045f32d1674d07180bd56b98c3af",
        image = "0a63776ee63fe3321043bec3fd13817fc592c2f62fc0cb28d84111140d992dfe",
        elf = "c4732e91f06e38ca9d30053e09e9645b3ff2c6be9faa798bfdbb0074186a22c6"
    ),
    twin!(
        "gemv_hfq4g256_residual_mq4v2",
        "gemv_hfq4g256_residual_mq4v2",
        &["gemv_mq4g256v2_residual"],
        source = "869f4a4bdd33adc3e5a3aa9a6faa7584b53d50b0930f1d9e737e19a77975dc0e",
        image = "8e2cb8e6bbefeca5d996d274e06e37532140cf42ca65606a565404f6e3790757",
        elf = "bca8ec25b5fdf77147df2ac7a0b925d688fc550ce50090ba0bd8d20616585c7d"
    ),
];

/// The accepted twin for `module`/`symbol`, if the flag is on, `arch` is
/// exactly `gfx1201` and `symbol` is one of its accepted exports.
pub fn select_pm_decode_twin(
    arch: &str,
    enabled: bool,
    module: &str,
    symbol: &str,
) -> Option<&'static NativeKernelBundle> {
    pm_decode_module(arch, enabled, module).filter(|twin| twin.symbols.contains(&symbol))
}

/// The accepted twin replacing HIP module `module` on `arch`, regardless of
/// the requested symbol (admission then refuses an unaccepted export).
pub fn pm_decode_module(
    arch: &str,
    enabled: bool,
    module: &str,
) -> Option<&'static NativeKernelBundle> {
    if !enabled || arch != "gfx1201" {
        return None;
    }
    GFX1201_TWINS.iter().find(|twin| twin.module == module)
}

impl NativeKernelBundle {
    /// Admission check, once per module load: the embedded image is the
    /// accepted one, the caller's HIP source is the one the oracle compared
    /// against, and the symbol is an accepted export. Fails closed.
    pub fn verify(&self, source: &str, symbol: &str) -> Result<(), String> {
        self.verify_image()?;
        let source_sha: [u8; 32] = Sha256::digest(source.as_bytes()).into();
        if source_sha != self.source_sha256 {
            return Err(format!(
                "pm_decode twin {}: incumbent source sha256 {} is not the accepted {}",
                self.module,
                hex(&source_sha),
                hex(&self.source_sha256)
            ));
        }
        if !self.symbols.contains(&symbol) {
            return Err(format!(
                "pm_decode twin {}: symbol {symbol:?} is not an accepted export {:?}",
                self.module, self.symbols
            ));
        }
        Ok(())
    }

    /// The embedded bundle still has its pinned digest.
    pub fn verify_image(&self) -> Result<(), String> {
        let image_sha: [u8; 32] = Sha256::digest(self.image).into();
        if image_sha != self.image_sha256 {
            return Err(format!(
                "pm_decode twin {}: embedded image sha256 {} is not the accepted {}",
                self.module,
                hex(&image_sha),
                hex(&self.image_sha256)
            ));
        }
        Ok(())
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gfx1201 code object inside a clang offload bundle (the bytes HIP
    /// loads), independent of any loose `.co` file next to it.
    fn gfx1201_payload(bundle: &[u8]) -> &[u8] {
        let u64_at = |at: usize| u64::from_le_bytes(bundle[at..at + 8].try_into().unwrap()) as usize;
        assert_eq!(&bundle[..24], b"__CLANG_OFFLOAD_BUNDLE__");
        let mut cursor = 32;
        let mut found = None;
        for _ in 0..u64_at(24) {
            let (offset, size, len) = (u64_at(cursor), u64_at(cursor + 8), u64_at(cursor + 16));
            let triple = &bundle[cursor + 24..cursor + 24 + len];
            cursor += 24 + len;
            if triple == b"hipv4-amdgcn-amd-amdhsa--gfx1201" {
                assert!(found.replace(&bundle[offset..offset + size]).is_none(), "two gfx1201 entries");
            }
        }
        found.expect("bundle has no gfx1201 entry")
    }

    #[test]
    fn every_twin_is_pinned_bundle_of_accepted_elf_with_valid_certificate() {
        assert_eq!(GFX1201_TWINS.len(), 5);
        for twin in GFX1201_TWINS {
            twin.verify_image().unwrap();
            let elf = gfx1201_payload(twin.image);
            let elf_sha: [u8; 32] = Sha256::digest(elf).into();
            assert_eq!(elf_sha, twin.accepted_elf_sha256, "{}: bundle does not carry the accepted ELF", twin.module);
            radiowave::CodeObjectCertification::from_json(twin.image, twin.radiowave_json)
                .unwrap_or_else(|e| panic!("{}: {e}", twin.module));
            for &symbol in twin.symbols {
                assert!(
                    twin.image.windows(symbol.len()).any(|w| w == symbol.as_bytes()),
                    "{}: export {symbol} missing",
                    twin.module
                );
            }
        }
    }

    #[test]
    fn selection_is_exact_gfx1201_and_flag_gated() {
        let m = "fused_qkvza_hfq4g256_mq4v2";
        let s = "fused_qkvza_mq4g256v2";
        assert!(select_pm_decode_twin("gfx1201", true, m, s).is_some());
        assert!(select_pm_decode_twin("gfx1201", false, m, s).is_none());
        for arch in ["gfx1200", "gfx12", "gfx1151", "gfx1100", "gfx942", "gfx1201:xnack-"] {
            assert!(select_pm_decode_twin(arch, true, m, s).is_none(), "{arch}");
        }
        assert!(select_pm_decode_twin("gfx1201", true, m, "other").is_none());
        assert!(pm_decode_module("gfx1201", true, "gemv_mq4g256v2_mq4v2").is_none());
    }

    #[test]
    fn digest_mismatch_is_refused() {
        let twin = &GFX1201_TWINS[0];
        let tampered = NativeKernelBundle {
            module: twin.module,
            source_sha256: twin.source_sha256,
            image_sha256: twin.image_sha256,
            accepted_elf_sha256: twin.accepted_elf_sha256,
            image: GFX1201_TWINS[1].image,
            radiowave_json: twin.radiowave_json,
            symbols: twin.symbols,
        };
        let err = tampered.verify_image().unwrap_err();
        assert!(err.contains("embedded image sha256"), "{err}");
        let err = twin.verify("// not the incumbent", twin.symbols[0]).unwrap_err();
        assert!(err.contains("incumbent source sha256"), "{err}");
        let err = tampered.verify("// not the incumbent", twin.symbols[0]).unwrap_err();
        assert!(err.contains("embedded image sha256"), "{err}");
    }
}
