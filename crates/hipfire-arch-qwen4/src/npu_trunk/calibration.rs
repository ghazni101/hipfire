// SPDX-License-Identifier: Apache-2.0
//! Stored measurement inputs. Loading never measures or recalibrates a device.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

pub const CALIBRATION_SCHEMA_VERSION: u32 = 4;
/// Change the contract suffix when serving numerics/transport change.
pub fn calibration_build_identity() -> String {
    format!("hipfire-{}:npu-trunk-bf16-rowmajor-v1", env!("CARGO_PKG_VERSION"))
}
pub(crate) const FAMILY_NAMES: [&str; 4] = ["qkv", "zab", "q", "kvidx"];
pub(crate) const OUT_FAMILY_NAMES: [&str; 2] = ["gdn.out", "qsa.o"];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceKey {
    pub arch: String,
    pub pci_bdf: String,
    pub npu_fw_version: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelKey {
    pub uname_release: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalKey {
    pub device: DeviceKey,
    /// Stable release + NPU numerical/transport contract, never a git SHA.
    #[serde(default = "calibration_build_identity")]
    pub build: String,
    pub kernel: KernelKey,
    /// SHA256 of HFQ bytes [0, data_offset), as in trunk-overlay binding.
    pub model_index_sha256: String,
    /// Always `bf16` (BF16-direct), matching NpuCalibration::epilogue.
    pub epilogue: String,
    /// GPU-native trunk identity: `stock` or `overlay:<artifact marker>`.
    pub trunk_route: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuCalibration {
    pub trunk_in_ms_per_8192: BTreeMap<String, f64>,
    pub uniform_per_layer_ms: f64,
    pub gdn_core_per_layer_ms: f64,
    pub qsa_per_layer_ms: f64,
    pub qsa_growth_per_layer_ms_per_8192_prefix: f64,
    pub fixed_per_chunk_ms: f64,
    pub boundary_tax_ms: f64,
    pub min_bank_rows: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyCalibration {
    pub layer: String,
    pub cols: usize,
    pub out_dtype: String,
    /// `false` keeps the family out of `select_families` until it is measured.
    #[serde(default = "selectable_default", skip_serializing_if = "is_true")]
    pub selectable: bool,
    /// Out-projection families only (`gdn.out`, `qsa.o`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<OutFamily>,
}
/// Out-projection shape (N = `cols`, K = `k`, `panel_cols` wide panels), its simulator rate alone, its NPU DRAM bytes
/// per kilo-op and the GPU o-proj GEMM time it removes from the layer per 8192 rows.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutFamily { pub k: usize, pub panel_cols: usize, pub tops_alone: f64, pub bytes_per_kop: f64, pub gpu_ms_per_8192: f64 }
fn selectable_default() -> bool { true }
fn is_true(v: &bool) -> bool { *v }
fn is_zero(v: &f64) -> bool { *v == 0.0 }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlueCalibration {
    pub pack_ms_8192: f64,
    /// GPU overhead per 8192 rows of a bank visit that moves families (events, split prepare/pack, wake).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub visit_ms_8192: f64,
    pub sync_ms: f64,
    pub consume_mode: String,
    pub consume_ms_per_col_8192: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NpuCalibration {
    pub tops_corun: f64,
    /// In-proj alone rate the co-run rate is measured against; the out families scale their alone rate by
    /// `tops_corun / tops_alone`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tops_alone: Option<f64>,
    pub bytes_per_kop: f64,
    pub epilogue: String,
    pub panel_cols: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentionCalibration {
    pub alpha_overlap: f64,
    /// Contention per NPU DRAM GB of the overlapped family chains (0 = burst model only).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ms_per_gb: f64,
    pub model: String,
}
/// Measured per-8192-row QSA phase times at the 16K pair (bank A prefix 0, bank B prefix 8192).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QsaOrder { pub core_a_ms: f64, pub core_b_ms: f64, pub post_ms: f64, pub inproj_chain_ms: f64 }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub schema_version: u32,
    pub key: CalKey,
    pub gpu: GpuCalibration,
    pub families: BTreeMap<String, FamilyCalibration>,
    pub glue: GlueCalibration,
    pub npu: NpuCalibration,
    pub contention: ContentionCalibration,
    pub margin: f64,
    /// Measured QSA phase windows for the order `C_A C_B P_A P_B` that moving `qsa.o` forces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qsa_order: Option<QsaOrder>,
}
impl Calibration {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let cal: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        cal.validate()?;
        Ok(cal)
    }
    /// Only a missing file falls back; invalid/unreadable files are authoritative.
    pub fn load_with_builtin(path: impl AsRef<Path>, route: &str) -> Result<(Self, &'static str), String> {
        let (bytes, source) = match fs::read(path) {
            Ok(bytes) => (std::borrow::Cow::Owned(bytes), "file"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let json = match route {
                    "stock" => include_str!("calibrations/halo-xdna2-stock-v2.json"),
                    _ => include_str!("calibrations/halo-xdna2-overlay-v4.json"),
                };
                (std::borrow::Cow::Borrowed(json.as_bytes()), "builtin")
            }
            Err(e) => return Err(e.to_string()),
        };
        let cal: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        cal.validate()?;
        Ok((cal, source))
    }
    pub fn store(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.validate()?;
        let json = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())
    }
    /// First mismatch in stable device/build/kernel/model/epilogue/trunk-route order. Caller
    /// must leave the route off and log this reason; never silently recalibrate.
    pub fn key_matches(&self, key: &CalKey) -> Result<(), String> {
        macro_rules! check { ($($field:ident).+) => {
            if self.key.$($field).+ != key.$($field).+ {
                return Err(format!("calibration key mismatch: {}", stringify!($($field).+)));
            }
        }; }
        check!(device.arch); check!(device.pci_bdf); check!(device.npu_fw_version);
        check!(build); check!(kernel.uname_release);
        check!(model_index_sha256); check!(epilogue); check!(trunk_route);
        Ok(())
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION { return Err("unsupported calibration schema_version".into()); }
        if !self.margin.is_finite() || !(0.0..1.0).contains(&self.margin) { return Err("margin must be finite and in [0,1)".into()); }
        if self.npu.panel_cols == 0 || self.gpu.min_bank_rows == 0 { return Err("panel_cols and min_bank_rows must be positive".into()); }
        if self.npu.epilogue != "bf16" { return Err("npu.epilogue must be bf16".into()); }
        if self.key.epilogue != self.npu.epilogue { return Err("key.epilogue differs from npu.epilogue".into()); }
        if self.key.trunk_route != "stock" && !self.key.trunk_route.strip_prefix("overlay:").is_some_and(|marker| !marker.trim().is_empty()) {
            return Err("key.trunk_route must be stock or overlay:<marker>".into());
        }
        if self.glue.consume_mode != "retile" && self.glue.consume_mode != "direct" {
            return Err("glue.consume_mode must be retile or direct".into());
        }
        if self.glue.consume_ms_per_col_8192.len() != 2 {
            return Err("expected bf16,f32 consume_ms_per_col_8192".into());
        }
        for dtype in ["bf16", "f32"] {
            let cost = self.glue.consume_ms_per_col_8192.get(dtype).ok_or_else(|| format!("missing glue.consume_ms_per_col_8192.{dtype}"))?;
            if !cost.is_finite() || *cost < 0.0 { return Err(format!("invalid glue.consume_ms_per_col_8192.{dtype}")); }
        }
        for (name, value) in [
            ("device.arch", self.key.device.arch.as_str()), ("device.pci_bdf", &self.key.device.pci_bdf),
            ("device.npu_fw_version", &self.key.device.npu_fw_version), ("build", &self.key.build),
            ("kernel.uname_release", &self.key.kernel.uname_release), ("contention.model", &self.contention.model),
        ] { if value.trim().is_empty() { return Err(format!("{name} must not be empty")); } }
        if self.key.model_index_sha256.len() != 64 || !self.key.model_index_sha256.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("model_index_sha256 must be 64 hex digits".into()); }
        if self.gpu.trunk_in_ms_per_8192.len() != 4 { return Err("expected qkv,zab,q,kvidx calibration families".into()); }
        for name in self.families.keys() {
            if !FAMILY_NAMES.contains(&name.as_str()) && !OUT_FAMILY_NAMES.contains(&name.as_str()) {
                return Err(format!("unknown families.{name}"));
            }
        }
        for (i, name) in FAMILY_NAMES.iter().enumerate() {
            let f = self.families.get(*name).ok_or_else(|| format!("missing families.{name}"))?;
            if f.cols == 0 || f.layer != if i < 2 { "gdn" } else { "qsa" } { return Err(format!("invalid families.{name}")); }
            if f.out_dtype != "bf16" && f.out_dtype != "f32" { return Err(format!("invalid families.{name}.out_dtype")); }
            let cost = self.gpu.trunk_in_ms_per_8192.get(*name).ok_or_else(|| format!("missing gpu.trunk_in_ms_per_8192.{name}"))?;
            if !cost.is_finite() || *cost < 0.0 { return Err(format!("invalid gpu.trunk_in_ms_per_8192.{name}")); }
        }
        for (name, value) in [
            ("uniform_per_layer_ms", self.gpu.uniform_per_layer_ms), ("gdn_core_per_layer_ms", self.gpu.gdn_core_per_layer_ms),
            ("qsa_per_layer_ms", self.gpu.qsa_per_layer_ms), ("qsa_growth_per_layer_ms_per_8192_prefix", self.gpu.qsa_growth_per_layer_ms_per_8192_prefix),
            ("fixed_per_chunk_ms", self.gpu.fixed_per_chunk_ms), ("boundary_tax_ms", self.gpu.boundary_tax_ms),
            ("pack_ms_8192", self.glue.pack_ms_8192), ("visit_ms_8192", self.glue.visit_ms_8192),
            ("sync_ms", self.glue.sync_ms), ("bytes_per_kop", self.npu.bytes_per_kop),
        ] { if !value.is_finite() || value < 0.0 { return Err(format!("{name} must be finite and nonnegative")); } }
        if !self.npu.tops_corun.is_finite() || self.npu.tops_corun <= 0.0 { return Err("tops_corun must be finite and positive".into()); }
        if !self.contention.alpha_overlap.is_finite() || self.contention.alpha_overlap < 1.0 { return Err("alpha_overlap must be finite and >= 1".into()); }
        if !self.contention.ms_per_gb.is_finite() || self.contention.ms_per_gb < 0.0 { return Err("ms_per_gb must be finite and nonnegative".into()); }
        self.validate_out_families()?;
        Ok(())
    }
    /// The optional out-projection families (`gdn.out`, `qsa.o`): their own shape, rate and GPU o-proj cost, the
    /// in-proj alone rate their co-run rate is scaled by, and (for `qsa.o`) the measured QSA phase windows.
    fn validate_out_families(&self) -> Result<(), String> {
        for (name, layer) in OUT_FAMILY_NAMES.into_iter().zip(["gdn", "qsa"]) {
            let Some(f) = self.families.get(name) else { continue };
            let Some(o) = &f.out else { return Err(format!("families.{name} needs an out block")); };
            if f.cols == 0 || f.layer != layer || f.out_dtype != "bf16" { return Err(format!("invalid families.{name}")); }
            if o.k == 0 || o.panel_cols == 0 { return Err(format!("invalid families.{name}.out")); }
            for (field, value) in [("tops_alone", o.tops_alone), ("bytes_per_kop", o.bytes_per_kop), ("gpu_ms_per_8192", o.gpu_ms_per_8192)] {
                if !value.is_finite() || value < 0.0 || (field == "tops_alone" && value == 0.0) {
                    return Err(format!("invalid families.{name}.out.{field}"));
                }
            }
            if !self.npu.tops_alone.is_some_and(|t| t.is_finite() && t > 0.0) { return Err("out families need a positive npu.tops_alone".into()); }
        }
        for name in FAMILY_NAMES {
            if self.families.get(name).is_some_and(|f| f.out.is_some()) { return Err(format!("families.{name} must not carry an out block")); }
        }
        if self.families.contains_key("qsa.o") {
            let q = self.qsa_order.as_ref().ok_or("qsa.o needs qsa_order")?;
            for (field, value) in [("core_a_ms", q.core_a_ms), ("core_b_ms", q.core_b_ms), ("post_ms", q.post_ms), ("inproj_chain_ms", q.inproj_chain_ms)] {
                if !value.is_finite() || value < 0.0 { return Err(format!("qsa_order.{field} must be finite and nonnegative")); }
            }
        } else if self.qsa_order.is_some() {
            return Err("qsa_order without families.qsa.o".into());
        }
        Ok(())
    }
}

/// Pure construction shared by runtime and CPU identity tests.
pub fn calibration_key(arch: &str, pci_bdf: &str, firmware: &str, release: &str, index: &str, trunk_route: &str) -> CalKey {
    CalKey {
        device: DeviceKey { arch: arch.into(), pci_bdf: pci_bdf.to_ascii_lowercase(), npu_fw_version: firmware.trim().into() },
        build: calibration_build_identity(),
        kernel: KernelKey { uname_release: release.trim().into() },
        model_index_sha256: index.into(), epilogue: "bf16".into(), trunk_route: trunk_route.into(),
    }
}

/// Probe the exact accel node railgun opens, without opening the runtime.
#[cfg(unix)]
pub(crate) fn npu_firmware() -> Result<String, String> {
    use std::os::unix::fs::FileTypeExt;
    let node = fs::metadata("/dev/accel/accel0").map_err(|e| format!("NPU accel0 unavailable: {e}"))?;
    if !node.file_type().is_char_device() { return Err("NPU accel0 is not an accel device".into()); }
    let device = Path::new("/sys/class/accel/accel0/device");
    let driver = fs::read_link(device.join("driver")).map_err(|e| format!("NPU driver: {e}"))?;
    if driver.file_name().is_none_or(|name| name != "amdxdna") { return Err("NPU accel0 is not amdxdna".into()); }
    let pci = fs::canonicalize(device).map_err(|e| format!("NPU PCI device: {e}"))?;
    if pci.file_name().is_none_or(|name| !name.to_string_lossy().contains(':')) {
        return Err("NPU accel0 PCI identity unavailable".into());
    }
    let firmware = fs::read_to_string(device.join("fw_version")).map_err(|e| format!("NPU fw_version: {e}"))?;
    if firmware.trim().is_empty() { return Err("NPU fw_version is empty".into()); }
    Ok(firmware)
}

#[cfg(not(unix))]
pub(crate) fn npu_firmware() -> Result<String, String> {
    Err("NPU accel0 unavailable on this platform".into())
}

/// Read only unprivileged metadata, never open a GPU/NPU or time work.
pub fn runtime_calibration_key(arch: &str, pci_bdf: &str, index: &str, trunk_route: &str) -> Result<CalKey, String> {
    // railgun opens accel0; key that exact device, not the first PCI directory.
    let firmware = npu_firmware()?;
    runtime_key_with_firmware(arch, pci_bdf, &firmware, index, trunk_route)
}

pub(crate) fn runtime_key_with_firmware(arch: &str, pci_bdf: &str, firmware: &str, index: &str, trunk_route: &str) -> Result<CalKey, String> {
    let release = std::process::Command::new("uname").arg("-r").output().map_err(|e| format!("uname -r: {e}"))?;
    if !release.status.success() { return Err("uname -r failed".into()); }
    let release = std::str::from_utf8(&release.stdout).map_err(|e| e.to_string())?;
    // PerfOpt's armed message is privileged dmesg, with no sysfs/proc ABI.
    // Schema 3 deliberately keys kernel identity on uname_release alone.
    Ok(calibration_key(arch, pci_bdf, firmware, release, index, trunk_route))
}

/// Capture the base before it moves into the source owner. Off performs no I/O.
pub fn model_index_identity(hfq: &hipfire_runtime::hfq::HfqFile, arch: &str) -> Option<Result<String, String>> {
    if super::NpuTrunkMode::from_env().ok().is_none_or(|mode| mode.resolve(arch).is_off()) { return None; }
    Some(hfq.index_sha256().map(|hash| {
        use std::fmt::Write;
        let mut hex = String::with_capacity(64);
        for byte in hash { write!(&mut hex, "{byte:02x}").expect("write to String"); }
        hex
    }).map_err(|e| format!("model index identity: {e}")))
}

#[cfg(test)]
mod builtin_tests {
    use super::*;

    #[test]
    fn measured_builtin_keys_validate_for_this_build() {
        for (json, route) in [
            (include_str!("calibrations/halo-xdna2-overlay-v4.json"), "overlay:gfx1151-zab-v1"),
            (include_str!("calibrations/halo-xdna2-stock-v2.json"), "stock"),
        ] {
            let cal: Calibration = serde_json::from_str(json).unwrap();
            cal.validate().unwrap();
            assert_eq!(cal.key.build, calibration_build_identity());
            assert_eq!(cal.key.trunk_route, route);
            let key = calibration_key("gfx1151", "0000:c0:00.0", "1.1.2.65",
                "7.3.0-rc7-perfopt+", "8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013", route);
            cal.key_matches(&key).unwrap();
            assert!(!super::super::auto_admission(&cal, &key).unwrap().is_empty());
            let mut mismatched = key;
            mismatched.device.npu_fw_version.push('x');
            assert_eq!(cal.key_matches(&mismatched), Err("calibration key mismatch: device.npu_fw_version".into()));
        }
    }

    #[test]
    fn file_precedes_builtin_and_invalid_file_never_falls_back() {
        let dir = std::env::temp_dir().join(format!("hipfire-builtin-cal-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("calibration.json");
        let (mut builtin, source) = Calibration::load_with_builtin(&path, "stock").unwrap();
        assert_eq!(source, "builtin");
        assert_eq!(builtin.key.trunk_route, "stock");
        builtin.key.device.npu_fw_version = "custom-file".into();
        builtin.store(&path).unwrap();
        let (file, source) = Calibration::load_with_builtin(&path, "stock").unwrap();
        assert_eq!(source, "file");
        assert_eq!(file.key.device.npu_fw_version, "custom-file");
        fs::write(&path, b"{}").unwrap();
        assert!(Calibration::load_with_builtin(&path, "stock").is_err());
        fs::remove_file(&path).unwrap();
        let (overlay, source) = Calibration::load_with_builtin(&path, "overlay:gfx1151-zab-v1").unwrap();
        assert_eq!(source, "builtin");
        assert_eq!(overlay.key.trunk_route, "overlay:gfx1151-zab-v1");
        fs::remove_dir(dir).unwrap();
    }
}
