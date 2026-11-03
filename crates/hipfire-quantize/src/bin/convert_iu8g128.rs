// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline qt59 converter: legacy `[K i8][K/32 f16]` rows -> compact
//! `[K i8][K/128 f16]` rows (one scale per 128). Non-qt59 tensors are copied
//! byte for byte. Usage:
//!
//!   convert_iu8g128 <old.iu4p> <new.iu4p>

use half::f16;
use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

struct Tensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn read_hfq(path: &Path) -> (u32, String, Vec<Tensor>) {
    let file = File::open(path).expect("open input");
    let mmap = unsafe { Mmap::map(&file).expect("mmap input") };
    assert_eq!(&mmap[0..4], HFQ_MAGIC, "bad magic");
    let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
    assert_eq!(version, HFQ_VERSION, "unsupported HFQ version {version}");
    let arch = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
    let n = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
    let meta_off = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
    let data_off = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
    let mut pos = meta_off;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let meta_end;
    loop {
        let b = mmap[pos];
        let c = b as char;
        if esc {
            esc = false;
        } else if c == '\\' && in_str {
            esc = true;
        } else if c == '"' {
            in_str = !in_str;
        } else if !in_str {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    meta_end = pos + 1;
                    break;
                }
            }
        }
        pos += 1;
    }
    let metadata = String::from_utf8_lossy(&mmap[meta_off..meta_end]).into_owned();
    let mut pos = meta_end;
    let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
    assert_eq!(idx_n, n, "tensor count mismatch");
    pos += 4;
    let mut out = Vec::with_capacity(n);
    let mut cumulative = data_off;
    for _ in 0..n {
        let l = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&mmap[pos..pos + l]).into_owned();
        pos += l;
        let qt = mmap[pos];
        pos += 1;
        let nd = mmap[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd {
            shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let data = mmap[cumulative..cumulative + data_size].to_vec();
        cumulative += data_size;
        out.push(Tensor {
            name,
            quant_type: qt,
            shape,
            group_size,
            data,
        });
    }
    (arch, metadata, out)
}

fn write_hfq(path: &Path, arch: u32, metadata: &str, tensors: &[Tensor]) {
    let mut f = File::create(path).expect("create output");
    let meta = metadata.as_bytes();
    let metadata_offset = 32u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let nb = t.name.as_bytes();
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(t.quant_type);
        index.push(t.shape.len() as u8);
        for &d in &t.shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&t.group_size.to_le_bytes());
        index.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }
    let data_start = metadata_offset + meta.len() as u64 + index.len() as u64;
    let data_offset = (data_start + 4095) & !4095;
    f.write_all(HFQ_MAGIC).unwrap();
    f.write_all(&HFQ_VERSION.to_le_bytes()).unwrap();
    f.write_all(&arch.to_le_bytes()).unwrap();
    f.write_all(&(tensors.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&metadata_offset.to_le_bytes()).unwrap();
    f.write_all(&data_offset.to_le_bytes()).unwrap();
    f.write_all(meta).unwrap();
    f.write_all(&index).unwrap();
    let pad = (data_offset - data_start) as usize;
    f.write_all(&vec![0u8; pad]).unwrap();
    for t in tensors {
        f.write_all(&t.data).unwrap();
    }
}

/// Legacy per-32 row -> compact per-128 row.
fn requant_row(src: &[u8], k: usize) -> Vec<u8> {
    let old_row = k + (k / 32) * 2;
    assert_eq!(src.len(), old_row, "legacy row length");
    let new_row = k + (k / 128) * 2;
    let mut out = vec![0u8; new_row];
    let mut w = [0f32; 128];
    for kb in 0..k / 128 {
        let mut amax = 0.0f32;
        for g in 0..4 {
            let sc = f16::from_bits(u16::from_le_bytes([
                src[k + (kb * 4 + g) * 2],
                src[k + (kb * 4 + g) * 2 + 1],
            ]))
            .to_f32();
            for i in 0..32 {
                let v = sc * f32::from(src[kb * 128 + g * 32 + i] as i8);
                w[g * 32 + i] = v;
                amax = amax.max(v.abs());
            }
        }
        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for i in 0..128 {
            let q = (w[i] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
            out[kb * 128 + i] = q as u8;
        }
        out[k + kb * 2..k + kb * 2 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 {
        eprintln!("usage: convert_iu8g128 <old.iu4p> <new.iu4p>");
        std::process::exit(2);
    }
    let (arch, metadata, mut tensors) = read_hfq(Path::new(&a[1]));
    let mut converted = 0usize;
    for t in &mut tensors {
        if t.quant_type != 59 {
            continue;
        }
        if t.shape.len() != 2 {
            panic!("qt59 tensor {} has shape {:?}", t.name, t.shape);
        }
        let (m, k) = (t.shape[0] as usize, t.shape[1] as usize);
        let old_row = k + (k / 32) * 2;
        if t.data.len() != m * old_row {
            panic!(
                "{}: legacy size {} != {}",
                t.name,
                t.data.len(),
                m * old_row
            );
        }
        let mut out = Vec::with_capacity(m * (k + (k / 128) * 2));
        for r in 0..m {
            out.extend_from_slice(&requant_row(&t.data[r * old_row..(r + 1) * old_row], k));
        }
        t.data = out;
        t.group_size = 128;
        converted += 1;
    }
    write_hfq(Path::new(&a[2]), arch, &metadata, &tensors);
    eprintln!(
        "wrote {} (arch {arch}, {} tensors, {converted} qt59 rows requantized)",
        a[2],
        tensors.len()
    );
}
