//! iu4_twin_check <iu.hfq> <mq.hfq>: proves an IU file's body rows are a pure
//! relayout of its MQ4V2 twin (qt 57) and its lifted qt 59 rows are the
//! deterministic per-128 requant of the Q8F16 twin (`iu_quant::repack_q8f16`).
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::iu_quant::{self, Layout};
use std::collections::BTreeMap;
use std::path::Path;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let iu = HfqFile::open(Path::new(&a[1])).unwrap();
    let mq = HfqFile::open(Path::new(&a[2])).unwrap();
    let mut stats: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut bad = 0;
    for t in iu.tensors().to_vec() {
        let (ti, ib) = iu.tensor_data(&t.name).unwrap();
        let Some((tm, mb)) = mq.tensor_data(&t.name) else {
            println!("MISSING_IN_MQ {}", t.name);
            bad += 1;
            continue;
        };
        let key = format!("iu{}<-mq{}", ti.quant_type, tm.quant_type);
        let shape = &ti.shape;
        let (m, k) = if shape.len() >= 2 {
            (
                shape[0] as usize,
                shape[1..].iter().map(|&s| s as usize).product(),
            )
        } else {
            (1, shape[0] as usize)
        };
        let ok = match (ti.quant_type, tm.quant_type) {
            (57, 44) => iu_quant::repack_mq4v2(mb, m, k, Layout::Iu4)
                .map(|v| v == ib)
                .unwrap_or(false),
            (59, _) if tm.quant_type != 59 => iu_quant::repack_q8f16(mb, m, k)
                .map(|v| v == ib)
                .unwrap_or(false),
            (x, y) if x == y => ib == mb,
            _ => false,
        };
        let e = stats.entry(key.clone()).or_insert((0, 0));
        if ok {
            e.0 += 1
        } else {
            e.1 += 1;
            bad += 1;
            if e.1 <= 3 {
                println!("MISMATCH {} {} shape={:?}", key, t.name, shape);
            }
        }
    }
    for (k, (ok, nb)) in &stats {
        println!("{k}: equal={ok} mismatch={nb}");
    }
    println!("TOTAL_BAD={bad}");
}
