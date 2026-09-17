// SPDX-License-Identifier: Apache-2.0
//! GPU rejection-law regression: distribution, row alignment, replay and invalid inputs.
use rdna_compute::{DType, Gpu};
fn run(gpu: &mut Gpu, p: &[f32], q: &[f32], tokens: &[u32], vocab: usize, temp: f32, seed: u32) -> Vec<[u32; 2]> {
    let mut tensors = Vec::new();
    for n in [p.len(), q.len(), tokens.len(), tokens.len()*2] {
        tensors.push(gpu.alloc_tensor(&[n], DType::F32).unwrap());
    }
    let pbytes: Vec<u8> = p.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let qbytes: Vec<u8> = q.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let tbytes: Vec<u8> = tokens.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&tensors[0].buf, &pbytes).unwrap();
    gpu.hip.memcpy_htod(&tensors[1].buf, &qbytes).unwrap();
    gpu.hip.memcpy_htod(&tensors[2].buf, &tbytes).unwrap();
    gpu.uno_verify_logits(&tensors[0], &tensors[1], &tensors[2], &tensors[3], tokens.len(), vocab, temp, seed).unwrap();
    let mut bytes = vec![0; tokens.len()*8];
    gpu.hip.memcpy_dtoh(&mut bytes, &tensors[3].buf).unwrap();
    let result = bytes.chunks_exact(8).map(|b| [u32::from_ne_bytes(b[..4].try_into().unwrap()), u32::from_ne_bytes(b[4..].try_into().unwrap())]).collect();
    for t in tensors { gpu.free_tensor(t).unwrap(); }
    result
}
fn main() {
    let mut gpu = Gpu::init().unwrap();
    let rows = 100_000;
    let temp = 1.5;
    let p: Vec<f32> = [0.1f32,0.6,0.3].map(|x| x.ln()*temp+7.).repeat(rows);
    let q: Vec<f32> = [0.7f32,0.2,0.1].map(|x| x.ln()*temp-3.).repeat(rows);
    // Stratified q proposals; GPU draws acceptance and residual independently.
    let tokens: Vec<u32> = (0..rows).map(|i| if i%10<7 {0} else if i%10<9 {1} else {2}).collect();
    let a = run(&mut gpu,&p,&q,&tokens,3,temp,12345);
    assert_eq!(a, run(&mut gpu,&p,&q,&tokens,3,temp,12345), "seed replay");
    let mut counts=[0usize;3]; let mut rejected=0;
    for (pair,t) in a.iter().zip(&tokens) {
        assert!(pair[0]<=1);
        let emitted=if pair[0]==1 {*t} else {rejected+=1; assert_ne!(pair[1],0); pair[1]};
        counts[emitted as usize]+=1;
    }
    for (count,expected) in counts.into_iter().zip([0.1,0.6,0.3]) { assert!((count as f64/rows as f64-expected).abs()<0.005); }
    assert!((rejected as f64/rows as f64-0.6).abs()<0.005);
    println!("rejection law PASS counts={counts:?} rejected={rejected}/{rows}; seed replay PASS");
    let vocab=250624;
    let mut p=vec![-1000.;3*vocab]; let mut q=p.clone();
    // Distinct rows, tails beyond multiple 256-lane iterations.
    for (r,(pt,qt)) in [(250623,249999),(123456,234567),(250600,250600)].into_iter().enumerate() {
        p[r*vocab+pt]=1000.; q[r*vocab+qt]=1000.;
    }
    let result=run(&mut gpu,&p,&q,&[249999,234567,250600],vocab,0.8,42);
    assert_eq!(result,[[0,250623],[0,123456],[1,0]]);
    println!("250624-vocab row alignment, disjoint support and identical distributions PASS");
    p[0]=f32::NAN;
    assert_eq!(run(&mut gpu,&p,&q,&[249999,234567,250600],vocab,0.8,42)[0][0],2);
    assert_eq!(run(&mut gpu,&[0.,0.],&[0.,0.],&[2],2,1.,42)[0][0],2);
    println!("invalid logits and out-of-range proposal fail closed PASS");
}
