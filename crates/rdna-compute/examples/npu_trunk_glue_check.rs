// SPDX-License-Identifier: Apache-2.0
//! Halo-only byte oracle. Run: cargo run --release -p rdna-compute --features lab --example npu_trunk_glue_check
//! Compares complete old/new packed arenas and sidecars, then reports pack/consume event timings.
use half::f16;
use peacemaker_npu::kernels::{gemm_g80::{self,bf16_output_offset},gemm_core::{Epilogue,Control},w8a8_epi::{grid15_up,bf16_direct}};
use rdna_compute::{Gpu,GpuTensor,DType,npu_trunk::{NpuBf16Layout,NpuTrunkOutput}};
const K:usize=2560;
const AP:usize=32;
const WP:usize=64;
// Same numeric DAG as lab quant_row(grid=1); repeated rows cache exact CPU dots,
// not GPU results. Every output and every padding byte is still compared.
fn activations()->(Vec<u16>,Vec<i8>,Vec<(u16,i16)>){
    let mut x=vec![0u16;AP*K];let mut a=vec![0i8;AP*K];let mut ds=vec![(0,0);AP];
    let mut seed=0xa8e001u32;
    for t in 0..AP {
        for i in 0..K {
            seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
            let h=if t==0 {0}
                else if t==1 {((seed&0x83ff)|1) as u16}
                else if t==2 {if i%2==0 {0x7bff}else{0xfbff}} // maximum finite F16
                else if t==3 {f16::from_f32([127.0,0.5,1.5,2.5,-0.5,-1.5,-2.5,126.5][i%8]).to_bits()}
                else if t==4 {f16::from_f32([127.0625,127.0,126.9375,-127.0625,0.5,-0.5][i%6]).to_bits()}
                else {((seed&0x83ff)|(((8+t%20) as u32)<<10)) as u16};
            x[t*K+i]=h;
        }
        let am=x[t*K..(t+1)*K].iter().map(|&h|f16::from_bits(h).to_f32().abs()).fold(0f32,f32::max);
        let (m,e)=grid15_up(am/127.0f32);ds[t]=(m,e);
        let d=(m as f64*2f64.powi(e as i32)) as f32;let inv=if am==0.0 {0.0}else{1.0f32/d};
        for i in 0..K {a[t*K+i]=(f16::from_bits(x[t*K+i]).to_f32()*inv).round_ties_even().clamp(-127.0,127.0) as i8;}
    }(x,a,ds)
}
fn sides(s:&[(u16,i16)])->Vec<u8>{s.iter().flat_map(|&(m,e)|m.to_le_bytes().into_iter().chain(e.to_le_bytes())).collect()}
fn audit(label:&str,t:&GpuTensor){
    assert_eq!(t.byte_size(),t.buf.size(),"{label}: shape {:?} dtype {:?} disagrees with allocation",t.shape,t.dtype);
}
fn bytes(gpu:&mut Gpu,b:&[u8])->GpuTensor{
    let t=gpu.upload_raw(b,&[b.len()]).expect("upload");audit("raw upload",&t);t
}
fn typed(gpu:&mut Gpu,b:&[u8],shape:&[usize],dtype:DType)->GpuTensor{
    assert_eq!(shape.iter().product::<usize>()*dtype.size(),b.len(),"typed upload byte count");
    let mut t=gpu.upload_raw(b,shape).expect("typed upload");t.dtype=dtype;
    audit("typed upload",&t);t
}
fn arena(gpu:&mut Gpu,n:usize)->GpuTensor{bytes(gpu,&vec![0xa5;n])}
fn download(gpu:&Gpu,label:&str,t:&GpuTensor)->Vec<u8>{
    audit(label,t);gpu.download_raw_bytes(t).expect(label)
}
fn compare(label:&str,got:&[u8],want:&[u8]){
    if let Some(i)=got.iter().zip(want).position(|(a,b)|a!=b){panic!("{label}: byte {i}: got {:02x}, expected {:02x}",got[i],want[i]);}
    assert_eq!(got.len(),want.len(),"{label} length");
}
fn timed(gpu:&mut Gpu,label:&str,mut f:impl FnMut(&mut Gpu))->f32 {
    eprintln!("PHASE timing {label}");
    f(gpu);gpu.hip.device_synchronize().unwrap();
    let start=gpu.hip.event_create().unwrap();let end=gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start,gpu.active_stream.as_ref()).unwrap();f(gpu);
    gpu.hip.event_record(&end,gpu.active_stream.as_ref()).unwrap();gpu.hip.event_synchronize(&end).unwrap();
    let ms=gpu.hip.event_elapsed_ms(&start,&end).unwrap();
    println!("timing {label}: {ms:.4} ms");
    gpu.hip.event_destroy(start).unwrap();gpu.hip.event_destroy(end).unwrap();
    ms
}
fn pack_checks(gpu:&mut Gpu,xp:&[u16],ap:&[i8],dp:&[(u16,i16)]) {
    for rows in [1usize,37,512,4647,8151,8192] {
        // Offset zero covers all mixed fixtures at large sizes; the other offsets
        // ensure even the one-row test exercises maxima, RNE ties and scale edges.
        for offset in [0usize,2,3,4] {
            let p=rows.div_ceil(256)*256;
            let layout=NpuBf16Layout{padded_rows:p,k:K,panels:1};
            let xbytes:Vec<u8>=(0..rows*K).flat_map(|i|xp[(i+offset*K)%(AP*K)].to_le_bytes()).collect();
            let x=typed(gpu,&xbytes,&[rows,K],DType::F16);
            let a_cpu:Vec<i8>=(0..p*K).map(|i|if i/K<rows {ap[(i+offset*K)%(AP*K)]}else{0}).collect();
            let d_cpu:Vec<_>=(0..p).map(|t|if t<rows {dp[(t+offset)%AP]}else{(0,0)}).collect();
            let a=arena(gpu,layout.a_bytes()+256);let d=arena(gpu,rows*4+256);
            let old_a=arena(gpu,layout.a_bytes()+256);let old_d=arena(gpu,rows*4+256);
            gpu.npu_trunk_pack_a_bf16_old(&x,rows,layout,&old_a,&old_d).unwrap();
            gpu.npu_trunk_pack_a_bf16(&x,rows,layout,&a,&d).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let design=gemm_g80::design_g80(p,1280,K,Epilogue::Bf16,Control::Fast);
            let mut expected_a=gemm_g80::pack_a_bf16(&design,&a_cpu,&d_cpu);
            expected_a.extend([0xa5;256]);
            let mut expected_d=sides(&d_cpu[..rows]);expected_d.extend([0xa5;256]);
            compare("pack new vs CPU (including padding/guard)",&download(gpu,"new A",&a),&expected_a);
            compare("pack old vs CPU (including padding/guard)",&download(gpu,"old A",&old_a),&expected_a);
            compare("pack new D vs CPU/guard",&download(gpu,"new D",&d),&expected_d);
            compare("pack old D vs CPU/guard",&download(gpu,"old D",&old_d),&expected_d);
            compare("pack old/new arena",&download(gpu,"new A",&a),&download(gpu,"old A",&old_a));
            compare("pack old/new sidecars",&download(gpu,"new D",&d),&download(gpu,"old D",&old_d));
            println!("pack_exact rows={rows} fixture_offset={offset} old_vs_new=byte-exact padding=ok guards=ok");
            if offset==0 {
                let old=timed(gpu,&format!("pack-old rows={rows} k={K}"),|g|g.npu_trunk_pack_a_bf16_old(&x,rows,layout,&old_a,&old_d).unwrap());
                let new=timed(gpu,&format!("pack-new rows={rows} k={K}"),|g|g.npu_trunk_pack_a_bf16(&x,rows,layout,&a,&d).unwrap());
                let traffic=(rows*K*2+layout.a_bytes()+rows*4) as f64;
                println!("pack_bw rows={rows} old_ms={old:.4} new_ms={new:.4} old_GBps={:.2} new_GBps={:.2} speedup={:.2} target_GBps=180",
                    traffic/(old as f64*1e6),traffic/(new as f64*1e6),old/new);
                compare("timed pack old/new arena",&download(gpu,"new A",&a),&download(gpu,"old A",&old_a));
                compare("timed pack old/new sidecars",&download(gpu,"new D",&d),&download(gpu,"old D",&old_d));
            }
            for t in [x,a,d,old_a,old_d]{gpu.free_tensor(t).unwrap();}
        }
    }
}
fn run(){
    eprintln!("PHASE GPU initialization");
    let mut gpu=Gpu::init().expect("GPU init");assert_eq!(gpu.arch,"gfx1151");
    eprintln!("PHASE CPU activation and integer-dot references");
    let (xp,ap,dp)=activations();
    pack_checks(&mut gpu,&xp,&ap,&dp);
    let wp:Vec<i8>=(0..WP*K).map(|i|(((i*73+(i/K)*31+(i%K)*(i%K)*3)%255) as i32-127) as i8).collect();
    let dots:Vec<i32>=(0..AP*WP).map(|i|ap[i/WP*K..(i/WP+1)*K].iter().zip(&wp[i%WP*K..(i%WP+1)*K]).map(|(&a,&w)|a as i32*w as i32).sum()).collect();
    // Frozen S1 full ColumnMaps, native projection widths in fused family order.
    for rows in [512usize,8192,4647] {
        eprintln!("PHASE activation/A/D preparation rows={rows}");
        let p=rows.div_ceil(256)*256;
        let xbytes:Vec<u8>=(0..rows*K).flat_map(|i|xp[i%(AP*K)].to_le_bytes()).collect();
        let x=typed(&mut gpu,&xbytes,&[rows,K],DType::F16);
        let a_cpu:Vec<i8>=(0..p*K).map(|i|if i/K<rows {ap[i%(AP*K)]}else{0}).collect();
        let d_cpu:Vec<_>=(0..p).map(|t|if t<rows {dp[t%AP]}else{(0,0)}).collect();
        let layout=NpuBf16Layout{padded_rows:p,k:K,panels:1};
        let a=arena(&mut gpu,layout.a_bytes());let d=arena(&mut gpu,rows*4);
        eprintln!("PHASE packed-A producer rows={rows}");
        gpu.npu_trunk_pack_a_bf16(&x,rows,layout,&a,&d).unwrap();gpu.hip.device_synchronize().unwrap();
        eprintln!("PHASE packed-A and compact-D comparison rows={rows}");
        let design=gemm_g80::design_g80(p,1280,K,Epilogue::Bf16,Control::Fast);
        compare("packed A (codes, replicated scales, padding)",&download(&gpu,"packed A",&a),&gemm_g80::pack_a_bf16(&design,&a_cpu,&d_cpu));
        compare("compact D",&download(&gpu,"compact D",&d),&sides(&d_cpu[..rows]));
        for (name,widths) in [("GdnIn",[10240usize,6144,48,48]),("QsaIn",[12288,512,512,640])] {
            eprintln!("PHASE weight/scales/I32/synthetic-C preparation rows={rows} shape={name}");
            let n:usize=widths.iter().sum();let layout=NpuBf16Layout{panels:n.div_ceil(1280),..layout};
            let w=bytes(&mut gpu,&(0..n*K).map(|i|wp[i%(WP*K)] as u8).collect::<Vec<_>>());
            // Include zero weights scales, ties, underflow and overflow exponents.
            let scales:Vec<_>=(0..n).map(|j|if j%31==0 {(0,0)}else{(16384+(j%16384) as u16,[-160,-90,-30,-10,30,100][j%6])}).collect();
            let s=bytes(&mut gpu,&sides(&scales));let sums=arena(&mut gpu,rows*n*4);
            let mut synthetic=vec![0xa5;layout.c_bytes()];
            for t in 0..p {for j in 0..layout.padded_cols(){
                let bits=((t*131+j*37) as u16).wrapping_add(0x8000);
                let off=(j/1280)*p*1280*2+bf16_output_offset(p,t,j%1280);
                synthetic[off..off+2].copy_from_slice(&bits.to_le_bytes());
            }}
            let c=bytes(&mut gpu,&synthetic);
            for dtype in [DType::BF16,DType::F32] {
                eprintln!("PHASE native-output preparation rows={rows} shape={name} dtype={dtype:?}");
                let tensors:Vec<_>=widths.iter().map(|&m|{
                    let elements=rows*m+256/dtype.size();
                    typed(&mut gpu,&vec![0xa5;elements*dtype.size()],&[elements],dtype)
                }).collect();
                let mut col0=0;let outs:Vec<_>=widths.iter().zip(&tensors).map(|(&cols,tensor)|{let o=NpuTrunkOutput{col0,cols,tensor};col0+=cols;o}).collect();
                let fast_tensors:Vec<_>=widths.iter().map(|&m|{
                    let elements=rows*m+256/dtype.size();
                    typed(&mut gpu,&vec![0xa5;elements*dtype.size()],&[elements],dtype)
                }).collect();
                let fast_outs:Vec<_>=outs.iter().zip(&fast_tensors).map(|(o,tensor)|NpuTrunkOutput{col0:o.col0,cols:o.cols,tensor}).collect();
                eprintln!("PHASE consume launch rows={rows} shape={name} dtype={dtype:?}");
                gpu.npu_trunk_consume_bf16(&c,layout,&outs,rows).unwrap();gpu.hip.device_synchronize().unwrap();
                gpu.npu_trunk_consume_bf16_fast(&c,layout,&fast_outs,rows).unwrap();gpu.hip.device_synchronize().unwrap();
                for (o,fast) in outs.iter().zip(&fast_outs) {
                    eprintln!("PHASE consume comparison rows={rows} shape={name} dtype={dtype:?} col0={} cols={}",o.col0,o.cols);
                    let mut expected=Vec::with_capacity(rows*o.cols*dtype.size()+256);
                    for t in 0..rows {for j in o.col0..o.col0+o.cols {
                        let bits=((t*131+j*37) as u16).wrapping_add(0x8000);
                        if dtype==DType::BF16 {expected.extend(bits.to_le_bytes())}else{expected.extend(((bits as u32)<<16).to_le_bytes())}
                    }}expected.extend([0xa5;256]);
                    let old=download(&gpu,"consume output",o.tensor);
                    compare("consume and guard",&old,&expected);
                    compare("fast consume vs PM and guard",&download(&gpu,"fast consume output",fast.tensor),&old);
                }
                println!("consume_exact rows={rows} shape={name} dtype={dtype:?} old_vs_fast=byte-exact guards=ok");
                if rows==8192 || rows==4647 {
                    let old=timed(&mut gpu,&format!("consume-old {name} rows={rows} {dtype:?}"),|g|g.npu_trunk_consume_bf16(&c,layout,&outs,rows).unwrap());
                    let fast=timed(&mut gpu,&format!("consume-fast {name} rows={rows} {dtype:?}"),|g|g.npu_trunk_consume_bf16_fast(&c,layout,&fast_outs,rows).unwrap());
                    let traffic=(rows*n*(2+dtype.size())) as f64;
                    println!("consume_bw rows={rows} shape={name} dtype={dtype:?} old_ms={old:.4} fast_ms={fast:.4} old_GBps={:.2} fast_GBps={:.2} speedup={:.2} target_GBps=180",
                        traffic/(old as f64*1e6),traffic/(fast as f64*1e6),old/fast);
                    // Check timed writes too, including untouched ragged-row/output guards.
                    for (o,fast) in outs.iter().zip(&fast_outs) {
                        compare("timed fast vs PM and guard",&download(&gpu,"timed fast",fast.tensor),&download(&gpu,"timed old",o.tensor));
                    }
                }
                eprintln!("PHASE emulate launch rows={rows} shape={name} dtype={dtype:?}");
                gpu.npu_trunk_emulate_bf16(&a,&w,&s,&d,&sums,layout,rows,n,&outs).unwrap();gpu.hip.device_synchronize().unwrap();
                for o in &outs {
                    eprintln!("PHASE emulate comparison rows={rows} shape={name} dtype={dtype:?} col0={} cols={}",o.col0,o.cols);
                    let mut expected=Vec::with_capacity(rows*o.cols*dtype.size()+256);
                    for t in 0..rows {for j in o.col0..o.col0+o.cols {
                        let (sm,se)=scales[j];let (dm,de)=dp[t%AP];let bits=bf16_direct(dots[(t%AP)*WP+j%WP],sm,se,dm,de);
                        if dtype==DType::BF16 {expected.extend(bits.to_le_bytes())}else{expected.extend(((bits as u32)<<16).to_le_bytes())}
                    }}expected.extend([0xa5;256]);
                    compare("emulate and guard",&download(&gpu,"emulate output",o.tensor),&expected);
                }
                println!("PASS rows={rows} shape={name} dtype={dtype:?}");
                for t in tensors {gpu.free_tensor(t).unwrap();}
                for t in fast_tensors {gpu.free_tensor(t).unwrap();}
            }
            if rows==8192 {timed(&mut gpu,&format!("emulate GEMM {name} rows8192 k2560 n{n}"),|g|g.npu_trunk_emulate_i32(&a,&w,&sums,layout,rows,n).unwrap());}
            for t in [w,s,sums,c]{gpu.free_tensor(t).unwrap();}
        }
        for t in [x,a,d]{gpu.free_tensor(t).unwrap();}
    }
    println!("RESULT PASS");
}
fn main(){if std::panic::catch_unwind(run).is_err(){eprintln!("RESULT FAIL");std::process::exit(1);}}
