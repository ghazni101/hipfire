// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use hipfire_isa::{Arch,Builder,KernelSpec,KernargLayout,RegPlan,Emitted};
use hipfire_isa::{reg::Live,insn::{Instruction,MemoryClass},ledger::Counter,vopd::{VopdOp,VopdF32,Operand}};
use std::{env,fs};

fn probe(arch:Arch)->Result<Emitted,String>{
 let mut regs=RegPlan::new(8,8)?;let mut v=Vec::new();for (name,n) in [("lane",0),("c",1),("w",2),("scale",3),("result",4)] {v.push(regs.v::<1>(name,n,Live::Whole)?)}
 let mut s=Vec::new();for (name,n) in [("arg0",0),("arg1",2),("arg2",4),("arg3",6)] {s.push(regs.s::<2>(name,n,Live::Whole)?)}
 let spec=KernelSpec{kernel_id:"fold_magic".into(),variant:"probe".into(),arch,symbol:"fold_magic".into(),kernargs:KernargLayout::new(32).pointer("out",0).pointer("weight",8).pointer("scale",16).pointer("c",24),user_sgpr_count:2,system_sgpr_workgroup_id_y:false,workgroup_size:1024,group_segment_fixed_size:0,wave32:true,cu_mode:false};let mut b=Builder::new(spec,regs);
 b.push(Instruction::new("s_load_b256 s[0:7], s[0:1], 0x0",s.iter().map(|r|r.reg()).collect(),vec![s[0].reg()]).memory(MemoryClass::SmemLoad))?;
 b.push(Instruction::new("v_lshlrev_b32_e32 v0, 2, v0",vec![v[0].reg()],vec![v[0].reg()]))?;
 b.wait(Counter::Km,0)?;
 b.clause(|b|{for (dst,ptr) in [(1,2),(2,0),(3,1),(4,3)] {b.push(Instruction::new(format!("global_load_b32 v{dst}, v0, s[{}:{}]",ptr*2,ptr*2+1),vec![v[dst].reg()],vec![v[0].reg(),s[ptr].reg()]).memory(MemoryClass::VmemLoad))?}Ok(())})?;
 b.vopd(VopdOp{op:VopdF32::Add,dst:1,src0:Operand::Lit(0xcb400000),src1:1},VopdOp{op:VopdF32::Mul,dst:2,src0:Operand::V(2),src1:3})?;
 if !arch.gfx12(){return Err("probe fold_magic uses a gfx12-only shared VMEM schedule; gfx11 is table-checked separately".into())}
 b.wait(Counter::Load,0)?;
 b.push(Instruction::new("s_delay_alu instid0(VALU_DEP_1)",vec![],vec![]))?;
 b.push(Instruction::new("v_fmac_f32_e32 v4, v2, v1",vec![v[4].reg()],vec![v[4].reg(),v[2].reg(),v[1].reg()]))?;
 b.push(Instruction::new("global_store_b32 v0, v4, s[6:7]",vec![],vec![v[0].reg(),v[4].reg(),s[3].reg()]).memory(MemoryClass::VmemStore))?;
 b.control(Instruction::new("s_endpgm",vec![],vec![]))?;
 b.finish()
}
const USAGE:&str="usage: hipfire-isa emit --kernel fold_magic --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel iu4_v2c [--epi set|add|silu|all] --arch gfx1100 --out FILE --proof FILE\n       hipfire-isa emit --kernel iu4_v2b --epi set|add|silu|all --arch gfx1151 --out FILE --proof FILE\n       hipfire-isa emit --kernel iu4_v2b_a4 --epi m512|a4|all --arch gfx1151 --out FILE --proof FILE\n       hipfire-isa emit --kernel iu4_gemm --fold k128 --tile 128x128x8|256x128x16 --cacc 1 --epi set|add|silu|silu-bf16|qkvzagdn|all [--alayout token|slab] --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel fp8_gemm --scale row|k128|both --epi set|add|silu|qkv|qkvza|all --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel gdn_scan --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel qsa_select --epi score|score-bf16|select|all|bf16 --arch gfx1151 --out FILE --proof FILE\n       hipfire-isa emit --kernel qwen4_mq6_x4 [--epi w4|w8|all] --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel qwen4_mq6_x4_gfx11 [--epi w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all] --arch gfx1151 --out FILE --proof FILE\n       hipfire-isa emit --kernel qwen4_mq4_x4_gfx11 [--epi w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all] --arch gfx1151 --out FILE --proof FILE\n       hipfire-isa emit --kernel mq4_rows_x4 [--epi entry|all] --arch gfx1151|gfx1201 --out FILE --proof FILE\n       hipfire-isa region-import --disassembly OBJDUMP.txt [--symbol gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3]";
const NATIVE_USAGE:&str="       emit options: [--co FILE] [--bundle FILE [--host-target TRIPLE]] also write the native code object / HIP offload bundle (no ROCm tools)\n       hipfire-isa emit --kernel pm_decode_<module> --arch gfx1201 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--shape FILE.shape.json] [--m7 FILE.m7.json]\n         (<module>: fused_gate_up_hfq4g256_mq4v2|gemv_hfq4g256_residual_mq4v2|fused_qkvza_hfq4g256_mq4v2|gemv_hfq4g256_multirow_default_mq4v2|fused_qkv_hfq4g256_mq4v2|gemv_hfq4g256_residual_xbatch_mq4v2|gemv_hfq4g256_xbatch_mq4v2; M7 runs on every export of the linked --co)\n       hipfire-isa emit --kernel mq4_verify [--epi <qkvza|qkv|gate_up|residual>_bt<4|8>w<4|8>|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]\n         (with --co, M7 runs on every export of the linked object; --shape writes DIR/mq4_verify.gfx1201.<entry>.contract.json)\n       hipfire-isa emit --kernel mq4_fp8_verify [--epi <qkvza|qkv|gate_up|residual|set>_bt<1|2|4|8>w<1|4|8>|all] --arch gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co [--bundle FILE.hxaco] [--shape DIR] [--m7 FILE.m7.json]]\n         (tiles bt1w1, bt{2,4,8}w{4,8}; with --co, M7 runs on every export of the linked object; --shape writes DIR/mq4_fp8_verify.gfx1201.<entry>.contract.json)\n       hipfire-isa emit --kernel qwen4_lane_rows --arch gfx1151 --out FILE.s --proof FILE.proof.json --co FILE.co [--bundle FILE.hxaco] [--m7 FILE.m7.json]\n         (the four-export lane-rows module qwen4_lane_rows_pm; --co is required and M7 runs on every export of the linked object)\n       hipfire-isa emit --kernel mq4_rows_x4 [--epi entry|all] --arch gfx1151|gfx1201 --out FILE.s --proof FILE.proof.json [--co FILE.co] [--bundle FILE.hxaco]\n         (the module mq4_rows_x4_pm_<arch> holds the single export gemv_mq4g256v2_rows_x4_pm_<arch>)";
/// `--epi all` emits the three epilogue symbols as one module (the product
/// code object the oracle loads); a single epilogue emits one symbol.
fn iu4_gemm(fold:&str,tile:&str,cacc:&str,epi:&str,act:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::iu4_gemm::{self,Spec};
 let (fold,tile,cacc,act)=(fold.parse()?,tile.parse()?,cacc.parse()?,act.parse()?);
 if epi=="all"{let (_,text,proof)=iu4_gemm::emit_module(fold,tile,cacc,act,arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let emitted=iu4_gemm::emit(Spec{fold,tile,cacc,epi:epi.parse()?,act,arch})?;
 Ok((emitted.s_text,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?))
}
/// gfx1100 V2C-equivalent GEMM: one epilogue symbol, or `--epi all` for the
/// product module (every epilogue symbol in one code object).
fn iu4_v2c(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::iu4_v2c::{self,Epi,Spec};
 if epi=="all"{let (_,text,proof)=iu4_v2c::module(arch,&Epi::ALL)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let e=iu4_v2c::emit(Spec{arch,epi:epi.parse()?})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// `--epi all` emits the SET, ADD and SiLU entries as one module.
fn iu4_v2b(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::iu4_v2b::{self,Spec};
 if epi=="all"{let (_,text,proof)=iu4_v2b::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let e=iu4_v2b::emit(Spec{arch,epi:epi.parse()?})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// `--epi all` emits the M512 gate/up entries (`m512` h twin, `a4` fused A4
/// epilogue) as one module: the separate A4-fusion code object.
fn iu4_v2b_a4(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::iu4_v2b_a4::{self,Spec};
 if epi=="all"{let (_,text,proof)=iu4_v2b_a4::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let e=iu4_v2b_a4::emit(Spec{arch,epi:epi.parse()?})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// `--epi all` emits every entry of the arch as one module; `gate_up`/`down`
/// one 16-slot entry, `gate_up_ntN`/`down_ntN` one expert-run entry,
/// `down_nt4rR` one expert-run down entry with R contiguous 64-row blocks per CTA.
fn qwen4_moe_sym(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qwen4_moe_sym::{self,Spec};
 if epi=="all"{let (_,text,proof)=qwen4_moe_sym::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let (kind,tile)=match epi.rsplit_once("_nt"){Some((k,n))=>(k,Some(n)),None=>(epi,None)};
 let (nt,rr)=match tile{
  Some(t)=>{let (n,r)=t.split_once('r').map_or((t,None),|(n,r)|(n,Some(r)));
   (n.parse::<u8>().map_err(|e|format!("--epi {epi}: {e}"))?,match r{Some(r)=>r.parse::<u8>().map_err(|e|format!("--epi {epi}: {e}"))?,None=>1})}
  None=>(1,1)};
 let e=qwen4_moe_sym::emit(Spec{arch,kind:kind.parse()?,nt,rr})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// `--epi all` (default) emits the convert and attention symbols as one module; `convert`/`attend` one symbol.
fn qsa_gather(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qsa_gather::{self,Kind,Spec};
 if epi=="all"{let (_,text,proof)=qsa_gather::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let kind=match epi{"convert"=>Kind::Convert,"attend"=>Kind::Attend,_=>return Err(format!("qsa_gather --epi {epi} (convert|attend|all)"))};
 let e=qsa_gather::emit(Spec{arch,kind})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// `all` (default) emits F32 score + select; `bf16` emits only BF16-pooled score.
fn qsa_select(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qsa_select::{self,Kind,Spec};
 if epi=="all"{let (_,text,proof)=qsa_select::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 if epi=="bf16"{let (_,text,proof)=qsa_select::emit_module_bf16(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let kind=match epi{"score"=>Kind::Score,"score-bf16"=>Kind::ScoreBf16,"select"=>Kind::Select,_=>return Err(format!("qsa_select --epi {epi} (score|score-bf16|select|all|bf16)"))};
 let e=qsa_select::emit(Spec{arch,kind})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// gfx1201 MQ6 trunk GEMM twin: `--epi w4|w8` one entry, `all` the module.
fn qwen4_mq6_x4(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qwen4_mq6_x4::{self,Spec};
 if epi=="all"{let (_,text,proof)=qwen4_mq6_x4::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let wr=match epi{"w4"=>4,"w8"=>8,_=>return Err(format!("qwen4_mq6_x4 --epi {epi} (w4|w8|all)"))};
 let e=qwen4_mq6_x4::emit(Spec{arch,wr})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// gfx1151 MQ6 trunk GEMM twin of hipcc's U3 BV8/RW8 entries: `--epi <wr>[_<kind>]`
/// (`w8`, `w4`, `w8_bf16out`, `w4_bf16out`, `w8_regions`, `w8_hcw`) one entry, `all` the module.
fn qwen4_mq6_x4_gfx11(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qwen4_mq6_x4_gfx11::{self,Kind,Spec};
 if epi=="all"{let (_,text,proof)=qwen4_mq6_x4_gfx11::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let (wr,kind)=match epi{"w4"=>(4,Kind::Plain),"w8"=>(8,Kind::Plain),"w4_bf16out"=>(4,Kind::Bf16),"w8_bf16out"=>(8,Kind::Bf16),"w8_regions"=>(8,Kind::Regions),"w8_hcw"=>(8,Kind::Hcw),
  _=>return Err(format!("qwen4_mq6_x4_gfx11 --epi {epi} (w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all)"))};
 let e=qwen4_mq6_x4_gfx11::emit(Spec{arch,wr,kind})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// gfx1151 Flash-Next lane-step row kernels: the whole `qwen4_lane_rows_pm` module (`--epi all` only).
fn qwen4_lane_rows(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qwen4_lane_rows;
 if epi!="all"{return Err(format!("qwen4_lane_rows --epi {epi} (all)"))}
 if arch!=Arch::Gfx1151{return Err("qwen4_lane_rows is exact-gfx1151 only".into())}
 let (_,text,proof)=qwen4_lane_rows::emit_module()?;
 Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))
}
/// gfx1151 MQ4G256V2 trunk GEMM twin of the hipcc MQ4 route (the `qwen4_mq6_x4_gfx11` builder over
/// the 136-byte MQ4 group): `--epi <wr>[_<kind>]` as there, `all` the module.
fn qwen4_mq4_x4_gfx11(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qwen4_mq4_x4_gfx11::{self,Kind,Spec};
 if epi=="all"{let (_,text,proof)=qwen4_mq4_x4_gfx11::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let (wr,kind)=match epi{"w4"=>(4,Kind::Plain),"w8"=>(8,Kind::Plain),"w4_bf16out"=>(4,Kind::Bf16),"w8_bf16out"=>(8,Kind::Bf16),"w8_regions"=>(8,Kind::Regions),"w8_hcw"=>(8,Kind::Hcw),
  _=>return Err(format!("qwen4_mq4_x4_gfx11 --epi {epi} (w4|w8|w4_bf16out|w8_bf16out|w8_regions|w8_hcw|all)"))};
 let e=qwen4_mq4_x4_gfx11::emit(Spec{arch,wr,kind})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// Few-row exact MQ4G256V2 GEMV over four matrices, gfx1151 and gfx1201:
/// the one entry (`--epi entry|all`, default `all`), which is also the module.
fn mq4_rows_x4(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::mq4_rows_x4::{self,Spec};
 if epi=="all"{let (_,text,proof)=mq4_rows_x4::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 if epi!="entry"{return Err(format!("mq4_rows_x4 --epi {epi} (entry|all)"))}
 let e=mq4_rows_x4::emit(Spec{arch})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// gfx1151 Flash-Next shared-expert gate/up (fused GEMM + BF16-recipe SwiGLU + selector sigmoid + F16 hidden):
/// `--epi w8|w4` one entry, `all` the module.
fn qwen4_shared_gu_gfx11(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::qwen4_shared_gu_gfx11::{self,Spec};
 if epi=="all"{let (_,text,proof)=qwen4_shared_gu_gfx11::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let wr=match epi{"w4"=>4,"w8"=>8,_=>return Err(format!("qwen4_shared_gu_gfx11 --epi {epi} (w4|w8|all)"))};
 let e=qwen4_shared_gu_gfx11::emit(Spec{arch,wr})?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// gfx1201 MQ4G256V2 wide exact-verify twins of hipcc's `_vt{4,8}w{4,8}_k32`
/// entries: `--epi <op>_bt<4|8>w<4|8>` one entry, `all` (default) the module.
fn mq4_verify_specs(epi:&str,arch:Arch)->Result<Vec<hipfire_isa::kernels::mq4_verify::Spec>,String>{
 use hipfire_isa::kernels::mq4_verify::Spec;
 if epi=="all"{return Ok(Spec::all(arch))}
 let bad=||format!("mq4_verify --epi {epi} (<qkvza|qkv|gate_up|residual>_bt<4|8>w<4|8>|all)");
 let (op,tile)=epi.rsplit_once("_bt").ok_or_else(bad)?;
 let (bt,waves)=tile.split_once('w').ok_or_else(bad)?;
 Ok(vec![Spec{arch,bt:bt.parse().map_err(|_|bad())?,waves:waves.parse().map_err(|_|bad())?,op:op.parse()?}])
}
fn mq4_verify(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::mq4_verify;
 if epi=="all"{let (_,text,proof)=mq4_verify::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let [spec]=mq4_verify_specs(epi,arch)?[..] else {return Err("mq4_verify: one entry expected".into())};
 let e=mq4_verify::emit(spec)?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
/// gfx1201 MQ4G256V2 symmetric FP8 verify GEMMs (1 <= N <= 128): `--epi <op>_bt<b>w<w>`
/// (`qkvza|qkv|gate_up|residual|set`, tiles `bt1w1`, `bt{2,4,8}w{4,8}`) one entry, `all` (default) the module.
fn mq4_fp8_verify_specs(epi:&str,arch:Arch)->Result<Vec<hipfire_isa::kernels::mq4_fp8_verify::Spec>,String>{
 use hipfire_isa::kernels::mq4_fp8_verify::Spec;
 if epi=="all"{return Ok(Spec::all(arch))}
 let bad=||format!("mq4_fp8_verify --epi {epi} (<qkvza|qkv|gate_up|residual|set>_bt<1|2|4|8>w<1|4|8>|all)");
 let (op,tile)=epi.rsplit_once("_bt").ok_or_else(bad)?;
 let (bt,waves)=tile.split_once('w').ok_or_else(bad)?;
 Ok(vec![Spec{arch,bt:bt.parse().map_err(|_|bad())?,waves:waves.parse().map_err(|_|bad())?,op:op.parse()?}])
}
fn mq4_fp8_verify(epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::mq4_fp8_verify;
 if epi=="all"{let (_,text,proof)=mq4_fp8_verify::emit_module(arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let [spec]=mq4_fp8_verify_specs(epi,arch)?[..] else {return Err("mq4_fp8_verify: one entry expected".into())};
 let e=mq4_fp8_verify::emit(spec)?;
 Ok((e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?))
}
fn fp8_gemm(scale:&str,epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::fp8_gemm::{self,Spec,ActScale,Epi};
 if scale=="both" {
     if epi!="all" {return Err("--scale both requires --epi all".into())}
     let epis=[Epi::Set,Epi::Add,Epi::GateUpSilu,Epi::Qkv,Epi::Qkvza];
     let mut emitted=Vec::with_capacity(10);
     for act_scale in [ActScale::Row,ActScale::K128] {
         for &epi in &epis {emitted.push(fp8_gemm::emit(Spec{arch,act_scale,epi})?);}
     }
     let (text,proof)=fp8_gemm::module(&emitted)?;
     return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))
 }
 let scale:ActScale=scale.parse()?;
 if matches!(epi,"all"|"initial") {
     let epis: &[Epi]=if epi=="initial" {&[Epi::Set,Epi::Add]} else {&[Epi::Set,Epi::Add,Epi::GateUpSilu,Epi::Qkv,Epi::Qkvza]};
     let (_,text,proof)=fp8_gemm::emit_module(arch,scale,epis)?;
     return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))
 }
 let emitted=fp8_gemm::emit(Spec{arch,act_scale:scale,epi:epi.parse()?})?;
 Ok((emitted.s_text,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?))
}
/// A PM decode twin module (`kernels::pm_decode`): every frozen export of the
/// HIP module combined into one code object source and proof.
fn pm_decode(module:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 if arch!=Arch::Gfx1201 {return Err(format!("pm_decode_{module} is exact-gfx1201 only"))}
 let (_,text,proof)=hipfire_isa::kernels::pm_decode::emit_module(module)?;
 Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))
}
/// Re-slice the SiLU region from a hipcc disassembly and require it to equal
/// the committed golden the gate/up epilogue instantiates.
fn region_import(mut args:impl Iterator<Item=String>)->Result<(),String>{
 use hipfire_isa::kernels::iu4_gemm::region;
 let mut dis=None;let mut symbol=None;let mut gdn=None;let mut write=false;let mut provenance=String::new();let mut arch=Arch::Gfx1201;
 while let Some(flag)=args.next(){if flag=="--write"{write=true;continue}let value=args.next().ok_or_else(||format!("missing value after {flag}"))?;match flag.as_str(){"--disassembly"=>dis=Some(value),"--symbol"=>symbol=Some(value),"--gdn-object"=>gdn=Some(value),"--provenance"=>provenance=value,"--arch"=>arch=value.parse()?,_=>return Err(format!("unknown flag {flag}\n{USAGE}"))}}
 if let Some(object)=gdn {return gdn_region_import(&object,write,&provenance)}
 let text=fs::read_to_string(dis.ok_or("missing --disassembly")?).map_err(|e|e.to_string())?;
 if !arch.gfx12(){
  // gfx1100: hipcc's V2C gate/up object (`--write` regenerates the golden, prefixing `--provenance`).
  let symbol=symbol.unwrap_or_else(||"gemm_mq4g256v2_gate_up_silu_iu4_v2c_gfx11".into());
  let slice=region::slice_silu_gfx11(&text,&symbol)?;
  region::Region::parse(&slice)?;
  let path=format!("{}/kernels/iu4_v2c.gfx1100.silu.region.s",env!("CARGO_MANIFEST_DIR"));
  if write {
   let head:String=provenance.split("\\n").map(|l|format!("; {l}\n")).collect();
   fs::write(&path,format!("{head}{slice}")).map_err(|e|e.to_string())?;
   eprintln!("region-import: wrote {path}");
   return Ok(())
  }
  print!("{slice}");
  if slice!=region::body_of(region::SILU_GOLDEN_GFX1100){return Err(format!("sliced region differs from {path}"))}
  eprintln!("region-import: {symbol} gfx1100 SiLU slice matches the committed golden");
  return Ok(())
 }
 let symbol=symbol.unwrap_or_else(||"gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3".into());
 let slice=region::slice_silu(&text,&symbol)?;
 print!("{slice}");
 if slice!=region::golden_body(){return Err("sliced region differs from kernels/iu4_gemm.silu.region.s".into())}
 region::Region::parse(&slice)?;
 eprintln!("region-import: {symbol} SiLU slice matches the committed golden");
 Ok(())
}
/// Lift `gdn_chunk_prep`'s hipcc code object with peacemaker (byte-exact
/// round trip), slice its per-token regions and require them to equal the
/// committed goldens (`--write` regenerates them, prefixing `--provenance`).
fn gdn_region_import(object:&str,write:bool,provenance:&str)->Result<(),String>{
 use hipfire_isa::kernels::fp8_gemm::gdn_region::{self,golden_body};
 use sha2::{Digest,Sha256};
 let bytes=fs::read(object).map_err(|e|format!("{object}: {e}"))?;
 let lifted=peacemaker_lift::lift_object(&bytes,peacemaker_lift::Options{frontend:peacemaker_ir::inst::Frontend::Hipcc}).map_err(|e|e.to_string())?;
 let kernel=lifted.program.kernels.iter().find(|k|k.symbol.0=="gdn_chunk_prep").ok_or("object has no gdn_chunk_prep kernel")?;
 let lines=peacemaker_lift::emit::text(kernel,peacemaker_ir::inst::Arch::Gfx1201).map_err(|e|e.to_string())?;
 let g=gdn_region::slice_gdn(&lines)?;
 let sha=format!("{:x}",Sha256::digest(&bytes));
 for (name,text,golden) in [("conv_silu",&g.conv_silu,gdn_region::CONV_SILU_GOLDEN),("norm_q",&g.norm_q,gdn_region::NORM_Q_GOLDEN),("norm_k",&g.norm_k,gdn_region::NORM_K_GOLDEN),("cvt_v",&g.cvt_v,gdn_region::CVT_V_GOLDEN)] {
  let path=format!("{}/kernels/fp8_gemm.gdn.{name}.region.s",env!("CARGO_MANIFEST_DIR"));
  if write {
   let head:String=provenance.split("\\n").map(|l|format!("; {l}\n")).collect();
   fs::write(&path,format!("{head}; Lifted with peacemaker-lift from {object} (sha256 {sha}); region `{name}`.\n; Regenerate/compare: `hipfire-isa region-import --gdn-object <object>` (feature `lift`).\n{text}")).map_err(|e|e.to_string())?;
   eprintln!("region-import: wrote {path}");
  } else if text!=&golden_body(golden) {return Err(format!("sliced {name} differs from {path}"))}
 }
 if !write {eprintln!("region-import: gdn_chunk_prep ({sha}) regions match the committed goldens")}
 Ok(())
}
fn run()->Result<(),String>{let mut args=env::args().skip(1);let command=args.next();if command.as_deref()==Some("region-import"){return region_import(args)}if command.as_deref()!=Some("emit"){return Err(format!("{USAGE}\n{NATIVE_USAGE}"))}let mut kernel=None;let mut arch=None;let mut out=None;let mut proof=None;let mut variant=None;let (mut fold,mut tile,mut cacc,mut epi,mut scale,mut alayout)=(None,None,None,None,None,None);let (mut co,mut bundle,mut host)=(None,None,hipfire_isa::native::DEFAULT_HOST_TARGET.to_owned());let (mut shape,mut m7)=(None::<String>,None::<String>);while let Some(flag)=args.next(){let value=args.next().ok_or_else(||format!("missing value after {flag}"))?;match flag.as_str(){"--kernel"=>kernel=Some(value),"--arch"=>arch=Some(value.parse::<Arch>()?),"--out"=>out=Some(value),"--proof"=>proof=Some(value),"--variant"=>variant=Some(value),"--fold"=>fold=Some(value),"--tile"=>tile=Some(value),"--cacc"=>cacc=Some(value),"--epi"=>epi=Some(value),"--scale"=>scale=Some(value),"--alayout"=>alayout=Some(value),"--co"=>co=Some(value),"--bundle"=>bundle=Some(value),"--host-target"=>host=value,"--shape"=>shape=Some(value),"--m7"=>m7=Some(value),_=>return Err(format!("unknown flag {flag}"))}}
 let kernel=kernel.ok_or("missing --kernel")?;let arch=arch.ok_or("missing --arch")?;
 let (text,proof_json)=match kernel.as_str(){
  "fold_magic"=>{if let Some(var)=variant {if var!="probe" {return Err("fold_magic supports only variant probe".into())}}let emitted=probe(arch)?;(emitted.s_text,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?)}
  "iu4_gemm"=>iu4_gemm(fold.as_deref().unwrap_or("k128"),tile.as_deref().ok_or("missing --tile")?,cacc.as_deref().unwrap_or("1"),epi.as_deref().ok_or("missing --epi")?,alayout.as_deref().unwrap_or("token"),arch)?,
  "gdn_scan"=>{let e=hipfire_isa::kernels::gdn_scan::emit(arch)?;(e.s_text,serde_json::to_vec_pretty(&e.proof).map_err(|e|e.to_string())?)}
  "iu4_v2c"=>iu4_v2c(epi.as_deref().unwrap_or("set"),arch)?,
  "iu4_v2b"=>iu4_v2b(epi.as_deref().ok_or("missing --epi")?,arch)?,
  "iu4_v2b_a4"=>iu4_v2b_a4(epi.as_deref().ok_or("missing --epi")?,arch)?,
  "qwen4_moe_sym"=>qwen4_moe_sym(epi.as_deref().ok_or("missing --epi")?,arch)?,
  "qsa_gather"=>qsa_gather(epi.as_deref().unwrap_or("all"),arch)?,
  "qsa_select"=>qsa_select(epi.as_deref().unwrap_or("all"),arch)?,
  "qwen4_mq6_x4"=>qwen4_mq6_x4(epi.as_deref().unwrap_or("all"),arch)?,
  "qwen4_mq6_x4_gfx11"=>qwen4_mq6_x4_gfx11(epi.as_deref().unwrap_or("all"),arch)?,
  "qwen4_lane_rows"=>qwen4_lane_rows(epi.as_deref().unwrap_or("all"),arch)?,
  "qwen4_mq4_x4_gfx11"=>qwen4_mq4_x4_gfx11(epi.as_deref().unwrap_or("all"),arch)?,
  "mq4_rows_x4"=>mq4_rows_x4(epi.as_deref().unwrap_or("all"),arch)?,
  "qwen4_shared_gu_gfx11"=>qwen4_shared_gu_gfx11(epi.as_deref().unwrap_or("all"),arch)?,
  "mq4_verify"=>mq4_verify(epi.as_deref().unwrap_or("all"),arch)?,
  "mq4_fp8_verify"=>mq4_fp8_verify(epi.as_deref().unwrap_or("all"),arch)?,
  "fp8_gemm"=>fp8_gemm(scale.as_deref().ok_or("missing --scale")?,epi.as_deref().ok_or("missing --epi")?,arch)?,
  k if k.starts_with("pm_decode_")=>pm_decode(&k["pm_decode_".len()..],arch)?,
  _=>return Err(format!("kernel {kernel} is not authored\n{USAGE}"))};
 let pm_module=kernel.strip_prefix("pm_decode_");
 let mq4v=kernel=="mq4_verify";
 let mq4f8=kernel=="mq4_fp8_verify";
 let lane_rows=kernel=="qwen4_lane_rows";
 if pm_module.is_none()&&!mq4v&&!mq4f8&&!lane_rows&&(shape.is_some()||m7.is_some()) {return Err("--shape/--m7 apply only to pm_decode_<module>, mq4_verify, mq4_fp8_verify and qwen4_lane_rows".into())}
 if lane_rows&&shape.is_some() {return Err("qwen4_lane_rows has no --shape contract (use --m7)".into())}
 if lane_rows&&co.is_none() {return Err("qwen4_lane_rows emission requires --co (M7 certifies the linked object)".into())}
 if pm_module.is_some()&&co.is_none() {return Err("pm_decode emission requires --co (M7 certifies the linked object)".into())}
 if mq4v&&co.is_none()&&(shape.is_some()||m7.is_some()) {return Err("mq4_verify --shape/--m7 require --co (M7 certifies the linked object)".into())}
 if mq4f8&&co.is_none()&&(shape.is_some()||m7.is_some()) {return Err("mq4_fp8_verify --shape/--m7 require --co (M7 certifies the linked object)".into())}
 // Native emission: the code object `llvm-mc` + `ld.lld -shared` would link, and its bundle.
 if co.is_some()||bundle.is_some(){
  let elf=hipfire_isa::native::assemble(&text,arch)?;
  if let Some(path)=&co{fs::write(path,&elf).map_err(|e|format!("{path}: {e}"))?}
  if let Some(path)=bundle{fs::write(&path,hipfire_isa::native::bundle(&elf,arch,&host)).map_err(|e|format!("{path}: {e}"))?}
 }
 // PM decode twins: M7 (lift byte-exact, obligations {}, no ambiguous delays) on every export of the linked object.
 #[cfg(not(feature="toolchain"))]
 if pm_module.is_some() {return Err("pm_decode emission certifies with M7: build hipfire-isa with --features toolchain".into())}
 #[cfg(feature="toolchain")]
 if let Some(module)=pm_module {
  let path=co.as_deref().ok_or("missing --co")?;
  let mut receipts=serde_json::Map::new();
  for &symbol in hipfire_isa::kernels::pm_decode::module(module)?.symbols {
   receipts.insert(symbol.to_owned(),hipfire_isa::pm_check::m7(std::path::Path::new(path),arch.name(),symbol)?);
  }
  let receipts=serde_json::to_vec_pretty(&receipts).map_err(|e|e.to_string())?;
  println!("{}",String::from_utf8_lossy(&receipts));
  if let Some(file)=&m7 {fs::write(file,&receipts).map_err(|e|format!("{file}: {e}"))?}
  if let Some(file)=&shape {fs::write(file,serde_json::to_vec_pretty(&hipfire_isa::kernels::pm_decode::shape_contract(module)?).map_err(|e|e.to_string())?).map_err(|e|format!("{file}: {e}"))?}
 }
 // Lane-step row kernels: M7 on every export of the linked object.
 #[cfg(not(feature="toolchain"))]
 if lane_rows {return Err("qwen4_lane_rows emission certifies with M7: build hipfire-isa with --features toolchain".into())}
 #[cfg(feature="toolchain")]
 if lane_rows {
  let path=co.as_deref().ok_or("missing --co")?;
  let mut receipts=serde_json::Map::new();
  for symbol in hipfire_isa::kernels::qwen4_lane_rows::SYMBOLS {
   receipts.insert(symbol.to_owned(),hipfire_isa::pm_check::m7(std::path::Path::new(path),arch.name(),symbol)?);
  }
  let receipts=serde_json::to_vec_pretty(&receipts).map_err(|e|e.to_string())?;
  println!("{}",String::from_utf8_lossy(&receipts));
  if let Some(file)=&m7 {fs::write(file,&receipts).map_err(|e|format!("{file}: {e}"))?}
 }
 // mq4_verify twins: with --co, M7 on every export of the linked object
 // (toolchain builds), plus each export's shape contract (--shape DIR).
 #[cfg(not(feature="toolchain"))]
 if mq4v&&(shape.is_some()||m7.is_some()) {return Err("mq4_verify --shape/--m7 certify with M7: build hipfire-isa with --features toolchain".into())}
 #[cfg(not(feature="toolchain"))]
 if mq4f8&&(shape.is_some()||m7.is_some()) {return Err("mq4_fp8_verify --shape/--m7 certify with M7: build hipfire-isa with --features toolchain".into())}
 #[cfg(feature="toolchain")]
 if mq4v&&co.is_some() {
  let path=co.as_deref().ok_or("missing --co")?;
  let mut receipts=serde_json::Map::new();
  for spec in mq4_verify_specs(epi.as_deref().unwrap_or("all"),arch)? {
   let symbol=spec.symbol();
   let m7=hipfire_isa::pm_check::m7(std::path::Path::new(path),arch.name(),&symbol)?;
   if let Some(dir)=&shape {
    let e=hipfire_isa::kernels::mq4_verify::emit(spec)?;
    let file=format!("{dir}/mq4_verify.{}.{}.contract.json",arch.name(),spec.tag());
    fs::write(&file,serde_json::to_vec_pretty(&hipfire_isa::kernels::mq4_verify::contract(spec,&e)).map_err(|e|e.to_string())?).map_err(|e|format!("{file}: {e}"))?;
   }
   receipts.insert(symbol,m7);
  }
  let receipts=serde_json::to_vec_pretty(&receipts).map_err(|e|e.to_string())?;
  println!("{}",String::from_utf8_lossy(&receipts));
  if let Some(file)=&m7 {fs::write(file,&receipts).map_err(|e|format!("{file}: {e}"))?}
 }
 // mq4_fp8_verify: the same, for the FP8 verify entries.
 #[cfg(feature="toolchain")]
 if mq4f8&&co.is_some() {
  let path=co.as_deref().ok_or("missing --co")?;
  let mut receipts=serde_json::Map::new();
  for spec in mq4_fp8_verify_specs(epi.as_deref().unwrap_or("all"),arch)? {
   let symbol=spec.symbol();
   let m7=hipfire_isa::pm_check::m7(std::path::Path::new(path),arch.name(),&symbol)?;
   if let Some(dir)=&shape {
    let e=hipfire_isa::kernels::mq4_fp8_verify::emit(spec)?;
    let file=format!("{dir}/mq4_fp8_verify.{}.{}.contract.json",arch.name(),spec.tag());
    fs::write(&file,serde_json::to_vec_pretty(&hipfire_isa::kernels::mq4_fp8_verify::contract(spec,&e)).map_err(|e|e.to_string())?).map_err(|e|format!("{file}: {e}"))?;
   }
   receipts.insert(symbol,m7);
  }
  let receipts=serde_json::to_vec_pretty(&receipts).map_err(|e|e.to_string())?;
  println!("{}",String::from_utf8_lossy(&receipts));
  if let Some(file)=&m7 {fs::write(file,&receipts).map_err(|e|format!("{file}: {e}"))?}
 }
 fs::write(out.ok_or("missing --out")?,text).map_err(|e|e.to_string())?;fs::write(proof.ok_or("missing --proof")?,proof_json).map_err(|e|e.to_string())?;Ok(())}
fn main(){if let Err(e)=run(){eprintln!("hipfire-isa: {e}");std::process::exit(1)}}
