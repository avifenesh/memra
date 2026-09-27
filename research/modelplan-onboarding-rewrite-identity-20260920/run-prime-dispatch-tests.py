#!/usr/bin/env python3
"""CPU routing/transaction controls using actual prime entry and eager-loop bodies.

CUDA allocation/copies, model T1 arithmetic and the legacy prefill backend are
explicit stand-ins. This proves dispatch and boundary handling, not native math.
The frozen 655 entry must fail the strict direct and batch routing tests.
"""
from pathlib import Path
import argparse, hashlib, json, re, subprocess

def function(source, name):
    start = source.index('    ' + ('pub fn ' if '    pub fn '+name+'(' in source else 'fn ') + name + '(')
    opening = source.index('{', start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise RuntimeError('unclosed function')

STUBS = r'''
#![allow(dead_code)]
use std::cell::{Cell, RefCell};
type Error = Box<dyn std::error::Error>;
#[derive(Clone, Debug, PartialEq)] struct CudaSlice<T>(Vec<T>);
mod memra_gguf { pub mod model_plan { pub enum OperationKind { PleNgramEmbedding } } pub mod execution_manifest { pub enum RewriteSurface { DecodeEager, CarriedPrime, Pipeline } } }
mod vision { pub struct EmbedOverlay; }
mod spec {
    thread_local! { static HPOST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    pub fn spec_hpost() -> bool { HPOST.with(|x| x.get()) }
    pub struct Scope(bool);
    impl Drop for Scope { fn drop(&mut self) { HPOST.with(|x| x.set(self.0)); } }
    pub fn scope(on: bool) -> Scope { Scope(HPOST.with(|x| x.replace(on))) }
}
fn normalized(row: &[f32], weights: &[f32], eps: f32) -> Vec<f32> {
    let rms=(row.iter().map(|x|x*x).sum::<f32>() / row.len() as f32 + eps).sqrt();
    row.iter().zip(weights).map(|(x,w)|x*w/rms).collect()
}
struct Tensor(CudaSlice<f32>);
impl Tensor { fn float_data(&self) -> &CudaSlice<f32> { &self.0 } }
mod pp { pub fn pp_cuts(_: usize) -> Option<Vec<usize>> { None } }
mod progress {
    thread_local! { static ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
    pub fn events() -> usize { ROWS.with(|r| r.get()) }
    pub fn note_prime_rows(n: usize) { ROWS.with(|r| r.set(r.get()+n)); }
    __ACTUAL_PRIME_CANCEL_BOUNDARY__
}
#[derive(Default)] struct Engine { fail_copy: Cell<bool>, fail_norm: Cell<bool>, norms: Cell<usize> }
impl Engine {
    fn rms_norm(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, out: &mut CudaSlice<f32>, cols: usize, rows: usize, eps: f32) -> Result<(), Error> {
        self.norms.set(self.norms.get()+1);
        if self.fail_norm.get() { return Err("normalization failure".into()); }
        assert_eq!(rows,1); assert_eq!(x.0.len(),cols);
        out.0=normalized(&x.0,&w.0,eps); Ok(())
    }
    fn uninit(&self, n: usize) -> Result<CudaSlice<f32>, Error> { Ok(CudaSlice(vec![f32::NAN; n])) }
    fn copy_into(&self, dst: &mut CudaSlice<f32>, off: usize, src: &CudaSlice<f32>, n: usize) -> Result<(), Error> {
        if self.fail_copy.get() { return Err("copy failure".into()); }
        dst.0[off..off+n].copy_from_slice(&src.0[..n]); Ok(())
    }
}
#[derive(Debug)] struct Cache { pos: usize, max_ctx: usize, tainted: bool, value: f32 }
impl Cache {
    fn at(pos: usize) -> Self { Self { pos, max_ctx: 256, tainted: false, value: pos as f32 } }
    fn ensure_usable(&self, _: &str) -> Result<(), Error> { if self.tainted { Err("tainted".into()) } else { Ok(()) } }
    fn mark_tainted(&mut self) { self.tainted = true; }
}
struct Config { n_embd: u32, rms_eps: f32 }
struct HybridModel {
    cfg: Config, layers: Vec<()>, eager: bool, prime: bool, gemma: bool, ple: bool, output_norm: Tensor,
    calls: RefCell<Vec<(usize, u32)>>, prefill: Cell<usize>, fail_at: Option<usize>,
}
impl HybridModel {
    fn new(prime: bool) -> Self { Self { cfg: Config { n_embd: 3, rms_eps: 0.000001 }, layers: vec![], eager: true,
        prime, gemma:false, ple:false, output_norm:Tensor(CudaSlice(vec![0.5,1.5,2.0])), calls: RefCell::new(vec![]), prefill: Cell::new(0), fail_at: None } }
    fn protect_rewrite_execution(&self) -> Result<(), Error> { Ok(()) }
    fn require_rewrite(&self, _: memra_gguf::execution_manifest::RewriteSurface) -> Result<(), Error> {
        if self.eager { Ok(()) } else { Err("unqualified eager".into()) }
    }
    fn rewrite_allowed(&self, surface: memra_gguf::execution_manifest::RewriteSurface) -> bool {
        match surface { memra_gguf::execution_manifest::RewriteSurface::DecodeEager => self.eager, _ => self.prime }
    }
    fn uses_gemma_program(&self) -> bool { self.gemma }
    fn has_plan_operation(&self, _: memra_gguf::model_plan::OperationKind) -> bool { self.ple }
    fn refuse_hyper(&self, _: &str) -> Result<(), Error> { Ok(()) }
    fn a4_prime_receipt_begin(&self) -> Option<()> { None }
    fn a4_prime_receipt_end(&self, _: &str, _: usize, _: Option<()>, _: Option<&Vec<f32>>) {}
    fn prime_cache_overlaid_inner(&self, _: &Engine, tokens: &[u32], cache: &mut Cache, _: usize,
        _: Option<&vision::EmbedOverlay>) -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Error> {
        self.prefill.set(self.prefill.get()+1); cache.pos += tokens.len();
        Ok((vec![-100.0], CudaSlice(vec![-100.0;3]), CudaSlice(vec![-100.0;tokens.len()*3])))
    }
    fn decode_step_h(&self, _: &Engine, token: u32, cache: &mut Cache) -> Result<(Vec<f32>, CudaSlice<f32>), Error> {
        cache.ensure_usable("t1")?;
        if self.fail_at == Some(cache.pos) { return Err("T1 failure".into()); }
        self.calls.borrow_mut().push((cache.pos, token));
        cache.value = cache.value * 1.03125 + token as f32; cache.pos += 1;
        let raw=vec![cache.value, cache.pos as f32, token as f32];
        let row=if !self.gemma && spec::spec_hpost() { normalized(&raw,&self.output_norm.0.0,self.cfg.rms_eps) } else { raw };
        Ok((vec![cache.value, -cache.value], CudaSlice(row)))
    }
}
'''

TESTS = r'''
#[test] fn strict_direct_preserves_t1_at_15_16_17_and_carried_positions() {
    for t in [15,16,17] { for base in [0,11,33] {
        let model=HybridModel::new(false); let mut cache=Cache::at(base); let engine=Engine::default();
        let tokens:Vec<u32>=(1..=t as u32).collect();
        let (logits, seed, hidden)=model.prime_cache(&engine,&tokens,&mut cache,23).unwrap();
        assert_eq!(model.prefill.get(),0,"unqualified prefill entered");
        assert_eq!(*model.calls.borrow(),tokens.iter().enumerate().map(|(i,&x)|(base+i,x)).collect::<Vec<_>>());
        assert_eq!(cache.pos,base+t); assert!(!cache.tainted);
        assert_eq!(hidden.0.len(),t*3); assert_eq!(seed.0,hidden.0[(t-1)*3..]);
        let mut expected=base as f32;
        for (i,&token) in tokens.iter().enumerate() {
            expected=expected*1.03125+token as f32;
            assert_eq!(&hidden.0[i*3..(i+1)*3], &[expected,(base+i+1)as f32,token as f32]);
        }
        assert_eq!(logits,vec![expected,-expected]);
    }}
}
#[test] fn strict_batch_uses_independent_t1_caches_across_the_threshold() {
    for lengths in [[15,16,17],[17,16,15]] {
        let model=HybridModel::new(false); let engine=Engine::default();
        let owned:Vec<Vec<u32>>=lengths.iter().map(|&n|(1..=n as u32).collect()).collect();
        let prompts:Vec<&[u32]>=owned.iter().map(Vec::as_slice).collect();
        let mut caches=vec![Cache::at(0),Cache::at(11),Cache::at(33)];
        let mut refs:Vec<&mut Cache>=caches.iter_mut().collect();
        let result=model.prime_cache_batch_inner(&engine,&prompts,&mut refs).unwrap();
        assert_eq!(model.prefill.get(),0); assert_eq!(model.calls.borrow().len(),48);
        for (i,base) in [0,11,33].into_iter().enumerate() {
            assert_eq!(caches[i].pos,base+lengths[i]); assert!(!caches[i].tainted);
            let mut expected=base as f32;
            for &token in &owned[i] { expected=expected*1.03125+token as f32; }
            assert_eq!(result[i].0,vec![expected,-expected]);
            assert_eq!(result[i].2.0.len(),lengths[i]*3);
        }
    }
}
#[test] fn allowed_prime_preserves_existing_prefill_route() {
    let model=HybridModel::new(true); let mut cache=Cache::at(0);
    model.prime_cache(&Engine::default(),&[1;16],&mut cache,0).unwrap();
    assert_eq!(model.prefill.get(),1); assert!(model.calls.borrow().is_empty());
}
#[test] fn strict_empty_capacity_overlay_and_pending_refuse_before_work() {
    for mode in 0..4 {
        let mut model=HybridModel::new(false); let mut cache=Cache::at(0); let engine=Engine::default();
        if mode==1 { cache.max_ctx=15; } if mode==3 { model.eager=false; }
        let tokens:&[u32]=if mode==0 { &[] } else { &[1;16] };
        let overlay=vision::EmbedOverlay;
        assert!(model.prime_cache_overlaid(&engine,tokens,&mut cache,0,if mode==2 { Some(&overlay) } else { None }).is_err());
        assert_eq!(cache.pos,0); assert!(!cache.tainted);
        assert_eq!(model.prefill.get(),0); assert!(model.calls.borrow().is_empty());
    }
}
#[test] fn partial_t1_or_hidden_copy_failure_taints_and_stops_the_cache() {
    for copy_fail in [false,true] {
        let mut model=HybridModel::new(false); model.fail_at=if copy_fail { None } else { Some(3) };
        let engine=Engine::default(); engine.fail_copy.set(copy_fail); let mut cache=Cache::at(0);
        assert!(model.prime_cache(&engine,&[1;16],&mut cache,0).is_err());
        assert!(cache.tainted); let calls=model.calls.borrow().len();
        assert!(model.prime_cache(&engine,&[1;16],&mut cache,0).is_err());
        assert_eq!(model.calls.borrow().len(),calls);
    }
}
#[test] fn failed_batch_taints_every_member_and_never_enters_prefill() {
    let mut model=HybridModel::new(false); model.fail_at=Some(18); let engine=Engine::default();
    let mut a=Cache::at(0); let mut b=Cache::at(17);
    assert!(model.prime_cache_batch_inner(&engine,&[&[1;16],&[2;16]],&mut[&mut a,&mut b]).is_err());
    assert!(a.tainted&&b.tainted); assert_eq!(a.pos,16); assert_eq!(b.pos,18);
    assert_eq!(model.prefill.get(),0);
}
#[test] fn selected_hidden_convention_is_preserved_for_each_row() {
    for gemma in [false,true] { for ple in [false,true] { for hpost in [false,true] {
        let _scope=spec::scope(hpost); let mut model=HybridModel::new(false);model.gemma=gemma;model.ple=ple;
        let engine=Engine::default();let mut cache=Cache::at(11);let tokens:Vec<u32>=(1..=17).collect();
        let (logits,seed,hidden)=model.prime_cache(&engine,&tokens,&mut cache,0).unwrap();
        let mut value=11f32;
        for (i,&token) in tokens.iter().enumerate() {
            value=value*1.03125+token as f32;
            let raw=vec![value,(12+i)as f32,token as f32];
            let expected=if hpost && (!gemma || !ple) { normalized(&raw,&[0.5,1.5,2.0],model.cfg.rms_eps) } else { raw };
            assert_eq!(&hidden.0[i*3..(i+1)*3],expected.as_slice(),"gemma={gemma} ple={ple} hpost={hpost}");
        }
        assert_eq!(seed.0,hidden.0[16*3..]);assert_eq!(logits,vec![value,-value]);
        assert_eq!(cache.value,value);assert_eq!(cache.pos,28);assert!(!cache.tainted);
        assert_eq!(model.calls.borrow().len(),17);
        assert_eq!(engine.norms.get(),if gemma && !ple && hpost {17}else{0});
    }}}
}
#[test] fn hidden_norm_failure_taints_before_export() {
    let _scope=spec::scope(true);let mut model=HybridModel::new(false);model.gemma=true;
    let engine=Engine::default();engine.fail_norm.set(true);let mut cache=Cache::at(0);
    assert!(model.prime_cache(&engine,&[2;16],&mut cache,0).is_err());
    assert!(cache.tainted);assert_eq!(cache.pos,1);assert_eq!(model.calls.borrow().len(),1);
}
'''

def run(root, out):
    root, out = root.resolve(), out.resolve()
    out.mkdir(exist_ok=False, parents=True)
    path = 'crates/memra-engine/src/hybrid_forward.rs'
    source = (root/path).read_text()
    frozen = subprocess.check_output(['git','show','65510819:'+path],cwd=root,text=True)
    frozen45 = subprocess.check_output(['git','show','45c0b1742:'+path],cwd=root,text=True)
    guard = source[source.index('struct CacheTaintGuard {'):source.index('#[derive(Debug, Clone, Copy, PartialEq, Eq)]\nstruct PrimePpWaveSlot')]
    progress_source = (root/'crates/memra-engine/src/progress.rs').read_text()
    cancel = progress_source[progress_source.index('thread_local! {\n    static PRIME_CANCEL'):progress_source.index('/// The odometer\'s observable state.')]
    stubs = STUBS.replace('__ACTUAL_PRIME_CANCEL_BOUNDARY__', 'use std::cell::RefCell;\n'+cancel)
    helper = function(source, 'prime_cache_eager')
    records = {}
    for mode, code in [('actual',source),('frozen-655',frozen),('frozen-45c',frozen45)]:
        batch = function(code,'prime_cache_batch_inner')
        cut = batch.index('        let _pp_walk =')
        batch = batch[:cut] + '        Err("legacy batch backend stub".into())\n    }'
        selected_helper = function(code,'prime_cache_eager') if 'fn prime_cache_eager(' in code else helper
        methods = '\n'.join([function(code,'prime_cache'),function(code,'prime_cache_overlaid'),selected_helper,batch])
        program = stubs+guard+'\nimpl HybridModel {\n'+methods+'\n}\n'+TESTS
        src, binary = out/(mode+'.rs'), out/mode
        src.write_text(program)
        built = subprocess.run(['rustc','--edition=2024','--test',str(src),'-o',str(binary)],capture_output=True)
        (out/(mode+'-build.log')).write_bytes(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('CPU harness failed to compile: '+mode)
        result = subprocess.run([str(binary),'--nocapture','--test-threads=1'],capture_output=True)
        raw=result.stdout+result.stderr
        (out/(mode+'.log')).write_bytes(raw)
        text=raw.decode(errors='replace')
        if mode=='actual':
            valid=result.returncode==0 and '8 passed; 0 failed; 0 ignored;' in text
        elif mode=='frozen-45c':
            valid=result.returncode!=0 and '6 passed; 2 failed; 0 ignored;' in text and all('test '+name+' ... FAILED' in text for name in ('selected_hidden_convention_is_preserved_for_each_row','hidden_norm_failure_taints_before_export'))
        else:
            valid=result.returncode!=0 and 'test result: FAILED.' in text and all('test '+name+' ... FAILED' in text for name in ('strict_direct_preserves_t1_at_15_16_17_and_carried_positions','strict_batch_uses_independent_t1_caches_across_the_threshold'))
        if not valid:raise RuntimeError('wrong non-vacuous result: '+mode)
        records[mode]={'returncode':result.returncode,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'log_sha256':hashlib.sha256(raw).hexdigest()}
    report={'scope':'CPU dispatch/transaction only; GPU arithmetic is stubbed, no native qualification',
            'progress_source_sha256':hashlib.sha256(progress_source.encode()).hexdigest(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'frozen_source_sha256':hashlib.sha256(frozen.encode()).hexdigest(),'frozen45_source_sha256':hashlib.sha256(frozen45.encode()).hexdigest(),'results':records}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
