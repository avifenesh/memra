"""Execute actual Gemma projection dispatch with CPU recording stand-ins for CUDA.

This checks operand provenance and dispatch, not GPU arithmetic. Native parity is
still required against the independent T1/logit and normalized-hidden controls.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def function(source, name):
    body = source[source.index('fn ' + name + '('):]
    opening = body.index('{')
    depth = 1
    for i in range(opening + 1, len(body)):
        depth += (body[i] == '{') - (body[i] == '}')
        if not depth:
            return body[:i + 1]
    raise ValueError('unclosed function')


class GemmaRawOperands(unittest.TestCase):
    def test_actual_softcapped_verify_wrappers_keep_terminal_f32_masks(self):
        source = (ROOT / 'crates/memra-engine/src/hybrid_forward.rs').read_text()
        methods = function(source, 'gemma4_decode_step_t_logits_dev') + function(source, 'gemma4_decode_step_t_h')
        fixture = r'''
#![allow(dead_code,unused_variables)]
type CudaSlice<T>=Vec<T>;
type R<T>=Result<T,Box<dyn std::error::Error>>;
thread_local! {static RAW:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
struct Engine;
impl Engine {
 fn stage_a_raw_needed()->bool {RAW.get()}
 fn softcap(&self,ld:&mut Vec<f32>,cap:f32,n:usize)->R<()> {assert_eq!(ld.len(),n);for v in ld {*v=cap*(*v/cap).tanh();} Ok(())}
 fn dtoh(&self,ld:&Vec<f32>)->R<Vec<f32>> {Ok(ld.clone())}
}
struct Gemma {final_logit_softcapping:f32}
struct Config {gemma4:Option<Gemma>}
struct Output;
impl Output {fn out_features(&self)->usize {3}}
struct Cache;
struct HybridModel {cfg:Config,output:Output}
impl HybridModel {
 fn gemma4_verify_trunk(&self,e:&Engine,tokens:&[u32],pos0:usize,cache:&mut Cache,tok_dev:Option<&Vec<u32>>)->R<(Vec<f32>,Vec<f32>)> {
  Ok(([0.25,f32::NEG_INFINITY,1.25].repeat(tokens.len()),vec![7.;tokens.len()]))
 }
 fn gemma4_suppress(&self,e:&Engine,ld:&mut Vec<f32>,t:usize)->R<()> {
  for row in 0..t {ld[row*3+1]=f32::NEG_INFINITY;} Ok(())
 }
 // WRAPPERS
}
#[test] fn mask_order_preserves_f32_program_and_fast_bits() {
 let m=HybridModel{cfg:Config{gemma4:Some(Gemma{final_logit_softcapping:30.})},output:Output};let e=Engine;
 for raw in [false,true] {RAW.set(raw);for t in [1,4,15,16,17] {
  let tokens=vec![1;t];let mut cache=Cache;
  for (ld,hn) in [m.gemma4_decode_step_t_logits_dev(&e,&tokens,t,11,&mut cache).unwrap(),m.gemma4_decode_step_t_h(&e,&tokens,11,&mut cache).unwrap()] {
   assert_eq!(hn,vec![7.;t]);
   for row in 0..t {
    assert_eq!(ld[row*3+1].to_bits(),if raw {f32::NEG_INFINITY.to_bits()} else {(-30f32).to_bits()});
    for (col,v) in [(0,0.25f32),(2,1.25f32)] {assert_eq!(ld[row*3+col].to_bits(),(30f32*(v/30f32).tanh()).to_bits());}
   }
  }
 }}
}
'''
        with tempfile.TemporaryDirectory(prefix='gemma-mask-') as tmp:
            path = Path(tmp) / 'test.rs'
            path.write_text(fixture.replace('// WRAPPERS', methods))
            binary = Path(tmp) / 'test'
            subprocess.run(['rustc', '--edition=2024', '--test', str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary), '--nocapture'], check=True)

    def test_actual_attention_and_dense_projection_dispatch(self):
        source = (ROOT / 'crates/memra-engine/src/hybrid_forward.rs').read_text()
        methods = function(source, 'gemma4_attention_raw')
        for name in ('gemma4_decode_attn', 'gemma4_verify_attn', 'gemma4_verify_attn_stream'):
            body = function(source, name).split('        let mut q =', 1)[0]
            body = body.replace('Result<CudaSlice<f32>, Box<dyn std::error::Error>>', 'Result<Triple, Box<dyn std::error::Error>>')
            methods += body + 'Ok((q0,k0,v0))\n}\n'
        tail = function(source, 'gemma4_layer_tail_core_pn')
        tail = tail[tail.index('            let pair_fast ='):tail.index('            let mut act =')]
        methods += '''fn dense(&self,e:&Engine,ffn_gate:&model::GpuTensor,ffn_up:&model::GpuTensor,zsh:CudaSlice<f32>,t:usize)->R<Pair>{
            let n_embd=4; let zpair=None;
        ''' + tail + 'Ok((gate,up))}\n'
        for name in ('gemma4_verify_trunk', 'gemma4_verify_t_am_stream'):
            body = function(source, name)
            prefix = body[body.index('{') + 1:body.index('        let n_embd =')]
            methods += 'fn scope_' + name + '(&self,e:&Engine,tokens:&[u32],t:usize,fail:bool)->R<bool>{' + prefix
            methods += 'if fail {return Err("injected".into());} Ok(*e.exact.borrow())}\n'
        body = function(source, 'gemma4_decode_step_dc_into')
        prefix = body[body.index('{') + 1:body.index('        let n_embd =')]
        methods += 'fn dc_entry(&self,mutations:&mut usize)->R<()>{' + prefix + '*mutations+=1; Ok(())}\n'
        decode = (ROOT / 'crates/memra-engine/src/decode.rs').read_text()
        for name in ('generate', 'generate_with'):
            body = decode[decode.index('    pub fn ' + name + ('(' if name == 'generate' else '<')):]
            predicate = body[body.index('if self.uses_gemma_program()') + 3:]
            predicate = predicate[:predicate.index('&& let Some(embd_gpu)')]
            methods += f'fn route_{name}(&self,sampler:&Sampler)->bool {{' + predicate + '}\n'
            predicate = body[body.index('if qwen_dc') + 3:]
            predicate = predicate[:predicate.index('&& let Some(embd_gpu)')]
            methods += f'fn route_qwen_{name}(&self,sampler:&Sampler,qwen_dc:bool,max_new:usize,budget:usize)->bool {{' + predicate + '}\n'
        fixture = Path(__file__).with_name('gemma_raw_operands_fixture.rs').read_text()
        program = fixture.replace('// ACTUAL_PRODUCTION_METHODS', methods)
        with tempfile.TemporaryDirectory(prefix='gemma-raw-') as tmp:
            path = Path(tmp) / 'test.rs'
            path.write_text(program)
            binary = Path(tmp) / 'test'
            subprocess.run(['rustc', '--edition=2024', '--test', '-C', 'debug-assertions=no', str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary), '--nocapture'], check=True)


if __name__ == '__main__':
    unittest.main()
