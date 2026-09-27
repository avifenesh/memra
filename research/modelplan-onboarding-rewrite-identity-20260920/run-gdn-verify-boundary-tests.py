#!/usr/bin/env python3
"""CPU admission controls extracted from actual GDN verify entrances.

The public decode_step_t body, refusal and canonical GDN predicate are verbatim.
Core/device/aux entrances execute their verbatim prefixes before the first work.
CUDA/math and previously tested identity guards are explicit stand-ins. This is
dispatch/no-work evidence, not native numerical qualification or an oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'crates/memra-engine/src/spec.rs'


def function(source, name):
    for qualifier in ('pub fn ', 'fn '):
        marker = '    ' + qualifier + name + '('
        if marker in source:
            start = source.index(marker)
            break
    else:
        return None
    opening = source.index('{', start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise ValueError('unclosed function: ' + name)


STUBS = r'''
#![allow(dead_code)]
use std::cell::Cell;
type Error = Box<dyn std::error::Error>;
mod memra_gguf {
    pub mod execution_manifest { pub enum RewriteSurface { DecodeEager } }
    pub mod model_plan { #[derive(PartialEq)] pub enum OperationKind { GatedDeltaNet } }
}
#[derive(Clone, Debug, PartialEq)] struct Cache { pos: usize, state: Vec<u32> }
struct Engine { work: Cell<usize> }
struct CudaSlice<T>(Vec<T>);
struct Plan { gdn: bool }
impl Plan {
    fn trunk_operations(&self) -> Vec<memra_gguf::model_plan::OperationKind> {
        if self.gdn { vec![memra_gguf::model_plan::OperationKind::GatedDeltaNet] } else { vec![] }
    }
}
struct HybridModel { plan: Plan, qualified: bool, eager: bool, gemma: bool }
impl HybridModel {
    fn protect_rewrite_execution(&self) -> Result<(), Error> { Ok(()) }
    fn rewrite_is_qualified(&self) -> bool { self.qualified }
    fn require_rewrite(&self, _: memra_gguf::execution_manifest::RewriteSurface) -> Result<(), String> {
        if self.eager { Ok(()) } else { Err("eager missing".into()) }
    }
    fn is_gemma4_e4b(&self) -> bool { false }
    fn gemma_batch_program(&self) -> bool { self.gemma }
    fn math(&self, e: &Engine, tokens: &[u32], cache: &mut Cache) -> Result<Vec<f32>, Error> {
        e.work.set(e.work.get() + 1); cache.pos += tokens.len();
        cache.state.extend_from_slice(tokens); Ok(vec![cache.pos as f32])
    }
    fn decode_step_t_h(&self, e: &Engine, tokens: &[u32], _: usize, cache: &mut Cache)
        -> Result<(Vec<f32>, CudaSlice<f32>), Error> {
        Ok((self.math(e, tokens, cache)?, CudaSlice(vec![])))
    }
    fn gemma4_e4b_decode_step_t_h(&self, e: &Engine, tokens: &[u32], pos: usize, cache: &mut Cache)
        -> Result<(Vec<f32>, CudaSlice<f32>), Error> { self.decode_step_t_h(e,tokens,pos,cache) }
    fn gemma4_decode_step_t(&self, e: &Engine, tokens: &[u32], _: usize, cache: &mut Cache)
        -> Result<Vec<f32>, Error> { self.math(e,tokens,cache) }
'''

TESTS = r'''
type Entry = fn(&HybridModel, &Engine, &[u32], usize, &mut Cache) -> Result<Vec<f32>, Error>;
fn entries() -> [Entry; 4] {
    [HybridModel::decode_step_t, HybridModel::core_boundary,
     HybridModel::device_boundary, HybridModel::aux_boundary]
}
#[test]
fn qualified_gdn_refuses_before_any_cache_or_math_work() {
    let model = HybridModel { plan: Plan { gdn: true }, qualified: true, eager: true, gemma: false };
    for entry in entries() { for rows in [0, 1, 4, 15, 16, 17] { for pos in [0, 11] {
        let e = Engine { work: Cell::new(0) };
        let mut cache = Cache { pos, state: vec![19, 23] }; let before = cache.clone();
        let result = entry(&model,&e,&vec![7;rows],pos,&mut cache);
        assert!(result.is_err(), "qualified GDN entered different numerical program");
        assert!(result.unwrap_err().to_string().contains("GDN verify numerical program"));
        assert_eq!(e.work.get(),0); assert_eq!(cache,before);
    }}}
}
#[test]
fn legacy_gdn_and_qualified_non_gdn_keep_the_original_path() {
    for (gdn,qualified,gemma) in [(true,false,false),(false,true,false),(false,true,true)] {
        let model = HybridModel { plan: Plan { gdn }, qualified, eager: true, gemma };
        for entry in entries() { for rows in [1, 4, 15, 16, 17] {
            let e = Engine { work: Cell::new(0) };
            let mut cache = Cache { pos: 11, state: vec![19,23] };
            assert_eq!(entry(&model,&e,&vec![7;rows],11,&mut cache).unwrap(),vec![(11+rows) as f32]);
            assert_eq!(e.work.get(),1); assert_eq!(cache.pos,11+rows);
            assert_eq!(&cache.state[2..],&vec![7;rows]);
        }}
    }
}
#[test]
fn existing_public_eager_refusal_still_precedes_work() {
    let model = HybridModel { plan: Plan { gdn: true }, qualified: false, eager: false, gemma: false };
    for entry in [HybridModel::decode_step_t as Entry,HybridModel::device_boundary,HybridModel::aux_boundary] {
        let e=Engine { work:Cell::new(0) }; let mut cache=Cache { pos:11,state:vec![23] };let before=cache.clone();
        assert_eq!(entry(&model,&e,&[7;16],11,&mut cache).unwrap_err().to_string(),"eager missing");
        assert_eq!(e.work.get(),0);assert_eq!(cache,before);
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ref')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = (subprocess.check_output(['git', 'show', f'{args.source_ref}:{SOURCE}'], cwd=ROOT).decode()
              if args.source_ref else (ROOT / SOURCE).read_text())
    parts = [STUBS]
    for name in ['refuse_unqualified_gdn_verify', 'batched_serving_numeric_class', 'decode_step_t']:
        body = function(source, name)
        if body:
            parts.append(body)
        elif name != 'refuse_unqualified_gdn_verify':
            raise ValueError('missing production body: ' + name)
    for name, method, stop in [
        ('decode_step_t_core_stream', 'core_boundary', '        // PP DOOR'),
        ('decode_step_t_h_emb_dev', 'device_boundary', '        cache.ensure_usable'),
        ('decode_step_t_aux2', 'aux_boundary', '        cache.ensure_usable'),
    ]:
        body = function(source, name)
        prefix = body[body.index('{')+1:body.index(stop)]
        parts.append(f'fn {method}(&self,e:&Engine,tokens:&[u32],_pos:usize,cache:&mut Cache)'
                     f' -> Result<Vec<f32>,Error> {{ {prefix} self.math(e,tokens,cache) }}')
    parts.extend(['}', TESTS])
    args.out.mkdir(parents=True, exist_ok=True)
    generated = args.out / 'actual-entry-controls.rs'
    generated.write_text('\n'.join(parts))
    with tempfile.TemporaryDirectory(prefix='gdn-verify-controls-', dir=ROOT / 'target') as temp:
        binary = Path(temp) / 'controls'
        subprocess.run(['rustc', '--edition=2024', '--test', str(generated), '-o', str(binary)], cwd=ROOT, check=True)
        result = subprocess.run([str(binary), '--nocapture'], capture_output=True, text=True)
    (args.out / 'output.log').write_text(result.stdout + result.stderr)
    (args.out / 'result.json').write_text(json.dumps({
        'source_ref': args.source_ref, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
        'exit_code': result.returncode,
        'scope': 'actual public body and pre-work prefixes; math/identity guard stand-ins; no native execution',
    }, indent=2) + '\n')
    print(result.stdout + result.stderr, end='')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
