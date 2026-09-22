
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

    fn refuse_unqualified_gdn_verify(&self) -> Result<(), String> {
        if self.rewrite_is_qualified() && self.batched_serving_numeric_class() {
            return Err(
                "GDN verify numerical program is not qualified; native-gdn-eager does not authorize batched verification"
                    .into(),
            );
        }
        Ok(())
    }
    fn batched_serving_numeric_class(&self) -> bool {
        self.plan
            .trunk_operations()
            .contains(&memra_gguf::model_plan::OperationKind::GatedDeltaNet)
    }
    pub fn decode_step_t(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Shared teacher-forced/prefill rows need a live eager baseline. Speculative
        // session entry points additionally require their MTP/GLM5 surface receipt.
        let _rewrite_execution = self.protect_rewrite_execution()?;
        self.require_rewrite(memra_gguf::execution_manifest::RewriteSurface::DecodeEager)?;
        self.refuse_unqualified_gdn_verify()?;
        if self.is_gemma4_e4b() {
            return Ok(self.gemma4_e4b_decode_step_t_h(e, tokens, pos0, cache)?.0);
        }
        if self.gemma_batch_program() {
            return self.gemma4_decode_step_t(e, tokens, pos0, cache);
        }
        Ok(self.decode_step_t_h(e, tokens, pos0, cache)?.0)
    }
fn core_boundary(&self,e:&Engine,tokens:&[u32],_pos:usize,cache:&mut Cache) -> Result<Vec<f32>,Error> { 
        let _rewrite_execution = self.protect_rewrite_execution()?;
        self.refuse_unqualified_gdn_verify()?;
 self.math(e,tokens,cache) }
fn device_boundary(&self,e:&Engine,tokens:&[u32],_pos:usize,cache:&mut Cache) -> Result<Vec<f32>,Error> { 
        // Shared teacher-forced/prefill rows need a live eager baseline. Speculative
        // session entry points additionally require their MTP/GLM5 surface receipt.
        let _rewrite_execution = self.protect_rewrite_execution()?;
        self.require_rewrite(memra_gguf::execution_manifest::RewriteSurface::DecodeEager)?;
        self.refuse_unqualified_gdn_verify()?;
 self.math(e,tokens,cache) }
fn aux_boundary(&self,e:&Engine,tokens:&[u32],_pos:usize,cache:&mut Cache) -> Result<Vec<f32>,Error> { 
        // Shared teacher-forced/prefill rows need a live eager baseline. Speculative
        // session entry points additionally require their MTP/GLM5 surface receipt.
        let _rewrite_execution = self.protect_rewrite_execution()?;
        self.require_rewrite(memra_gguf::execution_manifest::RewriteSurface::DecodeEager)?;
        self.refuse_unqualified_gdn_verify()?;
 self.math(e,tokens,cache) }
}

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
