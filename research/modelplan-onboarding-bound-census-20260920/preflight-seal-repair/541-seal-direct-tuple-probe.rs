use memra_gguf::{config::{HfConfig,ModelConfig},model_plan::ModelPlan,model_packs,source::{TensorSource,TensorView}};
struct Forged { cfg:ModelConfig, plan:ModelPlan }
impl TensorSource for Forged {
 fn config(&self)->ModelConfig { self.cfg.clone() }
 fn find(&self,_:&str)->Option<TensorView<'_>> {None}
 fn bound_program(&self)->Option<(&ModelConfig,&ModelPlan)> {Some((&self.cfg,&self.plan))}
}
fn main() {
 let cfg=ModelConfig::from_hf(&HfConfig::parse(r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,"vocab_size":32,"max_position_embeddings":128}"#));
 let plan=model_packs::compile_for_load(&cfg).unwrap();
 let mut bad=Forged{cfg:cfg.clone(),plan:plan.clone()};bad.cfg.hidden_act=Some("relu".into());
 assert!(model_packs::compile_for_load(&bad.cfg).is_err());
 let unsupported_accepted=model_packs::compile_for_source(&bad).is_ok();
 let mut swapped=Forged{cfg,plan};swapped.plan.vocab_size+=1;
 let mismatched_accepted=model_packs::compile_for_source(&swapped).is_ok_and(|(_,p)|p.vocab_size==swapped.plan.vocab_size);
 println!("unsupported_config_accepted={unsupported_accepted} mismatched_plan_accepted={mismatched_accepted}");
 assert!(!unsupported_accepted && !mismatched_accepted,"unbound tuple bypassed canonical preflight");
}
