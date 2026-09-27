use memra_gguf::{config::{HfConfig, ModelConfig}, model_packs};
fn main() {
 let base=r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,"vocab_size":32,"max_position_embeddings":128}"#;
 for (name, field) in [("literal", r#""hidden_act":"relu","#), ("escaped", r#""hidden\u005fact":"relu","#)] {
  let text=base.replacen('{',&format!("{{{field}"),1);
  let cfg=ModelConfig::from_hf(&HfConfig::parse(&text));
  match model_packs::compile_for_load(&cfg) { Ok(plan)=>println!("{name}: accepted {:?}",plan.layers[0].mlp),Err(error)=>println!("{name}: refused {error}") }
 }
}
