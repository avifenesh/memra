use memra_gguf::{config::{HfConfig, ModelConfig}, model_packs, execution_manifest::execution_rewrites};
fn main() {
 for path in std::env::args().skip(1) {
  let text=std::fs::read_to_string(&path).unwrap();
  let hf=HfConfig::parse(&text);
  let cfg=ModelConfig::from_hf(&hf);
  let plan=match model_packs::compile_for_load(&cfg) { Ok(plan)=>execution_rewrites(&plan)[0].plan_sha256.clone(), Err(error)=>format!("REFUSE:{error}") };
  println!("{path}\t{plan}\t{hf:?}");
 }
}
