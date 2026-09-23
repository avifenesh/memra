use std::collections::BTreeMap;
use memra_gguf::{bound_source::BoundTensorSource,config::{HfConfig,ModelConfig},model_packs,source::SafetensorsSource,surface_catalog::LoadScope,tensor_contract::{CheckpointDialect,ContractOptions,TensorId}};
struct T {shape:Vec<u64>,dtype:&'static str,bytes:Vec<u8>}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<_>=std::env::args().collect();
 let base=std::fs::read_to_string("crates/memra-gguf/src/model_packs/step35/contract-fixture.json")?;
 let raw=base.replacen('{',r#"{"vision_config":{"model_type":"perception_encoder","width":4,"layers":2,"heads":1,"image_size":4,"patch_size":2,"mlp_ratio":2.0},"#,1);
 let cfg=ModelConfig::from_hf(&HfConfig::parse(&raw));
 let pack=model_packs::for_config(&cfg).unwrap();let plan=pack.compile_plan(&cfg)?;
 let mut contract=pack.compile_tensor_contract(&cfg,&plan,CheckpointDialect::HfSafetensors,ContractOptions::default())?;
 for s in pack.additional_inventory(&cfg,CheckpointDialect::HfSafetensors,Some(&raw))? {contract.requirements.extend(s.requirements);}
 let mut ts=BTreeMap::new();
 for r in contract.requirements.iter().filter(|r|r.required && !matches!(r.id,TensorId::QuantAux{..})) {
  for name in &r.names {ts.insert(name.clone(),T{shape:r.shape.clone(),dtype:"F32",bytes:(0..r.shape.iter().product::<u64>()).flat_map(|_|0.25f32.to_le_bytes()).collect()});}
 }
 let bad=args.get(1).map(String::as_str)==Some("bad");
 if bad {ts.insert("vision_model.conv1.weight_scale".into(),T{shape:vec![13],dtype:"I64",bytes:vec![0;104]});}
 let dir=std::env::temp_dir().join(format!("memra-1770-aux-probe-{}",std::process::id()));std::fs::create_dir(&dir)?;
 let mut entries=Vec::new();let mut payload=Vec::new();
 for (name,t) in ts {let start=payload.len();payload.extend(t.bytes);entries.push(format!("{name:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{}]}}",t.dtype,t.shape,payload.len()));}
 let h=format!("{{{}}}",entries.join(","));let mut bytes=(h.len() as u64).to_le_bytes().to_vec();bytes.extend(h.as_bytes());bytes.extend(payload);
 std::fs::write(dir.join("config.json"),raw)?;std::fs::write(dir.join("model.safetensors"),bytes)?;
 let source=SafetensorsSource::open(&dir)?;
 let result=BoundTensorSource::compile_for_scope(&source,LoadScope::Text);
 if let Ok(bound)=&result {let record=bound.census().tensors.iter().find(|r|r.physical_name=="vision_model.conv1.weight").unwrap();println!("compile accepted; vision owner aux={:?}",record.auxiliaries);}
 else {println!("compile rejected: {}",result.as_ref().err().unwrap());}
 std::fs::remove_dir_all(dir)?;
 if bad {assert!(result.is_err(),"undeclared integer vision auxiliary must fail exact inventory schema");} else {assert!(result.is_ok(),"valid original fixture must compile");}
 Ok(())
}
