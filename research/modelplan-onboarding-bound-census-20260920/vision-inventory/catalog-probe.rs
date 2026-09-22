// Complete physical-header/catalog probe. No payload reads or vision execution claims.
use std::collections::BTreeMap;
use memra_gguf::config::{HfConfig,ModelConfig};
use memra_gguf::tensor_contract::{CheckpointDialect,ContractOptions,TensorOwner};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let dir=std::path::PathBuf::from(std::env::args().nth(1).ok_or("header directory required")?);
    let raw=std::fs::read_to_string(dir.join("config.json"))?;
    let cfg=ModelConfig::from_hf(&HfConfig::parse(&raw));
    let pack=memra_gguf::model_packs::for_config(&cfg).ok_or("no model pack")?;
    let plan=pack.compile_plan(&cfg)?;
    assert!(plan.vision.is_none(),"perception encoder must not be a fabricated Gemma plan");
    let mut contract=pack.compile_tensor_contract(&cfg,&plan,CheckpointDialect::HfSafetensors,ContractOptions::default())?;
    let inventory=pack.additional_inventory(&cfg,CheckpointDialect::HfSafetensors,Some(&raw))?;
    assert_eq!(inventory.len(),1);
    for surface in inventory {contract.requirements.extend(surface.requirements);}
    let mut headers=BTreeMap::new();
    for path in std::fs::read_dir(dir)? {
        let path=path?.path();if !path.to_string_lossy().ends_with(".safetensors.header.json") {continue;}
        for (name,info) in memra_gguf::safetensors::parse_header_json_checked(&std::fs::read_to_string(path)?)? {
            assert!(headers.insert(name,info).is_none(),"duplicate physical ownership");
        }
    }
    let census=memra_gguf::source::census_from_safetensors_headers(&headers)?;
    let entries:Vec<_>=census.tensors.iter().map(|r|r.entry.clone()).collect();
    let binding=contract.bind(&entries)?;
    let vision=binding.tensors.values().filter(|r|matches!(r.owner,TensorOwner::Vision(_))).count();
    println!("physical_headers={} folded_roles={} text_roles={} vision_inventory_roles={} residual_unclaimed=0",headers.len(),binding.tensors.len(),binding.tensors.len()-vision,vision);
    assert_eq!((headers.len(),binding.tensors.len(),vision),(1597,1471,667));
    println!("metadata_only=true no_vision_semantic_or_execution_qualification=true");
    Ok(())
}
