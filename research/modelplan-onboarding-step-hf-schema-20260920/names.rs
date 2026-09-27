// Metadata-name probe only: no tensor payload reads, no shape or native qualification.
use memra_gguf::config::{HfConfig,ModelConfig};
use memra_gguf::tensor_contract::{CheckpointDialect,ContractOptions,TensorId};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path=std::env::args().nth(1).ok_or("config.json path required")?;
    let cfg=ModelConfig::from_hf(&HfConfig::parse(&std::fs::read_to_string(path)?));
    let pack=memra_gguf::model_packs::for_config(&cfg).ok_or("no model pack")?;
    let plan=pack.compile_plan(&cfg)?;
    let contract=pack.compile_tensor_contract(&cfg,&plan,CheckpointDialect::HfSafetensors,ContractOptions::default())?;
    for r in contract.requirements {
        if matches!(r.id,TensorId::QuantAux{..}) {continue;}
        for name in r.names {println!("{}\t{:?}",name,r.id);}
    }
    Ok(())
}
