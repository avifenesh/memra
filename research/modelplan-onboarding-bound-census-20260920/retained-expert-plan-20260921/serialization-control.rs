use memra_gguf::{config::{HfConfig, ModelConfig}, model_plan::ModelPlan, execution_manifest::execution_rewrites};
use sha2::{Digest, Sha256};
fn main() {
    let cases = [
        ("qwen-moe", r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":64,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,"shared_expert_intermediate_size":48}"#),
        ("hy3-mtp", r#"{"model_type":"hy_v3","num_hidden_layers":2,"num_nextn_predict_layers":1,"hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"vocab_size":32,"max_position_embeddings":32,"first_k_dense_replace":1,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,"num_shared_experts":1,"moe_router_use_sigmoid":true,"moe_router_enable_expert_bias":true,"route_norm":true,"router_scaling_factor":2.826,"qk_norm":true}"#),
    ];
    for (name, config) in cases {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(config));
        let plan = ModelPlan::compile(&cfg).unwrap();
        println!("CASE\t{name}\t{:x}\t{:x}", Sha256::digest(format!("{plan:#?}\n").as_bytes()), Sha256::digest(format!("{:?}", execution_rewrites(&plan)).as_bytes()));
    }
}
