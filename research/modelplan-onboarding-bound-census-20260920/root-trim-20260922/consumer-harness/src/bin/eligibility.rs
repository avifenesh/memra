use memra_gguf::{
    GgufFile,
    bound_source::PreparedModelSource,
    config::ModelConfig,
    execution_manifest::{RewriteSurface, execution_rewrites},
    source::GgufSource,
};
fn main() {
    let path = std::env::args().nth(1).expect("metadata witness path");
    let file = GgufFile::open(&path).unwrap();
    assert_eq!(
        file.tensors.len(),
        1,
        "witness must remain an incomplete header subset"
    );
    assert_eq!(file.tensors[0].name, "token_embd.weight");
    let cfg = ModelConfig::from_gguf(&file);
    assert!(cfg.step35.is_none());
    assert!(cfg.n_vocab > 0);
    let plan = memra_gguf::model_packs::compile_for_load(&cfg).unwrap();
    let rows=execution_rewrites(&plan).into_iter().filter(|r|matches!(r.surface,RewriteSurface::DecodeEager|RewriteSurface::MtpSpec)).map(|r|serde_json::json!({"id":r.id,"surface":format!("{:?}",r.surface),"eligible":r.eligible(),"plan_sha256":r.plan_sha256,"blockers":format!("{:?}",r.blockers),"operations":format!("{:?}",r.canonical_operations)})).collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    let eligible = rows.iter().all(|r| r["eligible"] == true);
    assert!(
        PreparedModelSource::text(&GgufSource(&file)).is_err(),
        "witness must not mint model authority"
    );
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"scope":"actual captured metadata/embedding-header eligibility only; absent model weights and incomplete census cannot be a load/admission artifact","arch":format!("{:?}",cfg.arch),"layers":cfg.n_layer,"nextn":cfg.nextn_predict_layers,"hidden":cfg.n_embd,"vocab":cfg.n_vocab,"draft_source":format!("{:?}",plan.draft_source),"rewrites":rows})).unwrap());
    if !eligible {
        std::process::exit(2);
    }
}
