use memra_gguf::{
    bound_source::PreparedModelSource,
    checkpoint_binding::{RecordingSource, bind_loader_source},
    config::{HfConfig, ModelConfig},
    model_packs,
    source::{Hy3RepackSource, SafetensorsSource, TensorSource, TensorView},
    tensor_contract::{CheckpointDialect, TensorId, TensorMatch},
};
use std::{collections::BTreeMap, path::PathBuf};

struct Fixture(PathBuf);
impl Fixture {
    fn new(model: &str, tied: bool) -> Self {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "memra-recorded-binding-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let raw = format!(
            r#"{{"model_type":{model:?},"tie_word_embeddings":{tied},"num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":32,"vocab_size":32,"max_position_embeddings":64,"rms_norm_eps":0.000001,"rope_theta":10000,"layer_types":["full_attention"]}}"#
        );
        let cfg = ModelConfig::from_hf(&HfConfig::try_parse(&raw).unwrap());
        let pack = model_packs::for_config(&cfg).unwrap();
        let plan = pack.compile_plan(&cfg).unwrap();
        let contract = pack
            .compile_tensor_contract(
                &cfg,
                &plan,
                CheckpointDialect::HfSafetensors,
                pack.contract_options(&cfg),
            )
            .unwrap();
        let mut rows = BTreeMap::new();
        let mut data = Vec::new();
        for requirement in contract.requirements.iter().filter(|r| r.required) {
            let names = if requirement.match_mode == TensorMatch::All {
                &requirement.names[..]
            } else {
                &requirement.names[..1]
            };
            for name in names {
                let start = data.len();
                data.extend(
                    1.0f32
                        .to_le_bytes()
                        .repeat(requirement.shape.iter().product::<u64>() as usize),
                );
                rows.insert(name, serde_json::json!({"dtype":"F32","shape":requirement.shape,"data_offsets":[start,data.len()]}));
            }
        }
        let header = serde_json::to_vec(&rows).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        std::fs::write(root.join("model.safetensors"), bytes).unwrap();
        std::fs::write(root.join("config.json"), raw).unwrap();
        Self(root)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn recording_preserves_sealed_single_source_and_refusals() {
    let f = Fixture::new("qwen3", false);
    let src = SafetensorsSource::open(&f.0).unwrap();
    PreparedModelSource::text(&src)
        .unwrap()
        .with_runtime(|runtime| {
            let recorded = RecordingSource::new(runtime);
            let (cfg, plan) = model_packs::compile_for_source(&recorded).unwrap();
            assert_eq!(
                recorded.runtime_metadata().unwrap(),
                runtime.runtime_metadata().unwrap()
            );
            assert_eq!(
                recorded.bound_tensor_charges().unwrap(),
                runtime.bound_tensor_charges().unwrap()
            );
            assert_eq!(
                recorded.artifact_sha256().unwrap(),
                runtime.artifact_sha256().unwrap()
            );
            assert!(recorded.try_find("unplanned.weight").is_err());
            let binding = bind_loader_source(&recorded, &cfg, &plan).unwrap();
            let name = binding.require_ggml(&TensorId::OutputProjection).unwrap();
            assert!(recorded.try_find(&name).unwrap().is_some());
            assert_eq!(binding.output_head_ggml_name().unwrap(), "output.weight");
        })
        .unwrap();
}

#[test]
fn composite_consumer_keeps_component_dialects_and_overlay_bytes() {
    let f = Fixture::new("qwen3", false);
    let overlay = f.0.join("overlay");
    std::fs::create_dir(&overlay).unwrap();
    let expected = 1.5f32.to_le_bytes().repeat(32);
    std::fs::write(overlay.join("norm.bin"), &expected).unwrap();
    std::fs::write(overlay.join("manifest.json"), r#"{"format":"memra-expert-overlay-v2","source_dir":"..","tensors":{"output_norm.weight":{"file":"norm.bin","qtype":"F32","ne":[32],"bytes":128}}}"#).unwrap();
    let src = Hy3RepackSource::open(&overlay).unwrap();
    PreparedModelSource::text(&src)
        .unwrap()
        .with_runtime(|runtime| {
            let recorded = RecordingSource::new(runtime);
            assert!(recorded.tensor_census().is_err());
            let inventory = recorded.physical_tensor_inventory().unwrap();
            assert_eq!(inventory.components.len(), 2);
            assert_ne!(
                inventory.components[0].census.dialect,
                inventory.components[1].census.dialect
            );
            let (cfg, plan) = model_packs::compile_for_source(&recorded).unwrap();
            let binding = bind_loader_source(&recorded, &cfg, &plan).unwrap();
            assert_eq!(
                recorded
                    .try_find(&binding.require_ggml(&TensorId::OutputNorm).unwrap())
                    .unwrap()
                    .unwrap()
                    .bytes
                    .as_ref(),
                expected
            );
            assert!(recorded.try_find("blk.2.attn_q.weight").is_err());
            assert!(recorded.bound_tensor_charges().unwrap().is_some());
        })
        .unwrap();
}

#[test]
fn compiled_head_ownership_reaches_the_recording_consumer() {
    let f = Fixture::new("qwen3", true);
    let src = SafetensorsSource::open(&f.0).unwrap();
    PreparedModelSource::text(&src)
        .unwrap()
        .with_runtime(|runtime| {
            let recorded = RecordingSource::new(runtime);
            let (cfg, plan) = model_packs::compile_for_source(&recorded).unwrap();
            let binding = bind_loader_source(&recorded, &cfg, &plan).unwrap();
            assert_eq!(
                binding.output_head_ggml_name().unwrap(),
                "token_embd.weight"
            );
            assert!(recorded.try_find("output.weight").unwrap().is_some());
        })
        .unwrap();
}

#[test]
fn recording_does_not_fall_back_to_infallible_or_raw_access() {
    struct Refusing;
    impl TensorSource for Refusing {
        fn config(&self) -> ModelConfig {
            panic!("unexpected config read")
        }
        fn find(&self, _: &str) -> Option<TensorView<'_>> {
            panic!("infallible fallback")
        }
        fn try_find(&self, _: &str) -> Result<Option<TensorView<'_>>, String> {
            Err("bound read refused".into())
        }
        fn try_canonical_nvfp4_bank(
            &self,
            _: &str,
        ) -> Result<Option<memra_gguf::bound_disk::BoundDiskView>, String> {
            Err("producer refused".into())
        }
        fn try_find_expert_disk(
            &self,
            _: &str,
        ) -> Result<Option<memra_gguf::bound_disk::ExpertDiskView>, String> {
            Err("scoped disk refused".into())
        }
    }
    let recorded = RecordingSource::new(&Refusing);
    assert_eq!(
        recorded.try_find("x").err().as_deref(),
        Some("bound read refused")
    );
    assert_eq!(
        recorded.try_canonical_nvfp4_bank("x").err().as_deref(),
        Some("producer refused")
    );
    assert_eq!(
        recorded.try_find_expert_disk("x").err().as_deref(),
        Some("scoped disk refused")
    );
}

#[test]
fn inspection_and_prepared_sources_share_declared_head_ownership() {
    for tied in [false, true] {
        let f = Fixture::new("qwen3", tied);
        let src = SafetensorsSource::open(&f.0).unwrap();
        let cfg = src.try_config().unwrap();
        let pack = model_packs::for_config(&cfg).unwrap();
        let plan = pack.compile_plan(&cfg).unwrap();
        let census = src.tensor_census().unwrap();
        let inspection =
            memra_gguf::checkpoint_binding::bind_declared_census(pack, &cfg, &plan, &census)
                .unwrap();
        PreparedModelSource::text(&src)
            .unwrap()
            .with_runtime(|runtime| {
                let binding = bind_loader_source(runtime, &cfg, &plan).unwrap();
                assert_eq!(
                    inspection.output_head_ggml_name().unwrap(),
                    binding.output_head_ggml_name().unwrap()
                );
            })
            .unwrap();
        if !tied {
            let mut undeclared = cfg.clone();
            undeclared.tie_word_embeddings = None;
            let mut missing = census.clone();
            missing
                .tensors
                .retain(|row| row.entry.name != "lm_head.weight");
            assert!(
                memra_gguf::checkpoint_binding::bind_declared_census(
                    pack,
                    &undeclared,
                    &plan,
                    &missing,
                )
                .is_err(),
                "missing output must not choose tied ownership"
            );
        }
    }
}
