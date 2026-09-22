use super::*;
use crate::config::HfConfig;
use crate::source::{GgufSource, SafetensorsSource};
use crate::tensor_contract::{LayerTensor, StorageLayout, TensorMatch};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

mod draft_pair;
mod gemma_runtime;
mod head_trim;

const QWEN: &str = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":32,
"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,
"vocab_size":32,"max_position_embeddings":128}"#;
const HYBRID: &str = r#"{"model_type":"qwen3_5","num_hidden_layers":2,"hidden_size":32,
"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,
"vocab_size":32,"max_position_embeddings":128,"full_attention_interval":2,
"linear_conv_kernel_dim":3,"linear_key_head_dim":16,"linear_value_head_dim":16,
"linear_num_key_heads":1,"linear_num_value_heads":2}"#;

struct Fixture {
    dir: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}
struct Tensor {
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}
fn float(shape: Vec<u64>, value: f32) -> Tensor {
    Tensor {
        dtype: "F32",
        bytes: (0..shape.iter().product::<u64>())
            .flat_map(|_| value.to_le_bytes())
            .collect(),
        shape,
    }
}
fn fixture(config: &str, edit: impl FnOnce(&mut BTreeMap<String, Tensor>)) -> Fixture {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "memra-bound-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let cfg = ModelConfig::from_hf(&HfConfig::parse(config));
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
    let mut tensors = BTreeMap::new();
    for r in contract.requirements.iter().filter(|r| r.required) {
        let names = if r.match_mode == TensorMatch::All {
            &r.names[..]
        } else {
            &r.names[..1]
        };
        for name in names {
            tensors.insert(name.clone(), float(r.shape.clone(), 0.25));
        }
    }
    edit(&mut tensors);
    let mut rows = Vec::new();
    let mut bytes = Vec::new();
    for (name, t) in tensors {
        let start = bytes.len();
        bytes.extend(t.bytes);
        rows.push(format!(
            "{name:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{}]}}",
            t.dtype,
            t.shape,
            bytes.len()
        ));
    }
    let header = format!("{{{}}}", rows.join(","));
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend(bytes);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    Fixture { dir }
}
fn layer(tensor: LayerTensor) -> TensorId {
    TensorId::Layer { index: 0, tensor }
}
fn compile_error(source: &dyn TensorSource) -> String {
    match BoundTensorSource::compile(source) {
        Ok(_) => panic!("invalid checkpoint accepted"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn safetensors_consumes_bound_physical_name_and_transform() {
    let f = fixture(HYBRID, |ts| {
        // A wrapper prefix must be resolved once by the census, then retained as the exact target.
        let old = std::mem::take(ts);
        ts.extend(
            old.into_iter()
                .map(|(name, t)| (name.replacen("model.", "model.language_model.", 1), t)),
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::PreAttentionNorm);
    assert_eq!(
        bound.binding.tensors[&id].transform,
        TensorTransform::NormAddOne
    );
    let v = bound.tensor(&id).unwrap();
    assert!(
        v.bytes
            .chunks_exact(4)
            .all(|c| f32::from_le_bytes(c.try_into().unwrap()) == 1.25)
    );
    let legacy = source.find("blk.0.attn_norm.weight").unwrap();
    assert_eq!(v.bytes, legacy.bytes);
    assert_eq!(
        bound.tensor(&TensorId::TokenEmbedding).unwrap().ne,
        [32, 32]
    );
}

#[test]
fn separate_and_tied_heads_follow_config_not_absence() {
    let missing = fixture(QWEN, |ts| {
        ts.remove("lm_head.weight");
    });
    let source = SafetensorsSource::open(&missing.dir).unwrap();
    assert!(compile_error(&source).contains("OutputProjection"));
    let tied_cfg = QWEN.replacen('{', "{\"tie_word_embeddings\":true,", 1);
    let tied = fixture(&tied_cfg, |_| {});
    let source = SafetensorsSource::open(&tied.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    assert_eq!(bound.output_head(), OutputHead::TiedToEmbedding);
    let head = bound.tensor(&TensorId::OutputProjection).unwrap();
    let embed = bound.tensor(&TensorId::TokenEmbedding).unwrap();
    assert_eq!(head.bytes.as_ptr(), embed.bytes.as_ptr());
    let conflict = fixture(&tied_cfg, |ts| {
        ts.insert("lm_head.weight".into(), float(vec![32, 32], 0.5));
    });
    assert!(
        compile_error(&SafetensorsSource::open(&conflict.dir).unwrap())
            .contains("extra checkpoint tensors")
    );
}

struct Spy<'a> {
    source: &'a SafetensorsSource,
    reads: AtomicUsize,
}
impl TensorSource for Spy<'_> {
    fn bound_interpretation(&self) -> Result<crate::source::BoundSourceInterpretation, String> {
        self.source.bound_interpretation()
    }

    fn config(&self) -> ModelConfig {
        self.source.config()
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        self.source.tensor_census()
    }
    fn find(&self, _: &str) -> Option<TensorView<'_>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        panic!("payload read before bind")
    }
    fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
        self.source.validate_bound_metadata(r)
    }
}
#[test]
fn malformed_census_fails_before_any_payload_lookup() {
    for case in ["missing", "extra", "wrong-shape", "integer", "alias"] {
        let f = fixture(QWEN, |ts| match case {
            "missing" => {
                ts.remove("model.layers.0.self_attn.q_proj.weight");
            }
            "extra" => {
                ts.insert("undeclared.weight".into(), float(vec![1], 1.0));
            }
            "wrong-shape" => {
                ts.insert(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    float(vec![16, 64], 1.0),
                );
            }
            "integer" => {
                ts.insert(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    Tensor {
                        dtype: "I64",
                        shape: vec![32, 32],
                        bytes: vec![0; 8192],
                    },
                );
            }
            "alias" => {
                ts.insert(
                    "model.language_model.layers.0.self_attn.q_proj.weight".into(),
                    float(vec![32, 32], 1.0),
                );
            }
            _ => unreachable!(),
        });
        let source = SafetensorsSource::open(&f.dir).unwrap();
        let spy = Spy {
            source: &source,
            reads: AtomicUsize::new(0),
        };
        let error = compile_error(&spy);
        assert!(!error.is_empty());
        assert_eq!(spy.reads.load(Ordering::Relaxed), 0, "{case}");
    }
}
fn nvfp4(ts: &mut BTreeMap<String, Tensor>, macro_value: f32) {
    let stem = "model.layers.0.self_attn.q_proj";
    ts.insert(
        format!("{stem}.weight"),
        Tensor {
            dtype: "U8",
            shape: vec![32, 16],
            bytes: vec![0x22; 512],
        },
    );
    ts.insert(
        format!("{stem}.weight_scale"),
        Tensor {
            dtype: "F8_E4M3",
            shape: vec![32, 2],
            bytes: vec![0x38; 64],
        },
    );
    ts.insert(
        format!("{stem}.weight_scale_2"),
        float(vec![1], macro_value),
    );
}
#[test]
fn native_nvfp4_uses_bound_bytes_and_missing_auxiliary_refuses() {
    let f = fixture(QWEN, |ts| nvfp4(ts, 2.0));
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::Query);
    let native = bound.nvfp4(&id).unwrap().unwrap();
    assert_eq!(native.wbytes, &[0x22; 512]);
    assert_eq!((native.out_f, native.in_f), (32, 32));
    let legacy = source.find_nvfp4_native("blk.0.attn_q.weight").unwrap();
    assert_eq!(native.wbytes.as_ptr(), legacy.wbytes.as_ptr());
    assert!(bound.nvfp4(&TensorId::OutputProjection).unwrap().is_none());
    for suffix in ["weight_scale", "weight_scale_2"] {
        let f = fixture(QWEN, |ts| {
            nvfp4(ts, 2.0);
            ts.remove(&format!("model.layers.0.self_attn.q_proj.{suffix}"));
        });
        assert!(compile_error(&SafetensorsSource::open(&f.dir).unwrap()).contains("missing"));
    }
}
#[test]
fn invalid_native_payload_is_an_error_not_an_absent_representation() {
    let f = fixture(QWEN, |ts| nvfp4(ts, f32::NAN));
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    assert!(
        bound
            .nvfp4(&layer(LayerTensor::Query))
            .err()
            .unwrap()
            .contains("invalid auxiliary scale values")
    );
}
#[test]
fn metadata_digest_is_order_independent_and_covers_interpretation() {
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let digest =
        |binding: &BoundTensorContract, options: ContractOptions, census: &TensorCensus| {
            binding_digest(
                bound.config(),
                bound.plan(),
                bound.contract(),
                options,
                binding,
                census,
                &bound.interpretation,
                &bound.runtime_metadata,
                &bound.scope,
            )
        };
    let original = bound.binding_sha256();
    let mut changed = bound.binding.clone();
    changed
        .tensors
        .get_mut(&layer(LayerTensor::Query))
        .unwrap()
        .checkpoint_names = vec!["model.layers.0.self_attn.o_proj.weight".into()];
    assert_ne!(original, digest(&changed, bound.options, &bound.census));
    let mut changed = bound.binding.clone();
    changed
        .tensors
        .get_mut(&TensorId::OutputNorm)
        .unwrap()
        .transform = TensorTransform::NormAddOne;
    assert_ne!(original, digest(&changed, bound.options, &bound.census));
    assert_ne!(
        original,
        digest(
            &bound.binding,
            ContractOptions {
                output_head: OutputHead::TiedToEmbedding
            },
            &bound.census
        )
    );
    struct Reverse<'a>(&'a SafetensorsSource);
    impl TensorSource for Reverse<'_> {
        fn bound_interpretation(&self) -> Result<crate::source::BoundSourceInterpretation, String> {
            self.0.bound_interpretation()
        }

        fn config(&self) -> ModelConfig {
            self.0.config()
        }
        fn find(&self, n: &str) -> Option<TensorView<'_>> {
            self.0.find(n)
        }
        fn tensor_census(&self) -> Result<TensorCensus, String> {
            let mut c = self.0.tensor_census()?;
            c.tensors.reverse();
            Ok(c)
        }
        fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
            self.0.validate_bound_metadata(r)
        }
    }
    assert_eq!(
        original,
        BoundTensorSource::compile(&Reverse(&source))
            .unwrap()
            .binding_sha256()
    );
}
#[test]
fn gguf_binding_loads_original_opened_bytes() {
    let f = fixture(QWEN, |_| {});
    let path = f.dir.join("micro.gguf");
    crate::micro_gguf::write_glm_dsa_micro(&path, 541).unwrap();
    let (view, expected) = {
        let file = crate::GgufFile::open(&path).unwrap();
        let source = GgufSource(&file);
        let bound = BoundTensorSource::compile(&source).unwrap();
        let tensor = bound.tensor(&TensorId::TokenEmbedding).unwrap();
        let legacy = source.find("token_embd.weight").unwrap();
        assert_eq!(tensor.bytes.as_ptr(), legacy.bytes.as_ptr());
        assert_eq!(tensor.bytes, legacy.bytes);
        let identity = bound.artifact_identity().unwrap();
        assert_eq!(
            identity.opened_source_sha256,
            source.artifact_sha256().unwrap()
        );
        let runtime = bound.runtime().unwrap();
        assert_eq!(runtime.artifact_sha256().unwrap(), identity.artifact_sha256);
        let scoped_rows = runtime.gguf_tensor_metadata().unwrap().unwrap();
        let mut raw_rows = source.gguf_tensor_metadata().unwrap().unwrap();
        raw_rows.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(scoped_rows, raw_rows);
        assert_eq!(
            scoped_rows.iter().map(|r| r.physical_bytes).sum::<u64>(),
            file.tensors.iter().map(|t| t.n_bytes).sum::<u64>()
        );
        let metadata = runtime.runtime_metadata().unwrap();
        assert!(metadata.is_gguf && !metadata.is_safetensors);
        assert_eq!(metadata, source.runtime_metadata().unwrap());
        let (cfg, plan) = model_packs::compile_for_source(&runtime).unwrap();
        assert_eq!(&plan, bound.plan());
        assert_eq!(format!("{cfg:?}"), format!("{:?}", bound.config()));
        for derived in [BoundTensorView::MlaKey, BoundTensorView::MlaValue] {
            let mut request = bound.request(&TensorId::TokenEmbedding, None).unwrap();
            request.view = derived;
            assert!(source.disk_bound(&request).is_err());
        }
        let mut transformed = bound.request(&TensorId::TokenEmbedding, None).unwrap();
        transformed.transform = TensorTransform::NormAddOne;
        assert!(source.disk_bound(&transformed).is_err());
        assert!(
            source
                .try_find_expert_disk("token_embd.weight")
                .unwrap()
                .is_none()
        );
        assert!(
            runtime
                .try_find_expert_disk("token_embd.weight")
                .unwrap()
                .is_none()
        );
        assert!(!runtime.try_has_expert_mmap("token_embd.weight").unwrap());
        assert!(runtime.try_find_gguf_disk("invented.weight").is_err());
        let raw_disk = source
            .try_find_gguf_disk("token_embd.weight")
            .unwrap()
            .unwrap();
        assert_eq!(raw_disk.len(), tensor.bytes.len());
        let expected = tensor.bytes.to_vec();
        let view = bound.disk(&TensorId::TokenEmbedding).unwrap().unwrap();
        assert_eq!(view.bytes().as_ptr(), tensor.bytes.as_ptr());
        assert!(matches!(
            runtime.try_find_gguf_disk("token_embd.weight").unwrap(),
            Some(crate::bound_disk::ExpertDiskView::Bound(_))
        ));
        (view, expected)
    };
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, b"replacement is not the loaded checkpoint").unwrap();
    assert_eq!(view.bytes(), expected);
    let mut bytes = vec![0; expected.len()];
    assert_eq!(view.read_at(&mut bytes, 0).unwrap(), expected.len());
    assert_eq!(bytes, expected);
    assert!(view.read_at(&mut bytes, 1).is_err());
}
#[test]
fn duplicate_semantic_ids_and_census_names_are_rejected() {
    let cfg = ModelConfig::from_hf(&HfConfig::parse(QWEN));
    let pack = model_packs::for_config(&cfg).unwrap();
    let mut contract = pack
        .compile_tensor_contract(
            &cfg,
            &pack.compile_plan(&cfg).unwrap(),
            CheckpointDialect::HfSafetensors,
            pack.contract_options(&cfg),
        )
        .unwrap();
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let mut census: Vec<_> = source
        .tensor_census()
        .unwrap()
        .tensors
        .into_iter()
        .map(|r| r.entry)
        .collect();
    census.push(census[0].clone());
    assert!(
        contract
            .bind(&census)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    census.pop();
    contract.requirements.push(contract.requirements[0].clone());
    assert!(
        contract
            .bind(&census)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    assert!(matches!(census[0].storage, StorageLayout::Float(_)));
}

#[test]
fn auxiliary_values_and_metadata_remain_part_of_the_bound_program() {
    let f = fixture(QWEN, |ts| nvfp4(ts, 2.0));
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::Query);
    let macro_scale = bound
        .auxiliary(&id, QuantAuxTensor::WeightScale)
        .unwrap()
        .unwrap();
    assert_eq!(
        f32::from_le_bytes(macro_scale.bytes[..4].try_into().unwrap()),
        2.0
    );
    let mut changed = bound.census.clone();
    let row = changed
        .tensors
        .iter_mut()
        .find(|row| row.entry.name.ends_with("q_proj.weight"))
        .unwrap();
    row.auxiliaries[0].shape.push(1);
    assert_ne!(
        bound.binding_sha256(),
        binding_digest(
            &bound.config,
            &bound.plan,
            &bound.contract,
            bound.options,
            &bound.binding,
            &changed,
            &bound.interpretation,
            &bound.runtime_metadata,
            &bound.scope
        )
    );
    let f = fixture(QWEN, |ts| {
        nvfp4(ts, 4.0);
        let stem = "model.layers.0.self_attn.q_proj";
        let packed = ts.remove(&format!("{stem}.weight")).unwrap();
        ts.insert(format!("{stem}.weight_packed"), packed);
        let scale = ts.remove(&format!("{stem}.weight_scale_2")).unwrap();
        ts.insert(format!("{stem}.weight_global_scale"), scale);
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    assert_eq!(bound.nvfp4(&id).unwrap().unwrap().wbytes, &[0x22; 512]);
    let scale = bound
        .auxiliary(&id, QuantAuxTensor::WeightScale)
        .unwrap()
        .unwrap();
    assert_eq!(
        f32::from_le_bytes(scale.bytes[..4].try_into().unwrap()),
        0.25
    );
}

#[test]
fn fp8_scale_errors_do_not_become_native_misses() {
    for (bad_shape, bad_value) in [(false, false), (true, false), (false, true)] {
        let f = fixture(QWEN, |ts| {
            let stem = "model.layers.0.self_attn.q_proj";
            ts.insert(
                format!("{stem}.weight"),
                Tensor {
                    dtype: "F8_E4M3",
                    shape: vec![32, 32],
                    bytes: vec![0x38; 1024],
                },
            );
            ts.insert(
                format!("{stem}.weight_scale"),
                float(
                    if bad_shape { vec![3] } else { vec![1] },
                    if bad_value { f32::NAN } else { 2.0 },
                ),
            );
        });
        let source = SafetensorsSource::open(&f.dir).unwrap();
        if bad_shape {
            assert!(compile_error(&source).contains("FP8 scale shape"));
            continue;
        }
        let bound = BoundTensorSource::compile(&source).unwrap();
        let id = layer(LayerTensor::Query);
        if bad_value {
            assert!(
                bound
                    .fp8(&id)
                    .err()
                    .unwrap()
                    .contains("invalid auxiliary scale values")
            );
            assert!(bound.tensor(&id).is_err());
        } else {
            let native = bound.fp8(&id).unwrap().unwrap();
            assert_eq!(native.bytes.as_ref(), &[0x38; 1024]);
        }
    }
}

#[test]
fn replacing_a_checkpoint_path_does_not_rebind_an_open_source() {
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let before = bound
        .tensor(&TensorId::TokenEmbedding)
        .unwrap()
        .bytes
        .to_vec();
    let replacement = fixture(QWEN, |ts| {
        ts.insert("model.embed_tokens.weight".into(), float(vec![32, 32], 7.0));
    });
    std::fs::rename(
        replacement.dir.join("model.safetensors"),
        f.dir.join("model.safetensors"),
    )
    .unwrap();
    assert_eq!(
        bound
            .tensor(&TensorId::TokenEmbedding)
            .unwrap()
            .bytes
            .as_ref(),
        before
    );
    let reopened = SafetensorsSource::open(&f.dir).unwrap();
    assert_ne!(
        BoundTensorSource::compile(&reopened)
            .unwrap()
            .tensor(&TensorId::TokenEmbedding)
            .unwrap()
            .bytes
            .as_ref(),
        before
    );
}

#[test]
fn packed_scale_shape_and_ambiguous_aliases_refuse_before_materialization() {
    let f = fixture(QWEN, |ts| {
        nvfp4(ts, 2.0);
        ts.insert(
            "model.layers.0.self_attn.q_proj.weight_scale".into(),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![16, 4],
                bytes: vec![0x38; 64],
            },
        );
    });
    assert!(
        compile_error(&SafetensorsSource::open(&f.dir).unwrap())
            .contains("weight_scale shape/dtype")
    );
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let mut contract = bound.contract.clone();
    let requirement = contract
        .requirements
        .iter_mut()
        .find(|r| r.id == layer(LayerTensor::Query))
        .unwrap();
    requirement
        .names
        .push("model.layers.0.self_attn.o_proj.weight".into());
    let census: Vec<_> = bound
        .census
        .tensors
        .iter()
        .map(|r| r.entry.clone())
        .collect();
    assert!(
        contract
            .bind(&census)
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
}

#[test]
fn review_digest_distinguishes_declared_scale_layout() {
    let cfg = QWEN
        .replace("\"hidden_size\":32", "\"hidden_size\":128")
        .replace("\"num_attention_heads\":2", "\"num_attention_heads\":8")
        .replace("\"intermediate_size\":64", "\"intermediate_size\":256");
    let f = fixture(&cfg, |ts| {
        let stem = "model.layers.0.self_attn.q_proj";
        ts.insert(
            format!("{stem}.weight"),
            Tensor {
                dtype: "U8",
                shape: vec![128, 64],
                bytes: vec![0x22; 8192],
            },
        );
        ts.insert(
            format!("{stem}.weight_scale"),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![128, 8],
                bytes: (0..1024).map(|i| 0x38 + (i % 8) as u8).collect(),
            },
        );
        ts.insert(format!("{stem}.weight_scale_2"), float(vec![1], 2.0));
    });
    let linear = SafetensorsSource::open(&f.dir).unwrap();
    std::fs::write(
        f.dir.join("LAYOUT.json"),
        r#"{"nvfp4_scale":"Swizzle32x4x4"}"#,
    )
    .unwrap();
    let swizzled = SafetensorsSource::open(&f.dir).unwrap();
    let a = BoundTensorSource::compile(&linear).unwrap();
    let b = BoundTensorSource::compile(&swizzled).unwrap();
    let id = layer(LayerTensor::Query);
    assert_ne!(
        a.nvfp4(&id).unwrap().unwrap().wscale,
        b.nvfp4(&id).unwrap().unwrap().wscale,
        "must exercise different interpretations"
    );
    assert_ne!(
        a.binding_sha256(),
        b.binding_sha256(),
        "different scale-layout interpretation must change binding digest"
    );
}

#[test]
fn review_input_scale_shape_rejected_at_compile() {
    let f = fixture(QWEN, |ts| {
        nvfp4(ts, 2.0);
        ts.insert(
            "model.layers.0.self_attn.q_proj.input_scale".into(),
            float(vec![7], 2.0),
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let compiled = BoundTensorSource::compile(&source);
    if let Ok(bound) = &compiled {
        let aux = bound
            .auxiliary(&layer(LayerTensor::Query), QuantAuxTensor::InputScale)
            .unwrap()
            .unwrap();
        eprintln!(
            "accepted input_scale ne={:?}, bytes={}",
            aux.ne,
            aux.bytes.len()
        );
    }
    assert!(
        compiled.is_err(),
        "invalid NVFP4 input-scale shape should fail compile"
    );
}

#[test]
fn review_native_rejects_inverse_scale_overflow() {
    let f = fixture(QWEN, |ts| {
        nvfp4(ts, f32::from_bits(1));
        let stem = "model.layers.0.self_attn.q_proj";
        let packed = ts.remove(&format!("{stem}.weight")).unwrap();
        ts.insert(format!("{stem}.weight_packed"), packed);
        let scale = ts.remove(&format!("{stem}.weight_scale_2")).unwrap();
        ts.insert(format!("{stem}.weight_global_scale"), scale);
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::Query);
    assert!(bound.auxiliary(&id, QuantAuxTensor::WeightScale).is_err());
    assert!(
        bound.nvfp4(&id).is_err(),
        "malformed macro scale must be an error for native access too"
    );
}

#[test]
fn review_awq_input_scale_follows_bound_column_transform() {
    let cfg = HYBRID
        .replace("\"linear_num_key_heads\":1", "\"linear_num_key_heads\":2")
        .replace(
            "\"linear_num_value_heads\":2",
            "\"linear_num_value_heads\":4",
        );
    let f = fixture(&cfg, |ts| {
        let stem = "model.layers.0.linear_attn.out_proj";
        ts.insert(
            format!("{stem}.weight"),
            Tensor {
                dtype: "F32",
                shape: vec![32, 64],
                bytes: (0..2048)
                    .flat_map(|i| ((i % 64 + 1) as f32).to_le_bytes())
                    .collect(),
            },
        );
        ts.insert(
            format!("{stem}.pre_quant_scale"),
            Tensor {
                dtype: "F32",
                shape: vec![64],
                bytes: (1..=64).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            },
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::GdnOutput);
    assert_eq!(
        bound.binding().tensors[&id].transform,
        TensorTransform::OutReorderColumns
    );
    let weight = bound.tensor(&id).unwrap();
    let scale = bound
        .auxiliary(&id, QuantAuxTensor::PreQuantScale)
        .unwrap()
        .unwrap();
    let values = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    eprintln!("bound weight first-row={:?}", values(&weight.bytes[..256]));
    eprintln!("bound input-axis scale={:?}", values(&scale.bytes));
    assert_eq!(
        &weight.bytes[..256],
        scale.bytes.as_ref(),
        "input-axis auxiliary must follow the owner's column permutation"
    );
}

#[test]
fn review_transformed_nvfp4_bad_scale_is_not_optional_absence() {
    let f = fixture(HYBRID, |ts| {
        let stem = "model.layers.0.linear_attn.out_proj";
        ts.insert(
            format!("{stem}.weight"),
            Tensor {
                dtype: "U8",
                shape: vec![32, 16],
                bytes: vec![0x22; 512],
            },
        );
        ts.insert(
            format!("{stem}.weight_scale"),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![32, 2],
                bytes: vec![0x38; 64],
            },
        );
        ts.insert(format!("{stem}.weight_scale_2"), float(vec![1], f32::NAN));
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::GdnOutput);
    assert!(
        bound.nvfp4(&id).is_err(),
        "NaN scale must not become Ok(None) just because the native view cannot represent a transform"
    );
}

#[test]
fn review_repair_valid_linear_and_swizzled_layouts_preserve_native_values() {
    let cfg = QWEN
        .replace("\"hidden_size\":32", "\"hidden_size\":128")
        .replace("\"num_attention_heads\":2", "\"num_attention_heads\":8")
        .replace("\"intermediate_size\":64", "\"intermediate_size\":256");
    let linear_scales: Vec<u8> = (0..1024).map(|i| 0x38 + (i % 8) as u8).collect();
    let make = |swizzled: bool| {
        fixture(&cfg, |ts| {
            let stem = "model.layers.0.self_attn.q_proj";
            ts.insert(
                format!("{stem}.weight"),
                Tensor {
                    dtype: "U8",
                    shape: vec![128, 64],
                    bytes: vec![0x22; 8192],
                },
            );
            ts.insert(
                format!("{stem}.weight_scale"),
                Tensor {
                    dtype: "F8_E4M3",
                    shape: vec![128, 8],
                    bytes: if swizzled {
                        crate::nvfp4_repack::swizzle_blockscale(&linear_scales, 128, 128)
                    } else {
                        linear_scales.clone()
                    },
                },
            );
            ts.insert(format!("{stem}.weight_scale_2"), float(vec![1], 2.0));
        })
    };
    let linear_fixture = make(false);
    let swizzled_fixture = make(true);
    std::fs::write(
        swizzled_fixture.dir.join("LAYOUT.json"),
        r#"{"nvfp4_scale":"Swizzle32x4x4"}"#,
    )
    .unwrap();
    let linear = SafetensorsSource::open(&linear_fixture.dir).unwrap();
    let swizzled = SafetensorsSource::open(&swizzled_fixture.dir).unwrap();
    let a = BoundTensorSource::compile(&linear).unwrap();
    let b = BoundTensorSource::compile(&swizzled).unwrap();
    let id = layer(LayerTensor::Query);
    assert_eq!(
        a.nvfp4(&id).unwrap().unwrap().wscale,
        b.nvfp4(&id).unwrap().unwrap().wscale
    );
    assert_eq!(
        a.nvfp4(&id).unwrap().unwrap().wscale.as_ref(),
        linear_scales
    );
    assert_ne!(a.binding_sha256(), b.binding_sha256());
    let frozen = a.binding_sha256().to_string();
    std::fs::write(
        linear_fixture.dir.join("LAYOUT.json"),
        r#"{"nvfp4_scale":"Swizzle32x4x4"}"#,
    )
    .unwrap();
    assert_eq!(
        BoundTensorSource::compile(&linear)
            .unwrap()
            .binding_sha256(),
        frozen
    );
}

#[test]
fn review_repair_valid_input_scales_and_transformed_native_absence() {
    for shape in [vec![], vec![1]] {
        let f = fixture(QWEN, |ts| {
            nvfp4(ts, 2.0);
            ts.insert(
                "model.layers.0.self_attn.q_proj.input_scale".into(),
                float(shape.clone(), 3.0),
            );
        });
        let source = SafetensorsSource::open(&f.dir).unwrap();
        let bound = BoundTensorSource::compile(&source).unwrap();
        let id = layer(LayerTensor::Query);
        assert!(bound.nvfp4(&id).unwrap().is_some());
        let scale = bound
            .auxiliary(&id, QuantAuxTensor::InputScale)
            .unwrap()
            .unwrap();
        assert_eq!(scale.bytes.as_ref(), 3.0f32.to_le_bytes());
    }
    let f = fixture(HYBRID, |ts| {
        let stem = "model.layers.0.linear_attn.out_proj";
        ts.insert(
            format!("{stem}.weight"),
            Tensor {
                dtype: "U8",
                shape: vec![32, 16],
                bytes: vec![0x22; 512],
            },
        );
        ts.insert(
            format!("{stem}.weight_scale"),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![32, 2],
                bytes: vec![0x38; 64],
            },
        );
        ts.insert(format!("{stem}.weight_scale_2"), float(vec![1], 2.0));
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::GdnOutput);
    assert!(bound.nvfp4(&id).unwrap().is_none());
    assert!(
        bound
            .auxiliary(&id, QuantAuxTensor::WeightScale)
            .unwrap()
            .is_some()
    );
}

#[test]
fn review_repair_bf16_awq_permutation_preserves_encoding() {
    let cfg = HYBRID
        .replace("\"linear_num_key_heads\":1", "\"linear_num_key_heads\":2")
        .replace(
            "\"linear_num_value_heads\":2",
            "\"linear_num_value_heads\":4",
        );
    let original: Vec<u8> = (1..=64)
        .flat_map(|i| ((i as f32).to_bits() >> 16).to_le_bytes()[..2].to_vec())
        .collect();
    let f = fixture(&cfg, |ts| {
        ts.insert(
            "model.layers.0.linear_attn.out_proj.pre_quant_scale".into(),
            Tensor {
                dtype: "BF16",
                shape: vec![64],
                bytes: original.clone(),
            },
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let scale = bound
        .auxiliary(
            &layer(LayerTensor::GdnOutput),
            QuantAuxTensor::PreQuantScale,
        )
        .unwrap()
        .unwrap();
    assert_eq!(scale.ggml_type, crate::GgmlType::BF16);
    assert_eq!(scale.bytes.len(), original.len());
    assert_ne!(scale.bytes.as_ref(), original);
    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|b| crate::dequant::bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
            .collect::<Vec<_>>()
    };
    let expected: Vec<f32> = (1..=16)
        .chain(33..=48)
        .chain(17..=32)
        .chain(49..=64)
        .map(|x| x as f32)
        .collect();
    assert_eq!(decode(&scale.bytes), expected);
}

#[test]
fn review_repair_digest_includes_captured_dtype_preservation_metadata() {
    let f = fixture(QWEN, |_| {});
    let original = SafetensorsSource::open(&f.dir).unwrap();
    let config=QWEN.replacen('{',r#"{"quantization_config":{"modules_to_not_convert":["model.layers.0"],"quant_algo":"NVFP4"},"#,1);
    std::fs::write(f.dir.join("config.json"), config).unwrap();
    let changed = SafetensorsSource::open(&f.dir).unwrap();
    let a = BoundTensorSource::compile(&original).unwrap();
    let b = BoundTensorSource::compile(&changed).unwrap();
    assert_eq!(format!("{:?}", a.config()), format!("{:?}", b.config()));
    assert_eq!(a.census(), b.census());
    assert_ne!(a.binding_sha256(), b.binding_sha256());
}

#[test]
fn runtime_adapter_uses_semantic_roles_and_refuses_unrecognized_abi_names() {
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    for (name, id) in [
        ("token_embd.weight", TensorId::TokenEmbedding),
        ("output.weight", TensorId::OutputProjection),
        ("blk.0.attn_q.weight", layer(LayerTensor::Query)),
    ] {
        assert!(runtime.try_has(name).unwrap());
        assert_eq!(
            runtime.try_find(name).unwrap().unwrap().bytes,
            bound.tensor(&id).unwrap().bytes
        );
    }
    assert!(!runtime.try_has("blk.0.ffn_gate_up_exps.weight").unwrap());
    assert!(runtime.try_has("blk.0.misspelled.weight").is_err());
    assert_eq!(
        runtime.bound_interpretation().unwrap(),
        *bound.interpretation()
    );
}

#[test]
fn runtime_adapter_propagates_native_errors_and_tied_head_ownership() {
    let tied = QWEN.replacen('{', "{\"tie_word_embeddings\":true,", 1);
    let f = fixture(&tied, |ts| nvfp4(ts, f32::NAN));
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    assert!(
        runtime
            .try_find_nvfp4_native("blk.0.attn_q.weight")
            .is_err()
    );
    assert!(runtime.try_find("blk.0.attn_q.scale").is_err());
    let output = runtime.try_find("output.weight").unwrap().unwrap();
    let embed = runtime.try_find("token_embd.weight").unwrap().unwrap();
    assert_eq!(output.bytes.as_ptr(), embed.bytes.as_ptr());
}

#[test]
fn runtime_adapter_resolves_expert_members_without_fabricating_a_stacked_bank() {
    let cfg = QWEN.replace("\"qwen3\"", "\"qwen3_moe\"").replacen(
        '{',
        "{\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,",
        1,
    );
    let f = fixture(&cfg, |ts| {
        ts.insert(
            "model.layers.0.mlp.experts.1.gate_proj.weight".into(),
            float(vec![32, 32], 0.75),
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    assert!(!runtime.try_has("blk.0.ffn_gate_exps.weight").unwrap());
    assert!(runtime.try_has("blk.0.ffn_gate_exps.1.weight").unwrap());
    let value = runtime
        .try_find("blk.0.ffn_gate_exps.1.weight")
        .unwrap()
        .unwrap();
    assert!(
        value
            .bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 0.75)
    );
    assert!(runtime.try_find("blk.0.ffn_gate_exps.9.weight").is_err());
}

#[test]
fn runtime_adapter_derives_mla_planes_from_the_bound_fused_tensor() {
    let f = fixture(include_str!("mla-fixture.json"), |ts| {
        let t = ts
            .get_mut("model.layers.1.self_attn.kv_b_proj.weight")
            .unwrap();
        t.bytes = (0..t.shape.iter().product::<u64>())
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    for name in ["blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight"] {
        let v = runtime.try_find(name).unwrap().unwrap();
        let legacy = source.find(name).unwrap();
        assert_eq!(v.ne, legacy.ne);
        assert_eq!(v.bytes, legacy.bytes);
    }
    let k = runtime.try_find("blk.1.attn_k_b.weight").unwrap().unwrap();
    let values: Vec<f32> = k
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        values,
        [0., 3., 1., 4., 2., 5., 9., 12., 10., 13., 11., 14.]
    );
}

#[test]
fn ordered_optional_reads_never_convert_an_error_into_absence() {
    struct Refusing;
    impl TensorSource for Refusing {
        fn config(&self) -> ModelConfig {
            unreachable!()
        }
        fn find(&self, _: &str) -> Option<TensorView<'_>> {
            unreachable!()
        }
        fn try_find(&self, name: &str) -> Result<Option<TensorView<'_>>, String> {
            assert_eq!(name, "first");
            Err("bound payload refused".into())
        }
    }
    assert_eq!(
        Refusing
            .try_find_first(&["first", "second"])
            .err()
            .as_deref(),
        Some("bound payload refused")
    );
}

#[test]
fn step_hf_contract_binds_native_banks_private_heads_and_norm_transforms() {
    let f = fixture(
        include_str!("../model_packs/step35/contract-fixture.json"),
        |ts| {
            for (suffix, shape) in [("norm.weight", vec![16]), ("output.weight", vec![64, 16])] {
                ts.insert(
                    format!("model.layers.3.transformer.shared_head.{suffix}"),
                    float(shape, 0.75),
                );
            }
            for il in [1, 2] {
                for proj in ["gate", "up", "down"] {
                    let stem = format!("model.layers.{il}.moe.{proj}_proj");
                    let old = ts.get(&format!("{stem}.weight")).unwrap();
                    let shape = old.shape.clone();
                    let count = shape.iter().product::<u64>() as usize;
                    let grid = vec![shape[0], shape[1].div_ceil(128), shape[2].div_ceil(128)];
                    ts.insert(
                        format!("{stem}.weight"),
                        Tensor {
                            dtype: "F8_E4M3",
                            shape,
                            bytes: vec![0x38; count],
                        },
                    );
                    ts.insert(format!("{stem}.weight_scale_inv"), float(grid, 2.0));
                }
            }
        },
    );
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    assert!(matches!(
        bound.plan().mtp_blocks[0].layer.mlp,
        crate::model_plan::MlpPlan::Dense(_)
    ));
    let bank = runtime
        .try_find_fp8_stacked_native("blk.1.ffn_gate_exps.weight")
        .unwrap()
        .unwrap();
    assert_eq!((bank.n_expert, bank.out_f, bank.in_f), (6, 12, 16));
    let disk = runtime
        .try_find_expert_disk("blk.1.ffn_gate_exps.weight")
        .unwrap()
        .unwrap();
    let crate::bound_disk::ExpertDiskView::Bound(view) = disk else {
        panic!("bound runtime exported an unrestricted disk extent");
    };
    assert_eq!(view.bytes(), bank.bytes);
    let mut read = vec![0; bank.bytes.len()];
    assert_eq!(view.read_at(&mut read, 0).unwrap(), bank.bytes.len());
    assert_eq!(read, bank.bytes);
    assert!(view.read_at(&mut read, 1).is_err());
    assert!(
        runtime
            .try_has_expert_mmap("blk.1.ffn_gate_exps.weight")
            .unwrap()
    );
    assert!(
        runtime
            .try_find_expert_disk("blk.3.ffn_gate_exps.weight")
            .unwrap()
            .is_none()
    );
    assert!(runtime.try_find_expert_disk("invented.weight").is_err());

    assert!(runtime.try_has("blk.1.ffn_gate_exps.weight").unwrap());
    assert!(!runtime.try_has("blk.3.ffn_gate_exps.weight").unwrap());
    for name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.1.ffn_norm.weight",
        "blk.1.attn_gate.weight",
        "blk.3.nextn.enorm.weight",
        "blk.3.nextn.hnorm.weight",
        "blk.3.attn_norm.weight",
        "blk.3.nextn.shared_head_head.weight",
        "blk.3.nextn.shared_head_norm.weight",
        "blk.1.attn_q_norm.weight",
    ] {
        let actual = runtime.try_find(name).unwrap().unwrap();
        let legacy = source.find(name).unwrap();
        assert_eq!(actual.ne, legacy.ne, "{name}");
        assert_eq!(actual.bytes, legacy.bytes, "{name}");
    }
    let norm = runtime
        .try_find("blk.3.nextn.shared_head_norm.weight")
        .unwrap()
        .unwrap();
    assert!(
        norm.bytes
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 1.75)
    );
}

#[test]
fn step_hf_wrong_bank_and_private_head_shapes_fail_at_binding() {
    for (name, shape) in [
        ("model.layers.1.moe.gate_proj.weight", vec![6, 16, 12]),
        (
            "model.layers.3.transformer.shared_head.output.weight",
            vec![16, 64],
        ),
    ] {
        let f = fixture(
            include_str!("../model_packs/step35/contract-fixture.json"),
            |ts| {
                ts.insert(name.into(), float(shape, 1.0));
            },
        );
        let source = SafetensorsSource::open(&f.dir).unwrap();
        let error = compile_error(&source);
        assert!(error.contains(name), "{error}");
        assert!(error.contains("shape"), "{error}");
    }
}

#[test]
fn step_private_head_absence_keeps_the_planned_model_head_and_gguf_folds_stay_identity() {
    let f = fixture(
        include_str!("../model_packs/step35/contract-fixture.json"),
        |_| {},
    );
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    assert!(
        !runtime
            .try_has("blk.3.nextn.shared_head_head.weight")
            .unwrap()
    );
    assert!(runtime.try_has("output.weight").unwrap());
    let gguf =
        TensorContract::for_plan(bound.plan(), CheckpointDialect::Gguf, bound.options).unwrap();
    for id in [
        TensorId::OutputNorm,
        layer(LayerTensor::PreAttentionNorm),
        layer(LayerTensor::QueryNorm),
    ] {
        assert_eq!(
            gguf.requirements
                .iter()
                .find(|r| r.id == id)
                .unwrap()
                .transform,
            TensorTransform::Identity
        );
        assert_eq!(
            bound.binding().tensors[&id].transform,
            TensorTransform::NormAddOne
        );
    }
}

#[test]
fn bound_census_retains_physical_auxiliaries_omitted_from_the_index() {
    let f = fixture(QWEN, |ts| {
        let stem = "model.layers.0.self_attn.q_proj";
        ts.insert(
            format!("{stem}.weight"),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![32, 32],
                bytes: vec![0x38; 1024],
            },
        );
        ts.insert(format!("{stem}.weight_scale_inv"), float(vec![1], 2.0));
    });
    let single = SafetensorsSource::open(&f.dir).unwrap();
    let primary_names: Vec<_> = single
        .tensor_census()
        .unwrap()
        .tensors
        .into_iter()
        .map(|r| r.physical_name)
        .collect();
    let index = format!(
        "{{\"weight_map\":{{{}}}}}",
        primary_names
            .iter()
            .map(|n| format!("{n:?}:\"model.safetensors\""))
            .collect::<Vec<_>>()
            .join(",")
    );
    assert!(!index.contains("weight_scale_inv"));
    std::fs::write(f.dir.join("model.safetensors.index.json"), index).unwrap();
    let indexed = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&indexed).unwrap();
    let row = bound
        .census()
        .tensors
        .iter()
        .find(|r| r.entry.name.ends_with("q_proj.weight"))
        .unwrap();
    assert_eq!(row.entry.physical_bytes, 1028);
    assert_eq!(row.auxiliaries.len(), 1);
    assert_eq!(
        row.auxiliaries[0].physical_name,
        "model.layers.0.self_attn.q_proj.weight_scale_inv"
    );
    assert_eq!(row.auxiliaries[0].physical_bytes, 4);
    let view = bound.fp8(&layer(LayerTensor::Query)).unwrap().unwrap();
    assert_eq!(view.scale, 2.0);
}

#[test]
fn hybrid_full_attention_qk_norms_retain_the_declared_centered_convention() {
    let f = fixture(HYBRID, |ts| {
        for qk in ["q", "k"] {
            ts.insert(
                format!("model.layers.1.self_attn.{qk}_norm.weight"),
                float(vec![16], 0.25),
            );
        }
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    assert_eq!(
        bound.plan().layers[1].pre_attention_norm.weight_transform,
        crate::model_plan::WeightTransform::AddOne
    );
    for qk in ["q", "k"] {
        let name = format!("blk.1.attn_{qk}_norm.weight");
        let actual = runtime.try_find(&name).unwrap().unwrap();
        let legacy = source.find(&name).unwrap();
        assert_eq!(
            actual.bytes, legacy.bytes,
            "{name} must preserve the existing native program"
        );
        assert!(
            actual
                .bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 1.25)
        );
    }
}

#[test]
fn ordinary_qk_norms_and_gguf_folded_norms_are_not_shifted_again() {
    let f = fixture(QWEN, |ts| {
        for qk in ["q", "k"] {
            ts.insert(
                format!("model.layers.0.self_attn.{qk}_norm.weight"),
                float(vec![16], 0.25),
            );
        }
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    for qk in ["q", "k"] {
        let name = format!("blk.0.attn_{qk}_norm.weight");
        let actual = runtime.try_find(&name).unwrap().unwrap();
        let legacy = source.find(&name).unwrap();
        assert_eq!(actual.bytes, legacy.bytes);
        assert!(
            actual
                .bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 0.25)
        );
    }
    let cfg = ModelConfig::from_hf(&HfConfig::parse(HYBRID));
    let plan = model_packs::compile_for_load(&cfg).unwrap();
    let gguf = TensorContract::for_plan(&plan, CheckpointDialect::Gguf, ContractOptions::default())
        .unwrap();
    for tensor in [LayerTensor::QueryNorm, LayerTensor::KeyNorm] {
        let id = TensorId::Layer { index: 1, tensor };
        assert_eq!(
            gguf.requirements
                .iter()
                .find(|r| r.id == id)
                .unwrap()
                .transform,
            TensorTransform::Identity
        );
    }
}

fn step_with_vision_fixture(edit: impl FnOnce(&mut BTreeMap<String, Tensor>)) -> Fixture {
    let config=include_str!("../model_packs/step35/contract-fixture.json").replacen('{',r#"{"vision_config":{"model_type":"perception_encoder","width":4,"layers":2,"heads":1,"num_channels":3,"image_size":4,"patch_size":2,"mlp_ratio":2.0},"#,1);
    fixture(&config, |tensors| {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&config));
        let surfaces = model_packs::for_config(&cfg)
            .unwrap()
            .additional_inventory(&cfg, CheckpointDialect::HfSafetensors, Some(&config))
            .unwrap();
        for surface in surfaces {
            for r in surface.requirements {
                tensors.insert(r.names[0].clone(), float(r.shape, 0.25));
            }
        }
        edit(tensors);
    })
}

#[test]
fn scoped_text_catalog_validates_vision_without_materializing_it() {
    let f = step_with_vision_fixture(|_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    struct Count<'a> {
        source: &'a SafetensorsSource,
        reads: AtomicUsize,
    }
    impl TensorSource for Count<'_> {
        fn config(&self) -> ModelConfig {
            self.source.config()
        }
        fn raw_config_json(&self) -> Option<&str> {
            self.source.raw_config_json()
        }
        fn bound_interpretation(&self) -> Result<crate::source::BoundSourceInterpretation, String> {
            self.source.bound_interpretation()
        }
        fn tensor_census(&self) -> Result<TensorCensus, String> {
            self.source.tensor_census()
        }
        fn find(&self, name: &str) -> Option<TensorView<'_>> {
            assert_eq!(name, "rope_freqs.weight");
            None
        }
        fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
            self.source.validate_bound_metadata(r)
        }
        fn read_bound(&self, r: &BoundTensorRequest<'_>) -> Result<TensorView<'_>, String> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.source.read_bound(r)
        }
    }
    let spy = Count {
        source: &source,
        reads: AtomicUsize::new(0),
    };
    let bound = BoundTensorSource::compile_for_scope(&spy, crate::surface_catalog::LoadScope::Text)
        .unwrap();
    assert_eq!(spy.reads.load(Ordering::Relaxed), 0);
    let vision = bound
        .binding()
        .tensors
        .iter()
        .find(|(_, t)| matches!(t.owner, crate::tensor_contract::TensorOwner::Vision(_)))
        .unwrap()
        .0
        .clone();
    assert!(bound.contains(&vision));
    assert!(!bound.scope().permits(&vision));
    assert!(
        bound
            .tensor(&vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(
        bound
            .nvfp4(&vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(
        bound
            .fp8(&vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(
        bound
            .fp8_stacked(&vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(
        bound
            .nvfp4_stacked(&vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(
        bound
            .disk(&vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(
        bound
            .auxiliary(&vision, QuantAuxTensor::WeightScale)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert_eq!(spy.reads.load(Ordering::Relaxed), 0);
    bound.tensor(&TensorId::TokenEmbedding).unwrap();
    assert_eq!(spy.reads.load(Ordering::Relaxed), 1);
    assert!(compile_error(&source).contains("execution is unsupported"));
}

#[test]
fn invalid_unselected_inventory_still_refuses_text_scope() {
    for case in ["missing", "shape", "unknown", "quant"] {
        let f = step_with_vision_fixture(|ts| match case {
            "missing" => {
                ts.remove("vision_model.conv1.weight");
            }
            "shape" => {
                ts.insert(
                    "vision_model.conv1.weight".into(),
                    float(vec![4, 3, 1, 4], 0.25),
                );
            }
            "unknown" => {
                ts.insert(
                    "vision_model.unrecognized.weight".into(),
                    float(vec![1], 0.25),
                );
            }
            "quant" => {
                ts.insert(
                    "vision_model.positional_embedding".into(),
                    Tensor {
                        dtype: "F8_E4M3",
                        shape: vec![4, 4],
                        bytes: vec![0x38; 16],
                    },
                );
            }
            _ => unreachable!(),
        });
        let source = SafetensorsSource::open(&f.dir).unwrap();
        assert!(
            BoundTensorSource::compile_for_scope(&source, crate::surface_catalog::LoadScope::Text)
                .is_err(),
            "accepted {case}"
        );
    }
}

#[test]
fn scope_and_captured_inventory_declaration_are_immutable_identity_inputs() {
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let full = BoundTensorSource::compile(&source).unwrap();
    let text =
        BoundTensorSource::compile_for_scope(&source, crate::surface_catalog::LoadScope::Text)
            .unwrap();
    assert_ne!(full.binding_sha256(), text.binding_sha256());
    assert_eq!(
        full.tensor(&TensorId::TokenEmbedding).unwrap().bytes,
        text.tensor(&TensorId::TokenEmbedding).unwrap().bytes
    );
    let f = step_with_vision_fixture(|_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let original =
        BoundTensorSource::compile_for_scope(&source, crate::surface_catalog::LoadScope::Text)
            .unwrap();
    std::fs::write(f.dir.join("config.json"), "{}").unwrap();
    let again =
        BoundTensorSource::compile_for_scope(&source, crate::surface_catalog::LoadScope::Text)
            .unwrap();
    assert_eq!(original.binding_sha256(), again.binding_sha256());
}

#[test]
fn vision_materialization_preserves_float_encoding_instead_of_text_reencoding() {
    use crate::tensor_contract::VisionTensor;
    let physical = "model.vision_tower.encoder.layers.0.mlp.up_proj.weight";
    let f = fixture(QWEN, |ts| {
        ts.insert(
            physical.into(),
            Tensor {
                dtype: "BF16",
                shape: vec![1024, 1024],
                bytes: [0x80, 0x3e].repeat(1024 * 1024),
            },
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let census = source.tensor_census().unwrap();
    let record = census
        .tensors
        .iter()
        .find(|r| r.physical_name == physical)
        .unwrap();
    let id = TensorId::Vision {
        layer: Some(0),
        tensor: VisionTensor::MlpUp,
    };
    let request = BoundTensorRequest {
        id: &id,
        record,
        dialect: CheckpointDialect::HfSafetensors,
        transform: TensorTransform::Identity,
        view: BoundTensorView::Whole,
    };
    let value = source.read_bound(&request).unwrap();
    let raw = source.raw_hf(physical).unwrap();
    assert_eq!(value.ggml_type, crate::GgmlType::BF16);
    assert_eq!(value.bytes.as_ptr(), raw.bytes.as_ptr());
    assert_eq!(value.bytes, raw.bytes);
}

#[test]
fn inventory_expansion_and_unknown_component_declarations_fail_closed() {
    let cfg = ModelConfig::from_hf(&HfConfig::parse(include_str!(
        "../model_packs/step35/contract-fixture.json"
    )));
    let pack = model_packs::for_config(&cfg).unwrap();
    for extra in [
        r#""layers":4294967295"#,
        r#""model_type":"unrecognized_encoder""#,
        r#""model_type":42"#,
        r#""patch_size":0"#,
    ] {
        let raw = format!("{{\"vision_config\":{{{extra}}}}}");
        assert!(
            pack.additional_inventory(&cfg, CheckpointDialect::HfSafetensors, Some(&raw))
                .is_err(),
            "{extra}"
        );
    }
    assert!(
        pack.additional_inventory(
            &cfg,
            CheckpointDialect::Gguf,
            Some(r#"{"vision_config":{}}"#)
        )
        .is_err()
    );
}

#[test]
fn unselected_vision_rejects_undeclared_folded_auxiliaries_on_float_weights() {
    let valid = step_with_vision_fixture(|_| {});
    let valid_source = SafetensorsSource::open(&valid.dir).unwrap();
    assert!(
        BoundTensorSource::compile_for_scope(
            &valid_source,
            crate::surface_catalog::LoadScope::Text
        )
        .is_ok()
    );
    for (suffix, dtype, shape, bytes) in [
        ("weight_scale", "I64", vec![13], vec![0; 104]),
        (
            "weight_scale",
            "F32",
            vec![1],
            1.0f32.to_le_bytes().to_vec(),
        ),
        ("input_scale", "F32", vec![1], 1.0f32.to_le_bytes().to_vec()),
        (
            "pre_quant_scale",
            "F32",
            vec![2],
            1.0f32.to_le_bytes().repeat(2),
        ),
    ] {
        let f = step_with_vision_fixture(|ts| {
            ts.insert(
                format!("vision_model.conv1.{suffix}"),
                Tensor {
                    dtype,
                    shape,
                    bytes,
                },
            );
        });
        let source = SafetensorsSource::open(&f.dir).unwrap();
        let error = match BoundTensorSource::compile_for_scope(
            &source,
            crate::surface_catalog::LoadScope::Text,
        ) {
            Ok(_) => panic!("accepted undeclared {suffix}"),
            Err(e) => e.to_string(),
        };
        assert!(
            error.contains("auxiliary") && error.contains("vision_model.conv1"),
            "{error}"
        );
    }
}

#[test]
fn canonical_bind_checks_float_auxiliaries_without_a_runtime_wrapper() {
    let f = step_with_vision_fixture(|ts| {
        ts.insert(
            "vision_model.conv1.weight_scale".into(),
            Tensor {
                dtype: "I64",
                shape: vec![13],
                bytes: vec![0; 104],
            },
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let cfg = source.config();
    let pack = model_packs::for_config(&cfg).unwrap();
    let plan = pack.compile_plan(&cfg).unwrap();
    let mut contract = pack
        .compile_tensor_contract(
            &cfg,
            &plan,
            CheckpointDialect::HfSafetensors,
            pack.contract_options(&cfg),
        )
        .unwrap();
    for surface in pack
        .additional_inventory(
            &cfg,
            CheckpointDialect::HfSafetensors,
            source.raw_config_json(),
        )
        .unwrap()
    {
        contract.requirements.extend(surface.requirements);
    }
    let census = source.tensor_census().unwrap();
    let entries: Vec<_> = census.tensors.iter().map(|r| r.entry.clone()).collect();
    let error = contract.bind(&entries).unwrap_err().to_string();
    assert!(
        error.contains("auxiliary") && error.contains("vision_model.conv1.weight_scale"),
        "{error}"
    );
}

#[test]
fn composite_identity_binds_opened_bytes_and_scope_in_separate_domains() {
    let f = fixture(QWEN, |_| {});
    let changed = fixture(QWEN, |ts| {
        ts.insert("model.embed_tokens.weight".into(), float(vec![32, 32], 0.5));
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let other_source = SafetensorsSource::open(&changed.dir).unwrap();
    let full = BoundTensorSource::compile(&source).unwrap();
    let text =
        BoundTensorSource::compile_for_scope(&source, crate::surface_catalog::LoadScope::Text)
            .unwrap();
    let other = BoundTensorSource::compile(&other_source).unwrap();
    let a = full.artifact_identity().unwrap();
    let b = other.artifact_identity().unwrap();
    let t = text.artifact_identity().unwrap();
    assert_eq!(a.opened_source_sha256, source.artifact_sha256().unwrap());
    assert_eq!(a.semantic_scope_sha256, full.binding_sha256());
    assert_eq!(a.semantic_scope_sha256, b.semantic_scope_sha256);
    assert_ne!(a.opened_source_sha256, b.opened_source_sha256);
    assert_ne!(a.artifact_sha256, b.artifact_sha256);
    assert_eq!(a.opened_source_sha256, t.opened_source_sha256);
    assert_ne!(a.semantic_scope_sha256, t.semantic_scope_sha256);
    assert_ne!(a.artifact_sha256, t.artifact_sha256);
    assert_ne!(a.artifact_sha256, a.opened_source_sha256);
    assert_ne!(a.artifact_sha256, a.semantic_scope_sha256);
    assert_eq!(
        full.runtime().unwrap().artifact_sha256().unwrap(),
        a.artifact_sha256
    );
    // Captured config and opened mappings survive both pathname replacements.
    std::fs::rename(
        f.dir.join("model.safetensors"),
        f.dir.join("held.safetensors"),
    )
    .unwrap();
    std::fs::write(f.dir.join("model.safetensors"), b"replacement").unwrap();
    std::fs::write(f.dir.join("config.json"), b"replacement").unwrap();
    assert_eq!(full.artifact_identity().unwrap(), a);
}

#[test]
fn composite_identity_propagates_missing_and_malformed_source_identity() {
    struct IdentitySource<'a> {
        source: &'a SafetensorsSource,
        identity: Result<String, String>,
    }
    impl TensorSource for IdentitySource<'_> {
        fn artifact_sha256(&self) -> Result<String, String> {
            self.identity.clone()
        }
        fn config(&self) -> ModelConfig {
            self.source.config()
        }
        fn find(&self, name: &str) -> Option<TensorView<'_>> {
            self.source.find(name)
        }
        fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
            self.source.bound_interpretation()
        }
        fn tensor_census(&self) -> Result<TensorCensus, String> {
            self.source.tensor_census()
        }
        fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
            self.source.validate_bound_metadata(r)
        }
    }
    let f = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    for (identity, expected) in [
        (
            Err("opened identity unavailable".into()),
            "opened identity unavailable",
        ),
        (Ok("not-a-hash".into()), "invalid SHA-256"),
        (Ok("g".repeat(64)), "invalid SHA-256"),
    ] {
        let wrapper = IdentitySource {
            source: &source,
            identity,
        };
        let bound = BoundTensorSource::compile(&wrapper).unwrap();
        let error = bound.runtime().unwrap().artifact_sha256().unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn bound_preflight_reuses_sealed_step_program_and_handle_free_metadata() {
    let f = step_with_vision_fixture(|_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound =
        BoundTensorSource::compile_for_scope(&source, crate::surface_catalog::LoadScope::Text)
            .unwrap();
    let runtime = bound.runtime().unwrap();
    let (config, plan) = model_packs::compile_for_source(&runtime).unwrap();
    assert_eq!(format!("{config:?}"), format!("{:?}", bound.config()));
    assert_eq!(&plan, bound.plan());
    assert!(plan.vision.is_none());
    assert!(plan.multimodal.is_none());
    assert!(runtime.gguf_tensor_metadata().unwrap().is_none());
    assert!(!runtime.tensor_census().unwrap().tensors.is_empty());
    let meta = runtime.runtime_metadata().unwrap();
    assert_eq!(meta, source.runtime_metadata().unwrap());
    assert!(!meta.is_gguf && meta.is_safetensors);
    assert_eq!(
        meta.expert_activation_precision,
        runtime.expert_activation_precision()
    );
    assert_eq!(
        meta.preserve_expert_encodings,
        runtime.preserve_expert_encodings()
    );
    assert_eq!(meta.nvfp4_cache_tag, runtime.nvfp4_cache_tag());
    assert_eq!(
        runtime.artifact_sha256().unwrap(),
        bound.artifact_identity().unwrap().artifact_sha256
    );
}

#[test]
fn composite_identity_still_covers_unselected_vision_payloads() {
    let a = step_with_vision_fixture(|_| {});
    let b = step_with_vision_fixture(|ts| {
        let shape = ts["vision_model.conv1.weight"].shape.clone();
        ts.insert("vision_model.conv1.weight".into(), float(shape, 0.5));
    });
    let sa = SafetensorsSource::open(&a.dir).unwrap();
    let sb = SafetensorsSource::open(&b.dir).unwrap();
    let ba =
        BoundTensorSource::compile_for_scope(&sa, crate::surface_catalog::LoadScope::Text).unwrap();
    let bb =
        BoundTensorSource::compile_for_scope(&sb, crate::surface_catalog::LoadScope::Text).unwrap();
    let ia = ba.artifact_identity().unwrap();
    let ib = bb.artifact_identity().unwrap();
    assert_eq!(ia.semantic_scope_sha256, ib.semantic_scope_sha256);
    assert_ne!(ia.opened_source_sha256, ib.opened_source_sha256);
    assert_ne!(ia.artifact_sha256, ib.artifact_sha256);
    let vision = TensorId::Family {
        family: "step_perception_inventory",
        key: "vision_model.conv1.weight".into(),
    };
    assert!(ba.tensor(&vision).is_err());
    assert!(bb.tensor(&vision).is_err());
}

fn complete_repack_fixture(edit: impl FnOnce(&mut BTreeMap<String, Tensor>)) -> Fixture {
    complete_repack_fixture_size(32, edit)
}

fn complete_repack_fixture_size(
    width: u64,
    edit: impl FnOnce(&mut BTreeMap<String, Tensor>),
) -> Fixture {
    let config_json = QWEN
        .replace("\"hidden_size\":32", &format!("\"hidden_size\":{width}"))
        .replace("\"head_dim\":16", &format!("\"head_dim\":{}", width / 2));
    let cfg = config_json.replace("\"qwen3\"", "\"qwen3_moe\"").replacen(
        '{',
        &format!(
            "{{\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":{width},"
        ),
        1,
    );
    complete_repack_fixture_config(width, &cfg, edit)
}

fn complete_repack_fixture_config(
    width: u64,
    cfg: &str,
    edit: impl FnOnce(&mut BTreeMap<String, Tensor>),
) -> Fixture {
    let f = fixture(cfg, |_| {});
    let config = ModelConfig::from_hf(&HfConfig::parse(cfg));
    let pack = model_packs::for_config(&config).unwrap();
    let plan = pack.compile_plan(&config).unwrap();
    let contract = pack
        .compile_tensor_contract(
            &config,
            &plan,
            CheckpointDialect::Gguf,
            pack.contract_options(&config),
        )
        .unwrap();
    let mut tensors = BTreeMap::new();
    for r in contract.requirements.iter().filter(|r| r.required) {
        tensors.insert(r.names[0].clone(), float(r.shape.clone(), 0.25));
    }
    for (name, tensor) in &mut tensors {
        if name.contains("_exps.weight") {
            tensor.dtype = "Q8_0";
            tensor.bytes = crate::nvfp4_repack::f32_to_q8_0(&vec![
                0.25;
                tensor.shape.iter().product::<u64>()
                    as usize
            ]);
        }
    }
    // Retain the legacy repack's BF16-vector widening exactly.
    tensors.insert(
        "output_norm.weight".into(),
        Tensor {
            dtype: "BF16",
            shape: vec![width],
            bytes: 0x3f80u16.to_le_bytes().repeat(width as usize),
        },
    );
    edit(&mut tensors);
    let mut rows = Vec::new();
    let mut bytes = Vec::new();
    for (name, t) in tensors {
        let offset = bytes.len();
        bytes.extend(&t.bytes);
        rows.push(format!("{name:?}:{{\"file\":\"weights.bin\",\"offset\":{offset},\"qtype\":{:?},\"ne\":{:?},\"bytes\":{}}}",t.dtype,t.shape,t.bytes.len()));
    }
    std::fs::write(f.dir.join("weights.bin"), bytes).unwrap();
    std::fs::write(
        f.dir.join("manifest.json"),
        format!(
            "{{\"format\":\"memra-repack-v1\",\"tensors\":{{{}}}}}",
            rows.join(",")
        ),
    )
    .unwrap();
    f
}

#[test]
fn complete_repack_uses_bound_physical_targets_and_opened_identity() {
    let f = complete_repack_fixture(|_| {});
    let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    assert!(!runtime.runtime_metadata().unwrap().is_gguf);
    assert!(!runtime.runtime_metadata().unwrap().is_safetensors);
    assert!(runtime.gguf_tensor_metadata().unwrap().is_none());
    assert!(
        runtime
            .try_find_gguf_disk("blk.0.ffn_gate_exps.weight")
            .unwrap()
            .is_none()
    );
    for name in [
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.ffn_gate_exps.weight",
    ] {
        let expected = source.find(name).unwrap();
        let actual = runtime.try_find(name).unwrap().unwrap();
        assert_eq!(
            (actual.ggml_type, &actual.ne, &actual.bytes),
            (expected.ggml_type, &expected.ne, &expected.bytes)
        );
    }
    let disk = runtime
        .try_find_expert_disk("blk.0.ffn_gate_exps.weight")
        .unwrap()
        .unwrap();
    assert!(matches!(disk, crate::bound_disk::ExpertDiskView::Bound(_)));
    assert_eq!(
        runtime
            .try_find("blk.0.ffn_gate_exps.weight")
            .unwrap()
            .unwrap()
            .ggml_type,
        crate::GgmlType::Q8_0
    );
    assert_eq!(
        disk.bytes(),
        source
            .find("blk.0.ffn_gate_exps.weight")
            .unwrap()
            .bytes
            .as_ref()
    );
    assert!(
        runtime
            .try_find_nvfp4_native("blk.0.ffn_gate_exps.weight")
            .unwrap()
            .is_none()
    );
    let identity = bound.artifact_identity().unwrap();
    assert_eq!(
        identity.opened_source_sha256,
        source.artifact_sha256().unwrap()
    );
    assert_eq!(identity.artifact_sha256, runtime.artifact_sha256().unwrap());
    assert_ne!(identity.artifact_sha256, identity.opened_source_sha256);
    std::fs::remove_file(f.dir.join("manifest.json")).unwrap();
    std::fs::remove_file(f.dir.join("config.json")).unwrap();
    assert_eq!(identity, bound.artifact_identity().unwrap());
    let expected = disk.bytes().to_vec();
    drop(runtime);
    drop(bound);
    drop(source);
    std::fs::remove_file(f.dir.join("weights.bin")).unwrap();
    std::fs::write(f.dir.join("weights.bin"), b"replacement").unwrap();
    assert_eq!(disk.bytes(), expected);
    assert!(disk.subrange(0..expected.len() + 1).is_err());
}

#[test]
fn incomplete_repack_and_masked_uniform_bank_refuse_before_runtime_access() {
    for case in ["missing", "extra", "shape"] {
        let f = complete_repack_fixture(|ts| match case {
            "missing" => {
                ts.remove("output.weight");
            }
            "extra" => {
                ts.insert("unknown.weight".into(), float(vec![1], 1.0));
            }
            "shape" => {
                ts.insert("output.weight".into(), float(vec![16, 64], 1.0));
            }
            _ => unreachable!(),
        });
        let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
        assert!(BoundTensorSource::compile(&source).is_err(), "{case}");
    }
    let f = complete_repack_fixture(|_| {});
    let path = f.dir.join("manifest.json");
    let raw = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        raw.replacen('{', "{\"pruned_experts\":{\"0\":[3]},", 1),
    )
    .unwrap();
    let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    assert!(compile_error(&source).contains("missing tensor"));
    // A sparse overlay cannot borrow a complete fallback's apparent validity.
    let base = complete_repack_fixture(|_| {});
    let overlay = base.dir.join("overlay");
    std::fs::create_dir(&overlay).unwrap();
    std::fs::write(
        overlay.join("manifest.json"),
        r#"{"format":"memra-expert-overlay-v2","source_dir":"..","tensors":{}}"#,
    )
    .unwrap();
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(compile_error(&source).contains("composite tensor contract"));
}

fn retained_repack_fixture(edit: impl FnOnce(&mut BTreeMap<String, Tensor>)) -> Fixture {
    let f = complete_repack_fixture_size(256, |tensors| {
        for projection in ["gate", "up", "down"] {
            let name = format!("blk.0.ffn_{projection}_exps.weight");
            let bank = tensors.remove(&name).unwrap();
            let shape = bank.shape[..2].to_vec();
            let elements = shape.iter().product::<u64>() as usize;
            tensors.insert(
                format!("blk.0.ffn_{projection}_exps.1.weight"),
                Tensor {
                    dtype: "Q2_K",
                    shape: shape.clone(),
                    bytes: vec![0x11; elements / 256 * 84],
                },
            );
            tensors.insert(
                format!("blk.0.ffn_{projection}_exps.3.weight"),
                Tensor {
                    dtype: "NVFP4",
                    bytes: vec![0; crate::GgmlType::NVFP4.checked_nbytes(&shape).unwrap() as usize],
                    shape,
                },
            );
        }
        edit(tensors);
    });
    let path = f.dir.join("manifest.json");
    let manifest = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        path,
        manifest.replacen('{', "{\"pruned_experts\":{\"0\":[0,2]},", 1),
    )
    .unwrap();
    f
}

#[test]
fn retained_repack_binds_original_ids_encodings_masks_and_disk_windows() {
    let f = retained_repack_fixture(|_| {});
    let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let runtime = bound.runtime().unwrap();
    let crate::model_plan::MlpPlan::Moe(moe) = &bound.plan.layers[0].mlp else {
        unreachable!()
    };
    assert_eq!(moe.expert_count, 4);
    assert_eq!(moe.retained_experts, Some(vec![1, 3]));
    assert_eq!(
        runtime.active_experts(0),
        Some(&[false, true, false, true][..])
    );
    assert!(runtime.preserve_expert_encodings());
    let bank = TensorId::Layer {
        index: 0,
        tensor: LayerTensor::MoeExpertGateBank,
    };
    assert_eq!(
        bound.member_tensor(&bank, 3).unwrap().ggml_type,
        crate::GgmlType::NVFP4
    );
    assert!(
        bound
            .member_tensor(&bank, 0)
            .err()
            .unwrap()
            .contains("pruned")
    );
    assert!(
        runtime
            .try_find("blk.0.ffn_gate_exps.weight")
            .unwrap()
            .is_none()
    );
    assert!(
        runtime
            .try_find("blk.0.ffn_gate_exps.0.weight")
            .err()
            .unwrap()
            .contains("pruned")
    );
    let first = runtime
        .try_find("blk.0.ffn_gate_exps.1.weight")
        .unwrap()
        .unwrap();
    let third = runtime
        .try_find("blk.0.ffn_gate_exps.3.weight")
        .unwrap()
        .unwrap();
    assert_eq!(first.ggml_type, crate::GgmlType::Q2_K);
    assert_eq!(third.ggml_type, crate::GgmlType::NVFP4);
    assert_eq!(first.bytes.as_ref(), vec![0x11; 256 * 256 / 256 * 84]);
    assert_eq!(third.bytes.as_ref(), vec![0; 256 * 256 / 64 * 36]);
    let disk = runtime
        .try_find_expert_disk("blk.0.ffn_gate_exps.3.weight")
        .unwrap()
        .unwrap();
    assert_eq!(disk.bytes(), third.bytes.as_ref());
    let expected = disk.bytes().to_vec();
    let identity = bound.artifact_identity().unwrap();
    std::fs::remove_file(f.dir.join("manifest.json")).unwrap();
    std::fs::remove_file(f.dir.join("weights.bin")).unwrap();
    std::fs::write(f.dir.join("weights.bin"), b"replacement").unwrap();
    assert_eq!(bound.artifact_identity().unwrap(), identity);
    assert_eq!(
        runtime.active_experts(0),
        Some(&[false, true, false, true][..])
    );
    drop(runtime);
    drop(bound);
    drop(source);
    assert_eq!(disk.bytes(), expected);
    assert!(disk.subrange(0..expected.len() + 1).is_err());
}

#[test]
fn retained_repack_refuses_missing_pruned_wrong_shape_and_uniform_rows() {
    for case in ["missing", "pruned", "shape", "uniform"] {
        let f = retained_repack_fixture(|tensors| {
            let key = "blk.0.ffn_gate_exps.3.weight";
            match case {
                "missing" => {
                    tensors.remove(key);
                }
                "pruned" => {
                    let original = &tensors[key];
                    let value = Tensor {
                        dtype: original.dtype,
                        shape: original.shape.clone(),
                        bytes: original.bytes.clone(),
                    };
                    tensors.insert("blk.0.ffn_gate_exps.0.weight".into(), value);
                }
                "shape" => {
                    tensors.insert(key.into(), float(vec![32, 64], 0.25));
                }
                "uniform" => {
                    tensors.insert(
                        "blk.0.ffn_gate_exps.weight".into(),
                        float(vec![32, 32, 4], 0.25),
                    );
                }
                _ => unreachable!(),
            }
        });
        let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
        assert!(BoundTensorSource::compile(&source).is_err(), "{case}");
    }
}

#[test]
fn retained_repack_rejects_ambiguous_or_invalid_mask_metadata() {
    for mask in [
        r#"{"0":[0,0]}"#,
        r#"{"0":[0,1,2]}"#,
        r#"{"1":[0]}"#,
        r#"{"0":[9]}"#,
        r#"{"0":[0],"0":[2]}"#,
        r#"{"0":[0],"00":[2]}"#,
        r#"{"0":[0,"bad"]}"#,
        r#"{"0":[-1,0]}"#,
        r#"{"0":[0.5]}"#,
        r#"{"0":[0,]}"#,
        r#"null"#,
        r#"[]"#,
    ] {
        let f = retained_repack_fixture(|_| {});
        let path = f.dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replace(r#"{"0":[0,2]}"#, mask)).unwrap();
        assert!(
            crate::source::Hy3RepackSource::open(&f.dir).is_err(),
            "{mask}"
        );
    }
}

#[test]
fn retained_repack_masks_are_captured_by_the_compiler() {
    struct MutableMask<'a> {
        source: &'a crate::source::Hy3RepackSource,
        changed: std::sync::atomic::AtomicBool,
    }
    impl TensorSource for MutableMask<'_> {
        fn config(&self) -> ModelConfig {
            self.source.config()
        }
        fn find(&self, name: &str) -> Option<TensorView<'_>> {
            self.source.find(name)
        }
        fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
            self.source.bound_interpretation()
        }
        fn raw_config_json(&self) -> Option<&str> {
            self.source.raw_config_json()
        }
        fn runtime_metadata(&self) -> Result<crate::source::RuntimeSourceMetadata, String> {
            self.source.runtime_metadata()
        }
        fn tensor_census(&self) -> Result<TensorCensus, String> {
            self.source.tensor_census()
        }
        fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
            self.source.validate_bound_metadata(r)
        }
        fn active_experts(&self, layer: u32) -> Option<&[bool]> {
            if layer == 0 && self.changed.load(Ordering::SeqCst) {
                Some(&[false, false, true, true])
            } else {
                self.source.active_experts(layer)
            }
        }
    }
    let f = retained_repack_fixture(|_| {});
    let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    let mutable = MutableMask {
        source: &source,
        changed: std::sync::atomic::AtomicBool::new(false),
    };
    let bound = BoundTensorSource::compile(&mutable).unwrap();
    let runtime = bound.runtime().unwrap();
    mutable.changed.store(true, Ordering::SeqCst);
    assert_eq!(
        mutable.active_experts(0),
        Some(&[false, false, true, true][..])
    );
    assert_eq!(
        runtime.active_experts(0),
        Some(&[false, true, false, true][..])
    );
    assert!(compile_error(&mutable).contains("differs from its declared bound program"));
}

#[test]
fn retained_repack_rejects_mask_width_dense_and_unknown_layer_programs() {
    let f = complete_repack_fixture(|_| {});
    let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    for (index, mask) in [(0, vec![true; 3]), (1, vec![true; 4]), (0, vec![false; 4])] {
        let interpretation = BoundSourceInterpretation::RetainedRepack {
            expert_activation_precision: crate::source::ExpertActivationPrecision::F32,
            active_experts: BTreeMap::from([(index, mask)]),
        };
        assert!(bind_retained_experts(&mut bound.plan.clone(), &interpretation).is_err());
    }
    let config = ModelConfig::from_hf(&HfConfig::parse(QWEN));
    let mut dense = model_packs::compile_for_load(&config).unwrap();
    let interpretation = BoundSourceInterpretation::RetainedRepack {
        expert_activation_precision: crate::source::ExpertActivationPrecision::F32,
        active_experts: BTreeMap::from([(0, vec![false, true, false, true])]),
    };
    assert!(
        bind_retained_experts(&mut dense, &interpretation)
            .unwrap_err()
            .contains("no routed MoE operation")
    );
}

#[test]
fn retained_repack_rejects_malformed_numeric_or_duplicate_manifest_fields() {
    let mutations: &[(&str, &str)] = &[
        (r#""ne":[256, 256]"#, r#""ne":[256,"bad",256]"#),
        (r#""offset":0"#, r#""offset":"bad""#),
        (
            r#""qtype":"Q2_K""#,
            r#""expert_stride":"bad","qtype":"Q2_K""#,
        ),
        (r#""qtype":"Q2_K""#, r#""qtype":"NVFP4","qtype":"Q2_K""#),
    ];
    for &(from, to) in mutations {
        let f = retained_repack_fixture(|_| {});
        let path = f.dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(from));
        std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
        assert!(
            crate::source::Hy3RepackSource::open(&f.dir).is_err(),
            "{from}"
        );
    }
    let f = retained_repack_fixture(|_| {});
    let path = f.dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!("{text} trailing ignored content")).unwrap();
    assert!(crate::source::Hy3RepackSource::open(&f.dir).is_err());
}

#[test]
fn escaped_repack_declarations_bind_the_same_retained_program() {
    let f = retained_repack_fixture(|_| {});
    let literal = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    let literal_bound = BoundTensorSource::compile(&literal).unwrap();
    let path = f.dir.join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("pruned_experts", r"pruned\u005fexperts")
        .replace("weights.bin", r"we\u0069ghts.bin")
        .replace("Q2_K", r"Q\u0032_K")
        .replace(
            "blk.0.ffn_gate_exps.1.weight",
            r"blk.0.ffn_gate_exps.\u0031.weight",
        );
    std::fs::write(&path, text).unwrap();
    let escaped = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&escaped).unwrap();
    assert_eq!(bound.plan(), literal_bound.plan());
    assert_eq!(bound.binding_sha256(), literal_bound.binding_sha256());
    // Exact metadata bytes still distinguish artifact identities, even with equal semantics.
    assert_ne!(
        bound.artifact_identity().unwrap(),
        literal_bound.artifact_identity().unwrap()
    );
    let runtime = bound.runtime().unwrap();
    assert_eq!(
        runtime.active_experts(0),
        Some(&[false, true, false, true][..])
    );
    assert_eq!(
        runtime
            .try_find("blk.0.ffn_gate_exps.1.weight")
            .unwrap()
            .unwrap()
            .ggml_type,
        crate::GgmlType::Q2_K
    );
}

#[test]
fn escaped_mask_and_overlay_declarations_cannot_become_absence() {
    for declaration in [
        r#""pruned_experts":{"0":[0,2]}"#,
        r#""pruned\u005fexperts":{"0":[0,2]}"#,
    ] {
        let f = complete_repack_fixture(|_| {});
        let path = f.dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replacen('{', &format!("{{{declaration},"), 1)).unwrap();
        let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
        assert_eq!(
            source.active_experts(0),
            Some(&[false, true, false, true][..])
        );
        assert!(compile_error(&source).contains("missing tensor"));
    }
    for format in ["memra-expert-overlay-v2", r"memra-expert-overlay\u002dv2"] {
        let f = complete_repack_fixture(|_| {});
        let path = f.dir.join("manifest.json");
        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replace("memra-repack-v1", format);
        std::fs::write(&path, text).unwrap();
        let error = crate::source::Hy3RepackSource::open(&f.dir).err().unwrap();
        assert!(error.to_string().contains("missing source_dir"));
    }
    let f = retained_repack_fixture(|_| {});
    let path = f.dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replacen('{', r#"{"pruned\u005fexperts":{},"#, 1),
    )
    .unwrap();
    assert!(
        crate::source::Hy3RepackSource::open(&f.dir)
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate JSON key")
    );
}

fn write_inventory_overlay(directory: &Path, source: &str, value: f32) {
    std::fs::create_dir_all(directory).unwrap();
    std::fs::write(
        directory.join("expert.bin"),
        value.to_le_bytes().repeat(32 * 32),
    )
    .unwrap();
    std::fs::write(directory.join("manifest.json"), format!(r#"{{"format":"memra-expert-overlay-v2","source_dir":{source:?},"tensors":{{"blk.0.ffn_gate_exps.1.weight":{{"file":"expert.bin","qtype":"F32","ne":[32,32],"bytes":4096}}}}}}"#)).unwrap();
}

#[test]
fn overlay_physical_inventory_retains_shadowed_banks_and_component_names() {
    let base = complete_repack_fixture(|_| {});
    let lower = base.dir.join("lower");
    let upper = lower.join("upper");
    write_inventory_overlay(&lower, "..", 1.0);
    write_inventory_overlay(&upper, "..", 2.0);
    let source = crate::source::Hy3RepackSource::open(&upper).unwrap();
    let effective = source.tensor_census().unwrap();
    assert!(
        !effective
            .tensors
            .iter()
            .any(|r| r.physical_name == "blk.0.ffn_gate_exps.weight")
    );
    let inventory = source.physical_tensor_inventory().unwrap();
    assert_eq!(
        inventory
            .components
            .iter()
            .map(|c| c.component_path.clone())
            .collect::<Vec<_>>(),
        vec![vec![], vec![0], vec![0, 0]]
    );
    assert_eq!(
        inventory.components[0].census.tensors[0].physical_name,
        "blk.0.ffn_gate_exps.1.weight"
    );
    assert_eq!(
        inventory.components[1].census.tensors[0].physical_name,
        "blk.0.ffn_gate_exps.1.weight"
    );
    assert!(
        inventory.components[2]
            .census
            .tensors
            .iter()
            .any(|r| r.physical_name == "blk.0.ffn_gate_exps.weight")
    );
    assert!(compile_error(&source).contains("composite tensor contract"));
    assert!(source.artifact_sha256().is_err());
    // Inventory comes from opened declarations, without reopening any component path.
    std::fs::remove_file(upper.join("manifest.json")).unwrap();
    std::fs::remove_file(lower.join("manifest.json")).unwrap();
    std::fs::remove_file(base.dir.join("manifest.json")).unwrap();
    assert_eq!(source.physical_tensor_inventory().unwrap(), inventory);
    drop(source);
    assert_eq!(inventory.components.len(), 3);
}

#[test]
fn overlay_physical_inventory_keeps_hf_auxiliaries_and_gguf_override_dialects() {
    let base = fixture(QWEN, |tensors| {
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![32, 32],
                bytes: vec![0x38; 1024],
            },
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight_scale".into(),
            float(vec![1, 1], 1.0),
        );
    });
    let overlay = base.dir.join("overlay");
    std::fs::create_dir(&overlay).unwrap();
    std::fs::write(overlay.join("query.bin"), 2.0f32.to_le_bytes().repeat(1024)).unwrap();
    std::fs::write(overlay.join("manifest.json"), r#"{"format":"memra-expert-overlay-v2","source_dir":"..","tensors":{"blk.0.attn_q.weight":{"file":"query.bin","qtype":"F32","ne":[32,32],"bytes":4096}}}"#).unwrap();
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let inventory = source.physical_tensor_inventory().unwrap();
    assert_eq!(
        inventory.components[0].census.dialect,
        CheckpointDialect::Gguf
    );
    assert_eq!(
        inventory.components[1].census.dialect,
        CheckpointDialect::HfSafetensors
    );
    let parent = inventory.components[1]
        .census
        .tensors
        .iter()
        .find(|r| r.physical_name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(parent.dtype, "F8_E4M3");
    assert_eq!(parent.entry.physical_bytes, 1028);
    assert_eq!(parent.auxiliaries.len(), 1);
    assert_eq!(
        parent.auxiliaries[0].physical_name,
        "model.layers.0.self_attn.q_proj.weight_scale"
    );
    let effective = source.tensor_census().unwrap();
    let replacement = effective
        .tensors
        .iter()
        .find(|r| r.entry.name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(replacement.dtype, "F32");
    assert!(replacement.auxiliaries.is_empty());
    assert!(compile_error(&source).contains("composite tensor contract"));
}

#[test]
fn repack_source_cycles_and_excessive_depth_return_errors() {
    let f = complete_repack_fixture(|_| {});
    let a = f.dir.join("a");
    let b = f.dir.join("b");
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    let write = |dir: &Path, source: &str| {
        std::fs::write(
            dir.join("manifest.json"),
            format!(
                r#"{{"format":"memra-expert-overlay-v2","source_dir":{source:?},"tensors":{{}}}}"#
            ),
        )
        .unwrap()
    };
    write(&a, ".");
    assert!(
        crate::source::Hy3RepackSource::open(&a)
            .err()
            .unwrap()
            .to_string()
            .contains("source cycle")
    );
    write(&a, "../b");
    write(&b, "../a");
    assert!(
        crate::source::Hy3RepackSource::open(&a)
            .err()
            .unwrap()
            .to_string()
            .contains("source cycle")
    );
    for index in 0..65 {
        let dir = f.dir.join(format!("depth-{index}"));
        std::fs::create_dir(&dir).unwrap();
        write(
            &dir,
            &if index == 64 {
                "..".into()
            } else {
                format!("../depth-{}", index + 1)
            },
        );
    }
    assert!(
        crate::source::Hy3RepackSource::open(&f.dir.join("depth-0"))
            .err()
            .unwrap()
            .to_string()
            .contains("exceeds 64")
    );
    let maximum = crate::source::Hy3RepackSource::open(&f.dir.join("depth-2")).unwrap();
    assert_eq!(
        maximum
            .physical_tensor_inventory()
            .unwrap()
            .components
            .len(),
        64
    );
    assert!(maximum.find("token_embd.weight").is_some());
}

#[cfg(unix)]
#[test]
fn source_cycle_identity_uses_directory_context_and_detects_symlink_aliases() {
    let base = complete_repack_fixture(|_| {});
    let a = base.dir.join("a");
    let b = a.join("b");
    std::fs::create_dir_all(&b).unwrap();
    std::fs::write(
        a.join("manifest.json"),
        r#"{"format":"memra-expert-overlay-v2","source_dir":"..","tensors":{}}"#,
    )
    .unwrap();
    std::fs::hard_link(a.join("manifest.json"), b.join("manifest.json")).unwrap();
    // The same manifest inode in different directories has different relative-source meaning.
    let source = crate::source::Hy3RepackSource::open(&b).unwrap();
    assert_eq!(
        source.physical_tensor_inventory().unwrap().components.len(),
        3
    );
    drop(source);
    let cyclic = base.dir.join("cyclic");
    std::fs::create_dir(&cyclic).unwrap();
    std::os::unix::fs::symlink(".", cyclic.join("alias")).unwrap();
    std::fs::write(
        cyclic.join("manifest.json"),
        r#"{"format":"memra-expert-overlay-v2","source_dir":"alias","tensors":{}}"#,
    )
    .unwrap();
    assert!(
        crate::source::Hy3RepackSource::open(&cyclic)
            .err()
            .unwrap()
            .to_string()
            .contains("source cycle")
    );
}

#[cfg(unix)]
#[test]
fn source_context_does_not_require_directory_listing_permission() {
    use std::os::unix::fs::PermissionsExt;
    let f = complete_repack_fixture(|_| {});
    let original = std::fs::metadata(&f.dir).unwrap().permissions();
    std::fs::set_permissions(&f.dir, std::fs::Permissions::from_mode(0o111)).unwrap();
    let result = crate::source::Hy3RepackSource::open(&f.dir);
    std::fs::set_permissions(&f.dir, original).unwrap();
    assert!(result.is_ok());
}

#[cfg(unix)]
#[test]
fn source_manifest_fifo_refuses_without_waiting_for_a_writer() {
    use std::os::unix::ffi::OsStrExt;
    let f = complete_repack_fixture(|_| {});
    let path = f.dir.join("pipe-manifest");
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = crate::source::Hy3RepackSource::open(&path).err().unwrap();
    assert!(error.to_string().contains("not a regular file"));
}

fn composite_overlay(directory: &Path, tensors: &[(&str, Vec<u64>, f32)]) {
    std::fs::create_dir_all(directory).unwrap();
    let mut data = Vec::new();
    let mut rows = serde_json::Map::new();
    for (name, shape, value) in tensors {
        let offset = data.len();
        let bytes = value
            .to_le_bytes()
            .repeat(shape.iter().product::<u64>() as usize);
        data.extend(&bytes);
        rows.insert(
            (*name).into(),
            serde_json::json!({
                "file":"weights.bin", "qtype":"F32", "ne":shape,
                "offset":offset, "bytes":bytes.len()
            }),
        );
    }
    std::fs::write(directory.join("weights.bin"), data).unwrap();
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::json!({
            "format":"memra-expert-overlay-v2", "source_dir":"..", "tensors":rows
        })
        .to_string(),
    )
    .unwrap();
}

#[test]
fn composite_catalog_binds_precedence_and_preserves_shadowed_roles_without_runtime_authority() {
    use super::composite::CompositeTensorCatalog;
    let base = complete_repack_fixture(|_| {});
    let lower = base.dir.join("lower");
    let upper = lower.join("upper");
    composite_overlay(&lower, &[("output_norm.weight", vec![32], 1.0)]);
    composite_overlay(&upper, &[("output_norm.weight", vec![32], 2.0)]);
    let source = crate::source::Hy3RepackSource::open(&upper).unwrap();
    let bound = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
    assert_eq!(bound.components().len(), 3);
    assert_eq!(
        bound.selected()[&TensorId::OutputNorm].members[0].component_path,
        Vec::<u32>::new()
    );
    assert_eq!(
        bound.selected()[&TensorId::TokenEmbedding].members[0].component_path,
        [0, 0]
    );
    for component in bound.components() {
        assert!(
            component
                .binding()
                .tensors
                .contains_key(&TensorId::OutputNorm)
        );
    }
    let digest = bound.binding_sha256().to_owned();
    // The metadata result exports no runtime adapter or source handles. Existing entrypoints
    // still refuse; pathname loss changes neither the opened inventory nor its sealed selection.
    assert!(
        BoundTensorSource::compile(&source)
            .err()
            .unwrap()
            .to_string()
            .contains("composite tensor contract")
    );
    assert!(source.artifact_sha256().is_err());
    for directory in [&upper, &lower, &base.dir] {
        std::fs::remove_file(directory.join("manifest.json")).unwrap();
    }
    assert_eq!(
        CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .unwrap()
            .binding_sha256(),
        digest
    );
    drop(source);
    assert_eq!(bound.components().len(), 3);
}

#[test]
fn composite_catalog_preserves_member_dialect_and_transform() {
    use super::composite::CompositeTensorCatalog;
    let f = fixture(HYBRID, |_| {});
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.25)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
    let output = &bound.selected()[&TensorId::OutputNorm].members[0];
    assert_eq!(output.dialect, CheckpointDialect::Gguf);
    assert_eq!(output.transform, TensorTransform::Identity);
    let q_norm = TensorId::Layer {
        index: 1,
        tensor: LayerTensor::QueryNorm,
    };
    let member = &bound.selected()[&q_norm].members[0];
    assert_eq!(member.dialect, CheckpointDialect::HfSafetensors);
    assert_eq!(member.component_path, [0]);
    assert_eq!(member.transform, TensorTransform::NormAddOne);
}

#[test]
fn composite_catalog_rejects_shadowed_shape_dtype_and_auxiliary_errors() {
    use super::composite::CompositeTensorCatalog;
    for bad in ["shape", "dtype", "auxiliary", "scale"] {
        let f = fixture(QWEN, |tensors| {
            let key = "model.layers.0.self_attn.q_proj.weight";
            match bad {
                "shape" => {
                    tensors.insert(key.into(), float(vec![16, 32], 1.0));
                }
                "dtype" => {
                    tensors.insert(
                        key.into(),
                        Tensor {
                            dtype: "I64",
                            shape: vec![32, 32],
                            bytes: vec![0; 8192],
                        },
                    );
                }
                "auxiliary" => {
                    tensors.insert(
                        "model.layers.0.self_attn.q_proj.pre_quant_scale".into(),
                        float(vec![17], 1.0),
                    );
                }
                "scale" => {
                    tensors.insert(
                        key.into(),
                        Tensor {
                            dtype: "F8_E4M3",
                            shape: vec![32, 32],
                            bytes: vec![0x38; 1024],
                        },
                    );
                    tensors.insert(
                        "model.layers.0.self_attn.q_proj.weight_scale".into(),
                        float(vec![2, 3], 1.0),
                    );
                }
                _ => unreachable!(),
            }
        });
        let overlay = f.dir.join("overlay");
        composite_overlay(&overlay, &[("blk.0.attn_q.weight", vec![32, 32], 2.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let error = CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .err()
            .unwrap();
        assert!(error.contains("component [0]"), "{bad}: {error}");
    }
}

#[test]
fn composite_catalog_rejects_unknown_shadowed_rows_and_missing_final_roles() {
    use super::composite::CompositeTensorCatalog;
    for unknown in [false, true] {
        let f = fixture(QWEN, |tensors| {
            if unknown {
                tensors.insert("model.unclaimed.weight".into(), float(vec![32], 1.0));
            } else {
                tensors.remove("model.layers.0.self_attn.q_proj.weight");
            }
        });
        let overlay = f.dir.join("overlay");
        composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let error = CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .err()
            .unwrap();
        assert!(
            error.contains(if unknown {
                "unclaimed"
            } else {
                "missing required tensor"
            }),
            "{error}"
        );
    }
    let f = fixture(QWEN, |tensors| {
        tensors.remove("model.layers.0.self_attn.q_proj.weight");
    });
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("blk.0.attn_q.weight", vec![32, 32], 2.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(CompositeTensorCatalog::compile(&source, LoadScope::Full).is_ok());
}

#[test]
fn composite_catalog_digest_covers_shadowed_storage_and_execution_scope() {
    use super::composite::CompositeTensorCatalog;
    let mut digests = Vec::new();
    for quantized in [false, true] {
        let f = fixture(QWEN, |tensors| {
            if quantized {
                tensors.insert(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    Tensor {
                        dtype: "F8_E4M3",
                        shape: vec![32, 32],
                        bytes: vec![0x38; 1024],
                    },
                );
                tensors.insert(
                    "model.layers.0.self_attn.q_proj.weight_scale".into(),
                    float(vec![1, 1], 1.0),
                );
            }
        });
        let overlay = f.dir.join("overlay");
        composite_overlay(&overlay, &[("blk.0.attn_q.weight", vec![32, 32], 2.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let full = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
        let text = CompositeTensorCatalog::compile(&source, LoadScope::Text).unwrap();
        assert_ne!(full.binding_sha256(), text.binding_sha256());
        digests.push(full.binding_sha256().to_owned());
    }
    assert_ne!(digests[0], digests[1]);
}

#[test]
fn composite_catalog_refuses_incomplete_split_and_retained_banks() {
    use super::composite::CompositeTensorCatalog;
    let f = complete_repack_fixture(|_| {});
    let overlay = f.dir.join("overlay");
    write_inventory_overlay(&overlay, "..", 1.0);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(CompositeTensorCatalog::compile(&source, LoadScope::Full).is_err());
    let manifest = overlay.join("manifest.json");
    let mut declarations: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    declarations["pruned_experts"] = serde_json::json!({"0":[0,2]});
    std::fs::write(&manifest, declarations.to_string()).unwrap();
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(
        CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .err()
            .unwrap()
            .contains("missing group member 3")
    );
}

#[test]
fn composite_catalog_auxiliary_selection_follows_its_own_weight() {
    use super::composite::CompositeTensorCatalog;
    let f = complete_repack_fixture(|tensors| {
        tensors.insert("blk.0.attn_q.scale".into(), float(vec![1], 0.5));
    });
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("blk.0.attn_q.weight", vec![32, 32], 2.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let catalog = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
    let scale = TensorId::QuantAux {
        tensor: Box::new(TensorId::Layer {
            index: 0,
            tensor: LayerTensor::Query,
        }),
        kind: QuantAuxTensor::WeightScale,
    };
    assert!(
        catalog.components()[1]
            .binding()
            .tensors
            .contains_key(&scale)
    );
    assert!(!catalog.selected().contains_key(&scale));
    drop(source);
    composite_overlay(
        &overlay,
        &[
            ("blk.0.attn_q.weight", vec![32, 32], 2.0),
            ("blk.0.attn_q.scale", vec![1], 0.75),
        ],
    );
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let catalog = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
    assert_eq!(
        catalog.selected()[&scale].members[0].component_path,
        Vec::<u32>::new()
    );
    drop(source);
    composite_overlay(&overlay, &[("blk.0.attn_q.scale", vec![1], 0.75)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(
        CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .err()
            .unwrap()
            .contains("no owning weight")
    );
}

#[test]
fn composite_catalog_keeps_canonical_group_order_and_never_fabricates_a_member() {
    use super::composite::CompositeTensorCatalog;
    let config = QWEN.replace("\"qwen3\"", "\"qwen3_moe\"").replacen(
        '{',
        "{\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,",
        1,
    );
    for missing in [false, true] {
        let f = fixture(&config, |tensors| {
            if missing {
                tensors.remove("model.layers.0.mlp.experts.2.gate_proj.weight");
            }
        });
        let overlay = f.dir.join("overlay");
        composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let result = CompositeTensorCatalog::compile(&source, LoadScope::Full);
        if missing {
            assert!(result.err().unwrap().contains("missing group member 2"));
        } else {
            let catalog = result.unwrap();
            let id = TensorId::Layer {
                index: 0,
                tensor: LayerTensor::MoeExpertGateBank,
            };
            let members = &catalog.selected()[&id].members;
            assert_eq!(members.len(), 4);
            for (i, member) in members.iter().enumerate() {
                assert_eq!(
                    member.record.physical_name,
                    format!("model.layers.0.mlp.experts.{i}.gate_proj.weight")
                );
                assert_eq!(member.transform, TensorTransform::StackExperts);
            }
        }
        drop(source);
        // A complete explicit stacked replacement is a different declared representation; the
        // partial lower group is inventoried and validated, but never sliced or manufactured.
        composite_overlay(
            &overlay,
            &[("blk.0.ffn_gate_exps.weight", vec![32, 32, 4], 1.0)],
        );
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let catalog = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
        let id = TensorId::Layer {
            index: 0,
            tensor: LayerTensor::MoeExpertGateBank,
        };
        assert_eq!(catalog.selected()[&id].members.len(), 1);
        assert_eq!(
            catalog.selected()[&id].members[0].dialect,
            CheckpointDialect::Gguf
        );
    }
}

#[test]
fn composite_catalog_rejects_alias_collisions_and_respects_tied_head_policy() {
    use super::composite::CompositeTensorCatalog;
    let f = fixture(QWEN, |tensors| {
        tensors.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight".into(),
            float(vec![32, 32], 1.0),
        );
    });
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("blk.0.attn_q.weight", vec![32, 32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(CompositeTensorCatalog::compile(&source, LoadScope::Full).is_err());

    let config = QWEN.replacen('{', "{\"tie_word_embeddings\":true,", 1);
    let f = fixture(&config, |_| {});
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let catalog = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
    assert!(!catalog.selected().contains_key(&TensorId::OutputProjection));
    drop(source);
    composite_overlay(&overlay, &[("output.weight", vec![32, 32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(
        CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .err()
            .unwrap()
            .contains("output.weight")
    );
}

#[test]
fn composite_catalog_validates_unselected_vision_without_selecting_it() {
    use super::composite::CompositeTensorCatalog;
    for missing in [false, true] {
        let f = step_with_vision_fixture(|tensors| {
            if missing {
                tensors.remove("vision_model.conv1.weight");
            }
        });
        let overlay = f.dir.join("overlay");
        composite_overlay(&overlay, &[("output_norm.weight", vec![16], 1.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let result = CompositeTensorCatalog::compile(&source, LoadScope::Text);
        if missing {
            assert!(result.err().unwrap().contains("missing required tensor"));
        } else {
            let catalog = result.unwrap();
            assert!(
                catalog.components()[1]
                    .binding()
                    .tensors
                    .values()
                    .any(|tensor| matches!(
                        tensor.owner,
                        crate::tensor_contract::TensorOwner::Vision(_)
                    ))
            );
            assert!(catalog.selected().values().all(|tensor| !matches!(
                tensor.owner,
                crate::tensor_contract::TensorOwner::Vision(_)
            )));
            assert!(
                CompositeTensorCatalog::compile(&source, LoadScope::Full)
                    .err()
                    .unwrap()
                    .contains("execution is unsupported")
            );
        }
    }
}

#[test]
fn composite_catalog_still_requires_the_leafs_gguf_rope_tensor() {
    use super::composite::CompositeTensorCatalog;
    let raw = include_str!("../model_packs/step35/contract-fixture.json");
    let f = fixture(raw, |_| {});
    let config = ModelConfig::from_hf(&HfConfig::parse(raw));
    let pack = model_packs::for_config(&config).unwrap();
    let plan = pack.compile_plan(&config).unwrap();
    let contract = pack
        .compile_tensor_contract(
            &config,
            &plan,
            CheckpointDialect::Gguf,
            pack.contract_options(&config),
        )
        .unwrap();
    assert!(
        contract
            .requirements
            .iter()
            .any(|r| r.required && r.id == TensorId::RopeFactors)
    );
    let mut data = Vec::new();
    let mut rows = serde_json::Map::new();
    for requirement in contract.requirements.iter().filter(|r| r.required) {
        let names = if requirement.match_mode == TensorMatch::All {
            &requirement.names[..]
        } else {
            &requirement.names[..1]
        };
        for name in names {
            let bytes = 1.0f32
                .to_le_bytes()
                .repeat(requirement.shape.iter().product::<u64>() as usize);
            rows.insert(
                name.clone(),
                serde_json::json!({"file":"weights.bin","offset":data.len(),
                "qtype":"F32","ne":requirement.shape,"bytes":bytes.len()}),
            );
            data.extend(bytes);
        }
    }
    std::fs::write(f.dir.join("weights.bin"), data).unwrap();
    let manifest = f.dir.join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::json!({"format":"memra-repack-v1","tensors":rows}).to_string(),
    )
    .unwrap();
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![16], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let catalog = CompositeTensorCatalog::compile(&source, LoadScope::Full).unwrap();
    assert_eq!(
        catalog.selected()[&TensorId::RopeFactors].members[0].component_path,
        [0]
    );
    drop(source);
    rows.remove("rope_freqs.weight").unwrap();
    std::fs::write(
        &manifest,
        serde_json::json!({"format":"memra-repack-v1","tensors":rows}).to_string(),
    )
    .unwrap();
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(
        CompositeTensorCatalog::compile(&source, LoadScope::Full)
            .err()
            .unwrap()
            .contains("missing required tensor RopeFactors")
    );
}

#[test]
fn canonical_config_intake_preserves_explicit_bound_head_policy() {
    for field in [
        r#""tie_word_embeddings":true,"#,
        r#""tie_word_\u0065mbeddings":true,"#,
    ] {
        let raw = QWEN.replacen('{', &format!("{{{field}"), 1);
        let config = HfConfig::try_parse(&raw).unwrap();
        assert_eq!(config.tie_word_embeddings, Some(true));
        let f = fixture(&raw, |_| {});
        let source = SafetensorsSource::open(&f.dir).unwrap();
        assert_eq!(
            BoundTensorSource::compile(&source)
                .unwrap()
                .options
                .output_head,
            OutputHead::TiedToEmbedding
        );
    }
    for value in ["null", "1", "\"true\"", "[]"] {
        let raw = QWEN.replacen('{', &format!("{{\"tie_word_embeddings\":{value},"), 1);
        assert!(HfConfig::try_parse(&raw).is_err());
    }
}

#[test]
fn canonical_numeric_intake_keeps_manifest_integers_lexically_strict() {
    let f = complete_repack_fixture(|_| {});
    let path = f.dir.join("manifest.json");
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("\"offset\":0"));
    for value in [
        "0.0",
        "0e0",
        "-0",
        "0.0000000000000000001",
        r#"{"$serde_json::private::Number":"0"}"#,
    ] {
        std::fs::write(
            &path,
            raw.replacen("\"offset\":0", &format!("\"offset\":{value}"), 1),
        )
        .unwrap();
        assert!(
            crate::source::Hy3RepackSource::open(&f.dir).is_err(),
            "{value}"
        );
    }
    std::fs::write(&path, raw).unwrap();
    let source = crate::source::Hy3RepackSource::open(&f.dir).unwrap();
    assert!(BoundTensorSource::compile(&source).is_ok());
}

fn composite_encoded_overlay(
    directory: &Path,
    rows: Vec<(String, Tensor)>,
    pruned: Option<Vec<u32>>,
) {
    std::fs::create_dir_all(directory).unwrap();
    let mut payload = Vec::new();
    let mut tensors = serde_json::Map::new();
    for (name, tensor) in rows {
        tensors.insert(
            name,
            serde_json::json!({"file":"weights.bin","offset":payload.len(),
            "qtype":tensor.dtype,"ne":tensor.shape,"bytes":tensor.bytes.len()}),
        );
        payload.extend(tensor.bytes);
    }
    std::fs::write(directory.join("weights.bin"), payload).unwrap();
    let mut manifest =
        serde_json::json!({"format":"memra-expert-overlay-v2","source_dir":"..","tensors":tensors});
    if let Some(ids) = pruned {
        manifest["pruned_experts"] = serde_json::json!({"0":ids});
    }
    std::fs::write(directory.join("manifest.json"), manifest.to_string()).unwrap();
}

#[test]
fn composite_access_inherits_original_ids_and_retains_exact_disk_bytes_after_drop() {
    use super::composite::BoundCompositeSource;
    let base = retained_repack_fixture(|_| {});
    let overlay = base.dir.join("overlay");
    composite_encoded_overlay(
        &overlay,
        vec![(
            "blk.0.ffn_gate_exps.3.weight".into(),
            Tensor {
                dtype: "NVFP4",
                shape: vec![256, 256],
                bytes: vec![
                    0x22;
                    crate::GgmlType::NVFP4.checked_nbytes(&[256, 256]).unwrap() as usize
                ],
            },
        )],
        None,
    );
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let id = layer(LayerTensor::MoeExpertGateBank);
    assert_eq!(
        bound.active_experts(0),
        Some(&[false, true, false, true][..])
    );
    assert_eq!(bound.catalog().selected()[&id].member_ids, Some(vec![1, 3]));
    assert_eq!(
        bound.member_tensor(&id, 1).unwrap().ggml_type,
        crate::GgmlType::Q2_K
    );
    assert!(
        bound
            .member_tensor(&id, 3)
            .unwrap()
            .bytes
            .iter()
            .all(|&b| b == 0x22)
    );
    assert!(
        bound
            .member_tensor(&id, 0)
            .err()
            .unwrap()
            .contains("pruned")
    );
    assert!(bound.tensor(&id).is_err());
    for rewrite in crate::execution_manifest::execution_rewrites(bound.plan()) {
        assert!(
            rewrite
                .blockers
                .contains(&crate::model_plan::OperationKind::RetainedExpertRouting)
        );
    }
    let identity = bound.artifact_identity().unwrap();
    let disk = bound.disk(&id, Some(3)).unwrap().unwrap();
    std::fs::rename(overlay.join("weights.bin"), overlay.join("opened.bin")).unwrap();
    std::fs::write(overlay.join("weights.bin"), vec![0x33; disk.len()]).unwrap();
    assert_eq!(bound.artifact_identity().unwrap(), identity);
    let replacement = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert_ne!(
        BoundCompositeSource::compile(&replacement, LoadScope::Full)
            .unwrap()
            .artifact_identity()
            .unwrap(),
        identity
    );
    assert!(source.artifact_sha256().is_err());
    assert!(BoundTensorSource::compile(&source).is_err());
    drop(bound);
    drop(source);
    let mut bytes = [0; 16];
    disk.read_at(&mut bytes, 0).unwrap();
    assert_eq!(bytes, [0x22; 16]);
    assert!(disk.read_at(&mut bytes, disk.len() as u64 - 1).is_err());
}

#[test]
fn composite_access_binds_retained_split_overlay_over_uniform_base_without_slicing() {
    use super::composite::BoundCompositeSource;
    let base = complete_repack_fixture_size(256, |_| {});
    let overlay = base.dir.join("overlay");
    let rows = || {
        ["gate", "up", "down"]
            .into_iter()
            .flat_map(|projection| {
                [1u32, 3].into_iter().map(move |id| {
                    (
                        format!("blk.0.ffn_{projection}_exps.{id}.weight"),
                        Tensor {
                            dtype: "Q2_K",
                            shape: vec![256, 256],
                            bytes: vec![id as u8; 256 * 84],
                        },
                    )
                })
            })
            .collect()
    };
    composite_encoded_overlay(&overlay, rows(), Some(vec![0, 2]));
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let id = layer(LayerTensor::MoeExpertGateBank);
    assert!(
        bound
            .member_tensor(&id, 1)
            .unwrap()
            .bytes
            .iter()
            .all(|&b| b == 1)
    );
    assert!(
        bound
            .member_tensor(&id, 3)
            .unwrap()
            .bytes
            .iter()
            .all(|&b| b == 3)
    );
    assert_eq!(
        bound.catalog().components()[1].binding().tensors[&id].checkpoint_names,
        vec!["blk.0.ffn_gate_exps.weight"]
    );
    drop(bound);
    drop(source);
    let mut missing = rows();
    missing.retain(|(name, _)| name != "blk.0.ffn_gate_exps.3.weight");
    composite_encoded_overlay(&overlay, missing, Some(vec![0, 2]));
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(
        BoundCompositeSource::compile(&source, LoadScope::Full)
            .err()
            .unwrap()
            .contains("missing group member 3")
    );
    drop(source);
    let mut invalid = rows();
    invalid.push((
        "blk.0.ffn_gate_exps.0.weight".into(),
        float(vec![256, 256], 1.0),
    ));
    composite_encoded_overlay(&overlay, invalid, Some(vec![0, 2]));
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(BoundCompositeSource::compile(&source, LoadScope::Full).is_err());
}

#[test]
fn composite_access_refuses_inherited_pruned_weights_and_ambiguous_encodings() {
    use super::composite::BoundCompositeSource;
    let base = retained_repack_fixture(|_| {});
    let overlay = base.dir.join("overlay");
    composite_encoded_overlay(
        &overlay,
        vec![(
            "blk.0.ffn_gate_exps.0.weight".into(),
            float(vec![256, 256], 1.0),
        )],
        None,
    );
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(BoundCompositeSource::compile(&source, LoadScope::Full).is_err());
    let base = complete_repack_fixture(|_| {});
    let overlay = base.dir.join("overlay");
    composite_encoded_overlay(
        &overlay,
        vec![
            (
                "blk.0.ffn_gate_exps.1.weight".into(),
                float(vec![32, 32], 1.0),
            ),
            (
                "blk.0.ffn_gate_exps.weight".into(),
                float(vec![32, 32, 4], 1.0),
            ),
        ],
        None,
    );
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert!(
        BoundCompositeSource::compile(&source, LoadScope::Full)
            .err()
            .unwrap()
            .contains("both stacked and split")
    );
}

#[test]
fn composite_access_preserves_native_planes_transforms_and_shadowed_identity() {
    use super::composite::BoundCompositeSource;
    let f = fixture(QWEN, |ts| nvfp4(ts, 2.0));
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let id = layer(LayerTensor::Query);
    let native = bound.nvfp4(&id, None).unwrap().unwrap();
    assert_eq!(native.wbytes, &[0x22; 512]);
    assert_eq!((native.out_f, native.in_f), (32, 32));
    let identity = bound.artifact_identity().unwrap();
    std::fs::rename(
        f.dir.join("model.safetensors"),
        f.dir.join("old.safetensors"),
    )
    .unwrap();
    std::fs::copy(
        f.dir.join("old.safetensors"),
        f.dir.join("model.safetensors"),
    )
    .unwrap();
    let mut bytes = std::fs::read(f.dir.join("model.safetensors")).unwrap();
    // Modify a shadowed base norm's payload through its recorded offset, never its header.
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let offset = header["model.norm.weight"]["data_offsets"][0]
        .as_u64()
        .unwrap() as usize
        + 8
        + header_len;
    bytes[offset..offset + 4].copy_from_slice(&9.0f32.to_le_bytes());
    std::fs::write(f.dir.join("model.safetensors"), bytes).unwrap();
    assert_eq!(bound.artifact_identity().unwrap(), identity);
    let new = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let newbound = BoundCompositeSource::compile(&new, LoadScope::Full).unwrap();
    assert_eq!(
        newbound.catalog().binding_sha256(),
        bound.catalog().binding_sha256()
    );
    assert_ne!(
        newbound.artifact_identity().unwrap().opened_source_sha256,
        identity.opened_source_sha256
    );
    let f = fixture(HYBRID, |_| {});
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.25)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let id = TensorId::Layer {
        index: 1,
        tensor: LayerTensor::QueryNorm,
    };
    assert_eq!(
        f32::from_le_bytes(bound.tensor(&id).unwrap().bytes[..4].try_into().unwrap()),
        1.25
    );
}

#[test]
fn composite_access_refuses_invalid_scale_values_before_optional_native_selection() {
    use super::composite::BoundCompositeSource;
    for value in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        let f = fixture(QWEN, |ts| nvfp4(ts, value));
        let overlay = f.dir.join("overlay");
        // Even a replaced base query must not conceal its malformed declared scale.
        composite_overlay(&overlay, &[("blk.0.attn_q.weight", vec![32, 32], 1.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        assert!(
            BoundCompositeSource::compile(&source, LoadScope::Full)
                .err()
                .unwrap()
                .contains("scale values")
        );
    }
}

#[test]
fn composite_access_prepares_source_factors_and_preserves_text_scope() {
    use super::composite::BoundCompositeSource;
    let f = step_with_vision_fixture(|_| {});
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![16], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Text).unwrap();
    assert!(
        bound
            .config()
            .step35
            .as_ref()
            .unwrap()
            .rope_freq_factors
            .is_some()
    );
    let vision = bound.catalog().components()[1]
        .binding()
        .tensors
        .iter()
        .find(|(_, t)| matches!(t.owner, crate::tensor_contract::TensorOwner::Vision(_)))
        .unwrap()
        .0;
    assert!(
        bound
            .tensor(vision)
            .err()
            .unwrap()
            .contains("outside selected")
    );
    assert!(BoundCompositeSource::compile(&source, LoadScope::Full).is_err());
    let pack = model_packs::for_config(bound.config()).unwrap();
    let shape = pack
        .compile_tensor_contract(
            bound.config(),
            bound.plan(),
            CheckpointDialect::Gguf,
            pack.contract_options(bound.config()),
        )
        .unwrap()
        .requirements
        .into_iter()
        .find(|r| r.id == TensorId::RopeFactors)
        .unwrap()
        .shape;
    drop(bound);
    drop(source);
    composite_overlay(&overlay, &[("rope_freqs.weight", shape, 0.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let error = BoundCompositeSource::compile(&source, LoadScope::Text)
        .err()
        .unwrap();
    assert!(error.contains("finite positive factors"), "{error}");
}

#[test]
fn composite_access_preserves_stacked_fp8_and_nvfp4_planes() {
    use super::composite::BoundCompositeSource;
    let raw = include_str!("../model_packs/step35/contract-fixture.json");
    let config = ModelConfig::from_hf(&HfConfig::try_parse(raw).unwrap());
    let pack = model_packs::for_config(&config).unwrap();
    let plan = pack.compile_plan(&config).unwrap();
    let id = TensorId::Layer {
        index: 1,
        tensor: LayerTensor::MoeExpertGateBank,
    };
    let requirement = pack
        .compile_tensor_contract(
            &config,
            &plan,
            CheckpointDialect::HfSafetensors,
            pack.contract_options(&config),
        )
        .unwrap()
        .requirements
        .into_iter()
        .find(|r| r.id == id)
        .unwrap();
    let [experts, out, input]: [u64; 3] = requirement.shape.clone().try_into().unwrap();
    let stem = requirement.names[0].strip_suffix(".weight").unwrap();
    for packed in [false, true] {
        let f = fixture(raw, |tensors| {
            tensors.insert(
                requirement.names[0].clone(),
                Tensor {
                    dtype: if packed { "U8" } else { "F8_E4M3" },
                    shape: vec![experts, out, if packed { input / 2 } else { input }],
                    bytes: vec![
                        0x22;
                        (experts * out * input / if packed { 2 } else { 1 }) as usize
                    ],
                },
            );
            if packed {
                tensors.insert(
                    format!("{stem}.weight_scale"),
                    Tensor {
                        dtype: "F8_E4M3",
                        shape: vec![experts, out, input / 16],
                        bytes: vec![0x38; (experts * out * input / 16) as usize],
                    },
                );
                tensors.insert(format!("{stem}.weight_scale_2"), float(vec![experts], 2.0));
            } else {
                tensors.insert(
                    format!("{stem}.weight_scale_inv"),
                    float(vec![experts, out.div_ceil(128), input.div_ceil(128)], 0.5),
                );
            }
        });
        let overlay = f.dir.join("overlay");
        composite_overlay(&overlay, &[("output_norm.weight", vec![16], 1.0)]);
        let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
        let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
        if packed {
            let native = bound.nvfp4_stacked(&id).unwrap().unwrap();
            assert_eq!(
                native.codes,
                vec![0x22; (experts * out * input / 2) as usize]
            );
            assert_eq!(
                native.scales,
                vec![0x38; (experts * out * input / 16) as usize]
            );
            assert_eq!(native.macros, vec![2.0; experts as usize]);
            assert!(bound.fp8_stacked(&id).unwrap().is_none());
        } else {
            let native = bound.fp8_stacked(&id).unwrap().unwrap();
            assert_eq!(native.bytes, vec![0x22; (experts * out * input) as usize]);
            assert_eq!(native.scales, vec![0.5; experts as usize]);
            let disk = bound.disk(&id, None).unwrap().unwrap();
            assert_eq!(disk.bytes(), native.bytes);
            assert!(bound.nvfp4_stacked(&id).unwrap().is_none());
        }
        assert!(
            bound
                .member_tensor(&id, 0)
                .err()
                .unwrap()
                .contains("whole-bank slicing")
        );
    }
}

#[test]
fn composite_access_tied_head_and_opened_metadata_identity_remain_separate() {
    use super::composite::BoundCompositeSource;
    let raw = QWEN.replacen('{', "{\"tie_word_embeddings\":true,", 1);
    let f = fixture(&raw, |_| {});
    let overlay = f.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    assert_eq!(
        bound.tensor(&TensorId::OutputProjection).unwrap().bytes,
        bound.tensor(&TensorId::TokenEmbedding).unwrap().bytes
    );
    let identity = bound.artifact_identity().unwrap();
    let path = overlay.join("manifest.json");
    let manifest = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!(" {manifest}\n")).unwrap();
    assert_eq!(bound.artifact_identity().unwrap(), identity);
    let new = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let newbound = BoundCompositeSource::compile(&new, LoadScope::Full).unwrap();
    assert_eq!(
        bound.catalog().binding_sha256(),
        newbound.catalog().binding_sha256()
    );
    assert_ne!(
        identity.opened_source_sha256,
        newbound.artifact_identity().unwrap().opened_source_sha256
    );
    // Padding outside the declared tensor is also part of the opened component identity.
    std::fs::rename(overlay.join("weights.bin"), overlay.join("retained.bin")).unwrap();
    let mut bytes = std::fs::read(overlay.join("retained.bin")).unwrap();
    bytes.extend([1, 2, 3, 4]);
    std::fs::write(overlay.join("weights.bin"), bytes).unwrap();
    let padded = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    assert_ne!(
        newbound.artifact_identity().unwrap().opened_source_sha256,
        BoundCompositeSource::compile(&padded, LoadScope::Full)
            .unwrap()
            .artifact_identity()
            .unwrap()
            .opened_source_sha256
    );
}

#[test]
fn composite_access_tied_head_uses_the_embeddings_independent_scale() {
    use super::composite::BoundCompositeSource;
    // The independent embedding-scale contract does not require a MoE family whose
    // current pack forbids tied heads. Exercise it on an explicitly tied dense program.
    let cfg = QWEN.replacen('{', "{\"tie_word_embeddings\":true,", 1);
    let base = complete_repack_fixture_config(32, &cfg, |rows| {
        rows.insert("token_embd.scale".into(), float(vec![1], 0.5));
    });
    let overlay = base.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let scale = bound
        .auxiliary(
            &TensorId::OutputProjection,
            None,
            QuantAuxTensor::WeightScale,
        )
        .unwrap()
        .unwrap();
    assert_eq!(scale.bytes.as_ref(), &0.5f32.to_le_bytes());
}

#[test]
fn composite_access_mask_selects_original_hf_members_without_compact_index_confusion() {
    use super::composite::BoundCompositeSource;
    let raw = QWEN.replace("\"qwen3\"", "\"qwen3_moe\"").replacen(
        '{',
        "{\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,",
        1,
    );
    let base = fixture(&raw, |rows| {
        for original in 0..4 {
            rows.insert(
                format!("model.layers.0.mlp.experts.{original}.gate_proj.weight"),
                float(vec![32, 32], original as f32 + 1.0),
            );
        }
    });
    let overlay = base.dir.join("overlay");
    composite_encoded_overlay(&overlay, Vec::new(), Some(vec![0, 2]));
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let id = layer(LayerTensor::MoeExpertGateBank);
    for (original, expected) in [(1, 2.0f32), (3, 4.0)] {
        let value = bound.member_tensor(&id, original).unwrap();
        assert_eq!(
            f32::from_le_bytes(value.bytes[..4].try_into().unwrap()),
            expected
        );
    }
    assert!(bound.member_tensor(&id, 0).is_err());
    assert!(bound.member_tensor(&id, 2).is_err());
    // Pruned physical records remain inventoried in the original component, never selected.
    assert_eq!(
        bound.catalog().components()[1].binding().tensors[&id]
            .checkpoint_names
            .len(),
        4
    );
    assert_eq!(bound.catalog().selected()[&id].member_ids, Some(vec![1, 3]));
}

fn canonical_output_fixture(
    moe: bool,
    edit: impl FnOnce(&mut BTreeMap<String, Tensor>),
) -> Fixture {
    let mut raw = QWEN
        .replace("\"hidden_size\":32", "\"hidden_size\":64")
        .replace("\"head_dim\":16", "\"head_dim\":32");
    if moe {
        raw = raw.replace("\"qwen3\"", "\"qwen3_moe\"").replacen(
            '{',
            "{\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":64,",
            1,
        );
    }
    fixture(&raw, |rows| {
        let names = if moe {
            (0..4)
                .map(|id| format!("model.layers.0.mlp.experts.{id}.gate_proj"))
                .collect::<Vec<_>>()
        } else {
            vec!["model.layers.0.self_attn.q_proj".to_owned()]
        };
        for (index, stem) in names.into_iter().enumerate() {
            rows.insert(
                format!("{stem}.weight"),
                Tensor {
                    dtype: "U8",
                    shape: vec![64, 32],
                    bytes: vec![0x22 + index as u8; 2048],
                },
            );
            rows.insert(
                format!("{stem}.weight_scale"),
                Tensor {
                    dtype: "F8_E4M3",
                    shape: vec![64, 4],
                    bytes: vec![0x38; 256],
                },
            );
            rows.insert(
                format!("{stem}.weight_scale_2"),
                float(vec![1], index as f32 + 2.0),
            );
        }
        edit(rows);
    })
}

#[cfg(unix)]
#[test]
fn canonical_output_uses_bound_operands_and_retains_source_root_after_path_replacement() {
    let f = canonical_output_fixture(false, |_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::Query);
    let identity = bound.artifact_identity().unwrap();
    let native = bound.nvfp4(&id).unwrap().unwrap();
    let expected = crate::nvfp4_repack::repack_modelopt_to_gguf(
        native.wbytes,
        &native.wscale,
        native.out_f,
        native.in_f,
    );
    let retained = f.dir.with_extension("retained");
    std::fs::rename(&f.dir, &retained).unwrap();
    std::fs::create_dir(&f.dir).unwrap();
    std::fs::write(f.dir.join("sentinel"), b"replacement").unwrap();
    let view = bound.canonical_nvfp4(&id, None).unwrap().unwrap();
    assert_eq!(view.bytes(), expected);
    assert_eq!(bound.artifact_identity().unwrap(), identity);
    assert_eq!(std::fs::read_dir(&f.dir).unwrap().count(), 1);
    assert_eq!(
        std::fs::read_dir(retained.join(".memra-repack"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        f32::from_le_bytes(
            bound
                .auxiliary(&id, QuantAuxTensor::WeightScale)
                .unwrap()
                .unwrap()
                .bytes[..4]
                .try_into()
                .unwrap()
        ),
        2.0
    );
    drop(bound);
    drop(source);
    std::fs::remove_dir_all(&retained).unwrap();
    let mut bytes = vec![0; expected.len()];
    view.read_at(&mut bytes, 0).unwrap();
    assert_eq!(bytes, expected);
    assert!(view.read_at(&mut [0; 2], view.len() as u64 - 1).is_err());
}

#[cfg(unix)]
#[test]
fn canonical_output_ignores_poisoned_cache_bytes_and_refuses_bad_output_targets() {
    let f = canonical_output_fixture(false, |_| {});
    let cache = f.dir.join(".memra-repack");
    std::fs::create_dir(&cache).unwrap();
    std::fs::write(cache.join("query.nvfp4"), vec![0xff; 64 * 36]).unwrap();
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let id = layer(LayerTensor::Query);
    let native = bound.nvfp4(&id).unwrap().unwrap();
    let expected =
        crate::nvfp4_repack::repack_modelopt_to_gguf(native.wbytes, &native.wscale, 64, 64);
    let first = bound.canonical_nvfp4(&id, None).unwrap().unwrap();
    std::fs::write(cache.join("query.nvfp4"), vec![0x11; 64 * 36]).unwrap();
    let second = bound.canonical_nvfp4(&id, None).unwrap().unwrap();
    assert_eq!(first.bytes(), expected);
    assert_eq!(second.bytes(), expected);
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
    let bad = canonical_output_fixture(false, |_| {});
    let outside = bad.dir.join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, bad.dir.join(".memra-repack")).unwrap();
    let source = SafetensorsSource::open(&bad.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    assert!(bound.canonical_nvfp4(&id, None).is_err());
    assert_eq!(std::fs::read_dir(outside).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn canonical_output_gathers_original_composite_members_and_preserves_macros() {
    use super::composite::BoundCompositeSource;
    let f = canonical_output_fixture(true, |_| {});
    let overlay = f.dir.join("overlay");
    composite_encoded_overlay(&overlay, Vec::new(), Some(vec![0, 2]));
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let bound = BoundCompositeSource::compile(&source, LoadScope::Full).unwrap();
    let id = layer(LayerTensor::MoeExpertGateBank);
    let mut expected = Vec::new();
    for original in [1u32, 3] {
        let native = bound.nvfp4(&id, Some(original)).unwrap().unwrap();
        expected.extend(crate::nvfp4_repack::repack_modelopt_to_gguf(
            native.wbytes,
            &native.wscale,
            64,
            64,
        ));
        let scale = bound
            .auxiliary(&id, Some(original), QuantAuxTensor::WeightScale)
            .unwrap()
            .unwrap();
        assert_eq!(
            f32::from_le_bytes(scale.bytes[..4].try_into().unwrap()),
            original as f32 + 2.0
        );
    }
    let view = bound.canonical_nvfp4_bank(&id).unwrap().unwrap();
    assert_eq!(view.bytes(), expected);
    let member = bound.canonical_nvfp4(&id, Some(3)).unwrap().unwrap();
    assert_eq!(member.bytes(), &expected[64 * 36..]);
    assert!(bound.canonical_nvfp4(&id, Some(0)).is_err());
    assert_eq!(
        std::fs::read_dir(f.dir.join(".memra-repack"))
            .unwrap()
            .count(),
        0
    );
    assert!(!overlay.join(".memra-repack").exists());
}

#[cfg(unix)]
#[test]
fn canonical_output_member_failure_leaves_no_prefix_or_named_temporary() {
    let f = canonical_output_fixture(true, |rows| {
        for suffix in ["weight", "weight_scale", "weight_scale_2"] {
            rows.remove(&format!("model.layers.0.mlp.experts.2.gate_proj.{suffix}"));
        }
        rows.insert(
            "model.layers.0.mlp.experts.2.gate_proj.weight".into(),
            float(vec![64, 64], 0.25),
        );
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    assert!(
        bound
            .canonical_nvfp4_bank(&layer(LayerTensor::MoeExpertGateBank))
            .err()
            .unwrap()
            .contains("not native NVFP4")
    );
    assert_eq!(
        std::fs::read_dir(f.dir.join(".memra-repack"))
            .unwrap()
            .count(),
        0
    );
}

#[cfg(unix)]
#[test]
fn canonical_output_stacked_bank_matches_the_existing_codec() {
    let raw = include_str!("../model_packs/step35/contract-fixture.json")
        .replace("\"hidden_size\": 16", "\"hidden_size\": 64");
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&raw).unwrap());
    let pack = model_packs::for_config(&config).unwrap();
    let plan = pack.compile_plan(&config).unwrap();
    let id = TensorId::Layer {
        index: 1,
        tensor: LayerTensor::MoeExpertGateBank,
    };
    let r = pack
        .compile_tensor_contract(
            &config,
            &plan,
            CheckpointDialect::HfSafetensors,
            pack.contract_options(&config),
        )
        .unwrap()
        .requirements
        .into_iter()
        .find(|r| r.id == id)
        .unwrap();
    let [n, out, input]: [u64; 3] = r.shape.clone().try_into().unwrap();
    let stem = r.names[0].strip_suffix(".weight").unwrap();
    let f = fixture(&raw, |rows| {
        rows.insert(
            r.names[0].clone(),
            Tensor {
                dtype: "U8",
                shape: vec![n, out, input / 2],
                bytes: vec![0x22; (n * out * input / 2) as usize],
            },
        );
        rows.insert(
            format!("{stem}.weight_scale"),
            Tensor {
                dtype: "F8_E4M3",
                shape: vec![n, out, input / 16],
                bytes: vec![0x38; (n * out * input / 16) as usize],
            },
        );
        rows.insert(format!("{stem}.weight_scale_2"), float(vec![n], 2.0));
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let bound = BoundTensorSource::compile(&source).unwrap();
    let native = bound.nvfp4_stacked(&id).unwrap().unwrap();
    assert_eq!(native.macros, vec![2.0; n as usize]);
    let expected = crate::nvfp4_repack::repack_modelopt_to_gguf(
        &vec![0x22; (out * input / 2) as usize],
        &vec![0x38; (out * input / 16) as usize],
        out as usize,
        input as usize,
    )
    .repeat(n as usize);
    let view = bound.canonical_nvfp4_bank(&id).unwrap().unwrap();
    assert_eq!(view.bytes(), expected);
}

#[test]
fn prepared_root_validates_before_callback_and_never_uses_legacy_name_reads() {
    use super::PreparedModelSource;
    struct NoLegacy<'a>(&'a SafetensorsSource);
    impl TensorSource for NoLegacy<'_> {
        fn config(&self) -> ModelConfig {
            self.0.config()
        }
        fn bound_interpretation(&self) -> Result<BoundSourceInterpretation, String> {
            self.0.bound_interpretation()
        }
        fn tensor_census(&self) -> Result<TensorCensus, String> {
            self.0.tensor_census()
        }
        fn validate_bound_metadata(&self, r: &BoundTensorRequest<'_>) -> Result<(), String> {
            self.0.validate_bound_metadata(r)
        }
        fn read_bound(&self, r: &BoundTensorRequest<'_>) -> Result<TensorView<'_>, String> {
            self.0.read_bound(r)
        }
        fn find(&self, _: &str) -> Option<TensorView<'_>> {
            panic!("root used a legacy name read")
        }
    }
    let good = fixture(QWEN, |_| {});
    let source = SafetensorsSource::open(&good.dir).unwrap();
    let guarded = NoLegacy(&source);
    let prepared = PreparedModelSource::text(&guarded).unwrap();
    prepared
        .with_runtime(|runtime| {
            let (config, plan) = model_packs::compile_for_source(runtime).unwrap();
            assert_eq!(plan.hidden_size, config.n_embd);
            assert_eq!(
                runtime
                    .try_find("token_embd.weight")
                    .unwrap()
                    .unwrap()
                    .bytes,
                source.find("token_embd.weight").unwrap().bytes
            );
            assert!(runtime.try_find("rope_freqs.weight").unwrap().is_none());
            assert!(runtime.try_find("rope_freq.weight").is_err());
        })
        .unwrap();
    for bad in ["missing", "shape", "integer", "unknown", "alias"] {
        let f = fixture(QWEN, |rows| match bad {
            "missing" => {
                rows.remove("model.layers.0.self_attn.q_proj.weight");
            }
            "shape" => {
                rows.insert(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    float(vec![16, 32], 0.25),
                );
            }
            "integer" => {
                rows.insert(
                    "model.layers.0.self_attn.q_proj.weight".into(),
                    Tensor {
                        dtype: "I64",
                        shape: vec![32, 32],
                        bytes: vec![0; 8192],
                    },
                );
            }
            "unknown" => {
                rows.insert("model.surprise.weight".into(), float(vec![1], 0.25));
            }
            "alias" => {
                rows.insert(
                    "model.language_model.layers.0.self_attn.q_proj.weight".into(),
                    float(vec![32, 32], 0.25),
                );
            }
            _ => unreachable!(),
        });
        let source = SafetensorsSource::open(&f.dir).unwrap();
        assert!(
            PreparedModelSource::text(&NoLegacy(&source)).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn prepared_root_gguf_and_existing_bundle_preserve_bytes_and_identity() {
    use super::PreparedModelSource;
    let f = fixture(QWEN, |_| {});
    let path = f.dir.join("micro.gguf");
    crate::micro_gguf::write_glm_dsa_micro(&path, 7541).unwrap();
    let file = crate::GgufFile::open(&path).unwrap();
    let source = GgufSource(&file);
    let original = source.artifact_sha256().unwrap();
    let prepared = PreparedModelSource::text(&source).unwrap();
    let explicit = BoundTensorSource::compile_for_scope(&source, LoadScope::Text).unwrap();
    prepared
        .with_runtime(|runtime| {
            assert_eq!(
                runtime.artifact_sha256().unwrap(),
                explicit.artifact_identity().unwrap().artifact_sha256
            );
            assert_eq!(
                runtime
                    .try_find("token_embd.weight")
                    .unwrap()
                    .unwrap()
                    .bytes,
                source.find("token_embd.weight").unwrap().bytes
            );
            assert!(matches!(
                runtime.try_find_gguf_disk("token_embd.weight").unwrap(),
                Some(crate::bound_disk::ExpertDiskView::Bound(_))
            ));
            PreparedModelSource::text(runtime)
                .unwrap()
                .with_runtime(|again| {
                    assert_eq!(
                        again.artifact_sha256().unwrap(),
                        runtime.artifact_sha256().unwrap()
                    );
                })
                .unwrap();
        })
        .unwrap();
    assert_eq!(source.artifact_sha256().unwrap(), original);
}

#[test]
fn prepared_root_composite_uses_original_ids_and_keeps_disk_ownership() {
    use super::PreparedModelSource;
    let base = retained_repack_fixture(|_| {});
    let overlay = base.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![256], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let prepared = PreparedModelSource::text(&source).unwrap();
    let view = prepared
        .with_runtime(|runtime| {
            let (_, plan) = model_packs::compile_for_source(runtime).unwrap();
            assert!(
                plan.operations()
                    .contains(&crate::model_plan::OperationKind::RetainedExpertRouting)
            );
            assert_eq!(
                runtime.active_experts(0),
                Some(&[false, true, false, true][..])
            );
            assert!(!runtime.try_has("blk.0.ffn_gate_exps.weight").unwrap());
            assert_eq!(
                runtime
                    .try_find("blk.0.ffn_gate_exps.1.weight")
                    .unwrap()
                    .unwrap()
                    .ggml_type,
                crate::GgmlType::Q2_K
            );
            assert_eq!(
                runtime
                    .try_find("blk.0.ffn_gate_exps.3.weight")
                    .unwrap()
                    .unwrap()
                    .ggml_type,
                crate::GgmlType::NVFP4
            );
            assert!(runtime.try_find("blk.0.ffn_gate_exps.0.weight").is_err());
            assert!(runtime.try_find("blk.99.ffn_gate_exps.1.weight").is_err());
            assert!(runtime.try_find("blk.0.ffn_typo.weight").is_err());
            assert_eq!(
                runtime
                    .physical_tensor_inventory()
                    .unwrap()
                    .components
                    .len(),
                2
            );
            PreparedModelSource::text(runtime)
                .unwrap()
                .with_runtime(|again| {
                    assert_eq!(
                        again.artifact_sha256().unwrap(),
                        runtime.artifact_sha256().unwrap()
                    )
                })
                .unwrap();
            runtime
                .try_find_expert_disk("blk.0.ffn_gate_exps.1.weight")
                .unwrap()
                .unwrap()
        })
        .unwrap();
    drop(prepared);
    drop(source);
    assert!(matches!(view, crate::bound_disk::ExpertDiskView::Bound(_)));
    assert!(view.bytes().iter().all(|&b| b == 0x11));
}

#[test]
fn prepared_root_composite_native_and_canonical_routes_preserve_macro_scales() {
    use super::PreparedModelSource;
    let f = canonical_output_fixture(true, |_| {});
    let overlay = f.dir.join("overlay");
    composite_encoded_overlay(&overlay, Vec::new(), Some(vec![0, 2]));
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let prepared = PreparedModelSource::text(&source).unwrap();
    prepared
        .with_runtime(|runtime| {
            assert!(!runtime.try_has("blk.0.ffn_gate_exps.weight").unwrap());
            assert!(
                runtime
                    .try_find_nvfp4_stacked_native("blk.0.ffn_gate_exps.weight")
                    .unwrap()
                    .is_none()
            );
            let mut expected = Vec::new();
            for index in [1, 3] {
                let native = runtime
                    .try_find_nvfp4_native(&format!("blk.0.ffn_gate_exps.{index}.weight"))
                    .unwrap()
                    .unwrap();
                expected.extend(crate::nvfp4_repack::repack_modelopt_to_gguf(
                    native.wbytes,
                    &native.wscale,
                    64,
                    64,
                ));
                let scale = runtime
                    .try_find(&format!("blk.0.ffn_gate_exps.{index}.scale"))
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    f32::from_le_bytes(scale.bytes[..4].try_into().unwrap()),
                    index as f32 + 2.0
                );
            }
            assert_eq!(
                runtime
                    .try_canonical_nvfp4_bank("blk.0.ffn_gate_exps.weight")
                    .unwrap()
                    .unwrap()
                    .bytes(),
                expected
            );
            assert!(
                runtime
                    .try_canonical_nvfp4_bank("blk.0.ffn_up_exps.weight")
                    .is_err()
            );
            assert!(
                runtime
                    .try_find_nvfp4_native("blk.0.ffn_gate_exps.0.weight")
                    .is_err()
            );
        })
        .unwrap();
}

#[test]
fn prepared_root_text_scope_validates_vision_and_preserves_native_source_config() {
    use super::PreparedModelSource;
    let f = step_with_vision_fixture(|_| {});
    let source = SafetensorsSource::open(&f.dir).unwrap();
    let raw = source.raw_config_json().unwrap().to_owned();
    let before = source.artifact_sha256().unwrap();
    let prepared = PreparedModelSource::text(&source).unwrap();
    prepared
        .with_runtime(|runtime| {
            let (config, plan) = model_packs::compile_for_source(runtime).unwrap();
            assert!(plan.vision.is_none() && plan.multimodal.is_none());
            assert!(config.step35.is_some());
            assert!(runtime.try_find("vision_model.conv1.weight").is_err());
        })
        .unwrap();
    assert_eq!(source.raw_config_json(), Some(raw.as_str()));
    assert_eq!(source.artifact_sha256().unwrap(), before);
    let f = step_with_vision_fixture(|rows| {
        rows.remove("vision_model.conv1.weight");
    });
    let source = SafetensorsSource::open(&f.dir).unwrap();
    assert!(PreparedModelSource::text(&source).is_err());
}

#[test]
fn composite_preparation_retains_the_packs_separate_head_refusal() {
    let base = complete_repack_fixture(|rows| {
        rows.remove("output.weight");
    });
    let path = base.dir.join("config.json");
    let raw = std::fs::read_to_string(&path).unwrap();
    std::fs::write(path, raw.replacen('{', "{\"tie_word_embeddings\":true,", 1)).unwrap();
    let overlay = base.dir.join("overlay");
    composite_overlay(&overlay, &[("output_norm.weight", vec![32], 1.0)]);
    let source = crate::source::Hy3RepackSource::open(&overlay).unwrap();
    let error = match super::composite::BoundCompositeSource::compile(&source, LoadScope::Full) {
        Ok(_) => panic!("separate-head pack accepted a tied program"),
        Err(error) => error,
    };
    assert!(
        error.contains("requires a separate output projection"),
        "{error}"
    );
}
