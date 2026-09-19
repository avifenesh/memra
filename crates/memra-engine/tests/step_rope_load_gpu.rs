//! #537: the loaded Step HF RoPE buffer must equal the checkpoint-tensor path.
//! Run only under the coordinator's per-card lock wrapper, with one visible GPU:
//! NVIDIA_TF32_OVERRIDE=0 cargo test --release -p memra-engine --test step_rope_load_gpu \
//!     -- --ignored --test-threads=1 --nocapture
//! This synthetic gate does not qualify an official checkpoint or a quantization format.

use memra_engine::{Engine, hybrid::HybridModel};
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use std::borrow::Cow;
use std::collections::BTreeMap;

struct Tensor {
    bytes: Vec<u8>,
    shape: Vec<u64>,
    dtype: GgmlType,
}

struct Source {
    config: ModelConfig,
    tensors: BTreeMap<String, Tensor>,
}

impl TensorSource for Source {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }

    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let tensor = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&tensor.bytes),
            ne: tensor.shape.clone(),
            ggml_type: tensor.dtype,
        })
    }
}

fn fixture() -> Result<(Source, ModelPlan, Vec<f32>), Box<dyn std::error::Error>> {
    let config = ModelConfig::from_hf(&HfConfig::parse(
        r#"{"model_type":"step3p5","num_hidden_layers":2,"hidden_size":256,
        "intermediate_size":512,"num_attention_heads":2,"num_attention_groups":1,
        "head_dim":128,"vocab_size":32,"max_position_embeddings":256,
        "moe_num_experts":4,"moe_top_k":2,"moe_intermediate_size":128,
        "share_expert_dim":128,"moe_layers_enum":"1","norm_expert_weight":true,
        "moe_router_activation":"sigmoid","moe_router_scaling_factor":1.0,
        "sliding_window":16,"layer_types":["full_attention","sliding_attention"],
        "rope_theta":[10000,10000],"partial_rotary_factors":[0.5,1.0],
        "rope_scaling":{"rope_type":"llama3","factor":8.0,
        "original_max_position_embeddings":8,"low_freq_factor":1.0,"high_freq_factor":4.0},
        "attention_other_setting":{"num_attention_heads":2,"num_attention_groups":1},
        "swiglu_limits":[0,0],"swiglu_limits_shared":[0,0]}"#,
    ));
    let factors = config
        .step35
        .as_ref()
        .unwrap()
        .rope_freq_factors
        .clone()
        .unwrap();
    assert!(factors.iter().any(|&factor| factor > 1.0));
    let plan = memra_gguf::model_packs::compile_for_load(&config)?;
    assert_eq!(
        memra_gguf::execution_manifest::decode_batch_program(&plan),
        memra_gguf::execution_manifest::DecodeBatchProgram::SlidingGatedMoe
    );
    let fixture = memra_reference::deterministic_fixture(&plan)?;
    let contract = TensorContract::for_plan(
        &plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )?;
    let mut tensors = BTreeMap::new();
    for required in &contract.requirements {
        // HF has no factor tensor: normalization must carry its declared program.
        if required.id == TensorId::RopeFactors {
            continue;
        }
        let Some(weight) = fixture.weights.get(&required.id) else {
            assert!(!required.required, "missing {:?}", required.id);
            continue;
        };
        let expert = matches!(
            required.id,
            TensorId::Layer {
                tensor: LayerTensor::MoeExpertGateBank
                    | LayerTensor::MoeExpertUpBank
                    | LayerTensor::MoeExpertDownBank,
                ..
            }
        );
        let (bytes, dtype) = if expert {
            // The synthetic expert banks use a supported uniform layout. Both sides
            // read identical bytes; this is not evidence for a checkpoint's quantization.
            (
                memra_gguf::nvfp4_repack::f32_to_q8_0(&weight.data),
                GgmlType::Q8_0,
            )
        } else {
            (
                weight
                    .data
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                GgmlType::F32,
            )
        };
        let names = match required.match_mode {
            TensorMatch::OneOf => &required.names[..1],
            TensorMatch::All => required.names.as_slice(),
        };
        for name in names {
            tensors.insert(
                name.clone(),
                Tensor {
                    bytes: bytes.clone(),
                    shape: required.shape.clone(),
                    dtype,
                },
            );
        }
    }
    Ok((Source { config, tensors }, plan, factors))
}

fn logits(
    engine: &Engine,
    source: &Source,
    expected_factors: Option<&[f32]>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let model = HybridModel::load_from_source_without_mtp(engine, source)?;
    let uploaded = model
        .step35_aux
        .as_ref()
        .expect("Step auxiliary missing")
        .rope_freqs(engine)
        .map(|device| engine.dtoh(device))
        .transpose()?;
    assert_eq!(
        uploaded.as_deref(),
        expected_factors,
        "loaded factor bytes differ"
    );
    let tokens: Vec<u32> = (0..48).map(|index| 1 + (index * 7 % 30) as u32).collect();
    let mut cache = memra_engine::pp::new_cache(engine, &model.cfg, 64)?;
    let mut rows = Vec::new();
    for token in tokens {
        rows.extend(model.decode_step(engine, token, &mut cache)?);
    }
    assert!(rows.iter().all(|value| value.is_finite()));
    Ok(rows)
}

#[test]
#[ignore = "requires one exclusively locked RTX PRO 6000 Blackwell GPU"]
fn hf_factors_match_explicit_tensor_and_the_omitted_factor_mutation_diverges()
-> Result<(), Box<dyn std::error::Error>> {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES")?;
    assert!(!visible.is_empty() && visible.split(',').count() == 1);
    assert_eq!(std::env::var("NVIDIA_TF32_OVERRIDE").as_deref(), Ok("0"));
    let (mut source, _, factors) = fixture()?;
    let engine = Engine::new(0)?;
    let hf = logits(&engine, &source, Some(&factors))?;
    source.config.step35.as_mut().unwrap().rope_freq_factors = None;
    source.tensors.insert(
        "rope_freqs.weight".into(),
        Tensor {
            bytes: factors
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            shape: vec![factors.len() as u64],
            dtype: GgmlType::F32,
        },
    );
    let checkpoint = logits(&engine, &source, Some(&factors))?;
    assert_eq!(
        hf.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        checkpoint
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "HF and tensor-backed factors changed decode logits"
    );

    // The official GGUF storage extent is a full head. Keep its unused suffix,
    // and prove that the partial-rotary consumer still reads exactly its prefix.
    let mut full_head = factors.clone();
    full_head.resize(64, 999.0);
    source.tensors.insert(
        "rope_freqs.weight".into(),
        Tensor {
            bytes: full_head
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            shape: vec![64],
            dtype: GgmlType::F32,
        },
    );
    let stored = logits(&engine, &source, Some(&full_head))?;
    assert_eq!(
        hf.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        stored
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    // Diagnostic mutation recreates the old omission; it is never a supported model arm.
    source.tensors.remove("rope_freqs.weight");
    source.config.rope_scaling_hint = None;
    let omitted = logits(&engine, &source, None)?;
    let max_delta = hf
        .iter()
        .zip(omitted)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-6,
        "factor omission was numerically vacuous: {max_delta}"
    );
    println!(
        "STEP_ROPE_LOAD_PASS rows=48 hf_tensor_bit_identity=true omitted_max_delta={max_delta}"
    );
    Ok(())
}
