//! Text-only, layer-streamed reference runner for the pinned Xiaomi MXFP4 MiMo checkpoint.
//!
//! This is an offline numerical oracle driver. It reads the official checkpoint
//! through Memra's source decoder, but does not register native model support,
//! execute vision/audio/MTP, or serve requests.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs;
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::model_plan::{AttentionPlan, LayerPlan, MlpPlan, ModelPlan, TensorPresence};
use memra_gguf::safetensors::StModel;
use memra_gguf::source::SafetensorsSource;
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, ExpertTensor, LayerTensor, TensorContract, TensorId,
    TensorMatch, TensorRequirement, TensorTransform,
};
use memra_reference::{ReferenceTensor, ReferenceWeights, StreamedTrunkExecution};

type Fail = Box<dyn std::error::Error>;

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_v2_checkpoint_runner: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, tokens) = args.split_first().ok_or(
        "usage: mimo_v2_checkpoint_runner <source_dir> <token_id>... \
         (MEMRA_ORACLE_OUT=<tsv path>)",
    )?;
    if tokens.is_empty() {
        return Err("at least one token id is required".into());
    }
    if tokens.len() > 128 {
        return Err("reference parity runner accepts at most 128 input tokens".into());
    }
    let token_ids: Vec<u32> = tokens
        .iter()
        .map(|token| token.parse::<u32>())
        .collect::<Result<_, _>>()?;
    let out = std::env::var_os("MEMRA_ORACLE_OUT")
        .ok_or("MEMRA_ORACLE_OUT must name the output TSV path")?;
    run_checkpoint(Path::new(dir), &token_ids, Path::new(&out))
}

fn run_checkpoint(dir: &Path, token_ids: &[u32], out: &Path) -> Result<(), Fail> {
    if out.exists() {
        return Err(format!("refusing to overwrite {}", out.display()).into());
    }
    let started = Instant::now();
    let config_text = std::fs::read_to_string(dir.join("config.json"))?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let pack = model_packs::by_alias("mimo_v2_source")
        .ok_or("the MiMo source inspection profile is unavailable")?;
    let plan = pack.compile_plan(&config)?;
    let model = StModel::open(dir)?;
    let headers = model
        .names()
        .map(|name| (name.clone(), model.info(name).unwrap().clone()))
        .collect::<BTreeMap<_, _>>();
    let bound = inspect_pinned_source_headers(&config, &headers)?;
    eprintln!(
        "pinned source headers bound {} semantic tensors",
        bound.tensors.len()
    );
    drop(bound);
    drop(headers);
    drop(model);

    let source = SafetensorsSource::open(dir)?;
    let contract = pack.compile_tensor_contract(
        &config,
        &plan,
        CheckpointDialect::HfSafetensors,
        ContractOptions::default(),
    )?;
    let loader = Loader::new(&source, &contract);
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    let mut globals = ReferenceWeights::new();
    for (id, shape) in [
        (TensorId::TokenEmbedding, vec![vocab, hidden]),
        (TensorId::OutputNorm, vec![hidden]),
        (TensorId::OutputProjection, vec![vocab, hidden]),
    ] {
        globals.insert(id.clone(), loader.load(&id, shape)?);
    }
    eprintln!("globals loaded in {:.1}s", started.elapsed().as_secs_f32());

    let mut cursor = StreamedTrunkExecution::begin(&plan, &globals, token_ids)?;
    while let Some(layer) = cursor.next_layer() {
        let index = layer.index;
        let load_started = Instant::now();
        let weights = mimo_layer_weights(&loader, &plan, layer)?;
        eprintln!(
            "layer {index}: {} tensors decoded in {:.1}s",
            weights.len(),
            load_started.elapsed().as_secs_f32()
        );
        let step_started = Instant::now();
        cursor.step(&weights)?;
        eprintln!(
            "layer {index}: reference executed in {:.1}s",
            step_started.elapsed().as_secs_f32()
        );
    }
    let result = cursor.finish(&globals)?;
    let last = (result.tokens - 1) * result.vocab;
    let logits = &result.logits[last..last + result.vocab];
    let mut text = String::with_capacity(result.vocab * 20 + 256);
    text.push_str("format\tmemra-checkpoint-oracle-v1\n");
    text.push_str("engine\tmemra-reference-fp32\n");
    text.push_str("numeric_class\tsource-mxfp4-weights-float32-accumulation\n");
    writeln!(
        text,
        "tokens\t{}",
        token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )?;
    writeln!(text, "vocab\t{}", result.vocab)?;
    for (index, value) in logits.iter().enumerate() {
        writeln!(text, "logit\t{index}\t{:08x}", value.to_bits())?;
    }
    let tmp = out.with_file_name(format!(
        "{}.tmp",
        out.file_name()
            .ok_or("MEMRA_ORACLE_OUT must end in a file name")?
            .to_string_lossy()
    ));
    let mut staged = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    staged.write_all(text.as_bytes())?;
    staged.sync_all()?;
    drop(staged);
    std::fs::rename(&tmp, out)?;
    eprintln!(
        "wrote {} last-position logits in {:.1}s",
        logits.len(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

struct Loader<'a> {
    source: &'a SafetensorsSource,
    requirements: BTreeMap<&'a TensorId, &'a TensorRequirement>,
}

impl<'a> Loader<'a> {
    fn new(source: &'a SafetensorsSource, contract: &'a TensorContract) -> Self {
        Self {
            source,
            requirements: contract
                .requirements
                .iter()
                .map(|requirement| (&requirement.id, requirement))
                .collect(),
        }
    }

    fn values(&self, id: &TensorId) -> Result<Vec<f32>, Fail> {
        let requirement = self
            .requirements
            .get(id)
            .ok_or_else(|| format!("no MiMo source requirement for {id:?}"))?;
        if requirement.match_mode != TensorMatch::OneOf {
            return Err(format!("{id:?}: grouped tensor is not a MiMo source projection").into());
        }
        if requirement.transform != TensorTransform::Identity {
            return Err(format!(
                "{id:?}: unimplemented reference transform {:?}",
                requirement.transform
            )
            .into());
        }
        for name in &requirement.names {
            if let Some((values, _)) = self.source.dequant_f32_hf(name) {
                let expected: usize = requirement
                    .shape
                    .iter()
                    .try_fold(1usize, |total, &dim| {
                        total.checked_mul(usize::try_from(dim).ok()?)
                    })
                    .ok_or("MiMo tensor element count overflows")?;
                if values.len() != expected {
                    return Err(format!(
                        "{id:?}: decoded {} values, expected {expected}",
                        values.len()
                    )
                    .into());
                }
                return Ok(values);
            }
        }
        Err(format!("{id:?}: no declared source tensor resolved").into())
    }

    fn load(&self, id: &TensorId, shape: Vec<usize>) -> Result<ReferenceTensor, Fail> {
        let values = self.values(id)?;
        Ok(ReferenceTensor::new(shape, values)?)
    }

    fn expert_bank(
        &self,
        layer: u32,
        tensor: ExpertTensor,
        experts: usize,
        output: usize,
        input: usize,
    ) -> Result<ReferenceTensor, Fail> {
        let elements = experts
            .checked_mul(output)
            .and_then(|count| count.checked_mul(input))
            .ok_or("MiMo expert bank element count overflows")?;
        let mut bank = Vec::new();
        bank.try_reserve_exact(elements)?;
        for expert in 0..experts {
            let id = TensorId::Expert {
                layer,
                expert: expert as u32,
                tensor,
            };
            let values = self.values(&id)?;
            if values.len() != output * input {
                return Err(format!("{id:?}: decoded expert shape mismatch").into());
            }
            bank.extend(values);
        }
        Ok(ReferenceTensor::new(vec![experts, output, input], bank)?)
    }
}

struct LayerWeightShapes {
    tensors: Vec<(LayerTensor, Vec<usize>)>,
    banks: Vec<(LayerTensor, ExpertTensor, usize, usize)>,
}

fn layer_weight_shapes(plan: &ModelPlan, layer: &LayerPlan) -> Result<LayerWeightShapes, Fail> {
    let hidden = plan.hidden_size as usize;
    let index = layer.index;
    let mut tensors = vec![
        (LayerTensor::PreAttentionNorm, vec![hidden]),
        (LayerTensor::PreMlpNorm, vec![hidden]),
    ];
    let attention = match &layer.attention {
        AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
            attention
        }
        other => {
            return Err(format!("MiMo layer {index} has unsupported attention {other:?}").into());
        }
    };
    let q = attention.query_heads as usize * attention.key_head_dim as usize;
    let k = attention.kv_heads as usize * attention.key_head_dim as usize;
    let v = attention.kv_heads as usize * attention.value_head_dim as usize;
    tensors.push((LayerTensor::FusedQkv, vec![q + k + v, hidden]));
    tensors.push((
        LayerTensor::AttentionOutput,
        vec![
            hidden,
            attention.query_heads as usize * attention.value_head_dim as usize,
        ],
    ));
    if attention
        .mimo_math
        .is_some_and(|math| math.sink == TensorPresence::Required)
    {
        tensors.push((
            LayerTensor::AttentionSink,
            vec![attention.query_heads as usize],
        ));
    }
    let banks = match &layer.mlp {
        MlpPlan::Dense(dense) => {
            let width = dense.intermediate_size as usize;
            tensors.push((LayerTensor::MlpGate, vec![width, hidden]));
            tensors.push((LayerTensor::MlpUp, vec![width, hidden]));
            tensors.push((LayerTensor::MlpDown, vec![hidden, width]));
            Vec::new()
        }
        MlpPlan::Moe(moe) => {
            let experts = moe.expert_count as usize;
            let width = moe.expert_intermediate_size as usize;
            tensors.push((LayerTensor::MoeRouter, vec![experts, hidden]));
            tensors.push((LayerTensor::MoeRouterBias, vec![experts]));
            vec![
                (
                    LayerTensor::MoeExpertGateBank,
                    ExpertTensor::Gate,
                    width,
                    hidden,
                ),
                (
                    LayerTensor::MoeExpertUpBank,
                    ExpertTensor::Up,
                    width,
                    hidden,
                ),
                (
                    LayerTensor::MoeExpertDownBank,
                    ExpertTensor::Down,
                    hidden,
                    width,
                ),
            ]
        }
    };
    Ok(LayerWeightShapes { tensors, banks })
}

fn mimo_layer_weights(
    loader: &Loader<'_>,
    plan: &ModelPlan,
    layer: &LayerPlan,
) -> Result<ReferenceWeights, Fail> {
    let index = layer.index;
    let shapes = layer_weight_shapes(plan, layer)?;
    let mut weights = ReferenceWeights::new();
    for (tensor, shape) in shapes.tensors {
        let id = TensorId::Layer { index, tensor };
        weights.insert(id.clone(), loader.load(&id, shape)?);
    }
    if let MlpPlan::Moe(moe) = &layer.mlp {
        for (bank, tensor, output, input) in shapes.banks {
            weights.insert(
                TensorId::Layer {
                    index,
                    tensor: bank,
                },
                loader.expert_bank(index, tensor, moe.expert_count as usize, output, input)?,
            );
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn runner_requests_every_pinned_source_trunk_tensor_at_its_contract_shape() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let pack = model_packs::by_alias("mimo_v2_source").unwrap();
        let plan = pack.compile_plan(&config).unwrap();
        assert!(plan.drafter.is_none());
        assert!(plan.speech.is_none());
        let contract = pack
            .compile_tensor_contract(
                &config,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let by_id: BTreeMap<_, _> = contract
            .requirements
            .iter()
            .map(|requirement| (&requirement.id, requirement))
            .collect();
        let mut requested = BTreeSet::new();
        let mut experts = 0;
        for layer in &plan.layers {
            let shapes = layer_weight_shapes(&plan, layer).unwrap();
            for (tensor, shape) in shapes.tensors {
                let id = TensorId::Layer {
                    index: layer.index,
                    tensor,
                };
                let requirement = by_id[&id];
                assert_eq!(
                    requirement.shape,
                    shape.iter().map(|&dim| dim as u64).collect::<Vec<_>>(),
                    "{id:?}"
                );
                assert!(requirement.required);
                assert_eq!(requirement.transform, TensorTransform::Identity, "{id:?}");
                assert!(requested.insert(id));
            }
            if let MlpPlan::Moe(moe) = &layer.mlp {
                for (_, kind, output, input) in shapes.banks {
                    for expert in 0..moe.expert_count {
                        let id = TensorId::Expert {
                            layer: layer.index,
                            expert,
                            tensor: kind,
                        };
                        assert_eq!(
                            by_id[&id].shape,
                            vec![output as u64, input as u64],
                            "{id:?}"
                        );
                        assert_eq!(by_id[&id].transform, TensorTransform::Identity, "{id:?}");
                        experts += 1;
                    }
                }
            }
        }
        let required_layers: BTreeSet<_> = contract
            .requirements
            .iter()
            .filter(|requirement| {
                requirement.required && matches!(&requirement.id, TensorId::Layer { .. })
            })
            .map(|requirement| requirement.id.clone())
            .collect();
        assert_eq!(requested, required_layers);
        assert_eq!(experts, 36_096);
    }
}
