//! Native checkpoint-parity runner for `llama_dense` (DictaLM-3.0-24B and any other dense
//! llama/mistral checkpoint).
//!
//! Protocol (fixed by `run_native_checkpoint` in memra-cli): argv is
//! `<checkpoint_dir> <token_id>...`, `MEMRA_ORACLE_OUT` names the output TSV, and the output
//! is the `memra-checkpoint-oracle-v1` format `parse_checkpoint_oracle` consumes — the full
//! last-position logits row as `logit\t<index>\t<f32 bits hex>`.
//!
//! 24B weights cannot be resident as f32 (94 GB), so this driver streams residency layer by
//! layer through [`StreamedTrunkExecution`]: only the current layer's tensors, plus the token
//! embedding, output norm and LM head, are materialized at any time. Semantic-to-physical
//! mapping comes from the pack's tensor contract, and dtype handling (BF16, NVFP4/FP8
//! block-scale dequant) rides `SafetensorsSource::dequant_f32_hf`, so the same runner answers
//! for the BF16 source and for the NVFP4 mint of it.
//!
//! `--self-test` runs the same driver over the pack tiny plan's deterministic fixture and
//! asserts bit identity against the reference `execute()`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use memra_gguf::config::{Arch, AttentionGateKind, HfConfig, ModelConfig};
use memra_gguf::model_packs;
use memra_gguf::model_plan::{
    AttentionPlan, LayerPlan, MlpPlan, ModelPlan, ResidualTopology, ValueProjection,
};
use memra_gguf::source::SafetensorsSource;
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch, TensorRequirement, TensorTransform,
};
use memra_reference::{
    ReferenceOutput, ReferenceTensor, ReferenceWeights, StreamedTrunkExecution,
    deterministic_fixture, execute,
};

type Fail = Box<dyn std::error::Error>;

fn main() {
    if let Err(error) = run() {
        eprintln!("llama_dense_checkpoint_runner: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--self-test") {
        return self_test();
    }
    let (dir, token_args) = args.split_first().ok_or(
        "usage: llama_dense_checkpoint_runner <checkpoint_dir> <token_id>... \
         (MEMRA_ORACLE_OUT=<tsv path>) | llama_dense_checkpoint_runner --self-test",
    )?;
    if token_args.is_empty() {
        return Err("at least one token id is required".into());
    }
    let token_ids: Vec<u32> = token_args
        .iter()
        .map(|arg| {
            arg.parse::<u32>()
                .map_err(|error| format!("token id {arg:?}: {error}"))
        })
        .collect::<Result<_, _>>()?;
    let out = std::env::var_os("MEMRA_ORACLE_OUT")
        .ok_or("MEMRA_ORACLE_OUT must name the output TSV path")?;
    run_checkpoint(Path::new(dir), &token_ids, Path::new(&out))
}

/// Pins the streaming driver against the reference executor: the pack tiny plan and its
/// deterministic fixture, run through BOTH `execute()` and the streamed path, must agree
/// bit-for-bit.
fn self_test() -> Result<(), Fail> {
    let pack = model_packs::by_alias("llama_dense").ok_or("llama_dense pack is not registered")?;
    let plan = pack.compile_tiny_plan()?;
    let fixture = deterministic_fixture(&plan)?;
    let expected = execute(&plan, &fixture.weights, &fixture.token_ids)?;

    let mut globals = ReferenceWeights::new();
    let mut per_layer: BTreeMap<u32, ReferenceWeights> = BTreeMap::new();
    for (id, tensor) in &fixture.weights {
        match id {
            TensorId::Layer { index, .. } => {
                per_layer
                    .entry(*index)
                    .or_default()
                    .insert(id.clone(), tensor.clone());
            }
            _ => {
                globals.insert(id.clone(), tensor.clone());
            }
        }
    }
    let actual = drive_streamed(&plan, &globals, &fixture.token_ids, |layer| {
        Ok(per_layer.remove(&layer.index).unwrap_or_default())
    })?;

    // Compare on what the streamed path claims to produce. `layer_hidden` is the one
    // deliberate difference: `execute()` retains every layer's residual and the streamed path
    // documents that it cannot (retaining them would defeat the memory bound it exists for),
    // so a whole-struct `!=` can never be green here. Assert that difference explicitly
    // instead of comparing through it.
    if !actual.layer_hidden.is_empty() {
        return Err("streamed trunk retained layer_hidden; it is bounded by design".into());
    }
    if expected.layer_hidden.len() != plan.layers.len() {
        return Err("execute() no longer retains one residual per layer".into());
    }
    if (expected.tokens, expected.vocab) != (actual.tokens, actual.vocab)
        || expected.state != actual.state
        || expected.mtp != actual.mtp
        || expected.draft != actual.draft
        || expected.logits.len() != actual.logits.len()
    {
        return Err("streamed trunk output differs from execute() outside layer_hidden".into());
    }
    for (index, (reference, streamed)) in expected.logits.iter().zip(&actual.logits).enumerate() {
        if reference.to_bits() != streamed.to_bits() {
            return Err(format!(
                "logit {index} differs bitwise: execute()={:08x} streamed={:08x}",
                reference.to_bits(),
                streamed.to_bits()
            )
            .into());
        }
    }
    println!(
        "self-test PASS: streamed trunk matches execute() bit-for-bit \
         ({} layers, {} tokens, {} logits)",
        plan.layers.len(),
        expected.tokens,
        expected.logits.len()
    );
    Ok(())
}

fn run_checkpoint(dir: &Path, token_ids: &[u32], out: &Path) -> Result<(), Fail> {
    let started = Instant::now();
    let config_text = std::fs::read_to_string(dir.join("config.json"))?;
    let config = ModelConfig::from_hf(&HfConfig::parse(&config_text));
    if config.arch != Arch::Llama {
        return Err(format!("checkpoint arch {:?} is not the llama program", config.arch).into());
    }
    let pack = model_packs::for_config(&config).ok_or("no model pack matches this config")?;
    if pack.family != "llama_dense" {
        return Err(format!(
            "config selects the {} pack, not llama_dense; run that family's runner",
            pack.family
        )
        .into());
    }
    let plan = pack.compile_plan(&config)?;
    let source = SafetensorsSource::open(dir)?;
    let output_head = if source.raw_hf("lm_head.weight").is_some() {
        OutputHead::Separate
    } else {
        OutputHead::TiedToEmbedding
    };
    let contract = pack.compile_tensor_contract(
        &config,
        &plan,
        CheckpointDialect::HfSafetensors,
        ContractOptions { output_head },
    )?;
    let loader = Loader::new(&source, &contract);
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;

    let mut globals = ReferenceWeights::new();
    for (id, shape) in [
        (TensorId::TokenEmbedding, vec![vocab, hidden]),
        (TensorId::OutputNorm, vec![hidden]),
    ] {
        let tensor = loader.load(&id, shape)?;
        globals.insert(id, tensor);
    }
    if output_head == OutputHead::Separate {
        let tensor = loader.load(&TensorId::OutputProjection, vec![vocab, hidden])?;
        globals.insert(TensorId::OutputProjection, tensor);
    }
    eprintln!(
        "globals loaded (embedding, output norm{}) in {:.1}s",
        if output_head == OutputHead::Separate {
            ", lm_head"
        } else {
            "; tied head"
        },
        started.elapsed().as_secs_f32()
    );

    let output = drive_streamed(&plan, &globals, token_ids, |layer| {
        let load_started = Instant::now();
        let weights = layer_weights(&loader, &plan, layer)?;
        eprintln!(
            "layer {}: {} tensors loaded in {:.1}s",
            layer.index,
            weights.len(),
            load_started.elapsed().as_secs_f32()
        );
        Ok(weights)
    })?;

    let last = (output.tokens - 1) * output.vocab;
    let logits = &output.logits[last..last + output.vocab];
    let mut text = String::with_capacity(output.vocab * 20 + 256);
    text.push_str("format\tmemra-checkpoint-oracle-v1\n");
    text.push_str("engine\tmemra-reference-fp32\n");
    text.push_str("numeric_class\tsource-weights-float32-accumulation\n");
    writeln!(
        text,
        "tokens\t{}",
        token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )?;
    writeln!(text, "vocab\t{}", output.vocab)?;
    for (index, value) in logits.iter().enumerate() {
        writeln!(text, "logit\t{index}\t{:08x}", value.to_bits())?;
    }
    let tmp = out.with_file_name(format!(
        "{}.tmp",
        out.file_name()
            .ok_or("MEMRA_ORACLE_OUT must end in a file name")?
            .to_string_lossy()
    ));
    std::fs::write(&tmp, text.as_bytes())?;
    std::fs::rename(&tmp, out)?;
    eprintln!(
        "wrote {} ({} last-position logits) in {:.1}s total",
        out.display(),
        output.vocab,
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Shared driver for both modes: begin with globals, step every trunk layer with a weight map
/// that exists only for that step, then finish with globals.
fn drive_streamed(
    plan: &ModelPlan,
    globals: &ReferenceWeights,
    token_ids: &[u32],
    mut layer_weights: impl FnMut(&LayerPlan) -> Result<ReferenceWeights, Fail>,
) -> Result<ReferenceOutput, Fail> {
    let mut run = StreamedTrunkExecution::begin(plan, globals, token_ids)?;
    while let Some(layer) = run.next_layer() {
        let weights = layer_weights(layer)?;
        let step_started = Instant::now();
        run.step(&weights)?;
        eprintln!(
            "layer {}: executed in {:.1}s",
            layer.index,
            step_started.elapsed().as_secs_f32()
        );
    }
    Ok(run.finish(globals)?)
}

/// Resolves semantic tensor ids to checkpoint bytes through the pack contract: requirement
/// names in contract order, `SafetensorsSource` dequant, then the contract transform.
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

    fn requirement(&self, id: &TensorId) -> Result<&'a TensorRequirement, Fail> {
        self.requirements
            .get(id)
            .copied()
            .ok_or_else(|| format!("contract has no requirement for {id:?}").into())
    }

    fn load(&self, id: &TensorId, shape: Vec<usize>) -> Result<ReferenceTensor, Fail> {
        let requirement = self.requirement(id)?;
        if requirement.match_mode != TensorMatch::OneOf {
            return Err(format!(
                "{id:?}: match mode {:?} has no meaning for a dense llama stack (no expert banks)",
                requirement.match_mode
            )
            .into());
        }
        let mut data = None;
        for name in &requirement.names {
            if let Some((values, _)) = self.source.dequant_f32_hf(name) {
                data = Some(values);
                break;
            }
        }
        let mut data = data.ok_or_else(|| {
            format!(
                "{:?}: none of {:?} resolved in the checkpoint",
                requirement.id, requirement.names
            )
        })?;
        match requirement.transform {
            TensorTransform::Identity => {}
            TensorTransform::NormAddOne => {
                for value in &mut data {
                    *value += 1.0;
                }
            }
            other => {
                return Err(
                    format!("{id:?}: transform {other:?} is not supported by this runner").into(),
                );
            }
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "{id:?}: expected {expected} elements for reference shape {shape:?}, \
                 checkpoint provided {}",
                data.len()
            )
            .into());
        }
        Ok(ReferenceTensor::new(shape, data)?)
    }
}

/// Every tensor one dense llama layer owns, at the shapes `deterministic_fixture` builds for
/// the same plan — so the streamed path and the fixture path feed the executor identical
/// geometry.
fn layer_weights(
    loader: &Loader,
    plan: &ModelPlan,
    layer: &LayerPlan,
) -> Result<ReferenceWeights, Fail> {
    let hidden = plan.hidden_size as usize;
    let index = layer.index;
    let mut weights = ReferenceWeights::new();
    let put = |weights: &mut ReferenceWeights,
               tensor: LayerTensor,
               shape: Vec<usize>|
     -> Result<(), Fail> {
        let id = TensorId::Layer { index, tensor };
        let value = loader.load(&id, shape)?;
        weights.insert(id, value);
        Ok(())
    };

    if layer.residual != ResidualTopology::Serial {
        return Err(format!(
            "layer {index}: residual {:?} is not part of a dense llama trunk",
            layer.residual
        )
        .into());
    }

    let AttentionPlan::Full(attention) = &layer.attention else {
        return Err(format!(
            "layer {index}: attention {:?} is not the dense llama full-attention mixer",
            layer.attention
        )
        .into());
    };
    if attention.output_gate != AttentionGateKind::None {
        return Err(format!(
            "layer {index}: attention gate {:?} is not part of this family",
            attention.output_gate
        )
        .into());
    }
    if attention.value_projection != ValueProjection::Separate {
        return Err(format!(
            "layer {index}: value projection {:?} is not part of this family",
            attention.value_projection
        )
        .into());
    }
    let query_heads = attention.query_heads as usize;
    let kv_heads = attention.kv_heads as usize;
    let key_dim = attention.key_head_dim as usize;
    let value_dim = attention.value_head_dim as usize;

    put(&mut weights, LayerTensor::PreAttentionNorm, vec![hidden])?;
    put(
        &mut weights,
        LayerTensor::Query,
        vec![query_heads * key_dim, hidden],
    )?;
    put(
        &mut weights,
        LayerTensor::Key,
        vec![kv_heads * key_dim, hidden],
    )?;
    put(
        &mut weights,
        LayerTensor::Value,
        vec![kv_heads * value_dim, hidden],
    )?;
    put(
        &mut weights,
        LayerTensor::AttentionOutput,
        vec![hidden, query_heads * value_dim],
    )?;
    put(&mut weights, LayerTensor::PreMlpNorm, vec![hidden])?;

    let MlpPlan::Dense(mlp) = &layer.mlp else {
        return Err(format!("layer {index}: this runner executes dense MLPs only").into());
    };
    let intermediate = mlp.intermediate_size as usize;
    put(
        &mut weights,
        LayerTensor::MlpGate,
        vec![intermediate, hidden],
    )?;
    put(&mut weights, LayerTensor::MlpUp, vec![intermediate, hidden])?;
    put(
        &mut weights,
        LayerTensor::MlpDown,
        vec![hidden, intermediate],
    )?;

    Ok(weights)
}

#[cfg(test)]
mod tests {
    /// The self-test had no caller, which is how it bit-rotted: `ReferenceOutput` grew
    /// `layer_hidden`, the streamed path deliberately leaves it empty, and the whole-struct
    /// compare went red with nothing running it. `cargo test` is the caller now.
    #[test]
    fn streamed_trunk_self_test_passes() {
        super::self_test().expect("streamed trunk must match execute() bit-for-bit");
    }
}
