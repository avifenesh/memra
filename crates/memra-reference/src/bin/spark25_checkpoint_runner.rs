//! Native checkpoint-parity runner for `spark25` (XHToken Spark-X2.5-4B: the dense, gated,
//! sliding-window member of the step35 family).
//!
//! Protocol (fixed by `run_native_checkpoint` in memra-cli): argv is
//! `<checkpoint_dir> <token_id>...`, `MEMRA_ORACLE_OUT` names the output TSV, and the output
//! is the `memra-checkpoint-oracle-v1` format `parse_checkpoint_oracle` consumes: the full
//! last-position logits row as `logit\t<index>\t<f32 bits hex>`.
//!
//! Residency streams layer by layer through [`StreamedTrunkExecution`] exactly as the
//! llama_dense runner does (4B f32 is 16 GB resident; one layer is ~0.5 GB). Semantic-to-
//! physical mapping comes from the pack's tensor contract; the one family-specific step is the
//! fused `q_k_v_proj` (contract transform `SplitQkvRows`), which this runner dequantizes ONCE
//! per layer and slices into the executor's Query / Key / Value planes by the vendor's row
//! bands (`hf_mapping::spark_qkv_bands`). The head-wise gate is loaded as the plan's
//! `AttentionGate` (`[query_heads, hidden]`). There are no QK-norms and no norm `+1` fold.
//!
//! `--self-test` runs the same driver over the pack tiny plan's deterministic fixture and
//! asserts bit identity against the reference `execute()`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use memra_gguf::config::{Arch, AttentionGateKind, HfConfig, ModelConfig};
use memra_gguf::hf_mapping::spark_qkv_bands;
use memra_gguf::model_packs;
use memra_gguf::model_plan::{
    AttentionPlan, LayerPlan, MlpPlan, ModelPlan, ResidualTopology, TensorPresence, ValueProjection,
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

const FAMILY: &str = "spark25";

fn main() {
    if let Err(error) = run() {
        eprintln!("spark25_checkpoint_runner: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--self-test") {
        return self_test();
    }
    let (dir, token_args) = args.split_first().ok_or(
        "usage: spark25_checkpoint_runner <checkpoint_dir> <token_id>... \
         (MEMRA_ORACLE_OUT=<tsv path>) | spark25_checkpoint_runner --self-test",
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

/// Pins the streaming driver against the reference executor on the pack tiny plan: the
/// deterministic fixture run through BOTH `execute()` and the streamed path must agree
/// bit-for-bit. This is also the family's tiny-parity evidence: a sliding layer, a full
/// layer, the separate head gate, and the erf GELU all execute on the reference.
fn self_test() -> Result<(), Fail> {
    let pack = model_packs::by_alias(FAMILY).ok_or("spark25 pack is not registered")?;
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
    if config.arch != Arch::Step35 || !config.step35.as_ref().is_some_and(|s| s.is_spark()) {
        return Err(format!(
            "checkpoint arch {:?} is not the Spark-X2.5 dense step35 program",
            config.arch
        )
        .into());
    }
    let pack = model_packs::for_config(&config).ok_or("no model pack matches this config")?;
    if pack.family != FAMILY {
        return Err(format!(
            "config selects the {} pack, not {FAMILY}; run that family's runner",
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
    let loader = Loader::new(&source, &contract, &config);
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

/// Resolves semantic tensor ids to checkpoint bytes through the pack contract.
struct Loader<'a> {
    source: &'a SafetensorsSource,
    config: &'a ModelConfig,
    requirements: BTreeMap<&'a TensorId, &'a TensorRequirement>,
}

impl<'a> Loader<'a> {
    fn new(
        source: &'a SafetensorsSource,
        contract: &'a TensorContract,
        config: &'a ModelConfig,
    ) -> Self {
        Self {
            source,
            config,
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

    fn raw(&self, requirement: &TensorRequirement) -> Result<Vec<f32>, Fail> {
        if requirement.match_mode != TensorMatch::OneOf {
            return Err(format!(
                "{:?}: match mode {:?} has no meaning for a dense trunk (no expert banks)",
                requirement.id, requirement.match_mode
            )
            .into());
        }
        for name in &requirement.names {
            if let Some((values, _)) = self.source.dequant_f32_hf(name) {
                return Ok(values);
            }
        }
        Err(format!(
            "{:?}: none of {:?} resolved in the checkpoint",
            requirement.id, requirement.names
        )
        .into())
    }

    fn load(&self, id: &TensorId, shape: Vec<usize>) -> Result<ReferenceTensor, Fail> {
        let requirement = self.requirement(id)?;
        let data = self.raw(requirement)?;
        match requirement.transform {
            TensorTransform::Identity => {}
            other => {
                return Err(
                    format!("{id:?}: transform {other:?} is not supported by this runner").into(),
                );
            }
        }
        tensor(id, shape, data)
    }

    /// The fused `q_k_v_proj` (contract `SplitQkvRows`), dequantized once and sliced into the
    /// executor's three planes by the vendor's row bands.
    fn load_qkv(
        &self,
        index: u32,
        hidden: usize,
    ) -> Result<[(TensorId, ReferenceTensor); 3], Fail> {
        let source_id = TensorId::Layer {
            index,
            tensor: LayerTensor::QkvSource,
        };
        let requirement = self.requirement(&source_id)?;
        if requirement.transform != TensorTransform::SplitQkvRows {
            return Err(format!(
                "{source_id:?}: expected the SplitQkvRows transform, contract says {:?}",
                requirement.transform
            )
            .into());
        }
        let data = self.raw(requirement)?;
        let (q, k, v) = spark_qkv_bands(self.config);
        if data.len() != (q + k + v) * hidden {
            return Err(format!(
                "{source_id:?}: expected {} elements ({q}+{k}+{v} rows x {hidden}), checkpoint \
                 provided {}",
                (q + k + v) * hidden,
                data.len()
            )
            .into());
        }
        let slice = |start: usize, rows: usize, tensor: LayerTensor| {
            let id = TensorId::Layer { index, tensor };
            let plane = data[start * hidden..(start + rows) * hidden].to_vec();
            self::tensor(&id, vec![rows, hidden], plane).map(|t| (id, t))
        };
        Ok([
            slice(0, q, LayerTensor::Query)?,
            slice(q, k, LayerTensor::Key)?,
            slice(q + k, v, LayerTensor::Value)?,
        ])
    }
}

fn tensor(id: &TensorId, shape: Vec<usize>, data: Vec<f32>) -> Result<ReferenceTensor, Fail> {
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(format!(
            "{id:?}: expected {expected} elements for reference shape {shape:?}, checkpoint \
             provided {}",
            data.len()
        )
        .into());
    }
    Ok(ReferenceTensor::new(shape, data)?)
}

/// Every tensor one Spark layer owns, at the shapes `deterministic_fixture` builds for the
/// same plan, so the streamed path and the fixture path feed the executor identical geometry.
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
            "layer {index}: residual {:?} is not part of the Spark trunk",
            layer.residual
        )
        .into());
    }
    let attention = match &layer.attention {
        AttentionPlan::Full(attention) => attention,
        AttentionPlan::SlidingWindow { attention, .. } => attention,
        other => {
            return Err(format!(
                "layer {index}: attention {other:?} is not the Spark full/sliding mixer"
            )
            .into());
        }
    };
    if attention.output_gate != AttentionGateKind::SeparateHead {
        return Err(format!(
            "layer {index}: attention gate {:?}; Spark has the separate head-wise gate",
            attention.output_gate
        )
        .into());
    }
    if attention.qk_norm != TensorPresence::Absent {
        return Err(format!(
            "layer {index}: qk_norm {:?}; Spark has no QK-norm tensors",
            attention.qk_norm
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
    let value_dim = attention.value_head_dim as usize;

    put(&mut weights, LayerTensor::PreAttentionNorm, vec![hidden])?;
    for (id, value) in loader.load_qkv(index, hidden)? {
        weights.insert(id, value);
    }
    put(
        &mut weights,
        LayerTensor::AttentionGate,
        vec![query_heads, hidden],
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
    /// `cargo test` is the caller of the self-test, so it cannot bit-rot unrun.
    #[test]
    fn streamed_trunk_self_test_passes() {
        super::self_test().expect("streamed trunk must match execute() bit-for-bit");
    }
}
