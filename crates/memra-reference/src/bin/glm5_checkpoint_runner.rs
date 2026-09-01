//! Native checkpoint-parity runner for glm5_next (GLM-5.3-Flash).
//!
//! Protocol (fixed by `run_native_checkpoint` in memra-cli): argv is
//! `<checkpoint_dir> <token_id>...`, `MEMRA_ORACLE_OUT` names the output TSV, and the
//! output is the `memra-checkpoint-oracle-v1` format that `parse_checkpoint_oracle`
//! consumes — the full last-position logits row as `logit\t<index>\t<f32 bits hex>`.
//!
//! The 328 GB FP8 checkpoint cannot be resident as f32, so this driver streams weight
//! residency layer by layer through [`StreamedTrunkExecution`]: only the current trunk
//! layer's tensors (plus the token embedding, output norm, and LM head) are
//! materialized as f32 at any time. Semantic-to-physical mapping comes from the
//! glm5_next pack's tensor contract (the same census that binds this artifact);
//! FP8 e4m3 block-scale dequant and the `model.language_model.` wrapper-prefix
//! fallback ride `SafetensorsSource::dequant_f32_hf`.
//!
//! Deliberate scope: text-only trunk + final norm + LM head. Vision tensors are never
//! requested and the MTP (NextN) block is not executed. `--self-test` runs the same
//! driver over the pack tiny plan's deterministic fixture and asserts bit identity
//! against the reference `execute()`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use memra_gguf::config::{Arch, HfConfig, ModelConfig};
use memra_gguf::model_packs;
use memra_gguf::model_plan::{
    AttentionPlan, LayerPlan, MlaAttentionPlan, MlpPlan, ModelPlan, ResidualTopology, RouterPlan,
    SparseIndexPlan,
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
        eprintln!("glm5_checkpoint_runner: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--self-test") {
        return self_test();
    }
    let (dir, token_args) = args.split_first().ok_or(
        "usage: glm5_checkpoint_runner <checkpoint_dir> <token_id>... \
         (MEMRA_ORACLE_OUT=<tsv path>) | glm5_checkpoint_runner --self-test",
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

/// Pins the streaming driver against the reference executor: build the pack tiny plan
/// and its deterministic fixture, run BOTH `execute()` and the streamed path (globals
/// plus one per-layer weight map at a time), and require bit-for-bit identity.
fn self_test() -> Result<(), Fail> {
    self_test_split_mla_kv()?;
    let pack = model_packs::by_alias("glm5_next").ok_or("glm5_next pack is not registered")?;
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

    if expected != actual {
        return Err("streamed trunk output differs from execute() (struct compare)".into());
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
        "self-test PASS: SplitMlaKv convention pinned; streamed trunk matches \
         execute() bit-for-bit ({} layers, {} tokens, {} logits)",
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
    if config.arch != Arch::Glm5Next {
        return Err(format!("checkpoint arch {:?} is not glm5_next", config.arch).into());
    }
    let pack = model_packs::for_config(&config).ok_or("no model pack matches this config")?;
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
        let weights = glm5_layer_weights(&loader, &plan, layer)?;
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
    // Append (never replace) the extension so distinct outputs cannot share a tmp path.
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

/// Shared driver for both modes: begin with globals, step every trunk layer with a
/// weight map that exists only for that step, then finish with globals.
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

/// Resolves semantic tensor ids to checkpoint bytes through the pack contract:
/// requirement names (tried in contract order), `SafetensorsSource` dequant (FP8
/// block scales, BF16, wrapper-prefix fallback), and the contract transform.
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

    fn fetch_one_of(&self, requirement: &TensorRequirement) -> Result<Vec<f32>, Fail> {
        for name in &requirement.names {
            if let Some((data, _)) = self.source.dequant_f32_hf(name) {
                return Ok(data);
            }
        }
        Err(format!(
            "{:?}: none of {:?} resolved in the checkpoint",
            requirement.id, requirement.names
        )
        .into())
    }

    /// Load one semantic tensor at the reference executor's shape. `OneOf`
    /// requirements take the first alias present; `All` requirements concatenate
    /// every member in contract order (the expert-bank stacking), preallocated
    /// exactly so a 9.7 GB bank never transiently doubles.
    fn load(&self, id: &TensorId, shape: Vec<usize>) -> Result<ReferenceTensor, Fail> {
        let requirement = self.requirement(id)?;
        let expected: usize = shape.iter().product();
        let mut data = match requirement.match_mode {
            TensorMatch::OneOf => self.fetch_one_of(requirement)?,
            TensorMatch::All => {
                let mut bank = Vec::with_capacity(expected);
                for name in &requirement.names {
                    let (values, _) = self
                        .source
                        .dequant_f32_hf(name)
                        .ok_or_else(|| format!("{id:?}: {name} missing from the checkpoint"))?;
                    bank.extend_from_slice(&values);
                }
                bank
            }
        };
        match requirement.transform {
            TensorTransform::Identity | TensorTransform::StackExperts => {}
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

    /// The fused `kv_b_proj` split (contract transform `SplitMlaKv`). The HF lineage
    /// (`GlmMoeDsaAttention::expand_kv`, inherited by glm5_next) views the projection
    /// output as `[heads, nope + v]` and splits `[nope, v]` on the last dim, so the
    /// weight rows are per-head contiguous blocks: `nope` key rows first, then `v`
    /// value rows. Produces the reference `MlaKeyUp [heads, nope, kv_rank]` and
    /// `MlaValueUp [heads, v, kv_rank]` planes.
    fn load_mla_kv_split(
        &self,
        layer: u32,
        heads: usize,
        nope_dim: usize,
        value_dim: usize,
        kv_rank: usize,
    ) -> Result<(ReferenceTensor, ReferenceTensor), Fail> {
        let id = TensorId::Layer {
            index: layer,
            tensor: LayerTensor::MlaKvSource,
        };
        let requirement = self.requirement(&id)?;
        if requirement.transform != TensorTransform::SplitMlaKv {
            return Err(format!(
                "{id:?}: expected the SplitMlaKv transform, contract says {:?}",
                requirement.transform
            )
            .into());
        }
        let fused = self.fetch_one_of(requirement)?;
        let per_head = (nope_dim + value_dim) * kv_rank;
        if fused.len() != heads * per_head {
            return Err(format!(
                "{id:?}: expected {} elements ({heads} heads x {per_head}), \
                 checkpoint provided {}",
                heads * per_head,
                fused.len()
            )
            .into());
        }
        let (key, value) = split_mla_kv(&fused, heads, nope_dim, value_dim, kv_rank);
        // `TensorId::MlaKeyUp` is `[head][kv_rank][nope]` repo-wide (the tensor contract's GGUF
        // `ne = [nope, kv_rank, heads]`); the checkpoint stores `[head][nope][kv_rank]`. The
        // transpose lives here, not in `split_mla_kv`, so that function keeps pinning the ROW
        // convention `self_test_split_mla_kv` exists to pin.
        let mut key_t = vec![0.0f32; key.len()];
        for head in 0..heads {
            for out in 0..nope_dim {
                for rank in 0..kv_rank {
                    key_t[(head * kv_rank + rank) * nope_dim + out] =
                        key[(head * nope_dim + out) * kv_rank + rank];
                }
            }
        }
        Ok((
            ReferenceTensor::new(vec![heads, kv_rank, nope_dim], key_t)?,
            ReferenceTensor::new(vec![heads, value_dim, kv_rank], value)?,
        ))
    }
}

/// Row split of the fused `kv_b_proj` weight `[heads * (nope + v), kv_rank]` into the
/// key-up and value-up planes. Per-head contiguous blocks, `nope` key rows first, then
/// `v` value rows — the `expand_kv` convention. `fused.len()` must already be checked.
fn split_mla_kv(
    fused: &[f32],
    heads: usize,
    nope_dim: usize,
    value_dim: usize,
    kv_rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    let per_head = (nope_dim + value_dim) * kv_rank;
    let mut key = Vec::with_capacity(heads * nope_dim * kv_rank);
    let mut value = Vec::with_capacity(heads * value_dim * kv_rank);
    for head in 0..heads {
        let base = head * per_head;
        key.extend_from_slice(&fused[base..base + nope_dim * kv_rank]);
        value.extend_from_slice(&fused[base + nope_dim * kv_rank..base + per_head]);
    }
    (key, value)
}

/// Executed pin of the `expand_kv` row convention: torch views `kv_b_proj(latent)` as
/// `[heads, nope + v]` and splits `[nope, v]` on the last dim, which on the weight
/// means per-head blocks with the key rows first. A wrong convention preserves element
/// counts, so shape checks alone cannot catch it — this assertion can.
fn self_test_split_mla_kv() -> Result<(), Fail> {
    let (heads, nope_dim, value_dim, kv_rank) = (2, 2, 1, 3);
    // Encode each element as head*100 + row-within-head*10 + rank.
    let mut fused = Vec::new();
    for head in 0..heads {
        for row in 0..nope_dim + value_dim {
            for rank in 0..kv_rank {
                fused.push((head * 100 + row * 10 + rank) as f32);
            }
        }
    }
    let (key, value) = split_mla_kv(&fused, heads, nope_dim, value_dim, kv_rank);
    let expected_key: Vec<f32> = vec![
        0.0, 1.0, 2.0, 10.0, 11.0, 12.0, // head 0, nope rows 0..2
        100.0, 101.0, 102.0, 110.0, 111.0, 112.0, // head 1, nope rows 0..2
    ];
    let expected_value: Vec<f32> = vec![
        20.0, 21.0, 22.0, // head 0, value row (after the nope rows)
        120.0, 121.0, 122.0, // head 1, value row
    ];
    if key != expected_key || value != expected_value {
        return Err(format!(
            "SplitMlaKv convention broken: key={key:?} value={value:?} \
             (expected key={expected_key:?} value={expected_value:?})"
        )
        .into());
    }
    Ok(())
}

/// Materialize one trunk layer's tensors at the reference executor's shapes,
/// mirroring `deterministic_fixture`'s plan walk for the glm5_next layer classes
/// (KDA and MLA+kpool attention, hyper-connection residual, dense and sigmoid-MoE
/// MLPs). Physical layouts that differ only by a contiguous reshape (the
/// `[qkv, 1, kernel]` short convs) load directly at the reference shape.
fn glm5_layer_weights(
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

    match layer.residual {
        ResidualTopology::HyperConnections { streams, .. } => {
            let streams = streams as usize;
            let rows = (2 + streams) * streams;
            for (function, base, scale) in [
                (
                    LayerTensor::HyperAttentionFunction,
                    LayerTensor::HyperAttentionBase,
                    LayerTensor::HyperAttentionScale,
                ),
                (
                    LayerTensor::HyperMlpFunction,
                    LayerTensor::HyperMlpBase,
                    LayerTensor::HyperMlpScale,
                ),
            ] {
                put(&mut weights, function, vec![rows, streams * hidden])?;
                put(&mut weights, base, vec![rows])?;
                put(&mut weights, scale, vec![3])?;
            }
        }
        ResidualTopology::Serial => {}
        other => {
            return Err(format!(
                "layer {index}: residual {other:?} is not part of the glm5_next trunk"
            )
            .into());
        }
    }

    put(&mut weights, LayerTensor::PreAttentionNorm, vec![hidden])?;
    put(&mut weights, LayerTensor::PreMlpNorm, vec![hidden])?;

    match &layer.attention {
        AttentionPlan::KimiDeltaNet(kda) => {
            let heads = kda.num_heads as usize;
            let head_dim = kda.head_dim as usize;
            let kernel = kda.conv_kernel as usize;
            let qkv = heads * head_dim;
            for (tensor, shape) in [
                (LayerTensor::KdaQuery, vec![qkv, hidden]),
                (LayerTensor::KdaKey, vec![qkv, hidden]),
                (LayerTensor::KdaValue, vec![qkv, hidden]),
                (LayerTensor::KdaForgetDown, vec![head_dim, hidden]),
                (LayerTensor::KdaForgetUp, vec![qkv, head_dim]),
                (LayerTensor::KdaGateDown, vec![head_dim, hidden]),
                (LayerTensor::KdaGateUp, vec![qkv, head_dim]),
                (LayerTensor::KdaBeta, vec![heads, hidden]),
                (LayerTensor::KdaOutput, vec![hidden, qkv]),
                (LayerTensor::KdaQueryConv, vec![qkv, kernel]),
                (LayerTensor::KdaKeyConv, vec![qkv, kernel]),
                (LayerTensor::KdaValueConv, vec![qkv, kernel]),
                (LayerTensor::KdaALog, vec![heads]),
                (LayerTensor::KdaDtBias, vec![qkv]),
                (LayerTensor::KdaOutputNorm, vec![head_dim]),
            ] {
                put(&mut weights, tensor, shape)?;
            }
        }
        AttentionPlan::Mla(MlaAttentionPlan::LatentKv {
            query_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_head_dim,
            rope_head_dim,
            value_head_dim,
            rope: _,
            sparse_index,
        }) => {
            let heads = *query_heads as usize;
            let q_rank = *q_lora_rank as usize;
            let kv_rank = *kv_lora_rank as usize;
            let qk_dim = *qk_head_dim as usize;
            let rope_dim = *rope_head_dim as usize;
            let nope_dim = qk_dim - rope_dim;
            let value_dim = *value_head_dim as usize;
            for (tensor, shape) in [
                (LayerTensor::MlaQueryDown, vec![q_rank, hidden]),
                (LayerTensor::MlaQueryUp, vec![heads * qk_dim, q_rank]),
                (LayerTensor::MlaKvDown, vec![kv_rank + rope_dim, hidden]),
                (LayerTensor::MlaOutput, vec![hidden, heads * value_dim]),
                (LayerTensor::MlaQueryDownNorm, vec![q_rank]),
                (LayerTensor::MlaKvDownNorm, vec![kv_rank]),
            ] {
                put(&mut weights, tensor, shape)?;
            }
            let (key_up, value_up) =
                loader.load_mla_kv_split(index, heads, nope_dim, value_dim, kv_rank)?;
            weights.insert(
                TensorId::Layer {
                    index,
                    tensor: LayerTensor::MlaKeyUp,
                },
                key_up,
            );
            weights.insert(
                TensorId::Layer {
                    index,
                    tensor: LayerTensor::MlaValueUp,
                },
                value_up,
            );
            match sparse_index {
                SparseIndexPlan::Own {
                    heads: index_heads,
                    head_dim: index_dim,
                    top_k: _,
                    kpool: Some(kpool),
                } => {
                    let index_heads = *index_heads as usize;
                    let index_dim = *index_dim as usize;
                    let pool = kpool.pool as usize;
                    for (tensor, shape) in [
                        (
                            LayerTensor::SparseQuery,
                            vec![index_heads * index_dim, q_rank],
                        ),
                        (LayerTensor::SparseKey, vec![index_dim, hidden]),
                        (LayerTensor::SparseProjection, vec![index_heads, hidden]),
                        (LayerTensor::SparseKeyNorm, vec![index_dim]),
                        (LayerTensor::SparseKeyNormBias, vec![index_dim]),
                        (LayerTensor::SparseCompressorGate, vec![index_dim, hidden]),
                        (LayerTensor::SparseCompressorPosition, vec![pool, index_dim]),
                    ] {
                        put(&mut weights, tensor, shape)?;
                    }
                }
                other => {
                    return Err(format!(
                        "layer {index}: sparse index {other:?} is not the glm5_next \
                         k-pool indexer"
                    )
                    .into());
                }
            }
        }
        other => {
            return Err(format!(
                "layer {index}: attention {other:?} is not part of the glm5_next trunk"
            )
            .into());
        }
    }

    match &layer.mlp {
        MlpPlan::Dense(mlp) => {
            let intermediate = mlp.intermediate_size as usize;
            for (tensor, shape) in [
                (LayerTensor::MlpGate, vec![intermediate, hidden]),
                (LayerTensor::MlpUp, vec![intermediate, hidden]),
                (LayerTensor::MlpDown, vec![hidden, intermediate]),
            ] {
                put(&mut weights, tensor, shape)?;
            }
        }
        MlpPlan::Moe(moe) => {
            if matches!(moe.router, RouterPlan::TokenIdHash { .. }) {
                return Err(format!(
                    "layer {index}: token-id-hash routing is not a glm5_next router"
                )
                .into());
            }
            let experts = moe.expert_count as usize;
            let intermediate = moe.expert_intermediate_size as usize;
            put(&mut weights, LayerTensor::MoeRouter, vec![experts, hidden])?;
            if matches!(
                moe.router,
                RouterPlan::Sigmoid {
                    selection_bias: true,
                    ..
                } | RouterPlan::SqrtSoftplus {
                    selection_bias: true,
                    ..
                }
            ) {
                put(&mut weights, LayerTensor::MoeRouterBias, vec![experts])?;
            }
            for (tensor, shape) in [
                (
                    LayerTensor::MoeExpertGateBank,
                    vec![experts, intermediate, hidden],
                ),
                (
                    LayerTensor::MoeExpertUpBank,
                    vec![experts, intermediate, hidden],
                ),
                (
                    LayerTensor::MoeExpertDownBank,
                    vec![experts, hidden, intermediate],
                ),
            ] {
                put(&mut weights, tensor, shape)?;
            }
            if let Some(shared) = moe.shared.as_ref() {
                let shared_intermediate = shared.intermediate_size as usize;
                for (tensor, shape) in [
                    (
                        LayerTensor::SharedMlpGate,
                        vec![shared_intermediate, hidden],
                    ),
                    (LayerTensor::SharedMlpUp, vec![shared_intermediate, hidden]),
                    (
                        LayerTensor::SharedMlpDown,
                        vec![hidden, shared_intermediate],
                    ),
                ] {
                    put(&mut weights, tensor, shape)?;
                }
                if shared.gated {
                    put(&mut weights, LayerTensor::SharedMlpInputGate, vec![hidden])?;
                }
            }
        }
    }

    Ok(weights)
}
