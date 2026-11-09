//! DeepSeek-V4.1 reference execution: the program `ModelPlan::deepseek_v41` plans compile to,
//! run as the vendor `inference/model.py` runs it (one prefill chunk at position 0, then one
//! token per decode step), with the numerics of [`memra_gguf::dsv41_forward`].
//!
//! Per trunk block (vendor `Block.forward`, single-pass mHC): the attention input collapses the
//! stream with the pre mix the PREVIOUS block's FFN produced (identity one-hot at block 0), the
//! FFN input with the pre mix this block's attention mixes produced, and the exit collapses with
//! the last FFN's. Engram writes into the stream before its layer's mixes; DSpark taps the
//! stream mean of a target layer's input after Engram.
//!
//! CSA2 (vendor `Attention`): every layer projects its own query and window KV (FP8 per 32 after
//! RoPE) and keeps a ring of the last `window` rows. A Full layer compresses its group's KV
//! (ratio 1: a projection; ratio > 1: a softmax-gated pool of `ratio` tokens with the partial
//! group carried across decode steps), derives index keys from the RoPE-free latent, and runs
//! its indexer; a Reindex layer rescores the Full layer's index keys with its own query; a Reuse
//! layer takes the most recent index source's top-k. The candidate source keeps its best
//! candidate blocks and every later indexer scores only inside them. Attention concatenates the
//! window rows and the compressed rows into one sparse attention call.
//!
//! DSpark (vendor `DSparkBlock`, `forward_spec`): after the prefill, each DSpark block seeds its
//! window ring from the main-model taps; after each decode step it writes the step's tap into
//! the ring and drafts a block from `[input, noise...]` that attends the ring plus the whole
//! block, then the Markov head adds its bias position by position and the confidence head
//! scores each position.

use memra_gguf::dsv4_forward::FreqsCis;
use memra_gguf::dsv41_forward as f;
use memra_gguf::model_packs::deepseek_v41 as pack;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, BlockQuant, Csa2AttentionPlan, Csa2Candidates,
    Csa2CompressedState, Csa2Mode, Csa2Pooling, DrafterPlan, DsparkPlan, EngramPlan, LayerPlan,
    MlpPlan, ModelPlan, MoeMlpPlan, ResidualTopology, RopeFactors, RopePlan, RouterPlan,
    SinglePassHcPlan, StatePlan,
};
use memra_gguf::tensor_contract::{DsparkTensor, ExpertTensor, LayerTensor, TensorId};
use std::fmt::Write as _;

use crate::{
    ReferenceDecodeTrace, ReferenceDraftOutput, ReferenceError, ReferenceFixture,
    ReferenceLayerState, ReferenceOutput, ReferenceState, ReferenceTensor, ReferenceWeights,
    generated_tensor, layer_id, tensor,
};

const FAMILY: &str = "deepseek_v41";

/// The tokenizer-derived compressed token map Engram hashes over (vendor
/// `build_compressed_token_map`), `[vocab]` exact integers. Not a checkpoint tensor.
pub fn engram_token_map_id() -> TensorId {
    TensorId::Family {
        family: FAMILY,
        key: "engram.token_map".to_string(),
    }
}

fn family_id(prefix: &str, suffix: &str) -> TensorId {
    TensorId::Family {
        family: FAMILY,
        key: format!(
            "{}{suffix}",
            prefix
                .strip_prefix("layers.")
                .map(|rest| format!("trunk.{rest}"))
                .unwrap_or_else(|| prefix.to_string())
        ),
    }
}

fn invalid(layer: Option<u32>, reason: &'static str) -> ReferenceError {
    ReferenceError::InvalidPlan { layer, reason }
}

// ============================ plan geometry ============================

fn dspark_of(plan: &ModelPlan) -> Result<&DsparkPlan, ReferenceError> {
    match plan.drafter.as_ref() {
        Some(DrafterPlan::Dspark(dspark)) => Ok(dspark),
        _ => Err(invalid(
            None,
            "a deepseek_v41 plan carries its DSpark drafter",
        )),
    }
}

fn csa2_of(layer: &LayerPlan) -> Result<&Csa2AttentionPlan, ReferenceError> {
    match &layer.attention {
        AttentionPlan::Csa2(csa2) => Ok(csa2),
        _ => Err(invalid(Some(layer.index), "deepseek_v41 layers are CSA2")),
    }
}

fn moe_of(layer: &LayerPlan) -> Result<&MoeMlpPlan, ReferenceError> {
    match &layer.mlp {
        MlpPlan::Moe(moe) => Ok(moe),
        _ => Err(invalid(Some(layer.index), "deepseek_v41 layers are MoE")),
    }
}

/// The shared expert's intermediate width (vendor `MLP(dim, n_shared_experts * moe_inter_dim)`,
/// ungated: its output adds to the routed sum).
fn shared_width(layer: &LayerPlan, moe: &MoeMlpPlan) -> Result<usize, ReferenceError> {
    let shared = moe.shared.as_ref().ok_or(invalid(
        Some(layer.index),
        "deepseek_v41 MoE has a shared expert",
    ))?;
    if shared.gated {
        return Err(invalid(
            Some(layer.index),
            "deepseek_v41 shared experts are ungated; a gated shared expert is not implemented",
        ));
    }
    Ok(shared.intermediate_size as usize)
}

fn hc_of(layer: &LayerPlan) -> Result<&SinglePassHcPlan, ReferenceError> {
    match &layer.residual {
        ResidualTopology::SinglePassHyperConnections(hc) => Ok(hc),
        _ => Err(invalid(
            Some(layer.index),
            "deepseek_v41 layers carry single-pass mHC",
        )),
    }
}

fn prefix_of(plan_trunk: usize, layer: &LayerPlan) -> String {
    let index = layer.index as usize;
    if index < plan_trunk {
        format!("layers.{index}.")
    } else {
        format!("mtp.{}.", index - plan_trunk)
    }
}

// ============================ numerics ============================

/// The plan's checkpoint numerics, checked against what this executor implements before a tensor
/// is read. Storage fields name the one layout the loader decodes (FP8 dense weights in 32x32
/// ue8m0 blocks, e2m1 experts with a ue8m0 scale per 32, FP8 Engram rows with a ue8m0 scale per
/// 32). The activation, cache and table steps are read from the plan and applied as named
/// ([`f::QuantStep`]); each block must divide the row it quantizes. The two program switches
/// must name the program implemented, and each layer's state plan must be the state its
/// attention plan keeps. Anything else refuses by name.
#[derive(Debug, Clone, Copy)]
struct Numerics {
    /// The step on every quantized linear's input.
    linear: f::QuantStep,
}

const WINDOW_KV_QUANT: &str =
    "window_kv_quant names an unimplemented step or a block that does not divide head_dim";
const OUTPUT_QUANT: &str =
    "output_quant names an unimplemented step or a block that does not divide head_dim";
const LATENT_QUANT: &str =
    "latent_quant names an unimplemented step or a block that does not divide head_dim";
const KEY_QUANT: &str =
    "key_quant names an unimplemented step or a block that does not divide the index head_dim";
const QUERY_QUANT: &str =
    "query_quant names an unimplemented step or a block that does not divide the index head_dim";

fn step(
    quant: BlockQuant,
    layer: Option<u32>,
    reason: &'static str,
) -> Result<f::QuantStep, ReferenceError> {
    f::QuantStep::of(quant).map_err(|_| invalid(layer, reason))
}

fn row_step(
    quant: BlockQuant,
    width: u32,
    layer: Option<u32>,
    reason: &'static str,
) -> Result<f::QuantStep, ReferenceError> {
    let step = step(quant, layer, reason)?;
    if !(width as usize).is_multiple_of(step.block()) {
        return Err(invalid(layer, reason));
    }
    Ok(step)
}

fn numerics(plan: &ModelPlan) -> Result<Numerics, ReferenceError> {
    let numerics = plan.deepseek_v41.as_ref().ok_or(invalid(
        None,
        "a deepseek_v41 plan carries its checkpoint numerics",
    ))?;
    if numerics.dense_weight_block != 32 {
        return Err(invalid(
            None,
            "dense_weight_block: the loader decodes FP8 dense weights in 32x32 e8m0 blocks only",
        ));
    }
    if numerics.expert_weight != BlockQuant::FP4_32_UE8M0 {
        return Err(invalid(
            None,
            "expert_weight: the loader decodes e2m1 experts with a ue8m0 scale per 32 only",
        ));
    }
    if !numerics.routed_weight_before_down {
        return Err(invalid(
            None,
            "routed_weight_before_down: only the vendor order (routing weight before w2's \
             activation step) is implemented",
        ));
    }
    if !numerics.output_lora_a_dequantized_bf16 {
        return Err(invalid(
            None,
            "output_lora_a_dequantized_bf16: only wo_a as a bf16 einsum on the dequantized \
             weight is implemented",
        ));
    }
    let linear = step(
        numerics.linear_activation,
        None,
        "linear_activation names a quantize-dequantize this executor does not implement",
    )?;
    if !matches!(linear, f::QuantStep::Fp8Ue8m0(_)) {
        return Err(invalid(
            None,
            "linear_activation: quantized linears take FP8 ue8m0 activations (vendor linear()); \
             another activation GEMM is not implemented",
        ));
    }
    let dspark = dspark_of(plan)?;
    let hidden = plan.hidden_size;
    let mut linear_inputs = vec![hidden, hidden * dspark.target_layer_ids.len() as u32];
    for layer in plan.layers.iter().chain(&dspark.blocks) {
        let i = Some(layer.index);
        let csa2 = csa2_of(layer)?;
        let moe = moe_of(layer)?;
        linear_inputs.extend([
            csa2.q_lora_rank,
            csa2.output_groups * csa2.output_lora_rank,
            moe.expert_intermediate_size,
        ]);
        if let Some(shared) = moe.shared.as_ref() {
            linear_inputs.push(shared.intermediate_size);
        }
        row_step(csa2.window_kv_quant, csa2.head_dim, i, WINDOW_KV_QUANT)?;
        row_step(csa2.output_quant, csa2.head_dim, i, OUTPUT_QUANT)?;
        let indexer = match csa2.mode {
            Csa2Mode::Full {
                compressor,
                indexer,
                ..
            } => {
                row_step(compressor.latent_quant, csa2.head_dim, i, LATENT_QUANT)?;
                row_step(indexer.key_quant, indexer.head_dim, i, KEY_QUANT)?;
                Some(indexer)
            }
            Csa2Mode::Reindex {
                kv_source, indexer, ..
            } => {
                // A Reindex layer scores the keys its kv source wrote; its own key step is the
                // source's or the plan names two programs for one key cache.
                let source = plan
                    .layers
                    .get(kv_source as usize)
                    .ok_or(invalid(i, "a Reindex layer's kv source is a trunk layer"))?;
                let Csa2Mode::Full {
                    indexer: source_indexer,
                    ..
                } = csa2_of(source)?.mode
                else {
                    return Err(invalid(i, "a Reindex layer's kv source is a Full layer"));
                };
                if source_indexer.key_quant != indexer.key_quant {
                    return Err(invalid(
                        i,
                        "key_quant: a Reindex layer reads its kv source's index keys, so its key \
                         step is the source's",
                    ));
                }
                Some(indexer)
            }
            Csa2Mode::Window | Csa2Mode::Reuse { .. } => None,
        };
        if let Some(indexer) = indexer {
            row_step(indexer.query_quant, indexer.head_dim, i, QUERY_QUANT)?;
        }
        check_state(layer, csa2)?;
        if let Some(engram) = hc_of(layer)?.engram {
            if engram.table_quant != BlockQuant::FP8_32_UE8M0 {
                return Err(invalid(
                    i,
                    "table_quant: the loader decodes Engram tables as FP8 rows with a ue8m0 \
                     scale per 32 only",
                ));
            }
            linear_inputs.push(engram.hash_columns * engram.head_dim);
        }
    }
    if linear_inputs
        .iter()
        .any(|&width| !(width as usize).is_multiple_of(linear.block()))
    {
        return Err(invalid(
            None,
            "linear_activation: its block does not divide every quantized linear's input",
        ));
    }
    Ok(Numerics { linear })
}

/// The layer's state plan must be the state its attention plan keeps: the window ring of
/// `window` rows of `head_dim`, and the compressed state of its mode (a Full layer owns its
/// group's KV and index keys and, above ratio 1, carries the partial group across decode steps;
/// Reindex and Reuse layers read their kv source's).
fn check_state(layer: &LayerPlan, csa2: &Csa2AttentionPlan) -> Result<(), ReferenceError> {
    let i = Some(layer.index);
    let StatePlan::Csa2(state) = &layer.state else {
        return Err(invalid(i, "deepseek_v41 layers keep CSA2 state"));
    };
    if state.window != csa2.window || state.head_dim != csa2.head_dim {
        return Err(invalid(
            i,
            "the CSA2 state ring is the attention window of head_dim rows",
        ));
    }
    let expected = match csa2.mode {
        Csa2Mode::Window => Csa2CompressedState::None,
        Csa2Mode::Full { ratio, indexer, .. } => Csa2CompressedState::Own {
            ratio,
            index_head_dim: indexer.head_dim,
            partial_group: ratio > 1,
        },
        Csa2Mode::Reindex {
            ratio, kv_source, ..
        }
        | Csa2Mode::Reuse {
            ratio, kv_source, ..
        } => Csa2CompressedState::Shared {
            ratio,
            source_layer: kv_source,
        },
    };
    if state.compressed != expected {
        return Err(invalid(
            i,
            "partial_group / compressed state: only the vendor state is implemented (a ratio > 1 \
             Full layer carries its partial group, ratio 1 does not; shared layers read their \
             kv source)",
        ));
    }
    Ok(())
}

// ============================ checkpoint census ============================

/// How a checkpoint stores one tensor (vendor `Linear` and parameter dtypes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStorage {
    Bf16,
    F32,
    /// e4m3 with an e8m0 `.scale` per 32x32 block.
    Fp8Block32,
    /// e4m3 with an e8m0 `.scale` per 32 elements of a row (Engram tables).
    Fp8Row32,
    /// e2m1 pairs packed in I8 (low nibble first) with an e8m0 `.scale` per 32 inputs.
    Fp4,
}

/// One tensor the text program reads from a checkpoint, with its logical shape.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointTensor {
    pub name: String,
    pub id: TensorId,
    pub storage: CheckpointStorage,
    pub shape: Vec<usize>,
}

impl CheckpointTensor {
    /// The safetensors entries this tensor occupies: (name, dtype, physical shape).
    pub fn entries(&self) -> Vec<(String, &'static str, Vec<usize>)> {
        let stem = self.name.strip_suffix(".weight").unwrap_or(&self.name);
        let scale = format!("{stem}.scale");
        let (rows, cols) = (self.shape[0], *self.shape.last().unwrap());
        match self.storage {
            CheckpointStorage::Bf16 => vec![(self.name.clone(), "BF16", self.shape.clone())],
            CheckpointStorage::F32 => vec![(self.name.clone(), "F32", self.shape.clone())],
            CheckpointStorage::Fp8Block32 => vec![
                (self.name.clone(), "F8_E4M3", self.shape.clone()),
                (scale, "F8_E8M0", vec![rows.div_ceil(32), cols.div_ceil(32)]),
            ],
            CheckpointStorage::Fp8Row32 => vec![
                (self.name.clone(), "F8_E4M3", self.shape.clone()),
                (scale, "F8_E8M0", vec![rows, cols / 32]),
            ],
            CheckpointStorage::Fp4 => vec![
                (self.name.clone(), "I8", vec![rows, cols / 2]),
                (scale, "F8_E8M0", vec![rows, cols / 32]),
            ],
        }
    }
}

/// Every checkpoint tensor the plan's text program reads, derived from the plan alone (the pack
/// derives the same names from config math; a test pins the two against each other). The
/// vision tower, aligner, image-span embeddings and image-token router bias are not read.
pub fn checkpoint_tensors(plan: &ModelPlan) -> Result<Vec<CheckpointTensor>, ReferenceError> {
    use CheckpointStorage::{Bf16, F32, Fp4, Fp8Block32, Fp8Row32};
    numerics(plan)?;
    let dspark = dspark_of(plan)?;
    let hidden = plan.hidden_size as usize;
    let vocab = plan.vocab_size as usize;
    let trunk = plan.layers.len();
    let mut out = Vec::new();
    let mut put = |name: String, storage: CheckpointStorage, shape: Vec<usize>| {
        let id = pack::semantic_tensor_id(&name, dspark);
        out.push(CheckpointTensor {
            name,
            id,
            storage,
            shape,
        });
    };
    put("embed.weight".into(), Bf16, vec![vocab, hidden]);
    put("head.weight".into(), Bf16, vec![vocab, hidden]);
    put("norm.weight".into(), Bf16, vec![hidden]);
    let last_block = dspark.blocks.last().map(|block| block.index);
    for layer in plan.layers.iter().chain(&dspark.blocks) {
        let p = prefix_of(trunk, layer);
        let csa2 = csa2_of(layer)?;
        let moe = moe_of(layer)?;
        let hc = hc_of(layer)?;
        let streams = hc.streams as usize;
        let heads = csa2.query_heads as usize;
        let head_dim = csa2.head_dim as usize;
        let q_lora = csa2.q_lora_rank as usize;
        let groups = csa2.output_groups as usize;
        let o_lora = csa2.output_lora_rank as usize;
        put(format!("{p}attn_norm.weight"), Bf16, vec![hidden]);
        put(format!("{p}ffn_norm.weight"), Bf16, vec![hidden]);
        let mix = (2 + streams) * streams;
        for sub in ["attn", "ffn"] {
            put(format!("{p}hc_{sub}_fn"), F32, vec![mix, streams * hidden]);
            put(format!("{p}hc_{sub}_base"), F32, vec![mix]);
            put(format!("{p}hc_{sub}_scale"), F32, vec![3]);
        }
        put(format!("{p}attn.attn_sink"), F32, vec![heads]);
        put(
            format!("{p}attn.wq_a.weight"),
            Fp8Block32,
            vec![q_lora, hidden],
        );
        put(format!("{p}attn.q_norm.weight"), Bf16, vec![q_lora]);
        put(
            format!("{p}attn.wq_b.weight"),
            Fp8Block32,
            vec![heads * head_dim, q_lora],
        );
        put(
            format!("{p}attn.wkv.weight"),
            Fp8Block32,
            vec![head_dim, hidden],
        );
        put(format!("{p}attn.kv_norm.weight"), Bf16, vec![head_dim]);
        put(
            format!("{p}attn.wo_a.weight"),
            Fp8Block32,
            vec![groups * o_lora, heads * head_dim / groups],
        );
        put(
            format!("{p}attn.wo_b.weight"),
            Fp8Block32,
            vec![hidden, groups * o_lora],
        );
        let indexer = match csa2.mode {
            Csa2Mode::Full {
                ratio,
                compressor,
                indexer,
            } => {
                put(
                    format!("{p}attn.compressor.wkv.weight"),
                    Bf16,
                    vec![head_dim, hidden],
                );
                if ratio > 1 {
                    if compressor.pooling != Csa2Pooling::SoftmaxGated {
                        return Err(invalid(
                            Some(layer.index),
                            "a ratio > 1 compressor pools with a softmax gate",
                        ));
                    }
                    put(
                        format!("{p}attn.compressor.wgate.weight"),
                        Bf16,
                        vec![head_dim, hidden],
                    );
                }
                put(
                    format!("{p}attn.compressor.norm.weight"),
                    Bf16,
                    vec![head_dim],
                );
                put(
                    format!("{p}attn.indexer.wk.weight"),
                    Bf16,
                    vec![indexer.head_dim as usize, head_dim],
                );
                put(
                    format!("{p}attn.indexer.k_norm.weight"),
                    Bf16,
                    vec![indexer.head_dim as usize],
                );
                Some(indexer)
            }
            Csa2Mode::Reindex { indexer, .. } => Some(indexer),
            Csa2Mode::Window | Csa2Mode::Reuse { .. } => None,
        };
        if let Some(indexer) = indexer {
            put(
                format!("{p}attn.indexer.wq_b.weight"),
                Fp8Block32,
                vec![(indexer.heads * indexer.head_dim) as usize, q_lora],
            );
            put(
                format!("{p}attn.indexer.weights_proj.weight"),
                Bf16,
                vec![indexer.heads as usize, hidden],
            );
        }
        let experts = moe.expert_count as usize;
        let inter = moe.expert_intermediate_size as usize;
        put(format!("{p}ffn.gate.weight"), Bf16, vec![experts, hidden]);
        put(format!("{p}ffn.gate.bias"), F32, vec![experts]);
        for expert in 0..experts {
            let e = format!("{p}ffn.experts.{expert}.");
            put(format!("{e}w1.weight"), Fp4, vec![inter, hidden]);
            put(format!("{e}w2.weight"), Fp4, vec![hidden, inter]);
            put(format!("{e}w3.weight"), Fp4, vec![inter, hidden]);
        }
        let shared = shared_width(layer, moe)?;
        put(
            format!("{p}ffn.shared_experts.w1.weight"),
            Fp8Block32,
            vec![shared, hidden],
        );
        put(
            format!("{p}ffn.shared_experts.w2.weight"),
            Fp8Block32,
            vec![hidden, shared],
        );
        put(
            format!("{p}ffn.shared_experts.w3.weight"),
            Fp8Block32,
            vec![shared, hidden],
        );
        if let Some(engram) = hc.engram {
            let rows = usize::try_from(engram.table_rows)
                .map_err(|_| invalid(Some(layer.index), "engram table rows overflow usize"))?;
            let dim = engram.head_dim as usize;
            put(format!("{p}engram.embed.weight"), Fp8Row32, vec![rows, dim]);
            put(
                format!("{p}engram.wkv.weight"),
                Fp8Block32,
                vec![hidden * (streams + 1), engram.hash_columns as usize * dim],
            );
            put(format!("{p}engram.q_weight"), Bf16, vec![streams, hidden]);
            put(format!("{p}engram.k_weight"), Bf16, vec![streams, hidden]);
        }
        if layer.index as usize == trunk {
            let targets = dspark.target_layer_ids.len();
            put(
                format!("{p}main_proj.weight"),
                Fp8Block32,
                vec![hidden, targets * hidden],
            );
            put(format!("{p}main_norm.weight"), Bf16, vec![hidden]);
        }
        if Some(layer.index) == last_block {
            let rank = dspark.markov_rank as usize;
            put(format!("{p}norm.weight"), Bf16, vec![hidden]);
            put(
                format!("{p}markov_head.embed.weight"),
                Bf16,
                vec![vocab, rank],
            );
            put(
                format!("{p}markov_head.head.weight"),
                Bf16,
                vec![vocab, rank],
            );
            put(
                format!("{p}confidence_head.proj.weight"),
                Bf16,
                vec![1, hidden + rank],
            );
        }
    }
    Ok(out)
}

/// Dequantize one checkpoint tensor from its stored bytes (`scale` is the `.scale` plane for
/// FP8 and FP4 storage). Every stored value is exact in f32.
pub fn dequantize(
    tensor: &CheckpointTensor,
    primary: &[u8],
    scale: Option<&[u8]>,
) -> Result<ReferenceTensor, ReferenceError> {
    let elements: usize = tensor.shape.iter().product();
    let (rows, cols) = (tensor.shape[0], *tensor.shape.last().unwrap());
    let shape_error = || ReferenceError::TensorShape {
        id: Some(tensor.id.clone()),
        expected: tensor.shape.clone(),
        actual_elements: primary.len(),
    };
    let data = match tensor.storage {
        CheckpointStorage::Bf16 => {
            if primary.len() != elements * 2 {
                return Err(shape_error());
            }
            primary
                .chunks_exact(2)
                .map(|pair| f::bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
                .collect()
        }
        CheckpointStorage::F32 => {
            if primary.len() != elements * 4 {
                return Err(shape_error());
            }
            primary
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect()
        }
        CheckpointStorage::Fp8Block32 | CheckpointStorage::Fp8Row32 => {
            let scale = scale.ok_or_else(|| ReferenceError::MissingTensor(tensor.id.clone()))?;
            let block_rows = if tensor.storage == CheckpointStorage::Fp8Block32 {
                32
            } else {
                1
            };
            if primary.len() != elements
                || scale.len() != rows.div_ceil(block_rows) * cols.div_ceil(32)
            {
                return Err(shape_error());
            }
            f::dequant_fp8(primary, scale, rows, cols, block_rows)
        }
        CheckpointStorage::Fp4 => {
            let scale = scale.ok_or_else(|| ReferenceError::MissingTensor(tensor.id.clone()))?;
            if primary.len() * 2 != elements || scale.len() * 32 != elements {
                return Err(shape_error());
            }
            f::dequant_fp4(primary, scale, rows, cols)
        }
    };
    // A NaN e8m0 scale code (0xff) or a non-finite stored float is refused, never executed.
    if data.iter().any(|v: &f32| !v.is_finite()) {
        return Err(invalid(
            None,
            "checkpoint tensor holds or dequantizes to a non-finite value",
        ));
    }
    ReferenceTensor::new(tensor.shape.clone(), data)
}

// ============================ deterministic fixture ============================

fn fnv1a(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Seeded values for one fixture tensor: norms and Engram gate weights around 1, mHC function
/// rows small, scales around 1, the rest uniform with unit variance per input.
fn fixture_values(tensor: &CheckpointTensor) -> Result<Vec<f32>, ReferenceError> {
    let name = tensor.name.as_str();
    let fan_in = *tensor.shape.last().unwrap() as f32;
    let (scale, offset) = if name.ends_with("norm.weight")
        || name.ends_with("engram.q_weight")
        || name.ends_with("engram.k_weight")
    {
        (0.25, 1.0)
    } else if name.ends_with("_fn") {
        (0.1, 0.0)
    } else if name.ends_with("_scale") {
        (0.5, 1.0)
    } else if name.ends_with("_base") || name.ends_with("attn_sink") {
        (0.5, 0.0)
    } else if name.ends_with("ffn.gate.bias") {
        (0.1, 0.0)
    } else if name == "embed.weight"
        || name.ends_with("engram.embed.weight")
        || name.ends_with("markov_head.embed.weight")
    {
        (1.0, 0.0)
    } else {
        ((3.0 / fan_in).sqrt(), 0.0)
    };
    Ok(generated_tensor(&tensor.shape, fnv1a(name), scale)?
        .data
        .into_iter()
        .map(|value| value + offset)
        .collect())
}

/// Encode seeded values the way the checkpoint stores them: (primary bytes, scale bytes).
fn encode(tensor: &CheckpointTensor, values: &[f32]) -> (Vec<u8>, Option<Vec<u8>>) {
    let (rows, cols) = (tensor.shape[0], *tensor.shape.last().unwrap());
    match tensor.storage {
        CheckpointStorage::Bf16 => (
            values
                .iter()
                .flat_map(|&value| f::f32_to_bf16_bits(value).to_le_bytes())
                .collect(),
            None,
        ),
        CheckpointStorage::F32 => (
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            None,
        ),
        CheckpointStorage::Fp8Block32 => {
            let (codes, scales) = f::quantize_fp8(values, rows, cols, 32);
            (codes, Some(scales))
        }
        CheckpointStorage::Fp8Row32 => {
            let (codes, scales) = f::quantize_fp8(values, rows, cols, 1);
            (codes, Some(scales))
        }
        CheckpointStorage::Fp4 => {
            let (packed, scales) = f::quantize_fp4(values, rows, cols);
            (packed, Some(scales))
        }
    }
}

/// A minimal safetensors writer: little-endian header length, a JSON header, the data.
fn safetensors_bytes(entries: &[(String, &'static str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (index, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        write!(
            header,
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            shape.join(","),
            offset + bytes.len()
        )
        .unwrap();
        offset += bytes.len();
    }
    header.push('}');
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, bytes) in entries {
        out.extend_from_slice(bytes);
    }
    out
}

/// The tiny fixture's token stream: a cycle over the vocabulary with a repeated span (so later
/// positions see earlier n-grams again) and the Engram pad id as a real token.
fn fixture_tokens(vocab: usize, count: usize, pad: u32) -> Vec<u32> {
    let mut tokens: Vec<u32> = (0..count)
        .map(|i| ((i * 37 + (i / 5) * 11 + 3) % vocab) as u32)
        .collect();
    let span = 12.min(count / 4);
    for i in 0..span {
        tokens[count / 2 + i] = tokens[count / 4 + i];
    }
    if count > 5 {
        tokens[5] = pad;
    }
    tokens
}

fn json_list<T: std::fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let items: Vec<String> = values.into_iter().map(|value| value.to_string()).collect();
    format!("[{}]", items.join(", "))
}

/// The vendor `ModelArgs` the plan corresponds to, as the JSON the vendor reads
/// (`inference/config.json` field names), so an external oracle runs exactly this program.
pub fn vendor_model_args(plan: &ModelPlan) -> Result<String, ReferenceError> {
    numerics(plan)?;
    let (dtype, expert_dtype) = vendor_dtypes(plan)?;
    let n_shared_experts = vendor_shared_experts(plan)?;
    if plan.vision.is_some() {
        return Err(invalid(
            None,
            "the vendor arguments describe the text program; the plan carries a vision tower",
        ));
    }
    let dspark = dspark_of(plan)?;
    let trunk = &plan.layers;
    let first = trunk.first().ok_or(ReferenceError::EmptyInput)?;
    let csa2 = csa2_of(first)?;
    let hc = hc_of(first)?;
    let moe = moe_of(first)?;
    let (route_scale, norm_topk_prob) = match moe.router {
        RouterPlan::SqrtSoftplus {
            scaling_factor,
            normalize_selected,
            ..
        } => (scaling_factor, normalize_selected),
        _ => return Err(invalid(Some(0), "deepseek_v41 routes with sqrtsoftplus")),
    };
    let swiglu_limit = match moe.activation {
        ActivationPlan::SwiGluPreClamped { limit } => limit,
        _ => return Err(invalid(Some(0), "deepseek_v41 experts are clamped SwiGLU")),
    };
    let mut ratios = Vec::new();
    let mut kv_sources = Vec::new();
    let mut index_sources = Vec::new();
    let mut candidate = None;
    let mut indexer_plan = None;
    let mut window_rope = None;
    let mut compressed_rope = None;
    for layer in trunk.iter().chain(&dspark.blocks) {
        let plan = csa2_of(layer)?;
        ratios.push(plan.mode.ratio());
        match plan.mode {
            Csa2Mode::Window => window_rope = Some(plan.rope),
            Csa2Mode::Full { indexer, .. } => {
                kv_sources.push(layer.index);
                index_sources.push(layer.index);
                indexer_plan = Some(indexer);
                compressed_rope = Some(plan.rope);
                if let Csa2Candidates::Produce {
                    top_blocks,
                    block_size,
                } = indexer.candidates
                {
                    candidate = Some((layer.index, top_blocks, block_size));
                }
            }
            Csa2Mode::Reindex { .. } => index_sources.push(layer.index),
            Csa2Mode::Reuse { .. } => {}
        }
    }
    let indexer = indexer_plan.ok_or(invalid(None, "deepseek_v41 has a Full CSA2 layer"))?;
    let (candidate_layer, top_blocks, block_size) =
        candidate.ok_or(invalid(None, "deepseek_v41 has a candidate source"))?;
    let window_rope = window_rope.ok_or(invalid(None, "deepseek_v41 has window-only layers"))?;
    let compressed_rope = compressed_rope.ok_or(invalid(None, "deepseek_v41 compresses"))?;
    let RopeFactors::Yarn {
        factor,
        original_context,
        beta_fast,
        beta_slow,
    } = compressed_rope.factors
    else {
        return Err(invalid(None, "compressed CSA2 layers rotate with YaRN"));
    };
    let mut engram: Vec<(u32, EngramPlan)> = trunk
        .iter()
        .filter_map(|layer| hc_of(layer).ok()?.engram.map(|plan| (layer.index, plan)))
        .collect();
    engram.sort_by_key(|(_, plan)| plan.table_index);
    let engram_first = engram
        .first()
        .map(|(_, plan)| *plan)
        .ok_or(invalid(None, "deepseek_v41 has Engram layers"))?;
    let dspark_moe = moe_of(
        dspark
            .blocks
            .first()
            .ok_or(invalid(None, "DSpark has blocks"))?,
    )?;
    let mut json = String::from("{\n");
    let mut field = |key: &str, value: String| {
        writeln!(json, "  \"{key}\": {value},").unwrap();
    };
    field("vocab_size", plan.vocab_size.to_string());
    field("dim", plan.hidden_size.to_string());
    field("moe_inter_dim", moe.expert_intermediate_size.to_string());
    field("n_layers", trunk.len().to_string());
    field("n_mtp_layers", dspark.blocks.len().to_string());
    field("n_heads", csa2.query_heads.to_string());
    field("n_routed_experts", moe.expert_count.to_string());
    field("n_shared_experts", n_shared_experts.to_string());
    field("n_activated_experts", moe.experts_per_token.to_string());
    field("score_func", "\"sqrtsoftplus\"".into());
    field("norm_topk_prob", norm_topk_prob.to_string());
    field("route_scale", route_scale.to_string());
    field("swiglu_limit", swiglu_limit.to_string());
    field("q_lora_rank", csa2.q_lora_rank.to_string());
    field("head_dim", csa2.head_dim.to_string());
    field("rope_head_dim", csa2.rope_head_dim.to_string());
    field("norm_eps", format!("{:e}", plan.output_norm.epsilon));
    field("o_groups", csa2.output_groups.to_string());
    field("o_lora_rank", csa2.output_lora_rank.to_string());
    field("window_size", csa2.window.to_string());
    field("compress_ratios", json_list(ratios));
    field("kv_source_layers", json_list(kv_sources));
    field("index_source_layers", json_list(index_sources));
    field("compress_rope_theta", compressed_rope.base.to_string());
    field("original_seq_len", original_context.to_string());
    field("rope_theta", window_rope.base.to_string());
    field("rope_factor", factor.to_string());
    field("beta_fast", beta_fast.to_string());
    field("beta_slow", beta_slow.to_string());
    field("index_n_heads", indexer.heads.to_string());
    field("index_head_dim", indexer.head_dim.to_string());
    field("index_topk", indexer.top_k.to_string());
    field("candidate_source_layer", candidate_layer.to_string());
    field("candidate_topk_blocks", top_blocks.to_string());
    field("candidate_block_size", block_size.to_string());
    field("hc_mult", hc.streams.to_string());
    field("hc_sinkhorn_iters", hc.sinkhorn_iterations.to_string());
    field("hc_eps", format!("{:e}", hc.epsilon));
    field(
        "engram_layer_ids",
        json_list(engram.iter().map(|(layer, _)| *layer)),
    );
    field(
        "engram_num_embeddings",
        json_list(engram.iter().map(|(_, plan)| plan.table_rows)),
    );
    field("engram_max_ngram_size", engram_first.max_ngram.to_string());
    field("engram_vocab_size", engram_first.bucket_base.to_string());
    field("engram_n_heads", engram_first.heads.to_string());
    field("engram_head_dim", engram_first.head_dim.to_string());
    field("engram_pad_id", engram_first.pad_token_id.to_string());
    field(
        "engram_compressed_vocab_size",
        engram_first.compressed_vocab_size.to_string(),
    );
    field("dtype", format!("\"{dtype}\""));
    field("expert_dtype", format!("\"{expert_dtype}\""));
    field("vision_n_layers", "0".into());
    field("dspark_block_size", dspark.block_size.to_string());
    field("dspark_noise_token_id", dspark.noise_token_id.to_string());
    field(
        "dspark_target_layer_ids",
        json_list(dspark.target_layer_ids.iter()),
    );
    field("dspark_markov_rank", dspark.markov_rank.to_string());
    field(
        "dspark_n_routed_experts",
        dspark_moe.expert_count.to_string(),
    );
    field(
        "dspark_n_activated_experts",
        dspark_moe.experts_per_token.to_string(),
    );
    let mut json = json.trim_end_matches(",\n").to_string();
    json.push_str("\n}\n");
    Ok(json)
}

/// The vendor program's quantize-dequantize steps are module constants of `model.py`
/// (`fp8_block_size = fp4_block_size = 32`, `scale_fmt = "ue8m0"`, the compressed KV at 16 with an
/// e4m3 scale), and its `ModelArgs` selects only the storage dtypes: `dtype` "fp8" (FP8 dense
/// weights in 32x32 blocks, FP8 activations into every quantized linear) or "bf16" (no
/// quantization, which a V4.1 plan cannot express), and `expert_dtype` "fp4". A plan whose
/// numerics differ anywhere describes a program the vendor arguments cannot carry, so it refuses
/// by field name rather than exporting an oracle for another program.
fn vendor_dtypes(plan: &ModelPlan) -> Result<(&'static str, &'static str), ReferenceError> {
    let numerics = plan.deepseek_v41.as_ref().ok_or(invalid(
        None,
        "a deepseek_v41 plan carries its checkpoint numerics",
    ))?;
    if numerics.dense_weight_block != 32 || numerics.linear_activation != BlockQuant::FP8_32_UE8M0 {
        return Err(invalid(
            None,
            "dense_weight_block / linear_activation: ModelArgs dtype \"fp8\" is 32x32 FP8 \
             weights with FP8 ue8m0 activations per 32; the vendor program has no other",
        ));
    }
    if numerics.expert_weight != BlockQuant::FP4_32_UE8M0 {
        return Err(invalid(
            None,
            "expert_weight: ModelArgs expert_dtype \"fp4\" is e2m1 with a ue8m0 scale per 32; \
             the vendor program has no other",
        ));
    }
    let dspark = dspark_of(plan)?;
    for layer in plan.layers.iter().chain(&dspark.blocks) {
        let i = Some(layer.index);
        let csa2 = csa2_of(layer)?;
        if csa2.window_kv_quant != BlockQuant::FP8_32_UE8M0 {
            return Err(invalid(
                i,
                "window_kv_quant: the vendor program fixes it at FP8 per 32 with a ue8m0 scale",
            ));
        }
        if csa2.output_quant != BlockQuant::FP8_32_UE8M0 {
            return Err(invalid(
                i,
                "output_quant: the vendor program fixes it at FP8 per 32 with a ue8m0 scale",
            ));
        }
        let indexer = match csa2.mode {
            Csa2Mode::Full {
                compressor,
                indexer,
                ..
            } => {
                if compressor.latent_quant != BlockQuant::FP4_16_E4M3 {
                    return Err(invalid(
                        i,
                        "latent_quant: the vendor program fixes it at FP4 per 16 with an e4m3 scale",
                    ));
                }
                Some(indexer)
            }
            Csa2Mode::Reindex { indexer, .. } => Some(indexer),
            Csa2Mode::Window | Csa2Mode::Reuse { .. } => None,
        };
        if let Some(indexer) = indexer
            && (indexer.query_quant != BlockQuant::FP4_32_UE8M0
                || indexer.key_quant != BlockQuant::FP4_32_UE8M0)
        {
            return Err(invalid(
                i,
                "query_quant / key_quant: the vendor program fixes them at FP4 per 32 with a \
                 ue8m0 scale",
            ));
        }
    }
    Ok(("fp8", "fp4"))
}

/// `n_shared_experts`: the shared expert's width over `moe_inter_dim`, one value for the trunk
/// and the DSpark blocks (the vendor builds every `MoE` from the same `ModelArgs`).
fn vendor_shared_experts(plan: &ModelPlan) -> Result<u32, ReferenceError> {
    let dspark = dspark_of(plan)?;
    let mut count = None;
    for layer in plan.layers.iter().chain(&dspark.blocks) {
        let moe = moe_of(layer)?;
        let shared = shared_width(layer, moe)?;
        let inter = moe.expert_intermediate_size as usize;
        if inter == 0 || !shared.is_multiple_of(inter) || shared == 0 {
            return Err(invalid(
                Some(layer.index),
                "n_shared_experts: the shared expert is a whole number of moe_inter_dim widths",
            ));
        }
        let n = (shared / inter) as u32;
        if *count.get_or_insert(n) != n {
            return Err(invalid(
                Some(layer.index),
                "n_shared_experts: ModelArgs carries one shared-expert count for every block",
            ));
        }
    }
    count.ok_or(ReferenceError::EmptyInput)
}

/// The tiny-gate fixture: seeded weights in checkpoint storage, dequantized through
/// [`dequantize`]; a token stream whose prefill crosses the sliding window by 9 positions, then
/// 8 teacher-forced decode steps; the compressed token map `t -> t % compressed_vocab`; and the
/// checkpoint image (`model.safetensors`, the vendor `model-args.json`, `fixture.json`) an
/// external oracle loads to run the vendor program on the same bytes.
pub fn fixture(plan: &ModelPlan) -> Result<ReferenceFixture, ReferenceError> {
    let window = csa2_of(plan.layers.first().ok_or(ReferenceError::EmptyInput)?)?.window as usize;
    fixture_split(plan, window + 9)
}

/// [`fixture`]'s weights and token stream (`window + 17` tokens) with the prefill/decode split
/// at `decode_from` instead.
pub fn fixture_split(
    plan: &ModelPlan,
    decode_from: usize,
) -> Result<ReferenceFixture, ReferenceError> {
    let tensors = checkpoint_tensors(plan)?;
    let mut weights = ReferenceWeights::new();
    let mut entries = Vec::new();
    for tensor in &tensors {
        let values = fixture_values(tensor)?;
        let (primary, scale) = encode(tensor, &values);
        weights.insert(
            tensor.id.clone(),
            dequantize(tensor, &primary, scale.as_deref())?,
        );
        let layout = tensor.entries();
        entries.push((
            layout[0].0.clone(),
            layout[0].1,
            layout[0].2.clone(),
            primary,
        ));
        if let (Some(scale), Some(entry)) = (scale, layout.get(1)) {
            entries.push((entry.0.clone(), entry.1, entry.2.clone(), scale));
        }
    }
    let vocab = plan.vocab_size as usize;
    let engram = plan
        .layers
        .iter()
        .find_map(|layer| hc_of(layer).ok()?.engram)
        .ok_or(invalid(None, "deepseek_v41 has Engram layers"))?;
    let compressed = engram.compressed_vocab_size as i64;
    let token_map: Vec<i64> = (0..vocab as i64).map(|token| token % compressed).collect();
    weights.insert(
        engram_token_map_id(),
        ReferenceTensor::new_i64(vec![vocab], token_map.clone())?,
    );
    let window = csa2_of(&plan.layers[0])?.window as usize;
    let tokens = fixture_tokens(vocab, window + 17, engram.pad_token_id);
    if decode_from == 0 || decode_from >= tokens.len() {
        return Err(invalid(
            None,
            "the fixture's prefill is a non-empty proper prefix of its tokens",
        ));
    }
    let prefill = decode_from;
    let fixture_json = format!(
        "{{\n  \"token_ids\": {},\n  \"decode_from\": {prefill},\n  \"engram_token_map\": {}\n}}\n",
        json_list(&tokens),
        json_list(&token_map)
    );
    Ok(ReferenceFixture {
        token_ids: tokens,
        weights,
        vision: None,
        multimodal_token_ids: None,
        decode_from: Some(prefill),
        checkpoint_files: vec![
            ("model.safetensors".into(), safetensors_bytes(&entries)),
            (
                "model-args.json".into(),
                vendor_model_args(plan)?.into_bytes(),
            ),
            ("fixture.json".into(), fixture_json.into_bytes()),
        ],
    })
}

/// The tiny gate's variants beside [`fixture`], each pinned to its own vendor oracle, for the
/// splits the main fixture never runs. Below the window the vendor's prefill and decode programs
/// differ for the same row (`get_window_topk_idxs`: a prefill row is `min(seqlen, window)` wide
/// with its empty slots last, a decode row the ring oldest first with its empty slots first, and
/// `sparse_attn` rounds probabilities to bf16 per 64-entry block), so these pin that memra runs
/// both programs the vendor runs:
///
/// - `split60`: the tiny plan split at 60 of its window of 64: 21 decode steps, the first 4 while
///   the ring still fills, and 20 DSpark drafts, the first with 61 of its 64 ring rows filled;
/// - `window128`: the tiny plan at the 1017 window of 128 (two 64-entry blocks), split at 100:
///   45 decode steps, 28 while the ring fills, then a full two-block ring.
pub fn tiny_variants(plan: &ModelPlan) -> Result<Vec<crate::TinyVariant>, ReferenceError> {
    let wide = pack::tiny_window128_plan()
        .map_err(|_| invalid(None, "the deepseek_v41 window-128 tiny config compiles"))?;
    let fixture = fixture_split(plan, 60)?;
    let wide_fixture = fixture_split(&wide, 100)?;
    Ok(vec![
        crate::TinyVariant {
            name: "split60",
            plan: plan.clone(),
            fixture,
        },
        crate::TinyVariant {
            name: "window128",
            plan: wide,
            fixture: wide_fixture,
        },
    ])
}

// ============================ execution ============================

/// RoPE tables, one per distinct [`RopePlan`] in the plan.
struct RopeTables {
    tables: Vec<(RopePlan, FreqsCis)>,
}

impl RopeTables {
    fn new(
        plan: &ModelPlan,
        dspark: &DsparkPlan,
        positions: usize,
    ) -> Result<Self, ReferenceError> {
        let mut tables: Vec<(RopePlan, FreqsCis)> = Vec::new();
        for layer in plan.layers.iter().chain(&dspark.blocks) {
            let rope = csa2_of(layer)?.rope;
            if tables.iter().any(|(existing, _)| *existing == rope) {
                continue;
            }
            let table = match rope.factors {
                RopeFactors::None => f::precompute_freqs_cis(
                    rope.dimensions as usize,
                    positions,
                    0,
                    rope.base,
                    1.0,
                    0.0,
                    0.0,
                ),
                RopeFactors::Yarn {
                    factor,
                    original_context,
                    beta_fast,
                    beta_slow,
                } => f::precompute_freqs_cis(
                    rope.dimensions as usize,
                    positions,
                    original_context,
                    rope.base,
                    factor,
                    beta_fast,
                    beta_slow,
                ),
                _ => {
                    return Err(invalid(
                        Some(layer.index),
                        "CSA2 rotates without factors or with YaRN",
                    ));
                }
            };
            tables.push((rope, table));
        }
        Ok(Self { tables })
    }

    fn get(&self, rope: &RopePlan) -> &FreqsCis {
        &self
            .tables
            .iter()
            .find(|(existing, _)| existing == rope)
            .expect("every layer's rope table was built")
            .1
    }
}

/// Per-layer CSA2 state: the window ring and, for a Full layer, the group's compressed KV,
/// index keys and the partial-group carry.
#[derive(Clone)]
struct LayerState {
    ring: Vec<f32>,
    compressed: Vec<f32>,
    index_keys: Vec<f32>,
    kv_carry: Vec<f32>,
    score_carry: Vec<f32>,
}

/// What a layer hands down the stack within one step (vendor `SharedAttentionRuntime`).
#[derive(Default)]
struct Shared {
    /// The most recent index source's selection, `[queries, k]`, already offset.
    topk: Option<(Vec<i64>, usize)>,
    /// The candidate source's per-entry mask, `[queries, width]`.
    candidates: Option<(Vec<bool>, usize)>,
}

struct Session<'a> {
    plan: &'a ModelPlan,
    numerics: Numerics,
    weights: &'a ReferenceWeights,
    dspark: &'a DsparkPlan,
    hidden: usize,
    vocab: usize,
    streams: usize,
    trunk: usize,
    ropes: RopeTables,
    layers: Vec<LayerState>,
    dspark_rings: Vec<Vec<f32>>,
    /// Compressed token ids of every position so far (Engram look-back).
    history: Vec<Option<i64>>,
    engram: Vec<EngramTables>,
}

struct EngramTables {
    multipliers: Vec<i64>,
    primes: Vec<Vec<u64>>,
}

/// One forward step's outputs.
struct Step {
    logits: Vec<f32>,
    layer_hidden: Vec<Vec<f32>>,
    /// `[tokens, targets * hidden]`: the DSpark main-model taps.
    taps: Vec<f32>,
}

impl<'a> Session<'a> {
    fn new(
        plan: &'a ModelPlan,
        weights: &'a ReferenceWeights,
        positions: usize,
    ) -> Result<Self, ReferenceError> {
        let numerics = numerics(plan)?;
        let dspark = dspark_of(plan)?;
        let first = plan.layers.first().ok_or(ReferenceError::EmptyInput)?;
        let streams = hc_of(first)?.streams as usize;
        let mut layers = Vec::with_capacity(plan.layers.len());
        let mut engram_plans: Vec<(u32, EngramPlan)> = Vec::new();
        for layer in &plan.layers {
            let csa2 = csa2_of(layer)?;
            let head_dim = csa2.head_dim as usize;
            let ratio = csa2.mode.ratio() as usize;
            layers.push(LayerState {
                ring: vec![0.0; csa2.window as usize * head_dim],
                compressed: Vec::new(),
                index_keys: Vec::new(),
                kv_carry: vec![0.0; ratio * head_dim],
                score_carry: vec![f32::NEG_INFINITY; ratio * head_dim],
            });
            if hc_of(layer)?.streams as usize != streams {
                return Err(invalid(
                    Some(layer.index),
                    "every block has the same streams",
                ));
            }
            if let Some(engram) = hc_of(layer)?.engram {
                engram_plans.push((layer.index, engram));
            }
        }
        engram_plans.sort_by_key(|(_, plan)| plan.table_index);
        let primes = engram_plans.first().map(|(_, plan)| {
            f::engram_primes(
                engram_plans.len(),
                plan.max_ngram as usize,
                plan.heads as usize,
                u64::from(plan.bucket_base),
            )
        });
        let mut engram = Vec::new();
        for (slot, (layer, plan)) in engram_plans.iter().enumerate() {
            if plan.table_index as usize != slot {
                return Err(invalid(
                    Some(*layer),
                    "engram table indices are 0..n in order",
                ));
            }
            let primes = primes.as_ref().unwrap()[slot].clone();
            let rows: u64 = primes.iter().flatten().sum();
            if rows != plan.table_rows {
                return Err(invalid(
                    Some(*layer),
                    "engram table rows differ from the prime bucket sums",
                ));
            }
            engram.push(EngramTables {
                multipliers: f::engram_multipliers(
                    *layer,
                    plan.max_ngram as usize,
                    u64::from(plan.compressed_vocab_size),
                )
                .map_err(|_| invalid(Some(*layer), "engram multiplier draw is not ported"))?,
                primes,
            });
        }
        let dspark_rings = dspark
            .blocks
            .iter()
            .map(|block| {
                let csa2 = csa2_of(block)?;
                if csa2.mode != Csa2Mode::Window {
                    return Err(invalid(Some(block.index), "DSpark blocks are window-only"));
                }
                Ok(vec![0.0; csa2.window as usize * csa2.head_dim as usize])
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            plan,
            numerics,
            weights,
            dspark,
            hidden: plan.hidden_size as usize,
            vocab: plan.vocab_size as usize,
            streams,
            trunk: plan.layers.len(),
            ropes: RopeTables::new(plan, dspark, positions)?,
            layers,
            dspark_rings,
            history: Vec::new(),
            engram,
        })
    }

    fn t(&self, id: &TensorId, shape: &[usize]) -> Result<&'a [f32], ReferenceError> {
        tensor(self.weights, id, shape)
    }

    fn lt(
        &self,
        layer: u32,
        which: LayerTensor,
        shape: &[usize],
    ) -> Result<&'a [f32], ReferenceError> {
        tensor(self.weights, &layer_id(layer, which), shape)
    }

    fn ft(
        &self,
        layer: &LayerPlan,
        suffix: &str,
        shape: &[usize],
    ) -> Result<&'a [f32], ReferenceError> {
        let prefix = prefix_of(self.trunk, layer);
        tensor(self.weights, &family_id(&prefix, suffix), shape)
    }

    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, ReferenceError> {
        let table = self.t(&TensorId::TokenEmbedding, &[self.vocab, self.hidden])?;
        let mut out = Vec::with_capacity(tokens.len() * self.streams * self.hidden);
        for &token in tokens {
            let token = token as usize;
            if token >= self.vocab {
                return Err(ReferenceError::TokenOutOfRange {
                    token: token as u32,
                    vocab: self.vocab,
                });
            }
            let row = &table[token * self.hidden..(token + 1) * self.hidden];
            for _ in 0..self.streams {
                out.extend_from_slice(row);
            }
        }
        Ok(out)
    }

    fn identity_pre(&self, tokens: usize) -> Vec<f32> {
        let mut pre = vec![0.0; tokens * self.streams];
        for token in 0..tokens {
            pre[token * self.streams] = 1.0;
        }
        pre
    }

    /// One trunk step over `tokens` new tokens at `start`.
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Step, ReferenceError> {
        let s = tokens.len();
        let (hidden, streams) = (self.hidden, self.streams);
        let mut h = self.embed(tokens)?;
        // Engram: compressed ids, then each position's hash row per table.
        let hashes = if self.engram.is_empty() {
            Vec::new()
        } else {
            let map = self
                .weights
                .get(&engram_token_map_id())
                .and_then(|tensor| tensor.ints.as_ref())
                .ok_or_else(|| ReferenceError::IntegerTensorRequired(engram_token_map_id()))?;
            // The vendor map has one entry per tokenizer id, which can be fewer than the
            // embedding rows; a token outside it has no compressed id.
            for &token in tokens {
                let compressed = *map
                    .get(token as usize)
                    .ok_or(invalid(None, "token outside the engram token map"))?;
                self.history.push(Some(compressed));
            }
            let first = self
                .plan
                .layers
                .iter()
                .find_map(|layer| hc_of(layer).ok()?.engram)
                .expect("engram tables imply an engram layer");
            let pad = map[first.pad_token_id as usize];
            self.engram
                .iter()
                .map(|tables| {
                    (start..start + s)
                        .map(|position| {
                            f::engram_hash_row(
                                &self.history,
                                position,
                                pad,
                                &tables.multipliers,
                                &tables.primes,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let mut pre = self.identity_pre(s);
        let targets = &self.dspark.target_layer_ids;
        let mut taps = vec![0.0; s * targets.len() * hidden];
        let mut layer_hidden = Vec::with_capacity(self.trunk);
        let mut shared = Shared::default();
        for index in 0..self.trunk {
            let layer = &self.plan.layers[index];
            let hc = *hc_of(layer)?;
            if let Some(engram) = hc.engram {
                h = self.engram_layer(layer, &engram, &h, &hashes[engram.table_index as usize])?;
            }
            if let Some(target) = targets.iter().position(|&t| t == layer.index) {
                for token in 0..s {
                    for i in 0..hidden {
                        let sum: f64 = (0..streams)
                            .map(|stream| f64::from(h[(token * streams + stream) * hidden + i]))
                            .sum();
                        taps[(token * targets.len() + target) * hidden + i] =
                            f::bf16((sum / streams as f64) as f32);
                    }
                }
            }
            let (next, next_pre) = self.block(layer, &h, &pre, s, start, &mut shared, None)?;
            h = next;
            pre = next_pre;
            layer_hidden.push(h.clone());
        }
        let x = f::hc_pre(&h, s, streams, hidden, &pre);
        let x = f::rmsnorm_bf16(
            &x,
            self.t(&TensorId::OutputNorm, &[hidden])?,
            self.plan.output_norm.epsilon,
        );
        let head = self.t(&TensorId::OutputProjection, &[self.vocab, hidden])?;
        let logits = f::linear_f32(&x, s, hidden, head, self.vocab);
        Ok(Step {
            logits,
            layer_hidden,
            taps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        layer: &LayerPlan,
        x: &[f32],
        pre_mix: &[f32],
        s: usize,
        start: usize,
        shared: &mut Shared,
        dspark: Option<(usize, &[f32])>,
    ) -> Result<(Vec<f32>, Vec<f32>), ReferenceError> {
        let (hidden, streams) = (self.hidden, self.streams);
        let hc = *hc_of(layer)?;
        let mix = (2 + streams) * streams;
        let i = layer.index;
        let eps = layer.pre_attention_norm.epsilon;
        let (attn_pre, attn_post, attn_comb) = f::hc_mixes(
            x,
            s,
            streams,
            hidden,
            self.lt(
                i,
                LayerTensor::HyperAttentionFunction,
                &[mix, streams * hidden],
            )?,
            self.lt(i, LayerTensor::HyperAttentionScale, &[3])?,
            self.lt(i, LayerTensor::HyperAttentionBase, &[mix])?,
            hc.sinkhorn_iterations,
            hc.mix_norm_epsilon,
            hc.epsilon,
        );
        let y = f::hc_pre(x, s, streams, hidden, pre_mix);
        let y = f::rmsnorm_bf16(
            &y,
            self.lt(i, LayerTensor::PreAttentionNorm, &[hidden])?,
            eps,
        );
        let y = match dspark {
            None => self.attention(layer, &y, s, start, shared)?,
            Some((block, main_x)) => self.dspark_attention(layer, block, &y, s, start, main_x)?,
        };
        let x = f::hc_post(&y, x, s, streams, hidden, &attn_post, &attn_comb);
        let (ffn_pre, ffn_post, ffn_comb) = f::hc_mixes(
            &x,
            s,
            streams,
            hidden,
            self.lt(i, LayerTensor::HyperMlpFunction, &[mix, streams * hidden])?,
            self.lt(i, LayerTensor::HyperMlpScale, &[3])?,
            self.lt(i, LayerTensor::HyperMlpBase, &[mix])?,
            hc.sinkhorn_iterations,
            hc.mix_norm_epsilon,
            hc.epsilon,
        );
        let y = f::hc_pre(&x, s, streams, hidden, &attn_pre);
        let y = f::rmsnorm_bf16(
            &y,
            self.lt(i, LayerTensor::PreMlpNorm, &[hidden])?,
            layer.pre_mlp_norm.epsilon,
        );
        let y = self.moe(layer, &y, s)?;
        Ok((
            f::hc_post(&y, &x, s, streams, hidden, &ffn_post, &ffn_comb),
            ffn_pre,
        ))
    }

    fn engram_layer(
        &self,
        layer: &LayerPlan,
        engram: &EngramPlan,
        x: &[f32],
        hashes: &[Vec<u64>],
    ) -> Result<Vec<f32>, ReferenceError> {
        let (hidden, streams) = (self.hidden, self.streams);
        let s = hashes.len();
        let dim = engram.head_dim as usize;
        let columns = engram.hash_columns as usize;
        let rows = engram.table_rows as usize;
        let table = self.ft(layer, "engram.embed.weight", &[rows, dim])?;
        let wkv = self.ft(
            layer,
            "engram.wkv.weight",
            &[hidden * (streams + 1), columns * dim],
        )?;
        let q_weight = self.ft(layer, "engram.q_weight", &[streams, hidden])?;
        let k_weight = self.ft(layer, "engram.k_weight", &[streams, hidden])?;
        let mut gathered = Vec::with_capacity(s * columns * dim);
        for row in hashes {
            if row.len() != columns {
                return Err(invalid(Some(layer.index), "engram hash row width"));
            }
            for &bucket in row {
                let bucket = bucket as usize;
                if bucket >= rows {
                    return Err(invalid(
                        Some(layer.index),
                        "engram bucket outside the table",
                    ));
                }
                gathered.extend_from_slice(&table[bucket * dim..(bucket + 1) * dim]);
            }
        }
        let kv = f::linear_quant(
            &gathered,
            s,
            columns * dim,
            wkv,
            hidden * (streams + 1),
            self.numerics.linear,
        );
        let weight: Vec<f32> = q_weight.iter().zip(k_weight).map(|(q, k)| q * k).collect();
        let width = hidden * (streams + 1);
        let mut out = vec![0f32; x.len()];
        for token in 0..s {
            let value = &kv[token * width + streams * hidden..(token + 1) * width];
            for stream in 0..streams {
                let base = (token * streams + stream) * hidden;
                f::engram_mix(
                    &x[base..base + hidden],
                    &kv[token * width + stream * hidden..token * width + (stream + 1) * hidden],
                    &weight[stream * hidden..(stream + 1) * hidden],
                    value,
                    layer.pre_attention_norm.epsilon,
                    engram.gate_clamp,
                    &mut out[base..base + hidden],
                );
            }
        }
        Ok(out)
    }

    /// Query, window KV (FP8 per 32 after RoPE) shared by trunk and DSpark attention.
    fn project_query(
        &self,
        layer: &LayerPlan,
        csa2: &Csa2AttentionPlan,
        x: &[f32],
        s: usize,
        positions: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), ReferenceError> {
        let hidden = self.hidden;
        let (heads, head_dim, rd, q_lora) = (
            csa2.query_heads as usize,
            csa2.head_dim as usize,
            csa2.rope_head_dim as usize,
            csa2.q_lora_rank as usize,
        );
        let i = layer.index;
        let freqs = self.ropes.get(&csa2.rope);
        let qr = f::rmsnorm_bf16(
            &f::linear_quant(
                x,
                s,
                hidden,
                self.lt(i, LayerTensor::MlaQueryDown, &[q_lora, hidden])?,
                q_lora,
                self.numerics.linear,
            ),
            self.lt(i, LayerTensor::MlaQueryDownNorm, &[q_lora])?,
            layer.pre_attention_norm.epsilon,
        );
        let mut q = f::linear_quant(
            &qr,
            s,
            q_lora,
            self.lt(i, LayerTensor::MlaQueryUp, &[heads * head_dim, q_lora])?,
            heads * head_dim,
            self.numerics.linear,
        );
        f::rope_bf16(&mut q, head_dim, rd, heads, positions, freqs, false);
        Ok((qr, q))
    }

    fn window_kv(
        &self,
        layer: &LayerPlan,
        csa2: &Csa2AttentionPlan,
        x: &[f32],
        s: usize,
        positions: &[usize],
    ) -> Result<Vec<f32>, ReferenceError> {
        let head_dim = csa2.head_dim as usize;
        let i = layer.index;
        let mut kv = f::rmsnorm_bf16(
            &f::linear_quant(
                x,
                s,
                self.hidden,
                self.lt(i, LayerTensor::MlaKvDown, &[head_dim, self.hidden])?,
                head_dim,
                self.numerics.linear,
            ),
            self.lt(i, LayerTensor::MlaKvDownNorm, &[head_dim])?,
            layer.pre_attention_norm.epsilon,
        );
        f::rope_bf16(
            &mut kv,
            head_dim,
            csa2.rope_head_dim as usize,
            1,
            positions,
            self.ropes.get(&csa2.rope),
            false,
        );
        step(csa2.window_kv_quant, Some(i), WINDOW_KV_QUANT)?.apply(&mut kv);
        Ok(kv)
    }

    /// Sparse attention over `rows`, inverse RoPE, FP8 per 32, grouped `wo_a`, `wo_b`.
    #[allow(clippy::too_many_arguments)]
    fn attend_and_project(
        &self,
        layer: &LayerPlan,
        csa2: &Csa2AttentionPlan,
        q: &[f32],
        s: usize,
        rows: &[f32],
        indices: &[i64],
        k: usize,
        positions: &[usize],
    ) -> Result<Vec<f32>, ReferenceError> {
        let hidden = self.hidden;
        let (heads, head_dim, groups, o_lora) = (
            csa2.query_heads as usize,
            csa2.head_dim as usize,
            csa2.output_groups as usize,
            csa2.output_lora_rank as usize,
        );
        let i = layer.index;
        let sink = self.lt(i, LayerTensor::AttentionSink, &[heads])?;
        let scale = (head_dim as f64).powf(-0.5) as f32;
        let mut o = f::sparse_attention(q, s, heads, head_dim, rows, sink, indices, k, scale);
        f::rope_bf16(
            &mut o,
            head_dim,
            csa2.rope_head_dim as usize,
            heads,
            positions,
            self.ropes.get(&csa2.rope),
            true,
        );
        step(csa2.output_quant, Some(i), OUTPUT_QUANT)?.apply(&mut o);
        let group_width = heads * head_dim / groups;
        let wo_a = self.lt(
            i,
            LayerTensor::MlaOutputDown,
            &[groups * o_lora, group_width],
        )?;
        // einsum("bsgd,grd->bsgr") in bf16: each group projects only its own heads.
        let mut grouped = vec![0f32; s * groups * o_lora];
        for token in 0..s {
            for group in 0..groups {
                let input = &o[(token * groups + group) * group_width
                    ..(token * groups + group + 1) * group_width];
                for r in 0..o_lora {
                    let row = &wo_a[(group * o_lora + r) * group_width
                        ..(group * o_lora + r + 1) * group_width];
                    grouped[(token * groups + group) * o_lora + r] =
                        f::bf16(memra_gguf::dsv4_forward::dot(input, row));
                }
            }
        }
        Ok(f::linear_quant(
            &grouped,
            s,
            groups * o_lora,
            self.lt(i, LayerTensor::MlaOutput, &[hidden, groups * o_lora])?,
            hidden,
            self.numerics.linear,
        ))
    }

    fn attention(
        &mut self,
        layer: &LayerPlan,
        x: &[f32],
        s: usize,
        start: usize,
        shared: &mut Shared,
    ) -> Result<Vec<f32>, ReferenceError> {
        let csa2 = *csa2_of(layer)?;
        let head_dim = csa2.head_dim as usize;
        let window = csa2.window as usize;
        let index = layer.index as usize;
        let positions: Vec<usize> = (start..start + s).collect();
        let (qr, q) = self.project_query(layer, &csa2, x, s, &positions)?;
        let kv = self.window_kv(layer, &csa2, x, s, &positions)?;
        // Window: prefill attends its own chunk and seeds the ring; decode writes one slot and
        // attends the whole ring, oldest slot first.
        let (mut rows, window_indices, window_k) = if start == 0 {
            for position in s.saturating_sub(window)..s {
                self.layers[index].ring
                    [(position % window) * head_dim..(position % window + 1) * head_dim]
                    .copy_from_slice(&kv[position * head_dim..(position + 1) * head_dim]);
            }
            let width = s.min(window);
            let mut indices = vec![-1i64; s * width];
            for query in 0..s {
                let first = query.saturating_sub(window - 1);
                for j in 0..width {
                    let position = first + j;
                    if position <= query {
                        indices[query * width + j] = position as i64;
                    }
                }
            }
            (kv, indices, width)
        } else {
            if s != 1 {
                return Err(invalid(
                    Some(layer.index),
                    "decode steps one token at a time",
                ));
            }
            let slot = start % window;
            self.layers[index].ring[slot * head_dim..(slot + 1) * head_dim].copy_from_slice(&kv);
            let oldest = slot + 1;
            let indices = (oldest..window)
                .chain(0..oldest)
                .map(|ring_slot| {
                    if ring_slot > start {
                        -1
                    } else {
                        ring_slot as i64
                    }
                })
                .collect();
            (self.layers[index].ring.clone(), indices, window)
        };
        let offset = rows.len() / head_dim;
        let (indices, k) = if csa2.mode == Csa2Mode::Window {
            (window_indices, window_k)
        } else {
            let (compressed, compressed_indices, compressed_k) =
                self.compressed(layer, &csa2, x, &qr, s, start, offset, shared)?;
            rows.extend_from_slice(&compressed);
            let k = window_k + compressed_k;
            let mut indices = Vec::with_capacity(s * k);
            for query in 0..s {
                indices
                    .extend_from_slice(&window_indices[query * window_k..(query + 1) * window_k]);
                indices.extend_from_slice(
                    &compressed_indices[query * compressed_k..(query + 1) * compressed_k],
                );
            }
            (indices, k)
        };
        self.attend_and_project(layer, &csa2, &q, s, &rows, &indices, k, &positions)
    }

    /// The compressed rows this layer attends and each query's selection among them (offset).
    #[allow(clippy::too_many_arguments)]
    fn compressed(
        &mut self,
        layer: &LayerPlan,
        csa2: &Csa2AttentionPlan,
        x: &[f32],
        qr: &[f32],
        s: usize,
        start: usize,
        offset: usize,
        shared: &mut Shared,
    ) -> Result<(Vec<f32>, Vec<i64>, usize), ReferenceError> {
        let ratio = csa2.mode.ratio() as usize;
        let head_dim = csa2.head_dim as usize;
        let rd = csa2.rope_head_dim as usize;
        let index = layer.index as usize;
        let compress_len = (start + s) / ratio;
        let (kv_source, latent, indexer) = match csa2.mode {
            Csa2Mode::Full {
                indexer,
                compressor,
                ..
            } => (
                index,
                self.compressor(layer, csa2, x, s, start)?
                    .map(|latent| (latent, compressor.latent_quant)),
                Some(indexer),
            ),
            Csa2Mode::Reindex {
                kv_source, indexer, ..
            } => (kv_source as usize, None, Some(indexer)),
            Csa2Mode::Reuse { kv_source, .. } => (kv_source as usize, None, None),
            Csa2Mode::Window => unreachable!("window layers have no compressed rows"),
        };
        let (indices, k) = match indexer {
            Some(_) if compress_len == 0 => (Vec::new(), 0),
            Some(indexer) => self.indexer(
                layer,
                csa2,
                &indexer,
                x,
                qr,
                latent.as_ref().map(|(latent, _)| latent.as_slice()),
                s,
                start,
                offset,
                kv_source,
                shared,
            )?,
            None => shared.topk.clone().ok_or(invalid(
                Some(layer.index),
                "a Reuse layer follows an index source",
            ))?,
        };
        if indexer.is_some() {
            shared.topk = Some((indices.clone(), k));
        }
        if let Some((mut latent, latent_quant)) = latent {
            let groups = latent.len() / head_dim;
            let positions: Vec<usize> = if start == 0 {
                (0..groups).map(|group| group * ratio).collect()
            } else {
                vec![start + 1 - ratio]
            };
            f::rope_bf16(
                &mut latent,
                head_dim,
                rd,
                1,
                &positions,
                self.ropes.get(&csa2.rope),
                false,
            );
            step(latent_quant, Some(layer.index), LATENT_QUANT)?.apply(&mut latent);
            let state = &mut self.layers[index];
            if state.compressed.len() != (start / ratio) * head_dim {
                return Err(invalid(
                    Some(layer.index),
                    "compressed KV rows are contiguous",
                ));
            }
            state.compressed.extend_from_slice(&latent);
        }
        let source = &self.layers[kv_source].compressed;
        if source.len() < compress_len * head_dim {
            return Err(invalid(
                Some(layer.index),
                "kv source holds every complete group",
            ));
        }
        Ok((source[..compress_len * head_dim].to_vec(), indices, k))
    }

    /// Vendor `Compressor.forward`: the RoPE-free latent rows completed by this step.
    fn compressor(
        &mut self,
        layer: &LayerPlan,
        csa2: &Csa2AttentionPlan,
        x: &[f32],
        s: usize,
        start: usize,
    ) -> Result<Option<Vec<f32>>, ReferenceError> {
        let hidden = self.hidden;
        let head_dim = csa2.head_dim as usize;
        let ratio = csa2.mode.ratio() as usize;
        let index = layer.index as usize;
        let eps = layer.pre_attention_norm.epsilon;
        let wkv = self.lt(
            layer.index,
            LayerTensor::KvCompressorKeyValue,
            &[head_dim, hidden],
        )?;
        let norm = self.lt(layer.index, LayerTensor::KvCompressorNorm, &[head_dim])?;
        if ratio == 1 {
            return Ok(Some(f::rmsnorm_bf16(
                &f::linear_bf16(x, s, hidden, wkv, head_dim),
                norm,
                eps,
            )));
        }
        let wgate = self.lt(
            layer.index,
            LayerTensor::KvCompressorGate,
            &[head_dim, hidden],
        )?;
        let kv = f::linear_f32(x, s, hidden, wkv, head_dim);
        let score = f::linear_f32(x, s, hidden, wgate, head_dim);
        // `(kv * score.softmax(dim=ratio)).sum(dim=ratio)`, cast to bf16.
        let pool = |kv: &[f32], score: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; head_dim];
            for d in 0..head_dim {
                let column: Vec<f32> = (0..ratio).map(|r| score[r * head_dim + d]).collect();
                let pooled: f64 = f::softmax_cr(&column)
                    .iter()
                    .enumerate()
                    .map(|(r, &p)| f64::from(kv[r * head_dim + d] * p))
                    .sum();
                out[d] = f::bf16(pooled as f32);
            }
            out
        };
        let state = &mut self.layers[index];
        let pooled = if start == 0 {
            let remainder = s % ratio;
            let cutoff = s - remainder;
            state.kv_carry[..remainder * head_dim]
                .copy_from_slice(&kv[cutoff * head_dim..s * head_dim]);
            state.score_carry[..remainder * head_dim]
                .copy_from_slice(&score[cutoff * head_dim..s * head_dim]);
            if s < ratio {
                return Ok(None);
            }
            let mut rows = Vec::with_capacity(cutoff / ratio * head_dim);
            for group in 0..cutoff / ratio {
                let span = group * ratio * head_dim..(group + 1) * ratio * head_dim;
                rows.extend(pool(&kv[span.clone()], &score[span]));
            }
            rows
        } else {
            let slot = start % ratio;
            state.kv_carry[slot * head_dim..(slot + 1) * head_dim].copy_from_slice(&kv);
            state.score_carry[slot * head_dim..(slot + 1) * head_dim].copy_from_slice(&score);
            if !(start + 1).is_multiple_of(ratio) {
                return Ok(None);
            }
            pool(&state.kv_carry, &state.score_carry)
        };
        Ok(Some(f::rmsnorm_bf16(&pooled, norm, eps)))
    }

    /// Vendor `Indexer.forward`: appends this step's index keys (Full layers), scores every
    /// reachable compressed entry, applies the candidate pool and returns the top-k.
    #[allow(clippy::too_many_arguments)]
    fn indexer(
        &mut self,
        layer: &LayerPlan,
        csa2: &Csa2AttentionPlan,
        plan: &memra_gguf::model_plan::Csa2IndexerPlan,
        x: &[f32],
        qr: &[f32],
        latent: Option<&[f32]>,
        s: usize,
        start: usize,
        offset: usize,
        kv_source: usize,
        shared: &mut Shared,
    ) -> Result<(Vec<i64>, usize), ReferenceError> {
        let hidden = self.hidden;
        let ratio = csa2.mode.ratio() as usize;
        let head_dim = csa2.head_dim as usize;
        let rd = csa2.rope_head_dim as usize;
        let (heads, index_dim) = (plan.heads as usize, plan.head_dim as usize);
        let q_lora = csa2.q_lora_rank as usize;
        let eps = layer.pre_attention_norm.epsilon;
        let freqs = self.ropes.get(&csa2.rope);
        let end = start + s;
        if let Some(latent) = latent {
            let groups = latent.len() / head_dim;
            let positions: Vec<usize> = if start == 0 {
                (0..groups).map(|group| group * ratio).collect()
            } else {
                vec![start + 1 - ratio]
            };
            let wk = self.ft(layer, "attn.indexer.wk.weight", &[index_dim, head_dim])?;
            let k_norm = self.ft(layer, "attn.indexer.k_norm.weight", &[index_dim])?;
            let mut keys = f::rmsnorm_bf16(
                &f::linear_bf16(latent, groups, head_dim, wk, index_dim),
                k_norm,
                eps,
            );
            f::rope_bf16(&mut keys, index_dim, rd, 1, &positions, freqs, false);
            step(plan.key_quant, Some(layer.index), KEY_QUANT)?.apply(&mut keys);
            self.layers[layer.index as usize]
                .index_keys
                .extend_from_slice(&keys);
        }
        let width = end / ratio;
        let keys = &self.layers[kv_source].index_keys;
        if keys.len() < width * index_dim {
            return Err(invalid(
                Some(layer.index),
                "index keys cover every complete group",
            ));
        }
        let keys = &keys[..width * index_dim];
        let positions: Vec<usize> = (start..end).collect();
        let mut q = f::linear_quant(
            qr,
            s,
            q_lora,
            self.lt(
                layer.index,
                LayerTensor::SparseQuery,
                &[heads * index_dim, q_lora],
            )?,
            heads * index_dim,
            self.numerics.linear,
        );
        f::rope_bf16(&mut q, index_dim, rd, heads, &positions, freqs, false);
        step(plan.query_quant, Some(layer.index), QUERY_QUANT)?.apply(&mut q);
        let head_scale = ((index_dim as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
        let weights: Vec<f32> = f::linear_bf16(
            x,
            s,
            hidden,
            self.lt(layer.index, LayerTensor::SparseProjection, &[heads, hidden])?,
            heads,
        )
        .into_iter()
        .map(|w| f::bf16(w * head_scale))
        .collect();
        let mut scores = vec![0f32; s * width];
        for query in 0..s {
            let reachable = if start == 0 {
                (query + 1) / ratio
            } else {
                width
            };
            for entry in 0..width {
                if entry >= reachable {
                    scores[query * width + entry] = f32::NEG_INFINITY;
                    continue;
                }
                let key = &keys[entry * index_dim..(entry + 1) * index_dim];
                // einsum (bf16) -> relu -> * weights (bf16) -> sum over heads (bf16).
                let sum: f64 = (0..heads)
                    .map(|head| {
                        let q_row = &q[(query * heads + head) * index_dim
                            ..(query * heads + head + 1) * index_dim];
                        let score = f::bf16(memra_gguf::dsv4_forward::dot(q_row, key)).max(0.0);
                        f64::from(f::bf16(score * weights[query * heads + head]))
                    })
                    .sum();
                scores[query * width + entry] = f::bf16(sum as f32);
            }
        }
        match plan.candidates {
            Csa2Candidates::None => {}
            Csa2Candidates::Produce {
                top_blocks,
                block_size,
            } => {
                let mut mask = Vec::with_capacity(s * width);
                for query in 0..s {
                    let reachable = if start == 0 {
                        (query + 1) / ratio
                    } else {
                        width
                    };
                    mask.extend(f::select_candidate_blocks(
                        &scores[query * width..(query + 1) * width],
                        reachable,
                        top_blocks as usize,
                        block_size as usize,
                    ));
                }
                shared.candidates = Some((mask, width));
            }
            Csa2Candidates::Restrict { .. } => {
                let (mask, mask_width) = shared.candidates.as_ref().ok_or(invalid(
                    Some(layer.index),
                    "restricted indexer follows its source",
                ))?;
                if *mask_width != width {
                    return Err(invalid(Some(layer.index), "candidate mask width differs"));
                }
                for (score, &keep) in scores.iter_mut().zip(mask) {
                    if !keep {
                        *score = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let k = (plan.top_k as usize).min(width);
        let mut indices = Vec::with_capacity(s * k);
        for query in 0..s {
            let reachable = if start == 0 {
                (query + 1) / ratio
            } else {
                width
            };
            for entry in f::topk_ascending(&scores[query * width..(query + 1) * width], k) {
                indices.push(if entry < reachable {
                    (entry + offset) as i64
                } else {
                    -1
                });
            }
        }
        Ok((indices, k))
    }

    fn moe(&self, layer: &LayerPlan, x: &[f32], s: usize) -> Result<Vec<f32>, ReferenceError> {
        let hidden = self.hidden;
        let moe = moe_of(layer)?;
        let (experts, top_k, inter) = (
            moe.expert_count as usize,
            moe.experts_per_token as usize,
            moe.expert_intermediate_size as usize,
        );
        let RouterPlan::SqrtSoftplus {
            normalize_selected,
            scaling_factor,
            selection_bias: true,
        } = moe.router
        else {
            return Err(invalid(
                Some(layer.index),
                "deepseek_v41 routes with biased sqrtsoftplus",
            ));
        };
        let ActivationPlan::SwiGluPreClamped { limit } = moe.activation else {
            return Err(invalid(
                Some(layer.index),
                "deepseek_v41 experts are clamped SwiGLU",
            ));
        };
        let i = layer.index;
        let logits = f::linear_f32(
            x,
            s,
            hidden,
            self.lt(i, LayerTensor::MoeRouter, &[experts, hidden])?,
            experts,
        );
        let bias = self.lt(i, LayerTensor::MoeRouterBias, &[experts])?;
        let routes: Vec<(Vec<usize>, Vec<f32>)> = (0..s)
            .map(|token| {
                f::route_sqrtsoftplus(
                    &logits[token * experts..(token + 1) * experts],
                    bias,
                    top_k,
                    normalize_selected,
                    scaling_factor,
                )
            })
            .collect();
        let mut y = vec![0f32; s * hidden];
        for expert in 0..experts {
            let mut rows = Vec::new();
            let mut weights = Vec::new();
            for (token, (chosen, route_weights)) in routes.iter().enumerate() {
                if let Some(slot) = chosen.iter().position(|&e| e == expert) {
                    rows.push(token);
                    weights.push(route_weights[slot]);
                }
            }
            if rows.is_empty() {
                continue;
            }
            let mut input = Vec::with_capacity(rows.len() * hidden);
            for &token in &rows {
                input.extend_from_slice(&x[token * hidden..(token + 1) * hidden]);
            }
            let e = |tensor| TensorId::Expert {
                layer: i,
                expert: expert as u32,
                tensor,
            };
            let out = f::expert(
                &input,
                rows.len(),
                hidden,
                inter,
                self.t(&e(ExpertTensor::Gate), &[inter, hidden])?,
                self.t(&e(ExpertTensor::Down), &[hidden, inter])?,
                self.t(&e(ExpertTensor::Up), &[inter, hidden])?,
                limit,
                Some(&weights),
                self.numerics.linear,
            );
            for (row, &token) in rows.iter().enumerate() {
                for d in 0..hidden {
                    y[token * hidden + d] += out[row * hidden + d];
                }
            }
        }
        let shared = shared_width(layer, moe)?;
        let out = f::expert(
            x,
            s,
            hidden,
            shared,
            self.lt(i, LayerTensor::SharedMlpGate, &[shared, hidden])?,
            self.lt(i, LayerTensor::SharedMlpDown, &[hidden, shared])?,
            self.lt(i, LayerTensor::SharedMlpUp, &[shared, hidden])?,
            limit,
            None,
            self.numerics.linear,
        );
        for (value, &shared) in y.iter_mut().zip(&out) {
            *value = f::bf16(*value + shared);
        }
        Ok(y)
    }

    /// DSpark main-model fusion: `main_norm(main_proj(taps))`, `[tokens, hidden]`.
    fn dspark_main(&self, taps: &[f32], s: usize) -> Result<Vec<f32>, ReferenceError> {
        let hidden = self.hidden;
        let targets = self.dspark.target_layer_ids.len();
        let eps = self.plan.output_norm.epsilon;
        Ok(f::rmsnorm_bf16(
            &f::linear_quant(
                taps,
                s,
                targets * hidden,
                self.t(
                    &TensorId::Dspark(DsparkTensor::MainProjection),
                    &[hidden, targets * hidden],
                )?,
                hidden,
                self.numerics.linear,
            ),
            self.t(&TensorId::Dspark(DsparkTensor::MainNorm), &[hidden])?,
            eps,
        ))
    }

    /// Vendor `DSparkAttention` main-token KV: the window KV program on `main_x`.
    fn dspark_main_kv(
        &self,
        block: &LayerPlan,
        main_x: &[f32],
        positions: &[usize],
    ) -> Result<Vec<f32>, ReferenceError> {
        let csa2 = *csa2_of(block)?;
        self.window_kv(block, &csa2, main_x, positions.len(), positions)
    }

    /// `forward_spec(start_pos=0)`: every DSpark block seeds its ring from the prefill taps.
    fn dspark_seed(&mut self, taps: &[f32], s: usize) -> Result<(), ReferenceError> {
        let main_x = self.dspark_main(taps, s)?;
        let positions: Vec<usize> = (0..s).collect();
        for (block_index, block) in self.dspark.blocks.iter().enumerate() {
            let csa2 = csa2_of(block)?;
            let (window, head_dim) = (csa2.window as usize, csa2.head_dim as usize);
            let kv = self.dspark_main_kv(block, &main_x, &positions)?;
            for position in s.saturating_sub(window)..s {
                self.dspark_rings[block_index]
                    [(position % window) * head_dim..(position % window + 1) * head_dim]
                    .copy_from_slice(&kv[position * head_dim..(position + 1) * head_dim]);
            }
        }
        Ok(())
    }

    fn dspark_attention(
        &mut self,
        block: &LayerPlan,
        block_index: usize,
        x: &[f32],
        block_size: usize,
        start: usize,
        main_x: &[f32],
    ) -> Result<Vec<f32>, ReferenceError> {
        let csa2 = *csa2_of(block)?;
        let (window, head_dim) = (csa2.window as usize, csa2.head_dim as usize);
        let main_kv = self.dspark_main_kv(block, main_x, &[start])?;
        let positions: Vec<usize> = (start + 1..start + 1 + block_size).collect();
        let (_, q) = self.project_query(block, &csa2, x, block_size, &positions)?;
        let kv = self.window_kv(block, &csa2, x, block_size, &positions)?;
        let slot = start % window;
        self.dspark_rings[block_index][slot * head_dim..(slot + 1) * head_dim]
            .copy_from_slice(&main_kv);
        let mut rows = self.dspark_rings[block_index].clone();
        rows.extend_from_slice(&kv);
        let valid = window.min(start + 1);
        let k = valid + block_size;
        let mut indices = Vec::with_capacity(block_size * k);
        for _ in 0..block_size {
            indices.extend((0..valid).map(|slot| slot as i64));
            indices.extend((0..block_size).map(|row| (window + row) as i64));
        }
        self.attend_and_project(block, &csa2, &q, block_size, &rows, &indices, k, &positions)
    }

    /// `forward_spec(start_pos > 0)` after the decode step at `start`: one draft block.
    fn dspark_draft(
        &mut self,
        taps: &[f32],
        input_token: u32,
        start: usize,
    ) -> Result<ReferenceDraftOutput, ReferenceError> {
        let (hidden, streams, vocab) = (self.hidden, self.streams, self.vocab);
        let block_size = self.dspark.block_size as usize;
        let rank = self.dspark.markov_rank as usize;
        let main_x = self.dspark_main(taps, 1)?;
        let mut draft_ids = vec![self.dspark.noise_token_id; block_size];
        draft_ids[0] = input_token;
        let mut h = self.embed(&draft_ids)?;
        let mut pre = self.identity_pre(block_size);
        let blocks = self.dspark.blocks.clone();
        let mut shared = Shared::default();
        for (block_index, block) in blocks.iter().enumerate() {
            let (next, next_pre) = self.block(
                block,
                &h,
                &pre,
                block_size,
                start,
                &mut shared,
                Some((block_index, &main_x)),
            )?;
            h = next;
            pre = next_pre;
        }
        let collapsed = f::hc_pre(&h, block_size, streams, hidden, &pre);
        let normalized = f::rmsnorm_bf16(
            &collapsed,
            self.t(&TensorId::Dspark(DsparkTensor::OutputNorm), &[hidden])?,
            self.plan.output_norm.epsilon,
        );
        let mut logits = f::linear_f32(
            &normalized,
            block_size,
            hidden,
            self.t(&TensorId::OutputProjection, &[vocab, hidden])?,
            vocab,
        );
        let markov_embedding = self.t(
            &TensorId::Dspark(DsparkTensor::MarkovEmbedding),
            &[vocab, rank],
        )?;
        let markov_output = self.t(
            &TensorId::Dspark(DsparkTensor::MarkovOutput),
            &[vocab, rank],
        )?;
        let confidence_weight = self.t(
            &TensorId::Dspark(DsparkTensor::ConfidenceProjection),
            &[1, hidden + rank],
        )?;
        let mut output_ids = vec![input_token];
        let mut confidence = Vec::with_capacity(block_size);
        for position in 0..block_size {
            let previous = output_ids[position] as usize;
            let markov = &markov_embedding[previous * rank..(previous + 1) * rank];
            let bias = f::linear_f32(markov, 1, rank, markov_output, vocab);
            let row = &mut logits[position * vocab..(position + 1) * vocab];
            for (value, &bias) in row.iter_mut().zip(&bias) {
                *value += bias;
            }
            let next = row
                .iter()
                .enumerate()
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        .then_with(|| right_index.cmp(left_index))
                })
                .map(|(index, _)| index as u32)
                .unwrap();
            output_ids.push(next);
            let mut input = Vec::with_capacity(hidden + rank);
            input.extend_from_slice(&collapsed[position * hidden..(position + 1) * hidden]);
            input.extend_from_slice(markov);
            confidence.push(f::linear_f32(&input, 1, hidden + rank, confidence_weight, 1)[0]);
        }
        Ok(ReferenceDraftOutput {
            input_token,
            output_ids,
            confidence,
            logits,
            hidden: collapsed,
            block_size,
        })
    }

    fn state(&self) -> ReferenceState {
        ReferenceState {
            layers: self
                .plan
                .layers
                .iter()
                .zip(&self.layers)
                .map(|(layer, state)| {
                    let head_dim = csa2_of(layer).map(|c| c.head_dim as usize).unwrap_or(1);
                    ReferenceLayerState::Csa2 {
                        window: state.ring.clone(),
                        head_dim,
                        compressed: state.compressed.clone(),
                        index_keys: state.index_keys.clone(),
                    }
                })
                .collect(),
        }
    }
}

/// Run `token_ids` as one request: `token_ids[..decode_from]` as one prefill chunk at position
/// 0, then each later token as one decode step; after every decode step except the last, a
/// DSpark draft from the next token. `None` prefills every token and drafts nothing.
pub fn execute(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    decode_from: Option<usize>,
) -> Result<ReferenceOutput, ReferenceError> {
    if token_ids.is_empty() {
        return Err(ReferenceError::EmptyInput);
    }
    let prefill = decode_from.unwrap_or(token_ids.len());
    if prefill == 0 || prefill > token_ids.len() {
        return Err(invalid(
            None,
            "the prefill chunk is a non-empty prefix of the tokens",
        ));
    }
    let dspark = dspark_of(plan)?;
    let mut session = Session::new(
        plan,
        weights,
        token_ids.len() + dspark.block_size as usize + 1,
    )?;
    let vocab = plan.vocab_size as usize;
    let first = session.forward(&token_ids[..prefill], 0)?;
    let mut logits = first.logits;
    let mut layer_hidden = first.layer_hidden;
    let mut drafts = Vec::new();
    if decode_from.is_some() {
        session.dspark_seed(&first.taps, prefill)?;
        for position in prefill..token_ids.len() {
            let step = session.forward(&token_ids[position..position + 1], position)?;
            logits.extend_from_slice(&step.logits);
            for (rows, step_rows) in layer_hidden.iter_mut().zip(&step.layer_hidden) {
                rows.extend_from_slice(step_rows);
            }
            if position + 1 < token_ids.len() {
                drafts.push(session.dspark_draft(&step.taps, token_ids[position + 1], position)?);
            }
        }
    }
    debug_assert_eq!(logits.len(), token_ids.len() * vocab);
    Ok(ReferenceOutput {
        logits,
        tokens: token_ids.len(),
        vocab,
        state: session.state(),
        mtp: Vec::new(),
        draft: None,
        layer_hidden,
        decode: decode_from.map(|prefill_tokens| ReferenceDecodeTrace {
            prefill_tokens,
            drafts,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use std::collections::BTreeMap;

    const CONFIG_1017: &str =
        include_str!("../../memra-gguf/src/model_packs/deepseek_v41/fixtures/config.json");

    /// The plan-derived checkpoint census equals the pack's config-derived census for the 1017
    /// config, minus the tensors the text program does not read (tower, aligner, image-span
    /// embeddings, image-token router bias).
    #[test]
    fn plan_census_matches_the_pack_census() {
        let config = ModelConfig::from_hf(&HfConfig::parse(CONFIG_1017));
        let plan = pack::PACK.compile_plan(&config).unwrap();
        let mut from_plan = BTreeMap::new();
        for tensor in checkpoint_tensors(&plan).unwrap() {
            for (name, dtype, shape) in tensor.entries() {
                let shape: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
                assert!(from_plan.insert(name, (dtype, shape)).is_none());
            }
        }
        let from_pack: BTreeMap<String, (&str, Vec<u64>)> = pack::expected_census(&config)
            .into_iter()
            .filter(|(name, _)| {
                !(name.starts_with("vision.")
                    || name.starts_with("aligner.")
                    || name.starts_with("image_")
                    || name.ends_with(".bias_vl"))
            })
            .map(|(name, spec)| (name, (spec.dtype, spec.shape)))
            .collect();
        assert_eq!(from_plan.len(), from_pack.len());
        assert_eq!(from_plan, from_pack);
    }

    #[test]
    fn tiny_fixture_is_deterministic_and_runs() {
        let plan = pack::PACK.compile_tiny_plan().unwrap();
        let fixture = fixture(&plan).unwrap();
        assert_eq!(fixture, super::fixture(&plan).unwrap());
        let output = execute(
            &plan,
            &fixture.weights,
            &fixture.token_ids,
            fixture.decode_from,
        )
        .unwrap();
        assert!(output.logits.iter().all(|v| v.is_finite()));
        let trace = output.decode.as_ref().unwrap();
        assert_eq!(trace.prefill_tokens, 73);
        assert_eq!(trace.drafts.len(), 7);
    }

    /// At or past the window, a decode step runs the program the prefill runs for that
    /// position's row (the same window entries in the same order, the same compressed rows and
    /// selections; the prefill's extra empty slots sit after them and add nothing), so a prefill
    /// of every token and a prefill-then-decode split agree to the bit, for a one-block window
    /// (64) and a two-block one (128). Below the window they do not:
    /// [`below_the_window_prefill_and_decode_are_different_programs`].
    #[test]
    fn decode_at_or_past_the_window_continues_the_prefill_program_bit_for_bit() {
        let narrow = pack::PACK.compile_tiny_plan().unwrap();
        let wide = pack::tiny_window128_plan().unwrap();
        for (plan, splits) in [(narrow, [64, 70, 73, 80]), (wide, [128, 130, 137, 144])] {
            let fixture = fixture(&plan).unwrap();
            let tokens = &fixture.token_ids;
            let all = execute(&plan, &fixture.weights, tokens, None).unwrap();
            for split in splits {
                let stepped = execute(&plan, &fixture.weights, tokens, Some(split)).unwrap();
                assert_eq!(all.logits, stepped.logits, "split {split}");
                assert_eq!(all.layer_hidden, stepped.layer_hidden, "split {split}");
            }
        }
    }

    /// Below the window the vendor's prefill and decode are different numeric programs for the
    /// same row: `get_window_topk_idxs` lays a prefill row out `min(seqlen, window)` wide with
    /// its empty slots last and a decode row as the ring oldest first with its empty slots
    /// first, and `sparse_attn` rounds probabilities to bf16 per 64-entry block, so where a
    /// row's valid entries fall in those blocks changes its numbers. memra runs both programs
    /// as the vendor does (the `split60` and `window128` tiny variants are pinned to vendor
    /// oracles); this pins that the split itself moves the rows, a constraint any serve route
    /// that primes, chunks or splits a request below the window inherits.
    #[test]
    fn below_the_window_prefill_and_decode_are_different_programs() {
        let narrow = pack::PACK.compile_tiny_plan().unwrap();
        let wide = pack::tiny_window128_plan().unwrap();
        let vocab = narrow.vocab_size as usize;
        for (plan, split, first_moved) in [(narrow, 60, 0), (wide.clone(), 70, 64), (wide, 100, 64)]
        {
            let fixture = fixture(&plan).unwrap();
            let tokens = &fixture.token_ids;
            let all = execute(&plan, &fixture.weights, tokens, None).unwrap();
            let stepped = execute(&plan, &fixture.weights, tokens, Some(split)).unwrap();
            let moved: Vec<usize> = (0..tokens.len())
                .filter(|&p| {
                    all.logits[p * vocab..(p + 1) * vocab]
                        != stepped.logits[p * vocab..(p + 1) * vocab]
                })
                .collect();
            assert_eq!(moved.first(), Some(&first_moved), "split {split}");
            assert_eq!(moved.last(), Some(&(tokens.len() - 1)), "split {split}");
        }
    }

    type PlanEdit<'a> = Box<dyn Fn(&mut ModelPlan) + 'a>;

    fn refusal(result: Result<impl std::fmt::Debug, ReferenceError>) -> &'static str {
        match result {
            Err(ReferenceError::InvalidPlan { reason, .. }) => reason,
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    fn csa2_mut(layer: &mut LayerPlan) -> &mut Csa2AttentionPlan {
        match &mut layer.attention {
            AttentionPlan::Csa2(csa2) => csa2,
            _ => unreachable!("deepseek_v41 layers are CSA2"),
        }
    }

    /// The executor reads every quant field of the plan: a step it implements changes the run
    /// (and the vendor arguments refuse it, since `model.py` fixes its steps as constants); a
    /// step, storage or program switch it does not implement refuses by name. Nothing runs the
    /// 1017 program under another plan's numerics.
    #[test]
    fn the_plan_numerics_are_read_or_refused_by_name() {
        use memra_gguf::model_plan::{BlockElement, BlockScale};
        let plan = pack::PACK.compile_tiny_plan().unwrap();
        let fixture = fixture(&plan).unwrap();
        let tokens = &fixture.token_ids;
        let decode = fixture.decode_from;
        let base = execute(&plan, &fixture.weights, tokens, decode).unwrap();
        let args = vendor_model_args(&plan).unwrap();
        for line in [
            "\"dtype\": \"fp8\",",
            "\"expert_dtype\": \"fp4\",",
            "\"n_shared_experts\": 1,",
        ] {
            assert!(args.contains(line), "{line} not in {args}");
        }
        let run = |plan: &ModelPlan| execute(plan, &fixture.weights, tokens, decode);
        let fp8 = |block| BlockQuant {
            element: BlockElement::Fp8E4m3,
            block,
            scale: BlockScale::Ue8m0,
        };
        let all_layers = |plan: &mut ModelPlan, edit: &dyn Fn(&mut Csa2AttentionPlan)| {
            let DrafterPlan::Dspark(dspark) = plan.drafter.as_mut().unwrap() else {
                unreachable!()
            };
            for layer in plan.layers.iter_mut().chain(dspark.blocks.iter_mut()) {
                edit(csa2_mut(layer));
            }
        };
        // Implemented steps the plan can name: each moves the run, and the vendor arguments
        // refuse to describe the result as the vendor program.
        let implemented: Vec<(&str, PlanEdit<'_>)> = vec![
            (
                "window_kv_quant",
                Box::new(move |p| all_layers(p, &|c| c.window_kv_quant = BlockQuant::FP4_32_UE8M0)),
            ),
            (
                "output_quant",
                Box::new(move |p| all_layers(p, &|c| c.output_quant = BlockQuant::FP4_32_UE8M0)),
            ),
            (
                "latent_quant",
                Box::new(move |p| {
                    all_layers(p, &|c| {
                        if let Csa2Mode::Full { compressor, .. } = &mut c.mode {
                            compressor.latent_quant = BlockQuant::FP4_32_UE8M0;
                        }
                    })
                }),
            ),
            (
                "query_quant",
                Box::new(|p| {
                    if let Csa2Mode::Full { indexer, .. } = &mut csa2_mut(&mut p.layers[1]).mode {
                        indexer.query_quant = fp8(32);
                    }
                }),
            ),
            (
                // Layer 1's keys feed only its own group (no Reindex layer reads them).
                "key_quant",
                Box::new(|p| {
                    if let Csa2Mode::Full { indexer, .. } = &mut csa2_mut(&mut p.layers[1]).mode {
                        indexer.key_quant = BlockQuant::FP4_16_E4M3;
                    }
                }),
            ),
        ];
        for (field, edit) in &implemented {
            let mut edited = plan.clone();
            edit(&mut edited);
            assert_ne!(edited, plan, "{field}: the edit changes the plan");
            let moved = run(&edited).unwrap_or_else(|e| panic!("{field}: {e:?}"));
            assert_ne!(
                moved.logits, base.logits,
                "{field}: the run reads the field"
            );
            assert!(
                refusal(vendor_model_args(&edited)).contains(field),
                "{field}: the vendor arguments refuse it"
            );
        }
        // The linear activation step is read the same way. Its only implemented alternatives are
        // other FP8 ue8m0 blocks, and a power-of-two scale leaves every e4m3 value in the normal
        // range where it was, so the run barely moves; the resolved step shows the read.
        let mut edited = plan.clone();
        edited.deepseek_v41.as_mut().unwrap().linear_activation = fp8(16);
        assert_eq!(numerics(&plan).unwrap().linear, f::QuantStep::Fp8Ue8m0(32));
        assert_eq!(
            numerics(&edited).unwrap().linear,
            f::QuantStep::Fp8Ue8m0(16)
        );
        run(&edited).unwrap();
        assert!(refusal(vendor_model_args(&edited)).contains("linear_activation"));
        // Steps, storage and program switches it does not implement refuse by name, before a
        // tensor is read and before any fixture or vendor argument is written.
        let refused: Vec<(&str, PlanEdit<'_>)> = vec![
            (
                "window_kv_quant names an unimplemented step",
                Box::new(move |p| {
                    all_layers(p, &|c| {
                        c.window_kv_quant = BlockQuant {
                            scale: BlockScale::E4m3,
                            ..BlockQuant::FP8_32_UE8M0
                        }
                    })
                }),
            ),
            (
                "output_quant names an unimplemented step or a block that does not divide",
                Box::new(move |p| all_layers(p, &|c| c.output_quant = fp8(48))),
            ),
            (
                "key_quant: a Reindex layer reads its kv source's index keys",
                Box::new(|p| {
                    if let Csa2Mode::Full { indexer, .. } = &mut csa2_mut(&mut p.layers[5]).mode {
                        indexer.key_quant = fp8(32);
                    }
                }),
            ),
            (
                "dense_weight_block",
                Box::new(|p| p.deepseek_v41.as_mut().unwrap().dense_weight_block = 64),
            ),
            (
                "expert_weight",
                Box::new(|p| {
                    p.deepseek_v41.as_mut().unwrap().expert_weight = BlockQuant::FP4_16_E4M3
                }),
            ),
            (
                "linear_activation: quantized linears take FP8",
                Box::new(|p| {
                    p.deepseek_v41.as_mut().unwrap().linear_activation = BlockQuant::FP4_32_UE8M0
                }),
            ),
            (
                "linear_activation: its block does not divide",
                Box::new(move |p| p.deepseek_v41.as_mut().unwrap().linear_activation = fp8(64)),
            ),
            (
                "routed_weight_before_down",
                Box::new(|p| p.deepseek_v41.as_mut().unwrap().routed_weight_before_down = false),
            ),
            (
                "output_lora_a_dequantized_bf16",
                Box::new(|p| {
                    p.deepseek_v41
                        .as_mut()
                        .unwrap()
                        .output_lora_a_dequantized_bf16 = false
                }),
            ),
            (
                "table_quant",
                Box::new(move |p| {
                    let ResidualTopology::SinglePassHyperConnections(hc) =
                        &mut p.layers[1].residual
                    else {
                        unreachable!()
                    };
                    hc.engram.as_mut().unwrap().table_quant = fp8(16);
                }),
            ),
            (
                "partial_group",
                Box::new(|p| {
                    let StatePlan::Csa2(state) = &mut p.layers[1].state else {
                        unreachable!()
                    };
                    state.compressed = Csa2CompressedState::Own {
                        ratio: 2,
                        index_head_dim: 32,
                        partial_group: false,
                    };
                }),
            ),
            (
                "shared experts are ungated",
                Box::new(|p| {
                    let MlpPlan::Moe(moe) = &mut p.layers[0].mlp else {
                        unreachable!()
                    };
                    moe.shared.as_mut().unwrap().gated = true;
                }),
            ),
        ];
        for (needle, edit) in &refused {
            let mut edited = plan.clone();
            edit(&mut edited);
            assert_ne!(edited, plan, "{needle}: the edit changes the plan");
            for (what, reason) in [
                ("execute", refusal(run(&edited))),
                ("fixture", refusal(super::fixture(&edited))),
                ("vendor_model_args", refusal(vendor_model_args(&edited))),
            ] {
                assert!(
                    reason.contains(needle),
                    "{what}: {needle:?} not in {reason:?}"
                );
            }
        }
    }

    /// The loader's refusal of non-finite values reaches the FP8 NaN codes (torch
    /// `float8_e4m3fn` 0x7f / 0xff) and a NaN e8m0 scale (0xff), on dense and Engram storage.
    #[test]
    fn dequantize_refuses_fp8_nan_codes_and_nan_scales() {
        for storage in [CheckpointStorage::Fp8Block32, CheckpointStorage::Fp8Row32] {
            let tensor = CheckpointTensor {
                name: "probe.weight".into(),
                id: family_id("probe.", "weight"),
                storage,
                shape: vec![32, 32],
            };
            let scales = if storage == CheckpointStorage::Fp8Block32 {
                1
            } else {
                32
            };
            let codes = vec![0x38u8; 32 * 32];
            assert!(dequantize(&tensor, &codes, Some(&vec![127u8; scales])).is_ok());
            for nan in [0x7fu8, 0xff] {
                let mut codes = codes.clone();
                codes[100] = nan;
                assert_eq!(
                    refusal(dequantize(&tensor, &codes, Some(&vec![127u8; scales]))),
                    "checkpoint tensor holds or dequantizes to a non-finite value",
                    "{storage:?} code {nan:#04x}"
                );
            }
            let mut nan_scale = vec![127u8; scales];
            nan_scale[0] = 0xff;
            assert_eq!(
                refusal(dequantize(&tensor, &codes, Some(&nan_scale))),
                "checkpoint tensor holds or dequantizes to a non-finite value",
                "{storage:?} NaN scale"
            );
        }
    }

    /// Confusion guards: an Engram table, the candidate restriction and the DSpark weights each
    /// reach exactly the rows they must: the first layer that reads them moves, the layer before
    /// it does not (and the DSpark weights move the drafts, never the trunk).
    #[test]
    fn tiny_fixture_reaches_engram_candidates_and_the_dspark_ring() {
        let plan = pack::PACK.compile_tiny_plan().unwrap();
        let fixture = fixture(&plan).unwrap();
        let tokens = &fixture.token_ids;
        let decode = fixture.decode_from;
        let base = execute(&plan, &fixture.weights, tokens, decode).unwrap();
        let rerun = |weights: &ReferenceWeights| execute(&plan, weights, tokens, decode).unwrap();
        // Engram: zero the second table (layer 3); every position hashes into it.
        let mut weights = fixture.weights.clone();
        let table = family_id("layers.3.", "engram.embed.weight");
        weights.get_mut(&table).unwrap().data.fill(0.0);
        let moved = rerun(&weights);
        assert_ne!(moved.layer_hidden[3], base.layer_hidden[3]);
        assert_eq!(moved.layer_hidden[2], base.layer_hidden[2]);
        // Candidate pool: the same weights with layer 7's indexer unrestricted. Layer 6 is
        // untouched and layer 7 moves, so the restricted indexer does read the pool.
        let mut unrestricted = plan.clone();
        let AttentionPlan::Csa2(csa2) = &mut unrestricted.layers[7].attention else {
            unreachable!()
        };
        let Csa2Mode::Reindex { indexer, .. } = &mut csa2.mode else {
            panic!("tiny layer 7 is a Reindex layer");
        };
        assert!(matches!(
            indexer.candidates,
            Csa2Candidates::Restrict { .. }
        ));
        indexer.candidates = Csa2Candidates::None;
        let moved = execute(&unrestricted, &fixture.weights, tokens, decode).unwrap();
        assert_eq!(moved.layer_hidden[6], base.layer_hidden[6]);
        assert_ne!(moved.layer_hidden[7], base.layer_hidden[7]);
        // DSpark ring: the drafts read the ring every decode step writes.
        let mut weights = fixture.weights.clone();
        let wkv = layer_id(plan.layers.len() as u32, LayerTensor::MlaKvDown);
        for value in &mut weights.get_mut(&wkv).unwrap().data {
            *value = -*value;
        }
        let moved = rerun(&weights);
        assert_eq!(moved.logits, base.logits);
        let (a, b) = (base.decode.unwrap(), moved.decode.unwrap());
        assert_ne!(a.drafts[0].logits, b.drafts[0].logits);
    }
}
