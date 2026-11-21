//! Inkling 2 program types, compiled only by the `inkling2` pack.
//!
//! Semantic sources: the checkpoint's `config.json` and `hf_quant_config.json`
//! (`thinkingmachines/Inkling-2-Mini`), the model code of the vendor's serving image, and two
//! float-activation implementations used for triage (the transformers loop model and llama.cpp's
//! `inkling2.cpp` / `inkling2-loop.cpp`). The lane note is
//! `research/modelplan-onboarding-inkling2mini-20261121/SEMANTICS.md`.
//!
//! None of these types parameterizes another family's plan. No reference executor or tuned
//! surface implements them yet: every operation they emit is a registry row with no surface.

use super::{NormPlan, ValueNorm};
use crate::quant_groups::ActivationQuant;

/// Clip a vector to an L2 norm of at most `threshold`:
/// `x * min(1, threshold / (|x| + epsilon))`, after the optional learned scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L2ClipPlan {
    pub threshold: f32,
    pub epsilon: f32,
    pub scale: ClipScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipScale {
    None,
    /// One learned factor per head and channel, multiplied in before the norm is taken.
    PerHeadChannel,
}

/// Relative-position bias: `r = clip(scale * wr(x_q))` is a rank-`rank` vector per query head,
/// and `bias[h, d] = (r_h @ table[rank, extent])[d]` for distance `d = pos_q - pos_k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeBiasPlan {
    pub rank: u32,
    /// Columns of the bias table.
    pub extent: u32,
    pub clip: L2ClipPlan,
    pub beyond_extent: BeyondExtent,
}

/// What a key at distance `>= extent` gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeyondExtent {
    /// Full layers: the key is attended with a zero bias.
    ZeroBias,
    /// Sliding layers: the window ends at the table, so the key is masked.
    Masked,
}

/// `tau = 1 + alpha * ln(max(1, (pos + 1) / floor))`, multiplied into Q and R after their clips.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogPositionScalePlan {
    pub floor: u32,
    pub alpha: f32,
}

/// How a layer derives its three attention inputs from the residual mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionInputs {
    /// One mixed input and one norm (`attn_norm`) shared by Q, K and V. In a loop pass that
    /// carries a message the three inputs still differ, since each takes its own controller
    /// before the shared norm.
    SharedNorm,
    /// Three mixed inputs (three residual queries), each with its own norm
    /// (`q_input_norm`, `k_input_norm`, `v_input_norm`).
    SeparateQkvNorms,
}

/// The sink logit of the softmax denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuerySinkPlan {
    /// A per-query, per-head logit projected from the Q-side input (`attn_sink_proj`):
    /// `exp(s) / (exp(sink) + sum exp(s))`. Not a learned per-head constant.
    PerQueryProjection,
}

/// The attention output gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputGatePlan {
    /// `sigmoid(gate_attn(x_q))`, one value per head and channel, from its own tensor
    /// `[heads * head_dim, hidden]`. Applied before the output projection.
    SeparatePerDimSigmoid,
}

/// Tap layout of a short convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortConvTaps {
    /// Learned depthwise taps `[channels, 1, kernel]`.
    Static,
    /// Static taps `[kernel, channels]` plus input-dependent taps projected per group
    /// (`weight_proj [kernel * groups, hidden]`). Both are indexed by lag with the current token
    /// first.
    StaticPlusDynamic { groups: u32 },
}

/// Causal depthwise short convolution with a residual:
/// `(x + conv(x)) / sqrt(kernel + 1)`, or `sqrt(2 * kernel + 1)` with dynamic taps. Inputs before
/// position 0 are zero. Its state is the last `kernel - 1` inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvPlan {
    pub kernel: u32,
    pub channels: u32,
    pub taps: ShortConvTaps,
}

/// One attention layer of the family: GQA without RoPE, position through a relative bias and
/// (on full layers) a log scale, clipped Q and K, a weightless-RMS V, a per-query sink, a
/// separate per-dimension output gate, and short convolutions on K, V and the output.
/// Softmax scale `head_dim^-0.5`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeBiasAttentionPlan {
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// `Some(w)`: the query attends itself and the `w - 1` positions before it.
    pub window: Option<u32>,
    pub inputs: AttentionInputs,
    pub query_clip: L2ClipPlan,
    pub key_clip: L2ClipPlan,
    pub value_norm: ValueNorm,
    pub value_norm_epsilon: f32,
    pub relative: RelativeBiasPlan,
    /// Full layers only.
    pub log_scale: Option<LogPositionScalePlan>,
    pub sink: QuerySinkPlan,
    pub output_gate: OutputGatePlan,
    /// On the K projection, before the clip.
    pub key_conv: ShortConvPlan,
    /// On the V projection, before the norm.
    pub value_conv: ShortConvPlan,
    /// On the output projection.
    pub output_conv: ShortConvPlan,
}

/// Per-slot state: KV rows (a ring of `window` on sliding layers) and the short-conv history of
/// K, V, the attention output and the MLP output. Loop slots that share weights never share this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelativeBiasStatePlan {
    pub key_width: u32,
    pub value_width: u32,
    pub window: Option<u32>,
    /// `kernel - 1` rows of history.
    pub conv_history: u32,
    /// K + V + attention output + MLP output channels.
    pub conv_channels: u32,
}

/// Block attention residual. Layers are grouped in blocks of `block`. Every sublayer input is a
/// depth mix: `softmax(q . rms(s) / sqrt(hidden))` over the earlier block outputs and the
/// running stream, summing the RMS-normalized states. The first layer of a block pushes the
/// running stream onto the block list and starts a new one, so its attention output is not added
/// to its input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockAttentionPlan {
    pub block: u32,
    pub starts_block: bool,
    /// Residual queries of the attention sublayer: 0 on the first layer of the model, 1 on a
    /// sliding layer, 3 (Q, K, V) on a full layer.
    pub attention_queries: u32,
    /// The mix sums normalized states, not raw ones.
    pub normalize_values: bool,
    pub epsilon: f32,
}

/// Where a loop message enters a core layer. Order is the checkpoint's query order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMessageTarget {
    Query,
    Key,
    Value,
    Mlp,
}

/// One cache slot of the looped program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopSlotPlan {
    /// Cache slot; also the index into `ModelPlan::layers`.
    pub slot: u32,
    /// The physical layer whose weights the slot runs.
    pub physical_layer: u32,
    /// Loop pass, `None` for prelude and tail layers.
    pub pass: Option<u32>,
    /// The slot reads a message from the previous pass (passes after the first).
    pub message: bool,
}

/// The loop: core layers `core_start..core_end` run `passes` times over shared weights.
///
/// Every pass restarts from the prelude's stream and block list. A pass after the first reads a
/// message from the pass before it: a depth mix (the block-residual form, values normalized)
/// over that pass's block outputs after the first core block plus its final stream, with one
/// query per core layer and target. A message enters its target through a rank-`gate_rank`
/// controller: `g = down(cat(rms(x), rms(m)))`, `x - tanh(forget g) * x + tanh(gain g) * m`.
/// The message bank is per token; the loop adds no cross-token state.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopPlan {
    pub core_start: u32,
    pub core_end: u32,
    pub passes: u32,
    pub gate_rank: u32,
    pub controller_epsilon: f32,
    pub targets: [LoopMessageTarget; 4],
    /// One entry per cache slot, in slot order.
    pub slots: Vec<LoopSlotPlan>,
    /// Slots in execution order: prelude, every pass over the core, tail.
    pub execution_order: Vec<u32>,
}

/// Stored format of an expert bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertWeightFormat {
    /// e2m1 codes, two per byte with the even element in the low nibble, one e4m3 scale per 16
    /// inputs per row, and a per-expert factor `amax / outer_scale_denominator`.
    Nvfp4Amax,
    /// e4m3 values with one ue8m0 exponent per 32 inputs per row.
    Mxfp8Row32,
}

/// What one physical layer's expert GEMMs take as input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertActivationMode {
    /// The layer's declared quantizer: the token's row once before gate/up, and each routed row
    /// again before down.
    Declared,
    /// Float inputs against the exactly dequantized weights.
    Float,
}

/// Numerics of one physical layer's routed experts. The numeric class is set per layer, so a
/// per-layer measurement can move one layer without touching the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertNumericsPlan {
    pub physical_layer: u32,
    /// The stored format of both banks of the layer.
    pub weight: ExpertWeightFormat,
    /// The quantizer the layer's parameter group requires
    /// (`required_activation_quantization`).
    pub declared_activation: ActivationQuant,
    /// The input this plan runs the layer's experts on.
    pub activation: ExpertActivationMode,
}

/// Tensor classes outside the routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorClass {
    AttentionProjections,
    Router,
    RouterBias,
    LoopControllers,
    ShortConvs,
    Norms,
    ResidualQueries,
    Embedding,
    OutputProjection,
}

/// Numerics of one dense tensor class, from its parameter group: the weights run as stored (no
/// requantization) and the linear accumulates in f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseNumericsPlan {
    pub class: DenseTensorClass,
    /// The group's `compute_dtype`.
    pub weight: crate::quant_groups::ComputeDtype,
    /// The group declares an f32 forward output (`fwd_output_dtype`).
    pub output_f32: bool,
}

/// The numeric classes of the family. They are different functions of the same weights and are
/// never presented as one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inkling2NumericClass {
    /// The program the checkpoint declares as required: quantized expert inputs (NVFP4 W4A4,
    /// MXFP8 W8A8), BF16 elsewhere with f32 accumulation. The default and the support target.
    DeclaredActivationQuant,
    /// Float expert inputs against the exactly dequantized weights. A bring-up and oracle
    /// class: never the default, never a support claim.
    FloatActivation,
    /// Some layers run their declared quantizer and some run float inputs. A measurement
    /// class: it is neither of the two above and is never reported as either.
    PerLayerMixed,
}

/// Family-level numerics, carried on [`Inkling2Plan`].
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Numerics {
    /// The family's default class.
    pub default_class: Inkling2NumericClass,
    pub bring_up_class: Inkling2NumericClass,
    /// The class this plan runs, from its per-layer modes.
    pub class: Inkling2NumericClass,
    /// One entry per physical layer.
    pub experts: Vec<ExpertNumericsPlan>,
    /// One entry per dense tensor class.
    pub dense: Vec<DenseNumericsPlan>,
    /// Gate and up are one bank (`w13`) with interleaved output rows: row `2j` is gate `j`, row
    /// `2j + 1` is up `j`.
    pub gate_up_interleaved: bool,
    /// The routing weight multiplies the activation output before the down projection, so
    /// before the down input is quantized.
    pub routed_weight_before_down: bool,
    /// sha256 of the `hf_quant_config.json` the numerics were read from.
    pub quant_config_sha256: String,
}

/// What a drafter tap reads from the trunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterTap {
    /// The running residual stream on entry to `slot`: the partial sum of its block so far,
    /// before the depth mix and any norm.
    SlotInput { slot: u32 },
    /// The model's final hidden state after the output mix and norm.
    FinalNorm,
}

/// Head of the DSpark drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterHead {
    /// `markov_head_type: rnn_hidden`: a recurrent head of rank `rank` over the previous draft
    /// token, folded through the trunk's output projection. No confidence head.
    RnnHidden { rank: u32 },
}

/// The drafter shipped in `mtp.safetensors`, as declared. It is not the `DsparkPlan` program
/// (learned mask embedding, a fuse matrix over the taps, context K/V projections, a recurrent
/// head, no confidence head), and no executor implements it: the plan names it so the contract
/// can own its tensors and every drafting surface refuses it.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2DrafterPlan {
    pub blocks: u32,
    pub block_size: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub rope_base: f32,
    /// Blocks that see a window of context; the others see all of it.
    pub local_blocks: Vec<u32>,
    pub window: u32,
    pub head: DrafterHead,
    /// The checkpoint's `target_layer_ids`, which count layer visits in execution order.
    pub target_visit_ids: Vec<u32>,
    /// The same taps as cache slots.
    pub taps: Vec<DrafterTap>,
}

impl Inkling2Numerics {
    /// The class a set of per-layer modes amounts to.
    pub fn class_of(experts: &[ExpertNumericsPlan]) -> Inkling2NumericClass {
        let declared = experts
            .iter()
            .filter(|expert| expert.activation == ExpertActivationMode::Declared)
            .count();
        if declared == experts.len() {
            Inkling2NumericClass::DeclaredActivationQuant
        } else if declared == 0 {
            Inkling2NumericClass::FloatActivation
        } else {
            Inkling2NumericClass::PerLayerMixed
        }
    }
}

/// Program parts that belong to the whole model, carried on `ModelPlan::inkling2`.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Plan {
    /// RMS norm on token embeddings (`embed_norm`). Image embeddings would skip it.
    pub embedding_norm: NormPlan,
    pub loop_plan: LoopPlan,
    /// Short convolution on every MLP output.
    pub mlp_output_conv: ShortConvPlan,
    /// The output mix: one residual query over the block list and the final stream.
    pub output_queries: u32,
    pub numerics: Inkling2Numerics,
    pub drafter: Inkling2DrafterPlan,
    /// Vision tower tensors the contract owns and the text plan does not execute.
    pub vision_executed: bool,
}
