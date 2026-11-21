//! Inkling 2 program types, compiled only by the `inkling2` pack.
//!
//! Semantic sources: the checkpoint's `config.json` and `hf_quant_config.json`
//! (`thinkingmachines/Inkling-2-Mini`), the model code of the vendor's serving image, and two
//! float-activation implementations used for triage (the transformers loop model and llama.cpp's
//! `inkling2.cpp` / `inkling2-loop.cpp`). The lane note is
//! `research/modelplan-onboarding-inkling2mini-20261121/SEMANTICS.md`.
//!
//! None of these types parameterizes another family's plan. The reference executor
//! (memra-reference `inkling2`) runs every operation they emit except the drafter; no tuned
//! surface implements any of them, so each is a registry row with no surface.

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
    /// Learned depthwise taps `[channels, 1, kernel]`, a causal cross-correlation: indexed oldest
    /// input first, so the last tap weights the current token.
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
    /// MXFP8 W8A8) with the routed experts' bf16 rounding points, and an f32 stream outside the
    /// experts (BF16 and F32 weights as stored, f32 activations, f32 accumulation). The default
    /// and the support target. Running the stream outside the experts in BF16 is a different
    /// program and needs its own measured KL receipt before any route adopts it.
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
    ///
    /// READ, NOT PROVEN: the checkpoint does not state the direction; this is how the vendor's
    /// serving code and the transformers conversion read the bank. It is in the plan text (and
    /// so the plan hash) as the reading a test has to confirm against captured vectors, and the
    /// lane's SEMANTICS.md lists it as open.
    pub gate_up_interleaved: bool,
    /// The routing weight multiplies the activation output before the down projection, so
    /// before the down input is quantized. With quantized down inputs this placement changes
    /// the function, so it belongs to the numeric class.
    ///
    /// READ, NOT PROVEN: this is the vendor serving code's order for the declared class; the
    /// float-activation implementations multiply after the down projection. Same status as
    /// `gate_up_interleaved`.
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

/// What a draft block row is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftRowInput {
    /// The trunk's token embedding of the row's token, through the trunk's embedding norm.
    TrunkEmbeddingNormed,
    /// The learned `mask_embed` vector; no token id is embedded.
    LearnedMask,
}

/// One draft block: `rows` query rows at consecutive positions `P..P + rows`, `P` the position
/// of the anchor (the last committed token, which the trunk has not run yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftBlockPlan {
    pub rows: u32,
    /// Row 0.
    pub anchor: DraftRowInput,
    /// Rows 1..rows.
    pub mask: DraftRowInput,
    /// Rows that propose, each the token at its own position: rows 1..rows, so `rows - 1`
    /// proposals per block. The anchor row never proposes.
    pub proposals: u32,
}

/// The fuse matrix: `alpha = softmax(fuse_weight, over the taps)`, one row per draft layer and
/// a last row for the recurrent head. Each mix `sum_t alpha[row][t] * tap_t` is RMS-normalized
/// with its own weight (`dflare_norm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftFusePlan {
    /// `layers + 1`.
    pub rows: u32,
    /// `taps.len()`.
    pub taps: u32,
}

/// Rotary embedding of the draft: NeoX halves (dims `i` and `i + head_dim / 2` rotate
/// together), frequency `theta^(-2i / head_dim)`, angle `position * frequency` at the row's
/// absolute position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DraftRopePlan {
    pub theta: f32,
}

/// The draft layers' attention.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftAttentionPlan {
    /// Softmax scale `head_dim^-0.5`; no relative bias, output gate or sink.
    pub scale: f32,
    /// Per-head RMS norm (weight `q_norm`, `k_norm`) on the block's queries and keys, then
    /// RoPE; values are used raw.
    pub qk_norm: bool,
    pub rope: DraftRopePlan,
    /// Context keys and values: `k = rope(rms_head(k_proj_ctx(ctx)), k_norm_ctx)`,
    /// `v = v_proj_ctx(ctx)`, with `ctx` the layer's fuse mix of the committed token's taps.
    /// Written once per committed position; the block's own keys at `P..P + rows` are
    /// temporary.
    pub context_key_norm: bool,
    /// Every block row sees every block row (bidirectional) and every committed position
    /// before `P`, except on the local layers.
    pub bidirectional_block: bool,
    /// Layers that see only the `local_context` committed positions before `P` (plus the
    /// block): the keys at `P + rows - 1 - (local_context + rows - 1) ..= P + rows - 1`.
    pub local_layers: Vec<u32>,
    pub local_context: u32,
}

/// Inputs of the recurrent head's projection `rnn_proj`, in its column order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnInput {
    /// The recurrent state, `rank` wide, zero before the first proposal.
    State,
    /// `markov_w1[prev]`, `rank` wide: `prev` is the anchor for the first proposal, then
    /// the previous proposal.
    PreviousToken,
    /// The proposal row's head hidden (the final draft norm output), `hidden` wide.
    RowHidden,
    /// The head's fuse mix of the last committed token's taps (`f_rnn`), `hidden` wide.
    Feature,
}

/// Output chunks of `rnn_proj`, `rank` rows each, in row order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnnChunk {
    /// `g = sigmoid(gate)`.
    Gate,
    /// `c = tanh(candidate)`; `state' = g * state + (1 - g) * c`.
    Candidate,
    /// `latent = tanh(out)`, from the projection itself, not from the new state.
    Out,
}

/// The `rnn_hidden` recurrent Markov head, run once per proposal in order.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftRnnHeadPlan {
    pub rank: u32,
    pub inputs: [RnnInput; 4],
    pub chunks: [RnnChunk; 3],
    /// `bias = head @ hidden_out(latent)`, computed through the fold
    /// `fold_weight = head @ hidden_out.weight` (`[vocab, rank]`) and
    /// `fold_bias = head @ hidden_out.bias`, with `head` the trunk's output projection.
    pub folded_through_output: bool,
    /// `logits = softcap * tanh((base + bias) / softcap)` with `base = head @ row_hidden`;
    /// the trunk's final softcap, applied after the bias.
    pub softcap: f32,
}

/// The drafter shipped in `mtp.safetensors`. It is not the `DsparkPlan` program (learned mask
/// embedding, a fuse matrix over the taps, context K/V projections, a recurrent head, no
/// confidence head). The reference executor (memra-reference `inkling2_drafter`) runs it; its
/// numeric program is BF16 weights with f32 or f64 activations, and the proposals it makes
/// never change a committed token: verification runs the trunk's own program.
///
/// Per block: `x = rows inputs`; per layer `l`: `h = norm_in(x)`, `q, k, v = proj(h)`,
/// `q, k = rope(rms_head(q)), rope(rms_head(k))`, attention over context and block keys,
/// `x += o_proj(att)`, `x += down(silu(gate(norm_post(x))) * up(norm_post(x)))`; then
/// `hidden = norm(x)` and the recurrent head over rows `1..rows`.
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
    /// The checkpoint's `target_layer_ids`. They count layer visits in execution order; every
    /// id but the last taps the input of that visit, and the last is the visit count.
    pub target_visit_ids: Vec<u32>,
    /// The same taps typed: the input of the cache slot each visit runs, then the final hidden.
    pub taps: Vec<DrafterTap>,
    pub hidden_size: u32,
    /// RMS epsilon of every draft norm.
    pub epsilon: f32,
    pub block: DraftBlockPlan,
    pub fuse: DraftFusePlan,
    pub attention: DraftAttentionPlan,
    /// Dense SwiGLU of `intermediate_size`: `down(silu(gate(x)) * up(x))`, no softcap.
    pub mlp_swiglu: bool,
    pub rnn: DraftRnnHeadPlan,
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
