//! Typed variants of the DSpark drafter program (`DsparkPlan`).
//!
//! DeepSeek-V4 ships the first DSpark program: a noise-token block, the taps fused by one
//! projection, trunk-shaped draft blocks over their own KV rings, a rank-`r` bigram Markov
//! bias and a confidence head. Inkling 2 ships a second one (`mtp.safetensors`, the vendor's
//! `Inkling2DSparkModel`): an anchor row plus learned mask rows, a softmax fuse matrix over the
//! taps, bidirectional block attention over context K/V written once per committed position,
//! dense SwiGLU, and a recurrent (`rnn_hidden`) Markov head folded through the trunk's output
//! projection, with no confidence head. Each component is a variant here; an executor refuses
//! a variant it does not run.

/// How the block's rows enter the draft layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkRows {
    /// Row 0 embeds the input token, rows `1..block_size` embed `noise_token_id`, all through
    /// the shared trunk embedding (DeepSeek-V4).
    NoiseToken,
    /// Rows typed by a `DraftBlockPlan` (Inkling 2: the anchor through the trunk's normed
    /// embedding, then the learned mask vector).
    Block(DraftBlockPlan),
}

/// How the trunk's taps (`target_layer_ids`) become the draft's inputs.
#[derive(Debug, Clone, PartialEq)]
pub enum DsparkFusion {
    /// `main_x = main_norm(main_proj(concat(taps)))`, one input for every block (DeepSeek-V4).
    MainProjection,
    /// A softmax fuse matrix over the taps, one row per draft layer and one for the head, each
    /// mix RMS-normalized with its own weight (Inkling 2). `taps` types each tap.
    FuseWeights {
        fuse: DraftFusePlan,
        taps: Vec<DrafterTap>,
    },
}

/// The draft layers.
#[derive(Debug, Clone, PartialEq)]
pub enum DsparkLayers {
    /// `DsparkPlan::blocks`: full trunk-shaped blocks, each over its own KV ring (DeepSeek-V4).
    TrunkBlocks,
    /// Dense layers with bidirectional block attention over context K/V (Inkling 2).
    /// `DsparkPlan::blocks` is empty.
    Context(Box<DsparkContextLayers>),
}

/// The Inkling 2 draft layers: per layer `h = norm_in(x)`, `q, k, v = proj(h)`,
/// `q, k = rope(rms_head(q)), rope(rms_head(k))`, attention over context and block keys,
/// `x += o_proj(att)`, `x += down(silu(gate(norm_post(x))) * up(norm_post(x)))`; then
/// `hidden = norm(x)`.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkContextLayers {
    pub layers: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub hidden_size: u32,
    /// RMS epsilon of every draft norm.
    pub epsilon: f32,
    pub attention: DraftAttentionPlan,
    /// Dense SwiGLU of `intermediate_size`: `down(silu(gate(x)) * up(x))`, no softcap.
    pub mlp_swiglu: bool,
}

/// The Markov head, rank `DsparkPlan::markov_rank`.
#[derive(Debug, Clone, PartialEq)]
pub enum DsparkMarkov {
    /// `logits[v] += markov_w1[prev] . markov_w2[v]` (DeepSeek-V4).
    Bigram,
    /// The recurrent head over the previous draft token and the row hidden (Inkling 2).
    RnnHidden(DraftRnnHeadPlan),
}

/// The confidence head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkConfidence {
    /// `proj(concat(pre-norm hidden, markov_w1[prev]))`, one f32 per row (DeepSeek-V4).
    HiddenAndMarkovEmbedding,
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
