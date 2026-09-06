//! Arch-agnostic model configuration extracted from GGUF metadata.
//! One ModelConfig per loaded model; the forward pass reads it. Arch-specific
//! fields (SSM, MoE, MTP) are Option — present only for the arches that use them.

use crate::{GgufFile, MetaValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arch {
    Qwen3, // vanilla dense transformer
    Qwen3Moe,
    Qwen35, // hybrid: gated-deltanet linear-attn + periodic full-attn + MTP
    Qwen35Moe,
    Olmoe,     // dense full-attention + MoE FFN (no shared expert, no SSM, no MTP)
    MinimaxM3, // dense full-attention (MSA later) + MoE FFN: sigmoid router + shared expert,
    // gemma-norm, swigluoai, GQA 64/4 hd128 partial-RoPE, QK-norm
    Hy3,    // dense full-attention + MoE FFN: sigmoid router + bias + shared MLP, QK-norm
    Gemma4, // hybrid SWA(1024)/global 5:1, per-layer kv-heads+head_dim+rope, K=V globals,
    // 128-expert MoE + parallel shared FFN, gelu_tanh, softcap 30, layer_output_scale
    GlmDsa, // GLM-5/5.2: MLA attention (latent KV, MQA decode) + DSA sparse indexer +
    // deepseek-style MoE (sigmoid router + noaux_tc bias) + 1 NextN/MTP layer
    Step35, // StepFun Step-3.5/3.7-Flash: SWA(512) 3:1 + PER-LAYER q-head count (64 full /
    // 96 swa), head-wise attn gate (separate `attn_gate` tensor), dual rope base,
    // half-rotary on full layers, 288-expert sigmoid-router MoE + shared expert,
    // per-layer swiglu clamp arrays, 3 NextN/MTP blocks (shipped in a separate GGUF)
    DeepSeekV4, // DeepSeek-V4-Flash: MLA-lineage attention + per-layer KV compressor +
    // DSA lightning indexer (21 of 43 layers), sqrtsoftplus-scored 256-expert MoE with
    // 3 leading HASH-routed layers (tid2eid, still full expert banks — NOT dense FFN),
    // Sinkhorn hyper-connections (hc_*), 1 NextN/MTP layer. Loader lane only for now:
    // no forward pass — census-gated safetensors ingest (research/dsv4-flash-loader-20260818).
    Qwen4Exp, // Qwen3.8-Flash-Next (HF `qwen4_exp`): hybrid GDN 3:1 QSA (micro-block sparse
    // attention, MQA indexer 4Q/1K over 4-token blocks, budget 512 blocks), 512-expert
    // softmax top-k renormalized router (Qwen3NextTopKRouter — NOT sigmoid; the sigmoid
    // gate is on the shared expert only) + gated shared expert after EVERY layer,
    // 4-branch gated residual (rank-320 hyper-connections, 10240-wide stream), 20M-entry
    // n-gram/PLE embedding at layer 1 (0-based), fused-q attention gate, 1 NextN/MTP
    // full-attn block, ViT tower (research/qwen4exp-bringup-20260829).
    Glm5Next, // GLM-5.3-Flash: hybrid 34 KDA (Kimi Delta Attention) linear-attn + 11 MLA+DSA
    // sparse-attn layers, NoPE MLA (rope dim 0), k-pool-compressed indexer, sigmoid noaux_tc
    // 288-expert MoE + 1 shared, Sinkhorn hyper-connections (mHC), 1 NextN/MTP layer with its
    // own MLA attention
    Llama,
    Other(String),
}

impl Arch {
    pub fn parse(s: &str) -> Self {
        match s {
            "qwen3" => Arch::Qwen3,
            "qwen3moe" => Arch::Qwen3Moe,
            "qwen35" => Arch::Qwen35,
            "qwen35moe" => Arch::Qwen35Moe,
            // upstream llama.cpp writes the hybrid class as qwen3next (round 46: needed to
            // load public 27B GGUFs on the sm_90a board — same layer stack as qwen35).
            "qwen3next" => Arch::Qwen35,
            "qwen3nextmoe" => Arch::Qwen35Moe,
            "olmoe" => Arch::Olmoe,
            "minimax-m3" => Arch::MinimaxM3,
            "hy3" => Arch::Hy3,
            "gemma4" => Arch::Gemma4,
            "glm-dsa" => Arch::GlmDsa,
            // StepFun writes 3.5 AND 3.7-Flash under the same arch name (upstream llama.cpp
            // `step35`, PR #23845/#19283 — 3.7 is the 196B-A11B sibling of 3.5).
            "step35" => Arch::Step35,
            // No public GGUF writes this arch yet; the name is memra's own (safetensors-first).
            "deepseek-v4" => Arch::DeepSeekV4,
            // Qwen3.8-Flash-Next. Upstream llama.cpp has GGUFs for it, but memra's lane is
            // safetensors-first; the GGUF arch string is adopted when that entry lands.
            "qwen4exp" => Arch::Qwen4Exp,
            // No public GGUF writes this arch yet; the name is memra's own (safetensors-first).
            "glm5-next" => Arch::Glm5Next,
            "llama" => Arch::Llama,
            other => Arch::Other(other.to_string()),
        }
    }

    /// Map an HF `model_type` (config.json) to the ggml-style Arch. HF uses different strings
    /// than GGUF (`qwen3_moe` vs `qwen3moe`, `qwen3_5` vs `qwen35`), so normalize first.
    pub fn from_hf_model_type(mt: &str) -> Self {
        let ggml = match mt {
            "qwen3" => "qwen3",
            "qwen3_moe" => "qwen3moe",
            "qwen3_5" | "qwen3_5_text" | "qwen3_next" => "qwen35",
            "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next_moe" => "qwen35moe",
            "olmoe" => "olmoe",
            // MiniMax-M3 (incl the VL wrapper model_type; text_config flattening handles the rest)
            "minimax_m3" | "minimax_m3_vl" | "minimax_m3_text" => "minimax-m3",
            "hy_v3" | "hy3" => "hy3",
            // Step-3.7-Flash keeps the Step-3.5 text architecture/model_type spelling in the
            // official HF checkpoint (the outer VLM wrapper is `step3p7`). Both are the same
            // `step35` execution architecture used by the GGUF path.
            "step3p5" | "step3p7" => "step35",
            // GLM-5/5.2 (HF `GlmMoeDsaForCausalLM`, model_type `glm_moe_dsa`)
            "glm_moe_dsa" => "glm-dsa",
            // DeepSeek-V4-Flash (HF `DeepseekV4ForCausalLM`, model_type `deepseek_v4`)
            "deepseek_v4" => "deepseek-v4",
            // Qwen3.8-Flash-Next (HF `Qwen4ExpForConditionalGeneration`, model_type
            // `qwen4_exp`; the text_config carries `qwen4_exp_text`)
            "qwen4_exp" | "qwen4_exp_text" => "qwen4exp",
            // GLM-5.3-Flash (HF `Glm5NextForConditionalGeneration`): the VL wrapper model_type
            // is `glm5_next`, text_config's is `glm5_next_text` — same text architecture.
            "glm5_next" | "glm5_next_text" => "glm5-next",
            "gemma4" | "gemma4_text" => "gemma4",
            // Mistral dense (MistralForCausalLM) is the llama execution program: RMSNorm,
            // GQA full attention, rope over the whole head, SwiGLU, no QK-norm, no biases.
            "llama" | "mistral" => "llama",
            other => other,
        };
        Arch::parse(ggml)
    }
    /// MiniMax-M3: sigmoid router (+e_score_correction_bias), gemma-norm, swigluoai clamp,
    /// Mixtral-style expert tensor names. Full attention v0 (MSA is bit-exact-degenerate <=2048
    /// ctx — the sparse indexer selects everything; the MSA kernel is a later arc).
    pub fn is_minimax(&self) -> bool {
        matches!(self, Arch::MinimaxM3)
    }
    /// Tencent HunYuan/Hunyuan Hy3 (`hy_v3` in HF config.json).
    pub fn is_hy3(&self) -> bool {
        matches!(self, Arch::Hy3)
    }
    /// StepFun Step-3.5 / Step-3.7-Flash (GGUF arch `step35`).
    pub fn is_step35(&self) -> bool {
        matches!(self, Arch::Step35)
    }
    /// The attention output-gate LAYOUT this architecture declares. `None` = the architecture is
    /// not registered here at all, and NO layout may be assumed for it.
    ///
    /// Exhaustive on purpose: adding an `Arch` variant must fail to compile until bring-up
    /// declares its gate layout.
    ///
    /// Why this exists (2026-08-19). `ModelConfig::attn_out_gate()` used to answer the fused-gate
    /// question with a five-arch DENY-LIST plus a permissive `true` fallback — "not m3, not hy3,
    /// not gemma4, not mla, not step35, therefore qwen3.5 FusedQ". A new arm inherited `FusedQ`
    /// by default, and `q_gate_split` then reads 2x past the end of an `attn_q.weight` whose
    /// out-features are `n_head*head_dim` because the arm's gate is a SEPARATE tensor. A deny-list
    /// is the wrong default direction for a hazard whose failure mode is an out-of-bounds read:
    /// the safe answer for an undeclared arch is "no fused gate", and the only way to get `FusedQ`
    /// is to ask for it by name.
    ///
    /// This is the whole-model declaration. Migrated architectures additionally carry a PER-LAYER
    /// `AttentionGateKind` in `ArchGeometryTable`, which wins wherever it exists.
    pub fn attention_gate_kind(&self) -> Option<AttentionGateKind> {
        match self {
            // qwen3.5 packs [q|gate] per head inside `attn_q.weight` (out = 2*n_head*head_dim).
            Arch::Qwen35 | Arch::Qwen35Moe => Some(AttentionGateKind::FusedQ),
            // qwen4_exp keeps the fused layout: q_proj is [2*24*256, 2560] against 24 heads
            // of 256 (census 2026-08-29), sigmoid output gate per config.
            Arch::Qwen4Exp => Some(AttentionGateKind::FusedQ),
            // step35 projects one sigmoid scalar per head through a separate `attn_gate.weight`.
            Arch::Step35 => Some(AttentionGateKind::SeparateHead),
            // No attention output gate: wq out is exactly n_head*head_dim on all of these.
            Arch::MinimaxM3
            | Arch::Hy3
            | Arch::Gemma4
            | Arch::GlmDsa
            | Arch::Qwen3
            | Arch::Qwen3Moe
            | Arch::Olmoe
            | Arch::Llama => Some(AttentionGateKind::None),
            // DeepSeek-V4-Flash rides its own dsv4 lane (loader/bring-up), never the hybrid
            // q_gate_split path; its attn_q out-features carry no fused gate — declared None
            // like the other DSA-family arch (GlmDsa) rather than left to a fallback.
            Arch::DeepSeekV4 => Some(AttentionGateKind::None),
            // GLM-5.3-Flash MLA layers carry no fused q gate; KDA layers gate inside the mixer
            // via separate g_a_proj/g_b_proj tensors, not the attention q layout.
            Arch::Glm5Next => Some(AttentionGateKind::None),
            // An arch string we have never seen. Declares nothing, and callers must refuse
            // rather than pick a layout — see `ModelConfig::validate_attention_gate_layout`.
            Arch::Other(_) => None,
        }
    }

    /// DeepSeek-V4-Flash (HF `deepseek_v4`) — safetensors-first, loader lane only today.
    pub fn is_dsv4(&self) -> bool {
        matches!(self, Arch::DeepSeekV4)
    }

    /// Qwen3.8-Flash-Next (HF `qwen4_exp`) — safetensors-first, loader lane only today.
    pub fn is_qwen4exp(&self) -> bool {
        matches!(self, Arch::Qwen4Exp)
    }

    /// GLM-5.3-Flash (HF `glm5_next`/`glm5_next_text`) — safetensors-first bring-up.
    pub fn is_glm5_next(&self) -> bool {
        matches!(self, Arch::Glm5Next)
    }
}

/// What kind of token-mixing a given layer performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,   // softmax attention with growing KV cache
    LinearAttention, // gated-deltanet / SSM with fixed recurrent state
}

/// How an attention layer gates its output before the output projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionGateKind {
    None,
    /// Qwen3.5 packs a per-dimension sigmoid gate beside Q in `attn_q.weight`.
    FusedQ,
    /// Step35 projects one sigmoid scalar per head through `attn_gate.weight`.
    SeparateHead,
}

/// A model's architecture declares no attention output-gate layout, so no layout may be assumed.
/// Returned by `ModelConfig::validate_attention_gate_layout` and raised by the hybrid loader
/// before any tensor is split — see `Arch::attention_gate_kind`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndeclaredGateLayout {
    pub arch: String,
}

impl std::fmt::Display for UndeclaredGateLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "architecture {} declares no attention output-gate layout — refusing to load rather \
             than assume one. Register it in Arch::attention_gate_kind() (None / FusedQ / \
             SeparateHead) as part of bring-up; guessing FusedQ makes q_gate_split read 2x past \
             the end of attn_q.weight.",
            self.arch
        )
    }
}

impl std::error::Error for UndeclaredGateLayout {}

/// A fused `[q|gate]` split is about to read from a `wq` output that is not wide enough for it.
///
/// `q_gate_split` reads `2 * head_dim * n_head * t` floats from `qf`: q at `stride*hh + d`, gate at
/// `stride*hh + head_dim + d` with `stride = 2*head_dim`. On a checkpoint whose gate is a SEPARATE
/// tensor, `wq` produces only `n_head*head_dim` per token, so the read runs 2x off the end. That
/// is an out-of-bounds device read — no panic, no error, just whatever memory follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedQGateExtent {
    /// Elements the split will read.
    pub need: usize,
    /// Elements actually present in the wq output buffer.
    pub have: usize,
    pub head_dim: usize,
    pub n_head: usize,
    pub t: usize,
}

impl std::fmt::Display for FusedQGateExtent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fused [q|gate] split needs {} elements (2 * head_dim {} * n_head {} * T {}) but the \
             wq output holds {}. This layer's attn_q.weight carries NO fused gate — its \
             out-features are n_head*head_dim. Either the arch's AttentionGateKind is wrong \
             (FusedQ declared for a separate-gate or ungated checkpoint) or the geometry is.",
            self.need, self.head_dim, self.n_head, self.t, self.have
        )
    }
}

impl std::error::Error for FusedQGateExtent {}

/// Bounds contract for the fused `[q|gate]` split, checked at the read site.
///
/// Pure and arch-agnostic so it is unit-testable without a device: `q_gate_split` calls it with
/// `qf.len()` before the launch, and a layout mismatch becomes a typed error instead of an
/// out-of-bounds read.
pub fn check_fused_q_gate_extent(
    qf_len: usize,
    head_dim: usize,
    n_head: usize,
    t: usize,
) -> Result<(), FusedQGateExtent> {
    let need = 2 * head_dim * n_head * t;
    if qf_len >= need {
        Ok(())
    } else {
        Err(FusedQGateExtent {
            need,
            have: qf_len,
            head_dim,
            n_head,
            t,
        })
    }
}

/// Rotary width (`n_rot`) — ONE derivation, shared by every loader path.
///
/// The two readers see the same fact spelled three different ways, and they must not each
/// invent their own precedence for it:
///
/// * **GGUF** ships the answer pre-baked as `rope.dimension_count` (`explicit_dims`). The
///   converter already did the arithmetic.
/// * **HF `config.json`** ships either an explicit dim count (`rotary_dim` — the MiniMax-M3
///   spelling) *or* a FRACTION of `head_dim` (`partial_rotary_factor` — the Qwen3.5 family
///   spelling, present both top level and under `rope_parameters`), and the loader has to do
///   the multiply itself.
/// * **Absent from both** means full rope: every head dim rotates.
///
/// Precedence is explicit-dims > fraction > full width, because an explicit count is the more
/// specific declaration and is what a converter writes once it has resolved the fraction.
///
/// Why this is a function and not two `unwrap_or(head_dim_k)` expressions: the HF reader used
/// to honour only `rotary_dim`, so every `qwen35`/`qwen35moe` safetensors checkpoint — all of
/// which declare `partial_rotary_factor: 0.25` and none of which declare `rotary_dim` — got
/// `n_rot = head_dim = 256` instead of 64. Full rope over all 256 dims where 192 must pass
/// through unrotated: no shape mismatch, no error, fluent output, wrecked long context. The
/// GGUF twin of the same model was correct the whole time (`rope.dimension_count = 64`), which
/// is exactly the failure mode two independent implementations of one derivation produce.
///
/// `.max(2)` mirrors `Eagle3Draft::load`: rope consumes dim PAIRS, so a width of 0 or 1 is not
/// a representable rotation. A degenerate fraction is clamped rather than silently disabling
/// rope (which would look like a plausible model and be wrong).
pub fn resolve_rope_dim_count(
    explicit_dims: Option<u32>,
    partial_rotary_factor: Option<f32>,
    head_dim_k: u32,
) -> u32 {
    if let Some(dims) = explicit_dims {
        return dims;
    }
    match partial_rotary_factor {
        // Only a factor strictly inside (0, 1) truncates. 1.0 is full rope, and 0 / negative /
        // >1 are malformed — all four take the full width rather than a silently odd rotation.
        Some(factor) if factor > 0.0 && factor < 1.0 => {
            let dims = (factor * head_dim_k as f32).round() as u32;
            dims.clamp(2, head_dim_k)
        }
        _ => head_dim_k,
    }
}

/// Complete attention geometry for one architecture-defined layer class.
///
/// The table is intentionally small: it holds values that execution arms otherwise reconstruct
/// independently. Architecture-specific math, tensor names, clamps, and routing stay in their
/// existing configs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerGeometry {
    pub mixer: LayerKind,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_rot: u32,
    pub rope_base: f32,
    pub window: Option<u32>,
    pub rope_factors: bool,
    pub attention_gate: AttentionGateKind,
}

impl LayerGeometry {
    pub fn attention_scale(self) -> f32 {
        1.0 / (self.head_dim_k as f32).sqrt()
    }
}

/// Declarative per-architecture geometry: a compact class table plus one class id per layer.
///
/// Qwen3.5 and Step35 are the first migrated architectures. Other architectures keep their
/// existing scalar/per-arch config paths until they are deliberately migrated.
#[derive(Debug, Clone)]
pub struct ArchGeometryTable {
    classes: Vec<LayerGeometry>,
    layer_classes: Vec<u16>,
}

impl ArchGeometryTable {
    /// Interval-hybrid class table: (il+1) % full_attention_interval == 0 => full attention
    /// with the FusedQ gate, else GDN linear; MTP tail layers are full. Shared by qwen35 and
    /// qwen4_exp — qwen4_exp matches the rule exactly (interval 4, fused [q|gate] q_proj:
    /// 24 heads x 256 x 2 = q_out 12288 measured, census 2026-08-29).
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn qwen35(
        n_layer: u32,
        nextn: u32,
        full_attention_interval: u32,
        n_head: u32,
        n_head_kv: u32,
        head_dim_k: u32,
        head_dim_v: u32,
        n_rot: u32,
        rope_base: f32,
    ) -> Self {
        let linear = LayerGeometry {
            mixer: LayerKind::LinearAttention,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_rot,
            rope_base,
            window: None,
            rope_factors: false,
            attention_gate: AttentionGateKind::None,
        };
        let full = LayerGeometry {
            mixer: LayerKind::FullAttention,
            attention_gate: AttentionGateKind::FusedQ,
            ..linear
        };
        let n_trunk = n_layer.saturating_sub(nextn);
        let layer_classes = (0..n_layer)
            .map(|il| {
                let full_layer = il >= n_trunk
                    || full_attention_interval == 0
                    || (il + 1) % full_attention_interval == 0;
                if full_layer { 1 } else { 0 }
            })
            .collect();
        Self {
            classes: vec![linear, full],
            layer_classes,
        }
    }

    fn step35(n_layer: u32, head_dim_k: u32, head_dim_v: u32, step35: &Step35Config) -> Self {
        let mut classes = Vec::new();
        let mut layer_classes = Vec::with_capacity(n_layer as usize);
        for il in 0..n_layer {
            let swa = step35.is_swa(il);
            let geometry = LayerGeometry {
                mixer: LayerKind::FullAttention,
                n_head: step35.n_head(il),
                n_head_kv: step35.n_head_kv(il),
                head_dim_k,
                head_dim_v,
                n_rot: step35.n_rot(il),
                rope_base: step35.rope_base(il),
                window: swa.then_some(step35.sliding_window),
                rope_factors: !swa,
                attention_gate: AttentionGateKind::SeparateHead,
            };
            let class = match classes.iter().position(|candidate| *candidate == geometry) {
                Some(class) => class,
                None => {
                    classes.push(geometry);
                    classes.len() - 1
                }
            };
            assert!(
                class <= u16::MAX as usize,
                "too many architecture geometry classes"
            );
            layer_classes.push(class as u16);
        }
        Self {
            classes,
            layer_classes,
        }
    }

    pub fn classes(&self) -> &[LayerGeometry] {
        &self.classes
    }

    pub fn layer_classes(&self) -> &[u16] {
        &self.layer_classes
    }

    pub fn layer(&self, il: u32) -> Option<LayerGeometry> {
        let class = *self.layer_classes.get(il as usize)? as usize;
        self.classes.get(class).copied()
    }
}

#[derive(Debug, Clone)]
pub struct SsmConfig {
    pub conv_kernel: u32,
    pub inner_size: u32,
    pub state_size: u32,
    pub time_step_rank: u32,
    pub group_count: u32,
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub expert_shared_ff_length: u32, // NEW: qwen35moe.expert_shared_feed_forward_length = 512
}

/// MiniMax-M3-specific forward-pass knobs (config.json, minimax_m3_vl text_config).
#[derive(Debug, Clone)]
pub struct M3Config {
    pub use_gemma_norm: bool,   // (1+w) RMSNorm — folded into weights at load
    pub sigmoid_routing: bool,  // scoring_func == "sigmoid" (DeepSeek-V3 style)
    pub use_routing_bias: bool, // e_score_correction_bias on SELECTION only
    pub routed_scaling_factor: f32, // 2.0 — multiplies the normalized routing weights
    pub n_shared_experts: u32,  // 1
    pub swiglu_alpha: f32,      // swigluoai: gate*sigmoid(alpha*gate), clamp at limit
    pub swiglu_limit: f32,      // 7.0
    pub rotary_dim: u32,        // partial RoPE (64 of head_dim 128)
    pub dense_intermediate_size: u32, // dense-FFN layers' n_ff (12288)
    pub moe_layer_freq: Vec<u32>, // per-layer 0=dense 1=moe (len == n_layer)
}

/// Hy3-specific loader metadata. Forward/kernel support is a later GPU-gated lane; these fields
/// let the CPU-side loader distinguish REAP's dense layer 0 from routed layers 1..79 and preserve
/// the routing contract documented in the port dossier.
#[derive(Debug, Clone)]
pub struct Hy3Config {
    pub sigmoid_routing: bool,
    pub use_routing_bias: bool,
    pub route_norm: bool,
    pub router_scaling_factor: f32,
    pub n_shared_experts: u32,
    pub first_k_dense_replace: u32,
    pub qk_norm: bool,
    pub hidden_act: String,
    /// ModelOpt's explicit weight-only NVFP4 contract. When true, routed experts must retain
    /// floating-point activations; the q8_1 expert kernels are a different W4A8 numeric program.
    pub weight_only_nvfp4: bool,
}

/// Gemma-4 per-layer attention geometry + block extras (P0 census 2026-07-10).
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub head_count_kv: Vec<u32>, // per layer (8 SWA / 2 global on the 26B)
    pub swa_pattern: Vec<bool>,  // true = sliding-window layer
    pub sliding_window: u32,     // 1024
    pub key_length_global: u32,  // 512
    pub key_length_swa: u32,     // 256
    pub rope_base_global: f32,   // 1e6 (+ rope_freqs.weight factors tensor)
    pub rope_base_swa: f32,      // 1e4
    pub rope_dims_global: u32,   // 512 metadata (p-RoPE partial applies via rope_freqs)
    pub rope_dims_swa: u32,      // 256
    pub final_logit_softcapping: f32, // 30.0
    /// rope_parameters.full_attention.partial_rotary_factor (HF; 0.25 on the 31B).
    /// GGUF ships the derived rope_freqs tensor and never reads this; the native
    /// safetensors path synthesizes rope_freqs from it (hybrid.rs).
    pub partial_rotary_global: f32,
    // ---- E4B (per-layer-embedding + KV-sharing variant; 0 on 26B/31B) ----
    /// n_embd_per_layer (E4B: 256). 0 = no per-layer-embedding machinery.
    pub n_embd_per_layer: u32,
    /// trailing layers WITHOUT own KV (E4B: 18); they attend an earlier layer's cache:
    /// il >= n_layer - shared_kv_layers reads layer (n_layer - shared_kv_layers) - (swa ? 2 : 1).
    pub shared_kv_layers: u32,
    /// tokenizer.ggml.suppress_tokens — ids the model card forbids at sampling (the 12B QAT
    /// ships two control ids); empty on 26B/31B/E4B. Masked to -inf before every argmax/sample.
    pub suppress_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub context_length: u32,
    pub patch_size: u32,
    pub position_embedding_size: u32,
    pub position_axes: u32,
    pub pooling_kernel_size: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub activation: String,
    pub standardize: bool,
    pub clipped_linears: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultimodalConfig {
    pub image_token_id: u32,
    pub vision_soft_tokens_per_image: u32,
}

/// glm5_next vision tower config (`vision_config.model_type == "glm5_next_vision"`).
/// A DIFFERENT semantic program from the factored-additive `VisionConfig` tower (gemma-4):
/// fused-qkv ViT with per-head q/k RMS norms, 2D rope (no learned position table — upstream
/// `GlmOcrVisionModel.__init__` deletes `self.embeddings`), clamped-SwiGLU MLP with biases,
/// conv 2x2 downsample into a gated clamped merger. Field truth: banked
/// `research/glm53-flash-bringup-20260827/glm-config.json` + transformers 5.16.1
/// `models/glm5_next/modeling_glm5_next.py` (vision classes diffed identical to main).
#[derive(Debug, Clone, PartialEq)]
pub struct Glm5VisionConfig {
    pub depth: u32,                        // 24
    pub hidden_size: u32,                  // 1024
    pub num_heads: u32,                    // 16 (head_dim = hidden/heads = 64)
    pub intermediate_size: u32,            // 4096 (block MLP)
    pub patch_size: u32,                   // 14
    pub temporal_patch_size: u32,          // 2 (image frames duplicated by the processor)
    pub spatial_merge_size: u32,           // 2 (downsample conv kernel AND token merge)
    pub out_hidden_size: u32,              // 4096 == trunk n_embd (validated at plan compile)
    pub projection_intermediate_size: u32, // 10240 (merger gate/up width)
    pub swiglu_limit: f32,                 // 10.0 (gate max-clamp; up +/- clamp)
    pub rms_norm_eps: f32,                 // 1e-5
    pub in_channels: u32,                  // 3
    pub attention_bias: bool,              // true: qkv/proj AND block-MLP linears carry bias
    pub hidden_act: String,                // "silu"
    // ---- mm token splice ids (top-level config.json keys) ----
    pub image_token_id: u32, // 154854 <|image|> (placeholder the tower rows replace)
    pub video_token_id: u32, // 154855 <|video|>
    pub image_start_token_id: u32, // 154830 <|begin_of_image|>
    pub image_end_token_id: u32, // 154831 <|end_of_image|>
    pub video_start_token_id: u32, // 154832 <|begin_of_video|>
    pub video_end_token_id: u32, // 154833 <|end_of_video|>
}

/// StepFun Step-3.5/3.7-Flash (`step35`) per-layer geometry + block extras. Values in comments are
/// the 3.7-Flash 196B-A11B artifact (official IQ4_XS GGUF header, receipt
/// `research/step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt`).
///
/// Reference semantics: upstream llama.cpp `src/models/step35.cpp` (PR #23845, merged 2026-06-02)
/// + `llama-hparams.cpp` `n_rot()`/`is_swa()`. Three things make this arch different from every
///   arch memra already loads, and all three are per-LAYER:
///   1. `n_head` is an ARRAY (64 on full-attn layers, 96 on SWA layers) — KV heads are uniform 8,
///      so KV geometry is unaffected, but wq/wo/attn_gate out-features vary per layer.
///   2. RoPE: dual base (5e6 full / 1e4 SWA) AND half-rotary on the FULL layers only
///      (`n_rot_full = n_rot_full/2` = 64 of head_dim 128; SWA keeps the full 128).
///      `rope_freqs.weight` (llama3 factors) applies to the FULL layers only — SWA passes null.
///   3. The head-wise attention gate is a SEPARATE tensor `blk.N.attn_gate.weight [n_embd, n_head]`
///      producing ONE sigmoid scalar per head (broadcast over head_dim), NOT the qwen35 form where
///      the gate is fused into wq as a per-dim block. `ModelConfig::attn_out_gate()` must be false
///      for this arch or the wq split reads 2x out of bounds.
#[derive(Debug, Clone)]
pub struct Step35Config {
    /// Per-layer query-head count — `step35.attention.head_count` is an ARRAY (45 items: 64 on
    /// full-attn layers, 96 on SWA layers). len == n_layer_total (the MTP GGUF carries 48).
    pub head_count: Vec<u32>,
    /// Per-layer KV-head count (`attention.head_count_kv`, uniform 8 on 3.7 — kept as an array
    /// because the key IS an array in the artifact and a future sibling may vary it).
    pub head_count_kv: Vec<u32>,
    /// `attention.sliding_window_pattern` [bool; n_layer]: true = sliding-window layer.
    /// 3.7-Flash is 3:1 — [false,true,true,true] repeating = 12 full (il%4==0) + 33 SWA.
    pub swa_pattern: Vec<bool>,
    pub sliding_window: u32,   // attention.sliding_window = 512
    pub rope_base_global: f32, // rope.freq_base = 5e6 (full-attn layers)
    pub rope_base_swa: f32,    // rope.freq_base_swa = 1e4 (SWA layers)
    /// Rotary dims on FULL-attn layers = head_dim_k/2 (64). Upstream halves `n_rot_full` in
    /// `load_arch_hparams` AFTER the generic loader defaults it to `n_embd_head_k` (128).
    pub rope_dims_full: u32,
    /// Rotary dims on SWA layers = head_dim_k (128, unhalved — `n_rot_swa` is copied from
    /// `n_rot_full` BEFORE the arch hook halves it, so SWA keeps the full width).
    pub rope_dims_swa: u32,
    /// Llama-3-style per-frequency divisors synthesized from an HF `rope_scaling` object.
    /// GGUF sources carry the equivalent values in `rope_freqs.weight`, so this is `None`
    /// for that source class. Only full-attention layers consume the factors.
    pub rope_freq_factors: Option<Vec<f32>>,
    /// `swiglu_clamp_exp` [f32; n_layer] — routed-expert SwiGLU clamp limit per layer.
    /// Nonzero only on layers 43-44 of 3.7-Flash. Semantics (llama-graph.cpp:2146): the limit
    /// applies when > 1e-6 as `up = clamp(up, -L, L); act = min(silu(gate), L); out = act * up`.
    pub swiglu_clamp_exp: Vec<f32>,
    /// `swiglu_clamp_shexp` [f32; n_layer] — same for the shared expert (llama-graph.cpp:1751).
    pub swiglu_clamp_shexp: Vec<f32>,
    // ---- MoE (deepseek-V3-class sigmoid router; the Hy3/M3/glm-dsa recipe verbatim) ----
    pub sigmoid_routing: bool, // expert_gating_func == 2; ABSENT defaults to sigmoid (BC)
    pub routed_scaling_factor: f32, // expert_weights_scale = 3.0
    pub route_norm: bool,      // expert_weights_norm = true
    pub first_k_dense_replace: u32, // leading_dense_block_count = 3
}

impl Step35Config {
    /// True when layer `il` is a sliding-window layer. Out-of-range indices (the MTP blocks of a
    /// trunk-only GGUF) fall back to `true`: upstream's `is_swa_impl` array covers n_layer_all and
    /// every 3.7 MTP block is SWA-type (blocks 45/46/47, none at il%4==0).
    pub fn is_swa(&self, il: u32) -> bool {
        self.swa_pattern.get(il as usize).copied().unwrap_or(true)
    }
    /// Query-head count for layer `il` (64 full / 96 SWA on 3.7-Flash).
    pub fn n_head(&self, il: u32) -> u32 {
        self.head_count
            .get(il as usize)
            .copied()
            .or_else(|| self.head_count.last().copied())
            .expect("step35: attention.head_count array is empty")
    }
    /// KV-head count for layer `il` (uniform 8 on 3.7-Flash).
    pub fn n_head_kv(&self, il: u32) -> u32 {
        self.head_count_kv
            .get(il as usize)
            .copied()
            .or_else(|| self.head_count_kv.last().copied())
            .expect("step35: attention.head_count_kv array is empty")
    }
    /// Rotary width for layer `il` — upstream `llama_hparams::n_rot(il)`:
    /// `is_swa(il) ? n_rot_swa : n_rot_full` (128 SWA / 64 full on 3.7-Flash).
    pub fn n_rot(&self, il: u32) -> u32 {
        if self.is_swa(il) {
            self.rope_dims_swa
        } else {
            self.rope_dims_full
        }
    }
    /// RoPE base for layer `il` (1e4 SWA / 5e6 full).
    pub fn rope_base(&self, il: u32) -> f32 {
        if self.is_swa(il) {
            self.rope_base_swa
        } else {
            self.rope_base_global
        }
    }
    /// Routed-expert SwiGLU clamp for layer `il`, `None` when unset/<=eps (upstream uses a 1e-6
    /// epsilon, not != 0.0 — a tiny nonzero limit must not silently clamp everything to ~0).
    pub fn clamp_exp(&self, il: u32) -> Option<f32> {
        self.swiglu_clamp_exp
            .get(il as usize)
            .copied()
            .filter(|&l| l > 1e-6)
    }
    /// Shared-expert SwiGLU clamp for layer `il`.
    pub fn clamp_shexp(&self, il: u32) -> Option<f32> {
        self.swiglu_clamp_shexp
            .get(il as usize)
            .copied()
            .filter(|&l| l > 1e-6)
    }
    /// Count of full-attention (non-SWA) layers over the trunk — the layers whose KV cache grows
    /// unbounded with context. 12 on 3.7-Flash; the KV-budget arithmetic keys off this.
    pub fn n_full_attn(&self, n_trunk: u32) -> u32 {
        (0..n_trunk).filter(|&il| !self.is_swa(il)).count() as u32
    }
}

/// DeepSeek-V4-Flash (`deepseek_v4`) config. Values in comments are the 0731 Flash NVFP4
/// artifact (nvidia cast; census + geometry receipts in research/dsv4-flash-loader-20260818).
/// V3/GLM ancestry INFORMS these fields but proves nothing (no-generic-support law): the
/// scoring program (`sqrtsoftplus`), the hash-routed leading layers, the compressor/indexer
/// split, and the Sinkhorn hyper-connections are all V4's own semantic program.
///
/// Every field here is REQUIRED by the eventual forward pass; `from_hf` panics with the field
/// name when config.json omits one (loud refusal over a silently-defaulted different program).
#[derive(Debug, Clone)]
pub struct DeepSeekV4Config {
    // ---- MoE routing ----
    pub scoring_func: String, // "sqrtsoftplus" — NEW (the V3-class engines score sigmoid/softmax)
    pub topk_method: String,  // "noaux_tc"
    pub routed_scaling_factor: f32, // 1.5
    pub norm_topk_prob: bool, // true
    pub n_shared_experts: u32, // 1
    /// Leading layers routed by token-id table: `ffn.gate.tid2eid [n_vocab, top_k] I64` replaces
    /// score-based selection and `ffn.gate.bias` is ABSENT there. These layers still carry the
    /// full routed-expert bank (256 experts, measured) — they are hash-routed MoE, NOT dense FFN.
    pub num_hash_layers: u32, // 3 (layers 0..2 measured on the artifact)
    // ---- Sinkhorn hyper-connections (manifold-constrained; new engine work, no existing arm) ----
    pub hc_eps: f32,            // 1e-6
    pub hc_mult: u32,           // 4 (hc fn width = hc_mult * hidden; see dsv4.rs shape math)
    pub hc_sinkhorn_iters: u32, // 20
    // ---- attention: MLA lineage + per-layer KV compressor + DSA lightning indexer ----
    pub head_dim: u32, // 512 (latent head dim; wkv out = num_key_value_heads * head_dim)
    pub num_key_value_heads: u32, // 1
    pub q_lora_rank: u32, // 1024
    pub qk_rope_head_dim: u32, // 64
    pub o_lora_rank: u32, // 1024 (wo_a out = o_groups * o_lora_rank)
    pub o_groups: u32, // 8
    pub index_n_heads: u32, // 64
    pub index_head_dim: u32, // 128
    pub index_topk: u32, // 512
    /// Per-layer compression ratio, len == n_layer + nextn (43 + 1 on Flash; the LAST entries
    /// are the MTP layer(s)). Measured presence rules on the artifact:
    ///   0   -> no compressor, no indexer (layers 0, 1 and the MTP layer)
    ///   4   -> compressor + lightning indexer (21 layers: even il in 2..=42)
    ///   128 -> compressor only (20 layers: odd il in 3..=41)
    /// The 41/21 splits are DERIVED from this array — never hardcode the counts.
    pub compress_ratios: Vec<u32>,
    pub compress_rope_theta: f32, // 160000
    pub sliding_window: u32,      // 128
    // ---- activation / rope ----
    pub swiglu_limit: f32,        // 10.0
    pub rope_yarn_factor: f32,    // 16 (rope_scaling.type == "yarn")
    pub rope_yarn_orig_ctx: u32,  // 65536 (original_max_position_embeddings)
    pub rope_yarn_beta_fast: f32, // 32
    pub rope_yarn_beta_slow: f32, // 1
}

impl DeepSeekV4Config {
    /// Compression ratio for layer `il` (trunk 0..n_layer, then the MTP layer(s)). Out-of-range
    /// is a caller bug — the array length is pinned to n_layer + nextn at parse.
    pub fn compress_ratio(&self, il: u32) -> u32 {
        self.compress_ratios[il as usize]
    }
    /// Layer carries `attn.compressor.*` tensors (wkv/wgate/norm/ape) iff its ratio != 0.
    pub fn has_compressor(&self, il: u32) -> bool {
        self.compress_ratio(il) != 0
    }
    /// Layer carries `attn.indexer.*` tensors iff its ratio == 4. Measured coincidence on the
    /// Flash artifact (indexer on exactly the ratio-4 layers); a sibling with other ratio values
    /// re-verifies this rule through the census gate, which refuses loudly on any drift.
    pub fn has_indexer(&self, il: u32) -> bool {
        self.compress_ratio(il) == 4
    }
    /// Layer routes by the token-id table (`tid2eid`, no `gate.bias`). The leading-layers rule
    /// mirrors first_k_dense_replace conventions and is census-verified against the artifact.
    pub fn is_hash_layer(&self, il: u32) -> bool {
        il < self.num_hash_layers
    }
}

/// Qwen3.8-Flash-Next (`qwen4_exp`) config. Values in comments are the pinned artifact
/// (Qwen/Qwen3.8-Flash-Next @ de4b8e4d — geometry receipts research/qwen4exp-bringup-20260829/
/// ARCH.md, forward semantics SEMANTICS.md from transformers modular_qwen4_exp.py). qwen3_5
/// ancestry INFORMS these fields but proves nothing (no-generic-support law): the QSA
/// micro-block indexer, the 4-branch gated residual, the PLE n-gram block, the fused-3D
/// experts, and the sigmoid GDN output gate are all this family's own semantic program.
///
/// Every field here is REQUIRED by the eventual forward pass; `from_hf` panics with the field
/// name when config.json omits one (loud refusal over a silently-defaulted different program).
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    // ---- QSA micro-block sparse indexer (self_attn.indexer.*, on full-attention layers) ----
    pub indexer_n_heads: u32,        // 4 query heads
    pub indexer_kv_heads: u32,       // 1 shared key head (MQA)
    pub indexer_head_dim: u32,       // 128
    pub indexer_compress_ratio: u32, // 4 (tokens per scored micro-block)
    pub indexer_budget: u32,         // 2048 tokens = 512 micro-blocks
    // ---- gated residual (attn_/mlp_hyper_connection.* + one global hyper_connection_mixer) ----
    pub hc_count: u32,   // 4 streams — wide stream = hc_count * hidden = 10240
    pub hc_lowrank: u32, // 320 (input_mix_weight_down [320,10240] / _up [10240,320])
    // ---- n-gram / PLE ----
    pub ngram_size: u32,      // 3 (bigram + trigram histories)
    pub heads_per_ngram: u32, // 8 (=> 16 n-gram heads total)
    /// 20_000_000. Per-head vocab sizes are consecutive primes >= this base, shipped as
    /// checkpoint I64 buffers (ngram_heads_vocab_sizes/offsets) — LOAD, never re-derive
    /// (SEMANTICS.md §PLE).
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: u32, // 128
    pub split_ngram_parts: u32,                  // 128 shards, concatenated on dim 0
    /// ONE-indexed (config class docstring): `[2]` = checkpoint module `layers.1` (0-based).
    /// Receipt: census `model.language_model.layers.1.ple.conv1d.weight` (SEMANTICS.md §PLE).
    pub ple_layer_ids: Vec<u32>,
    pub ple_embed_dim: u32,        // 2560
    pub ple_conv_kernel_size: u32, // 4 (dilation 3, causal left-pad 9 — SEMANTICS.md §PLE)
    /// GDN gated output-norm activation: "sigmoid" here — the ONE numeric divergence from the
    /// qwen3_5 GDN program (SEMANTICS.md §GDN; config contract allows {"sigmoid","silu"}).
    pub output_gate_type: String,
    /// `eos_token_id` (248044 on the artifact; a list config takes its first entry —
    /// modular L621). PLE pads token history with it and resets n-gram segments at it
    /// (SEMANTICS.md §PLE), so it is REQUIRED whenever `ple_layer_ids` is non-empty;
    /// enforced at parse.
    pub eos_token_id: Option<u32>,
    // ---- mrope: STORED, not implemented. Text-only inputs degenerate to plain partial rope
    // (all 3 axes equal — SEMANTICS.md §Rope); the real 3-axis path is vision-lane work. ----
    pub mrope_section: Vec<u32>, // [11, 11, 10]
    pub mrope_interleaved: bool, // true
    // ---- MTP sub-object ----
    pub mtp_num_hidden_layers: u32, // 1 (full-attention QSA block, own indexer, no PLE)
    pub mtp_rope_theta: f32,        // 1e7 (same base as the trunk on this artifact)
    // ---- ViT tower (model.visual.*) ----
    /// `None` on a text-only sibling. The tower serves through the qwen3_5-style side-load
    /// path (MEMRA_VISION_DIR), NOT a plan-level encoder; this struct exists so the tensor
    /// contract can census the `model.visual.*` namespace. The generic `VisionConfig` parse
    /// speaks gemma key names and would fabricate wrong geometry for this file, so
    /// `ModelConfig::vision` stays `None` for the arch.
    pub vision: Option<Qwen4ExpVisionConfig>,
}

/// YaRN rope scaling as the qwen4_exp family consumes it (transformers 5.14.1
/// `_compute_yarn_parameters`, truncate=True arm): frequency interpolation over the
/// partial-rotary dims plus the derived `attention_factor = 0.1*ln(factor)+1` on cos/sin.
/// Parsed from `rope_parameters`/`rope_scaling` with `rope_type == "yarn"`; keys the twin
/// does NOT implement (explicit `attention_factor`, `mscale`, `mscale_all_dim`,
/// `truncate=false`) are REFUSED at parse rather than silently mis-scaled.
/// Receipt: research/qwen4exp-bringup-20260829/yarn/transformers-yarn-params.tsv.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnRopeConfig {
    /// Context extension factor. transformers uses the config value AS GIVEN when present
    /// and derives `max_position_embeddings / original_max_position_embeddings` only when
    /// the key is absent — mirrored here.
    pub factor: f32,
    pub original_context: u32,
    pub beta_fast: f32, // default 32 (paper defaults, transformers `or 32`)
    pub beta_slow: f32, // default 1
}

/// qwen4_exp `vision_config` geometry (census receipts in ARCH.md: 27 blocks, fused qkv
/// [3456,1152]+bias, mlp fc 4304, patch Conv3d [1152,3,2,16,16], pos_embed [2304,1152],
/// merger 4608 -> 2560).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4ExpVisionConfig {
    pub depth: u32,                   // 27
    pub hidden_size: u32,             // 1152
    pub intermediate_size: u32,       // 4304
    pub num_heads: u32,               // 16
    pub num_position_embeddings: u32, // 2304
    pub out_hidden_size: u32,         // 2560
    pub patch_size: u32,              // 16
    pub spatial_merge_size: u32,      // 2 (merger in = hidden * merge^2 = 4608)
    pub temporal_patch_size: u32,     // 2
    pub in_channels: u32,             // 3
}

impl Qwen4ExpVisionConfig {
    /// Merger input width: spatial-merged patch group (2x2 of hidden) = 4608.
    pub fn merger_in(&self) -> u32 {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }
}

impl Qwen4ExpConfig {
    /// Total n-gram heads: `heads_per_ngram` per n in 2..=ngram_size (16 on the artifact).
    pub fn ngram_heads(&self) -> u32 {
        self.heads_per_ngram * self.ngram_size.saturating_sub(1)
    }
    /// Per-head embedding row width: ple_embed_dim / ngram_heads = 160 (census receipt:
    /// ngram_embedding shard shape [2500012, 160]; 16 heads x 160 = 2560 = value_proj in).
    pub fn ngram_head_embed_dim(&self) -> u32 {
        let heads = self.ngram_heads();
        assert!(
            heads > 0 && self.ple_embed_dim.is_multiple_of(heads),
            "qwen4_exp ple_embed_dim {} not divisible by ngram heads {heads}",
            self.ple_embed_dim
        );
        self.ple_embed_dim / heads
    }
    /// Checkpoint (0-based) trunk-layer indices carrying the PLE module. `ple_layer_ids` is
    /// ONE-indexed: config `[2]` places the module at `layers.1` (SEMANTICS.md §PLE).
    pub fn ple_checkpoint_layers(&self) -> Vec<u32> {
        self.ple_layer_ids
            .iter()
            .map(|&id| {
                assert!(
                    id >= 1,
                    "qwen4_exp ple_layer_ids are ONE-indexed; got 0 (a 0 entry would wrap)"
                );
                id - 1
            })
            .collect()
    }
    /// Layer `il` (checkpoint 0-based) carries the PLE block.
    pub fn has_ple(&self, il: u32) -> bool {
        self.ple_checkpoint_layers().contains(&il)
    }
    /// Indexer budget in micro-blocks (2048 / 4 = 512).
    pub fn indexer_budget_blocks(&self) -> u32 {
        assert!(
            self.indexer_compress_ratio > 0
                && self
                    .indexer_budget
                    .is_multiple_of(self.indexer_compress_ratio),
            "qwen4_exp indexer_budget {} not divisible by compress ratio {}",
            self.indexer_budget,
            self.indexer_compress_ratio
        );
        self.indexer_budget / self.indexer_compress_ratio
    }
}

/// GLM-5.3-Flash (`glm5_next`) config. Values in comments are the zai-org/GLM-5.3-Flash FP8
/// checkpoint (census + banked config.json in research/glm53-flash-bringup-20260827). GlmDsa /
/// DeepSeek ancestry INFORMS these fields but proves nothing (no-generic-support law): the KDA
/// mixer, the NoPE MLA (rope dim 0), the k-pool-compressed indexer, and the mHC stream routing
/// are all this family's own semantic program.
///
/// Every field here is REQUIRED by the eventual forward pass; `from_hf` panics with the
/// config.json field path when one is missing (loud refusal over a silently-defaulted
/// different program).
#[derive(Debug, Clone)]
pub struct Glm5NextConfig {
    // ---- per-TRUNK-layer schedules (45 entries; the MTP layer is NOT in them — it carries
    // its own MLA attention + MoE, and its indexer rides `index_share_for_mtp_iteration`) ----
    /// true = "linear_attention" (KDA, 34 layers), false = "deepseek_sparse_attention"
    /// (MLA+DSA, 11 layers: il in {3, 7, ..., 43}). Derived from `layer_types` and
    /// cross-checked against `linear_attn_config.{kda_layers,full_attn_layers}` at parse.
    pub kda_layer: Vec<bool>,
    /// true = dense FFN (first 3 layers), false = 288-expert MoE. From `mlp_layer_types`,
    /// cross-checked against `first_k_dense_replace`.
    pub dense_layer: Vec<bool>,
    /// Per-layer DSA indexer mode: "full" = own top-k selection (+ indexer tensors),
    /// "shared" = reuse the previous full layer's selection. 45 x "full" on the checkpoint.
    pub indexer_types: Vec<String>,
    // ---- KDA mixer (linear_attn_config) ----
    pub linear_num_heads: u32,   // num_heads (64)
    pub linear_head_dim: u32,    // head_dim (128)
    pub linear_conv_kernel: u32, // short_conv_kernel_size (4)
    /// Forget-gate decay floor: gate = lower_bound * sigmoid(exp(A_log) * g). -5.0.
    pub gate_lower_bound: f32,
    // ---- MLA (NoPE: no rotary anywhere in the MLA path — nor in the indexer: the 5.3
    // reference's `Glm5NextTextIndexer.forward` applies no rope and never reads
    // `indexer_rope_interleave`; the key is carried below as checkpoint metadata only) ----
    pub q_lora_rank: u32,      // 1536
    pub kv_lora_rank: u32,     // 512
    pub qk_head_dim: u32,      // 256 (== qk_nope_head_dim + qk_rope_head_dim, checked at parse)
    pub qk_nope_head_dim: u32, // 256
    pub qk_rope_head_dim: u32, // 0 (NoPE)
    pub v_head_dim: u32,       // 256
    pub mla_use_nope: bool,    // true (requires qk_rope_head_dim == 0, checked at parse)
    // ---- DSA indexer (k-pool compressed) ----
    pub index_n_heads: u32,                   // 32
    pub index_head_dim: u32,                  // 128
    pub index_topk: u32,                      // 2048 (must divide by index_kpool, checked at parse)
    pub index_kpool: u32,                     // 4 (compressed token-group pool size)
    pub index_kpool_always_select_tail: bool, // true (incomplete tail group always attended)
    pub index_kpool_compress: bool,           // true (compress_ape/compress_gate tensors present)
    pub indexer_rope_interleave: bool,        // true
    pub index_share_for_mtp_iteration: bool,  // true (MTP layer reuses trunk indexer selection)
    // ---- MoE routing ----
    pub n_routed_experts: u32,      // 288
    pub num_experts_per_tok: u32,   // 8
    pub moe_intermediate_size: u32, // 2048
    pub n_shared_experts: u32,      // 1
    pub first_k_dense_replace: u32, // 3
    pub scoring_func: String,       // "sigmoid"
    pub topk_method: String,        // "noaux_tc"
    pub routed_scaling_factor: f32, // 2.5
    pub norm_topk_prob: bool,       // true
    pub moe_router_dtype: String,   // "float32"
    // ---- Sinkhorn hyper-connections (mHC) ----
    pub mhc: bool,              // true
    pub hc_mult: u32,           // 4 (residual stream count)
    pub hc_eps: f32,            // 1e-6
    pub hc_sinkhorn_iters: u32, // 20
    // ---- misc ----
    pub swiglu_limit: f32,             // 10.0
    pub num_nextn_predict_layers: u32, // 1
    /// KDA o_norm gate activation. The banked config.json carries NO `output_gate_type` key;
    /// the reference Glm5NextTextRMSNormGated hardcodes activation = "sigmoid", so absence
    /// defaults to "sigmoid" rather than panicking (the one non-required field here).
    pub output_gate_type: String,
}

impl Glm5NextConfig {
    /// Trunk layer `il` runs the KDA linear-attention mixer. The schedule vectors are
    /// trunk-indexed (len = num_hidden_layers); the plan compiler also asks about the
    /// NextN/MTP layer at index n_trunk, which is MLA+indexer+MoE by census — so
    /// out-of-range answers the MTP layer's class instead of panicking.
    pub fn is_kda_layer(&self, il: u32) -> bool {
        self.kda_layer.get(il as usize).copied().unwrap_or(false)
    }
    /// Trunk layer `il` runs a dense FFN instead of the routed-expert MoE. The MTP layer
    /// mirrors a MoE trunk layer (12,384/288 = 43 expert banks = 42 sparse trunk + MTP).
    pub fn is_dense_layer(&self, il: u32) -> bool {
        self.dense_layer.get(il as usize).copied().unwrap_or(false)
    }
    /// Layer `il` carries its own indexer tensors ("full"); "shared" layers reuse the
    /// previous full layer's top-k selection. The MTP layer owns the 12th indexer set.
    pub fn has_own_indexer(&self, il: u32) -> bool {
        self.indexer_types
            .get(il as usize)
            .is_none_or(|t| t == "full")
    }
}

/// DSA (DeepSeek Sparse Attention) lightning-indexer geometry (GLM-5.2). Parsed when the GGUF
/// carries the `attention.indexer.*` keys; consumed by increment 6 (indexer arm). GLM-5.2:
/// 32 heads x 128 (64 rope + 64 nope), top-k 2048, 21 "full" layers (own top-k) + 57 "shared"
/// (reuse the previous full layer's indices — IndexShare/IndexCache, arXiv 2603.12201).
#[derive(Debug, Clone)]
pub struct DsaConfig {
    pub index_n_heads: u32,  // glm-dsa.attention.indexer.head_count (32)
    pub index_head_dim: u32, // glm-dsa.attention.indexer.key_length (128)
    pub index_top_k: u32,    // glm-dsa.attention.indexer.top_k (2048)
    /// Per TRUNK layer: true = "full" indexer layer (own top-k selection + indexer tensors),
    /// false = "shared" (reuses the previous full layer's top-k; NO indexer tensors in the GGUF).
    /// glm-dsa.attention.indexer.types, [bool; n_trunk]. Empty if the key is absent (pre-5.2
    /// GLM GGUFs: all layers full — llama.cpp BC default).
    pub indexer_full: Vec<bool>,
}

/// llama.cpp `GLM_5_2_DEFAULT_INDEXER_TYPES` (glm-dsa.cpp): full-indexer layers at 0,1 then
/// every 4th from 2 — {0,1,2,6,10,...} — i.e. 21 full / 57 shared over 78 trunk layers. This is
/// NOT just BC: the 2026-06 unsloth GLM-5.2 GGUFs ship WITHOUT the `attention.indexer.types`
/// key (verified from the artifact header 2026-08-01, research/mla-inc2-20260801/ARTIFACT.md),
/// so the default table is what actually drives layer classification on the real artifact.
pub fn glm52_default_indexer_types(n_trunk: usize) -> Vec<bool> {
    (0..n_trunk).map(|i| i < 2 || (i - 2) % 4 == 0).collect()
}

/// MLA (multi-head latent attention) geometry + glm-dsa router knobs, parsed from the GGUF keys
/// the llama.cpp converter writes (pinned in research/mla-bringup-20260801/RECEIPTS.md §5).
/// GLM-5.2 values in comments. The latent KV-cache row is `latent_dim()` = kv_lora_rank +
/// qk_rope_head_dim (576); V is the first `kv_lora_rank` (512) elements of the SAME row.
#[derive(Debug, Clone)]
pub struct MlaConfig {
    pub q_lora_rank: u32,  // glm-dsa.attention.q_lora_rank (2048)
    pub kv_lora_rank: u32, // glm-dsa.attention.kv_lora_rank (512)
    /// Per-head qk dim AFTER decompression (nope + rope) — glm-dsa.attention.key_length_mla (256).
    /// The softmax scale is 1/sqrt(THIS), not of the absorbed 576 width (DESIGN.md §1.3).
    pub qk_head_dim: u32,
    pub qk_nope_head_dim: u32, // qk_head_dim - qk_rope_head_dim (192)
    pub qk_rope_head_dim: u32, // glm-dsa.rope.dimension_count (64)
    /// Per-head v dim after decompression — glm-dsa.attention.value_length_mla (256).
    pub v_head_dim: u32,
    // ---- deepseek-style sigmoid router (the Hy3/M3-class knobs, glm-dsa key names) ----
    pub sigmoid_routing: bool, // expert_gating_func == 2 (sigmoid); absent => sigmoid (BC)
    pub routed_scaling_factor: f32, // glm-dsa.expert_weights_scale (2.5)
    pub route_norm: bool,      // glm-dsa.expert_weights_norm (norm_topk_prob: true)
    pub n_shared_experts: u32, // glm-dsa.expert_shared_count (1)
    pub first_k_dense_replace: u32, // glm-dsa.leading_dense_block_count (3)
    // ---- DSA indexer (None when the GGUF carries no indexer keys) ----
    pub dsa: Option<DsaConfig>,
}

impl MlaConfig {
    /// Latent KV-cache row width: [rmsnorm(c_kv) | rope(k_pe)] — 576 on GLM-5.2. This is what
    /// `attention.key_length` carries in a glm-dsa GGUF (cross-checked at parse).
    pub fn latent_dim(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
    /// The V view is the first kv_lora_rank (512) elements of each latent row — what
    /// `attention.value_length` carries in a glm-dsa GGUF. No separate V plane exists.
    pub fn v_view_dim(&self) -> u32 {
        self.kv_lora_rank
    }
    /// Softmax scale: 1/sqrt(qk_head_dim) = 1/16 on GLM-5.2 (mscale = 1, no yarn).
    pub fn scale(&self) -> f32 {
        1.0 / (self.qk_head_dim as f32).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: Arch,
    pub name: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub context_length: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    pub rope_dim_count: u32, // partial rotary: only this many head dims get RoPE
    pub rope_sections: Vec<i32>, // M-RoPE sections (qwen35), empty if plain
    // hybrid (qwen35)
    pub full_attention_interval: u32, // 0 if not hybrid; else every Nth layer is full-attn
    pub ssm: Option<SsmConfig>,
    // moe
    pub moe: Option<MoeConfig>,
    // MiniMax-M3 extras (None for every other arch)
    pub m3: Option<M3Config>,
    // Hy3 extras (None for every other arch)
    pub hy3: Option<Hy3Config>,
    pub gemma4: Option<Gemma4Config>,
    pub vision: Option<VisionConfig>,
    /// glm5_next tower (mutually exclusive with `vision`; keyed by
    /// `vision_config.model_type` at parse, enforced at plan compile).
    pub vision_glm5: Option<Glm5VisionConfig>,
    pub multimodal: Option<MultimodalConfig>,
    // MLA extras — glm-dsa only (None for every other arch)
    pub mla: Option<MlaConfig>,
    // Step-3.5/3.7-Flash extras — `step35` only (None for every other arch)
    pub step35: Option<Step35Config>,
    // DeepSeek-V4-Flash extras — `deepseek_v4` only (None for every other arch)
    pub dsv4: Option<DeepSeekV4Config>,
    // Qwen3.8-Flash-Next extras — `qwen4_exp` only (None for every other arch)
    pub qwen4exp: Option<Qwen4ExpConfig>,
    /// YaRN rope scaling on the full-attention rope. Populated ONLY by the qwen4_exp HF
    /// arm today (scope: that family's long-context lane) — other arches keep their own
    /// rope-scaling handling (dsv4 flattens into DeepSeekV4Config, llama3 synthesizes
    /// rope_freqs) and are untouched.
    pub rope_yarn: Option<YarnRopeConfig>,
    // GLM-5.3-Flash extras — `glm5_next` only (None for every other arch)
    pub glm5: Option<Glm5NextConfig>,
    // Declarative per-layer geometry for migrated architectures.
    pub geometry: Option<ArchGeometryTable>,
    // multi-token-predict / NextN
    pub nextn_predict_layers: u32,
    pub n_layer_total: u32, // includes appended MTP layers
    /// The source's sliding-window key, carried VERBATIM for families whose plan does not
    /// consume it. Every family that serves a window parses it into its own sub-config
    /// (gemma4/step35/dsv4); a config that carries one and lands in a pack with no window
    /// support is a DIFFERENT attention program wearing the same shape (Mistral-7B-v0.1 is
    /// `mistral` with `sliding_window: 4096`). Packs refuse it here instead of silently
    /// compiling full attention (#216).
    pub window_hint: Option<u32>,
    /// The source's `rope_scaling.rope_type`, carried for the same reason: `llama3` scaling
    /// is a different RoPE program and the canonical plan compiles `RopeFactors::None`.
    /// `None` and `"default"` both mean identity.
    pub rope_scaling_hint: Option<String>,
}

/// qwen4_exp YaRN parse (scope: this family only — see `ModelConfig::rope_yarn`).
/// transformers 5.14.1 `_compute_yarn_parameters` semantics, mirrored:
/// - rope_type absent or "default" => no scaling;
/// - "yarn" => `original_max_position_embeddings` REQUIRED; `factor` used AS GIVEN when
///   present and derived `max_position_embeddings / original` only when absent;
/// - betas default 32/1 (a zero beta is refused — transformers' `or 32` would too);
/// - keys the engine twin does not implement (explicit attention_factor, mscale,
///   mscale_all_dim, truncate=false) are REFUSED loudly rather than silently mis-scaled.
fn qwen4exp_rope_yarn(c: &HfConfig) -> Option<YarnRopeConfig> {
    match c.rope_scaling_type.as_deref() {
        None | Some("default") => None,
        Some("yarn") => {
            assert!(
                c.rope_scaling_attention_factor.is_none(),
                "qwen4_exp yarn: explicit attention_factor is not implemented \
                 (the twin derives 0.1*ln(factor)+1 per transformers get_mscale)"
            );
            assert!(
                c.rope_scaling_mscale.is_none() && c.rope_scaling_mscale_all_dim.is_none(),
                "qwen4_exp yarn: mscale/mscale_all_dim are not implemented"
            );
            assert!(
                c.rope_scaling_truncate.unwrap_or(true),
                "qwen4_exp yarn: truncate=false is not implemented (the twin floors/ceils \
                 the correction range)"
            );
            let original_context = c.rope_scaling_original_context.unwrap_or_else(|| {
                panic!("qwen4_exp yarn rope scaling requires original_max_position_embeddings")
            });
            assert!(
                original_context > 0,
                "qwen4_exp yarn: original context is zero"
            );
            let factor = c.rope_scaling_factor.unwrap_or_else(|| {
                assert!(
                    c.max_position_embeddings > 0,
                    "qwen4_exp yarn: factor absent and max_position_embeddings unset \
                     (transformers derives factor = max_position_embeddings / original)"
                );
                c.max_position_embeddings as f32 / original_context as f32
            });
            assert!(
                factor.is_finite() && factor > 0.0,
                "qwen4_exp yarn: invalid factor {factor}"
            );
            let beta_fast = c.rope_yarn_beta_fast.unwrap_or(32.0);
            let beta_slow = c.rope_yarn_beta_slow.unwrap_or(1.0);
            assert!(
                beta_fast > 0.0 && beta_slow > 0.0,
                "qwen4_exp yarn: betas must be positive (got {beta_fast}/{beta_slow})"
            );
            Some(YarnRopeConfig {
                factor,
                original_context,
                beta_fast,
                beta_slow,
            })
        }
        Some(other) => panic!("qwen4_exp rope_type {other:?} is not supported (default|yarn)"),
    }
}

fn qwen3next_expert_alias_warning(raw_arch: &str, moe: Option<&MoeConfig>) -> Option<String> {
    let expert_count = moe?.expert_count;
    (raw_arch == "qwen3next" && expert_count > 0).then(|| {
        format!(
            "[gguf] WARN: architecture qwen3next carries {expert_count} experts but maps to \
             dense-enum alias Arch::Qwen35; this is a hybrid MoE loaded through a dense alias, \
             so subsystems keyed only on Arch may misclassify it"
        )
    })
}

impl ModelConfig {
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn from_gguf(g: &GgufFile) -> Self {
        let raw_arch = g.arch().unwrap_or("unknown");
        let arch = Arch::parse(raw_arch);
        let u = |k: &str| g.meta_arch(k).and_then(|v| v.as_u64()).map(|x| x as u32);
        let f = |k: &str| g.meta_arch(k).and_then(|v| v.as_f32());

        let n_layer = u("block_count").expect("block_count");
        let n_embd = u("embedding_length").expect("embedding_length");
        let head_dim_k = u("attention.key_length").unwrap_or_else(|| {
            // fall back to n_embd / n_head if not present
            n_embd / u("attention.head_count").unwrap_or(1)
        });
        let head_dim_v = u("attention.value_length").unwrap_or(head_dim_k);

        let rope_sections = match g.meta_arch("rope.dimension_sections") {
            Some(MetaValue::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as i32))
                .collect(),
            _ => Vec::new(),
        };

        let parsed_ssm = SsmConfig {
            conv_kernel: u("ssm.conv_kernel").unwrap_or(0),
            inner_size: u("ssm.inner_size").unwrap_or(0),
            state_size: u("ssm.state_size").unwrap_or(0),
            time_step_rank: u("ssm.time_step_rank").unwrap_or(0),
            group_count: u("ssm.group_count").unwrap_or(0),
        };
        let ssm = (parsed_ssm.conv_kernel > 0
            || parsed_ssm.inner_size > 0
            || parsed_ssm.state_size > 0
            || parsed_ssm.time_step_rank > 0
            || parsed_ssm.group_count > 0)
            .then_some(parsed_ssm);

        let expert_count = u("expert_count");
        let moe = if expert_count.is_some_and(|count| count > 0) {
            Some(MoeConfig {
                expert_count: expert_count.unwrap_or(0),
                expert_used_count: u("expert_used_count").unwrap_or(0),
                expert_ff_length: u("expert_feed_forward_length").unwrap_or(0),
                // meta_arch tries "qwen35moe.expert_shared_feed_forward_length" first, then bare key
                expert_shared_ff_length: u("expert_shared_feed_forward_length").unwrap_or(0),
            })
        } else {
            None
        };
        if let Some(warning) = qwen3next_expert_alias_warning(raw_arch, moe.as_ref()) {
            eprintln!("{warning}");
        }

        let nextn = u("nextn_predict_layers").unwrap_or(0);

        let gemma4 = if matches!(&arch, Arch::Gemma4) {
            let arr_u = |k: &str| -> Vec<u32> {
                match g.meta_arch(k) {
                    Some(MetaValue::Array(a)) => a
                        .iter()
                        .filter_map(|v| v.as_u64().map(|x| x as u32))
                        .collect(),
                    _ => Vec::new(),
                }
            };
            Some(Gemma4Config {
                head_count_kv: arr_u("attention.head_count_kv"),
                swa_pattern: arr_u("attention.sliding_window_pattern")
                    .iter()
                    .map(|&x| x == 1)
                    .collect(),
                sliding_window: u("attention.sliding_window").unwrap_or(1024),
                key_length_global: u("attention.key_length").unwrap_or(512),
                key_length_swa: u("attention.key_length_swa").unwrap_or(256),
                rope_base_global: f("rope.freq_base").unwrap_or(1e6),
                rope_base_swa: f("rope.freq_base_swa").unwrap_or(1e4),
                rope_dims_global: u("rope.dimension_count").unwrap_or(512),
                rope_dims_swa: u("rope.dimension_count_swa").unwrap_or(256),
                final_logit_softcapping: f("final_logit_softcapping").unwrap_or(30.0),
                partial_rotary_global: 0.25, // unused on GGUF (rope_freqs shipped)
                n_embd_per_layer: u("embedding_length_per_layer_input").unwrap_or(0),
                shared_kv_layers: u("attention.shared_kv_layers").unwrap_or(0),
                suppress_tokens: match g.metadata.get("tokenizer.ggml.suppress_tokens") {
                    Some(MetaValue::Array(a)) => a
                        .iter()
                        .filter_map(|v| v.as_u64().map(|x| x as u32))
                        .collect(),
                    _ => Vec::new(),
                },
            })
        } else {
            None
        };

        // step35 (Step-3.5/3.7-Flash). Reference: upstream `src/models/step35.cpp`
        // `load_arch_hparams` + `llama-model.cpp:1190-1235` (the generic n_rot defaulting that
        // runs BEFORE the arch hook halves n_rot_full).
        let step35 = if matches!(&arch, Arch::Step35) {
            let arr_u = |k: &str| -> Vec<u32> {
                match g.meta_arch(k) {
                    Some(MetaValue::Array(a)) => a
                        .iter()
                        .filter_map(|v| v.as_u64().map(|x| x as u32))
                        .collect(),
                    // The key may legitimately be a SCALAR (upstream `get_key_or_arr` accepts
                    // both, and a uniform-geometry sibling would write one) — broadcast it.
                    Some(v) => match v.as_u64() {
                        Some(x) => vec![x as u32; n_layer as usize],
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                }
            };
            let arr_f = |k: &str| -> Vec<f32> {
                match g.meta_arch(k) {
                    Some(MetaValue::Array(a)) => a.iter().filter_map(|v| v.as_f32()).collect(),
                    Some(v) => match v.as_f32() {
                        Some(x) => vec![x; n_layer as usize],
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                }
            };
            let head_count = arr_u("attention.head_count");
            assert!(
                !head_count.is_empty(),
                "step35: attention.head_count missing"
            );
            // The array covers every block INCLUDING the MTP ones (the 3.7 trunk GGUF writes 45,
            // the standalone MTP GGUF writes 48 = 45 trunk + 3 nextn). Short arrays are a
            // mis-converted file — a silent last-value broadcast would give the wrong wq width.
            assert!(
                head_count.len() as u32 >= n_layer,
                "step35: attention.head_count has {} entries, need >= block_count {n_layer}",
                head_count.len()
            );
            let swa_pattern: Vec<bool> = match g.meta_arch("attention.sliding_window_pattern") {
                // The artifact writes a BOOL array (llama.cpp `get_key_or_arr` into is_swa_impl).
                // `as_u64` maps Bool -> 0/1, so one reader covers bool and int serializations.
                Some(MetaValue::Array(a)) => a
                    .iter()
                    .filter_map(|v| v.as_u64().map(|x| x != 0))
                    .collect(),
                // Scalar form = llama.cpp's n_pattern convention: layer il is SWA unless
                // il % n_pattern == 0 (llama-hparams.cpp:11, set_swa_pattern).
                Some(v) => match v.as_u64() {
                    Some(np) if np > 0 => (0..n_layer).map(|il| il as u64 % np != 0).collect(),
                    _ => Vec::new(),
                },
                None => Vec::new(),
            };
            assert!(
                swa_pattern.len() as u32 >= n_layer,
                "step35: attention.sliding_window_pattern has {} entries, need >= {n_layer}",
                swa_pattern.len()
            );
            // Upstream: the generic loader sets n_rot_full = attention.key_length (128), copies
            // n_rot_swa from it, THEN step35.cpp halves n_rot_full -> 64. So SWA = 128, full = 64.
            let rope_dims_swa = u("rope.dimension_count").unwrap_or(head_dim_k);
            Some(Step35Config {
                head_count,
                head_count_kv: {
                    let kv = arr_u("attention.head_count_kv");
                    assert!(
                        kv.len() as u32 >= n_layer,
                        "step35: attention.head_count_kv has {} entries, need >= {n_layer}",
                        kv.len()
                    );
                    kv
                },
                swa_pattern,
                sliding_window: u("attention.sliding_window")
                    .expect("step35: attention.sliding_window (SWA layers need a window)"),
                rope_base_global: f("rope.freq_base").unwrap_or(10000.0),
                // ABSENT => same as global (upstream get_key(..., false) leaves the copied value).
                rope_base_swa: f("rope.freq_base_swa")
                    .unwrap_or_else(|| f("rope.freq_base").unwrap_or(10000.0)),
                rope_dims_full: rope_dims_swa / 2,
                rope_dims_swa,
                rope_freq_factors: None,
                swiglu_clamp_exp: arr_f("swiglu_clamp_exp"),
                swiglu_clamp_shexp: arr_f("swiglu_clamp_shexp"),
                // expert_gating_func 2 = sigmoid; ABSENT defaults to sigmoid (step35.cpp:19-21).
                sigmoid_routing: u("expert_gating_func").map(|v| v == 2).unwrap_or(true),
                routed_scaling_factor: f("expert_weights_scale").unwrap_or(1.0),
                route_norm: u("expert_weights_norm").map(|v| v != 0).unwrap_or(false),
                first_k_dense_replace: u("leading_dense_block_count").unwrap_or(0),
            })
        } else {
            None
        };

        // glm-dsa MLA + DSA keys (RECEIPTS.md §5 — the exact set the llama.cpp converter writes).
        let mla = if matches!(&arch, Arch::GlmDsa) {
            let q_lora_rank = u("attention.q_lora_rank").expect("glm-dsa: attention.q_lora_rank");
            let kv_lora_rank =
                u("attention.kv_lora_rank").expect("glm-dsa: attention.kv_lora_rank");
            let qk_head_dim =
                u("attention.key_length_mla").expect("glm-dsa: attention.key_length_mla");
            let v_head_dim =
                u("attention.value_length_mla").expect("glm-dsa: attention.value_length_mla");
            let qk_rope_head_dim =
                u("rope.dimension_count").expect("glm-dsa: rope.dimension_count");
            assert!(
                qk_rope_head_dim < qk_head_dim,
                "glm-dsa: rope dim {qk_rope_head_dim} >= qk head dim {qk_head_dim}"
            );
            // Cross-checks (DESIGN.md §3.1): attention.key_length is the LATENT cache row
            // (kv_lora_rank + rope), attention.value_length its V prefix view (kv_lora_rank).
            // A projection-wide mismatch here means a mis-converted GGUF — fail at load, loudly.
            if let Some(kl) = u("attention.key_length") {
                assert_eq!(
                    kl,
                    kv_lora_rank + qk_rope_head_dim,
                    "glm-dsa: attention.key_length {kl} != kv_lora_rank + rope {}",
                    kv_lora_rank + qk_rope_head_dim
                );
            }
            if let Some(vl) = u("attention.value_length") {
                assert_eq!(
                    vl, kv_lora_rank,
                    "glm-dsa: attention.value_length {vl} != kv_lora_rank {kv_lora_rank}"
                );
            }
            // Router: expert_gating_func 2 = sigmoid; ABSENT defaults to sigmoid (llama.cpp
            // glm-dsa BC — load_arch_hparams maps NONE -> SIGMOID).
            let sigmoid_routing = u("expert_gating_func").map(|v| v == 2).unwrap_or(true);
            // DSA indexer: present iff the converter wrote the indexer keys (GLM-5/5.1/5.2 all do).
            let dsa = u("attention.indexer.head_count").map(|index_n_heads| DsaConfig {
                index_n_heads,
                index_head_dim: u("attention.indexer.key_length")
                    .expect("glm-dsa: attention.indexer.key_length"),
                index_top_k: u("attention.indexer.top_k")
                    .expect("glm-dsa: attention.indexer.top_k"),
                indexer_full: match g.meta_arch("attention.indexer.types") {
                    Some(MetaValue::Array(a)) => a
                        .iter()
                        .filter_map(|v| v.as_u64().map(|x| x != 0))
                        .collect(),
                    // Key absent (the real 2026-06 unsloth GLM-5.2 GGUF!): llama.cpp BC —
                    // 5.2-class (ctx >= 1M) takes the hardcoded 21-full/57-shared table,
                    // pre-5.2 GLM (ctx < 1M) is all-full.
                    _ => {
                        let n_trunk = (n_layer - nextn) as usize;
                        if u("context_length").unwrap_or(0) >= 1_048_576 {
                            glm52_default_indexer_types(n_trunk)
                        } else {
                            vec![true; n_trunk]
                        }
                    }
                },
            });
            Some(MlaConfig {
                q_lora_rank,
                kv_lora_rank,
                qk_head_dim,
                qk_nope_head_dim: qk_head_dim - qk_rope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                sigmoid_routing,
                routed_scaling_factor: f("expert_weights_scale").unwrap_or(1.0),
                route_norm: u("expert_weights_norm").map(|v| v != 0).unwrap_or(false),
                n_shared_experts: u("expert_shared_count").unwrap_or(0),
                first_k_dense_replace: u("leading_dense_block_count").unwrap_or(0),
                dsa,
            })
        } else {
            None
        };

        let n_head = u("attention.head_count").unwrap_or_else(|| {
            step35
                .as_ref()
                .and_then(|s| s.head_count.iter().copied().max())
                .expect("head_count")
        });
        let n_head_kv = u("attention.head_count_kv")
            .or_else(|| {
                step35
                    .as_ref()
                    .and_then(|s| s.head_count_kv.iter().copied().max())
            })
            .unwrap_or_else(|| {
                u("attention.head_count")
                    .or_else(|| {
                        step35
                            .as_ref()
                            .and_then(|s| s.head_count.iter().copied().max())
                    })
                    .expect("head_count_kv fallback")
            });
        let rope_freq_base = f("rope.freq_base").unwrap_or(10000.0);
        // GGUF carries the resolved dim count; there is no fraction spelling on this side.
        // Shared with the HF reader so the two paths cannot drift (see resolve_rope_dim_count).
        let rope_dim_count = resolve_rope_dim_count(u("rope.dimension_count"), None, head_dim_k);
        let full_attention_interval = u("full_attention_interval").unwrap_or(0);
        let geometry = match &arch {
            // qwen4_exp reuses the interval-hybrid rule verbatim (see the table fn note).
            Arch::Qwen35 | Arch::Qwen35Moe | Arch::Qwen4Exp => Some(ArchGeometryTable::qwen35(
                n_layer,
                nextn,
                full_attention_interval,
                n_head,
                n_head_kv,
                head_dim_k,
                head_dim_v,
                rope_dim_count,
                rope_freq_base,
            )),
            Arch::Step35 => Some(ArchGeometryTable::step35(
                n_layer,
                head_dim_k,
                head_dim_v,
                step35
                    .as_ref()
                    .expect("step35 geometry needs step35 config"),
            )),
            _ => None,
        };

        ModelConfig {
            arch,
            window_hint: u("attention.sliding_window"),
            // GGUF spells llama3 rope scaling as per-frequency factors, not a type string;
            // `rope_factors` carries them and the packs that read them declare it.
            rope_scaling_hint: None,
            name: g
                .metadata
                .get("general.name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            n_layer,
            n_embd,
            // `attention.head_count` is a SCALAR on every arch but step35, where it is a
            // per-layer ARRAY (`as_u64` returns None on an Array, so the bare `.expect` would
            // panic). For step35 the global scalar is the MAX over layers: it sizes shared
            // scratch/workspace buffers, while every per-layer shape comes from
            // `step35.n_head(il)`. Max (96, not the 64 of a full layer) so no buffer under-sizes.
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_ff: u("feed_forward_length").unwrap_or(0),
            n_vocab: u("vocab_size").unwrap_or_else(|| {
                // vocab size from token_embd tensor's last dim if metadata absent
                g.find("token_embd.weight")
                    .map(|t| *t.ne.last().unwrap() as u32)
                    .unwrap_or(0)
            }),
            context_length: u("context_length").unwrap_or(0),
            rms_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope_freq_base,
            rope_dim_count,
            rope_sections,
            full_attention_interval,
            ssm,
            moe,
            m3: None,  // GGUF M3 metadata keys are a later arc (ST import first)
            hy3: None, // GGUF Hy3 metadata keys are a later arc (repack source first)
            gemma4,
            vision: None,
            vision_glm5: None, // GGUF glm5_next artifacts carry no tower (safetensors-first)
            multimodal: None,
            mla,
            step35,
            dsv4: None, // safetensors-first arch: no GGUF artifact exists (loader lane)
            qwen4exp: None, // safetensors-first arch: no GGUF artifact exists (loader lane)
            rope_yarn: None,
            glm5: None, // safetensors-first arch: no GGUF artifact exists (bring-up lane)
            geometry,
            nextn_predict_layers: nextn,
            n_layer_total: n_layer + nextn,
        }
    }

    /// Build a ModelConfig from an HF `config.json` (read parallel to a safetensors checkpoint).
    /// HF has no `{arch}.`-prefixed keys (unlike GGUF), so we read its flat field names. Hybrid
    /// (qwen3_5) nests the transformer fields under `text_config`; `from_config_json` flattens that
    /// before calling here. Lenient defaults mirror the GGUF fallbacks in `from_gguf`.
    pub fn from_hf(c: &HfConfig) -> Self {
        let arch = Arch::from_hf_model_type(&c.model_type);
        let is_gemma4 = matches!(&arch, Arch::Gemma4);
        // HF counts only trunk blocks; GGUF block_count includes appended NextN blocks.
        // qwen4_exp nests the depth in an `mtp` sub-object (its flat twin key usually rides
        // beside it; the object is the fallback for a sibling that drops the flat spelling).
        let nextn = c
            .num_nextn_predict_layers
            .or(c.mtp_num_hidden_layers)
            .or(c.qwen4exp_mtp_num_hidden_layers)
            .unwrap_or(0);
        let n_layer = c.num_hidden_layers + nextn;
        let base_n_head = c.num_attention_heads;
        let head_dim_k = c
            .head_dim
            .unwrap_or_else(|| c.hidden_size / base_n_head.max(1));
        let head_dim_v = head_dim_k;
        let base_n_head_kv = c.num_key_value_heads.unwrap_or(base_n_head);

        let expert_count = c.num_experts.or(c.num_local_experts).unwrap_or(0);
        let moe = if expert_count > 0 {
            let expert_ff_length = c
                .moe_intermediate_size
                .or(c.expert_hidden_dim)
                .unwrap_or(c.intermediate_size);
            let n_shared = c.n_shared_experts.unwrap_or(0);
            let shared_ff_length = c
                .shared_expert_intermediate_size
                .or(c.shared_intermediate_size)
                .or_else(|| {
                    if arch.is_hy3() && n_shared > 0 {
                        Some(expert_ff_length * n_shared)
                    } else if is_gemma4 {
                        Some(c.intermediate_size)
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            Some(MoeConfig {
                // M3 names the count `num_local_experts`, the shared FF `shared_intermediate_size`.
                expert_count,
                expert_used_count: c.num_experts_per_tok.unwrap_or(0),
                // OLMoE has no separate `moe_intermediate_size`; its experts use `intermediate_size`.
                expert_ff_length,
                expert_shared_ff_length: shared_ff_length,
            })
        } else {
            None
        };

        let m3 = if arch.is_minimax() {
            Some(M3Config {
                use_gemma_norm: c.use_gemma_norm.unwrap_or(false),
                sigmoid_routing: c.scoring_func.as_deref() == Some("sigmoid"),
                use_routing_bias: c.use_routing_bias.unwrap_or(false),
                routed_scaling_factor: c.routed_scaling_factor.unwrap_or(1.0),
                n_shared_experts: c.n_shared_experts.unwrap_or(0),
                swiglu_alpha: c.swiglu_alpha.unwrap_or(1.702),
                swiglu_limit: c.swiglu_limit.unwrap_or(7.0),
                rotary_dim: c.rotary_dim.unwrap_or(0),
                dense_intermediate_size: c.dense_intermediate_size.unwrap_or(c.intermediate_size),
                moe_layer_freq: c.moe_layer_freq.clone().unwrap_or_default(),
            })
        } else {
            None
        };

        let hy3 = if arch.is_hy3() {
            Some(Hy3Config {
                sigmoid_routing: c.moe_router_use_sigmoid.unwrap_or(false),
                use_routing_bias: c.moe_router_enable_expert_bias.unwrap_or(false),
                route_norm: c.route_norm.unwrap_or(false),
                router_scaling_factor: c.router_scaling_factor.unwrap_or(1.0),
                n_shared_experts: c.n_shared_experts.unwrap_or(0),
                first_k_dense_replace: c.first_k_dense_replace.unwrap_or(1),
                qk_norm: c.qk_norm.unwrap_or(false),
                hidden_act: c.hidden_act.clone().unwrap_or_else(|| "silu".to_string()),
                weight_only_nvfp4: c.quant_algo.as_deref() == Some("W4A16_NVFP4"),
            })
        } else {
            None
        };

        let step35 = if arch.is_step35() {
            let layer_types = c
                .layer_types
                .as_ref()
                .expect("step35 HF config needs layer_types");
            assert_eq!(
                layer_types.len(),
                n_layer as usize,
                "step35 layer_types length {} != trunk+NextN {n_layer}",
                layer_types.len()
            );
            let swa_pattern: Vec<bool> = layer_types
                .iter()
                .map(|kind| match kind.as_str() {
                    "sliding_attention" => true,
                    "full_attention" => false,
                    other => panic!("step35 unknown layer type {other}"),
                })
                .collect();
            let swa_heads = c
                .attention_other_num_heads
                .expect("step35 attention_other_setting.num_attention_heads");
            let swa_kv = c.attention_other_num_groups.unwrap_or(base_n_head_kv);
            let head_count = swa_pattern
                .iter()
                .map(|&swa| if swa { swa_heads } else { base_n_head })
                .collect();
            let head_count_kv = swa_pattern
                .iter()
                .map(|&swa| if swa { swa_kv } else { base_n_head_kv })
                .collect();

            let rope = c
                .rope_theta_layers
                .as_ref()
                .expect("step35 HF config needs per-layer rope_theta");
            assert_eq!(
                rope.len(),
                n_layer as usize,
                "step35 rope_theta length {} != trunk+NextN {n_layer}",
                rope.len()
            );
            let partial = c
                .partial_rotary_factors
                .as_ref()
                .expect("step35 HF config needs partial_rotary_factors");
            assert_eq!(
                partial.len(),
                n_layer as usize,
                "step35 partial_rotary_factors length {} != trunk+NextN {n_layer}",
                partial.len()
            );
            let first_full = swa_pattern
                .iter()
                .position(|&swa| !swa)
                .expect("step35 has no full-attention layer");
            let first_swa = swa_pattern
                .iter()
                .position(|&swa| swa)
                .expect("step35 has no sliding-attention layer");
            let clamps = c
                .swiglu_limits
                .clone()
                .unwrap_or_else(|| vec![0.0; n_layer as usize]);
            let shared_clamps = c
                .swiglu_limits_shared
                .clone()
                .unwrap_or_else(|| vec![0.0; n_layer as usize]);
            assert_eq!(
                clamps.len(),
                n_layer as usize,
                "step35 swiglu_limits length {} != trunk+NextN {n_layer}",
                clamps.len()
            );
            assert_eq!(
                shared_clamps.len(),
                n_layer as usize,
                "step35 swiglu_limits_shared length {} != trunk+NextN {n_layer}",
                shared_clamps.len()
            );
            let first_k_dense_replace = c
                .moe_layers_enum
                .as_deref()
                .and_then(|layers| {
                    layers
                        .split(',')
                        .filter_map(|v| v.trim().parse::<u32>().ok())
                        .min()
                })
                .or(c.first_k_dense_replace)
                .unwrap_or(0);
            let rope_dims_full = (partial[first_full] * head_dim_k as f32).round() as u32;
            Some(Step35Config {
                head_count,
                head_count_kv,
                swa_pattern,
                sliding_window: c.sliding_window.unwrap_or(512),
                rope_base_global: rope[first_full],
                rope_base_swa: rope[first_swa],
                rope_dims_full,
                rope_dims_swa: (partial[first_swa] * head_dim_k as f32).round() as u32,
                rope_freq_factors: c.llama3_rope_factors(rope[first_full], rope_dims_full),
                swiglu_clamp_exp: clamps,
                swiglu_clamp_shexp: shared_clamps,
                sigmoid_routing: c.moe_router_activation.as_deref() == Some("sigmoid"),
                routed_scaling_factor: c.router_scaling_factor.unwrap_or(1.0),
                route_norm: c.route_norm.unwrap_or(false),
                first_k_dense_replace,
            })
        } else {
            None
        };

        // Shared/global values size common scratch. Step has per-layer 64/96 query heads, so use
        // the maximum here; execution reads the exact value from the geometry table per layer.
        let n_head = step35
            .as_ref()
            .and_then(|s| s.head_count.iter().copied().max())
            .unwrap_or(base_n_head);
        let n_head_kv = step35
            .as_ref()
            .and_then(|s| s.head_count_kv.iter().copied().max())
            .unwrap_or(base_n_head_kv);

        // qwen3.5 linear-attn config keys (text_config). Presence, not architecture identity,
        // decides whether the recurrent program exists.
        let parsed_ssm = SsmConfig {
            conv_kernel: c.linear_conv_kernel_dim.unwrap_or(0),
            inner_size: c.linear_value_head_dim.unwrap_or(0)
                * c.linear_num_value_heads.unwrap_or(0),
            state_size: c.linear_key_head_dim.unwrap_or(0),
            time_step_rank: c.linear_num_value_heads.unwrap_or(0),
            group_count: c.linear_num_key_heads.unwrap_or(0),
        };
        let ssm = (parsed_ssm.conv_kernel > 0
            || parsed_ssm.inner_size > 0
            || parsed_ssm.state_size > 0
            || parsed_ssm.time_step_rank > 0
            || parsed_ssm.group_count > 0)
            .then_some(parsed_ssm);

        // NextN/MTP depth: 35B-MoE HF uses `num_nextn_predict_layers`; qwen3.6-27B (dense hybrid,
        // NVIDIA + local text ckpts) uses `mtp_num_hidden_layers`. Same meaning (head depth = 1).
        // qwen4_exp carries both a flat `mtp_num_hidden_layers` and an `mtp` sub-object; the
        // object is the fallback and is cross-checked against the flat key below.
        let mut nextn = c
            .num_nextn_predict_layers
            .or(c.mtp_num_hidden_layers)
            .or(c.qwen4exp_mtp_num_hidden_layers)
            .unwrap_or(0);
        // deepseek_v4: `num_nextn_predict_layers` is VESTIGIAL on 0731 — still 1 while the
        // drafter is 3 DSpark blocks (the repo's own inference/config.json: n_mtp_layers 3).
        // The load-bearing derivation is len(compress_ratios) − num_hidden_layers, and every
        // drafter entry must be ratio 0 (window-only block) — 0731 prep §1.1 trap. The preview
        // artifact (44 entries) derives 1, identical to its vestigial key: no behavior change.
        if arch.is_dsv4() {
            let ratios = c
                .compress_ratios
                .as_ref()
                .unwrap_or_else(|| panic!("deepseek_v4 config.json missing compress_ratios"));
            assert!(
                ratios.len() as u32 > c.num_hidden_layers,
                "deepseek_v4 compress_ratios len {} must exceed num_hidden_layers {}",
                ratios.len(),
                c.num_hidden_layers
            );
            for (k, &r) in ratios[c.num_hidden_layers as usize..].iter().enumerate() {
                assert_eq!(
                    r, 0,
                    "deepseek_v4 drafter compress_ratios entry {k} is {r}, expected 0 \
                     (window-only drafter block)"
                );
            }
            nextn = ratios.len() as u32 - c.num_hidden_layers;
        }

        // deepseek_v4: every field below feeds the forward pass; a missing one means a DIFFERENT
        // semantic program, so refuse by name instead of defaulting (from_gguf's expect style).
        let dsv4 = if arch.is_dsv4() {
            let req_u = |v: Option<u32>, k: &str| {
                v.unwrap_or_else(|| panic!("deepseek_v4 config.json missing required field {k}"))
            };
            let req_f = |v: Option<f32>, k: &str| {
                v.unwrap_or_else(|| panic!("deepseek_v4 config.json missing required field {k}"))
            };
            let compress_ratios = c
                .compress_ratios
                .clone()
                .unwrap_or_else(|| panic!("deepseek_v4 config.json missing compress_ratios"));
            assert_eq!(
                compress_ratios.len() as u32,
                c.num_hidden_layers + nextn,
                "deepseek_v4 compress_ratios len must equal num_hidden_layers + nextn \
                 (trunk entries then the MTP layer(s))"
            );
            Some(DeepSeekV4Config {
                scoring_func: c
                    .scoring_func
                    .clone()
                    .unwrap_or_else(|| panic!("deepseek_v4 config.json missing scoring_func")),
                topk_method: c
                    .topk_method
                    .clone()
                    .unwrap_or_else(|| panic!("deepseek_v4 config.json missing topk_method")),
                routed_scaling_factor: req_f(c.routed_scaling_factor, "routed_scaling_factor"),
                norm_topk_prob: c
                    .norm_topk_prob
                    .unwrap_or_else(|| panic!("deepseek_v4 config.json missing norm_topk_prob")),
                n_shared_experts: req_u(c.n_shared_experts, "n_shared_experts"),
                num_hash_layers: req_u(c.num_hash_layers, "num_hash_layers"),
                hc_eps: req_f(c.hc_eps, "hc_eps"),
                hc_mult: req_u(c.hc_mult, "hc_mult"),
                hc_sinkhorn_iters: req_u(c.hc_sinkhorn_iters, "hc_sinkhorn_iters"),
                head_dim: req_u(c.head_dim, "head_dim"),
                num_key_value_heads: req_u(c.num_key_value_heads, "num_key_value_heads"),
                q_lora_rank: req_u(c.q_lora_rank, "q_lora_rank"),
                qk_rope_head_dim: req_u(c.qk_rope_head_dim, "qk_rope_head_dim"),
                o_lora_rank: req_u(c.o_lora_rank, "o_lora_rank"),
                o_groups: req_u(c.o_groups, "o_groups"),
                index_n_heads: req_u(c.index_n_heads, "index_n_heads"),
                index_head_dim: req_u(c.index_head_dim, "index_head_dim"),
                index_topk: req_u(c.index_topk, "index_topk"),
                compress_ratios,
                compress_rope_theta: req_f(c.compress_rope_theta, "compress_rope_theta"),
                sliding_window: req_u(c.sliding_window, "sliding_window"),
                swiglu_limit: req_f(c.swiglu_limit, "swiglu_limit"),
                rope_yarn_factor: req_f(c.rope_yarn_factor, "rope_scaling.factor"),
                rope_yarn_orig_ctx: req_u(
                    c.rope_yarn_orig_ctx,
                    "rope_scaling.original_max_position_embeddings",
                ),
                rope_yarn_beta_fast: req_f(c.rope_yarn_beta_fast, "rope_scaling.beta_fast"),
                rope_yarn_beta_slow: req_f(c.rope_yarn_beta_slow, "rope_scaling.beta_slow"),
            })
        } else {
            None
        };

        // qwen4_exp: every field below feeds the forward pass; a missing one means a DIFFERENT
        // semantic program, so refuse by name instead of defaulting (the dsv4 expect style).
        let qwen4exp = if arch.is_qwen4exp() {
            let req_u = |v: Option<u32>, k: &str| {
                v.unwrap_or_else(|| panic!("qwen4_exp config.json missing required field {k}"))
            };
            let interval = req_u(c.full_attention_interval, "full_attention_interval");
            // layer_types cross-check. The shipped file spells the periodic layers
            // "full_attention"; the HF config class rewrites those entries to
            // "qwen_sparse_attention" when the indexer_* fields are present (SEMANTICS.md
            // §Loading) — accept both spellings, refuse anything off the interval pattern.
            if let Some(layer_types) = c.layer_types.as_ref() {
                assert_eq!(
                    layer_types.len() as u32,
                    c.num_hidden_layers,
                    "qwen4_exp layer_types length {} != num_hidden_layers {}",
                    layer_types.len(),
                    c.num_hidden_layers
                );
                for (il, kind) in layer_types.iter().enumerate() {
                    // manual_is_multiple_of: `interval` is a raw config value; `% 0` must
                    // keep panicking rather than take is_multiple_of's defined-result arm.
                    #[allow(clippy::manual_is_multiple_of)]
                    let full = (il as u32 + 1) % interval == 0;
                    let ok = if full {
                        kind == "full_attention" || kind == "qwen_sparse_attention"
                    } else {
                        kind == "linear_attention"
                    };
                    assert!(
                        ok,
                        "qwen4_exp layer_types[{il}] = {kind:?} contradicts \
                         full_attention_interval {interval}"
                    );
                }
            }
            let output_gate_type = c.output_gate_type.clone().unwrap_or_else(|| {
                panic!("qwen4_exp config.json missing required field output_gate_type")
            });
            assert!(
                matches!(output_gate_type.as_str(), "sigmoid" | "silu"),
                "qwen4_exp output_gate_type must be \"sigmoid\" or \"silu\" (the HF config \
                 contract), got {output_gate_type:?}"
            );
            let ple_layer_ids = c.ple_layer_ids.clone().unwrap_or_else(|| {
                panic!("qwen4_exp config.json missing required field ple_layer_ids")
            });
            let mtp_layers = req_u(c.qwen4exp_mtp_num_hidden_layers, "mtp.num_hidden_layers");
            assert_eq!(
                mtp_layers, nextn,
                "qwen4_exp mtp.num_hidden_layers {mtp_layers} != mtp_num_hidden_layers {nextn}"
            );
            let cfg = Qwen4ExpConfig {
                indexer_n_heads: req_u(c.indexer_n_heads, "indexer_n_heads"),
                indexer_kv_heads: req_u(c.indexer_kv_heads, "indexer_kv_heads"),
                indexer_head_dim: req_u(c.indexer_head_dim, "indexer_head_dim"),
                indexer_compress_ratio: req_u(c.indexer_compress_ratio, "indexer_compress_ratio"),
                indexer_budget: req_u(c.indexer_budget, "indexer_budget"),
                hc_count: req_u(c.hc_count, "hc_count"),
                hc_lowrank: req_u(c.hc_lowrank, "hc_lowrank"),
                ngram_size: req_u(c.ngram_size, "ngram_size"),
                heads_per_ngram: req_u(c.heads_per_ngram, "heads_per_ngram"),
                ngram_vocab_size_base: c.ngram_vocab_size_base.unwrap_or_else(|| {
                    panic!("qwen4_exp config.json missing required field ngram_vocab_size_base")
                }),
                make_ngram_vocab_size_divisible_by: req_u(
                    c.make_ngram_vocab_size_divisible_by,
                    "make_ngram_vocab_size_divisible_by",
                ),
                split_ngram_parts: req_u(c.split_ngram_parts, "split_ngram_parts"),
                ple_layer_ids,
                ple_embed_dim: req_u(c.ple_embed_dim, "ple_embed_dim"),
                ple_conv_kernel_size: req_u(c.ple_conv_kernel_size, "ple_conv_kernel_size"),
                output_gate_type,
                eos_token_id: c.eos_token_id,
                mrope_section: c.mrope_section.clone().unwrap_or_default(),
                mrope_interleaved: c.mrope_interleaved.unwrap_or(false),
                mtp_num_hidden_layers: mtp_layers,
                mtp_rope_theta: c.qwen4exp_mtp_rope_theta.unwrap_or(c.rope_theta),
                vision: c.qwen4exp_vision.clone(),
            };
            // ONE-indexed contract + divisibility checks run at parse, not first use.
            let ple_layers = cfg.ple_checkpoint_layers();
            for &il in &ple_layers {
                assert!(
                    il < c.num_hidden_layers,
                    "qwen4_exp ple layer {il} out of trunk range {}",
                    c.num_hidden_layers
                );
                // PLE is only valid on linear-attention layers (transformers validate).
                assert!(
                    (il + 1) % interval != 0,
                    "qwen4_exp ple layer {il} lands on a full-attention layer"
                );
            }
            // PLE token history pads with EOS and resets segments at it — a defaulted id
            // would be a silently different hash program (SEMANTICS.md §PLE).
            assert!(
                ple_layers.is_empty() || cfg.eos_token_id.is_some(),
                "qwen4_exp config.json missing required field eos_token_id (PLE layers present)"
            );
            let _ = cfg.ngram_head_embed_dim();
            let _ = cfg.indexer_budget_blocks();
            Some(cfg)
        } else {
            None
        };
        // YaRN long-context scaling — parsed for THIS family only (see ModelConfig::rope_yarn
        // scope note); other arches keep their existing rope-scaling handling untouched.
        let rope_yarn = if qwen4exp.is_some() {
            qwen4exp_rope_yarn(c)
        } else {
            None
        };

        // glm5_next: every field below feeds the forward pass; a missing one means a DIFFERENT
        // semantic program, so refuse by config.json field path instead of defaulting
        // (from_gguf's expect style, dsv4's message format).
        let glm5 = if arch.is_glm5_next() {
            let req_u = |v: Option<u32>, k: &str| {
                v.unwrap_or_else(|| panic!("glm5_next config.json missing required field {k}"))
            };
            let req_f = |v: Option<f32>, k: &str| {
                v.unwrap_or_else(|| panic!("glm5_next config.json missing required field {k}"))
            };
            let req_b = |v: Option<bool>, k: &str| {
                v.unwrap_or_else(|| panic!("glm5_next config.json missing required field {k}"))
            };
            let req_s = |v: &Option<String>, k: &str| {
                v.clone()
                    .unwrap_or_else(|| panic!("glm5_next config.json missing required field {k}"))
            };
            let n_trunk = c.num_hidden_layers as usize;
            let layer_types = c.layer_types.as_ref().unwrap_or_else(|| {
                panic!("glm5_next config.json missing required field layer_types")
            });
            assert_eq!(
                layer_types.len(),
                n_trunk,
                "glm5_next layer_types length {} != num_hidden_layers {n_trunk}",
                layer_types.len()
            );
            let kda_layer: Vec<bool> = layer_types
                .iter()
                .map(|kind| match kind.as_str() {
                    "linear_attention" => true,
                    "deepseek_sparse_attention" => false,
                    other => panic!("glm5_next unknown layer_types entry {other}"),
                })
                .collect();
            // layer_types and linear_attn_config carry the SAME schedule twice; a disagreement
            // means a corrupted or foreign config, so refuse rather than pick a winner.
            let kda_list = c.glm_kda_layers.as_ref().unwrap_or_else(|| {
                panic!("glm5_next config.json missing required field linear_attn_config.kda_layers")
            });
            let full_list = c.glm_full_attn_layers.as_ref().unwrap_or_else(|| {
                panic!(
                    "glm5_next config.json missing required field \
                     linear_attn_config.full_attn_layers"
                )
            });
            for (il, &kda) in kda_layer.iter().enumerate() {
                let il32 = il as u32;
                assert_eq!(
                    kda,
                    kda_list.contains(&il32),
                    "glm5_next layer {il}: layer_types says {} but \
                     linear_attn_config.kda_layers disagrees",
                    layer_types[il]
                );
                assert_eq!(
                    !kda,
                    full_list.contains(&il32),
                    "glm5_next layer {il}: layer_types says {} but \
                     linear_attn_config.full_attn_layers disagrees",
                    layer_types[il]
                );
            }
            let mlp_layer_types = c.mlp_layer_types.as_ref().unwrap_or_else(|| {
                panic!("glm5_next config.json missing required field mlp_layer_types")
            });
            assert_eq!(
                mlp_layer_types.len(),
                n_trunk,
                "glm5_next mlp_layer_types length {} != num_hidden_layers {n_trunk}",
                mlp_layer_types.len()
            );
            let dense_layer: Vec<bool> = mlp_layer_types
                .iter()
                .map(|kind| match kind.as_str() {
                    "dense" => true,
                    "sparse" => false,
                    other => panic!("glm5_next unknown mlp_layer_types entry {other}"),
                })
                .collect();
            let first_k_dense_replace = req_u(c.first_k_dense_replace, "first_k_dense_replace");
            for (il, &dense) in dense_layer.iter().enumerate() {
                assert_eq!(
                    dense,
                    (il as u32) < first_k_dense_replace,
                    "glm5_next layer {il}: mlp_layer_types disagrees with \
                     first_k_dense_replace {first_k_dense_replace}"
                );
            }
            let indexer_types = c.indexer_types.clone().unwrap_or_else(|| {
                panic!("glm5_next config.json missing required field indexer_types")
            });
            assert_eq!(
                indexer_types.len(),
                n_trunk,
                "glm5_next indexer_types length {} != num_hidden_layers {n_trunk}",
                indexer_types.len()
            );
            let qk_nope_head_dim = req_u(c.qk_nope_head_dim, "qk_nope_head_dim");
            let qk_rope_head_dim = req_u(c.qk_rope_head_dim, "qk_rope_head_dim");
            let qk_head_dim = req_u(c.qk_head_dim, "qk_head_dim");
            assert_eq!(
                qk_head_dim,
                qk_nope_head_dim + qk_rope_head_dim,
                "glm5_next qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim"
            );
            let mla_use_nope = req_b(c.mla_use_nope, "mla_use_nope");
            assert!(
                !mla_use_nope || qk_rope_head_dim == 0,
                "glm5_next mla_use_nope requires qk_rope_head_dim == 0, got {qk_rope_head_dim}"
            );
            let index_topk = req_u(c.index_topk, "index_topk");
            let index_kpool = req_u(c.index_kpool, "index_kpool");
            assert!(
                index_kpool > 0 && index_topk % index_kpool == 0,
                "glm5_next index_topk {index_topk} must be divisible by index_kpool {index_kpool}"
            );
            Some(Glm5NextConfig {
                kda_layer,
                dense_layer,
                indexer_types,
                linear_num_heads: req_u(c.glm_linear_num_heads, "linear_attn_config.num_heads"),
                linear_head_dim: req_u(c.glm_linear_head_dim, "linear_attn_config.head_dim"),
                linear_conv_kernel: req_u(
                    c.glm_linear_short_conv,
                    "linear_attn_config.short_conv_kernel_size",
                ),
                gate_lower_bound: req_f(
                    c.glm_gate_lower_bound,
                    "linear_attn_config.gate_lower_bound",
                ),
                q_lora_rank: req_u(c.q_lora_rank, "q_lora_rank"),
                kv_lora_rank: req_u(c.kv_lora_rank, "kv_lora_rank"),
                qk_head_dim,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim: req_u(c.v_head_dim, "v_head_dim"),
                mla_use_nope,
                index_n_heads: req_u(c.index_n_heads, "index_n_heads"),
                index_head_dim: req_u(c.index_head_dim, "index_head_dim"),
                index_topk,
                index_kpool,
                index_kpool_always_select_tail: req_b(
                    c.index_kpool_always_select_tail,
                    "index_kpool_always_select_tail",
                ),
                index_kpool_compress: req_b(c.index_kpool_compress, "index_kpool_compress"),
                indexer_rope_interleave: req_b(
                    c.indexer_rope_interleave,
                    "indexer_rope_interleave",
                ),
                index_share_for_mtp_iteration: req_b(
                    c.index_share_for_mtp_iteration,
                    "index_share_for_mtp_iteration",
                ),
                n_routed_experts: req_u(c.num_experts, "n_routed_experts"),
                num_experts_per_tok: req_u(c.num_experts_per_tok, "num_experts_per_tok"),
                moe_intermediate_size: req_u(c.moe_intermediate_size, "moe_intermediate_size"),
                n_shared_experts: req_u(c.n_shared_experts, "n_shared_experts"),
                first_k_dense_replace,
                scoring_func: req_s(&c.scoring_func, "scoring_func"),
                topk_method: req_s(&c.topk_method, "topk_method"),
                routed_scaling_factor: req_f(c.routed_scaling_factor, "routed_scaling_factor"),
                norm_topk_prob: req_b(c.norm_topk_prob, "norm_topk_prob"),
                moe_router_dtype: req_s(&c.moe_router_dtype, "moe_router_dtype"),
                mhc: req_b(c.mhc, "mhc"),
                hc_mult: req_u(c.hc_mult, "hc_mult"),
                hc_eps: req_f(c.hc_eps, "hc_eps"),
                hc_sinkhorn_iters: req_u(c.hc_sinkhorn_iters, "hc_sinkhorn_iters"),
                swiglu_limit: req_f(c.swiglu_limit, "swiglu_limit"),
                num_nextn_predict_layers: req_u(
                    c.num_nextn_predict_layers,
                    "num_nextn_predict_layers",
                ),
                // Not in the banked config.json; the reference gated norm hardcodes sigmoid.
                // Read the key when a checkpoint pins it, default "sigmoid" otherwise.
                output_gate_type: c
                    .output_gate_type
                    .clone()
                    .unwrap_or_else(|| "sigmoid".to_string()),
            })
        } else {
            None
        };
        let n_layer = c.num_hidden_layers + nextn;
        let full_attention_interval = c.full_attention_interval.unwrap_or(0);
        // Rotary width, resolved ONCE for this config and reused by both consumers below (the
        // geometry table and the scalar field). Two call sites deriving it independently is how
        // the geometry table and `rope_dim_count` were free to disagree.
        //
        // gemma-4 is deliberately excluded from the fraction arm: its partial rotary is expressed
        // as `rope_freqs` FREQ FACTORS over the FULL head_dim (1.0 for the first fraction of dim
        // pairs, ~1e30 beyond, so the tail rotates by ~0 — see the rope_freqs synthesis note in
        // hf_mapping.rs), NOT as a truncated n_rot. Its factor also lives under a different key
        // (`rope_parameters.full_attention.partial_rotary_factor` -> gemma4_partial_rotary_global),
        // so this is belt-and-braces against a future checkpoint hoisting the key to the top level.
        let rope_dim_count = resolve_rope_dim_count(
            c.rotary_dim,
            if is_gemma4 {
                None
            } else {
                c.partial_rotary_factor
            },
            head_dim_k,
        );
        let geometry = match &arch {
            // qwen4_exp reuses the interval-hybrid rule verbatim (see the table fn note).
            Arch::Qwen35 | Arch::Qwen35Moe | Arch::Qwen4Exp => Some(ArchGeometryTable::qwen35(
                n_layer,
                nextn,
                full_attention_interval,
                n_head,
                n_head_kv,
                head_dim_k,
                head_dim_v,
                rope_dim_count,
                c.rope_theta,
            )),
            Arch::Step35 => Some(ArchGeometryTable::step35(
                n_layer,
                head_dim_k,
                head_dim_v,
                step35
                    .as_ref()
                    .expect("step35 geometry needs step35 config"),
            )),
            _ => None,
        };
        // Step's published config omits this key because its configuration class defaults to
        // 1e-5. The generic HF fallback is 1e-6, so absence must be architecture-specific.
        let rms_eps = if arch.is_step35() && !c.rms_norm_eps_explicit {
            1e-5
        } else {
            c.rms_norm_eps
        };

        ModelConfig {
            arch,
            window_hint: c.sliding_window,
            rope_scaling_hint: c.rope_scaling_type.clone(),
            name: c.name.clone().unwrap_or_default(),
            // GGUF `block_count` INCLUDES the MTP/NextN block(s) (hybrid.rs n_trunk = n_layer -
            // nextn); HF `num_hidden_layers` EXCLUDES them. Add nextn so both sources agree.
            n_layer,
            n_embd: c.hidden_size,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_ff: c.intermediate_size,
            n_vocab: c.vocab_size,
            context_length: c.max_position_embeddings,
            rms_eps,
            rope_freq_base: c.rope_theta,
            // partial RoPE: M3 spells it `rotary_dim` (64 of head_dim 128), the Qwen3.5 family
            // spells it `partial_rotary_factor` (0.25 of head_dim 256 = 64). Both resolved above,
            // through the same function the GGUF reader uses.
            rope_dim_count,
            // mrope sections are STORED for arches that declare them (qwen4_exp [11,11,10]);
            // text-only positions degenerate to plain partial rope (SEMANTICS.md §Rope).
            rope_sections: c
                .mrope_section
                .as_ref()
                .map(|s| s.iter().map(|&x| x as i32).collect())
                .unwrap_or_default(),
            full_attention_interval,
            ssm,
            moe,
            m3,
            hy3,
            gemma4: if is_gemma4 {
                // Safetensors route (lane/gemma-vision native arc): derive the per-layer
                // geometry the GGUF metadata carries pre-baked. layer_types is the truth
                // for the 5:1 SWA:global pattern; global layers use global_head_dim +
                // num_global_key_value_heads (and K=V — the mapping arm's v_proj miss).
                let swa_pattern: Vec<bool> = c.gemma4_swa_pattern.clone().unwrap_or_default();
                let kv_swa = n_head_kv;
                let kv_global = c.num_global_key_value_heads.unwrap_or(kv_swa);
                let head_count_kv = swa_pattern
                    .iter()
                    .map(|&swa| if swa { kv_swa } else { kv_global })
                    .collect();
                let gdim = c.global_head_dim.unwrap_or(head_dim_k);
                Some(Gemma4Config {
                    head_count_kv,
                    swa_pattern,
                    sliding_window: c.sliding_window.unwrap_or(1024),
                    key_length_global: gdim,
                    key_length_swa: head_dim_k,
                    rope_base_global: c.gemma4_rope_theta_global.unwrap_or(1e6),
                    rope_base_swa: c.gemma4_rope_theta_swa.unwrap_or(1e4),
                    rope_dims_global: gdim,
                    rope_dims_swa: head_dim_k,
                    final_logit_softcapping: c.final_logit_softcapping.unwrap_or(30.0),
                    partial_rotary_global: c.gemma4_partial_rotary_global.unwrap_or(0.25),
                    n_embd_per_layer: c.hidden_size_per_layer_input.unwrap_or(0),
                    shared_kv_layers: c.num_kv_shared_layers.unwrap_or(0),
                    // HF ships no suppress list for the 31B (GGUF twin: key ABSENT too —
                    // verified 2026-08-16); parity holds with the empty set.
                    suppress_tokens: Vec::new(),
                })
            } else {
                None
            },
            // qwen4_exp: the generic VisionConfig parse reads gemma key names, so it would
            // fabricate a wrong tower geometry from this file's vision_config. The faithful
            // geometry lives in qwen4exp.vision; the tower itself side-loads at serving
            // (MEMRA_VISION_DIR, the qwen3_5 pattern) — never a plan-level encoder here.
            vision: if qwen4exp.is_some() {
                None
            } else {
                c.vision.clone()
            },
            vision_glm5: c.vision_glm5.clone(),
            multimodal: match (c.image_token_id, c.vision_soft_tokens_per_image) {
                (Some(image_token_id), Some(vision_soft_tokens_per_image)) => {
                    Some(MultimodalConfig {
                        image_token_id,
                        vision_soft_tokens_per_image,
                    })
                }
                _ => None,
            },
            mla: None, // GGUF-first arch (glm-dsa): HF/safetensors import is a later arc
            step35,
            dsv4,
            qwen4exp,
            rope_yarn,
            glm5,
            geometry,
            // NextN/MTP depth: 35B-MoE HF uses `num_nextn_predict_layers`; the 27B (dense hybrid)
            // uses `mtp_num_hidden_layers` (NVIDIA + local text ckpts) — same meaning, both = 1.
            // `nextn` above equals that expression for every arch EXCEPT deepseek_v4, where the
            // key is vestigial and the depth derives from compress_ratios (0731 trap).
            nextn_predict_layers: nextn,
            n_layer_total: c.num_hidden_layers + nextn,
        }
    }

    /// Read + parse an HF `config.json` directly from disk and build a ModelConfig.
    pub fn from_config_json(path: &std::path::Path) -> std::io::Result<Self> {
        let txt = std::fs::read_to_string(path)?;
        let cfg = HfConfig::parse(&txt);
        Ok(Self::from_hf(&cfg))
    }

    /// Classify a layer index. For hybrid models, layer il is full-attention when
    /// (il+1) % full_attention_interval == 0, else linear-attention (matches llama.cpp qwen35).
    /// Non-hybrid models are always full-attention.
    #[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
    pub fn layer_kind(&self, il: u32) -> LayerKind {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.mixer;
        }
        if self.full_attention_interval == 0 {
            return LayerKind::FullAttention;
        }
        if (il + 1) % self.full_attention_interval == 0 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        }
    }

    /// Count of full-attention layers (the ones that carry a growing KV cache).
    pub fn n_full_attn_layers(&self) -> u32 {
        (0..self.n_layer)
            .filter(|&il| self.layer_kind(il) == LayerKind::FullAttention)
            .count() as u32
    }

    /// qwen35-class FUSED [q|gate] attention output gate: wq packs q AND a per-head sigmoid gate
    /// (out = 2*n_head*head_dim) that `q_gate_split` separates. M3 and Hy3 have NO output gate —
    /// their wq out is exactly n_head*head_dim, and running the split would read 2x out of bounds.
    /// One predicate so every full-attn site (prefill/prime/decode/dc/spec) agrees.
    ///
    /// step35 is NOT in this class even though it HAS a head-wise gate: its gate is a separate
    /// `blk.N.attn_gate.weight [n_embd, n_head]` tensor (one scalar per head, broadcast over
    /// head_dim) and its wq out is exactly n_head*head_dim — see `attn_gate_separate()`. Running
    /// the fused split on it would read 2x out of bounds, which is why this deny-list must name it.
    pub fn attn_out_gate(&self) -> bool {
        if let Some(table) = self.geometry.as_ref() {
            return table
                .classes()
                .iter()
                .any(|geometry| geometry.attention_gate == AttentionGateKind::FusedQ);
        }
        match self.arch.attention_gate_kind() {
            Some(kind) => kind == AttentionGateKind::FusedQ,
            // Unregistered arch: NEVER inherit FusedQ. `q_gate_split` would read 2x past the end
            // of a wq whose out-features are n_head*head_dim. The hybrid loader refuses an
            // unregistered arch outright (`validate_attention_gate_layout`); this is the
            // belt-and-braces answer for any caller that reads the predicate first.
            None => false,
        }
    }

    /// Refuse a model whose architecture declares no attention-gate layout.
    ///
    /// Called from the hybrid loader before any tensor is split, so an unregistered arch is a
    /// clean typed load error instead of a `q_gate_split` that reads out of bounds. Registered
    /// arches — every `Arch` variant but `Other` — always pass.
    pub fn validate_attention_gate_layout(
        &self,
    ) -> Result<AttentionGateKind, UndeclaredGateLayout> {
        if let Some(table) = self.geometry.as_ref() {
            // A migrated arch declares per-layer; the whole-model answer is the strongest class.
            let classes = table.classes();
            if classes
                .iter()
                .any(|g| g.attention_gate == AttentionGateKind::FusedQ)
            {
                return Ok(AttentionGateKind::FusedQ);
            }
            if classes
                .iter()
                .any(|g| g.attention_gate == AttentionGateKind::SeparateHead)
            {
                return Ok(AttentionGateKind::SeparateHead);
            }
            return Ok(AttentionGateKind::None);
        }
        self.arch
            .attention_gate_kind()
            .ok_or_else(|| UndeclaredGateLayout {
                arch: format!("{:?}", self.arch),
            })
    }

    /// step35-class SEPARATE head-wise attention gate: `blk.N.attn_gate.weight [n_embd, n_head_l]`
    /// yields one pre-sigmoid scalar PER HEAD, broadcast across head_dim over the attention output
    /// before wo (upstream `step35.cpp:267-285`: `attn_out * sigmoid(g_proj(attn_norm_out))`).
    /// Distinct from `attn_out_gate()` (fused-in-wq, per-DIM) — the two are mutually exclusive.
    /// Note the gate input is the POST-attn_norm hidden state (`cur`), not the raw residual.
    pub fn attn_gate_separate(&self) -> bool {
        if let Some(table) = self.geometry.as_ref() {
            return table
                .classes()
                .iter()
                .any(|geometry| geometry.attention_gate == AttentionGateKind::SeparateHead);
        }
        // Same shape as `attn_out_gate()`: an explicit per-arch declaration, and an unregistered
        // arch gets `false` rather than a guess (loading it is refused before this is consulted).
        self.arch.attention_gate_kind() == Some(AttentionGateKind::SeparateHead)
    }

    /// Geometry row for a migrated architecture. `None` means the caller must use the existing
    /// architecture path; an out-of-range layer is never fabricated from another row.
    pub fn layer_geometry(&self, il: u32) -> Option<LayerGeometry> {
        self.geometry.as_ref()?.layer(il)
    }

    /// Resolve geometry for a full-attention execution arm. Migrated architectures read their
    /// declarative row; legacy architectures receive the exact scalar geometry they used before.
    pub fn full_attention_geometry_at(&self, il: u32) -> LayerGeometry {
        self.layer_geometry(il).unwrap_or(LayerGeometry {
            mixer: LayerKind::FullAttention,
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            head_dim_k: self.head_dim_k,
            head_dim_v: self.head_dim_v,
            n_rot: self.rope_dim_count,
            rope_base: self.rope_freq_base,
            window: None,
            rope_factors: false,
            attention_gate: self
                .arch
                .attention_gate_kind()
                .unwrap_or(AttentionGateKind::None),
        })
    }

    pub fn attn_out_gate_at(&self, il: u32) -> bool {
        self.layer_geometry(il)
            .map(|geometry| geometry.attention_gate == AttentionGateKind::FusedQ)
            .unwrap_or_else(|| self.attn_out_gate())
    }

    pub fn attn_gate_separate_at(&self, il: u32) -> bool {
        self.layer_geometry(il)
            .map(|geometry| geometry.attention_gate == AttentionGateKind::SeparateHead)
            .unwrap_or_else(|| self.attn_gate_separate())
    }

    /// DeepSeek-V3-class sigmoid routing knobs, arch-agnostic: `Some((scaling_factor, route_norm))`
    /// when the router scores with sigmoid (+ optional selection bias via `exp_probs_b`), `None`
    /// for the softmax archs. route_norm: sum-normalize the selected weights before scaling
    /// (true for M3 — its modeling code always normalizes — and for Hy3's `route_norm=true`).
    /// Sites that must NOT enter the fused SOFTMAX device-router arms key off `is_some()`.
    pub fn sigmoid_router(&self) -> Option<(f32, bool)> {
        // glm5_next (`noaux_tc`): sigmoid scores, selection-only `e_score_correction_bias`,
        // sum-normalize the selected weights (`norm_topk_prob`), then x `routed_scaling_factor`
        // (2.5) — the same DeepSeek-V3 recipe, under this family's own key names. Unconditional
        // on purpose: `model_plan::router` emits `RouterPlan::Sigmoid` for every glm5_next MoE
        // layer without consulting `scoring_func`, and an accessor that disagreed with the plan
        // is exactly the bug this arm closes (2026-08-28) — with no arm here the accessor
        // answered None, every `sigmoid_router().is_none()` dispatch predicate fired, and the
        // routed branch silently rode the SOFTMAX router: softmax scores instead of sigmoid,
        // no selection bias, and weights summing to 1 instead of 2.5.
        if let Some(g5) = self.glm5.as_ref() {
            return Some((g5.routed_scaling_factor, g5.norm_topk_prob));
        }
        if let Some(m3) = self.m3.as_ref()
            && m3.sigmoid_routing
        {
            return Some((m3.routed_scaling_factor, true));
        }
        if let Some(hy3) = self.hy3.as_ref()
            && hy3.sigmoid_routing
        {
            return Some((hy3.router_scaling_factor, hy3.route_norm));
        }
        // glm-dsa: sigmoid + noaux_tc selection bias (exp_probs_b) + routed_scaling 2.5,
        // norm_topk_prob=true — the exact DeepSeek-V3 recipe M3/Hy3 already ride.
        if let Some(mla) = self.mla.as_ref()
            && mla.sigmoid_routing
        {
            return Some((mla.routed_scaling_factor, mla.route_norm));
        }
        // step35: sigmoid + exp_probs_b selection bias + expert_weights_scale 3.0 +
        // expert_weights_norm true — the same DeepSeek-V3 recipe, different key names.
        if let Some(s) = self.step35.as_ref()
            && s.sigmoid_routing
        {
            return Some((s.routed_scaling_factor, s.route_norm));
        }
        None
    }

    /// Per-layer query-head count. Global scalar for every arch except step35, whose
    /// `attention.head_count` is an array (64 on full-attn layers, 96 on SWA). Sites that build
    /// wq/wo/attn_gate shapes or size per-head loops MUST use this, not the `n_head` field.
    pub fn n_head_at(&self, il: u32) -> u32 {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.n_head;
        }
        match self.step35.as_ref() {
            Some(s) => s.n_head(il),
            None => self.n_head,
        }
    }

    /// Per-layer KV-head count. gemma4 carries a per-layer array; step35's is uniform-8 but is
    /// serialized as an array. Every other arch is the global scalar.
    pub fn n_head_kv_at(&self, il: u32) -> u32 {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.n_head_kv;
        }
        if let Some(g) = self.gemma4.as_ref()
            && let Some(&n) = g.head_count_kv.get(il as usize)
        {
            return n;
        }
        if let Some(s) = self.step35.as_ref() {
            return s.n_head_kv(il);
        }
        self.n_head_kv
    }

    /// True when layer `il` uses sliding-window attention. gemma4 and step35 carry an explicit
    /// per-layer pattern; every other arch is unwindowed (returns false).
    pub fn is_swa_at(&self, il: u32) -> bool {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.window.is_some();
        }
        if let Some(g) = self.gemma4.as_ref() {
            return g.swa_pattern.get(il as usize).copied().unwrap_or(false);
        }
        if let Some(s) = self.step35.as_ref() {
            return s.is_swa(il);
        }
        false
    }

    /// Per-layer ROUTED-expert SwiGLU clamp, `None` when the arch/layer has none.
    /// step35 (`swiglu_clamp_exp`, live on layers 43-44 of 3.7-Flash) carries the POST form;
    /// glm5_next clamps every layer in the PRE form. The two are not interchangeable — see
    /// [`SwigluClamp`].
    pub fn clamp_exp_at(&self, il: u32) -> Option<SwigluClamp> {
        if let Some(g5) = self.glm5.as_ref() {
            return SwigluClamp::pre_if_live(g5.swiglu_limit);
        }
        self.step35
            .as_ref()
            .and_then(|s| s.clamp_exp(il))
            .map(SwigluClamp::Post)
    }

    /// Per-layer SHARED/DENSE-MLP SwiGLU clamp (step35 `swiglu_clamp_shexp`; glm5_next applies
    /// its single `swiglu_limit` to the dense MLP and the shared expert alike, both being
    /// `Glm5NextTextMLP` in the reference module).
    pub fn clamp_shexp_at(&self, il: u32) -> Option<SwigluClamp> {
        if let Some(g5) = self.glm5.as_ref() {
            return SwigluClamp::pre_if_live(g5.swiglu_limit);
        }
        self.step35
            .as_ref()
            .and_then(|s| s.clamp_shexp(il))
            .map(SwigluClamp::Post)
    }

    /// True when ANY FFN branch on layer `il` needs a clamped SwiGLU form. This is the
    /// FUSED-EPILOGUE DENY predicate: memra's fused SiLU epilogues (grouped-decode,
    /// moe_pairs_silu_mul, the dev-path kernels) hardcode plain `silu(gate)*up`, so a layer
    /// with a live clamp must fall through to the unfused `ffn_act_*` seam. Substituting the
    /// plain form compiles, runs, and produces plausible-but-wrong logits — exactly the
    /// failure mode `swiglu_clamped_mul_scaled_f32`'s kernel-check cell guards against.
    /// True for EVERY glm5_next layer, not just a two-layer tail as on step35.
    pub fn swiglu_clamped_at(&self, il: u32) -> bool {
        self.clamp_exp_at(il).is_some() || self.clamp_shexp_at(il).is_some()
    }

    /// True when the model has a live SwiGLU clamp on ANY layer — the cheap whole-model
    /// question the no-`il` `ffn_act` seam asserts against (a clamped model reaching a seam
    /// that cannot see `il` means the caller has to be migrated to `ffn_act_exp`/`_shexp`).
    pub fn swiglu_clamped_anywhere(&self) -> bool {
        if self
            .glm5
            .as_ref()
            .is_some_and(|g5| SwigluClamp::pre_if_live(g5.swiglu_limit).is_some())
        {
            return true;
        }
        self.step35.as_ref().is_some_and(|s| {
            s.swiglu_clamp_exp
                .iter()
                .chain(s.swiglu_clamp_shexp.iter())
                .any(|&l| l > 1e-6)
        })
    }
}

/// A live SwiGLU clamp, carrying its LIMIT and its FORM together. The two forms disagree
/// numerically wherever `gate > limit`, so no dispatch site may hold a bare `f32` limit and
/// pick a form by default — every consumer matches this enum exhaustively, which makes a new
/// arch a compile error rather than a silent substitution.
///
/// * `Post` — step35: `min(silu(gate*gs), limit) * clamp(up*us, ±limit)`. The clamp lands on the
///   silu OUTPUT (upper bound only). llama.cpp `llama-graph.cpp:2146-2165` / `:1751-1770`,
///   non-DEEPSEEK4 branch. Kernel: `swiglu_clamped_mul_scaled_f32`.
/// * `Pre` — glm5_next: `silu(min(gate*gs, limit)) * clamp(up*us, ±limit)`. The gate is clamped
///   BEFORE silu and one-sided (no lower bound). Reference `Glm5NextTextMLP.forward` and
///   `Glm5NextTextExperts._apply_gate`. Kernel: `swiglu_preclamped_mul_scaled_f32`.
///
/// Deliberately without a `limit()` accessor: a bare-f32 escape hatch is exactly the hole this
/// type exists to close.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwigluClamp {
    Post(f32),
    Pre(f32),
}

impl SwigluClamp {
    /// `Pre(limit)` above upstream's `> 1e-6` eps gate, `None` at or below it. At limit=0 a
    /// clamp kernel would zero every positive activation, so the plain silu path is the correct
    /// one there.
    fn pre_if_live(limit: f32) -> Option<Self> {
        (limit > 1e-6).then_some(Self::Pre(limit))
    }
}

/// Subset of HF `config.json` fields memra needs. Defaults mirror GGUF fallbacks. Hybrid models
/// (qwen3_5) nest the transformer fields under `text_config` — `parse` flattens that automatically.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub model_type: String,
    pub name: Option<String>,
    pub num_hidden_layers: u32,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rms_norm_eps_explicit: bool,
    pub rope_theta: f32,
    /// Fraction of `head_dim` that rotates — the Qwen3.5-family spelling of partial RoPE, read
    /// from top-level `partial_rotary_factor` OR `rope_parameters.partial_rotary_factor` (real
    /// checkpoints carry it in one or both places). Resolved into `n_rot` by
    /// `resolve_rope_dim_count`; `None` means the config declares no partial rotary.
    ///
    /// NOT the gemma-4 key: gemma-4 nests its factor under `rope_parameters.full_attention` and
    /// means something different by it (rope_freqs factors over the full width, not a truncated
    /// n_rot), so it lands in `gemma4_partial_rotary_global` instead.
    pub partial_rotary_factor: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context: Option<u32>,
    pub rope_scaling_low_freq_factor: Option<f32>,
    pub rope_scaling_high_freq_factor: Option<f32>,
    /// Yarn keys the qwen4_exp twin refuses rather than mis-scales (parsed so the refusal
    /// can be loud): explicit attention_factor / mscale / mscale_all_dim / truncate.
    pub rope_scaling_attention_factor: Option<f32>,
    pub rope_scaling_mscale: Option<f32>,
    pub rope_scaling_mscale_all_dim: Option<f32>,
    pub rope_scaling_truncate: Option<bool>,
    pub full_attention_interval: Option<u32>,
    // ---- gemma-4 (Arch::Gemma4 safetensors route, lane/gemma-vision) ----
    /// layer_types as swa flags (true = "sliding_attention"), parsed in apply().
    pub gemma4_swa_pattern: Option<Vec<bool>>,
    pub global_head_dim: Option<u32>,
    pub num_global_key_value_heads: Option<u32>,
    pub sliding_window: Option<u32>,
    pub final_logit_softcapping: Option<f32>,
    /// rope_parameters.{full_attention,sliding_attention}.rope_theta, flattened in apply().
    pub gemma4_rope_theta_global: Option<f32>,
    pub gemma4_rope_theta_swa: Option<f32>,
    /// rope_parameters.full_attention.partial_rotary_factor (rope_freqs synthesis input).
    pub gemma4_partial_rotary_global: Option<f32>,
    pub hidden_size_per_layer_input: Option<u32>,
    pub num_kv_shared_layers: Option<u32>,
    pub vision: Option<VisionConfig>,
    /// glm5_next tower (`vision_config.model_type == "glm5_next_vision"`); when this is
    /// Some, `vision` stays None — the two structs are different semantic programs.
    pub vision_glm5: Option<Glm5VisionConfig>,
    pub image_token_id: Option<u32>,
    pub vision_soft_tokens_per_image: Option<u32>,
    pub num_nextn_predict_layers: Option<u32>,
    pub mtp_num_hidden_layers: Option<u32>, // qwen3_5/3_6 HF key for the MTP head depth (27B: 1)
    // MoE
    pub num_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub moe_intermediate_size: Option<u32>,
    pub expert_hidden_dim: Option<u32>,
    pub shared_expert_intermediate_size: Option<u32>,
    // hybrid linear-attn (qwen3_5 text_config)
    pub linear_conv_kernel_dim: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
    // ---- MiniMax-M3 (minimax_m3_vl text_config) ----
    pub num_local_experts: Option<u32>, // M3 name for expert_count
    pub dense_intermediate_size: Option<u32>, // layers 0..2 dense FFN (12288)
    pub shared_intermediate_size: Option<u32>, // shared expert FF (3072)
    pub n_shared_experts: Option<u32>,
    pub rotary_dim: Option<u32>, // partial RoPE (64 of head_dim 128)
    pub use_gemma_norm: Option<bool>,
    pub scoring_func: Option<String>,       // "sigmoid"
    pub routed_scaling_factor: Option<f32>, // 2.0
    pub use_routing_bias: Option<bool>,
    pub swiglu_alpha: Option<f32>, // swigluoai clamp params
    pub swiglu_limit: Option<f32>,
    pub moe_layer_freq: Option<Vec<u32>>, // per-layer 0=dense 1=moe
    // ---- Hy3 (`hy_v3`) ----
    pub first_k_dense_replace: Option<u32>,
    pub moe_router_use_sigmoid: Option<bool>,
    pub moe_router_enable_expert_bias: Option<bool>,
    pub route_norm: Option<bool>,
    pub router_scaling_factor: Option<f32>,
    pub qk_norm: Option<bool>,
    pub hidden_act: Option<String>,
    // ---- DeepSeek-V4 (`deepseek_v4`) ----
    pub topk_method: Option<String>,
    pub norm_topk_prob: Option<bool>,
    pub num_hash_layers: Option<u32>,
    pub hc_eps: Option<f32>,
    pub hc_mult: Option<u32>,
    pub hc_sinkhorn_iters: Option<u32>,
    pub q_lora_rank: Option<u32>,
    pub qk_rope_head_dim: Option<u32>,
    pub o_lora_rank: Option<u32>,
    pub o_groups: Option<u32>,
    pub index_n_heads: Option<u32>,
    pub index_head_dim: Option<u32>,
    pub index_topk: Option<u32>,
    pub compress_ratios: Option<Vec<u32>>,
    pub compress_rope_theta: Option<f32>,
    // ---- GLM-5.3-Flash (`glm5_next` text_config) ----
    pub kv_lora_rank: Option<u32>,
    pub qk_head_dim: Option<u32>,
    pub qk_nope_head_dim: Option<u32>,
    pub v_head_dim: Option<u32>,
    pub mla_use_nope: Option<bool>,
    pub index_kpool: Option<u32>,
    pub index_kpool_always_select_tail: Option<bool>,
    pub index_kpool_compress: Option<bool>,
    pub indexer_rope_interleave: Option<bool>,
    pub index_share_for_mtp_iteration: Option<bool>,
    pub indexer_types: Option<Vec<String>>,
    pub mlp_layer_types: Option<Vec<String>>,
    pub moe_router_dtype: Option<String>,
    // output_gate_type is declared once further down (both qwen4_exp and glm5_next read it).
    pub mhc: Option<bool>,
    /// `linear_attn_config` nested object, flattened in apply() (the rope_parameters pattern).
    pub glm_linear_num_heads: Option<u32>,
    pub glm_linear_head_dim: Option<u32>,
    pub glm_linear_short_conv: Option<u32>,
    pub glm_gate_lower_bound: Option<f32>,
    pub glm_kda_layers: Option<Vec<u32>>,
    pub glm_full_attn_layers: Option<Vec<u32>>,
    /// rope_scaling.{factor, original_max_position_embeddings, beta_fast, beta_slow},
    /// flattened in apply() (yarn block; deepseek_v4 today).
    pub rope_yarn_factor: Option<f32>,
    pub rope_yarn_orig_ctx: Option<u32>,
    pub rope_yarn_beta_fast: Option<f32>,
    pub rope_yarn_beta_slow: Option<f32>,
    // ---- Qwen3.8-Flash-Next (`qwen4_exp` text_config) ----
    pub indexer_n_heads: Option<u32>,
    pub indexer_kv_heads: Option<u32>,
    pub indexer_head_dim: Option<u32>,
    pub indexer_compress_ratio: Option<u32>,
    pub indexer_budget: Option<u32>,
    pub hc_count: Option<u32>,
    pub hc_lowrank: Option<u32>,
    pub ngram_size: Option<u32>,
    pub heads_per_ngram: Option<u32>,
    pub ngram_vocab_size_base: Option<u64>,
    pub make_ngram_vocab_size_divisible_by: Option<u32>,
    pub split_ngram_parts: Option<u32>,
    pub ple_layer_ids: Option<Vec<u32>>,
    pub ple_embed_dim: Option<u32>,
    pub ple_conv_kernel_size: Option<u32>,
    pub output_gate_type: Option<String>,
    /// `eos_token_id`, scalar or first list entry (modular_qwen4_exp.py L621 takes
    /// `eos_token_id[0]` when it is a list). PLE hashes token history against it.
    pub eos_token_id: Option<u32>,
    /// rope_parameters.{mrope_section, mrope_interleaved}, flattened in apply().
    pub mrope_section: Option<Vec<u32>>,
    pub mrope_interleaved: Option<bool>,
    /// `mtp` sub-object {num_hidden_layers, rope_theta}, flattened in apply().
    pub qwen4exp_mtp_num_hidden_layers: Option<u32>,
    pub qwen4exp_mtp_rope_theta: Option<f32>,
    /// Parsed in `parse()` when vision_config.model_type == "qwen4_exp" (the generic
    /// VisionConfig parse reads gemma key names and does not speak this file).
    pub qwen4exp_vision: Option<Qwen4ExpVisionConfig>,
    // ---- Step-3.5 / Step-3.7-Flash (`step3p5` text config) ----
    pub attention_other_num_heads: Option<u32>,
    pub attention_other_num_groups: Option<u32>,
    pub layer_types: Option<Vec<String>>,
    pub rope_theta_layers: Option<Vec<f32>>,
    pub partial_rotary_factors: Option<Vec<f32>>,
    pub swiglu_limits: Option<Vec<f32>>,
    pub swiglu_limits_shared: Option<Vec<f32>>,
    pub moe_router_activation: Option<String>,
    pub moe_layers_enum: Option<String>,
    /// HF module prefixes that the checkpoint explicitly excludes from quantization.
    pub modules_to_not_convert: Vec<String>,
    /// Top-level `quantization_config.quant_algo`, retained so runtime dispatch can distinguish
    /// ModelOpt W4A16 from the activation-quantized NVFP4 program.
    pub quant_algo: Option<String>,
    /// The outer checkpoint contract requires every BF16 tensor to remain BF16. This is narrower
    /// than architecture identity: Step-3.5, base Hy3, and GGUF imports do not inherit a
    /// mixed-precision ModelOpt artifact's preservation rule.
    pub preserve_checkpoint_bf16: bool,
}

impl Default for HfConfig {
    fn default() -> Self {
        HfConfig {
            model_type: String::new(),
            name: None,
            num_hidden_layers: 0,
            hidden_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: None,
            head_dim: None,
            intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
            rms_norm_eps: 1e-6,
            rms_norm_eps_explicit: false,
            rope_theta: 10000.0,
            partial_rotary_factor: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rope_scaling_attention_factor: None,
            rope_scaling_mscale: None,
            rope_scaling_mscale_all_dim: None,
            rope_scaling_truncate: None,
            full_attention_interval: None,
            num_nextn_predict_layers: None,
            mtp_num_hidden_layers: None,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            expert_hidden_dim: None,
            shared_expert_intermediate_size: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            num_local_experts: None,
            dense_intermediate_size: None,
            shared_intermediate_size: None,
            n_shared_experts: None,
            rotary_dim: None,
            use_gemma_norm: None,
            scoring_func: None,
            routed_scaling_factor: None,
            use_routing_bias: None,
            swiglu_alpha: None,
            swiglu_limit: None,
            moe_layer_freq: None,
            first_k_dense_replace: None,
            moe_router_use_sigmoid: None,
            moe_router_enable_expert_bias: None,
            route_norm: None,
            router_scaling_factor: None,
            qk_norm: None,
            hidden_act: None,
            gemma4_swa_pattern: None,
            global_head_dim: None,
            num_global_key_value_heads: None,
            sliding_window: None,
            final_logit_softcapping: None,
            gemma4_rope_theta_global: None,
            gemma4_rope_theta_swa: None,
            gemma4_partial_rotary_global: None,
            hidden_size_per_layer_input: None,
            num_kv_shared_layers: None,
            vision: None,
            vision_glm5: None,
            image_token_id: None,
            vision_soft_tokens_per_image: None,
            topk_method: None,
            norm_topk_prob: None,
            num_hash_layers: None,
            hc_eps: None,
            hc_mult: None,
            hc_sinkhorn_iters: None,
            q_lora_rank: None,
            qk_rope_head_dim: None,
            o_lora_rank: None,
            o_groups: None,
            index_n_heads: None,
            index_head_dim: None,
            index_topk: None,
            compress_ratios: None,
            compress_rope_theta: None,
            kv_lora_rank: None,
            qk_head_dim: None,
            qk_nope_head_dim: None,
            v_head_dim: None,
            mla_use_nope: None,
            index_kpool: None,
            index_kpool_always_select_tail: None,
            index_kpool_compress: None,
            indexer_rope_interleave: None,
            index_share_for_mtp_iteration: None,
            indexer_types: None,
            mlp_layer_types: None,
            moe_router_dtype: None,
            mhc: None,
            glm_linear_num_heads: None,
            glm_linear_head_dim: None,
            glm_linear_short_conv: None,
            glm_gate_lower_bound: None,
            glm_kda_layers: None,
            glm_full_attn_layers: None,
            rope_yarn_factor: None,
            rope_yarn_orig_ctx: None,
            rope_yarn_beta_fast: None,
            rope_yarn_beta_slow: None,
            indexer_n_heads: None,
            indexer_kv_heads: None,
            indexer_head_dim: None,
            indexer_compress_ratio: None,
            indexer_budget: None,
            hc_count: None,
            hc_lowrank: None,
            ngram_size: None,
            heads_per_ngram: None,
            ngram_vocab_size_base: None,
            make_ngram_vocab_size_divisible_by: None,
            split_ngram_parts: None,
            ple_layer_ids: None,
            ple_embed_dim: None,
            ple_conv_kernel_size: None,
            output_gate_type: None,
            eos_token_id: None,
            mrope_section: None,
            mrope_interleaved: None,
            qwen4exp_mtp_num_hidden_layers: None,
            qwen4exp_mtp_rope_theta: None,
            qwen4exp_vision: None,
            attention_other_num_heads: None,
            attention_other_num_groups: None,
            layer_types: None,
            rope_theta_layers: None,
            partial_rotary_factors: None,
            swiglu_limits: None,
            swiglu_limits_shared: None,
            moe_router_activation: None,
            moe_layers_enum: None,
            modules_to_not_convert: Vec::new(),
            quant_algo: None,
            preserve_checkpoint_bf16: false,
        }
    }
}

impl HfConfig {
    fn llama3_rope_factors(&self, base: f32, n_dims: u32) -> Option<Vec<f32>> {
        if self.rope_scaling_type.as_deref() != Some("llama3") {
            return None;
        }
        let factor = self
            .rope_scaling_factor
            .expect("llama3 rope_scaling.factor missing");
        let original = self
            .rope_scaling_original_context
            .expect("llama3 rope_scaling.original_max_position_embeddings missing")
            as f32;
        let low = self
            .rope_scaling_low_freq_factor
            .expect("llama3 rope_scaling.low_freq_factor missing");
        let high = self
            .rope_scaling_high_freq_factor
            .expect("llama3 rope_scaling.high_freq_factor missing");
        assert!(
            factor >= 1.0 && original > 0.0 && low > 0.0 && high > low,
            "invalid llama3 rope_scaling values"
        );
        assert!(
            n_dims > 0 && n_dims.is_multiple_of(2),
            "llama3 rope dimension must be positive and even, got {n_dims}"
        );

        let low_wavelength = original / low;
        let high_wavelength = original / high;
        Some(
            (0..n_dims / 2)
                .map(|j| {
                    let inv_freq = base.powf(-((2 * j) as f32) / n_dims as f32);
                    let wavelength = std::f32::consts::TAU / inv_freq;
                    let scaled = if wavelength > low_wavelength {
                        inv_freq / factor
                    } else if wavelength < high_wavelength {
                        inv_freq
                    } else {
                        let smooth = (original / wavelength - low) / (high - low);
                        (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
                    };
                    inv_freq / scaled
                })
                .collect(),
        )
    }

    /// Parse an HF config.json. Reads scalar fields at the top level; if a `text_config`
    /// object is present (vision-language / hybrid wrappers like qwen3_5), its scalar fields
    /// override the top-level ones for the transformer config. `architectures[0]` and the
    /// top-level `model_type` seed the arch when `text_config.model_type` is more specific.
    pub fn parse(json: &str) -> Self {
        let top = JsonObj::parse(json);
        let mut cfg = HfConfig::default();
        cfg.apply(&top);
        cfg.quant_algo = top
            .object("quantization_config")
            .and_then(|quantization| quantization.string("quant_algo"));
        // Both official Step-3.7-Flash quantized artifacts keep everything OUTSIDE the routed
        // experts (attention, gates, shared experts, MTP, lm_head) as checkpoint BF16, and the
        // Step TP attention program requires those exact bytes. A ModelOpt Hy3 artifact likewise
        // uses physical BF16 to declare the deliberately unquantized half of a mixed profile.
        // Base checkpoints and unrelated formats keep the Q8_0 loader law.
        cfg.preserve_checkpoint_bf16 = top.object("quantization_config").is_some_and(|q| {
            (top.string("model_type").as_deref() == Some("step3p7")
                && ((q.string("quant_method").as_deref() == Some("fp8")
                    && q.string("activation_scheme").as_deref() == Some("dynamic")
                    && q.string("fmt").as_deref() == Some("e4m3")
                    && q.u32_array("weight_block_size").as_deref() == Some(&[128, 128]))
                    || (q.string("quant_method").as_deref() == Some("modelopt")
                        // W4A16_NVFP4 = the weight-only mint (glm5_next lane): identical
                        // weight/scale layout, no input_scale — same repack path.
                        && matches!(
                            q.string("quant_algo").as_deref(),
                            Some("NVFP4") | Some("W4A16_NVFP4")
                        ))))
                || (top.string("model_type").as_deref() == Some("hy_v3")
                    && matches!(
                        q.string("quant_method").as_deref(),
                        Some("modelopt" | "compressed-tensors")
                    ))
        });
        // qwen4_exp ViT tower: its own key spellings (depth / num_heads /
        // num_position_embeddings / spatial_merge_size / temporal_patch_size). Loud refusal
        // on a missing field — a defaulted tower geometry is a silently different program.
        cfg.qwen4exp_vision = top.object("vision_config").and_then(|vision| {
            (vision.string("model_type").as_deref() == Some("qwen4_exp")).then(|| {
                let req = |k: &str| {
                    vision.u32(k).unwrap_or_else(|| {
                        panic!("qwen4_exp vision_config missing required field {k}")
                    })
                };
                Qwen4ExpVisionConfig {
                    depth: req("depth"),
                    hidden_size: req("hidden_size"),
                    intermediate_size: req("intermediate_size"),
                    num_heads: req("num_heads"),
                    num_position_embeddings: req("num_position_embeddings"),
                    out_hidden_size: req("out_hidden_size"),
                    patch_size: req("patch_size"),
                    spatial_merge_size: req("spatial_merge_size"),
                    temporal_patch_size: req("temporal_patch_size"),
                    in_channels: req("in_channels"),
                }
            })
        });
        // vision_config dispatch is keyed by the tower's own model_type: the factored-additive
        // program (gemma-4 family) and the glm5_next fused-qkv program are different semantic
        // programs, and reading one's config through the other's key names produced a silently
        // wrong plan (caught in lane/glm5-vision: depth/num_heads/hidden_act all defaulted).
        if top
            .object("vision_config")
            .and_then(|v| v.string("model_type"))
            .as_deref()
            == Some("glm5_next_vision")
        {
            let v = top.object("vision_config").expect("checked above");
            let req_u = |x: Option<u32>, k: &str| {
                x.unwrap_or_else(|| {
                    panic!("glm5_next_vision config.json missing required field {k}")
                })
            };
            let req_f = |x: Option<f32>, k: &str| {
                x.unwrap_or_else(|| {
                    panic!("glm5_next_vision config.json missing required field {k}")
                })
            };
            cfg.vision_glm5 = Some(Glm5VisionConfig {
                depth: req_u(v.u32("depth"), "vision_config.depth"),
                hidden_size: req_u(v.u32("hidden_size"), "vision_config.hidden_size"),
                num_heads: req_u(v.u32("num_heads"), "vision_config.num_heads"),
                intermediate_size: req_u(
                    v.u32("intermediate_size"),
                    "vision_config.intermediate_size",
                ),
                patch_size: req_u(v.u32("patch_size"), "vision_config.patch_size"),
                temporal_patch_size: req_u(
                    v.u32("temporal_patch_size"),
                    "vision_config.temporal_patch_size",
                ),
                spatial_merge_size: req_u(
                    v.u32("spatial_merge_size"),
                    "vision_config.spatial_merge_size",
                ),
                out_hidden_size: req_u(v.u32("out_hidden_size"), "vision_config.out_hidden_size"),
                projection_intermediate_size: req_u(
                    v.u32("projection_intermediate_size"),
                    "vision_config.projection_intermediate_size",
                ),
                swiglu_limit: req_f(v.f32("swiglu_limit"), "vision_config.swiglu_limit"),
                rms_norm_eps: req_f(v.f32("rms_norm_eps"), "vision_config.rms_norm_eps"),
                in_channels: req_u(v.u32("in_channels"), "vision_config.in_channels"),
                attention_bias: v.boolean("attention_bias").unwrap_or_else(|| {
                    panic!("glm5_next_vision config.json missing required field vision_config.attention_bias")
                }),
                hidden_act: v.string("hidden_act").unwrap_or_else(|| {
                    panic!("glm5_next_vision config.json missing required field vision_config.hidden_act")
                }),
                image_token_id: req_u(top.u32("image_token_id"), "image_token_id"),
                video_token_id: req_u(top.u32("video_token_id"), "video_token_id"),
                image_start_token_id: req_u(
                    top.u32("image_start_token_id"),
                    "image_start_token_id",
                ),
                image_end_token_id: req_u(top.u32("image_end_token_id"), "image_end_token_id"),
                video_start_token_id: req_u(
                    top.u32("video_start_token_id"),
                    "video_start_token_id",
                ),
                video_end_token_id: req_u(top.u32("video_end_token_id"), "video_end_token_id"),
            });
        } else {
            cfg.vision = top.object("vision_config").map(|vision| VisionConfig {
                hidden_size: vision.u32("hidden_size").unwrap_or(768),
                intermediate_size: vision.u32("intermediate_size").unwrap_or(3072),
                layer_count: vision.u32("num_hidden_layers").unwrap_or(16),
                attention_heads: vision.u32("num_attention_heads").unwrap_or(12),
                kv_heads: vision.u32("num_key_value_heads").unwrap_or(12),
                head_dim: vision.u32("head_dim").unwrap_or(64),
                context_length: vision.u32("max_position_embeddings").unwrap_or(131_072),
                patch_size: vision.u32("patch_size").unwrap_or(16),
                position_embedding_size: vision.u32("position_embedding_size").unwrap_or(10_240),
                position_axes: 2,
                pooling_kernel_size: vision.u32("pooling_kernel_size").unwrap_or(3),
                rms_eps: vision.f32("rms_norm_eps").unwrap_or(1e-6),
                rope_theta: vision
                    .object("rope_parameters")
                    .and_then(|rope| rope.f32("rope_theta"))
                    .unwrap_or(100.0),
                activation: vision
                    .string("hidden_activation")
                    .unwrap_or_else(|| "gelu_pytorch_tanh".to_string()),
                standardize: vision.boolean("standardize").unwrap_or(false),
                clipped_linears: vision.boolean("use_clipped_linears").unwrap_or(false),
            });
        }
        // text_config (hybrid / VLM wrappers) — its transformer fields take precedence.
        if let Some(tc) = top.object("text_config") {
            cfg.apply(&tc);
        }
        // model_type fallback chain: text_config.model_type > model_type > architectures[0].
        if cfg.model_type.is_empty()
            && let Some(arch0) = top.first_string_in_array("architectures")
        {
            cfg.model_type = arch0;
        }
        cfg
    }

    fn apply(&mut self, o: &JsonObj) {
        if let Some(s) = o.string("model_type") {
            self.model_type = s;
        }
        if let Some(q) = o.object("quantization_config")
            && let Some(modules) = q.string_array("modules_to_not_convert")
        {
            self.modules_to_not_convert = modules;
        }
        if let Some(s) = o
            .string("name_or_path")
            .or_else(|| o.string("_name_or_path"))
        {
            self.name = Some(s);
        }
        if let Some(v) = o.u32("num_hidden_layers") {
            self.num_hidden_layers = v;
        }
        if let Some(v) = o.u32("image_token_id") {
            self.image_token_id = Some(v);
        }
        if let Some(v) = o.u32("vision_soft_tokens_per_image") {
            self.vision_soft_tokens_per_image = Some(v);
        }
        if let Some(v) = o.u32("hidden_size") {
            self.hidden_size = v;
        }
        if let Some(v) = o.u32("num_attention_heads") {
            self.num_attention_heads = v;
        }
        if let Some(v) = o
            .u32("num_key_value_heads")
            .or_else(|| o.u32("num_attention_groups"))
        {
            self.num_key_value_heads = Some(v);
        }
        if let Some(v) = o.u32("head_dim") {
            self.head_dim = Some(v);
        }
        if let Some(v) = o.u32("intermediate_size") {
            self.intermediate_size = v;
        }
        if let Some(v) = o.u32("vocab_size") {
            self.vocab_size = v;
        }
        if let Some(v) = o.u32("max_position_embeddings") {
            self.max_position_embeddings = v;
        }
        if let Some(v) = o.f32("rms_norm_eps") {
            self.rms_norm_eps = v;
            self.rms_norm_eps_explicit = true;
        }
        // gemma-4 (Arch::Gemma4) fields — read leniently, other arches never set them.
        if let Some(raw) = o.raw("layer_types") {
            let raw = raw.trim();
            if raw.starts_with('[') && raw.ends_with(']') {
                let flags: Vec<bool> = raw[1..raw.len() - 1]
                    .split(',')
                    .map(|s| s.trim().trim_matches('"') == "sliding_attention")
                    .collect();
                if !flags.is_empty() {
                    self.gemma4_swa_pattern = Some(flags);
                }
            }
        }
        if let Some(v) = o.u32("global_head_dim") {
            self.global_head_dim = Some(v);
        }
        if let Some(v) = o.u32("num_global_key_value_heads") {
            self.num_global_key_value_heads = Some(v);
        }
        if let Some(v) = o.u32("sliding_window") {
            self.sliding_window = Some(v);
        }
        if let Some(v) = o.f32("final_logit_softcapping") {
            self.final_logit_softcapping = Some(v);
        }
        if let Some(v) = o.u32("hidden_size_per_layer_input") {
            self.hidden_size_per_layer_input = Some(v);
        }
        if let Some(v) = o.u32("num_kv_shared_layers") {
            self.num_kv_shared_layers = Some(v);
        }
        if let Some(rp) = o.object("rope_parameters") {
            if let Some(fa) = rp.object("full_attention") {
                if let Some(t) = fa.f32("rope_theta") {
                    self.gemma4_rope_theta_global = Some(t);
                }
                if let Some(p) = fa.f32("partial_rotary_factor") {
                    self.gemma4_partial_rotary_global = Some(p);
                }
            }
            if let Some(sa) = rp.object("sliding_attention")
                && let Some(t) = sa.f32("rope_theta")
            {
                self.gemma4_rope_theta_swa = Some(t);
            }
        }
        if let Some(v) = o.f32("rope_theta") {
            self.rope_theta = v;
        }
        // Partial RoPE as a FRACTION of head_dim. Qwen3.5-family checkpoints write it top level
        // (Ornith-1.5-35B-A3B) or nested under `rope_parameters` (Qwen3.5-122B) or, most often,
        // BOTH with the same value — so read both, nested last, exactly like `rope_theta` above.
        // Missing entirely => full rope, which is what every non-partial arch wants.
        if let Some(v) = o.f32("partial_rotary_factor") {
            self.partial_rotary_factor = Some(v);
        }
        if let Some(rp) = o.object("rope_parameters") {
            if let Some(v) = rp.f32("rope_theta") {
                self.rope_theta = v;
            }
            if let Some(v) = rp.f32("partial_rotary_factor") {
                self.partial_rotary_factor = Some(v);
            }
        }
        if let Some(v) = o.f32_array("rope_theta") {
            if let Some(first) = v.first() {
                self.rope_theta = *first;
            }
            self.rope_theta_layers = Some(v);
        }
        if let Some(rp) = o.object("rope_parameters") {
            if let Some(v) = rp.f32("rope_theta") {
                self.rope_theta = v;
            }
            if let Some(v) = rp.f32("partial_rotary_factor") {
                self.partial_rotary_factor = Some(v);
            }
        }
        // Rope scaling lives under `rope_scaling` (classic) or `rope_parameters` (the
        // transformers 5.x spelling — qwen4_exp ships rope_type there). Read both, nested
        // last, exactly like `rope_theta` above; keys absent in the later object keep the
        // earlier value.
        for scaling_key in ["rope_scaling", "rope_parameters"] {
            let Some(rp) = o.object(scaling_key) else {
                continue;
            };
            if let Some(v) = rp.string("rope_type").or_else(|| rp.string("type")) {
                self.rope_scaling_type = Some(v);
            }
            if let Some(v) = rp.f32("factor") {
                self.rope_scaling_factor = Some(v);
            }
            if let Some(v) = rp.u32("original_max_position_embeddings") {
                self.rope_scaling_original_context = Some(v);
            }
            if let Some(v) = rp.f32("low_freq_factor") {
                self.rope_scaling_low_freq_factor = Some(v);
            }
            if let Some(v) = rp.f32("high_freq_factor") {
                self.rope_scaling_high_freq_factor = Some(v);
            }
            if let Some(v) = rp.f32("beta_fast") {
                self.rope_yarn_beta_fast = Some(v);
            }
            if let Some(v) = rp.f32("beta_slow") {
                self.rope_yarn_beta_slow = Some(v);
            }
            if let Some(v) = rp.f32("attention_factor") {
                self.rope_scaling_attention_factor = Some(v);
            }
            if let Some(v) = rp.f32("mscale") {
                self.rope_scaling_mscale = Some(v);
            }
            if let Some(v) = rp.f32("mscale_all_dim") {
                self.rope_scaling_mscale_all_dim = Some(v);
            }
            if let Some(v) = rp.boolean("truncate") {
                self.rope_scaling_truncate = Some(v);
            }
        }
        if let Some(v) = o.u32("full_attention_interval") {
            self.full_attention_interval = Some(v);
        }
        if let Some(v) = o.u32("num_nextn_predict_layers") {
            self.num_nextn_predict_layers = Some(v);
        }
        if let Some(v) = o.u32("top_k_experts") {
            self.num_experts_per_tok = Some(v);
        }
        if let Some(v) = o.u32("mtp_num_hidden_layers") {
            self.mtp_num_hidden_layers = Some(v);
        }
        if let Some(v) = o
            .u32("num_experts")
            .or_else(|| o.u32("num_local_experts"))
            // deepseek_v4 names the routed-expert count `n_routed_experts`.
            .or_else(|| o.u32("n_routed_experts"))
            .or_else(|| o.u32("moe_num_experts"))
        {
            self.num_experts = Some(v);
        }
        if let Some(v) = o.u32("num_experts_per_tok").or_else(|| o.u32("moe_top_k")) {
            self.num_experts_per_tok = Some(v);
        }
        if let Some(v) = o.u32("moe_intermediate_size") {
            self.moe_intermediate_size = Some(v);
        }
        if let Some(v) = o.u32("expert_hidden_dim") {
            self.expert_hidden_dim = Some(v);
        }
        if let Some(v) = o
            .u32("shared_expert_intermediate_size")
            .or_else(|| o.u32("share_expert_dim"))
        {
            self.shared_expert_intermediate_size = Some(v);
        }
        if let Some(v) = o.u32("linear_conv_kernel_dim") {
            self.linear_conv_kernel_dim = Some(v);
        }
        if let Some(v) = o.u32("linear_key_head_dim") {
            self.linear_key_head_dim = Some(v);
        }
        if let Some(v) = o.u32("linear_value_head_dim") {
            self.linear_value_head_dim = Some(v);
        }
        if let Some(v) = o.u32("linear_num_key_heads") {
            self.linear_num_key_heads = Some(v);
        }
        if let Some(v) = o.u32("linear_num_value_heads") {
            self.linear_num_value_heads = Some(v);
        }
        // ---- MiniMax-M3 keys ----
        if let Some(v) = o.u32("num_local_experts") {
            self.num_local_experts = Some(v);
        }
        if let Some(v) = o.u32("dense_intermediate_size") {
            self.dense_intermediate_size = Some(v);
        }
        if let Some(v) = o.u32("shared_intermediate_size") {
            self.shared_intermediate_size = Some(v);
        }
        if let Some(v) = o
            .u32("n_shared_experts")
            .or_else(|| o.u32("num_shared_experts"))
        {
            self.n_shared_experts = Some(v);
        }
        if let Some(v) = o.u32("rotary_dim") {
            self.rotary_dim = Some(v);
        }
        if let Some(v) = o.boolean("use_gemma_norm") {
            self.use_gemma_norm = Some(v);
        }
        if let Some(v) = o.string("scoring_func") {
            self.scoring_func = Some(v);
        }
        if let Some(v) = o.f32("routed_scaling_factor") {
            self.routed_scaling_factor = Some(v);
        }
        if let Some(v) = o.boolean("use_routing_bias") {
            self.use_routing_bias = Some(v);
        }
        if let Some(v) = o.f32("swiglu_alpha") {
            self.swiglu_alpha = Some(v);
        }
        if let Some(v) = o.f32("swiglu_limit") {
            self.swiglu_limit = Some(v);
        }
        if let Some(v) = o.u32_array("moe_layer_freq") {
            self.moe_layer_freq = Some(v);
        }
        // ---- Hy3 keys ----
        if let Some(v) = o.u32("first_k_dense_replace") {
            self.first_k_dense_replace = Some(v);
        }
        if let Some(v) = o.boolean("moe_router_use_sigmoid") {
            self.moe_router_use_sigmoid = Some(v);
        }
        if let Some(v) = o.boolean("moe_router_enable_expert_bias") {
            self.moe_router_enable_expert_bias = Some(v);
        }
        if let Some(v) = o.boolean("route_norm") {
            self.route_norm = Some(v);
        }
        if let Some(v) = o.f32("router_scaling_factor") {
            self.router_scaling_factor = Some(v);
        }
        if let Some(v) = o.boolean("qk_norm") {
            self.qk_norm = Some(v);
        }
        if let Some(v) = o.string("hidden_act") {
            self.hidden_act = Some(v);
        }
        // ---- DeepSeek-V4 keys ----
        if let Some(v) = o.string("topk_method") {
            self.topk_method = Some(v);
        }
        if let Some(v) = o.boolean("norm_topk_prob") {
            self.norm_topk_prob = Some(v);
        }
        if let Some(v) = o.u32("num_hash_layers") {
            self.num_hash_layers = Some(v);
        }
        if let Some(v) = o.f32("hc_eps") {
            self.hc_eps = Some(v);
        }
        if let Some(v) = o.u32("hc_mult") {
            self.hc_mult = Some(v);
        }
        if let Some(v) = o.u32("hc_sinkhorn_iters") {
            self.hc_sinkhorn_iters = Some(v);
        }
        if let Some(v) = o.u32("q_lora_rank") {
            self.q_lora_rank = Some(v);
        }
        if let Some(v) = o.u32("qk_rope_head_dim") {
            self.qk_rope_head_dim = Some(v);
        }
        if let Some(v) = o.u32("o_lora_rank") {
            self.o_lora_rank = Some(v);
        }
        if let Some(v) = o.u32("o_groups") {
            self.o_groups = Some(v);
        }
        if let Some(v) = o.u32("index_n_heads") {
            self.index_n_heads = Some(v);
        }
        if let Some(v) = o.u32("index_head_dim") {
            self.index_head_dim = Some(v);
        }
        if let Some(v) = o.u32("index_topk") {
            self.index_topk = Some(v);
        }
        if let Some(v) = o.u32_array("compress_ratios") {
            self.compress_ratios = Some(v);
        }
        if let Some(v) = o.f32("compress_rope_theta") {
            self.compress_rope_theta = Some(v);
        }
        // ---- GLM-5.3-Flash keys ----
        if let Some(v) = o.u32("kv_lora_rank") {
            self.kv_lora_rank = Some(v);
        }
        if let Some(v) = o.u32("qk_head_dim") {
            self.qk_head_dim = Some(v);
        }
        if let Some(v) = o.u32("qk_nope_head_dim") {
            self.qk_nope_head_dim = Some(v);
        }
        if let Some(v) = o.u32("v_head_dim") {
            self.v_head_dim = Some(v);
        }
        if let Some(v) = o.boolean("mla_use_nope") {
            self.mla_use_nope = Some(v);
        }
        if let Some(v) = o.u32("index_kpool") {
            self.index_kpool = Some(v);
        }
        if let Some(v) = o.boolean("index_kpool_always_select_tail") {
            self.index_kpool_always_select_tail = Some(v);
        }
        if let Some(v) = o.boolean("index_kpool_compress") {
            self.index_kpool_compress = Some(v);
        }
        if let Some(v) = o.boolean("indexer_rope_interleave") {
            self.indexer_rope_interleave = Some(v);
        }
        if let Some(v) = o.boolean("index_share_for_mtp_iteration") {
            self.index_share_for_mtp_iteration = Some(v);
        }
        if let Some(v) = o.string_array("indexer_types") {
            self.indexer_types = Some(v);
        }
        if let Some(v) = o.string_array("mlp_layer_types") {
            self.mlp_layer_types = Some(v);
        }
        if let Some(v) = o.string("moe_router_dtype") {
            self.moe_router_dtype = Some(v);
        }
        if let Some(v) = o.string("output_gate_type") {
            self.output_gate_type = Some(v);
        }
        if let Some(v) = o.boolean("mhc") {
            self.mhc = Some(v);
        }
        if let Some(la) = o.object("linear_attn_config") {
            self.glm_linear_num_heads = la.u32("num_heads");
            self.glm_linear_head_dim = la.u32("head_dim");
            self.glm_linear_short_conv = la.u32("short_conv_kernel_size");
            self.glm_gate_lower_bound = la.f32("gate_lower_bound");
            self.glm_kda_layers = la.u32_array("kda_layers");
            self.glm_full_attn_layers = la.u32_array("full_attn_layers");
        }
        if let Some(rs) = o.object("rope_scaling") {
            if let Some(v) = rs.f32("factor") {
                self.rope_yarn_factor = Some(v);
            }
            if let Some(v) = rs.u32("original_max_position_embeddings") {
                self.rope_yarn_orig_ctx = Some(v);
            }
            if let Some(v) = rs.f32("beta_fast") {
                self.rope_yarn_beta_fast = Some(v);
            }
            if let Some(v) = rs.f32("beta_slow") {
                self.rope_yarn_beta_slow = Some(v);
            }
        }
        // ---- Qwen4Exp keys (qwen4_exp text_config; SEMANTICS.md field notes) ----
        for (field, key) in [
            (&mut self.indexer_n_heads, "indexer_n_heads"),
            (&mut self.indexer_kv_heads, "indexer_kv_heads"),
            (&mut self.indexer_head_dim, "indexer_head_dim"),
            (&mut self.indexer_compress_ratio, "indexer_compress_ratio"),
            (&mut self.indexer_budget, "indexer_budget"),
            (&mut self.hc_count, "hc_count"),
            (&mut self.hc_lowrank, "hc_lowrank"),
            (&mut self.ngram_size, "ngram_size"),
            (&mut self.heads_per_ngram, "heads_per_ngram"),
            (
                &mut self.make_ngram_vocab_size_divisible_by,
                "make_ngram_vocab_size_divisible_by",
            ),
            (&mut self.split_ngram_parts, "split_ngram_parts"),
            (&mut self.ple_embed_dim, "ple_embed_dim"),
            (&mut self.ple_conv_kernel_size, "ple_conv_kernel_size"),
        ] {
            if let Some(v) = o.u32(key) {
                *field = Some(v);
            }
        }
        if let Some(v) = o.u64("ngram_vocab_size_base") {
            self.ngram_vocab_size_base = Some(v);
        }
        if let Some(v) = o.u32_array("ple_layer_ids") {
            self.ple_layer_ids = Some(v);
        }
        if let Some(v) = o.string("output_gate_type") {
            self.output_gate_type = Some(v);
        }
        // Scalar on the pinned artifact; a list takes its FIRST entry (modular L621).
        if let Some(v) = o
            .u32("eos_token_id")
            .or_else(|| o.u32_array("eos_token_id").and_then(|v| v.first().copied()))
        {
            self.eos_token_id = Some(v);
        }
        if let Some(rp) = o.object("rope_parameters") {
            if let Some(v) = rp.u32_array("mrope_section") {
                self.mrope_section = Some(v);
            }
            if let Some(v) = rp.boolean("mrope_interleaved") {
                self.mrope_interleaved = Some(v);
            }
        }
        if let Some(m) = o.object("mtp") {
            if let Some(v) = m.u32("num_hidden_layers") {
                self.qwen4exp_mtp_num_hidden_layers = Some(v);
            }
            if let Some(v) = m.f32("rope_theta") {
                self.qwen4exp_mtp_rope_theta = Some(v);
            }
        }
        // ---- Step35 keys ----
        if let Some(v) = o.object("attention_other_setting") {
            self.attention_other_num_heads = v.u32("num_attention_heads");
            self.attention_other_num_groups = v.u32("num_attention_groups");
        }
        if let Some(v) = o.string_array("layer_types") {
            self.layer_types = Some(v);
        }
        if let Some(v) = o.f32_array("partial_rotary_factors") {
            self.partial_rotary_factors = Some(v);
        }
        if let Some(v) = o.u32("sliding_window") {
            self.sliding_window = Some(v);
        }
        if let Some(v) = o.f32_array("swiglu_limits") {
            self.swiglu_limits = Some(v);
        }
        if let Some(v) = o.f32_array("swiglu_limits_shared") {
            self.swiglu_limits_shared = Some(v);
        }
        if let Some(v) = o.string("moe_router_activation") {
            self.moe_router_activation = Some(v);
        }
        if let Some(v) = o.string("moe_layers_enum") {
            self.moe_layers_enum = Some(v);
        }
        if let Some(v) = o.boolean("norm_expert_weight") {
            self.route_norm = Some(v);
        }
        if let Some(v) = o.f32("moe_router_scaling_factor") {
            self.router_scaling_factor = Some(v);
        }
    }
}

// ============================ minimal flat JSON object reader ============================
//
// config.json is a flat-ish object; we only need scalar fields + one level of nested object
// (text_config) + the architectures string array. Rather than add serde to memra-gguf, parse
// the value-bearing tokens for the keys we care about. Nested objects/arrays are captured as
// raw substrings so they can be re-parsed on demand.

// pub (was pub(crate)) since the dsv4 lane-4 verify bin reads its own small run-record
// JSON with it; still the same minimal hand parser, not a public serde substitute.
pub struct JsonObj {
    // key -> raw value substring (trimmed). Objects/arrays keep their braces/brackets.
    fields: std::collections::BTreeMap<String, String>,
}

impl JsonObj {
    pub fn parse(json: &str) -> Self {
        let b = json.as_bytes();
        let mut i = 0usize;
        let mut fields = std::collections::BTreeMap::new();
        // find opening brace
        while i < b.len() && b[i] != b'{' {
            i += 1;
        }
        if i >= b.len() {
            return JsonObj { fields };
        }
        i += 1; // past '{'
        loop {
            skip_ws(b, &mut i);
            if i >= b.len() || b[i] == b'}' {
                break;
            }
            if b[i] != b'"' {
                // unexpected; bail gracefully
                break;
            }
            let key = read_string(b, &mut i);
            skip_ws(b, &mut i);
            if i >= b.len() || b[i] != b':' {
                break;
            }
            i += 1; // ':'
            skip_ws(b, &mut i);
            let val = read_value_raw(b, &mut i);
            fields.insert(key, val);
            skip_ws(b, &mut i);
            if i < b.len() && b[i] == b',' {
                i += 1;
                continue;
            }
            break;
        }
        JsonObj { fields }
    }

    pub(crate) fn fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    // pub (was pub(crate)): the memra-engine ep-map reader (`ep_map.rs`) parses the
    // shared `memra-ep-map-v1` JSON with the same minimal house parser instead of
    // adding a serde dependency to the engine.
    pub fn raw(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    pub fn string(&self, key: &str) -> Option<String> {
        let v = self.raw(key)?.trim();
        if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
            Some(v[1..v.len() - 1].to_string())
        } else {
            None
        }
    }

    pub(crate) fn u32(&self, key: &str) -> Option<u32> {
        let v = self.raw(key)?.trim();
        if v == "null" {
            return None;
        }
        // accept integers (and floats that are whole, e.g. "8.0")
        v.parse::<u64>()
            .ok()
            .map(|x| x as u32)
            .or_else(|| v.parse::<f64>().ok().map(|x| x as u32))
    }

    pub(crate) fn u64(&self, key: &str) -> Option<u64> {
        let v = self.raw(key)?.trim();
        if v == "null" {
            return None;
        }
        v.parse::<u64>()
            .ok()
            .or_else(|| v.parse::<f64>().ok().map(|x| x as u64))
    }

    pub(crate) fn f32(&self, key: &str) -> Option<f32> {
        let v = self.raw(key)?.trim();
        if v == "null" {
            return None;
        }
        v.parse::<f32>().ok()
    }

    pub(crate) fn boolean(&self, key: &str) -> Option<bool> {
        match self.raw(key)?.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }

    /// Integer array field (e.g. moe_layer_freq: [0,0,0,1,...]).
    pub fn u32_array(&self, key: &str) -> Option<Vec<u32>> {
        let v = self.raw(key)?.trim();
        if !v.starts_with('[') || !v.ends_with(']') {
            return None;
        }
        Some(
            v[1..v.len() - 1]
                .split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect(),
        )
    }

    /// Flat f32 array field (0731 re-gate: banked per-step top-8 logit rows ride a flat
    /// aux JSON for the GPU teacher-forcing gate).
    pub fn f32_array(&self, key: &str) -> Option<Vec<f32>> {
        let v = self.raw(key)?.trim();
        if !v.starts_with('[') || !v.ends_with(']') {
            return None;
        }
        Some(
            v[1..v.len() - 1]
                .split(',')
                .filter_map(|x| x.trim().parse::<f32>().ok())
                .collect(),
        )
    }

    pub(crate) fn u64_array(&self, key: &str) -> Option<Vec<u64>> {
        let v = self.raw(key)?.trim();
        if !v.starts_with('[') || !v.ends_with(']') {
            return None;
        }
        Some(
            v[1..v.len() - 1]
                .split(',')
                .filter_map(|x| x.trim().parse::<u64>().ok())
                .collect(),
        )
    }

    pub(crate) fn string_array(&self, key: &str) -> Option<Vec<String>> {
        let v = self.raw(key)?.trim();
        if !v.starts_with('[') || !v.ends_with(']') {
            return None;
        }
        Some(
            v[1..v.len() - 1]
                .split(',')
                .filter_map(|x| {
                    let x = x.trim();
                    x.strip_prefix('"')
                        .and_then(|s| s.strip_suffix('"'))
                        .map(str::to_owned)
                })
                .collect(),
        )
    }

    pub(crate) fn object(&self, key: &str) -> Option<JsonObj> {
        let v = self.raw(key)?.trim();
        if v.starts_with('{') {
            Some(JsonObj::parse(v))
        } else {
            None
        }
    }

    /// Bare numeric field (fixture JSONs bank measured scalars, e.g. the contract fork).
    pub(crate) fn f64(&self, key: &str) -> Option<f64> {
        self.raw(key)?.trim().parse().ok()
    }

    /// First string element of a string array field (e.g. architectures[0]).
    pub(crate) fn first_string_in_array(&self, key: &str) -> Option<String> {
        let v = self.raw(key)?.trim();
        let inner = v.strip_prefix('[')?.trim_start();
        let q = inner.find('"')? + 1;
        let rest = &inner[q..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
}

fn read_string(b: &[u8], i: &mut usize) -> String {
    // assumes b[*i] == '"'
    *i += 1;
    let mut s = String::new();
    while *i < b.len() {
        let c = b[*i];
        *i += 1;
        match c {
            b'"' => break,
            b'\\' => {
                if *i < b.len() {
                    let e = b[*i];
                    *i += 1;
                    s.push(match e {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        other => other as char,
                    });
                }
            }
            _ => s.push(c as char),
        }
    }
    s
}

/// Read a raw value substring (string with quotes, number, bool/null, or a balanced {}/[] block).
fn read_value_raw(b: &[u8], i: &mut usize) -> String {
    skip_ws(b, i);
    let start = *i;
    match b.get(*i).copied() {
        Some(b'"') => {
            // string value — include the quotes
            *i += 1;
            while *i < b.len() {
                let c = b[*i];
                *i += 1;
                if c == b'\\' {
                    *i += 1;
                    continue;
                }
                if c == b'"' {
                    break;
                }
            }
            String::from_utf8_lossy(&b[start..*i]).into_owned()
        }
        Some(b'{') | Some(b'[') => {
            // balanced block, respecting strings inside
            let open = b[*i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0i32;
            let mut in_str = false;
            while *i < b.len() {
                let c = b[*i];
                *i += 1;
                if in_str {
                    if c == b'\\' {
                        *i += 1;
                    } else if c == b'"' {
                        in_str = false;
                    }
                    continue;
                }
                match c {
                    b'"' => in_str = true,
                    x if x == open => depth += 1,
                    x if x == close => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
            String::from_utf8_lossy(&b[start..*i]).into_owned()
        }
        _ => {
            // scalar: number / true / false / null — until , } ] or whitespace
            while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']') {
                *i += 1;
            }
            String::from_utf8_lossy(&b[start..*i]).trim().to_string()
        }
    }
}

#[cfg(test)]
pub(crate) mod hf_tests {
    use super::*;

    const QWEN3_17B: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"],
      "head_dim": 128,
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "max_position_embeddings": 40960,
      "model_type": "qwen3",
      "num_attention_heads": 16,
      "num_hidden_layers": 28,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "vocab_size": 151936
    }"#;

    #[test]
    fn parse_qwen3_dense() {
        let c = HfConfig::parse(QWEN3_17B);
        assert_eq!(c.model_type, "qwen3");
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, Some(8));
        assert_eq!(c.head_dim, Some(128));
        assert_eq!(c.intermediate_size, 6144);
        assert_eq!(c.vocab_size, 151936);
        assert_eq!(c.max_position_embeddings, 40960);
        assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((c.rope_theta - 1_000_000.0).abs() < 1.0);

        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen3);
        assert_eq!(mc.n_layer, 28);
        assert_eq!(mc.n_embd, 2048);
        assert_eq!(mc.n_head, 16);
        assert_eq!(mc.n_head_kv, 8);
        assert_eq!(mc.head_dim_k, 128);
        assert_eq!(mc.n_ff, 6144);
        assert_eq!(mc.n_vocab, 151936);
        assert!(mc.moe.is_none());
        assert!(mc.ssm.is_none());
        assert_eq!(mc.full_attention_interval, 0);
    }

    /// Every registered arch declares a gate layout, and the two predicates the forward paths key
    /// off agree with that declaration. This is the whole-model half; the per-layer half is pinned
    /// by the geometry-table assertions in micro_gguf.rs.
    #[test]
    fn every_registered_arch_declares_a_gate_layout() {
        let cases: &[(Arch, AttentionGateKind)] = &[
            (Arch::Qwen35, AttentionGateKind::FusedQ),
            (Arch::Qwen35Moe, AttentionGateKind::FusedQ),
            (Arch::Step35, AttentionGateKind::SeparateHead),
            (Arch::MinimaxM3, AttentionGateKind::None),
            (Arch::Hy3, AttentionGateKind::None),
            (Arch::Gemma4, AttentionGateKind::None),
            (Arch::GlmDsa, AttentionGateKind::None),
            (Arch::Qwen3, AttentionGateKind::None),
            (Arch::Qwen3Moe, AttentionGateKind::None),
            (Arch::Olmoe, AttentionGateKind::None),
            (Arch::Llama, AttentionGateKind::None),
        ];
        for (arch, want) in cases {
            assert_eq!(
                arch.attention_gate_kind(),
                Some(*want),
                "{arch:?} must declare its gate layout explicitly"
            );
        }
        // ...and every arch string `Arch::parse` recognizes is covered, so a new variant cannot
        // be reachable from metadata without a declaration (the match is exhaustive, so a new
        // variant fails to compile; this catches a variant declared but not parsed into).
        for name in [
            "qwen3",
            "qwen3moe",
            "qwen35",
            "qwen35moe",
            "qwen3next",
            "qwen3nextmoe",
            "olmoe",
            "minimax-m3",
            "hy3",
            "gemma4",
            "glm-dsa",
            "step35",
            "llama",
        ] {
            let arch = Arch::parse(name);
            assert!(
                arch.attention_gate_kind().is_some(),
                "arch string {name:?} parses to {arch:?}, which declares no gate layout"
            );
        }
    }

    /// The defect this hardening closes: an UNREGISTERED arch used to inherit the qwen3.5 FusedQ
    /// layout from a permissive deny-list fallback. It must now declare nothing, answer `false`,
    /// and refuse to validate.
    #[test]
    fn unregistered_arch_never_inherits_a_fused_gate() {
        let arch = Arch::parse("muse_glimmer");
        assert_eq!(arch, Arch::Other("muse_glimmer".into()));
        assert_eq!(
            arch.attention_gate_kind(),
            None,
            "an unregistered arch declares NO layout"
        );

        // A ModelConfig on that arch: no geometry table, no per-arch config struct — exactly the
        // shape that used to fall through the deny-list to `true`.
        let json = r#"{"model_type":"muse_glimmer","num_hidden_layers":4,"hidden_size":256,
            "num_attention_heads":8,"intermediate_size":512,"vocab_size":1000,
            "max_position_embeddings":2048}"#;
        let mc = ModelConfig::from_hf(&HfConfig::parse(json));
        assert_eq!(mc.arch, Arch::Other("muse_glimmer".into()));
        assert!(mc.geometry.is_none());
        assert!(
            !mc.attn_out_gate(),
            "unregistered arch must NOT be treated as qwen3.5 FusedQ — q_gate_split would read \
             2x out of bounds"
        );
        assert!(!mc.attn_gate_separate());
        assert!(!mc.attn_out_gate_at(0));
        assert_eq!(
            mc.full_attention_geometry_at(0).attention_gate,
            AttentionGateKind::None
        );

        // and the load-time gate refuses it with a typed error that names the arch and the fix
        let err = mc
            .validate_attention_gate_layout()
            .expect_err("an undeclared gate layout must refuse to load");
        assert_eq!(
            err,
            UndeclaredGateLayout {
                arch: "Other(\"muse_glimmer\")".to_string()
            }
        );
        let msg = err.to_string();
        assert!(msg.contains("muse_glimmer"), "{msg}");
        assert!(msg.contains("attention_gate_kind"), "{msg}");
        let _: &dyn std::error::Error = &err;
    }

    /// Registered arches validate, and report the layout their execution arms actually use.
    #[test]
    fn registered_arches_validate_their_layout() {
        let mc = ModelConfig::from_hf(&HfConfig::parse(QWEN3_17B));
        assert_eq!(
            mc.validate_attention_gate_layout(),
            Ok(AttentionGateKind::None)
        );

        // a geometry-table arch answers from the table, not from the arch declaration
        let hybrid = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":4,"hidden_size":256,
               "num_attention_heads":8,"num_key_value_heads":2,"intermediate_size":512,
               "vocab_size":1000,"max_position_embeddings":2048,"full_attention_interval":2}"#,
        ));
        assert_eq!(hybrid.arch, Arch::Qwen35);
        assert!(hybrid.geometry.is_some());
        assert_eq!(
            hybrid.validate_attention_gate_layout(),
            Ok(AttentionGateKind::FusedQ)
        );
        assert!(hybrid.attn_out_gate());
    }

    /// The read-site bounds contract for the fused split. A separate-gate checkpoint's wq output
    /// is exactly half of what the split reads, which is the out-of-bounds read verbatim.
    #[test]
    fn fused_q_gate_extent_catches_the_2x_overread() {
        let (head_dim, n_head, t) = (128usize, 32usize, 4usize);
        let fused = 2 * head_dim * n_head * t;
        let separate = head_dim * n_head * t;

        // a real fused wq output passes, exactly and with slack
        assert_eq!(
            check_fused_q_gate_extent(fused, head_dim, n_head, t),
            Ok(())
        );
        assert_eq!(
            check_fused_q_gate_extent(fused + 1, head_dim, n_head, t),
            Ok(())
        );

        // a separate-gate wq output is refused, and the error carries both extents
        let err = check_fused_q_gate_extent(separate, head_dim, n_head, t)
            .expect_err("half-width wq must be refused, not read past");
        assert_eq!(err.need, fused);
        assert_eq!(err.have, separate);
        assert_eq!(err.need, 2 * err.have, "the overread is exactly 2x");
        let msg = err.to_string();
        assert!(msg.contains("NO fused gate"), "{msg}");
        assert!(msg.contains("AttentionGateKind"), "{msg}");
        let _: &dyn std::error::Error = &err;

        // one element short is still refused (the kernel's last read is need-1)
        assert!(check_fused_q_gate_extent(fused - 1, head_dim, n_head, t).is_err());
        // and a zero-width buffer cannot slip through on a zero-token batch either
        assert!(check_fused_q_gate_extent(0, head_dim, n_head, 1).is_err());
    }

    #[test]
    fn head_dim_fallback() {
        // no head_dim -> hidden_size / num_attention_heads
        let json = r#"{"model_type":"llama","num_hidden_layers":2,"hidden_size":256,"num_attention_heads":8,"intermediate_size":512,"vocab_size":1000,"max_position_embeddings":2048}"#;
        let c = HfConfig::parse(json);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Llama);
        assert_eq!(mc.head_dim_k, 32); // 256/8
        assert_eq!(mc.n_head_kv, 8); // defaults to n_head when absent
    }

    /// A qwen3_5 `text_config` as real checkpoints ship it, INCLUDING the partial-rotary
    /// declaration. Every published qwen3_5/qwen3_5_moe config carries `partial_rotary_factor`
    /// (0.25) and none carries `rotary_dim`, so a fixture without it is not this architecture —
    /// which is how this test used to pin `n_rot == 256` (full rope over all 256 head dims) for
    /// a model whose GGUF twin says `rope.dimension_count = 64`.
    fn qwen35_text_config_json(partial_rotary: &str) -> String {
        format!(
            r#"{{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {{
            "model_type": "qwen3_5_text",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-06,
            "rope_theta": 5000000,
            {partial_rotary}
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32
          }}
        }}"#
        )
    }

    #[test]
    fn nested_text_config_hybrid() {
        // qwen3_5 wraps the transformer config in text_config and uses HF model_type "qwen3_5".
        let json = qwen35_text_config_json(r#""partial_rotary_factor": 0.25,"#);
        let c = HfConfig::parse(&json);
        // text_config fields win
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 32);
        assert_eq!(c.full_attention_interval, Some(4));
        assert_eq!(c.model_type, "qwen3_5_text");
        assert_eq!(c.partial_rotary_factor, Some(0.25));

        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen35);
        assert!(mc.uses_hybrid_executor());
        assert_eq!(mc.full_attention_interval, 4);
        assert!(mc.ssm.is_some());
        // periodic full-attn classification still works
        assert_eq!(mc.layer_kind(3), LayerKind::FullAttention); // (3+1)%4==0
        assert_eq!(mc.layer_kind(0), LayerKind::LinearAttention);
        let table = mc.geometry.as_ref().expect("qwen35 has a geometry table");
        assert_eq!(table.classes().len(), 2);
        assert_eq!(table.layer_classes().len(), 32);
        let linear = mc.layer_geometry(0).unwrap();
        assert_eq!(linear.mixer, LayerKind::LinearAttention);
        assert_eq!(linear.attention_gate, AttentionGateKind::None);
        let full = mc.layer_geometry(3).unwrap();
        assert_eq!(full.mixer, LayerKind::FullAttention);
        assert_eq!(full.n_head, 32);
        assert_eq!(full.n_head_kv, 8);
        assert_eq!(full.head_dim_k, 256);
        // PARTIAL rope: 0.25 * 256 = 64. Dims 64..255 must pass through unrotated. Rotating
        // them is silent — no shape mismatch, fluent output, destroyed long-context behaviour.
        assert_eq!(full.n_rot, 64);
        assert_eq!(full.rope_base, 5_000_000.0);
        assert_eq!(full.window, None);
        assert!(!full.rope_factors);
        assert_eq!(full.attention_gate, AttentionGateKind::FusedQ);

        // The scalar field and the geometry table are the SAME derivation, not two that happen
        // to agree: `full_attention_geometry_at` falls back to the scalar for unmigrated arches.
        assert_eq!(mc.rope_dim_count, 64);
        assert_eq!(mc.full_attention_geometry_at(3).n_rot, 64);
    }

    #[test]
    fn qwen35_hf_partial_rotary_under_rope_parameters_is_read_too() {
        // Qwen3.5-122B writes the factor ONLY inside `rope_parameters`; Ornith-1.5-35B-A3B writes
        // it in both places. One spelling missing must not fall back to full rope.
        let nested = qwen35_text_config_json(
            r#""rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},"#,
        );
        let c = HfConfig::parse(&nested);
        assert_eq!(c.partial_rotary_factor, Some(0.25));
        // rope_parameters.rope_theta also wins over the flat key, as it already did.
        assert_eq!(c.rope_theta, 10_000_000.0);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.rope_dim_count, 64);
        assert_eq!(mc.layer_geometry(3).unwrap().n_rot, 64);

        // Both spellings present with the same value (the Ornith shape) agree.
        let both = qwen35_text_config_json(
            r#""partial_rotary_factor": 0.25, "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},"#,
        );
        let mc_both = ModelConfig::from_hf(&HfConfig::parse(&both));
        assert_eq!(mc_both.rope_dim_count, 64);
    }

    #[test]
    fn qwen35_hf_without_a_partial_rotary_declaration_is_full_rope() {
        // The honest default, pinned separately so it can never again be mistaken for the
        // partial-rotary answer: a config that declares NO partial rotary rotates every head dim.
        // No published qwen3_5 checkpoint looks like this; the case exists so the fixture above
        // and this one cannot be conflated.
        let json = qwen35_text_config_json("");
        let c = HfConfig::parse(&json);
        assert_eq!(c.partial_rotary_factor, None);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.rope_dim_count, 256);
        assert_eq!(mc.layer_geometry(3).unwrap().n_rot, 256);
    }

    #[test]
    fn n_rot_agrees_across_the_gguf_and_hf_loader_paths() {
        // THE DIVERGENCE GATE. One model, two readers: the GGUF ships the resolved
        // `rope.dimension_count` (64), the HF config ships the fraction (0.25 of head_dim 256).
        // Both must land on the same n_rot, or the safetensors route serves a differently-roped
        // model than its own GGUF twin — which is exactly what shipped before this test existed.
        use crate::micro_gguf::{GgufWriter, MetaW};

        let path = std::env::temp_dir().join(format!(
            "memra-nrot-parity-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut writer = GgufWriter::new();
        writer.kv("general.architecture", MetaW::Str("qwen35"));
        writer.kv("qwen35.block_count", MetaW::U32(32));
        writer.kv("qwen35.embedding_length", MetaW::U32(4096));
        writer.kv("qwen35.feed_forward_length", MetaW::U32(12288));
        writer.kv("qwen35.attention.head_count", MetaW::U32(32));
        writer.kv("qwen35.attention.head_count_kv", MetaW::U32(8));
        writer.kv("qwen35.attention.key_length", MetaW::U32(256));
        writer.kv("qwen35.attention.value_length", MetaW::U32(256));
        writer.kv("qwen35.full_attention_interval", MetaW::U32(4));
        writer.kv("qwen35.rope.freq_base", MetaW::F32(5_000_000.0));
        // the resolved partial-rope width the converter wrote: 0.25 * 256
        writer.kv("qwen35.rope.dimension_count", MetaW::U32(64));
        writer.write(&path).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let from_gguf = ModelConfig::from_gguf(&gguf);
        std::fs::remove_file(&path).ok();

        let from_hf = ModelConfig::from_hf(&HfConfig::parse(&qwen35_text_config_json(
            r#""partial_rotary_factor": 0.25,"#,
        )));

        assert_eq!(from_gguf.head_dim_k, from_hf.head_dim_k, "head_dim_k");
        assert_eq!(
            from_gguf.rope_dim_count, from_hf.rope_dim_count,
            "the same model must have the same rotary width on both loader paths \
             (gguf rope.dimension_count vs hf partial_rotary_factor * head_dim)"
        );
        assert_eq!(from_hf.rope_dim_count, 64);
        assert_eq!(
            from_gguf.layer_geometry(3).unwrap().n_rot,
            from_hf.layer_geometry(3).unwrap().n_rot,
            "full-attention geometry rows must agree too, not just the scalar"
        );
    }

    #[test]
    fn resolve_rope_dim_count_precedence() {
        // explicit dim count wins over a fraction (a converter that resolved it already)
        assert_eq!(resolve_rope_dim_count(Some(64), Some(0.5), 256), 64);
        // fraction of head_dim, rounded
        assert_eq!(resolve_rope_dim_count(None, Some(0.25), 256), 64);
        assert_eq!(resolve_rope_dim_count(None, Some(0.5), 128), 64);
        assert_eq!(resolve_rope_dim_count(None, Some(0.375), 100), 38);
        // nothing declared, or a full-width factor => full rope
        assert_eq!(resolve_rope_dim_count(None, None, 256), 256);
        assert_eq!(resolve_rope_dim_count(None, Some(1.0), 256), 256);
        // degenerate factors cannot silently disable rope (it consumes dim PAIRS) or exceed the head
        assert_eq!(resolve_rope_dim_count(None, Some(0.0), 256), 256);
        assert_eq!(resolve_rope_dim_count(None, Some(-1.0), 256), 256);
        assert_eq!(resolve_rope_dim_count(None, Some(0.001), 256), 2);
        assert_eq!(resolve_rope_dim_count(None, Some(2.0), 256), 256);
    }

    #[test]
    fn gemma4_partial_rotary_does_not_truncate_n_rot() {
        // gemma-4 declares a partial rotary factor too, but means the OTHER thing by it: the
        // GGUF twin ships a `rope_freqs` tensor whose factors are 1.0 for the first fraction of
        // dim pairs and ~1e30 beyond, so all `head_dim` dims go through rope and the tail rotates
        // by ~0. Truncating n_rot here would change a shipped, gated arm. Its factor is also a
        // different key (nested under rope_parameters.full_attention) — assert both facts so a
        // future refactor that "unifies" the two spellings has to confront this.
        let json = r#"{
          "model_type": "gemma4",
          "num_hidden_layers": 4,
          "hidden_size": 2816,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "head_dim": 256,
          "global_head_dim": 512,
          "intermediate_size": 11264,
          "vocab_size": 262144,
          "max_position_embeddings": 131072,
          "rms_norm_eps": 1e-06,
          "sliding_window": 1024,
          "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention"],
          "rope_parameters": {
            "full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000}
          }
        }"#;
        let c = HfConfig::parse(json);
        assert_eq!(c.gemma4_partial_rotary_global, Some(0.25));
        assert_eq!(
            c.partial_rotary_factor, None,
            "the gemma-4 nested key must not feed the generic n_rot derivation"
        );
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Gemma4);
        assert_eq!(
            mc.rope_dim_count, 256,
            "gemma-4 rotates the full head_dim; its partial rotary rides rope_freqs factors"
        );
        let g = mc.gemma4.as_ref().expect("gemma4 config");
        assert_eq!(g.partial_rotary_global, 0.25);
        assert_eq!(g.rope_dims_global, 512);
        assert_eq!(g.rope_dims_swa, 256);
    }

    #[test]
    fn qwen35_mtp_layer_is_explicit_full_attention_geometry() {
        let json = r#"{
          "model_type": "qwen3_5",
          "num_hidden_layers": 32,
          "num_nextn_predict_layers": 1,
          "hidden_size": 4096,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 12288,
          "vocab_size": 151936,
          "max_position_embeddings": 262144,
          "full_attention_interval": 4
        }"#;
        let mc = ModelConfig::from_hf(&HfConfig::parse(json));
        assert_eq!(mc.n_layer, 33);
        assert_eq!(mc.layer_kind(31), LayerKind::FullAttention);
        assert_eq!(mc.layer_kind(32), LayerKind::FullAttention);
        assert_eq!(
            mc.layer_geometry(32).unwrap().attention_gate,
            AttentionGateKind::FusedQ
        );
    }

    #[test]
    fn parse_step37_per_layer_geometry_and_moe_aliases() {
        let json = r#"{
          "model_type":"step3p5",
          "num_hidden_layers":3,
          "num_nextn_predict_layers":1,
          "hidden_size":16,
          "intermediate_size":32,
          "num_attention_heads":2,
          "num_attention_groups":1,
          "head_dim":8,
          "vocab_size":64,
          "max_position_embeddings":262144,
          "moe_num_experts":6,
          "moe_top_k":2,
          "moe_intermediate_size":12,
          "share_expert_dim":12,
          "moe_layers_enum":"1,2",
          "norm_expert_weight":true,
          "moe_router_activation":"sigmoid",
          "moe_router_scaling_factor":3.0,
          "sliding_window":512,
          "layer_types":["full_attention","sliding_attention","sliding_attention","sliding_attention"],
          "rope_theta":[5000000,10000,10000,10000],
          "partial_rotary_factors":[0.5,1.0,1.0,1.0],
          "attention_other_setting":{"num_attention_heads":4,"num_attention_groups":1},
          "swiglu_limits":[0,7,0,0],
          "swiglu_limits_shared":[0,16,0,0]
        }"#;
        let mc = ModelConfig::from_hf(&HfConfig::parse(json));
        assert_eq!(mc.arch, Arch::Step35);
        assert_eq!(mc.n_layer, 4);
        assert_eq!(mc.nextn_predict_layers, 1);
        assert_eq!(mc.n_head, 4, "global scratch uses the per-layer maximum");
        assert_eq!(mc.n_head_kv, 1);
        assert!(
            mc.ssm.is_none(),
            "Step is a full-attention hybrid, not an SSM model"
        );
        let moe = mc.moe.as_ref().unwrap();
        assert_eq!((moe.expert_count, moe.expert_used_count), (6, 2));
        assert_eq!(
            (moe.expert_ff_length, moe.expert_shared_ff_length),
            (12, 12)
        );
        let step = mc.step35.as_ref().unwrap();
        assert_eq!(step.first_k_dense_replace, 1);
        assert_eq!(step.clamp_exp(1), Some(7.0));
        assert_eq!(step.clamp_shexp(1), Some(16.0));
        assert_eq!(mc.sigmoid_router(), Some((3.0, true)));
        let full = mc.layer_geometry(0).unwrap();
        assert_eq!(full.n_head, 2);
        assert_eq!(full.n_rot, 4);
        assert_eq!(full.rope_base, 5_000_000.0);
        assert_eq!(full.window, None);
        assert!(full.rope_factors);
        assert_eq!(full.attention_gate, AttentionGateKind::SeparateHead);
        let swa = mc.layer_geometry(1).unwrap();
        assert_eq!(swa.n_head, 4);
        assert_eq!(swa.n_rot, 8);
        assert_eq!(swa.rope_base, 10_000.0);
        assert_eq!(swa.window, Some(512));
        assert!(!swa.rope_factors);
    }

    #[test]
    fn step37_hf_defaults_and_llama3_rope_factors() {
        // The published Step-3.7 config intentionally omits rms_norm_eps (its Python config
        // defaults to 1e-5) and stores llama3 scaling parameters in config rather than a tensor.
        let json = r#"{
          "model_type":"step3p5",
          "num_hidden_layers":2,
          "hidden_size":256,
          "intermediate_size":512,
          "num_attention_heads":2,
          "num_attention_groups":1,
          "head_dim":128,
          "vocab_size":64,
          "max_position_embeddings":262144,
          "moe_num_experts":6,
          "moe_top_k":2,
          "moe_intermediate_size":128,
          "share_expert_dim":128,
          "moe_layers_enum":"1",
          "norm_expert_weight":true,
          "moe_router_activation":"sigmoid",
          "moe_router_scaling_factor":3.0,
          "sliding_window":512,
          "layer_types":["full_attention","sliding_attention"],
          "rope_theta":[5000000,10000],
          "partial_rotary_factors":[0.5,1.0],
          "rope_scaling":{
            "rope_type":"llama3",
            "factor":2.0,
            "original_max_position_embeddings":131072,
            "low_freq_factor":1.0,
            "high_freq_factor":32.0
          },
          "attention_other_setting":{"num_attention_heads":3,"num_attention_groups":1},
          "swiglu_limits":[0,0],
          "swiglu_limits_shared":[0,0]
        }"#;
        let parsed = HfConfig::parse(json);
        assert!(!parsed.rms_norm_eps_explicit);
        let mc = ModelConfig::from_hf(&parsed);
        assert_eq!(mc.rms_eps, 1e-5);
        let factors = mc
            .step35
            .as_ref()
            .unwrap()
            .rope_freq_factors
            .as_ref()
            .unwrap();
        assert_eq!(factors.len(), 32);
        assert!((factors[0] - 1.0).abs() < 1e-6);
        assert!((factors[31] - 2.0).abs() < 1e-6);
        assert!(factors.iter().all(|&factor| (1.0..=2.0).contains(&factor)));

        let explicit_json = json.replacen(
            "\"hidden_size\":256,",
            "\"hidden_size\":256,\"rms_norm_eps\":0.00002,",
            1,
        );
        let explicit = HfConfig::parse(&explicit_json);
        assert!(explicit.rms_norm_eps_explicit);
        assert_eq!(ModelConfig::from_hf(&explicit).rms_eps, 2e-5);
    }

    #[test]
    fn moe_config() {
        let json = r#"{"model_type":"qwen3_moe","num_hidden_layers":4,"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":4,"intermediate_size":6144,"vocab_size":151936,"max_position_embeddings":40960,"num_experts":128,"num_experts_per_tok":8,"moe_intermediate_size":768,"shared_expert_intermediate_size":0}"#;
        let c = HfConfig::parse(json);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen3Moe);
        let moe = mc.moe.expect("moe");
        assert_eq!(moe.expert_count, 128);
        assert_eq!(moe.expert_used_count, 8);
        assert_eq!(moe.expert_ff_length, 768);
    }

    #[test]
    fn qwen3next_gguf_experts_survive_the_dense_arch_alias() {
        use crate::micro_gguf::{GgufWriter, MetaW};

        let write_fixture = |path: &std::path::Path, expert_count| {
            let mut writer = GgufWriter::new();
            writer.kv("general.architecture", MetaW::Str("qwen3next"));
            writer.kv("general.name", MetaW::Str("Qwen3 Next alias fixture"));
            writer.kv("qwen3next.block_count", MetaW::U32(2));
            writer.kv("qwen3next.embedding_length", MetaW::U32(64));
            writer.kv("qwen3next.feed_forward_length", MetaW::U32(128));
            writer.kv("qwen3next.attention.head_count", MetaW::U32(4));
            writer.kv("qwen3next.attention.head_count_kv", MetaW::U32(2));
            writer.kv("qwen3next.full_attention_interval", MetaW::U32(4));
            writer.kv("qwen3next.expert_count", MetaW::U32(expert_count));
            writer.kv("qwen3next.expert_used_count", MetaW::U32(10));
            writer.kv("qwen3next.expert_feed_forward_length", MetaW::U32(512));
            writer.write(path).unwrap();
        };
        let path =
            std::env::temp_dir().join(format!("memra-qwen3next-alias-{}.gguf", std::process::id()));
        write_fixture(&path, 512);

        let gguf = GgufFile::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&gguf);
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg.arch, Arch::Qwen35);
        let moe = cfg
            .moe
            .as_ref()
            .expect("explicit GGUF experts must survive aliasing");
        assert_eq!(moe.expert_count, 512);
        assert_eq!(moe.expert_used_count, 10);
        let warning = qwen3next_expert_alias_warning("qwen3next", cfg.moe.as_ref())
            .expect("expert-bearing qwen3next aliases must warn");
        assert!(warning.contains("hybrid MoE loaded through a dense alias"));
        assert!(warning.contains("subsystems keyed only on Arch may misclassify it"));

        let zero_path = std::env::temp_dir().join(format!(
            "memra-qwen3next-alias-zero-{}.gguf",
            std::process::id()
        ));
        write_fixture(&zero_path, 0);
        let zero_gguf = GgufFile::open(&zero_path).unwrap();
        let zero_cfg = ModelConfig::from_gguf(&zero_gguf);
        std::fs::remove_file(&zero_path).ok();
        assert!(
            zero_cfg.moe.is_none(),
            "zero experts must retain the dense loader path"
        );
        assert!(qwen3next_expert_alias_warning("qwen3next", zero_cfg.moe.as_ref()).is_none());
    }

    #[test]
    fn parse_hy3_reap_config() {
        let json = r#"{
          "model_type": "hy_v3",
          "num_hidden_layers": 80,
          "hidden_size": 4096,
          "num_attention_heads": 64,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 13312,
          "vocab_size": 120832,
          "max_position_embeddings": 262144,
          "rms_norm_eps": 1e-05,
          "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
          "num_nextn_predict_layers": 1,
          "num_experts": 96,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 1536,
          "expert_hidden_dim": 1536,
          "num_shared_experts": 1,
          "moe_router_use_sigmoid": true,
          "moe_router_enable_expert_bias": true,
          "route_norm": true,
          "router_scaling_factor": 2.826,
          "qk_norm": true,
          "hidden_act": "silu"
        }"#;
        let c = HfConfig::parse(json);
        assert!(!c.preserve_checkpoint_bf16);
        assert_eq!(Arch::from_hf_model_type(&c.model_type), Arch::Hy3);
        assert!((c.rope_theta - 11_158_840.0).abs() < 1.0);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Hy3);
        assert!(mc.moe.as_ref().is_some_and(|moe| moe.expert_count > 0));
        // Degenerate hybrid (the M3 class): rides HybridModel with every layer full-attention.
        assert!(mc.uses_hybrid_executor());
        assert_eq!(
            mc.full_attention_interval, 0,
            "Hy3 has no linear-attention layers"
        );
        assert!(
            !mc.attn_out_gate(),
            "Hy3 wq has no fused [q|gate] output gate"
        );
        let (sf, norm) = mc.sigmoid_router().expect("Hy3 routes with sigmoid");
        assert!((sf - 2.826).abs() < 1e-6);
        assert!(norm);
        assert_eq!(
            mc.n_layer, 81,
            "HF config convention includes the appended MTP block"
        );
        assert_eq!(mc.nextn_predict_layers, 1);
        assert_eq!(mc.n_embd, 4096);
        assert_eq!(mc.n_head, 64);
        assert_eq!(mc.n_head_kv, 8);
        assert_eq!(mc.head_dim_k, 128);
        assert_eq!(mc.n_ff, 13312);
        assert_eq!(mc.n_vocab, 120832);
        assert_eq!(mc.context_length, 262144);
        assert_eq!(mc.rope_dim_count, 128);
        let moe = mc.moe.as_ref().unwrap();
        assert_eq!(moe.expert_count, 96);
        assert_eq!(moe.expert_used_count, 8);
        assert_eq!(moe.expert_ff_length, 1536);
        assert_eq!(moe.expert_shared_ff_length, 1536);
        let hy3 = mc.hy3.as_ref().unwrap();
        assert!(hy3.sigmoid_routing);
        assert!(hy3.use_routing_bias);
        assert!(hy3.route_norm);
        assert!((hy3.router_scaling_factor - 2.826).abs() < 1e-6);
        assert_eq!(hy3.n_shared_experts, 1);
        assert_eq!(hy3.first_k_dense_replace, 1);
        assert!(hy3.qk_norm);
        assert_eq!(hy3.hidden_act, "silu");
        assert!(!hy3.weight_only_nvfp4);

        let modelopt = HfConfig::parse(
            r#"{"model_type":"hy_v3","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"vocab_size":32,
            "max_position_embeddings":32,
            "quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION"}}"#,
        );
        assert!(modelopt.preserve_checkpoint_bf16);

        let w4a16 = HfConfig::parse(
            r#"{"model_type":"hy_v3","num_hidden_layers":2,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"vocab_size":32,
            "max_position_embeddings":32,
            "quantization_config":{"quant_method":"modelopt","quant_algo":"W4A16_NVFP4"}}"#,
        );
        assert_eq!(w4a16.quant_algo.as_deref(), Some("W4A16_NVFP4"));
        assert!(
            ModelConfig::from_hf(&w4a16)
                .hy3
                .as_ref()
                .unwrap()
                .weight_only_nvfp4
        );
    }

    /// qwen4_exp text_config as the pinned artifact ships it (Qwen/Qwen3.8-Flash-Next @
    /// de4b8e4d, research/qwen4exp-bringup-20260829/raw/config.json) — real values, every
    /// family key, nested under text_config like the real file.
    pub(crate) fn qwen4exp_config_json() -> String {
        let mut layer_types = Vec::new();
        for il in 0..48u32 {
            layer_types.push(if (il + 1) % 4 == 0 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            });
        }
        let lt = layer_types.join(",");
        format!(
            r#"{{
          "architectures": ["Qwen4ExpForConditionalGeneration"],
          "model_type": "qwen4_exp",
          "image_token_id": 248056,
          "text_config": {{
            "model_type": "qwen4_exp_text",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "head_dim": 256,
            "heads_per_ngram": 8,
            "hidden_size": 2560,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "layer_types": [{lt}],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "make_ngram_vocab_size_divisible_by": 128,
            "max_position_embeddings": 262144,
            "moe_intermediate_size": 640,
            "mtp": {{"hybrid": true, "layer_types": ["full_attention"],
                     "num_hidden_layers": 1, "rope_theta": 10000000}},
            "mtp_num_hidden_layers": 1,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "num_attention_heads": 24,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "output_gate_type": "sigmoid",
            "partial_rotary_factor": 0.25,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "rms_norm_eps": 1e-06,
            "rope_parameters": {{"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                                 "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                 "rope_type": "default"}},
            "shared_expert_intermediate_size": 640,
            "split_ngram_parts": 128,
            "tie_word_embeddings": false,
            "vocab_size": 248320
          }},
          "vision_config": {{
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen4_exp",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
          }}
        }}"#
        )
    }

    #[test]
    fn parse_qwen4exp_all_family_fields() {
        let c = HfConfig::parse(&qwen4exp_config_json());
        assert_eq!(c.model_type, "qwen4_exp_text");
        assert_eq!(Arch::from_hf_model_type(&c.model_type), Arch::Qwen4Exp);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen4Exp);
        assert_eq!(mc.n_layer, 49, "48 trunk + 1 MTP (GGUF convention)");
        assert_eq!(mc.nextn_predict_layers, 1);
        assert_eq!(mc.n_embd, 2560);
        assert_eq!(mc.n_head, 24);
        assert_eq!(mc.n_head_kv, 2);
        assert_eq!(mc.head_dim_k, 256);
        assert_eq!(mc.n_vocab, 248320);
        assert_eq!(mc.full_attention_interval, 4);
        // partial rope 0.25 * 256 = 64, theta 1e7 (rope_parameters wins)
        assert_eq!(mc.rope_dim_count, 64);
        assert_eq!(mc.rope_freq_base, 10_000_000.0);
        // mrope sections stored, not implemented (text-only positions are plain rope)
        assert_eq!(mc.rope_sections, vec![11, 11, 10]);
        // GDN geometry from the linear_* keys: QK 16x128, V 48x128, conv k=4
        let ssm = mc.ssm.as_ref().expect("qwen4_exp has GDN layers");
        assert_eq!(ssm.group_count, 16);
        assert_eq!(ssm.time_step_rank, 48);
        assert_eq!(ssm.state_size, 128);
        assert_eq!(ssm.inner_size, 6144);
        assert_eq!(ssm.conv_kernel, 4);
        // MoE: 512 experts top-10, intermediate 640, gated shared expert 640
        let moe = mc.moe.as_ref().expect("qwen4_exp is MoE");
        assert_eq!(
            (
                moe.expert_count,
                moe.expert_used_count,
                moe.expert_ff_length,
                moe.expert_shared_ff_length
            ),
            (512, 10, 640, 640)
        );
        // family sub-config
        let q = mc.qwen4exp.as_ref().expect("qwen4exp config");
        assert_eq!(
            (q.indexer_n_heads, q.indexer_kv_heads, q.indexer_head_dim),
            (4, 1, 128)
        );
        assert_eq!((q.indexer_compress_ratio, q.indexer_budget), (4, 2048));
        assert_eq!(q.indexer_budget_blocks(), 512);
        assert_eq!((q.hc_count, q.hc_lowrank), (4, 320));
        assert_eq!((q.ngram_size, q.heads_per_ngram), (3, 8));
        assert_eq!(q.ngram_heads(), 16);
        assert_eq!(q.ngram_head_embed_dim(), 160);
        assert_eq!(q.ngram_vocab_size_base, 20_000_000);
        assert_eq!(q.make_ngram_vocab_size_divisible_by, 128);
        assert_eq!(q.split_ngram_parts, 128);
        // ple_layer_ids is ONE-indexed: config [2] = checkpoint layers.1 (census receipt:
        // model.language_model.layers.1.ple.conv1d.weight)
        assert_eq!(q.ple_layer_ids, vec![2]);
        assert_eq!(q.ple_checkpoint_layers(), vec![1]);
        assert!(q.has_ple(1) && !q.has_ple(2));
        assert_eq!((q.ple_embed_dim, q.ple_conv_kernel_size), (2560, 4));
        assert_eq!(q.output_gate_type, "sigmoid");
        // scalar on the artifact; PLE history pad + segment-reset id
        assert_eq!(q.eos_token_id, Some(248_044));
        assert_eq!(q.mrope_section, vec![11, 11, 10]);
        assert!(q.mrope_interleaved);
        assert_eq!(q.mtp_num_hidden_layers, 1);
        assert_eq!(q.mtp_rope_theta, 10_000_000.0);
        // ViT tower: family struct is faithful; the GENERIC vision config stays None (the
        // gemma-keyed parse would fabricate wrong geometry, and the tower side-loads).
        assert!(mc.vision.is_none());
        assert!(mc.multimodal.is_none());
        let vt = q.vision.as_ref().expect("qwen4exp vision config");
        assert_eq!(
            (vt.depth, vt.hidden_size, vt.intermediate_size),
            (27, 1152, 4304)
        );
        assert_eq!((vt.num_heads, vt.num_position_embeddings), (16, 2304));
        assert_eq!(
            (vt.patch_size, vt.temporal_patch_size, vt.in_channels),
            (16, 2, 3)
        );
        assert_eq!((vt.out_hidden_size, vt.merger_in()), (2560, 4608));
        // layer classification: (il+1)%4==0 full, else GDN linear; MTP tail full
        assert_eq!(mc.layer_kind(3), LayerKind::FullAttention);
        assert_eq!(mc.layer_kind(0), LayerKind::LinearAttention);
        assert_eq!(mc.layer_kind(48), LayerKind::FullAttention);
        assert_eq!(mc.n_full_attn_layers(), 13, "12 QSA trunk + 1 MTP");
        // fused [q|gate] declared by the arch (q_proj [2*24*256, 2560] measured)
        assert!(mc.attn_out_gate());
        // geometry table: interval rule with FusedQ on full layers ONLY — a FusedQ row on a
        // GDN layer would make q_gate_split read past in_proj widths.
        let table = mc.geometry.as_ref().expect("qwen4_exp geometry table");
        assert_eq!(table.layer_classes().len(), 49);
        let linear = mc.layer_geometry(0).unwrap();
        assert_eq!(linear.mixer, LayerKind::LinearAttention);
        assert_eq!(linear.attention_gate, AttentionGateKind::None);
        let full = mc.layer_geometry(3).unwrap();
        assert_eq!(full.mixer, LayerKind::FullAttention);
        assert_eq!((full.n_head, full.n_head_kv), (24, 2));
        assert_eq!((full.head_dim_k, full.head_dim_v), (256, 256));
        assert_eq!(full.n_rot, 64);
        assert_eq!(full.rope_base, 10_000_000.0);
        assert_eq!(full.window, None);
        assert_eq!(full.attention_gate, AttentionGateKind::FusedQ);
        assert_eq!(
            mc.layer_geometry(48).unwrap().mixer,
            LayerKind::FullAttention,
            "MTP tail layer is full attention"
        );
    }

    /// deepseek_v4 config with `n_extra` drafter entries appended to the 43 trunk
    /// compress_ratios and the vestigial `num_nextn_predict_layers: 1` (both the preview
    /// artifact shape, n_extra = 1, and the 0731 shape, n_extra = 3, carry that key).
    fn dsv4_config_json(n_extra: usize) -> String {
        let mut ratios = vec![0u32, 0];
        for il in 2..43u32 {
            ratios.push(if il % 2 == 0 { 4 } else { 128 });
        }
        ratios.extend(std::iter::repeat_n(0u32, n_extra));
        let rj = ratios
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4",
            "num_hidden_layers":43,"hidden_size":4096,"num_attention_heads":64,
            "num_key_value_heads":1,"head_dim":512,"vocab_size":129280,
            "max_position_embeddings":1048576,"rms_norm_eps":1e-6,"rope_theta":10000,
            "n_routed_experts":256,"n_shared_experts":1,"num_experts_per_tok":6,
            "moe_intermediate_size":2048,"norm_topk_prob":true,"num_hash_layers":3,
            "num_nextn_predict_layers":1,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
            "routed_scaling_factor":1.5,"hc_eps":1e-6,"hc_mult":4,"hc_sinkhorn_iters":20,
            "q_lora_rank":1024,"qk_rope_head_dim":64,"o_lora_rank":1024,"o_groups":8,
            "index_n_heads":64,"index_head_dim":128,"index_topk":512,
            "compress_ratios":[{rj}],"compress_rope_theta":160000,
            "sliding_window":128,"swiglu_limit":10.0,
            "rope_scaling":{{"type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,
            "original_max_position_embeddings":65536}}}}"#
        )
    }

    /// 0731 vestigial-key trap (prep §1.1): num_nextn_predict_layers stays 1 while the
    /// drafter is 3 DSpark blocks; the depth must derive from compress_ratios.
    #[test]
    fn parse_dsv4_0731_derives_nextn_from_compress_ratios() {
        let mc = ModelConfig::from_hf(&HfConfig::parse(&dsv4_config_json(3)));
        assert_eq!(mc.nextn_predict_layers, 3, "derived, NOT the vestigial 1");
        assert_eq!(mc.n_layer, 46);
        assert_eq!(mc.n_layer_total, 46);
        assert_eq!(mc.dsv4.as_ref().unwrap().compress_ratios.len(), 46);
    }

    /// Preview shape (44 entries): the derivation lands on the same value the vestigial
    /// key carried — no behavior change for the existing preview gates.
    #[test]
    fn parse_dsv4_preview_nextn_unchanged() {
        let mc = ModelConfig::from_hf(&HfConfig::parse(&dsv4_config_json(1)));
        assert_eq!(mc.nextn_predict_layers, 1);
        assert_eq!(mc.n_layer, 44);
    }

    /// The REAL banked GLM-5.3-Flash config.json (zai-org/GLM-5.3-Flash @ main, 2026-08-27),
    /// tracked in this repo beside its tensor census.
    fn glm5_banked_json() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/glm53-flash-bringup-20260827/glm-config.json");
        std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "banked GLM-5.3-Flash config.json missing at {}: {e}",
                path.display()
            )
        })
    }

    #[test]
    fn parse_glm5_next_flash_banked_config() {
        let c = HfConfig::parse(&glm5_banked_json());
        // VL wrapper model_type is glm5_next; text_config overrides it with glm5_next_text —
        // both map to the same arch.
        assert_eq!(c.model_type, "glm5_next_text");
        assert_eq!(Arch::from_hf_model_type("glm5_next"), Arch::Glm5Next);
        assert_eq!(Arch::from_hf_model_type(&c.model_type), Arch::Glm5Next);
        assert_eq!(Arch::parse("glm5-next"), Arch::Glm5Next);
        assert_eq!(
            Arch::Glm5Next.attention_gate_kind(),
            Some(AttentionGateKind::None)
        );

        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Glm5Next);
        // 45 trunk layers + 1 NextN (HF config convention includes the appended MTP block).
        assert_eq!(mc.nextn_predict_layers, 1);
        assert_eq!(mc.n_layer, 46);
        assert_eq!(mc.n_layer_total, 46);
        assert_eq!(mc.n_embd, 4096);
        assert_eq!(mc.n_vocab, 154880);
        assert_eq!(mc.context_length, 1048576);
        assert!(mc.moe.as_ref().is_some_and(|moe| moe.expert_count == 288));

        let g = mc
            .glm5
            .as_ref()
            .expect("glm5_next arch carries Glm5NextConfig");
        // 34 KDA / 11 MLA+DSA over the 45-entry trunk schedule.
        assert_eq!(g.kda_layer.len(), 45);
        assert_eq!(g.kda_layer.iter().filter(|&&kda| kda).count(), 34);
        let full: Vec<u32> = (0..45).filter(|&il| !g.is_kda_layer(il)).collect();
        assert_eq!(full, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
        // KDA geometry (linear_attn_config).
        assert_eq!(g.linear_num_heads, 64);
        assert_eq!(g.linear_head_dim, 128);
        assert_eq!(g.linear_conv_kernel, 4);
        assert!((g.gate_lower_bound - (-5.0)).abs() < 1e-6);
        // NoPE MLA.
        assert_eq!(g.q_lora_rank, 1536);
        assert_eq!(g.kv_lora_rank, 512);
        assert_eq!(g.qk_head_dim, 256);
        assert_eq!(g.qk_nope_head_dim, 256);
        assert_eq!(g.qk_rope_head_dim, 0);
        assert_eq!(g.v_head_dim, 256);
        assert!(g.mla_use_nope);
        // Indexer.
        assert_eq!(g.index_n_heads, 32);
        assert_eq!(g.index_head_dim, 128);
        assert_eq!(g.index_topk, 2048);
        assert_eq!(g.index_kpool, 4);
        assert!(g.index_kpool_always_select_tail);
        assert!(g.index_kpool_compress);
        assert!(g.indexer_rope_interleave);
        assert!(g.index_share_for_mtp_iteration);
        assert!((0..45).all(|il| g.has_own_indexer(il)));
        // MoE.
        assert_eq!(g.n_routed_experts, 288);
        assert_eq!(g.num_experts_per_tok, 8);
        assert_eq!(g.moe_intermediate_size, 2048);
        assert_eq!(g.n_shared_experts, 1);
        assert_eq!(g.first_k_dense_replace, 3);
        assert!((0..3).all(|il| g.is_dense_layer(il)));
        assert!(!(3..45).any(|il| g.is_dense_layer(il)));
        assert_eq!(g.scoring_func, "sigmoid");
        assert_eq!(g.topk_method, "noaux_tc");
        assert!((g.routed_scaling_factor - 2.5).abs() < 1e-6);
        assert!(g.norm_topk_prob);
        assert_eq!(g.moe_router_dtype, "float32");
        // mHC.
        assert!(g.mhc);
        assert_eq!(g.hc_mult, 4);
        assert!((g.hc_eps - 1e-6).abs() < 1e-12);
        assert_eq!(g.hc_sinkhorn_iters, 20);
        // Misc.
        assert!((g.swiglu_limit - 10.0).abs() < 1e-6);
        assert_eq!(g.num_nextn_predict_layers, 1);
        assert_eq!(g.output_gate_type, "sigmoid");
    }

    /// layer_types vs linear_attn_config schedule disagreement refuses loudly instead of
    /// picking a winner.
    #[test]
    #[should_panic(expected = "linear_attn_config.kda_layers disagrees")]
    fn glm5_next_layer_types_crosscheck_panics() {
        let json = r#"{"model_type":"glm5_next_text","num_hidden_layers":4,
        "hidden_size":64,"num_attention_heads":4,"vocab_size":16,
        "max_position_embeddings":128,"intermediate_size":128,
        "layer_types":["linear_attention","linear_attention","linear_attention",
        "deepseek_sparse_attention"],
        "linear_attn_config":{"num_heads":4,"head_dim":16,"short_conv_kernel_size":4,
        "gate_lower_bound":-5.0,"kda_layers":[0,1,3],"full_attn_layers":[2]}}"#;
        ModelConfig::from_hf(&HfConfig::parse(json));
    }

    /// A missing required field names its config.json path.
    #[test]
    #[should_panic(expected = "missing required field mlp_layer_types")]
    fn glm5_next_missing_field_panics() {
        let json = r#"{"model_type":"glm5_next_text","num_hidden_layers":4,
        "hidden_size":64,"num_attention_heads":4,"vocab_size":16,
        "max_position_embeddings":128,"intermediate_size":128,
        "layer_types":["linear_attention","linear_attention","linear_attention",
        "deepseek_sparse_attention"],
        "linear_attn_config":{"num_heads":4,"head_dim":16,"short_conv_kernel_size":4,
        "gate_lower_bound":-5.0,"kda_layers":[0,1,2],"full_attn_layers":[3]}}"#;
        ModelConfig::from_hf(&HfConfig::parse(json));
    }
}

#[cfg(test)]
mod minimax_tests {
    use super::*;
    /// Checkpoint dir for the on-disk MiniMax tests below. Like `real_qwen3_17b_header`,
    /// they SKIP (not fail) when the model is absent from the box.
    const MINIMAX_DIR: &str = "/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50";

    #[test]
    fn parse_minimax_m3_vl() {
        let Ok(txt) = std::fs::read_to_string(format!("{MINIMAX_DIR}/config.json")) else {
            eprintln!("SKIP parse_minimax_m3_vl: no model at {MINIMAX_DIR}");
            return;
        };
        let cfg = HfConfig::parse(&txt);
        assert_eq!(Arch::from_hf_model_type(&cfg.model_type), Arch::MinimaxM3);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_local_experts, Some(64)); // REAP50 artifact
        assert_eq!(cfg.num_experts_per_tok, Some(4));
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.dense_intermediate_size, Some(12288));
        assert_eq!(cfg.shared_intermediate_size, Some(3072));
        assert_eq!(cfg.rotary_dim, Some(64));
        assert_eq!(cfg.use_gemma_norm, Some(true));
        assert_eq!(cfg.scoring_func.as_deref(), Some("sigmoid"));
        assert_eq!(cfg.routed_scaling_factor, Some(2.0));
        assert_eq!(
            cfg.moe_layer_freq.as_ref().map(|v| (v.len(), v[0], v[3])),
            Some((60, 0, 1))
        );
        let mc = ModelConfig::from_hf(&cfg);
        assert!(mc.moe.as_ref().is_some_and(|moe| moe.expert_count > 0));
        assert!(mc.arch.is_minimax());
        assert_eq!(mc.moe.as_ref().unwrap().expert_count, 64);
        assert_eq!(mc.moe.as_ref().unwrap().expert_shared_ff_length, 3072);
        assert_eq!(mc.rope_dim_count, 64); // partial RoPE from rotary_dim
        let m3 = mc.m3.as_ref().unwrap();
        assert!(m3.use_gemma_norm && m3.sigmoid_routing && m3.use_routing_bias);
        assert_eq!(m3.routed_scaling_factor, 2.0);
        assert_eq!(m3.n_shared_experts, 1);
        assert_eq!((m3.swiglu_alpha, m3.swiglu_limit), (1.702, 7.0));
        assert_eq!(m3.dense_intermediate_size, 12288);
        assert_eq!(m3.moe_layer_freq.iter().filter(|&&x| x == 0).count(), 3); // 3 dense layers
    }

    /// Name-mapping against the REAL REAP50 shard index: every text-model tensor pattern the
    /// loader will request must resolve to a name present in the safetensors index.
    #[test]
    fn minimax_name_mapping_against_index() {
        use crate::hf_mapping::{HfTarget, ggml_to_hf, hf_expert_name, resolve_ggml};
        let Ok(cfg_txt) = std::fs::read_to_string(format!("{MINIMAX_DIR}/config.json")) else {
            eprintln!("SKIP minimax_name_mapping_against_index: no model at {MINIMAX_DIR}");
            return;
        };
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&cfg_txt));
        let idx: std::collections::HashSet<String> = {
            let txt =
                std::fs::read_to_string(format!("{MINIMAX_DIR}/model.safetensors.index.json"))
                    .unwrap();
            // crude but sufficient: harvest every JSON key that looks like a tensor name
            txt.split('"')
                .filter(|s| s.contains('.') && !s.contains(' '))
                .map(|s| s.to_string())
                .collect()
        };
        // the VL wrapper prefixes the text model with `language_model.` — the source's lookup()
        // fallback strips/adds it; here emulate that for the assertion.
        let has = |hf: &str| idx.contains(hf) || idx.contains(&format!("language_model.{hf}"));

        // top-level + dense attention/norm names (layer 0 = dense-FFN layer, layer 3 = MoE)
        for g in ["token_embd.weight", "output_norm.weight", "output.weight"] {
            let hf = ggml_to_hf(g, &cfg.arch).unwrap();
            assert!(has(&hf), "{g} -> {hf} not in index");
        }
        for g in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.3.ffn_gate_inp.weight",
            "blk.3.exp_probs_b.bias",
            "blk.3.ffn_gate_shexp.weight",
            "blk.3.ffn_up_shexp.weight",
            "blk.3.ffn_down_shexp.weight",
        ] {
            let hf = ggml_to_hf(g, &cfg.arch).unwrap_or_else(|| panic!("{g} unmapped"));
            assert!(has(&hf), "{g} -> {hf} not in index");
        }
        // Mixtral-style per-expert names (w1=gate, w2=down, w3=up)
        for proj in ["gate", "down", "up"] {
            let hf = hf_expert_name(3, 63, proj, &cfg.arch);
            assert!(has(&hf), "expert {proj} -> {hf} not in index");
        }
        // gemma-norm fold: norms must resolve through the Transform(NormPlusOne) arm
        match resolve_ggml("blk.0.attn_norm.weight", &cfg) {
            Some(HfTarget::Transform {
                kind: crate::hf_mapping::TransformKind::NormPlusOne,
                ..
            }) => {}
            _ => panic!("gemma-norm fold not applied to attn_norm"),
        }
    }
}
