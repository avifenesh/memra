//! Inkling 2 Mini (`inkling2_loop`) configuration.
//!
//! The semantic source is the checkpoint's own `config.json` and `hf_quant_config.json`
//! (`thinkingmachines/Inkling-2-Mini`). Values in comments are that artifact's.
//!
//! Every field is REQUIRED by name. A missing field is another program, so the parse refuses
//! instead of defaulting, and [`Inkling2Config::validate`] refuses a declared topology the
//! program cannot run. Nothing here is shared with another family's config.

use crate::config_json::ConfigObject;
use crate::quant_groups::{ActivationQuant, ComputeDtype, QuantGroups};

const FAMILY: &str = "inkling2";

/// Upper bound on cache slots, the canonical compiler's layer cap. A loop multiplies layers by
/// passes, so an unbounded pass count is an unbounded plan.
pub const MAX_CACHE_SLOTS: u32 = 1_024;

/// Inputs per scale block of the widest expert format (MXFP8: 32; NVFP4's 16 and the
/// two-codes-per-byte packing divide it).
const EXPERT_INPUT_BLOCK: u32 = 32;

/// Text trunk, loop, vision tower, drafter and quantization declarations of one checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    // ---- trunk (`text_config`) ----
    pub model_max_length: u32,    // 1048576
    pub hidden_size: u32,         // 2304
    pub num_hidden_layers: u32,   // 24 physical layers
    pub vocab_size: u32,          // 201024: embed and unembed rows (padded)
    pub unpadded_vocab_size: u32, // 200058: logits past it are dropped
    pub num_attention_heads: u32, // 24
    pub num_key_value_heads: u32, // 8
    pub head_dim: u32,            // 128
    pub rms_norm_eps: f32,        // 1e-6
    // ---- relative-position attention ----
    pub d_rel: u32,                    // 16: rank of the relative-position projection
    pub rel_extent: u32,               // 1024: bias table width of a full-attention layer
    pub qk_l2_clip_threshold: f32,     // 22.627416997969522
    pub rel_l2_clip_threshold: f32,    // 8.0
    pub log_scaling_n_floor: u32,      // 8192
    pub log_scaling_alpha: f32,        // 0.1
    pub use_value_norm_no_gamma: bool, // true: V gets a weightless RMS norm
    pub q_bias: bool,                  // false
    pub o_bias: bool,                  // false
    /// Sliding-window layers; every other layer is full attention. 20 of 24:
    /// `[0-4, 6-10, 12-16, 18-22]`, so the full layers are 5, 11, 17 and 23.
    pub local_layer_ids: Vec<u32>,
    pub sliding_window_size: u32,         // 1024
    pub swa_head_dim: u32,                // 128
    pub swa_num_attention_heads: u32,     // 24
    pub swa_num_key_value_heads: u32,     // 8
    pub sconv_kernel_size: u32,           // 4
    pub n_layers_per_attn_res_block: u32, // 6
    pub attn_res_normalize_values: bool,  // true
    // ---- MoE ----
    pub n_routed_experts: u32,    // 128
    pub num_experts_per_tok: u32, // 8
    pub n_shared_experts: u32,    // 0
    pub intermediate_size: u32,   // 1152: one expert's width
    pub mlp_gate_softcap: f32,    // 4.0
    pub mlp_up_softcap: f32,      // 25.0
    pub route_scale: f32,         // 8.0
    pub use_gate_bias: bool,      // true
    // ---- head ----
    pub final_logit_softcapping: f32, // 30.0
    pub eos_token_id: u32,            // 200006
    // ---- loop (`text_config.loop_config`) ----
    pub loop_core_start: u32,  // 6
    pub loop_core_end: u32,    // 18 (exclusive)
    pub loop_num_passes: u32,  // 3
    pub loop_gate_rank: u32,   // 64
    pub loop_is_varloop: bool, // false
    // ---- `quantization_config` of config.json ----
    pub quant_method: String, // "modelopt"
    pub quant_algo: String,   // "NVFP4"
    pub vision: Inkling2VisionConfig,
    pub drafter: Inkling2DrafterConfig,
    /// `hf_quant_config.json`, attached by the source that read it
    /// ([`crate::config::ModelConfig::attach_quant_groups`]). The plan refuses to compile
    /// without it: the expert formats and the required activation quantizers live there.
    pub quant: Option<QuantGroups>,
}

/// `vision_config` (`inkling2_vision`). Contract-owned geometry; the text plan runs none of it.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2VisionConfig {
    pub text_hidden_size: u32,      // 2304
    pub temporal_patch_size: u32,   // 2
    pub native_patch_size: u32,     // 20
    pub adapter_merge_size: u32,    // 2
    pub projection_hidden_dim: u32, // 4096
    pub projector_bias: bool,       // false
    pub hidden_size: u32,           // 1024
    pub num_hidden_layers: u32,     // 23
    pub num_attention_heads: u32,   // 16
    pub mlp_ratio: f32,             // 4.0
    pub posemb_grid_size: u32,      // 32
    pub layer_norm_eps: f32,        // 1e-5
}

/// `mtp_config`: the DSpark drafter shipped in `mtp.safetensors`. Its geometry is declared;
/// its program has no public source code yet, so nothing executes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2DrafterConfig {
    pub hidden_size: u32,             // 2304
    pub num_hidden_layers: u32,       // 3
    pub num_attention_heads: u32,     // 24
    pub num_key_value_heads: u32,     // 8
    pub head_dim: u32,                // 128
    pub intermediate_size: u32,       // 9216 (dense SwiGLU)
    pub max_position_embeddings: u32, // 1048576
    pub q_bias: bool,                 // false
    pub o_bias: bool,                 // false
    pub rms_norm_eps: f32,            // 1e-6
    pub rope_theta: f32,              // 1e6
    pub local_layer_ids: Vec<u32>,    // [0, 2]
    pub sliding_window_size: u32,     // 1024
    pub block_size: u32,              // 8
    pub markov_rank: u32,             // 256
    pub markov_head_type: String,     // "rnn_hidden"
    /// What the drafter reads from the trunk. The ids count layer VISITS in execution order
    /// (prelude, each pass over the core, tail), not cache slots: every id but the last taps the
    /// INPUT of that visit (the running residual stream on entry), and the last equals the
    /// visit count and names the final hidden. Visit 23 is cache slot 29.
    pub target_layer_ids: Vec<u32>, // [5, 23, 41, 48]
}

/// Stored format of one physical layer's expert bank, with the activation quantizer its
/// parameter group requires. Both come from `hf_quant_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2ExpertQuant {
    pub compute: ComputeDtype,
    pub activation: ActivationQuant,
}

fn missing(field: &str) -> String {
    format!("{FAMILY} config.json missing required field {field}")
}

struct Fields<'a, 'b> {
    object: &'b ConfigObject<'a>,
    prefix: &'static str,
}

impl Fields<'_, '_> {
    fn name(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }
    fn u32(&self, key: &str) -> Result<u32, String> {
        self.object
            .u32(key)?
            .ok_or_else(|| missing(&self.name(key)))
    }
    fn f32(&self, key: &str) -> Result<f32, String> {
        self.object
            .f32(key)?
            .ok_or_else(|| missing(&self.name(key)))
    }
    fn boolean(&self, key: &str) -> Result<bool, String> {
        self.object
            .boolean(key)?
            .ok_or_else(|| missing(&self.name(key)))
    }
    fn string(&self, key: &str) -> Result<String, String> {
        self.object
            .string(key)?
            .ok_or_else(|| missing(&self.name(key)))
    }
    fn u32_array(&self, key: &str) -> Result<Vec<u32>, String> {
        self.object
            .u32_array(key)?
            .ok_or_else(|| missing(&self.name(key)))
    }
}

impl Inkling2Config {
    /// Parse the whole `config.json` object (the `inkling2_loop` wrapper).
    pub(crate) fn from_config(top: &ConfigObject<'_>) -> Result<Self, String> {
        let text = top
            .object("text_config")?
            .ok_or_else(|| missing("text_config"))?;
        let t = Fields {
            object: &text,
            prefix: "text_config.",
        };
        let loop_config = text
            .object("loop_config")?
            .ok_or_else(|| missing("text_config.loop_config"))?;
        let l = Fields {
            object: &loop_config,
            prefix: "text_config.loop_config.",
        };
        let vision = top
            .object("vision_config")?
            .ok_or_else(|| missing("vision_config"))?;
        let v = Fields {
            object: &vision,
            prefix: "vision_config.",
        };
        let mtp = top
            .object("mtp_config")?
            .ok_or_else(|| missing("mtp_config"))?;
        let m = Fields {
            object: &mtp,
            prefix: "mtp_config.",
        };
        let quant = top
            .object("quantization_config")?
            .ok_or_else(|| missing("quantization_config"))?;
        let q = Fields {
            object: &quant,
            prefix: "quantization_config.",
        };
        let vision_type = v.string("model_type")?;
        if vision_type != "inkling2_vision" {
            return Err(format!(
                "{FAMILY} config.json: vision_config.model_type is {vision_type:?}, expected \"inkling2_vision\""
            ));
        }
        let config = Self {
            model_max_length: t.u32("model_max_length")?,
            hidden_size: t.u32("hidden_size")?,
            num_hidden_layers: t.u32("num_hidden_layers")?,
            vocab_size: t.u32("vocab_size")?,
            unpadded_vocab_size: t.u32("unpadded_vocab_size")?,
            num_attention_heads: t.u32("num_attention_heads")?,
            num_key_value_heads: t.u32("num_key_value_heads")?,
            head_dim: t.u32("head_dim")?,
            rms_norm_eps: t.f32("rms_norm_eps")?,
            d_rel: t.u32("d_rel")?,
            rel_extent: t.u32("rel_extent")?,
            qk_l2_clip_threshold: t.f32("qk_l2_clip_threshold")?,
            rel_l2_clip_threshold: t.f32("rel_l2_clip_threshold")?,
            log_scaling_n_floor: t.u32("log_scaling_n_floor")?,
            log_scaling_alpha: t.f32("log_scaling_alpha")?,
            use_value_norm_no_gamma: t.boolean("use_value_norm_no_gamma")?,
            q_bias: t.boolean("q_bias")?,
            o_bias: t.boolean("o_bias")?,
            local_layer_ids: t.u32_array("local_layer_ids")?,
            sliding_window_size: t.u32("sliding_window_size")?,
            swa_head_dim: t.u32("swa_head_dim")?,
            swa_num_attention_heads: t.u32("swa_num_attention_heads")?,
            swa_num_key_value_heads: t.u32("swa_num_key_value_heads")?,
            sconv_kernel_size: t.u32("sconv_kernel_size")?,
            n_layers_per_attn_res_block: t.u32("n_layers_per_attn_res_block")?,
            attn_res_normalize_values: t.boolean("attn_res_normalize_values")?,
            n_routed_experts: t.u32("n_routed_experts")?,
            num_experts_per_tok: t.u32("num_experts_per_tok")?,
            n_shared_experts: text
                .u32("n_shared_experts")?
                .ok_or_else(|| missing("text_config.n_shared_experts"))?,
            intermediate_size: t.u32("intermediate_size")?,
            mlp_gate_softcap: t.f32("mlp_gate_softcap")?,
            mlp_up_softcap: t.f32("mlp_up_softcap")?,
            route_scale: t.f32("route_scale")?,
            use_gate_bias: t.boolean("use_gate_bias")?,
            final_logit_softcapping: text
                .f32("final_logit_softcapping")?
                .ok_or_else(|| missing("text_config.final_logit_softcapping"))?,
            eos_token_id: top
                .u32("eos_token_id")?
                .ok_or_else(|| missing("eos_token_id"))?,
            loop_core_start: l.u32("core_start")?,
            loop_core_end: l.u32("core_end")?,
            loop_num_passes: l.u32("num_passes")?,
            loop_gate_rank: l.u32("gate_rank")?,
            loop_is_varloop: l.boolean("is_varloop")?,
            quant_method: q.string("quant_method")?,
            quant_algo: q.string("quant_algo")?,
            vision: Inkling2VisionConfig {
                text_hidden_size: v.u32("text_hidden_size")?,
                temporal_patch_size: v.u32("temporal_patch_size")?,
                native_patch_size: v.u32("native_patch_size")?,
                adapter_merge_size: v.u32("adapter_merge_size")?,
                projection_hidden_dim: v.u32("projection_hidden_dim")?,
                projector_bias: v.boolean("projector_bias")?,
                hidden_size: v.u32("hidden_size")?,
                num_hidden_layers: v.u32("num_hidden_layers")?,
                num_attention_heads: v.u32("num_attention_heads")?,
                mlp_ratio: v.f32("mlp_ratio")?,
                posemb_grid_size: v.u32("posemb_grid_size")?,
                layer_norm_eps: v.f32("layer_norm_eps")?,
            },
            drafter: Inkling2DrafterConfig {
                hidden_size: m.u32("hidden_size")?,
                num_hidden_layers: m.u32("num_hidden_layers")?,
                num_attention_heads: m.u32("num_attention_heads")?,
                num_key_value_heads: mtp
                    .u32("num_key_value_heads")?
                    .ok_or_else(|| missing("mtp_config.num_key_value_heads"))?,
                head_dim: mtp
                    .u32("head_dim")?
                    .ok_or_else(|| missing("mtp_config.head_dim"))?,
                intermediate_size: m.u32("intermediate_size")?,
                max_position_embeddings: m.u32("max_position_embeddings")?,
                q_bias: m.boolean("q_bias")?,
                o_bias: m.boolean("o_bias")?,
                rms_norm_eps: m.f32("rms_norm_eps")?,
                rope_theta: m.f32("rope_theta")?,
                local_layer_ids: m.u32_array("local_layer_ids")?,
                sliding_window_size: m.u32("sliding_window_size")?,
                block_size: m.u32("block_size")?,
                markov_rank: m.u32("markov_rank")?,
                markov_head_type: m.string("markov_head_type")?,
                target_layer_ids: m.u32_array("target_layer_ids")?,
            },
            quant: None,
        };
        config
            .validate()
            .map_err(|why| format!("{FAMILY} config.json: {why}"))?;
        Ok(config)
    }

    /// Physical layers of the loop core.
    pub fn core_len(&self) -> u32 {
        self.loop_core_end - self.loop_core_start
    }

    /// Cache slots: one per physical layer, plus one per core layer for every pass after the
    /// first. 24 + 2 x 12 = 48. Each slot owns its KV rows and its conv state.
    pub fn slots(&self) -> u32 {
        self.num_hidden_layers + (self.loop_num_passes - 1) * self.core_len()
    }

    /// The physical layer whose weights slot `slot` runs.
    pub fn physical_layer(&self, slot: u32) -> u32 {
        if slot < self.num_hidden_layers {
            slot
        } else {
            self.loop_core_start + (slot - self.num_hidden_layers) % self.core_len()
        }
    }

    /// The loop pass slot `slot` belongs to; `None` outside the core.
    pub fn slot_pass(&self, slot: u32) -> Option<u32> {
        if slot >= self.num_hidden_layers {
            Some(1 + (slot - self.num_hidden_layers) / self.core_len())
        } else if (self.loop_core_start..self.loop_core_end).contains(&slot) {
            Some(0)
        } else {
            None
        }
    }

    /// The slot of core layer `physical` in pass `pass`.
    pub fn core_slot(&self, pass: u32, physical: u32) -> u32 {
        if pass == 0 {
            physical
        } else {
            self.num_hidden_layers
                + (pass - 1) * self.core_len()
                + (physical - self.loop_core_start)
        }
    }

    /// Slots in execution order: the prelude, every pass over the core, then the tail.
    pub fn execution_order(&self) -> Vec<u32> {
        let mut order: Vec<u32> = (0..self.loop_core_start).collect();
        for pass in 0..self.loop_num_passes {
            order.extend(
                (self.loop_core_start..self.loop_core_end)
                    .map(|physical| self.core_slot(pass, physical)),
            );
        }
        order.extend(self.loop_core_end..self.num_hidden_layers);
        order
    }

    /// Sliding-window layer (`local_layer_ids`); the others are full attention.
    pub fn is_local(&self, physical: u32) -> bool {
        self.local_layer_ids.contains(&physical)
    }

    /// First layer of an attention-residual block: it starts a new residual stream.
    pub fn starts_block(&self, physical: u32) -> bool {
        physical.is_multiple_of(self.n_layers_per_attn_res_block)
    }

    /// Short-conv state channels of one slot: K, V, the attention output and the MLP output.
    pub fn conv_state_channels(&self) -> u32 {
        2 * self.num_key_value_heads * self.head_dim + 2 * self.hidden_size
    }

    /// The expert bank format and required activation quantizer of `physical`, from the
    /// attached `hf_quant_config.json`. Both projections of a layer must sit in one group kind.
    pub fn expert_quant(&self, physical: u32) -> Result<Inkling2ExpertQuant, String> {
        let quant = self.quant.as_ref().ok_or_else(|| {
            "hf_quant_config.json is not attached: the expert formats and the required activation quantizers are declared there".to_owned()
        })?;
        let mut found: Option<Inkling2ExpertQuant> = None;
        for projection in ["w13_weight", "w2_weight"] {
            let name = format!("model.llm.layers.{physical}.mlp.experts.{projection}");
            let group = quant
                .model_group(&name)
                .ok_or_else(|| format!("hf_quant_config.json declares no group for {name}"))?;
            let activation = group.required_activation_quantization.ok_or_else(|| {
                format!("hf_quant_config.json: {name} has no required_activation_quantization")
            })?;
            let this = Inkling2ExpertQuant {
                compute: group.compute_dtype,
                activation,
            };
            if found.is_some_and(|first| first != this) {
                return Err(format!(
                    "hf_quant_config.json: layer {physical} declares its two expert projections in different formats"
                ));
            }
            found = Some(this);
        }
        Ok(found.expect("two projections were read"))
    }

    /// Attach a parsed `hf_quant_config.json` after checking that it is this family's file and
    /// that every physical layer's expert bank has a typed format.
    pub fn attach_quant(&mut self, quant: QuantGroups) -> Result<(), String> {
        if quant.quant_method != "inkling2" {
            return Err(format!(
                "hf_quant_config.json quant_method is {:?}; the {FAMILY} reader accepts \"inkling2\"",
                quant.quant_method
            ));
        }
        self.quant = Some(quant);
        for physical in 0..self.num_hidden_layers {
            if let Err(why) = self.expert_quant(physical) {
                self.quant = None;
                return Err(why);
            }
        }
        Ok(())
    }

    /// The structural invariants the program relies on (llama.cpp `inkling2.cpp` and
    /// `inkling2-loop.cpp` `load_arch_hparams`, plus what the typed plan cannot express).
    pub fn validate(&self) -> Result<(), String> {
        let expect = |ok: bool, what: String| if ok { Ok(()) } else { Err(what) };
        let layers = self.num_hidden_layers;
        let block = self.n_layers_per_attn_res_block;
        for (field, value) in [
            ("model_max_length", self.model_max_length),
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("d_rel", self.d_rel),
            ("rel_extent", self.rel_extent),
            ("sliding_window_size", self.sliding_window_size),
            ("log_scaling_n_floor", self.log_scaling_n_floor),
            ("n_layers_per_attn_res_block", block),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("intermediate_size", self.intermediate_size),
            ("unpadded_vocab_size", self.unpadded_vocab_size),
            ("loop_config.gate_rank", self.loop_gate_rank),
            ("loop_config.core_start", self.loop_core_start),
            ("vision_config.hidden_size", self.vision.hidden_size),
            (
                "vision_config.num_hidden_layers",
                self.vision.num_hidden_layers,
            ),
            (
                "vision_config.num_attention_heads",
                self.vision.num_attention_heads,
            ),
            (
                "vision_config.native_patch_size",
                self.vision.native_patch_size,
            ),
            (
                "vision_config.adapter_merge_size",
                self.vision.adapter_merge_size,
            ),
            (
                "vision_config.projection_hidden_dim",
                self.vision.projection_hidden_dim,
            ),
            (
                "vision_config.posemb_grid_size",
                self.vision.posemb_grid_size,
            ),
            ("mtp_config.hidden_size", self.drafter.hidden_size),
            (
                "mtp_config.num_hidden_layers",
                self.drafter.num_hidden_layers,
            ),
            (
                "mtp_config.num_attention_heads",
                self.drafter.num_attention_heads,
            ),
            (
                "mtp_config.num_key_value_heads",
                self.drafter.num_key_value_heads,
            ),
            ("mtp_config.head_dim", self.drafter.head_dim),
            (
                "mtp_config.intermediate_size",
                self.drafter.intermediate_size,
            ),
            (
                "mtp_config.sliding_window_size",
                self.drafter.sliding_window_size,
            ),
            ("mtp_config.markov_rank", self.drafter.markov_rank),
        ] {
            expect(value > 0, format!("{field} is 0"))?;
        }
        // The expert banks are packed two codes per byte and scaled per 16 (NVFP4) or 32 (MXFP8)
        // inputs, over both the hidden width (gate/up input) and the expert width (down input).
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
        ] {
            expect(
                value.is_multiple_of(EXPERT_INPUT_BLOCK),
                format!(
                    "{field} {value} is not a whole number of {EXPERT_INPUT_BLOCK}-input scale blocks"
                ),
            )?;
        }
        // The ViT MLP width is hidden * mlp_ratio and must be a whole number.
        let vision_mlp = f64::from(self.vision.hidden_size) * f64::from(self.vision.mlp_ratio);
        expect(
            vision_mlp.is_finite() && vision_mlp >= 1.0 && vision_mlp.fract() == 0.0,
            format!(
                "vision_config.hidden_size {} x mlp_ratio {} is not a whole width",
                self.vision.hidden_size, self.vision.mlp_ratio
            ),
        )?;
        for (field, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("qk_l2_clip_threshold", self.qk_l2_clip_threshold),
            ("rel_l2_clip_threshold", self.rel_l2_clip_threshold),
            ("mlp_gate_softcap", self.mlp_gate_softcap),
            ("mlp_up_softcap", self.mlp_up_softcap),
            ("route_scale", self.route_scale),
            ("final_logit_softcapping", self.final_logit_softcapping),
        ] {
            expect(
                value.is_finite() && value > 0.0,
                format!("{field} is {value}; it must be finite and positive"),
            )?;
        }
        expect(
            self.log_scaling_alpha.is_finite() && self.log_scaling_alpha >= 0.0,
            format!("log_scaling_alpha is {}", self.log_scaling_alpha),
        )?;
        expect(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            format!(
                "num_attention_heads {} is not a multiple of num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            ),
        )?;
        // One attention geometry serves both layer classes in the typed plan.
        expect(
            self.swa_head_dim == self.head_dim
                && self.swa_num_attention_heads == self.num_attention_heads
                && self.swa_num_key_value_heads == self.num_key_value_heads,
            format!(
                "sliding-window geometry {}q/{}kv x {} differs from the full-attention geometry {}q/{}kv x {}",
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim
            ),
        )?;
        for (field, value, want) in [
            ("q_bias", self.q_bias, false),
            ("o_bias", self.o_bias, false),
            (
                "use_value_norm_no_gamma",
                self.use_value_norm_no_gamma,
                true,
            ),
            (
                "attn_res_normalize_values",
                self.attn_res_normalize_values,
                true,
            ),
            ("use_gate_bias", self.use_gate_bias, true),
            ("loop_config.is_varloop", self.loop_is_varloop, false),
        ] {
            expect(
                value == want,
                format!("{field} is {value}; the typed program is {want}"),
            )?;
        }
        expect(
            self.n_shared_experts == 0,
            format!(
                "n_shared_experts is {}; the program has none",
                self.n_shared_experts
            ),
        )?;
        expect(
            self.num_experts_per_tok <= self.n_routed_experts,
            format!(
                "num_experts_per_tok {} exceeds n_routed_experts {}",
                self.num_experts_per_tok, self.n_routed_experts
            ),
        )?;
        expect(
            (2..=64).contains(&self.sconv_kernel_size),
            format!(
                "sconv_kernel_size {} is outside 2..=64",
                self.sconv_kernel_size
            ),
        )?;
        expect(
            self.unpadded_vocab_size <= self.vocab_size,
            format!(
                "unpadded_vocab_size {} exceeds vocab_size {}",
                self.unpadded_vocab_size, self.vocab_size
            ),
        )?;
        expect(
            self.eos_token_id < self.unpadded_vocab_size,
            format!(
                "eos_token_id {} is outside the unpadded vocabulary",
                self.eos_token_id
            ),
        )?;
        expect(
            self.local_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1])
                && self.local_layer_ids.iter().all(|&layer| layer < layers),
            format!(
                "local_layer_ids {:?} is not a strictly increasing list of layers below {layers}",
                self.local_layer_ids
            ),
        )?;
        // Loop: at least two passes of a fixed loop, a non-empty prelude, block-aligned bounds.
        expect(
            self.loop_num_passes >= 2,
            format!(
                "loop_config.num_passes is {}; a fixed loop needs at least 2",
                self.loop_num_passes
            ),
        )?;
        expect(
            self.loop_core_start < self.loop_core_end && self.loop_core_end <= layers,
            format!(
                "loop core [{}, {}) is not inside {layers} layers",
                self.loop_core_start, self.loop_core_end
            ),
        )?;
        // Every pass after the first adds one cache slot per core layer. Checked arithmetic:
        // `slots()` and everything after this line may use the plain form.
        let slots = (self.loop_num_passes - 1)
            .checked_mul(self.loop_core_end - self.loop_core_start)
            .and_then(|looped| looped.checked_add(layers))
            .filter(|&slots| slots <= MAX_CACHE_SLOTS);
        let Some(slots) = slots else {
            return Err(format!(
                "{layers} layers with {} passes over a core of {} need more than {MAX_CACHE_SLOTS} cache slots",
                self.loop_num_passes,
                self.loop_core_end - self.loop_core_start
            ));
        };
        expect(
            self.loop_core_start.is_multiple_of(block) && self.loop_core_end.is_multiple_of(block),
            format!(
                "loop core [{}, {}) is not aligned to the attention-residual block of {block}",
                self.loop_core_start, self.loop_core_end
            ),
        )?;
        expect(
            self.quant_method == "modelopt" && self.quant_algo == "NVFP4",
            format!(
                "quantization_config is {}/{}; the pinned declaration is modelopt/NVFP4",
                self.quant_method, self.quant_algo
            ),
        )?;
        expect(
            self.vision.text_hidden_size == self.hidden_size,
            format!(
                "vision_config.text_hidden_size {} differs from hidden_size {}",
                self.vision.text_hidden_size, self.hidden_size
            ),
        )?;
        // Drafter: declared geometry must sit on this trunk.
        let d = &self.drafter;
        expect(
            d.hidden_size == self.hidden_size,
            format!(
                "mtp_config.hidden_size {} differs from hidden_size {}",
                d.hidden_size, self.hidden_size
            ),
        )?;
        expect(
            d.markov_head_type == "rnn_hidden",
            format!(
                "mtp_config.markov_head_type is {:?}; only \"rnn_hidden\" is declared",
                d.markov_head_type
            ),
        )?;
        expect(
            d.local_layer_ids.windows(2).all(|pair| pair[0] < pair[1])
                && d.local_layer_ids
                    .iter()
                    .all(|&layer| layer < d.num_hidden_layers),
            format!(
                "mtp_config.local_layer_ids {:?} is not a strictly increasing list of blocks below {}",
                d.local_layer_ids, d.num_hidden_layers
            ),
        )?;
        for (field, value) in [
            ("mtp_config.q_bias", d.q_bias),
            ("mtp_config.o_bias", d.o_bias),
        ] {
            expect(
                !value,
                format!("{field} is true; the drafter's projections carry no bias tensor"),
            )?;
        }
        expect(
            d.num_attention_heads.is_multiple_of(d.num_key_value_heads),
            format!(
                "mtp_config.num_attention_heads {} is not a multiple of mtp_config.num_key_value_heads {}",
                d.num_attention_heads, d.num_key_value_heads
            ),
        )?;
        // One anchor row plus at least one drafted row.
        expect(
            d.block_size >= 2,
            format!(
                "mtp_config.block_size is {}; a block is the anchor plus at least one drafted row",
                d.block_size
            ),
        )?;
        for (field, value) in [
            ("mtp_config.rms_norm_eps", d.rms_norm_eps),
            ("mtp_config.rope_theta", d.rope_theta),
        ] {
            expect(
                value.is_finite() && value > 0.0,
                format!("{field} is {value}; it must be finite and positive"),
            )?;
        }
        // The ids count layer VISITS in execution order (there are as many visits as cache
        // slots). Every id but the last names a visit whose input is tapped; the last must be
        // the visit count itself, which names the final hidden.
        let taps = &d.target_layer_ids;
        expect(
            taps.len() >= 2,
            format!(
                "mtp_config.target_layer_ids {taps:?} needs at least two entries: a visit and the final hidden"
            ),
        )?;
        expect(
            taps.windows(2).all(|pair| pair[0] < pair[1]),
            format!("mtp_config.target_layer_ids {taps:?} is not strictly increasing"),
        )?;
        expect(
            taps.last() == Some(&slots),
            format!(
                "mtp_config.target_layer_ids {taps:?} does not end at {slots}, the visit count that names the final hidden"
            ),
        )?;
        Ok(())
    }
}
