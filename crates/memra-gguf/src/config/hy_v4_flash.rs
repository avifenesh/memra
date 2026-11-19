//! Hy4-flash (`hy_v4_flash`) config: every `text_config` field the transformers
//! `modeling_hy_v4_flash.py` program reads, required by name, plus the per-layer ModelOpt map
//! of the NVFP4 artifact and the vision-tower geometry its census needs.
//!
//! Semantic source: transformers PR #50903 at b865850d0532 (`configuration_hy_v4_flash.py`,
//! `modeling_hy_v4_flash.py`), pinned in research/modelplan-onboarding-hy4flash-20261119/. A
//! switch that selects a different program than the one the pack compiles (Sinkhorn
//! hyper-connections, a softmax router, a fine indexer, an output LoRA, ...) is refused by name
//! rather than ignored.

use crate::config_json::ConfigObject;
use crate::modelopt_quant::LayerQuantMap;

/// Attention layer type (`layer_types`, `swa_pattern`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyV4LayerKind {
    /// `deepseek_sparse_attention` (transformers `indexed_attention`): full causal attention
    /// narrowed by the compressed indexer once the context passes `index_topk`.
    Indexed,
    /// `sliding_attention`: the last `sliding_window` keys, with its own head count.
    Sliding,
}

/// Head geometry of one attention layer type (`per_layer_config`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyV4HeadGeometry {
    pub heads: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub rope_theta: f32,
}

/// The vision tower geometry (`vision_config`). The text plan does not run the tower; the
/// tensor contract owns its tensors from this geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyV4VisionConfig {
    pub num_hidden_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_channels: u32,
    pub patch_size: u32,
    pub max_vit_seq_len: u32,
    pub spatial_merge_size: u32,
    pub out_hidden_size: u32,
}

/// DFly2 drafter geometry (top-level `dfly_*`). The text plan carries no drafter yet; the
/// tensor contract owns its tensors from this geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyV4DflyConfig {
    pub num_hidden_layers: u32,
    pub mask_token_id: u32,
    /// 0-based target layers whose KV and indexer top-k the drafter reads.
    pub target_kv_layers: Vec<u32>,
    pub target_index_layers: Vec<u32>,
    pub hidden_correction: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyV4FlashConfig {
    pub num_hidden_layers: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: u32,
    pub layer_kinds: Vec<HyV4LayerKind>,
    /// True for a dense MLP layer (`mlp_layer_types` "dense"), false for the latent MoE.
    pub dense_layers: Vec<bool>,
    pub indexed: HyV4HeadGeometry,
    pub sliding: HyV4HeadGeometry,
    pub sliding_window: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    /// `compressed_indexer_compress_ratio` (transformers `index_kpool`).
    pub index_pool: u32,
    pub hc_mult: u32,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub intermediate_size: u32,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub routed_expert_hidden_size: u32,
    pub n_shared_experts: u32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub swiglu_limit: f32,
    /// 1-based layer ids (the module sits in 0-based decoder layer `id - 1`).
    pub engram_layer_ids: Vec<u32>,
    pub engram_max_ngram_size: u32,
    pub engram_n_head_per_ngram: u32,
    pub engram_n_embed_per_ngram: u32,
    /// Per engram layer, in `engram_layer_ids` order: the prime moduli of orders 2..max, each
    /// `engram_n_head_per_ngram` long, flattened order-major.
    pub engram_head_vocab_sizes: Vec<Vec<u64>>,
    pub engram_pad_id: u32,
    pub engram_seed: u64,
    pub engram_fp8_embedding: bool,
    /// transformers `engram_lookup_table_size`: not in the artifact's config, so the
    /// configuration class default (120818) applies; the census checks the tensor's shape.
    pub engram_lookup_table_size: u32,
    pub tie_word_embeddings: bool,
    pub quant: LayerQuantMap,
    pub vision: HyV4VisionConfig,
    pub dfly: HyV4DflyConfig,
}

/// transformers `HYV4FlashTextConfig.engram_lookup_table_size` default.
pub const ENGRAM_LOOKUP_TABLE_SIZE: u32 = 120_818;

impl HyV4FlashConfig {
    pub fn trunk_layers(&self) -> u32 {
        self.num_hidden_layers
    }

    pub fn head_geometry(&self, layer: u32) -> HyV4HeadGeometry {
        match self.layer_kinds[layer as usize] {
            HyV4LayerKind::Indexed => self.indexed,
            HyV4LayerKind::Sliding => self.sliding,
        }
    }

    /// The engram slot (index into `engram_layer_ids`) whose module runs in 0-based decoder
    /// layer `layer`: the one whose 1-based id is `layer + 1`.
    pub fn engram_slot(&self, layer: u32) -> Option<usize> {
        self.engram_layer_ids.iter().position(|id| *id == layer + 1)
    }

    /// Rows of an engram table: the sum of its hash moduli.
    pub fn engram_table_rows(&self, slot: usize) -> u64 {
        self.engram_head_vocab_sizes[slot].iter().sum()
    }

    pub fn engram_heads(&self) -> u32 {
        (self.engram_max_ngram_size - 1) * self.engram_n_head_per_ngram
    }

    pub fn engram_head_dim(&self) -> u32 {
        self.engram_n_embed_per_ngram / self.engram_n_head_per_ngram
    }

    /// The hash-multiplier seed of an engram layer: `engram_seed + 10007 * layer_id` with the
    /// 1-based id (transformers `draw_hash_multipliers`, vLLM `FlashNgramHashState`).
    pub fn engram_hash_seed(&self, slot: usize) -> u64 {
        self.engram_seed + 10_007 * u64::from(self.engram_layer_ids[slot])
    }

    /// Parse from the top-level config object and its `text_config`.
    pub(crate) fn parse(top: &ConfigObject, text: &ConfigObject) -> Result<Self, String> {
        let missing = |key: &str| format!("hy_v4_flash config.json missing required field {key}");
        let refuse = |key: &str, value: &dyn std::fmt::Debug, why: &str| {
            format!("hy_v4_flash config.json {key} = {value:?}: {why}")
        };
        let u = |key: &str| text.u32(key)?.ok_or_else(|| missing(key));
        let f = |key: &str| text.f32(key)?.ok_or_else(|| missing(key));
        let b = |key: &str| text.boolean(key)?.ok_or_else(|| missing(key));
        let s = |key: &str| text.string(key)?.ok_or_else(|| missing(key));
        let expect_bool = |key: &str, want: bool, why: &str| -> Result<(), String> {
            let got = b(key)?;
            if got != want {
                return Err(refuse(key, &got, why));
            }
            Ok(())
        };
        let expect_u = |key: &str, want: u32, why: &str| -> Result<(), String> {
            let got = u(key)?;
            if got != want {
                return Err(refuse(key, &got, why));
            }
            Ok(())
        };
        let expect_s = |key: &str, want: &str, why: &str| -> Result<(), String> {
            let got = s(key)?;
            if got != want {
                return Err(refuse(key, &got, why));
            }
            Ok(())
        };

        let layers = u("num_hidden_layers")?;
        let hidden = u("hidden_size")?;
        if layers == 0 || hidden == 0 {
            return Err(refuse(
                "num_hidden_layers/hidden_size",
                &(layers, hidden),
                "empty model",
            ));
        }

        // The program switches. Each value below is the one the transformers program runs; any
        // other selects math this pack does not compile.
        expect_bool("use_mla", true, "the pack compiles gated MLA only")?;
        expect_bool(
            "gated_mla",
            true,
            "the attention output carries a sigmoid gate",
        )?;
        expect_s(
            "gating_type",
            "elementwise",
            "the gate is one sigmoid per value element",
        )?;
        expect_s(
            "gate_lora_op",
            "none",
            "a low-rank gate projection is another program",
        )?;
        expect_s(
            "o_lora_op",
            "none",
            "a low-rank output projection is another program",
        )?;
        expect_u(
            "o_groups",
            1,
            "grouped output projection is another program",
        )?;
        expect_u("gate_groups", 1, "grouped gates are another program")?;
        expect_bool("learnable_sink", true, "every layer carries a learned sink")?;
        expect_bool(
            "qk_norm",
            true,
            "q_a_layernorm and kv_a_layernorm are part of the program",
        )?;
        expect_bool(
            "sandwich_norm",
            true,
            "post_proj_norm and the MLP output norm are part of the program",
        )?;
        expect_bool(
            "dsa_skip_attn_post_norm",
            false,
            "indexed layers keep post_proj_norm",
        )?;
        expect_bool(
            "full_nope",
            false,
            "the trailing qk_rope_head_dim dims rotate",
        )?;
        expect_bool("attention_bias", false, "the projections carry no bias")?;
        expect_bool(
            "enable_ihc",
            true,
            "independent hyper-connections (no Sinkhorn)",
        )?;
        expect_bool("enable_lm_head_fp32", true, "the head is computed in F32")?;
        expect_bool("use_dsa", true, "indexed layers are part of the program")?;
        expect_bool(
            "use_compressed_indexer",
            true,
            "the indexer pools token keys",
        )?;
        expect_bool(
            "compressed_indexer_skip_fine_select",
            true,
            "a fine per-token selection after the pooled one is another program",
        )?;
        expect_bool(
            "compressed_indexer_reuse_fine_indexer",
            false,
            "indexer reuse is another program",
        )?;
        expect_s(
            "compressed_indexer_key_pooling",
            "gated",
            "the pool is a softmax-gated sum",
        )?;
        expect_bool(
            "moe_router_use_sigmoid",
            true,
            "the router scores are sigmoids",
        )?;
        expect_bool(
            "moe_router_enable_expert_bias",
            true,
            "e_score_correction_bias steers selection",
        )?;
        expect_u("n_group", 1, "grouped expert selection is another program")?;
        expect_u(
            "topk_group",
            1,
            "grouped expert selection is another program",
        )?;
        expect_s("hidden_act", "silu", "the MLPs are SwiGLU over silu")?;
        expect_bool(
            "engram_isolate_fffd_namespace",
            false,
            "an isolated U+FFFD namespace changes the n-gram hash",
        )?;
        expect_bool(
            "engram_fp8_embedding",
            true,
            "the engram rows are read through a per-row FP8 quantization",
        )?;
        if let Some(rank) = text.u32("o_lora_rank")? {
            return Err(refuse(
                "o_lora_rank",
                &rank,
                "no output LoRA in this program",
            ));
        }
        if let Some(rank) = text.u32("gate_lora_rank")? {
            return Err(refuse(
                "gate_lora_rank",
                &rank,
                "no gate LoRA in this program",
            ));
        }
        let tie = text
            .boolean("tie_word_embeddings")?
            .or(top.boolean("tie_word_embeddings")?)
            .ok_or_else(|| missing("tie_word_embeddings"))?;
        if tie {
            return Err(refuse(
                "tie_word_embeddings",
                &tie,
                "the artifact ships lm_head.weight",
            ));
        }

        // Layer types: `layer_types` and `swa_pattern` must say the same thing.
        let layer_types = text
            .string_array("layer_types")?
            .ok_or_else(|| missing("layer_types"))?;
        let swa_pattern = text
            .u32_array("swa_pattern")?
            .ok_or_else(|| missing("swa_pattern"))?;
        if layer_types.len() != layers as usize || swa_pattern.len() != layers as usize {
            return Err(refuse(
                "layer_types/swa_pattern length",
                &(layer_types.len(), swa_pattern.len()),
                &format!("num_hidden_layers is {layers}"),
            ));
        }
        let mut layer_kinds = Vec::with_capacity(layers as usize);
        for (il, (kind, swa)) in layer_types.iter().zip(&swa_pattern).enumerate() {
            let parsed = match kind.as_str() {
                "deepseek_sparse_attention" | "indexed_attention" => HyV4LayerKind::Indexed,
                "sliding_attention" => HyV4LayerKind::Sliding,
                other => {
                    return Err(refuse(
                        &format!("layer_types[{il}]"),
                        &other,
                        "not an attention type of this program",
                    ));
                }
            };
            if (parsed == HyV4LayerKind::Sliding) != (*swa == 1) {
                return Err(refuse(
                    &format!("swa_pattern[{il}]"),
                    swa,
                    &format!("contradicts layer_types[{il}] = {kind:?}"),
                ));
            }
            layer_kinds.push(parsed);
        }
        let mlp_types = text
            .string_array("mlp_layer_types")?
            .ok_or_else(|| missing("mlp_layer_types"))?;
        if mlp_types.len() != layers as usize {
            return Err(refuse(
                "mlp_layer_types length",
                &mlp_types.len(),
                &format!("num_hidden_layers is {layers}"),
            ));
        }
        let dense_layers = mlp_types
            .iter()
            .enumerate()
            .map(|(il, kind)| match kind.as_str() {
                "dense" => Ok(true),
                "sparse" => Ok(false),
                other => Err(refuse(
                    &format!("mlp_layer_types[{il}]"),
                    &other,
                    "neither dense nor sparse",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let first_dense = u("first_k_dense_replace")?;
        let leading_dense = dense_layers.iter().take_while(|dense| **dense).count() as u32;
        if leading_dense != first_dense
            || dense_layers.iter().filter(|d| **d).count() as u32 != first_dense
        {
            return Err(refuse(
                "first_k_dense_replace",
                &first_dense,
                "mlp_layer_types must be that many leading dense layers, then sparse",
            ));
        }

        // RoPE per layer type: plain parameters (text positions on all three M-RoPE axes).
        let rope = text
            .object("rope_parameters")?
            .ok_or_else(|| missing("rope_parameters"))?;
        let rope_theta = |kind: &str, rope_dims: u32| -> Result<f32, String> {
            let params = rope
                .object(kind)?
                .ok_or_else(|| missing(&format!("rope_parameters.{kind}")))?;
            let key = |k: &str| format!("rope_parameters.{kind}.{k}");
            match params.string("rope_type")?.as_deref() {
                Some("default") => {}
                other => return Err(refuse(&key("rope_type"), &other, "only unscaled RoPE")),
            }
            match params.boolean("mrope_interleaved")? {
                Some(true) => {}
                other => {
                    return Err(refuse(
                        &key("mrope_interleaved"),
                        &other,
                        "interleaved M-RoPE",
                    ));
                }
            }
            let section = params
                .u32_array("mrope_section")?
                .ok_or_else(|| missing(&key("mrope_section")))?;
            if section.len() != 3 || section.iter().sum::<u32>() * 2 != rope_dims {
                return Err(refuse(
                    &key("mrope_section"),
                    &section,
                    "three axes summing to half the rotary width",
                ));
            }
            params
                .f32("rope_theta")?
                .ok_or_else(|| missing(&key("rope_theta")))
        };
        let qk_rope = u("qk_rope_head_dim")?;
        let swa_rope = u("swa_qk_rope_head_dim")?;
        let indexed = HyV4HeadGeometry {
            heads: u("num_attention_heads")?,
            qk_nope_head_dim: u("qk_nope_head_dim")?,
            qk_rope_head_dim: qk_rope,
            v_head_dim: u("v_head_dim")?,
            rope_theta: rope_theta("deepseek_sparse_attention", qk_rope)?,
        };
        let sliding = HyV4HeadGeometry {
            heads: u("swa_num_attention_heads")?,
            qk_nope_head_dim: u("swa_qk_nope_head_dim")?,
            qk_rope_head_dim: swa_rope,
            v_head_dim: u("swa_v_head_dim")?,
            rope_theta: rope_theta("sliding_attention", swa_rope)?,
        };
        for (key, geometry, combined) in [
            ("qk_head_dim", indexed, u("qk_head_dim")?),
            ("swa_qk_head_dim", sliding, u("swa_qk_head_dim")?),
        ] {
            if geometry.qk_nope_head_dim + geometry.qk_rope_head_dim != combined {
                return Err(refuse(key, &combined, "must be nope + rope"));
            }
        }
        // MLA expands one latent to one key/value per query head.
        if u("num_key_value_heads")? != indexed.heads {
            return Err(refuse(
                "num_key_value_heads",
                &u("num_key_value_heads")?,
                "MLA keeps one key/value head per query head",
            ));
        }
        let sliding_window = u("sliding_window")?;
        let window_size = text
            .u32_array("window_size")?
            .ok_or_else(|| missing("window_size"))?;
        if sliding_window == 0 || window_size != [sliding_window - 1, 0] {
            return Err(refuse(
                "window_size",
                &window_size,
                &format!(
                    "a causal window of sliding_window {sliding_window} is [{}, 0]",
                    sliding_window.saturating_sub(1)
                ),
            ));
        }

        // The compressed indexer: the checkpoint's aliases must agree with the canonical names.
        let index_n_heads = u("index_n_heads")?;
        let index_head_dim = u("index_head_dim")?;
        let index_topk = u("index_topk")?;
        let index_pool = u("compressed_indexer_compress_ratio")?;
        if index_pool == 0 || index_topk % index_pool != 0 {
            return Err(refuse(
                "compressed_indexer_compress_ratio",
                &index_pool,
                "index_topk must be a whole number of pools",
            ));
        }
        for (key, want) in [
            ("compressed_indexer_num_attention_heads", index_n_heads),
            ("compressed_indexer_head_dim", index_head_dim),
            ("compressed_indexer_candidate_topk", index_topk),
            ("compressed_indexer_group_topk", index_topk / index_pool),
        ] {
            expect_u(key, want, "disagrees with the canonical index_* fields")?;
        }
        if index_head_dim < qk_rope {
            return Err(refuse(
                "index_head_dim",
                &index_head_dim,
                "narrower than the rotary tail",
            ));
        }

        let engram_layer_ids = text
            .u32_array("engram_layer_ids")?
            .ok_or_else(|| missing("engram_layer_ids"))?;
        let engram_max_ngram_size = u("engram_max_ngram_size")?;
        let engram_n_head_per_ngram = u("engram_n_head_per_ngram")?;
        let engram_n_embed_per_ngram = u("engram_n_embed_per_ngram")?;
        if engram_max_ngram_size < 2
            || engram_n_head_per_ngram == 0
            || engram_n_embed_per_ngram % engram_n_head_per_ngram != 0
        {
            return Err(refuse(
                "engram_max_ngram_size/engram_n_head_per_ngram/engram_n_embed_per_ngram",
                &(
                    engram_max_ngram_size,
                    engram_n_head_per_ngram,
                    engram_n_embed_per_ngram,
                ),
                "n-gram orders from 2 and a whole head width",
            ));
        }
        let sizes = text
            .object("engram_layer_head_vocab_sizes")?
            .ok_or_else(|| missing("engram_layer_head_vocab_sizes"))?;
        let mut engram_head_vocab_sizes = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for id in &engram_layer_ids {
            if *id == 0 || *id > layers || !seen.insert(*id) {
                return Err(refuse(
                    "engram_layer_ids",
                    &engram_layer_ids,
                    &format!("unique 1-based ids in 1..={layers}"),
                ));
            }
            let key = format!("engram_layer_head_vocab_sizes.{id}");
            let orders = sizes
                .u64_matrix(&id.to_string())?
                .ok_or_else(|| missing(&key))?;
            if orders.len() != (engram_max_ngram_size - 1) as usize
                || orders
                    .iter()
                    .any(|order| order.len() != engram_n_head_per_ngram as usize)
                || orders.iter().flatten().any(|size| *size == 0)
            {
                return Err(refuse(
                    &key,
                    &orders.iter().map(Vec::len).collect::<Vec<_>>(),
                    "one list of engram_n_head_per_ngram positive moduli per n-gram order",
                ));
            }
            engram_head_vocab_sizes.push(orders.into_iter().flatten().collect());
        }
        let engram_pad_id = u("engram_pad_id")?;
        if engram_pad_id >= ENGRAM_LOOKUP_TABLE_SIZE {
            return Err(refuse(
                "engram_pad_id",
                &engram_pad_id,
                "must lie inside the lookup table",
            ));
        }

        let hc_mult = u("hc_mult")?;
        let n_routed_experts = u("n_routed_experts")?;
        let num_experts_per_tok = u("num_experts_per_tok")?;
        if hc_mult == 0 || num_experts_per_tok == 0 || num_experts_per_tok > n_routed_experts {
            return Err(refuse(
                "hc_mult/num_experts_per_tok",
                &(hc_mult, num_experts_per_tok),
                "positive, and top-k within the expert count",
            ));
        }
        let quant = top
            .raw_object("quantization_config")?
            .ok_or_else(|| {
                missing("quantization_config (the pack onboards the ModelOpt NVFP4 artifact)")
            })
            .and_then(LayerQuantMap::from_config_quantization)?;

        let vision = top
            .object("vision_config")?
            .ok_or_else(|| missing("vision_config"))?;
        let vu = |key: &str| {
            vision
                .u32(key)?
                .ok_or_else(|| missing(&format!("vision_config.{key}")))
        };
        let vision = HyV4VisionConfig {
            num_hidden_layers: vu("num_hidden_layers")?,
            hidden_size: vu("hidden_size")?,
            intermediate_size: vu("intermediate_size")?,
            num_channels: vu("num_channels")?,
            patch_size: vu("patch_size")?,
            max_vit_seq_len: vu("max_vit_seq_len")?,
            spatial_merge_size: vu("spatial_merge_size")?,
            out_hidden_size: vu("out_hidden_size")?,
        };
        if vision.out_hidden_size != hidden {
            return Err(refuse(
                "vision_config.out_hidden_size",
                &vision.out_hidden_size,
                "the tower projects into the text width",
            ));
        }

        let tu = |key: &str| top.u32(key)?.ok_or_else(|| missing(key));
        let target_list = |key: &str| -> Result<Vec<u32>, String> {
            let list = top.u32_array(key)?.ok_or_else(|| missing(key))?;
            if list.iter().any(|layer| *layer >= layers) {
                return Err(refuse(key, &list, "0-based trunk layers"));
            }
            Ok(list)
        };
        let dfly = HyV4DflyConfig {
            num_hidden_layers: tu("dfly_num_hidden_layers")?,
            mask_token_id: tu("dfly_mask_token_id")?,
            target_kv_layers: target_list("dfly_target_kv_layers")?,
            target_index_layers: target_list("dfly_target_index_layer")?,
            hidden_correction: top
                .boolean("dfly_enable_hidden_correction")?
                .ok_or_else(|| missing("dfly_enable_hidden_correction"))?,
        };
        if dfly.target_kv_layers.len() != dfly.num_hidden_layers as usize
            || dfly.target_index_layers != dfly.target_kv_layers
        {
            return Err(refuse(
                "dfly_target_kv_layers/dfly_target_index_layer",
                &(&dfly.target_kv_layers, &dfly.target_index_layers),
                "one shared target layer per drafter block",
            ));
        }
        if let Some(layer) = dfly
            .target_kv_layers
            .iter()
            .find(|layer| layer_kinds[**layer as usize] != HyV4LayerKind::Indexed)
        {
            return Err(refuse(
                "dfly_target_kv_layers",
                layer,
                "the drafter reads indexer top-k, so its targets are indexed layers",
            ));
        }

        Ok(Self {
            num_hidden_layers: layers,
            hidden_size: hidden,
            vocab_size: u("vocab_size")?,
            rms_norm_eps: f("rms_norm_eps")?,
            max_position_embeddings: u("max_position_embeddings")?,
            layer_kinds,
            dense_layers,
            indexed,
            sliding,
            sliding_window,
            q_lora_rank: u("q_lora_rank")?,
            kv_lora_rank: u("kv_lora_rank")?,
            index_n_heads,
            index_head_dim,
            index_topk,
            index_pool,
            hc_mult,
            hc_eps: f("hc_eps")?,
            hc_magnitude: f("hc_magnitude")?,
            intermediate_size: u("intermediate_size")?,
            n_routed_experts,
            num_experts_per_tok,
            moe_intermediate_size: u("moe_intermediate_size")?,
            routed_expert_hidden_size: u("routed_expert_hidden_size")?,
            n_shared_experts: u("n_shared_experts")?,
            routed_scaling_factor: f("routed_scaling_factor")?,
            norm_topk_prob: b("norm_topk_prob")?,
            swiglu_limit: f("swiglu_limit")?,
            engram_layer_ids,
            engram_max_ngram_size,
            engram_n_head_per_ngram,
            engram_n_embed_per_ngram,
            engram_head_vocab_sizes,
            engram_pad_id,
            engram_seed: text
                .u64("engram_seed")?
                .ok_or_else(|| missing("engram_seed"))?,
            engram_fp8_embedding: b("engram_fp8_embedding")?,
            engram_lookup_table_size: ENGRAM_LOOKUP_TABLE_SIZE,
            tie_word_embeddings: tie,
            quant,
            vision,
            dfly,
        })
    }
}
