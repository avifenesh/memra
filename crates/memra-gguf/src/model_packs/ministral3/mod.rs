use super::*;
use crate::config::HfConfig;

/// Pinned Ministral-3-8B text decoder. Pixtral and the projector are excluded.
pub static PACK: ModelPack = ModelPack {
    family: "ministral3",
    aliases: &["ministral3", "mistral3"],
    config_layout: ConfigLayout::FlatOrTextConfig,
    tokenizer_sources: &[TokenizerSource::TokenizerJson],
    template: TemplateContract::ArtifactRequired,
    support: Some(NativeSupport::NativeReference), // Native tiny executor and boundary test, 2026-09-09.
    gates: &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ],
    checkpoint_parity: Some(CheckpointParityGate {
        max_abs: 0.005,
        max_rel: 0.005,
        require_argmax: true,
    }),
    matches_config: |c| {
        c.arch == Arch::Ministral3
            && c.window_hint.is_none()
            && c.rope_yarn.is_some()
            && c.moe.is_none()
    },
    plan_builder: ministral_plan,
    tensor_schema: canonical_tensor_schema,
    tiny_plan: Some(tiny_plan),
};

fn ministral_plan(cfg: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    let mut plan = canonical_plan(cfg)?;
    // Publisher guidance is temperature below 0.1 for daily use; examples use 0.15.
    // Choose a sampled default within the daily-use recommendation, never greedy.
    plan.sampling_defaults = Some(crate::model_plan::SamplingDefaultsPlan {
        temperature: 0.05,
        top_p: 1.0,
    });
    Ok(plan)
}

fn tiny_plan() -> Result<ModelPlan, PlanCompileError> {
    let mut cfg = ModelConfig::from_hf(&HfConfig::parse(include_str!("config.json")));
    cfg.n_layer = 2;
    cfg.n_layer_total = 2;
    cfg.n_embd = 8;
    cfg.n_ff = 16;
    cfg.n_head = 2;
    cfg.n_head_kv = 1;
    cfg.head_dim_k = 4;
    cfg.head_dim_v = 4;
    cfg.rope_dim_count = 4;
    cfg.n_vocab = 32;
    canonical_plan(&cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_plan::{AttentionPlan, RopeFactors, TensorPresence};
    #[test]
    fn pinned_config_compiles_text_only() {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(include_str!("config.json")));
        assert_eq!(super::super::for_config(&cfg).unwrap().family, "ministral3");
        let plan = PACK.compile_plan(&cfg).unwrap();
        assert_eq!(plan.layers.len(), 34);
        assert!(plan.vision.is_none());
        for l in &plan.layers {
            let AttentionPlan::Full(a) = &l.attention else {
                panic!("full causal required")
            };
            assert_eq!(a.qk_norm, TensorPresence::Absent);
            assert!(matches!(
                a.rope.factors,
                RopeFactors::YarnQueryScaled {
                    original_context: 16384,
                    ..
                }
            ));
        }
    }
    #[test]
    fn pinned_text_tensor_census() {
        use crate::safetensors::StInfo;
        use crate::source::census_from_safetensors_headers;
        use std::collections::BTreeMap;
        let mut headers = BTreeMap::new();
        for line in include_str!("text-census.tsv").lines() {
            let c: Vec<_> = line.split('\t').collect();
            headers.insert(
                c[0].to_string(),
                StInfo {
                    dtype: c[1].into(),
                    shape: c[2]
                        .split(',')
                        .filter(|x| !x.is_empty())
                        .map(|x| x.parse().unwrap())
                        .collect(),
                    data_offsets: [c[3].parse().unwrap(), c[4].parse().unwrap()],
                },
            );
        }
        assert_eq!(headers.len(), 785);
        let census = census_from_safetensors_headers(&headers).unwrap();
        assert_eq!(census.tensors.len(), 309);
        assert_eq!(
            census
                .tensors
                .iter()
                .map(|r| r.entry.physical_bytes)
                .sum::<u64>(),
            9_563_579_320
        );
        let cfg = ModelConfig::from_hf(&HfConfig::parse(include_str!("config.json")));
        let plan = PACK.compile_plan(&cfg).unwrap();
        let contract = PACK
            .compile_tensor_contract(
                &cfg,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let entries: Vec<_> = census.tensors.into_iter().map(|r| r.entry).collect();
        contract.bind(&entries).unwrap();
    }

    #[test]
    fn changed_math_is_refused() {
        for (from, to) in [
            ("\"sliding_window\": null", "\"sliding_window\": 4096"),
            ("\"mscale\": 1.0", "\"mscale\": 2.0"),
            (
                "\"tie_word_embeddings\": false",
                "\"tie_word_embeddings\": true",
            ),
            ("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""),
        ] {
            let json = include_str!("config.json").replace(from, to);
            assert_ne!(json, include_str!("config.json"));
            assert!(std::panic::catch_unwind(|| HfConfig::parse(&json)).is_err());
        }
    }
    #[test]
    fn tiny_program_compiles() {
        PACK.compile_tiny_plan().unwrap();
    }
}
