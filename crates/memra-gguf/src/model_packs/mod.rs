//! Statically linked model-family packs used by onboarding and load-time compilation.
//!
//! A pack is data plus ordinary Rust functions. There is no dynamic plugin ABI and no dispatch
//! inside a token loop: alias resolution and compilation happen once before weights are loaded.

use crate::config::{Arch, ModelConfig};
use crate::model_plan::{ModelPlan, PlanCompileError};
use crate::tensor_contract::{
    CheckpointDialect, ContractOptions, TensorContract, TensorContractError,
};

pub mod deepseek_v4;
pub mod gemma4_dense;
pub mod gemma4_moe;
pub mod glm5_next;
pub mod glm_dsa;
pub mod hy3;
pub mod llama_dense;
/// Speech packs use their own config normalization until the CLI accepts audio artifacts.
pub mod nemotron_rnnt;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35_moe;
pub mod qwen3_moe;
pub mod qwen4_exp;
pub mod step35;
pub mod whisper;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigLayout {
    Flat,
    FlatOrTextConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSource {
    TokenizerJson,
    GgufMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateContract {
    /// The artifact carries its own chat dialect, as a `tokenizer_config.json`
    /// `chat_template` or a `chat_template.jinja`. Absent means FAIL.
    ArtifactRequired,
    /// The family's chat dialect is CODE in this engine, because the vendor ships it
    /// as code and no artifact in the family carries a template.
    /// deepseek-v4 is the case this exists for: `deepseek-ai/DeepSeek-V4-Flash-0731`
    /// at its pinned revision ships `encoding/encoding_dsv4.py` and NO template in any
    /// form, so demanding one from the artifact refuses a correct artifact forever.
    ///
    /// This is NOT a fallback and must never become one. It names the renderer that owns
    /// the dialect, and the gate is two-sided: a family declaring a renderer must also
    /// refuse an artifact that carries a template of its own, because a template over an
    /// engine-owned dialect is a franken artifact whose renders silently disagree with
    /// the engine's. The renderer's own byte oracle is the vendor's `encoding/tests`
    /// fixtures, gated in memra-server and memra-tokenizer, not here.
    EngineRenderer(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NativeSupport {
    NativeReference,
    NativeQualified,
    NativeTuned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    Config,
    TokenizerTemplate,
    TensorCensus,
    TinyParity,
    CheckpointParity,
    RewriteParity,
    Serve,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckpointParityGate {
    pub max_abs: f32,
    pub max_rel: f32,
    pub require_argmax: bool,
}

pub struct ModelPack {
    pub family: &'static str,
    pub aliases: &'static [&'static str],
    pub config_layout: ConfigLayout,
    pub tokenizer_sources: &'static [TokenizerSource],
    pub template: TemplateContract,
    /// ModelPlan-pack qualification state, not a claim about legacy handwritten execution paths.
    /// `None` means inspection works but native plan execution is still unsupported.
    pub support: Option<NativeSupport>,
    pub gates: &'static [Gate],
    pub checkpoint_parity: Option<CheckpointParityGate>,
    matches_config: fn(&ModelConfig) -> bool,
    plan_builder: fn(&ModelConfig) -> Result<ModelPlan, PlanCompileError>,
    tensor_schema: fn(
        &ModelConfig,
        &ModelPlan,
        CheckpointDialect,
        ContractOptions,
    ) -> Result<TensorContract, TensorContractError>,
    tiny_plan: Option<fn() -> Result<ModelPlan, PlanCompileError>>,
}

impl ModelPack {
    pub fn matches_config(&self, config: &ModelConfig) -> bool {
        (self.matches_config)(config)
    }

    pub fn compile_plan(&self, config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
        if !self.matches_config(config) {
            return Err(PlanCompileError::ModelPackMismatch {
                pack: self.family,
                arch: format!("{:?}", config.arch),
            });
        }
        (self.plan_builder)(config)
    }

    #[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
    pub fn compile_tensor_contract(
        &self,
        config: &ModelConfig,
        plan: &ModelPlan,
        dialect: CheckpointDialect,
        options: ContractOptions,
    ) -> Result<TensorContract, TensorContractError> {
        (self.tensor_schema)(config, plan, dialect, options)
    }

    pub fn compile_tiny_plan(&self) -> Result<ModelPlan, PlanCompileError> {
        self.tiny_plan
            .ok_or(PlanCompileError::MissingTinyFixture { pack: self.family })?()
    }
}

pub const PACKS: &[&ModelPack] = &[
    &qwen3::PACK,
    &qwen3_moe::PACK,
    &qwen35::PACK,
    &qwen35_moe::PACK,
    &glm5_next::PACK,
    &glm_dsa::PACK,
    &gemma4_dense::PACK,
    &gemma4_moe::PACK,
    &deepseek_v4::PACK,
    &deepseek_v4::DSPARK_PACK,
    &qwen4_exp::PACK,
    &step35::PACK,
    &hy3::PACK,
    // Last: the plainest dense stack, so a family with its own pack is always matched first.
    &llama_dense::PACK,
];

/// Explicit artifact-storage profiles used by `model inspect`. They never participate in
/// automatic family selection because several profiles can intentionally share one ModelPlan.
pub const ONBOARDING_PROFILES: &[&ModelPack] = &[&hy3::NVFP4_PACK];

pub fn by_alias(alias: &str) -> Option<&'static ModelPack> {
    PACKS
        .iter()
        .chain(ONBOARDING_PROFILES)
        .copied()
        .find(|pack| pack.family == alias || pack.aliases.contains(&alias))
}

pub fn for_config(config: &ModelConfig) -> Option<&'static ModelPack> {
    PACKS
        .iter()
        .copied()
        .find(|pack| pack.matches_config(config))
}

pub(super) fn canonical_plan(config: &ModelConfig) -> Result<ModelPlan, PlanCompileError> {
    ModelPlan::compile(config)
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
pub(super) fn canonical_tensor_schema(
    _config: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    TensorContract::for_plan(plan, dialect, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HfConfig;
    use std::collections::BTreeSet;

    #[test]
    fn pack_aliases_are_unique_and_manifests_are_complete() {
        let mut aliases = BTreeSet::new();
        for pack in PACKS.iter().chain(ONBOARDING_PROFILES) {
            assert!(!pack.family.is_empty());
            assert!(!pack.aliases.is_empty());
            assert!(!pack.tokenizer_sources.is_empty());
            // Every pack declares WHERE its chat dialect comes from, and a renderer arm
            // must name the renderer rather than leaving it blank: a nameless renderer
            // contract is indistinguishable from a fallback.
            match pack.template {
                TemplateContract::ArtifactRequired => {}
                TemplateContract::EngineRenderer(renderer) => {
                    assert!(!renderer.is_empty(), "{}", pack.family)
                }
            }
            assert_eq!(
                pack.gates,
                &[
                    Gate::Config,
                    Gate::TokenizerTemplate,
                    Gate::TensorCensus,
                    Gate::TinyParity,
                    Gate::CheckpointParity,
                    Gate::RewriteParity,
                    Gate::Serve,
                ]
            );
            for alias in pack.aliases {
                assert!(aliases.insert(*alias), "duplicate model-pack alias {alias}");
                assert_eq!(by_alias(alias).unwrap().family, pack.family);
            }
            if pack.support.is_none() {
                // A loader-lane pack may still carry a tiny fixture (it protects the family's
                // from_hf parse before any native execution exists) — but never a parity gate.
                match pack.tiny_plan {
                    Some(_) => {
                        pack.compile_tiny_plan()
                            .expect("loader-lane tiny fixture must compile");
                    }
                    None => assert!(matches!(
                        pack.compile_tiny_plan(),
                        Err(PlanCompileError::MissingTinyFixture { .. })
                    )),
                }
                assert!(pack.checkpoint_parity.is_none());
            } else {
                let gate = pack
                    .checkpoint_parity
                    .expect("native reference packs require a checkpoint parity gate");
                assert!(gate.max_abs >= 0.0 && gate.max_rel >= 0.0);
                assert!(
                    gate.max_abs > 0.0 || gate.max_rel > 0.0,
                    "checkpoint parity gate must carry a positive absolute or relative bound"
                );
            }
        }
    }

    #[test]
    fn registered_pack_compiles_plan_and_schema_and_refuses_other_family() {
        let qwen = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        ));
        let qwen_pack = for_config(&qwen).unwrap();
        assert_eq!(qwen_pack.family, "qwen3");
        let plan = qwen_pack.compile_plan(&qwen).unwrap();
        qwen_pack.compile_tiny_plan().unwrap();
        qwen_pack
            .compile_tensor_contract(
                &qwen,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();

        let llama = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"llama","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        ));
        assert!(matches!(
            qwen_pack.compile_plan(&llama),
            Err(PlanCompileError::ModelPackMismatch { pack: "qwen3", .. })
        ));
    }
}
