//! Compare the portable ModelPlan executor with Memra's native CUDA eager rewrite.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorMatch,
};
use memra_gguf::{GgmlType, model_plan::ModelPlan};
use memra_reference::{ReferenceTensor, deterministic_fixture, execute};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeMap;

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
}

struct FixtureSource {
    cfg: ModelConfig,
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }

    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let tensor = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&tensor.bytes),
            ggml_type: GgmlType::F32,
            ne: tensor.ne.clone(),
        })
    }
}

fn fixture_source(
    cfg: &ModelConfig,
    plan: &ModelPlan,
    weights: &BTreeMap<memra_gguf::tensor_contract::TensorId, ReferenceTensor>,
) -> Result<FixtureSource, Box<dyn std::error::Error>> {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )?;
    let mut tensors = BTreeMap::new();
    for requirement in contract
        .requirements
        .iter()
        .filter(|requirement| requirement.required || weights.contains_key(&requirement.id))
    {
        let tensor = weights
            .get(&requirement.id)
            .ok_or_else(|| format!("reference fixture is missing {:?}", requirement.id))?;
        let elements: usize = requirement.shape.iter().map(|&dim| dim as usize).product();
        if elements != tensor.data.len() {
            return Err(format!(
                "reference fixture {:?} has {} elements, contract requires {elements}",
                requirement.id,
                tensor.data.len()
            )
            .into());
        }
        let bytes: Vec<u8> = tensor
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let names = match requirement.match_mode {
            TensorMatch::OneOf => &requirement.names[..1],
            TensorMatch::All => requirement.names.as_slice(),
        };
        for name in names {
            if tensors
                .insert(
                    name.clone(),
                    OwnedTensor {
                        bytes: bytes.clone(),
                        ne: requirement.shape.clone(),
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate fixture tensor {name}").into());
            }
        }
    }
    Ok(FixtureSource {
        cfg: cfg.clone(),
        tensors,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let config_path = args
        .next()
        .ok_or("usage: modelplan-reference-gate <config.json> <receipt.tsv>")?;
    let receipt_path = args
        .next()
        .ok_or("usage: modelplan-reference-gate <config.json> <receipt.tsv>")?;
    let cfg = ModelConfig::from_hf(&HfConfig::parse(&std::fs::read_to_string(config_path)?));
    let pack = memra_gguf::model_packs::for_config(&cfg)
        .ok_or("reference gate requires a registered model pack")?;
    let plan = pack.compile_plan(&cfg)?;
    let fixture = deterministic_fixture(&plan)?;
    let reference = execute(&plan, &fixture.weights, &fixture.token_ids)?;
    let source = fixture_source(&cfg, &plan, &fixture.weights)?;

    let engine = Engine::new(0)?;
    let model = HybridModel::load_from_source(&engine, &source)?;
    let candidate = model.forward_last(&engine, &fixture.token_ids)?;
    let start = (reference.tokens - 1) * reference.vocab;
    let expected = &reference.logits[start..start + reference.vocab];

    let rewrite = memra_engine::plan_backend::execution_rewrites(&plan)
        .into_iter()
        .find(|rewrite| rewrite.surface == memra_engine::plan_backend::RewriteSurface::DecodeEager)
        .ok_or("decode-eager rewrite manifest is missing")?;
    let executable = std::fs::read(std::env::current_exe()?)?;
    let executable_sha256 = Sha256::digest(&executable)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let receipt = rewrite.verify_logits(
        &executable_sha256,
        expected,
        &candidate,
        memra_engine::plan_backend::RewriteParityPolicy {
            max_abs: 0.01,
            max_rel: 0.01,
            require_argmax: true,
        },
    )?;
    if !receipt.passed {
        return Err(format!(
            "native eager rewrite diverged from ModelPlan reference: max_abs={} max_rel={} first={:?}",
            receipt.max_abs, receipt.max_rel, receipt.first_violation
        )
        .into());
    }
    let receipt = memra_engine::plan_backend::bind_rewrite_artifact(receipt)?;
    receipt.validate_for(&rewrite)?;
    std::fs::write(&receipt_path, receipt.to_tsv())?;
    println!(
        "ModelPlan reference parity passed: values={} max_abs={} max_rel={} receipt={receipt_path}",
        receipt.values, receipt.max_abs, receipt.max_rel
    );
    Ok(())
}
