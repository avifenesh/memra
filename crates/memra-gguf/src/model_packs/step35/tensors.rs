//! Step safetensors tensor dialect. Names match the existing native source mapping; shapes and
//! required operations come from ModelPlan. This does not promote pack or rewrite qualification.
use crate::model_plan::{ModelPlan, MoeMlpPlan, RouterPlan};
use crate::tensor_contract::{
    LayerTensor, MtpTensor, QuantConstraint, TensorContract, TensorContractError, TensorId,
    TensorMatch, TensorOwner, TensorRequirement, TensorTransform,
};

fn requirement(
    id: TensorId,
    name: String,
    shape: Vec<u64>,
    owner: TensorOwner,
    quant: QuantConstraint,
    transform: TensorTransform,
    required: bool,
) -> TensorRequirement {
    TensorRequirement {
        id,
        names: vec![name],
        match_mode: TensorMatch::OneOf,
        shape,
        owner,
        transform,
        quant,
        auxiliaries: None,
        required,
    }
}

#[allow(clippy::result_large_err)] // allow: preserve the shared contract diagnostic type
pub(crate) fn hf_moe_requirements(
    plan: &ModelPlan,
    index: u32,
    moe: &MoeMlpPlan,
) -> Result<Vec<TensorRequirement>, TensorContractError> {
    let mut rows = Vec::new();
    let owner = TensorOwner::Layer(index);
    let h = u64::from(plan.hidden_size);
    let f = u64::from(moe.expert_intermediate_size);
    let e = u64::from(moe.expert_count);
    let mut add = |tensor, suffix: &str, shape, quant| {
        rows.push(requirement(
            TensorId::Layer { index, tensor },
            format!("model.layers.{index}.{suffix}"),
            shape,
            owner,
            quant,
            TensorTransform::Identity,
            true,
        ))
    };
    add(
        LayerTensor::MoeRouter,
        "moe.gate.weight",
        vec![e, h],
        QuantConstraint::FloatOnly,
    );
    if matches!(
        moe.router,
        RouterPlan::Sigmoid {
            selection_bias: true,
            ..
        }
    ) {
        add(
            LayerTensor::MoeRouterBias,
            "moe.router_bias",
            vec![e],
            QuantConstraint::FloatOnly,
        );
    }
    for (tensor, suffix, shape) in [
        (
            LayerTensor::MoeExpertGateBank,
            "moe.gate_proj.weight",
            vec![e, f, h],
        ),
        (
            LayerTensor::MoeExpertUpBank,
            "moe.up_proj.weight",
            vec![e, f, h],
        ),
        (
            LayerTensor::MoeExpertDownBank,
            "moe.down_proj.weight",
            vec![e, h, f],
        ),
    ] {
        add(tensor, suffix, shape, QuantConstraint::Weight);
    }
    if let Some(shared) = &moe.shared {
        if shared.gated {
            return Err(TensorContractError::UnsupportedPlanOperation {
                operation: "Step HF gated shared expert",
            });
        }
        let s = u64::from(shared.intermediate_size);
        for (tensor, suffix, shape) in [
            (
                LayerTensor::SharedMlpGate,
                "share_expert.gate_proj.weight",
                vec![s, h],
            ),
            (
                LayerTensor::SharedMlpUp,
                "share_expert.up_proj.weight",
                vec![s, h],
            ),
            (
                LayerTensor::SharedMlpDown,
                "share_expert.down_proj.weight",
                vec![h, s],
            ),
        ] {
            add(tensor, suffix, shape, QuantConstraint::Weight);
        }
    }
    Ok(rows)
}

pub(crate) fn normalize_hf_contract(contract: &mut TensorContract, plan: &ModelPlan) {
    for tensor in &mut contract.requirements {
        match &tensor.id {
            TensorId::Layer {
                index,
                tensor: LayerTensor::AttentionGate,
            } => {
                tensor.names = vec![format!("model.layers.{index}.self_attn.g_proj.weight")];
            }
            TensorId::Layer {
                tensor: LayerTensor::QueryNorm | LayerTensor::KeyNorm,
                ..
            } => {
                // Step3p7RMSNorm folds +1 for these norms as well as block/output norms.
                tensor.transform = TensorTransform::NormAddOne;
            }
            TensorId::Mtp {
                depth,
                tensor: MtpTensor::OutputNorm,
            } => {
                let block = plan
                    .mtp_blocks
                    .iter()
                    .find(|b| b.depth == *depth)
                    .expect("MTP requirement has a planned block");
                tensor.names = vec![format!(
                    "model.layers.{}.transformer.shared_head.norm.weight",
                    block.layer.index
                )];
            }
            _ => {}
        }
    }
    for block in &plan.mtp_blocks {
        contract.requirements.push(requirement(
            TensorId::Mtp {
                depth: block.depth,
                tensor: MtpTensor::OutputProjection,
            },
            format!(
                "model.layers.{}.transformer.shared_head.output.weight",
                block.layer.index
            ),
            vec![u64::from(plan.vocab_size), u64::from(plan.hidden_size)],
            TensorOwner::Mtp(block.depth),
            QuantConstraint::Weight,
            TensorTransform::Identity,
            false,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};
    use crate::hf_mapping::{HfTarget, TransformKind, resolve_ggml};
    use crate::tensor_contract::{
        CheckpointDialect, ContractOptions, FloatType, IntegerType, StorageLayout,
        TensorCensusEntry,
    };

    fn contracts() -> (ModelConfig, ModelPlan, TensorContract, TensorContract) {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(include_str!("contract-fixture.json")));
        let plan = super::super::PACK.compile_plan(&cfg).unwrap();
        let hf = super::super::PACK
            .compile_tensor_contract(
                &cfg,
                &plan,
                CheckpointDialect::HfSafetensors,
                ContractOptions::default(),
            )
            .unwrap();
        let gguf = super::super::PACK
            .compile_tensor_contract(
                &cfg,
                &plan,
                CheckpointDialect::Gguf,
                ContractOptions::default(),
            )
            .unwrap();
        (cfg, plan, hf, gguf)
    }

    #[test]
    fn step_declared_hf_names_and_folds_match_the_existing_native_mapping() {
        let (cfg, plan, hf, gguf) = contracts();
        assert!(matches!(
            plan.mtp_blocks[0].layer.mlp,
            crate::model_plan::MlpPlan::Dense(_)
        ));
        let mut checked = 0;
        for requirement in &gguf.requirements {
            if matches!(
                requirement.id,
                TensorId::QuantAux { .. } | TensorId::RopeFactors
            ) {
                continue;
            }
            let target = hf
                .requirements
                .iter()
                .find(|r| r.id == requirement.id)
                .unwrap_or_else(|| panic!("missing HF role {:?}", requirement.id));
            let name = &requirement.names[0];
            let (mapped, transform) =
                match resolve_ggml(name, &cfg).unwrap_or_else(|| panic!("unmapped {name}")) {
                    HfTarget::Plain(name) => (name, TensorTransform::Identity),
                    HfTarget::Transform {
                        hf,
                        kind: TransformKind::NormPlusOne,
                    } => (hf, TensorTransform::NormAddOne),
                    _ => panic!("unexpected transform for {name}"),
                };
            assert!(
                target.names.contains(&mapped),
                "{name}: {:?} != {mapped}",
                target.names
            );
            assert_eq!(target.transform, transform, "{name}");
            assert_eq!(
                requirement.transform,
                TensorTransform::Identity,
                "GGUF tensors are already folded"
            );
            checked += 1;
        }
        assert_eq!(
            checked,
            3 + 2 * 12 + 2 * 17 + 5,
            "all four blocks, including dense MTP and its private head, must participate"
        );
    }

    #[test]
    fn step_full_census_binds_and_rejects_missing_misowned_and_misshaped_tensors() {
        let (_, _, hf, _) = contracts();
        let mut rows = Vec::new();
        for requirement in &hf.requirements {
            if matches!(requirement.id, TensorId::QuantAux { .. }) {
                continue;
            }
            for name in &requirement.names {
                rows.push(TensorCensusEntry {
                    name: name.clone(),
                    shape: requirement.shape.clone(),
                    storage: StorageLayout::Float(FloatType::Bf16),
                    physical_bytes: 2 * requirement.shape.iter().product::<u64>(),
                });
            }
        }
        let bound = hf.bind(&rows).unwrap();
        let gate = TensorId::Layer {
            index: 1,
            tensor: LayerTensor::MoeExpertGateBank,
        };
        assert_eq!(bound.tensors[&gate].shapes, [vec![6, 12, 16]]);
        let head = TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::OutputProjection,
        };
        assert_eq!(bound.tensors[&head].owner, TensorOwner::Mtp(0));
        assert_eq!(
            bound.tensors[&head].checkpoint_names,
            ["model.layers.3.transformer.shared_head.output.weight"]
        );
        let bank = rows
            .iter()
            .position(|r| r.name == "model.layers.1.moe.gate_proj.weight")
            .unwrap();
        let mut missing = rows.clone();
        missing.remove(bank);
        assert!(
            hf.bind(&missing)
                .unwrap_err()
                .to_string()
                .contains("missing")
        );
        let mut wrong = rows.clone();
        wrong[bank].shape = vec![6, 16, 12];
        assert!(hf.bind(&wrong).unwrap_err().to_string().contains("shape"));
        let router = rows
            .iter()
            .position(|r| r.name == "model.layers.1.moe.gate.weight")
            .unwrap();
        let mut wrong = rows.clone();
        wrong[router].storage = StorageLayout::Integer(IntegerType::I64);
        assert!(hf.bind(&wrong).is_err());
        let mut wrong = rows.clone();
        wrong.push(TensorCensusEntry {
            name: "model.layers.2.transformer.shared_head.output.weight".into(),
            shape: vec![64, 16],
            storage: StorageLayout::Float(FloatType::Bf16),
            physical_bytes: 2048,
        });
        assert!(
            hf.bind(&wrong)
                .unwrap_err()
                .to_string()
                .contains("extra checkpoint tensors")
        );
    }
}
