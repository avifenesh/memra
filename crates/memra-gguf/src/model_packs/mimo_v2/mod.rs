//! MiMo V2.6 tensor schema slices. The model pack remains unregistered until
//! checkpoint and executable modality contracts are complete.

pub(crate) mod mtp;

use crate::model_plan::{ModelPlan, MoeMlpPlan};
use crate::tensor_contract::{
    ExpertTensor, QuantConstraint, TensorId, TensorMatch, TensorOwner, TensorRequirement,
    TensorTransform,
};

pub(crate) fn mint_expert_requirements(
    plan: &ModelPlan,
    layer: u32,
    moe: &MoeMlpPlan,
) -> Vec<TensorRequirement> {
    let mut requirements = Vec::with_capacity(moe.expert_count as usize * 3);
    for expert in 0..moe.expert_count {
        for (tensor, projection, output, input) in [
            (
                ExpertTensor::Gate,
                "gate",
                moe.expert_intermediate_size,
                plan.hidden_size,
            ),
            (
                ExpertTensor::Up,
                "up",
                moe.expert_intermediate_size,
                plan.hidden_size,
            ),
            (
                ExpertTensor::Down,
                "down",
                plan.hidden_size,
                moe.expert_intermediate_size,
            ),
        ] {
            let name = crate::hf_mapping::hf_expert_name(layer, expert, projection, &plan.arch);
            let stem = name.strip_suffix(".weight").unwrap();
            let auxiliaries = vec![
                format!("{stem}.weight_scale"),
                format!("{stem}.weight_scale_2"),
                format!("{stem}.input_scale"),
            ];
            requirements.push(TensorRequirement {
                id: TensorId::Expert {
                    layer,
                    expert,
                    tensor,
                },
                names: vec![name],
                match_mode: TensorMatch::OneOf,
                shape: vec![output as u64, input as u64],
                owner: TensorOwner::Layer(layer),
                transform: TensorTransform::Identity,
                quant: QuantConstraint::Nvfp4,
                auxiliaries: Some(auxiliaries),
                required: true,
            });
        }
    }
    requirements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};
    use crate::tensor_contract::{
        CheckpointDialect, ContractOptions, QuantLayout, StorageLayout, TensorCensusEntry,
        TensorContract, TensorContractError,
    };

    #[test]
    fn pinned_mint_owns_every_routed_expert_projection_separately() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = ModelPlan::compile(&config).unwrap();
        let contract = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        let experts: Vec<_> = contract
            .requirements
            .iter()
            .filter(|requirement| matches!(&requirement.id, TensorId::Expert { .. }))
            .collect();
        assert_eq!(experts.len(), 47 * 256 * 3);
        for (tensor, projection, shape) in [
            (ExpertTensor::Gate, "gate", vec![2_048, 4_096]),
            (ExpertTensor::Up, "up", vec![2_048, 4_096]),
            (ExpertTensor::Down, "down", vec![4_096, 2_048]),
        ] {
            let requirement = experts
                .iter()
                .copied()
                .find(|requirement| {
                    requirement.id
                        == TensorId::Expert {
                            layer: 1,
                            expert: 0,
                            tensor,
                        }
                })
                .unwrap();
            let stem = format!("model.layers.1.mlp.experts.0.{projection}_proj");
            assert_eq!(requirement.names, vec![format!("{stem}.weight")]);
            assert_eq!(requirement.shape, shape);
            assert_eq!(requirement.quant, QuantConstraint::Nvfp4);
            assert_eq!(
                requirement.auxiliaries,
                Some(vec![
                    format!("{stem}.weight_scale"),
                    format!("{stem}.weight_scale_2"),
                    format!("{stem}.input_scale"),
                ])
            );
            assert!(requirement.required);
        }
    }

    #[test]
    fn mint_expert_refuses_source_format_and_missing_input_scale() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let plan = ModelPlan::compile(&config).unwrap();
        let requirement = mint_expert_requirements(
            &plan,
            1,
            match &plan.layers[1].mlp {
                crate::model_plan::MlpPlan::Moe(moe) => moe,
                _ => panic!("layer 1 must be MoE"),
            },
        )
        .into_iter()
        .next()
        .unwrap();
        let contract = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: vec![requirement.clone()],
        };
        let mut entry = TensorCensusEntry {
            name: requirement.names[0].clone(),
            shape: requirement.shape.clone(),
            storage: StorageLayout::Quantized(QuantLayout {
                format: "MXFP4".to_owned(),
                block_shape: vec![32],
                auxiliaries: vec![requirement.auxiliaries.as_ref().unwrap()[0].clone()],
            }),
            physical_bytes: 1,
        };
        assert!(matches!(
            contract.bind(&[entry.clone()]),
            Err(TensorContractError::QuantLayoutMismatch { .. })
        ));
        if let StorageLayout::Quantized(layout) = &mut entry.storage {
            layout.format = "NVFP4".to_owned();
            layout.block_shape = vec![16];
        }
        assert!(matches!(
            contract.bind(&[entry.clone()]),
            Err(TensorContractError::AuxiliaryLayoutMismatch { .. })
        ));
        if let StorageLayout::Quantized(layout) = &mut entry.storage {
            layout.auxiliaries = requirement.auxiliaries.unwrap();
        }
        contract.bind(&[entry]).unwrap();
    }
}
