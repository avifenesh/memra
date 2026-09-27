//! Tensor ownership for MiMo's separately stored three-block MTP draft.
//! This is a schema slice. No draft execution is admitted by the model pack.

use crate::config::{Arch, ModelConfig};
use crate::tensor_contract::{
    FloatType, MtpTensor, QuantConstraint, TensorContractError, TensorId, TensorMatch, TensorOwner,
    TensorRequirement, TensorTransform,
};

#[allow(clippy::result_large_err)] // the tensor contract error keeps the diagnostic tensor identity
#[allow(dead_code)] // consumed by the subsequent full MiMo mint inspector, never a load path
pub(crate) fn mint_mtp_requirements(
    config: &ModelConfig,
) -> Result<Vec<TensorRequirement>, TensorContractError> {
    if config.arch != Arch::MiMoV2
        || config.n_embd != 4_096
        || config.n_layer != 48
        || config
            .mimo
            .as_ref()
            .and_then(|mimo| mimo.separate_mtp_layers)
            != Some(3)
    {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "MiMo MTP draft geometry differs from the pinned three-block artifact",
        });
    }

    let mut rows = Vec::with_capacity(3 * 12);
    for depth in 0..3 {
        for (tensor, suffix, shape, fp8) in [
            (
                MtpTensor::FusionProjection,
                "eh_proj.weight",
                vec![4_096, 8_192],
                false,
            ),
            (MtpTensor::EmbeddingNorm, "enorm.weight", vec![4_096], false),
            (
                MtpTensor::OutputNorm,
                "final_layernorm.weight",
                vec![4_096],
                false,
            ),
            (MtpTensor::HiddenNorm, "hnorm.weight", vec![4_096], false),
            (
                MtpTensor::PreAttentionNorm,
                "input_layernorm.weight",
                vec![4_096],
                false,
            ),
            (
                MtpTensor::MlpDown,
                "mlp.down_proj.weight",
                vec![4_096, 16_384],
                true,
            ),
            (
                MtpTensor::MlpGate,
                "mlp.gate_proj.weight",
                vec![16_384, 4_096],
                true,
            ),
            (
                MtpTensor::MlpUp,
                "mlp.up_proj.weight",
                vec![16_384, 4_096],
                true,
            ),
            (
                MtpTensor::PreMlpNorm,
                "pre_mlp_layernorm.weight",
                vec![4_096],
                false,
            ),
            (
                MtpTensor::AttentionSink,
                "self_attn.attention_sink_bias",
                vec![64],
                false,
            ),
            (
                MtpTensor::AttentionOutput,
                "self_attn.o_proj.weight",
                vec![4_096, 8_192],
                false,
            ),
            (
                MtpTensor::FusedQkv,
                "self_attn.qkv_proj.weight",
                vec![14_848, 4_096],
                true,
            ),
        ] {
            let name = format!("model.mtp.layers.{depth}.{suffix}");
            let auxiliaries = fp8.then(|| {
                vec![format!(
                    "{}.weight_scale_inv",
                    name.strip_suffix(".weight").unwrap()
                )]
            });
            rows.push(TensorRequirement {
                id: TensorId::Mtp { depth, tensor },
                names: vec![name],
                match_mode: TensorMatch::OneOf,
                shape,
                owner: TensorOwner::Mtp(depth),
                transform: TensorTransform::Identity,
                quant: if fp8 {
                    QuantConstraint::Fp8Block128
                } else {
                    QuantConstraint::ExactFloat(FloatType::Bf16)
                },
                auxiliaries,
                required: true,
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HfConfig;
    use crate::tensor_contract::{
        CheckpointDialect, QuantLayout, StorageLayout, TensorCensusEntry, TensorContract,
    };

    fn pinned() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")))
    }

    #[test]
    fn pinned_mtp_sidecar_has_three_complete_draft_blocks() {
        let rows = mint_mtp_requirements(&pinned()).unwrap();
        assert_eq!(rows.len(), 36);
        for depth in 0..3 {
            let draft: Vec<_> = rows
                .iter()
                .filter(|row| row.owner == TensorOwner::Mtp(depth))
                .collect();
            assert_eq!(draft.len(), 12);
            let qkv = draft
                .iter()
                .find(|row| {
                    row.id
                        == TensorId::Mtp {
                            depth,
                            tensor: MtpTensor::FusedQkv,
                        }
                })
                .unwrap();
            assert_eq!(qkv.shape, vec![14_848, 4_096]);
            assert_eq!(
                qkv.auxiliaries,
                Some(vec![format!(
                    "model.mtp.layers.{depth}.self_attn.qkv_proj.weight_scale_inv"
                )])
            );
            assert_eq!(qkv.quant, QuantConstraint::Fp8Block128);
        }
        let mut changed = pinned();
        changed.mimo.as_mut().unwrap().separate_mtp_layers = Some(2);
        assert!(mint_mtp_requirements(&changed).is_err());
    }

    #[test]
    fn mtp_scale_sibling_is_required_by_the_header_contract() {
        let row = mint_mtp_requirements(&pinned())
            .unwrap()
            .into_iter()
            .find(|row| {
                row.id
                    == TensorId::Mtp {
                        depth: 0,
                        tensor: MtpTensor::FusedQkv,
                    }
            })
            .unwrap();
        let contract = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: vec![row.clone()],
        };
        let mut entry = TensorCensusEntry {
            name: row.names[0].clone(),
            shape: row.shape.clone(),
            storage: StorageLayout::Quantized(QuantLayout {
                format: "FP8_E4M3".to_owned(),
                block_shape: vec![128, 128],
                auxiliaries: vec![],
            }),
            physical_bytes: 1,
        };
        assert!(matches!(
            contract.bind(&[entry.clone()]),
            Err(TensorContractError::AuxiliaryLayoutMismatch { .. })
        ));
        let StorageLayout::Quantized(layout) = &mut entry.storage else {
            unreachable!()
        };
        layout.auxiliaries = row.auxiliaries.unwrap();
        contract.bind(&[entry]).unwrap();
    }
}
